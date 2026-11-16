// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! All independent FC/SD/PSO/DSO coupling components and spin channels
//! against independent finite complex SCF and direct semicanonical PT2.
//! All 925 component/channel fits pass across five parent families; source
//! normalization is exact, all failed intervals retained, and TPSS includes
//! the ordinary current-kernel zero-correlation limit.
#[path = "support/complex_pt2.rs"]
mod complex_pt2;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/nonlocal_complex_oracle.rs"]
mod nonlocal_complex_oracle;
#[path = "support/stencils.rs"]
mod stencils;
use dft_rs::constants::{ELECTRON_G_FACTOR, FINE_STRUCTURE};
use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::grid::{eval::eval_aos, GridOptions, MolecularGrid};
use dft_rs::integrals::london::{repulsion_series, ComplexMatrix, FieldPath};
use dft_rs::nlc::Vv10Params;
use dft_rs::properties::correlated_spin_spin::correlated_spin_spin_couplings_for_model;
use dft_rs::response::complex::Complex;
use dft_rs::response::field_path::FieldEnergyModel;
use dft_rs::response::spin_spin::{nuclear_magnetic_sites, spin_spin_couplings};
use dft_rs::response::ResponseSolverConfig;
use dft_rs::scf::magnetic::{magnetic_field_scf_with_core, MagneticFieldOptions};
use dft_rs::xc::double_hybrid::Pt2Coefficients;
use dft_rs::Matrix;

#[test]
fn every_closed_shell_pt2_j_term_has_complete_independent_component_fits() {
    fit_couplings(false);
}

#[test]
fn every_nonlocal_pt2_j_term_has_complete_independent_component_fits() {
    fit_couplings(true);
}

fn fit_couplings(nonlocal: bool) {
    let models: &[&str] = if nonlocal {
        &["hf", "pbe0"]
    } else {
        &["hf", "svwn5", "pbe", "tpss", "pbe0"]
    };
    let selected = std::env::var("DFT_RS_TEST_MODEL").ok();
    assert!(selected
        .as_ref()
        .is_none_or(|v| models.contains(&v.as_str())));
    for &xc in models {
        if selected.as_ref().is_some_and(|v| v != xc) {
            continue;
        }
        let mut positions = vec![
            [0.13, -0.21, 0.19],
            [1.5, -0.04, 0.06],
            [0.44, 3.02, 0.66],
            [1.94, 3.24, 1.15],
        ];
        for a in 0..3 {
            let center = positions.iter().map(|r| r[a]).sum::<f64>() / 4.0;
            for r in &mut positions {
                r[a] -= center;
            }
        }
        let method = MolecularMethod {
            xc: Some(xc.into()),
            basis: "sto-3g".into(),
            conv_tol: 1e-14,
            conv_tol_dp: 1e-14,
            conv_tol_grad: Some(1e-14),
            max_iter: 400,
            guess: Some("core".into()),
            dispersion: if nonlocal { "vv10" } else { "none" }.into(),
            vv10_grid_level: 0,
            grid_level: 0,
            grid_weight_cutoff: 0.0,
            ..Default::default()
        };
        let mut setup = prepare_molecular(&[1; 4], &positions, &method).unwrap();
        if setup.grid.is_none() {
            setup.grid = Some(
                MolecularGrid::build(
                    &setup.molecule,
                    &GridOptions {
                        level: 0,
                        weight_cutoff: 0.0,
                        ..Default::default()
                    },
                )
                .unwrap(),
            );
        }
        let reference = run_molecular_reference(&setup).unwrap();
        assert!(reference.scf.converged);
        let config = ResponseSolverConfig {
            tolerance: 1e-12,
            ..Default::default()
        };
        let coefficients = Pt2Coefficients {
            same_spin: 0.31,
            opposite_spin: 0.47,
        };
        let target_coefficient = 0.37;
        let target_parameters = Vv10Params::WB97XV;
        let target = nonlocal.then(|| {
            FieldEnergyModel::new(&setup.functional)
                .with_vv10(target_coefficient, target_parameters)
                .unwrap()
        });
        let nonlocal_grid = nonlocal.then(|| setup.options.nonlocal_grid(&setup.molecule).unwrap());
        let analytic = correlated_spin_spin_couplings_for_model(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            target,
            setup.grid.as_ref(),
            &reference.scf,
            &setup.options,
            coefficients,
            &[0, 2],
            false,
            2048,
            &config,
        )
        .unwrap();
        assert!(analytic.response.converged);
        println!(
            "{xc} J raw residual {:.12e}",
            analytic.response.value.residual
        );
        let parent = (!nonlocal).then(|| {
            spin_spin_couplings(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                &reference.scf,
                &setup.options,
                &[0, 2],
                false,
                &config,
            )
            .unwrap()
        });
        let channels = [
            &analytic.response.value,
            &analytic.parent,
            &analytic.target,
            &analytic.correlation,
            &analytic.same_spin_alpha,
            &analytic.opposite_spin,
            &analytic.same_spin_beta,
            &analytic.parent_nonlocal,
            &analytic.target_nonlocal,
        ];
        let selected_channels: &[usize] = if nonlocal {
            &[0, 1, 2, 3, 4, 5, 6, 7, 8]
        } else {
            &[0, 1, 4, 5, 6]
        };
        let alpha4 = FINE_STRUCTURE.powi(4);
        let mut parent_defect = 0.0_f64;
        // Every ordinary term uses the same current-invariant model.
        for a in 0..3 {
            for b in 0..3 {
                if let Some(parent) = &parent {
                    for (ordinary, correlated) in [
                        (
                            parent.diamagnetic[0][a][b],
                            analytic.parent.diamagnetic[0][a][b],
                        ),
                        (
                            parent.fermi_contact[0][a][b],
                            analytic.parent.fermi_contact[0][a][b],
                        ),
                        (
                            parent.spin_dipole[0][a][b],
                            analytic.parent.spin_dipole[0][a][b],
                        ),
                    ] {
                        parent_defect = parent_defect.max((ordinary - correlated).abs() / alpha4);
                    }
                    {
                        parent_defect = parent_defect.max(
                            (parent.paramagnetic[0][a][b] - analytic.parent.paramagnetic[0][a][b])
                                .abs()
                                / alpha4,
                        );
                    }
                }
                assert!(
                    (analytic.response.value.total[0][a][b]
                        - analytic.target.total[0][a][b]
                        - analytic.correlation.total[0][a][b])
                        .abs()
                        / alpha4
                        < 1e-10
                );
            }
        }
        if parent.is_some() {
            println!("{xc} J full parent term limit {parent_defect:.12e} (K/alpha^4)");
            assert!(parent_defect < 1e-8);
        }
        let sites = nuclear_magnetic_sites(&setup.molecule, &setup.basis, &[0, 2], false).unwrap();
        let n = setup.basis.nao_sph;
        let blank = || [ComplexMatrix::zeros(n, n), ComplexMatrix::zeros(n, n)];
        let spin = |operator: &Matrix, scale: f64| {
            let mut h = blank();
            for (s, matrix) in h.iter_mut().enumerate() {
                for (v, a) in matrix.re.as_mut_slice().iter_mut().zip(operator.as_slice()) {
                    *v = 0.5 * scale * a * if s == 0 { 1.0 } else { -1.0 };
                }
            }
            h
        };
        let orbital = |operator: &Matrix| {
            let mut h = blank();
            for matrix in &mut h {
                for (v, a) in matrix.im.as_mut_slice().iter_mut().zip(operator.as_slice()) {
                    *v = -*a;
                }
            }
            h
        };
        let raw = repulsion_series(
            &setup.basis,
            &FieldPath {
                electric: [0.0; 3],
                magnetic: [0.0; 3],
                origin: [0.0; 3],
                london: false,
                order: 0,
            },
            None,
        )
        .unwrap();
        let g: Vec<_> = raw[0]
            .0
            .iter()
            .zip(&raw[0].1)
            .map(|(&re, &im)| Complex::new(re, im))
            .collect();
        let mut open = reference.scf.clone();
        open.unrestricted = true;
        let mut options = setup.options.clone();
        options.reference = dft_rs::scf::Reference::Uhf;
        let finite = |h: &[ComplexMatrix; 2]| {
            let state = magnetic_field_scf_with_core(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                &open,
                &options,
                &MagneticFieldOptions {
                    field: [0.0; 3],
                    spin_zeeman: false,
                    series_order: Some(2),
                    series_tolerance: 1e-15,
                },
                h,
            )
            .unwrap();
            let e = complex_pt2::finite_channels(
                n,
                &g,
                &state.coefficients,
                &state.orbital_energies,
                [state.n_alpha, state.n_beta],
            );
            let target_nlc = if let Some(grid) = &nonlocal_grid {
                let shift = dft_rs::Vec3::new(state.shift[0], state.shift[1], state.shift[2]);
                let mut shifted_basis = setup.basis.clone();
                for shell in &mut shifted_basis.shells {
                    shell.center += shift;
                }
                let mut shifted_grid = grid.clone();
                for point in &mut shifted_grid.points {
                    *point += shift;
                }
                let parent_nlc = nonlocal_complex_oracle::energy(
                    &shifted_basis,
                    &shifted_grid,
                    [0.0; 3],
                    &state.coefficients,
                    [state.n_alpha, state.n_beta],
                    Vv10Params::ORIGINAL,
                );
                assert!((parent_nlc - state.e_nlc).abs() < 1e-12);
                target_coefficient
                    * nonlocal_complex_oracle::energy(
                        &shifted_basis,
                        &shifted_grid,
                        [0.0; 3],
                        &state.coefficients,
                        [state.n_alpha, state.n_beta],
                        target_parameters,
                    )
            } else {
                0.0
            };
            let target_energy = state.energy - state.e_nlc + target_nlc;
            let correlation =
                coefficients.same_spin * (e[0] + e[2]) + coefficients.opposite_spin * e[1];
            [
                target_energy + correlation,
                state.energy,
                target_energy,
                correlation,
                e[0],
                e[1],
                e[2],
                state.e_nlc,
                target_nlc,
            ]
        };
        let finite_mixed = |left: &[ComplexMatrix; 2],
                            right: &[ComplexMatrix; 2],
                            h: f64,
                            stencil: stencils::Stencil| {
            // Nuclear sources have widely different physical magnitudes.
            // Normalize validation directions, then restore both factors
            // exactly. This changes only finite-difference coordinates,
            // not the production operator or the fit/error thresholds.
            let norm = |source: &[ComplexMatrix; 2]| {
                source
                    .iter()
                    .flat_map(|m| m.re.as_slice().iter().chain(m.im.as_slice()))
                    .map(|v| v.abs())
                    .fold(0.0_f64, f64::max)
            };
            let left_norm = norm(left);
            let right_norm = norm(right);
            assert!(left_norm > 0.0 && right_norm > 0.0);
            let at = |x: f64, y: f64| {
                let mut core = blank();
                for s in 0..2 {
                    for (v, (a, b)) in core[s]
                        .re
                        .as_mut_slice()
                        .iter_mut()
                        .zip(left[s].re.as_slice().iter().zip(right[s].re.as_slice()))
                    {
                        *v = x * a / left_norm + y * b / right_norm;
                    }
                    for (v, (a, b)) in core[s]
                        .im
                        .as_mut_slice()
                        .iter_mut()
                        .zip(left[s].im.as_slice().iter().zip(right[s].im.as_slice()))
                    {
                        *v = x * a / left_norm + y * b / right_norm;
                    }
                }
                finite(&core)
            };
            let along = |x| {
                let plus = at(x, h);
                let minus = at(x, -h);
                let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                    (at(x, 2.0 * h), at(x, -2.0 * h))
                } else {
                    ([0.0; 9], [0.0; 9])
                };
                std::array::from_fn::<_, 9, _>(|c| {
                    stencil.first(plus[c], minus[c], plus2[c], minus2[c], h)
                })
            };
            let plus = along(h);
            let minus = along(-h);
            let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                (along(2.0 * h), along(-2.0 * h))
            } else {
                ([0.0; 9], [0.0; 9])
            };
            std::array::from_fn::<_, 9, _>(|c| {
                left_norm * right_norm * stencil.first(plus[c], minus[c], plus2[c], minus2[c], h)
            })
        };
        let stages = [
            ([0.008, 0.006, 0.004, 0.003], stencils::Stencil::Second),
            ([0.002, 0.0015, 0.001, 0.00075], stencils::Stencil::Second),
            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Second),
            ([0.04, 0.03, 0.02, 0.015], stencils::Stencil::Fourth),
            ([0.02, 0.0175, 0.015, 0.0125], stencils::Stencil::Fourth),
            ([0.006, 0.005, 0.004, 0.003], stencils::Stencil::Second),
        ];
        let mut failures = Vec::new();
        let names = |part: &str, a, b| {
            let labels = [
                "total",
                "parent",
                "target",
                "correlation",
                "SS-alpha",
                "OS",
                "SS-beta",
                "parent-NLC",
                "target-NLC",
            ];
            selected_channels
                .iter()
                .map(|&c| format!("{xc} NLC={nonlocal} J {part} {} ({a},{b})", labels[c]))
                .collect::<Vec<_>>()
        };
        let fc_scale = 0.5 * ELECTRON_G_FACTOR * 8.0 * std::f64::consts::PI / 3.0;
        let fc_left = spin(&sites[0].contact, fc_scale);
        let fc_right = spin(&sites[1].contact, fc_scale);
        stencils::fit_components_staged(
            &names("FC", 0, 0),
            &stages,
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; selected_channels.len()];
                for (k, &h) in steps.iter().enumerate() {
                    let values = finite_mixed(&fc_left, &fc_right, h, stencil);
                    for (index, &c) in selected_channels.iter().enumerate() {
                        errors[index][k] =
                            (values[c] - channels[c].fermi_contact[0][0][0] / alpha4).abs();
                    }
                }
                errors
            },
            &mut failures,
        );
        for a in 0..3 {
            for b in 0..3 {
                let mut dso = Matrix::zeros(n, n);
                let grid = setup.grid.as_ref().unwrap();
                // Independent vector-potential dot product, rather than the
                // production index formula or relaxed-density expectation.
                for (start, points) in grid.points.chunks(256).enumerate() {
                    let ao = eval_aos(&setup.basis, points, false);
                    for (i, p) in points.iter().enumerate() {
                        let dm = *p - setup.molecule.atoms[0].position;
                        let dn = *p - setup.molecule.atoms[2].position;
                        if dm.norm() == 0.0 || dn.norm() == 0.0 {
                            continue;
                        }
                        let mut ea = [0.0; 3];
                        ea[a] = 1.0;
                        let mut eb = [0.0; 3];
                        eb[b] = 1.0;
                        let am = dft_rs::Vec3::new(ea[0], ea[1], ea[2]).cross(dm)
                            * (1.0 / dm.norm().powi(3));
                        let an = dft_rs::Vec3::new(eb[0], eb[1], eb[2]).cross(dn)
                            * (1.0 / dn.norm().powi(3));
                        let w = grid.weights[start * 256 + i] * am.dot(an);
                        for mu in 0..n {
                            for nu in 0..n {
                                dso[(mu, nu)] += w * ao.phi[(i, mu)] * ao.phi[(i, nu)];
                            }
                        }
                    }
                }
                stencils::fit_components_staged(
                    &names("all-terms", a, b)
                        .into_iter()
                        .chain(names("PSO", a, b))
                        .chain(names("SD", a, b))
                        .chain(names("DSO", a, b))
                        .collect::<Vec<_>>(),
                    &stages,
                    |steps, stencil| {
                        let mut errors = vec![[0.0; 4]; 4 * selected_channels.len()];
                        for (k, &h) in steps.iter().enumerate() {
                            let pso = finite_mixed(
                                &orbital(&sites[0].orbital[a]),
                                &orbital(&sites[1].orbital[b]),
                                h,
                                stencil,
                            );
                            let fc = if a == b {
                                finite_mixed(&fc_left, &fc_right, h, stencil)
                            } else {
                                [0.0; 9]
                            };
                            let mut sd = fc.map(|v| -v);
                            for axis in 0..3 {
                                let v = finite_mixed(
                                    &spin(
                                        &sites[0].triplet[3 * axis + a],
                                        1.0 / FINE_STRUCTURE.powi(2),
                                    ),
                                    &spin(
                                        &sites[1].triplet[3 * axis + b],
                                        1.0 / FINE_STRUCTURE.powi(2),
                                    ),
                                    h,
                                    stencil,
                                );
                                for c in 0..9 {
                                    sd[c] += v[c];
                                }
                            }
                            let dso_norm = dso
                                .as_slice()
                                .iter()
                                .map(|v| v.abs())
                                .fold(0.0_f64, f64::max);
                            assert!(dso_norm > 0.0);
                            let at = |x: f64| {
                                let mut core = blank();
                                for matrix in &mut core {
                                    for (v, a) in
                                        matrix.re.as_mut_slice().iter_mut().zip(dso.as_slice())
                                    {
                                        *v = x * a / dso_norm;
                                    }
                                }
                                finite(&core)
                            };
                            let plus = at(h);
                            let minus = at(-h);
                            let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                                (at(2.0 * h), at(-2.0 * h))
                            } else {
                                ([0.0; 9], [0.0; 9])
                            };
                            for (index, &c) in selected_channels.iter().enumerate() {
                                let dia = dso_norm
                                    * stencil.first(plus[c], minus[c], plus2[c], minus2[c], h);
                                let exact = channels[c];
                                for (part, computed, value) in [
                                    (0, fc[c] + sd[c] + pso[c] + dia, exact.total[0][a][b]),
                                    (1, pso[c], exact.paramagnetic[0][a][b]),
                                    (2, sd[c], exact.spin_dipole[0][a][b]),
                                    (3, dia, exact.diamagnetic[0][a][b]),
                                ] {
                                    errors[part * selected_channels.len() + index][k] =
                                        (computed - value / alpha4).abs();
                                }
                            }
                        }
                        errors
                    },
                    &mut failures,
                );
            }
        }
        components::assert_no_failures("complete post-SCF J tensors", &failures);
    }
}
