// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent ingredient and molecular-energy oracles for the meta-GGA
//! double hybrids, including four-step first through fourth derivatives.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::driver::method::{prepare_molecular, MolecularMethod};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::integrals::{eri_incore, kinetic_matrix, nuclear_matrix, CoulombKernel};
use dft_rs::linalg::Matrix;
use dft_rs::properties::pt2::{from_ao, OrbitalSpace};
use dft_rs::xc::double_hybrid::DoubleHybridFunctional;
use dft_rs::xc::{b97, dh_components, XcFunctional, XcTaylorInput};
use dft_rs::xc::{HybridParams, ResolvedFunctional};
use std::sync::Arc;

const BASE: [f64; 7] = [0.27, 0.19, 0.13, 0.02, 0.09, 0.42, 0.31];
const DIRECTION: [f64; 7] = [0.21, -0.08, 0.12, -0.02, 0.04, 0.17, -0.09];

fn parts() -> Vec<(&'static str, Box<dyn XcFunctional>)> {
    let mut result = vec![
        ("pwp_x", dh_components::pwpb95_x()),
        ("pwp_c", dh_components::pwpb95_c()),
    ];
    // Arc providers stay owned by the test below; use their public factories
    // through the same generic Taylor interface.
    result.shrink_to_fit();
    result
}
fn number(v: &Value, key: &str) -> f64 {
    v.get(key).unwrap().as_f64().unwrap()
}

/// Compare physical operators on a common independent orbital state. Two
/// independently optimized densities need not have separately equal kinetic
/// and potential terms, even when their variational total energies agree.
fn fixed_state_terms(system: &System, model: &str, name: &str) -> Vec<(&'static str, f64)> {
    let recipe = DoubleHybridFunctional::named(model).unwrap();
    let method = MolecularMethod {
        xc: Some("tpss".into()),
        grid_level: 6,
        dispersion: "none".into(),
        ..MolecularMethod::default()
    };
    let setup = prepare_molecular(&system.numbers, &system.positions, &method).unwrap();
    let n = setup.basis.nao_sph;
    let mut coeff = vec![0.; n * n];
    let mut eps = vec![0.; n];
    let mut occupied = 0;
    for line in include_str!("oracle_data/requested_functional_orbitals.csv")
        .lines()
        .skip(1)
    {
        let f: Vec<_> = line.split(',').collect();
        if f[0] != model || f[1] != name {
            continue;
        }
        let i: usize = f[2].parse().unwrap();
        let a: usize = f[3].parse().unwrap();
        coeff[n * i + a] = f[4].parse().unwrap();
        eps[a] = f[5].parse().unwrap();
        if i == 0 {
            occupied += f[6].parse::<usize>().unwrap();
        }
    }
    let c = Matrix::from_row_major(n, n, coeff.clone());
    let spin = c.leading_columns_gram(occupied, 1.0);
    let p = c.leading_columns_gram(occupied, 2.0);
    let t = kinetic_matrix(&setup.basis);
    let v = nuclear_matrix(&setup.basis, &setup.molecule, None);
    let one = t.frobenius_dot(&p) + v.frobenius_dot(&p);
    let eri = eri_incore(&setup.basis, CoulombKernel::Coulomb, 2048).unwrap();
    let mut tensor = Vec::new();
    let mut hartree = 0.;
    let mut exchange = 0.;
    let functional = recipe.target.as_ref().unwrap_or(&recipe.parent);
    let lr = (functional.hybrid.clr != 0.0).then(|| {
        eri_incore(
            &setup.basis,
            CoulombKernel::Erf {
                omega: functional.hybrid.omega,
            },
            2048,
        )
        .unwrap()
    });
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for l in 0..n {
                    let g = eri.get(i, j, k, l);
                    tensor.push(g);
                    hartree += 0.5 * p[(i, j)] * p[(k, l)] * g;
                    let gx = functional.hybrid.cx * g
                        + lr.as_ref()
                            .map_or(0., |e| functional.hybrid.clr * e.get(i, j, k, l));
                    exchange -= spin[(i, k)] * spin[(j, l)] * gx;
                }
            }
        }
    }
    let mut result = vec![("one", one), ("hartree", hartree), ("exchange", exchange)];
    for ((weight, part), key) in functional.parts.iter().zip(["part0", "part1", "part2"]) {
        let single = ResolvedFunctional::custom(
            "independent-state ingredient",
            vec![(*weight, Arc::clone(part))],
            HybridParams::PURE,
            part.family(),
        );
        let energy = dft_rs::ks::xc_contribution(
            &single,
            &setup.basis,
            setup.grid.as_ref().unwrap(),
            [&spin, &spin],
        )
        .unwrap()
        .exc;
        result.push((key, energy));
    }
    let grid = dft_rs::grid::MolecularGrid::build(
        &setup.molecule,
        &dft_rs::grid::GridOptions {
            level: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let parent_nlc = recipe.reference_vv10.map_or(0., |params| {
        dft_rs::nlc::vv10_energy(&setup.basis, &grid, &p, params).unwrap()
    });
    let target_nlc = recipe.target_vv10.map_or(parent_nlc, |(scale, params)| {
        scale * dft_rs::nlc::vv10_energy(&setup.basis, &grid, &p, params).unwrap()
    });
    result.push(("parent_nlc", parent_nlc));
    result.push(("nlc", target_nlc));
    let orbitals = std::array::from_fn(|_| OrbitalSpace {
        energies: eps.clone(),
        coefficients: coeff.clone(),
        occupied,
    });
    let pt2 = from_ao(n, &tensor, &orbitals)
        .unwrap()
        .solve(recipe.correlation)
        .unwrap();
    result.push(("same_spin", pt2.same_spin_alpha + pt2.same_spin_beta));
    result.push(("opposite_spin", pt2.opposite_spin));
    result
}

fn check_part(name: &str, part: &dyn XcFunctional) {
    let data = include_str!("oracle_data/requested_functional_points.csv");
    let header: Vec<_> = data.lines().next().unwrap().split(',').collect();
    let column = header.iter().position(|h| *h == name).unwrap();
    let at = |t: f64, column: usize| -> f64 {
        data.lines()
            .skip(1)
            .find_map(|line| {
                let fields: Vec<_> = line.split(',').collect();
                let x: f64 = fields[0].parse().unwrap();
                ((x - t).abs() < 1e-14).then(|| fields[column].parse().unwrap())
            })
            .expect("frozen oracle point")
    };
    let oracle = |t: f64| at(t, column);
    let coefficients = part
        .taylor_energy(&XcTaylorInput::linear(BASE, DIRECTION, 4).unwrap())
        .unwrap();
    assert!(
        (coefficients[0] - oracle(0.0)).abs() < 1e-12,
        "{name} energy {} {}",
        coefficients[0],
        oracle(0.0)
    );
    for (order, factorial) in [(1, 1.0), (2, 2.0), (3, 6.0), (4, 24.0)] {
        let exact = coefficients[order] * factorial;
        let independent_hessian = name == "pwp_c" && order == 4;
        let estimate = |h: f64| {
            if independent_hessian {
                let index = header.iter().position(|x| *x == "pwp_c_second").unwrap();
                assert!((at(0.0, index) - 2.0 * coefficients[2]).abs() < 1e-12);
                (at(h, index) - 2.0 * at(0.0, index) + at(-h, index)) / h.powi(2)
            } else {
                match order {
                    1 => (oracle(h) - oracle(-h)) / (2. * h),
                    2 => (oracle(h) - 2. * oracle(0.) + oracle(-h)) / h.powi(2),
                    3 => {
                        (oracle(2. * h) - 2. * oracle(h) + 2. * oracle(-h) - oracle(-2. * h))
                            / (2. * h.powi(3))
                    }
                    4 => {
                        (oracle(2. * h) - 4. * oracle(h) + 6. * oracle(0.) - 4. * oracle(-h)
                            + oracle(-2. * h))
                            / h.powi(4)
                    }
                    _ => unreachable!(),
                }
            }
        };
        // The initial [0.32,0.16,0.08,0.04] PW-exchange third derivative
        // had p=1.713018998: its coarsest step was not asymptotic. Retain
        // those oracle points and document that failed interval separately.
        // B95 has a very small sixth derivative on this path. The energy
        // fourth-difference interval had p=3.049563606, mixing h^2 and h^4
        // errors. An independent analytic-Hessian stencil resolves smaller
        // steps without fourth differences amplifying energy roundoff.
        let steps = if independent_hessian {
            [0.02, 0.01, 0.005, 0.0025]
        } else {
            [0.16, 0.08, 0.04, 0.02]
        };
        let errors: Vec<_> = steps.iter().map(|h| (estimate(*h) - exact).abs()).collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let richardson = ((4. * estimate(steps[3]) - estimate(steps[2])) / 3. - exact).abs();
        println!("{name} order={order} analytic={exact:0.12e} steps={steps:?} errors={errors:?} p={:0.9} R2={:0.12} Richardson={richardson:0.12e}",fit.exponent,fit.r_squared);
        assert!(
            (fit.exponent - 2.).abs() < 0.2 && fit.r_squared > 0.99,
            "{name} order {order}"
        );
        assert!(richardson < 1e-5, "{name} order {order}");
    }
}

#[test]
fn independently_evaluated_components_and_fourth_order_paths() {
    for (name, part) in parts() {
        check_part(name, &*part);
    }
    for ((_, part), name) in b97::wb97m2_parts()
        .iter()
        .zip(["omega_x", "omega_ss", "omega_os"])
    {
        check_part(name, &**part);
    }
}

#[test]
fn b95_one_spin_one_orbital_limit_and_exchange_uniform_limit() {
    assert!(dft_rs::xc::double_hybrid::DoubleHybridFunctional::named("ωB97M(2)").is_some());
    let rho: f64 = 0.3;
    let sigma: f64 = 0.2;
    let values = [rho, 0., sigma, 0., 0., sigma / (8. * rho), 0.];
    let result = dh_components::pwpb95_c()
        .taylor_energy(&XcTaylorInput::linear(values, [0.; 7], 0).unwrap())
        .unwrap();
    assert!(result[0].abs() < 1e-17);
    let values = [rho, rho, 0., 0., 0., 0.5, 0.5];
    let result = dh_components::pwpb95_x()
        .taylor_energy(&XcTaylorInput::linear(values, [0.; 7], 0).unwrap())
        .unwrap();
    assert!((result[0] + 2. * 0.9305257363491001 * rho.powf(4. / 3.)).abs() < 1e-15);
}

#[test]
fn requested_models_match_independent_molecular_terms_and_spin_components() {
    let data = include_str!("oracle_data/requested_functional_energy.csv");
    let header: Vec<_> = data.lines().next().unwrap().split(',').collect();
    for line in data.lines().skip(1) {
        let fields: Vec<_> = line.split(',').collect();
        let expected = |key: &str| {
            fields[header.iter().position(|h| *h == key).unwrap()]
                .parse::<f64>()
                .unwrap()
        };
        let model = fields[0];
        let name = fields[1];
        let system = if name == "hydrogen" {
            System::molecule(vec![1, 1], vec![[0., 0., -0.7], [0., 0., 0.7]])
        } else {
            System::molecule(
                vec![8, 1, 1],
                vec![
                    [0., 0., 0.2217],
                    [0., 1.4309, -0.8867],
                    [0., -1.4309, -0.8867],
                ],
            )
        };
        let controls = Controls::from_value(
            &Value::object()
                .with("xc", model)
                .with("basis", "sto-3g")
                .with("grid_level", 6usize)
                .with("vv10_grid_level", 1usize)
                .with("conv_tol", 1e-13)
                .with("conv_tol_grad", 1e-10)
                .with("conv_tol_dp", 1e-12),
        )
        .unwrap();
        let report = run("energy", &system, &controls).unwrap();
        // Total energies and PT2 test the independently optimized reference.
        // All individual operators are additionally checked on the shared
        // frozen orbital result, at the same unchanged term threshold.
        for (key, actual) in [
            ("parent", number(&report, "orbital_reference_energy")),
            ("target", number(&report, "reference_energy")),
            ("energy", number(&report, "energy")),
            ("dispersion", number(&report, "model_dispersion_energy")),
            (
                "same_spin",
                number(&report, "pt2_same_spin_alpha") + number(&report, "pt2_same_spin_beta"),
            ),
            ("opposite_spin", number(&report, "pt2_opposite_spin")),
        ] {
            let defect = (actual - expected(key)).abs();
            println!(
                "{model} {name} {key} actual={actual:0.15e} oracle={:0.15e} defect={defect:0.12e}",
                expected(key)
            );
            assert!(defect < 1e-6, "{model} {name} {key}");
        }
        for (key, actual) in fixed_state_terms(&system, model, name) {
            let defect = (actual - expected(key)).abs();
            println!("{model} {name} fixed_state {key} actual={actual:.15e} oracle={:.15e} defect={defect:.12e}",expected(key));
            assert!(defect < 1e-6, "{model} {name} fixed-state {key}");
        }
        for key in [
            "reference_energy_sum_residual",
            "semilocal_component_sum_residual",
            "pt2_amplitude_residual",
        ] {
            println!("{model} {name} {key}={:0.12e}", number(&report, key));
            assert!(number(&report, key) < 1e-9);
        }
        if model == "wb97m2" {
            assert_eq!(number(&report, "target_vv10_b"), 10.);
            assert_eq!(number(&report, "target_vv10_scale"), 0.65904);
        }
    }
}
