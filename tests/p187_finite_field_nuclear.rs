// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independently displaced finite-field SCF energies/analytic forces against
//! production nuclear response coefficients. Differences are validation only.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::math::Vec3;
use dft_rs::response::{nuclear::magnetic_nuclear_taylor, ResponseSolverConfig};
use dft_rs::scf::magnetic::{
    magnetic_field_gradient, magnetic_field_scf_from_core, MagneticFieldOptions,
};
use dft_rs::scf::ScfOptions;
use dft_rs::{resolve, Atom, FunctionalSpec, Molecule};

fn accept(
    numbers: &[u8],
    positions: &[[f64; 3]],
    directions: &[[f64; 3]],
    model: &str,
    b: [f64; 3],
    e: [f64; 3],
) {
    accept_reference(numbers, positions, directions, model, b, e, 0, 1, false);
}

fn accept_reference(
    numbers: &[u8],
    positions: &[[f64; 3]],
    directions: &[[f64; 3]],
    model: &str,
    b: [f64; 3],
    e: [f64; 3],
    charge: i32,
    multiplicity: usize,
    spin_zeeman: bool,
) {
    let options = ScfOptions {
        e_tol: 1e-14,
        p_tol: 1e-13,
        conv_tol_grad: 1e-13,
        max_iter: 500,
        external_field: Some(Vec3::new(e[0], e[1], e[2])),
        ..Default::default()
    };
    let field = MagneticFieldOptions {
        field: b,
        spin_zeeman,
        series_order: Some(8),
        series_tolerance: 1e-14,
    };
    let functional = resolve(&FunctionalSpec::Combined(model.into())).unwrap();
    let grid_options = GridOptions {
        radial_override: Some(10),
        angular_override: Some(7),
        prune: false,
        weight_cutoff: 0.0,
        ..Default::default()
    };
    let prepare = |t: f64| {
        let mut molecule = Molecule::new(
            numbers
                .iter()
                .zip(positions)
                .zip(directions)
                .map(|((z, p), d)| Atom {
                    z: *z,
                    position: Vec3::new(p[0] + t * d[0], p[1] + t * d[1], p[2] + t * d[2]),
                })
                .collect(),
        );
        molecule.charge = charge;
        molecule.multiplicity = multiplicity;
        let basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
        let grid = (!functional.parts.is_empty())
            .then(|| MolecularGrid::build(&molecule, &grid_options).unwrap());
        let state = magnetic_field_scf_from_core(
            &molecule,
            &basis,
            &functional,
            grid.as_ref(),
            &options,
            &field,
        )
        .unwrap();
        assert!(state.converged && state.residual < 1e-12);
        if state.n_alpha + state.n_beta == 1
            && functional.parts.is_empty()
            && functional.hybrid.cx == 1.0
            && functional.hybrid.clr == 0.0
        {
            let core = dft_rs::periodic::BlochMatrix {
                re: state.core.re.clone(),
                im: state.core.im.clone(),
            };
            let overlap = dft_rs::periodic::BlochMatrix {
                re: state.overlap.re.clone(),
                im: state.overlap.im.clone(),
            };
            let bare =
                dft_rs::dfpt::orbitals::generalized_orbitals(&core, &overlap, 1e-12).unwrap();
            let electric = Vec3::new(e[0], e[1], e[2]);
            let shift = Vec3::new(state.shift[0], state.shift[1], state.shift[2]);
            let nuclear_field: f64 = molecule
                .atoms
                .iter()
                .map(|a| -(a.z as f64) * electric.dot(a.position))
                .sum();
            let zeeman = if spin_zeeman {
                0.25 * dft_rs::constants::ELECTRON_G_FACTOR
                    * (b.iter().map(|x| x * x).sum::<f64>()).sqrt()
                    * (state.n_alpha as f64 - state.n_beta as f64)
            } else {
                0.0
            };
            let oracle = bare.energies[0] + molecule.nuclear_repulsion() + nuclear_field
                - electric.dot(shift)
                + zeeman;
            println!("one-electron finite-field energy oracle t={t:.16e} charge={charge} spin_zeeman={spin_zeeman} defect={:.16e}",(state.energy-oracle).abs());
            assert!((state.energy - oracle).abs() < 2e-12);
            assert!(state.unrestricted);
        }
        (molecule, basis, grid, state)
    };
    let (molecule, basis, grid, state) = prepare(0.0);
    let analytic = magnetic_nuclear_taylor(
        &molecule,
        &basis,
        &functional,
        grid.as_ref(),
        &options,
        &field,
        &state,
        directions,
        4,
        &ResponseSolverConfig {
            tolerance: 1e-12,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(analytic.residuals.iter().all(|r| *r <= 1e-12));
    assert!(analytic.projector_defects.iter().all(|r| *r < 1e-10));
    assert!(analytic.stationarity_defects.iter().all(|r| *r < 1e-10));
    let mut cache = std::collections::BTreeMap::new();
    let mut point = |t: f64| {
        *cache.entry(t.to_bits()).or_insert_with(|| {
            let (molecule,basis,grid,state)=prepare(t);
            let gradient=magnetic_field_gradient(&molecule,&basis,&functional,grid.as_ref(),&field,&state).unwrap();
            let projected=gradient.iter().zip(directions).map(|(g,d)| (0..3).map(|a| g[a]*d[a]).sum::<f64>()).sum::<f64>();
            println!("finite-field nuclear model={model} charge={charge} multiplicity={multiplicity} spin_zeeman={spin_zeeman} atoms={} b={b:?} e={e:?} t={t:.16e} energy={:.16e} gradient={projected:.16e} reference_residual={:.16e} phase_tail={:.16e}", numbers.len(),state.energy,state.residual,state.series_tail);
            (state.energy,projected)
        })
    };
    let zero = point(0.0);
    let steps = [0.12, 0.09, 0.06, 0.03, 0.015, 0.0075];
    let mut estimates = Vec::new();
    for h in steps {
        let (a, b, c, d) = (point(h), point(-h), point(2.0 * h), point(-2.0 * h));
        estimates.push([
            (a.0 - b.0) / (2.0 * h),
            (a.1 - b.1) / (2.0 * h),
            (a.1 - 2.0 * zero.1 + b.1) / (h * h),
            (c.1 - 2.0 * a.1 + 2.0 * b.1 - d.1) / (2.0 * h * h * h),
        ]);
    }
    let mut results = Vec::new();
    for n in 1..=4 {
        let target = (1..=n).product::<usize>() as f64 * analytic.energy[n];
        let errors: Vec<_> = estimates
            .iter()
            .map(|v: &[f64; 4]| (v[n - 1] - target).abs())
            .collect();
        let coarse = convergence::error_power_fit(&steps[..4], &errors[..4]);
        let coarse_rich = ((4.0 * estimates[3][n - 1] - estimates[2][n - 1]) / 3.0 - target).abs();
        println!("finite-field nuclear coarse model={model} charge={charge} multiplicity={multiplicity} spin_zeeman={spin_zeeman} atoms={} b={b:?} e={e:?} order={n} p={:.9} R2={:.12} Richardson={coarse_rich:.16e}", numbers.len(),coarse.exponent,coarse.r_squared);
        let interval = if n == 4 { 2..6 } else { 0..4 };
        let fit = convergence::error_power_fit(&steps[interval.clone()], &errors[interval.clone()]);
        let fine = interval.end - 1;
        let rich =
            ((4.0 * estimates[fine][n - 1] - estimates[fine - 1][n - 1]) / 3.0 - target).abs();
        println!("finite-field nuclear fit model={model} charge={charge} multiplicity={multiplicity} spin_zeeman={spin_zeeman} atoms={} b={b:?} e={e:?} order={n} target={target:.16e} steps={steps:?} errors={errors:?} accepted_interval={:?} p={:.9} R2={:.12} Richardson={rich:.16e} response_residuals={:?}", numbers.len(),&steps[interval],fit.exponent,fit.r_squared,analytic.residuals);
        results.push((n, fit, rich));
    }
    for (n, fit, rich) in results {
        assert!(
            (fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.9999 && rich < 2e-6,
            "model={model} nuclear order={n}"
        );
    }
}

#[test]
fn unrestricted_charged_hf_joint_bias_with_spin_zeeman() {
    accept_reference(
        &[1, 1],
        &[[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
        &[[0.02, 0.0, -0.3], [0.02, 0.0, 0.3]],
        "hf",
        [0.0, 0.012, 0.0],
        [0.0007, 0.0005, -0.001],
        1,
        2,
        true,
    );
}

#[test]
fn molecular_hf_electric_bias() {
    let positions = [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]];
    let directions = [[0.02, 0.0, -0.3], [0.02, 0.0, 0.3]];
    accept(
        &[1, 1],
        &positions,
        &directions,
        "hf",
        [0.0; 3],
        [0.0007, 0.0005, -0.001],
    );
}

#[test]
fn molecular_hf_magnetic_bias() {
    accept(
        &[1, 1],
        &[[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
        &[[0.02, 0.0, -0.3], [0.02, 0.0, 0.3]],
        "hf",
        [0.0, 0.012, 0.0],
        [0.0; 3],
    );
}

#[test]
fn molecular_hf_joint_bias() {
    accept(
        &[1, 1],
        &[[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
        &[[0.02, 0.0, -0.3], [0.02, 0.0, 0.3]],
        "hf",
        [0.0, 0.012, 0.0],
        [0.0007, 0.0005, -0.001],
    );
}

fn semilocal(model: &str) {
    accept(
        &[1, 1],
        &[[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
        &[[0.02, 0.0, -0.3], [0.02, 0.0, 0.3]],
        model,
        [0.0, 0.012, 0.0],
        [0.0007, 0.0005, -0.001],
    );
}

#[test]
fn gga_moving_grid_nuclear_orders() {
    semilocal("pbe");
}

#[test]
fn meta_gga_moving_grid_nuclear_orders() {
    semilocal("tpss");
}

#[test]
fn hybrid_moving_grid_nuclear_orders() {
    semilocal("pbe0");
}

#[test]
fn water_hybrid_joint_field_nuclear_orders_have_independent_fits() {
    accept(
        &[8, 1, 1],
        &[[0.0, 0.0, 0.0], [1.43, 0.0, 1.1], [-1.43, 0.0, 1.1]],
        &[[0.0, 0.02, -0.1], [0.15, 0.02, 0.25], [-0.15, 0.02, 0.2]],
        "pbe0",
        [0.0, 0.0, 0.006],
        [0.0007, 0.0005, -0.001],
    );
}
