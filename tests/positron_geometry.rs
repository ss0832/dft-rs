// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent SCF energy differences on a C1 dimer with a free Gaussian site.

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::driver::{Controls, System, Value, run};
use std::collections::BTreeMap;
use stencils::{Stencil, fit_components_staged};

const NUCLEI: usize = 4;
const SITES: usize = NUCLEI + 1;
const CART: usize = 3 * SITES;
const FIELD_ORIGIN: [f64; 3] = [0.21, -0.13, 0.31];
const NUMBERS: [u8; NUCLEI] = [1; NUCLEI];
const COORDINATES: [[f64; 3]; SITES] = [
    [0.13, -0.27, 0.19],
    [1.77, 0.36, -0.08],
    [2.44, 2.35, 1.20],
    [3.97, 2.88, 1.67],
    [0.98, 0.73, 0.24],
];

fn calculate(coordinates: &[[f64; 3]; SITES], order: usize) -> Value {
    calculate_with(coordinates, order, |_| {})
}

fn calculate_with(
    coordinates: &[[f64; 3]; SITES],
    order: usize,
    input: impl FnOnce(&mut Value),
) -> Value {
    let system = System::molecule(NUMBERS.to_vec(), coordinates[..NUCLEI].to_vec());
    let mut values = Value::object()
        .with("xc", "pbe")
        .with("reference", "rks")
        .with("basis", "sto-3g")
        .with("dispersion", "none")
        .with("grid_level", 0_usize)
        .with("grid_weight_cutoff", 0.0)
        .with("cpu_threads", 1_usize)
        .with("conv_tol", 1e-13)
        .with("conv_tol_grad", 1e-11)
        .with("conv_tol_dp", 1e-11)
        .with("species_tolerance", 1e-11)
        .with("species_max_iterations", 500_usize)
        .with("species_derivative_order", order)
        .with(
            "particles",
            vec![
                Value::object()
                    .with("kind", "positron")
                    .with("position", coordinates[NUCLEI])
                    .with("exponents", [0.18, 0.64])
                    .with("lmax", 0_usize)
                    .with("correlation", "drummond-2011"),
            ],
        );
    input(&mut values);
    let report = run(
        "multicomponent",
        &system,
        &Controls::from_value(&values).unwrap(),
    )
    .unwrap();
    assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
    report
}

fn array(record: &Value, name: &str) -> Vec<f64> {
    record.get(name).unwrap().numeric_array().unwrap().1
}

struct EnergyOracle<const DIMENSION: usize> {
    base: f64,
    displaced: BTreeMap<([i8; DIMENSION], u64), f64>,
    evaluate: fn([f64; DIMENSION]) -> f64,
}

impl<const DIMENSION: usize> EnergyOracle<DIMENSION> {
    fn energy(&mut self, h: f64, shifts: &[(usize, i8)]) -> f64 {
        let mut coordinate = [0_i8; DIMENSION];
        for &(i, step) in shifts {
            coordinate[i] += step;
        }
        *self
            .displaced
            .entry((coordinate, h.to_bits()))
            .or_insert_with(|| (self.evaluate)(coordinate.map(|step| f64::from(step) * h)))
    }

    fn second(&mut self, i: usize, j: usize, h: f64) -> f64 {
        if i == j {
            ((self.energy(h, &[(i, 1)]) - self.base) + (self.energy(h, &[(i, -1)]) - self.base))
                / (h * h)
        } else {
            let a = self.energy(h, &[(i, 1), (j, 1)]) - self.energy(h, &[(i, 1), (j, -1)]);
            let b = self.energy(h, &[(i, -1), (j, 1)]) - self.energy(h, &[(i, -1), (j, -1)]);
            (a - b) / (4.0 * h * h)
        }
    }

    fn derivative(&mut self, i: usize, j: Option<usize>, h: f64, stencil: Stencil) -> f64 {
        match j {
            None => {
                let (p, m) = if stencil == Stencil::Fourth {
                    (self.energy(h, &[(i, 2)]), self.energy(h, &[(i, -2)]))
                } else {
                    (0.0, 0.0)
                };
                stencil.first(
                    self.energy(h, &[(i, 1)]),
                    self.energy(h, &[(i, -1)]),
                    p,
                    m,
                    h,
                )
            }
            Some(j) => {
                let value = self.second(i, j, h);
                if stencil == Stencil::Fourth {
                    (4.0 * value - self.second(i, j, 2.0 * h)) / 3.0
                } else {
                    value
                }
            }
        }
    }
}

fn total_energy(report: &Value) -> f64 {
    report.get("energy").unwrap().as_f64().unwrap()
}

#[test]
fn added_positron_full_geometry_fits_all_independent_components() {
    c1_geometry::assert_c1(&NUMBERS, &COORDINATES[..NUCLEI]);
    let report = calculate(&COORDINATES, 2);
    let derivative = report.get("geometry_derivatives").unwrap();
    assert_eq!(array(derivative, "added_particle_coordinate_order"), [0.0]);
    assert_eq!(
        array(derivative, "hessian_shape"),
        [CART as f64, CART as f64]
    );
    let gradient = array(derivative, "gradient");
    let hessian = array(derivative, "hessian");
    let mut definitions: Vec<_> = (0..CART).map(|i| (i, None, gradient[i])).collect();
    definitions.extend((0..CART).flat_map(|i| {
        let hessian = &hessian;
        (0..=i).map(move |j| (i, Some(j), hessian[i * CART + j]))
    }));
    let names: Vec<_> = definitions
        .iter()
        .map(|&(i, j, _)| match j {
            None => format!("positron geometry gradient component {i}"),
            Some(j) => format!("positron geometry Hessian component {i},{j}"),
        })
        .collect();
    println!(
        "C1 H4 with positron E {:.16e} gradient {:?} added-center diagonal {:?} reference energy errors {:?}",
        report.get("energy").unwrap().as_f64().unwrap(),
        gradient,
        (3 * NUCLEI..CART)
            .map(|a| hessian[a * CART + a])
            .collect::<Vec<_>>(),
        array(derivative, "reference_energy_errors")
    );
    let mut oracle = EnergyOracle {
        base: total_energy(&calculate(&COORDINATES, 0)),
        displaced: BTreeMap::new(),
        evaluate: |displacement: [f64; CART]| {
            let mut coordinates = COORDINATES;
            for (i, value) in displacement.into_iter().enumerate() {
                coordinates[i / 3][i % 3] += value;
            }
            total_energy(&calculate(&coordinates, 0))
        },
    };
    let mut failures = Vec::new();
    fit_components_staged(
        &names,
        &[
            ([0.036, 0.024, 0.016, 0.012], Stencil::Second),
            ([0.018, 0.012, 0.008, 0.006], Stencil::Second),
            ([0.048, 0.036, 0.024, 0.016], Stencil::Fourth),
        ],
        |steps, stencil| {
            definitions
                .iter()
                .map(|&(i, j, value)| {
                    steps.map(|h| (oracle.derivative(i, j, h, stencil) - value).abs())
                })
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures("positron Cartesian derivatives", &failures);
    let mut max_defects = [0.0_f64; 2];
    for &(i, j, value) in &definitions {
        let estimate = oracle.derivative(i, j, 0.006, Stencil::Fourth);
        let defect = (estimate - value).abs();
        let order = usize::from(j.is_some());
        max_defects[order] = max_defects[order].max(defect);
        assert!(defect < [5e-8, 2e-7][order], "{i},{j:?} error {defect:.6e}");
    }
    for axis in 0..3 {
        let translation: f64 = (axis..CART).step_by(3).map(|i| gradient[i]).sum();
        assert!(translation.abs() < 1e-9);
        for i in 0..CART {
            let sum: f64 = (axis..CART).step_by(3).map(|j| hessian[i * CART + j]).sum();
            assert!(sum.abs() < 1e-8);
        }
    }
    println!(
        "independent fourth-order absolute checks {max_defects:?}; SCF calls {}",
        oracle.displaced.len()
    );
}

fn field_report(displacement: [f64; 8], response: bool) -> Value {
    let bias = [
        0.0005, -0.0007, 0.0004, 0.0008, -0.0004, 0.0005, -0.0007, 0.0006,
    ];
    let field: [f64; 8] = std::array::from_fn(|i| bias[i] + displacement[i]);
    calculate_with(&COORDINATES, 0, |input| {
        input.set("field", field[..3].to_vec());
        input.set("species_field_gradient", field[3..].to_vec());
        input.set("origin", FIELD_ORIGIN);
        input.set("species_polarizability", response);
        input.set("species_quadrupole_polarizability", response);
    })
}

#[test]
fn biased_positron_quadrupole_matches_every_energy_derivative() {
    c1_geometry::assert_c1(&NUMBERS, &COORDINATES[..NUCLEI]);
    let base = field_report([0.0; 8], true);
    let quadrupole = base.get("quadrupole").unwrap();
    let polarizability = base.get("polarizability").unwrap();
    let qq = array(quadrupole, "polarizability");
    let qd = array(quadrupole, "field_conjugate_moment_from_field");
    let dd = array(polarizability, "tensor");
    let moment = array(quadrupole, "moment");
    let dipole = array(quadrupole, "dipole_moment");
    let raw: [f64; 8] = array(quadrupole, "response_residuals").try_into().unwrap();
    assert!(raw.iter().all(|r| *r < 1e-9));
    // The homogeneous field couples at laboratory zero. Translate the
    // dipole from the explicit quadrupole origin for the net +1 system.
    let mut definitions: Vec<_> = dipole
        .iter()
        .zip(FIELD_ORIGIN)
        .enumerate()
        .map(|(axis, (&moment, origin))| (axis, None, -moment - origin))
        .collect();
    definitions.extend(
        moment
            .iter()
            .enumerate()
            .map(|(axis, &moment)| (axis + 3, None, -moment / 3.0)),
    );
    for a in 0..8 {
        for b in 0..=a {
            let response = match (a < 3, b < 3) {
                (true, _) => dd[3 * a + b],
                (false, true) => qd[3 * (a - 3) + b],
                (false, false) => qq[5 * (a - 3) + (b - 3)] / 3.0,
            };
            definitions.push((a, Some(b), -response));
        }
    }
    let names: Vec<_> = definitions
        .iter()
        .map(|&(a, b, _)| match b {
            None => format!("joint electric moment {a}"),
            Some(b) => format!("joint electric Hessian {a},{b}"),
        })
        .collect();
    println!(
        "biased C1 H4 positron E {:.16e}; Buckingham moments {moment:?}; C00 {:.12e}; raw responses {raw:?}",
        total_energy(&base),
        qq[0],
    );
    let mut oracle = EnergyOracle {
        base: total_energy(&base),
        displaced: BTreeMap::new(),
        evaluate: |displacement| total_energy(&field_report(displacement, false)),
    };
    let mut failures = Vec::new();
    fit_components_staged(
        &names,
        &[
            ([0.004, 0.003, 0.002, 0.0015], Stencil::Second),
            ([0.002, 0.0015, 0.001, 0.00075], Stencil::Second),
            ([0.006, 0.004, 0.003, 0.002], Stencil::Fourth),
        ],
        |steps, stencil| {
            definitions
                .iter()
                .map(|&(a, b, exact)| {
                    steps.map(|h| (oracle.derivative(a, b, h, stencil) - exact).abs())
                })
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures("joint electric moments and response", &failures);
    let mut defects = [0.0_f64; 2];
    for &(a, b, exact) in &definitions {
        let error = (oracle.derivative(a, b, 0.00075, Stencil::Fourth) - exact).abs();
        let order = usize::from(b.is_some());
        defects[order] = defects[order].max(error);
        assert!(error < [2e-7, 5e-6][order], "{a},{b:?} error {error:.6e}");
    }
    println!("independent fourth-order absolute defects {defects:?}");
}
