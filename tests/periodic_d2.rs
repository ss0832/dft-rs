// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::dispersion::Dispersion;
use dft_rs::math::Vec3;
use dft_rs::periodic::dispersion::{
    periodic_d3, periodic_dispersion_force_constant_tensor, periodic_dispersion_hessian,
    PeriodicDispersion, ThreeBody,
};
use dft_rs::periodic::Lattice;
use stencils::Stencil;

const NUMBERS: [u8; 4] = [7, 1, 1, 1];
const POINTS: [[f64; 3]; 4] = [
    [0.13, -0.27, 0.19],
    [1.78, 0.34, -0.08],
    [-0.49, 1.65, 0.41],
    [0.31, -0.78, 1.91],
];
const ROWS: [[f64; 3]; 3] = [[9.1, 0.2, 0.1], [1.2, 8.4, -0.1], [0.6, 1.3, 10.2]];
const CUTOFF: f64 = 12.37;
const DIMENSION: usize = 3 * NUMBERS.len();

fn vector([x, y, z]: [f64; 3]) -> Vec3 {
    Vec3::new(x, y, z)
}

#[derive(Clone, Copy)]
struct Pair {
    i: usize,
    j: usize,
    translation: [i32; 3],
    c6: f64,
    radius: f64,
}

impl Pair {
    fn vector(self, points: &[[f64; 3]; 4], rows: &[[f64; 3]; 3]) -> [f64; 3] {
        std::array::from_fn(|axis| {
            points[self.j][axis] - points[self.i][axis]
                + (0..3)
                    .map(|i| f64::from(self.translation[i]) * rows[i][axis])
                    .sum::<f64>()
        })
    }

    // Independently differentiate the printed D2 pair formula through the
    // third Cartesian derivative. The test differences these lower orders.
    fn derivative(self, points: &[[f64; 3]; 4], rows: &[[f64; 3]; 3], axes: &[usize]) -> f64 {
        let sign = axes.iter().fold(1.0, |product, &axis| {
            product * (f64::from(axis / 3 == self.j) - f64::from(axis / 3 == self.i))
        });
        if sign == 0.0 {
            return 0.0;
        }
        let displacement = self.vector(points, rows);
        let r = vector(displacement).norm();
        let u = displacement.map(|x| x / r);
        let a = 20.0 / self.radius;
        let damping = 1.0 / (1.0 + (-a * (r - self.radius)).exp());
        let first = a * damping * (1.0 - damping);
        let second = a * first * (1.0 - 2.0 * damping);
        let third = a * a * first * (1.0 - 6.0 * damping * (1.0 - damping));
        let f1 = first - 6.0 * damping / r;
        let f2 = second - 12.0 * first / r + 42.0 * damping / r.powi(2);
        let f3 =
            third - 18.0 * second / r + 126.0 * first / r.powi(2) - 336.0 * damping / r.powi(3);
        let component = match axes {
            [] => damping,
            [a] => f1 * u[a % 3],
            [a, b] => {
                let (a, b) = (a % 3, b % 3);
                (f2 - f1 / r) * u[a] * u[b] + (f1 / r) * f64::from(a == b)
            }
            [a, b, c] => {
                let (a, b, c) = (a % 3, b % 3, c % 3);
                let t = (f2 - f1 / r) / r;
                (f3 - 3.0 * t) * u[a] * u[b] * u[c]
                    + t * (f64::from(a == b) * u[c]
                        + f64::from(a == c) * u[b]
                        + f64::from(b == c) * u[a])
            }
            _ => unreachable!(),
        };
        -0.5 * 0.75 * self.c6 * sign * component / r.powi(6)
    }
}

fn oracle_pairs() -> Vec<Pair> {
    let mut atoms = [(0.14_f64, 1.001_f64); NUMBERS.len()];
    atoms[0] = (1.23, 1.397);
    // Table 1 coefficients are J nm^6/mol and radii are Angstrom. Enumerate
    // an integer cube large enough for the fixed, skewed 12.37 Bohr sphere.
    let to_bohr = dft_rs::constants::ANGSTROM_TO_BOHR;
    let conversion = 1e3 * to_bohr.powi(6) / dft_rs::constants::HARTREE_TO_KJ_MOL;
    let mut pairs = Vec::new();
    for (i, (c, r)) in atoms.into_iter().enumerate() {
        for (j, (cp, rp)) in atoms.into_iter().enumerate() {
            for a in -3..=3 {
                for b in -3..=3 {
                    for c_index in -3..=3 {
                        let translation = [a, b, c_index];
                        if i == j && translation == [0; 3] {
                            continue;
                        }
                        let pair = Pair {
                            i,
                            j,
                            translation,
                            c6: (c * cp).sqrt() * conversion,
                            radius: (r + rp) * to_bohr,
                        };
                        let distance = vector(pair.vector(&POINTS, &ROWS)).norm();
                        assert!((distance - CUTOFF).abs() > 0.14);
                        if distance < CUTOFF {
                            pairs.push(pair);
                        }
                    }
                }
            }
        }
    }
    pairs
}

fn independent_axes(order: usize) -> Vec<Vec<usize>> {
    (0..order).fold(vec![Vec::new()], |lower, _| {
        lower
            .into_iter()
            .flat_map(|axes| {
                (axes.last().copied().unwrap_or(0)..DIMENSION).map(move |axis| {
                    let mut next = axes.clone();
                    next.push(axis);
                    next
                })
            })
            .collect()
    })
}

#[test]
fn periodic_d2_energy_and_full_gamma_cartesian_tensors_through_order_four() {
    c1_geometry::assert_c1(&NUMBERS, &POINTS);
    let [a, b, c] = ROWS.map(vector);
    let lattice = Lattice::new(a, b, c);
    let points = POINTS.map(vector);
    let model = PeriodicDispersion::from_model(&Dispersion::D2, "pbe", 7.0, CUTOFF)
        .unwrap()
        .unwrap();
    let result = periodic_d3(&NUMBERS, &points, &lattice, &model).unwrap();
    let hessian = periodic_dispersion_hessian(&NUMBERS, &points, &lattice, &model).unwrap();
    let cubic =
        periodic_dispersion_force_constant_tensor(&NUMBERS, &points, &lattice, &model, 3).unwrap();
    let quartic =
        periodic_dispersion_force_constant_tensor(&NUMBERS, &points, &lattice, &model, 4).unwrap();
    assert!(model.with_three_body(ThreeBody::default()).is_err());
    let pairs = oracle_pairs();
    let oracle = |points: &[[f64; 3]; 4], rows: &[[f64; 3]; 3], axes: &[usize]| {
        pairs
            .iter()
            .map(|p| p.derivative(points, rows, axes))
            .sum::<f64>()
    };
    let expected = oracle(&POINTS, &ROWS, &[]);
    assert!((result.energy - expected).abs() < 1e-14);
    assert_eq!(result.energy, cubic.energy);
    assert_eq!(result.energy, quartic.energy);
    assert!(result.coordination_numbers.iter().all(|v| *v == 0.0));
    assert!(cubic.acoustic_residual < 1e-11 && cubic.permutation_residual < 1e-11);
    assert!(quartic.acoustic_residual < 1e-11 && quartic.permutation_residual < 1e-11);
    println!("periodic D2 ammonia: E={:.12e} Eh oracle={expected:.12e} Eh images={} force={:?} strain={:?} cubic acoustic/permutation={:.4e}/{:.4e} quartic acoustic/permutation={:.4e}/{:.4e}",
        result.energy, pairs.len(), result.forces, result.strain_derivative,
        cubic.acoustic_residual, cubic.permutation_residual, quartic.acoustic_residual, quartic.permutation_residual);

    let stages = [
        ([0.008, 0.004, 0.002, 0.001], Stencil::Second),
        ([0.002, 0.001, 0.0005, 0.00025], Stencil::Second),
        ([0.016, 0.008, 0.004, 0.002], Stencil::Fourth),
    ];
    let mut failures = Vec::new();
    for order in 1..=4 {
        let exact_component = |labels: &[usize]| {
            let sequential = labels.iter().fold(0, |i, &a| i * DIMENSION + a);
            match labels {
                [a] => -result.forces[a / 3].get(a % 3),
                [a, b] => hessian[(*a, *b)],
                [_, _, _] => cubic.tensor[sequential],
                [_, _, _, _] => quartic.tensor[sequential],
                _ => unreachable!(),
            }
        };
        let (axes, structural_zeros): (Vec<_>, Vec<_>) =
            independent_axes(order).into_iter().partition(|labels| {
                labels
                    .windows(2)
                    .filter(|ab| ab[0] / 3 != ab[1] / 3)
                    .count()
                    < 2
            });
        // A pair potential has a zero mixed partial for any three distinct
        // atoms. Check the closed-form zeros directly at every displaced step.
        for labels in &structural_zeros {
            assert_eq!(exact_component(labels), 0.0, "order {order} {labels:?}");
            let (&axis, lower) = labels.split_last().unwrap();
            for h in stages[0].0 {
                let mut plus = POINTS;
                let mut minus = POINTS;
                plus[axis / 3][axis % 3] += h;
                minus[axis / 3][axis % 3] -= h;
                assert_eq!(oracle(&plus, &ROWS, lower), oracle(&minus, &ROWS, lower));
            }
        }
        println!(
            "periodic D2 order {order}: {} component fits, {} structural zeros",
            axes.len(),
            structural_zeros.len()
        );
        let names = axes
            .iter()
            .map(|labels| format!("D2 order {order} {labels:?}"))
            .collect::<Vec<_>>();
        stencils::fit_components_staged(
            &names,
            &stages,
            |steps, stencil| {
                axes.iter()
                    .map(|labels| {
                        let (&axis, lower) = labels.split_last().unwrap();
                        let analytic = exact_component(labels);
                        let shifted = |step: f64| {
                            let mut points = POINTS;
                            points[axis / 3][axis % 3] += step;
                            oracle(&points, &ROWS, lower)
                        };
                        steps.map(|h| {
                            let (plus2, minus2) = if stencil == Stencil::Fourth {
                                (shifted(2.0 * h), shifted(-2.0 * h))
                            } else {
                                (0.0, 0.0)
                            };
                            (stencil.first(shifted(h), shifted(-h), plus2, minus2, h) - analytic)
                                .abs()
                        })
                    })
                    .collect()
            },
            &mut failures,
        );
    }

    let strain_axes: Vec<_> = (0..3).flat_map(|a| (0..=a).map(move |b| (a, b))).collect();
    let names = strain_axes
        .iter()
        .map(|(a, b)| format!("D2 strain [{a},{b}]"))
        .collect::<Vec<_>>();
    let small_steps = [([0.002, 0.001, 0.0005, 0.00025], Stencil::Second)];
    stencils::fit_components_staged(
        &names,
        &small_steps,
        |steps, stencil| {
            strain_axes
                .iter()
                .map(|&(a, b)| {
                    let shifted = |eps: f64| {
                        let shear = |mut v: [f64; 3]| {
                            v[a] += eps * v[b];
                            v
                        };
                        oracle(&POINTS.map(shear), &ROWS.map(shear), &[])
                    };
                    steps.map(|h| {
                        (stencil.first(shifted(h), shifted(-h), 0.0, 0.0, h)
                            - result.strain_derivative[a][b])
                            .abs()
                    })
                })
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures("periodic D2 Cartesian and strain derivatives", &failures);
}
