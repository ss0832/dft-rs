// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Common-orbital projector coefficients with retained summation remainders.
//! The same coefficients feed AO densities, stationarity and raw constraints.

use super::super::precision::{self, MatrixParts, Series};
use super::Classes;
use crate::linalg::{AccurateSum, Matrix};

#[derive(Clone)]
struct Coefficient {
    high: Matrix,
    low: Matrix,
}

impl Coefficient {
    fn from_matrix(high: Matrix) -> Self {
        let low = Matrix::zeros(high.rows, high.cols);
        Self { high, low }
    }

    fn zero(size: usize) -> Self {
        Self::from_matrix(Matrix::zeros(size, size))
    }

    fn rounded(&self) -> Matrix {
        let mut value = self.high.clone();
        for (v, low) in value.as_mut_slice().iter_mut().zip(self.low.as_slice()) {
            *v += low;
        }
        value
    }

    fn add(&mut self, source: &Self, sign: f64) {
        for i in 0..self.high.rows {
            for j in 0..self.high.cols {
                let mut sum = AccurateSum::default();
                sum.add_product(1.0, self.high[(i, j)]);
                sum.add_product(1.0, self.low[(i, j)]);
                sum.add_product(sign, source.high[(i, j)]);
                sum.add_product(sign, source.low[(i, j)]);
                (self.high[(i, j)], self.low[(i, j)]) = sum.parts();
            }
        }
    }

    fn set(&mut self, i: usize, j: usize, source: &Self, sign: f64) {
        self.high[(i, j)] = sign * source.high[(i, j)];
        self.low[(i, j)] = sign * source.low[(i, j)];
    }

    fn largest(&self) -> f64 {
        self.high
            .as_slice()
            .iter()
            .zip(self.low.as_slice())
            .map(|(a, b)| (a + b).abs())
            .fold(0.0, f64::max)
    }
}

fn product(a: &[Coefficient], b: &[Coefficient], n: usize) -> Coefficient {
    let mut value = Coefficient::from_matrix(Matrix::zeros(a[0].high.rows, b[0].high.cols));
    for i in 0..value.high.rows {
        for j in 0..value.high.cols {
            let mut sum = AccurateSum::default();
            for k in 0..=n {
                for p in 0..a[k].high.cols {
                    for x in [&a[k].high, &a[k].low] {
                        for y in [&b[n - k].high, &b[n - k].low] {
                            sum.add_product(x[(i, p)], y[(p, j)]);
                        }
                    }
                }
            }
            (value.high[(i, j)], value.low[(i, j)]) = sum.parts();
        }
    }
    value
}

#[derive(Clone)]
pub(super) struct Flag {
    closed: Vec<Coefficient>,
    open: Vec<Coefficient>,
    classes: Classes,
}

impl Flag {
    pub fn new(closed: Matrix, open: Matrix, classes: Classes) -> Self {
        Self {
            closed: vec![Coefficient::from_matrix(closed)],
            open: vec![Coefficient::from_matrix(open)],
            classes,
        }
    }

    pub fn advance(&mut self, n: usize) {
        let size = self.classes.total;
        let (lower_closed, lower_open, mixed) = if n > 1 {
            (
                product(&self.closed[1..n], &self.closed[1..n], n - 2),
                product(&self.open[1..n], &self.open[1..n], n - 2),
                product(&self.closed[1..n], &self.open[1..n], n - 2),
            )
        } else {
            (
                Coefficient::zero(size),
                Coefficient::zero(size),
                Coefficient::zero(size),
            )
        };
        let mut closed = Coefficient::zero(size);
        let mut open = Coefficient::zero(size);
        for i in 0..size {
            for j in 0..size {
                let (ci, cj) = (self.classes.class(i), self.classes.class(j));
                if ci == 0 && cj == 0 {
                    closed.set(i, j, &lower_closed, -1.0);
                } else if ci != 0 && cj != 0 {
                    closed.set(i, j, &lower_closed, 1.0);
                }
                if ci == 1 && cj == 1 {
                    open.set(i, j, &lower_open, -1.0);
                } else if ci != 1 && cj != 1 {
                    open.set(i, j, &lower_open, 1.0);
                } else if ci == 0 && cj == 1 {
                    open.set(i, j, &mixed, -1.0);
                    open.high[(j, i)] = open.high[(i, j)];
                    open.low[(j, i)] = open.low[(i, j)];
                }
            }
        }
        self.closed.push(closed);
        self.open.push(open);
    }

    pub fn rotate(&mut self, n: usize, changes: [Matrix; 2]) {
        let [closed, open] = changes;
        self.closed[n].add(&Coefficient::from_matrix(closed), 1.0);
        self.open[n].add(&Coefficient::from_matrix(open), 1.0);
    }

    fn alpha(&self) -> Vec<Coefficient> {
        self.closed
            .iter()
            .zip(&self.open)
            .map(|(c, o)| {
                let mut value = c.clone();
                value.add(o, 1.0);
                value
            })
            .collect()
    }

    pub fn stationarity(&self, fock: &[Series; 2], n: usize) -> MatrixParts {
        let mut value = MatrixParts::zeros(self.classes.total, self.classes.total);
        let alpha = self.alpha();
        for (f, density) in fock.iter().zip([&alpha, &self.closed]) {
            let density: Series = density
                .iter()
                .map(|d| MatrixParts {
                    high: d.high.clone(),
                    low: d.low.clone(),
                })
                .collect();
            value.add_scaled(&precision::commutator(f, &density, n), 1.0);
        }
        value
    }

    pub fn densities(&self, frame: &[MatrixParts], order: usize) -> [Series; 2] {
        let alpha = self.alpha();
        [&alpha, &self.closed].map(|p| {
            let density: Series = p
                .iter()
                .map(|d| MatrixParts {
                    high: d.high.clone(),
                    low: d.low.clone(),
                })
                .collect();
            precision::density(frame, &density, order)
        })
    }

    pub fn defect(&self, n: usize) -> f64 {
        let mut closed = product(&self.closed, &self.closed, n);
        let mut open = product(&self.open, &self.open, n);
        closed.add(&self.closed[n], -1.0);
        open.add(&self.open[n], -1.0);
        closed
            .largest()
            .max(open.largest())
            .max(product(&self.closed, &self.open, n).largest())
    }

    pub fn rounded(&self) -> (Vec<Matrix>, Vec<Matrix>) {
        (
            self.closed.iter().map(Coefficient::rounded).collect(),
            self.open.iter().map(Coefficient::rounded).collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acceptance::{components, stencils};
    use crate::dual::{Scalar, Taylor};

    type Small<S> = [[S; 3]; 3];
    struct ClosedModel<S: Scalar> {
        density: [Small<S>; 2],
        ao_density: [Small<S>; 2],
        ao_fock: [Small<S>; 2],
        overlap: Small<S>,
    }
    fn closed_model<S: Scalar>(t: S) -> ClosedModel<S> {
        let f = S::from_f64;
        let product = |a: Small<S>, b: Small<S>| {
            std::array::from_fn(|i| {
                std::array::from_fn(|j| (0..3).fold(f(0.0), |sum, k| sum + a[i][k] * b[k][j]))
            })
        };
        let mut u: Small<S> =
            std::array::from_fn(|i| std::array::from_fn(|j| f(f64::from(i == j))));
        for ((i, j), (a, b)) in
            [(0, 1), (1, 2), (0, 2)]
                .into_iter()
                .zip([(0.31, 0.07), (-0.23, 0.05), (0.29, -0.04)])
        {
            let theta = f(a) * t + f(b) * t * t;
            let denominator = f(1.0) + theta * theta;
            let cos = (f(1.0) - theta * theta) / denominator;
            let sin = f(2.0) * theta / denominator;
            let mut r: Small<S> =
                std::array::from_fn(|i| std::array::from_fn(|j| f(f64::from(i == j))));
            r[i][i] = cos;
            r[j][j] = cos;
            r[i][j] = -sin;
            r[j][i] = sin;
            u = product(u, r);
        }
        let metric: [S; 3] = std::array::from_fn(|i| {
            f(1.0) + f(0.1 * (i + 1) as f64) * t + f(0.03 * (3 - i) as f64) * t * t
        });
        let root = metric.map(Scalar::sqrt);
        let density = std::array::from_fn(|s| {
            std::array::from_fn(|i| {
                std::array::from_fn(|j| (0..2 - s).fold(f(0.0), |v, k| v + u[i][k] * u[j][k]))
            })
        });
        let ao_density = std::array::from_fn(|s| {
            std::array::from_fn(|i| std::array::from_fn(|j| density[s][i][j] / (root[i] * root[j])))
        });
        let ao_fock = std::array::from_fn(|s| {
            std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    let energies = [[-1.2, -0.1, 0.8], [-1.1, 0.2, 0.9]][s];
                    root[i]
                        * root[j]
                        * (0..3).fold(f(0.0), |v, k| v + u[i][k] * f(energies[k]) * u[j][k])
                })
            })
        });
        let overlap = std::array::from_fn(|i| {
            std::array::from_fn(|j| if i == j { metric[i] } else { f(0.0) })
        });
        ClosedModel {
            density,
            ao_density,
            ao_fock,
            overlap,
        }
    }
    fn coefficients(value: Small<Taylor<5>>) -> Vec<Matrix> {
        (0..5)
            .map(|n| {
                let mut matrix = Matrix::zeros(3, 3);
                for i in 0..3 {
                    for j in 0..3 {
                        matrix[(i, j)] = value[i][j].c[n];
                    }
                }
                matrix
            })
            .collect()
    }

    #[test]
    fn every_retained_flag_ao_coefficient_has_independent_four_step_fits() {
        let model = closed_model(Taylor::<5>::from_coefficients(&[0.0, 1.0]));
        let d = model.density.map(coefficients);
        let classes = Classes {
            closed: 1,
            open: 1,
            total: 3,
        };
        let mut closed = Matrix::zeros(3, 3);
        closed[(0, 0)] = 1.0;
        let mut open = Matrix::zeros(3, 3);
        open[(1, 1)] = 1.0;
        let mut flag = Flag::new(closed, open, classes);
        for n in 1..=4 {
            flag.advance(n);
            let (c, o) = flag.rounded();
            // Supply independently known free rotation blocks. This checks
            // the retained algebra, not the molecular response solver.
            let amplitudes = [
                d[1][n][(0, 1)] - c[n][(0, 1)],
                d[1][n][(0, 2)] - c[n][(0, 2)],
                d[0][n][(1, 2)] - d[1][n][(1, 2)] - o[n][(1, 2)],
            ];
            flag.rotate(n, classes.rotation(&amplitudes));
        }
        let b = precision::transport(&Matrix::identity(3), &coefficients(model.overlap), 4);
        let p = flag.densities(&b, 4);
        let f = std::array::from_fn(|s| {
            precision::congruence(&b, &precision::retain(&coefficients(model.ao_fock[s])), 4)
        });
        for n in 1..=4 {
            assert!(flag.defect(n) < 1e-12);
            assert!(flag.stationarity(&f, n).max_abs() < 1e-12);
        }
        let entries: Vec<_> = (0..2)
            .flat_map(|s| (0..3).flat_map(move |i| (i..3).map(move |j| (s, i, j))))
            .collect();
        let stages = [
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
            ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Fourth),
        ];
        let mut failures = Vec::new();
        for n in 1..=4 {
            let names: Vec<_> = entries
                .iter()
                .map(|(s, i, j)| format!("retained flag AO spin={s} derivative={n} i={i} j={j}"))
                .collect();
            stencils::fit_components_staged(
                &names,
                &stages,
                |steps, stencil| {
                    entries
                        .iter()
                        .map(|&(s, i, j)| {
                            std::array::from_fn(|k| {
                                let h = steps[k];
                                let sample = |offset| {
                                    closed_model(Taylor::<5>::from_coefficients(&[offset, 1.0]))
                                        .ao_density[s][i][j]
                                        .c[n - 1]
                                };
                                let (p2, m2) = if stencil == stencils::Stencil::Fourth {
                                    (sample(2.0 * h), sample(-2.0 * h))
                                } else {
                                    (0.0, 0.0)
                                };
                                let target =
                                    n as f64 * (p[s][n].high[(i, j)] + p[s][n].low[(i, j)]);
                                (stencil.first(sample(h), sample(-h), p2, m2, h) - target).abs()
                            })
                        })
                        .collect()
                },
                &mut failures,
            );
        }
        components::assert_no_failures("all retained flag AO coefficients", &failures);
    }

    #[test]
    fn large_fourth_order_projectors_preserve_raw_constraints_without_a_floor() {
        let classes = Classes {
            closed: 1,
            open: 1,
            total: 3,
        };
        let mut closed = Matrix::zeros(3, 3);
        let mut open = Matrix::zeros(3, 3);
        closed[(0, 0)] = 1.0;
        open[(1, 1)] = 1.0;
        let mut flag = Flag::new(closed, open, classes);
        for n in 1..=4 {
            flag.advance(n);
            if n <= 2 {
                let rotations = if n == 1 {
                    [12.31, -8.47, 14.92]
                } else {
                    [-21.6, 31.5, 0.81]
                };
                flag.rotate(n, classes.rotation(&rotations));
            }
            let defect = flag.defect(n);
            println!(
                "retained-remainder common-orbital projector order={n} raw_defect={defect:.12e}"
            );
            assert!(defect < 1e-12);
        }
        let (c, o) = flag.rounded();
        let mut defect = super::super::super::matrix_coefficient(&c, &c, 4);
        super::super::super::add(&mut defect, &c[4], -1.0);
        let cross = super::super::super::matrix_coefficient(&c, &o, 4);
        let rounded = defect
            .as_slice()
            .iter()
            .chain(cross.as_slice())
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        println!("rounded-only fourth-order projector raw defect={rounded:.12e}; full retained coefficients are used in density and stationarity");
        assert!(rounded > 1e-12);
    }
}
