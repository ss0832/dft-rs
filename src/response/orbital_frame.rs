// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Functional-independent real orbital-frame contractions. Full Fock
//! blocks preserve occupied/virtual rotation covariance; no physical gap
//! or matrix element is replaced by a preconditioner floor.
use crate::linalg::{AccurateSum, Matrix};

pub(super) struct OrbitalFrame<'a> {
    coefficients: &'a [Matrix; 2],
    occupied: [usize; 2],
    virtuals: [usize; 2],
    spins: usize,
    fock: [Matrix; 2],
}

impl<'a> OrbitalFrame<'a> {
    pub(super) fn new(
        coefficients: &'a [Matrix; 2],
        fock: &[Matrix; 2],
        occupied: [usize; 2],
        unrestricted: bool,
    ) -> Self {
        let virtuals = std::array::from_fn(|s| coefficients[s].cols - occupied[s]);
        Self {
            coefficients,
            occupied,
            virtuals,
            spins: if unrestricted { 2 } else { 1 },
            fock: std::array::from_fn(|s| coefficients[s].congruence_accurate(&fock[s])),
        }
    }

    /// Shared exponential retraction; the SCF optimizer supplies its
    /// generator and orthogonal exponential, so no rotation algebra is copied.
    pub(super) fn rotations(&self, amplitudes: &[f64]) -> [Matrix; 2] {
        assert_eq!(amplitudes.len(), self.dimension());
        let mut offset = 0;
        let mut rotations = std::array::from_fn(|s| Matrix::identity(self.coefficients[s].cols));
        for (s, rotation) in rotations.iter_mut().enumerate().take(self.spins) {
            let count = self.occupied[s] * self.virtuals[s];
            let generator = crate::scf::soscf::generator(
                self.coefficients[s].cols,
                self.occupied[s],
                &amplitudes[offset..offset + count],
            );
            *rotation = crate::scf::soscf::orthogonal_exponential(&generator);
            offset += count;
        }
        if self.spins == 1 {
            rotations[1] = rotations[0].clone();
        }
        rotations
    }

    pub(super) fn occupied_density(&self) -> [Matrix; 2] {
        std::array::from_fn(|s| {
            let c = &self.coefficients[s];
            let mut occupations = Matrix::zeros(c.cols, c.cols);
            for i in 0..self.occupied[s] {
                occupations[(i, i)] = 1.0;
            }
            c.transpose().congruence_accurate(&occupations)
        })
    }

    pub(super) fn dimension(&self) -> usize {
        (0..self.spins)
            .map(|s| self.occupied[s] * self.virtuals[s])
            .sum()
    }

    pub(super) fn density(&self, amplitudes: &[f64]) -> [Matrix; 2] {
        assert_eq!(amplitudes.len(), self.dimension());
        let mut off = 0;
        let mut density = std::array::from_fn(|s| {
            Matrix::zeros(self.coefficients[s].rows, self.coefficients[s].rows)
        });
        for (s, value) in density.iter_mut().enumerate().take(self.spins) {
            let c = &self.coefficients[s];
            let (no, nv) = (self.occupied[s], self.virtuals[s]);
            let mut orbital = Matrix::zeros(c.cols, c.cols);
            for a in 0..nv {
                for i in 0..no {
                    orbital[(no + a, i)] = amplitudes[off + a * no + i];
                    orbital[(i, no + a)] = amplitudes[off + a * no + i];
                }
            }
            *value = c.transpose().congruence_accurate(&orbital);
            off += no * nv;
        }
        if self.spins == 1 {
            density[1] = density[0].clone();
        }
        density
    }

    pub(super) fn project(&self, operator: [&Matrix; 2], out: &mut [f64]) {
        assert_eq!(out.len(), self.dimension());
        let mut off = 0;
        for (s, operator) in operator.iter().enumerate().take(self.spins) {
            let (no, nv) = (self.occupied[s], self.virtuals[s]);
            let projected = self.coefficients[s].congruence_accurate(operator);
            for a in 0..nv {
                for i in 0..no {
                    out[off + a * no + i] = projected[(no + a, i)];
                }
            }
            off += no * nv;
        }
    }

    pub(super) fn add_fock_action(&self, amplitudes: &[f64], out: &mut [f64]) {
        assert_eq!(amplitudes.len(), self.dimension());
        assert_eq!(out.len(), self.dimension());
        let mut off = 0;
        for s in 0..self.spins {
            let (no, nv) = (self.occupied[s], self.virtuals[s]);
            let f = &self.fock[s];
            for a in 0..nv {
                for i in 0..no {
                    let mut sum = AccurateSum::default();
                    sum.add_product(out[off + a * no + i], 1.0);
                    for b in 0..nv {
                        sum.add_product(f[(no + a, no + b)], amplitudes[off + b * no + i]);
                    }
                    for j in 0..no {
                        sum.add_product(-amplitudes[off + a * no + j], f[(j, i)]);
                    }
                    out[off + a * no + i] = sum.value();
                }
            }
            off += no * nv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acceptance::{components, stencils};

    fn matrix(values: &[f64]) -> Matrix {
        let mut m = Matrix::zeros(4, 4);
        m.as_mut_slice().copy_from_slice(values);
        m
    }

    fn coefficients() -> [Matrix; 2] {
        [
            matrix(&[
                0.83, -0.31, 0.19, 0.27, 0.23, 0.91, -0.17, 0.13, -0.29, 0.11, 0.87, -0.21, 0.16,
                -0.24, 0.32, 0.94,
            ]),
            matrix(&[
                0.92, 0.17, -0.23, 0.11, -0.19, 0.86, 0.27, -0.14, 0.21, -0.13, 0.94, 0.18, -0.12,
                0.26, -0.16, 0.89,
            ]),
        ]
    }

    fn core() -> [Matrix; 2] {
        [
            matrix(&[
                -1.3, 0.17, -0.23, 0.11, 0.17, -0.71, 0.29, -0.19, -0.23, 0.29, 0.43, 0.31, 0.11,
                -0.19, 0.31, 0.87,
            ]),
            matrix(&[
                -1.11, -0.21, 0.13, 0.19, -0.21, -0.63, 0.17, 0.23, 0.13, 0.17, 0.57, -0.27, 0.19,
                0.23, -0.27, 0.96,
            ]),
        ]
    }

    fn density(c: &[Matrix; 2], occupied: [usize; 2]) -> [Matrix; 2] {
        std::array::from_fn(|s| {
            let mut p = Matrix::zeros(4, 4);
            for a in 0..occupied[s] {
                for i in 0..4 {
                    for j in 0..4 {
                        p[(i, j)] += c[s][(i, a)] * c[s][(j, a)];
                    }
                }
            }
            p
        })
    }

    fn response(p: &[Matrix; 2]) -> [Matrix; 2] {
        std::array::from_fn(|s| {
            let mut f = Matrix::zeros(4, 4);
            for i in 0..4 {
                for j in 0..4 {
                    f[(i, j)] = 0.23 * (p[0][(i, j)] + p[1][(i, j)]) - 0.17 * p[s][(i, j)];
                }
            }
            f
        })
    }

    fn fock(c: &[Matrix; 2], occupied: [usize; 2], spins: usize) -> [Matrix; 2] {
        let mut f = core();
        if spins == 1 {
            f[1] = f[0].clone();
        }
        let r = response(&density(c, occupied));
        for s in 0..2 {
            for (a, b) in f[s].as_mut_slice().iter_mut().zip(r[s].as_slice()) {
                *a += b;
            }
        }
        f
    }

    fn gradient(c: &[Matrix; 2], occupied: [usize; 2], spins: usize) -> Vec<f64> {
        let f = fock(c, occupied, spins);
        let mut g = Vec::new();
        for s in 0..spins {
            let projected = c[s].transpose().matmul_seq(&f[s]).matmul_seq(&c[s]);
            for a in occupied[s]..4 {
                for i in 0..occupied[s] {
                    g.push(projected[(a, i)]);
                }
            }
        }
        g
    }

    fn rotate(
        c: &[Matrix; 2],
        occupied: [usize; 2],
        spins: usize,
        axis: usize,
        t: f64,
    ) -> [Matrix; 2] {
        let mut out = c.clone();
        let mut remaining = axis;
        for s in 0..spins {
            let no = occupied[s];
            let length = no * (4 - no);
            if remaining >= length {
                remaining -= length;
                continue;
            }
            let (a, i) = (no + remaining / no, remaining % no);
            let (sin, cos) = t.sin_cos();
            for mu in 0..4 {
                out[s][(mu, i)] = cos * c[s][(mu, i)] + sin * c[s][(mu, a)];
                out[s][(mu, a)] = cos * c[s][(mu, a)] - sin * c[s][(mu, i)];
            }
            if spins == 1 {
                for mu in 0..4 {
                    out[1][(mu, i)] = cos * c[1][(mu, i)] + sin * c[1][(mu, a)];
                    out[1][(mu, a)] = cos * c[1][(mu, a)] - sin * c[1][(mu, i)];
                }
            }
            break;
        }
        out
    }

    #[test]
    fn every_noncanonical_kernel_entry_fits_an_independent_orbital_gradient() {
        // E=sum tr(H_s P_s)+0.23/2 tr(P_total^2)-0.17/2 sum tr(P_s^2).
        // Its independently evaluated gradient supplies the FD oracle.
        // All matrices are dense and all OV columns/rows are checked.
        for unrestricted in [false, true] {
            let occupied = if unrestricted { [2, 1] } else { [2, 2] };
            let spins = if unrestricted { 2 } else { 1 };
            let mut c = coefficients();
            if !unrestricted {
                c[1] = c[0].clone();
            }
            let f = fock(&c, occupied, spins);
            let frame = OrbitalFrame::new(&c, &f, occupied, unrestricted);
            let density = frame.occupied_density();
            let expected = super::tests::density(&c, occupied);
            for s in 0..2 {
                assert!(density[s].max_abs_diff(&expected[s]) < 1e-15);
            }
            let dimension = frame.dimension();
            let mut failures = Vec::new();
            for axis in 0..dimension {
                let mut u = vec![0.0; dimension];
                u[axis] = 1.0;
                let df = response(&frame.density(&u));
                let mut analytic = vec![0.0; dimension];
                frame.project([&df[0], &df[1]], &mut analytic);
                frame.add_fock_action(&u, &mut analytic);
                let names:Vec<_>=(0..dimension).map(|row|format!("noncanonical unrestricted={unrestricted} kernel row={row} column={axis}")).collect();
                let stages = [
                    ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
                    ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
                    ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
                    ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Fourth),
                ];
                stencils::fit_components_staged(
                    &names,
                    &stages,
                    |steps, stencil| {
                        let mut errors = vec![[0.0; 4]; dimension];
                        for (k, &h) in steps.iter().enumerate() {
                            let plus =
                                gradient(&rotate(&c, occupied, spins, axis, h), occupied, spins);
                            let minus =
                                gradient(&rotate(&c, occupied, spins, axis, -h), occupied, spins);
                            let (plus2, minus2) = if stencil == stencils::Stencil::Fourth {
                                (
                                    gradient(
                                        &rotate(&c, occupied, spins, axis, 2. * h),
                                        occupied,
                                        spins,
                                    ),
                                    gradient(
                                        &rotate(&c, occupied, spins, axis, -2. * h),
                                        occupied,
                                        spins,
                                    ),
                                )
                            } else {
                                (vec![0.; dimension], vec![0.; dimension])
                            };
                            for row in 0..dimension {
                                errors[row][k] = (stencil.first(
                                    plus[row],
                                    minus[row],
                                    plus2[row],
                                    minus2[row],
                                    h,
                                ) - analytic[row])
                                    .abs();
                            }
                        }
                        errors
                    },
                    &mut failures,
                );
            }
            components::assert_no_failures("complete noncanonical real orbital Hessian", &failures);
        }
    }

    #[test]
    fn every_dense_orbital_rotation_entry_has_an_independent_four_step_fit() {
        for unrestricted in [false, true] {
            let occupied = if unrestricted { [2, 1] } else { [2, 2] };
            let spins = if unrestricted { 2 } else { 1 };
            let mut c = coefficients();
            if !unrestricted {
                c[1] = c[0].clone();
            }
            let f = fock(&c, occupied, spins);
            let frame = OrbitalFrame::new(&c, &f, occupied, unrestricted);
            let amplitudes: Vec<_> = (0..frame.dimension())
                .map(|k| 0.17 + ((k * 7) % 13) as f64 * 0.019)
                .collect();
            let mut analytic = Vec::new();
            let mut names = Vec::new();
            let mut offset = 0;
            for s in 0..spins {
                let no = occupied[s];
                for mu in 0..4 {
                    for col in 0..4 {
                        let value = if col < no {
                            (0..4 - no)
                                .map(|a| c[s][(mu, no + a)] * amplitudes[offset + a * no + col])
                                .sum::<f64>()
                        } else {
                            -(0..no)
                                .map(|i| c[s][(mu, i)] * amplitudes[offset + (col - no) * no + i])
                                .sum::<f64>()
                        };
                        assert!(value.abs() > 1e-9);
                        analytic.push(value);
                        names.push(format!("orbital rotation unrestricted={unrestricted} spin={s} row={mu} col={col}"));
                    }
                }
                offset += no * (4 - no);
            }
            let stages = [
                ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
                ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
                ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
                ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Fourth),
            ];
            let mut failures = Vec::new();
            stencils::fit_components_staged(
                &names,
                &stages,
                |steps, stencil| {
                    let mut errors = vec![[0.0; 4]; analytic.len()];
                    let at = |t: f64| {
                        let step: Vec<_> = amplitudes.iter().map(|v| v * t).collect();
                        let rotations = frame.rotations(&step);
                        c.iter()
                            .zip(rotations)
                            .take(spins)
                            .flat_map(|(c, r)| c.matmul_accurate(&r, false).as_slice().to_vec())
                            .collect::<Vec<_>>()
                    };
                    for (k, &h) in steps.iter().enumerate() {
                        let plus = at(h);
                        let minus = at(-h);
                        let plus2 = at(2.0 * h);
                        let minus2 = at(-2.0 * h);
                        for (i, &value) in analytic.iter().enumerate() {
                            errors[i][k] =
                                (stencil.first(plus[i], minus[i], plus2[i], minus2[i], h) - value)
                                    .abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
            components::assert_no_failures("all dense orbital rotation derivatives", &failures);
        }
    }
}
