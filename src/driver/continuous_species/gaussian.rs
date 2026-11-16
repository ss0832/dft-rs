// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed-exponent local Gaussian potential and analytic center derivatives.
//! This kernel has no electron number, atomic label or basis selection logic.

use crate::basis::{aos, cart_components, MolBasis};
use crate::integrals::hermite::ETable;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::{DftError, Result};
use std::f64::consts::PI;

/// Matrices for the unit-amplitude potential exp(-exponent |r-center|^2).
/// Multiplying by an amplitude in Hartree gives an electronic Hamiltonian.
pub struct GaussianPotential {
    pub operator: Matrix,
    /// Derivative with respect to the potential's center, per Bohr.
    pub center_derivatives: [Matrix; 3],
    /// Derivative through moving AO centers, with potential center fixed.
    pub atom_derivatives: Vec<[Matrix; 3]>,
}

struct Pair {
    e: [ETable; 3],
    r: [Vec<f64>; 3],
    prefactor: f64,
}

impl Pair {
    #[allow(clippy::too_many_arguments)]
    fn new(
        a: f64,
        b: f64,
        ac: Vec3,
        bc: Vec3,
        la: usize,
        lb: usize,
        center: Vec3,
        exponent: f64,
    ) -> Self {
        let p = a + b;
        let rho = p * exponent / (p + exponent);
        let product = (ac * a + bc * b) * (1.0 / p);
        let e = std::array::from_fn(|i| ETable::build(la + 1, lb + 1, ac.get(i) - bc.get(i), a, b));
        let r = std::array::from_fn(|i| {
            let d = product.get(i) - center.get(i);
            let mut values = vec![0.0; la + lb + 2];
            values[0] = (-rho * d * d).exp();
            if values.len() > 1 {
                values[1] = -2.0 * rho * d * values[0];
            }
            for t in 1..values.len() - 1 {
                values[t + 1] = -2.0 * rho * (d * values[t] + t as f64 * values[t - 1]);
            }
            values
        });
        Self {
            e,
            r,
            prefactor: (PI / (p + exponent)).powf(1.5),
        }
    }

    fn integral(&self, a: [i32; 3], b: [i32; 3], center_axis: Option<usize>) -> f64 {
        if a.iter().chain(&b).any(|&l| l < 0) {
            return 0.0;
        }
        let mut product = self.prefactor;
        for axis in 0..3 {
            let shift = usize::from(center_axis == Some(axis));
            let mut value = 0.0;
            for t in 0..=(a[axis] + b[axis]) as usize {
                value += self.e[axis].get(a[axis] as usize, b[axis] as usize, t)
                    * self.r[axis][t + shift];
            }
            product *= if shift == 1 { -value } else { value };
        }
        product
    }

    fn orbital_derivative(
        &self,
        a: [i32; 3],
        b: [i32; 3],
        axis: usize,
        first: bool,
        exponent: f64,
    ) -> f64 {
        let mut raised = if first { a } else { b };
        let mut lowered = raised;
        raised[axis] += 1;
        lowered[axis] -= 1;
        let l = if first { a[axis] } else { b[axis] };
        let high = if first {
            self.integral(raised, b, None)
        } else {
            self.integral(a, raised, None)
        };
        let low = if first {
            self.integral(lowered, b, None)
        } else {
            self.integral(a, lowered, None)
        };
        2.0 * exponent * high - l as f64 * low
    }
}

/// Contract Cartesian primitive integrals and transform using the common
/// spherical AO convention. Exponents and contraction coefficients stay fixed.
pub fn gaussian_potential(
    basis: &MolBasis,
    atoms: usize,
    center: Vec3,
    exponent: f64,
) -> Result<GaussianPotential> {
    if !exponent.is_finite()
        || exponent <= 0.0
        || (0..3).any(|i| !center.get(i).is_finite())
        || basis.shells.iter().any(|s| s.atom >= atoms)
    {
        return Err(DftError::InvalidInput("Gaussian potentials require finite centers, positive exponents and valid AO atom indices".into()));
    }
    let n = basis.nao_sph;
    let mut result = GaussianPotential {
        operator: Matrix::zeros(n, n),
        center_derivatives: std::array::from_fn(|_| Matrix::zeros(n, n)),
        atom_derivatives: (0..atoms)
            .map(|_| std::array::from_fn(|_| Matrix::zeros(n, n)))
            .collect(),
    };
    for sa in 0..basis.shells.len() {
        for sb in sa..basis.shells.len() {
            let a = &basis.shells[sa];
            let b = &basis.shells[sb];
            let ca = cart_components(a.l);
            let cb = cart_components(b.l);
            // value, potential center xyz, first AO center xyz, second AO center xyz.
            let mut blocks: [Matrix; 10] =
                std::array::from_fn(|_| Matrix::zeros(ca.len(), cb.len()));
            for (&ea, &wa) in a.exps.iter().zip(&a.coefs) {
                for (&eb, &wb) in b.exps.iter().zip(&b.coefs) {
                    let pair = Pair::new(
                        ea,
                        eb,
                        a.center,
                        b.center,
                        a.l as usize,
                        b.l as usize,
                        center,
                        exponent,
                    );
                    for (i, &(ax, ay, az)) in ca.iter().enumerate() {
                        for (j, &(bx, by, bz)) in cb.iter().enumerate() {
                            let ac = [ax, ay, az];
                            let bc = [bx, by, bz];
                            let w = wa * wb;
                            blocks[0][(i, j)] += w * pair.integral(ac, bc, None);
                            for axis in 0..3 {
                                blocks[1 + axis][(i, j)] += w * pair.integral(ac, bc, Some(axis));
                                blocks[4 + axis][(i, j)] +=
                                    w * pair.orbital_derivative(ac, bc, axis, true, ea);
                                blocks[7 + axis][(i, j)] +=
                                    w * pair.orbital_derivative(ac, bc, axis, false, eb);
                            }
                        }
                    }
                }
            }
            for (index, block) in blocks.iter().enumerate() {
                let spherical = aos::sph_block(a.l, b.l, block);
                let target = match index {
                    0 => &mut result.operator,
                    1..=3 => &mut result.center_derivatives[index - 1],
                    4..=6 => &mut result.atom_derivatives[a.atom][index - 4],
                    _ => &mut result.atom_derivatives[b.atom][index - 7],
                };
                for i in 0..spherical.rows {
                    for j in 0..spherical.cols {
                        target[(a.sph_off + i, b.sph_off + j)] += spherical[(i, j)];
                        if sa != sb {
                            target[(b.sph_off + j, a.sph_off + i)] += spherical[(i, j)];
                        }
                    }
                }
            }
        }
    }
    Ok(result)
}
