// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Moving all-electron contact and spin-dipolar nuclear operators.
//! Nuclear path coefficients and the two independent Coulomb-probe
//! derivatives are distinct AD axes. No finite differences or trace repair.

use crate::basis::{aos::cart_to_sph, cart_components, n_sph, MolBasis};
use crate::dual::{DualN, Scalar};
use crate::integrals::analytic::GaussianKernel;
use crate::integrals::analytic_basis::{self as jets, AnalyticNucleus, AnalyticShell};
use crate::linalg::Matrix;
use crate::response::series::{evaluate_series_many_result, SeriesScalar};
use crate::system::Molecule;
use crate::{DftError, Result};
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct SpinDipoleOperatorPath {
    pub contact: Vec<Matrix>,
    /// [3*spin_axis+nuclear_moment_axis][Taylor coefficient].
    pub dipolar: [Vec<Matrix>; 9],
    /// Raw maximum AO Poisson and Hermiticity defects for each order.
    pub poisson_residuals: Vec<f64>,
    pub hermiticity_residuals: Vec<f64>,
}

/// Coefficient k is the path derivative divided by k!, in Bohr^-3.
/// The target nucleus and all Gaussian centres follow the same motion.
pub fn spin_dipole_operator_path(
    mol: &Molecule,
    basis: &MolBasis,
    nucleus: usize,
    direction: &[[f64; 3]],
    order: usize,
) -> Result<SpinDipoleOperatorPath> {
    if nucleus >= mol.len()
        || direction.len() != mol.len()
        || order > 2
        || direction.iter().flatten().any(|v| !v.is_finite())
        || basis.shells.iter().any(|s| s.atom >= mol.len())
        || !basis.ecps.is_empty()
        || basis.nao_sph == 0
    {
        return Err(DftError::InvalidInput("moving spin-dipolar operators require all-electron AOs, a valid target/direction and order0..2".into()));
    }
    let n = basis.nao_sph;
    let count = crate::resources::checked_bytes(&[10, n, n], "moving spin operator entries")?;
    let raw_bytes = count
        .checked_mul(
            std::mem::size_of::<Vec<f64>>()
                + (order + 1) * std::mem::size_of::<f64>()
                + std::mem::size_of::<SeriesScalar>(),
        )
        .ok_or_else(|| DftError::InvalidInput("moving spin operator workspace overflow".into()))?;
    // Matrix storage has its own ledger. This lease covers the coexisting
    // raw coefficient arrays/handles, not the unregistered AD arena or RSS.
    let _raw_workspace =
        crate::resources::reserve(raw_bytes, "moving spin operator coefficient arrays")?;
    let raw = raw_spin_dipole_series(mol, basis, nucleus, direction, order)?;
    let mut contact = vec![Matrix::zeros(n, n); order + 1];
    let mut dipolar = std::array::from_fn::<_, 9, _>(|_| vec![Matrix::zeros(n, n); order + 1]);
    let mut poisson_residuals = vec![0.0_f64; order + 1];
    let mut hermiticity_residuals = vec![0.0_f64; order + 1];
    for degree in 0..=order {
        for mu in 0..n {
            for nu in 0..n {
                contact[degree][(mu, nu)] = raw[mu * n + nu][degree];
                for (component, matrices) in dipolar.iter_mut().enumerate() {
                    matrices[degree][(mu, nu)] = raw[(1 + component) * n * n + mu * n + nu][degree];
                }
            }
        }
        for mu in 0..n {
            for nu in 0..n {
                poisson_residuals[degree] = poisson_residuals[degree].max(
                    (dipolar[0][degree][(mu, nu)]
                        + dipolar[4][degree][(mu, nu)]
                        + dipolar[8][degree][(mu, nu)])
                        .abs(),
                );
                for matrix in dipolar
                    .iter()
                    .map(|v| &v[degree])
                    .chain(std::iter::once(&contact[degree]))
                {
                    if !matrix[(mu, nu)].is_finite() {
                        return Err(DftError::InvalidInput(
                            "nonfinite moving spin operator coefficient".into(),
                        ));
                    }
                    hermiticity_residuals[degree] = hermiticity_residuals[degree]
                        .max((matrix[(mu, nu)] - matrix[(nu, mu)]).abs());
                }
            }
        }
    }
    Ok(SpinDipoleOperatorPath {
        contact,
        dipolar,
        poisson_residuals,
        hermiticity_residuals,
    })
}

fn point_values(
    shell: &AnalyticShell<SeriesScalar>,
    point: [SeriesScalar; 3],
) -> Vec<SeriesScalar> {
    let d = std::array::from_fn::<_, 3, _>(|a| point[a] - shell.center[a]);
    let r2 = d
        .iter()
        .fold(SeriesScalar::from_f64(0.0), |sum, x| sum + *x * *x);
    let radial = shell
        .exponents
        .iter()
        .zip(&shell.coefficients)
        .fold(SeriesScalar::from_f64(0.0), |sum, (a, c)| {
            sum + *c * (-*a * r2).exp()
        });
    let cart: Vec<_> = cart_components(shell.l)
        .iter()
        .map(|&(x, y, z)| radial * d[0].powi(x) * d[1].powi(y) * d[2].powi(z))
        .collect();
    let transform = cart_to_sph(shell.l);
    (0..n_sph(shell.l))
        .map(|s| {
            cart.iter()
                .enumerate()
                .fold(SeriesScalar::from_f64(0.0), |sum, (c, v)| {
                    sum + *v * SeriesScalar::from_f64(transform[(c, s)])
                })
        })
        .collect()
}

/// All contact and ordered dipolar AO entries along a moving nuclear path.
/// Nested probe derivatives act on the Coulomb operator independently of t.
fn raw_spin_dipole_series(
    mol: &Molecule,
    basis: &MolBasis,
    nucleus: usize,
    direction: &[[f64; 3]],
    order: usize,
) -> Result<Vec<Vec<f64>>> {
    let inputs: Vec<_> = mol
        .atoms
        .iter()
        .zip(direction)
        .flat_map(|(atom, velocity)| {
            atom.position
                .to_array()
                .into_iter()
                .zip(velocity)
                .map(move |(position, speed)| {
                    let mut value = vec![0.0; order + 1];
                    value[0] = position;
                    if order > 0 {
                        value[1] = *speed;
                    }
                    value
                })
        })
        .collect();
    evaluate_series_many_result(&inputs, |v| {
        type Inner = DualN<SeriesScalar, 3>;
        type Outer = DualN<Inner, 3>;
        let lift = |value| Outer::constant(Inner::constant(value));
        let point = std::array::from_fn(|a| v[3 * nucleus + a]);
        let probe = std::array::from_fn(|a| Outer::variable(Inner::variable(point[a], a), a));
        let source = [AnalyticNucleus {
            center: probe,
            charge: Outer::from_f64(-1.0),
        }];
        let shells: Vec<_> = basis
            .shells
            .iter()
            .map(|shell| {
                let offset = shell.center - mol.atoms[shell.atom].position;
                let offset = offset.to_array();
                let center = std::array::from_fn(|a| {
                    v[3 * shell.atom + a] + SeriesScalar::from_f64(offset[a])
                });
                AnalyticShell::from_placed(shell, center)
            })
            .collect::<Result<_>>()?;
        let lifted: Vec<_> = shells
            .iter()
            .map(|shell| AnalyticShell {
                l: shell.l,
                center: shell.center.map(lift),
                exponents: shell.exponents.iter().copied().map(lift).collect(),
                coefficients: shell.coefficients.iter().copied().map(lift).collect(),
            })
            .collect();
        let values: Vec<_> = shells
            .iter()
            .map(|shell| point_values(shell, point))
            .collect();
        let nao = basis.nao_sph;
        let count = nao
            .checked_mul(nao)
            .and_then(|v| v.checked_mul(10))
            .ok_or_else(|| DftError::InvalidInput("hyperfine operator size overflow".into()))?;
        let mut out = vec![SeriesScalar::from_f64(0.0); count];
        for (a, sa) in basis.shells.iter().enumerate() {
            for (b, sb) in basis.shells.iter().enumerate().take(a + 1) {
                let block = jets::attraction(
                    &lifted[a],
                    &lifted[b],
                    &source,
                    GaussianKernel::Coulomb,
                    order + 2,
                )?;
                for i in 0..n_sph(sa.l) {
                    for j in 0..n_sph(sb.l) {
                        let contact = values[a][i] * values[b][j];
                        let entry = block.values[i * n_sph(sb.l) + j];
                        let mu = sa.sph_off + i;
                        let nu = sb.sph_off + j;
                        for (row, col) in [(mu, nu), (nu, mu)] {
                            out[row * nao + col] = contact;
                            for u in 0..3 {
                                for w in 0..3 {
                                    let correction = if u == w {
                                        contact * SeriesScalar::from_f64(4.0 * PI / 3.0)
                                    } else {
                                        SeriesScalar::from_f64(0.0)
                                    };
                                    out[(1 + 3 * u + w) * nao * nao + row * nao + col] =
                                        entry.im[u].im[w] + correction;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    })
}
