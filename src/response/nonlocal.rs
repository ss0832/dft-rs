// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Nonlinear VV10 response on a fixed molecular quadrature grid. Kernel,
//! local density/gradient dependence and AO chain-rule weights all carry
//! Taylor coefficients. Pair sums use the same vacuum cutoff as SCF.

use super::series::evaluate_series_many;
use crate::basis::MolBasis;
use crate::dual::{Scalar, Taylor};
use crate::grid::{eval::eval_aos_full, MolecularGrid};
use crate::linalg::Matrix;
use crate::nlc::{Vv10Params, MOLECULAR_VV10_RHO_CUT};
use crate::{DftError, Result};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct NonlocalTaylor {
    pub energy_coefficients: Vec<f64>,
    pub fock_coefficients: Vec<Matrix>,
}

pub fn vv10_taylor(
    mb: &MolBasis,
    grid: &MolecularGrid,
    densities: &[Matrix],
    params: Vv10Params,
) -> Result<NonlocalTaylor> {
    let width = densities.len();
    let nao = mb.nao_sph;
    let count = grid.len();
    if width == 0
        || grid.weights.len() != count
        || grid.weights.iter().any(|x| !x.is_finite())
        || grid
            .points
            .iter()
            .any(|p| !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite())
        || densities
            .iter()
            .any(|d| d.rows != nao || d.cols != nao || d.as_slice().iter().any(|x| !x.is_finite()))
        || !params.b.is_finite()
        || params.b <= 0.0
        || !params.c.is_finite()
        || params.c < 0.0
    {
        return Err(DftError::InvalidInput(
            "invalid VV10 density Taylor path, grid or parameters".into(),
        ));
    }
    let mut rho = vec![vec![0.0; width]; count];
    let mut grad: Vec<[Vec<f64>; 3]> = (0..count)
        .map(|_| std::array::from_fn(|_| vec![0.0; width]))
        .collect();
    for start in (0..count).step_by(256) {
        let end = (start + 256).min(count);
        let aos = eval_aos_full(mb, &grid.points[start..end], true, false);
        let g = aos.grad.as_ref().expect("AO gradients");
        for (n, d) in densities.iter().enumerate() {
            let pd = aos.phi.matmul_seq(d);
            for i in 0..end - start {
                for mu in 0..nao {
                    rho[start + i][n] += pd[(i, mu)] * aos.phi[(i, mu)];
                    for a in 0..3 {
                        grad[start + i][a][n] += 2.0 * pd[(i, mu)] * g[a][(i, mu)];
                    }
                }
            }
        }
    }
    let fields = vv10_density_taylor(grid, &rho, &grad, params)?;
    let energy = fields.energy;
    let vrho = fields.vrho;
    let vsigma = fields.vsigma;
    let mut fock: Vec<_> = (0..width).map(|_| Matrix::zeros(nao, nao)).collect();
    for start in (0..count).step_by(256) {
        let end = (start + 256).min(count);
        let aos = eval_aos_full(mb, &grid.points[start..end], true, false);
        let g = aos.grad.as_ref().expect("AO gradients");
        for p in 0..end - start {
            let i = start + p;
            if rho[i][0] <= MOLECULAR_VV10_RHO_CUT {
                continue;
            }
            for n in 0..width {
                let b: [f64; 3] = std::array::from_fn(|a| {
                    2.0 * (0..=n)
                        .map(|k| vsigma[i][k] * grad[i][a][n - k])
                        .sum::<f64>()
                });
                for mu in 0..nao {
                    for nu in 0..nao {
                        let mut v = vrho[i][n] * aos.phi[(p, mu)] * aos.phi[(p, nu)];
                        for a in 0..3 {
                            v += b[a]
                                * (g[a][(p, mu)] * aos.phi[(p, nu)]
                                    + aos.phi[(p, mu)] * g[a][(p, nu)]);
                        }
                        fock[n][(mu, nu)] += grid.weights[i] * v;
                    }
                }
            }
        }
    }
    Ok(NonlocalTaylor {
        energy_coefficients: energy,
        fock_coefficients: fock,
    })
}

/// Complete two-endpoint energy and unweighted density/sigma partials.
/// The AO basis and its magnetic phases are deliberately outside this body.
pub(crate) struct Vv10DensityTaylor {
    pub energy: Vec<f64>,
    pub vrho: Vec<Vec<f64>>,
    pub vsigma: Vec<Vec<f64>>,
}

pub(crate) fn vv10_density_taylor(
    grid: &MolecularGrid,
    rho: &[Vec<f64>],
    grad: &[[Vec<f64>; 3]],
    params: Vv10Params,
) -> Result<Vv10DensityTaylor> {
    let count = grid.len();
    let width = rho.first().map_or(0, Vec::len);
    if count == 0
        || width == 0
        || rho.len() != count
        || grad.len() != count
        || grid.weights.len() != count
        || !params.b.is_finite()
        || params.b <= 0.0
        || !params.c.is_finite()
        || params.c < 0.0
        || rho
            .iter()
            .any(|r| r.len() != width || r.iter().any(|v| !v.is_finite()))
        || grad
            .iter()
            .flatten()
            .any(|g| g.len() != width || g.iter().any(|v| !v.is_finite()))
        || grid.weights.iter().any(|v| !v.is_finite())
        || grid.points.iter().any(|p| !p.norm().is_finite())
    {
        return Err(DftError::InvalidInput(
            "invalid VV10 field coefficient dimensions, values or parameters".into(),
        ));
    }
    let indices: Vec<_> = (0..count)
        .filter(|&i| rho[i][0] > MOLECULAR_VV10_RHO_CUT)
        .collect();
    let mut sigma = vec![vec![0.0; width]; count];
    for &i in &indices {
        for n in 0..width {
            for k in 0..=n {
                for a in 0..3 {
                    sigma[i][n] += grad[i][a][k] * grad[i][a][n - k];
                }
            }
        }
    }
    // Exact support reduction: if every active endpoint contains only powers
    // of t^s, evaluate in u=t^s and expand the output. Constant input needs
    // only the scalar body. This never thresholds a small coefficient and
    // retains the requested nonlinear order even for a linear input path.
    let mut stride = 0;
    for &i in &indices {
        for n in 1..width {
            if rho[i][n] != 0.0 || sigma[i][n] != 0.0 {
                let (mut a, mut b) = (stride, n);
                while b != 0 {
                    (a, b) = (b, a % b);
                }
                stride = a;
            }
        }
    }
    if width > 1 && stride != 1 {
        let stride = if stride == 0 { width } else { stride };
        let reduced_width = 1 + (width - 1) / stride;
        let reduce = |values: &[Vec<f64>]| -> Vec<Vec<f64>> {
            values
                .iter()
                .map(|v| (0..reduced_width).map(|n| v[n * stride]).collect())
                .collect()
        };
        let result =
            direct_density_series(grid, &reduce(rho), &reduce(&sigma), params, reduced_width)?;
        let expand = |values: Vec<f64>| {
            let mut output = vec![0.0; width];
            for (n, v) in values.into_iter().enumerate() {
                output[n * stride] = v;
            }
            output
        };
        return Ok(Vv10DensityTaylor {
            energy: expand(result.energy),
            vrho: result.vrho.into_iter().map(&expand).collect(),
            vsigma: result.vsigma.into_iter().map(expand).collect(),
        });
    }
    direct_density_series(grid, rho, &sigma, params, width)
}

fn direct_density_series(
    grid: &MolecularGrid,
    rho: &[Vec<f64>],
    sigma: &[Vec<f64>],
    params: Vv10Params,
    width: usize,
) -> Result<Vv10DensityTaylor> {
    let indices: Vec<_> = (0..grid.len())
        .filter(|&i| rho[i][0] > MOLECULAR_VV10_RHO_CUT)
        .collect();
    match width {
        1 => return owned_fields::<f64>(grid, rho, sigma, params, width),
        2 => return owned_fields::<Taylor<2>>(grid, rho, sigma, params, width),
        3 => return owned_fields::<Taylor<3>>(grid, rho, sigma, params, width),
        4 => return owned_fields::<Taylor<4>>(grid, rho, sigma, params, width),
        5..=8 => return owned_fields::<Taylor<8>>(grid, rho, sigma, params, width),
        _ => {}
    }
    let local: Result<Vec<_>> = indices
        .par_iter()
        .map(|&i| {
            evaluate_series_many(&[rho[i].clone(), sigma[i].clone()], |v| {
                let (omega, kappa) = crate::nlc::vv10_local_fields(v[0], v[1], params);
                vec![omega, kappa]
            })
        })
        .collect();
    let local = local?;
    // A row scope shares the precomputed local series of every point.
    let mut inputs: Vec<Vec<f64>> = indices.iter().map(|&i| rho[i].clone()).collect();
    inputs.extend(local.iter().map(|v| v[0].clone()));
    inputs.extend(local.iter().map(|v| v[1].clone()));
    let m = indices.len();
    let rows: Result<Vec<_>> = (0..m)
        .into_par_iter()
        .map(|a| {
            let i = indices[a];
            let mut row_inputs = inputs.clone();
            row_inputs.push(sigma[i].clone());
            evaluate_series_many(&row_inputs, |v| {
                crate::nlc::vv10_density_row(
                    grid,
                    &indices,
                    a,
                    &v[..m],
                    &v[m..2 * m],
                    &v[2 * m..3 * m],
                    v[3 * m],
                    params,
                )
                .to_vec()
            })
        })
        .collect();
    let rows = rows?;
    let mut vrho = vec![vec![0.0; width]; grid.len()];
    let mut vsigma = vrho.clone();
    let mut energy = vec![0.0; width];
    for (a, &i) in indices.iter().enumerate() {
        for n in 0..width {
            energy[n] += grid.weights[i] * rows[a][0][n];
        }
        vrho[i] = rows[a][1].clone();
        vsigma[i] = rows[a][2].clone();
    }
    Ok(Vv10DensityTaylor {
        energy,
        vrho,
        vsigma,
    })
}

trait OwnedCoefficient: Scalar + Send + Sync {
    fn input(values: &[f64]) -> Self;
    fn output(&self, width: usize) -> Vec<f64>;
}
impl OwnedCoefficient for f64 {
    fn input(values: &[f64]) -> Self {
        values[0]
    }
    fn output(&self, _: usize) -> Vec<f64> {
        vec![*self]
    }
}
impl<const W: usize> OwnedCoefficient for Taylor<W> {
    fn input(values: &[f64]) -> Self {
        Self::from_coefficients(values)
    }
    fn output(&self, width: usize) -> Vec<f64> {
        self.c[..width].to_vec()
    }
}
fn owned_fields<S: OwnedCoefficient>(
    grid: &MolecularGrid,
    rho: &[Vec<f64>],
    sigma: &[Vec<f64>],
    parameters: Vv10Params,
    width: usize,
) -> Result<Vv10DensityTaylor> {
    let rho: Vec<_> = rho.iter().map(|r| S::input(r)).collect();
    let sigma: Vec<_> = sigma.iter().map(|r| S::input(r)).collect();
    let fields = crate::nlc::vv10_density_fields(grid, &rho, &sigma, parameters)?;
    Ok(Vv10DensityTaylor {
        energy: fields.energy.output(width),
        vrho: fields.d_rho.iter().map(|v| v.output(width)).collect(),
        vsigma: fields.d_sigma.iter().map(|v| v.output(width)).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acceptance::components::{assert_no_failures, fit_components};
    use crate::math::Vec3;

    #[test]
    fn exact_sparse_support_preserves_every_two_endpoint_coefficient() {
        let grid = MolecularGrid {
            points: vec![Vec3::new(0.1, -0.2, 0.3), Vec3::new(0.7, 0.4, -0.1)],
            weights: vec![0.7, 0.9],
            raw_weights: vec![0.7, 0.9],
            atom_of: vec![0, 1],
        };
        for width in [2, 3, 4, 8, 13] {
            for stride in [0, 1, 2, 3, 4] {
                let mut rho = vec![vec![0.0; width]; 2];
                let mut grad: Vec<[Vec<f64>; 3]> = (0..2)
                    .map(|_| std::array::from_fn(|_| vec![0.0; width]))
                    .collect();
                let mut sigma = rho.clone();
                for i in 0..2 {
                    rho[i][0] = 0.3 + 0.2 * i as f64;
                    if stride > 0 && stride < width {
                        rho[i][stride] = 0.03;
                    }
                    for a in 0..3 {
                        let g = 0.07 + 0.02 * (i + a) as f64;
                        grad[i][a][0] = g;
                        sigma[i][0] += g * g;
                        if stride > 0 && stride < width {
                            grad[i][a][stride] = 0.004;
                            sigma[i][stride] += 2.0 * g * 0.004;
                            if 2 * stride < width {
                                sigma[i][2 * stride] += 0.004_f64.powi(2);
                            }
                        }
                    }
                }
                let sparse = vv10_density_taylor(&grid, &rho, &grad, Vv10Params::ORIGINAL).unwrap();
                let direct =
                    direct_density_series(&grid, &rho, &sigma, Vv10Params::ORIGINAL, width)
                        .unwrap();
                let mut defect = 0.0_f64;
                for (a, b) in sparse
                    .energy
                    .iter()
                    .chain(sparse.vrho.iter().flatten())
                    .chain(sparse.vsigma.iter().flatten())
                    .zip(
                        direct
                            .energy
                            .iter()
                            .chain(direct.vrho.iter().flatten())
                            .chain(direct.vsigma.iter().flatten()),
                    )
                {
                    defect = defect.max((a - b).abs());
                }
                println!("nonlocal exact sparse width={width} stride={stride} full coefficient defect {defect:.12e}");
                assert!(defect < 1e-12);
            }
        }
        // A tiny nonzero odd coefficient must not be discarded as a symmetry.
        let rho = vec![vec![0.3, 1e-20, 0.03], vec![0.5, 2e-20, 0.02]];
        let grad = vec![std::array::from_fn(|_| vec![0.0; 3]); 2];
        let result = vv10_density_taylor(&grid, &rho, &grad, Vv10Params::ORIGINAL).unwrap();
        assert!(result.energy[1] != 0.0);
    }

    #[test]
    fn sparse_even_energy_and_potential_orders_have_independent_fits() {
        let grid = MolecularGrid {
            points: vec![Vec3::new(0.1, -0.2, 0.3)],
            weights: vec![0.7],
            raw_weights: vec![0.7],
            atom_of: vec![0],
        };
        let parameters = Vv10Params::ORIGINAL;
        let r0: f64 = 0.4;
        let r2: f64 = 0.07;
        let beta = (3.0 / parameters.b.powi(2)).powf(0.75) / 32.0;
        let kc = parameters.b * 1.5 * std::f64::consts::PI
            / (9.0 * std::f64::consts::PI).powf(1.0 / 6.0);
        let derivative = |observable: usize, n: usize, t: f64| {
            let (amplitude, power) = if observable == 0 {
                (-3.0 / 8.0 * grid.weights[0].powi(2) / kc.powi(3), 1.5)
            } else {
                (-9.0 / 16.0 * grid.weights[0] / kc.powi(3), 0.5)
            };
            let mut value = if observable == 0 {
                match n {
                    0 => beta * grid.weights[0] * (r0 + r2 * t * t),
                    1 => 2.0 * beta * grid.weights[0] * r2 * t,
                    2 => 2.0 * beta * grid.weights[0] * r2,
                    _ => 0.0,
                }
            } else if n == 0 {
                beta
            } else {
                0.0
            };
            let factorial = |n: usize| (1..=n).map(|v| v as f64).product::<f64>();
            let mut binomial_power = 1.0;
            for j in 0..=n {
                if j > 0 {
                    binomial_power *= (power - (j - 1) as f64) / j as f64;
                }
                if 2 * j < n {
                    continue;
                }
                let choose = factorial(j) / (factorial(n - j) * factorial(2 * j - n));
                value += amplitude
                    * factorial(n)
                    * binomial_power
                    * choose
                    * r2.powi(j as i32)
                    * (2.0 * t).powi((2 * j - n) as i32)
                    * (r0 + r2 * t * t).powf(power - j as f64);
            }
            value
        };
        let mut rho = vec![0.0; 13];
        rho[0] = r0;
        rho[2] = r2;
        let result = vv10_density_taylor(
            &grid,
            &[rho],
            &[std::array::from_fn(|_| vec![0.0; 13])],
            parameters,
        )
        .unwrap();
        let mut failures = Vec::new();
        for n in 1..=12 {
            let factorial = (1..=n).map(|v| v as f64).product::<f64>();
            let exact = [result.energy[n] * factorial, result.vrho[0][n] * factorial];
            for (observable, &actual) in exact.iter().enumerate() {
                assert!((actual - derivative(observable, n, 0.0)).abs() < 1e-10);
            }
            if n % 2 == 1 {
                assert_eq!(exact, [0.0; 2]);
                continue;
            }
            let names: Vec<_> = (0..2)
                .map(|o| format!("nonlocal sparse even observable={o} order={n}"))
                .collect();
            fit_components(
                &names,
                &[[0.08, 0.06, 0.04, 0.03], [0.02, 0.015, 0.01, 0.0075]],
                |steps| {
                    (0..2)
                        .map(|o| {
                            std::array::from_fn(|k| {
                                let h = steps[k];
                                ((derivative(o, n - 1, h) - derivative(o, n - 1, -h)) / (2.0 * h)
                                    - exact[o])
                                    .abs()
                            })
                        })
                        .collect()
                },
                &mut failures,
            );
        }
        assert_no_failures(
            "nonlocal sparse even derivatives through order twelve",
            &failures,
        );
    }

    #[test]
    fn owned_and_runtime_orders_match_independent_self_pair_closed_forms_and_fits() {
        fit_independent_self_pair(0.23, 0.017, 12);
    }

    #[test]
    fn molecular_vv10_tail_crosses_the_old_cutoff_with_the_literal_derivatives() {
        // Both signs of every independent sample lie on opposite sides
        // of the old molecular screen, where the unscreened self pair
        // still has well-defined energy and potential derivatives.
        fit_independent_self_pair(1e-8, 1.4e-9, 4);
    }

    fn fit_independent_self_pair(r0: f64, r1: f64, fit_order: usize) {
        let grid = MolecularGrid {
            points: vec![Vec3::new(0.13, -0.21, 0.19)],
            weights: vec![0.73],
            raw_weights: vec![0.73],
            atom_of: vec![0],
        };
        let parameters = Vv10Params::ORIGINAL;
        let beta = (3.0 / parameters.b.powi(2)).powf(0.75) / 32.0;
        let k = parameters.b * 1.5 * std::f64::consts::PI
            / (9.0 * std::f64::consts::PI).powf(1.0 / 6.0);
        // At R=0 the exact self-pair energy is beta*w*rho
        // minus (3/8)*w^2*rho^(3/2)/k^3, independent of sigma.
        let derivative = |observable: usize, n: usize, t: f64| {
            let power = if observable == 0 { 1.5 } else { 0.5 };
            let amplitude = if observable == 0 {
                -0.375 * grid.weights[0].powi(2) / k.powi(3)
            } else {
                -0.5625 * grid.weights[0] / k.powi(3)
            };
            let product = (0..n).fold(1.0, |v, j| v * (power - j as f64));
            let local = if observable == 0 {
                match n {
                    0 => beta * grid.weights[0] * (r0 + r1 * t),
                    1 => beta * grid.weights[0] * r1,
                    _ => 0.0,
                }
            } else if n == 0 {
                beta
            } else {
                0.0
            };
            local + amplitude * product * r1.powi(n as i32) * (r0 + r1 * t).powf(power - n as f64)
        };
        let mut last = None;
        for width in [1, 2, 3, 4, 8, 13] {
            let mut rho = vec![0.0; width];
            rho[0] = r0;
            if width > 1 {
                rho[1] = r1;
            }
            let grad = vec![std::array::from_fn(|_| vec![0.0; width])];
            let result = vv10_density_taylor(&grid, &[rho], &grad, parameters).unwrap();
            let mut factorial = 1.0;
            for n in 0..width {
                if n > 0 {
                    factorial *= n as f64;
                }
                for (observable, actual) in [
                    (0, result.energy[n] * factorial),
                    (1, result.vrho[0][n] * factorial),
                ] {
                    let expected = derivative(observable, n, 0.0);
                    let relative = ((actual - expected) / expected).abs();
                    println!("nonlocal self-pair width={width} observable={observable} order={n} exact={expected:.12e} relative={relative:.12e}");
                    assert!(relative < 1e-9);
                }
                assert_eq!(result.vsigma[0][n], 0.0);
            }
            last = Some(result);
        }
        let result = last.unwrap();
        let mut factorial = 1.0;
        let mut failures = Vec::new();
        for n in 1..=fit_order {
            factorial *= n as f64;
            let names: Vec<_> = (0..2)
                .map(|observable| format!("nonlocal self-pair observable={observable} order={n}"))
                .collect();
            fit_components(
                &names,
                &[[0.08, 0.06, 0.04, 0.03], [0.02, 0.015, 0.01, 0.0075]],
                |steps| {
                    let mut errors = vec![[0.0; 4]; 2];
                    for observable in 0..2 {
                        let exact = if observable == 0 {
                            result.energy[n] * factorial
                        } else {
                            result.vrho[0][n] * factorial
                        };
                        for (i, &h) in steps.iter().enumerate() {
                            let fd = (derivative(observable, n - 1, h)
                                - derivative(observable, n - 1, -h))
                                / (2.0 * h);
                            errors[observable][i] = (fd - exact).abs();
                        }
                    }
                    errors
                },
                &mut failures,
            );
        }
        assert_no_failures(
            "owned/runtime nonlocal self-pair derivatives through order twelve",
            &failures,
        );
    }
}
