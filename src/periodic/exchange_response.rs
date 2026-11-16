// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Displacement derivatives of the periodic exact-exchange image sum.
//!
//! For one spin density with real image blocks `D(T)` the exchange energy is
//! the truncated real-space sum of [`super::exchange`],
//!
//! ```text
//! E = c sum_{T,C,D} D_mu,nu(T) D_lambda,sigma(D-C)
//!       (mu(0) lambda(C) | sigma(D) nu(T)),
//! ```
//!
//! over the quartets selected by the separations of their shell centres
//! (`super::exchange_quartets`). A Gamma-point
//! displacement moves every image of an atom, so each of the four centres of
//! a quartet contributes the derivative with respect to its home atom.
//!
//! Three objects follow from the same quartet list:
//!
//! * the second derivatives of the sum at fixed density (skeleton term);
//! * the derivative of the exchange matrix that the self-consistent field
//!   builds, `K^x(T) = sum D(D-C) d/dx (mu lambda | sigma nu)`, projected
//!   onto Hermitian image parity exactly like the matrix itself (the source
//!   of the coupled-perturbed equations);
//! * the density derivative of the exchange gradient: the average of `K^x`
//!   and of its adjoint map, which collects the derivative quartets on the
//!   image `D - C` of the inner density. The pair selection is symmetric
//!   under the exchange of the two density factors, so the two maps agree
//!   up to rounding.
//!
//! With these the force constants are the exact derivative of the analytic
//! gradient at any cutoff.

use crate::basis::{MolBasis, MolShell};
use crate::integrals::deriv::eri_quartet_center_derivatives;
use crate::integrals::deriv2::{quartet_mixed_second, PAIRS};
use crate::integrals::CoulombKernel;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::exchange_quartets::ExchangeCells;
use crate::periodic::integrals::ImageBlock;
use crate::periodic::strain::StrainSecond;
use crate::periodic::Lattice;
use rayon::prelude::*;

/// Density weights below this size are skipped, as in the gradient.
const NEGLIGIBLE: f64 = 1e-14;

fn image_index(images: &[Vec3], t: Vec3) -> Option<usize> {
    images.iter().position(|r| (*r - t).norm() < 1e-8)
}

/// Project image matrices onto the Hermitian parity `M(T)_ij = M(-T)_ji`.
fn project_hermitian(images: &[Vec3], matrices: &mut [Matrix]) {
    for left in 0..images.len() {
        let Some(right) = image_index(images, Vec3::zero() - images[left]) else {
            continue;
        };
        if right < left {
            continue;
        }
        let a = matrices[left].clone();
        let b = matrices[right].clone();
        for i in 0..a.rows {
            for j in 0..a.cols {
                let value = 0.5 * (a[(i, j)] + b[(j, i)]);
                matrices[left][(i, j)] = value;
                matrices[right][(j, i)] = value;
            }
        }
    }
}

fn blocks(images: &[Vec3], matrices: Vec<Matrix>) -> Vec<ImageBlock> {
    images
        .iter()
        .zip(matrices)
        .map(|(translation, matrix)| ImageBlock {
            translation: *translation,
            matrix,
        })
        .collect()
}

/// First-order exchange matrices of every Gamma-point displacement
/// `x = 3 atom + axis`, as image blocks `[x][image]` without the exchange
/// coefficient.
pub(crate) struct ExchangeFirstOrder {
    /// Derivative of the exchange matrix of the self-consistent field.
    pub(crate) source: Vec<Vec<ImageBlock>>,
    /// Half the density derivative of the exchange gradient
    /// `sum D D d/dx (mu lambda | sigma nu)`.
    pub(crate) pairing: Vec<Vec<ImageBlock>>,
}

pub(crate) fn exchange_first_order(
    mb: &MolBasis,
    lattice: &Lattice,
    density: &[ImageBlock],
    cutoff: f64,
    kernel: CoulombKernel,
    natoms: usize,
) -> ExchangeFirstOrder {
    first_order(
        mb,
        lattice,
        density,
        cutoff,
        kernel,
        3 * natoms,
        &|shell, axis, components| components.push((3 * shell.atom + axis, 1.0)),
    )
}

/// First-order exchange matrices of the nine strain components `3a + b` of
/// `x -> (1 + epsilon) x`, as image blocks `[component][image]` without the
/// exchange coefficient, at density images labelled by integer translations:
/// every centre of a quartet contributes its derivative along `a` times its
/// coordinate `b`. The quartets are translation invariant, so the origin of
/// the coordinates does not matter.
pub(crate) fn exchange_strain_first_order(
    mb: &MolBasis,
    lattice: &Lattice,
    density: &[ImageBlock],
    cutoff: f64,
    kernel: CoulombKernel,
) -> ExchangeFirstOrder {
    first_order(
        mb,
        lattice,
        density,
        cutoff,
        kernel,
        9,
        &|shell, axis, components| {
            for b in 0..3 {
                components.push((3 * axis + b, shell.center.get(b)));
            }
        },
    )
}

/// The perturbation components `(index, scale)` fed by the derivative of a
/// quartet along one axis at the centre of a shell.
type Perturbation<'a> = &'a (dyn Fn(&MolShell, usize, &mut Vec<(usize, f64)>) + Sync);

fn first_order(
    mb: &MolBasis,
    lattice: &Lattice,
    density: &[ImageBlock],
    cutoff: f64,
    kernel: CoulombKernel,
    dim: usize,
    perturbation: Perturbation<'_>,
) -> ExchangeFirstOrder {
    let n = mb.nao_sph;
    let cells = ExchangeCells::for_lattice(mb, lattice, cutoff);
    let ntarget = cells.near.len();
    let densities = cells.densities(density);
    let zeros = || -> Vec<Matrix> { (0..dim * ntarget).map(|_| Matrix::zeros(n, n)).collect() };
    let add = |mut a: Vec<Matrix>, b: Vec<Matrix>| -> Vec<Matrix> {
        for (x, y) in a.iter_mut().zip(&b) {
            for (p, q) in x.as_mut_slice().iter_mut().zip(y.as_slice()) {
                *p += *q;
            }
        }
        a
    };
    let (direct, adjoint) = (0..ntarget)
        .into_par_iter()
        .fold(
            || (zeros(), zeros()),
            |(mut direct, mut adjoint), target| {
                let outer = densities[target];
                let mut components: Vec<(usize, f64)> = Vec::new();
                cells.visit(
                    mb,
                    target,
                    |_, _| true,
                    |source| outer.is_some() || densities[source].is_some(),
                    |q| {
                        let source = densities[q.source];
                        let [offset_mu, offset_lambda, offset_sigma, offset_nu] = q.offsets;
                        let derivatives = eri_quartet_center_derivatives(q.shells, kernel);
                        for (center, per_axis) in derivatives.iter().enumerate() {
                            for (axis, quartet) in per_axis.iter().enumerate() {
                                components.clear();
                                perturbation(q.shells[center], axis, &mut components);
                                for &(x, scale) in &components {
                                    let direct_block = &mut direct[x * ntarget + target];
                                    let adjoint_block = &mut adjoint[x * ntarget + q.source];
                                    for i in 0..quartet.na {
                                        for j in 0..quartet.nd {
                                            let weight = outer
                                                .map_or(0.0, |p| p[(offset_mu + i, offset_nu + j)]);
                                            let mut value = 0.0;
                                            for k in 0..quartet.nb {
                                                for l in 0..quartet.nc {
                                                    let integral = quartet.get(i, k, l, j);
                                                    if let Some(p) = source {
                                                        value += p
                                                            [(offset_lambda + k, offset_sigma + l)]
                                                            * integral;
                                                    }
                                                    adjoint_block
                                                        [(offset_lambda + k, offset_sigma + l)] +=
                                                        scale * weight * integral;
                                                }
                                            }
                                            direct_block[(offset_mu + i, offset_nu + j)] +=
                                                scale * value;
                                        }
                                    }
                                }
                            }
                        }
                    },
                );
                (direct, adjoint)
            },
        )
        .reduce(|| (zeros(), zeros()), |a, b| (add(a.0, b.0), add(a.1, b.1)));
    let targets = &cells.near;
    let mut source = Vec::with_capacity(dim);
    let mut pairing = Vec::with_capacity(dim);
    for x in 0..dim {
        let mut on_targets: Vec<Matrix> = (0..ntarget)
            .map(|t| direct[x * ntarget + t].clone())
            .collect();
        project_hermitian(targets, &mut on_targets);
        source.push(blocks(targets, on_targets));
        let mut on_targets: Vec<Matrix> = (0..ntarget)
            .map(|t| {
                let mut average = direct[x * ntarget + t].clone();
                for (p, q) in average
                    .as_mut_slice()
                    .iter_mut()
                    .zip(adjoint[x * ntarget + t].as_slice())
                {
                    *p = 0.5 * (*p + *q);
                }
                average
            })
            .collect();
        project_hermitian(targets, &mut on_targets);
        pairing.push(blocks(targets, on_targets));
    }
    ExchangeFirstOrder { source, pairing }
}

/// `sum_{T,C,D} D(T) D(D-C) d2/dx dy (mu(0) lambda(C) | sigma(D) nu(T))`
/// for the Gamma-point displacements `x, y = 3 atom + axis`, without the
/// exchange coefficient.
pub(crate) fn exchange_second_derivatives(
    mb: &MolBasis,
    lattice: &Lattice,
    density: &[ImageBlock],
    cutoff: f64,
    kernel: CoulombKernel,
    natoms: usize,
) -> Matrix {
    let dim = 3 * natoms;
    let cells = ExchangeCells::for_lattice(mb, lattice, cutoff);
    let densities = cells.densities(density);
    let partials: Vec<Matrix> = (0..cells.near.len())
        .into_par_iter()
        .filter_map(|target| {
            let outer = densities[target]?;
            let mut hessian = Matrix::zeros(dim, dim);
            cells.visit(
                mb,
                target,
                |sa, sb| {
                    let (offset_mu, offset_nu) = (mb.shells[sa].sph_off, mb.shells[sb].sph_off);
                    let (n_mu, n_nu) = (
                        crate::basis::n_sph(mb.shells[sa].l),
                        crate::basis::n_sph(mb.shells[sb].l),
                    );
                    let mut outer_max = 0.0_f64;
                    for i in 0..n_mu {
                        for j in 0..n_nu {
                            outer_max = outer_max.max(outer[(offset_mu + i, offset_nu + j)].abs());
                        }
                    }
                    outer_max >= NEGLIGIBLE
                },
                |source| densities[source].is_some(),
                |q| {
                    let source = densities[q.source].expect("screened source");
                    let shells = q.shells;
                    let [mu, lambda, sigma, nu] = shells;
                    let [offset_mu, offset_lambda, offset_sigma, offset_nu] = q.offsets;
                    let [n_mu, n_lambda, n_sigma, n_nu] = shells.map(|s| crate::basis::n_sph(s.l));
                    let atoms = [mu.atom, lambda.atom, sigma.atom, nu.atom];
                    // A quartet whose four centres move together
                    // does not change.
                    if atoms.iter().all(|a| *a == atoms[0]) {
                        return;
                    }
                    // Density weights in the element order of the
                    // quartet (mu, lambda, sigma, nu).
                    let mut weights = Vec::with_capacity(n_mu * n_lambda * n_sigma * n_nu);
                    let mut largest = 0.0_f64;
                    for i in 0..n_mu {
                        for k in 0..n_lambda {
                            for l in 0..n_sigma {
                                for j in 0..n_nu {
                                    let w = outer[(offset_mu + i, offset_nu + j)]
                                        * source[(offset_lambda + k, offset_sigma + l)];
                                    largest = largest.max(w.abs());
                                    weights.push(w);
                                }
                            }
                        }
                    }
                    if largest < NEGLIGIBLE {
                        return;
                    }
                    let second = quartet_mixed_second(shells, kernel);
                    for (pair, &(si, sj)) in PAIRS.iter().enumerate() {
                        for a in 0..3 {
                            for b in 0..3 {
                                let m: f64 = weights
                                    .iter()
                                    .zip(&second[pair][3 * a + b])
                                    .map(|(w, v)| w * v)
                                    .sum();
                                let (row, column) = (3 * atoms[si] + a, 3 * atoms[sj] + b);
                                hessian[(row, column)] += m;
                                hessian[(column, row)] += m;
                                // Same-centre blocks from the
                                // translational invariance of
                                // the quartet.
                                hessian[(row, 3 * atoms[si] + b)] -= m;
                                hessian[(column, 3 * atoms[sj] + a)] -= m;
                            }
                        }
                    }
                },
            );
            Some(hessian)
        })
        .collect();
    let mut total = Matrix::zeros(dim, dim);
    for partial in partials {
        for (t, v) in total.as_mut_slice().iter_mut().zip(partial.as_slice()) {
            *t += *v;
        }
    }
    total
}

/// Strain second derivatives of `sum_{T,C,D} D(T) D(D-C) (mu(0) lambda(C) |
/// sigma(D) nu(T))` at fixed density images, without the exchange
/// coefficient: `[3a+b][3c+d]` for two strains and `[3a+b][3 atom + c]` for a
/// strain and a Gamma-point displacement applied before it.
///
/// With `M_ij = d2/dR_i dR_j` of a quartet and `r_ij = R_i - R_j`, the
/// translational invariance of the quartet gives
/// `W_(ab),(cd) = -sum_(i != j) M_ij^(ac) (R_i - R_0)^b r_ij^d` for any
/// origin `R_0` (the two coordinate indices are not interchangeable for
/// more than two centres) and
/// `Lambda_(ab),(jc) = sum_(i != j) M_ij^(ac) r_ij^b + delta_bc dI/dR_j^a`.
pub(crate) fn exchange_strain_second(
    mb: &MolBasis,
    lattice: &Lattice,
    density: &[ImageBlock],
    cutoff: f64,
    kernel: CoulombKernel,
    natoms: usize,
) -> StrainSecond {
    let dim = 3 * natoms;
    let cells = ExchangeCells::for_lattice(mb, lattice, cutoff);
    let densities = cells.densities(density);
    let partials: Vec<StrainSecond> = (0..cells.near.len())
        .into_par_iter()
        .filter_map(|target| {
            let outer = densities[target]?;
            let mut out = StrainSecond::zeros(natoms);
            cells.visit(
                mb,
                target,
                |sa, sb| {
                    let (offset_mu, offset_nu) = (mb.shells[sa].sph_off, mb.shells[sb].sph_off);
                    let (n_mu, n_nu) = (
                        crate::basis::n_sph(mb.shells[sa].l),
                        crate::basis::n_sph(mb.shells[sb].l),
                    );
                    let mut outer_max = 0.0_f64;
                    for i in 0..n_mu {
                        for j in 0..n_nu {
                            outer_max = outer_max.max(outer[(offset_mu + i, offset_nu + j)].abs());
                        }
                    }
                    outer_max >= NEGLIGIBLE
                },
                |source| densities[source].is_some(),
                |q| {
                    let source = densities[q.source].expect("screened source");
                    let shells = q.shells;
                    let [mu, lambda, sigma, nu] = shells;
                    let [offset_mu, offset_lambda, offset_sigma, offset_nu] = q.offsets;
                    let [n_mu, n_lambda, n_sigma, n_nu] = shells.map(|s| crate::basis::n_sph(s.l));
                    let atoms = [mu.atom, lambda.atom, sigma.atom, nu.atom];
                    let centres = [mu.center, lambda.center, sigma.center, nu.center];
                    // A quartet on one centre does not change.
                    if centres.iter().all(|r| (*r - centres[0]).norm() < 1e-12) {
                        return;
                    }
                    // Density weights in the element order of the
                    // quartet (mu, lambda, sigma, nu).
                    let mut weights = Vec::with_capacity(n_mu * n_lambda * n_sigma * n_nu);
                    let mut largest = 0.0_f64;
                    for i in 0..n_mu {
                        for k in 0..n_lambda {
                            for l in 0..n_sigma {
                                for j in 0..n_nu {
                                    let w = outer[(offset_mu + i, offset_nu + j)]
                                        * source[(offset_lambda + k, offset_sigma + l)];
                                    largest = largest.max(w.abs());
                                    weights.push(w);
                                }
                            }
                        }
                    }
                    if largest < NEGLIGIBLE {
                        return;
                    }
                    let second = quartet_mixed_second(shells, kernel);
                    for (pair, &(si, sj)) in PAIRS.iter().enumerate() {
                        let r = centres[si] - centres[sj];
                        let mut m = [[0.0_f64; 3]; 3];
                        for a in 0..3 {
                            for b in 0..3 {
                                m[a][b] = weights
                                    .iter()
                                    .zip(&second[pair][3 * a + b])
                                    .map(|(w, v)| w * v)
                                    .sum();
                            }
                        }
                        // Coordinates relative to the first
                        // centre: the sum does not depend on the
                        // origin.
                        let from_i = centres[si] - centres[0];
                        let from_j = centres[sj] - centres[0];
                        for a in 0..3 {
                            for c2 in 0..3 {
                                for b in 0..3 {
                                    for d2 in 0..3 {
                                        out.strain[(3 * a + b) * 9 + 3 * c2 + d2] +=
                                            (m[c2][a] * from_j.get(b) - m[a][c2] * from_i.get(b))
                                                * r.get(d2);
                                    }
                                    out.mixed[(3 * a + b) * dim + 3 * atoms[sj] + c2] +=
                                        m[a][c2] * r.get(b);
                                    out.mixed[(3 * a + b) * dim + 3 * atoms[si] + c2] -=
                                        m[c2][a] * r.get(b);
                                }
                            }
                        }
                    }
                    // The displacement is applied before the
                    // strain: the gradient of the quartet enters
                    // the mixed derivative.
                    let first = eri_quartet_center_derivatives(shells, kernel);
                    for (centre, per_axis) in first.iter().enumerate() {
                        for (a, quartet) in per_axis.iter().enumerate() {
                            let mut gradient = 0.0;
                            let mut weight = weights.iter();
                            for i in 0..quartet.na {
                                for k in 0..quartet.nb {
                                    for l in 0..quartet.nc {
                                        for j in 0..quartet.nd {
                                            gradient +=
                                                weight.next().expect("one weight per element")
                                                    * quartet.get(i, k, l, j);
                                        }
                                    }
                                }
                            }
                            for b in 0..3 {
                                out.mixed[(3 * a + b) * dim + 3 * atoms[centre] + b] += gradient;
                            }
                        }
                    }
                },
            );
            Some(out)
        })
        .collect();
    let mut total = StrainSecond::zeros(natoms);
    for partial in partials {
        total.add(&partial);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::periodic::exchange::exchange_image_blocks;
    use crate::periodic::gradient::real_two_electron_gradient;
    use crate::periodic::hessian::tests::{check_columns, images, lattice, system, BASIS, STEPS};
    use crate::periodic::Cell;

    fn density_at(density: &[ImageBlock], t: Vec3) -> Option<&Matrix> {
        density
            .iter()
            .find(|block| (block.translation - t).norm() < 1e-8)
            .map(|block| &block.matrix)
    }

    /// Reaches the two shortest lattice vectors of the test cell.
    const CUTOFF: f64 = 4.7;

    fn cell(atom: usize, axis: usize, h: f64) -> Cell {
        let (molecule, _) = system(atom, axis, h);
        let set = crate::basis::g94::parse("exchange-response-test", BASIS).unwrap();
        Cell::new(molecule, lattice(), &set, CUTOFF).unwrap()
    }

    fn kernels() -> [(&'static str, CoulombKernel); 2] {
        [
            ("full-range", CoulombKernel::Coulomb),
            ("long-range", CoulombKernel::Erf { omega: 0.4 }),
        ]
    }

    fn flat_gradient(cell: &Cell, density: &[ImageBlock], kernel: CoulombKernel) -> Vec<f64> {
        let (gradient, _) = real_two_electron_gradient(cell, density, density, CUTOFF, kernel, 1.0);
        gradient.iter().flat_map(|g| [g.x, g.y, g.z]).collect()
    }

    fn contraction(density: &[ImageBlock], operator: &[ImageBlock]) -> f64 {
        operator
            .iter()
            .map(|block| {
                density_at(density, block.translation)
                    .map_or(0.0, |d| d.frobenius_dot(&block.matrix))
            })
            .sum()
    }

    #[test]
    fn second_derivatives_pass_four_step_fits_against_the_gradient() {
        let reference = cell(0, 0, 0.0);
        let density = images(&reference.lattice, reference.nao());
        for (label, kernel) in kernels() {
            let hessian = exchange_second_derivatives(
                &reference.basis,
                &reference.lattice,
                &density,
                CUTOFF,
                kernel,
                2,
            );
            check_columns(
                &format!("exchange second derivatives ({label})"),
                &hessian,
                |atom, axis, h| flat_gradient(&cell(atom, axis, h), &density, kernel),
            );
        }
    }

    #[test]
    fn source_matrices_pass_four_step_fits_against_the_exchange_matrix() {
        let reference = cell(0, 0, 0.0);
        let density = images(&reference.lattice, reference.nao());
        for (label, kernel) in kernels() {
            let first = exchange_first_order(
                &reference.basis,
                &reference.lattice,
                &density,
                CUTOFF,
                kernel,
                2,
            );
            for (atom, axis) in [(0usize, 0usize), (1, 2)] {
                let x = 3 * atom + axis;
                let exchange = |h: f64| {
                    let displaced = cell(atom, axis, h);
                    exchange_image_blocks(
                        &displaced.basis,
                        &displaced.lattice,
                        &density,
                        CUTOFF,
                        kernel,
                    )
                };
                let errors: Vec<f64> = STEPS
                    .iter()
                    .map(|&h| {
                        let (plus, minus) = (exchange(h), exchange(-h));
                        let mut worst = 0.0_f64;
                        for block in &first.source[x] {
                            let p = density_at(&plus, block.translation);
                            let m = density_at(&minus, block.translation);
                            for i in 0..block.matrix.rows {
                                for j in 0..block.matrix.cols {
                                    let fd = (p.map_or(0.0, |v| v[(i, j)])
                                        - m.map_or(0.0, |v| v[(i, j)]))
                                        / (2.0 * h);
                                    worst = worst.max((fd - block.matrix[(i, j)]).abs());
                                }
                            }
                        }
                        worst
                    })
                    .collect();
                let (lx, ly): (Vec<f64>, Vec<f64>) = STEPS
                    .iter()
                    .zip(&errors)
                    .map(|(h, e)| (h.ln(), e.ln()))
                    .unzip();
                let count = lx.len() as f64;
                let (mx, my) = (
                    lx.iter().sum::<f64>() / count,
                    ly.iter().sum::<f64>() / count,
                );
                let sxy: f64 = lx.iter().zip(&ly).map(|(a, b)| (a - mx) * (b - my)).sum();
                let sxx: f64 = lx.iter().map(|a| (a - mx).powi(2)).sum();
                let syy: f64 = ly.iter().map(|b| (b - my).powi(2)).sum();
                let (p, r2) = (sxy / sxx, sxy * sxy / (sxx * syy));
                println!(
                    "exchange matrix derivative ({label}) displacement ({atom},{axis}): errors {errors:?}, h 0.0005..0.004, p {p:.6}, R^2 {r2:.8}"
                );
                assert!((p - 2.0).abs() < 0.05 && r2 > 0.9999);
            }
        }
    }

    #[test]
    fn pairing_matrices_are_the_density_derivative_of_the_gradient() {
        // The gradient is quadratic in the density, so its change along a
        // density direction is exact for any step.
        let reference = cell(0, 0, 0.0);
        let density = images(&reference.lattice, reference.nao());
        let direction: Vec<ImageBlock> = density
            .iter()
            .map(|block| ImageBlock {
                translation: block.translation,
                matrix: Matrix::from_row_major(
                    block.matrix.rows,
                    block.matrix.cols,
                    block
                        .matrix
                        .as_slice()
                        .iter()
                        .map(|v| 0.1 * (3.0 * v).sin())
                        .collect(),
                ),
            })
            .collect();
        let moved = |sign: f64| -> Vec<ImageBlock> {
            density
                .iter()
                .zip(&direction)
                .map(|(d, e)| ImageBlock {
                    translation: d.translation,
                    matrix: Matrix::from_row_major(
                        d.matrix.rows,
                        d.matrix.cols,
                        d.matrix
                            .as_slice()
                            .iter()
                            .zip(e.matrix.as_slice())
                            .map(|(a, b)| a + sign * b)
                            .collect(),
                    ),
                })
                .collect()
        };
        for (label, kernel) in kernels() {
            let first = exchange_first_order(
                &reference.basis,
                &reference.lattice,
                &density,
                CUTOFF,
                kernel,
                2,
            );
            let plus = flat_gradient(&reference, &moved(1.0), kernel);
            let minus = flat_gradient(&reference, &moved(-1.0), kernel);
            let mut worst = 0.0_f64;
            let mut truncation = 0.0_f64;
            let mut scale = 0.0_f64;
            for x in 0..6 {
                let expected = 0.5 * (plus[x] - minus[x]);
                let found = 2.0 * contraction(&direction, &first.pairing[x]);
                let from_source = 2.0 * contraction(&direction, &first.source[x]);
                worst = worst.max((found - expected).abs());
                truncation = truncation.max((from_source - expected).abs());
                scale = scale.max(expected.abs());
            }
            println!(
                "exchange gradient density derivative ({label}): largest value {scale:e}, pairing error {worst:e}, error of the exchange-matrix derivative in its place {truncation:e}"
            );
            assert!(worst < 1e-11 * scale.max(1.0));
        }
    }
}
