// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Skala contributions to the analytic Hessian at fixed density matrices.
//!
//! The energy is `E = sum_p w_p e_p`, with the energy density `e_p` a
//! non-local function of the densities at all grid points of an atom. Grid
//! points ride with their atom, so every field at a point (spin densities,
//! their gradients, kinetic energy densities, partition weight) has total
//! derivatives `D_x` with respect to the nuclear coordinates. With
//! `a_p = dE/drho_p`, `b_p = dE/dgrad rho_p` and `c_p = dE/dtau_p`
//!
//! ```text
//! D_x E     = sum_p [a D_x rho + b . D_x grad rho + c D_x tau + e D_x w]
//! D_y D_x E = sum_p [D_y a D_x rho + D_y b . D_x grad rho + D_y c D_x tau + D_y e D_x w]
//!           + sum_p [a D_y D_x rho + b . D_y D_x grad rho + c D_y D_x tau + e D_y D_x w]
//! ```
//!
//! The derivatives `D_y a`, `D_y b`, `D_y c` and `D_y e` of the first line
//! are directional derivatives of the reverse pass of the network along the
//! field changes `D_y (rho, grad rho, tau, w)`: one evaluation per nuclear
//! coordinate on numbers that carry a tangent, exact for the non-local
//! layers. The second line holds the second derivatives of the fields (atomic
//! orbital derivatives up to third order) and of the partition weights.
//!
//! A field at a point of atom `O` depends on the positions only through
//! `R_O - R_B` for the other atoms `B`, so its derivative with respect to
//! `R_B` moves the orbitals of `B` at a fixed point and its derivative with
//! respect to `R_O` is minus the sum over the other atoms.
//!
//! The derivative of the Kohn-Sham matrix at fixed density matrices has the
//! same two parts: the matrix assembled from `D_y (a, b, c)`, and the
//! product rule on the orbital values at the moving points.

use crate::basis::MolBasis;
use crate::error::Result;
use crate::grid::eval::{eval_aos_deriv3, third_index};
use crate::grid::{becke_weight_gradient, MolecularGrid};
use crate::integrals::deriv_mat::AtomDerivs;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::skala::{
    assemble, batch_ranges, grid_geometry, SkalaInput, SkalaModel, SkalaOutput, Tangent,
};
use crate::system::Molecule;
use rayon::prelude::*;

use super::xc_terms::{MolecularWeights, WeightResponse, XcHessianTerms};

/// Packed second-derivative index (xx, xy, xz, yy, yz, zz).
const H6: [[usize; 3]; 3] = [[0, 1, 2], [1, 3, 4], [2, 4, 5]];

/// Fields at the grid points and their total first derivatives.
struct Fields {
    rho: [Vec<f64>; 2],
    gradient: [Vec<[f64; 3]>; 2],
    tau: [Vec<f64>; 2],
    /// `[coordinate][spin][point]`.
    d_rho: Vec<[Vec<f64>; 2]>,
    d_gradient: Vec<[Vec<[f64; 3]>; 2]>,
    d_tau: Vec<[Vec<f64>; 2]>,
    /// Derivatives of the partitioned weights, `[coordinate][point]`.
    d_weight: Vec<Vec<f64>>,
}

impl Fields {
    fn new(dim: usize) -> Self {
        Self {
            rho: [Vec::new(), Vec::new()],
            gradient: [Vec::new(), Vec::new()],
            tau: [Vec::new(), Vec::new()],
            d_rho: (0..dim).map(|_| [Vec::new(), Vec::new()]).collect(),
            d_gradient: (0..dim).map(|_| [Vec::new(), Vec::new()]).collect(),
            d_tau: (0..dim).map(|_| [Vec::new(), Vec::new()]).collect(),
            d_weight: vec![Vec::new(); dim],
        }
    }

    fn append(&mut self, mut other: Fields) {
        for spin in 0..2 {
            self.rho[spin].append(&mut other.rho[spin]);
            self.gradient[spin].append(&mut other.gradient[spin]);
            self.tau[spin].append(&mut other.tau[spin]);
        }
        for x in 0..self.d_weight.len() {
            for spin in 0..2 {
                self.d_rho[x][spin].append(&mut other.d_rho[x][spin]);
                self.d_gradient[x][spin].append(&mut other.d_gradient[x][spin]);
                self.d_tau[x][spin].append(&mut other.d_tau[x][spin]);
            }
            self.d_weight[x].append(&mut other.d_weight[x]);
        }
    }
}

/// Fields and first derivatives of one batch of points.
fn batch_fields(
    mb: &MolBasis,
    grid: &MolecularGrid,
    p: [&Matrix; 2],
    centers: &[Vec3],
    aij: &[f64],
    ao_atom: &[usize],
    start: usize,
    end: usize,
) -> Fields {
    let natoms = centers.len();
    let dim = 3 * natoms;
    let nao = mb.nao_sph;
    let npts = end - start;
    let pts = &grid.points[start..end];
    let batch = eval_aos_deriv3(mb, pts, true, true, false);
    let g = batch.grad.as_ref().expect("gradients were requested");
    let h = batch
        .hess
        .as_ref()
        .expect("second derivatives were requested");
    let mut out = Fields::new(dim);
    for x in 0..dim {
        out.d_weight[x] = vec![0.0; npts];
        for spin in 0..2 {
            out.d_rho[x][spin] = vec![0.0; npts];
            out.d_gradient[x][spin] = vec![[0.0; 3]; npts];
            out.d_tau[x][spin] = vec![0.0; npts];
        }
    }
    for spin in 0..2 {
        let x = batch.phi.matmul_seq(p[spin]);
        let y: [Matrix; 3] = std::array::from_fn(|axis| g[axis].matmul_seq(p[spin]));
        out.rho[spin] = vec![0.0; npts];
        out.gradient[spin] = vec![[0.0; 3]; npts];
        out.tau[spin] = vec![0.0; npts];
        for i in 0..npts {
            let owner = grid.atom_of[start + i];
            let mut rho = 0.0;
            let mut gradient = [0.0_f64; 3];
            let mut kinetic = 0.0;
            // Derivatives with the orbitals of one atom moving at a fixed
            // point: `[atom][axis]` and, for the gradient, `[component]`.
            let mut bare_rho = vec![[0.0_f64; 3]; natoms];
            let mut bare_gradient = vec![[[0.0_f64; 3]; 3]; natoms];
            let mut bare_tau = vec![[0.0_f64; 3]; natoms];
            for mu in 0..nao {
                let xv = x[(i, mu)];
                rho += xv * batch.phi[(i, mu)];
                for u in 0..3 {
                    gradient[u] += 2.0 * xv * g[u][(i, mu)];
                    kinetic += 0.5 * y[u][(i, mu)] * g[u][(i, mu)];
                }
                let atom = ao_atom[mu];
                if atom == owner {
                    continue;
                }
                for a in 0..3 {
                    let first = g[a][(i, mu)];
                    bare_rho[atom][a] -= 2.0 * first * xv;
                    for u in 0..3 {
                        let second = h[H6[a][u]][(i, mu)];
                        bare_gradient[atom][a][u] -= 2.0 * (second * xv + first * y[u][(i, mu)]);
                        bare_tau[atom][a] -= second * y[u][(i, mu)];
                    }
                }
            }
            out.rho[spin][i] = rho;
            out.gradient[spin][i] = gradient;
            out.tau[spin][i] = kinetic;
            for atom in 0..natoms {
                if atom == owner {
                    continue;
                }
                for a in 0..3 {
                    out.d_rho[3 * atom + a][spin][i] = bare_rho[atom][a];
                    out.d_rho[3 * owner + a][spin][i] -= bare_rho[atom][a];
                    out.d_tau[3 * atom + a][spin][i] = bare_tau[atom][a];
                    out.d_tau[3 * owner + a][spin][i] -= bare_tau[atom][a];
                    for u in 0..3 {
                        out.d_gradient[3 * atom + a][spin][i][u] = bare_gradient[atom][a][u];
                        out.d_gradient[3 * owner + a][spin][i][u] -= bare_gradient[atom][a][u];
                    }
                }
            }
        }
    }
    for i in 0..npts {
        let owner = grid.atom_of[start + i];
        let raw = grid.raw_weights[start + i];
        let (_, slopes) = becke_weight_gradient(centers, aij, natoms, owner, pts[i]);
        let mut sum = Vec3::zero();
        for slope in &slopes {
            sum += *slope;
        }
        for (atom, slope) in slopes.iter().enumerate() {
            for a in 0..3 {
                let riding = if atom == owner { sum.get(a) } else { 0.0 };
                out.d_weight[3 * atom + a][i] = raw * (slope.get(a) - riding);
            }
        }
    }
    out
}

/// Second-derivative part of one batch: its contribution to the skeleton
/// (`dim x dim`, row-major) and the product-rule part of the derivative of
/// the Kohn-Sham matrices, `[coordinate][spin]`.
#[allow(clippy::too_many_arguments)]
fn batch_second(
    mb: &MolBasis,
    grid: &MolecularGrid,
    p: [&Matrix; 2],
    energy: &SkalaOutput,
    weights: &MolecularWeights<'_>,
    ao_atom: &[usize],
    members: &[Vec<usize>],
    start: usize,
    end: usize,
) -> (Vec<f64>, Vec<[Matrix; 2]>) {
    let natoms = members.len();
    let dim = 3 * natoms;
    let nao = mb.nao_sph;
    let npts = end - start;
    let pts = &grid.points[start..end];
    let batch = eval_aos_deriv3(mb, pts, true, true, true);
    let g = batch.grad.as_ref().expect("gradients were requested");
    let h = batch
        .hess
        .as_ref()
        .expect("second derivatives were requested");
    let t = batch
        .third
        .as_ref()
        .expect("third derivatives were requested");
    let mut skeleton = vec![0.0_f64; dim * dim];
    let mut matrices: Vec<[Matrix; 2]> = (0..dim)
        .map(|_| [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)])
        .collect();
    let mut k = vec![0.0_f64; dim * dim];
    for spin in 0..2 {
        let density = p[spin];
        let x = batch.phi.matmul_seq(density);
        let y: [Matrix; 3] = std::array::from_fn(|axis| g[axis].matmul_seq(density));
        // V = sum_p phi1 (a phi2 + b . G2) + G1 . (b phi2 + (c / 2) G2).
        let mut a1 = Matrix::zeros(npts, nao);
        let mut a2: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(npts, nao));
        for i in 0..npts {
            let point = start + i;
            let dr = energy.d_rho[spin][point];
            let dg = energy.d_gradient[spin][point];
            let dt = energy.d_tau[spin][point];
            for nu in 0..nao {
                let value = batch.phi[(i, nu)];
                a1[(i, nu)] = dr * value
                    + dg[0] * g[0][(i, nu)]
                    + dg[1] * g[1][(i, nu)]
                    + dg[2] * g[2][(i, nu)];
                for u in 0..3 {
                    a2[u][(i, nu)] = dg[u] * value + 0.5 * dt * g[u][(i, nu)];
                }
            }
        }
        for (coordinate, target) in matrices.iter_mut().enumerate() {
            let (atom, b) = (coordinate / 3, coordinate % 3);
            let mut moved = Matrix::zeros(npts, nao);
            let mut moved_gradient: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(npts, nao));
            let mut any = false;
            for i in 0..npts {
                let riding = grid.atom_of[start + i] == atom;
                for mu in 0..nao {
                    let sign = match (riding, ao_atom[mu] == atom) {
                        (true, false) => 1.0,
                        (false, true) => -1.0,
                        _ => continue,
                    };
                    any = true;
                    moved[(i, mu)] = sign * g[b][(i, mu)];
                    for u in 0..3 {
                        moved_gradient[u][(i, mu)] = sign * h[H6[b][u]][(i, mu)];
                    }
                }
            }
            if !any {
                continue;
            }
            let mut m = moved.transpose_matmul_seq(&a1);
            for u in 0..3 {
                let part = moved_gradient[u].transpose_matmul_seq(&a2[u]);
                for (value, extra) in m.as_mut_slice().iter_mut().zip(part.as_slice()) {
                    *value += *extra;
                }
            }
            for mu in 0..nao {
                for nu in 0..nao {
                    target[spin][(mu, nu)] += m[(mu, nu)] + m[(nu, mu)];
                }
            }
        }
        // Second derivatives of the fields contracted with (a, b, c).
        for i in 0..npts {
            let point = start + i;
            let owner = grid.atom_of[point];
            let dr = energy.d_rho[spin][point];
            let dg = energy.d_gradient[spin][point];
            let dt = energy.d_tau[spin][point];
            k.iter_mut().for_each(|value| *value = 0.0);
            for xa in 0..natoms {
                if xa == owner {
                    continue;
                }
                // Orbitals of `xa` contracted with the density matrix.
                let mut gp = [vec![0.0_f64; nao], vec![0.0_f64; nao], vec![0.0_f64; nao]];
                let mut hp: [[Vec<f64>; 3]; 3] =
                    std::array::from_fn(|_| std::array::from_fn(|_| vec![0.0_f64; nao]));
                for &mu in &members[xa] {
                    for nu in 0..nao {
                        let pv = density[(mu, nu)];
                        if pv == 0.0 {
                            continue;
                        }
                        for a in 0..3 {
                            gp[a][nu] += g[a][(i, mu)] * pv;
                            for u in 0..3 {
                                hp[a][u][nu] += h[H6[a][u]][(i, mu)] * pv;
                            }
                        }
                    }
                }
                for ya in 0..natoms {
                    if ya == owner {
                        continue;
                    }
                    for a in 0..3 {
                        for b in 0..3 {
                            let mut gg = 0.0;
                            let mut hg = [0.0_f64; 3];
                            let mut hh = 0.0;
                            for &nu in &members[ya] {
                                let first = g[b][(i, nu)];
                                gg += gp[a][nu] * first;
                                for u in 0..3 {
                                    hg[u] += hp[a][u][nu] * first;
                                    hh += hp[a][u][nu] * h[H6[b][u]][(i, nu)];
                                }
                            }
                            let cross = 2.0 * (dg[0] * hg[0] + dg[1] * hg[1] + dg[2] * hg[2]);
                            k[(3 * xa + a) * dim + 3 * ya + b] += 2.0 * dr * gg + cross + dt * hh;
                            k[(3 * ya + b) * dim + 3 * xa + a] += cross;
                        }
                    }
                }
                for a in 0..3 {
                    for b in 0..3 {
                        let mut value = 0.0;
                        for &mu in &members[xa] {
                            let second = h[H6[a][b]][(i, mu)];
                            let xv = x[(i, mu)];
                            value += 2.0 * dr * second * xv;
                            for u in 0..3 {
                                let third = t[third_index(a, b, u)][(i, mu)];
                                let yv = y[u][(i, mu)];
                                value += 2.0 * dg[u] * (third * xv + second * yv) + dt * third * yv;
                            }
                        }
                        k[(3 * xa + a) * dim + 3 * xa + b] += value;
                    }
                }
            }
            // The owner coordinate is minus the sum over the other atoms.
            for xa in 0..natoms {
                if xa == owner {
                    continue;
                }
                for ya in 0..natoms {
                    if ya == owner {
                        continue;
                    }
                    for a in 0..3 {
                        for b in 0..3 {
                            let value = k[(3 * xa + a) * dim + 3 * ya + b];
                            skeleton[(3 * xa + a) * dim + 3 * ya + b] += value;
                            skeleton[(3 * owner + a) * dim + 3 * ya + b] -= value;
                            skeleton[(3 * xa + a) * dim + 3 * owner + b] -= value;
                            skeleton[(3 * owner + a) * dim + 3 * owner + b] += value;
                        }
                    }
                }
            }
        }
    }
    // Second derivatives of the partition weights.
    for i in 0..npts {
        let point = start + i;
        let scale = grid.raw_weights[point] * energy.energy_density[point];
        if scale == 0.0 {
            continue;
        }
        let (_, second) = weights.derivatives(pts[i], grid.atom_of[point]);
        for (target, value) in skeleton.iter_mut().zip(&second) {
            *target += scale * value;
        }
    }
    (skeleton, matrices)
}

/// Skala pieces of the analytic Hessian at fixed spin density matrices
/// `p`: the second derivatives of the energy and the derivatives of the
/// Kohn-Sham matrices with respect to the nuclear coordinates.
pub fn skala_hessian_terms(
    model: &SkalaModel,
    mol: &Molecule,
    mb: &MolBasis,
    grid: &MolecularGrid,
    p: [&Matrix; 2],
) -> Result<XcHessianTerms> {
    let natoms = mol.atoms.len();
    let dim = 3 * natoms;
    let nao = mb.nao_sph;
    let n = grid.len();
    let ranges = batch_ranges(n);
    let ao_atom = mb.sph_ao_atoms();
    let mut members = vec![Vec::new(); natoms];
    for (mu, atom) in ao_atom.iter().enumerate() {
        members[*atom].push(mu);
    }
    let centers: Vec<Vec3> = mol.atoms.iter().map(|atom| atom.position).collect();
    let aij = MolecularGrid::becke_aij(mol);
    let (points, positions) = grid_geometry(mol, grid);

    // Fields, their first derivatives and the reverse pass on reals.
    let parts: Vec<Fields> = ranges
        .par_iter()
        .map(|&(start, end)| batch_fields(mb, grid, p, &centers, &aij, &ao_atom, start, end))
        .collect();
    let mut fields = Fields::new(dim);
    for part in parts {
        fields.append(part);
    }
    let energy = model.evaluate(
        &SkalaInput {
            atom_of: &grid.atom_of,
            points: &points,
            centers: &positions,
            weights: &grid.weights,
            atomic_weights: &grid.raw_weights,
            rho: [&fields.rho[0], &fields.rho[1]],
            gradient: [&fields.gradient[0], &fields.gradient[1]],
            tau: [&fields.tau[0], &fields.tau[1]],
        },
        true,
    )?;

    // Second derivatives of the fields and of the weights.
    let weights = MolecularWeights {
        centers: &centers,
        aij: &aij,
    };
    let locals: Vec<(Vec<f64>, Vec<[Matrix; 2]>)> = ranges
        .par_iter()
        .map(|&(start, end)| {
            batch_second(
                mb, grid, p, &energy, &weights, &ao_atom, &members, start, end,
            )
        })
        .collect();
    let mut skeleton = Matrix::zeros(dim, dim);
    let mut derivatives: Vec<[Matrix; 2]> = (0..dim)
        .map(|_| [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)])
        .collect();
    for (part, matrices) in locals {
        for (target, value) in skeleton.as_mut_slice().iter_mut().zip(&part) {
            *target += *value;
        }
        for (target, source) in derivatives.iter_mut().zip(&matrices) {
            for spin in 0..2 {
                for (value, extra) in target[spin]
                    .as_mut_slice()
                    .iter_mut()
                    .zip(source[spin].as_slice())
                {
                    *value += *extra;
                }
            }
        }
    }

    // One tangent evaluation per nuclear coordinate.
    let joined = |value: f64, tangent: f64| Tangent {
        re: value,
        im: [tangent],
    };
    for y in 0..dim {
        let scalars = |values: &[f64], tangents: &[f64]| -> Vec<Tangent> {
            values
                .iter()
                .zip(tangents)
                .map(|(v, t)| joined(*v, *t))
                .collect()
        };
        let rho: [Vec<Tangent>; 2] =
            std::array::from_fn(|s| scalars(&fields.rho[s], &fields.d_rho[y][s]));
        let tau: [Vec<Tangent>; 2] =
            std::array::from_fn(|s| scalars(&fields.tau[s], &fields.d_tau[y][s]));
        let gradient: [Vec<[Tangent; 3]>; 2] = std::array::from_fn(|s| {
            fields.gradient[s]
                .iter()
                .zip(&fields.d_gradient[y][s])
                .map(|(v, t)| std::array::from_fn(|axis| joined(v[axis], t[axis])))
                .collect()
        });
        let moving = scalars(&grid.weights, &fields.d_weight[y]);
        let out = model.evaluate(
            &SkalaInput {
                atom_of: &grid.atom_of,
                points: &points,
                centers: &positions,
                weights: &moving,
                atomic_weights: &grid.raw_weights,
                rho: [&rho[0], &rho[1]],
                gradient: [&gradient[0], &gradient[1]],
                tau: [&tau[0], &tau[1]],
            },
            true,
        )?;
        let slopes = |values: &[Tangent]| -> Vec<f64> { values.iter().map(|v| v.im[0]).collect() };
        let v_rho = [slopes(&out.d_rho[0]), slopes(&out.d_rho[1])];
        let v_tau = [slopes(&out.d_tau[0]), slopes(&out.d_tau[1])];
        let v_gradient: [Vec<[f64; 3]>; 2] = std::array::from_fn(|s| {
            out.d_gradient[s]
                .iter()
                .map(|g| [g[0].im[0], g[1].im[0], g[2].im[0]])
                .collect()
        });
        let v_energy = slopes(&out.energy_density);
        let column: Vec<f64> = (0..dim)
            .into_par_iter()
            .map(|x| {
                let mut sum = 0.0;
                for point in 0..n {
                    for s in 0..2 {
                        let dg = fields.d_gradient[x][s][point];
                        let vg = v_gradient[s][point];
                        sum += v_rho[s][point] * fields.d_rho[x][s][point]
                            + vg[0] * dg[0]
                            + vg[1] * dg[1]
                            + vg[2] * dg[2]
                            + v_tau[s][point] * fields.d_tau[x][s][point];
                    }
                    sum += v_energy[point] * fields.d_weight[x][point];
                }
                sum
            })
            .collect();
        for (x, value) in column.iter().enumerate() {
            skeleton[(x, y)] += *value;
        }
        let kernel = assemble(
            mb,
            grid,
            &ranges,
            [&v_rho[0], &v_rho[1]],
            [&v_gradient[0], &v_gradient[1]],
            [&v_tau[0], &v_tau[1]],
        );
        for spin in 0..2 {
            for (value, extra) in derivatives[y][spin]
                .as_mut_slice()
                .iter_mut()
                .zip(kernel[spin].as_slice())
            {
                *value += *extra;
            }
        }
    }

    let mut vxc_deriv: [Vec<AtomDerivs>; 2] = [Vec::new(), Vec::new()];
    let mut remaining = derivatives.into_iter();
    for _ in 0..natoms {
        let mut per_spin: [Vec<Matrix>; 2] = [Vec::new(), Vec::new()];
        for _ in 0..3 {
            let [alpha, beta] = remaining.next().expect("one entry per coordinate");
            per_spin[0].push(alpha);
            per_spin[1].push(beta);
        }
        for (spin, matrices) in per_spin.into_iter().enumerate() {
            let [mx, my, mz]: [Matrix; 3] = matrices
                .try_into()
                .unwrap_or_else(|_| unreachable!("three axes per atom"));
            vxc_deriv[spin].push([mx, my, mz]);
        }
    }
    Ok(XcHessianTerms {
        skeleton,
        vxc_deriv,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;
    use crate::grid::GridOptions;
    use crate::skala::{random_tensors, skala_fock, skala_gradient, SkalaSizes};
    use crate::system::Atom;

    const SIZES: SkalaSizes = SkalaSizes {
        width: 12,
        channels: 4,
        shells: 3,
        correlation: 3,
        layers: 2,
        post: 5,
    };

    fn water(displaced: Option<(usize, usize, f64)>) -> Molecule {
        let mut positions = [[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]];
        if let Some((atom, axis, h)) = displaced {
            positions[atom][axis] += h;
        }
        Molecule::new(
            [8u8, 1, 1]
                .iter()
                .zip(positions)
                .map(|(z, r)| Atom {
                    z: *z,
                    position: Vec3::new(r[0], r[1], r[2]),
                })
                .collect(),
        )
    }

    fn setup(molecule: &Molecule) -> (MolBasis, MolecularGrid) {
        let basis = MolBasis::build(molecule, &data::load("sto-3g").unwrap()).unwrap();
        // Every point is kept: the atomic sums of the network are continuous
        // in the nuclear positions only on the complete atomic grids.
        let grid = MolecularGrid::build(
            molecule,
            &GridOptions {
                level: 1,
                weight_cutoff: 0.0,
                ..GridOptions::default()
            },
        )
        .unwrap();
        (basis, grid)
    }

    fn power_fit(h: &[f64], errors: &[f64]) -> (f64, f64) {
        let xs: Vec<f64> = h.iter().map(|v| v.ln()).collect();
        let ys: Vec<f64> = errors.iter().map(|v| v.ln()).collect();
        let n = xs.len() as f64;
        let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
        let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
        let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
        let syy: f64 = ys.iter().map(|y| (y - my).powi(2)).sum();
        (sxy / sxx, sxy * sxy / (sxx * syy))
    }

    /// Fixed symmetric density matrices of the two spins.
    fn densities(nao: usize) -> [Matrix; 2] {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        std::array::from_fn(|spin| {
            let mut m = Matrix::zeros(nao, nao);
            for i in 0..nao {
                for j in 0..=i {
                    let value = 0.08 * next()
                        + if i == j {
                            0.35 - 0.05 * spin as f64
                        } else {
                            0.0
                        };
                    m[(i, j)] = value;
                    m[(j, i)] = value;
                }
            }
            m
        })
    }

    #[test]
    fn skeleton_and_potential_derivatives_pass_a_four_step_fit() {
        let model = SkalaModel::from_tensors(&random_tensors(61, SIZES)).unwrap();
        let reference = water(None);
        let (basis, grid) = setup(&reference);
        let nao = basis.nao_sph;
        let p = densities(nao);
        let p_ref = [&p[0], &p[1]];
        let terms = skala_hessian_terms(&model, &reference, &basis, &grid, p_ref).unwrap();
        let dim = 9;
        let mut asymmetry = 0.0_f64;
        let mut scale = 0.0_f64;
        for x in 0..dim {
            for y in 0..dim {
                asymmetry = asymmetry.max((terms.skeleton[(x, y)] - terms.skeleton[(y, x)]).abs());
                scale = scale.max(terms.skeleton[(x, y)].abs());
            }
        }
        // Translational invariance of every column.
        let mut translation = 0.0_f64;
        for y in 0..dim {
            for axis in 0..3 {
                let sum: f64 = (0..3)
                    .map(|atom| terms.skeleton[(3 * atom + axis, y)])
                    .sum();
                translation = translation.max(sum.abs());
            }
        }
        println!(
            "skeleton: largest element {scale:e}, asymmetry {asymmetry:e}, translational sum {translation:e}"
        );
        assert!(asymmetry < 1e-9 * scale.max(1.0));
        assert!(translation < 1e-9 * scale.max(1.0));

        let at = |atom: usize, axis: usize, h: f64| {
            let molecule = water(Some((atom, axis, h)));
            let (basis, grid) = setup(&molecule);
            let gradient = skala_gradient(&model, &molecule, &basis, &grid, p_ref).unwrap();
            let matrices = skala_fock(&model, &molecule, &basis, &grid, p_ref)
                .unwrap()
                .1;
            (gradient, matrices)
        };
        let steps = [0.02, 0.01, 0.005, 0.0025];
        for (atom, axis) in [(0usize, 2usize), (1, 1), (2, 0)] {
            let y = 3 * atom + axis;
            let mut skeleton_errors = Vec::new();
            let mut matrix_errors = Vec::new();
            let mut skeleton_slopes = Vec::new();
            for &h in &steps {
                let (plus, minus) = (at(atom, axis, h), at(atom, axis, -h));
                let mut worst = 0.0_f64;
                let mut column = Vec::new();
                for x in 0..dim {
                    let slope = (plus.0[x / 3].get(x % 3) - minus.0[x / 3].get(x % 3)) / (2.0 * h);
                    worst = worst.max((slope - terms.skeleton[(x, y)]).abs());
                    column.push(slope);
                }
                skeleton_errors.push(worst);
                skeleton_slopes.push(column);
                let mut worst = 0.0_f64;
                for spin in 0..2 {
                    let analytic = &terms.vxc_deriv[spin][atom][axis];
                    for ((a, b), d) in plus.1[spin]
                        .as_slice()
                        .iter()
                        .zip(minus.1[spin].as_slice())
                        .zip(analytic.as_slice())
                    {
                        worst = worst.max(((a - b) / (2.0 * h) - d).abs());
                    }
                }
                matrix_errors.push(worst);
            }
            let (p_skeleton, r_skeleton) = power_fit(&steps, &skeleton_errors);
            let (p_matrix, r_matrix) = power_fit(&steps, &matrix_errors);
            let mut richardson = 0.0_f64;
            for x in 0..dim {
                let extrapolated = (4.0 * skeleton_slopes[3][x] - skeleton_slopes[2][x]) / 3.0;
                richardson = richardson.max((extrapolated - terms.skeleton[(x, y)]).abs());
            }
            println!(
                "coordinate ({atom},{axis}): steps {steps:?} skeleton column errors {skeleton_errors:?} p={p_skeleton:.6} R2={r_skeleton:.8} Richardson difference {richardson:e}; Kohn-Sham matrix derivative errors {matrix_errors:?} p={p_matrix:.6} R2={r_matrix:.8}"
            );
            assert!((p_skeleton - 2.0).abs() < 0.1 && r_skeleton > 0.999);
            assert!((p_matrix - 2.0).abs() < 0.1 && r_matrix > 0.999);
            assert!(richardson < 1e-6);
        }
    }
}
