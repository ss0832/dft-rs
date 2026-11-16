// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Second derivatives of a learned exchange-correlation functional in
//! periodic cells at fixed density images.
//!
//! The energy is `E = sum_p w_p e_p`: the energy density `e_p` is a non-local
//! function of the fields (spin densities, their gradients and kinetic energy
//! densities) at the grid points of one atom, on an atomic grid that moves
//! rigidly with its atom, and `w_p` is the periodic partition weight. With
//! `a = dE/drho`, `b = dE/dgrad rho` and `c = dE/dtau` of the reverse pass of
//! the network, the derivatives along two coordinates `x` and `y` (atomic
//! displacements at the Gamma point or strain components) are
//!
//! ```text
//! D_y D_x E = sum_p [D_y a D_x rho + D_y b . D_x grad rho + D_y c D_x tau + D_y e D_x w]
//!           + sum_p [a D_y D_x rho + b . D_y D_x grad rho + c D_y D_x tau + e D_y D_x w]
//! ```
//!
//! The fields and the weights of the local image basis are evaluated as
//! forward-mode jets in the coordinates by the batches of
//! [`super::xc_strain`]. The first line needs `D_y (a, b, c, e)`: the network
//! is evaluated once on numbers that carry the field derivatives of a group
//! of coordinates, which is exact for the non-local layers. The second line
//! is the second derivative of `sum_p [a rho + b . grad rho + c tau + e w]`
//! with `(a, b, c, e)` held at the reference, evaluated on nested jets.
//!
//! The derivative of the Kohn-Sham matrices at fixed density images has the
//! same two parts: the matrices `sum_p [phi_m (a phi_n + b . grad phi_n) +
//! grad phi_m . (b phi_n + c grad phi_n / 2)]` of the local image basis are
//! assembled with the orbitals and `(a, b, c)` as first-order jets and folded
//! into image blocks by the translation difference of the two orbitals.

use crate::basis::n_sph;
use crate::dual::{DualN, Jet2, Scalar};
use crate::error::Result;
use crate::linalg::Matrix;
use crate::periodic::grid::PeriodicGrid;
use crate::periodic::integrals::ImageBlock;
use crate::periodic::strain::StrainSecond;
use crate::periodic::strain_fock::{merge_strain_blocks, StrainBlock};
use crate::periodic::xc_hessian::PeriodicXcHessian;
use crate::periodic::xc_strain::{xc_batches, BatchJets};
use crate::periodic::Cell;
use crate::skala::{Number, SkalaInput, SkalaModel, SkalaOutput};
use crate::xc::ResolvedFunctional;

type First3 = DualN<f64, 3>;
type Second3 = DualN<First3, 3>;
type First9 = DualN<f64, 9>;
type Mixed = DualN<First9, 3>;

/// Fields at every grid point.
struct Fields<S> {
    rho: [Vec<S>; 2],
    gradient: [Vec<[S; 3]>; 2],
    tau: [Vec<S>; 2],
    weight: Vec<S>,
}

/// First derivatives of the fields at every point along one coordinate.
struct Moved {
    rho: [Vec<f64>; 2],
    gradient: [Vec<[f64; 3]>; 2],
    tau: [Vec<f64>; 2],
    weight: Vec<f64>,
}

/// First derivatives of the reverse pass of the network along one
/// coordinate: of `(a, b, c)` and of the energy density.
struct Slopes {
    d_rho: [Vec<f64>; 2],
    d_gradient: [Vec<[f64; 3]>; 2],
    d_tau: [Vec<f64>; 2],
    energy_density: Vec<f64>,
}

/// Field derivatives of the `N` coordinates of first-order jets.
fn moved<const N: usize>(fields: &Fields<DualN<f64, N>>) -> Vec<Moved> {
    (0..N)
        .map(|k| Moved {
            rho: std::array::from_fn(|s| fields.rho[s].iter().map(|v| v.im[k]).collect()),
            gradient: std::array::from_fn(|s| {
                fields.gradient[s]
                    .iter()
                    .map(|g| [g[0].im[k], g[1].im[k], g[2].im[k]])
                    .collect()
            }),
            tau: std::array::from_fn(|s| fields.tau[s].iter().map(|v| v.im[k]).collect()),
            weight: fields.weight.iter().map(|v| v.im[k]).collect(),
        })
        .collect()
}

/// `sum_p [D_y a D_x rho + D_y b . D_x grad rho + D_y c D_x tau + D_y e D_x w]`
/// with the field derivatives of `x` and the network derivatives of `y`.
fn pairing(x: &Moved, y: &Slopes) -> f64 {
    let mut sum = 0.0;
    for p in 0..x.weight.len() {
        for s in 0..2 {
            let (g, v) = (x.gradient[s][p], y.d_gradient[s][p]);
            sum += y.d_rho[s][p] * x.rho[s][p]
                + v[0] * g[0]
                + v[1] * g[1]
                + v[2] * g[2]
                + y.d_tau[s][p] * x.tau[s][p];
        }
        sum += y.energy_density[p] * x.weight[p];
    }
    sum
}

/// `(a, b, c)` of a point and spin as first-order jets: the reference values
/// with the derivatives along the `N` coordinates of `slopes`.
fn potential_jet<'a, const N: usize>(
    energy: &'a SkalaOutput,
    slopes: &'a [Slopes],
) -> impl Fn(usize, usize) -> (DualN<f64, N>, [DualN<f64, N>; 3], DualN<f64, N>) + Sync + 'a {
    move |p, s| {
        let a = DualN {
            re: energy.d_rho[s][p],
            im: std::array::from_fn(|k| slopes[k].d_rho[s][p]),
        };
        let b = std::array::from_fn(|axis| DualN {
            re: energy.d_gradient[s][p][axis],
            im: std::array::from_fn(|k| slopes[k].d_gradient[s][p][axis]),
        });
        let c = DualN {
            re: energy.d_tau[s][p],
            im: std::array::from_fn(|k| slopes[k].d_tau[s][p]),
        };
        (a, b, c)
    }
}

/// The displacement of `atom` as the three variables of a first-order jet.
fn moving(atom: usize) -> impl Fn(usize) -> [First3; 3] + Sync {
    move |other| {
        if other == atom {
            std::array::from_fn(|c| First3::variable(0.0, c))
        } else {
            [First3::constant(0.0); 3]
        }
    }
}

/// Strain components as the nine variables of a first-order jet.
fn strain_first() -> [[First9; 3]; 3] {
    std::array::from_fn(|a| std::array::from_fn(|b| First9::variable(0.0, 3 * a + b)))
}

/// A learned functional on the periodic quadrature of fixed density images.
struct Context<'a> {
    cell: &'a Cell,
    model: &'a SkalaModel,
    density: [&'a [ImageBlock]; 2],
    grid: PeriodicGrid,
    grid_shell: i64,
    ao_cutoff: f64,
    /// Selects the fields of the batches (densities, gradients and kinetic
    /// energy densities); it has no semilocal part of its own.
    carrier: ResolvedFunctional,
    points: Vec<[f64; 3]>,
    centers: Vec<[f64; 3]>,
}

impl<'a> Context<'a> {
    fn new(
        cell: &'a Cell,
        model: &'a SkalaModel,
        density: [&'a [ImageBlock]; 2],
        grid_options: &crate::grid::GridOptions,
        grid_shell: i64,
        ao_cutoff: f64,
    ) -> Result<Self> {
        let grid = PeriodicGrid::build(&cell.molecule, &cell.lattice, grid_options, grid_shell)?;
        let points = grid.points.iter().map(|p| p.to_array()).collect();
        let centers = cell
            .molecule
            .atoms
            .iter()
            .map(|atom| atom.position.to_array())
            .collect();
        Ok(Self {
            cell,
            model,
            density,
            grid,
            grid_shell,
            ao_cutoff,
            carrier: ResolvedFunctional::model_based("learned functional"),
            points,
            centers,
        })
    }

    /// Fields of every grid point as jets in the strain `eps` and the seeded
    /// atomic displacements.
    fn fields<S: Scalar + Send + Sync>(
        &self,
        eps: &[[S; 3]; 3],
        displacement: &(dyn Fn(usize) -> [S; 3] + Sync),
    ) -> Result<Fields<S>> {
        let parts = xc_batches(
            self.cell,
            &self.carrier,
            self.density,
            &self.grid,
            self.grid_shell,
            self.ao_cutoff,
            eps,
            displacement,
            &|batch: &BatchJets<S>| {
                (
                    batch.rho.clone(),
                    batch.grad.clone(),
                    batch.tau.clone(),
                    batch.weight.clone(),
                )
            },
        )?;
        let mut out = Fields {
            rho: [Vec::new(), Vec::new()],
            gradient: [Vec::new(), Vec::new()],
            tau: [Vec::new(), Vec::new()],
            weight: Vec::new(),
        };
        for (rho, gradient, tau, weight) in parts {
            for i in 0..weight.len() {
                for spin in 0..2 {
                    out.rho[spin].push(rho[i][spin]);
                    out.gradient[spin].push(gradient[i][spin]);
                    out.tau[spin].push(tau[i][spin]);
                }
                out.weight.push(weight[i]);
            }
        }
        Ok(out)
    }

    /// The network with its reverse pass on the fields. Quadrature roundoff
    /// below zero is removed from the densities as in the self-consistent
    /// evaluation.
    fn evaluate<S: Number>(&self, fields: &Fields<S>) -> Result<SkalaOutput<S>> {
        let rho: [Vec<S>; 2] = std::array::from_fn(|s| {
            fields.rho[s]
                .iter()
                .map(|v| {
                    let value = v.value();
                    if value < 0.0 && value > -1e-12 {
                        *v - S::from_f64(value)
                    } else {
                        *v
                    }
                })
                .collect()
        });
        self.model.evaluate(
            &SkalaInput {
                atom_of: &self.grid.atom_of,
                points: &self.points,
                centers: &self.centers,
                weights: &fields.weight,
                atomic_weights: &self.grid.raw_weights,
                rho: [&rho[0], &rho[1]],
                gradient: [&fields.gradient[0], &fields.gradient[1]],
                tau: [&fields.tau[0], &fields.tau[1]],
            },
            true,
        )
    }

    /// The reverse pass at the reference and its derivatives along the `N`
    /// coordinates of first-order field jets, from one evaluation.
    fn linearize<const N: usize>(
        &self,
        fields: &Fields<DualN<f64, N>>,
    ) -> Result<(SkalaOutput, Vec<Slopes>)> {
        let out = self.evaluate(fields)?;
        let energy = SkalaOutput {
            energy: out.energy.re,
            energy_density: out.energy_density.iter().map(|v| v.re).collect(),
            d_rho: std::array::from_fn(|s| out.d_rho[s].iter().map(|v| v.re).collect()),
            d_gradient: std::array::from_fn(|s| {
                out.d_gradient[s]
                    .iter()
                    .map(|g| [g[0].re, g[1].re, g[2].re])
                    .collect()
            }),
            d_tau: std::array::from_fn(|s| out.d_tau[s].iter().map(|v| v.re).collect()),
        };
        let slopes = (0..N)
            .map(|k| Slopes {
                d_rho: std::array::from_fn(|s| out.d_rho[s].iter().map(|v| v.im[k]).collect()),
                d_gradient: std::array::from_fn(|s| {
                    out.d_gradient[s]
                        .iter()
                        .map(|g| [g[0].im[k], g[1].im[k], g[2].im[k]])
                        .collect()
                }),
                d_tau: std::array::from_fn(|s| out.d_tau[s].iter().map(|v| v.im[k]).collect()),
                energy_density: out.energy_density.iter().map(|v| v.im[k]).collect(),
            })
            .collect();
        Ok((energy, slopes))
    }

    /// `sum_p [a rho + b . grad rho + c tau + e w]` with the fields as jets
    /// and the reverse pass `energy` held at the reference.
    fn frozen<S: Scalar + Send + Sync>(
        &self,
        energy: &SkalaOutput,
        eps: &[[S; 3]; 3],
        displacement: &(dyn Fn(usize) -> [S; 3] + Sync),
    ) -> Result<S> {
        let parts = xc_batches(
            self.cell,
            &self.carrier,
            self.density,
            &self.grid,
            self.grid_shell,
            self.ao_cutoff,
            eps,
            displacement,
            &|batch: &BatchJets<S>| {
                let mut sum = S::from_f64(0.0);
                for i in 0..batch.weight.len() {
                    let p = batch.start + i;
                    for spin in 0..2 {
                        sum = sum
                            + batch.rho[i][spin].scale(energy.d_rho[spin][p])
                            + batch.tau[i][spin].scale(energy.d_tau[spin][p]);
                        for axis in 0..3 {
                            sum = sum
                                + batch.grad[i][spin][axis].scale(energy.d_gradient[spin][p][axis]);
                        }
                    }
                    sum = sum + batch.weight[i].scale(energy.energy_density[p]);
                }
                sum
            },
        )?;
        Ok(parts.into_iter().fold(S::from_f64(0.0), |a, b| a + b))
    }

    /// Kohn-Sham matrices of the local image basis as jets, with `(a, b, c)`
    /// of a point and spin from `potential`, scattered into image blocks
    /// with `components` matrices from `extract(value, component)`.
    fn potential_jets<S: Scalar + Send + Sync>(
        &self,
        eps: &[[S; 3]; 3],
        displacement: &(dyn Fn(usize) -> [S; 3] + Sync),
        potential: &(dyn Fn(usize, usize) -> (S, [S; 3], S) + Sync),
        components: usize,
        extract: &(dyn Fn(S, usize) -> f64 + Sync),
    ) -> Result<[Vec<StrainBlock>; 2]> {
        let basis = &self.cell.basis;
        let n = basis.nao_sph;
        let per_batch = xc_batches(
            self.cell,
            &self.carrier,
            self.density,
            &self.grid,
            self.grid_shell,
            self.ao_cutoff,
            eps,
            displacement,
            &|batch: &BatchJets<S>| {
                let m = batch.m;
                let zero = S::from_f64(0.0);
                let mut local: [Vec<S>; 2] = [vec![zero; m * m], vec![zero; m * m]];
                for i in 0..batch.weight.len() {
                    let phi = &batch.phi[i];
                    let dphi = &batch.dphi[i];
                    let list = &batch.significant[i];
                    for (spin, target) in local.iter_mut().enumerate() {
                        let (a, b, c) = potential(batch.start + i, spin);
                        let half = c.scale(0.5);
                        // A_m = a phi_m / 2 + b . grad phi_m.
                        let mut left: Vec<S> = vec![zero; m];
                        for &mu in list {
                            left[mu] = (a * phi[mu]).scale(0.5)
                                + b[0] * dphi[mu][0]
                                + b[1] * dphi[mu][1]
                                + b[2] * dphi[mu][2];
                        }
                        for &mu in list {
                            for &nu in list {
                                target[mu * m + nu] = target[mu * m + nu]
                                    + left[mu] * phi[nu]
                                    + phi[mu] * left[nu]
                                    + half
                                        * (dphi[mu][0] * dphi[nu][0]
                                            + dphi[mu][1] * dphi[nu][1]
                                            + dphi[mu][2] * dphi[nu][2]);
                            }
                        }
                    }
                }
                // Scatter into image blocks at T_b - T_a.
                let mut blocks: [Vec<StrainBlock>; 2] = [Vec::new(), Vec::new()];
                for spin in 0..2 {
                    for (a, (sa, ta)) in batch.local.iter().enumerate() {
                        let home_a = &basis.shells[*sa];
                        for (b, (sb, tb)) in batch.local.iter().enumerate() {
                            let home_b = &basis.shells[*sb];
                            let t = *tb - *ta;
                            let index = match blocks[spin]
                                .iter()
                                .position(|x| (x.translation - t).norm() < 1e-8)
                            {
                                Some(index) => index,
                                None => {
                                    blocks[spin]
                                        .push(StrainBlock::with_components(t, n, components));
                                    blocks[spin].len() - 1
                                }
                            };
                            let block = &mut blocks[spin][index];
                            for i in 0..n_sph(home_a.l) {
                                for j in 0..n_sph(home_b.l) {
                                    let v = local[spin]
                                        [(batch.offsets[a] + i) * m + batch.offsets[b] + j];
                                    for k in 0..components {
                                        block.d[k][(home_a.sph_off + i, home_b.sph_off + j)] +=
                                            extract(v, k);
                                    }
                                }
                            }
                        }
                    }
                }
                blocks
            },
        )?;
        let mut out: [Vec<StrainBlock>; 2] = [Vec::new(), Vec::new()];
        for [alpha, beta] in per_batch {
            out[0].extend(alpha);
            out[1].extend(beta);
            // Merge as we go to bound the memory.
            out[0] = merge_strain_blocks(std::mem::take(&mut out[0]));
            out[1] = merge_strain_blocks(std::mem::take(&mut out[1]));
        }
        Ok(out)
    }
}

/// Spin Kohn-Sham image blocks `<m(0)| v_s |n(T)>` of the learned functional
/// for fixed density images (which must reach twice `ao_cutoff`). Their
/// Bloch sums are the matrices of the self-consistent calculation.
pub(crate) fn learned_potential_blocks(
    cell: &Cell,
    model: &SkalaModel,
    density: [&[ImageBlock]; 2],
    grid_options: &crate::grid::GridOptions,
    grid_shell: i64,
    ao_cutoff: f64,
) -> Result<[Vec<ImageBlock>; 2]> {
    let context = Context::new(cell, model, density, grid_options, grid_shell, ao_cutoff)?;
    let eps = [[0.0_f64; 3]; 3];
    let still = |_: usize| [0.0_f64; 3];
    let energy = context.evaluate(&context.fields(&eps, &still)?)?;
    let potential = |p: usize, s: usize| {
        (
            energy.d_rho[s][p],
            energy.d_gradient[s][p],
            energy.d_tau[s][p],
        )
    };
    let blocks = context.potential_jets(&eps, &still, &potential, 1, &|v: f64, _| v)?;
    Ok(blocks.map(|list| {
        list.into_iter()
            .map(|mut b| ImageBlock {
                translation: b.translation,
                matrix: b.d.swap_remove(0),
            })
            .collect()
    }))
}

/// Gamma-point displacement terms of the learned functional at fixed density
/// images: the second derivatives of the energy and the derivatives of the
/// Kohn-Sham image blocks, for `x = 3 atom + axis`.
pub(crate) fn learned_displacement_terms(
    cell: &Cell,
    model: &SkalaModel,
    density: [&[ImageBlock]; 2],
    grid_options: &crate::grid::GridOptions,
    grid_shell: i64,
    ao_cutoff: f64,
) -> Result<PeriodicXcHessian> {
    let context = Context::new(cell, model, density, grid_options, grid_shell, ao_cutoff)?;
    let natoms = cell.molecule.len();
    let dim = 3 * natoms;
    let unstrained = [[First3::constant(0.0); 3]; 3];
    let mut reference: Option<SkalaOutput> = None;
    let mut fields_moved: Vec<Moved> = Vec::with_capacity(dim);
    let mut slopes: Vec<Slopes> = Vec::with_capacity(dim);
    for atom in 0..natoms {
        let fields = context.fields(&unstrained, &moving(atom))?;
        let (energy, part) = context.linearize(&fields)?;
        fields_moved.extend(moved(&fields));
        slopes.extend(part);
        if reference.is_none() {
            reference = Some(energy);
        }
    }
    let mut skeleton = Matrix::zeros(dim, dim);
    let mut vxc_derivative: [Vec<Vec<ImageBlock>>; 2] =
        [Vec::with_capacity(dim), Vec::with_capacity(dim)];
    let Some(energy) = reference else {
        return Ok(PeriodicXcHessian {
            skeleton,
            vxc_derivative,
        });
    };
    for x in 0..dim {
        for y in 0..dim {
            skeleton[(x, y)] = pairing(&fields_moved[x], &slopes[y]);
        }
    }
    // Second derivatives of the fields and of the weights, one pair of
    // atoms at a time.
    let unstrained_second = [[Second3::constant(First3::constant(0.0)); 3]; 3];
    for u in 0..natoms {
        for v in u..natoms {
            let seeded = move |atom: usize| -> [Second3; 3] {
                std::array::from_fn(|c| {
                    let mut x = Second3::constant(First3::constant(0.0));
                    if atom == u {
                        x.im[c] = First3::constant(1.0);
                    }
                    if atom == v {
                        x.re = First3::variable(0.0, c);
                    }
                    x
                })
            };
            let value = context.frozen(&energy, &unstrained_second, &seeded)?;
            for c in 0..3 {
                for d in 0..3 {
                    let second = value.im[c].im[d];
                    skeleton[(3 * u + c, 3 * v + d)] += second;
                    if u != v {
                        skeleton[(3 * v + d, 3 * u + c)] += second;
                    }
                }
            }
        }
    }
    for atom in 0..natoms {
        let potential = potential_jet::<3>(&energy, &slopes[3 * atom..3 * atom + 3]);
        let blocks = context.potential_jets(
            &unstrained,
            &moving(atom),
            &potential,
            3,
            &|v: First3, k| v.im[k],
        )?;
        for (spin, list) in blocks.into_iter().enumerate() {
            for c in 0..3 {
                vxc_derivative[spin].push(
                    list.iter()
                        .map(|b| ImageBlock {
                            translation: b.translation,
                            matrix: b.d[c].clone(),
                        })
                        .collect(),
                );
            }
        }
    }
    Ok(PeriodicXcHessian {
        skeleton,
        vxc_derivative,
    })
}

/// First-order strain matrices `dV_s(T)/d(epsilon_ab)` of the learned
/// Kohn-Sham image blocks at fixed density images, component `3a + b`.
pub(crate) fn learned_strain_fock(
    cell: &Cell,
    model: &SkalaModel,
    density: [&[ImageBlock]; 2],
    grid_options: &crate::grid::GridOptions,
    grid_shell: i64,
    ao_cutoff: f64,
) -> Result<[Vec<StrainBlock>; 2]> {
    let context = Context::new(cell, model, density, grid_options, grid_shell, ao_cutoff)?;
    let eps = strain_first();
    let still = |_: usize| [First9::constant(0.0); 3];
    let (energy, slopes) = context.linearize(&context.fields(&eps, &still)?)?;
    let potential = potential_jet::<9>(&energy, &slopes);
    context.potential_jets(&eps, &still, &potential, 9, &|v: First9, k| v.im[k])
}

/// [`StrainSecond`] of the learned exchange-correlation energy at fixed
/// density images: strain-strain and strain-displacement second derivatives.
pub(crate) fn learned_strain_second(
    cell: &Cell,
    model: &SkalaModel,
    density: [&[ImageBlock]; 2],
    grid_options: &crate::grid::GridOptions,
    grid_shell: i64,
    ao_cutoff: f64,
) -> Result<StrainSecond> {
    let context = Context::new(cell, model, density, grid_options, grid_shell, ao_cutoff)?;
    let natoms = cell.molecule.len();
    let eps = strain_first();
    let still = |_: usize| [First9::constant(0.0); 3];
    let fields = context.fields(&eps, &still)?;
    let (energy, slopes) = context.linearize(&fields)?;
    let strained = moved(&fields);
    let mut out = StrainSecond::zeros(natoms);
    let fixed = |_: usize| [<Jet2 as Scalar>::from_f64(0.0); 3];
    let value = context.frozen(&energy, &crate::periodic::strain::strain_seed(), &fixed)?;
    for k in 0..9 {
        for l in 0..9 {
            out.strain[k * 9 + l] = pairing(&strained[k], &slopes[l]) + value.second(k, l);
        }
    }
    let unstrained = [[First3::constant(0.0); 3]; 3];
    let inner: [[Mixed; 3]; 3] = std::array::from_fn(|a| {
        std::array::from_fn(|b| Mixed::constant(First9::variable(0.0, 3 * a + b)))
    });
    for u in 0..natoms {
        let displaced = moved(&context.fields(&unstrained, &moving(u))?);
        let seeded = move |atom: usize| -> [Mixed; 3] {
            std::array::from_fn(|c| {
                let mut x = Mixed::constant(First9::constant(0.0));
                if atom == u {
                    x.im[c] = First9::constant(1.0);
                }
                x
            })
        };
        let value = context.frozen(&energy, &inner, &seeded)?;
        for k in 0..9 {
            for c in 0..3 {
                out.mixed[k * 3 * natoms + 3 * u + c] =
                    pairing(&displaced[c], &slopes[k]) + value.im[c].im[k];
            }
        }
    }
    Ok(out)
}
