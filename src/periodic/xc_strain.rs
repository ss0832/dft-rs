// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Strain and strain-displacement second derivatives of the periodic XC
//! energy at fixed density images by exact forward-mode jets.
//!
//! At a grid point owned by atom `O` with rigid offset `delta`, the strained
//! and displaced point is `r' = (1+eps)(R_O + u_O) + delta`, every AO image
//! centre `C' = (1+eps)(A + T + u_A)` and every Becke image centre likewise.
//! The density, its gradient and the kinetic energy density of the local
//! image basis, and the periodic Becke weight, are evaluated as jets; the
//! functional enters through its exact second-order expansion with the
//! analytic derivatives, which is exact for nested second-order jets.

use crate::basis::aos::cart_to_sph;
use crate::basis::{cart_components, n_sph, MolShell};
use crate::dual::Scalar;
use crate::error::Result;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::grid::{pair_factor, size_adjustment, PeriodicGrid, Switch};
use crate::periodic::integrals::ImageBlock;
use crate::periodic::strain::StrainSecond;
use crate::periodic::strain_jets::strain_apply;
use crate::periodic::xc::xc_ao_images;
use crate::periodic::Cell;
use crate::xc::{Family, GridVars, ResolvedFunctional, XcOut};
use rayon::prelude::*;

fn jet<S: Scalar>(v: Vec3) -> [S; 3] {
    [S::from_f64(v.x), S::from_f64(v.y), S::from_f64(v.z)]
}

/// Spherical AO values and point gradients of one shell at `d = r - C`.
pub(crate) fn shell_values<S: Scalar>(shell: &MolShell, d: [S; 3]) -> (Vec<S>, Vec<[S; 3]>) {
    let comps = cart_components(shell.l);
    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    let zero = S::from_f64(0.0);
    let mut radial = zero;
    let mut radial_slope = zero; // d radial / d(r2)
    for (&alpha, &c) in shell.exps.iter().zip(&shell.coefs) {
        let g = (r2 * S::from_f64(-alpha)).exp() * S::from_f64(c);
        radial = radial + g;
        radial_slope = radial_slope + g * S::from_f64(-alpha);
    }
    let power = |x: S, n: i32| -> S {
        let mut v = S::from_f64(1.0);
        for _ in 0..n {
            v = v * x;
        }
        v
    };
    let mut cart_values = Vec::with_capacity(comps.len());
    let mut cart_grads: Vec<[S; 3]> = Vec::with_capacity(comps.len());
    for &(i, j, k) in comps.iter() {
        let (px, py, pz) = (power(d[0], i), power(d[1], j), power(d[2], k));
        let poly = px * py * pz;
        cart_values.push(poly * radial);
        let dpoly = [
            if i > 0 {
                S::from_f64(i as f64) * power(d[0], i - 1) * py * pz
            } else {
                zero
            },
            if j > 0 {
                S::from_f64(j as f64) * px * power(d[1], j - 1) * pz
            } else {
                zero
            },
            if k > 0 {
                S::from_f64(k as f64) * px * py * power(d[2], k - 1)
            } else {
                zero
            },
        ];
        let two = S::from_f64(2.0);
        cart_grads.push(std::array::from_fn(|a| {
            dpoly[a] * radial + poly * radial_slope * two * d[a]
        }));
    }
    let c2s = cart_to_sph(shell.l);
    let nsph = n_sph(shell.l);
    let mut values = vec![zero; nsph];
    let mut grads = vec![[zero; 3]; nsph];
    for (c, (v, g)) in cart_values.iter().zip(&cart_grads).enumerate() {
        for m in 0..nsph {
            let w = c2s[(c, m)];
            if w == 0.0 {
                continue;
            }
            let w = S::from_f64(w);
            values[m] = values[m] + w * *v;
            for a in 0..3 {
                grads[m][a] = grads[m][a] + w * g[a];
            }
        }
    }
    (values, grads)
}

/// Becke cells below this fraction of the partition sum are left out of the
/// weight jets. A cell `P_b` enters `w = P_t / sum P` and its derivatives
/// only through `P_b d(ln P_b)`, so the omitted terms are below
/// `1e-24 |d ln P_b|` relative to the weight, far under roundoff; the cut
/// changes the weight itself by less than `1e-24` relative (jump bound
/// 1e-24 times the XC energy of the point).
const BECKE_CELL_SCREEN: f64 = 1e-24;

/// The switched periodic Becke weight of a point over the local image
/// centres, as a jet: `P_b = g_b prod_c [1 - g_c (1 - s(nu_bc))]` with the
/// switching function `g` of [`Switch`] on the point-centre distances in
/// units of the strained length per atom (`length_scale = det(1 + eps)^(1/3)`). Cells
/// that vanish exactly in f64 carry no derivative and are left out, as are
/// cells below [`BECKE_CELL_SCREEN`] of the partition sum (except the target)
/// and factors that equal one exactly.
pub(crate) fn becke_weight_jet<S: Scalar>(
    centres: &[(Vec3, f64, usize)],
    centre_jets: &[[S; 3]],
    target: usize,
    point: [S; 3],
    sw: &Switch,
    length_scale: S,
) -> S {
    let n = centres.len();
    let p0 = Vec3::new(point[0].value(), point[1].value(), point[2].value());
    let distance = |v: [S; 3]| -> S {
        let d: [S; 3] = std::array::from_fn(|k| point[k] - v[k]);
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    };
    let d0: Vec<f64> = centres.iter().map(|(c, _, _)| (p0 - *c).norm()).collect();
    let g0: Vec<f64> = d0.iter().map(|d| sw.value_slope(*d).0).collect();
    let factor_f64 = |b: usize, c: usize| -> f64 {
        let rbc = (centres[b].0 - centres[c].0).norm();
        pair_factor(
            d0[b],
            d0[c],
            rbc,
            size_adjustment(centres[b].1, centres[c].1),
            g0[c],
        )
    };
    let skipped = |b: usize, c: usize| -> bool {
        c == b || g0[c] == 0.0 || (centres[b].0 - centres[c].0).norm() < 1e-12
    };
    // f64 pre-pass: cell values; dead cells carry nothing.
    let cells: Vec<f64> = (0..n)
        .map(|b| {
            let mut value = g0[b];
            for c in 0..n {
                if value == 0.0 {
                    break;
                }
                if skipped(b, c) {
                    continue;
                }
                value *= factor_f64(b, c);
            }
            value
        })
        .collect();
    let sum: f64 = cells.iter().sum();
    let distances: Vec<S> = centre_jets.iter().map(|c| distance(*c)).collect();
    let switching: Vec<S> = (0..n)
        .map(|b| {
            if g0[b] == 0.0 {
                S::from_f64(0.0)
            } else {
                sw.value(distances[b] / length_scale)
            }
        })
        .collect();
    let mut total = S::from_f64(0.0);
    let mut own = S::from_f64(0.0);
    for b in 0..n {
        if cells[b] == 0.0 || (b != target && cells[b] < BECKE_CELL_SCREEN * sum) {
            continue;
        }
        let db = distances[b];
        let mut cell = switching[b];
        for c in 0..n {
            if skipped(b, c) || factor_f64(b, c) == 1.0 {
                continue;
            }
            let rbc_v: [S; 3] = std::array::from_fn(|k| centre_jets[b][k] - centre_jets[c][k]);
            let rbc = (rbc_v[0] * rbc_v[0] + rbc_v[1] * rbc_v[1] + rbc_v[2] * rbc_v[2]).sqrt();
            cell = cell
                * pair_factor(
                    db,
                    distances[c],
                    rbc,
                    size_adjustment(centres[b].1, centres[c].1),
                    switching[c],
                );
        }
        if b == target {
            own = cell;
        }
        total = total + cell;
    }
    if total.value() <= 0.0 {
        S::from_f64(0.0)
    } else {
        own / total
    }
}

/// Jets of one grid batch: the local image basis, AO values and gradients,
/// spin densities, gradients and kinetic energy densities, partition
/// weights, and the functional with its analytic first and second
/// derivatives at the reference values.
pub(super) struct BatchJets<S: Scalar> {
    /// Index of the first point of the batch in the grid.
    pub(super) start: usize,
    /// Local image basis: (home shell, translation).
    pub(super) local: Vec<(usize, Vec3)>,
    pub(super) offsets: Vec<usize>,
    pub(super) m: usize,
    pub(super) phi: Vec<Vec<S>>,
    pub(super) dphi: Vec<Vec<[S; 3]>>,
    /// Local AO indices inside the cutoff of each point.
    pub(super) significant: Vec<Vec<usize>>,
    /// Spin densities, their gradients and kinetic energy densities.
    pub(super) rho: Vec<[S; 2]>,
    pub(super) grad: Vec<[[S; 3]; 2]>,
    pub(super) tau: Vec<[S; 2]>,
    pub(super) weight: Vec<S>,
    /// Variations `x - x_0` of rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb,
    /// tau_a, tau_b.
    dx: Vec<[S; 7]>,
    e0: Vec<f64>,
    first: Vec<[f64; 7]>,
    second: Vec<[[f64; 7]; 7]>,
}

/// Evaluate every batch of the grid as jets in the strain `eps` and the
/// seeded atomic displacements, and reduce the per-batch results.
#[allow(clippy::too_many_arguments)]
pub(super) fn xc_batches<S: Scalar + Send + Sync, R: Send>(
    cell: &Cell,
    functional: &ResolvedFunctional,
    density: [&[ImageBlock]; 2],
    grid: &PeriodicGrid,
    grid_shell: i64,
    ao_cutoff: f64,
    eps: &[[S; 3]; 3],
    displacement: &(dyn Fn(usize) -> [S; 3] + Sync),
    visit: &(dyn Fn(&BatchJets<S>) -> R + Sync),
) -> Result<Vec<R>> {
    let gga = functional.family != Family::Lda;
    let meta = functional.family == Family::MetaGga;
    let molecule = &cell.molecule;
    let lattice = &cell.lattice;
    let basis = &cell.basis;
    let find = |spin: usize, t: Vec3| -> Option<&Matrix> {
        density[spin]
            .iter()
            .find(|b| (b.translation - t).norm() < 1e-8)
            .map(|b| &b.matrix)
    };
    const BATCH: usize = 32;
    let ranges: Vec<(usize, usize)> = (0..grid.points.len())
        .step_by(BATCH)
        .map(|s| (s, (s + BATCH).min(grid.points.len())))
        .collect();
    let sw = Switch::periodic(molecule, lattice, grid_shell);
    let keep = sw.cut;
    // The length per atom of the strained cell, relative to the reference:
    // det(1 + eps)^(1/3).
    let length_scale: S = {
        let m = |a: usize, b: usize| eps[a][b] + S::from_f64(if a == b { 1.0 } else { 0.0 });
        let det = m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1))
            - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
            + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
        det.cbrt()
    };
    ranges
        .par_iter()
        .map(|&(start, end)| -> Result<R> {
            let points = &grid.points[start..end];
            let images = xc_ao_images(basis, lattice, points, ao_cutoff);
            let mut local: Vec<(usize, Vec3)> = Vec::new();
            // Local shells inside the AO cutoff of each point, exactly as the
            // SCF quadrature screens them.
            let mut point_shells: Vec<Vec<usize>> = vec![Vec::new(); end - start];
            for image in &images {
                let mut selected: Vec<usize> = image.point_shells.iter().map(|&(_, s)| s).collect();
                selected.sort_unstable();
                selected.dedup();
                let first = local.len();
                for &s in &selected {
                    local.push((s, image.translation));
                }
                for &(point, s) in &image.point_shells {
                    point_shells[point]
                        .push(first + selected.binary_search(&s).expect("selected shell"));
                }
            }
            let offsets: Vec<usize> = local
                .iter()
                .scan(0usize, |acc, (s, _)| {
                    let o = *acc;
                    *acc += n_sph(basis.shells[*s].l);
                    Some(o)
                })
                .collect();
            let m = offsets
                .last()
                .map_or(0, |o| o + n_sph(basis.shells[local.last().unwrap().0].l));
            let mut p = [Matrix::zeros(m, m), Matrix::zeros(m, m)];
            for (a, (sa, ta)) in local.iter().enumerate() {
                let home_a = &basis.shells[*sa];
                for (b, (sb, tb)) in local.iter().enumerate() {
                    let home_b = &basis.shells[*sb];
                    for spin in 0..2 {
                        let Some(d) = find(spin, *tb - *ta) else {
                            continue;
                        };
                        for i in 0..n_sph(home_a.l) {
                            for j in 0..n_sph(home_b.l) {
                                p[spin][(offsets[a] + i, offsets[b] + j)] =
                                    d[(home_a.sph_off + i, home_b.sph_off + j)];
                            }
                        }
                    }
                }
            }
            // Jets of the local AO centres.
            let centres: Vec<[S; 3]> = local
                .iter()
                .map(|(s, t)| {
                    let shell = &basis.shells[*s];
                    let u = displacement(shell.atom);
                    let base = jet::<S>(shell.center + *t);
                    strain_apply(eps, std::array::from_fn(|k| base[k] + u[k]))
                })
                .collect();
            let npts = end - start;
            let zero = S::from_f64(0.0);
            let mut rho = vec![[zero; 2]; npts];
            let mut grad = vec![[[zero; 3]; 2]; npts];
            let mut tau = vec![[zero; 2]; npts];
            let mut point_jets = Vec::with_capacity(npts);
            let mut phis = Vec::with_capacity(npts);
            let mut dphis = Vec::with_capacity(npts);
            let mut significant: Vec<Vec<usize>> = Vec::with_capacity(npts);
            for (i, point) in points.iter().enumerate() {
                let owner = grid.atom_of[start + i];
                let home = molecule.atoms[owner].position;
                let u = displacement(owner);
                let base = jet::<S>(home);
                let owner_strained = strain_apply(eps, std::array::from_fn(|k| base[k] + u[k]));
                let offset = *point - home;
                let r: [S; 3] =
                    std::array::from_fn(|k| owner_strained[k] + S::from_f64(offset.get(k)));
                point_jets.push(r);
                let mut phi = vec![zero; m];
                let mut dphi = vec![[zero; 3]; m];
                let mut list: Vec<usize> = Vec::new();
                for &a in &point_shells[i] {
                    let shell = &basis.shells[local[a].0];
                    let d: [S; 3] = std::array::from_fn(|k| r[k] - centres[a][k]);
                    let (v, g) = shell_values(shell, d);
                    for (c, (vv, gg)) in v.into_iter().zip(g).enumerate() {
                        phi[offsets[a] + c] = vv;
                        dphi[offsets[a] + c] = gg;
                        list.push(offsets[a] + c);
                    }
                }
                for spin in 0..2 {
                    for &row in &list {
                        let mut x = zero;
                        let mut y = [zero; 3];
                        for &col in &list {
                            let pv = p[spin][(row, col)];
                            if pv == 0.0 {
                                continue;
                            }
                            x = x + phi[col].scale(pv);
                            if meta {
                                for k in 0..3 {
                                    y[k] = y[k] + dphi[col][k].scale(pv);
                                }
                            }
                        }
                        rho[i][spin] = rho[i][spin] + phi[row] * x;
                        if gga {
                            for k in 0..3 {
                                grad[i][spin][k] = grad[i][spin][k] + (dphi[row][k] * x).scale(2.0);
                            }
                        }
                        if meta {
                            for k in 0..3 {
                                tau[i][spin] = tau[i][spin] + (dphi[row][k] * y[k]).scale(0.5);
                            }
                        }
                    }
                }
                phis.push(phi);
                dphis.push(dphi);
                significant.push(list);
            }
            // Functional derivatives at the reference point values.
            let values = |f: &dyn Fn(usize) -> f64| (0..npts).map(f).collect::<Vec<f64>>();
            let rho_v = [
                values(&|i| rho[i][0].value().max(0.0)),
                values(&|i| rho[i][1].value().max(0.0)),
            ];
            let sigma_v = [
                values(&|i| {
                    (0..3)
                        .map(|k| grad[i][0][k].value() * grad[i][0][k].value())
                        .sum()
                }),
                values(&|i| {
                    (0..3)
                        .map(|k| grad[i][0][k].value() * grad[i][1][k].value())
                        .sum()
                }),
                values(&|i| {
                    (0..3)
                        .map(|k| grad[i][1][k].value() * grad[i][1][k].value())
                        .sum()
                }),
            ];
            let tau_v = [
                values(&|i| tau[i][0].value()),
                values(&|i| tau[i][1].value()),
            ];
            let vars = GridVars {
                rho: [&rho_v[0], &rho_v[1]],
                sigma: gga.then_some([&sigma_v[0], &sigma_v[1], &sigma_v[2]]),
                tau: meta.then_some([&tau_v[0], &tau_v[1]]),
            };
            let mut out = XcOut::default();
            functional.eval(&vars, 2, &mut out)?;
            let mut dxs = Vec::with_capacity(npts);
            let mut firsts = Vec::with_capacity(npts);
            let mut seconds = Vec::with_capacity(npts);
            let mut weights = Vec::with_capacity(npts);
            for i in 0..npts {
                let mut dx = [zero; 7];
                dx[0] = rho[i][0] - S::from_f64(rho[i][0].value());
                dx[1] = rho[i][1] - S::from_f64(rho[i][1].value());
                if gga {
                    let s = |a: usize, b: usize| {
                        (0..3).fold(zero, |acc, k| acc + grad[i][a][k] * grad[i][b][k])
                    };
                    let (saa, sab, sbb) = (s(0, 0), s(0, 1), s(1, 1));
                    dx[2] = saa - S::from_f64(saa.value());
                    dx[3] = sab - S::from_f64(sab.value());
                    dx[4] = sbb - S::from_f64(sbb.value());
                }
                if meta {
                    dx[5] = tau[i][0] - S::from_f64(tau[i][0].value());
                    dx[6] = tau[i][1] - S::from_f64(tau[i][1].value());
                }
                let mut first = [0.0; 7];
                let mut second = [[0.0; 7]; 7];
                first[0] = out.vrho[0][i];
                first[1] = out.vrho[1][i];
                second[0][0] = out.v2rho2[0][i];
                second[0][1] = out.v2rho2[1][i];
                second[1][1] = out.v2rho2[2][i];
                if gga {
                    for pch in 0..3 {
                        first[2 + pch] = out.vsigma[pch][i];
                        second[0][2 + pch] = out.v2rhosigma[pch][i];
                        second[1][2 + pch] = out.v2rhosigma[3 + pch][i];
                        for qch in pch..3 {
                            let idx = match (pch, qch) {
                                (0, 0) => 0,
                                (0, 1) => 1,
                                (0, 2) => 2,
                                (1, 1) => 3,
                                (1, 2) => 4,
                                _ => 5,
                            };
                            second[2 + pch][2 + qch] = out.v2sigma2[idx][i];
                        }
                    }
                }
                if meta {
                    first[5] = out.vtau[0][i];
                    first[6] = out.vtau[1][i];
                    second[0][5] = out.v2rhotau[0][i];
                    second[0][6] = out.v2rhotau[1][i];
                    second[1][5] = out.v2rhotau[2][i];
                    second[1][6] = out.v2rhotau[3][i];
                    for pch in 0..3 {
                        second[2 + pch][5] = out.v2sigmatau[2 * pch][i];
                        second[2 + pch][6] = out.v2sigmatau[2 * pch + 1][i];
                    }
                    second[5][5] = out.v2tau2[0][i];
                    second[5][6] = out.v2tau2[1][i];
                    second[6][6] = out.v2tau2[2][i];
                }
                for r in 0..7 {
                    for c in 0..r {
                        second[r][c] = second[c][r];
                    }
                }
                // Partition weight of the strained point.
                let raw = grid.raw_weights[start + i];
                let weight = if (grid.weights[start + i] - raw).abs() <= 1e-14 * raw.abs() {
                    S::from_f64(grid.weights[start + i])
                } else {
                    let owner = grid.atom_of[start + i];
                    let centres =
                        crate::periodic::grid::local_images_of(molecule, lattice, points[i], keep);
                    let target = centres.iter().position(|(c, _, atom)| {
                        *atom == owner && (*c - molecule.atoms[owner].position).norm() < 1e-10
                    });
                    let centre_jets: Vec<[S; 3]> = centres
                        .iter()
                        .map(|(c, _, atom)| {
                            let u = displacement(*atom);
                            let base = jet::<S>(*c);
                            strain_apply(eps, std::array::from_fn(|k| base[k] + u[k]))
                        })
                        .collect();
                    // A point outside the partition radius of its owner has
                    // no weight in a neighbourhood of the reference.
                    match target {
                        Some(target) => {
                            S::from_f64(raw)
                                * becke_weight_jet(
                                    &centres,
                                    &centre_jets,
                                    target,
                                    point_jets[i],
                                    &sw,
                                    length_scale,
                                )
                        }
                        None => S::from_f64(0.0),
                    }
                };
                dxs.push(dx);
                firsts.push(first);
                seconds.push(second);
                weights.push(weight);
            }
            Ok(visit(&BatchJets {
                start,
                local,
                offsets,
                m,
                phi: phis,
                dphi: dphis,
                significant,
                rho,
                grad,
                tau,
                weight: weights,
                dx: dxs,
                e0: out.e.clone(),
                first: firsts,
                second: seconds,
            }))
        })
        .collect()
}

/// The XC energy of the fixed density images as a jet in the strain `eps`
/// and the seeded atomic displacements; the functional enters through its
/// exact second-order expansion.
#[allow(clippy::too_many_arguments)]
fn xc_energy_jet<S: Scalar + Send + Sync>(
    cell: &Cell,
    functional: &ResolvedFunctional,
    density: [&[ImageBlock]; 2],
    grid: &PeriodicGrid,
    grid_shell: i64,
    ao_cutoff: f64,
    eps: &[[S; 3]; 3],
    displacement: &(dyn Fn(usize) -> [S; 3] + Sync),
) -> Result<S> {
    let energies = xc_batches(
        cell,
        functional,
        density,
        grid,
        grid_shell,
        ao_cutoff,
        eps,
        displacement,
        &|batch| {
            let mut energy = S::from_f64(0.0);
            for (i, dx) in batch.dx.iter().enumerate() {
                let mut e = S::from_f64(batch.e0[i]);
                for r in 0..7 {
                    if batch.first[i][r] != 0.0 {
                        e = e + S::from_f64(batch.first[i][r]) * dx[r];
                    }
                    for c in 0..7 {
                        if batch.second[i][r][c] != 0.0 {
                            e = e + S::from_f64(0.5 * batch.second[i][r][c]) * dx[r] * dx[c];
                        }
                    }
                }
                energy = energy + batch.weight[i] * e;
            }
            energy
        },
    )?;
    Ok(energies.into_iter().fold(S::from_f64(0.0), |a, b| a + b))
}

/// First-order strain matrices `dV_xc,s(T)/d(epsilon_ab)` of both spin XC
/// potentials at fixed density images (which must reach twice `ao_cutoff`):
/// the derivative of `sum_p w_p [v_rho phi_m phi_n + g . grad(phi_m phi_n) +
/// v_tau grad phi_m . grad phi_n / 2]` with every point riding its owner,
/// every AO image centre, the Becke weight and the functional derivatives
/// (through `f_xc`) carried by exact first-order jets. The value parts are
/// the potentials themselves.
pub(crate) fn xc_strain_fock(
    cell: &Cell,
    functional: &ResolvedFunctional,
    density: [&[ImageBlock]; 2],
    grid_options: &crate::grid::GridOptions,
    grid_shell: i64,
    ao_cutoff: f64,
) -> Result<[Vec<crate::periodic::strain_fock::StrainBlock>; 2]> {
    type Jet = crate::dual::DualN<f64, 9>;
    let eps: [[Jet; 3]; 3] =
        std::array::from_fn(|a| std::array::from_fn(|b| Jet::variable(0.0, 3 * a + b)));
    xc_potential_jets(
        cell,
        functional,
        density,
        grid_options,
        grid_shell,
        ao_cutoff,
        &eps,
        9,
        &|v: Jet, k| v.im[k],
    )
}

/// Spin XC potential image blocks `<m(0)| v_xc,s |n(T)>` of fixed density
/// images on the SCF quadrature (local image basis, AO cutoff screening per
/// point). Their Bloch sums are the SCF potentials.
pub(crate) fn xc_potential_blocks(
    cell: &Cell,
    functional: &ResolvedFunctional,
    density: [&[ImageBlock]; 2],
    grid_options: &crate::grid::GridOptions,
    grid_shell: i64,
    ao_cutoff: f64,
) -> Result<[Vec<ImageBlock>; 2]> {
    let eps = [[0.0_f64; 3]; 3];
    let blocks = xc_potential_jets(
        cell,
        functional,
        density,
        grid_options,
        grid_shell,
        ao_cutoff,
        &eps,
        1,
        &|v: f64, _| v,
    )?;
    Ok(blocks.map(|list| {
        list.into_iter()
            .map(|mut b| ImageBlock {
                translation: b.translation,
                matrix: b.d.swap_remove(0),
            })
            .collect()
    }))
}

/// XC potential matrices as jets in the strain `eps`, scattered into image
/// blocks with `components` matrices from `extract(value, component)`.
#[allow(clippy::too_many_arguments)]
fn xc_potential_jets<S: Scalar + Send + Sync>(
    cell: &Cell,
    functional: &ResolvedFunctional,
    density: [&[ImageBlock]; 2],
    grid_options: &crate::grid::GridOptions,
    grid_shell: i64,
    ao_cutoff: f64,
    eps: &[[S; 3]; 3],
    components: usize,
    extract: &(dyn Fn(S, usize) -> f64 + Sync),
) -> Result<[Vec<crate::periodic::strain_fock::StrainBlock>; 2]> {
    use crate::periodic::strain_fock::{merge_strain_blocks, StrainBlock};
    let gga = functional.family != Family::Lda;
    let meta = functional.family == Family::MetaGga;
    let basis = &cell.basis;
    let n = basis.nao_sph;
    let grid = PeriodicGrid::build(&cell.molecule, &cell.lattice, grid_options, grid_shell)?;
    let still = |_: usize| [S::from_f64(0.0); 3];
    let per_batch = xc_batches(
        cell,
        functional,
        density,
        &grid,
        grid_shell,
        ao_cutoff,
        eps,
        &still,
        &|batch| {
            let m = batch.m;
            let zero = S::from_f64(0.0);
            let mut local: [Vec<S>; 2] = [vec![zero; m * m], vec![zero; m * m]];
            for i in 0..batch.dx.len() {
                let w = batch.weight[i];
                let phi = &batch.phi[i];
                let dphi = &batch.dphi[i];
                let expand = |r: usize| -> S {
                    let mut v = S::from_f64(batch.first[i][r]);
                    for c in 0..7 {
                        if batch.second[i][r][c] != 0.0 {
                            v = v + S::from_f64(batch.second[i][r][c]) * batch.dx[i][c];
                        }
                    }
                    v
                };
                let v_sigma = if gga {
                    [expand(2), expand(3), expand(4)]
                } else {
                    [zero; 3]
                };
                for spin in 0..2 {
                    let v_rho = expand(spin);
                    // g_s = 2 v_sigma_ss grad rho_s + v_sigma_ab grad rho_s'.
                    let own = if spin == 0 { v_sigma[0] } else { v_sigma[2] };
                    let g: [S; 3] = std::array::from_fn(|k| {
                        S::from_f64(2.0) * own * batch.grad[i][spin][k]
                            + v_sigma[1] * batch.grad[i][1 - spin][k]
                    });
                    let half_tau = if meta {
                        S::from_f64(0.5) * expand(5 + spin) * w
                    } else {
                        zero
                    };
                    // A_m = w (v_rho phi_m / 2 + g . grad phi_m).
                    let list = &batch.significant[i];
                    let mut a: Vec<S> = vec![zero; m];
                    for &mu in list {
                        let mut x = (v_rho * phi[mu]).scale(0.5);
                        if gga {
                            for k in 0..3 {
                                x = x + g[k] * dphi[mu][k];
                            }
                        }
                        a[mu] = w * x;
                    }
                    let target = &mut local[spin];
                    for &mu in list {
                        for &nu in list {
                            let mut v = a[mu] * phi[nu] + phi[mu] * a[nu];
                            if meta {
                                v = v + half_tau
                                    * (dphi[mu][0] * dphi[nu][0]
                                        + dphi[mu][1] * dphi[nu][1]
                                        + dphi[mu][2] * dphi[nu][2]);
                            }
                            target[mu * m + nu] = target[mu * m + nu] + v;
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
                                blocks[spin].push(StrainBlock::with_components(t, n, components));
                                blocks[spin].len() - 1
                            }
                        };
                        let block = &mut blocks[spin][index];
                        for i in 0..n_sph(home_a.l) {
                            for j in 0..n_sph(home_b.l) {
                                let v =
                                    local[spin][(batch.offsets[a] + i) * m + batch.offsets[b] + j];
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

/// [`StrainSecond`] of the periodic XC energy at fixed spin density images
/// (which must reach twice `ao_cutoff`), on the grid of `grid_options` and
/// `grid_shell`.
pub(crate) fn xc_strain_second(
    cell: &Cell,
    functional: &ResolvedFunctional,
    density: [&[ImageBlock]; 2],
    grid_options: &crate::grid::GridOptions,
    grid_shell: i64,
    ao_cutoff: f64,
) -> Result<(StrainSecond, [[f64; 3]; 3])> {
    use crate::dual::DualN;
    type MixedJet = DualN<DualN<f64, 9>, 3>;
    let natoms = cell.molecule.len();
    let grid = PeriodicGrid::build(&cell.molecule, &cell.lattice, grid_options, grid_shell)?;
    let eps = crate::periodic::strain::strain_seed();
    let still = |_: usize| [<crate::dual::Jet2 as Scalar>::from_f64(0.0); 3];
    let energy = xc_energy_jet(
        cell, functional, density, &grid, grid_shell, ao_cutoff, &eps, &still,
    )?;
    let mut out = StrainSecond::zeros(natoms);
    let mut first = [[0.0; 3]; 3];
    for k in 0..9 {
        first[k / 3][k % 3] = energy.gradient[k];
        for l in 0..9 {
            out.strain[k * 9 + l] = energy.second(k, l);
        }
    }
    let inner: [[MixedJet; 3]; 3] = std::array::from_fn(|a| {
        std::array::from_fn(|b| MixedJet::constant(DualN::<f64, 9>::variable(0.0, 3 * a + b)))
    });
    for u in 0..natoms {
        let moving = move |atom: usize| -> [MixedJet; 3] {
            if atom == u {
                std::array::from_fn(|c| {
                    let mut x = MixedJet::constant(DualN::<f64, 9>::constant(0.0));
                    x.im[c] = DualN::<f64, 9>::constant(1.0);
                    x
                })
            } else {
                [<MixedJet as Scalar>::from_f64(0.0); 3]
            }
        };
        let energy = xc_energy_jet(
            cell, functional, density, &grid, grid_shell, ao_cutoff, &inner, &moving,
        )?;
        for k in 0..9 {
            for c in 0..3 {
                out.mixed[k * 3 * natoms + 3 * u + c] = energy.im[c].im[k];
            }
        }
    }
    Ok((out, first))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::GridOptions;
    use crate::periodic::bloch::BlochMatrix;
    use crate::periodic::scf::{PeriodicScfOptions, PeriodicScfResult};
    use crate::periodic::Lattice;
    use crate::system::{Atom, Molecule};
    use crate::xc::{resolve, FunctionalSpec};

    const BASIS: &str = "H 0\nS 2 1.00\n 2.4 0.6\n 0.55 0.5\nP 2 1.00\n 1.6 0.6\n 0.5 0.5\n****\n";
    const SCALE: f64 = 2.2;
    const CUTOFF: f64 = 9.0;

    fn deform(v: Vec3, eps: &[[f64; 3]; 3]) -> Vec3 {
        Vec3::new(
            v.x + eps[0][0] * v.x + eps[0][1] * v.y + eps[0][2] * v.z,
            v.y + eps[1][0] * v.x + eps[1][1] * v.y + eps[1][2] * v.z,
            v.z + eps[2][0] * v.x + eps[2][1] * v.y + eps[2][2] * v.z,
        )
    }

    fn cell(eps: &[[f64; 3]; 3], atom: usize, axis: usize, u: f64) -> Cell {
        let base = [
            Vec3::new(4.2, 0.0, 0.0),
            Vec3::new(0.3, 4.6, 0.0),
            Vec3::new(0.2, -0.25, 4.9),
        ]
        .map(|v| v * SCALE);
        let lattice = Lattice::new(
            deform(base[0], eps),
            deform(base[1], eps),
            deform(base[2], eps),
        );
        let mut positions = [Vec3::new(0.1, 0.2, -0.3), Vec3::new(1.2, 0.9, 0.8)];
        match axis {
            0 => positions[atom].x += u,
            1 => positions[atom].y += u,
            _ => positions[atom].z += u,
        }
        let molecule = Molecule::new(
            positions
                .iter()
                .map(|p| Atom {
                    z: 1,
                    position: deform(*p, eps),
                })
                .collect(),
        );
        let set = crate::basis::g94::parse("xc-strain-test", BASIS).unwrap();
        Cell::new(molecule, lattice, &set, CUTOFF).unwrap()
    }

    fn options() -> PeriodicScfOptions {
        PeriodicScfOptions {
            k_mesh: [2, 1, 1],
            xc_ao_cutoff: Some(CUTOFF),
            grid_shell: 3,
            grid: GridOptions {
                radial_override: Some(12),
                angular_override: Some(26),
                prune: false,
                weight_cutoff: 0.0,
                ..GridOptions::default()
            },
            ..PeriodicScfOptions::default()
        }
    }

    fn result(cell: &Cell, k_points: Vec<Vec3>) -> PeriodicScfResult {
        let n = cell.nao();
        let density: [Vec<BlochMatrix>; 2] = std::array::from_fn(|spin| {
            let mut re = Matrix::identity(n);
            for p in 0..n {
                re[(p, p)] = 0.30 + 0.03 * p as f64 + 0.02 * spin as f64;
                for q in 0..n {
                    if p != q {
                        re[(p, q)] = 0.012 * ((p * 5 + q * 5 + spin) as f64 * 0.7).cos();
                    }
                }
            }
            k_points
                .iter()
                .map(|_| BlochMatrix {
                    re: re.clone(),
                    im: Matrix::zeros(n, n),
                })
                .collect()
        });
        let zero = || {
            k_points
                .iter()
                .map(|_| BlochMatrix::zeros(n, n))
                .collect::<Vec<_>>()
        };
        PeriodicScfResult {
            orbital_conjugation_constraint: true,
            energy: 0.0,
            free_energy: 0.0,
            entropy_term: 0.0,
            e_nuc: 0.0,
            e_elec: 0.0,
            e_xc: 0.0,
            e_nlc: 0.0,
            k_points: k_points.clone(),
            mo_energy: std::array::from_fn(|_| Vec::new()),
            occupations: std::array::from_fn(|_| Vec::new()),
            chemical_potential: [None, None],
            density,
            energy_weighted_density: std::array::from_fn(|_| zero()),
            fock: std::array::from_fn(|_| zero()),
            overlap: zero(),
            h_core: zero(),
            n_alpha: 1,
            n_beta: 1,
            unrestricted: true,
            grid_electron_count: None,
            converged: true,
            iterations: 0,
            orbital_residual: None,
            derivative_plan: None,
        }
    }

    fn fit(steps: &[f64], errors: &[f64]) -> (f64, f64) {
        let (x, y): (Vec<f64>, Vec<f64>) = steps
            .iter()
            .zip(errors)
            .map(|(h, e)| (h.ln(), e.ln()))
            .unzip();
        let n = x.len() as f64;
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
        let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
        let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
        (sxy / sxx, sxy * sxy / (sxx * syy))
    }

    /// The images crossing the Becke neighbourhood make the quadrature
    /// discontinuous (periodic.md section 11); the cell is dilated by 2.2
    /// with three Becke shells and the fit uses 6.4e-2..8e-3. Recorded with
    /// SVWN5 on every column: strain-strain [0][0], [0][1], [2][1] gave
    /// p = 1.997907, 2.007498, 2.002431 and strain-displacement (0,0)
    /// p = 1.996021; each central difference rebuilds the periodic grid, so
    /// the regression keeps one strain-strain and one strain-displacement
    /// column per functional on different components.
    #[test]
    fn xc_strain_second_derivatives_pass_four_step_fits() {
        let steps = [6.4e-2, 3.2e-2, 1.6e-2, 8e-3];
        let options = options();
        let zero = [[0.0; 3]; 3];
        for (name, strain_columns, displacement_columns) in [
            ("svwn5", [(0usize, 1usize)], [(1usize, 2usize)]),
            ("pbe", [(2, 1)], [(0, 0)]),
        ] {
            let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
            let base = cell(&zero, 0, 0, 0.0);
            let k_points = base.k_points(options.k_mesh);
            let reference = result(&base, k_points.clone());
            let images: Vec<Vec<ImageBlock>> = (0..2)
                .map(|spin| {
                    base.density_images_from_k(
                        &k_points,
                        &reference.density[spin]
                            .iter()
                            .map(|p| (p.re.clone(), p.im.clone()))
                            .collect::<Vec<_>>(),
                        2.0 * CUTOFF + 1.0,
                    )
                    .unwrap()
                })
                .collect();
            let (analytic, first_jet) = xc_strain_second(
                &base,
                &functional,
                [&images[0], &images[1]],
                &options.grid,
                options.grid_shell,
                CUTOFF,
            )
            .unwrap();
            let first = |eps: &[[f64; 3]; 3], atom: usize, axis: usize, u: f64| -> [[f64; 3]; 3] {
                let c = cell(eps, atom, axis, u);
                let r = result(&c, c.k_points(options.k_mesh));
                crate::periodic::xc_gradient::xc_strain_derivative(&c, &functional, &r, &options)
                    .unwrap()
            };
            let s0 = first(&zero, 0, 0, 0.0);
            let mut consistency = 0.0_f64;
            for a in 0..3 {
                for b in 0..3 {
                    consistency = consistency.max((s0[a][b] - first_jet[a][b]).abs());
                }
            }
            println!("XC {name}: jet first derivative against the virial route {consistency:e}");
            assert!(consistency < 1e-8);
            for (c, dd) in strain_columns {
                let errors: Vec<f64> = steps
                    .iter()
                    .map(|&h| {
                        let mut ep = zero;
                        ep[c][dd] = h;
                        let mut em = zero;
                        em[c][dd] = -h;
                        let (plus, minus) = (first(&ep, 0, 0, 0.0), first(&em, 0, 0, 0.0));
                        let mut worst = 0.0_f64;
                        for a in 0..3 {
                            for b in 0..3 {
                                let composed = if b == c { s0[a][dd] } else { 0.0 };
                                let fd = (plus[a][b] - minus[a][b]) / (2.0 * h);
                                worst = worst.max(
                                    (fd - analytic.strain[(3 * a + b) * 9 + 3 * c + dd] - composed)
                                        .abs(),
                                );
                            }
                        }
                        worst
                    })
                    .collect();
                let (p, r2) = fit(&steps, &errors);
                println!("XC {name} strain-strain column [{c}][{dd}]: errors {errors:?} p = {p:.6} R2 = {r2:.8}");
                assert!((p - 2.0).abs() < 0.05 && r2 > 0.9999);
            }
            for (atom, axis) in displacement_columns {
                let errors: Vec<f64> = steps
                    .iter()
                    .map(|&h| {
                        let (plus, minus) =
                            (first(&zero, atom, axis, h), first(&zero, atom, axis, -h));
                        let mut worst = 0.0_f64;
                        for a in 0..3 {
                            for b in 0..3 {
                                let fd = (plus[a][b] - minus[a][b]) / (2.0 * h);
                                worst = worst.max(
                                    (fd - analytic.mixed[(3 * a + b) * 6 + 3 * atom + axis]).abs(),
                                );
                            }
                        }
                        worst
                    })
                    .collect();
                let (p, r2) = fit(&steps, &errors);
                println!("XC {name} strain-displacement column ({atom},{axis}): errors {errors:?} p = {p:.6} R2 = {r2:.8}");
                assert!((p - 2.0).abs() < 0.05 && r2 > 0.9999);
            }
        }
    }

    /// The XC strain matrices against central differences of the SCF
    /// potential at fixed Bloch densities, on the dilated cell of the
    /// energy test.
    #[test]
    fn xc_strain_matrices_pass_four_step_fits() {
        use crate::periodic::strain_fock::strain_bloch;
        use crate::periodic::xc::{xc_contribution, AoDerivativeOrder, PeriodicXcGrid};
        let steps = [6.4e-2, 3.2e-2, 1.6e-2, 8e-3];
        let options = options();
        let zero = [[0.0; 3]; 3];
        for name in ["svwn5", "pbe"] {
            let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
            let base = cell(&zero, 0, 0, 0.0);
            let n = base.nao();
            let k_points = base.k_points(options.k_mesh);
            let reference = result(&base, k_points.clone());
            let images: Vec<Vec<ImageBlock>> = (0..2)
                .map(|spin| {
                    base.density_images_from_k(
                        &k_points,
                        &reference.density[spin]
                            .iter()
                            .map(|p| (p.re.clone(), p.im.clone()))
                            .collect::<Vec<_>>(),
                        2.0 * CUTOFF + 1.0,
                    )
                    .unwrap()
                })
                .collect();
            let analytic = xc_strain_fock(
                &base,
                &functional,
                [&images[0], &images[1]],
                &options.grid,
                options.grid_shell,
                CUTOFF,
            )
            .unwrap();
            let potential = |eps: &[[f64; 3]; 3]| -> [Vec<BlochMatrix>; 2] {
                let c = cell(eps, 0, 0, 0.0);
                let ks = c.k_points(options.k_mesh);
                let r = result(&c, ks.clone());
                let grid = PeriodicXcGrid::build(
                    &c,
                    &ks,
                    &options.grid,
                    options.grid_shell,
                    CUTOFF,
                    AoDerivativeOrder::for_scf(functional.family),
                )
                .unwrap();
                xc_contribution(&functional, &grid, &r.density).unwrap().vxc
            };
            for (c, dd) in [(0usize, 0usize), (0, 1), (2, 1)] {
                let errors: Vec<f64> = steps
                    .iter()
                    .map(|&h| {
                        let mut ep = zero;
                        ep[c][dd] = h;
                        let mut em = zero;
                        em[c][dd] = -h;
                        let (plus, minus) = (potential(&ep), potential(&em));
                        let mut worst = 0.0_f64;
                        for spin in 0..2 {
                            for (ik, k) in k_points.iter().enumerate() {
                                let a = &strain_bloch(&analytic[spin], *k, n)[3 * c + dd];
                                for i in 0..n {
                                    for j in 0..n {
                                        let re = (plus[spin][ik].re[(i, j)]
                                            - minus[spin][ik].re[(i, j)])
                                            / (2.0 * h);
                                        let im = (plus[spin][ik].im[(i, j)]
                                            - minus[spin][ik].im[(i, j)])
                                            / (2.0 * h);
                                        worst = worst
                                            .max((re - a.re[(i, j)]).abs())
                                            .max((im - a.im[(i, j)]).abs());
                                    }
                                }
                            }
                        }
                        worst
                    })
                    .collect();
                let (p, r2) = fit(&steps, &errors);
                println!("XC {name} strain matrix [{c}][{dd}]: errors {errors:?} p = {p:.6} R2 = {r2:.8}");
                assert!((p - 2.0).abs() < 0.05 && r2 > 0.9999);
            }
        }
    }
}
