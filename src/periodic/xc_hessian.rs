// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Gamma-point XC skeleton Hessian and XC Fock derivatives of a periodic
//! density.
//!
//! At a grid point only the AO images within the AO cutoff contribute, so
//! the periodic density is the molecular-form density of a local basis of
//! lattice images, `rho(r) = sum_ij P_ij phi_i(r) phi_j(r)` with
//! `P_ij = D(T_j - T_i)_{mu_i nu_j}`. Labelling every image by its parent
//! atom makes a Gamma-point displacement of an atom the molecular
//! displacement of that label, and the grid point rides its owner. The
//! molecular XC Hessian kernel therefore applies batch by batch, with the
//! periodic Becke weight derivatives of [`super::grid`]. The XC Fock
//! derivatives `D_A V_xc` of the local basis fold back into image blocks by
//! the translation difference `T_j - T_i`.

use crate::basis::{n_cart, n_sph, MolBasis, MolShell};
use crate::error::Result;
use crate::hessian::mgga_terms::mgga_batch_terms;
use crate::hessian::xc_terms::{batch_terms, BatchPoints, WeightResponse};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::grid::{periodic_weight_hessian, PeriodicGrid};
use crate::periodic::integrals::ImageBlock;
use crate::periodic::xc::xc_ao_images;
use crate::periodic::{Cell, Lattice};
use crate::system::Molecule;
use crate::xc::{Family, GridVars, ResolvedFunctional, XcOut};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};

struct PeriodicWeights<'a> {
    molecule: &'a Molecule,
    lattice: &'a Lattice,
    shell: i64,
}

impl WeightResponse for PeriodicWeights<'_> {
    fn derivatives(&self, point: Vec3, owner: usize) -> (Vec<f64>, Vec<f64>) {
        let w = periodic_weight_hessian(self.molecule, self.lattice, point, owner, self.shell);
        (w.first, w.second)
    }
}

/// XC pieces of the Gamma-point Hessian at a fixed density.
pub struct PeriodicXcHessian {
    /// `D_A D_B E_xc`, `3N x 3N` (Hartree/Bohr^2).
    pub skeleton: Matrix,
    /// `D_A V_xc^spin` as image blocks, `[spin][3 * atom + axis]`.
    pub vxc_derivative: [Vec<Vec<ImageBlock>>; 2],
}

/// Integer lattice index of a translation.
fn index_of(lattice: &Lattice, t: Vec3) -> [i64; 3] {
    let f = lattice.to_fractional(t);
    [
        f[0].round() as i64,
        f[1].round() as i64,
        f[2].round() as i64,
    ]
}

/// Gamma-point XC skeleton Hessian and XC Fock derivatives of the spin
/// density images `density[spin]` (which must reach twice `ao_cutoff`),
/// on the periodic grid of `grid_options`/`grid_shell`.
pub fn periodic_xc_hessian(
    cell: &Cell,
    functional: &ResolvedFunctional,
    density: [&[ImageBlock]; 2],
    grid_options: &crate::grid::GridOptions,
    grid_shell: i64,
    ao_cutoff: f64,
) -> Result<PeriodicXcHessian> {
    let molecule = &cell.molecule;
    let lattice = &cell.lattice;
    let basis = &cell.basis;
    let natoms = molecule.len();
    let n = basis.nao_sph;
    let dim = 3 * natoms;
    let grid = PeriodicGrid::build(molecule, lattice, grid_options, grid_shell)?;
    let weights = PeriodicWeights {
        molecule,
        lattice,
        shell: grid_shell,
    };
    let kernel = |vars: &GridVars, _start: usize, _npts: usize| -> Result<XcOut> {
        let mut out = XcOut::default();
        functional.eval(vars, 2, &mut out)?;
        Ok(out)
    };
    // Density image blocks by integer lattice index: the local density of a
    // batch looks up one block per pair of image shells.
    let images_of: [HashMap<[i64; 3], &Matrix>; 2] = std::array::from_fn(|spin| {
        density[spin]
            .iter()
            .map(|b| (index_of(lattice, b.translation), &b.matrix))
            .collect()
    });
    const BATCH: usize = 64;
    let ranges: Vec<(usize, usize)> = (0..grid.points.len())
        .step_by(BATCH)
        .map(|s| (s, (s + BATCH).min(grid.points.len())))
        .collect();
    type Folded = [Vec<BTreeMap<[i64; 3], Matrix>>; 2];
    let batch_terms_at = |&(start, end): &(usize, usize)| -> Result<(Matrix, Folded)> {
        let points = &grid.points[start..end];
        // Local basis: every (shell, image) within the AO cutoff of a
        // point of the batch, labelled by its parent atom.
        let images = xc_ao_images(basis, lattice, points, ao_cutoff);
        let mut shells: Vec<MolShell> = Vec::new();
        let mut origin: Vec<(usize, Vec3)> = Vec::new();
        let meta = functional.family == Family::MetaGga;
        let mut active_shells = Vec::new();
        let (mut sph, mut cart, mut max_l) = (0usize, 0usize, 0u8);
        let mut selected = vec![Vec::new(); basis.shells.len()];
        for image in &images {
            for &(point, shell) in &image.point_shells {
                selected[shell].push(point);
            }
            for (s, selected_points) in selected.iter_mut().enumerate() {
                if selected_points.is_empty() {
                    continue;
                }
                let mut shell = basis.shells[s].clone();
                shell.center += image.translation;
                shell.sph_off = sph;
                shell.cart_off = cart;
                sph += n_sph(shell.l);
                cart += n_cart(shell.l);
                max_l = max_l.max(shell.l);
                shells.push(shell);
                origin.push((s, image.translation));
                let mut active = vec![false; points.len()];
                for point in selected_points.drain(..) {
                    active[point] = true;
                }
                active_shells.push(active);
            }
        }
        let local = MolBasis {
            shells,
            nao_sph: sph,
            nao_cart: cart,
            max_l,
            ecps: Vec::new(),
            n_core_electrons: 0,
        };
        // Lattice index of every local shell; a pair of shells uses the
        // image block of the difference of their indices.
        let index: Vec<[i64; 3]> = origin
            .iter()
            .map(|(_, translation)| index_of(lattice, *translation))
            .collect();
        let key =
            |a: usize, b: usize| -> [i64; 3] { std::array::from_fn(|k| index[b][k] - index[a][k]) };
        // Local density P_ij = D(T_j - T_i)_{mu_i nu_j}.
        let mut p = [Matrix::zeros(sph, sph), Matrix::zeros(sph, sph)];
        for (a, (sa, _)) in origin.iter().enumerate() {
            let shell_a = &local.shells[a];
            let home_a = &basis.shells[*sa];
            for (b, (sb, _)) in origin.iter().enumerate() {
                let shell_b = &local.shells[b];
                let home_b = &basis.shells[*sb];
                let pair = key(a, b);
                for spin in 0..2 {
                    let Some(d) = images_of[spin].get(&pair) else {
                        continue;
                    };
                    for i in 0..n_sph(shell_a.l) {
                        for j in 0..n_sph(shell_b.l) {
                            p[spin][(shell_a.sph_off + i, shell_b.sph_off + j)] =
                                d[(home_a.sph_off + i, home_b.sph_off + j)];
                        }
                    }
                }
            }
        }
        let mut skeleton = Matrix::zeros(dim, dim);
        let zeros = || -> [Matrix; 3] { std::array::from_fn(|_| Matrix::zeros(sph, sph)) };
        let mut vxc_local: [Vec<[Matrix; 3]>; 2] =
            std::array::from_fn(|_| (0..natoms).map(|_| zeros()).collect());
        let batch = BatchPoints {
            points,
            weights: &grid.weights[start..end],
            raw_weights: &grid.raw_weights[start..end],
            owners: &grid.atom_of[start..end],
            active_shells: Some(&active_shells),
        };
        if meta {
            mgga_batch_terms(
                &batch,
                &local,
                [&p[0], &p[1]],
                natoms,
                functional,
                &weights,
                &mut skeleton,
                &mut vxc_local,
            )?;
        } else {
            batch_terms(
                start,
                &batch,
                &local,
                [&p[0], &p[1]],
                natoms,
                functional.family == Family::Gga,
                &kernel,
                &weights,
                &mut skeleton,
                &mut vxc_local,
            )?;
        }
        // Fold the local Fock derivatives into image blocks: one slot per
        // distinct pair key, found once per pair of shells.
        let mut slot_of: HashMap<[i64; 3], usize> = HashMap::new();
        let mut keys: Vec<[i64; 3]> = Vec::new();
        let mut pair_slots = vec![0usize; origin.len() * origin.len()];
        for a in 0..origin.len() {
            for b in 0..origin.len() {
                let pair = key(a, b);
                let slot = *slot_of.entry(pair).or_insert_with(|| {
                    keys.push(pair);
                    keys.len() - 1
                });
                pair_slots[a * origin.len() + b] = slot;
            }
        }
        let mut blocks: [Vec<Vec<Matrix>>; 2] = std::array::from_fn(|_| {
            (0..dim)
                .map(|_| (0..keys.len()).map(|_| Matrix::zeros(n, n)).collect())
                .collect()
        });
        for (a, (sa, _)) in origin.iter().enumerate() {
            let shell_a = &local.shells[a];
            let home_a = &basis.shells[*sa];
            for (b, (sb, _)) in origin.iter().enumerate() {
                let shell_b = &local.shells[b];
                let home_b = &basis.shells[*sb];
                let slot = pair_slots[a * origin.len() + b];
                for spin in 0..2 {
                    for atom in 0..natoms {
                        for axis in 0..3 {
                            let m = &vxc_local[spin][atom][axis];
                            let block = &mut blocks[spin][3 * atom + axis][slot];
                            for i in 0..n_sph(shell_a.l) {
                                for j in 0..n_sph(shell_b.l) {
                                    block[(home_a.sph_off + i, home_b.sph_off + j)] +=
                                        m[(shell_a.sph_off + i, shell_b.sph_off + j)];
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut folded: Folded = std::array::from_fn(|_| vec![BTreeMap::new(); dim]);
        for spin in 0..2 {
            for (coordinate, per_key) in blocks[spin].iter_mut().enumerate() {
                for (slot, block) in per_key.drain(..).enumerate() {
                    folded[spin][coordinate].insert(keys[slot], block);
                }
            }
        }
        Ok((skeleton, folded))
    };
    let mut skeleton = Matrix::zeros(dim, dim);
    let mut folded: Folded = std::array::from_fn(|_| vec![BTreeMap::new(); dim]);
    // Keep at most one wave of folded Fock matrices live. Reduce in the
    // original batch order so the floating-point sum is reproducible.
    let wave_size = 2 * rayon::current_num_threads().max(1);
    for wave in ranges.chunks(wave_size) {
        let partials: Vec<_> = wave.par_iter().map(batch_terms_at).collect();
        for part in partials {
            let (s, f) = part?;
            for (x, y) in skeleton.as_mut_slice().iter_mut().zip(s.as_slice()) {
                *x += *y;
            }
            for (target_spin, source_spin) in folded.iter_mut().zip(f) {
                for (target, source) in target_spin.iter_mut().zip(source_spin) {
                    for (key, block) in source {
                        let entry = target.entry(key).or_insert_with(|| Matrix::zeros(n, n));
                        for (x, y) in entry.as_mut_slice().iter_mut().zip(block.as_slice()) {
                            *x += *y;
                        }
                    }
                }
            }
        }
    }
    let vxc_derivative = folded.map(|per_spin| {
        per_spin
            .into_iter()
            .map(|blocks| {
                blocks
                    .into_iter()
                    .map(|(index, matrix)| ImageBlock {
                        translation: lattice.translation(index),
                        matrix,
                    })
                    .collect()
            })
            .collect()
    });
    Ok(PeriodicXcHessian {
        skeleton,
        vxc_derivative,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acceptance::{components, stencils};
    use crate::grid::GridOptions;
    use crate::periodic::bloch::BlochMatrix;
    use crate::periodic::integrals::bloch;
    use crate::periodic::scf::{PeriodicScfOptions, PeriodicScfResult};
    use crate::periodic::xc::{xc_contribution, AoDerivativeOrder, PeriodicXcGrid};
    use crate::system::Atom;
    use crate::xc::{resolve, FunctionalSpec};
    use stencils::Stencil;

    const BASIS: &str = "H 0\nS 2 1.00\n 2.4 0.6\n 0.55 0.5\nP 2 1.00\n 1.6 0.6\n 0.5 0.5\n****\n";
    const SCALE: f64 = 2.2;
    const CUTOFF: f64 = 9.0;

    fn cell(atom: usize, axis: usize, h: f64) -> Cell {
        displaced_cell(atom, axis, h, false)
    }

    fn displaced_cell(atom: usize, axis: usize, h: f64, three_atoms: bool) -> Cell {
        let scale = if three_atoms { 1.4 } else { SCALE };
        let lattice = Lattice::new(
            Vec3::new(4.2, 0.0, 0.0) * scale,
            Vec3::new(0.3, 4.6, 0.0) * scale,
            Vec3::new(0.2, -0.25, 4.9) * scale,
        );
        let mut atoms = vec![
            Atom {
                z: 1,
                position: Vec3::new(0.1, 0.2, -0.3),
            },
            Atom {
                z: 1,
                position: Vec3::new(1.2, 0.9, 0.8),
            },
        ];
        if three_atoms {
            atoms.push(Atom {
                z: 1,
                position: Vec3::new(-0.42, 1.64, 0.57),
            });
        }
        match axis {
            0 => atoms[atom].position.x += h,
            1 => atoms[atom].position.y += h,
            _ => atoms[atom].position.z += h,
        }
        let set = crate::basis::g94::parse("xc-hessian-test", BASIS).unwrap();
        Cell::new(Molecule::new(atoms), lattice, &set, CUTOFF).unwrap()
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

    /// Positive-definite spin densities, different for the two spins.
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

    #[test]
    fn periodic_semilocal_hessian_and_fock_fit_all_components() {
        for functional in ["svwn5", "pbe", "tpss"] {
            fit_hessian_and_fock(functional);
        }
    }

    fn fit_hessian_and_fock(name: &str) {
        let options = PeriodicScfOptions {
            k_mesh: [3, 1, 1],
            ..options()
        };
        let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
        let base = displaced_cell(0, 0, 0.0, true);
        let k_points = base.k_points(options.k_mesh);
        let mut reference = result(&base, k_points.clone());
        // Gamma is self-conjugate. The two nonzero points are conjugates
        // modulo a reciprocal vector and probe the imaginary Fock elements.
        for (s, spin) in reference.density.iter_mut().enumerate() {
            for i in 0..base.nao() {
                for j in i + 1..base.nao() {
                    let value = 0.009 * (0.34 + (3 * i + j + s) as f64 * 0.47).sin();
                    for (k, density) in spin.iter_mut().enumerate() {
                        let sign = [0.0, 1.0, -1.0][k];
                        density.im[(i, j)] = sign * value;
                        density.im[(j, i)] = -sign * value;
                    }
                }
            }
        }
        let images = reference.density.each_ref().map(|spin| {
            base.density_images_from_k(
                &k_points,
                &spin
                    .iter()
                    .map(|p| (p.re.clone(), p.im.clone()))
                    .collect::<Vec<_>>(),
                2.0 * CUTOFF + 1.0,
            )
            .unwrap()
        });
        let analytic = periodic_xc_hessian(
            &base,
            &functional,
            images.each_ref().map(Vec::as_slice),
            &options.grid,
            options.grid_shell,
            CUTOFF,
        )
        .unwrap();
        let n = base.nao();
        let dim = 3 * base.molecule.len();
        // Test Gamma and one representative of the conjugate k pair.
        let entries: Vec<_> = (0..2)
            .flat_map(|k| {
                (0..n).flat_map(move |i| {
                    (i..n).flat_map(move |j| {
                        (0..1 + usize::from(k != 0 && i != j)).map(move |part| (k, i, j, part))
                    })
                })
            })
            .collect();
        let mut labels = Vec::new();
        let mut exact = Vec::new();
        for col in 0..dim {
            for row in 0..dim {
                labels.push(format!("periodic {name} skeleton [{row},{col}]"));
                exact.push(analytic.skeleton[(row, col)]);
            }
            for spin in 0..2 {
                let matrices =
                    [0, 1].map(|k| bloch(&analytic.vxc_derivative[spin][col], k_points[k]));
                for &(k, i, j, part) in &entries {
                    let tag = ["real", "imag"][part];
                    labels.push(format!(
                        "periodic {name} Fock {tag} [{spin},{col},{k},{i},{j}]"
                    ));
                    let (re, im) = &matrices[k];
                    exact.push([re, im][part][(i, j)]);
                }
            }
        }
        let stride = dim + 2 * entries.len();
        let evaluate = |col: usize, h: f64| {
            let moved = displaced_cell(col / 3, col % 3, h, true);
            let mut values: Vec<_> = crate::periodic::xc_gradient::xc_gradient(
                &moved,
                &functional,
                &crate::periodic::stationary_density::StationaryDensity::from_scf(&reference),
                &options,
            )
            .unwrap()
            .iter()
            .flat_map(|g| [g.x, g.y, g.z])
            .collect();
            let grid = PeriodicXcGrid::build(
                &moved,
                &k_points,
                &options.grid,
                options.grid_shell,
                CUTOFF,
                AoDerivativeOrder::for_scf(functional.family),
            )
            .unwrap();
            let potential = xc_contribution(&functional, &grid, &reference.density)
                .unwrap()
                .vxc;
            for spin in &potential {
                for &(k, i, j, part) in &entries {
                    values.push([&spin[k].re, &spin[k].im][part][(i, j)]);
                }
            }
            values
        };
        let mut samples = BTreeMap::new();
        let mut failures = Vec::new();
        stencils::fit_components_staged(
            &labels,
            &[
                ([0.032, 0.016, 0.008, 0.004], Stencil::Second),
                ([0.016, 0.008, 0.004, 0.002], Stencil::Second),
                ([0.04, 0.03, 0.02, 0.01], Stencil::Fourth),
            ],
            |steps, stencil| {
                let mut errors = vec![[0.0; 4]; exact.len()];
                for col in 0..dim {
                    for (s, &h) in steps.iter().enumerate() {
                        let mut at = |d: f64| {
                            samples
                                .entry((col, d.to_bits()))
                                .or_insert_with(|| evaluate(col, d))
                                .clone()
                        };
                        let (plus, minus) = (at(h), at(-h));
                        let (plus2, minus2) = if stencil == Stencil::Fourth {
                            (at(2.0 * h), at(-2.0 * h))
                        } else {
                            (vec![0.0; stride], vec![0.0; stride])
                        };
                        for i in 0..stride {
                            let index = col * stride + i;
                            errors[index][s] = (stencil
                                .first(plus[i], minus[i], plus2[i], minus2[i], h)
                                - exact[index])
                                .abs();
                        }
                    }
                }
                errors
            },
            &mut failures,
        );
        components::assert_no_failures(name, &failures);
    }

    #[test]
    fn periodic_xc_hessian_and_fock_derivatives_pass_four_step_fits() {
        let steps = [1.6e-2, 8e-3, 4e-3, 2e-3];
        let options = options();
        for name in ["svwn5", "pbe"] {
            let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
            let base = cell(0, 0, 0.0);
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
            let analytic = periodic_xc_hessian(
                &base,
                &functional,
                [&images[0], &images[1]],
                &options.grid,
                options.grid_shell,
                CUTOFF,
            )
            .unwrap();
            let gradient = |atom: usize, axis: usize, h: f64| -> Vec<f64> {
                let moved = cell(atom, axis, h);
                crate::periodic::xc_gradient::xc_gradient(
                    &moved,
                    &functional,
                    &crate::periodic::stationary_density::StationaryDensity::from_scf(&reference),
                    &options,
                )
                .unwrap()
                .iter()
                .flat_map(|g| [g.x, g.y, g.z])
                .collect()
            };
            let vxc = |atom: usize, axis: usize, h: f64| -> [Vec<BlochMatrix>; 2] {
                let moved = cell(atom, axis, h);
                let grid = PeriodicXcGrid::build(
                    &moved,
                    &k_points,
                    &options.grid,
                    options.grid_shell,
                    CUTOFF,
                    AoDerivativeOrder::for_scf(functional.family),
                )
                .unwrap();
                xc_contribution(&functional, &grid, &reference.density)
                    .unwrap()
                    .vxc
            };
            for (atom, axis) in [(0usize, 0usize), (1, 2)] {
                let column = 3 * atom + axis;
                let mut skeleton_errors = Vec::new();
                let mut fock_errors = Vec::new();
                for &h in &steps {
                    let (plus, minus) = (gradient(atom, axis, h), gradient(atom, axis, -h));
                    skeleton_errors.push(
                        (0..6)
                            .map(|row| {
                                ((plus[row] - minus[row]) / (2.0 * h)
                                    - analytic.skeleton[(row, column)])
                                    .abs()
                            })
                            .fold(0.0_f64, |m, e| {
                                if e.is_finite() {
                                    m.max(e)
                                } else {
                                    f64::INFINITY
                                }
                            }),
                    );
                    let (vp, vm) = (vxc(atom, axis, h), vxc(atom, axis, -h));
                    let mut worst = 0.0_f64;
                    for spin in 0..2 {
                        for (ik, k) in k_points.iter().enumerate() {
                            let (re, im) = bloch(&analytic.vxc_derivative[spin][column], *k);
                            for (plus_part, minus_part, exact) in [
                                (&vp[spin][ik].re, &vm[spin][ik].re, &re),
                                (&vp[spin][ik].im, &vm[spin][ik].im, &im),
                            ] {
                                for (idx, value) in exact.as_slice().iter().enumerate() {
                                    let fd = (plus_part.as_slice()[idx]
                                        - minus_part.as_slice()[idx])
                                        / (2.0 * h);
                                    let e = (fd - value).abs();
                                    worst = if e.is_finite() {
                                        worst.max(e)
                                    } else {
                                        f64::INFINITY
                                    };
                                }
                            }
                        }
                    }
                    fock_errors.push(worst);
                }
                for (label, errors) in [
                    ("skeleton", &skeleton_errors),
                    ("Fock derivative", &fock_errors),
                ] {
                    let (p, r2) = fit(&steps, errors);
                    println!("XC {name} {label} column ({atom},{axis}): errors {errors:?} p = {p:.6} R2 = {r2:.8}");
                    assert!(
                        (p - 2.0).abs() < 0.05 && r2 > 0.9999,
                        "{name} {label} ({atom},{axis}): p = {p}, R2 = {r2}"
                    );
                }
            }
            for row in 0..6 {
                for col in 0..6 {
                    assert!(
                        (analytic.skeleton[(row, col)] - analytic.skeleton[(col, row)]).abs()
                            < 1e-10
                    );
                }
            }
        }
    }
}
