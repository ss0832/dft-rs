// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! The shell quartets of the truncated periodic exchange sum
//!
//! ```text
//! sum_{T,C,D} D_mu,nu(T) D_lambda,sigma(D-C) (mu(0) lambda(C) | sigma(D) nu(T)),
//! ```
//!
//! selected by the separations of shell centres. A quartet is kept when
//! both charge distributions (`mu-lambda`, `sigma-nu`) and both density
//! pairs (`mu-nu`, `lambda-sigma`) have their centres at most `cutoff`
//! apart. The criterion depends only on separations, so a supercell selects
//! the quartets of its primitive cell and the truncated sum keeps the
//! translation symmetry of the crystal. (Selecting the cells instead by
//! `|T|, |C|, |D| <= cutoff` broke that symmetry: a two-cell PBE0 chain at a
//! 14 Bohr cutoff differed from its primitive cell by 5.4e-7 Hartree.) The
//! criterion is also symmetric under the exchange of the two density
//! factors, so the truncated energy is a symmetric quadratic form of the
//! density and the exchange matrix is its exact density derivative.
//!
//! Every kernel that differentiates the sum (gradient, strain, force
//! constants, first-order exchange matrices) enumerates the quartets
//! through [`ExchangeCells::visit`], so they differentiate the same sum.

use crate::basis::{MolBasis, MolShell};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::integrals::ImageBlock;
use crate::periodic::{Lattice, PeriodicGeometry};
use std::collections::HashMap;

/// Integer coordinates in the periodic subspace, with zero on open axes.
fn image_label(dual: [Vec3; 3], translation: Vec3) -> [i64; 3] {
    dual.map(|axis| translation.dot(axis).round() as i64)
}

/// Whether two shell centres are at most `cutoff` apart.
fn within(a: &MolShell, b: &MolShell, cutoff: f64) -> bool {
    (a.center - b.center).norm() <= cutoff
}

/// The largest distance between two shell centres of the home cell.
fn shell_spread(mb: &MolBasis) -> f64 {
    mb.shells
        .iter()
        .flat_map(|a| mb.shells.iter().map(move |b| (a.center - b.center).norm()))
        .fold(0.0_f64, f64::max)
}

/// One selected quartet `(mu(0) lambda(C) | sigma(D) nu(T))`.
pub(crate) struct Quartet<'a> {
    /// The shells `[mu, lambda, sigma, nu]` at their translated centres.
    pub(crate) shells: [&'a MolShell; 4],
    /// Home-cell spherical offsets of the four shells.
    pub(crate) offsets: [usize; 4],
    /// Index in [`ExchangeCells::near`] of the inner density image `D - C`.
    pub(crate) source: usize,
    /// Translation C of lambda(C), needed by an independent (k+q,k) density.
    pub(crate) lambda_translation: Vec3,
}

/// The cells and translated shells of the selected exchange quartets.
pub(crate) struct ExchangeCells {
    /// Cells that can hold `nu(T)` or `lambda(C)` of a selected quartet,
    /// and every inner density image `D - C`: the translations within
    /// `cutoff` plus the spread of the home-cell shells.
    pub(crate) near: Vec<Vec3>,
    near_labels: Vec<[i64; 3]>,
    dual: [Vec3; 3],
    /// The basis translated into each cell of `far` (`far[i]` holds the
    /// shells `nsh i .. nsh (i + 1)`).
    shells: Vec<MolShell>,
    /// Cells that can hold `sigma(D)`, `D = T - U` for `T`, `U` in `near`.
    far: HashMap<[i64; 3], usize>,
    near_index: HashMap<[i64; 3], usize>,
    nsh: usize,
    cutoff: f64,
}

impl ExchangeCells {
    /// Reach of every density pair selected by the separation-based sum.
    pub(crate) fn density_image_radius(mb: &MolBasis, cutoff: f64) -> f64 {
        cutoff + shell_spread(mb)
    }
    /// The cells for the exchange sum of `mb` with the translations
    /// returned by `translations(radius)` (all lattice translations of
    /// length at most `radius`, sorted by length, the home cell first).
    fn new(
        mb: &MolBasis,
        cutoff: f64,
        reciprocal: [Vec3; 3],
        translations: impl Fn(f64) -> crate::Result<Vec<Vec3>>,
    ) -> crate::Result<Self> {
        let reach = Self::density_image_radius(mb, cutoff);
        let near = translations(reach)?;
        // Keep boundary differences whose recomputed Cartesian norm
        // rounds above the triangle-inequality bound.
        let far_cells = translations(2.0 * reach * (1.0 + 32.0 * f64::EPSILON))?;
        let dual = reciprocal.map(|axis| axis / std::f64::consts::TAU);
        let near_labels: Vec<_> = near.iter().map(|&t| image_label(dual, t)).collect();
        let nsh = mb.shells.len();
        let mut shells = Vec::with_capacity(nsh * far_cells.len());
        for t in &far_cells {
            for shell in &mb.shells {
                let mut moved = shell.clone();
                moved.center += *t;
                shells.push(moved);
            }
        }
        let far = far_cells
            .iter()
            .enumerate()
            .map(|(i, &t)| (image_label(dual, t), i))
            .collect();
        let near_index = near_labels.iter().copied().zip(0..near.len()).collect();
        Ok(Self {
            near,
            near_labels,
            dual,
            shells,
            far,
            near_index,
            nsh,
            cutoff,
        })
    }

    /// The cells of a three-dimensional lattice.
    pub(crate) fn for_lattice(mb: &MolBasis, lattice: &Lattice, cutoff: f64) -> Self {
        Self::new(mb, cutoff, lattice.reciprocal(), |radius| {
            Ok(lattice.translations_within(radius))
        })
        .expect("lattice translations")
    }

    pub(crate) fn for_geometry(
        mb: &MolBasis,
        geometry: &PeriodicGeometry,
        cutoff: f64,
    ) -> crate::Result<Self> {
        Self::new(mb, cutoff, geometry.reciprocal(), |radius| {
            geometry.translations_within(radius)
        })
    }

    /// The density block of every cell of `near`.
    pub(crate) fn densities<'a>(&self, density: &'a [ImageBlock]) -> Vec<Option<&'a Matrix>> {
        let mut out = vec![None; self.near.len()];
        for block in density {
            let label = image_label(self.dual, block.translation);
            if let Some(&index) = self.near_index.get(&label) {
                let delta = (block.translation - self.near[index]).to_array();
                if delta.iter().all(|v| v.abs() <= 1e-6) {
                    out[index] = Some(&block.matrix);
                }
            }
        }
        out
    }

    fn shell(&self, cell: usize, s: usize) -> &MolShell {
        &self.shells[cell * self.nsh + s]
    }

    /// Visit every selected quartet with `nu` in the cell `near[target]`.
    /// `pair(sa, sb)` may reject a shell pair `(mu, nu)` before its quartets
    /// are enumerated; `source(c)` may reject an inner density image.
    pub(crate) fn visit<'a>(
        &'a self,
        mb: &MolBasis,
        target: usize,
        mut pair: impl FnMut(usize, usize) -> bool,
        mut source: impl FnMut(usize) -> bool,
        mut f: impl FnMut(&Quartet<'a>),
    ) {
        let nsh = self.nsh;
        let t = self.near_labels[target];
        let nu_cell = self.far[&t];
        for sa in 0..nsh {
            let mu = self.shell(0, sa);
            for sb in 0..nsh {
                let nu = self.shell(nu_cell, sb);
                if !within(mu, nu, self.cutoff) || !pair(sa, sb) {
                    continue;
                }
                for (&c, &translation) in self.near_labels.iter().zip(&self.near) {
                    let lambda_cell = self.far[&c];
                    let lambdas: Vec<usize> = (0..nsh)
                        .filter(|&sc| within(mu, self.shell(lambda_cell, sc), self.cutoff))
                        .collect();
                    if lambdas.is_empty() {
                        continue;
                    }
                    for &u in &self.near_labels {
                        let d: [i64; 3] = std::array::from_fn(|i| t[i] - u[i]);
                        let inner = std::array::from_fn(|i| d[i] - c[i]);
                        let Some(&inner) = self.near_index.get(&inner) else {
                            continue;
                        };
                        if !source(inner) {
                            continue;
                        }
                        let sigma_cell = self.far[&d];
                        for &sc in &lambdas {
                            let lambda = self.shell(lambda_cell, sc);
                            for sd in 0..nsh {
                                let sigma = self.shell(sigma_cell, sd);
                                if !within(sigma, nu, self.cutoff)
                                    || !within(lambda, sigma, self.cutoff)
                                {
                                    continue;
                                }
                                f(&Quartet {
                                    shells: [mu, lambda, sigma, nu],
                                    offsets: [
                                        mb.shells[sa].sph_off,
                                        mb.shells[sc].sph_off,
                                        mb.shells[sd].sph_off,
                                        mb.shells[sb].sph_off,
                                    ],
                                    source: inner,
                                    lambda_translation: translation,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::{run, Controls, System, Value};
    use crate::periodic::features::PeriodicBoundary;
    use crate::system::{Atom, Molecule};

    #[test]
    fn strained_exchange_images_use_integer_lattice_differences() {
        let shear = |mut v: Vec3| {
            v.x += 0.0025 * v.y;
            v
        };
        let lattice = Lattice::new(
            shear(Vec3::new(8.4, 0.7, 0.3) * 0.55),
            shear(Vec3::new(0.4, 8.9, 0.6) * 0.55),
            shear(Vec3::new(0.2, -0.5, 9.2) * 0.55),
        );
        let molecule = Molecule::new(
            [
                Vec3::new(0.13, -0.21, 0.07),
                Vec3::new(2.0, 0.45, -0.31),
                Vec3::new(-0.46, 1.29, 1.62),
            ]
            .map(|position| Atom {
                z: 1,
                position: shear(position),
            })
            .to_vec(),
        );
        let mb = MolBasis::build(&molecule, &crate::basis::data::load("sto-3g").unwrap()).unwrap();
        for axes in [[true; 3], [true, false, true]] {
            let geometry =
                PeriodicGeometry::new(lattice.clone(), PeriodicBoundary { axes }).unwrap();
            let cells = ExchangeCells::for_geometry(&mb, &geometry, 3.8).unwrap();
            let target = cells.near_index[&[-1, 0, 0]];
            assert!(cells.near_index.contains_key(&[-1, 0, 1]));
            let mut n = 0;
            cells.visit(&mb, target, |_, _| true, |_| true, |_| n += 1);
            assert!(n > 0, "{axes:?}");
        }
    }

    /// `cells` primitive cells of a two-atom hydrogen chain along x, with
    /// the second atom of the cell `moved` displaced by `shift` along x.
    fn chain(cells: usize, moved: usize, shift: f64) -> System {
        let mut positions = Vec::new();
        for c in 0..cells {
            let x = 3.6 * c as f64;
            positions.push([x, 0.0, 0.0]);
            let dx = if c == moved { shift } else { 0.0 };
            positions.push([x + 1.5 + dx, 0.1, 0.0]);
        }
        System {
            numbers: vec![1; 2 * cells],
            positions,
            cell: Some([
                [3.6 * cells as f64, 0.0, 0.0],
                [0.0, 7.0, 0.0],
                [0.0, 0.0, 7.0],
            ]),
            pbc: [true; 3],
        }
    }

    fn energy_per_cell(xc: &str, cells: usize, moved: usize, shift: f64) -> f64 {
        let controls = Controls::from_value(&Value::Object(vec![
            ("xc".to_string(), Value::Str(xc.into())),
            ("basis".to_string(), Value::Str("sto-3g".into())),
            (
                "k_mesh".to_string(),
                Value::Array(vec![
                    Value::Int((4 / cells) as i64),
                    Value::Int(1),
                    Value::Int(1),
                ]),
            ),
            ("overlap_cutoff".to_string(), Value::Float(12.0)),
            ("exchange_cutoff".to_string(), Value::Float(14.0)),
            ("grid_level".to_string(), Value::Int(2)),
            ("grid_shell".to_string(), Value::Int(1)),
            ("conv_tol".to_string(), Value::Float(1e-12)),
            ("conv_tol_dp".to_string(), Value::Float(1e-10)),
            ("max_iter".to_string(), Value::Int(200)),
        ]))
        .unwrap();
        let report = run("periodic-energy", &chain(cells, moved, shift), &controls).unwrap();
        report.get("energy").and_then(Value::as_f64).unwrap() / cells as f64
    }

    /// A doubled Hartree-Fock or hybrid chain has the energy per cell of its
    /// primitive cell, and moving an atom or its image in the other
    /// primitive cell of the supercell gives the same energy. (With the cells selected by
    /// `|T|, |C|, |D| <= cutoff` the doubled PBE0 chain differed from the
    /// primitive one by 5.4e-7 Hartree per cell at this cutoff.)
    #[test]
    fn a_supercell_has_the_exchange_of_its_primitive_cell() {
        for xc in ["hf", "pbe0"] {
            let primitive = energy_per_cell(xc, 1, 0, 0.0);
            let doubled = energy_per_cell(xc, 2, usize::MAX, 0.0);
            let first = energy_per_cell(xc, 2, 0, 0.05);
            let second = energy_per_cell(xc, 2, 1, 0.05);
            println!(
                "{xc}: primitive {primitive:.12}, doubled {doubled:.12} (difference {:.3e}); atom moved in cell 0 {first:.12}, in cell 1 {second:.12} (difference {:.3e})",
                doubled - primitive,
                second - first
            );
            assert!((doubled - primitive).abs() < 1e-9, "{xc}");
            assert!((second - first).abs() < 1e-9, "{xc}");
        }
    }
}
