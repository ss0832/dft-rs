// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Periodic boundary conditions: the lattice and its sums.
//!
//! This layer contains the geometry of a periodic cell, Ewald electrostatics,
//! Bloch-summed one-electron/ECP integrals, Hartree and exact exchange,
//! finite-k LDA/GGA/meta-GGA SCF with insulating or Fermi-Dirac occupations,
//! and analytic Mermin-free-energy forces. Masked geometry and dimensional
//! point/Gaussian Coulomb energy and forces support 1D/2D/3D, including
//! explicitly compensated charged cells. The reduced-dimensional electronic
//! Gaussian overlap/kinetic, atomic derivatives and open-direction quadrature
//! are supplied by `basis_boundary`, contracted nuclear/Hartree operators by
//! `coulomb_boundary`, and spin/Bloch semilocal potentials by `xc_boundary`.
//! Complete dimensional SCF/DFPT and periodic VV10/CHG remain tracked work.
//!
//! # Why this layer is separate, and why it is testable alone
//!
//! Periodic quantum chemistry has no convenient reference implementation to
//! diff against the way a molecular code can be diffed against PySCF, so
//! the correctness argument has to come from identities the answer must
//! satisfy regardless of implementation. The lattice sums are where those
//! identities are sharpest: the Madelung constant of rocksalt is a known
//! number to ten digits, the Ewald splitting parameter must divide out
//! exactly, and a supercell must give the same energy per formula unit as
//! the primitive cell. Getting those right first means the electronic
//! layer is built on something already pinned.

pub mod band_transport;
pub mod basis_boundary;
pub mod berry_phase;
pub mod bloch;
pub mod cell;
pub mod coherent_transport;
pub mod coulomb;
pub mod coulomb_boundary;
pub mod dielectric;
pub mod dispersion;
pub mod elastic;
pub mod electron_phonon;
pub mod electrostatics;
pub mod eliashberg;
pub mod eos;
pub mod ewald;
pub mod exchange;
pub(crate) mod exchange_quartets;
pub(crate) mod exchange_response;
pub mod features;
pub mod fermi_thermodynamics;
pub mod field;
pub mod forces;
pub mod gaussian_coulomb;
pub mod geometry;
pub mod gradient;
pub mod grid;
pub mod gruneisen;
pub mod hessian;
pub mod integrals;
pub(crate) mod learned_response;
pub mod magnetic_cell;
pub mod magnetic_scf;
pub mod nonlocal;
pub mod nuclear;
pub mod occupations;
pub mod optimizer;
pub(crate) mod path_series;
pub mod phonon;
pub mod phonon_bands;
pub mod planewave;
pub mod quasi_harmonic;
pub mod quasiperiodic;
pub mod roks;
pub mod scattering;
pub mod scf;
pub mod scf_boundary;
pub mod screened_coulomb;
pub mod spin_susceptibility;
pub mod spinor_gradient;
pub mod spinor_scf;
pub mod spinor_xc;
pub(crate) mod stationary_density;
pub mod strain;
pub mod strain_fock;
pub mod strain_jets;
pub mod stress;
pub mod uniform_open;
pub mod xc;
pub mod xc_boundary;
pub(crate) mod xc_gradient;
pub mod xc_hessian;
pub(crate) mod xc_strain;
pub(crate) mod xc_terms;

pub use basis_boundary::{periodic_basis, PeriodicBasisOptions, PeriodicBasisReport};
pub use bloch::BlochMatrix;
pub use cell::Cell;
pub use geometry::PeriodicGeometry;

use crate::math::Vec3;

/// A three-dimensional Bravais lattice.
#[derive(Clone, Debug)]
pub struct Lattice {
    /// Real-space vectors as rows: `a[0]`, `a[1]`, `a[2]` (Bohr).
    pub a: [Vec3; 3],
}

impl Lattice {
    pub fn new(a1: Vec3, a2: Vec3, a3: Vec3) -> Self {
        Self { a: [a1, a2, a3] }
    }

    /// Cubic cell of edge `l`.
    pub fn cubic(l: f64) -> Self {
        Self::new(
            Vec3::new(l, 0.0, 0.0),
            Vec3::new(0.0, l, 0.0),
            Vec3::new(0.0, 0.0, l),
        )
    }

    /// Signed cell volume `a1 . (a2 x a3)`.
    pub fn volume(&self) -> f64 {
        self.a[0].dot(cross(self.a[1], self.a[2]))
    }

    /// Reciprocal vectors `b_i` with `a_i . b_j = 2 pi delta_ij`.
    ///
    /// The `2 pi` is in the vectors themselves, so a plane wave is
    /// `exp(i G . r)` with `G = m1 b1 + m2 b2 + m3 b3` and no further
    /// factor — the convention that keeps the Ewald reciprocal sum free of
    /// stray `2 pi`s.
    pub fn reciprocal(&self) -> [Vec3; 3] {
        let v = self.volume();
        let f = 2.0 * std::f64::consts::PI / v;
        [
            cross(self.a[1], self.a[2]) * f,
            cross(self.a[2], self.a[0]) * f,
            cross(self.a[0], self.a[1]) * f,
        ]
    }

    /// Lattice translation `n0 a0 + n1 a1 + n2 a2`.
    pub fn translation(&self, n: [i64; 3]) -> Vec3 {
        self.a[0] * (n[0] as f64) + self.a[1] * (n[1] as f64) + self.a[2] * (n[2] as f64)
    }

    /// Fractional coordinates of a Cartesian point.
    pub fn to_fractional(&self, r: Vec3) -> [f64; 3] {
        let b = self.reciprocal();
        let inv = 1.0 / (2.0 * std::f64::consts::PI);
        [r.dot(b[0]) * inv, r.dot(b[1]) * inv, r.dot(b[2]) * inv]
    }

    /// Cartesian point from fractional coordinates.
    pub fn to_cartesian(&self, f: [f64; 3]) -> Vec3 {
        self.a[0] * f[0] + self.a[1] * f[1] + self.a[2] * f[2]
    }

    /// Wrap a point into the `[0,1)` fractional cell.
    pub fn wrap(&self, r: Vec3) -> Vec3 {
        let mut f = self.to_fractional(r);
        for v in f.iter_mut() {
            *v -= v.floor();
        }
        self.to_cartesian(f)
    }

    /// Exact shortest image of `d` under an arbitrary, possibly unreduced,
    /// lattice basis.
    ///
    /// Fractional rounding and a fixed neighbouring-cell search both fail
    /// for sufficiently skewed bases.  Starting from the rounded candidate
    /// gives an upper bound `R`.  Any better Cartesian vector `r` satisfies
    /// `|r . b_i|/(2 pi) <= R |b_i|/(2 pi)`, which gives finite guaranteed
    /// bounds for every lattice index.  Enumerating that box is therefore
    /// exact without assuming a Minkowski-reduced input cell.
    pub fn minimum_image(&self, d: Vec3) -> Vec3 {
        let f = self.to_fractional(d);
        let nearest = [
            f[0].round() as i64,
            f[1].round() as i64,
            f[2].round() as i64,
        ];
        let mut best = d - self.translation(nearest);
        let mut best2 = best.norm2();
        let reciprocal = self.reciprocal();
        let radius = best2.sqrt();
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut lower = [0i64; 3];
        let mut upper = [0i64; 3];
        for axis in 0..3 {
            let bound = radius * reciprocal[axis].norm() / two_pi + 16.0 * f64::EPSILON;
            lower[axis] = (f[axis] - bound).ceil() as i64;
            upper[axis] = (f[axis] + bound).floor() as i64;
        }
        for i in lower[0]..=upper[0] {
            for j in lower[1]..=upper[1] {
                for k in lower[2]..=upper[2] {
                    let cand = d - self.translation([i, j, k]);
                    let n2 = cand.norm2();
                    if n2 < best2 {
                        best2 = n2;
                        best = cand;
                    }
                }
            }
        }
        best
    }

    /// All translations whose length is at most `rcut`.
    ///
    /// The loop bounds come from the perpendicular widths of the cell, so
    /// no shell inside `rcut` is missed however skewed the cell is — a
    /// naive cubic bound on the indices silently truncates for triclinic
    /// lattices.
    pub fn translations_within(&self, rcut: f64) -> Vec<Vec3> {
        let b = self.reciprocal();
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut nmax = [0i64; 3];
        for i in 0..3 {
            // Perpendicular width along a_i is 2 pi / |b_i|.
            let width = two_pi / b[i].norm();
            nmax[i] = (rcut / width).ceil() as i64 + 1;
        }
        let mut out = Vec::new();
        for i in -nmax[0]..=nmax[0] {
            for j in -nmax[1]..=nmax[1] {
                for k in -nmax[2]..=nmax[2] {
                    let t = self.translation([i, j, k]);
                    if t.norm() <= rcut {
                        out.push(t);
                    }
                }
            }
        }
        // A number of supercell builders use the first block as the home
        // cell. Distance ordering makes that invariant explicit and also
        // lets screened real-space loops encounter the largest terms first.
        out.sort_by(|a, b| {
            a.norm2()
                .total_cmp(&b.norm2())
                .then_with(|| a.x.total_cmp(&b.x))
                .then_with(|| a.y.total_cmp(&b.y))
                .then_with(|| a.z.total_cmp(&b.z))
        });
        out
    }

    /// All reciprocal vectors with `|G| <= gcut`, excluding `G = 0`.
    pub fn reciprocal_within(&self, gcut: f64) -> Vec<Vec3> {
        let b = self.reciprocal();
        // Perpendicular width in reciprocal space along b_i.
        let mut nmax = [0i64; 3];
        for i in 0..3 {
            let width = 2.0 * std::f64::consts::PI / self.a[i].norm();
            nmax[i] = (gcut / width).ceil() as i64 + 1;
        }
        let mut out = Vec::new();
        for i in -nmax[0]..=nmax[0] {
            for j in -nmax[1]..=nmax[1] {
                for k in -nmax[2]..=nmax[2] {
                    if i == 0 && j == 0 && k == 0 {
                        continue;
                    }
                    let g = b[0] * (i as f64) + b[1] * (j as f64) + b[2] * (k as f64);
                    if g.norm() <= gcut {
                        out.push(g);
                    }
                }
            }
        }
        out
    }

    /// Supercell repeating this lattice `n` times along each vector.
    pub fn supercell(&self, n: [usize; 3]) -> Lattice {
        Lattice::new(
            self.a[0] * (n[0] as f64),
            self.a[1] * (n[1] as f64),
            self.a[2] * (n[2] as f64),
        )
    }
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `a_i . b_j = 2 pi delta_ij` is the definition; check it on a skewed
    /// cell, where a sign or a transpose slip actually shows up.
    #[test]
    fn reciprocal_lattice_is_dual_to_the_real_one() {
        let lat = Lattice::new(
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(1.3, 3.7, 0.0),
            Vec3::new(0.4, -0.9, 5.1),
        );
        let b = lat.reciprocal();
        let two_pi = 2.0 * std::f64::consts::PI;
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { two_pi } else { 0.0 };
                let got = lat.a[i].dot(b[j]);
                assert!(
                    (got - want).abs() < 1e-12,
                    "a{i}.b{j} = {got}, expected {want}"
                );
            }
        }
    }

    /// Fractional and Cartesian coordinates must round-trip exactly.
    #[test]
    fn fractional_round_trip() {
        let lat = Lattice::new(
            Vec3::new(3.1, 0.2, -0.4),
            Vec3::new(0.7, 4.4, 0.1),
            Vec3::new(-0.3, 0.5, 5.9),
        );
        for r in [
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-7.3, 0.4, 11.2),
            Vec3::zero(),
        ] {
            let back = lat.to_cartesian(lat.to_fractional(r));
            assert!((back - r).norm() < 1e-12, "round trip failed for {r:?}");
        }
    }

    /// A fixed +/-1 neighbourhood is not sufficient for an unreduced
    /// lattice: the shortest vector can differ by many basis translations.
    /// Compare the guaranteed search against a deliberately broad oracle.
    #[test]
    fn minimum_image_handles_strongly_skewed_unreduced_cells() {
        let lattice = Lattice::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(13.776_874_061, 0.231_375_795_7, 0.0),
            Vec3::new(0.0, 0.0, 8.0),
        );
        let fractional = [-0.079_428_419_2, -0.241_083_249_7, 0.0];
        let displacement = lattice.to_cartesian(fractional);
        let got = lattice.minimum_image(displacement);
        let mut oracle = displacement;
        let mut oracle2 = oracle.norm2();
        for i in -30..=30 {
            for j in -4..=4 {
                for k in -1..=1 {
                    let candidate = displacement - lattice.translation([i, j, k]);
                    if candidate.norm2() < oracle2 {
                        oracle = candidate;
                        oracle2 = candidate.norm2();
                    }
                }
            }
        }
        assert!(
            (got - oracle).norm() < 1e-12,
            "minimum image {got:?} differs from broad-search oracle {oracle:?}"
        );
        assert!(got.norm2() < 0.12);
    }

    /// The minimum image must be the true shortest translation image. A
    /// skewed cell is the case where rounding fractional coordinates alone
    /// gets it wrong, so that is what is tested.
    #[test]
    fn minimum_image_is_the_shortest_image() {
        let lat = Lattice::new(
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(3.6, 1.8, 0.0),
            Vec3::new(0.0, 0.0, 4.0),
        );
        for d in [
            Vec3::new(1.9, 0.9, 0.3),
            Vec3::new(3.5, 1.7, 1.9),
            Vec3::new(-2.2, 0.4, -1.8),
        ] {
            let img = lat.minimum_image(d);
            // Brute force over a generous window.
            let mut best = f64::INFINITY;
            for i in -3..=3 {
                for j in -3..=3 {
                    for k in -3..=3 {
                        let c = d + lat.translation([i, j, k]);
                        best = best.min(c.norm());
                    }
                }
            }
            assert!(
                (img.norm() - best).abs() < 1e-10,
                "minimum image {} vs brute force {best}",
                img.norm()
            );
        }
    }

    /// Translation enumeration must not miss a shell. Counting against a
    /// brute-force window catches the classic bug of bounding the indices
    /// cubically on a skewed cell.
    #[test]
    fn translation_enumeration_is_complete() {
        let lat = Lattice::new(
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(3.6, 1.8, 0.0),
            Vec3::new(0.5, 0.3, 4.0),
        );
        let rcut = 9.0;
        let got = lat.translations_within(rcut).len();
        let mut want = 0;
        for i in -12..=12 {
            for j in -12..=12 {
                for k in -12..=12 {
                    if lat.translation([i, j, k]).norm() <= rcut {
                        want += 1;
                    }
                }
            }
        }
        assert_eq!(got, want, "translation shell is incomplete");
        let translations = lat.translations_within(rcut);
        assert!(translations[0].norm2() < 1e-24, "home cell is not first");
        for t in &translations {
            assert!(
                translations.iter().any(|u| (*u + *t).norm() < 1e-12),
                "translation shell has no inversion partner for {t:?}"
            );
        }
    }

    /// Volume scales with the supercell, and the reciprocal cell shrinks to
    /// match.
    #[test]
    fn supercell_volume_and_reciprocal_scale() {
        let lat = Lattice::cubic(5.0);
        let sc = lat.supercell([2, 3, 1]);
        assert!((sc.volume() - 6.0 * lat.volume()).abs() < 1e-10);
        let (b, bs) = (lat.reciprocal(), sc.reciprocal());
        assert!((bs[0].norm() * 2.0 - b[0].norm()).abs() < 1e-12);
        assert!((bs[1].norm() * 3.0 - b[1].norm()).abs() < 1e-12);
    }
}
