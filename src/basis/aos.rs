// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Real solid harmonics: the Cartesian → spherical AO transformation with a
//! fixed canonical component order.
//!
//! Ordering conventions (frozen; see docs/THEORY.md):
//! - Cartesian components: x-major lexicographic (`basis::cart_components`).
//! - Spherical components: m = 0, +1, −1, +2, −2, …, +l, −l for every l.
//!   (The Molden writer maps p shells to its x, y, z expectation; the PySCF
//!   oracle generator permutes reference data into this order.)
//!
//! Construction: the real-solid-harmonic Cartesian expansion follows the
//! standard closed form (Helgaker/Jørgensen/Olsen, *Molecular
//! Electronic-Structure Theory*, eq. 6.4.47). Columns are then normalized
//! **exactly** against the analytic one-center overlap metric of the
//! radially-normalized Cartesian components, so the spherical AOs of a
//! normalized shell are orthonormal by construction — the unit tests assert
//! both orthonormality and fixed sign conventions.

use super::{cart_components, double_factorial, n_cart, n_sph};
use crate::linalg::Matrix;
use std::sync::OnceLock;

/// Highest angular momentum with a precomputed transform (i shells).
pub const MAX_L: u8 = 6;

/// The `ncart × nsph` coefficient matrix `C` for shell `l`:
/// `φ_sph = Σ_cart C[cart][sph] · φ_cart`, with all Cartesian components
/// carrying the shared `(l,0,0)` radial normalization.
pub fn cart_to_sph(l: u8) -> &'static Matrix {
    static CACHE: OnceLock<Vec<Matrix>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| (0..=MAX_L).map(build_cart_to_sph).collect());
    &cache[l as usize]
}

/// Angular part of the one-center overlap between two Cartesian components
/// of the same shell (shared radial normalization, `(l,0,0)` self-overlap 1):
/// `⟨a|b⟩ = (ax+bx−1)!!(ay+by−1)!!(az+bz−1)!! / (2l−1)!!` when all component
/// sums are even, else 0.
fn cart_metric(l: u8) -> Matrix {
    let comps = cart_components(l);
    let n = comps.len();
    let norm = double_factorial(2 * l as i32 - 1);
    let mut m = Matrix::zeros(n, n);
    for (i, &(ax, ay, az)) in comps.iter().enumerate() {
        for (j, &(bx, by, bz)) in comps.iter().enumerate() {
            if (ax + bx) % 2 == 0 && (ay + by) % 2 == 0 && (az + bz) % 2 == 0 {
                m[(i, j)] = double_factorial(ax + bx - 1)
                    * double_factorial(ay + by - 1)
                    * double_factorial(az + bz - 1)
                    / norm;
            }
        }
    }
    m
}

fn factorial(n: i32) -> f64 {
    (1..=n).map(|k| k as f64).product()
}

fn binom(n: i32, k: i32) -> f64 {
    if k < 0 || k > n {
        return 0.0;
    }
    factorial(n) / (factorial(k) * factorial(n - k))
}

/// Raw (unnormalized-column) Cartesian expansion coefficient of the real
/// solid harmonic (l, m) for the monomial x^lx y^ly z^lz.
pub(crate) fn solid_harmonic_coef(l: i32, m: i32, lx: i32, ly: i32, lz: i32) -> f64 {
    debug_assert_eq!(lx + ly + lz, l, "monomial degree must equal l");
    let ma = m.abs();
    if lz > l - ma || (l - ma - lz) % 2 != 0 {
        return 0.0;
    }
    let t = (l - ma - lz) / 2; // z exponent fixes t
                               // k = 2v: even for the cosine (m ≥ 0) form, odd for sine (m < 0).
    let k0 = if m < 0 { 1 } else { 0 };
    let mut sum = 0.0;
    let mut k = k0;
    while k <= ma {
        // x exponent: 2t + ma − 2u − k = lx → u = (2t + ma − k − lx)/2.
        let two_u = 2 * t + ma - k - lx;
        if two_u >= 0 && two_u % 2 == 0 {
            let u = two_u / 2;
            if u <= t {
                let vfloor = (k - k0) / 2;
                let sign = if (t + vfloor) % 2 == 0 { 1.0 } else { -1.0 };
                sum += sign
                    * 0.25_f64.powi(t)
                    * binom(l, t)
                    * binom(l - t, ma + t)
                    * binom(t, u)
                    * binom(ma, k);
            }
        }
        k += 2;
    }
    sum
}

fn build_cart_to_sph(l: u8) -> Matrix {
    let comps = cart_components(l);
    let (nc, ns) = (n_cart(l), n_sph(l));
    let li = l as i32;
    let mut c = Matrix::zeros(nc, ns);
    // Canonical m order: 0, +1, −1, +2, −2, …
    let m_order: Vec<i32> = std::iter::once(0)
        .chain((1..=li).flat_map(|m| [m, -m]))
        .collect();
    for (col, &m) in m_order.iter().enumerate() {
        for (row, &(lx, ly, lz)) in comps.iter().enumerate() {
            c[(row, col)] = solid_harmonic_coef(li, m, lx, ly, lz);
        }
    }
    // Exact column normalization against the one-center metric. Distinct-m
    // harmonics are orthogonal under the isotropic metric, so scaling
    // columns is exact (orthogonality is asserted in tests).
    let metric = cart_metric(l);
    let g = c.transpose_matmul(&metric.matmul(&c));
    for col in 0..ns {
        let norm = g[(col, col)].sqrt();
        assert!(norm > 0.0, "null solid-harmonic column l={l} col={col}");
        for row in 0..nc {
            c[(row, col)] /= norm;
        }
    }
    c
}

/// Transform a Cartesian shell-pair block `B` (`ncart_a × ncart_b`) into the
/// spherical basis: `C_aᵀ · B · C_b`.
pub fn sph_block(la: u8, lb: u8, cart_block: &Matrix) -> Matrix {
    let ca = cart_to_sph(la);
    let cb = cart_to_sph(lb);
    ca.transpose_matmul_seq(&cart_block.matmul_seq(cb))
}

/// Kronecker-product pair transform `W = C_la ⊗ C_lb`
/// (`(ncart_a·ncart_b) × (nsph_a·nsph_b)`), used to convert ERI shell-quartet
/// blocks two indices at a time. Lazily built and cached per (la, lb).
pub fn cart_to_sph_pair(la: u8, lb: u8) -> &'static Matrix {
    static CACHE: OnceLock<Vec<Vec<OnceLock<Matrix>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| {
        (0..=MAX_L)
            .map(|_| (0..=MAX_L).map(|_| OnceLock::new()).collect())
            .collect()
    });
    cache[la as usize][lb as usize].get_or_init(|| {
        let ca = cart_to_sph(la);
        let cb = cart_to_sph(lb);
        let (nca, nsa) = (ca.rows, ca.cols);
        let (ncb, nsb) = (cb.rows, cb.cols);
        let mut w = Matrix::zeros(nca * ncb, nsa * nsb);
        for ia in 0..nca {
            for ib in 0..ncb {
                for ja in 0..nsa {
                    for jb in 0..nsb {
                        w[(ia * ncb + ib, ja * nsb + jb)] = ca[(ia, ja)] * cb[(ib, jb)];
                    }
                }
            }
        }
        w
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The transform must render every shell orthonormal under the exact
    /// one-center metric — this validates both the solid-harmonic
    /// coefficients and the shared-radial-normalization convention.
    #[test]
    fn spherical_components_are_orthonormal() {
        for l in 0..=MAX_L {
            let c = cart_to_sph(l);
            let g = c.transpose_matmul(&cart_metric(l).matmul(c));
            let id = Matrix::identity(n_sph(l));
            assert!(
                g.max_abs_diff(&id) < 1e-12,
                "l={l}: Cᵀ M C deviates from identity by {}",
                g.max_abs_diff(&id)
            );
        }
    }

    /// s and p shells are permutations: s → 1; p order (m=0,+1,−1) = (z,x,y).
    #[test]
    fn s_and_p_are_permutations() {
        let s = cart_to_sph(0);
        assert!((s[(0, 0)] - 1.0).abs() < 1e-14);
        let p = cart_to_sph(1);
        // Cartesian rows: x, y, z. Columns: m=0(z), m=+1(x), m=−1(y).
        let expect = [
            [0.0, 1.0, 0.0], // x row
            [0.0, 0.0, 1.0], // y row
            [1.0, 0.0, 0.0], // z row
        ];
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (p[(i, j)] - expect[i][j]).abs() < 1e-14,
                    "p transform ({i},{j})"
                );
            }
        }
    }

    /// d-shell sign conventions: d0 ∝ 2z²−x²−y² (positive z² coefficient),
    /// d+2 ∝ x²−y², d−2 ∝ xy (positive), d+1 ∝ xz, d−1 ∝ yz.
    #[test]
    fn d_shell_signs_and_structure() {
        let d = cart_to_sph(2);
        // Cartesian order: xx, xy, xz, yy, yz, zz. Columns: 0,+1,−1,+2,−2.
        let (xx, xy, xz, yy, yz, zz) = (0, 1, 2, 3, 4, 5);
        // d0: zz positive, xx = yy negative, off monomials zero.
        assert!(d[(zz, 0)] > 0.0);
        assert!(d[(xx, 0)] < 0.0);
        assert!((d[(xx, 0)] - d[(yy, 0)]).abs() < 1e-14);
        assert!(d[(xy, 0)].abs() < 1e-14);
        // d+1 = xz only; d−1 = yz only (positive).
        assert!(d[(xz, 1)] > 0.0);
        assert!(d[(yz, 2)] > 0.0);
        assert!(d[(xx, 1)].abs() < 1e-14 && d[(yy, 2)].abs() < 1e-14);
        // d+2: xx positive, yy negative, equal magnitude; d−2: xy positive.
        assert!(d[(xx, 3)] > 0.0 && d[(yy, 3)] < 0.0);
        assert!((d[(xx, 3)] + d[(yy, 3)]).abs() < 1e-14);
        assert!(d[(xy, 4)] > 0.0);
        // Known normalized values: d−2 column is √3·(xy metric)⁻... the xy
        // coefficient must be √3 under the shared-radial convention
        // (⟨xy|xy⟩ = 1/3 ⇒ normalized harmonic = √3·xy).
        assert!((d[(xy, 4)] - 3.0_f64.sqrt()).abs() < 1e-12);
    }

    /// f-shell spot value: the f−3 harmonic ∝ (3x² − y²)y; under the metric
    /// its normalization is fixed — verify a known ratio instead of an
    /// absolute table: coef(y³)/coef(x²y) = −1/3.
    #[test]
    fn f_shell_known_ratio() {
        let f = cart_to_sph(3);
        let comps = cart_components(3);
        let row = |t: (i32, i32, i32)| comps.iter().position(|&c| c == t).unwrap();
        // Column order: 0,+1,−1,+2,−2,+3,−3 → f−3 is column 6.
        let x2y = f[(row((2, 1, 0)), 6)];
        let y3 = f[(row((0, 3, 0)), 6)];
        assert!(x2y > 0.0);
        assert!((y3 / x2y + 1.0 / 3.0).abs() < 1e-12);
    }
}
