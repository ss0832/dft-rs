// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Lebedev-Laikov angular quadrature: octahedrally invariant rules exact for
//! spherical polynomials up to degree 131, expanded from the published orbit
//! generators in [`super::lebedev_data`]. Every rule is checked against the
//! closed-form monomial integrals in the module tests.

use super::lebedev_data::RULES;
use crate::math::Vec3;
use std::sync::OnceLock;

/// Degrees of the available rules, increasing.
pub(crate) fn degrees() -> impl Iterator<Item = usize> {
    RULES.iter().map(|(degree, _, _)| *degree)
}

/// The rule of lowest degree that is exact to at least `degree`, as unit
/// directions and weights summing to 4π, or `None` above degree 131.
pub(crate) fn rule(degree: usize) -> Option<&'static [(Vec3, f64)]> {
    static EXPANDED: OnceLock<Vec<Vec<(Vec3, f64)>>> = OnceLock::new();
    let index = RULES.iter().position(|(d, _, _)| *d >= degree)?;
    let rules = EXPANDED.get_or_init(|| {
        RULES
            .iter()
            .map(|(_, points, orbits)| {
                let mut out = Vec::with_capacity(*points);
                for &(code, a, b, v) in orbits.iter() {
                    expand(code, a, b, v * 4.0 * std::f64::consts::PI, &mut out);
                }
                debug_assert_eq!(out.len(), *points);
                out
            })
            .collect()
    });
    Some(&rules[index])
}

/// Points of one octahedral orbit with weight `w`:
/// code 0, the 6 vertices (±1, 0, 0); code 1, the 12 edge centres
/// (0, ±√½, ±√½); code 2, the 8 face centres (±√⅓, ±√⅓, ±√⅓); code 3, the 24
/// points (±a, ±a, ±b) with b = √(1 − 2a²); code 4, the 24 points
/// (±a, ±b, 0) with b = √(1 − a²); code 5, the 48 points (±a, ±b, ±c) with
/// c = √(1 − a² − b²); each with all coordinate permutations.
fn expand(code: u8, a: f64, b: f64, w: f64, out: &mut Vec<(Vec3, f64)>) {
    let base = match code {
        0 => [1.0, 0.0, 0.0],
        1 => [0.0, 0.5_f64.sqrt(), 0.5_f64.sqrt()],
        2 => [(1.0_f64 / 3.0).sqrt(); 3],
        3 => [a, a, (1.0 - 2.0 * a * a).sqrt()],
        4 => [a, (1.0 - a * a).sqrt(), 0.0],
        _ => [a, b, (1.0 - a * a - b * b).sqrt()],
    };
    const PERMUTATIONS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let start = out.len();
    for perm in PERMUTATIONS {
        let p = [base[perm[0]], base[perm[1]], base[perm[2]]];
        for signs in 0..8 {
            let mut q = p;
            for (axis, value) in q.iter_mut().enumerate() {
                if (signs >> axis) & 1 == 1 {
                    *value = -*value;
                }
            }
            // Distinct points only: a zero coordinate has one sign (−0 equals
            // +0) and equal coordinates give repeated permutations.
            let duplicate = out[start..]
                .iter()
                .any(|(d, _)| d.x == q[0] && d.y == q[1] && d.z == q[2]);
            if !duplicate {
                out.push((Vec3::new(q[0], q[1], q[2]), w));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::double_factorial;

    /// Every rule integrates every monomial x^a y^b z^c with a + b + c up
    /// to its degree: ∫ dΩ = 4π (a−1)!!(b−1)!!(c−1)!!/(a+b+c+1)!! for even
    /// exponents and zero otherwise. The point counts are the published
    /// ones.
    #[test]
    fn lebedev_rules_are_exact_to_their_degree() {
        let four_pi = 4.0 * std::f64::consts::PI;
        for (degree, points, _) in RULES.iter() {
            let rule = rule(*degree).unwrap();
            assert_eq!(rule.len(), *points, "degree {degree}");
            for (d, _) in rule {
                assert!((d.norm() - 1.0).abs() < 1e-14);
            }
            let mut worst = 0.0_f64;
            for a in 0..=*degree {
                for b in 0..=(*degree - a) {
                    // Total degrees `degree` and `degree - 1` suffice: on the
                    // sphere a monomial of degree n - 2 equals the sum of
                    // three of degree n (multiply by x² + y² + z²).
                    let c_max = *degree - a - b;
                    for c in [c_max, c_max.saturating_sub(1)] {
                        let got: f64 = rule
                            .iter()
                            .map(|(d, w)| {
                                w * d.x.powi(a as i32) * d.y.powi(b as i32) * d.z.powi(c as i32)
                            })
                            .sum();
                        let want = if a % 2 == 0 && b % 2 == 0 && c % 2 == 0 {
                            four_pi
                                * double_factorial(a as i32 - 1)
                                * double_factorial(b as i32 - 1)
                                * double_factorial(c as i32 - 1)
                                / double_factorial((a + b + c) as i32 + 1)
                        } else {
                            0.0
                        };
                        worst = worst.max((got - want).abs());
                    }
                }
            }
            assert!(
                worst < 1e-12,
                "degree {degree}: worst monomial error {worst:e}"
            );
        }
    }

    #[test]
    fn lebedev_rule_rounds_up_to_the_next_degree() {
        assert_eq!(rule(26).unwrap().len(), 266);
        assert_eq!(rule(29).unwrap().len(), 302);
        assert!(rule(133).is_none());
    }
}
