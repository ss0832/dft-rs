// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Phase 1 gate: basis + integral engine against closed forms, literature
//! values, and structural identities (no external oracle needed here; the
//! frozen-PySCF matrix comparison is added with the oracle tooling).

use dft_rs::basis::{data, MolBasis};
use dft_rs::integrals::{
    dipole_matrices, eri_incore, eri_shell_quartet, kinetic_matrix, moment_matrices,
    nuclear_matrix, overlap_matrix, schwarz_matrix, CoulombKernel,
};
use dft_rs::math::Vec3;
use dft_rs::system::{Atom, Molecule};

fn h2_sto3g() -> (Molecule, MolBasis) {
    // Szabo & Ostrund's H2: R = 1.4 Bohr, STO-3G (ζ = 1.24 scaled H).
    let mol = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::zero(),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.0, 0.0, 1.4),
        },
    ]);
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    (mol, mb)
}

/// Literature gate (Szabo & Ostlund, *Modern Quantum Chemistry*, §3.5.2,
/// H2/STO-3G at R = 1.4 a0; tabulated to 4 decimals): overlap, kinetic,
/// per-nucleus attraction, and the unique two-electron integrals.
#[test]
fn h2_sto3g_literature_values() {
    let (mol, mb) = h2_sto3g();
    let s = overlap_matrix(&mb);
    assert!((s[(0, 0)] - 1.0).abs() < 1e-10, "S11 = {}", s[(0, 0)]);
    assert!((s[(0, 1)] - 0.6593).abs() < 1e-4, "S12 = {}", s[(0, 1)]);

    let t = kinetic_matrix(&mb);
    assert!((t[(0, 0)] - 0.7600).abs() < 1e-4, "T11 = {}", t[(0, 0)]);
    assert!((t[(0, 1)] - 0.2365).abs() < 1e-4, "T12 = {}", t[(0, 1)]);

    // Attraction to nucleus 1 only (charges = [1, 0]).
    let v1 = nuclear_matrix(&mb, &mol, Some(&[1.0, 0.0]));
    assert!(
        (v1[(0, 0)] - -1.2266).abs() < 1e-4,
        "V1_11 = {}",
        v1[(0, 0)]
    );
    assert!(
        (v1[(0, 1)] - -0.5974).abs() < 1e-4,
        "V1_12 = {}",
        v1[(0, 1)]
    );
    assert!(
        (v1[(1, 1)] - -0.6538).abs() < 1e-4,
        "V1_22 = {}",
        v1[(1, 1)]
    );

    // Both nuclei = sum of the single-nucleus matrices (linearity + symmetry).
    let v = nuclear_matrix(&mb, &mol, None);
    assert!((v[(0, 0)] - (v1[(0, 0)] + v1[(1, 1)])).abs() < 1e-10);

    // Unique ERIs (chemists' notation).
    let eri = eri_incore(&mb, CoulombKernel::Coulomb, 64).unwrap();
    assert!((eri.get(0, 0, 0, 0) - 0.7746).abs() < 1e-4, "(11|11)");
    assert!((eri.get(0, 0, 1, 1) - 0.5697).abs() < 1e-4, "(11|22)");
    assert!((eri.get(1, 0, 0, 0) - 0.4441).abs() < 1e-4, "(21|11)");
    assert!((eri.get(1, 0, 1, 0) - 0.2970).abs() < 1e-4, "(21|21)");
}

/// The nuclear-attraction matrix of a single H atom must match a direct
/// radial quadrature of ⟨φ|1/r|φ⟩ for the contracted s function — an
/// independent numerical route through none of the Boys/Hermite code.
#[test]
fn h_atom_nuclear_attraction_vs_quadrature() {
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: Vec3::zero(),
    }]);
    let set = data::load("sto-3g").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let v = nuclear_matrix(&mb, &mol, None);

    // φ(r) = Σ c_i (2a_i/π)^{3/4}... — MolBasis already holds the normalized
    // contraction; evaluate ⟨φ|1/r|φ⟩ = ∫ φ(r)² (1/r) 4π r² dr numerically.
    let shell = &mb.shells[0];
    let phi = |r: f64| -> f64 {
        shell
            .exps
            .iter()
            .zip(&shell.coefs)
            .map(|(&a, &c)| c * (-a * r * r).exp())
            .sum()
    };
    // Composite Simpson on [0, 30] Bohr, fine grid (integrand is smooth;
    // r² kills the 1/r singularity).
    let n = 300_000;
    let h = 30.0 / n as f64;
    let mut integral = 0.0;
    for k in 0..n {
        let r0 = k as f64 * h;
        let rm = r0 + 0.5 * h;
        let r1 = r0 + h;
        let f = |r: f64| {
            if r == 0.0 {
                0.0
            } else {
                phi(r) * phi(r) * 4.0 * std::f64::consts::PI * r
            }
        };
        integral += h / 6.0 * (f(r0) + 4.0 * f(rm) + f(r1));
    }
    assert!(
        (v[(0, 0)] + integral).abs() < 1e-9,
        "V11 = {} vs quadrature {}",
        v[(0, 0)],
        -integral
    );
}

/// Same-center d shell: after the spherical transform the overlap block must
/// be the 5×5 identity — locks normalization + solid harmonics + assembly.
#[test]
fn single_shell_orthonormality_def2svp_carbon() {
    let mol = Molecule::new(vec![Atom {
        z: 6,
        position: Vec3::new(0.3, -0.4, 0.9),
    }]);
    let set = data::load("def2-svp").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();
    let s = overlap_matrix(&mb);
    for i in 0..mb.nao_sph {
        for j in 0..mb.nao_sph {
            if i == j {
                assert!(
                    (s[(i, i)] - 1.0).abs() < 1e-10,
                    "S[{i},{i}] = {}",
                    s[(i, i)]
                );
            } else {
                assert!(s[(i, j)].abs() < 0.999, "off-diagonal ≥ 1: S[{i},{j}]");
            }
        }
    }
    // The d-block specifically must be exactly orthonormal (single shell).
    let d_off = mb
        .shells
        .iter()
        .find(|sh| sh.l == 2)
        .map(|sh| sh.sph_off)
        .unwrap();
    for i in 0..5 {
        for j in 0..5 {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (s[(d_off + i, d_off + j)] - want).abs() < 1e-10,
                "d-block S[{i},{j}]"
            );
        }
    }
}

/// S and T are invariant under rigid translation of the whole molecule.
#[test]
fn translation_invariance_of_s_and_t() {
    let set = data::load("def2-svp").unwrap();
    let make = |shift: Vec3| {
        let mol = Molecule::new(vec![
            Atom {
                z: 8,
                position: Vec3::new(0.0, 0.0, 0.0) + shift,
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 1.8) + shift,
            },
        ]);
        let mb = MolBasis::build(&mol, &set).unwrap();
        (overlap_matrix(&mb), kinetic_matrix(&mb))
    };
    let (s0, t0) = make(Vec3::zero());
    let (s1, t1) = make(Vec3::new(3.7, -1.2, 0.4));
    assert!(s0.max_abs_diff(&s1) < 1e-11);
    assert!(t0.max_abs_diff(&t1) < 1e-11);
}

/// ERI permutational symmetry on a shell quartet with p functions and four
/// distinct centers: (ab|cd) = (ba|cd) = (ab|dc) = (cd|ab).
#[test]
fn eri_permutation_symmetry() {
    let set = data::load("sto-3g").unwrap();
    let mol = Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.0, 2.3),
        },
        Atom {
            z: 8,
            position: Vec3::new(2.1, 0.0, 0.0),
        },
        Atom {
            z: 8,
            position: Vec3::new(0.0, 1.9, 0.7),
        },
    ]);
    let mb = MolBasis::build(&mol, &set).unwrap();
    // O in STO-3G: shells [S, S, P] per atom → pick the P shell (index 2) of
    // each atom: shells 2, 5, 8, 11.
    let (a, b, c, d) = (2usize, 5, 8, 11);
    let abcd = eri_shell_quartet(&mb, a, b, c, d, CoulombKernel::Coulomb);
    let bacd = eri_shell_quartet(&mb, b, a, c, d, CoulombKernel::Coulomb);
    let abdc = eri_shell_quartet(&mb, a, b, d, c, CoulombKernel::Coulomb);
    let cdab = eri_shell_quartet(&mb, c, d, a, b, CoulombKernel::Coulomb);
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    let v = abcd.get(i, j, k, l);
                    assert!((v - bacd.get(j, i, k, l)).abs() < 1e-12);
                    assert!((v - abdc.get(i, j, l, k)).abs() < 1e-12);
                    assert!((v - cdab.get(k, l, i, j)).abs() < 1e-12);
                }
            }
        }
    }
}

/// Range separation: erf + erfc kernels must sum to the Coulomb ERI, and the
/// Schwarz bound must hold for every computed quartet element.
#[test]
fn kernel_split_and_schwarz_bound() {
    let (_, mb) = h2_sto3g();
    let omega = 0.33;
    let full = eri_incore(&mb, CoulombKernel::Coulomb, 64).unwrap();
    let lr = eri_incore(&mb, CoulombKernel::Erf { omega }, 64).unwrap();
    let sr = eri_incore(&mb, CoulombKernel::Erfc { omega }, 64).unwrap();
    let n = mb.nao_sph;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for l in 0..n {
                    let f = full.get(i, j, k, l);
                    assert!((lr.get(i, j, k, l) + sr.get(i, j, k, l) - f).abs() < 1e-12);
                }
            }
        }
    }
    let q = schwarz_matrix(&mb, CoulombKernel::Coulomb);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for l in 0..n {
                    assert!(
                        full.get(i, j, k, l).abs() <= q[(i, j)] * q[(k, l)] + 1e-12,
                        "Schwarz bound violated at ({i}{j}|{k}{l})"
                    );
                }
            }
        }
    }
}

/// Moment integrals: closed forms on a single primitive s shell —
/// ⟨s|(x−Ax)|s⟩ = 0, ⟨s|(x−Cx)|s⟩ = (Ax−Cx)·S, ⟨s|(x−Ax)²|s⟩ = 1/(4a).
#[test]
fn moment_closed_forms() {
    use dft_rs::basis::{BasisSet, ElementBasis, Shell};
    let a_exp = 0.8;
    let center = Vec3::new(0.7, -0.3, 1.1);
    let mut bs = BasisSet::new("prim");
    bs.insert(ElementBasis {
        z: 1,
        shells: vec![Shell {
            l: 0,
            exps: vec![a_exp],
            coefs: vec![1.0],
        }],
    });
    let mol = Molecule::new(vec![Atom {
        z: 1,
        position: center,
    }]);
    let mb = MolBasis::build(&mol, &bs).unwrap();

    let dip_at_center = dipole_matrices(&mb, center);
    for m in &dip_at_center {
        assert!(m[(0, 0)].abs() < 1e-14);
    }
    let origin = Vec3::zero();
    let dip = dipole_matrices(&mb, origin);
    assert!((dip[0][(0, 0)] - center.x).abs() < 1e-12);
    assert!((dip[1][(0, 0)] - center.y).abs() < 1e-12);
    assert!((dip[2][(0, 0)] - center.z).abs() < 1e-12);

    let second = moment_matrices(&mb, center, 2);
    for ((e, f, g), m) in second {
        let want = if e == 2 || f == 2 || g == 2 {
            1.0 / (4.0 * a_exp)
        } else {
            0.0 // mixed ⟨xy⟩ vanish by symmetry
        };
        assert!(
            (m[(0, 0)] - want).abs() < 1e-13,
            "⟨({e},{f},{g})⟩ = {} vs {want}",
            m[(0, 0)]
        );
    }
}

/// The dense in-core tensor must agree with individually computed quartets
/// (scatter/symmetry bookkeeping check) on a mixed s/p/d system.
#[test]
fn incore_matches_direct_quartets() {
    let set = data::load("def2-svp").unwrap();
    let mol = Molecule::new(vec![
        Atom {
            z: 1,
            position: Vec3::zero(),
        },
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.0, 1.81),
        },
    ]);
    let mb = MolBasis::build(&mol, &set).unwrap();
    let eri = eri_incore(&mb, CoulombKernel::Coulomb, 256).unwrap();
    // Spot-check a handful of shell quartets, including d shells.
    let ns = mb.shells.len();
    let picks = [
        (0, 0, 0, 0),
        (0, 1, 2, 3 % ns),
        (ns - 1, ns - 1, ns - 1, ns - 1),
        (1, ns - 1, 0, ns - 2),
    ];
    for &(a, b, c, d) in &picks {
        let quartet = eri_shell_quartet(&mb, a, b, c, d, CoulombKernel::Coulomb);
        let (oa, ob, oc, od) = (
            mb.shells[a].sph_off,
            mb.shells[b].sph_off,
            mb.shells[c].sph_off,
            mb.shells[d].sph_off,
        );
        for i in 0..quartet.na {
            for j in 0..quartet.nb {
                for k in 0..quartet.nc {
                    for l in 0..quartet.nd {
                        assert!(
                            (quartet.get(i, j, k, l) - eri.get(oa + i, ob + j, oc + k, od + l))
                                .abs()
                                < 1e-12
                        );
                    }
                }
            }
        }
    }
}

/// The eightfold permutational symmetry of the two-electron integrals holds
/// for every angular momentum and every screened kernel, not just the case
/// the original test covered.
///
/// Classification: **exact**. `(ab|cd)` is invariant under `a<->b`, `c<->d`
/// and `(ab)<->(cd)` because the integrand is, so any deviation is an
/// implementation defect rather than a quadrature or basis effect.
///
/// The existing `eri_permutation_symmetry` covers p shells, four distinct
/// centres and the bare Coulomb operator, and checks three of the four
/// relations. This widens it on the three axes where a bug could hide:
///
/// - **d and f shells.** The Cartesian-to-spherical transform is applied
///   per index, so a transposed or mis-ordered transform block shows up as
///   an asymmetry that s and p shells (whose transforms are the identity and
///   a permutation) cannot expose.
/// - **The erf and erfc kernels.** Range separation enters only through the
///   Boys-function kernel, which is shared by all four indices; a kernel
///   applied with the wrong index pairing breaks bra-ket exchange while
///   leaving the bra and ket swaps intact.
/// - **The fourth relation** `(ab|cd) == (dc|ba)`, which the original omits
///   and which is the composition of the other three.
#[test]
fn eri_permutation_symmetry_holds_for_high_l_and_screened_kernels() {
    // Four distinct centres so no accidental degeneracy can mask an error,
    // and a basis carrying d (and f, on the heavier centre) functions.
    let mol = Molecule::new(vec![
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        Atom {
            z: 6,
            position: Vec3::new(1.9, 0.3, -0.4),
        },
        Atom {
            z: 1,
            position: Vec3::new(-0.7, 1.6, 0.5),
        },
        Atom {
            z: 1,
            position: Vec3::new(0.4, -1.3, 1.7),
        },
    ]);
    let set = data::load("cc-pvtz").unwrap();
    let mb = MolBasis::build(&mol, &set).unwrap();

    // Pick one shell per angular momentum actually present, on as many
    // different centres as possible.
    let mut by_l: Vec<(u8, usize)> = Vec::new();
    for (i, sh) in mb.shells.iter().enumerate() {
        if !by_l.iter().any(|(l, _)| *l == sh.l) {
            by_l.push((sh.l, i));
        }
    }
    by_l.sort_by_key(|(l, _)| *l);
    assert!(
        by_l.iter().any(|(l, _)| *l >= 2),
        "the chosen basis carries no d shell, so this test would not exercise \
         a non-trivial cart->sph transform"
    );

    let kernels = [
        CoulombKernel::Coulomb,
        CoulombKernel::Erf { omega: 0.33 },
        CoulombKernel::Erfc { omega: 0.33 },
    ];

    for kernel in kernels {
        for &(la, a) in &by_l {
            for &(lb, b) in &by_l {
                for &(lc, c) in &by_l {
                    for &(ld, d) in &by_l {
                        let base = eri_shell_quartet(&mb, a, b, c, d, kernel);
                        let ba = eri_shell_quartet(&mb, b, a, c, d, kernel);
                        let dc = eri_shell_quartet(&mb, a, b, d, c, kernel);
                        let cd = eri_shell_quartet(&mb, c, d, a, b, kernel);
                        let rev = eri_shell_quartet(&mb, d, c, b, a, kernel);
                        let (na, nb, nc, nd) = (base.na, base.nb, base.nc, base.nd);
                        for i in 0..na {
                            for j in 0..nb {
                                for k in 0..nc {
                                    for l in 0..nd {
                                        let v = base.get(i, j, k, l);
                                        let tol = 1e-11 * (1.0 + v.abs());
                                        let cases = [
                                            ("(ba|cd)", ba.get(j, i, k, l)),
                                            ("(ab|dc)", dc.get(i, j, l, k)),
                                            ("(cd|ab)", cd.get(k, l, i, j)),
                                            ("(dc|ba)", rev.get(l, k, j, i)),
                                        ];
                                        for (what, got) in cases {
                                            assert!(
                                                (v - got).abs() < tol,
                                                "{kernel:?} l=({la},{lb},{lc},{ld}) \
                                                 element ({i},{j},{k},{l}): {what} = {got} \
                                                 but (ab|cd) = {v}"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
