// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! The local body frame of a pseudo-substituent site, and its Jacobian.
//!
//! # The frame
//!
//! A site `G` attached to a real atom `X` needs a body frame to express its
//! (future, not-yet-fitted) anisotropic operator in:
//!
//! ```text
//! ê_z = (R_G − R_X) / |R_G − R_X|
//! R_ref = the bonded reference candidate chosen by priority: highest Z, then lowest index
//! ê_x = normalize( (R_ref − R_X) − ê_z ((R_ref − R_X)·ê_z) )
//! ê_y = ê_z × ê_x
//! ```
//!
//! `U = [ê_x ê_y ê_z]`, **columns**, matching `math::Mat3`'s own
//! column-of-`Vec3` layout (`Mat3::from_columns`).
//!
//! The physical API, [`build_local_frame_with_connectivity`], takes X and the
//! bonded reference candidates explicitly. The backward-compatible
//! [`build_local_frame`] discovers X as the nearest other atom to G, but only
//! proceeds for a non-cylindrical site when at most one possible reference
//! atom remains; it refuses a larger molecule because `Molecule` contains no
//! bond graph and an all-atom priority search would let a distant high-Z atom
//! steal the frame. A cylindrical site needs only its unique X/G axis.
//!
//! # Refusals
//!
//! Conditions that make the frame impossible or ill-conditioned are refused
//! **by name** — `DftError::Unsupported`,
//! never a silently-picked frame — in the tone of `pcm::PcmSolver::require_k_is_s`
//! (`src/pcm.rs`): the caller asked for something reasonable, this code
//! cannot deliver it *for this geometry*, and the message says exactly why.
//!
//! 1. **X not unique.** Two or more atoms sit within [`EQUIDISTANT_TOL_BOHR`]
//!    of the shortest G-to-other-atom distance.
//! 2. **No reference atom.** No bonded reference candidate was supplied, *and*
//!    `symmetry != SiteSymmetry::Cylindrical`. When the symmetry *is*
//!    Cylindrical this is not refused: [`build_local_frame`] returns
//!    `Ok(None)` instead, because a cylindrically-symmetric operator's
//!    value cannot depend on rotation about `ê_z`, so `ê_x`/`ê_y` are a
//!    gauge freedom the parameters are constrained to ignore, and `ê_z`
//!    alone (which needs no reference atom) is all a Cylindrical site ever
//!    reads.
//! 3. **Near-linear.** `(R_ref − R_X)` is within [`NEAR_LINEAR_TOL_RAD`] of
//!    parallel (or antiparallel) to `ê_z`; the projection step that builds
//!    `ê_x` divides by a quantity that vanishes at exact linearity, and
//!    the tolerance keeps that division well away from that singularity
//!    (see the constant's own doc comment for the numeric argument).
//!
//! Malformed connectivity (out-of-range, repeated, G/X included as a
//! reference) is caller input and returns `DftError::InvalidInput`.
//!
//! # The Jacobian (`frame_jacobian`)
//!
//! `U` depends on three atoms (`R_X`, `R_G`, `R_ref` — 9 Cartesian
//! coordinates), so a gradient through the frame distributes over three
//! atoms at once. [`build_frame_generic`] is written once, generically over
//! `dual::Scalar`, and used for **both** the plain `f64` value (inside
//! [`build_local_frame`]) and, seeded with `DualN<f64, 9>`, the analytic
//! Jacobian in [`frame_jacobian`] — the same arithmetic (normalization,
//! projection, cross product) runs both times, so a value/derivative
//! mismatch would require the *same* bug to appear in both call sites
//! simultaneously, rather than two independently-written formulas quietly
//! disagreeing. The finite-difference gate in the tests below is still
//! mandatory and is what actually establishes correctness — the shared
//! code path only rules out one class of bug (transcription drift between
//! a value formula and a hand-derived gradient formula), not all of them.

use crate::dual::{DualN, Scalar};
use crate::error::{DftError, Result};
use crate::linalg::Matrix;
use crate::math::{Mat3, Vec3};
use crate::system::Molecule;

use super::SiteSymmetry;

/// Two candidate bond partners closer to G than this to each other (in
/// their distance *to* G) are treated as tied rather than one being picked
/// arbitrarily. SCF/optimizer geometries carry positional noise of order
/// `1e-6`-`1e-5` bohr; the smallest chemically meaningful difference
/// between two distinct bond lengths is at least hundredths of a bohr
/// (`1e-2`). `1e-4` bohr sits two orders of magnitude above the noise
/// floor and two below the smallest real difference, so it never
/// mistakes noise for a tie and never mistakes two genuinely different
/// bond partners for one.
pub const EQUIDISTANT_TOL_BOHR: f64 = 1e-4;

/// `(R_ref − R_X)` is refused as too close to parallel/antiparallel to
/// `ê_z` once the angle between them drops below this. The `ê_x`
/// projection step divides its normalization by a vector whose magnitude
/// is `sin(angle)·|R_ref − R_X|`; at `5°`, `sin(5°) ≈ 0.0872`, so
/// normalization noise is amplified by at most `~11.5x`. That keeps the
/// central-difference Jacobian gate (`h = 1e-5` bohr, tolerance `1e-8`)
/// well-conditioned all the way up to the refusal boundary — the gate's
/// own "near-but-not-past" geometry sits at `7°`, inside the allowed
/// region, and still matches finite differences to the full `1e-8`.
pub const NEAR_LINEAR_TOL_RAD: f64 = 0.08726646259971647; // 5 degrees

#[inline]
fn sub3<S: Scalar>(a: [S; 3], b: [S; 3]) -> [S; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn dot3<S: Scalar>(a: [S; 3], b: [S; 3]) -> S {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn cross3<S: Scalar>(a: [S; 3], b: [S; 3]) -> [S; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
fn scale3<S: Scalar>(a: [S; 3], s: S) -> [S; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
fn normalize3<S: Scalar>(a: [S; 3]) -> [S; 3] {
    let n = dot3(a, a).sqrt();
    scale3(a, n.recip())
}

/// The frame construction, generic over the scalar type. See the module
/// docs for why this single generic function backs both the numeric value
/// ([`build_local_frame`], `S = f64`) and the analytic Jacobian
/// ([`frame_jacobian`], `S = DualN<f64, 9>`).
///
/// Callers are responsible for having already ruled out the degenerate
/// cases [`build_local_frame`] refuses (X coincident with G, near-linear
/// `R_ref`): this function does not itself guard against a division by
/// (near-)zero.
fn build_frame_generic<S: Scalar>(r_x: [S; 3], r_g: [S; 3], r_ref: [S; 3]) -> [[S; 3]; 3] {
    let ez = normalize3(sub3(r_g, r_x));
    let d_ref = sub3(r_ref, r_x);
    let proj = dot3(d_ref, ez);
    let ortho = sub3(d_ref, scale3(ez, proj));
    let ex = normalize3(ortho);
    let ey = cross3(ez, ex);
    [ex, ey, ez]
}

/// A pseudo-substituent site's local body frame. `u`'s columns are
/// `[ê_x, ê_y, ê_z]`, in that order.
#[derive(Clone, Copy, Debug)]
pub struct LocalFrame {
    /// The real atom X the site is attached to (`ê_z` runs from X to G).
    pub x_atom: usize,
    /// The reference atom used to fix `ê_x` (highest-Z, then lowest-index
    /// atom other than X and G).
    pub ref_atom: usize,
    /// `[ê_x ê_y ê_z]`, columns.
    pub u: Mat3,
}

/// Explicit topology needed to define a pseudo-substituent body frame.
///
/// The caller attests that `x_atom` is the atom bonded to G and every entry
/// in `reference_candidates` is another atom bonded to X. This module does
/// not infer bonds from empirical radii or a distance cutoff. Candidates are
/// validated as indices and then ranked by highest atomic number, breaking a
/// same-Z tie by lowest atom index.
#[derive(Clone, Copy, Debug)]
pub struct FrameConnectivity<'a> {
    pub x_atom: usize,
    pub reference_candidates: &'a [usize],
}

/// Build the local frame for the pseudo-substituent site sitting on atom
/// `g_atom`. See the module docs for the refusal conditions and the
/// `Ok(None)` sentinel.
///
/// If a non-cylindrical site has more than one possible reference atom after
/// the unique-nearest X search, this compatibility API refuses rather than
/// ranking all atoms. Physical callers should use
/// [`build_local_frame_with_connectivity`].
pub fn build_local_frame(
    mol: &Molecule,
    g_atom: usize,
    symmetry: SiteSymmetry,
) -> Result<Option<LocalFrame>> {
    let r_g = mol
        .atoms
        .get(g_atom)
        .ok_or_else(|| {
            DftError::InvalidInput(format!(
                "local frame: G atom index {g_atom} is outside a molecule with {} atoms",
                mol.len()
            ))
        })?
        .position;

    let dists: Vec<(usize, f64)> = mol
        .atoms
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != g_atom)
        .map(|(i, a)| (i, (a.position - r_g).norm()))
        .collect();
    if dists.is_empty() {
        return Err(DftError::Unsupported {
            feature: format!("local frame for the pseudo-substituent site at atom {g_atom}"),
            missing: "the molecule contains no atom other than the site itself; a bonded \
                      partner X is required"
                .to_string(),
        });
    }
    let min_d = dists.iter().map(|&(_, d)| d).fold(f64::INFINITY, f64::min);
    let x_candidates: Vec<usize> = dists
        .iter()
        .filter(|&&(_, d)| (d - min_d).abs() < EQUIDISTANT_TOL_BOHR)
        .map(|&(i, _)| i)
        .collect();
    if x_candidates.len() != 1 {
        return Err(DftError::Unsupported {
            feature: format!("local frame for the pseudo-substituent site at atom {g_atom}"),
            missing: format!(
                "the bonded partner X is not uniquely defined: candidate atoms \
                 {x_candidates:?} are within {EQUIDISTANT_TOL_BOHR:.1e} bohr of the shortest \
                 G-X distance ({min_d:.6} bohr)"
            ),
        });
    }
    let x_atom = x_candidates[0];
    if symmetry == SiteSymmetry::Cylindrical {
        return build_local_frame_with_connectivity(
            mol,
            g_atom,
            FrameConnectivity {
                x_atom,
                reference_candidates: &[],
            },
            symmetry,
        );
    }
    let ref_candidates: Vec<usize> = (0..mol.len())
        .filter(|&i| i != x_atom && i != g_atom)
        .collect();
    if ref_candidates.len() > 1 {
        return Err(DftError::Unsupported {
            feature: format!("local frame for the pseudo-substituent site at atom {g_atom}"),
            missing: format!(
                "the molecule has {} possible reference atoms after selecting X = atom \
                 {x_atom}, but carries no bond graph that can identify X's bonded neighbours; \
                 call build_local_frame_with_connectivity with an explicit bonded-reference \
                 candidate list",
                ref_candidates.len()
            ),
        });
    }

    build_local_frame_with_connectivity(
        mol,
        g_atom,
        FrameConnectivity {
            x_atom,
            reference_candidates: &ref_candidates,
        },
        symmetry,
    )
}

/// Build a local frame using explicit bonded connectivity.
///
/// No bond is inferred here. `connectivity.x_atom` must be the atom bonded to
/// G, and `reference_candidates` must contain only X's other bonded
/// neighbours. A non-cylindrical site needs at least one candidate;
/// cylindrical sites return `Ok(None)` because their x/y axes are gauge.
pub fn build_local_frame_with_connectivity(
    mol: &Molecule,
    g_atom: usize,
    connectivity: FrameConnectivity<'_>,
    symmetry: SiteSymmetry,
) -> Result<Option<LocalFrame>> {
    let r_g = mol
        .atoms
        .get(g_atom)
        .ok_or_else(|| {
            DftError::InvalidInput(format!(
                "local frame: G atom index {g_atom} is outside a molecule with {} atoms",
                mol.len()
            ))
        })?
        .position;
    let x_atom = connectivity.x_atom;
    let r_x = mol
        .atoms
        .get(x_atom)
        .ok_or_else(|| {
            DftError::InvalidInput(format!(
                "local frame: X atom index {x_atom} is outside a molecule with {} atoms",
                mol.len()
            ))
        })?
        .position;
    if x_atom == g_atom {
        return Err(DftError::InvalidInput(format!(
            "local frame: G and its bonded partner X both name atom {g_atom}"
        )));
    }
    let gx = r_g - r_x;
    if !gx.norm().is_finite() || gx.norm() == 0.0 {
        return Err(DftError::Unsupported {
            feature: format!("local frame for the pseudo-substituent site at atom {g_atom}"),
            missing: format!(
                "G atom {g_atom} and X atom {x_atom} do not define a finite nonzero bond axis"
            ),
        });
    }

    let ref_candidates = connectivity.reference_candidates;
    let mut seen = Vec::with_capacity(ref_candidates.len());
    for &candidate in ref_candidates {
        if candidate >= mol.len() {
            return Err(DftError::InvalidInput(format!(
                "local frame: reference atom index {candidate} is outside a molecule with {} atoms",
                mol.len()
            )));
        }
        if candidate == g_atom || candidate == x_atom {
            return Err(DftError::InvalidInput(format!(
                "local frame: reference candidate atom {candidate} is G or X, not another bonded neighbour of X"
            )));
        }
        if seen.contains(&candidate) {
            return Err(DftError::InvalidInput(format!(
                "local frame: reference candidate atom {candidate} is repeated"
            )));
        }
        seen.push(candidate);
    }

    if symmetry == SiteSymmetry::Cylindrical {
        return Ok(None);
    }
    if ref_candidates.is_empty() {
        return Err(DftError::Unsupported {
            feature: format!("local frame for the pseudo-substituent site at atom {g_atom}"),
            missing: format!(
                "atom {x_atom} (X) has no neighbour supplied as a bonded reference, so ê_x is \
                 undetermined; a {symmetry:?} site needs one (only SiteSymmetry::Cylindrical \
                 is exempt, since its ê_x/ê_y would be gauge freedom)"
            ),
        });
    }

    // Priority is applied only inside the caller-supplied bonded-neighbour
    // set. Duplicate indices were rejected above.
    let max_z = ref_candidates
        .iter()
        .map(|&i| mol.atoms[i].z)
        .max()
        .unwrap();
    let top_z: Vec<usize> = ref_candidates
        .iter()
        .copied()
        .filter(|&i| mol.atoms[i].z == max_z)
        .collect();
    let ref_atom = *top_z.iter().min().unwrap();
    let r_ref = mol.atoms[ref_atom].position;

    let ez = (r_g - r_x).normalized();
    let d_ref = r_ref - r_x;
    if !d_ref.norm().is_finite() || d_ref.norm() == 0.0 {
        return Err(DftError::Unsupported {
            feature: format!("local frame for the pseudo-substituent site at atom {g_atom}"),
            missing: format!(
                "R_ref atom {ref_atom} and X atom {x_atom} do not define a finite nonzero reference direction"
            ),
        });
    }
    let sin_angle = d_ref.normalized().cross(ez).norm();
    if sin_angle < NEAR_LINEAR_TOL_RAD.sin() {
        return Err(DftError::Unsupported {
            feature: format!("local frame for the pseudo-substituent site at atom {g_atom}"),
            missing: format!(
                "(R_ref - R_X) is within {:.2} degrees of parallel/antiparallel to ê_z (R_ref = \
                 atom {ref_atom}, X = atom {x_atom}), below the {:.1}-degree near-linear \
                 tolerance -- the ê_x projection would be ill-conditioned",
                sin_angle.asin().to_degrees(),
                NEAR_LINEAR_TOL_RAD.to_degrees()
            ),
        });
    }

    let [ex, ey, ez2] = build_frame_generic(
        [r_x.x, r_x.y, r_x.z],
        [r_g.x, r_g.y, r_g.z],
        [r_ref.x, r_ref.y, r_ref.z],
    );
    let u = Mat3::from_columns(
        Vec3::new(ex[0], ex[1], ex[2]),
        Vec3::new(ey[0], ey[1], ey[2]),
        Vec3::new(ez2[0], ez2[1], ez2[2]),
    );
    Ok(Some(LocalFrame {
        x_atom,
        ref_atom,
        u,
    }))
}

/// Analytic Jacobian `∂U/∂R`, via forward-mode AD (`DualN<f64, 9>`) through
/// the exact same [`build_frame_generic`] formula [`build_local_frame`]
/// uses for the value.
///
/// Input coordinate order (9 columns): `R_X.x,y,z`, `R_G.x,y,z`,
/// `R_ref.x,y,z`. Output row order (9 rows): `ê_x.x,y,z`, `ê_y.x,y,z`,
/// `ê_z.x,y,z` -- i.e. row `3*c + r` holds `∂(U's column c, row r)/∂(input
/// coordinate)`, matching `Mat3`'s column-of-`Vec3` layout.
///
/// Does not itself check the near-linear/degenerate conditions
/// [`build_local_frame`] refuses on; callers differentiating through a
/// frame that would have been refused get a numerically unstable (not
/// panicking) result, same as [`build_frame_generic`] itself.
pub fn frame_jacobian(r_x: Vec3, r_g: Vec3, r_ref: Vec3) -> Matrix {
    type D9 = DualN<f64, 9>;
    let rx = [
        D9::variable(r_x.x, 0),
        D9::variable(r_x.y, 1),
        D9::variable(r_x.z, 2),
    ];
    let rg = [
        D9::variable(r_g.x, 3),
        D9::variable(r_g.y, 4),
        D9::variable(r_g.z, 5),
    ];
    let rr = [
        D9::variable(r_ref.x, 6),
        D9::variable(r_ref.y, 7),
        D9::variable(r_ref.z, 8),
    ];
    let vecs = build_frame_generic(rx, rg, rr);
    let mut j = Matrix::zeros(9, 9);
    for (block, v) in vecs.iter().enumerate() {
        for (comp, d) in v.iter().enumerate() {
            let row = block * 3 + comp;
            for (k, &im) in d.im.iter().enumerate() {
                j[(row, k)] = im;
            }
        }
    }
    j
}

/// Value and analytic two-direction derivatives of the declared frame.
///
/// Inputs follow `frame_jacobian`: X, site, then reference, each xyz. Output
/// is `[column][Cartesian component][value, d_a, d_b, d_a_d_b]`. The directions
/// are constant Cartesian directions. Nested forward duals differentiate the
/// same frame arithmetic used by the value and Jacobian; no finite differences
/// or frame-axis reselection occur here. Like `frame_jacobian`, this low-level
/// function requires the caller to validate the frame with the guarded builder.
pub fn frame_directional_second(
    r_x: Vec3,
    r_g: Vec3,
    r_ref: Vec3,
    direction_a: [f64; 9],
    direction_b: [f64; 9],
) -> [[[f64; 4]; 3]; 3] {
    type D1 = DualN<f64, 1>;
    type D11 = DualN<D1, 1>;
    let coordinates = [
        r_x.x, r_x.y, r_x.z, r_g.x, r_g.y, r_g.z, r_ref.x, r_ref.y, r_ref.z,
    ];
    let seeded: [D11; 9] = std::array::from_fn(|i| D11 {
        re: D1 {
            re: coordinates[i],
            im: [direction_b[i]],
        },
        im: [D1 {
            re: direction_a[i],
            im: [0.0],
        }],
    });
    let frame = build_frame_generic(
        [seeded[0], seeded[1], seeded[2]],
        [seeded[3], seeded[4], seeded[5]],
        [seeded[6], seeded[7], seeded[8]],
    );
    std::array::from_fn(|column| {
        std::array::from_fn(|axis| {
            let value = frame[column][axis];
            [
                value.re.re,
                value.im[0].re,
                value.re.im[0],
                value.im[0].im[0],
            ]
        })
    })
}

/// All analytic frame Hessians. Entry `3*column+axis` is a 9 by 9 matrix in
/// the same input-coordinate order as `frame_jacobian`. Both index orders are
/// evaluated independently; the implementation does not impose symmetry by
/// copying one triangle. The guarded frame builder remains the caller's domain
/// check, as for the first-derivative interface.
pub fn frame_hessian(r_x: Vec3, r_g: Vec3, r_ref: Vec3) -> Vec<Matrix> {
    let mut result = (0..9).map(|_| Matrix::zeros(9, 9)).collect::<Vec<_>>();
    for a in 0..9 {
        let mut direction_a = [0.0; 9];
        direction_a[a] = 1.0;
        for b in 0..9 {
            let mut direction_b = [0.0; 9];
            direction_b[b] = 1.0;
            let derivative = frame_directional_second(r_x, r_g, r_ref, direction_a, direction_b);
            for column in 0..3 {
                for axis in 0..3 {
                    result[3 * column + axis][(a, b)] = derivative[column][axis][3];
                }
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::Atom;

    fn mol_of(positions: &[(u8, Vec3)]) -> Molecule {
        Molecule::new(
            positions
                .iter()
                .map(|&(z, position)| Atom { z, position })
                .collect(),
        )
    }

    // ---------------------------------------------------------------
    // Refusal / success gates for build_local_frame.
    // ---------------------------------------------------------------

    #[test]
    fn refuses_when_x_not_unique() {
        let mol = mol_of(&[
            (1, Vec3::new(0.0, 0.0, 0.0)),  // G, atom 0
            (6, Vec3::new(1.5, 0.0, 0.0)),  // atom 1
            (6, Vec3::new(-1.5, 0.0, 0.0)), // atom 2, same distance
        ]);
        match build_local_frame(&mol, 0, SiteSymmetry::None) {
            Err(DftError::Unsupported { feature, missing }) => {
                assert!(feature.contains("atom 0"));
                assert!(missing.contains("not unique"));
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn refuses_when_no_reference_and_not_cylindrical() {
        let mol = mol_of(&[
            (6, Vec3::new(0.0, 0.0, 0.0)), // X, atom 0
            (1, Vec3::new(0.0, 0.0, 2.0)), // G, atom 1
        ]);
        match build_local_frame(&mol, 1, SiteSymmetry::C2v) {
            Err(DftError::Unsupported { missing, .. }) => {
                assert!(missing.contains("no neighbour"));
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn cylindrical_with_no_reference_returns_none() {
        let mol = mol_of(&[(6, Vec3::new(0.0, 0.0, 0.0)), (1, Vec3::new(0.0, 0.0, 2.0))]);
        let result = build_local_frame(&mol, 1, SiteSymmetry::Cylindrical).unwrap();
        assert!(result.is_none());
    }

    /// `R_ref` is placed at a small deviation angle from the direction
    /// **antiparallel** to `ê_z` (i.e. roughly opposite G, on the far side
    /// of X) rather than parallel to it: the near-linear tolerance is
    /// symmetric under `d_ref -> -d_ref` (only `sin(angle)` enters the
    /// check, see `NEAR_LINEAR_TOL_RAD`'s doc comment), so this is an
    /// equally valid trigger for it, and it keeps `R_ref` far from `G` so
    /// the nearest-neighbour search that discovers `X` is not confused by
    /// `R_ref` itself looking like a closer bond partner for `G` than `X`
    /// is (which a same-side, small-angle placement would do, since it
    /// would then sit almost on the X-G segment, close to G).
    fn antiparallel_ref(deviation: f64, bond: f64) -> Vec3 {
        Vec3::new(bond * deviation.sin(), 0.0, -bond * deviation.cos())
    }

    #[test]
    fn refuses_when_near_linear() {
        let deviation = 2.0_f64.to_radians(); // inside the 5-degree tolerance
        let mol = mol_of(&[
            (6, Vec3::new(0.0, 0.0, 0.0)),         // X, atom 0
            (1, Vec3::new(0.0, 0.0, 2.0)),         // G, atom 1
            (7, antiparallel_ref(deviation, 1.5)), // atom 2: nearly antiparallel to ê_z
        ]);
        match build_local_frame(&mol, 1, SiteSymmetry::None) {
            Err(DftError::Unsupported { missing, .. }) => {
                assert!(missing.contains("parallel"));
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn succeeds_just_past_near_linear_threshold_and_frame_is_orthonormal_right_handed() {
        let deviation = 7.0_f64.to_radians(); // just past the 5-degree tolerance
        let mol = mol_of(&[
            (6, Vec3::new(0.0, 0.0, 0.0)),
            (1, Vec3::new(0.0, 0.0, 2.0)),
            (7, antiparallel_ref(deviation, 1.5)),
        ]);
        let frame = build_local_frame(&mol, 1, SiteSymmetry::None)
            .unwrap()
            .unwrap();
        let u = frame.u;
        assert!((u.col[0].norm() - 1.0).abs() < 1e-12);
        assert!((u.col[1].norm() - 1.0).abs() < 1e-12);
        assert!((u.col[2].norm() - 1.0).abs() < 1e-12);
        assert!(u.col[0].dot(u.col[1]).abs() < 1e-12);
        assert!(u.col[1].dot(u.col[2]).abs() < 1e-12);
        assert!(u.col[0].dot(u.col[2]).abs() < 1e-12);
        // Right-handed by construction: ê_x x ê_y == ê_z.
        let cross = u.col[0].cross(u.col[1]);
        assert!((cross - u.col[2]).norm() < 1e-12);
    }

    #[test]
    fn priority_picks_highest_z_then_lowest_index() {
        let mol = mol_of(&[
            (6, Vec3::new(0.0, 0.0, 0.0)),  // X, atom 0
            (1, Vec3::new(0.0, 0.0, 2.0)),  // G, atom 1
            (6, Vec3::new(1.5, 0.3, 0.0)),  // atom 2: carbon
            (7, Vec3::new(-1.2, 0.9, 0.0)), // atom 3: nitrogen -- higher Z, wins
        ]);
        let frame = build_local_frame_with_connectivity(
            &mol,
            1,
            FrameConnectivity {
                x_atom: 0,
                reference_candidates: &[2, 3],
            },
            SiteSymmetry::None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(frame.x_atom, 0);
        assert_eq!(frame.ref_atom, 3);
    }

    #[test]
    fn priority_ties_on_z_break_by_lowest_index() {
        let mol = mol_of(&[
            (6, Vec3::new(0.0, 0.0, 0.0)),  // X, atom 0
            (1, Vec3::new(0.0, 0.0, 2.0)),  // G, atom 1
            (7, Vec3::new(1.5, 0.3, 0.0)),  // atom 2: nitrogen
            (7, Vec3::new(-1.2, 0.9, 0.0)), // atom 3: nitrogen, same Z, later index
        ]);
        let frame = build_local_frame_with_connectivity(
            &mol,
            1,
            FrameConnectivity {
                x_atom: 0,
                reference_candidates: &[2, 3],
            },
            SiteSymmetry::None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(frame.ref_atom, 2);
    }

    #[test]
    fn explicit_connectivity_prevents_remote_high_z_from_stealing_reference() {
        let base = mol_of(&[
            (6, Vec3::new(0.0, 0.0, 0.0)),  // X, atom 0
            (1, Vec3::new(0.0, 0.0, 2.0)),  // G, atom 1
            (6, Vec3::new(1.4, 0.3, 0.0)),  // bonded C candidate, atom 2
            (7, Vec3::new(-1.1, 0.8, 0.0)), // bonded N candidate, atom 3
        ]);
        let with_remote_iodine = mol_of(&[
            (6, Vec3::new(0.0, 0.0, 0.0)),
            (1, Vec3::new(0.0, 0.0, 2.0)),
            (6, Vec3::new(1.4, 0.3, 0.0)),
            (7, Vec3::new(-1.1, 0.8, 0.0)),
            (53, Vec3::new(40.0, -30.0, 20.0)), // not bonded to X
        ]);
        let connectivity = FrameConnectivity {
            x_atom: 0,
            reference_candidates: &[2, 3],
        };
        let frame_base =
            build_local_frame_with_connectivity(&base, 1, connectivity, SiteSymmetry::None)
                .unwrap()
                .unwrap();
        let frame_remote = build_local_frame_with_connectivity(
            &with_remote_iodine,
            1,
            connectivity,
            SiteSymmetry::None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(frame_base.ref_atom, 3);
        assert_eq!(frame_remote.ref_atom, 3);
        for axis in 0..3 {
            assert_eq!(frame_base.u.col[axis], frame_remote.u.col[axis]);
        }

        match build_local_frame(&with_remote_iodine, 1, SiteSymmetry::None) {
            Err(DftError::Unsupported { missing, .. }) => {
                assert!(missing.contains("no bond graph"));
                assert!(missing.contains("build_local_frame_with_connectivity"));
            }
            other => panic!("legacy API must refuse ambiguous reference atoms, got {other:?}"),
        }
    }

    #[test]
    fn explicit_connectivity_rejects_malformed_candidate_sets() {
        let mol = mol_of(&[
            (6, Vec3::new(0.0, 0.0, 0.0)),
            (1, Vec3::new(0.0, 0.0, 2.0)),
            (7, Vec3::new(1.0, 0.6, 0.0)),
        ]);
        for connectivity in [
            FrameConnectivity {
                x_atom: 1,
                reference_candidates: &[2],
            },
            FrameConnectivity {
                x_atom: 0,
                reference_candidates: &[0],
            },
            FrameConnectivity {
                x_atom: 0,
                reference_candidates: &[2, 2],
            },
            FrameConnectivity {
                x_atom: 0,
                reference_candidates: &[3],
            },
        ] {
            assert!(matches!(
                build_local_frame_with_connectivity(&mol, 1, connectivity, SiteSymmetry::None),
                Err(DftError::InvalidInput(_))
            ));
        }
    }

    // ---------------------------------------------------------------
    // Deliverable 2b: the Jacobian, gated in isolation against central
    // finite differences.
    // ---------------------------------------------------------------

    fn frame_value(r_x: Vec3, r_g: Vec3, r_ref: Vec3) -> [[f64; 3]; 3] {
        build_frame_generic(
            [r_x.x, r_x.y, r_x.z],
            [r_g.x, r_g.y, r_g.z],
            [r_ref.x, r_ref.y, r_ref.z],
        )
    }

    fn unpack(c: [f64; 9]) -> (Vec3, Vec3, Vec3) {
        (
            Vec3::new(c[0], c[1], c[2]),
            Vec3::new(c[3], c[4], c[5]),
            Vec3::new(c[6], c[7], c[8]),
        )
    }

    fn fd_jacobian(r_x: Vec3, r_g: Vec3, r_ref: Vec3, h: f64) -> Matrix {
        let mut coords = [
            r_x.x, r_x.y, r_x.z, r_g.x, r_g.y, r_g.z, r_ref.x, r_ref.y, r_ref.z,
        ];
        let mut j = Matrix::zeros(9, 9);
        for k in 0..9 {
            let base = coords[k];
            coords[k] = base + h;
            let (a, b, c) = unpack(coords);
            let plus = frame_value(a, b, c);
            coords[k] = base - h;
            let (a, b, c) = unpack(coords);
            let minus = frame_value(a, b, c);
            coords[k] = base;
            for block in 0..3 {
                for comp in 0..3 {
                    let row = block * 3 + comp;
                    j[(row, k)] = (plus[block][comp] - minus[block][comp]) / (2.0 * h);
                }
            }
        }
        j
    }

    /// Deliverable 2b's gate, standalone: `frame_jacobian` (analytic, via
    /// `DualN<f64, 9>`) against central finite differences (`h = 1e-5`
    /// bohr) on all 9 coordinates, for three geometries -- a generic
    /// asymmetric one, a second generic one, and one just inside the
    /// near-linear refusal boundary (7 degrees, refusal is at 5).
    #[test]
    fn jacobian_matches_finite_differences() {
        let h = 1e-5;
        let near_linear_angle = 7.0_f64.to_radians();
        let geometries = [
            (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 2.0),
                Vec3::new(1.3, 0.5, -0.3),
            ),
            (
                Vec3::new(0.4, -0.2, 0.1),
                Vec3::new(-0.6, 0.3, 2.5),
                Vec3::new(-1.1, 1.4, 0.6),
            ),
            (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 2.0),
                Vec3::new(
                    1.5 * near_linear_angle.sin(),
                    0.0,
                    1.5 * near_linear_angle.cos(),
                ),
            ),
        ];
        let mut worst = 0.0_f64;
        for &(r_x, r_g, r_ref) in &geometries {
            let analytic = frame_jacobian(r_x, r_g, r_ref);
            let fd = fd_jacobian(r_x, r_g, r_ref, h);
            worst = worst.max(analytic.max_abs_diff(&fd));
        }
        assert!(
            worst < 1e-8,
            "jacobian worst-case |analytic - FD| = {worst:e}"
        );
        eprintln!("frame jacobian worst-case error: {worst:e}");
    }
}
