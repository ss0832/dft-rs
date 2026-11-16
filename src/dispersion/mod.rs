// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Empirical dispersion corrections: Grimme D2 (2006) and D3 (2010) with
//! zero and Becke–Johnson damping.
//!
//! - Model: S. Grimme et al., J. Chem. Phys. 132, 154104 (2010);
//!   S. Grimme, J. Comput. Chem. 27, 1787 (2006); BJ damping:
//!   S. Grimme, S. Ehrlich, L. Goerigk, J. Comput. Chem. 32, 1456 (2011).
//! - D3 reference data (`data/*.csv`): via MOPAC v23.2.5 (Apache-2.0),
//!   PROVENANCE headers in each file.
//! - Everything is `Scalar`-generic: gradients AND Hessians come from dual
//!   seeding of the coordinates — analytic, not finite-difference. The
//!   Hessian nests one level deeper (`DualN<DualN<f64,3>,3>`, per `dual.rs`'s
//!   "nests to arbitrary order" contract) and reuses exactly the same energy
//!   kernels the gradient does.

pub mod d1;
pub(crate) mod hirshfeld;
pub mod mbd;
pub mod ts;
pub mod xdm;

use crate::dual::{DualN, Scalar};
use crate::error::{DftError, Result};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::system::Molecule;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Which dispersion correction to apply (off by default).
#[derive(Clone, Debug, PartialEq, Default)]
pub enum Dispersion {
    #[default]
    None,
    /// Original D1 with the caller's atomic parameters in molecular atom order.
    D1 {
        atoms: Vec<d1::AtomParameters>,
        s6: f64,
    },
    D2,
    D3Zero,
    D3Bj,
    /// D3 with zero damping plus the three-body (Axilrod-Teller-Muto) term.
    D3ZeroAbc,
    /// D3 with Becke-Johnson damping plus the three-body term.
    D3BjAbc,
    /// Chai–Head-Gordon damping over the D2 C6 table (the ωB97X-D tail):
    /// E = −Σ C6ij/R⁶ · [1 + a(R/Rr)⁻¹²]⁻¹ with a = 6.0
    /// (Chai & Head-Gordon, PCCP 10, 6615 (2008)).
    Chg,
}

impl Dispersion {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "none" => Some(Self::None),
            "d2" => Some(Self::D2),
            "d3" | "d3zero" | "d3(0)" => Some(Self::D3Zero),
            "d3bj" | "d3(bj)" => Some(Self::D3Bj),
            "d3abc" | "d3-abc" | "d3zero-abc" => Some(Self::D3ZeroAbc),
            "d3bjabc" | "d3bj-abc" => Some(Self::D3BjAbc),
            "chg" => Some(Self::Chg),
            _ => None,
        }
    }
}

/// Zero-damping D3 parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct D3ZeroParams {
    pub s6: f64,
    pub s8: f64,
    pub rs6: f64,
    pub rs8: f64,
    pub alp6: f64,
    pub alp8: f64,
}

/// BJ-damping D3 parameters.
#[derive(Clone, Copy, Debug)]
pub struct D3BjParams {
    pub s6: f64,
    pub s8: f64,
    pub a1: f64,
    pub a2: f64,
}

/// Published per-functional D3(zero) parameters (Grimme 2010 + website
/// tables; s6 = 1, α6 = 14, α8 = 16, rs8 = 1).
pub fn d3_zero_params(functional: &str) -> Result<D3ZeroParams> {
    let name = functional.trim().to_ascii_lowercase();
    if crate::xc::normalize(&name) == "wb97xd3" {
        return Ok(crate::xc::wb97x_d3::PAIR_PARAMETERS);
    }
    let (rs6, s8) = match name.as_str() {
        "b3lyp" | "b3lyp5" => (1.261, 1.703),
        "blyp" => (1.094, 1.682),
        "pbe" => (1.217, 0.722),
        "pbe0" => (1.287, 0.928),
        "camb3lyp" | "cam-b3lyp" => (1.378, 1.217),
        "hf" => (1.158, 1.746),
        "revpbe" => (0.923, 1.010),
        "tpss" => (1.166, 1.105),
        "bp86" => (1.139, 1.683),
        "m06-2x" | "m062x" => (1.619, 0.0),
        "b3pw91" => (1.176, 1.775),
        "hse06" => (1.129, 0.109),
        "lc-wpbe" | "lcwpbe" => (1.355, 1.279),
        "o3lyp" => (1.4060, 1.8058),
        other => {
            return Err(DftError::Unsupported {
                feature: format!("D3(zero) for functional `{other}`"),
                missing: "a published damping-parameter fit".into(),
            })
        }
    };
    Ok(D3ZeroParams {
        s6: 1.0,
        s8,
        rs6,
        rs8: 1.0,
        alp6: 14.0,
        alp8: 16.0,
    })
}

/// Published per-functional D3(BJ) parameters (Grimme/Goerigk 2011).
pub fn d3_bj_params(functional: &str) -> Result<D3BjParams> {
    let (a1, s8, a2) = match functional.to_ascii_lowercase().as_str() {
        "b3lyp" | "b3lyp5" => (0.3981, 1.9889, 4.4211),
        "blyp" => (0.4298, 2.6996, 4.2359),
        "pbe" => (0.4289, 0.7875, 4.4407),
        "pbe0" => (0.4145, 1.2177, 4.8593),
        "camb3lyp" | "cam-b3lyp" => (0.3708, 2.0674, 5.4743),
        "hf" => (0.3385, 0.9171, 2.8830),
        "revpbe" => (0.5238, 2.3550, 3.5016),
        "tpss" => (0.4535, 1.9435, 4.4752),
        "bp86" => (0.3946, 3.2822, 4.8516),
        "r2scan" => (0.49484001, 0.78981345, 5.73083694),
        "b3pw91" => (0.4312, 2.8524, 4.4693),
        "hse06" => (0.3830, 2.3100, 5.6850),
        "lc-wpbe" | "lcwpbe" => (0.3919, 1.8541, 5.0897),
        "o3lyp" => (0.0963, 1.8171, 5.9940),
        other => {
            return Err(DftError::Unsupported {
                feature: format!("D3(BJ) for functional `{other}`"),
                missing: "a published damping-parameter fit".into(),
            })
        }
    };
    Ok(D3BjParams {
        s6: 1.0,
        s8,
        a1,
        a2,
    })
}

/// D2 s6 per functional (Grimme 2006).
pub fn d2_s6(functional: &str) -> Result<f64> {
    Ok(match functional.to_ascii_lowercase().as_str() {
        "b3lyp" | "b3lyp5" => 1.05,
        "blyp" => 1.2,
        "pbe" => 0.75,
        "pbe0" => 0.6,
        "bp86" => 1.05,
        "tpss" => 1.0,
        other => {
            return Err(DftError::Unsupported {
                feature: format!("D2 for functional `{other}`"),
                missing: "a published global s6 fit".into(),
            })
        }
    })
}

// ---------------------------------------------------------------------------
// D3 reference data (CSV-embedded; PROVENANCE headers in the files).
// ---------------------------------------------------------------------------

const D3_C6_CSV: &str = include_str!("data/d3_c6_reference.csv");
const D3_R0AB_CSV: &str = include_str!("data/d3_r0ab.csv");
const D3_RADII_CSV: &str = include_str!("data/d3_radii.csv");

pub(crate) struct D3Data {
    /// `(max(z), min(z))` → reference `(c6, cn_a, cn_b)` tuples.
    c6: HashMap<(u8, u8), Vec<(f64, f64, f64)>>,
    /// Cutoff radii (Bohr), 94×94, 0-indexed by Z−1.
    pub(crate) r0ab: Vec<Vec<f64>>,
    /// `sqrt(0.5·r2r4_raw·√Z)` per element.
    pub(crate) r2r4: Vec<f64>,
    /// Scaled covalent radii (Bohr).
    pub(crate) rcov: Vec<f64>,
    pub(crate) max_elem: usize,
}

fn parse_csv(text: &str) -> Vec<Vec<f64>> {
    text.lines()
        .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
        .skip(1) // header row
        .map(|l| {
            l.split(',')
                .map(|t| t.trim().parse::<f64>().unwrap_or(f64::NAN))
                .collect()
        })
        .collect()
}

pub(crate) fn data() -> &'static D3Data {
    static D: OnceLock<D3Data> = OnceLock::new();
    D.get_or_init(|| {
        const MAX: usize = 94;
        let mut c6: HashMap<(u8, u8), Vec<(f64, f64, f64)>> = HashMap::new();
        for row in parse_csv(D3_C6_CSV) {
            let (iat, jat) = (row[0] as u8, row[1] as u8);
            let key = (iat.max(jat), iat.min(jat));
            let entry = if iat >= jat {
                (row[2], row[3], row[4])
            } else {
                (row[2], row[4], row[3])
            };
            c6.entry(key).or_default().push(entry);
            // The table of a homonuclear pair lists each pair of reference
            // coordination numbers once (`cn_a >= cn_b`); the interpolation
            // needs the mirrored point as well, or C6(CN_i, CN_j) differs from
            // C6(CN_j, CN_i) and the energy depends on the order of the atoms.
            if iat == jat && row[3] != row[4] {
                c6.entry(key).or_default().push((row[2], row[4], row[3]));
            }
        }
        let packed: Vec<f64> = parse_csv(D3_R0AB_CSV).into_iter().map(|r| r[0]).collect();
        let mut r0ab = vec![vec![0.0; MAX]; MAX];
        let mut k = 0;
        for i in 0..MAX {
            for j in 0..=i {
                let v = packed[k] * crate::constants::ANGSTROM_TO_BOHR;
                r0ab[i][j] = v;
                r0ab[j][i] = v;
                k += 1;
            }
        }
        let mut r2r4 = vec![0.0; MAX + 1];
        let mut rcov = vec![0.0; MAX + 1];
        for row in parse_csv(D3_RADII_CSV) {
            let z = row[0] as usize;
            if z == 0 || z > MAX {
                continue;
            }
            r2r4[z] = (0.5 * row[1] * (z as f64).sqrt()).sqrt();
            rcov[z] = 4.0 / 3.0 * row[2] * crate::constants::ANGSTROM_TO_BOHR;
        }
        D3Data {
            c6,
            r0ab,
            r2r4,
            rcov,
            max_elem: MAX,
        }
    })
}

// ---------------------------------------------------------------------------
// Generic D3 machinery.
// ---------------------------------------------------------------------------

fn dist_g<S: Scalar>(a: &[S; 3], b: &[S; 3]) -> S {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// DFTD3 coordination numbers (k1 = 16).
fn coordination_numbers_g<S: Scalar>(numbers: &[u8], pos: &[[S; 3]], d: &D3Data) -> Vec<S> {
    const K1: f64 = 16.0;
    let n = numbers.len();
    let one = S::from_f64(1.0);
    let mut cn = vec![S::from_f64(0.0); n];
    for i in 0..n {
        let zi = numbers[i] as usize;
        if zi == 0 || zi > d.max_elem {
            continue;
        }
        for j in 0..n {
            if i == j {
                continue;
            }
            let zj = numbers[j] as usize;
            if zj == 0 || zj > d.max_elem {
                continue;
            }
            let r = dist_g(&pos[j], &pos[i]);
            let rco = d.rcov[zi] + d.rcov[zj];
            cn[i] = cn[i]
                + (((r.recip() * S::from_f64(rco) - one) * S::from_f64(-K1)).exp() + one).recip();
        }
    }
    cn
}

/// Gaussian-weighted C6 interpolation (k3 = −4).
pub(crate) fn getc6_g<S: Scalar>(d: &D3Data, zi: u8, zj: u8, cni: S, cnj: S) -> S {
    const K3: f64 = -4.0;
    let key = (zi.max(zj), zi.min(zj));
    let refs = match d.c6.get(&key) {
        Some(r) => r,
        None => return S::from_f64(0.0),
    };
    let (cn_i, cn_j) = if zi >= zj { (cni, cnj) } else { (cnj, cni) };
    // Naively exponentiating exponent_k = K3·(Δcn_a² + Δcn_b²) per reference
    // point and normalizing (Σ c6_k·w_k)/(Σ w_k) underflows every w_k to a
    // hard 0.0 once ΔCN gets large enough (4·ΔCN² > ~745, the point where
    // exp() underflows f64's smallest subnormal) — turning a legitimate
    // reference key into a silent 0/0. This is reachable in practice: a
    // pathological geometry step (optimizer overshoot, MD blowup) that
    // crushes several atoms together drives the coordination number a good
    // ten-plus past any tabulated reference CN. That 0/0 then propagates
    // into a NaN BJ damping radius downstream (see the R0 comment below).
    //
    // Fix: log-sum-exp. Find the largest exponent first (using only the
    // real/primal CN value — this is a discrete "which reference is
    // closest" choice, not something that needs to be differentiated
    // through, matching how the rest of this module branches on
    // `.value()`) and shift every exponent by it before exponentiating.
    // Subtracting the same constant from numerator and denominator leaves
    // both the value AND its derivative unchanged (it's a common factor
    // exp(-max_exp) on every term), but now the closest reference point's
    // shifted weight is exactly 1 and every other weight is <= 1 — so the
    // normalized average is always well-defined, and it smoothly asymptotes
    // to the closest reference's C6 as ΔCN → ∞ instead of collapsing to 0.
    let cni_v = cn_i.value();
    let cnj_v = cn_j.value();
    let mut max_exp = f64::NEG_INFINITY;
    for &(_, cna, cnb) in refs {
        let da = cni_v - cna;
        let db = cnj_v - cnb;
        let e = K3 * (da * da + db * db);
        if e > max_exp {
            max_exp = e;
        }
    }
    let max_exp_s = S::from_f64(max_exp);
    let mut num = S::from_f64(0.0);
    let mut den = S::from_f64(0.0);
    for &(c6, cna, cnb) in refs {
        let da = cn_i - S::from_f64(cna);
        let db = cn_j - S::from_f64(cnb);
        let t = ((da * da + db * db) * S::from_f64(K3) - max_exp_s).exp();
        num = num + t * S::from_f64(c6);
        den = den + t;
    }
    num / den
}

/// D3 dispersion energy (Hartree), generic over the scalar so coordinate
/// seeding yields analytic derivatives INCLUDING the CN coupling.
pub fn d3_energy_g<S: Scalar>(
    numbers: &[u8],
    pos: &[[S; 3]],
    zero: Option<&D3ZeroParams>,
    bj: Option<&D3BjParams>,
) -> S {
    let d = data();
    let cn = coordination_numbers_g(numbers, pos, d);
    let n = numbers.len();
    let one = S::from_f64(1.0);
    let mut energy = S::from_f64(0.0);
    for i in 0..n.saturating_sub(1) {
        let zi = numbers[i];
        if zi == 0 || zi as usize > d.max_elem {
            continue;
        }
        for j in i + 1..n {
            let zj = numbers[j];
            if zj == 0 || zj as usize > d.max_elem {
                continue;
            }
            let r = dist_g(&pos[j], &pos[i]);
            let r2 = r * r;
            let r6 = r2 * r2 * r2;
            let r8 = r6 * r2;
            let c6 = getc6_g(d, zi, zj, cn[i], cn[j]);
            let c8fac = 3.0 * d.r2r4[zi as usize] * d.r2r4[zj as usize];
            let c8 = c6 * S::from_f64(c8fac);
            if let Some(p) = zero {
                let rr = r.recip() * S::from_f64(d.r0ab[zi as usize - 1][zj as usize - 1]);
                let damp6 =
                    ((rr * S::from_f64(p.rs6)).powf(p.alp6) * S::from_f64(6.0) + one).recip();
                let damp8 =
                    ((rr * S::from_f64(p.rs8)).powf(p.alp8) * S::from_f64(6.0) + one).recip();
                energy = energy
                    - c6 * damp6 / r6 * S::from_f64(p.s6)
                    - c8 * damp8 / r8 * S::from_f64(p.s8);
            }
            if let Some(p) = bj {
                // R0_BJ = √(C8/C6); f = a1·R0 + a2. Computing R0 from the
                // live c6/c8 values re-introduces a 0/0 whenever getc6_g's
                // CN-interpolation weights fully underflow: C6 → 0 forces
                // C8 = C6·c8fac → 0 right along with it, so (c8/c6).sqrt()
                // becomes (0/0).sqrt() = NaN, silently NaN-poisoning the
                // whole energy/gradient. But c8fac = 3·r2r4_i·r2r4_j never
                // touches C6 at all — it IS C8/C6 for any nonzero C6 by
                // construction — so R0 doesn't actually depend on C6 and
                // can be taken straight from c8fac, sidestepping the
                // division (and its 0/0) entirely.
                let r0 = S::from_f64(c8fac.sqrt());
                let f = r0 * S::from_f64(p.a1) + S::from_f64(p.a2);
                let f2 = f * f;
                let f6 = f2 * f2 * f2;
                let f8 = f6 * f2;
                energy = energy
                    - c6 / (r6 + f6) * S::from_f64(p.s6)
                    - c8 / (r8 + f8) * S::from_f64(p.s8);
            }
        }
    }
    energy
}

// ---------------------------------------------------------------------------
// D2 (Grimme 2006): C6 in J·nm⁶/mol and R0 in Å per element (paper Table 1,
// H–Kr; Sc–Zn share the K/Ca value per the paper's prescription).
// ---------------------------------------------------------------------------

const D2_C6_JNM6: [f64; 37] = [
    0.0, 0.14, 0.08, 1.61, 1.61, 3.13, 1.75, 1.23, 0.70, 0.75, 0.63, 5.71, 5.71, 10.79, 9.23, 7.84,
    5.57, 5.07, 4.61, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8,
    16.99, 17.10, 16.37, 12.64, 12.47, 12.01,
];
const D2_R0_ANG: [f64; 37] = [
    0.0, 1.001, 1.012, 0.825, 1.408, 1.485, 1.452, 1.397, 1.342, 1.287, 1.243, 1.144, 1.364, 1.639,
    1.716, 1.705, 1.683, 1.639, 1.595, 1.485, 1.474, 1.562, 1.562, 1.562, 1.562, 1.562, 1.562,
    1.562, 1.562, 1.562, 1.562, 1.650, 1.727, 1.760, 1.771, 1.749, 1.727,
];

/// D2 energy (Hartree). `E = −s6 Σ √(C6i C6j)/R⁶ · 1/(1+e^{−20(R/Rr−1)})`.
pub fn d2_energy_g<S: Scalar>(numbers: &[u8], pos: &[[S; 3]], s6: f64) -> S {
    let n = numbers.len();
    let mut e = S::from_f64(0.0);
    for i in 0..n.saturating_sub(1) {
        for j in i + 1..n {
            if let Some((c6, rr)) = d2_pair_parameters(numbers[i], numbers[j]) {
                let r = dist_g(&pos[j], &pos[i]);
                e = e - d2_radial_g(r, rr).scale(s6 * c6);
            }
        }
    }
    e
}

/// D2's damped inverse-sixth power of distance, shared by molecular and
/// periodic pair sums. Scale and pair coefficient are applied by the caller.
pub(crate) fn d2_radial_g<S: Scalar>(r: S, radius: f64) -> S {
    let r2 = r * r;
    let r6 = r2 * r2 * r2;
    logistic_damping(r, radius, 20.0) / r6
}

pub(crate) fn logistic_damping<S: Scalar>(r: S, radius: f64, steepness: f64) -> S {
    let one = S::from_f64(1.0);
    (((r / S::from_f64(radius) - one).scale(-steepness)).exp() + one).recip()
}

/// CHG-damped dispersion energy (Hartree) on the D2 C6/R0 tables:
/// `E = −Σ √(C6i C6j)/R⁶ · [1 + a(R/Rr)⁻¹²]⁻¹` (s6 ≡ 1; ωB97X-D: a = 6).
pub fn chg_energy_g<S: Scalar>(numbers: &[u8], pos: &[[S; 3]], a: f64) -> S {
    let ang6 = crate::constants::ANGSTROM_TO_BOHR.powi(6);
    let jnm6_to_au = 1.0e6 * ang6 / (crate::constants::HARTREE_TO_KJ_MOL * 1000.0);
    let one = S::from_f64(1.0);
    let n = numbers.len();
    let mut e = S::from_f64(0.0);
    for i in 0..n.saturating_sub(1) {
        let zi = numbers[i] as usize;
        if zi == 0 || zi >= D2_C6_JNM6.len() {
            continue;
        }
        for j in i + 1..n {
            let zj = numbers[j] as usize;
            if zj == 0 || zj >= D2_C6_JNM6.len() {
                continue;
            }
            let c6 = (D2_C6_JNM6[zi] * D2_C6_JNM6[zj]).sqrt() * jnm6_to_au;
            let rr = (D2_R0_ANG[zi] + D2_R0_ANG[zj]) * crate::constants::ANGSTROM_TO_BOHR;
            let r = dist_g(&pos[j], &pos[i]);
            let r2 = r * r;
            let r6 = r2 * r2 * r2;
            let ratio = r / S::from_f64(rr);
            let t2 = ratio * ratio;
            let t4 = t2 * t2;
            let t12 = t4 * t4 * t4;
            let fdmp = (one + S::from_f64(a) / t12).recip();
            e = e - fdmp / r6 * S::from_f64(c6);
        }
    }
    e
}

/// Axilrod-Teller-Muto energy of one triple of atoms from its three squared
/// distances: `c9 [3 cos(a) cos(b) cos(c) + 1] / (r_ab r_ac r_bc)^3` with the
/// zero damping `1 / (1 + 6 (r0 / (r_ab r_ac r_bc))^(alpha/3))`, where `r0`
/// is the product of the three scaled pair radii and `c9` is positive
/// (S. Grimme, J. Antony, S. Ehrlich, H. Krieg, J. Chem. Phys. 132, 154104
/// (2010), eq. 14).
pub fn atm_term_g<S: Scalar>(r2ab: S, r2ac: S, r2bc: S, c9: S, r0: f64, alpha: f64) -> S {
    let r2 = r2ab * r2ac * r2bc;
    let r1 = r2.sqrt();
    let r3 = r2 * r1;
    let r5 = r3 * r2;
    let angular = ((r2ab + r2bc - r2ac) * (r2ab - r2bc + r2ac) * (r2ac + r2bc - r2ab)).scale(0.375)
        / r5
        + r3.recip();
    let damping = (r1.recip().scale(r0).powf(alpha / 3.0).scale(6.0) + S::from_f64(1.0)).recip();
    c9 * angular * damping
}

/// Three-body (Axilrod-Teller-Muto) energy of a molecule: every triple of
/// atoms with `C9 = sqrt(C6_ab C6_ac C6_bc)` on the D3 coordination numbers,
/// pair radii scaled by 4/3 and the damping exponent 16 (see
/// [`atm_term_g`]).
pub fn d3_three_body_g<S: Scalar>(numbers: &[u8], pos: &[[S; 3]]) -> S {
    let d = data();
    let cn = coordination_numbers_g(numbers, pos, d);
    let n = numbers.len();
    let valid = |z: u8| z != 0 && z as usize <= d.max_elem;
    let squared = |a: usize, b: usize| {
        let r = dist_g(&pos[a], &pos[b]);
        r * r
    };
    let radius =
        |a: usize, b: usize| 4.0 / 3.0 * d.r0ab[numbers[a] as usize - 1][numbers[b] as usize - 1];
    let mut energy = S::from_f64(0.0);
    for a in 0..n {
        for b in a + 1..n {
            for c in b + 1..n {
                if !(valid(numbers[a]) && valid(numbers[b]) && valid(numbers[c])) {
                    continue;
                }
                let product = getc6_g(d, numbers[a], numbers[b], cn[a], cn[b])
                    * getc6_g(d, numbers[a], numbers[c], cn[a], cn[c])
                    * getc6_g(d, numbers[b], numbers[c], cn[b], cn[c]);
                if !(product.value() > 0.0) {
                    continue;
                }
                energy = energy
                    + atm_term_g(
                        squared(a, b),
                        squared(a, c),
                        squared(b, c),
                        product.sqrt(),
                        radius(a, b) * radius(a, c) * radius(b, c),
                        16.0,
                    );
            }
        }
    }
    energy
}

/// The ωB97X-D nonlinear damping strength.
pub const CHG_A_WB97XD: f64 = 6.0;

/// `C6` (Hartree Bohr^6) and the sum of the van der Waals radii `R_r` (Bohr)
/// of an atom pair for the CHG-damped term, on the D2 tables; `None` for an
/// element outside them.
pub fn chg_pair_parameters(zi: u8, zj: u8) -> Option<(f64, f64)> {
    d2_pair_parameters(zi, zj)
}

pub(crate) fn d2_pair_parameters(zi: u8, zj: u8) -> Option<(f64, f64)> {
    let (zi, zj) = (zi as usize, zj as usize);
    if zi == 0 || zj == 0 || zi >= D2_C6_JNM6.len() || zj >= D2_C6_JNM6.len() {
        return None;
    }
    let ang6 = crate::constants::ANGSTROM_TO_BOHR.powi(6);
    let jnm6_to_au = 1.0e6 * ang6 / (crate::constants::HARTREE_TO_KJ_MOL * 1000.0);
    Some((
        (D2_C6_JNM6[zi] * D2_C6_JNM6[zj]).sqrt() * jnm6_to_au,
        (D2_R0_ANG[zi] + D2_R0_ANG[zj]) * crate::constants::ANGSTROM_TO_BOHR,
    ))
}

// ---------------------------------------------------------------------------
// Public entry points: energy + analytic gradient (dual-seeded).
// ---------------------------------------------------------------------------

fn geometry(mol: &Molecule) -> (Vec<u8>, Vec<[f64; 3]>) {
    (
        mol.atoms.iter().map(|a| a.z).collect(),
        mol.atoms
            .iter()
            .map(|a| [a.position.x, a.position.y, a.position.z])
            .collect(),
    )
}

/// Dispersion energy (Hartree) for the resolved functional name.
pub fn dispersion_energy(mol: &Molecule, kind: &Dispersion, functional: &str) -> Result<f64> {
    let (numbers, pos) = geometry(mol);
    dispersion_energy_g(&numbers, &pos, kind, functional)
}

/// The same resolved dispersion energy over an analytic scalar geometry.
pub(crate) fn dispersion_energy_g<S: Scalar>(
    numbers: &[u8],
    pos: &[[S; 3]],
    kind: &Dispersion,
    functional: &str,
) -> Result<S> {
    PairKernel::resolve(kind, functional)?.energy(numbers, pos)
}

/// Bind fitted parameters once when an analytic derivative evaluates
/// several differently seeded copies of the same molecular geometry.
enum PairKernel<'a> {
    None,
    D1(&'a [d1::AtomParameters], f64),
    D2(f64),
    D3(Option<D3ZeroParams>, Option<D3BjParams>, bool),
    Chg,
}

impl<'a> PairKernel<'a> {
    fn resolve(kind: &'a Dispersion, functional: &str) -> Result<Self> {
        Ok(match kind {
            Dispersion::None => Self::None,
            Dispersion::D1 { atoms, s6 } => Self::D1(atoms, *s6),
            Dispersion::D2 => Self::D2(d2_s6(functional)?),
            Dispersion::D3Zero | Dispersion::D3ZeroAbc => Self::D3(
                Some(d3_zero_params(functional)?),
                None,
                *kind == Dispersion::D3ZeroAbc,
            ),
            Dispersion::D3Bj | Dispersion::D3BjAbc => Self::D3(
                None,
                Some(d3_bj_params(functional)?),
                *kind == Dispersion::D3BjAbc,
            ),
            Dispersion::Chg => Self::Chg,
        })
    }

    fn energy<S: Scalar>(&self, numbers: &[u8], positions: &[[S; 3]]) -> Result<S> {
        Ok(match self {
            Self::None => S::from_f64(0.0),
            Self::D1(atoms, scale) => return d1::energy_geometry(positions, atoms, *scale),
            Self::D2(s6) => d2_energy_g(numbers, positions, *s6),
            Self::D3(zero, bj, three_body) => {
                let pair = d3_energy_g(numbers, positions, zero.as_ref(), bj.as_ref());
                if *three_body {
                    pair + d3_three_body_g(numbers, positions)
                } else {
                    pair
                }
            }
            Self::Chg => chg_energy_g(numbers, positions, CHG_A_WB97XD),
        })
    }
}

/// Analytic dispersion gradient via per-atom dual seeding (the CN coupling
/// differentiates exactly).
pub fn dispersion_gradient(
    mol: &Molecule,
    kind: &Dispersion,
    functional: &str,
) -> Result<Vec<Vec3>> {
    if *kind == Dispersion::None {
        return Ok(vec![Vec3::zero(); mol.len()]);
    }
    let (numbers, pos_f) = geometry(mol);
    type D3v = DualN<f64, 3>;
    let mut grad = vec![Vec3::zero(); mol.len()];
    let kernel = PairKernel::resolve(kind, functional)?;
    for a in 0..mol.len() {
        let pos: Vec<[D3v; 3]> = pos_f
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if i == a {
                    [
                        D3v::variable(p[0], 0),
                        D3v::variable(p[1], 1),
                        D3v::variable(p[2], 2),
                    ]
                } else {
                    [
                        D3v::constant(p[0]),
                        D3v::constant(p[1]),
                        D3v::constant(p[2]),
                    ]
                }
            })
            .collect();
        let e = kernel.energy(&numbers, &pos)?;
        grad[a] = Vec3::new(e.im[0], e.im[1], e.im[2]);
    }
    Ok(grad)
}

/// Analytic dispersion Hessian (Hartree/Bohr^2), `3N x 3N` Cartesian, via
/// nested dual seeding: one level deeper than `dispersion_gradient`.
///
/// Seeding shape: for every unordered atom pair `(a, b)` (including `a ==
/// b`), the SAME `DualN<f64, 3>` type `dispersion_gradient` already uses per
/// atom (`D3v` there, `Inner` here) seeds atom `a`'s three Cartesian
/// coordinates in the OUTER dual and atom `b`'s three coordinates in the
/// nested INNER dual, so a single energy-kernel evaluation returns the whole
/// `3x3` block `d^2E / dR_a dR_b` at once:
///
/// - atom `a` only (`a != b`): outer-seeded, inner-constant
///   (`Outer::variable(Inner::constant(x), c)`) — first-order sensitivity to
///   `a`, no dependence on `b`'s coordinates.
/// - atom `b` only: inner-seeded, outer-constant
///   (`Outer::constant(Inner::variable(x, c))`) — the mirror image.
/// - `a == b` (the diagonal block): BOTH levels seeded at the matching
///   index (`Outer::variable(Inner::variable(x, c), c)`) — exactly the
///   nested pattern `dual.rs`'s `second_derivatives_match_fd_of_gradient`
///   test validates for a full multivariable second-order dual (there for
///   two independent scalars `x`, `y`; here for the atom's own `x, y, z`).
/// - every other atom: fully constant at both levels.
///
/// This was chosen over "one coordinate in the outer dual, all `3N` in the
/// inner" because the inner dual's width is a compile-time const generic
/// (`DualN<S, N>`) and `N` (atom count) is only known at runtime; reusing the
/// existing per-atom `N = 3` width — looped over atom PAIRS instead of
/// `dispersion_gradient`'s loop over single atoms — mirrors the existing
/// code almost exactly and needs no new const-generic plumbing. Cost is
/// `O(atoms^2)` kernel evaluations (each itself `O(atoms^2)` for the
/// pairwise sum), which is negligible next to the ab initio Hessian this
/// feeds into.
pub fn dispersion_hessian(mol: &Molecule, kind: &Dispersion, functional: &str) -> Result<Matrix> {
    let n = mol.len();
    if *kind == Dispersion::None {
        return Ok(Matrix::zeros(3 * n, 3 * n));
    }
    let (numbers, pos_f) = geometry(mol);
    type Inner = DualN<f64, 3>;
    type Outer = DualN<Inner, 3>;
    let kernel = PairKernel::resolve(kind, functional)?;
    let mut h = Matrix::zeros(3 * n, 3 * n);
    for a in 0..n {
        for b in a..n {
            let pos: Vec<[Outer; 3]> = pos_f
                .iter()
                .enumerate()
                .map(|(k, p)| {
                    if k == a && k == b {
                        [
                            Outer::variable(Inner::variable(p[0], 0), 0),
                            Outer::variable(Inner::variable(p[1], 1), 1),
                            Outer::variable(Inner::variable(p[2], 2), 2),
                        ]
                    } else if k == a {
                        [
                            Outer::variable(Inner::constant(p[0]), 0),
                            Outer::variable(Inner::constant(p[1]), 1),
                            Outer::variable(Inner::constant(p[2]), 2),
                        ]
                    } else if k == b {
                        [
                            Outer::constant(Inner::variable(p[0], 0)),
                            Outer::constant(Inner::variable(p[1], 1)),
                            Outer::constant(Inner::variable(p[2], 2)),
                        ]
                    } else {
                        [
                            Outer::constant(Inner::constant(p[0])),
                            Outer::constant(Inner::constant(p[1])),
                            Outer::constant(Inner::constant(p[2])),
                        ]
                    }
                })
                .collect();
            let e = kernel.energy(&numbers, &pos)?;
            for oc in 0..3 {
                for ic in 0..3 {
                    // e.im[oc].im[ic] = d^2E / d(atom a, coord oc) d(atom b, coord ic).
                    let val = e.im[oc].im[ic];
                    h[(3 * a + oc, 3 * b + ic)] = val;
                    if a != b {
                        // Clairaut symmetry: H[3b+ic, 3a+oc] is the same
                        // mixed partial taken in the other order.
                        h[(3 * b + ic, 3 * a + oc)] = val;
                    }
                }
            }
        }
    }
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::Atom;

    fn water_dimer() -> Molecule {
        Molecule::from_xyz_str(
            "6\nwd\nO -1.551007 -0.114520 0.0\nH -1.934259 0.762503 0.0\nH -0.599677 \
                0.040712 0.0\nO 1.350625 0.111469 0.0\nH 1.680398 -0.373741 \
                -0.758561\nH 1.680398 -0.373741 0.758561\n",
        )
        .unwrap()
    }

    /// C6 of a homonuclear pair is symmetric in the two coordination numbers:
    /// the energy must not depend on the order of the atoms.
    #[test]
    fn homonuclear_c6_is_symmetric_and_the_energy_is_order_independent() {
        let d = data();
        for z in [1u8, 6, 8] {
            for (x, y) in [(0.3, 1.7), (1.0, 0.1), (2.2, 0.9)] {
                let (a, b) = (getc6_g::<f64>(d, z, z, x, y), getc6_g::<f64>(d, z, z, y, x));
                assert!(
                    (a - b).abs() < 1e-12 * a.abs(),
                    "Z {z}: C6({x},{y}) = {a} vs C6({y},{x}) = {b}"
                );
            }
        }
        let mol = water_dimer();
        let mut reversed = mol.clone();
        reversed.atoms.reverse();
        for kind in [
            Dispersion::D3Zero,
            Dispersion::D3Bj,
            Dispersion::D3ZeroAbc,
            Dispersion::D3BjAbc,
        ] {
            let (a, b) = (
                dispersion_energy(&mol, &kind, "b3lyp").unwrap(),
                dispersion_energy(&reversed, &kind, "b3lyp").unwrap(),
            );
            assert!((a - b).abs() < 1e-13, "{kind:?}: {a} vs {b}");
        }
    }

    #[test]
    fn d3_data_and_cn_interpolation() {
        let d = data();
        assert!(d.c6.contains_key(&(6, 6)));
        assert!(d.c6[&(8, 8)].len() >= 5);
        let h_iso = getc6_g::<f64>(d, 1, 1, 0.0, 0.0);
        let h_bond = getc6_g::<f64>(d, 1, 1, 1.0, 1.0);
        assert!(h_iso > 7.0 && h_iso < 7.6 && h_iso > h_bond);
        let (numbers, pos) = geometry(&water_dimer());
        let cn = coordination_numbers_g::<f64>(&numbers, &pos, d);
        assert!((cn[0] - 2.0).abs() < 0.1);
        assert!((cn[1] - 1.0).abs() < 0.1);
    }

    /// Published D3 and D2 damping tables are public data, but pinning every
    /// newly exposed row here prevents aliases or tuple ordering from silently
    /// swapping `s8` and a damping length.
    #[test]
    fn expanded_published_damping_parameters_are_exact() {
        let zero = [
            ("revpbe", 0.923, 1.010),
            ("tpss", 1.166, 1.105),
            ("bp86", 1.139, 1.683),
            ("m06-2x", 1.619, 0.0),
            ("b3pw91", 1.176, 1.775),
            ("hse06", 1.129, 0.109),
            ("lc-wpbe", 1.355, 1.279),
            ("o3lyp", 1.4060, 1.8058),
        ];
        for (name, rs6, s8) in zero {
            let p = d3_zero_params(name).unwrap();
            assert_eq!((p.rs6, p.s8), (rs6, s8), "D3(0) {name}");
        }

        let bj = [
            ("revpbe", 0.5238, 2.3550, 3.5016),
            ("tpss", 0.4535, 1.9435, 4.4752),
            ("bp86", 0.3946, 3.2822, 4.8516),
            ("r2scan", 0.49484001, 0.78981345, 5.73083694),
            ("b3pw91", 0.4312, 2.8524, 4.4693),
            ("hse06", 0.3830, 2.3100, 5.6850),
            ("lc-wpbe", 0.3919, 1.8541, 5.0897),
            ("o3lyp", 0.0963, 1.8171, 5.9940),
        ];
        for (name, a1, s8, a2) in bj {
            let p = d3_bj_params(name).unwrap();
            assert_eq!((p.a1, p.s8, p.a2), (a1, s8, a2), "D3(BJ) {name}");
        }

        assert_eq!(d2_s6("bp86").unwrap(), 1.05);
        assert_eq!(d2_s6("tpss").unwrap(), 1.0);
        assert!(d3_bj_params("m06-2x").unwrap_err().is_unsupported());
        assert!(d3_zero_params("r2scan").unwrap_err().is_unsupported());
    }

    #[test]
    fn dispersion_energies_negative_and_sane() {
        let mol = water_dimer();
        for kind in [Dispersion::D2, Dispersion::D3Zero, Dispersion::D3Bj] {
            let e = dispersion_energy(&mol, &kind, "b3lyp").unwrap();
            // Water dimer dispersion: a few tenths of kcal/mol up to ~2.
            assert!(
                e < 0.0 && e > -0.01,
                "{kind:?}: E_disp = {e} Ha out of range"
            );
        }
        assert!(dispersion_energy(&mol, &Dispersion::None, "b3lyp").unwrap() == 0.0);
    }

    /// CHG damping (ωB97X-D tail): closed-form two-atom pin at R = Rr
    /// (f = 1/(1+a)) and the undamped −C6/R⁶ long-range limit.
    #[test]
    fn chg_damping_closed_form_and_asymptote() {
        let ang6 = crate::constants::ANGSTROM_TO_BOHR.powi(6);
        let jnm6_to_au = 1.0e6 * ang6 / (crate::constants::HARTREE_TO_KJ_MOL * 1000.0);
        let c6_hh = 0.14 * jnm6_to_au; // H–H, D2 table
        let rr_hh = 2.0 * 1.001 * crate::constants::ANGSTROM_TO_BOHR;
        // At exactly R = Rr the damping factor is 1/(1 + a).
        let numbers = [1u8, 1u8];
        let pos = [[0.0, 0.0, 0.0], [0.0, 0.0, rr_hh]];
        let e = chg_energy_g(&numbers, &pos, CHG_A_WB97XD);
        let want = -c6_hh / rr_hh.powi(6) / (1.0 + CHG_A_WB97XD);
        assert!((e - want).abs() < 1e-15, "{e} vs closed form {want}");
        // At 20 Bohr the damping is ≈ 1: E → −C6/R⁶.
        let pos_far = [[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]];
        let e_far = chg_energy_g(&numbers, &pos_far, CHG_A_WB97XD);
        let bare = -c6_hh / 20.0f64.powi(6);
        assert!(
            (e_far / bare - 1.0).abs() < 1e-6,
            "long-range ratio {} ≠ 1",
            e_far / bare
        );
    }

    /// At non-overlapping separation every supported atom-pair correction
    /// must approach the London -C6/R^6 tail. Damping and the D3 C8 term may
    /// change short range, but cannot change this leading power law.
    #[test]
    fn empirical_dispersion_has_the_london_r6_asymptote() {
        let make = |r: f64| {
            Molecule::new(vec![
                Atom {
                    z: 1,
                    position: Vec3::zero(),
                },
                Atom {
                    z: 1,
                    position: Vec3::new(0.0, 0.0, r),
                },
            ])
        };
        let (r1, r2) = (40.0_f64, 60.0_f64);
        for kind in [
            Dispersion::D2,
            Dispersion::D3Zero,
            Dispersion::D3Bj,
            Dispersion::Chg,
        ] {
            let c6_1 = -dispersion_energy(&make(r1), &kind, "b3lyp").unwrap() * r1.powi(6);
            let c6_2 = -dispersion_energy(&make(r2), &kind, "b3lyp").unwrap() * r2.powi(6);
            assert!(
                c6_1 > 0.0 && c6_2 > 0.0,
                "{kind:?} lost its attractive tail"
            );
            assert!(
                (c6_1 / c6_2 - 1.0).abs() < 0.03,
                "{kind:?} does not approach -C6/R^6: {c6_1} vs {c6_2}"
            );
        }
    }

    /// Direct unit test of `getc6_g`'s CN-interpolation weighting, isolated
    /// from the geometry/CN machinery. ΔCN ≈ 16 from the closest C-C
    /// reference point (cn = 3.9844) makes every Gaussian weight
    /// `exp(-4·ΔCN²)` underflow to exactly 0.0 in f64 (the closest term
    /// alone needs exp(-4·2·16.0156²) ≈ exp(-2052), far past the smallest
    /// subnormal magnitude, exp(x) underflows around x ≈ -745). The naive
    /// weighted average num/den is then 0/0.
    #[test]
    fn getc6_g_underflow_does_not_collapse_to_zero() {
        let d = data();
        let c6 = getc6_g::<f64>(d, 6, 6, 20.0, 20.0);
        assert!(c6.is_finite(), "C6 is not finite: {c6}");
        assert!(c6 > 0.0, "C6 collapsed to zero under CN underflow: {c6}");
    }

    /// As ΔCN runs away, the interpolated C6 must smoothly asymptote to the
    /// *nearest* tabulated reference point's C6 (a normalized Gaussian
    /// average has to land somewhere on the convex hull of its inputs), not
    /// fall to zero. For the C-C table the point with the largest cna+cnb
    /// sum — (cna, cnb) = (3.9844, 3.9844), C6 = 18.2067 — is the one
    /// closest to (T, T) as T → ∞, so that's the value to converge to.
    #[test]
    fn getc6_g_asymptotes_to_the_nearest_reference_point() {
        let d = data();
        let c6 = getc6_g::<f64>(d, 6, 6, 50.0, 50.0);
        assert!(
            (c6 - 18.2067).abs() < 1e-6,
            "C6 = {c6} did not asymptote to the nearest reference C6 (18.2067)"
        );
    }

    /// A genuinely-missing (Z_i, Z_j) key must stay distinct from the
    /// CN-underflow case above: it is still exactly 0, via the `None` arm,
    /// not via the Gaussian-weight normalization.
    #[test]
    fn getc6_g_missing_key_is_still_zero() {
        let d = data();
        let c6 = getc6_g::<f64>(d, 0, 6, 1.0, 1.0);
        assert_eq!(c6, 0.0);
    }

    /// A pathological geometry step (e.g. an optimizer overshoot) can crush
    /// a whole cluster of atoms almost on top of one another for one
    /// iteration. D3's CN is a sum of per-neighbor sigmoids that saturate
    /// at 1 for very short distances, so a compact N-atom cluster drives
    /// every atom's CN up to roughly N-1 — comfortably past the threshold
    /// where `getc6_g`'s weights fully underflow. Before the fix this made
    /// C6 = 0/0 = 0, and then the BJ radius R0 = √(C8/C6) = √(0/0) = NaN,
    /// silently NaN-poisoning the whole dispersion energy and gradient.
    fn overlapping_carbon_cluster() -> Molecule {
        let mut atoms = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..2 {
                    atoms.push(Atom {
                        z: 6,
                        position: Vec3::new(i as f64 * 0.05, j as f64 * 0.05, k as f64 * 0.05),
                    });
                }
            }
        }
        Molecule::new(atoms)
    }

    #[test]
    fn d3bj_energy_and_gradient_stay_finite_under_extreme_coordination_number() {
        let mol = overlapping_carbon_cluster();
        // Sanity check that this geometry really does reach the pathological
        // CN regime the fix targets (ΔCN from the closest C-C reference,
        // 3.9844, comfortably above the ~9.64 needed to fully underflow).
        let (numbers, pos) = geometry(&mol);
        let cn = coordination_numbers_g::<f64>(&numbers, &pos, data());
        assert!(cn[0] > 13.0, "test geometry only reaches CN = {}", cn[0]);

        let e = dispersion_energy(&mol, &Dispersion::D3Bj, "pbe").unwrap();
        assert!(
            e.is_finite(),
            "D3Bj energy is not finite (got {e}) under extreme CN"
        );

        let g = dispersion_gradient(&mol, &Dispersion::D3Bj, "pbe").unwrap();
        for (a, ga) in g.iter().enumerate() {
            assert!(
                ga.x.is_finite() && ga.y.is_finite() && ga.z.is_finite(),
                "D3Bj gradient on atom {a} is not finite: {ga:?}"
            );
        }
    }

    /// Analytic dispersion gradients vs FD (all three models; includes the
    /// CN-coupling terms for D3).
    #[test]
    fn dispersion_gradients_match_fd() {
        let mol = water_dimer();
        let h = 1e-6;
        for kind in [
            Dispersion::D2,
            Dispersion::D3Zero,
            Dispersion::D3Bj,
            Dispersion::Chg,
            Dispersion::D3ZeroAbc,
            Dispersion::D3BjAbc,
        ] {
            let g = dispersion_gradient(&mol, &kind, "pbe").unwrap();
            // Sum rule.
            let s: Vec3 = g.iter().fold(Vec3::zero(), |a, &v| a + v);
            assert!(s.norm() < 1e-12, "{kind:?} sum rule {}", s.norm());
            for a in 0..mol.len() {
                for ax in 0..3 {
                    let mut mp = mol.clone();
                    let mut mm = mol.clone();
                    let bump = |at: &mut Atom, sgn: f64| match ax {
                        0 => at.position.x += sgn * h,
                        1 => at.position.y += sgn * h,
                        _ => at.position.z += sgn * h,
                    };
                    bump(&mut mp.atoms[a], 1.0);
                    bump(&mut mm.atoms[a], -1.0);
                    let fd = (dispersion_energy(&mp, &kind, "pbe").unwrap()
                        - dispersion_energy(&mm, &kind, "pbe").unwrap())
                        / (2.0 * h);
                    let an = g[a].get(ax);
                    assert!(
                        (an - fd).abs() < 1e-9,
                        "{kind:?} atom {a} ax {ax}: {an} vs {fd}"
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // dispersion_hessian gates.
    // -----------------------------------------------------------------

    /// Staggered ethane (idealized tetrahedral geometry, C-C = 1.54 Ang,
    /// C-H = 1.09 Ang). Deliberately NOT a dispersion-bound dimer: every
    /// atom sits in a real covalent bonding network, so both carbons and
    /// all six hydrogens carry non-trivial (non-reference, coupled)
    /// coordination numbers and the CN interpolation in `getc6_g` is
    /// actually exercised across atom pairs, unlike a weakly-bound dimer
    /// where cross-monomer CN contributions are negligible.
    fn ethane() -> Molecule {
        let d_cc = 1.54_f64;
        let d_ch = 1.09_f64;
        let cos_t = -1.0 / 3.0; // ideal tetrahedral angle, cos(109.47 deg)
        let sin_t = (1.0 - cos_t * cos_t).sqrt();
        let h_on = |c: Vec3, axis: f64, phis_deg: [f64; 3]| -> Vec<Atom> {
            phis_deg
                .iter()
                .map(|&phi_deg| {
                    let phi = phi_deg.to_radians();
                    Atom {
                        z: 1,
                        position: Vec3::new(
                            c.x + d_ch * sin_t * phi.cos(),
                            c.y + d_ch * sin_t * phi.sin(),
                            c.z + d_ch * cos_t * axis,
                        ),
                    }
                })
                .collect()
        };
        let c1 = Vec3::new(0.0, 0.0, 0.0);
        let c2 = Vec3::new(0.0, 0.0, d_cc);
        let mut atoms = vec![Atom { z: 6, position: c1 }, Atom { z: 6, position: c2 }];
        atoms.extend(h_on(c1, 1.0, [0.0, 120.0, 240.0]));
        atoms.extend(h_on(c2, -1.0, [60.0, 180.0, 300.0]));
        Molecule::new(atoms)
    }

    const DISP_KINDS: [Dispersion; 6] = [
        Dispersion::D2,
        Dispersion::D3Zero,
        Dispersion::D3Bj,
        Dispersion::Chg,
        Dispersion::D3ZeroAbc,
        Dispersion::D3BjAbc,
    ];

    /// Sanity check on the `ethane()` fixture: coordination numbers must be
    /// away from the trivial isolated-atom (CN = 0) and dimer-like
    /// near-integer cases the task explicitly warns against, confirming the
    /// FD/symmetry/translation gates below actually exercise the CN
    /// coupling.
    #[test]
    fn ethane_fixture_has_nontrivial_coordination_numbers() {
        let mol = ethane();
        let (numbers, pos) = geometry(&mol);
        let cn = coordination_numbers_g::<f64>(&numbers, &pos, data());
        // Carbons: 1 C + 3 H neighbors each, plus a smaller tail from the
        // farther atoms (the DFTD3 CN sigmoid decays smoothly, not sharply).
        assert!(cn[0] > 3.5 && cn[1] > 3.5, "carbon CN too low: {cn:?}");
        // Hydrogens: bonded to one carbon, with a real (non-negligible)
        // additional contribution from the other two hydrogens on the same
        // carbon and a smaller tail from the rest of the molecule -- this
        // is exactly the multi-neighbor coupling the FD/symmetry/
        // translation gates below need, so it must be well above the
        // single-neighbor value of 1 and comfortably below the carbons'.
        for &c in &cn[2..8] {
            assert!(c > 1.5 && c < 3.5, "hydrogen CN out of range: {c}");
        }
    }

    /// Analytic dispersion Hessian vs central differences of the analytic
    /// (dual-seeded, exact) gradient, for all four models, on a small
    /// hydrocarbon so the CN-coupling second derivative is actually
    /// exercised.
    #[test]
    fn dispersion_hessian_matches_fd_of_gradient() {
        let mol = ethane();
        let h = 1e-5;
        for kind in DISP_KINDS {
            let hess = dispersion_hessian(&mol, &kind, "pbe").unwrap();
            for b in 0..mol.len() {
                for bx in 0..3 {
                    let mut mp = mol.clone();
                    let mut mm = mol.clone();
                    let bump = |at: &mut Atom, sgn: f64| match bx {
                        0 => at.position.x += sgn * h,
                        1 => at.position.y += sgn * h,
                        _ => at.position.z += sgn * h,
                    };
                    bump(&mut mp.atoms[b], 1.0);
                    bump(&mut mm.atoms[b], -1.0);
                    let gp = dispersion_gradient(&mp, &kind, "pbe").unwrap();
                    let gm = dispersion_gradient(&mm, &kind, "pbe").unwrap();
                    for a in 0..mol.len() {
                        for ax in 0..3 {
                            let fd = (gp[a].get(ax) - gm[a].get(ax)) / (2.0 * h);
                            let an = hess[(3 * a + ax, 3 * b + bx)];
                            assert!(
                                (an - fd).abs() < 1e-6,
                                "{kind:?} H[atom {a} ax {ax}][atom {b} ax {bx}]: {an} vs FD {fd}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// H must be exactly symmetric (well, to nested-dual round-off — see
    /// `dual.rs`'s own mixed-partial-symmetry test for the same tolerance
    /// class) for every model.
    #[test]
    fn dispersion_hessian_is_symmetric() {
        let mol = ethane();
        let dim = 3 * mol.len();
        for kind in DISP_KINDS {
            let hess = dispersion_hessian(&mol, &kind, "pbe").unwrap();
            for i in 0..dim {
                for j in 0..dim {
                    assert!(
                        (hess[(i, j)] - hess[(j, i)]).abs() < 1e-10,
                        "{kind:?} H[{i},{j}] = {} vs H[{j},{i}] = {}",
                        hess[(i, j)],
                        hess[(j, i)]
                    );
                }
            }
        }
    }

    /// Translational invariance: since the energy depends only on relative
    /// coordinates, `sum_B dE/dR_B = 0` identically as a function of
    /// geometry, so its derivative w.r.t. R_A is exactly zero too — every
    /// `3x3` block ROW of H must sum to zero. Exact for a pairwise term (no
    /// SCF/quadrature noise anywhere in this path), so this gets a tight
    /// tolerance.
    #[test]
    fn dispersion_hessian_translational_invariance() {
        let mol = ethane();
        let n = mol.len();
        for kind in DISP_KINDS {
            let hess = dispersion_hessian(&mol, &kind, "pbe").unwrap();
            for a in 0..n {
                for i in 0..3 {
                    for j in 0..3 {
                        let s: f64 = (0..n).map(|b| hess[(3 * a + i, 3 * b + j)]).sum();
                        assert!(
                            s.abs() < 1e-9,
                            "{kind:?} atom {a} block row ({i},{j}) sums to {s:.3e}, not 0"
                        );
                    }
                }
            }
        }
    }

    /// `Dispersion::None` returns an exact zero matrix, mirroring
    /// `dispersion_gradient`'s zero-vector behaviour.
    #[test]
    fn dispersion_hessian_none_is_zero() {
        let mol = ethane();
        let h = dispersion_hessian(&mol, &Dispersion::None, "pbe").unwrap();
        let dim = 3 * mol.len();
        assert_eq!((h.rows, h.cols), (dim, dim));
        for v in h.as_slice() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn three_body_energy_matches_the_closed_form_of_a_triangle() {
        let d = data();
        let raw: [[f64; 3]; 3] = [[0.0, 0.0, 0.0], [7.0, 0.5, 0.0], [3.0, 6.5, 1.0]];
        let numbers = [6u8, 7, 8];
        let mol = Molecule::new(
            numbers
                .iter()
                .zip(&raw)
                .map(|(z, p)| Atom {
                    z: *z,
                    position: Vec3::new(p[0], p[1], p[2]),
                })
                .collect(),
        );
        let found = dispersion_energy(&mol, &Dispersion::D3BjAbc, "pbe").unwrap()
            - dispersion_energy(&mol, &Dispersion::D3Bj, "pbe").unwrap();
        // Independent evaluation with explicit angles.
        let cn = coordination_numbers_g(&numbers, &raw, d);
        let c6 = |a: usize, b: usize| getc6_g(d, numbers[a], numbers[b], cn[a], cn[b]);
        let c9 = (c6(0, 1) * c6(0, 2) * c6(1, 2)).sqrt();
        let p: Vec<Vec3> = raw.iter().map(|r| Vec3::new(r[0], r[1], r[2])).collect();
        let side = |a: usize, b: usize| (p[b] - p[a]).norm();
        let cosine = |v: usize, q: usize, r: usize| {
            (p[q] - p[v]).dot(p[r] - p[v]) / (side(v, q) * side(v, r))
        };
        let angular = 3.0 * cosine(0, 1, 2) * cosine(1, 0, 2) * cosine(2, 0, 1) + 1.0;
        let product = side(0, 1) * side(0, 2) * side(1, 2);
        let radius = |a: usize, b: usize| {
            4.0 / 3.0 * d.r0ab[numbers[a] as usize - 1][numbers[b] as usize - 1]
        };
        let ratio = radius(0, 1) * radius(0, 2) * radius(1, 2) / product;
        let expected = c9 * angular / product.powi(3) / (1.0 + 6.0 * ratio.powf(16.0 / 3.0));
        println!("three-body energy {found:.12e}, closed form {expected:.12e}");
        assert!((found - expected).abs() < 1e-12 * expected.abs());
    }

    /// Log-log least squares `log|e| = log C + p log h`: exponent, R^2.
    fn power_fit(h: &[f64], errors: &[f64]) -> (f64, f64) {
        let x: Vec<f64> = h.iter().map(|v| v.ln()).collect();
        let y: Vec<f64> = errors.iter().map(|e| e.abs().ln()).collect();
        let n = x.len() as f64;
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
        let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
        let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
        (sxy / sxx, sxy * sxy / (sxx * syy))
    }

    /// Four-step error fits of the three-body gradient and Hessian: the
    /// analytic derivatives of the three-body part alone (model with the
    /// term minus the pair model) against central differences.
    #[test]
    fn three_body_derivatives_pass_a_four_step_fit() {
        let mol = ethane();
        let steps: [f64; 4] = [0.008, 0.004, 0.002, 0.001];
        let shifted = |atom: usize, axis: usize, shift: f64| {
            let mut m = mol.clone();
            match axis {
                0 => m.atoms[atom].position.x += shift,
                1 => m.atoms[atom].position.y += shift,
                _ => m.atoms[atom].position.z += shift,
            }
            m
        };
        for (with, without) in [
            (Dispersion::D3ZeroAbc, Dispersion::D3Zero),
            (Dispersion::D3BjAbc, Dispersion::D3Bj),
        ] {
            // The three-body energy itself: a difference of two total
            // energies would lose it to rounding.
            let energy = |m: &Molecule| {
                let (numbers, pos) = geometry(m);
                d3_three_body_g::<f64>(&numbers, &pos)
            };
            let gradient = |m: &Molecule| -> Vec<f64> {
                let a = dispersion_gradient(m, &with, "pbe").unwrap();
                let b = dispersion_gradient(m, &without, "pbe").unwrap();
                (0..3 * m.len())
                    .map(|i| a[i / 3].get(i % 3) - b[i / 3].get(i % 3))
                    .collect()
            };
            let g0 = gradient(&mol);
            let h_with = dispersion_hessian(&mol, &with, "pbe").unwrap();
            let h_without = dispersion_hessian(&mol, &without, "pbe").unwrap();
            for (atom, axis) in [(0usize, 2usize), (3, 0), (5, 1)] {
                let column = 3 * atom + axis;
                let errors: Vec<f64> = steps
                    .iter()
                    .map(|&h| {
                        (energy(&shifted(atom, axis, h)) - energy(&shifted(atom, axis, -h)))
                            / (2.0 * h)
                            - g0[column]
                    })
                    .collect();
                let (p, r2) = power_fit(&steps, &errors);
                println!(
                    "{with:?} three-body gradient atom {atom} axis {axis}: analytic {:.10e}, errors {errors:?}, h 0.001..0.008, p {p:.6}, R^2 {r2:.8}",
                    g0[column]
                );
                assert!((p - 2.0).abs() < 0.02 && r2 > 0.9999);
                // The largest element of the Hessian column.
                let analytic: Vec<f64> = (0..3 * mol.len())
                    .map(|i| h_with[(i, column)] - h_without[(i, column)])
                    .collect();
                let row = (0..analytic.len())
                    .max_by(|a, b| analytic[*a].abs().total_cmp(&analytic[*b].abs()))
                    .unwrap();
                let errors: Vec<f64> = steps
                    .iter()
                    .map(|&h| {
                        (gradient(&shifted(atom, axis, h))[row]
                            - gradient(&shifted(atom, axis, -h))[row])
                            / (2.0 * h)
                            - analytic[row]
                    })
                    .collect();
                let (p, r2) = power_fit(&steps, &errors);
                println!(
                    "{with:?} three-body Hessian element ({row}, {column}): analytic {:.10e}, errors {errors:?}, h 0.001..0.008, p {p:.6}, R^2 {r2:.8}",
                    analytic[row]
                );
                assert!((p - 2.0).abs() < 0.02 && r2 > 0.9999);
            }
        }
    }
}
