// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Semi-local effective-core-potential integrals for Gaussian94 ECP blocks
//! `U(r) = U_L(r) + sum_{l<L} sum_m |lm> (U_l - U_L) <lm|`, with radial
//! terms `c * r^{n-2} * exp(-eta r^2)` (`EcpTerm.power` stores n).
//!
//! Method (McMurchie-Davidson 1981 structure, semi-numerical radial part):
//! about the ECP center C every Cartesian AO factors into a polynomial in
//! s = r - C (binomial shift), a radial Gaussian, and the plane-wave-like
//! factor exp(2 alpha s.d) with d = A - C, expanded as
//! `4 pi sum_{lam,mu} i_lam(2 alpha d r) Y_lam,mu(s_hat) Y_lam,mu(d_hat)`
//! (modified spherical Bessel i_lam). Angular factors reduce to exact
//! closed-form monomial integrals over the unit sphere; radial integrals
//! are evaluated by Gauss-Legendre quadrature on the peak interval with
//! exp(-x)-scaled Bessel functions (total exponents stay non-positive, so
//! nothing overflows).
//!
//! - Type-1 (local channel): one joint expansion around
//!   P = alpha d_A + beta d_B; the r^2 volume element combines with the
//!   potential to r^n (n = `power`).
//! - Type-2 (projected channels): independent bra/ket expansions coupled
//!   by the radial integral over U_l - U_L.
//!
//! Projector convention: sum_m |lm><lm| over an orthonormal real spherical
//! harmonic basis (any orthogonal choice of the m functions is equivalent;
//! we reuse the solid-harmonic Cartesian expansion, L2-normalized on the
//! sphere via exact monomial integrals).

use crate::basis::aos::{solid_harmonic_coef, sph_block};
use crate::basis::{cart_components, double_factorial, EcpPotential, MolBasis, MolShell};
use crate::error::Result;
use crate::grid::gauss_legendre;
use crate::linalg::Matrix;
use crate::math::Vec3;
use rayon::prelude::*;
use std::sync::OnceLock;

mod taylor;
pub use taylor::ecp_nuclear_taylor;
pub(crate) use taylor::{check_ecp_second_order, ecp_pair_path};

/// Capacity for Bessel/harmonic orders: lam <= l_AO,max + l_proj,max + poly.
const LAMBDA_CAP: usize = 16;
/// Radial quadrature order (Gauss-Legendre on the peak interval).
const NRAD: usize = 100;
/// Primitive-pair screening threshold on the peak exponent.
const EXP_CUTOFF: f64 = -46.0;

// ---------------------------------------------------------------------------
// Exact angular algebra
// ---------------------------------------------------------------------------

/// Unit-sphere monomial integral (closed form; zero unless all even):
/// `int Ox^a Oy^b Oz^c dO = 4 pi (a-1)!!(b-1)!!(c-1)!!/(a+b+c+1)!!`.
fn mono_sphere(a: i32, b: i32, c: i32) -> f64 {
    if a % 2 != 0 || b % 2 != 0 || c % 2 != 0 {
        return 0.0;
    }
    4.0 * std::f64::consts::PI
        * double_factorial(a - 1)
        * double_factorial(b - 1)
        * double_factorial(c - 1)
        / double_factorial(a + b + c + 1)
}

/// One orthonormal real spherical harmonic as sum of c * O^(lx,ly,lz).
type YFunc = Vec<((i32, i32, i32), f64)>;

/// Per-l tables in the canonical m order (0, +1, -1, ...).
fn y_tables() -> &'static Vec<Vec<YFunc>> {
    static CACHE: OnceLock<Vec<Vec<YFunc>>> = OnceLock::new();
    CACHE.get_or_init(|| {
        (0..=LAMBDA_CAP as i32)
            .map(|l| {
                let comps = cart_components(l as u8);
                let m_order: Vec<i32> = std::iter::once(0)
                    .chain((1..=l).flat_map(|m| [m, -m]))
                    .collect();
                m_order
                    .iter()
                    .map(|&m| {
                        let mut f: YFunc = comps
                            .iter()
                            .filter_map(|&(lx, ly, lz)| {
                                let c = solid_harmonic_coef(l, m, lx, ly, lz);
                                (c != 0.0).then_some(((lx, ly, lz), c))
                            })
                            .collect();
                        // Exact L2 normalization on the sphere.
                        let mut n2 = 0.0;
                        for &((ax, ay, az), ca) in &f {
                            for &((bx, by, bz), cb) in &f {
                                n2 += ca * cb * mono_sphere(ax + bx, ay + by, az + bz);
                            }
                        }
                        let n = n2.sqrt();
                        for e in &mut f {
                            e.1 /= n;
                        }
                        f
                    })
                    .collect()
            })
            .collect()
    })
}

/// Evaluate Y at a unit vector.
fn y_eval(f: &YFunc, u: Vec3) -> f64 {
    f.iter()
        .map(|&((lx, ly, lz), c)| c * u.x.powi(lx) * u.y.powi(ly) * u.z.powi(lz))
        .sum()
}

/// `int O^j Y dO` (type-1 angular factor).
fn omega1(j: (i32, i32, i32), y: &YFunc) -> f64 {
    y.iter()
        .map(|&((lx, ly, lz), c)| c * mono_sphere(j.0 + lx, j.1 + ly, j.2 + lz))
        .sum()
}

fn monomials(max_degree: usize) -> Vec<(i32, i32, i32)> {
    let mut values = Vec::new();
    for degree in 0..=max_degree as i32 {
        for x in (0..=degree).rev() {
            for y in (0..=degree - x).rev() {
                values.push((x, y, degree - x - y));
            }
        }
    }
    values
}

fn monomial_index((x, y, z): (i32, i32, i32)) -> usize {
    debug_assert!(x >= 0 && y >= 0 && z >= 0);
    let degree = (x + y + z) as usize;
    let remainder = degree - x as usize;
    degree * (degree + 1) * (degree + 2) / 6 + remainder * (remainder + 1) / 2 + remainder
        - y as usize
}

/// Geometry-independent sphere moments, initialized only for requested
/// monomials. Their finite capacity is fixed by the existing harmonic limit.
fn local_moments(j: (i32, i32, i32)) -> &'static [Vec<f64>] {
    type Moments = Vec<Vec<f64>>;
    static CACHE: OnceLock<Vec<OnceLock<Moments>>> = OnceLock::new();
    let count = (LAMBDA_CAP + 1) * (LAMBDA_CAP + 2) * (LAMBDA_CAP + 3) / 6;
    let cache = CACHE.get_or_init(|| (0..count).map(|_| OnceLock::new()).collect());
    let degree = (j.0 + j.1 + j.2) as usize;
    debug_assert!(degree <= LAMBDA_CAP);
    cache[monomial_index(j)].get_or_init(|| {
        (0..=degree)
            .map(|l| y_tables()[l].iter().map(|y| omega1(j, y)).collect())
            .collect()
    })
}

/// One primitive pair's directional angular contractions. Preserve the
/// original harmonic summation order while reusing it across Cartesian
/// products and radial terms; no quadrature or screening rule changes.
fn local_angles(max_degree: usize, direction: Vec3) -> Vec<Vec<f64>> {
    let harmonics = y_tables()[..=max_degree]
        .iter()
        .map(|ys| ys.iter().map(|y| y_eval(y, direction)).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    monomials(max_degree)
        .into_iter()
        .map(|j| {
            local_moments(j)
                .iter()
                .enumerate()
                .map(|(l, moments)| {
                    let mut value = 0.0;
                    for (mu, &moment) in moments.iter().enumerate() {
                        let y = harmonics[l][mu];
                        if y != 0.0 {
                            value += y * moment;
                        }
                    }
                    value
                })
                .collect()
        })
        .collect()
}

/// `int O^j Ya Yb dO` (type-2 angular factor).
fn omega2(j: (i32, i32, i32), ya: &YFunc, yb: &YFunc) -> f64 {
    let mut s = 0.0;
    for &((ax, ay, az), ca) in ya {
        for &((bx, by, bz), cb) in yb {
            s += ca * cb * mono_sphere(j.0 + ax + bx, j.1 + ay + by, j.2 + az + bz);
        }
    }
    s
}

// ---------------------------------------------------------------------------
// Scaled modified spherical Bessel functions i~_lam(x) = exp(-x) i_lam(x)
// ---------------------------------------------------------------------------

/// Fill `out[0..=lmax]` with i~_lam(x), x >= 0.
fn scaled_bessel(lmax: usize, x: f64, out: &mut [f64]) {
    debug_assert!(out.len() > lmax);
    if x < 1e-10 {
        // i~_lam -> x^lam/(2 lam + 1)!! * (1 - x + ...).
        out[0] = 1.0 - x;
        let mut p = 1.0;
        for (l, o) in out.iter_mut().enumerate().take(lmax + 1).skip(1) {
            p *= x / (2 * l + 1) as f64;
            *o = p;
        }
        return;
    }
    let i0 = (1.0 - (-2.0 * x).exp()) / (2.0 * x);
    if lmax == 0 {
        out[0] = i0;
        return;
    }
    if x > 1.5 * lmax as f64 + 2.0 {
        // Upward recursion i_{l+1} = i_{l-1} - (2l+1)/x i_l is stable for
        // x well above lam.
        let e2 = (-2.0 * x).exp();
        let i1 = (1.0 + e2) / (2.0 * x) - (1.0 - e2) / (2.0 * x * x);
        out[0] = i0;
        out[1] = i1;
        for l in 1..lmax {
            out[l + 1] = out[l - 1] - (2 * l + 1) as f64 / x * out[l];
            if out[l + 1] < 0.0 {
                // Cancellation floor: the sequence is positive decreasing.
                out[l + 1] = 0.0;
            }
        }
    } else {
        // Downward (Miller) recursion normalized by the closed-form i~_0.
        let top = lmax + 24;
        let mut t_hi = 0.0f64;
        let mut t_lo = 1e-280f64;
        let mut buf = vec![0.0; top + 1];
        buf[top] = t_lo;
        for l in (1..=top).rev() {
            let t_prev = t_hi + (2 * l + 1) as f64 / x * t_lo;
            t_hi = t_lo;
            t_lo = t_prev;
            buf[l - 1] = t_lo;
            if t_lo > 1e260 {
                let s = 1e-260;
                t_lo *= s;
                t_hi *= s;
                for b in buf.iter_mut() {
                    *b *= s;
                }
            }
        }
        let scale = i0 / buf[0];
        for (l, o) in out.iter_mut().enumerate().take(lmax + 1) {
            *o = buf[l] * scale;
        }
    }
}

// ---------------------------------------------------------------------------
// Radial quadrature
// ---------------------------------------------------------------------------

/// Gauss-Legendre nodes/weights on the peak interval of
/// `r^m exp(-p r^2 + k r + shift)`.
fn radial_grid(p: f64, k: f64) -> (Vec<f64>, Vec<f64>) {
    static GL: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    let (xs, ws) = GL.get_or_init(|| gauss_legendre(NRAD));
    let r0 = (k / (2.0 * p)).max(0.0);
    let w = (46.0 / p).sqrt();
    let lo = (r0 - w).max(0.0);
    let hi = r0 + w;
    let half = 0.5 * (hi - lo);
    let mid = 0.5 * (hi + lo);
    let nodes: Vec<f64> = xs.iter().map(|&x| mid + half * x).collect();
    let weights: Vec<f64> = ws.iter().map(|&w| w * half).collect();
    (nodes, weights)
}

// ---------------------------------------------------------------------------
// Shell-pair evaluation
// ---------------------------------------------------------------------------

/// Sub-monomials `j <= u` (component-wise) of a Cartesian power tuple, with
/// the shifted-binomial coefficients `prod binom(u_c, j_c) (-d_c)^{u_c-j_c}`
/// from `(x - A_x)^{u_x} = ((x - C_x) - d_x)^{u_x}` etc.
fn binomial_shift(u: (i32, i32, i32), d: Vec3) -> Vec<((i32, i32, i32), f64)> {
    let binom = |n: i32, k: i32| -> f64 {
        let mut v = 1.0;
        for i in 0..k {
            v = v * (n - i) as f64 / (i + 1) as f64;
        }
        v
    };
    let mut out = Vec::new();
    for jx in 0..=u.0 {
        for jy in 0..=u.1 {
            for jz in 0..=u.2 {
                let c = binom(u.0, jx)
                    * binom(u.1, jy)
                    * binom(u.2, jz)
                    * (-d.x).powi(u.0 - jx)
                    * (-d.y).powi(u.1 - jy)
                    * (-d.z).powi(u.2 - jz);
                if c != 0.0 {
                    out.push(((jx, jy, jz), c));
                }
            }
        }
    }
    out
}

/// Unit vector of `d`, or z_hat when d ~ 0 (higher-lam Bessel factors vanish
/// exactly there, so the direction is arbitrary).
fn unit_or_z(d: Vec3, n: f64) -> Vec3 {
    if n < 1e-12 {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        d * (1.0 / n)
    }
}

/// Full ECP Cartesian block for one shell pair and one ECP center.
pub(crate) fn ecp_shell_pair(sha: &MolShell, shb: &MolShell, c: Vec3, pot: &EcpPotential) -> Matrix {
    let comps_a = cart_components(sha.l);
    let comps_b = cart_components(shb.l);
    let mut block = Matrix::zeros(comps_a.len(), comps_b.len());
    let d_a = sha.center - c;
    let d_b = shb.center - c;
    let (da_n, db_n) = (d_a.norm(), d_b.norm());
    let (ua, ub) = (unit_or_z(d_a, da_n), unit_or_z(d_b, db_n));
    let la = sha.l as usize;
    let lb = shb.l as usize;
    let ytab = y_tables();

    // Shifted-binomial expansions per Cartesian component (primitive-free).
    let shifts_a: Vec<_> = comps_a.iter().map(|&u| binomial_shift(u, d_a)).collect();
    let shifts_b: Vec<_> = comps_b.iter().map(|&u| binomial_shift(u, d_b)).collect();

    // All monomials up to the shell degree, for the type-2 angular tables.
    let monos_a = monomials(la);
    let monos_b = monomials(lb);

    // Type-2 angular tables (primitive-free):
    // W[chan][mono][lam][m] = sum_mu Y_lam,mu(d_hat) int O^j Y_lam,mu Y_lm dO.
    let n_chan = pot.semilocal.len();
    let build_w = |mlist: &[(i32, i32, i32)], u: Vec3| -> Vec<Vec<Vec<Vec<f64>>>> {
        (0..n_chan)
            .map(|l| {
                let yl = &ytab[l];
                mlist
                    .iter()
                    .map(|&j| {
                        let jdeg = (j.0 + j.1 + j.2) as usize;
                        let lam_max = (l + jdeg).min(LAMBDA_CAP);
                        (0..=lam_max)
                            .map(|lam| {
                                let ylam = &ytab[lam];
                                (0..2 * l + 1)
                                    .map(|mi| {
                                        (0..2 * lam + 1)
                                            .map(|mu| {
                                                let yv = y_eval(&ylam[mu], u);
                                                if yv == 0.0 {
                                                    0.0
                                                } else {
                                                    yv * omega2(j, &ylam[mu], &yl[mi])
                                                }
                                            })
                                            .sum()
                                    })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    };
    let w_a = build_w(&monos_a, ua);
    let w_b = build_w(&monos_b, ub);

    let four_pi = 4.0 * std::f64::consts::PI;
    let mut bes_a = vec![0.0; LAMBDA_CAP + 1];
    let mut bes_b = vec![0.0; LAMBDA_CAP + 1];

    for (ia, &alpha) in sha.exps.iter().enumerate() {
        let ca = sha.coefs[ia];
        for (ib, &beta) in shb.exps.iter().enumerate() {
            let cb = shb.coefs[ib];
            let cc = ca * cb;
            let shift = -alpha * da_n * da_n - beta * db_n * db_n;
            let k_a = 2.0 * alpha * da_n;
            let k_b = 2.0 * beta * db_n;

            // ---------------- Type-1: local channel U_L ----------------
            let pvec = d_a * alpha + d_b * beta;
            let p_n = pvec.norm();
            let up = unit_or_z(pvec, p_n);
            let k_p = 2.0 * p_n;
            let nmax1 = la + lb;
            let mut angles = None;
            for term in &pot.local {
                let p_tot = alpha + beta + term.exp;
                if k_p * k_p / (4.0 * p_tot) + shift < EXP_CUTOFF {
                    continue;
                }
                // rad1[lam][N] = int r^{n+N} i~_lam(kP r)
                //                exp(-p r^2 + kP r + shift) dr
                // (r^2 volume element folded with the r^{n-2} radial law).
                let (nodes, wts) = radial_grid(p_tot, k_p);
                let mut rad1 = vec![vec![0.0; nmax1 + 1]; nmax1 + 1];
                for (r, w) in nodes.iter().zip(&wts) {
                    let ex = (-p_tot * r * r + k_p * r + shift).exp();
                    if ex == 0.0 {
                        continue;
                    }
                    scaled_bessel(nmax1, k_p * r, &mut bes_a);
                    let base = w * r.powi(term.power) * ex;
                    let mut rn = base;
                    for n in 0..=nmax1 {
                        for (lam, r1) in rad1.iter_mut().enumerate().take(nmax1 + 1) {
                            r1[n] += rn * bes_a[lam];
                        }
                        rn *= r;
                    }
                }
                let angles = angles.get_or_insert_with(|| local_angles(nmax1, up));
                for (row, sa) in shifts_a.iter().enumerate() {
                    for (col, sb) in shifts_b.iter().enumerate() {
                        let mut v = 0.0;
                        for &(j, tj) in sa {
                            for &(k, tk) in sb {
                                let u = (j.0 + k.0, j.1 + k.1, j.2 + k.2);
                                let ndeg = (u.0 + u.1 + u.2) as usize;
                                for (lam, r1) in rad1.iter().enumerate().take(ndeg + 1) {
                                    let ang = angles[monomial_index(u)][lam];
                                    if ang != 0.0 {
                                        v += tj * tk * ang * r1[ndeg];
                                    }
                                }
                            }
                        }
                        block[(row, col)] += four_pi * cc * term.coef * v;
                    }
                }
            }

            // ---------------- Type-2: projected channels ----------------
            for (l, terms) in pot.semilocal.iter().enumerate() {
                if terms.is_empty() {
                    continue;
                }
                let lam_a_max = (l + la).min(LAMBDA_CAP);
                let lam_b_max = (l + lb).min(LAMBDA_CAP);
                let nmax = la + lb;
                for term in terms {
                    let p_tot = alpha + beta + term.exp;
                    let k_sum = k_a + k_b;
                    if k_sum * k_sum / (4.0 * p_tot) + shift < EXP_CUTOFF {
                        continue;
                    }
                    // rad2[lam_a][lam_b][N] = int r^{n+N} i~_a i~_b
                    //                         exp(-p r^2 + (kA+kB) r + shift) dr.
                    let (nodes, wts) = radial_grid(p_tot, k_sum);
                    let mut rad2 = vec![vec![vec![0.0; nmax + 1]; lam_b_max + 1]; lam_a_max + 1];
                    for (r, w) in nodes.iter().zip(&wts) {
                        let ex = (-p_tot * r * r + k_sum * r + shift).exp();
                        if ex == 0.0 {
                            continue;
                        }
                        scaled_bessel(lam_a_max, k_a * r, &mut bes_a);
                        scaled_bessel(lam_b_max, k_b * r, &mut bes_b);
                        let base = w * r.powi(term.power) * ex;
                        for (la_i, r2a) in rad2.iter_mut().enumerate().take(lam_a_max + 1) {
                            let ba = bes_a[la_i] * base;
                            if ba == 0.0 {
                                continue;
                            }
                            for (lb_i, r2b) in r2a.iter_mut().enumerate().take(lam_b_max + 1) {
                                let bab = ba * bes_b[lb_i];
                                let mut rn = bab;
                                for r2n in r2b.iter_mut().take(nmax + 1) {
                                    *r2n += rn;
                                    rn *= r;
                                }
                            }
                        }
                    }
                    // A shifted monomial pair recurs in many Cartesian
                    // components. Its complete radial/angular contraction is
                    // identical for this primitive, channel and radial term.
                    let mut pair_values = vec![None; monos_a.len() * monos_b.len()];
                    for (row, sa) in shifts_a.iter().enumerate() {
                        for (col, sb) in shifts_b.iter().enumerate() {
                            let mut v = 0.0;
                            for &(j, tj) in sa {
                                let jdeg = (j.0 + j.1 + j.2) as usize;
                                let wa = &w_a[l][monomial_index(j)];
                                for &(k, tk) in sb {
                                    let kdeg = (k.0 + k.1 + k.2) as usize;
                                    let wb = &w_b[l][monomial_index(k)];
                                    let slot =
                                        monomial_index(j) * monos_b.len() + monomial_index(k);
                                    let s = *pair_values[slot].get_or_insert_with(|| {
                                        let mut value = 0.0;
                                        for (la_i, wa_l) in wa.iter().enumerate() {
                                            for (lb_i, wb_l) in wb.iter().enumerate() {
                                                let mut msum = 0.0;
                                                for mi in 0..2 * l + 1 {
                                                    msum += wa_l[mi] * wb_l[mi];
                                                }
                                                if msum != 0.0 {
                                                    value += msum * rad2[la_i][lb_i][jdeg + kdeg];
                                                }
                                            }
                                        }
                                        value
                                    });
                                    v += tj * tk * s;
                                }
                            }
                            block[(row, col)] += four_pi * four_pi * cc * term.coef * v;
                        }
                    }
                }
            }
        }
    }
    block
}

/// Assemble the full spherical-AO ECP matrix `sum_C U_C` over all
/// ECP-carrying atoms of the basis.
pub fn ecp_matrix(mb: &MolBasis) -> Result<Matrix> {
    let mut u = Matrix::zeros(mb.nao_sph, mb.nao_sph);
    if mb.ecps.is_empty() {
        return Ok(u);
    }
    let pairs: Vec<(usize, usize)> = (0..mb.shells.len())
        .flat_map(|i| (i..mb.shells.len()).map(move |j| (i, j)))
        .collect();
    let blocks: Vec<((usize, usize), Matrix)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let (sha, shb) = (&mb.shells[i], &mb.shells[j]);
            let mut cart =
                Matrix::zeros(cart_components(sha.l).len(), cart_components(shb.l).len());
            for ecp in &mb.ecps {
                let b = ecp_shell_pair(sha, shb, ecp.center, &ecp.potential);
                for r in 0..cart.rows {
                    for c in 0..cart.cols {
                        cart[(r, c)] += b[(r, c)];
                    }
                }
            }
            ((i, j), sph_block(sha.l, shb.l, &cart))
        })
        .collect();
    for ((i, j), sph) in blocks {
        let (sha, shb) = (&mb.shells[i], &mb.shells[j]);
        for r in 0..sph.rows {
            for c in 0..sph.cols {
                u[(sha.sph_off + r, shb.sph_off + c)] += sph[(r, c)];
                if i != j {
                    u[(shb.sph_off + c, sha.sph_off + r)] += sph[(r, c)];
                }
            }
        }
    }
    Ok(u)
}

/// Analytic nuclear gradient of the ECP energy `sum_mn P_mn <m|U|n>`:
/// bra-shell derivatives via the raised/lowered shells (the ordered-pair
/// loop covers each element's bra side once; operator symmetry makes the
/// ket side equal, hence the overall factor 2), and the ECP-center
/// derivative from translational invariance dU/dC = -(dA + dB).
pub fn ecp_gradient(mb: &MolBasis, natoms: usize, p: &Matrix) -> Vec<Vec3> {
    use super::deriv::{combine_bra_derivative, lowered, raised};
    if mb.ecps.is_empty() {
        return vec![Vec3::zero(); natoms];
    }
    let ns = mb.shells.len();
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let partials: Vec<Vec<Vec3>> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let mut out = vec![Vec3::zero(); natoms];
            let (sh_a, sh_b) = (&mb.shells[a], &mb.shells[b]);
            let ra = raised(sh_a);
            let lo = lowered(sh_a);
            for ecp in &mb.ecps {
                let raise_blk = ecp_shell_pair(&ra, sh_b, ecp.center, &ecp.potential);
                let lower_blk = lo
                    .as_ref()
                    .map(|l| ecp_shell_pair(l, sh_b, ecp.center, &ecp.potential));
                for ax in 0..3 {
                    let cart =
                        combine_bra_derivative(sh_a.l, sh_b.l, ax, &raise_blk, lower_blk.as_ref());
                    let sph = crate::basis::aos::sph_block(sh_a.l, sh_b.l, &cart);
                    let mut v = 0.0;
                    for i in 0..sph.rows {
                        for j in 0..sph.cols {
                            v += p[(sh_a.sph_off + i, sh_b.sph_off + j)] * sph[(i, j)];
                        }
                    }
                    let v2 = 2.0 * v;
                    let vec = match ax {
                        0 => Vec3::new(v2, 0.0, 0.0),
                        1 => Vec3::new(0.0, v2, 0.0),
                        _ => Vec3::new(0.0, 0.0, v2),
                    };
                    out[sh_a.atom] += vec;
                    out[ecp.atom] -= vec;
                }
            }
            out
        })
        .collect();
    let mut total = vec![Vec3::zero(); natoms];
    for pt in partials {
        for (t, v) in total.iter_mut().zip(pt) {
            *t += v;
        }
    }
    total
}

/// Derivative MATRICES `dU_mn/dR_a` of the semi-local ECP operator, one
/// `[x, y, z]` triple per atom — the ECP's contribution to the perturbed
/// core Hamiltonian that CPKS needs.
///
/// [`ecp_gradient`] returns the energy gradient; this returns the operator
/// derivative it is built from, which is what a response calculation
/// requires.
///
/// `U` is symmetric at every geometry, so `dU/dR` is symmetric too. That
/// makes the construction simple: accumulate only the BRA-side blocks over
/// ordered shell pairs, then symmetrize. The transpose supplies exactly the
/// ket-side term, and contracting the result with a symmetric density
/// reproduces the factor 2 that [`ecp_gradient`] applies by hand.
///
/// The ECP CENTRE derivative comes from translational invariance,
/// `dU/dC = -(dU/dA + dU/dB)` — the same bookkeeping as the nuclear
/// attraction, so no third derivative class is needed. Summing the returned
/// matrices over atoms therefore gives zero, which is what the test checks.
pub fn ecp_derivative_matrices(mb: &MolBasis, natoms: usize) -> Vec<super::deriv_mat::AtomDerivs> {
    let n = mb.nao_sph;
    let mut acc: Vec<[Matrix; 3]> = (0..natoms)
        .map(|_| {
            [
                Matrix::zeros(n, n),
                Matrix::zeros(n, n),
                Matrix::zeros(n, n),
            ]
        })
        .collect();
    if mb.ecps.is_empty() {
        return acc;
    }
    let ns = mb.shells.len();
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    // (bra atom, ECP atom, axis, shell a, shell b, block)
    let partials: Vec<Vec<(usize, usize, usize, usize, usize, Matrix)>> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let (sh_a, sh_b) = (&mb.shells[a], &mb.shells[b]);
            let bra = EcpBraDerivative::new(sh_a);
            let mut out = Vec::new();
            for ecp in &mb.ecps {
                for (ax, sph) in bra
                    .spherical(sh_b, ecp.center, &ecp.potential)
                    .into_iter()
                    .enumerate()
                {
                    out.push((sh_a.atom, ecp.atom, ax, a, b, sph));
                }
            }
            out
        })
        .collect();
    for chunk in partials {
        for (atom_bra, atom_ecp, ax, a, b, sph) in chunk {
            let (oa, ob) = (mb.shells[a].sph_off, mb.shells[b].sph_off);
            for i in 0..sph.rows {
                for j in 0..sph.cols {
                    let v = sph[(i, j)];
                    if v == 0.0 {
                        continue;
                    }
                    acc[atom_bra][ax][(oa + i, ob + j)] += v;
                    acc[atom_ecp][ax][(oa + i, ob + j)] -= v;
                }
            }
        }
    }
    for atom in acc.iter_mut() {
        for m in atom.iter_mut() {
            let t = m.transpose();
            for (v, tv) in m.as_mut_slice().iter_mut().zip(t.as_slice()) {
                *v += *tv;
            }
        }
    }
    acc
}

/// Spherical AO derivatives for one bra centre against translated ECP centres.
pub(crate) struct EcpBraDerivative {
    angular_momentum: u8,
    raised: MolShell,
    lowered: Option<MolShell>,
}

impl EcpBraDerivative {
    pub(crate) fn new(bra: &MolShell) -> Self {
        Self {
            angular_momentum: bra.l,
            raised: super::deriv::raised(bra),
            lowered: super::deriv::lowered(bra),
        }
    }

    pub(crate) fn spherical(
        &self,
        ket: &MolShell,
        center: Vec3,
        potential: &EcpPotential,
    ) -> [Matrix; 3] {
        let raised = ecp_shell_pair(&self.raised, ket, center, potential);
        let lowered = self
            .lowered
            .as_ref()
            .map(|bra| ecp_shell_pair(bra, ket, center, potential));
        std::array::from_fn(|axis| {
            let cartesian = super::deriv::combine_bra_derivative(
                self.angular_momentum,
                ket.l,
                axis,
                &raised,
                lowered.as_ref(),
            );
            crate::basis::aos::sph_block(self.angular_momentum, ket.l, &cartesian)
        })
    }
}

/// Second derivative of the ECP energy `sum_mn P_mn <m|U|n>`, contracted
/// into the 3N x 3N nuclear block.
///
/// Same construction as `ecp_gradient` one order up: the bra and ket
/// derivatives come from raised/lowered shells, and the ECP CENTRE follows
/// from translational invariance without a third derivative class —
/// `AC = -AA - AB`, `BC = -BA - BB`, `CC = AA + AB + BA + BB`. That is the
/// identical bookkeeping `deriv2` uses for the nuclear attraction, which is
/// why nothing new had to be derived here.
///
/// Unlike the nuclear-attraction case the operator is SEMI-LOCAL, so the
/// bra and ket sides are not related by a simple transpose; `AB` and `BA`
/// are both formed explicitly.
pub fn ecp_second_contraction(mb: &MolBasis, natoms: usize, p: &Matrix) -> Matrix {
    use super::deriv::{combine_bra_derivative, lowered, raised};
    use super::deriv2::{combine_bra_second, combine_ket_derivative, combine_ket_second};
    if mb.ecps.is_empty() {
        return Matrix::zeros(3 * natoms, 3 * natoms);
    }
    let ns = mb.shells.len();
    let pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (0..ns).map(move |b| (a, b))).collect();
    let contract = |m: &Matrix, oa: usize, ob: usize| -> f64 {
        let mut v = 0.0;
        for i in 0..m.rows {
            for j in 0..m.cols {
                v += p[(oa + i, ob + j)] * m[(i, j)];
            }
        }
        v
    };
    let partials: Vec<Matrix> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let mut h = Matrix::zeros(3 * natoms, 3 * natoms);
            let (sh_a, sh_b) = (&mb.shells[a], &mb.shells[b]);
            let ra = raised(sh_a);
            let rra = raised(&ra);
            let la = lowered(sh_a);
            let lla = la.as_ref().and_then(lowered);
            let rb = raised(sh_b);
            let rrb = raised(&rb);
            let lb = lowered(sh_b);
            let llb = lb.as_ref().and_then(lowered);
            let rl_a = lowered(&ra).expect("l+1 is always lowerable");
            let rl_b = lowered(&rb).expect("l+1 is always lowerable");
            let (ia, ib) = (sh_a.atom, sh_b.atom);
            for ecp in &mb.ecps {
                let c = ecp.center;
                let pot = &ecp.potential;
                let blk = |x: &MolShell, y: &MolShell| ecp_shell_pair(x, y, c, pot);
                // Nested on the bra, nested on the ket, and mixed.
                let aa_rr = blk(&rra, sh_b);
                let aa_rl = blk(&rl_a, sh_b);
                let aa_ll = lla.as_ref().map(|l| blk(l, sh_b));
                let bb_rr = blk(sh_a, &rrb);
                let bb_rl = blk(sh_a, &rl_b);
                let bb_ll = llb.as_ref().map(|l| blk(sh_a, l));
                let m_rr = blk(&ra, &rb);
                let m_rl = lb.as_ref().map(|l| blk(&ra, l));
                let m_lr = la.as_ref().map(|l| blk(l, &rb));
                let m_ll = la.as_ref().and_then(|l| lb.as_ref().map(|r| blk(l, r)));
                let (oa, ob) = (sh_a.sph_off, sh_b.sph_off);
                for ax in 0..3 {
                    let bra_r =
                        combine_bra_derivative(sh_a.l, sh_b.l + 1, ax, &m_rr, m_lr.as_ref());
                    let bra_l = m_rl.as_ref().map(|rlb| {
                        combine_bra_derivative(sh_a.l, sh_b.l - 1, ax, rlb, m_ll.as_ref())
                    });
                    for by in 0..3 {
                        let aa = contract(
                            &crate::basis::aos::sph_block(
                                sh_a.l,
                                sh_b.l,
                                &combine_bra_second(
                                    sh_a.l,
                                    sh_b.l,
                                    ax,
                                    by,
                                    &aa_rr,
                                    &aa_rl,
                                    aa_ll.as_ref(),
                                ),
                            ),
                            oa,
                            ob,
                        );
                        let bb = contract(
                            &crate::basis::aos::sph_block(
                                sh_a.l,
                                sh_b.l,
                                &combine_ket_second(
                                    sh_a.l,
                                    sh_b.l,
                                    ax,
                                    by,
                                    &bb_rr,
                                    &bb_rl,
                                    bb_ll.as_ref(),
                                ),
                            ),
                            oa,
                            ob,
                        );
                        let ab = contract(
                            &crate::basis::aos::sph_block(
                                sh_a.l,
                                sh_b.l,
                                &combine_ket_derivative(sh_a.l, sh_b.l, by, &bra_r, bra_l.as_ref()),
                            ),
                            oa,
                            ob,
                        );
                        // BA(ax, by) = the mixed block with the axes swapped.
                        let bra_r2 =
                            combine_bra_derivative(sh_a.l, sh_b.l + 1, by, &m_rr, m_lr.as_ref());
                        let bra_l2 = m_rl.as_ref().map(|rlb| {
                            combine_bra_derivative(sh_a.l, sh_b.l - 1, by, rlb, m_ll.as_ref())
                        });
                        let ba = contract(
                            &crate::basis::aos::sph_block(
                                sh_a.l,
                                sh_b.l,
                                &combine_ket_derivative(
                                    sh_a.l,
                                    sh_b.l,
                                    ax,
                                    &bra_r2,
                                    bra_l2.as_ref(),
                                ),
                            ),
                            oa,
                            ob,
                        );
                        let ac = -aa - ab;
                        let bc = -ba - bb;
                        let cc = aa + ab + ba + bb;
                        let ec = ecp.atom;
                        let mut add = |r: usize, rx: usize, s: usize, sy: usize, v: f64| {
                            h[(3 * r + rx, 3 * s + sy)] += v;
                        };
                        add(ia, ax, ia, by, aa);
                        add(ia, ax, ib, by, ab);
                        add(ib, ax, ia, by, ba);
                        add(ib, ax, ib, by, bb);
                        add(ia, ax, ec, by, ac);
                        add(ec, by, ia, ax, ac);
                        add(ib, ax, ec, by, bc);
                        add(ec, by, ib, ax, bc);
                        add(ec, ax, ec, by, cc);
                    }
                }
            }
            h
        })
        .collect();
    let mut total = Matrix::zeros(3 * natoms, 3 * natoms);
    for pt in partials {
        for (t, v) in total.as_mut_slice().iter_mut().zip(pt.as_slice()) {
            *t += *v;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    #[test]
    fn local_angles_preserve_direct_sphere_sums_and_graded_indices() {
        let mut comparisons = 0usize;
        for d in [
            Vec3::zero(),
            Vec3::new(0.31, -0.47, 0.83),
            Vec3::new(-0.19, 0.61, 0.37),
            Vec3::new(-0.41, -0.23, -0.79),
        ] {
            let direction = unit_or_z(d, d.norm());
            let angles = local_angles(LAMBDA_CAP, direction);
            let powers = monomials(LAMBDA_CAP);
            assert_eq!(angles.len(), powers.len());
            for (index, j) in powers.into_iter().enumerate() {
                assert_eq!(monomial_index(j), index);
                let degree = (j.0 + j.1 + j.2) as usize;
                assert_eq!(angles[index].len(), degree + 1);
                for (l, &cached) in angles[index].iter().enumerate() {
                    let mut direct = 0.0;
                    for y in &y_tables()[l] {
                        let value = y_eval(y, direction);
                        if value != 0.0 {
                            direct += value * omega1(j, y);
                        }
                    }
                    assert_eq!(cached.to_bits(), direct.to_bits(), "j={j:?} l={l}");
                    comparisons += 1;
                }
            }
            assert!((angles[0][0] - 1.0).abs() < 1e-14);
        }
        let bytes = monomials(LAMBDA_CAP)
            .into_iter()
            .map(|j| {
                let moments = local_moments(j);
                std::mem::size_of::<OnceLock<Vec<Vec<f64>>>>()
                    + std::mem::size_of_val(moments)
                    + moments
                        .iter()
                        .map(|m| std::mem::size_of_val(m.as_slice()))
                        .sum::<usize>()
            })
            .sum::<usize>();
        println!("ECP local angular direct comparisons={comparisons} bounded_cache_bytes={bytes}");
        assert!(bytes < 4 * 1024 * 1024);
    }

    /// Is `ecp_shell_pair(a,b)` the transpose of `ecp_shell_pair(b,a)`?
    ///
    /// `ecp_matrix` only ever evaluates the upper triangle and mirrors it,
    /// so nothing in the energy path depends on the answer — but
    /// `ecp_gradient` and everything built on it loop over ALL ordered
    /// pairs, and the derivative bookkeeping is completely different
    /// depending on which way this falls. Measuring it directly is worth
    /// more than reasoning about it: the operator is Hermitian, so the
    /// answer "should" be yes, and the ECP Hessian discrepancy is evidence
    /// that something in that chain is not what it appears.
    #[test]
    fn ecp_shell_pair_transpose_symmetry() {
        use crate::basis::data;
        use crate::system::{Atom, Molecule};

        let mol = Molecule::new(vec![
            Atom {
                z: 16,
                position: Vec3::new(0.0, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 1.81, 1.45),
            },
        ]);
        let set = data::load("lanl2dz").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        let ecp = &mb.ecps[0];
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for i in 0..mb.shells.len() {
            for j in 0..mb.shells.len() {
                let ab = ecp_shell_pair(&mb.shells[i], &mb.shells[j], ecp.center, &ecp.potential);
                let ba = ecp_shell_pair(&mb.shells[j], &mb.shells[i], ecp.center, &ecp.potential);
                for r in 0..ab.rows {
                    for c in 0..ab.cols {
                        worst = worst.max((ab[(r, c)] - ba[(c, r)]).abs());
                        scale = scale.max(ab[(r, c)].abs());
                    }
                }
            }
        }
        assert!(
            worst < 1e-12 * scale.max(1.0),
            "ecp_shell_pair is NOT transpose-symmetric: worst {worst:e} on a \
             scale of {scale:e}. The derivative bookkeeping in ecp_gradient \
             and ecp_derivative_matrices assumes it is."
        );
    }

    /// `ecp_gradient` against central differences of the ECP energy
    /// `Tr[P U(R)]` at FIXED P.
    ///
    /// This is the gate that settles the convention question the Hessian
    /// work ran into: `ecp_gradient` sums bra-side derivatives over ALL
    /// ordered shell pairs and then doubles, while an ordered-pair loop
    /// already visits both `(a,b)` and `(b,a)`. Whether that doubling is
    /// right cannot be read off the code — but it can be measured, and a
    /// fixed density removes the SCF from the comparison so nothing else
    /// can absorb an error.
    #[test]
    fn ecp_gradient_matches_fd_of_the_ecp_energy() {
        use crate::basis::data;
        use crate::system::{Atom, Molecule};

        let build = |shift: (usize, usize, f64)| -> Molecule {
            let mut atoms = vec![
                Atom {
                    z: 16,
                    position: Vec3::new(0.0, 0.0, 0.0),
                },
                Atom {
                    z: 1,
                    position: Vec3::new(0.0, 1.81, 1.45),
                },
                Atom {
                    z: 1,
                    position: Vec3::new(0.3, -1.75, 1.50),
                },
            ];
            let (at, ax, h) = shift;
            match ax {
                0 => atoms[at].position.x += h,
                1 => atoms[at].position.y += h,
                _ => atoms[at].position.z += h,
            }
            Molecule::new(atoms)
        };
        let set = data::load("lanl2dz").unwrap();
        let mol0 = build((0, 0, 0.0));
        let mb0 = MolBasis::build(&mol0, &set).unwrap();
        let nao = mb0.nao_sph;
        let mut p = Matrix::zeros(nao, nao);
        for mu in 0..nao {
            for nu in mu..nao {
                let v = 0.1 * (((mu * 5 + nu * 3) % 7) as f64 / 7.0 - 0.5);
                p[(mu, nu)] = v;
                p[(nu, mu)] = v;
            }
        }
        let g = ecp_gradient(&mb0, 3, &p);
        let energy = |shift: (usize, usize, f64)| -> f64 {
            let m = build(shift);
            let mbx = MolBasis::build(&m, &set).unwrap();
            p.frobenius_dot(&ecp_matrix(&mbx).unwrap())
        };
        let h = 1e-4;
        let mut worst = 0.0_f64;
        for at in 0..3 {
            for ax in 0..3 {
                let fd = (energy((at, ax, h)) - energy((at, ax, -h))) / (2.0 * h);
                let got = [g[at].x, g[at].y, g[at].z][ax];
                worst = worst.max((got - fd).abs());
            }
        }
        assert!(
            worst < 1e-7,
            "ecp_gradient vs FD of Tr[P U]: worst {worst:e} \
             (a factor-of-two convention error shows up here as ~100% off)"
        );
    }

    /// `ecp_derivative_matrices` against central differences of
    /// `ecp_matrix`, ELEMENT BY ELEMENT.
    ///
    /// This is the gate that should have been written first. Contracting
    /// with a density and comparing against the gradient — which is what
    /// the test below does — cannot distinguish the right matrix from any
    /// other with the same trace against that one density, and the ECP
    /// Hessian work was misled by exactly that gap.
    #[test]
    fn ecp_derivative_matrices_match_fd_of_the_ecp_matrix() {
        use crate::basis::data;
        use crate::system::{Atom, Molecule};

        let build = |shift: (usize, usize, f64)| -> Molecule {
            let mut atoms = vec![
                Atom {
                    z: 16,
                    position: Vec3::new(0.0, 0.0, 0.0),
                },
                Atom {
                    z: 1,
                    position: Vec3::new(0.0, 1.81, 1.45),
                },
                Atom {
                    z: 1,
                    position: Vec3::new(0.3, -1.75, 1.50),
                },
            ];
            let (at, ax, h) = shift;
            match ax {
                0 => atoms[at].position.x += h,
                1 => atoms[at].position.y += h,
                _ => atoms[at].position.z += h,
            }
            Molecule::new(atoms)
        };
        let set = data::load("lanl2dz").unwrap();
        let mb0 = MolBasis::build(&build((0, 0, 0.0)), &set).unwrap();
        let du = ecp_derivative_matrices(&mb0, 3);
        let u_at = |shift: (usize, usize, f64)| -> Matrix {
            let mbx = MolBasis::build(&build(shift), &set).unwrap();
            ecp_matrix(&mbx).unwrap()
        };
        let h = 1e-4;
        let n = mb0.nao_sph;
        let mut worst = 0.0_f64;
        let mut worst_at = (0, 0, 0, 0);
        for at in 0..3 {
            for ax in 0..3 {
                let (up, um) = (u_at((at, ax, h)), u_at((at, ax, -h)));
                for i in 0..n {
                    for j in 0..n {
                        let fd = (up[(i, j)] - um[(i, j)]) / (2.0 * h);
                        let d = (du[at][ax][(i, j)] - fd).abs();
                        if d > worst {
                            worst = d;
                            worst_at = (at, ax, i, j);
                        }
                    }
                }
            }
        }
        let (at, ax, i, j) = worst_at;
        assert!(
            worst < 1e-6,
            "dU/dR element by element: worst {worst:e} at atom {at} axis {ax} \
             element ({i},{j}) — analytic {} vs FD",
            du[at][ax][(i, j)]
        );
    }

    /// The ECP derivative MATRICES must reproduce the ECP energy gradient
    /// when contracted with the density, and must sum to zero over atoms.
    ///
    /// The first identity ties the new matrices to the FD-gated gradient,
    /// so they inherit that verification instead of needing their own
    /// finite differences. The second is translational invariance: shifting
    /// the whole molecule cannot change any integral, so the per-atom
    /// derivatives must cancel exactly — the check that catches the ECP
    /// centre's term being mis-signed or omitted, which is precisely the
    /// piece that has no independent integral class.
    #[test]
    fn ecp_derivative_matrices_reproduce_the_gradient_and_sum_to_zero() {
        use crate::basis::data;
        use crate::system::{Atom, Molecule};

        let d = 1.052;
        let mol = Molecule::new(vec![
            Atom {
                z: 28,
                position: Vec3::new(0.0, 0.0, 0.0),
            },
            Atom {
                z: 6,
                position: Vec3::new(d, d, d),
            },
            Atom {
                z: 6,
                position: Vec3::new(-d, -d, d),
            },
        ]);
        let set = data::load("lanl2dz").unwrap();
        let mb = MolBasis::build(&mol, &set).unwrap();
        let nao = mb.nao_sph;
        let mut p = Matrix::zeros(nao, nao);
        for mu in 0..nao {
            for nu in mu..nao {
                let v = 0.1 * (((mu * 5 + nu * 3) % 7) as f64 / 7.0 - 0.5);
                p[(mu, nu)] = v;
                p[(nu, mu)] = v;
            }
        }
        let du = ecp_derivative_matrices(&mb, 3);
        let g = ecp_gradient(&mb, 3, &p);
        for at in 0..3 {
            for (ax, want) in [g[at].x, g[at].y, g[at].z].into_iter().enumerate() {
                let got = p.frobenius_dot(&du[at][ax]);
                assert!(
                    (got - want).abs() < 1e-9 * want.abs().max(1e-6),
                    "atom {at} axis {ax}: Tr[P dU] = {got:e} vs gradient {want:e}"
                );
            }
        }
        for ax in 0..3 {
            let mut sum = Matrix::zeros(nao, nao);
            for atom in du.iter() {
                for (s, v) in sum.as_mut_slice().iter_mut().zip(atom[ax].as_slice()) {
                    *s += *v;
                }
            }
            let worst = sum.as_slice().iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
            assert!(
                worst < 1e-10,
                "axis {ax}: the per-atom ECP derivatives do not sum to zero ({worst:e})"
            );
        }
    }

    /// The ECP second derivative against central differences of the ECP
    /// GRADIENT (which itself is FD-gated against the energy). Ni(CO)4 in
    /// LANL2DZ exercises a real semi-local potential on a metal centre,
    /// where the ket and bra sides are genuinely different — the case a
    /// transpose-based shortcut would silently get wrong.
    #[test]
    fn ecp_second_derivative_matches_fd() {
        use crate::basis::data;
        use crate::system::{Atom, Molecule};

        let build = |shift: (usize, usize, f64)| -> Molecule {
            let d = 1.052;
            let mut atoms = vec![
                Atom {
                    z: 28,
                    position: Vec3::new(0.0, 0.0, 0.0),
                },
                Atom {
                    z: 6,
                    position: Vec3::new(d, d, d),
                },
                Atom {
                    z: 6,
                    position: Vec3::new(-d, -d, d),
                },
            ];
            let (at, ax, h) = shift;
            match ax {
                0 => atoms[at].position.x += h,
                1 => atoms[at].position.y += h,
                _ => atoms[at].position.z += h,
            }
            Molecule::new(atoms)
        };
        let set = data::load("lanl2dz").unwrap();
        let mol0 = build((0, 0, 0.0));
        let mb0 = MolBasis::build(&mol0, &set).unwrap();
        let nao = mb0.nao_sph;
        // A fixed, deterministic symmetric density: the identity being
        // gated holds for any P, and fixing it isolates the integrals.
        let mut pd = Matrix::zeros(nao, nao);
        for mu in 0..nao {
            for nu in mu..nao {
                let v = 0.1 * (((mu * 5 + nu * 3) % 7) as f64 / 7.0 - 0.5);
                pd[(mu, nu)] = v;
                pd[(nu, mu)] = v;
            }
        }
        let an = ecp_second_contraction(&mb0, 3, &pd);
        let grad = |shift: (usize, usize, f64)| -> Vec<Vec3> {
            let mm = build(shift);
            let mbx = MolBasis::build(&mm, &set).unwrap();
            ecp_gradient(&mbx, 3, &pd)
        };
        let h = 1e-4;
        for at in 0..3 {
            for ax in 0..3 {
                let gp = grad((at, ax, h));
                let gm = grad((at, ax, -h));
                for bt in 0..3 {
                    for by in 0..3 {
                        let fd = (gp[bt].get(by) - gm[bt].get(by)) / (2.0 * h);
                        let a = an[(3 * at + ax, 3 * bt + by)];
                        assert!(
                            (a - fd).abs() < 2e-5 + 1e-3 * fd.abs(),
                            "ECP d2 ({at},{ax})({bt},{by}): analytic {a} vs FD {fd}"
                        );
                    }
                }
            }
        }
    }
    use super::*;
    use crate::basis::EcpTerm;

    /// int Y Y dO = delta over l and m - validates the monomial-integral +
    /// table pipeline.
    #[test]
    fn spherical_harmonics_orthonormal_on_sphere() {
        let t = y_tables();
        for l in 0..=6usize {
            for (mi, yi) in t[l].iter().enumerate() {
                for (mj, yj) in t[l].iter().enumerate() {
                    let s = omega2((0, 0, 0), yi, yj);
                    let want = if mi == mj { 1.0 } else { 0.0 };
                    assert!((s - want).abs() < 1e-12, "l={l} {mi},{mj}: {s}");
                }
            }
        }
        let s = omega2((0, 0, 0), &t[2][1], &t[4][3]);
        assert!(s.abs() < 1e-12);
    }

    /// i~ against the exact i~_0 and the three-term recursion residual.
    #[test]
    fn scaled_bessel_consistency() {
        let mut buf = vec![0.0; 13];
        for &x in &[1e-6, 0.03, 0.7, 3.0, 12.0, 45.0, 300.0] {
            scaled_bessel(12, x, &mut buf);
            if x > 1e-4 {
                let i0 = (1.0 - (-2.0f64 * x).exp()) / (2.0 * x);
                assert!((buf[0] - i0).abs() < 1e-13 * (1.0 + i0), "x={x}");
            }
            for l in 1..11usize {
                let res = buf[l - 1] - buf[l + 1] - (2 * l + 1) as f64 / x * buf[l];
                let scale = buf[l - 1].abs().max(1e-300);
                assert!(
                    (res / scale).abs() < 1e-9,
                    "x={x} lam={l}: rel residual {}",
                    res / scale
                );
            }
            for l in 0..12usize {
                assert!(buf[l] >= buf[l + 1] - 1e-300, "x={x} lam={l}");
            }
        }
    }

    fn shell(l: u8, center: Vec3, exps: Vec<f64>, coefs: Vec<f64>) -> MolShell {
        MolShell {
            l,
            atom: 0,
            center,
            exps,
            coefs,
            sph_off: 0,
            cart_off: 0,
        }
    }

    /// Independent spherical-coordinates reference about the ECP center:
    /// radial Gauss-Legendre x product angular grid; the r^2 Jacobian tames
    /// every G94 radial law (n = 0, 1, 2). No Bessel expansion, no monomial
    /// algebra, no binomial shifts.
    fn spherical_reference(
        sha: &MolShell,
        ca: (i32, i32, i32),
        shb: &MolShell,
        cb: (i32, i32, i32),
        c: Vec3,
        chan_l: Option<usize>,
        terms: &[EcpTerm],
    ) -> f64 {
        let ytab = y_tables();
        let (nt, np) = (40, 81);
        let (cts, wts) = gauss_legendre(nt);
        let nr = 240;
        let (rx, rw) = gauss_legendre(nr);
        let rmax = 12.0;
        let chi = |sh: &MolShell, comp: (i32, i32, i32), pt: Vec3| -> f64 {
            let rr = pt - sh.center;
            let mut g = 0.0;
            for (i, &al) in sh.exps.iter().enumerate() {
                g += sh.coefs[i] * (-al * rr.norm2()).exp();
            }
            g * rr.x.powi(comp.0) * rr.y.powi(comp.1) * rr.z.powi(comp.2)
        };
        let mut total = 0.0;
        for (ir, &xr) in rx.iter().enumerate() {
            let r = rmax * 0.5 * (xr + 1.0);
            let wr = rw[ir] * rmax * 0.5;
            let mut u = 0.0;
            for t in terms {
                u += t.coef * r.powi(t.power - 2) * (-t.exp * r * r).exp();
            }
            if u == 0.0 {
                continue;
            }
            match chan_l {
                None => {
                    // Local channel: plain 3D integral in spherical coords.
                    let mut acc = 0.0;
                    for (it, &ct) in cts.iter().enumerate() {
                        let st = (1.0 - ct * ct).sqrt();
                        for ip in 0..np {
                            let phi = 2.0 * std::f64::consts::PI * ip as f64 / np as f64;
                            let om = Vec3::new(st * phi.cos(), st * phi.sin(), ct);
                            let wang = wts[it] * 2.0 * std::f64::consts::PI / np as f64;
                            let pt = c + om * r;
                            acc += wang * chi(sha, ca, pt) * chi(shb, cb, pt);
                        }
                    }
                    total += wr * r * r * u * acc;
                }
                Some(l) => {
                    // Projected channel: project both sides onto Y_lm at
                    // radius r by angular quadrature.
                    for mi in 0..2 * l + 1 {
                        let mut fa = 0.0;
                        let mut fb = 0.0;
                        for (it, &ct) in cts.iter().enumerate() {
                            let st = (1.0 - ct * ct).sqrt();
                            for ip in 0..np {
                                let phi = 2.0 * std::f64::consts::PI * ip as f64 / np as f64;
                                let om = Vec3::new(st * phi.cos(), st * phi.sin(), ct);
                                let wang = wts[it] * 2.0 * std::f64::consts::PI / np as f64;
                                let y = y_eval(&ytab[l][mi], om);
                                let pt = c + om * r;
                                fa += wang * y * chi(sha, ca, pt);
                                fb += wang * y * chi(shb, cb, pt);
                            }
                        }
                        total += wr * r * r * u * fa * fb;
                    }
                }
            }
        }
        total
    }

    /// Closed form for the s-s local Gaussian term (n = 2): the
    /// three-Gaussian product integral
    /// `int exp(-a|r-A|^2 - b|r-B|^2 - eta|r-C|^2) d3r`.
    #[test]
    fn type1_ss_gaussian_closed_form() {
        let a = Vec3::new(0.4, -0.3, 0.7);
        let b = Vec3::new(-0.5, 0.6, -0.2);
        let c = Vec3::new(0.1, 0.2, -0.3);
        let (alpha, beta, eta, coef) = (0.8, 1.1, 1.3, 2.0);
        let pot = EcpPotential {
            n_core: 0,
            lmax: 0,
            local: vec![EcpTerm {
                power: 2,
                exp: eta,
                coef,
            }],
            semilocal: vec![],
        };
        let sha = shell(0, a, vec![alpha], vec![1.0]);
        let shb = shell(0, b, vec![beta], vec![1.0]);
        let block = ecp_shell_pair(&sha, &shb, c, &pot);
        let p = alpha + beta + eta;
        let q = (a * alpha + b * beta + c * eta) * (1.0 / p);
        let w = alpha * a.norm2() + beta * b.norm2() + eta * c.norm2() - p * q.norm2();
        let want = coef * (-w).exp() * (std::f64::consts::PI / p).powf(1.5);
        assert!(
            (block[(0, 0)] - want).abs() < 1e-12 * want.abs().max(1.0),
            "{} vs closed form {want}",
            block[(0, 0)]
        );
    }

    /// Type-1 against the independent spherical-coordinates reference for
    /// s/p/d shells and all three G94 radial laws (n = 0, 1, 2).
    #[test]
    fn type1_matches_spherical_reference() {
        let a = Vec3::new(0.4, -0.3, 0.7);
        let b = Vec3::new(-0.5, 0.6, -0.2);
        let c = Vec3::new(0.1, 0.2, -0.3);
        let terms = vec![
            EcpTerm {
                power: 2,
                exp: 1.3,
                coef: 2.0,
            },
            EcpTerm {
                power: 1,
                exp: 0.9,
                coef: -0.7,
            },
            EcpTerm {
                power: 0,
                exp: 1.6,
                coef: 0.4,
            },
        ];
        let pot = EcpPotential {
            n_core: 0,
            lmax: 0,
            local: terms.clone(),
            semilocal: vec![],
        };
        for (la, lb) in [(0u8, 0u8), (1, 0), (1, 1), (2, 1)] {
            let sha = shell(la, a, vec![0.8, 2.2], vec![0.6, 0.5]);
            let shb = shell(lb, b, vec![1.1], vec![0.9]);
            let block = ecp_shell_pair(&sha, &shb, c, &pot);
            let comps_a = cart_components(la);
            let comps_b = cart_components(lb);
            for (i, &cai) in comps_a.iter().enumerate() {
                for (j, &cbj) in comps_b.iter().enumerate() {
                    let want = spherical_reference(&sha, cai, &shb, cbj, c, None, &terms);
                    assert!(
                        (block[(i, j)] - want).abs() < 1e-8,
                        "l=({la},{lb}) comp ({i},{j}): {} vs {want}",
                        block[(i, j)]
                    );
                }
            }
        }
    }

    /// Type-2 against the independent projector-quadrature reference for
    /// s/p projectors with s/p/d AOs off-center.
    #[test]
    fn type2_matches_projector_quadrature() {
        let a = Vec3::new(0.5, -0.2, 0.6);
        let b = Vec3::new(-0.4, 0.5, -0.3);
        let c = Vec3::new(0.0, 0.1, -0.2);
        let terms_s = vec![EcpTerm {
            power: 2,
            exp: 1.7,
            coef: 3.0,
        }];
        let terms_p = vec![EcpTerm {
            power: 1,
            exp: 1.1,
            coef: -1.2,
        }];
        let pot = EcpPotential {
            n_core: 0,
            lmax: 2,
            local: vec![],
            semilocal: vec![terms_s.clone(), terms_p.clone()],
        };
        for (la, lb) in [(0u8, 0u8), (1, 0), (2, 1)] {
            let sha = shell(la, a, vec![0.9], vec![0.8]);
            let shb = shell(lb, b, vec![1.3], vec![0.7]);
            let block = ecp_shell_pair(&sha, &shb, c, &pot);
            let comps_a = cart_components(la);
            let comps_b = cart_components(lb);
            for (i, &cai) in comps_a.iter().enumerate() {
                for (j, &cbj) in comps_b.iter().enumerate() {
                    let want = spherical_reference(&sha, cai, &shb, cbj, c, Some(0), &terms_s)
                        + spherical_reference(&sha, cai, &shb, cbj, c, Some(1), &terms_p);
                    assert!(
                        (block[(i, j)] - want).abs() < 5e-8,
                        "l=({la},{lb}) comp ({i},{j}): {} vs {want}",
                        block[(i, j)]
                    );
                }
            }
        }
    }
}
