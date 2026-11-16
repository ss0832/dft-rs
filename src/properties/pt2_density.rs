// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Relaxed one-particle density of a spin-component PT2 (double-hybrid)
//! energy on a semicanonical SCF reference.
//!
//! With the amplitudes `t` (antisymmetrized same-spin, direct
//! opposite-spin) and the spin-component weights `w`, the stationary
//! amplitude functional gives the unrelaxed occupied and virtual blocks
//! (spin orbitals)
//! `P_ab = 1/2 sum_(ijc) w t_ij^ac t_ij^bc`,
//! `P_ij = -1/2 sum_(kab) w t_ik^ab t_jk^ab`,
//! and the orbital gradient of the total energy `E_target[P] + E_PT2`
//! under an occupied-virtual rotation `x_ai` of the reference,
//! `g_ai = 2 F^target_ai + 2 (C^T G[P^(2)] C)_ai
//!        + 2 sum_(jbc) w/2 t_ij^bc <aj||bc> - 2 sum_(jkb) w/2 t_jk^ab <jk||ib>`,
//! where `G` is the Fock response kernel of the parent functional. The
//! parent orbital Hessian `A` gives the multipliers `z = -A^-1 g`, and the
//! relaxed density `P_ref + C P^(2) C^T + 1/2 sum z_ai (C_a C_i^T + C_i C_a^T)`
//! reproduces `dE/dlambda = Tr(D h_lambda)` for every one-electron
//! perturbation. Assumptions: integer aufbau occupations, PT2 over all
//! electrons (no frozen core), Hartree-Fock-like amplitudes without singles.

use crate::integrals::eri::EriTensor;
use crate::linalg::Matrix;
use crate::properties::pt2::Pt2Problem;
use crate::response::ResponseSolver;
use crate::scf::ScfResult;
use crate::xc::double_hybrid::Pt2Coefficients;
use crate::{DftError, Result};
use rayon::prelude::*;

/// Relaxed PT2 density and its parts, AO basis per spin.
#[derive(Clone, Debug)]
pub struct Pt2RelaxedDensity {
    /// Unrelaxed PT2 correction `C P^(2) C^T`.
    pub unrelaxed: [Matrix; 2],
    /// Orbital relaxation from the multipliers.
    pub relaxation: [Matrix; 2],
    /// Reference plus both corrections.
    pub relaxed: [Matrix; 2],
    /// Largest orbital-gradient element `|g_ai|`.
    pub gradient_norm: f64,
    /// `|Tr(P^(2) S)|` per spin (zero for the exact unrelaxed density).
    pub unrelaxed_trace: [f64; 2],
}

fn columns(c: &Matrix, start: usize, end: usize) -> Matrix {
    let mut out = Matrix::zeros(c.rows, end - start);
    for i in 0..c.rows {
        for j in start..end {
            out[(i, j - start)] = c[(i, j)];
        }
    }
    out
}

/// `(pq|rs)` with the four index sets given by coefficient matrices,
/// `[p][q][r][s]` row-major.
pub(crate) fn transform(eri: &EriTensor, c: [&Matrix; 4]) -> Vec<f64> {
    let n = eri.nao;
    let data = eri.as_slice();
    let [c1, c2, c3, c4] = c;
    let (n1, n2, n3, n4) = (c1.cols, c2.cols, c3.cols, c4.cols);
    // (p nu | lambda sigma)
    let first: Vec<f64> = (0..n1)
        .into_par_iter()
        .flat_map_iter(|p| {
            let mut out = vec![0.0; n * n * n];
            for mu in 0..n {
                let cp = c1[(mu, p)];
                if cp == 0.0 {
                    continue;
                }
                let block = &data[mu * n * n * n..(mu + 1) * n * n * n];
                for (o, v) in out.iter_mut().zip(block) {
                    *o += cp * v;
                }
            }
            out
        })
        .collect();
    // (p q | lambda sigma)
    let second: Vec<f64> = (0..n1)
        .into_par_iter()
        .flat_map_iter(|p| {
            let mut out = vec![0.0; n2 * n * n];
            for q in 0..n2 {
                for nu in 0..n {
                    let cq = c2[(nu, q)];
                    if cq == 0.0 {
                        continue;
                    }
                    let block = &first[(p * n + nu) * n * n..(p * n + nu + 1) * n * n];
                    for (o, v) in out[q * n * n..(q + 1) * n * n].iter_mut().zip(block) {
                        *o += cq * v;
                    }
                }
            }
            out
        })
        .collect();
    // (p q | r s)
    (0..n1 * n2)
        .into_par_iter()
        .flat_map_iter(|pq| {
            let block = &second[pq * n * n..(pq + 1) * n * n];
            let mut half = vec![0.0; n3 * n];
            for r in 0..n3 {
                for lambda in 0..n {
                    let cr = c3[(lambda, r)];
                    if cr == 0.0 {
                        continue;
                    }
                    for (o, v) in half[r * n..(r + 1) * n]
                        .iter_mut()
                        .zip(&block[lambda * n..(lambda + 1) * n])
                    {
                        *o += cr * v;
                    }
                }
            }
            let mut out = vec![0.0; n3 * n4];
            for r in 0..n3 {
                for s in 0..n4 {
                    out[r * n4 + s] = (0..n)
                        .map(|sigma| half[r * n + sigma] * c4[(sigma, s)])
                        .sum();
                }
            }
            out
        })
        .collect()
}

/// Unrelaxed second-order one-particle density in the semicanonical MO
/// frame. Reused by the stationary energy and transition propagator.
pub(crate) fn unrelaxed_mo_density(
    problem: &Pt2Problem<f64>,
    amplitudes: &[Vec<f64>; 3],
    coefficients: Pt2Coefficients,
) -> Result<[Matrix; 2]> {
    coefficients.validate()?;
    let occupied = problem.occupied;
    let nmo = [problem.energies[0].len(), problem.energies[1].len()];
    if occupied.iter().zip(nmo).any(|(&o, n)| o > n) {
        return Err(DftError::InvalidInput(
            "PT2 occupations exceed the orbital space".into(),
        ));
    }
    let virtuals = [nmo[0] - occupied[0], nmo[1] - occupied[1]];
    let (oa, ob, va, vb) = (occupied[0], occupied[1], virtuals[0], virtuals[1]);
    let expected = [oa * va * oa * va, oa * va * ob * vb, ob * vb * ob * vb];
    if amplitudes
        .iter()
        .zip(expected)
        .any(|(t, e)| t.len() != e || t.iter().any(|v| !v.is_finite()))
    {
        return Err(DftError::InvalidInput(
            "PT2 density amplitude dimensions disagree".into(),
        ));
    }
    let (css, cos) = (coefficients.same_spin, coefficients.opposite_spin);
    let same = |s: usize, i: usize, a: usize, j: usize, b: usize| -> f64 {
        let (o, v) = (occupied[s], virtuals[s]);
        amplitudes[if s == 0 { 0 } else { 2 }][((i * v + a) * o + j) * v + b]
    };
    let mixed = |i: usize, a: usize, j: usize, b: usize| -> f64 {
        amplitudes[1][((i * va + a) * ob + j) * vb + b]
    };
    // Unrelaxed blocks.
    let mut p_oo = [Matrix::zeros(oa, oa), Matrix::zeros(ob, ob)];
    let mut p_vv = [Matrix::zeros(va, va), Matrix::zeros(vb, vb)];
    for (s, (vv, oo)) in p_vv.iter_mut().zip(p_oo.iter_mut()).enumerate() {
        let (o, v) = (occupied[s], virtuals[s]);
        for a in 0..v {
            for c in 0..v {
                let mut sum = 0.0;
                for i in 0..o {
                    for j in 0..o {
                        for b in 0..v {
                            sum += 0.5 * css * same(s, i, a, j, b) * same(s, i, c, j, b);
                        }
                    }
                }
                vv[(a, c)] = sum;
            }
        }
        for k in 0..o {
            for i in 0..o {
                let mut sum = 0.0;
                for j in 0..o {
                    for a in 0..v {
                        for b in 0..v {
                            sum -= 0.5 * css * same(s, i, a, j, b) * same(s, k, a, j, b);
                        }
                    }
                }
                oo[(k, i)] = sum;
            }
        }
    }
    for a in 0..va {
        for c in 0..va {
            let mut sum = 0.0;
            for i in 0..oa {
                for j in 0..ob {
                    for b in 0..vb {
                        sum += cos * mixed(i, a, j, b) * mixed(i, c, j, b);
                    }
                }
            }
            p_vv[0][(a, c)] += sum;
        }
    }
    for k in 0..oa {
        for i in 0..oa {
            let mut sum = 0.0;
            for a in 0..va {
                for j in 0..ob {
                    for b in 0..vb {
                        sum -= cos * mixed(i, a, j, b) * mixed(k, a, j, b);
                    }
                }
            }
            p_oo[0][(k, i)] += sum;
        }
    }
    for a in 0..vb {
        for c in 0..vb {
            let mut sum = 0.0;
            for i in 0..oa {
                for b in 0..va {
                    for j in 0..ob {
                        sum += cos * mixed(i, b, j, a) * mixed(i, b, j, c);
                    }
                }
            }
            p_vv[1][(a, c)] += sum;
        }
    }
    for k in 0..ob {
        for i in 0..ob {
            let mut sum = 0.0;
            for j in 0..oa {
                for b in 0..va {
                    for a in 0..vb {
                        sum -= cos * mixed(j, b, i, a) * mixed(j, b, k, a);
                    }
                }
            }
            p_oo[1][(k, i)] += sum;
        }
    }
    Ok(std::array::from_fn(|s| {
        let mut p = Matrix::zeros(nmo[s], nmo[s]);
        for i in 0..occupied[s] {
            for j in 0..occupied[s] {
                p[(i, j)] = p_oo[s][(i, j)];
            }
        }
        for a in 0..virtuals[s] {
            for b in 0..virtuals[s] {
                p[(occupied[s] + a, occupied[s] + b)] = p_vv[s][(a, b)];
            }
        }
        p
    }))
}

/// Relaxed density of the spin-component PT2 energy with the amplitudes of
/// `problem`. `target_delta` is `F_target - F_parent` (AO, per spin) at the
/// reference density, zero when the energy functional is the parent; the
/// solver carries the parent orbital Hessian.
pub fn relaxed_density(
    scf: &ScfResult,
    problem: &Pt2Problem<f64>,
    amplitudes: &[Vec<f64>; 3],
    coefficients: Pt2Coefficients,
    eri: &EriTensor,
    solver: &ResponseSolver,
    target_delta: &[Matrix; 2],
) -> Result<Pt2RelaxedDensity> {
    let n = eri.nao;
    let occupied = problem.occupied;
    let nmo = [problem.energies[0].len(), problem.energies[1].len()];
    let virtuals = [nmo[0] - occupied[0], nmo[1] - occupied[1]];
    let (oa, ob, va, vb) = (occupied[0], occupied[1], virtuals[0], virtuals[1]);
    let expected = [oa * va * oa * va, oa * va * ob * vb, ob * vb * ob * vb];
    if amplitudes.iter().zip(expected).any(|(t, e)| t.len() != e)
        || scf.mo_coeff.iter().any(|c| c.rows != n)
    {
        return Err(DftError::InvalidInput(
            "PT2 amplitudes, orbitals and integrals of the relaxed density disagree".into(),
        ));
    }
    let (css, cos) = (coefficients.same_spin, coefficients.opposite_spin);
    let co = [
        columns(&scf.mo_coeff[0], 0, oa),
        columns(&scf.mo_coeff[1], 0, ob),
    ];
    let cv = [
        columns(&scf.mo_coeff[0], oa, nmo[0]),
        columns(&scf.mo_coeff[1], ob, nmo[1]),
    ];
    let same = |s: usize, i: usize, a: usize, j: usize, b: usize| -> f64 {
        let (o, v) = (occupied[s], virtuals[s]);
        amplitudes[if s == 0 { 0 } else { 2 }][((i * v + a) * o + j) * v + b]
    };
    // Opposite spin: i, a alpha; j, b beta.
    let mixed = |i: usize, a: usize, j: usize, b: usize| -> f64 {
        amplitudes[1][((i * va + a) * ob + j) * vb + b]
    };
    let mo_density = unrelaxed_mo_density(problem, amplitudes, coefficients)?;
    let unrelaxed: [Matrix; 2] = std::array::from_fn(|s| {
        scf.mo_coeff[s]
            .matmul(&mo_density[s])
            .matmul_transpose_seq(&scf.mo_coeff[s])
    });
    let unrelaxed_trace: [f64; 2] =
        std::array::from_fn(|s| unrelaxed[s].frobenius_dot(&scf.overlap).abs());
    // Orbital gradient.
    let response = solver.fock_response(&unrelaxed)?;
    let mut gradient: [Matrix; 2] = std::array::from_fn(|s| {
        let mut f = scf.fock[s].clone();
        for ((x, d), r) in f
            .as_mut_slice()
            .iter_mut()
            .zip(target_delta[s].as_slice())
            .zip(response[s].as_slice())
        {
            *x = 2.0 * (*x + d + r);
        }
        cv[s].transpose_matmul(&f.matmul(&co[s]))
    });
    for (s, g) in gradient.iter_mut().enumerate() {
        let t = 1 - s;
        let (o, v, ot, vt) = (occupied[s], virtuals[s], occupied[t], virtuals[t]);
        // (a b | j c), same spin and with the other spin's (j c).
        let vvov_same = transform(eri, [&cv[s], &cv[s], &co[s], &cv[s]]);
        let vvov_other = transform(eri, [&cv[s], &cv[s], &co[t], &cv[t]]);
        // (j i | k b), same spin and with the other spin's (k b).
        let ooov_same = transform(eri, [&co[s], &co[s], &co[s], &cv[s]]);
        let ooov_other = transform(eri, [&co[s], &co[s], &co[t], &cv[t]]);
        let opposite = |own_i: usize, own_a: usize, other_j: usize, other_b: usize| -> f64 {
            if s == 0 {
                mixed(own_i, own_a, other_j, other_b)
            } else {
                mixed(other_j, other_b, own_i, own_a)
            }
        };
        for a in 0..v {
            for i in 0..o {
                let mut value = 0.0;
                for j in 0..o {
                    for b in 0..v {
                        for c in 0..v {
                            value += 2.0
                                * css
                                * same(s, i, b, j, c)
                                * vvov_same[((a * v + b) * o + j) * v + c];
                        }
                    }
                }
                for j in 0..ot {
                    for b in 0..v {
                        for c in 0..vt {
                            value += 2.0
                                * cos
                                * opposite(i, b, j, c)
                                * vvov_other[((a * v + b) * ot + j) * vt + c];
                        }
                    }
                }
                for j in 0..o {
                    for k in 0..o {
                        for b in 0..v {
                            value -= 2.0
                                * css
                                * same(s, j, a, k, b)
                                * ooov_same[((j * o + i) * o + k) * v + b];
                        }
                    }
                }
                for j in 0..o {
                    for k in 0..ot {
                        for b in 0..vt {
                            value -= 2.0
                                * cos
                                * opposite(j, a, k, b)
                                * ooov_other[((j * o + i) * ot + k) * vt + b];
                        }
                    }
                }
                g[(a, i)] += value;
            }
        }
    }
    let gradient_norm = gradient
        .iter()
        .flat_map(|g| g.as_slice())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    // Multipliers z = -A^-1 g on the occupied-virtual rotations of the
    // solver (one shared spin block for a restricted reference).
    let spins = if scf.unrestricted { 2 } else { 1 };
    let mut rhs = Vec::new();
    for g in gradient.iter().take(spins) {
        for a in 0..g.rows {
            for i in 0..g.cols {
                rhs.push(if spins == 1 {
                    0.5 * (g[(a, i)] + gradient[1][(a, i)])
                } else {
                    g[(a, i)]
                });
            }
        }
    }
    let z = solver.solve_ov(&rhs)?;
    let delta = solver.delta_density(&z);
    let relaxation: [Matrix; 2] = std::array::from_fn(|s| {
        let mut m = delta[s].clone();
        m.as_mut_slice().iter_mut().for_each(|x| *x *= 0.5);
        m
    });
    let relaxed: [Matrix; 2] = std::array::from_fn(|s| {
        let mut p = scf.density[s].clone();
        for ((x, u), r) in p
            .as_mut_slice()
            .iter_mut()
            .zip(unrelaxed[s].as_slice())
            .zip(relaxation[s].as_slice())
        {
            *x += u + r;
        }
        p
    });
    Ok(Pt2RelaxedDensity {
        unrelaxed,
        relaxation,
        relaxed,
        gradient_norm,
        unrelaxed_trace,
    })
}
