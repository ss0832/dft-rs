// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Truncated second-order correlated induction.
//!
//! For each monomer, the same-direction term is the quadratic field
//! coefficient of the doubles-only MP2 expression in the occupied and
//! virtual spaces of the one-body operator F + lambda Omega. The crossed
//! term uses the order-W^2 unrelaxed density of one monomer in the ordinary
//! uncoupled induction response of the other. The double-excitation
//! projection retains the diagrams in Hohenstein and Sherrill, Eqs. 48
//! and 58 (DOI: 10.1063/1.3451077).

use crate::linalg::Matrix;
use crate::mcscf::ci::{ActiveHamiltonian, CiSpace};
use crate::Result;

pub(super) struct Corrections {
    pub(super) rotation: Matrix,
    pub(super) induction: f64,
}

fn commutator(a: &Matrix, b: &Matrix) -> Matrix {
    let mut ab = a.matmul(b);
    let ba = b.matmul(a);
    for (x, y) in ab.as_mut_slice().iter_mut().zip(ba.as_slice()) {
        *x -= y;
    }
    ab
}

fn one_axis(eri: &[f64], r: &Matrix, axis: usize, result: &mut [f64]) {
    let n = r.rows;
    let stride = n.pow(3 - axis as u32);
    for (i, value) in result.iter_mut().enumerate() {
        let p = (i / stride) % n;
        let first = i - p * stride;
        *value += (0..n)
            .map(|q| r[(q, p)] * eri[first + q * stride])
            .sum::<f64>();
    }
}

fn rotated_eri(g: &[f64], u: &Matrix, second: &Matrix) -> [Vec<f64>; 2] {
    let mut first_eri = vec![0.0; g.len()];
    let mut second_eri = vec![0.0; g.len()];
    let mut axis_first = vec![0.0; g.len()];
    for a in 0..4 {
        axis_first.fill(0.0);
        one_axis(g, u, a, &mut axis_first);
        one_axis(g, second, a, &mut second_eri);
        for b in 0..a {
            one_axis(&axis_first, u, b, &mut second_eri);
        }
        for (x, y) in first_eri.iter_mut().zip(&axis_first) {
            *x += y;
        }
    }
    [first_eri, second_eri]
}

pub(super) fn monomer(
    space: &CiSpace,
    g: &[f64],
    field: &Matrix,
    energies: &[f64],
    gaps: &[f64],
    t: &[f64],
    reference: &[f64],
) -> Result<Corrections> {
    let no = space.n_alpha;
    let n = space.n_orb;
    let nv = n - no;
    let mut first = Matrix::zeros(n, n);
    let mut rotation = Matrix::zeros(nv, no);
    let mut block = field.clone();
    let mut off = Matrix::zeros(n, n);
    for a in no..n {
        for i in 0..no {
            let value = field[(a, i)] / (energies[i] - energies[a]);
            rotation[(a - no, i)] = value;
            first[(a, i)] = value;
            first[(i, a)] = -value;
            block[(a, i)] = 0.0;
            block[(i, a)] = 0.0;
            off[(a, i)] = field[(a, i)];
            off[(i, a)] = field[(i, a)];
        }
    }
    let mut second = first.matmul(&first);
    for x in second.as_mut_slice() {
        *x *= 0.5;
    }
    let z = commutator(&block, &first);
    for a in no..n {
        for i in 0..no {
            let value = z[(a, i)] / (energies[i] - energies[a]);
            second[(a, i)] += value;
            second[(i, a)] -= value;
        }
    }
    let [first_eri, second_eri] = rotated_eri(g, &first, &second);
    let b1 = space.sigma(
        &ActiveHamiltonian {
            n,
            constant: 0.0,
            h: Matrix::zeros(n, n),
            g: first_eri,
        },
        reference,
    )?;
    let b2 = space.sigma(
        &ActiveHamiltonian {
            n,
            constant: 0.0,
            h: Matrix::zeros(n, n),
            g: second_eri,
        },
        reference,
    )?;
    let mut block2 = commutator(&off, &first);
    for x in block2.as_mut_slice() {
        *x *= 0.5;
    }
    let centered_fock = |h: Matrix| ActiveHamiltonian {
        n,
        constant: -2.0 * (0..no).map(|i| h[(i, i)]).sum::<f64>(),
        h,
        g: vec![0.0; g.len()],
    };
    let g1t = space.sigma(&centered_fock(block), t)?;
    let g2t = space.sigma(&centered_fock(block2), t)?;

    let mut induction = 0.0;
    for i in 1..t.len() {
        let occupied = (1u64 << no) - 1;
        let rank: u32 = space
            .occupations(i)
            .iter()
            .map(|&mask| (mask & !occupied).count_ones())
            .sum();
        if rank == 2 {
            let x = b1[i] + g1t[i];
            induction -= x * x / gaps[i];
            induction += t[i] * (2.0 * b2[i] + g2t[i]);
        }
    }
    Ok(Corrections {
        rotation,
        induction,
    })
}
