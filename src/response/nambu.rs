// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Full finite-temperature particle/hole density and interband pairing
//! response on a supplied normal electronic space. This kinematic provider
//! does not replace a self-consistent normal/pairing energy functional.

use super::thermal_matrix::{fermi_matrix_taylor, ThermalMatrixTaylor};
use super::ResponseSolverConfig;
use crate::periodic::BlochMatrix;
use crate::{DftError, Result};

mod causal;
pub use causal::{nambu_causal_response, NambuCausalResponse, NambuProbe, NambuReference};

#[derive(Clone, Debug)]
pub struct NambuTaylor {
    pub thermal: ThermalMatrixTaylor,
    pub normal_density: Vec<BlochMatrix>,
    /// chi_ij=<c_i c_j>=-R_eh, antisymmetric in the full electronic space.
    pub anomalous_density: Vec<BlochMatrix>,
    pub electron_number: Vec<f64>,
    pub particle_hole_defects: Vec<f64>,
    pub pairing_antisymmetry_defects: Vec<f64>,
    pub number_imaginary_defects: Vec<f64>,
    pub hamiltonian: Vec<BlochMatrix>,
    pub overlap: Vec<BlochMatrix>,
}

/// Overlay particle/hole space on any full-rank normal electronic space.
/// Normal H/S are Hermitian and Delta=-Delta^T in that full space (including
/// spin). Nambu order is all electrons, then their conjugate holes.
///
/// H_N=[[h-mu*S,Delta],[Delta^H,-(h-mu*S)*]], S_N=diag(S,S*).
/// Inputs and outputs are factorial-normalized coefficients. kT is Hartree.
/// A general matrix Delta is retained, without a band-diagonal assumption.
/// The ensemble is grand canonical at the supplied chemical-potential path;
/// no normal-density or pairing-potential self-consistency is implied.
pub fn nambu_density_taylor(
    normal: &[BlochMatrix],
    pairing: &[BlochMatrix],
    overlap: &[BlochMatrix],
    chemical_potential: &[f64],
    temperature_energy: &[f64],
    order: usize,
    overlap_cutoff: f64,
    config: &ResponseSolverConfig,
) -> Result<NambuTaylor> {
    let (h, s) = nambu_matrices(normal, pairing, overlap, chemical_potential, order)?;
    let n = normal[0].rows();
    let dimension = h[0].rows();
    let thermal = fermi_matrix_taylor(&h, &s, temperature_energy, order, overlap_cutoff, config)?;
    let mut rho = Vec::new();
    let mut chi = Vec::new();
    let mut particle_hole = Vec::new();
    let mut antisymmetry = Vec::new();
    let mut numbers = Vec::new();
    let mut number_imaginary = Vec::new();
    for k in 0..=order {
        let density = &thermal.density[k];
        let inverse = &thermal.metric_inverse[k];
        let mut r = BlochMatrix::zeros(n, n);
        let mut c = r.clone();
        let mut ph = 0.0_f64;
        let mut anti = 0.0_f64;
        for i in 0..dimension {
            for j in 0..dimension {
                let (a, b) = ((i + n) % dimension, (j + n) % dimension);
                ph = ph
                    .max((density.re[(i, j)] + density.re[(a, b)] - inverse.re[(i, j)]).abs())
                    .max((density.im[(i, j)] - density.im[(a, b)] - inverse.im[(i, j)]).abs());
            }
        }
        for i in 0..n {
            for j in 0..n {
                r.re[(i, j)] = density.re[(i, j)];
                r.im[(i, j)] = density.im[(i, j)];
                c.re[(i, j)] = -density.re[(i, n + j)];
                c.im[(i, j)] = -density.im[(i, n + j)];
                anti = anti
                    .max((density.re[(i, n + j)] + density.re[(j, n + i)]).abs())
                    .max((density.im[(i, n + j)] + density.im[(j, n + i)]).abs());
            }
        }
        let (mut number, mut imag) = (0.0, 0.0);
        for l in 0..=k {
            for i in 0..n {
                for j in 0..n {
                    let a = &s[l];
                    let b = &thermal.density[k - l];
                    number += a.re[(i, j)] * b.re[(j, i)] - a.im[(i, j)] * b.im[(j, i)];
                    imag += a.re[(i, j)] * b.im[(j, i)] + a.im[(i, j)] * b.re[(j, i)];
                }
            }
        }
        rho.push(r);
        chi.push(c);
        particle_hole.push(ph);
        antisymmetry.push(anti);
        numbers.push(number);
        number_imaginary.push(imag.abs());
    }
    if particle_hole
        .iter()
        .chain(&antisymmetry)
        .chain(&number_imaginary)
        .any(|x| !x.is_finite() || *x > config.acceptance())
    {
        return Err(DftError::LinearAlgebra(
            "Nambu particle-hole/pairing/number residual exceeds acceptance".into(),
        ));
    }
    Ok(NambuTaylor {
        thermal,
        normal_density: rho,
        anomalous_density: chi,
        electron_number: numbers,
        particle_hole_defects: particle_hole,
        pairing_antisymmetry_defects: antisymmetry,
        number_imaginary_defects: number_imaginary,
        hamiltonian: h,
        overlap: s,
    })
}

fn nambu_matrices(
    normal: &[BlochMatrix],
    pairing: &[BlochMatrix],
    overlap: &[BlochMatrix],
    chemical_potential: &[f64],
    order: usize,
) -> Result<(Vec<BlochMatrix>, Vec<BlochMatrix>)> {
    let n = normal.first().map_or(0, BlochMatrix::rows);
    let dimension = n
        .checked_mul(2)
        .ok_or_else(|| DftError::InvalidInput("Nambu dimension overflow".into()))?;
    if n == 0
        || order > 4
        || normal.len() > order + 1
        || pairing.is_empty()
        || pairing.len() > order + 1
        || overlap.is_empty()
        || overlap.len() > order + 1
        || chemical_potential.is_empty()
        || chemical_potential.len() > order + 1
        || chemical_potential.iter().any(|x| !x.is_finite())
        || normal.iter().chain(pairing).chain(overlap).any(|a| {
            a.rows() != n
                || a.cols() != n
                || a.im.rows != n
                || a.im.cols != n
                || a.re
                    .as_slice()
                    .iter()
                    .chain(a.im.as_slice())
                    .any(|x| !x.is_finite())
        })
    {
        return Err(DftError::InvalidInput("Nambu density needs finite matching normal/pairing/overlap matrices, chemical potential and order zero through four".into()));
    }
    if normal
        .iter()
        .chain(overlap)
        .any(|a| a.hermiticity_error() > 1e-10)
    {
        return Err(DftError::InvalidInput(
            "normal Nambu Hamiltonian and overlap must be Hermitian without projection".into(),
        ));
    }
    for a in pairing {
        for i in 0..n {
            for j in 0..n {
                if (a.re[(i, j)] + a.re[(j, i)])
                    .abs()
                    .max((a.im[(i, j)] + a.im[(j, i)]).abs())
                    > 1e-10
                {
                    return Err(DftError::InvalidInput("full electronic-space pairing must obey Delta=-Delta^T, including spin indices".into()));
                }
            }
        }
    }
    let mut h = vec![BlochMatrix::zeros(dimension, dimension); order + 1];
    let mut s = h.clone();
    for k in 0..=order {
        for i in 0..n {
            for j in 0..n {
                let mut re = normal.get(k).map_or(0.0, |a| a.re[(i, j)]);
                let mut im = normal.get(k).map_or(0.0, |a| a.im[(i, j)]);
                for l in 0..=k {
                    let mu = chemical_potential.get(l).copied().unwrap_or(0.0);
                    if let Some(a) = overlap.get(k - l) {
                        re -= mu * a.re[(i, j)];
                        im -= mu * a.im[(i, j)];
                    }
                }
                h[k].re[(i, j)] = re;
                h[k].im[(i, j)] = im;
                h[k].re[(n + i, n + j)] = -re;
                h[k].im[(n + i, n + j)] = im;
                if let Some(a) = pairing.get(k) {
                    h[k].re[(i, n + j)] = a.re[(i, j)];
                    h[k].im[(i, n + j)] = a.im[(i, j)];
                    h[k].re[(n + j, i)] = a.re[(i, j)];
                    h[k].im[(n + j, i)] = -a.im[(i, j)];
                }
                if let Some(a) = overlap.get(k) {
                    s[k].re[(i, j)] = a.re[(i, j)];
                    s[k].im[(i, j)] = a.im[(i, j)];
                    s[k].re[(n + i, n + j)] = a.re[(i, j)];
                    s[k].im[(n + i, n + j)] = -a.im[(i, j)];
                }
            }
        }
    }
    Ok((h, s))
}
