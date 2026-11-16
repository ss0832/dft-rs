// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Length-gauge rotatory strengths with London orbitals.
//!
//! A London orbital carries `exp(-i/2 (B x (R_n - O)) . r)`. For a magnetic
//! field oscillating at the frequency `w` the basis moves with the field,
//! which adds `-i <m| d/dt |n> = i w W` to the first-order operator, with
//! the real ket-centre matrix
//!
//! ```text
//! W_mn = (1/2) [(R_n - O) x <m| r - O |n>].
//! ```
//!
//! With the static London property gradient `g_ai = Fhat_ai - e_i Shat_ai`
//! of the magnetic response equations (the explicit London derivative of
//! the Kohn-Sham matrix, the exchange of the occupied-occupied overlap
//! response and the overlap derivative, all with the factor `i` removed),
//! the residue of the electric dipole - magnetic field response at the
//! excitation `n` is
//!
//! ```text
//! R_n = d_n . [ g . (X - Y)_n + w_n W . (X + Y)_n ],   d_n = r . (X + Y)_n.
//! ```
//!
//! For one common origin `g = -L / 2` and `W = 0`, which is the usual
//! length-gauge expression. The second term changes by `-(w / 2) a x d_n`
//! when the origin `O` moves by `a`, which is orthogonal to `d_n`.

use super::krylov::ResponseSolverConfig;
use super::tddft::{TdaSolver, TdaSpin, TdaState};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::properties::magnetic::{magnetic_density_response, Gauge};
use crate::properties::nmr::{
    angular_momentum_operator, ao_centres, curl_terms, gradient_operator,
};
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

/// Rotatory strengths of one excited state.
#[derive(Clone, Debug)]
pub struct LondonRotatoryStrength {
    pub energy: f64,
    /// `<0| r - O |n>`.
    pub transition_dipole: [f64; 3],
    /// `g . (X - Y)`: the static London property gradient on the state.
    pub static_part: [f64; 3],
    /// `w W . (X + Y)`: the part of the moving basis.
    pub centre_part: [f64; 3],
    /// Length gauge with London orbitals.
    pub london: f64,
    /// Length gauge with the common origin `O`; a paired state carries
    /// `X - Y` on the magnetic moment.
    pub common_origin: f64,
    /// Velocity gauge; a paired state carries `X - Y` on both moments.
    pub velocity: f64,
}

#[derive(Clone, Debug)]
pub struct LondonRotatoryStrengths {
    pub states: Vec<LondonRotatoryStrength>,
    /// Largest relative residual of the static magnetic response equations.
    pub magnetic_residual: f64,
}

/// Rotatory strengths of `states` (one spin manifold of `scf`) in atomic
/// units, with London phases written about `origin`.
#[allow(clippy::too_many_arguments)]
pub fn london_rotatory_strengths(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    states: &[TdaState],
    origin: Vec3,
    config: &ResponseSolverConfig,
) -> Result<LondonRotatoryStrengths> {
    let Some(first) = states.first() else {
        return Ok(LondonRotatoryStrengths {
            states: Vec::new(),
            magnetic_residual: 0.0,
        });
    };
    let spin = first.spin;
    if states.iter().any(|s| s.spin != spin) {
        return Err(DftError::InvalidInput(
            "the states of one request share one spin manifold".into(),
        ));
    }
    // Spin blocks of the amplitudes and their factors: the alpha block of a
    // spin-adapted singlet stands for both spins.
    let blocks: &[(usize, f64)] = match spin {
        TdaSpin::Singlet => &[(0, std::f64::consts::SQRT_2)],
        TdaSpin::SpinResolved => &[(0, 1.0), (1, 1.0)],
        TdaSpin::Triplet => &[],
    };
    if blocks.is_empty() {
        // A triplet of a closed shell has no electric or magnetic dipole
        // transition moment.
        return Ok(LondonRotatoryStrengths {
            states: states
                .iter()
                .map(|state| LondonRotatoryStrength {
                    energy: state.energy,
                    transition_dipole: [0.0; 3],
                    static_part: [0.0; 3],
                    centre_part: [0.0; 3],
                    london: 0.0,
                    common_origin: 0.0,
                    velocity: 0.0,
                })
                .collect(),
            magnetic_residual: 0.0,
        });
    }
    let response = magnetic_density_response(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        Gauge::Giao,
        origin,
        config,
    )?;
    let magnetic_residual = response.residuals.iter().copied().fold(0.0_f64, f64::max);
    let solver = TdaSolver::new(mol, mb, functional, grid, scf, opts, spin)?;
    let nao = mb.nao_sph;
    let dipoles = crate::integrals::dipole_matrices(mb, origin);
    let momentum = angular_momentum_operator(mb, origin);
    let gradient = gradient_operator(mb);
    let centre = ao_centres(mb);
    let ket: [Matrix; 3] = std::array::from_fn(|axis| {
        let (j, k, _) = curl_terms(axis)[0];
        let mut out = Matrix::zeros(nao, nao);
        for m in 0..nao {
            for p in 0..nao {
                let v = centre[p] - origin;
                out[(m, p)] = 0.5 * (v.get(j) * dipoles[k][(m, p)] - v.get(k) * dipoles[j][(m, p)]);
            }
        }
        out
    });
    // Virtual-occupied blocks in the order of the amplitudes, per axis, with
    // the spin factor.
    let occupied = [scf.n_alpha, scf.n_beta];
    let project = |operators: &[Matrix; 3]| -> [Vec<f64>; 3] {
        std::array::from_fn(|axis| {
            let mut out = Vec::new();
            for &(s, factor) in blocks {
                let c = &scf.mo_coeff[s];
                let no = occupied[s];
                let mo = c.transpose_matmul_seq(&operators[axis].matmul_seq(c));
                for a in 0..c.cols - no {
                    for i in 0..no {
                        out.push(factor * mo[(no + a, i)]);
                    }
                }
            }
            out
        })
    };
    let electric = project(&dipoles);
    let centred = project(&ket);
    let angular = project(&momentum);
    let velocity = project(&gradient);
    let property: [Vec<f64>; 3] = std::array::from_fn(|axis| {
        let mut out = Vec::new();
        for &(s, factor) in blocks {
            out.extend(
                response.value.gradient_hat[s][axis]
                    .iter()
                    .map(|g| factor * g),
            );
        }
        out
    });
    let dot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    let mut out = Vec::with_capacity(states.len());
    for state in states {
        let mut plus = Vec::with_capacity(solver.dim());
        for &(s, _) in blocks {
            let x = &state.x[s];
            for a in 0..x.rows {
                for i in 0..x.cols {
                    plus.push(x[(a, i)]);
                }
            }
        }
        if plus.len() != solver.dim() || plus.len() != property[0].len() {
            return Err(DftError::InvalidInput(
                "the excited state does not belong to this reference".into(),
            ));
        }
        // (A + B)(X + Y) = w (X - Y).
        let minus: Vec<f64> = if state.tamm_dancoff {
            plus.clone()
        } else {
            if !(state.energy > 0.0) {
                return Err(DftError::InvalidInput(
                    "a paired excited state needs a positive excitation energy".into(),
                ));
            }
            solver
                .apb_vec(&plus)?
                .iter()
                .map(|v| v / state.energy)
                .collect()
        };
        let transition_dipole: [f64; 3] = std::array::from_fn(|a| dot(&electric[a], &plus));
        let static_part: [f64; 3] = std::array::from_fn(|a| dot(&property[a], &minus));
        let centre_part: [f64; 3] = std::array::from_fn(|a| state.energy * dot(&centred[a], &plus));
        let l: [f64; 3] = std::array::from_fn(|a| dot(&angular[a], &minus));
        let p: [f64; 3] = std::array::from_fn(|a| dot(&velocity[a], &minus));
        let mut london = 0.0;
        let mut common_origin = 0.0;
        let mut velocity_gauge = 0.0;
        for a in 0..3 {
            london += transition_dipole[a] * (static_part[a] + centre_part[a]);
            // m = -(1/2) L in atomic units.
            common_origin -= 0.5 * transition_dipole[a] * l[a];
            velocity_gauge += 0.5 * p[a] * l[a];
        }
        out.push(LondonRotatoryStrength {
            energy: state.energy,
            transition_dipole,
            static_part,
            centre_part,
            london,
            common_origin,
            velocity: if state.energy.abs() > 1e-12 {
                velocity_gauge / state.energy
            } else {
                0.0
            },
        });
    }
    Ok(LondonRotatoryStrengths {
        states: out,
        magnetic_residual,
    })
}
