// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Magnetic circular dichroism of a closed-shell molecule: Faraday B terms
//! (and A terms of degenerate states) from the complete manifold of singlet
//! Tamm-Dancoff states, routed through the common driver.
//!
//! With real amplitudes `X` (unit norm over the orbital pairs), the electric
//! dipole `mu = -r` and the magnetic dipole `m = i mh`, `mh = (r x grad)/2`:
//!
//! ```text
//! <0|mu|J> = sqrt(2) sum_ai X_ai mu_ai,     <K|m|0> = i sqrt(2) sum_ai X_ai mh_ai
//! <J|O|K> = sum_iab X^J_ai X^K_bi O_ab - sum_aij X^J_ai X^K_aj O_ji + delta_JK <0|O|0>
//! B_J = sum_K mh_K0 . (mu_0J x (mu_JK - delta_JK mu_00)) / w_K
//!     + sum_(K != J) mh_JK . (mu_0J x mu_0K) / (w_K - w_J)
//! A_J(canonical) = (1/2) sum_(K in the level of J) mh_JK . (mu_0J x mu_0K)
//! a_terms[J] = -A_J(canonical), faraday_a_terms[J] = A_J(canonical)
//! ```
//!
//! Both B sums here run over excited states. The ground-state part of the
//! final-state response therefore supplies the explicit dipole difference
//! in the first sum. Adding the permanent dipole again loses that term.
//! Canonical A/B use the exact-state convention of Faber et al., J. Chem.
//! Phys. 153, 114105 (2020), DOI 10.1063/5.0013398, Eqs. (3)--(5).
//! B is the cyclic sum of d Im[mu_a(0J) mu_b(J0)]/dB_c, before orientation
//! and circular-polarization prefactors. Legacy a_terms is the signed first
//! line moment; the coefficient of a derivative in observation frequency
//! is its negative, returned separately as faraday_a_terms.
//!
//! States closer in energy than `degeneracy_tolerance` form one level: they
//! are left out of the second sum of `B` and give the `A` term instead. The
//! sums run over every excited state of the Tamm-Dancoff matrix, which is
//! diagonalized completely; that limits the size of the molecule.

use super::controls::Controls;
use super::method::run_molecular_reference;
use super::molecular::{reference_report, setup, units};
use super::value::Value;
use super::System;
use crate::integrals::dipole_matrices;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::properties::nmr::angular_momentum_operator;
use crate::response::tddft::{TdaSolver, TdaSpin};
use crate::{DftError, Result};

#[path = "mcd/thermal.rs"]
mod thermal;

#[path = "mcd/complex.rs"]
mod complex;

#[path = "mcd/casci.rs"]
mod casci;

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `<J|O|K>` without the ground-state expectation value, for the amplitudes
/// `xj`, `xk` (rows virtual, columns occupied) and the orbital matrix `o`.
fn between(xj: &Matrix, xk: &Matrix, o: &Matrix, no: usize) -> f64 {
    let nv = xj.rows;
    let mut sum = 0.0;
    // sum_ab (X^J X^K^T)_ab O_ab.
    let particle = xj.matmul_transpose_seq(xk);
    for a in 0..nv {
        for b in 0..nv {
            sum += particle[(a, b)] * o[(no + a, no + b)];
        }
    }
    // sum_ij (X^J^T X^K)_ij O_ji.
    let hole = xj.transpose_matmul(xk);
    for i in 0..no {
        for j in 0..no {
            sum -= hole[(i, j)] * o[(j, i)];
        }
    }
    sum
}

pub(super) fn mcd(system: &System, controls: &Controls) -> Result<Value> {
    match controls.str("mcd_model", "closed-shell-tda")?.as_str() {
        "thermal-manifold" => return thermal::run(controls),
        "thermal-complex-manifold" => return complex::run(controls),
        "molecular-casci-soc" => return casci::run(system, controls),
        "closed-shell-tda" => {}
        other => {
            return Err(DftError::InvalidInput(format!(
                "unknown mcd_model `{other}`; use closed-shell-tda, thermal-manifold, thermal-complex-manifold or molecular-casci-soc"
            )))
        }
    }
    let nstates = controls.usize("nstates", 5)?;
    let explicit = controls.opt_vec3("gauge_origin")?;
    let tolerance = controls.f64("degeneracy_tolerance", 1e-6)?;
    let limit = controls.usize("max_states", 4000)?;
    let setup = setup(system, controls)?;
    controls.finish("mcd")?;
    if nstates == 0 || !(tolerance.is_finite() && tolerance >= 0.0) {
        return Err(DftError::InvalidInput(
            "nstates is at least 1 and degeneracy_tolerance is not negative".into(),
        ));
    }
    let reference = run_molecular_reference(&setup)?;
    let scf = &reference.scf;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    if scf.unrestricted || scf.electronic_temperature > 0.0 {
        return Err(DftError::Unsupported {
            feature: "magnetic circular dichroism of an open-shell or smeared reference".into(),
            missing: "the Faraday C terms and the spin-resolved state-to-state moments".into(),
        });
    }
    let solver = TdaSolver::new(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        scf,
        &setup.options,
        TdaSpin::Singlet,
    )?;
    let dim = solver.dim();
    if dim == 0 {
        return Err(DftError::InvalidInput(
            "the basis has no empty orbitals: there are no excitations".into(),
        ));
    }
    if dim > limit {
        return Err(DftError::Unsupported {
            feature: format!("magnetic circular dichroism with {dim} orbital pairs"),
            missing: format!(
                "an iterative response route; the complete diagonalization is limited to \
                 max_states = {limit}"
            ),
        });
    }
    let states = solver.solve_dense(dim)?;
    let mut charge_sum = Vec3::zero();
    let mut charge = 0.0;
    for atom in &setup.molecule.atoms {
        let z = atom.z as f64;
        charge_sum += atom.position * z;
        charge += z;
    }
    let origin = match explicit {
        Some(origin) => origin,
        None if charge > 0.0 => charge_sum * (1.0 / charge),
        None => Vec3::zero(),
    };
    let c = &scf.mo_coeff[0];
    let no = scf.n_alpha;
    let nv = c.cols - no;
    let position = dipole_matrices(&setup.basis, origin);
    let angular = angular_momentum_operator(&setup.basis, origin);
    // Orbital matrices of mu = -r and of mh = (r x grad) / 2.
    let electric: Vec<Matrix> = position
        .iter()
        .map(|r| {
            let mut m = c.transpose_matmul(&r.matmul(c));
            for v in m.as_mut_slice() {
                *v = -*v;
            }
            m
        })
        .collect();
    let magnetic: Vec<Matrix> = angular
        .iter()
        .map(|l| {
            let mut m = c.transpose_matmul(&l.matmul(c));
            for v in m.as_mut_slice() {
                *v *= 0.5;
            }
            m
        })
        .collect();
    let root2 = std::f64::consts::SQRT_2;
    let from_ground = |x: &Matrix, o: &[Matrix]| -> [f64; 3] {
        std::array::from_fn(|axis| {
            let mut sum = 0.0;
            for a in 0..nv {
                for i in 0..no {
                    sum += x[(a, i)] * o[axis][(no + a, i)];
                }
            }
            root2 * sum
        })
    };
    let electric_0: Vec<[f64; 3]> = states
        .iter()
        .map(|s| from_ground(&s.x[0], &electric))
        .collect();
    let magnetic_0: Vec<[f64; 3]> = states
        .iter()
        .map(|s| from_ground(&s.x[0], &magnetic))
        .collect();
    let count = nstates.min(states.len());
    let mut b_terms = Vec::with_capacity(count);
    let mut a_terms = Vec::with_capacity(count);
    let mut level_sizes = Vec::with_capacity(count);
    for j in 0..count {
        let xj = &states[j].x[0];
        let wj = states[j].energy;
        let mut b_term = 0.0;
        let mut a_term = 0.0;
        let mut level = 0_usize;
        for (k, state) in states.iter().enumerate() {
            let xk = &state.x[0];
            let wk = state.energy;
            // For K=J this is mu_JJ-mu_00: the K=0 contribution
            // to the final-state response cancels the permanent dipole.
            let electric_jk: [f64; 3] =
                std::array::from_fn(|axis| between(xj, xk, &electric[axis], no));
            if wk != 0.0 {
                b_term += dot(magnetic_0[k], cross(electric_0[j], electric_jk)) / wk;
            }
            let degenerate = (wk - wj).abs() <= tolerance;
            if degenerate {
                level += 1;
            }
            if k == j {
                continue;
            }
            let magnetic_jk: [f64; 3] =
                std::array::from_fn(|axis| between(xj, xk, &magnetic[axis], no));
            if degenerate {
                // mh_l'l with l = j, l' = k is minus mh_jk.
                a_term -= 0.5 * dot(magnetic_jk, cross(electric_0[j], electric_0[k]));
            } else {
                b_term += dot(magnetic_jk, cross(electric_0[j], electric_0[k])) / (wk - wj);
            }
        }
        b_terms.push(b_term);
        a_terms.push(a_term);
        level_sizes.push(level);
    }
    let canonical_a: Vec<f64> = a_terms.iter().map(|a| -*a).collect();
    let mut report = reference_report(
        &setup,
        &reference,
        "magnetic circular dichroism: canonical Faraday B with ground-state dipole \
         subtraction and canonical/legacy A representations from the complete manifold \
         of singlet Tamm-Dancoff states; no orientation or helicity prefactors applied",
    )
    .with("gauge_origin", origin)
    .with(
        "excitation_energies",
        states[..count].iter().map(|s| s.energy).collect::<Vec<_>>(),
    )
    .with(
        "dipole_strengths",
        electric_0[..count]
            .iter()
            .map(|d| dot(*d, *d))
            .collect::<Vec<_>>(),
    )
    .with("b_terms", b_terms)
    .with("a_terms", a_terms)
    .with("faraday_a_terms", canonical_a)
    .with("mcd_convention", "b_terms and faraday_a_terms follow Faber et al. (2020), DOI 10.1063/5.0013398, Eqs. (3)-(5), with m=i*(r cross grad)/2 and H_B=-m.B. b_terms is one half of the Levi-Civita contraction of the transition-strength derivative, equivalently the three cyclic derivatives of Im[mu_a(0J)*mu_b(J0)]. a_terms retains the legacy first-line-moment convention; faraday_a_terms=-a_terms is the canonical coefficient for a line-shape derivative in observation frequency. Sum complete degenerate levels; individual shares depend on the chosen basis. Orientation averages and circular-helicity prefactors are not applied.")
    .with("level_sizes", level_sizes)
    .with("states_in_sum", states.len())
    .with("provenance", "analytic");
    report.set(
        "units",
        units(&[
            ("energy", "hartree"),
            ("gauge_origin", "bohr"),
            ("excitation_energies", "hartree"),
            ("dipole_strengths", "e^2*bohr^2"),
            (
                "b_terms",
                "e^3*bohr^2*hbar/m_e/hartree, Faraday B term of the state",
            ),
            (
                "a_terms",
                "e^3*bohr^2*hbar/m_e, legacy first-line-moment share; negative of canonical Faraday A",
            ),
            (
                "faraday_a_terms",
                "e^3*bohr^2*hbar/m_e, canonical Faraday A share; sum complete degenerate levels",
            ),
        ]),
    );
    Ok(report)
}
