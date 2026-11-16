// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Orbital shielding on the actual common-orbital or full-spin stationary
//! manifold. The same one-body response providers serve electric, nuclear
//! and magnetic observables; this module only constructs and contracts the
//! magnetic operators.

use super::*;
use crate::integrals::london::ComplexMatrix;
use crate::periodic::spinor_scf::{pauli_components, spin_diagonal, SpinorScfReport};
use crate::periodic::BlochMatrix;
use crate::response::complex::Complex;
use crate::response::nuclear::{
    open_shell_cross_operator_response, spinor_constrained_one_body_response, SpinorOneBodyReport,
};
use crate::response::{Frequency, ResponseSolverConfig};

/// Shielding tensors and diagnostics of a reference-preserving magnetic solve.
pub struct ReferenceShielding {
    pub value: Vec<Shielding>,
    pub residuals: Vec<f64>,
    pub iterations: Vec<usize>,
    pub reference_stationarity: f64,
    pub metric_defect: f64,
    pub spinor_response: Option<SpinorOneBodyReport>,
}

fn imaginary(mut value: Matrix, scale: f64) -> ComplexMatrix {
    for v in value.as_mut_slice() {
        *v *= scale;
    }
    ComplexMatrix {
        re: Matrix::zeros(value.rows, value.cols),
        im: value,
    }
}

/// Orbital Zeeman derivative L/2 on both diagonal spin blocks.
pub(crate) fn spinor_orbital_field_sources(basis: &MolBasis, origin: Vec3) -> [BlochMatrix; 3] {
    angular_momentum_operator(basis, origin).map(|m| {
        let h = imaginary(m, -0.5);
        spin_diagonal(&BlochMatrix { re: h.re, im: h.im })
    })
}

fn require_common_origin(basis: &MolBasis, gauge: Gauge, origin: Vec3) -> Result<()> {
    if gauge != Gauge::CommonOrigin {
        return Err(DftError::Unsupported {
            feature: "general common-orbital or spinor London NMR response".into(),
            missing: "reference-specific London overlap and explicit phase derivatives; the common-origin response is implemented independently".into(),
        });
    }
    if !basis.ecps.is_empty() {
        return Err(DftError::Unsupported {
            feature: "reference-preserving NMR with ECP".into(),
            missing: "core reconstruction and magnetic nonlocal-potential derivatives".into(),
        });
    }
    if origin.to_array().iter().any(|x| !x.is_finite()) {
        return Err(DftError::InvalidInput(
            "the NMR gauge origin must be finite".into(),
        ));
    }
    Ok(())
}

/// London or common-origin orbital shielding using constrained complex ROKS rotations.
/// Closed-open, closed-virtual and open-virtual blocks retain their actual
/// occupation metric. No unconstrained spin Fock equation is substituted.
pub fn common_orbital_nmr_response(
    molecule: &Molecule,
    basis: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    orbitals: &Matrix,
    options: &ScfOptions,
    gauge: Gauge,
    origin: Vec3,
    config: &ResponseSolverConfig,
) -> Result<ReferenceShielding> {
    if !basis.ecps.is_empty() || origin.to_array().iter().any(|x| !x.is_finite()) {
        require_common_origin(basis, Gauge::CommonOrigin, origin)?;
    }
    let (overlap_hat, explicit, _, _) = collinear_magnetic_sources(
        molecule, basis, functional, grid, reference, options, gauge, origin,
    )?;
    // In the parallel-transport frame, C^dagger S C=1 gives
    // dP_metric = -(S^-1 dS P + P dS S^-1)/2. Transport both the
    // density-dependent Fock and the orbital metric before solving ROKS.
    let inverse = orbitals.matmul_seq(&orbitals.transpose());
    let kernel = crate::response::ResponseSolver::kernel_only(
        molecule, basis, functional, grid, reference, options,
    )?;
    let mut metric_density = Vec::with_capacity(3);
    let mut sources = Vec::with_capacity(3);
    for axis in 0..3 {
        let metric: [Matrix; 2] = std::array::from_fn(|spin| {
            let p = &reference.density[spin];
            let mut d = inverse.matmul_seq(&overlap_hat[axis]).matmul_seq(p);
            add_scaled(
                &mut d,
                &p.matmul_seq(&overlap_hat[axis]).matmul_seq(&inverse),
                1.0,
            );
            for v in d.as_mut_slice() {
                *v *= -0.5;
            }
            d
        });
        let induced = kernel.imaginary_fock_response(&metric)?;
        let source = std::array::from_fn(|spin| {
            let f = &reference.fock[spin];
            let mut h = explicit[spin][axis].clone();
            add_scaled(&mut h, &induced[spin], 1.0);
            add_scaled(
                &mut h,
                &overlap_hat[axis].matmul_seq(&inverse).matmul_seq(f),
                -0.5,
            );
            add_scaled(
                &mut h,
                &f.matmul_seq(&inverse).matmul_seq(&overlap_hat[axis]),
                -0.5,
            );
            imaginary(h, 1.0)
        });
        sources.push(source);
        metric_density.push(metric);
    }
    // Only the three field responses are needed. Each nuclear moment is a
    // measured operator contracted with those same three relaxed densities.
    let mut observables = Vec::with_capacity(3 * molecule.len());
    for atom in &molecule.atoms {
        for m in paramagnetic_operator(basis, atom.position) {
            let h = imaginary(m, -1.0);
            observables.push([h.clone(), h]);
        }
    }
    let response = open_shell_cross_operator_response(
        molecule,
        basis,
        functional,
        grid,
        reference,
        orbitals,
        options,
        &sources,
        &observables,
        Frequency::Static,
        config,
    )?;
    let density = reference.p_total();
    let scale = FINE_STRUCTURE.powi(2) * 1e6;
    let value = molecule
        .atoms
        .iter()
        .enumerate()
        .map(|(n, atom)| {
            let mixed =
                shielding_mixed_operator(basis, atom.position, origin, gauge == Gauge::Giao);
            let diamagnetic = std::array::from_fn(|a| {
                std::array::from_fn(|b| scale * trace_product(&density, &mixed[b][a]))
            });
            let pso = paramagnetic_operator(basis, atom.position);
            let paramagnetic = std::array::from_fn(|a| {
                std::array::from_fn(|b| {
                    scale
                        * (response.susceptibility[3 * n + a][b].re
                            + metric_density[b]
                                .iter()
                                .map(|d| trace_product(d, &pso[a]))
                                .sum::<f64>())
                })
            });
            Shielding {
                tensor: std::array::from_fn(|a| {
                    std::array::from_fn(|b| diamagnetic[a][b] + paramagnetic[a][b])
                }),
                diamagnetic,
                paramagnetic,
            }
        })
        .collect();
    Ok(ReferenceShielding {
        value,
        residuals: response.residuals,
        iterations: response.iterations,
        reference_stationarity: response.reference_stationarity,
        metric_defect: response.metric_defect,
        spinor_response: None,
    })
}

/// Common-origin orbital shielding of a general complex spinor reference.
/// The external field is an orbital minimal-coupling source. Spin-Zeeman,
/// Curie and thermal population shifts remain separately defined responses.
pub fn spinor_nmr_response(
    molecule: &Molecule,
    basis: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    reference: &SpinorScfReport,
    options: &ScfOptions,
    operator: &BlochMatrix,
    constraints: &[BlochMatrix],
    gauge: Gauge,
    origin: Vec3,
    config: &ResponseSolverConfig,
) -> Result<ReferenceShielding> {
    require_common_origin(basis, gauge, origin)?;
    let sources = spinor_orbital_field_sources(basis, origin);
    let response = spinor_constrained_one_body_response(
        molecule,
        basis,
        grid,
        functional,
        reference,
        options,
        operator,
        &sources,
        constraints,
        &[Complex::ZERO],
        config,
    )?;
    let raw = &response.values[0];
    let density = &pauli_components(&reference.density[0])?[0].re;
    let scale = FINE_STRUCTURE.powi(2) * 1e6;
    let value = molecule
        .atoms
        .iter()
        .map(|atom| {
            let mixed = shielding_mixed_operator(basis, atom.position, origin, false);
            let operators = paramagnetic_operator(basis, atom.position).map(|m| {
                let h = imaginary(m, -1.0);
                spin_diagonal(&BlochMatrix { re: h.re, im: h.im })
            });
            let diamagnetic = std::array::from_fn(|a| {
                std::array::from_fn(|b| scale * trace_product(density, &mixed[b][a]))
            });
            let paramagnetic = std::array::from_fn(|a| {
                std::array::from_fn(|b| {
                    scale * operators[a].trace_product(&raw.density_responses[b])
                })
            });
            Shielding {
                tensor: std::array::from_fn(|a| {
                    std::array::from_fn(|b| diamagnetic[a][b] + paramagnetic[a][b])
                }),
                diamagnetic,
                paramagnetic,
            }
        })
        .collect();
    Ok(ReferenceShielding {
        value,
        residuals: raw.residuals.clone(),
        iterations: raw.iterations.clone(),
        reference_stationarity: reference.orbital_residual,
        metric_defect: reference.metric_defect,
        spinor_response: Some(response),
    })
}
