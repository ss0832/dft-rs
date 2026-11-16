// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Ramsey tensors in the full complex spin/AO space. The electronic
//! source set is shared by unrestricted and generalized references; no
//! Cartesian or spin component is copied from another component.

use super::*;
use crate::periodic::spinor_scf::{from_pauli, pauli_components, SpinorScfReport};
use crate::periodic::BlochMatrix;
use crate::response::complex::Complex;
use crate::response::nuclear::{spinor_constrained_one_body_response, SpinorOneBodyReport};

/// Complete generalized Ramsey decomposition and reference/constraint diagnostics.
pub struct SpinorSpinSpinCouplings {
    /// Total includes `spin_orbit_mixed` in addition to the four Ramsey parts.
    pub value: SpinSpinCouplings,
    /// FC/SD with PSO, both source/observable orders, [pair][a][b].
    pub spin_orbit_mixed: Vec<[[f64; 3]; 3]>,
    pub response: SpinorOneBodyReport,
}

/// Full Hermitian FC, SD and PSO sources. The layout is
/// [nucleus][term FC,SD,PSO][nuclear moment axis]; contact-only has FC only.
pub fn spinor_nuclear_magnetic_sources(
    molecule: &Molecule,
    basis: &MolBasis,
    nuclei: &[usize],
    contact_only: bool,
) -> Result<Vec<BlochMatrix>> {
    let sites = nuclear_magnetic_sites(molecule, basis, nuclei, contact_only)?;
    let n = basis.nao_sph;
    let contact_scale =
        0.5 * ELECTRON_G_FACTOR * FINE_STRUCTURE.powi(2) * 8.0 * std::f64::consts::PI / 3.0;
    let mut sources = Vec::new();
    for site in sites {
        for term in 0..if contact_only { 1 } else { 3 } {
            for axis in 0..3 {
                let mut pauli = std::array::from_fn(|_| BlochMatrix::zeros(n, n));
                if term == 2 {
                    pauli[0].im = site.orbital[axis].clone();
                    for v in pauli[0].im.as_mut_slice() {
                        *v *= -2.0 * FINE_STRUCTURE.powi(2);
                    }
                } else {
                    for spin in 0..3 {
                        if term == 1 {
                            pauli[spin + 1].re = site.triplet[3 * spin + axis].clone();
                        }
                        if spin == axis {
                            add(
                                &mut pauli[spin + 1].re,
                                &site.contact,
                                if term == 0 {
                                    contact_scale
                                } else {
                                    -contact_scale
                                },
                            );
                        }
                    }
                }
                sources.push(from_pauli(&pauli)?);
            }
        }
    }
    Ok(sources)
}

/// Total electronic spin operators Sx,Sy,Sz in the nonorthogonal AO metric.
/// They constrain expectation derivatives, rather than shifting Goldstone modes.
pub fn total_spin_constraints(overlap: &Matrix) -> Result<Vec<BlochMatrix>> {
    (0..3)
        .map(|axis| {
            let mut pauli = std::array::from_fn(|_| BlochMatrix::zeros(overlap.rows, overlap.cols));
            pauli[axis + 1].re = overlap.clone();
            from_pauli(&pauli)
        })
        .collect()
}

/// Two regular constraints fixing the direction of a nonzero total spin.
/// Its magnitude may relax. Constraining all three components at a collinear
/// determinant is not a regular finite-field constraint: the longitudinal
/// Jacobian vanishes although the longitudinal expectation can change at
/// second order. It must not be replaced by a dropped linear row.
pub fn spin_direction_constraints(
    density: &BlochMatrix,
    overlap: &Matrix,
) -> Result<Vec<BlochMatrix>> {
    let spin = total_spin_constraints(overlap)?;
    pauli_components(density)?;
    if density.rows() != 2 * overlap.rows || density.cols() != 2 * overlap.cols {
        return Err(DftError::InvalidInput(
            "spin-direction constraints need a matching spin/AO density and overlap".into(),
        ));
    }
    let moment: [f64; 3] = std::array::from_fn(|a| spin[a].trace_product(density));
    let length = moment.iter().fold(0.0_f64, |a, &b| a.hypot(b));
    let norm = |m: &BlochMatrix| {
        m.re.as_slice()
            .iter()
            .chain(m.im.as_slice())
            .fold(0.0_f64, |a, &b| a.hypot(b))
    };
    let bound = 64.0
        * f64::EPSILON
        * density.rows() as f64
        * norm(density)
        * spin.iter().map(norm).fold(0.0_f64, f64::max);
    if !length.is_finite() || !bound.is_finite() || length <= bound {
        return Err(DftError::Unsupported {feature:"fixed spin direction at an unresolved or zero total moment".into(),
            missing:"a nonzero resolved reference moment; choose free response for a spin-paired state, or supply local magnetic constraints for a compensated magnetic state".into()});
    }
    let direction = moment.map(|v| v / length);
    let axis = (0..3)
        .min_by(|&a, &b| direction[a].abs().total_cmp(&direction[b].abs()))
        .expect("three spin axes");
    let mut first: [f64; 3] = std::array::from_fn(|a| if a == axis { 1.0 } else { 0.0 });
    for a in 0..3 {
        first[a] -= direction[axis] * direction[a];
    }
    let scale = first.iter().fold(0.0_f64, |a, &b| a.hypot(b));
    first = first.map(|v| v / scale);
    let second = [
        direction[1] * first[2] - direction[2] * first[1],
        direction[2] * first[0] - direction[0] * first[2],
        direction[0] * first[1] - direction[1] * first[0],
    ];
    [first, second]
        .iter()
        .map(|axis| {
            let mut op = BlochMatrix::zeros(density.rows(), density.cols());
            for a in 0..3 {
                op.add_scaled(&spin[a], axis[a]);
            }
            Ok(op)
        })
        .collect()
}

/// Static FC/SD/PSO/DSO tensors of the supplied stationary complex reference.
/// Optional constraint operators specify a fixed-expectation ensemble. An
/// unconstrained singular spin response is reported by the physical solver.
pub fn spinor_spin_spin_couplings(
    molecule: &Molecule,
    basis: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    reference: &SpinorScfReport,
    options: &ScfOptions,
    operator: &BlochMatrix,
    constraints: &[BlochMatrix],
    nuclei: &[usize],
    contact_only: bool,
    config: &ResponseSolverConfig,
) -> Result<SpinorSpinSpinCouplings> {
    let sources = spinor_nuclear_magnetic_sources(molecule, basis, nuclei, contact_only)?;
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
    let terms = if contact_only { 1 } else { 3 };
    let mut value = SpinSpinCouplings {
        pairs: Vec::new(),
        diamagnetic: Vec::new(),
        paramagnetic: Vec::new(),
        fermi_contact: Vec::new(),
        spin_dipole: Vec::new(),
        total: Vec::new(),
        residual: raw.residuals.iter().copied().fold(0.0_f64, f64::max),
    };
    let mut mixed = Vec::new();
    let built;
    let quadrature = match grid {
        Some(g) => g,
        None => {
            built = MolecularGrid::build(
                molecule,
                &GridOptions {
                    weight_cutoff: 0.0,
                    ..Default::default()
                },
            )?;
            &built
        }
    };
    let total = pauli_components(&reference.density[0])?[0].re.clone();
    let mut density = vec![0.0; quadrature.len()];
    if !contact_only {
        for first in (0..quadrature.len()).step_by(256) {
            let last = (first + 256).min(quadrature.len());
            let ao = eval_aos(basis, &quadrature.points[first..last], false);
            let product = ao.phi.matmul(&total);
            for g in 0..last - first {
                density[first + g] = (0..basis.nao_sph)
                    .map(|i| ao.phi[(g, i)] * product[(g, i)])
                    .sum();
            }
        }
    }
    for m in 0..nuclei.len() {
        for n in m + 1..nuclei.len() {
            let entry = |left: usize, right: usize, a: usize, b: usize| {
                raw.susceptibility[(m * terms + left) * 3 + a][(n * terms + right) * 3 + b].re
            };
            let fc = std::array::from_fn(|a| std::array::from_fn(|b| entry(0, 0, a, b)));
            let sd = std::array::from_fn(|a| {
                std::array::from_fn(|b| {
                    if contact_only {
                        0.0
                    } else {
                        entry(0, 1, a, b) + entry(1, 0, a, b) + entry(1, 1, a, b)
                    }
                })
            });
            let pso = std::array::from_fn(|a| {
                std::array::from_fn(|b| if contact_only { 0.0 } else { entry(2, 2, a, b) })
            });
            let cross = std::array::from_fn(|a| {
                std::array::from_fn(|b| {
                    if contact_only {
                        0.0
                    } else {
                        entry(0, 2, a, b)
                            + entry(1, 2, a, b)
                            + entry(2, 0, a, b)
                            + entry(2, 1, a, b)
                    }
                })
            });
            let mut dso = [[0.0; 3]; 3];
            if !contact_only {
                for (g, point) in quadrature.points.iter().enumerate() {
                    if let Some(kernel) = diamagnetic_spin_orbit_kernel(
                        point.to_array(),
                        molecule.atoms[nuclei[m]].position.to_array(),
                        molecule.atoms[nuclei[n]].position.to_array(),
                    ) {
                        for a in 0..3 {
                            for b in 0..3 {
                                dso[a][b] += FINE_STRUCTURE.powi(4)
                                    * quadrature.weights[g]
                                    * density[g]
                                    * kernel[a][b];
                            }
                        }
                    }
                }
            }
            value.pairs.push((nuclei[m], nuclei[n]));
            value.total.push(std::array::from_fn(|a| {
                std::array::from_fn(|b| fc[a][b] + sd[a][b] + pso[a][b] + cross[a][b] + dso[a][b])
            }));
            value.fermi_contact.push(fc);
            value.spin_dipole.push(sd);
            value.paramagnetic.push(pso);
            value.diamagnetic.push(dso);
            mixed.push(cross);
        }
    }
    Ok(SpinorSpinSpinCouplings {
        value,
        spin_orbit_mixed: mixed,
        response,
    })
}
