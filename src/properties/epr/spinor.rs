// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Full-complex spin/orbital hyperfine fields. A noncollinear texture has
//! a well-defined nuclear field but does not, from one determinant alone,
//! determine an effective-spin coupling tensor.

use crate::basis::MolBasis;
use crate::constants::{
    ELECTRON_G_FACTOR, FINE_STRUCTURE, HARTREE_TO_JOULE, HBAR_J_S, PROTON_ELECTRON_MASS_RATIO,
};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::spinor_scf::pauli_components;
use crate::periodic::BlochMatrix;
use crate::{DftError, Result};
use std::f64::consts::PI;

#[path = "spinor/rotation.rs"]
pub mod rotation;

fn add_real_pauli(target: &mut BlochMatrix, coefficient: &Matrix, axis: usize, scale: f64) {
    let n = coefficient.rows;
    for i in 0..n {
        for j in 0..n {
            let value = scale * coefficient[(i, j)];
            match axis {
                0 => {
                    target.re[(i, n + j)] += value;
                    target.re[(n + i, j)] += value;
                }
                1 => {
                    target.im[(i, n + j)] -= value;
                    target.im[(n + i, j)] += value;
                }
                _ => {
                    target.re[(i, j)] += value;
                    target.re[(n + i, n + j)] -= value;
                }
            }
        }
    }
}

/// One-electron Breit-Pauli SOC, -i alpha^2/4 sum_A Z_A Lhat_A . sigma.
/// This is a physical Hamiltonian input to a self-consistent spinor solver;
/// the two-electron spin-orbit operator is not supplied by this function.
pub fn one_electron_soc_operator(
    basis: &MolBasis,
    molecule: &crate::system::Molecule,
    soc_charges: &[f64],
) -> Result<BlochMatrix> {
    if !basis.ecps.is_empty()
        || soc_charges.len() != molecule.len()
        || soc_charges.iter().any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput(
            "one-electron SOC requires matching finite charges and an all-electron basis".into(),
        ));
    }
    let n = basis.nao_sph;
    let mut result = BlochMatrix::zeros(2 * n, 2 * n);
    for (atom, charge) in molecule.atoms.iter().zip(soc_charges) {
        let orbital = crate::properties::nmr::paramagnetic_operator(basis, atom.position);
        let scale = FINE_STRUCTURE.powi(2) * charge / 4.0;
        for i in 0..n {
            for j in 0..n {
                result.im[(i, n + j)] -= scale * orbital[0][(i, j)];
                result.im[(n + i, j)] -= scale * orbital[0][(i, j)];
                result.re[(i, n + j)] -= scale * orbital[1][(i, j)];
                result.re[(n + i, j)] += scale * orbital[1][(i, j)];
                result.im[(i, j)] -= scale * orbital[2][(i, j)];
                result.im[(n + i, n + j)] += scale * orbital[2][(i, j)];
            }
        }
    }
    Ok(result)
}

/// dH/dB for an explicitly common-origin one-electron Breit-Pauli model.
/// H_eff = mu_B B_a g_ab sigma_b/2, mu_B=1/2, so four times the
/// projected Pauli coefficients gives g. SOC orbital mixing belongs to
/// the supplied spinor state; no perturbative SO correction is added twice.
pub fn common_origin_zeeman_operators(
    basis: &MolBasis,
    molecule: &crate::system::Molecule,
    overlap: &Matrix,
    origin: Vec3,
    soc_charges: &[f64],
) -> Result<[BlochMatrix; 3]> {
    let n = basis.nao_sph;
    if !basis.ecps.is_empty()
        || soc_charges.len() != molecule.len()
        || soc_charges.iter().any(|v| !v.is_finite())
        || !origin.norm().is_finite()
        || overlap.rows != n
        || overlap.cols != n
    {
        return Err(DftError::InvalidInput("common-origin Zeeman operators require finite matching all-electron geometry, charges and overlap".into()));
    }
    let mut result = std::array::from_fn(|_| BlochMatrix::zeros(2 * n, 2 * n));
    let angular = crate::properties::nmr::angular_momentum_operator(basis, origin);
    let kinetic = crate::integrals::kinetic_matrix(basis);
    for axis in 0..3 {
        add_real_pauli(&mut result[axis], overlap, axis, ELECTRON_G_FACTOR / 4.0);
        add_real_pauli(
            &mut result[axis],
            &kinetic,
            axis,
            -FINE_STRUCTURE.powi(2) * ELECTRON_G_FACTOR / 4.0,
        );
        for spin in 0..2 {
            for i in 0..n {
                for j in 0..n {
                    result[axis].im[(spin * n + i, spin * n + j)] -= 0.5 * angular[axis][(i, j)];
                }
            }
        }
    }
    for (atom, charge) in molecule.atoms.iter().zip(soc_charges) {
        let mixed =
            crate::properties::nmr::shielding_mixed_operator(basis, atom.position, origin, false);
        for field in 0..3 {
            for spin in 0..3 {
                add_real_pauli(
                    &mut result[field],
                    &mixed[field][spin],
                    spin,
                    FINE_STRUCTURE.powi(2) * charge / 4.0,
                );
            }
        }
    }
    Ok(result)
}

/// Explicit optional mean-field single-pair projection. Tensor columns
/// refer to the ordered determinant pair, not automatically to Cartesian
/// electronic spin. gg^T and determinant(g) are pseudospin invariants.
pub struct ProjectedSpinorEpr {
    pub g_tensor: [[f64; 3]; 3],
    pub g_gram: [[f64; 3]; 3],
    pub contact_per_g_mhz: Vec<[[f64; 3]; 3]>,
    pub dipolar_per_g_mhz: Vec<[[f64; 3]; 3]>,
    pub orbital_per_g_mhz: Vec<[[f64; 3]; 3]>,
    pub total_per_g_mhz: Vec<[[f64; 3]; 3]>,
    pub spin_expectation: [f64; 3],
    pub spin_squared: f64,
    pub occupied_metric_defect: f64,
    pub pair_overlap: f64,
    pub hamiltonian_time_reversal_defect: f64,
}

/// Use the actual converged GKS/GHF spinor SCF state. The extra zero-field
/// one-electron Hamiltonian must be time even. This is a declared pair
/// approximation: no isolated-doublet guarantee, spin purification,
/// two-electron SOC, London response or general-multiplet claim is made.
#[allow(clippy::too_many_arguments)]
pub fn projected_spinor_epr(
    molecule: &crate::system::Molecule,
    basis: &MolBasis,
    overlap: &Matrix,
    reference: &crate::periodic::spinor_scf::SpinorScfReport,
    extra_operator: &BlochMatrix,
    origin: Vec3,
    soc_charges: &[f64],
    tolerance: f64,
) -> Result<ProjectedSpinorEpr> {
    if !reference.fields.operator_converged
        || reference
            .fields
            .xc_diagnostics
            .as_ref()
            .is_some_and(|d| !d.converged)
    {
        let residual = reference.fields.xc_diagnostics.as_ref().map_or(
            reference.fields.coulomb_operator_defect,
            |d| {
                d.angular_energy_defect
                    .max(d.angular_operator_defect)
                    .max(d.hermiticity_defect)
                    .max(d.component_energy_defect)
                    .max(reference.fields.coulomb_operator_defect)
            },
        );
        return Err(DftError::ResponseNotConverged {
            iterations: reference.iterations,
            residual,
        });
    }
    if extra_operator.rows() != 2 * basis.nao_sph || extra_operator.cols() != 2 * basis.nao_sph {
        return Err(DftError::InvalidInput(
            "EPR zero-field operator dimensions disagree with the molecular spin/AO basis".into(),
        ));
    }
    let time_defect = super::doublet::time_reversal_defect(extra_operator, 1)?;
    if time_defect > tolerance || extra_operator.hermiticity_error() > tolerance {
        return Err(DftError::InvalidInput(
            "single Kramers-pair EPR requires a Hermitian time-even zero-field Hamiltonian".into(),
        ));
    }
    let pair = super::doublet::KramersPair::from_reference(reference, overlap, tolerance)?;
    let zeeman = common_origin_zeeman_operators(basis, molecule, overlap, origin, soc_charges)?;
    let g_tensor = pair.tensor(&zeeman, 4.0, tolerance)?;
    let g_gram = std::array::from_fn(|a| {
        std::array::from_fn(|b| (0..3).map(|c| g_tensor[a][c] * g_tensor[b][c]).sum())
    });
    let sites = (0..molecule.len()).collect::<Vec<_>>();
    let sources = crate::response::spin_spin::spinor_nuclear_magnetic_sources(
        molecule, basis, &sites, false,
    )?;
    let scale = HARTREE_TO_JOULE / (2.0 * PI * HBAR_J_S) * 1e-6 / PROTON_ELECTRON_MASS_RATIO;
    let mut contact = Vec::new();
    let mut dipolar = Vec::new();
    let mut orbital = Vec::new();
    let mut total = Vec::new();
    for atom in 0..molecule.len() {
        let mut terms = Vec::new();
        for term in 0..3 {
            let operators = std::array::from_fn(|axis| sources[9 * atom + 3 * term + axis].clone());
            terms.push(pair.tensor(&operators, scale, tolerance)?);
        }
        total.push(std::array::from_fn(|a| {
            std::array::from_fn(|b| terms.iter().map(|t| t[a][b]).sum())
        }));
        contact.push(terms[0]);
        dipolar.push(terms[1]);
        orbital.push(terms[2]);
    }
    let density = &reference.density[0];
    let spins = crate::response::spin_spin::total_spin_constraints(overlap)?;
    let mut moment = [0.0; 3];
    let mut exchange = 0.0;
    for a in 0..3 {
        let product = crate::dfpt::orbitals::matmul(density, &spins[a])?;
        moment[a] = (0..product.rows()).map(|i| product.re[(i, i)]).sum::<f64>();
        let square = crate::dfpt::orbitals::matmul(&product, &product)?;
        exchange += (0..square.rows()).map(|i| square.re[(i, i)]).sum::<f64>();
    }
    let electrons = reference.occupations[0].iter().sum::<f64>();
    let spin_squared = moment.iter().map(|v| v * v).sum::<f64>() + 0.75 * electrons - exchange;
    Ok(ProjectedSpinorEpr {
        g_tensor,
        g_gram,
        contact_per_g_mhz: contact,
        dipolar_per_g_mhz: dipolar,
        orbital_per_g_mhz: orbital,
        total_per_g_mhz: total,
        spin_expectation: moment,
        spin_squared,
        occupied_metric_defect: pair.occupied_metric_defect,
        pair_overlap: pair.determinant_overlap.abs(),
        hamiltonian_time_reversal_defect: time_defect,
    })
}

pub struct SpinorHyperfineField {
    /// Coefficients of nuclear spin I, MHz per nuclear g; includes the
    /// expectation of the actual electronic spin, not normalization by S.
    pub contact: [f64; 3],
    pub dipolar: [f64; 3],
    pub orbital: [f64; 3],
    pub total: [f64; 3],
    pub spin_expectation: [f64; 3],
    pub collinearity_defect: f64,
    /// Only present when every complex AO spin-density component is a
    /// multiple of the same spin axis. Otherwise an explicit effective
    /// manifold is needed; no tensor is fabricated from a scalar moment.
    pub collinear_spin_tensor: Option<crate::properties::hyperfine::HyperfineTensor>,
    pub poisson_residual: f64,
    pub hermiticity_residual: f64,
}

/// Contract the supplied Hermitian GKS/GHF density with point-nucleus FC,
/// SD and orbital operators, retaining all complex Pauli blocks. The spin
/// operators are real, so their trace with an imaginary antisymmetric
/// density vanishes by Hermiticity, rather than by a collinear replacement.
pub fn spinor_hyperfine_field(
    basis: &MolBasis,
    overlap: &Matrix,
    density: &BlochMatrix,
    center: Vec3,
    collinearity_tolerance: f64,
) -> Result<SpinorHyperfineField> {
    let n = basis.nao_sph;
    if !basis.ecps.is_empty()
        || overlap.rows != n
        || overlap.cols != n
        || density.rows() != 2 * n
        || density.cols() != 2 * n
        || !collinearity_tolerance.is_finite()
        || collinearity_tolerance <= 0.0
        || overlap.as_slice().iter().any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput("spinor hyperfine fields require matching all-electron Hermitian AOs, metric and a finite positive collinearity tolerance".into()));
    }
    let pauli = pauli_components(density)?;
    let trace = |p: &Matrix, o: &Matrix| -> f64 {
        (0..n)
            .map(|i| (0..n).map(|j| p[(i, j)] * o[(j, i)]).sum::<f64>())
            .sum()
    };
    let moment = std::array::from_fn::<_, 3, _>(|a| 0.5 * trace(&pauli[a + 1].re, overlap));
    let length = moment.iter().fold(0.0_f64, |value, &x| value.hypot(x));
    let operators = crate::properties::hyperfine::spin_dipole_operators(basis, center)?;
    let scale = FINE_STRUCTURE.powi(2) * ELECTRON_G_FACTOR / (8.0 * PROTON_ELECTRON_MASS_RATIO)
        * HARTREE_TO_JOULE
        / (2.0 * PI * HBAR_J_S)
        * 1e-6;
    let contact = std::array::from_fn(|a| {
        scale * (8.0 * PI / 3.0) * trace(&pauli[a + 1].re, &operators.contact)
    });
    let dipolar = std::array::from_fn(|a| {
        (0..3)
            .map(|b| scale * trace(&pauli[b + 1].re, &operators.dipolar[3 * a + b]))
            .sum::<f64>()
    });
    let orbital_operator = crate::properties::nmr::paramagnetic_operator(basis, center);
    let orbital_scale = FINE_STRUCTURE.powi(2) / (2.0 * PROTON_ELECTRON_MASS_RATIO)
        * HARTREE_TO_JOULE
        / (2.0 * PI * HBAR_J_S)
        * 1e-6;
    // The charge Pauli block contains orbital current. For O=-i alpha^2 Lhat,
    // Re Tr[P O] = alpha^2 Tr[Im(P) Lhat]; do not discard this block.
    let orbital =
        std::array::from_fn(|a| orbital_scale * trace(&pauli[0].im, &orbital_operator[a]));
    let total = std::array::from_fn(|a| contact[a] + dipolar[a] + orbital[a]);
    let mut projected = BlochMatrix::zeros(n, n);
    if length > 0.0 {
        for a in 0..3 {
            projected.add_scaled(&pauli[a + 1], moment[a] / length);
        }
    }
    let mut collinearity_defect = 0.0_f64;
    for a in 0..3 {
        for i in 0..n {
            for j in 0..n {
                let axis = if length > 0.0 {
                    moment[a] / length
                } else {
                    0.0
                };
                collinearity_defect = collinearity_defect
                    .max((pauli[a + 1].re[(i, j)] - axis * projected.re[(i, j)]).abs())
                    .max((pauli[a + 1].im[(i, j)] - axis * projected.im[(i, j)]).abs());
            }
        }
    }
    let collinear_spin_tensor = if length > 0.0 && collinearity_defect <= collinearity_tolerance {
        Some(crate::properties::hyperfine::spin_hyperfine_tensor(
            basis,
            &projected.re,
            center,
            2.0 * length,
        )?)
    } else {
        None
    };
    Ok(SpinorHyperfineField {
        contact,
        dipolar,
        orbital,
        total,
        spin_expectation: moment,
        collinearity_defect,
        collinear_spin_tensor,
        poisson_residual: operators.poisson_residual,
        hermiticity_residual: operators.hermiticity_residual,
    })
}
