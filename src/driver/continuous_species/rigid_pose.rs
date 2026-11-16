// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic center/orientation jets for an explicit collection of static cores.
//!
//! Coordinates are atom-major Cartesian centers, followed by three laboratory
//! Euler angles per fragment. U = Rz(gamma) Ry(beta) Rx(alpha) U_reference.
//! Angles are radians. The rotation connection is differentiated here exactly
//! once. These operator/classical jets require the moving-basis SCF response
//! driver before they can be called derivatives of a converged electronic energy.
//! No finite differences, optimizer, internal fragment deformation or implicit
//! chemical accuracy claim is supplied by this module.

use super::core_jet2::{
    self, BaseFieldConvention, GaussianDistributionPair, PairCoefficients, PairGeometry,
    PairScalar, PointIonScope, Projectors,
};
use super::static_core::StaticCoreParameters;
/// Fixed core dictionary parameters; no element identity or fitted coefficient
/// is inferred from the retained center's ordinary point-ion label.
pub use super::static_core::StaticCoreParameters as RigidCoreParameters;
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::linalg::Matrix;
use crate::math::{Mat3, Vec3};
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

/// Immutable primitive-eight-projector core on one retained point-ion center.
/// The point ion remains in the ordinary Hamiltonian and is never repeated.
#[derive(Clone)]
pub struct RigidFragment {
    pub atom: usize,
    pub reference_orientation: Mat3,
    pub euler_angles: [f64; 3],
    pub parameters: StaticCoreParameters,
    pub local_exponents: [f64; 3],
    pub local_amplitudes: [f64; 3],
    /// Constant E0. Added once to the total energy, not to its derivatives.
    pub internal_energy_reference: f64,
}

pub struct RigidPosePair {
    /// Complete one-body correction [value,Da,Db,Dab], including moving AOs.
    pub operator: [Matrix; 4],
    /// Gaussian/point and inter-fragment Gaussian/Gaussian terms. The base
    /// point/point repulsion and the separate constant E0 are excluded.
    pub classical_correction: [f64; 4],
    /// Laboratory permanent-dipole correction relative to the point ions.
    /// [Cartesian component][value,Da,Db,Dab], in e*bohr and its derivatives.
    pub permanent_dipole: [[f64; 4]; 3],
    pub internal_energy_reference: f64,
    pub positions: Vec<[PairScalar; 3]>,
    pub coordinate_map: u64,
}

pub struct RigidPoseGradient {
    /// Same coordinate order as pair: Cartesian centers, then Euler angles.
    /// Center entries are Eh/bohr; angle entries are Eh/radian. Physical forces
    /// and rotation torques have the opposite sign. Euler entries are chart
    /// covector components and are not three laboratory torque components.
    pub gradient: Vec<f64>,
    pub total_energy: f64,
    pub orbital_residual: f64,
}

/// Complete first derivative of the converged reduced electronic energy.
/// Ordinary moving-AO/Pulay, ERI and moving-grid contributions come from the
/// common analytic gradient. This adapter adds Tr(P dDelta) and the classical
/// correction once. No density response is needed at a stationary reference.
#[allow(clippy::too_many_arguments)]
pub fn assemble_gradient(
    molecule: &Molecule,
    basis: &MolBasis,
    fragments: &[RigidFragment],
    functional: &ResolvedFunctional,
    grid: &MolecularGrid,
    reference: &ScfResult,
    options: &ScfOptions,
) -> Result<RigidPoseGradient> {
    if reference.unrestricted
        || reference.n_alpha != reference.n_beta
        || reference.electronic_temperature != 0.
        || !reference.has_integer_occupations()
        || options.external_field.is_some()
        || options.asymptotic_correction.is_some()
        || options.density_fit.is_some()
        || options.vv10.is_some()
        || options.skala.is_some()
        || options.pcm.is_some()
        || !basis.ecps.is_empty()
        || basis.n_core_electrons != 0
        || !matches!(options.kernel, crate::integrals::CoulombKernel::Coulomb)
    {
        return Err(DftError::Unsupported {
            feature: "rigid reduced-core center/orientation derivatives for this reference or environment".into(),
            missing: "the current adapter requires a restricted integer all-electron molecular reference at zero field without RI-J, nonlocal correlation, learned XC or solvation".into(),
        });
    }
    let diagnostics = crate::response::diagnostics::reference_diagnostics(reference)?;
    if !reference.converged
        || diagnostics.orbital_residual > options.conv_tol_grad
        || diagnostics.projector_defect > 1e-8
        || diagnostics.particle_number_error > 1e-8
    {
        return Err(DftError::ScfNotConverged {
            iterations: reference.iterations,
            error: diagnostics.orbital_residual,
        });
    }
    if (reference.e_nuc - molecule.nuclear_repulsion()).abs() > 1e-11 {
        return invalid("rigid reduced-core reference must retain the bare point-ion classical ledger; corrections and E0 are added separately once");
    }
    let count = 3 * molecule.len() + 3 * fragments.len();
    let zero = vec![0.; count];
    let base_pair = pair(molecule, basis, fragments, &zero, &zero)?;
    let Some(extra) = &options.extra_hcore else {
        return invalid(
            "rigid reduced-core reference must include its complete one-body correction",
        );
    };
    let n = basis.nao_sph;
    if extra.rows != n
        || extra.cols != n
        || reference.h_core.rows != n
        || reference.h_core.cols != n
    {
        return invalid("rigid reduced-core reference AO dimensions do not match the model");
    }
    let mut expected = crate::integrals::kinetic_matrix(basis);
    let nuclear = crate::integrals::nuclear_matrix(basis, molecule, None);
    for i in 0..n {
        for j in 0..n {
            expected[(i, j)] += nuclear[(i, j)] + base_pair.operator[0][(i, j)];
            if !extra[(i, j)].is_finite()
                || !reference.h_core[(i, j)].is_finite()
                || (extra[(i, j)] - base_pair.operator[0][(i, j)]).abs() > 1e-11
                || (reference.h_core[(i, j)] - expected[(i, j)]).abs() > 1e-11
            {
                return invalid(
                    "rigid reduced-core reference does not match the complete current operator",
                );
            }
        }
    }
    let ordinary = crate::gradient::assemble_gradient(
        molecule,
        basis,
        functional,
        Some(grid),
        reference,
        options,
    )?;
    let density = reference.p_total();
    let mut gradient = Vec::with_capacity(count);
    for i in 0..count {
        let mut direction = zero.clone();
        direction[i] = 1.;
        let derivative = pair(molecule, basis, fragments, &direction, &zero)?;
        let base = if i < 3 * molecule.len() {
            ordinary[i / 3].get(i % 3)
        } else {
            0.
        };
        gradient.push(
            base + density.frobenius_dot(&derivative.operator[1])
                + derivative.classical_correction[1],
        );
    }
    if gradient.iter().any(|v| !v.is_finite()) {
        return invalid("rigid reduced-core analytic gradient overflowed");
    }
    Ok(RigidPoseGradient {
        gradient,
        total_energy: reference.energy
            + base_pair.classical_correction[0]
            + base_pair.internal_energy_reference,
        orbital_residual: diagnostics.orbital_residual,
    })
}

fn invalid<T>(message: &str) -> Result<T> {
    Err(DftError::InvalidInput(message.into()))
}

fn constant(value: f64) -> PairScalar {
    core_jet2::seed(value, 0., 0., 0.)
}

// Composition with sin/cos uses their exact first and second derivatives,
// including preexisting chart curvature. It does not difference a rotation.
fn sin_cos(value: PairScalar) -> (PairScalar, PairScalar) {
    let [x, a, b, ab] = core_jet2::lanes(value);
    let (s, c) = x.sin_cos();
    (
        core_jet2::seed(s, c * a, c * b, c * ab - s * a * b),
        core_jet2::seed(c, -s * a, -s * b, -s * ab - c * a * b),
    )
}

/// Proper rotation columns and all mixed chain-rule terms in the fixed Euler
/// chart. Euler singularities do not discard lanes; inverse torque conversion
/// at gimbal lock must be rejected separately by a caller requesting it.
pub fn euler_rotation_pair(
    reference: Mat3,
    angles: [PairScalar; 3],
) -> Result<[[PairScalar; 3]; 3]> {
    if reference
        .col
        .iter()
        .flat_map(|v| v.to_array())
        .any(|v| !v.is_finite())
        || angles
            .iter()
            .flat_map(|&v| core_jet2::lanes(v))
            .any(|v| !v.is_finite())
    {
        return invalid("rigid Euler chart requires finite reference axes and angle jets");
    }
    for i in 0..3 {
        for j in 0..3 {
            if (reference.col[i].dot(reference.col[j]) - if i == j { 1. } else { 0. }).abs() > 1e-12
            {
                return invalid("rigid Euler reference axes must be orthonormal");
            }
        }
    }
    if (reference.col[0]
        .cross(reference.col[1])
        .dot(reference.col[2])
        - 1.)
        .abs()
        > 1e-12
    {
        return invalid("rigid Euler reference axes must be right handed");
    }
    let (sa, ca) = sin_cos(angles[0]);
    let (sb, cb) = sin_cos(angles[1]);
    let (sg, cg) = sin_cos(angles[2]);
    Ok(reference.col.map(|axis| {
        let [x, y, z] = axis.to_array().map(constant);
        let y1 = ca * y - sa * z;
        let z1 = sa * y + ca * z;
        let x2 = cb * x + sb * z1;
        let z2 = -sb * x + cb * z1;
        [cg * x2 - sg * y1, sg * x2 + cg * y1, z2]
    }))
}

/// Assemble a complete static correction for multiple cores at zero field.
/// Derivative directions have length 3*N_atoms+3*N_fragments. No electron,
/// functional, quadrature or SCF policy is inferred from these sources.
pub fn pair(
    molecule: &Molecule,
    basis: &MolBasis,
    fragments: &[RigidFragment],
    direction_a: &[f64],
    direction_b: &[f64],
) -> Result<RigidPosePair> {
    let n = molecule.len();
    let coordinates = n
        .checked_mul(3)
        .and_then(|n| {
            fragments
                .len()
                .checked_mul(3)
                .and_then(|m| n.checked_add(m))
        })
        .ok_or_else(|| DftError::InvalidInput("rigid pose coordinate count overflow".into()))?;
    if n == 0
        || fragments.is_empty()
        || direction_a.len() != coordinates
        || direction_b.len() != coordinates
        || direction_a
            .iter()
            .chain(direction_b)
            .any(|v| !v.is_finite())
        || fragments.iter().any(|f| f.atom >= n)
        || fragments
            .iter()
            .enumerate()
            .any(|(i, f)| fragments[..i].iter().any(|p| p.atom == f.atom))
    {
        return invalid(
            "rigid pose needs finite complete directions and distinct valid fragment centers",
        );
    }
    let positions: Vec<_> = molecule
        .atoms
        .iter()
        .enumerate()
        .map(|(i, atom)| {
            std::array::from_fn(|k| {
                core_jet2::seed(
                    atom.position.get(k),
                    direction_a[3 * i + k],
                    direction_b[3 * i + k],
                    0.,
                )
            })
        })
        .collect();
    let mut identity = super::Fingerprint::new();
    identity.string("rigid-multi-core-center-euler-v1");
    identity.number(n);
    for atom in &molecule.atoms {
        identity.number(usize::from(atom.z));
        for x in atom.position.to_array() {
            identity.float(x);
        }
    }
    for f in fragments {
        identity.number(f.atom);
        for x in f
            .reference_orientation
            .col
            .iter()
            .flat_map(|v| v.to_array())
            .chain(f.euler_angles)
        {
            identity.float(x);
        }
    }
    let coordinate_map = identity.0;
    let mut operator = std::array::from_fn(|_| Matrix::zeros(basis.nao_sph, basis.nao_sph));
    let mut classical_correction = [0.; 4];
    let mut permanent_dipole = [[0.; 4]; 3];
    let mut internal_energy_reference = 0.;
    let mut distributions = Vec::with_capacity(fragments.len());
    let ionic_charges: Vec<_> = molecule.atoms.iter().map(|a| f64::from(a.z)).collect();
    for (i, fragment) in fragments.iter().enumerate() {
        if fragment.local_amplitudes.iter().any(|v| !v.is_finite())
            || !fragment.internal_energy_reference.is_finite()
        {
            return invalid("rigid core amplitudes and separate E0 must be finite");
        }
        let offset = 3 * n + 3 * i;
        let angles = std::array::from_fn(|k| {
            core_jet2::seed(
                fragment.euler_angles[k],
                direction_a[offset + k],
                direction_b[offset + k],
                0.,
            )
        });
        let rotation = euler_rotation_pair(fragment.reference_orientation, angles)?;
        let geometry = PairGeometry::rigid_pose(
            molecule,
            fragment.atom,
            positions.clone(),
            rotation,
            coordinate_map,
        )?;
        let coefficients = PairCoefficients::constant(
            &ionic_charges,
            &fragment.parameters,
            fragment.local_amplitudes,
            0.,
        )?;
        let value = core_jet2::evaluate_pair(
            molecule,
            basis,
            &geometry,
            &coefficients,
            fragment.parameters.charge_exponents,
            fragment.local_exponents,
            Projectors::Primitive {
                s_exponents: fragment.parameters.projector_s_exponents,
                p_exponents: fragment.parameters.projector_p_exponents,
            },
            Vec3::zero(),
            [constant(0.); 3],
            PointIonScope::CorrectionOnly,
            BaseFieldConvention::AlreadyShifted,
        )?;
        for lane in 0..4 {
            for (x, y) in operator[lane]
                .as_mut_slice()
                .iter_mut()
                .zip(value.operator[lane].as_slice())
            {
                *x += y;
            }
            classical_correction[lane] += value.classical_energy[lane];
            for axis in 0..3 {
                permanent_dipole[axis][lane] += value.permanent_dipole[axis][lane];
            }
        }
        let source: [GaussianDistributionPair; 3] =
            std::array::from_fn(|s| GaussianDistributionPair {
                center: positions[fragment.atom],
                exponent: fragment.parameters.charge_exponents[s],
                charge: constant(fragment.parameters.charge_coefficients[s]),
                dipole: std::array::from_fn(|world| {
                    (0..3).fold(constant(0.), |sum, body| {
                        sum + rotation[body][world]
                            * constant(fragment.parameters.body_dipoles[s].get(body))
                    })
                }),
            });
        distributions.push(source);
        internal_energy_reference += fragment.internal_energy_reference;
    }
    for i in 0..distributions.len() {
        for j in 0..i {
            for a in &distributions[i] {
                for b in &distributions[j] {
                    let value = core_jet2::gaussian_distribution_pair(a, b)?;
                    for lane in 0..4 {
                        classical_correction[lane] += value[lane];
                    }
                }
            }
        }
    }
    if operator
        .iter()
        .flat_map(|v| v.as_slice())
        .chain(&classical_correction)
        .chain(std::iter::once(&internal_energy_reference))
        .any(|v| !v.is_finite())
    {
        return invalid("rigid multi-core correction overflowed");
    }
    Ok(RigidPosePair {
        operator,
        classical_correction,
        permanent_dipole,
        internal_energy_reference,
        positions,
        coordinate_map,
    })
}
