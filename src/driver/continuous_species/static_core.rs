// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed generated-core corrections to a separately counted +1 point ion.
//! The scalar classical contribution is evaluated at zero external field.
//! Its permanent dipole and derivative are returned separately for the caller
//! to add -F.mu to energy and -F_i mu_i,R to the nuclear gradient exactly once.

use crate::basis::MolBasis;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::pseudo_sub::reduction_geometry::BodyFrameKinematics;
use crate::pseudo_sub::reduction_kernels::{
    charge_moments, charge_operator, core_interactions, projector_overlaps,
    symmetric_projector_coefficients, GaussianCharge, ProjectorTemplate,
};
use crate::{DftError, Result};

/// Fixed dictionary dimensions; these are coefficients, not orbital AOs.
#[derive(Clone, Debug)]
pub struct StaticCoreParameters {
    pub charge_exponents: [f64; 3],
    /// Signed neutral correction: the baseline +1 point ion is not repeated.
    pub charge_coefficients: [f64; 3],
    pub body_dipoles: [Vec3; 3],
    pub projector_s_exponents: [f64; 2],
    pub projector_p_exponents: [f64; 2],
    /// Upper triangle in (u,v), u<=v order. No PSD constraint is imposed.
    pub nonlocal_coefficients: [f64; 36],
}

pub struct StaticCoreCorrection {
    pub operator: Matrix,
    pub atom_derivative_matrices: Vec<[Matrix; 3]>,
    pub classical_energy: f64,
    pub classical_gradients: Vec<Vec3>,
    pub permanent_dipole: Vec3,
    /// [atom][dipole Cartesian component][atom Cartesian component].
    pub permanent_dipole_derivatives: Vec<[[f64; 3]; 3]>,
    pub charge_correction: f64,
    pub electrostatic_sites: Vec<GaussianCharge>,
    pub origin: Vec3,
}

fn invalid<T>(message: &str) -> Result<T> {
    Err(DftError::InvalidInput(message.into()))
}

/// Internal opt-in policy for explicit AO-only centers. Ordinary chart APIs
/// always use Positive; the auxiliary mask cannot enter through JSON controls.
#[derive(Clone, Debug)]
pub(super) enum IonicChargePolicy {
    Positive,
    AuxiliaryZeros(Vec<bool>),
}

impl IonicChargePolicy {
    pub(super) fn validate_charges(&self, charges: &[f64], atoms: usize) -> Result<()> {
        if charges.len() != atoms || atoms == 0 {
            return invalid("ionic charge count must match the nonempty role layout");
        }
        match self {
            Self::Positive => {
                if charges.iter().any(|z| !z.is_finite() || *z <= 0.0) {
                    return invalid("physical nuclear charges must be positive and finite");
                }
            }
            Self::AuxiliaryZeros(mask) => {
                if mask.len() != atoms
                    || charges
                        .iter()
                        .zip(mask)
                        .any(|(&z, &aux)| !z.is_finite() || if aux { z != 0.0 } else { z <= 0.0 })
                {
                    return invalid("zero ionic charges must match exactly the explicit auxiliary-role mask; every other ion must be positive");
                }
            }
        }
        Ok(())
    }
}

fn finite(v: Vec3) -> bool {
    (0..3).all(|axis| v.get(axis).is_finite())
}

fn finite_matrix(matrix: &Matrix, rows: usize, cols: usize) -> bool {
    matrix.rows == rows && matrix.cols == cols && matrix.as_slice().iter().all(|x| x.is_finite())
}

fn symmetric(matrix: &Matrix) -> bool {
    let scale = matrix
        .as_slice()
        .iter()
        .fold(1.0_f64, |a, &x| a.max(x.abs()));
    let tolerance = 128.0 * f64::EPSILON * scale * matrix.rows.max(1) as f64;
    (0..matrix.rows).all(|i| (0..i).all(|j| (matrix[(i, j)] - matrix[(j, i)]).abs() <= tolerance))
}

fn add_scaled(target: &mut Matrix, source: &Matrix, scale: f64) {
    for (value, &element) in target.as_mut_slice().iter_mut().zip(source.as_slice()) {
        *value += scale * element;
    }
}

impl StaticCoreParameters {
    pub fn validate(&self) -> Result<()> {
        if self
            .charge_exponents
            .iter()
            .chain(&self.projector_s_exponents)
            .chain(&self.projector_p_exponents)
            .any(|x| !x.is_finite() || *x <= 0.0)
            || self
                .charge_coefficients
                .iter()
                .chain(&self.nonlocal_coefficients)
                .any(|x| !x.is_finite())
            || self.body_dipoles.iter().any(|&p| !finite(p))
        {
            return invalid("static core dictionaries require finite coefficients and positive fixed Gaussian exponents");
        }
        let total: f64 = self.charge_coefficients.iter().sum();
        let scale: f64 = self.charge_coefficients.iter().map(|x| x.abs()).sum();
        if total.abs() > 64.0 * f64::EPSILON * scale.max(1.0) {
            return invalid("the generated Gaussian charge correction must be neutral; it is not silently renormalized");
        }
        Ok(())
    }
}

impl StaticCoreCorrection {
    /// Validate a frozen provider before attaching it to a molecular model.
    /// Roundoff symmetry is checked, never repaired by averaging.
    pub fn validate(&self, nao: usize, atoms: usize) -> Result<()> {
        if !finite_matrix(&self.operator, nao, nao)
            || !symmetric(&self.operator)
            || self.atom_derivative_matrices.len() != atoms
            || self
                .atom_derivative_matrices
                .iter()
                .flatten()
                .any(|m| !finite_matrix(m, nao, nao) || !symmetric(m))
            || self.classical_gradients.len() != atoms
            || self.classical_gradients.iter().any(|&g| !finite(g))
            || self.permanent_dipole_derivatives.len() != atoms
            || self
                .permanent_dipole_derivatives
                .iter()
                .flatten()
                .flatten()
                .any(|x| !x.is_finite())
            || !self.classical_energy.is_finite()
            || !finite(self.permanent_dipole)
            || !self.charge_correction.is_finite()
            || !finite(self.origin)
            || self.electrostatic_sites.len() != 3
        {
            return invalid("static core correction has incompatible dimensions, nonfinite values or a non-Hermitian operator");
        }
        let (charge, dipole) = charge_moments(&self.electrostatic_sites, self.origin)?;
        let scale = self
            .electrostatic_sites
            .iter()
            .map(|s| s.charge.abs())
            .sum::<f64>()
            .max(1.0);
        if charge.abs() > 64.0 * f64::EPSILON * scale
            || (charge - self.charge_correction).abs() > 64.0 * f64::EPSILON * scale
            || (dipole - self.permanent_dipole).norm()
                > 128.0 * f64::EPSILON * (1.0 + dipole.norm())
        {
            return invalid("static core raw charge and permanent moments are inconsistent with the saved neutral Gaussian distribution");
        }
        Ok(())
    }
}

/// Build a fixed, density-independent correction in the current reduced AO
/// basis. Retained cores interact with the Gaussian correction once, while the
/// own point-ion self term and all frozen internal/self energies belong to E0.
/// There is no short-range projector interaction with nuclei and no shape term.
#[allow(clippy::too_many_arguments)]
pub fn static_core_correction(
    basis: &MolBasis,
    positions: &[Vec3],
    ionic_charges: &[f64],
    own_ion: usize,
    frame: &BodyFrameKinematics,
    parameters: &StaticCoreParameters,
    origin: Vec3,
) -> Result<StaticCoreCorrection> {
    static_core_correction_impl(
        basis,
        positions,
        ionic_charges,
        own_ion,
        frame,
        parameters,
        origin,
        &IonicChargePolicy::Positive,
    )
}

/// Internal opt-in for explicitly absent auxiliary ions. The ordinary wrapper
/// remains strictly positive and no public controls expose this role mask.
#[allow(clippy::too_many_arguments)]
pub(crate) fn static_core_correction_with_auxiliary_roles(
    basis: &MolBasis,
    positions: &[Vec3],
    ionic_charges: &[f64],
    own_ion: usize,
    frame: &BodyFrameKinematics,
    parameters: &StaticCoreParameters,
    origin: Vec3,
    auxiliary_roles: &[bool],
) -> Result<StaticCoreCorrection> {
    static_core_correction_impl(
        basis,
        positions,
        ionic_charges,
        own_ion,
        frame,
        parameters,
        origin,
        &IonicChargePolicy::AuxiliaryZeros(auxiliary_roles.to_vec()),
    )
}

#[allow(clippy::too_many_arguments)]
fn static_core_correction_impl(
    basis: &MolBasis,
    positions: &[Vec3],
    ionic_charges: &[f64],
    own_ion: usize,
    frame: &BodyFrameKinematics,
    parameters: &StaticCoreParameters,
    origin: Vec3,
    ionic_charge_policy: &IonicChargePolicy,
) -> Result<StaticCoreCorrection> {
    parameters.validate()?;
    let atoms = positions.len();
    ionic_charge_policy.validate_charges(ionic_charges, atoms)?;
    let coordinates = atoms.checked_mul(3).ok_or_else(|| {
        DftError::InvalidInput("static core coordinate dimension overflow".into())
    })?;
    if atoms == 0
        || own_ion >= atoms
        || ionic_charges.len() != atoms
        || !basis.ecps.is_empty()
        || basis.n_core_electrons != 0
        || positions.iter().any(|&p| !finite(p))
        || (ionic_charges[own_ion] - 1.0).abs() > 64.0 * f64::EPSILON
        || !finite(origin)
        || !finite(frame.center)
        || !finite_matrix(&frame.center_jacobian, 3, coordinates)
        || !finite_matrix(&frame.orientation_jacobian, 9, coordinates)
        || basis.shells.iter().any(|shell| {
            shell.atom >= atoms
                || (shell.center - positions[shell.atom]).norm()
                    > 1e-12 * (1.0 + shell.center.norm())
        })
    {
        return invalid("static core requires matching reduced AO/ionic positions, a +1 own ion and finite compatible frame Jacobians");
    }
    let template = ProjectorTemplate {
        s_exponents: parameters.projector_s_exponents,
        p_exponents: parameters.projector_p_exponents,
        orientation: frame.orientation,
    };
    template.validate()?;
    let electrostatic_sites: Vec<_> = (0..3)
        .map(|s| GaussianCharge {
            center: frame.center,
            exponent: parameters.charge_exponents[s],
            charge: parameters.charge_coefficients[s],
            dipole: frame.orientation.mul_vec(parameters.body_dipoles[s]),
        })
        .collect();
    let (charge_correction, permanent_dipole) = charge_moments(&electrostatic_sites, origin)?;
    let electrostatic = charge_operator(basis, atoms, &electrostatic_sites)?;
    let overlaps = projector_overlaps(basis, atoms, frame.center, &template)?;
    let nonlocal = overlaps.operator(&symmetric_projector_coefficients(
        &parameters.nonlocal_coefficients,
    )?)?;
    let mut operator = electrostatic.operator;
    add_scaled(&mut operator, &nonlocal.operator, 1.0);
    let mut atom_derivative_matrices = electrostatic.atom_derivatives;
    for atom in 0..atoms {
        for axis in 0..3 {
            add_scaled(
                &mut atom_derivative_matrices[atom][axis],
                &nonlocal.atom_derivatives[atom][axis],
                1.0,
            );
        }
    }
    let mut center_derivatives = nonlocal.center_derivatives;
    for site in 0..3 {
        for axis in 0..3 {
            add_scaled(
                &mut center_derivatives[axis],
                &electrostatic.center_derivatives[site][axis],
                1.0,
            );
        }
    }
    // d(U p_body)_i / d y_j. Frame rows are column-major, unlike kernel
    // orientation derivative arrays, whose index is 3*world_row+local_axis.
    let dipole_jacobians: [Matrix; 3] = std::array::from_fn(|s| {
        let mut matrix = Matrix::zeros(3, coordinates);
        for world in 0..3 {
            for coordinate in 0..coordinates {
                matrix[(world, coordinate)] = (0..3)
                    .map(|local| {
                        parameters.body_dipoles[s].get(local)
                            * frame.orientation_jacobian[(3 * local + world, coordinate)]
                    })
                    .sum();
            }
        }
        matrix
    });
    for atom in 0..atoms {
        for axis in 0..3 {
            let coordinate = 3 * atom + axis;
            let target = &mut atom_derivative_matrices[atom][axis];
            for world in 0..3 {
                add_scaled(
                    target,
                    &center_derivatives[world],
                    frame.center_jacobian[(world, coordinate)],
                );
                for site in 0..3 {
                    add_scaled(
                        target,
                        &electrostatic.dipole_operators[site][world],
                        dipole_jacobians[site][(world, coordinate)],
                    );
                }
                for local in 0..3 {
                    add_scaled(
                        target,
                        &nonlocal.orientation_derivatives[3 * world + local],
                        frame.orientation_jacobian[(3 * local + world, coordinate)],
                    );
                }
            }
        }
    }
    let retained_indices: Vec<_> = (0..atoms).filter(|&atom| atom != own_ion).collect();
    let cores: Vec<_> = retained_indices
        .iter()
        .map(|&atom| (positions[atom], ionic_charges[atom]))
        .collect();
    let core = core_interactions(&electrostatic_sites, &cores)?;
    let mut classical_gradients = vec![Vec3::zero(); atoms];
    for (&atom, &gradient) in retained_indices.iter().zip(&core.core_gradients) {
        classical_gradients[atom] += gradient;
    }
    let mut permanent_dipole_derivatives = vec![[[0.0; 3]; 3]; atoms];
    for atom in 0..atoms {
        for axis in 0..3 {
            let coordinate = 3 * atom + axis;
            let mut classical = 0.0;
            for world in 0..3 {
                for site in 0..3 {
                    classical += core.center_gradients[site].get(world)
                        * frame.center_jacobian[(world, coordinate)]
                        + core.dipole_gradients[site].get(world)
                            * dipole_jacobians[site][(world, coordinate)];
                }
                permanent_dipole_derivatives[atom][world][axis] = charge_correction
                    * frame.center_jacobian[(world, coordinate)]
                    + (0..3)
                        .map(|s| dipole_jacobians[s][(world, coordinate)])
                        .sum::<f64>();
            }
            let mut value = classical_gradients[atom].to_array();
            value[axis] += classical;
            classical_gradients[atom] = Vec3::new(value[0], value[1], value[2]);
        }
    }
    let result = StaticCoreCorrection {
        operator,
        atom_derivative_matrices,
        classical_energy: core.energy,
        classical_gradients,
        permanent_dipole,
        permanent_dipole_derivatives,
        charge_correction,
        electrostatic_sites,
        origin,
    };
    result.validate(basis.nao_sph, atoms)?;
    Ok(result)
}

#[cfg(test)]
mod auxiliary_role_tests {
    use super::*;
    use crate::basis::data;
    use crate::math::Mat3;
    use crate::system::{Atom, Molecule};

    #[test]
    fn strict_core_wrapper_stays_positive_and_scoped_wrapper_requires_exact_mask() {
        let positions = [
            Vec3::zero(),
            Vec3::new(1.5, 0.3, 0.0),
            Vec3::new(-0.4, 1.3, 0.7),
        ];
        let mol = Molecule::new(
            positions
                .iter()
                .map(|&position| Atom { z: 1, position })
                .collect(),
        );
        let basis = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
        let mut center_jacobian = Matrix::zeros(3, 9);
        for axis in 0..3 {
            center_jacobian[(axis, axis)] = 1.0;
        }
        let frame = BodyFrameKinematics {
            center: positions[0],
            orientation: Mat3::from_columns(
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ),
            orientation_jacobian: Matrix::zeros(9, 9),
            center_jacobian,
        };
        let params = StaticCoreParameters {
            charge_exponents: [0.2, 0.8, 3.2],
            charge_coefficients: [0.0; 3],
            body_dipoles: [Vec3::zero(); 3],
            projector_s_exponents: [0.35, 1.4],
            projector_p_exponents: [0.35, 1.4],
            nonlocal_coefficients: [0.0; 36],
        };
        assert!(static_core_correction(
            &basis,
            &positions,
            &[1.0, 1.0, 0.0],
            0,
            &frame,
            &params,
            Vec3::zero()
        )
        .is_err());
        for mask in [
            &[false, false][..],
            &[false, false, false][..],
            &[true, false, true][..],
        ] {
            assert!(static_core_correction_with_auxiliary_roles(
                &basis,
                &positions,
                &[1.0, 1.0, 0.0],
                0,
                &frame,
                &params,
                Vec3::zero(),
                mask
            )
            .is_err());
        }
        let zero = static_core_correction_with_auxiliary_roles(
            &basis,
            &positions,
            &[1.0, 1.0, 0.0],
            0,
            &frame,
            &params,
            Vec3::zero(),
            &[false, false, true],
        )
        .unwrap();
        assert_eq!(zero.classical_energy, 0.0);
        assert_eq!(zero.charge_correction, 0.0);
        let ordinary = static_core_correction(
            &basis,
            &positions,
            &[1.0; 3],
            0,
            &frame,
            &params,
            Vec3::zero(),
        )
        .unwrap();
        let scoped = static_core_correction_with_auxiliary_roles(
            &basis,
            &positions,
            &[1.0; 3],
            0,
            &frame,
            &params,
            Vec3::zero(),
            &[false; 3],
        )
        .unwrap();
        assert_eq!(ordinary.operator.as_slice(), scoped.operator.as_slice());
        assert_eq!(ordinary.classical_energy, scoped.classical_energy);
    }
}
