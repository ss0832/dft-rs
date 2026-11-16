// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Explicit fixed one-, two- and three-center roles for experimental reduced models.
use crate::basis::MolBasis;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::pseudo_sub::reduction_basis::{build_reduction_basis, OneCenterBasisTemplate};
use crate::pseudo_sub::reduction_geometry::{BodyFrameKinematics, RigidExpansion};
use crate::system::{Atom, Molecule};
use crate::{DftError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CenterTemplate {
    CoreSp,
    CoreSpH,
    CoreSpHH,
}
impl CenterTemplate {
    pub fn center_count(self) -> usize {
        match self {
            Self::CoreSp => 1,
            Self::CoreSpH => 2,
            Self::CoreSpHH => 3,
        }
    }
    pub fn id(self) -> &'static str {
        match self {
            Self::CoreSp => "one-center-core-valence-sp-local3-projector8-charge3-v1",
            Self::CoreSpH => "two-center-core-valence-sp-h1s-local3-projector8-charge3-v1",
            Self::CoreSpHH => "three-center-core-valence-sp-h1s-h1s-local3-projector8-charge3-v1",
        }
    }
}

/// Basis role labels are NOT physical nuclear charges. Never use this
/// molecule's n_electrons() or ordinary SCF/nuclear repulsion to define energy.
pub struct RoleLayout {
    pub template: CenterTemplate,
    pub basis_role_molecule: Molecule,
    pub ionic_charges: Vec<f64>,
    pub main_ion: usize,
    /// Retained physical nuclei only; the main effective ion and AO-only
    /// auxiliary roles have separate provenance and are not added here.
    pub retained_nucleus_roles: Vec<usize>,
    pub auxiliary_roles: Vec<usize>,
    /// Ordered X/G/reference roles used by the existing frame contract.
    pub frame_anchor_roles: [usize; 3],
    pub role_to_full_geometry: Vec<usize>,
    /// All role centers are quadrature sites in the proposed production grid.
    /// Their immutable positive labels supply partition radii, not ionic charge.
    pub grid_site_roles: Vec<usize>,
    /// d(role Cartesian coordinates)/d(existing rigid reduced coordinates).
    pub geometry_jacobian: Matrix,
    /// Partial frame derivatives in the independent role-center coordinates.
    /// Aux columns zero; pullback through geometry_jacobian happens once later.
    pub frame_in_role_coordinates: BodyFrameKinematics,
}

pub fn layout(
    map: &RigidExpansion,
    source: &Molecule,
    y: &[Vec3],
    template: CenterTemplate,
) -> Result<RoleLayout> {
    let invalid = |s: &str| DftError::InvalidInput(s.into());
    let spec = map
        .fragment_spec()
        .ok_or_else(|| invalid("few-center roles require an explicit methyl map"))?;
    if spec.electronic_centers != 1
        || source.len() != map.full_atom_count()
        || source
            .atoms
            .iter()
            .zip(&map.reference_molecule().atoms)
            .any(|(a, b)| a.z != b.z)
    {
        return Err(invalid(
            "few-center role layout requires the validated original methyl atom order",
        ));
    }
    let full = map.evaluate(y)?;
    let frame = map
        .body_frame_kinematics(y)?
        .ok_or_else(|| invalid("missing rigid port frame"))?;
    let main = map
        .electronic_center_index()
        .ok_or_else(|| invalid("missing effective ionic center"))?;
    if main + 1 != y.len() {
        return Err(invalid(
            "rigid reduced order must contain retained atoms followed by main center",
        ));
    }
    let mut role_to_full = map.retained_atoms().to_vec();
    role_to_full.push(spec.port.fragment_atom);
    // Exact explicit ordered H correspondence; no energy-based choice or fit.
    for &(full_h, _) in map
        .internal_offsets()
        .iter()
        .take(template.center_count() - 1)
    {
        role_to_full.push(full_h);
    }
    let n = role_to_full.len();
    let mut atoms = Vec::with_capacity(n);
    let mut ionic = Vec::with_capacity(n);
    for (role, &full_id) in role_to_full.iter().enumerate() {
        let (z, charge) = match role.cmp(&main) {
            std::cmp::Ordering::Less => {
                (source.atoms[full_id].z, f64::from(source.atoms[full_id].z))
            }
            std::cmp::Ordering::Equal => (1, 1.0),
            std::cmp::Ordering::Greater => (1, 0.0),
        };
        atoms.push(Atom {
            z,
            position: full.molecule.atoms[full_id].position,
        });
        ionic.push(charge);
    }
    let mut j = Matrix::zeros(3 * n, 3 * y.len());
    for (role, &full_id) in role_to_full.iter().enumerate() {
        for axis in 0..3 {
            for col in 0..j.cols {
                j[(3 * role + axis, col)] = full.jacobian[(3 * full_id + axis, col)];
            }
        }
    }
    let mut uj = Matrix::zeros(9, 3 * n);
    let mut cj = Matrix::zeros(3, 3 * n);
    for row in 0..9 {
        for col in 0..3 * y.len() {
            uj[(row, col)] = frame.orientation_jacobian[(row, col)];
        }
    }
    for row in 0..3 {
        for col in 0..3 * y.len() {
            cj[(row, col)] = frame.center_jacobian[(row, col)];
        }
    }
    let retained_role = |full| {
        map.retained_atoms()
            .iter()
            .position(|&i| i == full)
            .ok_or_else(|| invalid("a declared frame anchor is not retained"))
    };
    let frame_anchor_roles = [
        retained_role(spec.port.host_atom)?,
        main,
        retained_role(spec.port.reference_atom)?,
    ];
    Ok(RoleLayout {
        template,
        basis_role_molecule: Molecule::new(atoms),
        ionic_charges: ionic,
        main_ion: main,
        retained_nucleus_roles: (0..main).collect(),
        auxiliary_roles: ((main + 1)..n).collect(),
        frame_anchor_roles,
        role_to_full_geometry: role_to_full,
        grid_site_roles: (0..n).collect(),
        geometry_jacobian: j,
        frame_in_role_coordinates: BodyFrameKinematics {
            center: frame.center,
            orientation: frame.orientation,
            orientation_jacobian: uj,
            center_jacobian: cj,
        },
    })
}

impl RoleLayout {
    pub fn basis(&self) -> Result<MolBasis> {
        build_reduction_basis(
            &self.basis_role_molecule,
            self.main_ion,
            OneCenterBasisTemplate::CarbonCoreValenceSp,
        )
    }
    /// Use gradients or forces consistently on both sides; no torque is added
    /// again after this pullback. Occupancy is supplied by the separate ledger.
    pub fn pullback(&self, role_values: &[Vec3]) -> Result<Vec<Vec3>> {
        if role_values.len() != self.basis_role_molecule.len()
            || role_values
                .iter()
                .any(|v| (0..3).any(|axis| !v.get(axis).is_finite()))
        {
            return Err(DftError::InvalidInput(
                "role pullback requires one finite vector per role center".into(),
            ));
        }
        let mut result = vec![Vec3::zero(); self.geometry_jacobian.cols / 3];
        for (column, value) in result.iter_mut().enumerate() {
            let components = std::array::from_fn::<_, 3, _>(|axis| {
                role_values
                    .iter()
                    .enumerate()
                    .map(|(role, v)| {
                        (0..3)
                            .map(|a| {
                                self.geometry_jacobian[(3 * role + a, 3 * column + axis)] * v.get(a)
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>()
            });
            *value = Vec3::new(components[0], components[1], components[2]);
        }
        Ok(result)
    }
}
