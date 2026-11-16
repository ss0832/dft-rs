// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Point-nucleus contact and spin-dipolar hyperfine tensors of a supplied
//! collinear all-electron spin density. Couplings are in MHz per nuclear g.
//! The tensor belongs to that electronic state; no degenerate-state average
//! or relativistic effective-spin projection is implied.

pub mod nuclear;

use crate::basis::MolBasis;
use crate::constants::{
    ELECTRON_G_FACTOR, FINE_STRUCTURE, HARTREE_TO_JOULE, HBAR_J_S, PROTON_ELECTRON_MASS_RATIO,
};
use crate::grid::eval::eval_aos;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::{DftError, Result};
use std::f64::consts::PI;

/// Principal-value dipolar and contact AO operators, in inverse Bohr cubed.
/// The raw Poisson and AO Hermiticity residuals precede any projection.
pub struct SpinDipoleOperators {
    pub contact: Matrix,
    /// Cartesian order `[3*a+b][bra AO][ket AO]`.
    pub dipolar: [Matrix; 9],
    pub poisson_residual: f64,
    pub hermiticity_residual: f64,
}

pub fn spin_dipole_operators(basis: &MolBasis, center: Vec3) -> Result<SpinDipoleOperators> {
    let n = basis.nao_sph;
    if n == 0 || !center.norm().is_finite() {
        return Err(DftError::InvalidInput(
            "hyperfine operators require AOs and a finite probe position".into(),
        ));
    }
    let ao = eval_aos(basis, &[center], false);
    let mut contact = Matrix::zeros(n, n);
    for a in 0..n {
        for b in 0..n {
            contact[(a, b)] = ao.phi[(0, a)] * ao.phi[(0, b)];
        }
    }
    let hessian = crate::response::spin_spin::field_gradient_operator(basis, center.to_array())?;
    let dipolar = std::array::from_fn(|component| {
        let mut matrix = hessian[component].clone();
        if component / 3 == component % 3 {
            for (value, rho) in matrix.as_mut_slice().iter_mut().zip(contact.as_slice()) {
                *value += 4.0 * PI / 3.0 * rho;
            }
        }
        matrix
    });
    let mut poisson_residual = 0.0_f64;
    let mut hermiticity_residual = 0.0_f64;
    for a in 0..n {
        for b in 0..n {
            poisson_residual = poisson_residual
                .max((dipolar[0][(a, b)] + dipolar[4][(a, b)] + dipolar[8][(a, b)]).abs());
            for matrix in &dipolar {
                hermiticity_residual =
                    hermiticity_residual.max((matrix[(a, b)] - matrix[(b, a)]).abs());
            }
        }
    }
    if dipolar
        .iter()
        .flat_map(|m| m.as_slice())
        .chain(contact.as_slice())
        .any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput(
            "non-finite contact or spin-dipolar operator".into(),
        ));
    }
    Ok(SpinDipoleOperators {
        contact,
        dipolar,
        poisson_residual,
        hermiticity_residual,
    })
}

/// `total = contact I + dipolar`, without a symmetry or trace projection.
/// Values returned by `spin_hyperfine_tensor` are MHz per nuclear g;
/// `scaled(g)` returns MHz. Principal values are ascending, axes are columns.
#[derive(Clone, Debug)]
pub struct HyperfineTensor {
    pub contact: f64,
    pub dipolar: [[f64; 3]; 3],
    pub total: [[f64; 3]; 3],
    pub principal_values: [f64; 3],
    pub principal_axes: [[f64; 3]; 3],
    pub dipolar_trace: f64,
    pub symmetry_residual: f64,
    pub eigen_residual: f64,
    pub axes_orthogonality_residual: f64,
}
impl HyperfineTensor {
    fn new(contact: f64, dipolar: [[f64; 3]; 3]) -> Result<Self> {
        if !contact.is_finite() || dipolar.iter().flatten().any(|v| !v.is_finite()) {
            return Err(DftError::InvalidInput(
                "hyperfine tensor entries must be finite".into(),
            ));
        }
        let total = std::array::from_fn(|a| {
            std::array::from_fn(|b| dipolar[a][b] + if a == b { contact } else { 0.0 })
        });
        let mut matrix = Matrix::zeros(3, 3);
        for a in 0..3 {
            for b in 0..3 {
                matrix[(a, b)] = total[a][b];
            }
        }
        let (values, axes) = symmetric_eigen(&matrix)?;
        let principal_values = std::array::from_fn(|a| values[a]);
        let principal_axes = std::array::from_fn(|a| std::array::from_fn(|b| axes[(a, b)]));
        let mut symmetry_residual = 0.0_f64;
        let mut eigen_residual = 0.0_f64;
        let mut axes_orthogonality_residual = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                symmetry_residual = symmetry_residual.max((total[a][b] - total[b][a]).abs());
                let action = (0..3)
                    .map(|c| total[a][c] * principal_axes[c][b])
                    .sum::<f64>();
                eigen_residual =
                    eigen_residual.max((action - principal_axes[a][b] * principal_values[b]).abs());
                let metric = (0..3)
                    .map(|c| principal_axes[c][a] * principal_axes[c][b])
                    .sum::<f64>();
                axes_orthogonality_residual = axes_orthogonality_residual
                    .max((metric - if a == b { 1.0 } else { 0.0 }).abs());
            }
        }
        let dipolar_trace = (0..3).map(|a| dipolar[a][a]).sum();
        Ok(Self {
            contact,
            dipolar,
            total,
            principal_values,
            principal_axes,
            dipolar_trace,
            symmetry_residual,
            eigen_residual,
            axes_orthogonality_residual,
        })
    }
    /// Re-diagonalize the signed physical tensor, including negative or zero
    /// nuclear g factors, so values and matching axes retain ascending order.
    pub fn scaled(&self, nuclear_g: f64) -> Result<Self> {
        if !nuclear_g.is_finite() {
            return Err(DftError::InvalidInput(
                "nuclear g factor must be finite".into(),
            ));
        }
        Self::new(
            self.contact * nuclear_g,
            self.dipolar.map(|row| row.map(|v| v * nuclear_g)),
        )
    }
    /// Relative algebraic residual normalized by the actual full tensor norm.
    /// The identically zero tensor has zero residual; no floor is introduced.
    pub fn relative_residual(&self) -> f64 {
        let norm = self
            .total
            .iter()
            .flatten()
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        let residual = self
            .dipolar_trace
            .abs()
            .max(self.symmetry_residual)
            .max(self.eigen_residual);
        if norm == 0.0 {
            if residual == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            residual / norm
        }
    }
}

/// Functional-independent expectation value in a collinear spin density
/// `P_alpha-P_beta`, normalized by `2 M_S`. The molecular common driver
/// supplies the converged reference and masks ECP nuclei before calling it.
pub fn spin_hyperfine_tensor(
    basis: &MolBasis,
    spin_density: &Matrix,
    center: Vec3,
    twice_spin: f64,
) -> Result<HyperfineTensor> {
    let n = basis.nao_sph;
    if spin_density.rows != n
        || spin_density.cols != n
        || !twice_spin.is_finite()
        || twice_spin == 0.0
        || spin_density.as_slice().iter().any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput("hyperfine tensor requires a finite AO spin density and a finite nonzero spin projection".into()));
    }
    let operator = spin_dipole_operators(basis, center)?;
    let scale = FINE_STRUCTURE * FINE_STRUCTURE * ELECTRON_G_FACTOR
        / (4.0 * PROTON_ELECTRON_MASS_RATIO * twice_spin)
        * HARTREE_TO_JOULE
        / (2.0 * PI * HBAR_J_S)
        * 1e-6;
    let contact = 8.0 * PI / 3.0 * spin_density.frobenius_dot(&operator.contact) * scale;
    let dipolar = std::array::from_fn(|a| {
        std::array::from_fn(|b| spin_density.frobenius_dot(&operator.dipolar[3 * a + b]) * scale)
    });
    HyperfineTensor::new(contact, dipolar)
}
