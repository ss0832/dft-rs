// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Two-component transformations of molecular kinetically balanced operators.

use crate::dfpt::orbitals::{adjoint, matmul};
use crate::periodic::BlochMatrix;
use crate::{DftError, Result};

mod approximate;
mod grid;
mod matrix;
mod molecular;
mod x2c;
pub use approximate::{dkh2_taylor, matrix_zora_taylor};
pub use molecular::{
    DiracCoreSourceResponse, MolecularRelativisticOptions, MolecularTwoComponent,
    TwoComponentMethod,
};
pub(crate) use molecular::{NuclearMagneticFields, UniformMagneticBasis};
pub use x2c::x2c_taylor;

/// Scalar treatment discards the spin-vector part of the nuclear one-body
/// operator before decoupling. Spin-orbit treatment keeps all Pauli blocks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelativisticSpin {
    Scalar,
    SpinOrbit,
}

/// Take half the Pauli spin trace and repeat it in each spin diagonal.
/// Both the large and the small component sectors are spin-major.
pub fn spin_scalar_operator(operator: &BlochMatrix) -> Result<BlochMatrix> {
    let size = operator.rows();
    if size == 0
        || size % 4 != 0
        || operator.cols() != size
        || operator.im.rows != size
        || operator.im.cols != size
    {
        return Err(DftError::InvalidInput(
            "the scalar relativistic operator requires equal four-component spin-major blocks"
                .into(),
        ));
    }
    let n = size / 4;
    let mut scalar = BlochMatrix::zeros(size, size);
    for sector_i in [0, 2 * n] {
        for sector_j in [0, 2 * n] {
            for i in 0..n {
                for j in 0..n {
                    let (p, q) = (sector_i + i, sector_j + j);
                    let re = 0.5 * (operator.re[(p, q)] + operator.re[(p + n, q + n)]);
                    let im = 0.5 * (operator.im[(p, q)] + operator.im[(p + n, q + n)]);
                    scalar.re[(p, q)] = re;
                    scalar.re[(p + n, q + n)] = re;
                    scalar.im[(p, q)] = im;
                    scalar.im[(p + n, q + n)] = im;
                }
            }
        }
    }
    Ok(scalar)
}

/// One common embedding for the two-component Hamiltonian and observables.
/// All series coefficients are factorial-normalized. The target AO chart
/// is orthonormal and spin-major with the axes of the original large basis.
#[derive(Clone, Debug)]
pub struct TwoComponentTaylor {
    embedding: Vec<BlochMatrix>,
    /// Rest-energy-subtracted electronic core Hamiltonian in Hartree.
    pub hamiltonian: Vec<BlochMatrix>,
    /// Raw maximum element of the metric-identity defect at each order.
    pub metric_defects: Vec<f64>,
    /// Raw maximum element of H B - S B H2 at each order, in Hartree.
    /// For approximate decouplings this also measures model truncation.
    pub decoupling_defects: Vec<f64>,
}

impl TwoComponentTaylor {
    pub fn embedding(&self) -> &[BlochMatrix] {
        &self.embedding
    }

    /// Change the picture of a covariant one-electron operator, B† O B.
    /// Missing higher coefficients of the operator represent exact zero.
    pub fn transform(&self, operator: &[BlochMatrix]) -> Result<Vec<BlochMatrix>> {
        let order = self.embedding.len() - 1;
        matrix::validate_hermitian(operator, self.embedding[0].rows(), order)?;
        matrix::congruence(&self.embedding, operator, order)
    }

    /// Lift the two-component contravariant density to the balanced AO
    /// picture. Contraction with the full and transformed operators agrees.
    pub fn lift_density(&self, density: &BlochMatrix) -> Result<BlochMatrix> {
        let b = &self.embedding[0];
        matrix::validate_hermitian(std::slice::from_ref(density), b.cols(), 0)?;
        matmul(b, &matmul(density, &adjoint(b)?)?)
    }
}

fn finish_embedding(
    embedding: Vec<BlochMatrix>,
    effective: Vec<BlochMatrix>,
    h: &[BlochMatrix],
    s: &[BlochMatrix],
) -> Result<TwoComponentTaylor> {
    use matrix::{congruence, identity, max_abs, product};
    let order = embedding.len() - 1;
    let mut metric = congruence(&embedding, s, order)?;
    metric[0].add_scaled(&identity(embedding[0].cols()), -1.0);
    let mut residual = product(h, &embedding, order)?;
    let rhs = product(&product(s, &embedding, order)?, &effective, order)?;
    for (left, right) in residual.iter_mut().zip(rhs) {
        left.add_scaled(&right, -1.0);
    }
    if embedding
        .iter()
        .chain(&effective)
        .chain(&metric)
        .chain(&residual)
        .any(|a| {
            a.re.as_slice()
                .iter()
                .chain(a.im.as_slice())
                .any(|v| !v.is_finite())
        })
    {
        return Err(DftError::LinearAlgebra(
            "nonfinite two-component embedding, Hamiltonian or residual".into(),
        ));
    }
    Ok(TwoComponentTaylor {
        embedding,
        hamiltonian: effective,
        metric_defects: metric.iter().map(max_abs).collect(),
        decoupling_defects: residual.iter().map(max_abs).collect(),
    })
}
