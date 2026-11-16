// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Spin-covariant Hubbard fields at fixed real, spin-independent projectors.

use super::{eri_index, trace, HubbardModel, Interaction, IntersiteEnergy};
use crate::linalg::Matrix;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::occupations::OccupationScheme;
use crate::periodic::spinor_scf::{SpinorEnergyBackend, SpinorFields};
use crate::{DftError, Result};

/// Site energy terms in Hartree and occupations in spin-major site order.
pub struct SpinorSiteEnergy {
    pub interaction: f64,
    pub double_counting: f64,
    pub occupations: BlochMatrix,
}

/// Hubbard energy and variational AO potential for a molecular spinor density.
pub struct SpinorHubbardFields {
    pub energy: f64,
    pub potential: BlochMatrix,
    pub sites: Vec<SpinorSiteEnergy>,
    pub pairs: Vec<IntersiteEnergy>,
}

impl HubbardModel {
    /// Apply the same spatial projectors to all four blocks of the density.
    pub fn evaluate_spinor(&self, density: &BlochMatrix) -> Result<SpinorHubbardFields> {
        self.spinor_fields(density, true)
    }

    /// Analytic derivative of the Hubbard potential at fixed projectors.
    pub fn response_spinor(&self, direction: &BlochMatrix) -> Result<BlochMatrix> {
        Ok(self.spinor_fields(direction, false)?.potential)
    }

    fn spinor_fields(
        &self,
        density: &BlochMatrix,
        include_constant: bool,
    ) -> Result<SpinorHubbardFields> {
        let width = 2 * self.dimension;
        if [&density.re, &density.im].iter().any(|matrix| {
            matrix.rows != width
                || matrix.cols != width
                || matrix.as_slice().iter().any(|entry| !entry.is_finite())
        }) || density.hermiticity_error() > 1e-12
        {
            return Err(DftError::InvalidInput(
                "Hubbard spinor density must be a finite Hermitian matrix in spin-major AO order"
                    .into(),
            ));
        }
        let mut fields = SpinorHubbardFields {
            energy: 0.0,
            potential: BlochMatrix::zeros(width, width),
            sites: Vec::with_capacity(self.sites.len()),
            pairs: Vec::with_capacity(self.pairs.len()),
        };
        for site in &self.sites {
            let occupations = transform_blocks(density, &site.covariant);
            let (interaction, double_counting, potential) = site
                .interaction
                .spinor_fields(&occupations, include_constant);
            fields.potential.add_scaled(
                &transform_blocks(&potential, &site.covariant.transpose()),
                1.0,
            );
            fields.energy += interaction - double_counting;
            fields.sites.push(SpinorSiteEnergy {
                interaction,
                double_counting,
                occupations,
            });
        }
        for &pair in &self.pairs {
            let [left, right] = pair.sites.map(|i| &self.sites[i].covariant);
            let left_width = left.cols;
            let site_width = left.cols + right.cols;
            let mut projectors = Matrix::zeros(self.dimension, site_width);
            for i in 0..self.dimension {
                for j in 0..left.cols {
                    projectors[(i, j)] = left[(i, j)];
                }
                for j in 0..right.cols {
                    projectors[(i, left_width + j)] = right[(i, j)];
                }
            }
            // Sum all spin blocks of the intersite coherence. This is the
            // spin-covariant Frobenius norm of n^{IJ} and reduces to separate
            // alpha and beta terms for a collinear density.
            let mut mixed = transform_blocks(density, &projectors);
            let mut bond_order = 0.0;
            for i in 0..2 * site_width {
                for j in 0..2 * site_width {
                    let left_i = i % site_width < left_width;
                    let left_j = j % site_width < left_width;
                    let re = &mut mixed.re[(i, j)];
                    let im = &mut mixed.im[(i, j)];
                    if left_i && !left_j {
                        bond_order += re.powi(2) + im.powi(2);
                    }
                    let scale = if left_i == left_j { 0.0 } else { -pair.v };
                    *re *= scale;
                    *im *= scale;
                }
            }
            fields.potential.add_scaled(&transform_blocks(&mixed, &projectors.transpose()), 1.0);
            let [left, right] = pair.sites.map(|i| trace(&fields.sites[i].occupations.re));
            let terms = pair.energy(left, right, bond_order);
            fields.energy += terms.correction;
            fields.pairs.push(terms);
        }
        Ok(fields)
    }
}

/// Congruence with the same real (possibly rectangular) map in each spin block.
fn transform_blocks(matrix: &BlochMatrix, transform: &Matrix) -> BlochMatrix {
    let mut result = BlochMatrix::zeros(2 * transform.cols, 2 * transform.cols);
    for (source, target) in [(&matrix.re, &mut result.re), (&matrix.im, &mut result.im)] {
        for s in 0..2 {
            for t in 0..2 {
                let mut block = Matrix::zeros(transform.rows, transform.rows);
                for a in 0..transform.rows {
                    for b in 0..transform.rows {
                        block[(a, b)] = source[(s * transform.rows + a, t * transform.rows + b)];
                    }
                }
                let transformed = transform.transpose_matmul_seq(&block.matmul_seq(transform));
                for a in 0..transform.cols {
                    for b in 0..transform.cols {
                        target[(s * transform.cols + a, t * transform.cols + b)] =
                            transformed[(a, b)];
                    }
                }
            }
        }
    }
    result
}

impl Interaction {
    fn spinor_fields(
        &self,
        occupation: &BlochMatrix,
        include_constant: bool,
    ) -> (f64, f64, BlochMatrix) {
        let half = if include_constant { 0.5 } else { 0.0 };
        let total = trace(&occupation.re);
        match self {
            Self::Dudarev { u_eff } => {
                let energy = u_eff * (half * total - 0.5 * occupation.trace_product(occupation));
                let mut potential = occupation.clone();
                potential.scale(-u_eff);
                for a in 0..potential.rows() {
                    potential.re[(a, a)] += half * u_eff;
                }
                (energy, 0.0, potential)
            }
            Self::Liechtenstein { u, j, integrals } => {
                let m = occupation.rows() / 2;
                let mut potential = BlochMatrix::zeros(2 * m, 2 * m);
                // The direct operator sees the spin trace. The exchange
                // operator contracts every spin block, including the phases.
                for a in 0..m {
                    for b in 0..m {
                        let mut direct = 0.0;
                        for c in 0..m {
                            for d in 0..m {
                                direct += integrals[eri_index(m, a, b, c, d)]
                                    * (occupation.re[(c, d)] + occupation.re[(m + c, m + d)]);
                                let exchange = integrals[eri_index(m, a, c, b, d)];
                                for s in 0..2 {
                                    for t in 0..2 {
                                        let target = (s * m + a, t * m + b);
                                        let source = (s * m + c, t * m + d);
                                        potential.re[target] -= exchange * occupation.re[source];
                                        potential.im[target] -= exchange * occupation.im[source];
                                    }
                                }
                            }
                        }
                        potential.re[(a, b)] += direct;
                        potential.re[(m + a, m + b)] += direct;
                    }
                }
                let energy = 0.5 * occupation.trace_product(&potential);
                // FLL uses the spin density traced over spatial orbitals.
                // Tr(T^2) gives the squared spin populations in any spin frame.
                let mut spin = BlochMatrix::zeros(2, 2);
                for s in 0..2 {
                    for t in 0..2 {
                        for a in 0..m {
                            spin.re[(s, t)] += occupation.re[(s * m + a, t * m + a)];
                            spin.im[(s, t)] += occupation.im[(s * m + a, t * m + a)];
                        }
                    }
                }
                let double_counting = 0.5
                    * (u * total * (total - 2.0 * half)
                        - j * (spin.trace_product(&spin) - 2.0 * half * total));
                for s in 0..2 {
                    for t in 0..2 {
                        for a in 0..m {
                            let target = (s * m + a, t * m + a);
                            potential.re[target] += j * spin.re[(s, t)];
                            potential.im[target] += j * spin.im[(s, t)];
                            if s == t {
                                potential.re[target] -= u * (total - half) + j * half;
                            }
                        }
                    }
                }
                (energy, double_counting, potential)
            }
        }
    }
}

/// Adds the fixed-projector density functional to the molecular spinor Fock.
pub(crate) struct HubbardSpinorBackend<'a, B> {
    pub(crate) inner: &'a B,
    pub(crate) hubbard: &'a HubbardModel,
}

impl<B: SpinorEnergyBackend> SpinorEnergyBackend for HubbardSpinorBackend<'_, B> {
    fn overlap(&self) -> &[BlochMatrix] {
        self.inner.overlap()
    }

    fn initial_fock(&self) -> &[BlochMatrix] {
        self.inner.initial_fock()
    }

    fn electrons(&self) -> f64 {
        self.inner.electrons()
    }

    fn validate_occupation(&self, occupation: OccupationScheme) -> Result<()> {
        self.inner.validate_occupation(occupation)
    }

    fn quadratic_density_energy(&self) -> bool {
        self.inner.quadratic_density_energy()
    }

    fn evaluate(&self, density: &[BlochMatrix]) -> Result<SpinorFields> {
        let [density] = density else {
            return Err(DftError::InvalidInput(
                "molecular Hubbard spinors require a single AO density".into(),
            ));
        };
        let mut fields = self.inner.evaluate(std::slice::from_ref(density))?;
        let correction = self.hubbard.evaluate_spinor(density)?;
        fields.fock[0].add_scaled(&correction.potential, 1.0);
        // The report adds Hubbard as a separate term; the parent slots retain
        // their meaning for energies provided by the wrapped backend.
        fields.energy += correction.energy;
        Ok(fields)
    }
}
