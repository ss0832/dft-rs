// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Density differential of the periodic spinor Fock at a fixed basis and mesh.

use super::{
    from_pauli, lda_collocate, lda_potential, lda_spin_frame, pauli_components, spin_diagonal,
    valid, BlochMatrix, CoulombKernel, Family, GridVars, SpinorDensityKernel, SpinorDftBackend,
    XcOut,
};
use crate::periodic::spinor_xc::GridKernel;
use crate::{DftError, Result};

pub(super) struct DensityKernel<'a, 'system> {
    backend: &'a SpinorDftBackend<'system>,
    local: Option<LocalKernel>,
}

enum LocalKernel {
    Lda(Vec<LdaPointKernel>),
    Projected(GridKernel),
}

impl<'a, 'system> DensityKernel<'a, 'system> {
    pub(super) fn new(
        backend: &'a SpinorDftBackend<'system>,
        density: &[BlochMatrix],
    ) -> Result<Self> {
        if density.len() != backend.overlap.len()
            || density.iter().any(|p| !valid(p, backend.overlap[0].rows()))
        {
            return Err(DftError::InvalidInput(
                "periodic spinor kernel requires finite Hermitian reference densities on its SCF mesh"
                    .into(),
            ));
        }
        let local = if let Some(grid) = backend.scalar.xc_grid() {
            if grid.nonlocal_options().is_some() {
                return Err(DftError::Unsupported {
                    feature: "periodic spinor density kernel".into(),
                    missing: "the nonlocal XC density kernel for the selected provider".into(),
                });
            }
            let pauli = density
                .iter()
                .map(pauli_components)
                .collect::<Result<Vec<_>>>()?;
            let aos: Vec<_> = grid.aos.iter().map(|ao| &ao.phi).collect();
            if backend.functional.family != Family::Lda {
                let momentum: Vec<_> = backend.xc_momentum.iter().collect();
                let projected = GridKernel::new(
                    backend.functional,
                    &grid.weights,
                    &aos,
                    &momentum,
                    &pauli,
                    backend.scalar.xc_tolerances()[0],
                )?;
                return Ok(Self {
                    backend,
                    local: Some(LocalKernel::Projected(projected)),
                });
            }
            let (rho, frame) = lda_spin_frame(&lda_collocate(grid.weights.len(), &aos, &pauli))?;
            let mut out = XcOut::default();
            backend.functional.eval(
                &GridVars {
                    rho: [&rho[0], &rho[1]],
                    sigma: None,
                    tau: None,
                },
                2,
                &mut out,
            )?;
            let lda = frame
                .into_iter()
                .enumerate()
                .map(|(g, (magnitude, axis))| {
                    let (v, f) = out.potential_and_kernel(g, Family::Lda)?;
                    let parallel = 0.5 * (f[0][0] - 2.0 * f[0][1] + f[1][1]);
                    let density = rho[0][g] + rho[1][g];
                    // The divided difference tends to the isotropic spin
                    // curvature as |m| approaches zero. Use that limit before
                    // subtraction of the two spin potentials loses precision.
                    let transverse = if magnitude > density * f64::EPSILON.sqrt() {
                        (v[0] - v[1]) / magnitude
                    } else {
                        parallel
                    };
                    Ok(LdaPointKernel {
                        axis,
                        charge: 0.5 * (f[0][0] + 2.0 * f[0][1] + f[1][1]),
                        mixed: 0.5 * (f[0][0] - f[1][1]),
                        parallel,
                        transverse,
                    })
                })
                .collect::<Result<_>>()?;
            Some(LocalKernel::Lda(lda))
        } else {
            None
        };
        Ok(Self { backend, local })
    }
}

impl SpinorDensityKernel for DensityKernel<'_, '_> {
    fn apply(&self, direction: &[BlochMatrix]) -> Result<Vec<BlochMatrix>> {
        let backend = self.backend;
        if direction.len() != backend.overlap.len()
            || direction
                .iter()
                .any(|p| !valid(p, backend.overlap[0].rows()))
        {
            return Err(DftError::InvalidInput(
                "periodic spinor kernel requires one finite Hermitian direction per k point".into(),
            ));
        }
        let pauli = direction
            .iter()
            .map(pauli_components)
            .collect::<Result<Vec<_>>>()?;
        let charge: Vec<_> = pauli.iter().map(|p| p[0].clone()).collect();
        let mut result: Vec<_> = backend
            .scalar
            .coulomb_backend()
            .hartree_action(&charge)?
            .iter()
            .map(spin_diagonal)
            .collect();
        let hybrid = backend.functional.hybrid;
        for (coefficient, kernel) in [
            (hybrid.cx, CoulombKernel::Coulomb),
            (
                hybrid.clr,
                CoulombKernel::Erf {
                    omega: hybrid.omega,
                },
            ),
        ] {
            if coefficient == 0.0 {
                continue;
            }
            let n = charge[0].rows();
            let mut exchange: Vec<[BlochMatrix; 4]> = (0..charge.len())
                .map(|_| std::array::from_fn(|_| BlochMatrix::zeros(n, n)))
                .collect();
            // Exchange acts independently on each Pauli component. The
            // inverse Pauli transform restores the full complex spin blocks.
            for component in 0..4 {
                let density: Vec<_> = pauli.iter().map(|p| p[component].clone()).collect();
                for (blocks, k) in exchange
                    .iter_mut()
                    .zip(backend.scalar.exchange(&density, kernel)?)
                {
                    blocks[component] = k;
                }
            }
            for (value, blocks) in result.iter_mut().zip(&exchange) {
                value.add_scaled(&from_pauli(blocks)?, -coefficient);
            }
        }
        if let (Some(grid), Some(kernel)) = (backend.scalar.xc_grid(), &self.local) {
            let aos: Vec<_> = grid.aos.iter().map(|ao| &ao.phi).collect();
            let potential = match kernel {
                LocalKernel::Lda(lda) => {
                    let local = lda_collocate(grid.weights.len(), &aos, &pauli);
                    let values: Vec<_> = lda
                        .iter()
                        .zip(local)
                        .map(|(point, delta)| point.apply(delta))
                        .collect();
                    lda_potential(charge[0].rows(), &grid.weights, &aos, &values)?
                }
                LocalKernel::Projected(projected) => {
                    let momentum: Vec<_> = backend.xc_momentum.iter().collect();
                    projected.apply(&grid.weights, &aos, &momentum, &pauli)?
                }
            };
            for (value, v) in result.iter_mut().zip(potential) {
                value.add_scaled(&v, 1.0);
            }
        }
        Ok(result)
    }
}

struct LdaPointKernel {
    axis: [f64; 3],
    charge: f64,
    mixed: f64,
    parallel: f64,
    transverse: f64,
}

impl LdaPointKernel {
    fn apply(&self, d: [f64; 4]) -> [f64; 4] {
        let spin = (0..3).map(|a| self.axis[a] * d[a + 1]).sum::<f64>();
        let along_axis = self.mixed * d[0] + (self.parallel - self.transverse) * spin;
        [
            self.charge * d[0] + self.mixed * spin,
            self.transverse * d[1] + self.axis[0] * along_axis,
            self.transverse * d[2] + self.axis[1] * along_axis,
            self.transverse * d[3] + self.axis[2] * along_axis,
        ]
    }
}
