// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed-geometry, real orbital Perdew-Zunger self-interaction correction.

#[path = "flo_sic.rs"]
pub mod flo_sic;

use super::fields::MolecularDensityModel;
use super::{run_scf, JkEngine, Reference, ScfOptions};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::integrals::CoulombKernel;
use crate::linalg::Matrix;
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

/// Self interaction removed from one occupied orbital, in Hartree.
pub struct SelfEnergy {
    pub hartree: f64,
    pub exchange: f64,
    pub long_range_exchange: f64,
    pub semilocal: f64,
    pub nonlocal: f64,
}

impl SelfEnergy {
    pub fn total(&self) -> f64 {
        self.hartree + self.exchange + self.long_range_exchange + self.semilocal + self.nonlocal
    }
}

/// Stationary corrected energy and physical orbital-gradient diagnostics.
pub struct SicResult {
    pub energy: f64,
    pub parent_energy: f64,
    pub initial_parent_energy: f64,
    pub terms: [Vec<SelfEnergy>; 2],
    pub orbital_gradient: f64,
    pub iterations: usize,
}

struct Point {
    energy: f64,
    parent_energy: f64,
    terms: [Vec<SelfEnergy>; 2],
    gradient: [Matrix; 2],
    inverse_curvature: [Matrix; 2],
}

impl Point {
    fn max_gradient(&self) -> f64 {
        self.gradient
            .iter()
            .flat_map(Matrix::as_slice)
            .map(|x| x.abs())
            .fold(0.0, f64::max)
    }
}

impl MolecularDensityModel<'_> {
    fn single_orbital(&self, density: &Matrix, zero: &Matrix) -> Result<(SelfEnergy, Matrix)> {
        let functional = self.functional;
        let cx = functional.hybrid.cx;
        let clr = functional.hybrid.clr;
        let (j, k) = self
            .jk
            .build(self.mb, density, [density, zero], cx != 0.0)?;
        let mut terms = SelfEnergy {
            hartree: 0.5 * density.frobenius_dot(&j),
            exchange: -0.5 * cx * density.frobenius_dot(&k[0]),
            long_range_exchange: 0.0,
            semilocal: 0.0,
            nonlocal: 0.0,
        };
        let mut potential = j;
        add_scaled(&mut potential, &k[0], -cx);
        if let Some(engine) = self.jk_lr {
            let lr = engine.build_k_only(self.mb, [density, zero])?;
            terms.long_range_exchange = -0.5 * clr * density.frobenius_dot(&lr[0]);
            add_scaled(&mut potential, &lr[0], -clr);
        }
        self.orbital_xc(density, zero, &mut terms, &mut potential)?;
        Ok((terms, potential))
    }

    fn single_complex_orbital(
        &self,
        density: &Matrix,
        zero: &Matrix,
    ) -> Result<(SelfEnergy, Matrix)> {
        // density is Re(c c^H) in a real AO basis. For one occupied spin
        // orbital each self-exchange ERI equals the corresponding
        // self-Coulomb ERI, including for a complex orbital.
        let (mut potential, _) = self.jk.build(self.mb, density, [zero, zero], false)?;
        let hartree = 0.5 * density.frobenius_dot(&potential);
        let cx = self.functional.hybrid.cx;
        let clr = self.functional.hybrid.clr;
        let mut terms = SelfEnergy {
            hartree,
            exchange: -cx * hartree,
            long_range_exchange: 0.0,
            semilocal: 0.0,
            nonlocal: 0.0,
        };
        for x in potential.as_mut_slice() {
            *x *= 1.0 - cx;
        }
        if let Some(engine) = self.jk_lr {
            let (lr, _) = engine.build(self.mb, density, [zero, zero], false)?;
            terms.long_range_exchange = -0.5 * clr * density.frobenius_dot(&lr);
            add_scaled(&mut potential, &lr, -clr);
        }
        self.orbital_xc(density, zero, &mut terms, &mut potential)?;
        Ok((terms, potential))
    }

    fn orbital_xc(
        &self,
        density: &Matrix,
        zero: &Matrix,
        terms: &mut SelfEnergy,
        potential: &mut Matrix,
    ) -> Result<()> {
        if !self.functional.parts.is_empty() {
            let grid = self
                .grid
                .ok_or_else(|| DftError::InvalidInput("SIC needs an XC grid".into()))?;
            let xc = crate::ks::xc_contribution(self.functional, self.mb, grid, [density, zero])?;
            terms.semilocal = xc.exc;
            add_scaled(potential, &xc.vxc[0], 1.0);
        }
        if let (Some(params), Some(grid)) = (self.opts.vv10, self.nlc_grid) {
            let (energy, matrix) = crate::nlc::vv10_potential(self.mb, grid, density, params)?;
            terms.nonlocal = energy;
            add_scaled(potential, &matrix, 1.0);
        }
        Ok(())
    }
}

fn add_scaled(target: &mut Matrix, matrix: &Matrix, scale: f64) {
    for (t, v) in target.as_mut_slice().iter_mut().zip(matrix.as_slice()) {
        *t += scale * v;
    }
}

fn evaluate(
    model: &MolecularDensityModel<'_>,
    orbitals: &[Matrix; 2],
    counts: [usize; 2],
    nuclear: f64,
) -> Result<Point> {
    let density = std::array::from_fn(|s| orbitals[s].leading_columns_gram(counts[s], 1.0));
    let fields = model.evaluate(&density)?;
    let parent_energy = fields.electronic_energy + nuclear;
    let mut energy = parent_energy;
    let mut terms = [Vec::with_capacity(counts[0]), Vec::with_capacity(counts[1])];
    let zero = Matrix::zeros(model.mb.nao_sph, model.mb.nao_sph);
    let mut gradients = orbitals.each_ref().map(|c| Matrix::zeros(c.cols, c.cols));
    let mut inverse_curvature = gradients.clone();
    for s in 0..2 {
        let c = &orbitals[s];
        let mo_fock = c.transpose_matmul_seq(&fields.fock[s].matmul_seq(c));
        let mut image = Matrix::zeros(c.rows, counts[s]);
        for i in 0..counts[s] {
            let orbital =
                Matrix::from_row_major(c.rows, 1, (0..c.rows).map(|a| c[(a, i)]).collect());
            let (self_energy, potential) =
                model.single_orbital(&orbital.leading_columns_gram(1, 1.0), &zero)?;
            energy -= self_energy.total();
            terms[s].push(self_energy);
            let mut differentiated = fields.fock[s].clone();
            add_scaled(&mut differentiated, &potential, -1.0);
            let column = differentiated.matmul_seq(&orbital);
            for a in 0..c.rows {
                image[(a, i)] = column[(a, 0)];
            }
        }
        // The occupied-occupied commutator retains the orbital localization
        // derivative. Both orbitals change even though the spin density does not.
        let rotated_fock = c.transpose_matmul_seq(&image);
        for i in 0..counts[s] {
            for j in i + 1..c.cols {
                let qi = rotated_fock[(j, i)];
                let qj = if j < counts[s] {
                    rotated_fock[(i, j)]
                } else {
                    0.0
                };
                gradients[s][(j, i)] = 2.0 * (qi - qj);
                // Parent orbital gaps approximate occupied-virtual curvature.
                // Occupied rotations leave the parent density invariant and
                // have only the smaller orbital-dependent SIC curvature.
                let curvature = if j < counts[s] {
                    0.2
                } else {
                    (2.0 * (mo_fock[(j, j)] - mo_fock[(i, i)]).abs()).max(0.2)
                };
                inverse_curvature[s][(j, i)] = 1.0 / curvature;
            }
        }
    }
    if !energy.is_finite()
        || gradients
            .iter()
            .flat_map(Matrix::as_slice)
            .any(|x| !x.is_finite())
    {
        return Err(DftError::InvalidInput(
            "non-finite PZ-SIC energy or orbital gradient".into(),
        ));
    }
    Ok(Point {
        energy,
        parent_energy,
        terms,
        gradient: gradients,
        inverse_curvature,
    })
}

fn rotate(orbitals: &[Matrix; 2], direction: &[Matrix; 2], scale: f64) -> [Matrix; 2] {
    std::array::from_fn(|s| {
        let c = &orbitals[s];
        let mut skew = Matrix::zeros(c.cols, c.cols);
        for i in 0..c.cols {
            for j in 0..i {
                skew[(i, j)] = scale * direction[s][(i, j)];
                skew[(j, i)] = -skew[(i, j)];
            }
        }
        c.matmul_seq(&super::soscf::orthogonal_exponential(&skew))
    })
}

/// Optimize occupied and occupied-virtual real orbital rotations, starting
/// from the integer UKS determinant. The solution is a local stationary state.
pub fn run_pz(
    mol: &Molecule,
    basis: &MolBasis,
    options: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
) -> Result<SicResult> {
    if options.occupation.is_some()
        || options.fixed_occupations.is_some()
        || options.electronic_temperature != 0.0
        || options.electron_offset != [0.0; 2]
        || options.asymptotic_correction.is_some()
        || options.skala.is_some()
        || options.pcm.is_some()
        || options.density_fit.is_some()
    {
        return Err(DftError::Unsupported {
            feature: "PZ-SIC with the requested density model or occupations".into(),
            missing: "a variational gas-phase integer determinant with exact Coulomb and exchange contractions".into(),
        });
    }
    let mut options = options.clone();
    options.reference = Reference::Uhf;
    options.eri_mode = options.effective_eri_mode(basis.nao_sph);
    let seed = run_scf(mol, basis, &options, functional, grid)?;
    let jk = JkEngine::from_options(mol, basis, &options)?;
    let jk_lr = if functional.hybrid.clr != 0.0 {
        Some(JkEngine::new(
            basis,
            options.eri_mode,
            CoulombKernel::Erf {
                omega: functional.hybrid.omega,
            },
        )?)
    } else {
        None
    };
    let nlc_grid = options
        .vv10
        .map(|_| options.nonlocal_grid(mol))
        .transpose()?;
    let model = MolecularDensityModel {
        mol,
        mb: basis,
        opts: &options,
        functional,
        grid,
        h_core: &seed.h_core,
        extension: None,
        jk: &jk,
        jk_lr: jk_lr.as_ref(),
        pcm_solver: None,
        nlc_grid: nlc_grid.as_ref(),
        restricted: false,
    };
    let mut orbitals = seed.mo_coeff;
    let counts = [seed.n_alpha, seed.n_beta];
    let mut point = evaluate(&model, &orbitals, counts, seed.e_nuc)?;
    let mut scale = 0.25;
    for iteration in 0..=options.max_iter {
        let residual = point.max_gradient();
        crate::logger::log(&format!(
            "[sic] iter {iteration:>3} E={:.12} residual={residual:.3e}",
            point.energy
        ));
        if residual <= options.conv_tol_grad {
            return Ok(SicResult {
                energy: point.energy,
                parent_energy: point.parent_energy,
                initial_parent_energy: seed.energy,
                terms: point.terms,
                orbital_gradient: residual,
                iterations: iteration,
            });
        }
        if iteration == options.max_iter {
            break;
        }
        let direction = std::array::from_fn(|s| {
            let mut step = point.gradient[s].clone();
            for (value, inverse) in step
                .as_mut_slice()
                .iter_mut()
                .zip(point.inverse_curvature[s].as_slice())
            {
                *value *= -*inverse;
            }
            step
        });
        let slope: f64 = point
            .gradient
            .iter()
            .zip(&direction)
            .map(|(g, d)| g.frobenius_dot(d))
            .sum();
        let max_angle = direction
            .iter()
            .flat_map(Matrix::as_slice)
            .map(|x| x.abs())
            .fold(0.0, f64::max);
        scale = (scale * 1.5_f64).min(0.2 / max_angle);
        let mut accepted = None;
        for _ in 0..24 {
            let next = rotate(&orbitals, &direction, scale);
            let trial = evaluate(&model, &next, counts, seed.e_nuc)?;
            if trial.energy <= point.energy + 1e-4 * scale * slope
                || trial.energy <= point.energy + 8.0 * f64::EPSILON * point.energy.abs()
                    && trial.max_gradient() < 0.95 * residual
            {
                accepted = Some((next, trial));
                break;
            }
            scale *= 0.5;
        }
        let Some((next, trial)) = accepted else {
            return Err(DftError::ScfNotConverged {
                iterations: iteration,
                error: residual,
            });
        };
        orbitals = next;
        point = trial;
    }
    Err(DftError::ScfNotConverged {
        iterations: options.max_iter,
        error: point.max_gradient(),
    })
}

#[cfg(test)]
#[path = "sic_tests.rs"]
mod tests;
