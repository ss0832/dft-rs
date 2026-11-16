// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Constrained extended-Lagrangian dynamics in a moving Löwdin AO frame.

use super::fields::{one_electron, MolecularDensityModel};
use super::{JkEngine, ScfOptions};
use crate::basis::{BasisSet, MolBasis};
use crate::dispersion::{dispersion_energy, dispersion_gradient, Dispersion};
use crate::grid::{GridOptions, MolecularGrid};
use crate::integrals::{overlap_matrix, CoulombKernel};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::periodic::phonon::{adjoint, product};
use crate::periodic::BlochMatrix;
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

pub(crate) struct Model<'a> {
    pub basis_set: &'a BasisSet,
    pub functional: &'a ResolvedFunctional,
    pub grid_options: &'a GridOptions,
    pub options: &'a ScfOptions,
    pub dispersion: &'a Dispersion,
    pub dispersion_functional: &'a str,
    pub spins: [usize; 2],
    /// Both spin occupations use one spatial frame when true.
    pub common: bool,
}

pub(crate) struct Evaluation {
    pub energy: f64,
    pub nuclear_gradient: Vec<Vec3>,
    pub orbital_gradient: Vec<Matrix>,
    pub minimum_overlap: f64,
}

pub(crate) struct Frame {
    pub positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub electronic_kinetic: f64,
    pub nuclear_kinetic: f64,
    pub potential_energy: f64,
    pub nuclear_gradient: Vec<Vec3>,
    pub orthogonality_defect: f64,
    pub tangency_defect: f64,
    pub minimum_overlap: f64,
}

fn add_scaled(out: &mut Matrix, input: &Matrix, scale: f64) {
    for (x, &y) in out.as_mut_slice().iter_mut().zip(input.as_slice()) {
        *x += scale * y;
    }
}

fn columns(m: &Matrix, count: usize) -> Matrix {
    let mut out = Matrix::zeros(m.rows, count);
    for i in 0..m.rows {
        for j in 0..count {
            out[(i, j)] = m[(i, j)];
        }
    }
    out
}

fn metric(basis: &MolBasis, threshold: f64) -> Result<(Vec<f64>, Matrix, Matrix)> {
    let (eigenvalues, eigenvectors) = symmetric_eigen(&overlap_matrix(basis))?;
    if eigenvalues.is_empty() || eigenvalues[0] <= threshold {
        return Err(DftError::Unsupported {
            feature: "Car-Parrinello moving Löwdin metric".into(),
            missing: "a full-rank AO overlap above the SCF overlap cutoff along the trajectory"
                .into(),
        });
    }
    let mut scaled = eigenvectors.clone();
    for j in 0..scaled.cols {
        let x = eigenvalues[j].sqrt().recip();
        for i in 0..scaled.rows {
            scaled[(i, j)] *= x;
        }
    }
    let transform = scaled.matmul_transpose_seq(&eigenvectors);
    Ok((eigenvalues, eigenvectors, transform))
}

pub(crate) fn initial_coordinates(
    basis: &MolBasis,
    threshold: f64,
    coefficients: [&Matrix; 2],
    spins: [usize; 2],
    common: bool,
) -> Result<Vec<Matrix>> {
    let (_, _, transform) = metric(basis, threshold)?;
    let to_orthogonal = overlap_matrix(basis).matmul_seq(&transform);
    let count = if common { 1 } else { 2 };
    Ok((0..count)
        .map(|s| to_orthogonal.matmul_seq(&columns(coefficients[s], spins[s])))
        .collect())
}

pub(crate) fn initial_spinor_coordinates(
    basis: &MolBasis,
    threshold: f64,
    coefficients: &BlochMatrix,
    occupied: usize,
) -> Result<BlochMatrix> {
    let (_, _, transform) = metric(basis, threshold)?;
    let to_orthogonal = overlap_matrix(basis).matmul_seq(&transform);
    Ok(BlochMatrix {
        re: spatial_apply(&to_orthogonal, &columns(&coefficients.re, occupied)),
        im: spatial_apply(&to_orthogonal, &columns(&coefficients.im, occupied)),
    })
}

fn spatial_apply(operator: &Matrix, spinors: &Matrix) -> Matrix {
    let n = operator.rows;
    let mut result = Matrix::zeros(2 * n, spinors.cols);
    for spin in 0..2 {
        let offset = spin * n;
        for i in 0..n {
            for j in 0..spinors.cols {
                result[(offset + i, j)] = (0..n)
                    .map(|k| operator[(i, k)] * spinors[(offset + k, j)])
                    .sum();
            }
        }
    }
    result
}

fn metric_covector(eigenvalues: &[f64], eigenvectors: &Matrix, covector: &Matrix) -> Matrix {
    // Minus the divided difference of S^(-1/2), for the convention -W:dS.
    let mut w = eigenvectors
        .transpose_matmul(covector)
        .matmul_seq(eigenvectors);
    let roots: Vec<_> = eigenvalues.iter().map(|s| s.sqrt()).collect();
    for i in 0..w.rows {
        for j in 0..w.cols {
            w[(i, j)] /= roots[i] * roots[j] * (roots[i] + roots[j]);
        }
    }
    eigenvectors
        .matmul_seq(&w)
        .matmul_transpose_seq(eigenvectors)
}

impl Model<'_> {
    #[cfg(test)]
    fn spinor_energy(
        &self,
        molecule: &Molecule,
        orbitals: &BlochMatrix,
        tolerance: f64,
    ) -> Result<f64> {
        let basis = MolBasis::build(molecule, self.basis_set)?;
        let (_, _, transform) = metric(&basis, self.options.orth_threshold)?;
        let coefficients = BlochMatrix {
            re: spatial_apply(&transform, &orbitals.re),
            im: spatial_apply(&transform, &orbitals.im),
        };
        let density = product(&coefficients, &adjoint(&coefficients));
        let grid = (!self.functional.parts.is_empty())
            .then(|| MolecularGrid::build(molecule, self.grid_options))
            .transpose()?;
        Ok(crate::response::nuclear::spinor_off_shell_energy(
            molecule,
            &basis,
            grid.as_ref(),
            self.functional,
            self.options,
            &density,
            tolerance,
        )? + dispersion_energy(molecule, self.dispersion, self.dispersion_functional)?)
    }

    fn evaluate_spinor(
        &self,
        molecule: &Molecule,
        orbitals: &BlochMatrix,
        tolerance: f64,
    ) -> Result<(f64, Vec<Vec3>, BlochMatrix, f64)> {
        let basis = MolBasis::build(molecule, self.basis_set)?;
        let (eigenvalues, eigenvectors, transform) = metric(&basis, self.options.orth_threshold)?;
        let coefficients = BlochMatrix {
            re: spatial_apply(&transform, &orbitals.re),
            im: spatial_apply(&transform, &orbitals.im),
        };
        let density = product(&coefficients, &adjoint(&coefficients));
        let grid = (!self.functional.parts.is_empty())
            .then(|| MolecularGrid::build(molecule, self.grid_options))
            .transpose()?;
        let (mut energy, fock, mut gradient) = crate::response::nuclear::spinor_off_shell_geometry(
            molecule,
            &basis,
            grid.as_ref(),
            self.functional,
            self.options,
            &density,
            tolerance,
        )?;
        energy += dispersion_energy(molecule, self.dispersion, self.dispersion_functional)?;
        let fc = product(&fock, &coefficients);
        let mut orbital_gradient = BlochMatrix {
            re: spatial_apply(&transform, &fc.re),
            im: spatial_apply(&transform, &fc.im),
        };
        orbital_gradient.scale(2.0);
        // Trace the metric covector over both spin blocks of (F C) Q^dagger.
        let contraction = product(&fc, &adjoint(orbitals));
        let n = basis.nao_sph;
        let mut covector = Matrix::zeros(n, n);
        for spin in 0..2 {
            let offset = spin * n;
            for i in 0..n {
                for j in 0..n {
                    covector[(i, j)] += contraction.re[(offset + i, offset + j)]
                        + contraction.re[(offset + j, offset + i)];
                }
            }
        }
        let w = metric_covector(&eigenvalues, &eigenvectors, &covector);
        let metric_gradient =
            crate::integrals::deriv::overlap_gradient(&basis, &w, molecule.len(), -1.0);
        let dispersion =
            dispersion_gradient(molecule, self.dispersion, self.dispersion_functional)?;
        for ((g, metric), d) in gradient.iter_mut().zip(metric_gradient).zip(dispersion) {
            *g += metric + d;
        }
        Ok((energy, gradient, orbital_gradient, eigenvalues[0]))
    }
}

impl Model<'_> {
    pub fn evaluate(&self, molecule: &Molecule, orbitals: &[Matrix]) -> Result<Evaluation> {
        let basis = MolBasis::build(molecule, self.basis_set)?;
        let (eigenvalues, eigenvectors, transform) = metric(&basis, self.options.orth_threshold)?;
        let coefficients: [Matrix; 2] = std::array::from_fn(|spin| {
            let q = &orbitals[if self.common { 0 } else { spin }];
            transform.matmul_seq(&columns(q, self.spins[spin]))
        });
        let density = coefficients.each_ref().map(|c| c.matmul_transpose_seq(c));
        let grid = (!self.functional.parts.is_empty() || self.options.skala.is_some())
            .then(|| MolecularGrid::build(molecule, self.grid_options))
            .transpose()?;
        let charges = basis.effective_charges(molecule);
        let (core, classical) = one_electron(molecule, &basis, self.options, &charges)?;
        let mut options = self.options.clone();
        options.eri_mode = options.effective_eri_mode(basis.nao_sph);
        let jk = JkEngine::from_options(molecule, &basis, &options)?;
        let jk_lr = (self.functional.hybrid.clr != 0.0)
            .then(|| {
                JkEngine::new(
                    &basis,
                    options.eri_mode,
                    CoulombKernel::Erf {
                        omega: self.functional.hybrid.omega,
                    },
                )
            })
            .transpose()?;
        let nlc_grid = options
            .vv10
            .map(|_| options.nonlocal_grid(molecule))
            .transpose()?;
        let pcm = options
            .pcm
            .as_ref()
            .map(|opts| crate::pcm::PcmSolver::build(molecule, &basis, &charges, opts))
            .transpose()?;
        let fields = MolecularDensityModel {
            mol: molecule,
            mb: &basis,
            opts: &options,
            functional: self.functional,
            grid: grid.as_ref(),
            h_core: &core,
            extension: None,
            jk: &jk,
            jk_lr: jk_lr.as_ref(),
            pcm_solver: pcm.as_ref(),
            nlc_grid: nlc_grid.as_ref(),
            restricted: false,
        }
        .evaluate(&density)?;
        let energy = fields.electronic_energy
            + classical
            + dispersion_energy(molecule, self.dispersion, self.dispersion_functional)?;
        let mut orbital_gradient: Vec<_> = orbitals
            .iter()
            .map(|q| Matrix::zeros(q.rows, q.cols))
            .collect();
        let mut covector = Matrix::zeros(basis.nao_sph, basis.nao_sph);
        for spin in 0..2 {
            let fc = fields.fock[spin].matmul_seq(&coefficients[spin]);
            let gradient = transform.matmul_seq(&fc);
            let output = &mut orbital_gradient[if self.common { 0 } else { spin }];
            for i in 0..gradient.rows {
                for j in 0..gradient.cols {
                    output[(i, j)] += 2.0 * gradient[(i, j)];
                }
            }
            let q = columns(
                &orbitals[if self.common { 0 } else { spin }],
                self.spins[spin],
            );
            let product = fc.matmul_transpose_seq(&q);
            add_scaled(&mut covector, &product, 1.0);
            add_scaled(&mut covector, &product.transpose(), 1.0);
        }
        // Energy chain through X=S^(-1/2). The divided difference below
        // also has the correct coincident-eigenvalue limit. The gradient
        // kernel contracts -W with dS, hence the minus sign is already used.
        let w = metric_covector(&eigenvalues, &eigenvectors, &covector);
        let mut nuclear_gradient = crate::gradient::assemble_gradient_density(
            molecule,
            &basis,
            self.functional,
            grid.as_ref(),
            [&density[0], &density[1]],
            &options,
            &w,
        )?;
        let dispersion =
            dispersion_gradient(molecule, self.dispersion, self.dispersion_functional)?;
        for (g, d) in nuclear_gradient.iter_mut().zip(dispersion) {
            *g += d;
        }
        if !energy.is_finite()
            || nuclear_gradient.iter().any(|v| !v.norm().is_finite())
            || orbital_gradient
                .iter()
                .any(|g| g.as_slice().iter().any(|v| !v.is_finite()))
        {
            return Err(DftError::LinearAlgebra(
                "nonfinite extended-Lagrangian molecular fields".into(),
            ));
        }
        Ok(Evaluation {
            energy,
            nuclear_gradient,
            orbital_gradient,
            minimum_overlap: eigenvalues[0],
        })
    }
}

fn maximum_abs(m: &Matrix) -> f64 {
    m.as_slice().iter().map(|v| v.abs()).fold(0.0, f64::max)
}

trait OrbitalFrame: Clone {
    fn zeros_like(&self) -> Self;
    fn add(&mut self, value: &Self, scale: f64);
    fn gram(&self, other: &Self) -> Self;
    fn multiply(&self, other: &Self) -> Self;
    fn dagger(&self) -> Self;
    fn subtract_identity(&mut self);
    fn max_abs(&self) -> f64;
    fn norm2(&self) -> f64;
}

impl OrbitalFrame for Matrix {
    fn zeros_like(&self) -> Self {
        Self::zeros(self.rows, self.cols)
    }
    fn add(&mut self, value: &Self, scale: f64) {
        add_scaled(self, value, scale);
    }
    fn gram(&self, other: &Self) -> Self {
        self.transpose_matmul(other)
    }
    fn multiply(&self, other: &Self) -> Self {
        self.matmul_seq(other)
    }
    fn dagger(&self) -> Self {
        self.transpose()
    }
    fn subtract_identity(&mut self) {
        for i in 0..self.rows {
            self[(i, i)] -= 1.0;
        }
    }
    fn max_abs(&self) -> f64 {
        maximum_abs(self)
    }
    fn norm2(&self) -> f64 {
        self.frobenius_dot(self)
    }
}

impl OrbitalFrame for BlochMatrix {
    fn zeros_like(&self) -> Self {
        Self::zeros(self.rows(), self.cols())
    }
    fn add(&mut self, value: &Self, scale: f64) {
        self.add_scaled(value, scale);
    }
    fn gram(&self, other: &Self) -> Self {
        product(&adjoint(self), other)
    }
    fn multiply(&self, other: &Self) -> Self {
        product(self, other)
    }
    fn dagger(&self) -> Self {
        adjoint(self)
    }
    fn subtract_identity(&mut self) {
        self.re.subtract_identity();
    }
    fn max_abs(&self) -> f64 {
        self.re
            .as_slice()
            .iter()
            .zip(self.im.as_slice())
            .map(|(r, i)| r.hypot(*i))
            .fold(0.0, f64::max)
    }
    fn norm2(&self) -> f64 {
        self.re.norm2() + self.im.norm2()
    }
}

fn position_constraint<Q: OrbitalFrame>(q: &Q, v: &mut Q, dt: f64, tolerance: f64) -> Result<Q> {
    let mut trial = q.clone();
    trial.add(v, dt);
    let mut multiplier = q.gram(q).zeros_like();
    for _ in 0..40 {
        let adjustment = q.multiply(&multiplier);
        let mut next = trial.clone();
        next.add(&adjustment, 1.0);
        let mut residual = next.gram(&next);
        residual.subtract_identity();
        if residual.max_abs() <= tolerance {
            v.add(&adjustment, dt.recip());
            return Ok(next);
        }
        // Linearized position constraints are 2I at the previous frame.
        // The nearby root is selected continuously from zero multiplier.
        multiplier.add(&residual, -0.5);
    }
    Err(DftError::LinearAlgebra(
        "Car-Parrinello RATTLE position constraints failed; use a smaller timestep".into(),
    ))
}

fn tangent_defect<Q: OrbitalFrame>(q: &Q, v: &Q) -> Q {
    let mut normal = q.gram(v);
    normal.add(&normal.dagger(), 1.0);
    normal
}

pub(crate) fn propagate(
    model: &Model<'_>,
    molecule: Molecule,
    orbitals: Vec<Matrix>,
    velocities: Vec<Vec3>,
    masses: &[f64],
    fictitious_mass: f64,
    timestep: f64,
    steps: usize,
    tolerance: f64,
) -> Result<Vec<Frame>> {
    propagate_frames(
        molecule,
        orbitals,
        velocities,
        masses,
        fictitious_mass,
        timestep,
        steps,
        tolerance,
        |m, q| {
            let e = model.evaluate(m, q)?;
            Ok((
                e.energy,
                e.nuclear_gradient,
                e.orbital_gradient,
                e.minimum_overlap,
            ))
        },
    )
}

pub(crate) fn propagate_spinor(
    model: &Model<'_>,
    molecule: Molecule,
    orbitals: BlochMatrix,
    velocities: Vec<Vec3>,
    masses: &[f64],
    fictitious_mass: f64,
    timestep: f64,
    steps: usize,
    tolerance: f64,
    xc_tolerance: f64,
) -> Result<Vec<Frame>> {
    propagate_frames(
        molecule,
        vec![orbitals],
        velocities,
        masses,
        fictitious_mass,
        timestep,
        steps,
        tolerance,
        |m, q| {
            let (energy, gradient, orbital_gradient, overlap) =
                model.evaluate_spinor(m, &q[0], xc_tolerance)?;
            Ok((energy, gradient, vec![orbital_gradient], overlap))
        },
    )
}

type FrameEvaluation<Q> = (f64, Vec<Vec3>, Vec<Q>, f64);

fn propagate_frames<Q: OrbitalFrame>(
    mut molecule: Molecule,
    mut orbitals: Vec<Q>,
    mut velocities: Vec<Vec3>,
    masses: &[f64],
    fictitious_mass: f64,
    timestep: f64,
    steps: usize,
    tolerance: f64,
    evaluate: impl Fn(&Molecule, &[Q]) -> Result<FrameEvaluation<Q>>,
) -> Result<Vec<Frame>> {
    let mut orbital_velocities: Vec<_> = orbitals.iter().map(OrbitalFrame::zeros_like).collect();
    let (mut energy, mut nuclear_gradient, mut orbital_gradient, mut minimum_overlap) =
        evaluate(&molecule, &orbitals)?;
    let mut frames = Vec::with_capacity(steps + 1);
    for step in 0..=steps {
        let mut orthogonality = 0.0_f64;
        let mut tangency = 0.0_f64;
        for (q, v) in orbitals.iter().zip(&orbital_velocities) {
            let mut gram = q.gram(q);
            gram.subtract_identity();
            orthogonality = orthogonality.max(gram.max_abs());
            tangency = tangency.max(tangent_defect(q, v).max_abs());
        }
        if orthogonality > tolerance || tangency > tolerance {
            return Err(DftError::LinearAlgebra(format!("Car-Parrinello constraint defect at step {step}: metric {orthogonality:.6e}, tangent {tangency:.6e}")));
        }
        let electronic = orbital_velocities
            .iter()
            .map(|v| 0.5 * fictitious_mass * v.norm2())
            .sum::<f64>();
        let nuclear = velocities
            .iter()
            .zip(masses)
            .map(|(v, m)| 0.5 * m * v.norm2())
            .sum::<f64>();
        crate::logger::log(&format!(
            "[car-parrinello] step {step} E_ext={:.12} T_e={electronic:.6e} |Q^TQ-I|={orthogonality:.3e} |tangent|={tangency:.3e}",
            energy+nuclear+electronic,
        ));
        frames.push(Frame {
            positions: molecule.atoms.iter().map(|a| a.position).collect(),
            velocities: velocities.clone(),
            electronic_kinetic: electronic,
            nuclear_kinetic: nuclear,
            potential_energy: energy,
            nuclear_gradient: nuclear_gradient.clone(),
            orthogonality_defect: orthogonality,
            tangency_defect: tangency,
            minimum_overlap,
        });
        if step == steps {
            break;
        }
        let half = 0.5 * timestep;
        for (((atom, v), &mass), gradient) in molecule
            .atoms
            .iter_mut()
            .zip(&mut velocities)
            .zip(masses)
            .zip(&nuclear_gradient)
        {
            *v -= *gradient * (half / mass);
            atom.position += *v * timestep;
        }
        for ((q, v), gradient) in orbitals
            .iter_mut()
            .zip(&mut orbital_velocities)
            .zip(&orbital_gradient)
        {
            v.add(gradient, -half / fictitious_mass);
            *q = position_constraint(q, v, timestep, tolerance)?;
        }
        (energy, nuclear_gradient, orbital_gradient, minimum_overlap) =
            evaluate(&molecule, &orbitals)?;
        for ((v, &mass), gradient) in velocities.iter_mut().zip(masses).zip(&nuclear_gradient) {
            *v -= *gradient * (half / mass);
        }
        for ((q, v), gradient) in orbitals
            .iter()
            .zip(&mut orbital_velocities)
            .zip(&orbital_gradient)
        {
            v.add(gradient, -half / fictitious_mass);
            v.add(&q.multiply(&tangent_defect(q, v)), -0.5);
        }
    }
    Ok(frames)
}

#[cfg(test)]
#[path = "car_parrinello_tests.rs"]
mod tests;
