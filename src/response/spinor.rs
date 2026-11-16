// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Two-component molecular reference with a one-electron spin-orbit operator
//! in its Fock matrix, and the frequency-dependent dipole response of its
//! spinors.
//!
//! The spinor density is a Hermitian `2 nao x 2 nao` matrix in spin-major
//! order (up AOs, then down AOs); the spin axis of the collinear start is
//! the quantization axis. Coulomb and exact exchange act on the whole spinor
//! density. Every density-functional term (semilocal, nonlocal correlation,
//! learned functional, reaction field) is the collinear functional of the two
//! spin-diagonal blocks. The same functional defines the reference and the
//! response kernel, so the response operator is the second derivative of the
//! energy that the SCF made stationary.

use super::complex::Complex;
use super::krylov::{self, LinearOperator, OperatorStructure, ResponseSolverConfig};
use super::ResponseSolver;
use crate::basis::MolBasis;
use crate::dfpt::orbitals::{adjoint, matmul};
use crate::grid::MolecularGrid;
use crate::integrals::{dipole_matrices, CoulombKernel};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::spinor_scf::{
    from_pauli, pauli_components, run_spinor_scf, spin_diagonal, SpinorEnergyBackend, SpinorFields,
    SpinorScfConfig,
};
use crate::periodic::BlochMatrix;
use crate::scf::{JkEngine, ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

mod constraints;

/// Embed an already converged integer collinear reference into the full
/// complex spin space, without reoptimizing it or changing its occupations.
/// The spin-covariant provider must reproduce its Fock and stationarity.
/// This is not an adapter for common-orbital ROKS stationarity: its separate
/// orbital constraint has a different Hessian. Iteration-change fields in
/// the returned representation are zero because no new SCF step is taken.
pub fn embedded_collinear_spinor_reference(
    molecule: &Molecule,
    basis: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    options: &ScfOptions,
    tolerance: f64,
) -> Result<crate::periodic::spinor_scf::SpinorScfReport> {
    if !reference.converged
        || !reference.has_integer_occupations()
        || reference.electronic_temperature != 0.0
        || !tolerance.is_finite()
        || tolerance <= 0.0
    {
        return Err(DftError::InvalidInput("full-spin embedding requires a converged integer collinear reference and positive tolerance".into()));
    }
    let n = basis.nao_sph;
    let mut density = BlochMatrix::zeros(2 * n, 2 * n);
    let mut fock = density.clone();
    for s in 0..2 {
        for i in 0..n {
            for j in 0..n {
                density.re[(s * n + i, s * n + j)] = reference.density[s][(i, j)];
                fock.re[(s * n + i, s * n + j)] = reference.fock[s][(i, j)];
            }
        }
    }
    let occupied = [reference.n_alpha, reference.n_beta];
    let mut columns = Vec::new();
    for empty in [false, true] {
        for (s, &count) in occupied.iter().enumerate() {
            for i in 0..reference.mo_coeff[s].cols {
                if (i >= count) == empty {
                    columns.push((s, i));
                }
            }
        }
    }
    let mut coefficients = BlochMatrix::zeros(2 * n, columns.len());
    let mut energies = Vec::new();
    for (col, &(s, i)) in columns.iter().enumerate() {
        energies.push(reference.mo_energy[s][i]);
        for j in 0..n {
            coefficients.re[(s * n + j, col)] = reference.mo_coeff[s][(j, i)];
        }
    }
    let backend = MolecularSpinorBackend::new(
        molecule,
        basis,
        functional,
        grid,
        reference,
        options,
        BlochMatrix::zeros(2 * n, 2 * n),
        [0.0, 0.0, 1.0],
        MolecularSpinorFunctionalModel::SpinCovariant,
        tolerance,
    )?;
    let fields = backend.evaluate(std::slice::from_ref(&density))?;
    let fock_defect = fields.fock[0]
        .re
        .max_abs_diff(&fock.re)
        .max(fields.fock[0].im.max_abs_diff(&fock.im));
    let fc = matmul(&fields.fock[0], &coefficients)?;
    let sc = matmul(&backend.overlap()[0], &coefficients)?;
    let mut orbital_residual = 0.0_f64;
    for i in 0..2 * n {
        for (j, &energy) in energies.iter().enumerate() {
            orbital_residual = orbital_residual.max(
                (fc.re[(i, j)] - sc.re[(i, j)] * energy)
                    .hypot(fc.im[(i, j)] - sc.im[(i, j)] * energy),
            );
        }
    }
    if fock_defect > tolerance || orbital_residual > tolerance {
        return Err(DftError::ResponseNotConverged {
            iterations: 0,
            residual: fock_defect.max(orbital_residual),
        });
    }
    let mut metric = matmul(
        &matmul(&adjoint(&coefficients)?, &backend.overlap()[0])?,
        &coefficients,
    )?;
    for i in 0..metric.rows() {
        metric.re[(i, i)] -= 1.0;
    }
    let metric_defect = metric
        .re
        .as_slice()
        .iter()
        .chain(metric.im.as_slice())
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    let particles = occupied.iter().sum::<usize>();
    let particle_number_error =
        (density.trace_product(&backend.overlap()[0]) - particles as f64).abs();
    if metric_defect > tolerance || particle_number_error > tolerance {
        return Err(DftError::ResponseNotConverged {
            iterations: 0,
            residual: metric_defect.max(particle_number_error),
        });
    }
    let energy = fields.energy;
    Ok(crate::periodic::spinor_scf::SpinorScfReport {
        density: vec![density],
        fields,
        bands: vec![crate::dfpt::orbitals::BandOrbitals {
            coefficients,
            energies,
            eigen_residual: orbital_residual,
            metric_defect,
        }],
        occupations: vec![(0..columns.len())
            .map(|i| if i < particles { 1.0 } else { 0.0 })
            .collect()],
        chemical_potential: None,
        entropy_term: 0.0,
        free_energy: energy,
        iterations: reference.iterations,
        converged: true,
        density_change: 0.0,
        orbital_residual,
        occupation_consistency: 0.0,
        particle_number_error,
        metric_defect,
        energy_change: 0.0,
    })
}

fn add(target: &mut Matrix, source: &Matrix, weight: f64) {
    for (t, s) in target.as_mut_slice().iter_mut().zip(source.as_slice()) {
        *t += weight * s;
    }
}

fn symmetrized(a: &Matrix) -> Matrix {
    let mut out = a.transpose();
    for (t, s) in out.as_mut_slice().iter_mut().zip(a.as_slice()) {
        *t = 0.5 * (*t + s);
    }
    out
}

fn vanishes(a: &Matrix) -> bool {
    a.as_slice().iter().all(|v| *v == 0.0)
}

fn lifted(a: &Matrix) -> BlochMatrix {
    BlochMatrix {
        re: a.clone(),
        im: Matrix::zeros(a.rows, a.cols),
    }
}

/// Real and imaginary parts of the four spin blocks `[s][t]` of a spinor
/// matrix.
struct Blocks {
    re: [[Matrix; 2]; 2],
    im: [[Matrix; 2]; 2],
}

impl Blocks {
    fn zeros(n: usize) -> Self {
        let empty = || -> [[Matrix; 2]; 2] {
            std::array::from_fn(|_| std::array::from_fn(|_| Matrix::zeros(n, n)))
        };
        Self {
            re: empty(),
            im: empty(),
        }
    }

    fn of(p: &BlochMatrix) -> Self {
        let n = p.rows() / 2;
        let part = |source: &Matrix| -> [[Matrix; 2]; 2] {
            std::array::from_fn(|s| {
                std::array::from_fn(|t| {
                    let mut out = Matrix::zeros(n, n);
                    for i in 0..n {
                        for j in 0..n {
                            out[(i, j)] = source[(s * n + i, t * n + j)];
                        }
                    }
                    out
                })
            })
        };
        Self {
            re: part(&p.re),
            im: part(&p.im),
        }
    }

    fn assemble(&self) -> BlochMatrix {
        let n = self.re[0][0].rows;
        let mut out = BlochMatrix::zeros(2 * n, 2 * n);
        for s in 0..2 {
            for t in 0..2 {
                for i in 0..n {
                    for j in 0..n {
                        out.re[(s * n + i, t * n + j)] = self.re[s][t][(i, j)];
                        out.im[(s * n + i, t * n + j)] = self.im[s][t][(i, j)];
                    }
                }
            }
        }
        out
    }
}

/// Two-electron engines and the handles of the density-functional terms.
#[derive(Clone)]
pub(crate) struct Interaction<'a> {
    mol: &'a Molecule,
    mb: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    opts: &'a ScfOptions,
    jk: JkEngine,
    jk_lr: Option<JkEngine>,
}

impl<'a> Interaction<'a> {
    /// Bind the existing SCF pair engines to one full-spin reference.
    /// Freeze direct screening once; never rescreen with response amplitudes.
    pub(crate) fn at_spinor_reference(
        mol: &'a Molecule,
        mb: &'a MolBasis,
        functional: &'a ResolvedFunctional,
        grid: Option<&'a MolecularGrid>,
        opts: &'a ScfOptions,
        density: &BlochMatrix,
    ) -> Result<Self> {
        let n = mb.nao_sph;
        if density.rows() != 2 * n
            || density.cols() != 2 * n
            || density.im.rows != 2 * n
            || density.im.cols != 2 * n
        {
            return Err(DftError::InvalidInput(
                "spinor pair kernel requires the full matching spin/AO density".into(),
            ));
        }
        let blocks = Blocks::of(density);
        Self::new(
            mol,
            mb,
            functional,
            grid,
            opts,
            Some([&blocks.re[0][0], &blocks.re[1][1]]),
        )
    }

    /// Full complex pair-potential action, including non-Hermitian transition
    /// densities. Share Coulomb/exchange weights and zero-block skipping with
    /// the SCF implementation; no nuclear Taylor coefficients are needed.
    pub(crate) fn pair_response(&self, density: &BlochMatrix) -> Result<BlochMatrix> {
        let b = Blocks::of(density);
        let (j, mut blocks) = self.coulomb_exchange(&b)?;
        for spin in 0..2 {
            add(&mut blocks.re[spin][spin], &j[0], 1.0);
            add(&mut blocks.im[spin][spin], &j[1], 1.0);
        }
        Ok(blocks.assemble())
    }
    /// `screening` freezes the integral screening of a response operator at
    /// the reference spin densities.
    fn new(
        mol: &'a Molecule,
        mb: &'a MolBasis,
        functional: &'a ResolvedFunctional,
        grid: Option<&'a MolecularGrid>,
        opts: &'a ScfOptions,
        screening: Option<[&Matrix; 2]>,
    ) -> Result<Self> {
        let frozen = |engine: JkEngine| match screening {
            Some(spin) => {
                let mut total = spin[0].clone();
                add(&mut total, spin[1], 1.0);
                engine.with_reference_screening(&total, spin)
            }
            None => engine,
        };
        // Apply the same bounded in-core selection as collinear SCF. Public
        // setup options retain Direct until resolved for an actual AO space;
        // rebuilding shell integrals at every Krylov action would bypass the
        // existing memory-aware cache policy.
        let mut pair_options = opts.clone();
        pair_options.eri_mode = opts.effective_eri_mode(mb.nao_sph);
        let jk = frozen(JkEngine::from_options(mol, mb, &pair_options)?);
        let jk_lr = if functional.hybrid.clr != 0.0 {
            Some(frozen(JkEngine::new(
                mb,
                pair_options.eri_mode,
                CoulombKernel::Erf {
                    omega: functional.hybrid.omega,
                },
            )?))
        } else {
            None
        };
        Ok(Self {
            mol,
            mb,
            functional,
            grid,
            opts,
            jk,
            jk_lr,
        })
    }

    /// Coulomb matrices of the real and imaginary parts of the total density
    /// and minus the weighted exchange matrices of every spin block. Both
    /// are linear in the spinor matrix, which need not be Hermitian.
    fn coulomb_exchange(&self, b: &Blocks) -> Result<([Matrix; 2], Blocks)> {
        let n = self.mb.nao_sph;
        let cx = self.functional.hybrid.cx;
        let clr = self.functional.hybrid.clr;
        let zero = Matrix::zeros(n, n);
        let mut coulomb = [zero.clone(), zero.clone()];
        let mut exchange = Blocks::zeros(n);
        for (part, target) in [&b.re, &b.im].into_iter().zip(coulomb.iter_mut()) {
            let mut total = part[0][0].clone();
            add(&mut total, &part[1][1], 1.0);
            let total = symmetrized(&total);
            if !vanishes(&total) {
                *target = self.jk.build(self.mb, &total, [&zero, &zero], false)?.0;
            }
        }
        for (part, target) in [(&b.re, &mut exchange.re), (&b.im, &mut exchange.im)] {
            for pair in [[(0, 0), (1, 1)], [(0, 1), (1, 0)]] {
                let sources = [&part[pair[0].0][pair[0].1], &part[pair[1].0][pair[1].1]];
                if vanishes(sources[0]) && vanishes(sources[1]) {
                    continue;
                }
                if cx != 0.0 {
                    let k = self.jk.build_k_only(self.mb, sources)?;
                    for (slot, matrix) in pair.iter().zip(&k) {
                        add(&mut target[slot.0][slot.1], matrix, -cx);
                    }
                }
                if let Some(engine) = &self.jk_lr {
                    let k = engine.build_k_only(self.mb, sources)?;
                    for (slot, matrix) in pair.iter().zip(&k) {
                        add(&mut target[slot.0][slot.1], matrix, -clr);
                    }
                }
            }
        }
        Ok((coulomb, exchange))
    }
}

/// Energy and spin potentials of the density-functional terms.
struct Terms {
    energy: f64,
    potential: [Matrix; 2],
    grid_electrons: Option<f64>,
}

/// Semilocal extension used by a molecular spinor energy provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MolecularSpinorFunctionalModel {
    /// Historical functional of the two spin-diagonal density blocks.
    SpinDiagonal,
    /// Global spin projection of the complete Pauli density, gradients,
    /// kinetic density and current, using the shared semilocal provider.
    SpinCovariant,
}

struct CovariantGrid {
    phi: BlochMatrix,
    momentum: [BlochMatrix; 3],
    tolerance: f64,
}

/// Reusable isolated-molecule energy/Fock provider for the common complex
/// spinor SCF solver. The complete Hermitian spin/AO density is retained.
/// It also accepts trial densities for independent variational checks.
pub struct MolecularSpinorBackend<'a> {
    seed: &'a ScfResult,
    interaction: Interaction<'a>,
    nlc_grid: Option<MolecularGrid>,
    pcm: Option<crate::pcm::PcmSolver>,
    h_core: &'a Matrix,
    e_nuc: f64,
    soc: BlochMatrix,
    overlap: Vec<BlochMatrix>,
    initial: Vec<BlochMatrix>,
    electrons: f64,
    covariant_grid: Option<CovariantGrid>,
    functional_model: MolecularSpinorFunctionalModel,
}

impl<'a> MolecularSpinorBackend<'a> {
    /// Build a spinor provider from a converged collinear seed. `operator`
    /// is an additional Hermitian one-electron spin/AO operator in Hartree;
    /// `axis` rotates only the seed, without constraining the final spin.
    pub fn new(
        mol: &'a Molecule,
        mb: &'a MolBasis,
        functional: &'a ResolvedFunctional,
        grid: Option<&'a MolecularGrid>,
        seed: &'a ScfResult,
        opts: &'a ScfOptions,
        operator: BlochMatrix,
        axis: [f64; 3],
        model: MolecularSpinorFunctionalModel,
        xc_tolerance: f64,
    ) -> Result<Self> {
        if functional.required_vv10().is_some() && opts.vv10.is_none() {
            return Err(DftError::InvalidInput(
                "the spinor functional requires its configured nonlocal correlation provider"
                    .into(),
            ));
        }
        if opts.occupation.is_some() || opts.fixed_occupations.is_some() {
            return Err(DftError::Unsupported {
                feature: "occupation-selected molecular spinor reference".into(),
                missing: "a complex occupied-projector selector for the full spinor space".into(),
            });
        }
        let n = mb.nao_sph;
        let norm = axis.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !seed.converged
            || seed.overlap.rows != n
            || operator.rows() != 2 * n
            || operator.cols() != 2 * n
            || operator.im.rows != 2 * n
            || operator.im.cols != 2 * n
            || operator.hermiticity_error() > 1e-10
            || operator
                .re
                .as_slice()
                .iter()
                .chain(operator.im.as_slice())
                .any(|v| !v.is_finite())
            || !norm.is_finite()
            || norm == 0.0
            || !xc_tolerance.is_finite()
            || xc_tolerance <= 0.0
        {
            return Err(DftError::InvalidInput("molecular spinor provider requires a converged matching seed, a finite Hermitian operator, a nonzero spin axis and a positive XC tolerance".into()));
        }
        if model == MolecularSpinorFunctionalModel::SpinCovariant && opts.skala.is_some() {
            return Err(DftError::Unsupported {
                feature: "spin-covariant molecular learned functional".into(),
                missing: "a spin-covariant learned energy and variational operator provider".into(),
            });
        }
        let mut components = std::array::from_fn(|_| BlochMatrix::zeros(n, n));
        components[0] = lifted(&seed.fock[0]);
        components[0].add_scaled(&lifted(&seed.fock[1]), 1.0);
        let mut difference = lifted(&seed.fock[0]);
        difference.add_scaled(&lifted(&seed.fock[1]), -1.0);
        for a in 0..3 {
            components[a + 1] = difference.clone();
            components[a + 1].scale(axis[a] / norm);
        }
        let mut initial = from_pauli(&components)?;
        initial.add_scaled(&operator, 1.0);
        let covariant_grid = if model == MolecularSpinorFunctionalModel::SpinCovariant
            && !functional.parts.is_empty()
        {
            let grid = grid.ok_or_else(|| {
                DftError::InvalidInput("semilocal GKS needs an integration grid".into())
            })?;
            let values = crate::grid::eval::eval_aos(mb, &grid.points, true);
            let gradients = values.grad.expect("AO gradients requested");
            Some(CovariantGrid {
                phi: lifted(&values.phi),
                momentum: std::array::from_fn(|a| {
                    let mut im = gradients[a].clone();
                    for value in im.as_mut_slice() {
                        *value = -*value;
                    }
                    BlochMatrix {
                        re: Matrix::zeros(im.rows, im.cols),
                        im,
                    }
                }),
                tolerance: xc_tolerance,
            })
        } else {
            None
        };
        Ok(Self {
            seed,
            interaction: Interaction::new(mol, mb, functional, grid, opts, None)?,
            nlc_grid: match opts.vv10 {
                Some(_) => Some(opts.nonlocal_grid(mol)?),
                None => None,
            },
            pcm: match &opts.pcm {
                Some(p) => Some(crate::pcm::PcmSolver::build(
                    mol,
                    mb,
                    &mb.effective_charges(mol),
                    p,
                )?),
                None => None,
            },
            h_core: &seed.h_core,
            e_nuc: seed.e_nuc,
            soc: operator,
            overlap: vec![spin_diagonal(&lifted(&seed.overlap))],
            initial: vec![initial],
            electrons: (seed.n_alpha + seed.n_beta) as f64
                + opts.electron_offset.iter().sum::<f64>(),
            covariant_grid,
            functional_model: model,
        })
    }

    pub fn functional_model(&self) -> MolecularSpinorFunctionalModel {
        self.functional_model
    }
}

impl MolecularSpinorBackend<'_> {
    /// Collinear functional of the two spin-diagonal densities.
    fn functional_terms(&self, spin: [&Matrix; 2], total: &Matrix) -> Result<Terms> {
        let it = &self.interaction;
        let n = total.rows;
        let mut energy = 0.0;
        let mut potential = [Matrix::zeros(n, n), Matrix::zeros(n, n)];
        let mut grid_electrons = None;
        if !it.functional.parts.is_empty() && self.covariant_grid.is_none() {
            let grid = it.grid.ok_or_else(|| {
                DftError::InvalidInput("a density functional needs an integration grid".into())
            })?;
            let xc = crate::ks::xc_contribution(it.functional, it.mb, grid, spin)?;
            energy += xc.exc;
            for (target, source) in potential.iter_mut().zip(&xc.vxc) {
                add(target, source, 1.0);
            }
            grid_electrons = Some(xc.n_elec);
        }
        if let (Some(params), Some(grid)) = (&it.opts.vv10, &self.nlc_grid) {
            let (e, v) = crate::logger::timed("Molecular GKS VV10 potential", || {
                crate::nlc::vv10_potential(it.mb, grid, total, *params)
            })?;
            energy += e;
            for target in potential.iter_mut() {
                add(target, &v, 1.0);
            }
        }
        if let (Some(model), Some(grid)) = (&it.opts.skala, it.grid) {
            let (e, v) = crate::skala::skala_fock(model, it.mol, it.mb, grid, spin)?;
            energy += e;
            for (target, source) in potential.iter_mut().zip(&v) {
                add(target, source, 1.0);
            }
        }
        if let Some(solver) = &self.pcm {
            let (e, v) = solver.energy_and_fock(total)?;
            energy += e;
            for target in potential.iter_mut() {
                add(target, &v, 1.0);
            }
        }
        Ok(Terms {
            energy,
            potential,
            grid_electrons,
        })
    }
}

impl SpinorEnergyBackend for MolecularSpinorBackend<'_> {
    fn orbital_kernel<'a>(
        &'a self,
        density: &'a [BlochMatrix],
    ) -> Result<Box<dyn crate::periodic::spinor_scf::SpinorDensityKernel + 'a>> {
        let [p] = density else {
            return Err(DftError::InvalidInput(
                "molecular orbital kernel requires one density".into(),
            ));
        };
        let b = Blocks::of(p);
        let mut reference = self.seed.clone();
        reference.converged = false;
        reference.density = [symmetrized(&b.re[0][0]), symmetrized(&b.re[1][1])];
        reference.unrestricted = true;
        let mut interaction = self.interaction.clone();
        let total = reference.p_total();
        let spin = [&reference.density[0], &reference.density[1]];
        interaction.jk = interaction.jk.with_reference_screening(&total, spin);
        interaction.jk_lr = interaction
            .jk_lr
            .map(|j| j.with_reference_screening(&total, spin));
        let xc = if let Some(ao) = &self.covariant_grid {
            Some(crate::response::nuclear::SpinorScfXcKernel::new(
                interaction.functional,
                interaction.mb,
                interaction.grid.expect("spinor XC grid"),
                p,
                ao.tolerance,
            )?)
        } else {
            None
        };
        Ok(Box::new(MolecularOrbitalKernel {
            interaction,
            reference,
            xc,
        }))
    }

    fn quadratic_density_energy(&self) -> bool {
        self.interaction.functional.parts.is_empty()
            && self.interaction.opts.vv10.is_none()
            && self.interaction.opts.skala.is_none()
    }

    fn overlap(&self) -> &[BlochMatrix] {
        &self.overlap
    }
    fn initial_fock(&self) -> &[BlochMatrix] {
        &self.initial
    }
    fn electrons(&self) -> f64 {
        self.electrons
    }
    fn evaluate(&self, density: &[BlochMatrix]) -> Result<SpinorFields> {
        let n = self.h_core.rows;
        let [p] = density else {
            return Err(DftError::InvalidInput(
                "a molecular spinor density is one 2 nao x 2 nao matrix".into(),
            ));
        };
        if p.rows() != 2 * n || p.cols() != 2 * n {
            return Err(DftError::InvalidInput(
                "a molecular spinor density is one 2 nao x 2 nao matrix".into(),
            ));
        }
        let b = Blocks::of(p);
        let (coulomb, exchange) =
            crate::logger::timed("Molecular GKS Coulomb and exchange", || {
                self.interaction.coulomb_exchange(&b)
            })?;
        let spin = [symmetrized(&b.re[0][0]), symmetrized(&b.re[1][1])];
        let mut total = spin[0].clone();
        add(&mut total, &spin[1], 1.0);
        let terms = self.functional_terms([&spin[0], &spin[1]], &total)?;
        let one_electron = total.frobenius_dot(self.h_core);
        let hartree = 0.5 * total.frobenius_dot(&coulomb[0]);
        let exchange_energy = 0.5 * p.trace_product(&exchange.assemble());
        let soc_energy = p.trace_product(&self.soc);
        let mut blocks = exchange;
        for s in 0..2 {
            add(&mut blocks.re[s][s], self.h_core, 1.0);
            add(&mut blocks.re[s][s], &coulomb[0], 1.0);
            add(&mut blocks.re[s][s], &terms.potential[s], 1.0);
        }
        let mut fock = blocks.assemble();
        fock.add_scaled(&self.soc, 1.0);
        let mut semilocal_energy = 0.0;
        let mut semilocal_parts = Vec::new();
        let mut xc_diagnostics = None;
        let mut grid_number_defect = None;
        if let Some(ao) = &self.covariant_grid {
            let grid = self.interaction.grid.expect("cached semilocal grid");
            let components = pauli_components(p)?;
            // Keep angular refinement separate from exchange and nonlocal
            // correlation in timing reports; these have different scaling.
            let (energy, potential, parts, diagnostics) =
                crate::logger::timed("Molecular GKS spin-covariant semilocal XC", || {
                    crate::periodic::spinor_xc::spinor_semilocal_grid_converged(
                        self.interaction.functional,
                        &grid.weights,
                        &[&ao.phi],
                        &[&ao.momentum],
                        std::slice::from_ref(&components),
                        ao.tolerance,
                    )
                })?;
            semilocal_energy = energy;
            semilocal_parts = parts;
            fock.add_scaled(&potential[0], 1.0);
            let mut count = 0.0;
            for g in 0..grid.len() {
                for i in 0..n {
                    for j in 0..n {
                        count += grid.weights[g]
                            * components[0].re[(i, j)]
                            * ao.phi.re[(g, i)]
                            * ao.phi.re[(g, j)];
                    }
                }
            }
            grid_number_defect = Some((count - self.electrons).abs());
            xc_diagnostics = Some(diagnostics);
        }
        // Nuclear repulsion, core Hamiltonian, Hartree, exchange of both
        // ranges, all density-functional terms and the spin-orbit operator
        // in the slots of the periodic component list.
        let components = [
            self.e_nuc,
            one_electron,
            0.0,
            0.0,
            hartree,
            exchange_energy,
            0.0,
            terms.energy + semilocal_energy,
            0.0,
            soc_energy,
        ];
        let energy = components.iter().sum::<f64>();
        if !energy.is_finite() {
            return Err(DftError::InvalidInput(
                "non-finite molecular spinor energy".into(),
            ));
        }
        Ok(SpinorFields {
            fock: vec![fock],
            energy,
            energy_components: components,
            xc_component_energies: semilocal_parts,
            nuclear_soc_energy: soc_energy,
            operator_converged: xc_diagnostics.as_ref().is_none_or(|d| d.converged),
            coulomb_operator_defect: 0.0,
            xc_grid_number_defect: grid_number_defect.unwrap_or_else(|| {
                terms
                    .grid_electrons
                    .map_or(0.0, |count| (count - self.electrons).abs())
            }),
            xc_diagnostics,
        })
    }
}

/// Self-consistent two-component state with a spin-orbit operator.
#[derive(Clone, Debug)]
pub struct SpinorReference {
    /// Total energy of the spinor state, including the nuclear repulsion.
    pub energy: f64,
    /// Expectation value of the spin-orbit operator.
    pub soc_energy: f64,
    /// Ascending spinor energies.
    pub energies: Vec<f64>,
    /// Spinor coefficients as columns, `2 nao x n_spinor`.
    pub coefficients: BlochMatrix,
    pub density: BlochMatrix,
    pub fock: BlochMatrix,
    /// Number of occupied spinors.
    pub occupied: usize,
    /// Spin expectation value along the axes of the spin frame.
    pub spin: [f64; 3],
    pub iterations: usize,
    pub orbital_residual: f64,
    pub grid_electron_defect: f64,
    /// The collinear start with the spin-diagonal blocks of the spinor
    /// density as its spin densities: the point at which the collinear
    /// kernel of the response is evaluated.
    pub blocks: ScfResult,
}

/// Converge the two-component state from a collinear reference. `orbital`
/// holds the real antisymmetric matrices `O_l` of the spin-orbit operator
/// `sum_l (-i O_l) s_l` along the axes of the spin frame, whose third axis is
/// the spin axis of the collinear reference.
pub fn spin_orbit_spinor_reference(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    orbital: &[Matrix; 3],
    config: &SpinorScfConfig,
) -> Result<SpinorReference> {
    spin_orbit_spinor_reference_with_model(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        orbital,
        config,
        MolecularSpinorFunctionalModel::SpinDiagonal,
        1e-10,
    )
}

/// Reuse the same one-electron SOC operator and SCF seed with the selected
/// molecular spinor functional extension.
pub(crate) fn spin_orbit_spinor_reference_with_model(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    orbital: &[Matrix; 3],
    config: &SpinorScfConfig,
    model: MolecularSpinorFunctionalModel,
    xc_tolerance: f64,
) -> Result<SpinorReference> {
    if opts.occupation.is_some()
        || opts.fixed_occupations.is_some()
        || scf.occupation_tracking.is_some()
        || !scf.has_integer_occupations()
    {
        return Err(DftError::Unsupported {
            feature: "occupation-selected noncollinear spinor SCF".into(),
            missing: "a complex-metric occupied-projector selector for the full spinor space"
                .into(),
        });
    }
    let n = mb.nao_sph;
    if !scf.converged || scf.electronic_temperature > 0.0 {
        return Err(DftError::InvalidInput(
            "the spinor reference starts from a converged collinear reference with integer \
             occupations"
                .into(),
        ));
    }
    if scf.overlap.rows != n || orbital.iter().any(|o| o.rows != n || o.cols != n) {
        return Err(DftError::InvalidInput(
            "spin-orbit matrices and the reference must share the AO basis".into(),
        ));
    }
    let pauli: [BlochMatrix; 4] = std::array::from_fn(|m| {
        if m == 0 {
            BlochMatrix::zeros(n, n)
        } else {
            let mut im = orbital[m - 1].clone();
            for v in im.as_mut_slice() {
                *v = -*v;
            }
            BlochMatrix {
                re: Matrix::zeros(n, n),
                im,
            }
        }
    });
    let soc = from_pauli(&pauli)?;
    let backend = MolecularSpinorBackend::new(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        soc,
        [0.0, 0.0, 1.0],
        model,
        xc_tolerance,
    )?;
    let report = run_spinor_scf(&backend, config)?;
    if !report.converged {
        return Err(DftError::ScfNotConverged {
            iterations: report.iterations,
            error: report.orbital_residual,
        });
    }
    let missing = || DftError::InvalidInput("the spinor SCF returned no state".into());
    let density = report.density.into_iter().next().ok_or_else(missing)?;
    let fock = report.fields.fock.into_iter().next().ok_or_else(missing)?;
    let band = report.bands.into_iter().next().ok_or_else(missing)?;
    let components = pauli_components(&density)?;
    let metric = lifted(&scf.overlap);
    let mut spin = [0.0; 3];
    for (a, value) in spin.iter_mut().enumerate() {
        *value = 0.5 * components[a + 1].trace_product(&metric);
    }
    let b = Blocks::of(&density);
    let mut blocks = scf.clone();
    blocks.density = [symmetrized(&b.re[0][0]), symmetrized(&b.re[1][1])];
    blocks.unrestricted = true;
    Ok(SpinorReference {
        energy: report.fields.energy,
        soc_energy: report.fields.nuclear_soc_energy,
        energies: band.energies,
        coefficients: band.coefficients,
        density,
        fock,
        occupied: scf.n_alpha + scf.n_beta,
        spin,
        iterations: report.iterations,
        orbital_residual: report.orbital_residual,
        grid_electron_defect: report.fields.xc_grid_number_defect,
        blocks,
    })
}

/// Occupied-empty and empty-occupied density blocks for one orbital space.
pub(crate) struct PairedOrbitals {
    occupied: BlochMatrix,
    empty: BlochMatrix,
    gaps: Vec<f64>,
}

impl PairedOrbitals {
    pub(crate) fn new(coefficients: &BlochMatrix, energies: &[f64], occupied: usize) -> Self {
        let rows = coefficients.rows();
        let columns = |first, count| {
            let mut out = BlochMatrix::zeros(rows, count);
            for r in 0..rows {
                for c in 0..count {
                    out.re[(r, c)] = coefficients.re[(r, first + c)];
                    out.im[(r, c)] = coefficients.im[(r, first + c)];
                }
            }
            out
        };
        let gaps = energies[occupied..]
            .iter()
            .flat_map(|empty| {
                energies[..occupied]
                    .iter()
                    .map(move |filled| empty - filled)
            })
            .collect();
        Self {
            occupied: columns(0, occupied),
            empty: columns(occupied, energies.len() - occupied),
            gaps,
        }
    }

    pub(crate) fn dimension(&self) -> usize {
        2 * self.gaps.len()
    }

    /// Add the independent-particle action with gaps minus z in the
    /// empty-occupied block and gaps plus z in its backward partner.
    pub(crate) fn add_free_action(&self, out: &mut [Complex], x: &[Complex], frequency: Complex) {
        let n = self.gaps.len();
        for (k, gap) in self.gaps.iter().enumerate() {
            out[k] += x[k] * *gap - frequency * x[k];
            out[n + k] += x[n + k] * *gap + frequency * x[n + k];
        }
    }

    fn sizes(&self) -> (usize, usize) {
        (self.occupied.cols(), self.empty.cols())
    }

    /// First-order AO density from both independent orbital blocks.
    pub(crate) fn density_response(&self, x: &[Complex]) -> Result<BlochMatrix> {
        let (no, nv) = self.sizes();
        if nv == 0 {
            return Ok(BlochMatrix::zeros(
                self.occupied.rows(),
                self.occupied.rows(),
            ));
        }
        let mut up = BlochMatrix::zeros(nv, no);
        let mut down = BlochMatrix::zeros(no, nv);
        for a in 0..nv {
            for i in 0..no {
                let (p, q) = (x[a * no + i], x[nv * no + a * no + i]);
                up.re[(a, i)] = p.re;
                up.im[(a, i)] = p.im;
                down.re[(i, a)] = q.re;
                down.im[(i, a)] = q.im;
            }
        }
        let mut density = matmul(&matmul(&self.empty, &up)?, &adjoint(&self.occupied)?)?;
        density.add_scaled(
            &matmul(&matmul(&self.occupied, &down)?, &adjoint(&self.empty)?)?,
            1.0,
        );
        Ok(density)
    }

    /// Empty-occupied and occupied-empty blocks of a spinor matrix in the
    /// order of the amplitudes.
    pub(crate) fn projected(&self, matrix: &BlochMatrix) -> Result<Vec<Complex>> {
        let (no, nv) = self.sizes();
        if nv == 0 {
            return Ok(Vec::new());
        }
        let vo = matmul(&matmul(&adjoint(&self.empty)?, matrix)?, &self.occupied)?;
        let ov = matmul(&matmul(&adjoint(&self.occupied)?, matrix)?, &self.empty)?;
        let mut out = vec![Complex::ZERO; 2 * nv * no];
        for a in 0..nv {
            for i in 0..no {
                out[a * no + i] = Complex::new(vo.re[(a, i)], vo.im[(a, i)]);
                out[nv * no + a * no + i] = Complex::new(ov.re[(i, a)], ov.im[(i, a)]);
            }
        }
        Ok(out)
    }
}

struct Response<'a> {
    kernel: &'a dyn Fn(&BlochMatrix) -> Result<BlochMatrix>,
    orbitals: PairedOrbitals,
    frequency: Complex,
}

impl LinearOperator for Response<'_> {
    fn dimension(&self) -> usize {
        self.orbitals.dimension()
    }
    fn structure(&self) -> OperatorStructure {
        if self.frequency.im == 0.0 {
            OperatorStructure::Hermitian
        } else {
            OperatorStructure::General
        }
    }
    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        if x.len() != self.dimension() {
            return Err(DftError::InvalidInput(
                "invalid spinor amplitude dimension".into(),
            ));
        }
        let density = self.orbitals.density_response(x)?;
        let mut out = self.orbitals.projected(&(self.kernel)(&density)?)?;
        self.orbitals.add_free_action(&mut out, x, self.frequency);
        Ok(out)
    }
}

/// Polarizability of the spinor state at one complex frequency.
#[derive(Clone, Debug)]
pub struct SpinorPolarizability {
    pub frequency: Complex,
    /// `alpha[observable axis][field axis]`.
    pub alpha: [[Complex; 3]; 3],
    pub residuals: [f64; 3],
    pub iterations: [usize; 3],
}

/// General one-body susceptibility in the full complex spin/AO space.
/// The matrix element is d Tr[O_a P] / d source_b, without a dipole sign.
/// Source and observable operators are the same supplied list. Density
/// responses retain every spin block for other mixed properties.
pub struct SpinorOperatorResponse {
    pub frequency: Complex,
    pub susceptibility: Vec<Vec<Complex>>,
    pub density_responses: Vec<BlochMatrix>,
    pub residuals: Vec<f64>,
    pub iterations: Vec<usize>,
    /// Original constraint expectation derivatives, [source][constraint].
    pub constraint_residuals: Vec<Vec<Complex>>,
    /// Lagrange-field derivatives in the supplied constraint basis.
    pub constraint_multipliers: Vec<Vec<Complex>>,
    pub constraint_rank: usize,
    /// Projection norms removed only by floating-point rank revelation.
    pub constraint_discarded_norms: Vec<f64>,
}

/// Paired response of arbitrary one-body operators, sharing the electric
/// response's matrix-free full-spin Hessian and residual checks. Operators
/// may contain real/imaginary diagonal or transverse spin blocks; no
/// longitudinal-spin or reciprocal-component replacement is performed.
pub fn spinor_operator_response_with_kernel(
    coefficients: &BlochMatrix,
    energies: &[f64],
    occupied: usize,
    kernel: &dyn Fn(&BlochMatrix) -> Result<BlochMatrix>,
    operators: &[BlochMatrix],
    frequencies: &[Complex],
    config: &ResponseSolverConfig,
) -> Result<Vec<SpinorOperatorResponse>> {
    constrained_spinor_operator_response_with_kernel(
        coefficients,
        energies,
        occupied,
        kernel,
        operators,
        &[],
        frequencies,
        config,
    )
}

/// Equality-constrained full-spin susceptibility. Each supplied Hermitian
/// constraint Q imposes Tr[Q dP]=0 through the unmodified Hessian's saddle
/// system [A V; V^dagger 0]. This is a fixed-expectation response, not a
/// free-spin thermal/Curie response. Redundant projected constraints are
/// rank-revealed with a floating-point matrix-product bound; every original
/// constraint residual is returned. No physical denominator is replaced.
pub fn constrained_spinor_operator_response_with_kernel(
    coefficients: &BlochMatrix,
    energies: &[f64],
    occupied: usize,
    kernel: &dyn Fn(&BlochMatrix) -> Result<BlochMatrix>,
    operators: &[BlochMatrix],
    constraints: &[BlochMatrix],
    frequencies: &[Complex],
    config: &ResponseSolverConfig,
) -> Result<Vec<SpinorOperatorResponse>> {
    config.validate()?;
    let spinors = energies.len();
    let rows = coefficients.rows();
    if occupied == 0
        || occupied >= spinors
        || coefficients.cols() != spinors
        || coefficients.im.rows != rows
        || coefficients.im.cols != spinors
        || coefficients
            .re
            .as_slice()
            .iter()
            .chain(coefficients.im.as_slice())
            .any(|v| !v.is_finite())
        || energies.iter().any(|v| !v.is_finite())
        || operators.iter().chain(constraints).any(|o| {
            o.rows() != rows
                || o.cols() != rows
                || o.im.rows != rows
                || o.im.cols != rows
                || o.re
                    .as_slice()
                    .iter()
                    .chain(o.im.as_slice())
                    .any(|v| !v.is_finite())
        })
    {
        return Err(DftError::InvalidInput("spinor operator response needs a finite matching occupied/empty reference and full spin/AO operators".into()));
    }
    if constraints.iter().any(|o| {
        o.hermiticity_error()
            > 32.0
                * f64::EPSILON
                * rows as f64
                * o.re
                    .as_slice()
                    .iter()
                    .chain(o.im.as_slice())
                    .fold(0.0_f64, |a, &b| a.hypot(b))
    }) {
        return Err(DftError::InvalidInput(
            "spinor expectation constraints must be Hermitian".into(),
        ));
    }
    let mut response = Response {
        kernel,
        orbitals: PairedOrbitals::new(coefficients, energies, occupied),
        frequency: Complex::ZERO,
    };
    let count = response.dimension() / 2;
    let vertices = operators
        .iter()
        .map(|operator| response.orbitals.projected(operator))
        .collect::<Result<Vec<_>>>()?;
    let space = constraints::ConstraintSpace::new(&response, constraints)?;
    let mut reports = Vec::with_capacity(frequencies.len());
    for &frequency in frequencies {
        if !frequency.is_finite() {
            return Err(DftError::InvalidInput(
                "spinor response frequencies must be finite".into(),
            ));
        }
        response.frequency = frequency;
        // Positive magnitudes of the forward/backward independent-particle
        // denominators scale the solve, never shift the physical Hessian.
        // A constraint multiplier uses the diagonal Schur-complement estimate
        // V^dagger D^-1 V; positive two-sided scaling preserves the saddle form.
        let mut diagonal = Vec::with_capacity(response.dimension() + space.basis.len());
        for sign in [-1.0, 1.0] {
            for &gap in &response.orbitals.gaps {
                let denominator = (Complex::from(gap) + frequency * sign).abs();
                if denominator < config.min_denominator {
                    return Err(DftError::Unsupported {
                        feature: format!(
                            "spinor response with a zeroth-order denominator of {denominator:.3e}"
                        ),
                        missing: "a resolved response above response_min_denominator".into(),
                    });
                }
                diagonal.push(denominator.max(config.preconditioner_floor));
            }
        }
        let orbital_dimension = diagonal.len();
        for column in &space.basis {
            let schur: f64 = column
                .iter()
                .zip(&diagonal[..orbital_dimension])
                .map(|(v, &d)| v.abs().powi(2) / d)
                .sum();
            diagonal.push(schur.max(config.preconditioner_floor));
        }
        let mut solve_config = config.clone();
        // The physical orbital denominators were checked above. Schur scales
        // have inverse-energy units and are not resonance denominators.
        solve_config.min_denominator = 0.0;
        let mut susceptibility = vec![vec![Complex::ZERO; operators.len()]; operators.len()];
        let mut density_responses = Vec::with_capacity(operators.len());
        let mut residuals = Vec::with_capacity(operators.len());
        let mut iterations = Vec::with_capacity(operators.len());
        let mut constraint_residuals = Vec::with_capacity(operators.len());
        let mut constraint_multipliers = Vec::with_capacity(operators.len());
        for (b, vertex) in vertices.iter().enumerate() {
            let mut rhs = vertex.iter().map(|v| -*v).collect::<Vec<_>>();
            rhs.resize(response.dimension() + space.basis.len(), Complex::ZERO);
            let operator = constraints::Constrained {
                response: &response,
                constraints: &space,
            };
            let solved = krylov::solve(&operator, &rhs, Some(&diagonal), &solve_config)?;
            crate::logger::log(&format!("[spinor-response] source {}/{} frequency=({:.6e},{:.6e}) method={} iterations={} residual={:.6e}",
                b + 1, vertices.len(), frequency.re, frequency.im, solved.method,
                solved.iterations, solved.relative_residual));
            for (a, observable) in vertices.iter().enumerate() {
                for k in 0..count {
                    susceptibility[a][b] += observable[count + k] * solved.amplitudes[k]
                        + observable[k] * solved.amplitudes[count + k];
                }
            }
            let (errors, multipliers) = space.diagnostics(&solved.amplitudes, response.dimension());
            constraint_residuals.push(errors);
            constraint_multipliers.push(multipliers);
            density_responses.push(
                response
                    .orbitals
                    .density_response(&solved.amplitudes[..response.dimension()])?,
            );
            residuals.push(solved.relative_residual);
            iterations.push(solved.iterations);
        }
        reports.push(SpinorOperatorResponse {
            frequency,
            susceptibility,
            density_responses,
            residuals,
            iterations,
            constraint_residuals,
            constraint_multipliers,
            constraint_rank: space.basis.len(),
            constraint_discarded_norms: space.discarded_norms.clone(),
        });
    }
    Ok(reports)
}

/// `alpha_ab(z) = -Tr[r_a D_b(z)]` for the perturbation `+E.r exp(-i z t)`:
/// the first-order density solves
/// `(e_a - e_i - z) X_ai + F1_ai = 0` and `(e_a - e_i + z) Y_ia + F1_ia = 0`
/// with `F1 = r_b + G[D]`, where `G` holds the Coulomb, exchange and
/// collinear kernel terms of the reference functional.
pub fn spinor_dynamic_polarizability<'a>(
    mol: &'a Molecule,
    mb: &'a MolBasis,
    functional: &'a ResolvedFunctional,
    grid: Option<&'a MolecularGrid>,
    opts: &'a ScfOptions,
    reference: &'a SpinorReference,
    frequencies: &[Complex],
    config: &ResponseSolverConfig,
) -> Result<Vec<SpinorPolarizability>> {
    let view = &reference.blocks;
    let interaction = Interaction::new(
        mol,
        mb,
        functional,
        grid,
        opts,
        Some([&view.density[0], &view.density[1]]),
    )?;
    let has_kernel = !functional.parts.is_empty()
        || opts.vv10.is_some()
        || opts.pcm.is_some()
        || opts.skala.is_some();
    let kernel = if has_kernel {
        Some(ResponseSolver::kernel_only(
            mol, mb, functional, grid, view, opts,
        )?)
    } else {
        None
    };
    let induced = |density: &BlochMatrix| {
        let b = Blocks::of(density);
        let (coulomb, mut out) = interaction.coulomb_exchange(&b)?;
        for s in 0..2 {
            add(&mut out.re[s][s], &coulomb[0], 1.0);
            add(&mut out.im[s][s], &coulomb[1], 1.0);
        }
        if let Some(kernel) = &kernel {
            for (part, target) in [(&b.re, &mut out.re), (&b.im, &mut out.im)] {
                let spin = [symmetrized(&part[0][0]), symmetrized(&part[1][1])];
                if vanishes(&spin[0]) && vanishes(&spin[1]) {
                    continue;
                }
                let v = kernel.fxc_contraction(&spin)?;
                for s in 0..2 {
                    add(&mut target[s][s], &v[s], 1.0);
                }
            }
        }
        Ok(out.assemble())
    };
    spinor_polarizability_with_kernel(
        mb,
        &reference.coefficients,
        &reference.energies,
        reference.occupied,
        &induced,
        frequencies,
        config,
    )
}

/// Paired complex projector response with an explicit reference-bound Hessian
/// action. Providers retain all spin/AO blocks and extend their Hermitian
/// density derivative complex-linearly to transition densities. This is the
/// shared operator for spin-diagonal and spin-covariant functional models.
pub fn spinor_polarizability_with_kernel(
    mb: &MolBasis,
    coefficients: &BlochMatrix,
    energies: &[f64],
    occupied: usize,
    kernel: &dyn Fn(&BlochMatrix) -> Result<BlochMatrix>,
    frequencies: &[Complex],
    config: &ResponseSolverConfig,
) -> Result<Vec<SpinorPolarizability>> {
    if coefficients.rows() != 2 * mb.nao_sph {
        return Err(DftError::InvalidInput(
            "electric spinor response basis and coefficients disagree".into(),
        ));
    }
    let operators = dipole_matrices(mb, Vec3::zero())
        .iter()
        .map(|d| spin_diagonal(&lifted(d)))
        .collect::<Vec<_>>();
    spinor_operator_response_with_kernel(
        coefficients,
        energies,
        occupied,
        kernel,
        &operators,
        frequencies,
        config,
    )?
    .into_iter()
    .map(|report| {
        Ok(SpinorPolarizability {
            frequency: report.frequency,
            alpha: std::array::from_fn(|a| std::array::from_fn(|b| -report.susceptibility[a][b])),
            residuals: report.residuals.try_into().map_err(|_| {
                DftError::InvalidInput("electric response requires three source residuals".into())
            })?,
            iterations: report.iterations.try_into().map_err(|_| {
                DftError::InvalidInput("electric response requires three iteration counts".into())
            })?,
        })
    })
    .collect()
}

/// Shared full-spin Coulomb/exchange plus the same collinear/noncollinear
/// XC and nonlocal kernels as the physical molecular energy provider.
struct MolecularOrbitalKernel<'a> {
    interaction: Interaction<'a>,
    reference: ScfResult,
    xc: Option<crate::response::nuclear::SpinorScfXcKernel<'a>>,
}
impl crate::periodic::spinor_scf::SpinorDensityKernel for MolecularOrbitalKernel<'_> {
    fn apply(&self, direction: &[BlochMatrix]) -> Result<Vec<BlochMatrix>> {
        let [p] = direction else {
            return Err(DftError::InvalidInput(
                "molecular kernel requires one direction".into(),
            ));
        };
        let it = &self.interaction;
        let b = Blocks::of(p);
        let mut f = it.pair_response(p)?;
        let mut linear = ResolvedFunctional::hartree_fock();
        linear.hybrid = it.functional.hybrid;
        let functional = if self.xc.is_some() {
            &linear
        } else {
            it.functional
        };
        if let Some(xc) = &self.xc {
            f.add_scaled(&xc.apply(p)?, 1.0);
        }
        if !functional.parts.is_empty()
            || it.opts.vv10.is_some()
            || it.opts.skala.is_some()
            || it.opts.pcm.is_some()
        {
            let kernel = ResponseSolver::kernel_with_engines(
                it.mol,
                it.mb,
                functional,
                it.grid,
                &self.reference,
                it.opts,
                &it.jk,
                it.jk_lr.as_ref(),
            )?;
            let dp = [symmetrized(&b.re[0][0]), symmetrized(&b.re[1][1])];
            let v = kernel.fxc_contraction(&dp)?;
            for s in 0..2 {
                for i in 0..it.mb.nao_sph {
                    for j in 0..it.mb.nao_sph {
                        f.re[(s * it.mb.nao_sph + i, s * it.mb.nao_sph + j)] += v[s][(i, j)];
                    }
                }
            }
        }
        Ok(vec![f])
    }
}
