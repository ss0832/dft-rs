// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed-geometry full-spin fields using the same stationary reference,
//! Coulomb/exchange/XC jets and physical residual gates as nuclear response.
use super::*;
use crate::properties::pt2_response::{
    complex_projector_frames, generalized_taylor_response, spinor_transform_path,
    GeneralizedPt2Taylor,
};
use crate::response::complex::Complex;
use crate::response::field_path::{adjoint, series_product, stationary_retained_complex_path};

pub struct SpinorWavefunctionPath {
    pub parent: FieldPathTaylor,
    /// Physical AO-by-spinor coefficients, including both spin components.
    pub orbitals: Vec<Cm>,
    /// Full MO Fock in precisely the same parallel occupied/virtual frame.
    pub mo_fock: Vec<Cm>,
    pub density: Vec<Cm>,
    pub occupied: usize,
    pub frame_defect: f64,
    pub particle_number_defects: Vec<f64>,
    pub angular_history: Vec<[f64; 3]>,
}

struct WithCore<'a, 'b> {
    base: &'a SpinorGeometry<'b>,
    core: &'a [Cm],
}
impl ComplexPathBackend for WithCore<'_, '_> {
    fn overlap(&self) -> &[Cm] {
        self.base.overlap()
    }
    fn fock_coefficient(&self, p: &[Vec<Cm>; 2], order: usize) -> Result<[Cm; 2]> {
        let mut f = self.base.fock_coefficient(p, order)?;
        add(&mut f[0], &self.core[order]);
        Ok(f)
    }
    fn linear_kernel(&self, p: &[Cm; 2]) -> Result<[Cm; 2]> {
        self.base.linear_kernel(p)
    }
    fn energy(&self, p: &[Vec<Cm>; 2], order: usize) -> Result<Vec<f64>> {
        let mut e = self.base.energy(p, order)?;
        for n in 0..=order {
            for k in 0..=n {
                e[n] += p[0][k].real_trace_product(&self.core[n - k]);
            }
        }
        Ok(e)
    }
    fn dipole(&self, p: &[Vec<Cm>; 2], order: usize) -> Vec<[f64; 3]> {
        self.base.dipole(p, order)
    }
}

impl RetainedComplexBackend for WithCore<'_, '_> {
    fn retained_fock_coefficient(
        &self,
        high: &[Vec<Cm>; 2],
        low: &[Vec<Cm>; 2],
        order: usize,
    ) -> Result<([Cm; 2], [Cm; 2])> {
        let (mut high, mut low) = self.base.retained_fock_coefficient(high, low, order)?;
        let h = &mut high[0];
        let l = &mut low[0];
        for (h, l, source) in [
            (&mut h.re, &mut l.re, &self.core[order].re),
            (&mut h.im, &mut l.im, &self.core[order].im),
        ] {
            for ((h, l), &v) in h
                .as_mut_slice()
                .iter_mut()
                .zip(l.as_mut_slice())
                .zip(source.as_slice())
            {
                let mut sum = crate::linalg::AccurateSum::default();
                sum.add_product(*h, 1.0);
                sum.add_product(*l, 1.0);
                sum.add_product(v, 1.0);
                (*h, *l) = sum.parts();
            }
        }
        Ok((high, low))
    }
}

/// Analytic full-spin one-body paths, including transverse spin fields and
/// mixed operators at higher order. Coefficients are factorial normalized;
/// core[0] must be zero because `operator` belongs to the reference already.
/// A fixed AO source implies no nuclear, London-phase or SOC source derivative.
pub fn spinor_wavefunction_path(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: Option<&MolecularGrid>,
    functional: &ResolvedFunctional,
    reference: &SpinorScfReport,
    options: &ScfOptions,
    operator: &BlochMatrix,
    core: &[Cm],
    config: &ResponseSolverConfig,
) -> Result<SpinorWavefunctionPath> {
    validate_core(2 * basis.nao_sph, core, config)?;
    let (mut base, state) = prepare(
        molecule,
        basis,
        grid,
        functional,
        reference,
        options,
        operator,
        core.len() - 1,
        config,
    )?;
    base.cache = descriptor_cache(&base, core.len() - 1, options.incore_mb)?;
    solve(
        &WithCore { base: &base, core },
        &base,
        &state,
        core.len() - 1,
        false,
        config,
    )
}

fn validate_core(dimension: usize, core: &[Cm], config: &ResponseSolverConfig) -> Result<()> {
    if !(2..=5).contains(&core.len())
        || core.iter().any(|h| {
            h.re.rows != dimension
                || h.re.cols != dimension
                || h.im.rows != dimension
                || h.im.cols != dimension
                || h.re
                    .as_slice()
                    .iter()
                    .chain(h.im.as_slice())
                    .any(|v| !v.is_finite())
                || (0..dimension).any(|i| {
                    (0..dimension).any(|j| {
                        (h.re[(i, j)] - h.re[(j, i)]).abs() > config.acceptance()
                            || (h.im[(i, j)] + h.im[(j, i)]).abs() > config.acceptance()
                    })
                })
        })
        || core[0]
            .re
            .as_slice()
            .iter()
            .chain(core[0].im.as_slice())
            .any(|v| *v != 0.0)
    {
        return Err(DftError::InvalidInput("spinor field paths require finite Hermitian coefficients of orders zero through 1..4 and an exactly zero added base source".into()));
    }
    Ok(())
}

fn prepare<'a>(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: Option<&MolecularGrid>,
    functional: &'a ResolvedFunctional,
    reference: &SpinorScfReport,
    options: &ScfOptions,
    operator: &BlochMatrix,
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<(SpinorGeometry<'a>, ComplexPathReference)> {
    prepare_spinor_path(
        molecule,
        basis,
        grid,
        functional,
        reference,
        options,
        operator,
        &vec![[0.0; 3]; molecule.len()],
        [0.0; 3],
        order,
        config,
    )
}

fn solve(
    backend: &impl RetainedComplexBackend,
    base: &SpinorGeometry<'_>,
    reference: &ComplexPathReference,
    order: usize,
    retained: bool,
    config: &ResponseSolverConfig,
) -> Result<SpinorWavefunctionPath> {
    let (parent, state) = if retained {
        stationary_retained_complex_path(backend, reference, order, config)?
    } else {
        stationary_complex_path(backend, reference, order, config)?
    };
    let occupied = reference.occupations[0];
    let (orbitals, mo_fock, frame_defect) = if let Some([frame, _]) = state.retained_orbital_frames
    {
        frame
    } else {
        let (frame, defect) = complex_projector_frames(&state.d[0], occupied)?;
        let orbitals = series_product(&state.b[0], &frame, order);
        let dagger = frame.iter().map(adjoint).collect::<Vec<_>>();
        let fock = series_product(&series_product(&dagger, &state.f[0], order), &frame, order);
        (orbitals, fock, defect)
    };
    let particle_number_defects = match state.retained_number_defects {
        Some([spinor, _]) => spinor,
        None => (0..=order)
            .map(|n| {
                ((0..=n)
                    .map(|k| state.p[0][k].real_trace_product(&base.overlap[n - k]))
                    .sum::<f64>()
                    - if n == 0 { occupied as f64 } else { 0.0 })
                .abs()
            })
            .collect(),
    };
    Ok(SpinorWavefunctionPath {
        parent,
        orbitals,
        mo_fock,
        density: state.p[0].clone(),
        occupied,
        frame_defect,
        particle_number_defects,
        angular_history: base.angular_history.borrow().clone(),
    })
}

pub struct SpinorCorrelatedPath {
    pub parent: SpinorWavefunctionPath,
    /// Spin-complete unscaled correlation, with no artificial SS/OS labels.
    pub pairs: GeneralizedPt2Taylor,
    /// Parent energy plus a common correlation coefficient times PT2.
    pub energy: Vec<f64>,
    pub imaginary_energy_defect: f64,
}

/// Full orbital and pair-amplitude response of a stationary GHF/GKS
/// reference to arbitrary Hermitian one-body operators. Standard generalized
/// PT2 is spin-rotation invariant; `coefficient` scales the complete energy.
/// A distinct target, unequal SS/OS scaling and orientational constraints
/// need their explicit additional Lagrangians and are not inferred here.
pub fn spinor_correlated_one_body_taylor(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: Option<&MolecularGrid>,
    functional: &ResolvedFunctional,
    reference: &SpinorScfReport,
    options: &ScfOptions,
    operator: &BlochMatrix,
    core: &[Cm],
    coefficient: f64,
    config: &ResponseSolverConfig,
) -> Result<SpinorCorrelatedPath> {
    validate_core(2 * basis.nao_sph, core, config)?;
    SpinorCorrelatedSolver::new(
        molecule,
        basis,
        grid,
        functional,
        None,
        reference,
        options,
        operator,
        core.len() - 1,
        coefficient,
        options.incore_mb,
        config,
    )?
    .solve(core)
}

/// Shared stationary reference and spatial integral series for several
/// one-electron perturbations of the same Taylor order.
pub(crate) struct SpinorCorrelatedSolver<'a> {
    base: SpinorGeometry<'a>,
    target: Option<SpinorGeometry<'a>>,
    reference: ComplexPathReference,
    integrals: Vec<Vec<Complex>>,
    nao: usize,
    coefficient: f64,
    config: ResponseSolverConfig,
}

impl<'a> SpinorCorrelatedSolver<'a> {
    pub(crate) fn new(
        molecule: &Molecule,
        basis: &MolBasis,
        grid: Option<&MolecularGrid>,
        functional: &'a ResolvedFunctional,
        target: Option<&'a ResolvedFunctional>,
        reference: &SpinorScfReport,
        options: &ScfOptions,
        operator: &BlochMatrix,
        order: usize,
        coefficient: f64,
        memory_mb: usize,
        config: &ResponseSolverConfig,
    ) -> Result<Self> {
        if !coefficient.is_finite() || !(1..=4).contains(&order) {
            return Err(DftError::InvalidInput(
                "spinor PT2 requires a finite coefficient and Taylor order 1..4".into(),
            ));
        }
        let nao = basis.nao_sph;
        check_spinor_pt2_workspace(nao, order + 1, memory_mb.min(options.incore_mb))?;
        let (mut base, stationary) = prepare(
            molecule, basis, grid, functional, reference, options, operator, order, config,
        )?;
        base.cache = descriptor_cache(&base, order, options.incore_mb)?;
        let target = target
            .map(|functional| {
                let (mut backend, _) = prepare(
                    molecule, basis, grid, functional, reference, options, operator, order, config,
                )?;
                backend.cache = descriptor_cache(&backend, order, options.incore_mb)?;
                Ok::<_, DftError>(backend)
            })
            .transpose()?;
        let integrals = base
            .geometry
            .coulomb_coefficients()
            .iter()
            .map(|g| g.iter().copied().map(Complex::from).collect())
            .collect();
        Ok(Self {
            base,
            target,
            reference: stationary,
            integrals,
            nao,
            coefficient,
            config: config.clone(),
        })
    }

    /// Evaluate the target on the converged parent density series with the
    /// same fixed one-body perturbation. The scalar correction therefore
    /// includes the relaxed parent density through the requested order.
    pub(crate) fn target_correction(
        &self,
        core: &[Cm],
        parent: &SpinorWavefunctionPath,
    ) -> Result<Option<Vec<f64>>> {
        let Some(target) = &self.target else {
            return Ok(None);
        };
        target.angular_history.borrow_mut().clear();
        let energy = WithCore { base: target, core }
            .energy(&[parent.density.clone(), Vec::new()], core.len() - 1)?;
        Ok(Some(
            energy
                .into_iter()
                .zip(&parent.parent.energy)
                .map(|(target, parent)| target - parent)
                .collect(),
        ))
    }

    pub(crate) fn solve(&self, core: &[Cm]) -> Result<SpinorCorrelatedPath> {
        if core.len() != self.integrals.len() {
            return Err(DftError::InvalidInput(
                "spinor PT2 source and integral Taylor orders differ".into(),
            ));
        }
        validate_core(2 * self.nao, core, &self.config)?;
        self.base.angular_history.borrow_mut().clear();
        let path = solve(
            &WithCore {
                base: &self.base,
                core,
            },
            &self.base,
            &self.reference,
            core.len() - 1,
            core.len() > 3,
            &self.config,
        )?;
        correlate(
            path,
            self.nao,
            &self.integrals,
            &self.reference.energies[0],
            self.coefficient,
            &self.config,
        )
    }
}

impl SpinorNuclearContext<'_> {
    /// First nuclear derivative on the integer spinor parent. Both the
    /// relaxed spinors and the moving AO integrals enter the pair source.
    pub(crate) fn correlated_gradient_path(
        &self,
        target: Option<(&Self, f64)>,
        displacement: &[[f64; 3]],
        coefficient: f64,
        memory_mb: usize,
    ) -> Result<(SpinorCorrelatedPath, Option<Vec<f64>>)> {
        if self.options.electronic_temperature != 0.0 || !coefficient.is_finite() {
            return Err(DftError::InvalidInput(
                "nuclear spinor PT2 requires integer occupations and a finite coefficient".into(),
            ));
        }
        let nao = self.basis.nao_sph;
        check_spinor_pt2_workspace(nao, 2, memory_mb.min(self.options.incore_mb))?;
        let direction = super::super::MolecularDirection::nuclear(displacement.to_vec());
        let base = self.path_geometry(&direction, 1)?;
        let reference = integer_path_reference(self.reference, nao);
        let parent = solve(&base, &base, &reference, 1, false, self.config)?;
        let integrals = base
            .geometry
            .coulomb_coefficients()
            .iter()
            .map(|g| g.iter().copied().map(Complex::from).collect())
            .collect::<Vec<_>>();
        let target_correction = target
            .map(|(target, scale)| {
                let energy = target
                    .path_geometry(&direction, 1)?
                    .energy_with_vv10_scale(&parent.density, 1, scale)?;
                Ok::<_, DftError>(
                    energy
                        .into_iter()
                        .zip(&parent.parent.energy)
                        .map(|(target, parent)| target - parent)
                        .collect(),
                )
            })
            .transpose()?;
        let value = correlate(
            parent,
            nao,
            &integrals,
            &reference.energies[0],
            coefficient,
            self.config,
        )?;
        Ok((value, target_correction))
    }
}

fn correlate(
    parent: SpinorWavefunctionPath,
    nao: usize,
    integrals: &[Vec<Complex>],
    energies: &[f64],
    coefficient: f64,
    config: &ResponseSolverConfig,
) -> Result<SpinorCorrelatedPath> {
    let source =
        spinor_transform_path(nao, integrals, &parent.orbitals, energies, parent.occupied)?;
    let pairs = generalized_taylor_response(energies, parent.occupied, &source, &parent.mo_fock)?;
    let imaginary_energy_defect = check_pair_response(&pairs, config.acceptance())?;
    let energy = parent
        .parent
        .energy
        .iter()
        .zip(&pairs.energy)
        .map(|(parent, pair)| parent + coefficient * pair.re)
        .collect();
    Ok(SpinorCorrelatedPath {
        parent,
        pairs,
        energy,
        imaginary_energy_defect,
    })
}

fn check_spinor_pt2_workspace(nao: usize, width: usize, limit_mb: usize) -> Result<()> {
    // Bound the spatial AO series and packed spinor transform. The
    // transformation never expands the AO tensor to the spinor dimension.
    let bytes = nao
        .checked_pow(4)
        .and_then(|n| n.checked_mul(width))
        .and_then(|n| n.checked_mul(16 * 32))
        .ok_or_else(|| DftError::InvalidInput("spinor PT2 workspace overflows".into()))?;
    let required_mb = bytes.div_ceil(1024 * 1024);
    if required_mb > limit_mb {
        return Err(DftError::ResourceLimit {
            operation: "spinor PT2 coefficient workspace",
            required_mb,
            limit_mb,
        });
    }
    Ok(())
}

fn check_pair_response(pairs: &GeneralizedPt2Taylor, tolerance: f64) -> Result<f64> {
    let imaginary_energy_defect = pairs
        .energy
        .iter()
        .map(|e| e.im.abs())
        .fold(0.0_f64, f64::max);
    let absolute_residual = pairs.residuals.iter().copied().fold(0.0, f64::max);
    let relative_residual = pairs.relative_residuals.iter().copied().fold(0.0, f64::max);
    let residual = imaginary_energy_defect
        .max(absolute_residual)
        .max(relative_residual);
    if residual > tolerance {
        crate::logger::warn(&format!(
            "Generalized PT2 pair response: equation residual={absolute_residual:.6e} Hartree, normalized residual={relative_residual:.6e}, imaginary energy={imaginary_energy_defect:.6e} Hartree"
        ));
        return Err(DftError::ResponseNotConverged {
            iterations: pairs.energy.len() - 1,
            residual,
        });
    }
    Ok(imaginary_energy_defect)
}

/// Zeroth order of the same generalized PT2 pair equations used by the
/// response, without a field-path geometry or an orbital-response solve.
pub(crate) fn stationary_spinor_pt2(
    basis: &MolBasis,
    reference: &SpinorScfReport,
    memory_mb: usize,
) -> Result<GeneralizedPt2Taylor> {
    let ([band], [occupation]) = (reference.bands.as_slice(), reference.occupations.as_slice())
    else {
        return Err(DftError::InvalidInput(
            "molecular generalized PT2 needs one spinor sector".into(),
        ));
    };
    let occupied = occupation.iter().take_while(|&&f| f == 1.0).count();
    if !reference.converged || occupation[occupied..].iter().any(|&f| f != 0.0) {
        return Err(DftError::InvalidInput(
            "molecular generalized PT2 needs a converged integer occupied projector".into(),
        ));
    }
    check_spinor_pt2_workspace(basis.nao_sph, 1, memory_mb)?;
    let eri =
        crate::integrals::eri_incore(basis, crate::integrals::CoulombKernel::Coulomb, memory_mb)?;
    let integrals = [eri.as_slice().iter().copied().map(Complex::from).collect()];
    drop(eri);
    let energies = &band.energies;
    let mut fock = Cm::zeros(energies.len(), energies.len());
    for (i, &energy) in energies.iter().enumerate() {
        fock.re[(i, i)] = energy;
    }
    let source = spinor_transform_path(
        basis.nao_sph,
        &integrals,
        &[cm(&band.coefficients)],
        energies,
        occupied,
    )?;
    let pairs = generalized_taylor_response(energies, occupied, &source, &[fock])?;
    check_pair_response(&pairs, ResponseSolverConfig::default().acceptance())?;
    Ok(pairs)
}
