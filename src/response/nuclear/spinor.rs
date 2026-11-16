// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complex spinor nuclear paths sharing the existing geometry jets and
//! stationary complex-projector solver. The present interaction provider
//! covers Coulomb, full/range-separated exchange and globally projected
//! semilocal density/gradient/tau/current paths on the moving Becke grid.
use super::geometry::{GeometryTaylor, NonlocalTaylor};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::integrals::london::ComplexMatrix as Cm;
use crate::linalg::Matrix;
use crate::periodic::spinor_scf::SpinorScfReport;
use crate::periodic::BlochMatrix;
use crate::response::field_path::{
    stationary_complex_path, stationary_retained_complex_energy_path, ComplexPathBackend,
    ComplexPathReference, FieldPathTaylor, RetainedComplexBackend,
};
use crate::response::ResponseSolverConfig;
use crate::scf::ScfOptions;
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};
use std::cell::RefCell;
use std::sync::Arc;

mod excited;
mod excited_kernel;
mod excited_path;
mod field;
#[cfg(test)]
mod pair_kernel_tests;
mod retained;
mod semilocal;
mod thermal;
pub use excited::{spinor_excitations, SpinorExcitationReport};
pub use excited_path::{
    spinor_excitation_path, spinor_excited_state_hessian, SpinorExcitationPath,
};
pub use field::{
    spinor_correlated_one_body_taylor, spinor_wavefunction_path, SpinorCorrelatedPath,
    SpinorWavefunctionPath,
};
pub(crate) use field::{stationary_spinor_pt2, SpinorCorrelatedSolver};

pub struct SpinorElectricReport {
    pub values: Vec<crate::response::spinor::SpinorPolarizability>,
    pub reference_fock_defect: f64,
    pub angular_history: Vec<[f64; 3]>,
    pub cache_bound_history: Vec<[f64; 3]>,
}

pub struct SpinorOneBodyReport {
    pub values: Vec<crate::response::spinor::SpinorOperatorResponse>,
    pub reference_fock_defect: f64,
    pub angular_history: Vec<[f64; 3]>,
    pub cache_bound_history: Vec<[f64; 3]>,
}

/// Frequency-resolved electric response of the same spin-covariant complex
/// functional as the molecular GKS reference. Reuse the nuclear provider at
/// fixed geometry, including its descriptor-space AD Hessian and angular
/// convergence checks, instead of substituting a spin-diagonal kernel.
pub fn spinor_electric_response(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: Option<&MolecularGrid>,
    functional: &ResolvedFunctional,
    reference: &SpinorScfReport,
    options: &ScfOptions,
    operator: &BlochMatrix,
    frequencies: &[crate::response::complex::Complex],
    config: &ResponseSolverConfig,
) -> Result<SpinorElectricReport> {
    let sources = crate::integrals::dipole_matrices(basis, crate::math::Vec3::zero())
        .iter()
        .map(|d| {
            let m = scalar(d);
            BlochMatrix { re: m.re, im: m.im }
        })
        .collect::<Vec<_>>();
    let result = spinor_one_body_response(
        molecule,
        basis,
        grid,
        functional,
        reference,
        options,
        operator,
        &sources,
        frequencies,
        config,
    )?;
    let mut values = Vec::new();
    for report in result.values {
        values.push(crate::response::spinor::SpinorPolarizability {
            frequency: report.frequency,
            alpha: std::array::from_fn(|a| std::array::from_fn(|b| -report.susceptibility[a][b])),
            residuals: report.residuals.try_into().map_err(|_| {
                DftError::InvalidInput("electric spinor response needs three sources".into())
            })?,
            iterations: report.iterations.try_into().map_err(|_| {
                DftError::InvalidInput("electric spinor response needs three sources".into())
            })?,
        });
    }
    Ok(SpinorElectricReport {
        values,
        reference_fock_defect: result.reference_fock_defect,
        angular_history: result.angular_history,
        cache_bound_history: result.cache_bound_history,
    })
}

/// General full-spin one-body response of the same immutable molecular
/// functional/reference used for electric and nuclear response.
pub fn spinor_one_body_response(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: Option<&MolecularGrid>,
    functional: &ResolvedFunctional,
    reference: &SpinorScfReport,
    options: &ScfOptions,
    operator: &BlochMatrix,
    sources: &[BlochMatrix],
    frequencies: &[crate::response::complex::Complex],
    config: &ResponseSolverConfig,
) -> Result<SpinorOneBodyReport> {
    spinor_constrained_one_body_response(
        molecule,
        basis,
        grid,
        functional,
        reference,
        options,
        operator,
        sources,
        &[],
        frequencies,
        config,
    )
}

/// Full-spin fixed-expectation response, using the identical generic
/// molecular Hessian as the unconstrained electric/nuclear provider.
pub fn spinor_constrained_one_body_response(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: Option<&MolecularGrid>,
    functional: &ResolvedFunctional,
    reference: &SpinorScfReport,
    options: &ScfOptions,
    operator: &BlochMatrix,
    sources: &[BlochMatrix],
    constraints: &[BlochMatrix],
    frequencies: &[crate::response::complex::Complex],
    config: &ResponseSolverConfig,
) -> Result<SpinorOneBodyReport> {
    let (mut backend, state) = prepare_spinor_path(
        molecule,
        basis,
        grid,
        functional,
        reference,
        options,
        operator,
        &vec![[0.0; 3]; molecule.len()],
        [0.0; 3],
        1,
        config,
    )?;
    // One-body sources keep nuclei and basis fixed. The SCF pair engines
    // evaluate this same linear action without expensive Taylor/compensated
    // contractions for absent nuclear derivatives. Keep the nuclear provider
    // below for actual moving-basis paths and as an independent test oracle.
    backend.fixed_pairs = Some(crate::response::spinor::Interaction::at_spinor_reference(
        molecule,
        basis,
        functional,
        grid,
        options,
        &reference.density[0],
    )?);
    crate::logger::log("[spinor-response] fixed-geometry Coulomb/exchange cache ready");
    backend.cache = descriptor_cache(&backend, 2, options.incore_mb)?;
    let zero = Cm::zeros(state.density[0].re.rows, state.density[0].re.cols);
    let actual = backend.fock_coefficient(&[vec![state.density[0].clone()], vec![zero]], 0)?;
    let reference_fock_defect = actual[0]
        .re
        .max_abs_diff(&reference.fields.fock[0].re)
        .max(actual[0].im.max_abs_diff(&reference.fields.fock[0].im));
    if reference_fock_defect > config.acceptance() {
        return Err(DftError::ResponseNotConverged {
            iterations: 0,
            residual: reference_fock_defect,
        });
    }
    let kernel = |direction: &BlochMatrix| {
        // A transition density at complex frequency is not Hermitian. The
        // real Hessian acts on its two independent Hermitian parts H+iA.
        let mut h = Cm::zeros(direction.rows(), direction.cols());
        let mut a = h.clone();
        for i in 0..direction.rows() {
            for j in 0..direction.cols() {
                h.re[(i, j)] = 0.5 * (direction.re[(i, j)] + direction.re[(j, i)]);
                h.im[(i, j)] = 0.5 * (direction.im[(i, j)] - direction.im[(j, i)]);
                a.re[(i, j)] = 0.5 * (direction.im[(i, j)] + direction.im[(j, i)]);
                a.im[(i, j)] = -0.5 * (direction.re[(i, j)] - direction.re[(j, i)]);
            }
        }
        let zero = Cm::zeros(direction.rows(), direction.cols());
        let vh = backend.linear_kernel(&[h, zero.clone()])?;
        let va = backend.linear_kernel(&[a, zero])?;
        let mut out = BlochMatrix {
            re: vh[0].re.clone(),
            im: vh[0].im.clone(),
        };
        for i in 0..out.rows() {
            for j in 0..out.cols() {
                out.re[(i, j)] -= va[0].im[(i, j)];
                out.im[(i, j)] += va[0].re[(i, j)];
            }
        }
        Ok(out)
    };
    let values = crate::response::spinor::constrained_spinor_operator_response_with_kernel(
        &reference.bands[0].coefficients,
        &reference.bands[0].energies,
        state.occupations[0],
        &kernel,
        sources,
        constraints,
        frequencies,
        config,
    )?;
    Ok(SpinorOneBodyReport {
        values,
        reference_fock_defect,
        angular_history: backend.angular_history.into_inner(),
        cache_bound_history: backend
            .cache
            .as_ref()
            .map_or_else(Vec::new, |c| c.angular_history.clone()),
    })
}

fn cm(m: &BlochMatrix) -> Cm {
    Cm {
        re: m.re.clone(),
        im: m.im.clone(),
    }
}
fn scalar(m: &Matrix) -> Cm {
    let n = m.rows;
    let mut out = Cm::zeros(2 * n, 2 * n);
    for s in 0..2 {
        for i in 0..n {
            for j in 0..n {
                out.re[(s * n + i, s * n + j)] = m[(i, j)];
            }
        }
    }
    out
}
fn add(a: &mut Cm, b: &Cm) {
    for (x, y) in a.re.as_mut_slice().iter_mut().zip(b.re.as_slice()) {
        *x += y;
    }
    for (x, y) in a.im.as_mut_slice().iter_mut().zip(b.im.as_slice()) {
        *x += y;
    }
}
struct SpinorGeometry<'a> {
    nonlocal: Option<NonlocalTaylor>,
    geometry: GeometryTaylor,
    fixed_pairs: Option<crate::response::spinor::Interaction<'a>>,
    functional: &'a ResolvedFunctional,
    linear_functional: ResolvedFunctional,
    reference_density: Cm,
    angular_order: usize,
    angular_tolerance: f64,
    angular_history: RefCell<Vec<[f64; 3]>>,
    cache: Option<Arc<semilocal::GridCache>>,
    overlap: Vec<Cm>,
    operator: Cm,
}
impl SpinorGeometry<'_> {
    fn semilocal(
        &self,
        p: &[Cm],
        order: usize,
        moving: bool,
        potentials: bool,
    ) -> Result<(Vec<f64>, Vec<Cm>)> {
        let evaluate = |rule, cache: Option<&semilocal::GridCache>| {
            semilocal::evaluate(
                self.geometry.grid_points(),
                self.functional,
                p,
                order,
                rule,
                moving,
                potentials,
                cache,
            )
        };
        let compare = |rule: usize,
                       current: &(Vec<f64>, Vec<Cm>),
                       previous: &(Vec<f64>, Vec<Cm>)| {
            let energy_error = current
                .0
                .iter()
                .zip(&previous.0)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let potential_error = current
                .1
                .iter()
                .zip(&previous.1)
                .map(|(a, b)| a.re.max_abs_diff(&b.re).max(a.im.max_abs_diff(&b.im)))
                .fold(0.0_f64, f64::max);
            self.angular_history
                .borrow_mut()
                .push([rule as f64, energy_error, potential_error]);
            energy_error.max(potential_error)
        };
        if let Some(cache) = self.cache.as_ref().filter(|_| {
            p[0].re.as_slice() == self.reference_density.re.as_slice()
                && p[0].im.as_slice() == self.reference_density.im.as_slice()
        }) {
            if if potentials { order <= 1 } else { order <= 2 } {
                let current = evaluate(cache.rule, Some(cache))?;
                let prior = cache
                    .previous
                    .as_deref()
                    .expect("a converged descriptor cache retains the previous rule");
                let previous = evaluate(prior.rule, Some(prior))?;
                if compare(cache.rule, &current, &previous) <= self.angular_tolerance {
                    return Ok(current);
                }
            }
        }
        let mut rule = self.angular_order.min(32);
        let mut previous = evaluate(rule, None)?;
        loop {
            rule *= 2;
            let current = evaluate(rule, None)?;
            let error = compare(rule, &current, &previous);
            if error <= self.angular_tolerance {
                return Ok(current);
            }
            if rule == crate::periodic::spinor_xc::MAX_SPIN_ANGULAR_ORDER {
                return Err(DftError::ResponseNotConverged {
                    iterations: rule,
                    residual: error,
                });
            }
            previous = current;
        }
    }
    /// Reuse the existing spatial Coulomb/exchange contraction. Opposite
    /// auxiliary channels cancel Coulomb for off-diagonal spin blocks;
    /// neither their real nor imaginary AO parts are symmetrized.
    fn interaction(&self, p: &[Cm], order: usize) -> Result<Cm> {
        if order == 0 {
            if let Some(engine) = &self.fixed_pairs {
                let value = engine.pair_response(&BlochMatrix {
                    re: p[0].re.clone(),
                    im: p[0].im.clone(),
                })?;
                return Ok(Cm {
                    re: value.re,
                    im: value.im,
                });
            }
        }
        let n = self.geometry.overlap[0].rows;
        let zeros: [Vec<Matrix>; 2] = std::array::from_fn(|_| vec![Matrix::zeros(n, n); order + 1]);
        let mut out = Cm::zeros(2 * n, 2 * n);
        for imaginary in [false, true] {
            let block = |s: usize, t: usize| -> Vec<Matrix> {
                p[..=order]
                    .iter()
                    .map(|p| {
                        let source = if imaginary { &p.im } else { &p.re };
                        let mut m = Matrix::zeros(n, n);
                        for i in 0..n {
                            for j in 0..n {
                                m[(i, j)] = source[(s * n + i, t * n + j)];
                            }
                        }
                        m
                    })
                    .collect()
            };
            let diagonal = [block(0, 0), block(1, 1)];
            let fields =
                self.geometry
                    .evaluate_linear(&self.linear_functional, &zeros, &diagonal, order)?;
            let target = if imaginary { &mut out.im } else { &mut out.re };
            for s in 0..2 {
                for i in 0..n {
                    for j in 0..n {
                        target[(s * n + i, s * n + j)] = fields[s][order][(i, j)];
                    }
                }
            }
            for (s, t) in [(0, 1), (1, 0)] {
                let q = block(s, t);
                let mut negative = q.clone();
                for m in &mut negative {
                    for v in m.as_mut_slice() {
                        *v = -*v;
                    }
                }
                let fields = self.geometry.evaluate_linear(
                    &self.linear_functional,
                    &zeros,
                    &[q, negative],
                    order,
                )?;
                for i in 0..n {
                    for j in 0..n {
                        target[(s * n + i, t * n + j)] = fields[0][order][(i, j)];
                    }
                }
            }
        }
        Ok(out)
    }
    fn core(&self, order: usize) -> Cm {
        let mut h = scalar(&self.geometry.core_coefficients()[order]);
        if order == 0 {
            add(&mut h, &self.operator);
        }
        h
    }
}
impl ComplexPathBackend for SpinorGeometry<'_> {
    fn overlap(&self) -> &[Cm] {
        &self.overlap
    }
    fn fock_coefficient(&self, p: &[Vec<Cm>; 2], order: usize) -> Result<[Cm; 2]> {
        let mut f = self.interaction(&p[0], order)?;
        add(&mut f, &self.core(order));
        if !self.functional.parts.is_empty() {
            let (_, v) = self.semilocal(&p[0], order, true, true)?;
            add(&mut f, &v[order]);
        }
        if let Some(nonlocal) = &self.nonlocal {
            let (_, v) = nonlocal.evaluate(&[], &spin_diagonal_real_path(&p[0]), None, order)?;
            add(&mut f, &scalar(&v[0][order]));
        }
        Ok([
            f,
            Cm::zeros(
                2 * self.geometry.overlap[0].rows,
                2 * self.geometry.overlap[0].rows,
            ),
        ])
    }
    fn linear_kernel(&self, direction: &[Cm; 2]) -> Result<[Cm; 2]> {
        let mut f = self.interaction(std::slice::from_ref(&direction[0]), 0)?;
        if !self.functional.parts.is_empty() {
            let (_, v) = self.semilocal(
                &[self.reference_density.clone(), direction[0].clone()],
                1,
                false,
                true,
            )?;
            add(&mut f, &v[1]);
        }
        if let Some(nonlocal) = &self.nonlocal {
            let (_, v) = nonlocal.evaluate(
                &[],
                &spin_diagonal_real_path(std::slice::from_ref(&self.reference_density)),
                Some(&spin_diagonal_real_path(std::slice::from_ref(
                    &direction[0],
                ))),
                0,
            )?;
            add(&mut f, &scalar(&v[0][0]));
        }
        Ok([
            f,
            Cm::zeros(
                2 * self.geometry.overlap[0].rows,
                2 * self.geometry.overlap[0].rows,
            ),
        ])
    }
    fn energy(&self, p: &[Vec<Cm>; 2], order: usize) -> Result<Vec<f64>> {
        self.energy_with_vv10_scale(&p[0], order, 1.0)
    }
    fn dipole(&self, p: &[Vec<Cm>; 2], order: usize) -> Vec<[f64; 3]> {
        (0..=order)
            .map(|k| {
                std::array::from_fn(|a| {
                    self.geometry.nuclear_dipole[k][a]
                        - (0..=k)
                            .map(|m| {
                                p[0][m]
                                    .real_trace_product(&scalar(&self.geometry.dipoles[a][k - m]))
                            })
                            .sum::<f64>()
                })
            })
            .collect()
    }
}

impl SpinorGeometry<'_> {
    /// Evaluate the energy on an already supplied density series. A distinct
    /// target may have a different VV10 weight from the stationary parent.
    fn energy_with_vv10_scale(&self, p: &[Cm], order: usize, scale: f64) -> Result<Vec<f64>> {
        let interaction = (0..=order)
            .map(|k| self.interaction(p, k))
            .collect::<Result<Vec<_>>>()?;
        let mut e = vec![0.0; order + 1];
        for k in 0..=order {
            e[k] = self.geometry.nuclear_coefficients()[k]
                + self.geometry.nuclear_field_coefficients()[k];
            for m in 0..=k {
                e[k] += p[m].real_trace_product(&self.core(k - m))
                    + 0.5 * p[m].real_trace_product(&interaction[k - m]);
            }
        }
        if !self.functional.parts.is_empty() {
            let (xc, _) = self.semilocal(p, order, true, false)?;
            for (e, xc) in e.iter_mut().zip(xc) {
                *e += xc;
            }
        }
        if let Some(nonlocal) = &self.nonlocal {
            let (values, _) = nonlocal.evaluate(&[], &spin_diagonal_real_path(p), None, order)?;
            for (energy, value) in e.iter_mut().zip(values) {
                *energy += scale * value;
            }
        }
        Ok(e)
    }
}

// Hermitian spinors have a real symmetric total charge density. VV10 acts
// identically on the two spin diagonals and has no transverse spin term.
fn spin_diagonal_real_path(path: &[Cm]) -> [Vec<Matrix>; 2] {
    spin_diagonal_path(path, |p| &p.re)
}

fn spin_diagonal_path(path: &[Cm], channel: impl Fn(&Cm) -> &Matrix) -> [Vec<Matrix>; 2] {
    let n = path[0].re.rows / 2;
    std::array::from_fn(|spin| {
        let offset = spin * n;
        path.iter()
            .map(|p| {
                let mut block = Matrix::zeros(n, n);
                for i in 0..n {
                    for j in 0..n {
                        block[(i, j)] = channel(p)[(offset + i, offset + j)];
                    }
                }
                block
            })
            .collect()
    })
}

/// Analytic nuclear energy coefficients of a full complex spinor reference.
/// `operator` is a fixed AO one-electron matrix (no implied nuclear SOC
/// derivative). Geometry, metric/Pulay and all spin exchange endpoints are
/// differentiated through the existing jet and projector providers.
pub fn spinor_nuclear_taylor(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: Option<&MolecularGrid>,
    functional: &ResolvedFunctional,
    reference: &SpinorScfReport,
    options: &ScfOptions,
    operator: &BlochMatrix,
    direction: &[[f64; 3]],
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<FieldPathTaylor> {
    let (backend, state) = prepare_spinor_path(
        molecule, basis, grid, functional, reference, options, operator, direction, [0.0; 3],
        order, config,
    )?;
    Ok(stationary_complex_path(&backend, &state, order, config)?.0)
}

/// Stationary energy paths use the shared 2n+1 theorem. Unlike the complete
/// density path above, they solve only floor(order/2) orbital equations and
/// do not claim higher-order density or dipole coefficients.
pub struct SpinorNuclearEnergyTaylor {
    pub energy: Vec<f64>,
    pub solved_order: usize,
    pub residuals: Vec<f64>,
    pub reference_defect: f64,
    /// Retained zero-order energy minus the supplied SCF energy, Hartree.
    pub reference_energy_defect: f64,
    pub projector_defects: Vec<f64>,
    pub stationarity_defects: Vec<f64>,
    /// Trace defects of positive-order density coefficients at fixed total N.
    pub thermal_number_defects: Vec<f64>,
    /// Factorial-normalized chemical potential coefficients for a Fermi ensemble.
    pub thermal_chemical_potentials: Vec<f64>,
    pub angular_refinement_history: Vec<[f64; 3]>,
    /// Conservative descriptor-space bounds, not physical operator residuals.
    pub angular_cache_bound_history: Vec<[f64; 3]>,
}

/// Analytic stationary nuclear energy coefficients with complete metric and
/// grid order and only the required lower-order orbital response.
pub fn spinor_nuclear_energy_taylor(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: Option<&MolecularGrid>,
    functional: &ResolvedFunctional,
    reference: &SpinorScfReport,
    options: &ScfOptions,
    operator: &BlochMatrix,
    direction: &[[f64; 3]],
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<SpinorNuclearEnergyTaylor> {
    SpinorNuclearContext::new(
        molecule, basis, grid, functional, reference, options, operator, order, config,
    )?
    .energy_taylor(direction, order)
}

fn descriptor_cache(
    backend: &SpinorGeometry<'_>,
    max_order: usize,
    memory_mb: usize,
) -> Result<Option<Arc<semilocal::GridCache>>> {
    let active = match backend.functional.family {
        crate::xc::Family::Lda => 4usize,
        crate::xc::Family::Gga => 16,
        crate::xc::Family::MetaGga => 32,
    };
    let bytes = backend
        .geometry
        .grid_points()
        .len()
        .checked_mul(2 * 8 * (if max_order >= 2 { active * active } else { 0 } + 65))
        .ok_or_else(|| DftError::InvalidInput("spinor descriptor cache size overflow".into()))?;
    let cache = if backend.functional.parts.is_empty() || bytes.div_ceil(1024 * 1024) > memory_mb {
        None
    } else {
        Some(Arc::new(semilocal::GridCache::new(
            backend.geometry.grid_points(),
            backend.functional,
            &backend.reference_density,
            backend.angular_order,
            backend.angular_tolerance,
            max_order >= 2,
        )?))
    };
    Ok(cache)
}

/// Immutable reference-bound context sharing descriptor kernels across all
/// Cartesian nuclear paths. No AO fourth-rank response tensor is stored.
pub struct SpinorNuclearContext<'a> {
    molecule: &'a Molecule,
    basis: &'a MolBasis,
    grid: Option<&'a MolecularGrid>,
    functional: &'a ResolvedFunctional,
    reference: &'a SpinorScfReport,
    options: &'a ScfOptions,
    operator: &'a BlochMatrix,
    max_order: usize,
    config: &'a ResponseSolverConfig,
    cache: Option<Arc<semilocal::GridCache>>,
}
impl<'a> SpinorNuclearContext<'a> {
    pub fn new(
        molecule: &'a Molecule,
        basis: &'a MolBasis,
        grid: Option<&'a MolecularGrid>,
        functional: &'a ResolvedFunctional,
        reference: &'a SpinorScfReport,
        options: &'a ScfOptions,
        operator: &'a BlochMatrix,
        max_order: usize,
        config: &'a ResponseSolverConfig,
    ) -> Result<Self> {
        let backend = prepare_spinor_geometry(
            molecule,
            basis,
            grid,
            functional,
            reference,
            options,
            operator,
            &vec![[0.0; 3]; molecule.len()],
            [0.0; 3],
            1,
            config,
            true,
        )?;
        if !(1..=4).contains(&max_order) {
            return Err(DftError::InvalidInput(
                "spinor nuclear context order must be one through four".into(),
            ));
        }
        let cache = descriptor_cache(&backend, max_order, options.incore_mb)?;
        Ok(Self {
            molecule,
            basis,
            grid,
            functional,
            reference,
            options,
            operator,
            max_order,
            config,
            cache,
        })
    }
    pub fn energy_taylor(
        &self,
        direction: &[[f64; 3]],
        order: usize,
    ) -> Result<SpinorNuclearEnergyTaylor> {
        self.molecular_energy_taylor(
            &super::MolecularDirection::nuclear(direction.to_vec()),
            order,
        )
    }

    /// Stationary full-complex energy for a simultaneous nuclear/electric
    /// path, sharing reference kernels and constraints with nuclear tensors.
    pub fn molecular_energy_taylor(
        &self,
        direction: &super::MolecularDirection,
        order: usize,
    ) -> Result<SpinorNuclearEnergyTaylor> {
        let backend = self.path_geometry(direction, order)?;
        if self.options.electronic_temperature > 0.0 {
            return thermal::solve(
                &backend,
                self.reference,
                self.options.electronic_temperature,
                order,
                self.config,
            );
        }
        let state = integer_path_reference(self.reference, self.basis.nao_sph);
        let value = stationary_retained_complex_energy_path(&backend, &state, order, self.config)?;
        let reference_energy_defect = value.energy[0] - self.reference.fields.energy;
        let bounds = self
            .cache
            .as_ref()
            .map_or_else(Vec::new, |c| c.angular_history.clone());
        Ok(SpinorNuclearEnergyTaylor {
            energy: value.energy,
            solved_order: value.solved_order,
            residuals: value.residuals,
            reference_defect: value.reference_defect,
            reference_energy_defect,
            projector_defects: value.projector_defects,
            stationarity_defects: value.stationarity_defects,
            thermal_number_defects: Vec::new(),
            thermal_chemical_potentials: Vec::new(),
            angular_refinement_history: backend.angular_history.into_inner(),
            angular_cache_bound_history: bounds,
        })
    }

    fn path_geometry(
        &self,
        direction: &super::MolecularDirection,
        order: usize,
    ) -> Result<SpinorGeometry<'a>> {
        if order > self.max_order || order == 0 {
            return Err(DftError::InvalidInput(
                "nuclear energy order exceeds its prepared context".into(),
            ));
        }
        let mut backend = prepare_spinor_geometry(
            self.molecule,
            self.basis,
            self.grid,
            self.functional,
            self.reference,
            self.options,
            self.operator,
            &direction.nuclear,
            direction.electric,
            order,
            self.config,
            true,
        )?;
        backend.cache = self.cache.clone();
        Ok(backend)
    }
}

fn prepare_spinor_path<'a>(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: Option<&MolecularGrid>,
    functional: &'a ResolvedFunctional,
    reference: &SpinorScfReport,
    options: &ScfOptions,
    operator: &BlochMatrix,
    direction: &[[f64; 3]],
    electric: [f64; 3],
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<(SpinorGeometry<'a>, ComplexPathReference)> {
    let backend = prepare_spinor_geometry(
        molecule, basis, grid, functional, reference, options, operator, direction, electric,
        order, config, false,
    )?;
    Ok((backend, integer_path_reference(reference, basis.nao_sph)))
}

#[allow(clippy::too_many_arguments)]
fn prepare_spinor_geometry<'a>(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: Option<&MolecularGrid>,
    functional: &'a ResolvedFunctional,
    reference: &SpinorScfReport,
    options: &ScfOptions,
    operator: &BlochMatrix,
    direction: &[[f64; 3]],
    electric: [f64; 3],
    order: usize,
    config: &ResponseSolverConfig,
    allow_thermal: bool,
) -> Result<SpinorGeometry<'a>> {
    config.validate()?;
    let thermal = allow_thermal && options.electronic_temperature > 0.0;
    if options.vv10.is_some() && order > 1 && !thermal
        || functional.required_vv10().is_some() && options.vv10.is_none()
        || options.skala.is_some()
        || options.pcm.is_some()
        || options.density_fit.is_some()
        || !matches!(options.kernel, crate::integrals::CoulombKernel::Coulomb)
    {
        return Err(DftError::Unsupported {
            feature: "complex spinor nuclear path with additional functional/core/solvent terms"
                .into(),
            missing:
                "the corresponding full-spin analytic geometry and transverse response provider"
                    .into(),
        });
    }
    let dimension = 2 * basis.nao_sph;
    if operator.rows() != dimension
        || operator.cols() != dimension
        || operator.im.rows != dimension
        || operator.im.cols != dimension
        || operator.hermiticity_error() > 1e-10
        || operator
            .re
            .as_slice()
            .iter()
            .chain(operator.im.as_slice())
            .any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput(
            "spinor nuclear source must be a finite matching Hermitian spin/AO operator".into(),
        ));
    }
    if !reference.converged
        || reference.density.len() != 1
        || reference.bands.len() != 1
        || (!allow_thermal && options.electronic_temperature != 0.0)
        || order == 0
        || order > 4
        || direction.len() != molecule.len()
    {
        return Err(DftError::InvalidInput("spinor nuclear paths require a converged single-sector reference, one direction per nucleus and order 1..4".into()));
    }
    if options.electronic_temperature == 0.0
        && reference.occupations[0]
            .iter()
            .any(|f| *f != 0.0 && *f != 1.0)
    {
        return Err(DftError::InvalidInput(
            "spinor nuclear projector response requires integer occupations".into(),
        ));
    }
    let angular_order = reference
        .fields
        .xc_diagnostics
        .as_ref()
        .map_or(8, |d| d.angular_order.max(8));
    spinor_geometry(
        molecule,
        basis,
        grid,
        functional,
        options,
        operator,
        direction,
        electric,
        order,
        cm(&reference.density[0]),
        angular_order,
        config.tolerance,
    )
}

fn integer_path_reference(reference: &SpinorScfReport, nao: usize) -> ComplexPathReference {
    let c = cm(&reference.bands[0].coefficients);
    let zero = Cm::zeros(c.re.rows, c.re.rows);
    ComplexPathReference {
        coefficients: [c.clone(), c],
        energies: [
            reference.bands[0].energies.clone(),
            vec![0.0; reference.bands[0].energies.len()],
        ],
        occupations: [
            reference.occupations[0]
                .iter()
                .filter(|f| **f == 1.0)
                .count(),
            0,
        ],
        density: [cm(&reference.density[0]), zero],
        fock: [cm(&reference.fields.fock[0]), Cm::zeros(2 * nao, 2 * nao)],
        unrestricted: true,
    }
}

#[allow(clippy::too_many_arguments)]
fn spinor_geometry<'a>(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: Option<&MolecularGrid>,
    functional: &'a ResolvedFunctional,
    options: &ScfOptions,
    operator: &BlochMatrix,
    direction: &[[f64; 3]],
    electric: [f64; 3],
    order: usize,
    reference_density: Cm,
    angular_order: usize,
    angular_tolerance: f64,
) -> Result<SpinorGeometry<'a>> {
    let geometry = GeometryTaylor::new(
        molecule,
        basis,
        grid,
        functional,
        None,
        direction,
        options.external_field.map_or([0.0; 3], |v| v.to_array()),
        electric,
        order,
        options.incore_mb,
    )?;
    let overlap = geometry.overlap.iter().map(scalar).collect();
    let mut linear_functional = ResolvedFunctional::hartree_fock();
    linear_functional.hybrid = functional.hybrid;
    let mut source = cm(operator);
    if let Some(extra) = &options.extra_hcore {
        if extra.rows != basis.nao_sph
            || extra.cols != basis.nao_sph
            || extra.as_slice().iter().any(|v| !v.is_finite())
        {
            return Err(DftError::InvalidInput(
                "fixed extra core matrix must match the spatial AO space".into(),
            ));
        }
        add(&mut source, &scalar(extra));
    }
    Ok(SpinorGeometry {
        nonlocal: options
            .vv10
            .map(|parameters| {
                NonlocalTaylor::molecular(
                    molecule,
                    basis,
                    &options.nonlocal_grid(molecule)?,
                    parameters,
                    direction,
                    order,
                )
            })
            .transpose()?,
        geometry,
        fixed_pairs: None,
        functional,
        linear_functional,
        reference_density,
        angular_order,
        angular_tolerance,
        angular_history: RefCell::new(Vec::new()),
        cache: None,
        overlap,
        operator: source,
    })
}

/// Fixed spinor AO density energy, Fock and explicit nuclear derivative.
/// The caller supplies its own metric chain term for its orbital coordinates.
pub(crate) fn spinor_off_shell_geometry(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: Option<&MolecularGrid>,
    functional: &ResolvedFunctional,
    options: &ScfOptions,
    density: &BlochMatrix,
    tolerance: f64,
) -> Result<(f64, BlochMatrix, Vec<crate::math::Vec3>)> {
    let components = crate::periodic::spinor_scf::pauli_components(density)?;
    let charge = &components[0].re;
    let n = density.rows();
    let zero = Cm::zeros(n, n);
    let operator = BlochMatrix::zeros(n, n);
    let mut direction = vec![[0.0; 3]; molecule.len()];
    // VV10 depends on the spin trace. Its batched first derivative can
    // handle every Cartesian component without rebuilding a pair jet.
    let mut local_options = options.clone();
    local_options.vv10 = None;
    let base = |direction: &[[f64; 3]], order| {
        spinor_geometry(
            molecule,
            basis,
            grid,
            functional,
            &local_options,
            &operator,
            direction,
            [0.0; 3],
            order,
            cm(density),
            8,
            tolerance,
        )
    };
    let mut reference = base(&direction, 0)?;
    // Reuse descriptor energy and first derivatives for every nuclear path
    // at this same instantaneous density and quadrature.
    reference.cache = descriptor_cache(&reference, 1, options.incore_mb)?;
    let at_zero = [vec![cm(density)], vec![zero.clone()]];
    let [mut fock, _] = reference.fock_coefficient(&at_zero, 0)?;
    let mut energy = reference.energy(&at_zero, 0)?[0];
    let mut gradient = Vec::with_capacity(molecule.len());
    let path = [vec![cm(density), zero.clone()], vec![zero.clone(), zero]];
    for atom in 0..molecule.len() {
        let mut axes = [0.0; 3];
        for (axis, derivative) in axes.iter_mut().enumerate() {
            direction[atom][axis] = 1.0;
            let mut displaced = base(&direction, 1)?;
            displaced.cache = reference.cache.clone();
            *derivative = displaced.energy(&path, 1)?[1];
            direction[atom][axis] = 0.0;
        }
        gradient.push(crate::math::Vec3::new(axes[0], axes[1], axes[2]));
    }
    if let Some(parameters) = options.vv10 {
        let grid = options.nonlocal_grid(molecule)?;
        let (value, potential) = crate::nlc::vv10_potential(basis, &grid, charge, parameters)?;
        energy += value;
        add(&mut fock, &scalar(&potential));
        let nonlocal = crate::nlc::vv10_gradient(molecule, basis, &grid, charge, parameters)?;
        for (g, term) in gradient.iter_mut().zip(nonlocal) {
            *g += term;
        }
    }
    Ok((
        energy,
        BlochMatrix {
            re: fock.re,
            im: fock.im,
        },
        gradient,
    ))
}

#[cfg(test)]
pub(crate) fn spinor_off_shell_energy(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: Option<&MolecularGrid>,
    functional: &ResolvedFunctional,
    options: &ScfOptions,
    density: &BlochMatrix,
    tolerance: f64,
) -> Result<f64> {
    let n = density.rows();
    let geometry = spinor_geometry(
        molecule,
        basis,
        grid,
        functional,
        options,
        &BlochMatrix::zeros(n, n),
        &vec![[0.0; 3]; molecule.len()],
        [0.0; 3],
        0,
        cm(density),
        8,
        tolerance,
    )?;
    Ok(geometry.energy(&[vec![cm(density)], vec![Cm::zeros(n, n)]], 0)?[0])
}

/// Fixed-geometry reuse of the nuclear-response descriptor Hessian. The same
/// Pauli density, current and tau definitions serve the spinor SCF optimizer.
pub(crate) struct SpinorScfXcKernel<'a> {
    functional: &'a ResolvedFunctional,
    points: Vec<super::geometry::GridPointTaylor>,
    reference: Cm,
    cache: semilocal::GridCache,
}
impl<'a> SpinorScfXcKernel<'a> {
    pub(crate) fn new(
        functional: &'a ResolvedFunctional,
        basis: &MolBasis,
        grid: &MolecularGrid,
        density: &BlochMatrix,
        tolerance: f64,
    ) -> Result<Self> {
        let ao = crate::grid::eval::eval_aos(basis, &grid.points, true);
        let grad = ao.grad.expect("requested orbital gradients");
        let points: Vec<_> = (0..grid.len())
            .map(|g| super::geometry::GridPointTaylor {
                weight: vec![grid.weights[g]],
                aos: (0..basis.nao_sph)
                    .map(|i| {
                        std::array::from_fn(|d| {
                            vec![if d == 0 {
                                ao.phi[(g, i)]
                            } else {
                                grad[d - 1][(g, i)]
                            }]
                        })
                    })
                    .collect(),
            })
            .collect();
        let reference = Cm {
            re: density.re.clone(),
            im: density.im.clone(),
        };
        let cache = semilocal::GridCache::new(&points, functional, &reference, 8, tolerance, true)?;
        for row in &cache.angular_history {
            crate::logger::log(&format!(
                "[spinor-scf] TRAH XC angular_order={} energy_bound={:.12e} kernel_bound={:.12e}",
                row[0], row[1], row[2]
            ));
        }
        Ok(Self {
            functional,
            points,
            reference,
            cache,
        })
    }
    pub(crate) fn apply(&self, direction: &BlochMatrix) -> Result<BlochMatrix> {
        let (_, f) = semilocal::evaluate(
            &self.points,
            self.functional,
            &[
                self.reference.clone(),
                Cm {
                    re: direction.re.clone(),
                    im: direction.im.clone(),
                },
            ],
            1,
            self.cache.rule,
            false,
            true,
            Some(&self.cache),
        )?;
        Ok(BlochMatrix {
            re: f[1].re.clone(),
            im: f[1].im.clone(),
        })
    }
}
