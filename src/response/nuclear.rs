// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Nuclear Taylor response in a moving Gaussian basis and integration grid.
//! Metric transport preserves B(t)^T S(t) B(t) = I. Projector and
//! stationarity equations then determine every density coefficient through
//! the physical orbital Hessian. No displaced SCF enters this driver.

mod correlated;
mod correlated_excited;
pub use correlated_excited::{
    correlated_excitation_path, correlated_excited_state_hessian, CorrelatedExcitationPath,
    CorrelatedExcitedHessian,
};
mod frame;
pub use correlated::{
    correlated_molecular_taylor, correlated_molecular_taylor_for_model, MolecularCorrelatedTaylor,
};
pub(crate) mod extra_core;
pub(crate) mod geometry;
mod lattice;
mod lattice_correlated;
mod lattice_dc;
mod lattice_nmr;
pub use lattice::{
    lattice_mode_gruneisen_with_dispersion, lattice_volume_gruneisen,
    lattice_volume_gruneisen_with_dispersion, LatticeVolumeGruneisen,
};
pub mod kinetic_operators;
mod magnetic;
pub mod magnetic_operators;
mod mixed_reference;
pub mod one_body;
mod open_shell;
pub mod spin_orbit_operators;
pub mod spin_spin_operators;
pub(crate) use open_shell::open_shell_cross_operator_response;
pub(crate) use open_shell::open_shell_molecular_energy_taylor_retained;
pub use open_shell::{
    open_shell_excitation_path, open_shell_excited_state_hessian, open_shell_tda_states,
    OpenShellExcitationPath, OpenShellExcitations, OpenShellTdaState,
};
mod spinor;
mod thermal_roks;
#[cfg(test)]
pub(crate) use spinor::spinor_off_shell_energy;
pub(crate) use spinor::spinor_off_shell_geometry;
pub use spinor::{
    spinor_constrained_one_body_response, spinor_correlated_one_body_taylor,
    spinor_electric_response, spinor_excitation_path, spinor_excitations,
    spinor_excited_state_hessian, spinor_nuclear_energy_taylor, spinor_nuclear_taylor,
    spinor_one_body_response, spinor_wavefunction_path, SpinorCorrelatedPath, SpinorElectricReport,
    SpinorExcitationPath, SpinorExcitationReport, SpinorNuclearContext, SpinorNuclearEnergyTaylor,
    SpinorOneBodyReport, SpinorWavefunctionPath,
};
pub(crate) use spinor::{stationary_spinor_pt2, SpinorCorrelatedSolver, SpinorScfXcKernel};
pub(crate) use thermal_roks::ThermalRoksNuclearContext;
pub(crate) mod precision;
mod reduced;

pub use crate::periodic::path_series::{LatticePath, ScalarProbe};
pub use frame::{
    excitation_path, excited_state_hessian, optical_tensor_path, ExcitationPath,
    ExcitedStateHessian, OpticalTensorPath,
};
pub(crate) use geometry::{becke_weight, learned_fock_coefficient};
pub(crate) use geometry::{GeometrySources, GeometryTaylor, ParticleSource};
pub use lattice::{
    lattice_density_response, lattice_elastic_constants, lattice_elastic_constants_at_field,
    lattice_elastic_constants_with_dispersion, lattice_electric_tensor,
    lattice_electric_tensor_at_field, lattice_field_response, lattice_field_response_at_field,
    lattice_finite_field, lattice_force_constant_tensor, lattice_force_constant_tensor_at_field,
    lattice_force_constants_along, lattice_force_constants_along_at_field,
    lattice_harmonic_force_constants, lattice_mode_gruneisen, lattice_orbital_susceptibility,
    lattice_path_energy, lattice_path_energy_at_field, DensityResponse, ElectricTensorSymmetry,
    FieldRequest, LatticeDensityResponse, LatticeElastic, LatticeElectricTensor,
    LatticeFieldResponse, LatticeFiniteField, LatticeForceConstants, LatticeGruneisen,
    LatticeOrbitalSusceptibility, LatticeTensor, OrbitalSusceptibility, RelaxedFieldResponse,
    SeriesReference,
};
pub(crate) use lattice::{
    lattice_electronic_finite_field, lattice_electronic_force_constant_tensor,
    lattice_electronic_force_constants_along,
};
pub use lattice_correlated::{LatticeCorrelatedSolver, LatticeCorrelatedTaylor};
pub use lattice_dc::{lattice_divide_conquer, LatticeDivideConquer};
pub use lattice_nmr::{lattice_nmr_shielding, LatticeShielding, NucleusShielding, ShieldingPoint};
pub use magnetic::{
    magnetic_nuclear_field_taylor, magnetic_nuclear_taylor, MixedMagneticNuclearTaylor,
};
pub use mixed_reference::{
    mixed_reference_path, mixed_reference_surface, MixedReferencePath, MixedReferenceState,
    MixedReferenceSurface,
};
pub use open_shell::{
    open_shell_dynamic_polarizability, open_shell_electric, open_shell_hessian,
    open_shell_molecular_energy_taylor, open_shell_nuclear_energy_taylor,
    open_shell_operator_response, OpenShellDynamicPolarizability, OpenShellElectric,
    OpenShellHessian, OpenShellOperatorResponse, OpenShellTaylor,
};
pub use reduced::{
    cubic_force_constants_along, cubic_force_constants_reduced, quartic_force_constants_along,
    reduced_order_taylor, ContractedForceConstants, CubicForceConstants, ReducedTaylor,
};

use super::{ResponseReport, ResponseSolver, ResponseSolverConfig};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::linalg::Matrix;
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

pub(super) fn add(a: &mut Matrix, b: &Matrix, scale: f64) {
    for (x, y) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *x += scale * y;
    }
}
pub(crate) fn product(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    (0..=n).map(|k| product_coefficient(a, b, k)).collect()
}

pub(crate) fn product_into(a: &[f64], b: &[f64], coefficients: &mut [f64]) {
    for (order, value) in coefficients.iter_mut().enumerate() {
        *value = product_coefficient(a, b, order);
    }
}

fn product_coefficient(a: &[f64], b: &[f64], order: usize) -> f64 {
    (0..=order).map(|j| a[j] * b[order - j]).sum()
}
/// One convolution coefficient, with product and summation roundoff
/// recovered before large projector/metric terms cancel.
pub(crate) fn matrix_coefficient(a: &[Matrix], b: &[Matrix], n: usize) -> Matrix {
    let mut value = Matrix::zeros(a[0].rows, b[0].cols);
    for i in 0..value.rows {
        for j in 0..value.cols {
            let mut sum = crate::linalg::AccurateSum::default();
            for k in 0..=n {
                for p in 0..a[k].cols {
                    sum.add_product(a[k][(i, p)], b[n - k][(p, j)]);
                }
            }
            value[(i, j)] = sum.value();
        }
    }
    value
}

pub(crate) fn matrix_product(a: &[Matrix], b: &[Matrix], order: usize) -> Vec<Matrix> {
    (0..=order).map(|n| matrix_coefficient(a, b, n)).collect()
}
/// Complete B(t)^T F(t) B(t), carrying the low intermediate products
/// through the final convolution. Every ordered matrix entry is evaluated.
pub(super) fn matrix_congruence(b: &[Matrix], f: &[Matrix], order: usize) -> Vec<Matrix> {
    let mut high = Vec::new();
    let mut low = Vec::new();
    for n in 0..=order {
        let mut h = Matrix::zeros(f[0].rows, b[0].cols);
        let mut l = h.clone();
        for i in 0..h.rows {
            for j in 0..h.cols {
                let mut sum = crate::linalg::AccurateSum::default();
                for k in 0..=n {
                    for a in 0..f[0].cols {
                        sum.add_product(f[k][(i, a)], b[n - k][(a, j)]);
                    }
                }
                (h[(i, j)], l[(i, j)]) = sum.parts();
            }
        }
        high.push(h);
        low.push(l);
    }
    (0..=order)
        .map(|n| {
            let mut v = Matrix::zeros(b[0].cols, b[0].cols);
            for i in 0..v.rows {
                for j in 0..v.cols {
                    let mut sum = crate::linalg::AccurateSum::default();
                    for k in 0..=n {
                        for a in 0..b[0].rows {
                            sum.add_product(b[k][(a, i)], high[n - k][(a, j)]);
                            sum.add_product(b[k][(a, i)], low[n - k][(a, j)]);
                        }
                    }
                    v[(i, j)] = sum.value();
                }
            }
            v
        })
        .collect()
}

fn max_abs(m: &Matrix) -> f64 {
    m.as_slice().iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// Factorial-normalized energy components along a nuclear path.
#[derive(Clone, Debug, Default)]
pub struct NuclearEnergyCoefficient {
    pub nuclear_repulsion: f64,
    pub nuclear_field: f64,
    pub one_electron: f64,
    pub hartree: f64,
    pub exchange: f64,
    pub long_range_exchange: f64,
    pub semilocal_xc: f64,
    pub nonlocal_correlation: f64,
    /// Contribution -kT S/k_B to the electronic Mermin free energy.
    pub electronic_entropy_energy: f64,
}
impl NuclearEnergyCoefficient {
    pub fn total(&self) -> f64 {
        self.nuclear_repulsion
            + self.nuclear_field
            + self.one_electron
            + self.hartree
            + self.exchange
            + self.long_range_exchange
            + self.semilocal_xc
            + self.nonlocal_correlation
            + self.electronic_entropy_energy
    }
}

#[derive(Clone, Debug)]
pub struct CoefficientEquationDiagnostics {
    /// Norm of the initial reconstructed order-n orbital source (Hartree).
    pub source_norm: f64,
    /// Norm after rebuilding the retained physical coefficient (Hartree).
    pub reconstructed_norm: f64,
    /// Reconstructed norm divided by the initial source norm, without a floor.
    /// This is not the relative residual of the linear response operator.
    pub reconstructed_relative_residual: f64,
    /// Fresh relative residual of the returned coefficient's original equation.
    pub linear_relative_residual: f64,
    /// Largest fresh relative residual of the individual correction solves.
    pub inner_linear_relative_residual: f64,
}

#[derive(Clone, Debug)]
pub struct NuclearTaylor {
    /// R_A(t) = R_A(0) + t direction[A], positions and directions in Bohr.
    pub direction: Vec<[f64; 3]>,
    /// E(t) = E_reference + t field_direction, in atomic electric field units.
    pub field_direction: [f64; 3],
    /// E(t) = sum_n energy_coefficients[n] t^n, in Hartree.
    pub energy_coefficients: Vec<f64>,
    pub energy_components: Vec<NuclearEnergyCoefficient>,
    /// Molecular dipole series, in electron*Bohr, including moving AO
    /// position operators and the explicit nuclear dipole derivatives.
    pub dipole_coefficients: Vec<[f64; 3]>,
    pub density_coefficients: Vec<[Matrix; 2]>,
    /// Low parts retained in the physical density and energy contractions.
    /// Add these to `density_coefficients` in compensated arithmetic.
    pub density_corrections: Vec<[Matrix; 2]>,
    pub overlap_coefficients: Vec<Matrix>,
    /// Factorial-normalized B(t)^T F_s(t) B(t), with B(t)^T S(t) B(t)=I
    /// and B(0) equal to their metric-polished reference frame. These are operator
    /// coefficients in the transported reference frame, not eigenvalues.
    pub orthonormal_fock_coefficients: Vec<[Matrix; 2]>,
    pub projector_defects: Vec<f64>,
    /// Active fixed-spectrum constraints; raw projector defects are retained.
    pub fixed_population: Option<super::ensemble::FixedPopulationDiagnostics>,
    /// Active canonical Fermi population constraint at positive temperature.
    pub thermal_population: Option<super::ensemble::FermiPathDiagnostics>,
    pub stationarity_defects: Vec<f64>,
    pub metric_defects: Vec<f64>,
    /// Coefficients of Tr[P_s(t) S(t)] minus its constant value.
    pub particle_number_coefficients: Vec<[f64; 2]>,
    pub reference_energy_error: f64,
    pub reference_fock_error: f64,
    /// Retained coefficient reconstruction diagnostics, indexed by order
    /// one through the requested order. None for other response backends.
    pub coefficient_equations: Option<Vec<CoefficientEquationDiagnostics>>,
}

/// Nuclear and electric perturbations share the same metric-aware driver.
pub type MolecularTaylor = NuclearTaylor;

#[derive(Clone, Debug)]
pub struct MolecularDirection {
    pub nuclear: Vec<[f64; 3]>,
    pub electric: [f64; 3],
}
impl MolecularDirection {
    pub fn nuclear(direction: Vec<[f64; 3]>) -> Self {
        Self {
            nuclear: direction,
            electric: [0.0; 3],
        }
    }
    pub fn electric(n_atoms: usize, direction: [f64; 3]) -> Self {
        Self {
            nuclear: vec![[0.0; 3]; n_atoms],
            electric: direction,
        }
    }
    fn vector(&self) -> Vec<f64> {
        self.nuclear
            .iter()
            .flatten()
            .chain(self.electric.iter())
            .copied()
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct MixedMolecularResponse {
    pub directions: Vec<MolecularDirection>,
    pub energy_derivative: f64,
    pub dipole_derivative: [f64; 3],
    pub energy_accumulation_roundoff: f64,
    pub dipole_accumulation_roundoff: [f64; 3],
    pub directional_evaluations: usize,
}

#[derive(Clone, Debug)]
pub struct StaticSpectroscopicDerivative {
    pub nuclear_direction: Vec<[f64; 3]>,
    /// 1: directional APT; 2: directional Raman polarizability derivative;
    /// 3: directional hyper-Raman first-hyperpolarizability derivative.
    pub electric_order: usize,
    /// Row-major Cartesian tensor; the output dipole index varies slowest.
    pub tensor: Vec<f64>,
    pub accumulation_roundoff: Vec<f64>,
    pub permutation_defect: f64,
    /// Number of comparisons of distinct independently evaluated index orders.
    pub permutation_comparisons: usize,
    pub directional_evaluations: usize,
}

#[derive(Clone, Debug)]
pub struct MixedNuclearResponse {
    pub directions: Vec<Vec<[f64; 3]>>,
    /// The actual mixed energy derivative; no factorial normalization.
    pub energy_derivative: f64,
    pub accumulation_roundoff: f64,
    pub directional_evaluations: usize,
}

pub(crate) fn metric_transport(c: &Matrix, overlap: &[Matrix], order: usize) -> Vec<Matrix> {
    // The analytic principal inverse square root of I + X. Matrix powers
    // retain all noncommuting coefficient products; no elementwise root.
    let mut x: Vec<_> = overlap.iter().map(|s| c.congruence_accurate(s)).collect();
    // Reference MO coefficients are orthonormal only to floating-point
    // precision. Retain that constant defect: discarding it magnifies it
    // into high-order number/metric errors when response coefficients grow.
    for i in 0..c.cols {
        x[0][(i, i)] -= 1.0;
    }
    let mut power = vec![Matrix::zeros(c.cols, c.cols); order + 1];
    power[0] = Matrix::identity(c.cols);
    let mut inverse = power.clone();
    let mut binomial = 1.0;
    // Four extra powers retain constant-defect insertions through every
    // requested Taylor order; the neglected terms contain at least five
    // powers of the reference's near-machine-precision metric defect.
    for k in 1..=order + 4 {
        power = matrix_product(&power, &x, order);
        binomial *= (-0.5 - (k - 1) as f64) / k as f64;
        for n in 0..=order {
            add(&mut inverse[n], &power[n], binomial);
        }
    }
    inverse
        .iter()
        .map(|m| c.matmul_accurate(m, false))
        .collect()
}

/// Complete coefficient of [F(t), D(t)] before rounding its cancelling terms.
fn commutator_coefficient(f: &[Matrix], d: &[Matrix], n: usize) -> Matrix {
    let mut value = Matrix::zeros(d[0].rows, d[0].cols);
    for i in 0..value.rows {
        for j in 0..value.cols {
            let mut sum = crate::linalg::AccurateSum::default();
            for k in 0..=n {
                for a in 0..value.cols {
                    sum.add_product(f[k][(i, a)], d[n - k][(a, j)]);
                    sum.add_product(-d[n - k][(i, a)], f[k][(a, j)]);
                }
            }
            value[(i, j)] = sum.value();
        }
    }
    value
}

fn orbital_source(solver: &ResponseSolver<'_>, commutator: &[Matrix; 2]) -> Vec<f64> {
    let mut source = Vec::with_capacity(solver.u_len());
    for (s, commutator) in commutator.iter().enumerate().take(solver.spins()) {
        let no = solver.dims.n_occ[s];
        for a in 0..solver.dims.n_vir[s] {
            for i in 0..no {
                source.push(commutator[(no + a, i)]);
            }
        }
    }
    source
}

fn add_orbital_coefficient(
    solver: &ResponseSolver<'_>,
    d: &mut [Vec<Matrix>; 2],
    n: usize,
    amplitudes: &[f64],
) -> bool {
    assert_eq!(amplitudes.len(), solver.u_len());
    let mut index = 0;
    let mut changed = false;
    for (s, d) in d.iter_mut().enumerate().take(solver.spins()) {
        let no = solver.dims.n_occ[s];
        for a in 0..solver.dims.n_vir[s] {
            for i in 0..no {
                let old = d[n][(no + a, i)];
                let value = old + amplitudes[index];
                changed |= old.to_bits() != value.to_bits();
                d[n][(no + a, i)] = value;
                d[n][(i, no + a)] = value;
                index += 1;
            }
        }
    }
    if solver.spins() == 1 {
        d[1][n] = d[0][n].clone();
    }
    changed
}

pub(super) fn density_path(b: &[Matrix], d: &[Matrix], order: usize) -> Vec<Matrix> {
    let transpose: Vec<_> = b.iter().map(Matrix::transpose).collect();
    matrix_congruence(&transpose, d, order)
}

/// Analytic molecular HF / semilocal / hybrid / RSH energy response,
/// including cubic and quartic derivatives at fixed electronic temperature.
/// The integration topology and the retained reference AO subspace are held
/// fixed, while coordinates, Becke weights and AO values are differentiated.
#[allow(clippy::too_many_arguments)]
pub fn nuclear_taylor(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    direction: &[[f64; 3]],
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<NuclearTaylor>> {
    molecular_taylor(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        &MolecularDirection::nuclear(direction.to_vec()),
        order,
        config,
    )
}

/// Analytic simultaneous nuclear/electric energy and dipole Taylor series.
#[allow(clippy::too_many_arguments)]
pub fn molecular_taylor(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    direction: &MolecularDirection,
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<MolecularTaylor>> {
    Ok(molecular_taylor_reference(
        mol, mb, functional, grid, scf, opts, direction, order, config, None, None,
    )?
    .0)
}

/// `molecular_taylor` together with the transported reference solution that
/// the path response is built on.
#[allow(clippy::too_many_arguments)]
fn molecular_taylor_parts(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    direction: &MolecularDirection,
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<(ResponseReport<MolecularTaylor>, frame::TransportedPath)> {
    // Excitation and reduced-order continuations still impose a projector.
    // Keep their existing physical constraint until their own population
    // equations are connected; the full static path below is isospectral.
    super::diagnostics::reference_diagnostics(scf)?;
    molecular_taylor_reference(
        mol, mb, functional, grid, scf, opts, direction, order, config, None, None,
    )
}

/// [`molecular_taylor_parts`] on a moving-geometry series that the caller
/// built for the same direction at an order of at least `order`, so that a
/// higher-order energy evaluation along the same path does not build the
/// integral and grid series twice. The series is returned in the path.
#[allow(clippy::too_many_arguments)]
fn molecular_taylor_parts_on(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    direction: &MolecularDirection,
    order: usize,
    config: &ResponseSolverConfig,
    geometry: geometry::GeometryTaylor,
) -> Result<(ResponseReport<MolecularTaylor>, frame::TransportedPath)> {
    super::diagnostics::reference_diagnostics(scf)?;
    molecular_taylor_reference(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        direction,
        order,
        config,
        Some(geometry),
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn molecular_taylor_reference(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    direction: &MolecularDirection,
    order: usize,
    config: &ResponseSolverConfig,
    prebuilt: Option<geometry::GeometryTaylor>,
    extra_core: Option<&extra_core::ExtraCoreTaylor>,
) -> Result<(ResponseReport<MolecularTaylor>, frame::TransportedPath)> {
    let field_direction = direction.electric;
    let direction = &direction.nuclear;
    let reference_field = opts.external_field.map_or([0.0; 3], |f| [f.x, f.y, f.z]);
    config.validate()?;
    let fixed = scf.fixed_occupation_residual.is_some();
    let thermal = scf.electronic_temperature > 0.0;
    let integer_reference = !fixed && !thermal;
    if thermal && (fixed || order > 4 || extra_core.is_some()) {
        return Err(DftError::Unsupported {
            feature: "canonical finite-temperature molecular Taylor response".into(),
            missing: "orders above four, extra-core perturbations and simultaneous prescribed nonthermal populations".into(),
        });
    }
    let reference = if !integer_reference {
        super::diagnostics::density_diagnostics(scf)?
    } else {
        super::diagnostics::reference_diagnostics(scf)?
    };
    if order == usize::MAX
        || direction.len() != mol.len()
        || direction.iter().flatten().any(|v| !v.is_finite())
        || field_direction
            .iter()
            .chain(reference_field.iter())
            .any(|v| !v.is_finite())
        || mb.nao_sph != scf.overlap.rows
        || !scf.converged
        || reference.orthonormality_defect > 1e-8
        || integer_reference && reference.projector_defect > 1e-8
    {
        return Err(DftError::InvalidInput(
            "nuclear Taylor response requires finite per-atom directions and an \
            orthonormal converged reference with a supported population constraint"
                .into(),
        ));
    }
    if opts.pcm.is_some()
        || functional.required_nonlocal_kind().is_some_and(|kind| {
            kind != crate::xc::RequiredNonlocalTerm::Vv10 || opts.vv10.is_none()
        })
        || opts.density_fit.is_some()
        || (opts.extra_hcore.is_some() && extra_core.is_none())
        || !matches!(opts.kernel, crate::integrals::CoulombKernel::Coulomb)
    {
        return Err(DftError::Unsupported {
            feature: "nuclear Taylor response with additional Hamiltonian terms".into(),
            missing: "moving-geometry Taylor kernels for solvent, the selected nonlocal composition, RI-J and additional one-electron operators".into(),
        });
    }
    if (!functional.parts.is_empty() || opts.skala.is_some()) && grid.is_none() {
        return Err(DftError::InvalidInput(
            "nuclear semilocal response requires the reference molecular grid".into(),
        ));
    }
    let mut geometry = match prebuilt {
        Some(shared) if shared.overlap.len() > order => shared,
        Some(_) => {
            return Err(DftError::InvalidInput(
                "a shared moving-geometry series is shorter than the response order".into(),
            ))
        }
        None => geometry::GeometryTaylor::new(
            mol,
            mb,
            grid,
            functional,
            opts.skala.as_ref(),
            direction,
            reference_field,
            field_direction,
            order,
            opts.incore_mb,
        )?,
    };
    if let Some(parameters) = opts.vv10 {
        let grid = opts.nonlocal_grid(mol)?;
        let geometry_order = geometry.overlap.len() - 1;
        geometry.attach_molecular_nonlocal(
            mol,
            mb,
            &grid,
            parameters,
            1.0,
            direction,
            geometry_order,
            opts.incore_mb,
        )?;
    }
    if let Some(source) = extra_core {
        source.attach(
            &mut geometry,
            mol,
            opts,
            &MolecularDirection {
                nuclear: direction.to_vec(),
                electric: field_direction,
            },
        )?;
        // Validate the unpolished ACTUAL reference before allowing orbital
        // refinement to conceal an incomplete zeroth-order source.
        let p = std::array::from_fn(|s| vec![scf.density[s].clone()]);
        let value = geometry.evaluate(functional, &p, 0)?;
        let tolerance = config.acceptance();
        let fock_error = (0..2)
            .map(|s| value.fock[s][0].max_abs_diff(&scf.fock[s]))
            .fold(0.0_f64, f64::max);
        if geometry.reference_core().max_abs_diff(&scf.h_core) > tolerance
            || fock_error > tolerance
            || (value.coefficient_total(0) - scf.energy).abs() > tolerance
            || (geometry.classical_coefficient(0) - scf.e_nuc).abs() > tolerance
        {
            return Err(DftError::InvalidInput(
                "extra-core zeroth operator/Fock/classical/full energy differs from actual reference".into(),
            ));
        }
    }
    let overlap = &geometry.overlap[..=order];
    let mut b: [Vec<Matrix>; 2] =
        std::array::from_fn(|s| metric_transport(&scf.mo_coeff[s], overlap, order));
    // Kernel and moving-density contractions use the same polished zeroth
    // frame and density. Keep the original reference diagnostics separately.
    let mut kernel_reference = scf.clone();
    kernel_reference.overlap = overlap[0].clone();
    kernel_reference.mo_coeff = std::array::from_fn(|s| b[s][0].clone());
    let base_density: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
        let mut populations = Matrix::zeros(b[s][0].cols, b[s][0].cols);
        for (i, &occupation) in scf.occupations[s].iter().enumerate() {
            populations[(i, i)] = occupation;
        }
        density_path(&b[s][..1], &[populations], 0)
    });
    kernel_reference.density = std::array::from_fn(|s| base_density[s][0].clone());
    let base_evaluation = geometry.evaluate(functional, &base_density, 0)?;
    kernel_reference.fock = std::array::from_fn(|s| base_evaluation.fock[s][0].clone());
    let polish = if integer_reference && (scf.unrestricted || scf.n_alpha == scf.n_beta) {
        let initial = ResponseSolver::new(mol, mb, functional, grid, &kernel_reference, opts)?;
        Some(initial.polish_reference(config, |density| {
            let density = std::array::from_fn(|s| vec![density[s].clone()]);
            let value = geometry.evaluate(functional, &density, 0)?;
            Ok((
                value.coefficient_total(0),
                std::array::from_fn(|s| value.fock[s][0].clone()),
            ))
        })?)
    } else {
        None
    };
    if let Some(polish) = &polish {
        if polish.iterations > 0 {
            for s in 0..2 {
                for coefficient in &mut b[s] {
                    *coefficient = coefficient.matmul_accurate(&polish.rotations[s], false);
                }
            }
            kernel_reference.mo_coeff = std::array::from_fn(|s| b[s][0].clone());
            let density = super::orbital_frame::OrbitalFrame::new(
                &kernel_reference.mo_coeff,
                &kernel_reference.fock,
                [scf.n_alpha, scf.n_beta],
                scf.unrestricted,
            )
            .occupied_density();
            kernel_reference.density = density.clone();
            let series = std::array::from_fn(|s| vec![density[s].clone()]);
            let value = geometry.evaluate(functional, &series, 0)?;
            kernel_reference.fock = std::array::from_fn(|s| value.fock[s][0].clone());
        }
    }
    let ensemble = if !integer_reference {
        Some(super::ensemble::EnsembleSolver::new(
            mol,
            mb,
            functional,
            grid,
            &kernel_reference,
            opts,
        )?)
    } else {
        None
    };
    let integer = if !integer_reference {
        None
    } else {
        Some(ResponseSolver::new(
            mol,
            mb,
            functional,
            grid,
            &kernel_reference,
            opts,
        )?)
    };
    let solver = match &ensemble {
        Some(s) => &s.kernel,
        None => integer
            .as_ref()
            .expect("integer response kernel constructed"),
    };
    let mut d: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
        let mut p = Matrix::zeros(scf.mo_coeff[s].cols, scf.mo_coeff[s].cols);
        if !integer_reference {
            for (i, &f) in scf.occupations[s].iter().enumerate() {
                p[(i, i)] = f;
            }
        } else {
            for i in 0..solver.dims.n_occ[s] {
                p[(i, i)] = 1.;
            }
        }
        vec![p]
    });
    let mut f: [Vec<Matrix>; 2] =
        std::array::from_fn(|s| vec![b[s][0].congruence_accurate(&kernel_reference.fock[s])]);
    let mut residuals = Vec::new();
    let mut iterations = Vec::new();
    let mut methods = Vec::new();
    let mut fixed_population = None;
    let mut fixed_density = None;
    let retained = if integer_reference && opts.skala.is_none() {
        Some(precision::integer_path(
            &geometry,
            functional,
            solver,
            &kernel_reference.mo_coeff,
            order,
            config,
        )?)
    } else {
        None
    };
    if let Some(path) = &retained {
        b = std::array::from_fn(|s| precision::high(&path.b[s]));
        d = std::array::from_fn(|s| precision::high(&path.d[s]));
        residuals = path.residuals.clone();
        iterations = path.iterations.clone();
    }
    let integer_order = if !integer_reference || retained.is_some() {
        0
    } else {
        order
    };
    for n in 1..=integer_order {
        for s in 0..2 {
            let dim = d[s][0].rows;
            d[s].push(Matrix::zeros(dim, dim));
            // The current zero coefficient makes both endpoint terms zero.
            // Reuse the compensated complete convolution for the lower orders.
            let lower = matrix_coefficient(&d[s], &d[s], n);
            let no = solver.dims.n_occ[s];
            for i in 0..dim {
                for j in 0..dim {
                    if i < no && j < no {
                        d[s][n][(i, j)] = -lower[(i, j)];
                    }
                    if i >= no && j >= no {
                        d[s][n][(i, j)] = lower[(i, j)];
                    }
                }
            }
        }
        let p: [Vec<Matrix>; 2] = std::array::from_fn(|s| density_path(&b[s], &d[s], n));
        let evaluation = geometry.evaluate(functional, &p, n)?;
        for s in 0..2 {
            f[s].push(matrix_congruence(&b[s], &evaluation.fock[s], n)[n].clone());
        }
        let commutator = std::array::from_fn(|s| commutator_coefficient(&f[s], &d[s], n));
        let source = orbital_source(solver, &commutator);
        let source_norm = source.iter().fold(0.0_f64, |norm, &v| norm.hypot(v));
        let solve_config = config.for_absolute_residual(source_norm)?;
        let solution = solver.solve_ov_report_with_structure(
            &source,
            &solve_config,
            super::krylov::OperatorStructure::Hermitian,
        )?;
        let u: Vec<_> = solution.amplitudes.iter().map(|z| z.re).collect();
        add_orbital_coefficient(solver, &mut d, n, &u);
        let mut residual = solution.relative_residual;
        let mut count = solution.iterations;
        // The common kernel and the moving-geometry evaluator have different
        // contraction schedules. Rebuild the actual coefficient equation;
        // carrying a separately rounded linearized F_n into the next order
        // otherwise amplifies its low-order error. Bounded iterative
        // refinement changes only the numerical solution, never the operator.
        for refinement in 0..=2 {
            let physical_density = std::array::from_fn(|s| density_path(&b[s], &d[s], n));
            let actual = geometry.evaluate(functional, &physical_density, n)?;
            f = std::array::from_fn(|s| matrix_congruence(&b[s], &actual.fock[s], n));
            let commutator = std::array::from_fn(|s| commutator_coefficient(&f[s], &d[s], n));
            let defect = orbital_source(solver, &commutator);
            let norm = defect.iter().fold(0.0_f64, |m, &v| m.hypot(v));
            if refinement == 2 || norm == 0.0 {
                break;
            }
            let correction = solver.solve_ov_report_with_structure(
                &defect,
                &config.for_absolute_residual(norm)?,
                super::krylov::OperatorStructure::Hermitian,
            )?;
            residual = residual.max(correction.relative_residual);
            count += correction.iterations;
            let amplitudes: Vec<_> = correction.amplitudes.iter().map(|z| z.re).collect();
            if !add_orbital_coefficient(solver, &mut d, n, &amplitudes) {
                break;
            }
        }
        residuals.push(residual);
        iterations.push(count);
        methods.push(solution.method);
    }
    let mut thermal_population = None;
    let mut entropy_energy = None;
    if let Some(ensemble) = &ensemble {
        if thermal {
            let path =
                super::ensemble::canonical_fermi_path(ensemble, &b, d, order, config, |p, n| {
                    Ok(geometry.evaluate(functional, p, n)?.fock)
                })?;
            d = path.density_in_frame;
            entropy_energy = Some(path.entropy_energy);
            thermal_population = Some(path.diagnostics);
            residuals = path.residuals;
            iterations = path.iterations;
        } else {
            let path = super::ensemble::fixed_population_path(
                ensemble,
                &b,
                overlap,
                order,
                config,
                |density, n| {
                    let p: [Vec<Matrix>; 2] =
                        std::array::from_fn(|s| density.iter().map(|v| v[s].clone()).collect());
                    let evaluation = geometry.evaluate(functional, &p, n)?;
                    Ok([evaluation.fock[0][n].clone(), evaluation.fock[1][n].clone()])
                },
            )?;
            d = path.density_in_frame;
            fixed_density = Some(std::array::from_fn(|s| {
                path.density
                    .iter()
                    .map(|p| p[s].clone())
                    .collect::<Vec<_>>()
            }));
            fixed_population = Some(path.diagnostics);
            residuals = path.residuals;
            iterations = path.iterations;
        }
    }
    let p: [Vec<Matrix>; 2] = fixed_density.unwrap_or_else(|| match &retained {
        Some(path) => std::array::from_fn(|s| precision::high(&path.p[s])),
        None => std::array::from_fn(|s| density_path(&b[s], &d[s], order)),
    });
    // Rebuild from the actual moving integrals, grid and full density path.
    // This also detects disagreement with reference integral screening.
    let mut evaluation = match &retained {
        Some(path) => path.evaluation.clone(),
        None => geometry.evaluate(functional, &p, order)?,
    };
    if let Some(entropy) = entropy_energy {
        for (component, value) in evaluation.energy.iter_mut().zip(entropy) {
            component.electronic_entropy_energy = value;
        }
    }
    let actual_f: [Vec<Matrix>; 2] = std::array::from_fn(|s| match &retained {
        Some(path) => precision::high(&path.f[s]),
        None => matrix_congruence(&b[s], &evaluation.fock[s], order),
    });
    let mut projector_defects = Vec::new();
    let mut stationarity_defects = Vec::new();
    let mut metric_defects = Vec::new();
    let mut particle_number_coefficients = Vec::new();
    let metric: [Vec<Matrix>; 2] =
        std::array::from_fn(|s| matrix_congruence(&b[s], overlap, order));
    for n in 1..=order {
        let mut pd = 0.0_f64;
        let mut sd = 0.0_f64;
        let mut md = 0.0_f64;
        let mut number = [0.0; 2];
        for s in 0..2 {
            if let Some(path) = &retained {
                let mut projector = precision::coefficient(&path.d[s], &path.d[s], n);
                projector.add_scaled(&path.d[s][n], -1.0);
                pd = pd.max(projector.max_abs());
                sd = sd.max(precision::commutator(&path.f[s], &path.d[s], n).max_abs());
                let overlap = precision::retain(overlap);
                md = md.max(precision::congruence(&path.b[s], &overlap, order)[n].max_abs());
                number[s] = precision::trace_product(&path.p[s], &overlap, n);
                continue;
            }
            // Accumulate each complete physical convolution before rounding.
            // Separately rounded matrix products and trace fragments lose the
            // small residual when high-order coefficients cancel.
            for i in 0..d[s][0].rows {
                for j in 0..d[s][0].cols {
                    let mut projector = crate::linalg::AccurateSum::default();
                    let mut stationary = crate::linalg::AccurateSum::default();
                    projector.add_product(-1.0, d[s][n][(i, j)]);
                    for k in 0..=n {
                        for a in 0..d[s][0].cols {
                            projector.add_product(d[s][k][(i, a)], d[s][n - k][(a, j)]);
                            stationary.add_product(actual_f[s][k][(i, a)], d[s][n - k][(a, j)]);
                            stationary.add_product(-d[s][n - k][(i, a)], actual_f[s][k][(a, j)]);
                        }
                    }
                    pd = pd.max(projector.value().abs());
                    sd = sd.max(stationary.value().abs());
                }
            }
            let mut trace = crate::linalg::AccurateSum::default();
            for k in 0..=n {
                for (&density, &overlap) in p[s][k]
                    .as_slice()
                    .iter()
                    .zip(geometry.overlap[n - k].as_slice())
                {
                    trace.add_product(density, overlap);
                }
            }
            number[s] = trace.value();
            md = md.max(max_abs(&metric[s][n]));
        }
        projector_defects.push(pd);
        stationarity_defects.push(sd);
        metric_defects.push(md);
        particle_number_coefficients.push(number);
    }
    let energy_coefficients: Vec<_> = if extra_core.is_some() {
        evaluation
            .total_energy()
            .into_iter()
            .zip(evaluation.energy_remainders())
            .map(|(high, low)| extra_core::accurate_total(high, low))
            .collect()
    } else {
        evaluation
            .energy
            .iter()
            .map(NuclearEnergyCoefficient::total)
            .collect()
    };
    let reference_energy_error = energy_coefficients[0] - scf.energy;
    let dipole_coefficients = (0..=order)
        .map(|n| {
            std::array::from_fn(|axis| {
                let mut value = geometry.nuclear_dipole[n][axis];
                if let Some(path) = &retained {
                    let operator = precision::retain(&geometry.dipoles[axis]);
                    for spin in 0..2 {
                        value -= precision::trace_product(&path.p[spin], &operator, n);
                    }
                    return value;
                }
                for k in 0..=n {
                    for spin in 0..2 {
                        value -= p[spin][k].frobenius_dot(&geometry.dipoles[axis][n - k]);
                    }
                }
                value
            })
        })
        .collect();
    let reference_fock_error = (0..2)
        .map(|s| evaluation.fock[s][0].max_abs_diff(&scf.fock[s]))
        .fold(0.0_f64, f64::max);
    let mut warnings = Vec::new();
    if let Some(polish) = polish {
        if polish.iterations > 0 {
            warnings.push(format!("zeroth-order orbital frame polished in {} steps from {:.6e} to {:.6e} Hartree; original reference energy/Fock diagnostics retained",polish.iterations,polish.initial_residual,polish.final_residual));
        }
    }
    let acceptance = if !integer_reference {
        config.acceptance()
    } else {
        config
            .tolerance
            .max(reference.orbital_residual)
            .max(reference.eigen_residual)
            .max(1e-14)
    };
    let active_spectrum = match (&fixed_population, &thermal_population) {
        (Some(d), _) => d.population_defects.as_slice(),
        (_, Some(d)) => d.density_defects.as_slice(),
        _ => projector_defects.as_slice(),
    };
    let converged = stationarity_defects
        .iter()
        .chain(active_spectrum)
        .chain(&metric_defects)
        .all(|x| *x <= acceptance)
        && particle_number_coefficients
            .iter()
            .flatten()
            .all(|x| x.abs() <= acceptance)
        && reference_energy_error.abs() <= acceptance
        && reference_fock_error <= acceptance
        && fixed_population
            .as_ref()
            .is_none_or(|d| d.maximum_defect() <= acceptance);
    if !converged {
        warnings.push(if thermal {
            "canonical Fermi Taylor diagnostics exceed the requested tolerance; inspect the raw density defects".into()
        } else if fixed {
            "moving-basis fixed-population diagnostics exceed the requested tolerance; inspect the raw coefficient defects".into()
        } else {
            "moving-basis response diagnostics exceed the larger of the requested tolerance and the reference residual; inspect individual coefficient defects".into()
        });
    }
    if order > 0 {
        warnings.push(format!(
            "orbital equations solved by {}; integration topology and reference AO coefficient subspace are fixed",
            if methods.is_empty() { "shared full-MO spectral kernel".into() } else { methods[0].to_string() }
        ));
    }
    let report = ResponseReport {
        value: NuclearTaylor {
            direction: direction.to_vec(), field_direction, energy_coefficients,
            energy_components: evaluation.energy, dipole_coefficients,
            density_coefficients: (0..=order).map(|n| [p[0][n].clone(),p[1][n].clone()]).collect(),
            density_corrections: (0..=order).map(|n|std::array::from_fn(|s|retained.as_ref().map_or_else(||Matrix::zeros(p[s][n].rows,p[s][n].cols),|path|path.p[s][n].low.clone()))).collect(),
            overlap_coefficients: geometry.overlap.clone(), projector_defects, stationarity_defects,
            orthonormal_fock_coefficients: (0..=order).map(|n| [actual_f[0][n].clone(), actual_f[1][n].clone()]).collect(),
            metric_defects, particle_number_coefficients, reference_energy_error, reference_fock_error, fixed_population, thermal_population,
            coefficient_equations: retained.as_ref().map(|path| path.coefficient_equations.clone()),
        },
        shape: vec![order+1], functional: functional.description.clone(), reference,
        units: "Hartree; coefficient n = directional energy derivative n / n!",
        model: if thermal {
            "static canonical Fermi-Dirac Mermin free energy at fixed electronic temperature; analytic moving Gaussian basis and Becke grid"
        } else if fixed {
            "static all-electron prescribed-population KS/HF; analytic moving Gaussian basis and Becke grid; isospectral orbital response; complete connected energy derivatives"
        } else {
            "static all-electron integer-occupation KS/HF; analytic moving Gaussian basis and Becke grid; adiabatic semilocal XC"
        },
        axis_convention: "[Taylor order]; R_A(t)=R_A(0)+t direction[A], E(t)=E_reference+t field_direction; Bohr and atomic electric field units",
        converged, residuals, iterations, warnings,
    };
    let path = frame::TransportedPath {
        geometry,
        b,
        d,
        f: actual_f,
        p,
        n_occ: solver.dims.n_occ,
        unrestricted: scf.unrestricted,
        energy: report.value.energy_coefficients.clone(),
        retained,
    };
    Ok((report, path))
}

/// Exact mixed nuclear/electric derivatives of both energy and dipole.
/// A dipole derivative of perturbation order n is an energy derivative of
/// total order n+1, with the additional electric index and opposite sign.
#[allow(clippy::too_many_arguments)]
pub fn mixed_molecular_response(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    directions: &[MolecularDirection],
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<MixedMolecularResponse>> {
    config.validate()?;
    if directions.iter().any(|d| d.nuclear.len() != mol.len()) {
        return Err(DftError::InvalidInput(
            "mixed molecular directions require one nuclear displacement per atom".into(),
        ));
    }
    let vectors: Vec<_> = directions.iter().map(MolecularDirection::vector).collect();
    let mut residuals = Vec::new();
    let mut iterations = Vec::new();
    let mut warnings = Vec::new();
    let mut converged = true;
    let mixed = super::mixed::polarized_mixed(3 * mol.len() + 3, &vectors, |d, n| {
        let direction = MolecularDirection {
            nuclear: d[..3 * mol.len()]
                .chunks_exact(3)
                .map(|v| [v[0], v[1], v[2]])
                .collect(),
            electric: [d[3 * mol.len()], d[3 * mol.len() + 1], d[3 * mol.len() + 2]],
        };
        let report = molecular_taylor(mol, mb, functional, grid, scf, opts, &direction, n, config)?;
        converged &= report.converged;
        residuals.extend(report.residuals);
        iterations.extend(report.iterations);
        warnings.extend(report.warnings);
        let mu = report.value.dipole_coefficients[n];
        Ok(vec![
            report.value.energy_coefficients[n],
            mu[0],
            mu[1],
            mu[2],
        ])
    })?;
    for i in 0..4 {
        if mixed.accumulation_roundoff[i] > config.tolerance * mixed.value[i].abs().max(1.0) {
            converged = false;
            warnings.push(format!(
                "mixed molecular component {i} exceeds the polarization roundoff scale"
            ));
        }
    }
    warnings.sort();
    warnings.dedup();
    Ok(ResponseReport {
        value:MixedMolecularResponse {directions:directions.to_vec(),energy_derivative:mixed.value[0],
            dipole_derivative:std::array::from_fn(|a|mixed.value[a+1]),energy_accumulation_roundoff:mixed.accumulation_roundoff[0],
            dipole_accumulation_roundoff:std::array::from_fn(|a|mixed.accumulation_roundoff[a+1]),directional_evaluations:mixed.directional_evaluations},
        shape:vec![4],functional:functional.description.clone(),reference:super::diagnostics::reference_diagnostics(scf)?,
        units:"atomic units; unnormalized mixed derivatives of energy and dipole",
        model:"static all-electron integer-occupation KS/HF; analytic moving basis, Becke grid and electric field; polarization identity",
        axis_convention:"energy scalar then Cartesian dipole vector; each direction combines per-atom Bohr displacements and atomic electric field components",
        converged,residuals,iterations,warnings,
    })
}

/// Static directional APT, Raman and hyper-Raman tensors from analytic
/// mixed dipole response. No displaced polarizability calculation occurs.
/// This low-level entry differentiates the supplied reference functional.
/// Complete post-SCF target and PT2 contributions use
/// `driver::run("static-spectroscopy", ...)` and the same tensor assembly.
#[allow(clippy::too_many_arguments)]
pub fn static_spectroscopic_derivative(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    nuclear_direction: &[[f64; 3]],
    electric_order: usize,
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<StaticSpectroscopicDerivative>> {
    static_spectroscopic_from_mixed(
        mol.len(),
        nuclear_direction,
        electric_order,
        config,
        "static all-electron integer-occupation KS/HF; analytic APT/Raman/hyper-Raman mixed response; adiabatic semilocal XC",
        |directions| mixed_molecular_response(mol, mb, functional, grid, scf, opts, directions, config),
    )
}

/// Contract a mixed dipole-response provider without choosing its energy
/// composition. The ordinary parent and post-SCF correlated providers share
/// every ordered tensor slot, roundoff check and permutation diagnostic.
pub(crate) fn static_spectroscopic_from_mixed(
    natoms: usize,
    nuclear_direction: &[[f64; 3]],
    electric_order: usize,
    config: &ResponseSolverConfig,
    model: &'static str,
    mut mixed: impl FnMut(&[MolecularDirection]) -> Result<ResponseReport<MixedMolecularResponse>>,
) -> Result<ResponseReport<StaticSpectroscopicDerivative>> {
    config.validate()?;
    if !(1..=3).contains(&electric_order)
        || nuclear_direction.len() != natoms
        || nuclear_direction.iter().flatten().any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput("spectroscopic derivatives require finite per-atom displacements and electric_order=1, 2 or 3".into()));
    }
    let nuclear = MolecularDirection::nuclear(nuclear_direction.to_vec());
    let size = 3usize.pow(electric_order as u32);
    let mut tensor = vec![0.0; size];
    let mut roundoff = vec![0.0; size];
    let mut residuals = Vec::new();
    let mut iterations = Vec::new();
    let mut warnings = Vec::new();
    let mut converged = true;
    let mut evaluations = 0;
    let mut metadata = None;
    for b in 0..if electric_order == 1 { 1 } else { 3 } {
        // Keep every ordered field pair: permutation defects must compare
        // independently evaluated slots, rather than copied tensor entries.
        for c in 0..if electric_order == 3 { 3 } else { 1 } {
            let mut directions = vec![nuclear.clone()];
            if electric_order >= 2 {
                let mut field = [0.0; 3];
                field[b] = 1.0;
                directions.push(MolecularDirection::electric(natoms, field));
            }
            if electric_order == 3 {
                let mut field = [0.0; 3];
                field[c] = 1.0;
                directions.push(MolecularDirection::electric(natoms, field));
            }
            let report = mixed(&directions)?;
            if metadata.is_none() {
                metadata = Some((report.functional.clone(), report.reference.clone()));
            }
            converged &= report.converged;
            residuals.extend(report.residuals);
            iterations.extend(report.iterations);
            warnings.extend(report.warnings);
            evaluations += report.value.directional_evaluations;
            for a in 0..3 {
                let ix = match electric_order {
                    1 => a,
                    2 => 3 * a + b,
                    _ => 9 * a + 3 * b + c,
                };
                tensor[ix] = report.value.dipole_derivative[a];
                roundoff[ix] = report.value.dipole_accumulation_roundoff[a];
            }
        }
    }
    let mut permutation_defect = 0.0_f64;
    let mut permutation_comparisons = 0;
    if electric_order >= 2 {
        for a in 0..3 {
            for b in 0..3 {
                if electric_order == 2 {
                    if a != b {
                        permutation_comparisons += 1;
                    }
                    permutation_defect =
                        permutation_defect.max((tensor[3 * a + b] - tensor[3 * b + a]).abs());
                } else {
                    for c in 0..3 {
                        permutation_comparisons += usize::from(a != b) + usize::from(b != c);
                        permutation_defect = permutation_defect
                            .max((tensor[9 * a + 3 * b + c] - tensor[9 * b + 3 * a + c]).abs())
                            .max((tensor[9 * a + 3 * b + c] - tensor[9 * a + 3 * c + b]).abs());
                    }
                }
            }
        }
    }
    let scale = tensor.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    if permutation_defect > config.tolerance * scale {
        converged = false;
        warnings.push(
            "static electric-index permutation symmetry exceeds the requested relative tolerance"
                .into(),
        );
    }
    warnings.sort();
    warnings.dedup();
    let (functional, reference) = metadata.ok_or_else(|| {
        DftError::InvalidInput(
            "a spectroscopic response requires at least one mixed tensor slot".into(),
        )
    })?;
    Ok(ResponseReport {
        value:StaticSpectroscopicDerivative {nuclear_direction:nuclear_direction.to_vec(),electric_order,tensor,
            accumulation_roundoff:roundoff,permutation_defect,permutation_comparisons,directional_evaluations:evaluations},
        shape:vec![3;electric_order],functional,reference,
        units:"atomic units; dipole / electric_field^(electric_order-1) per supplied nuclear path",
        model,
        axis_convention:"row-major Cartesian electric indices; output dipole index first; one nuclear directional derivative",
        converged,residuals,iterations,warnings,
    })
}

/// Cubic, quartic and higher mixed force constants by the exact polarization
/// identity. Each direction is an atom-by-Cartesian displacement in Bohr.
#[allow(clippy::too_many_arguments)]
pub fn mixed_nuclear_response(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    directions: &[Vec<[f64; 3]>],
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<MixedNuclearResponse>> {
    if directions.iter().any(|d| d.len() != mol.len()) {
        return Err(DftError::InvalidInput(
            "each mixed nuclear direction must contain one displacement per atom".into(),
        ));
    }
    let vectors: Vec<Vec<f64>> = directions
        .iter()
        .map(|d| d.iter().flatten().copied().collect())
        .collect();
    let mut residuals = Vec::new();
    let mut iterations = Vec::new();
    let mut warnings = Vec::new();
    let mut converged = true;
    let mixed = super::mixed::polarized_mixed(3 * mol.len(), &vectors, |d, n| {
        let direction: Vec<_> = d.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        let report = nuclear_taylor(mol, mb, functional, grid, scf, opts, &direction, n, config)?;
        converged &= report.converged;
        residuals.extend(report.residuals);
        iterations.extend(report.iterations);
        warnings.extend(report.warnings);
        Ok(vec![report.value.energy_coefficients[n]])
    })?;
    if mixed.accumulation_roundoff[0] > config.tolerance * mixed.value[0].abs().max(1.0) {
        warnings.push(
            "mixed nuclear polarization accumulation exceeds the requested roundoff scale".into(),
        );
        converged = false;
    }
    warnings.sort();
    warnings.dedup();
    Ok(ResponseReport {
        value: MixedNuclearResponse { directions:directions.to_vec(), energy_derivative:mixed.value[0],
            accumulation_roundoff:mixed.accumulation_roundoff[0], directional_evaluations:mixed.directional_evaluations },
        shape:vec![], functional:functional.description.clone(), reference:super::diagnostics::reference_diagnostics(scf)?,
        units:"Hartree; unnormalized mixed energy derivative",
        model:"static all-electron integer-occupation KS/HF; analytic moving basis and grid; polarization identity",
        axis_convention:"one nuclear directional derivative per input direction, directions in Bohr",
        converged,residuals,iterations,warnings,
    })
}

#[cfg(test)]
mod metric_transport_tests {
    use super::*;
    use crate::acceptance::{components, stencils};

    fn overlap() -> Vec<Matrix> {
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.31, 0.17, 0.17, -0.23],
            [-0.09, 0.21, 0.21, 0.13],
            [0.12, -0.08, -0.08, 0.19],
            [-0.11, 0.16, 0.16, 0.07],
        ]
        .iter()
        .map(|v| {
            let mut m = Matrix::zeros(2, 2);
            m.as_mut_slice().copy_from_slice(v);
            m
        })
        .collect()
    }

    // Independent closed form for the positive inverse square root of a
    // two-by-two SPD matrix; no Taylor recurrence enters this oracle.
    fn dense(c: &Matrix, overlap: &[Matrix], t: f64) -> Matrix {
        let mut s = Matrix::zeros(2, 2);
        for (n, m) in overlap.iter().enumerate() {
            add(&mut s, m, t.powi(n as i32));
        }
        let m = c.transpose().matmul_seq(&s).matmul_seq(c);
        let determinant = (m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)]).sqrt();
        let trace = (m[(0, 0)] + m[(1, 1)] + 2.0 * determinant).sqrt();
        let denominator =
            (m[(0, 0)] + determinant) * (m[(1, 1)] + determinant) - m[(0, 1)] * m[(1, 0)];
        let mut inverse = Matrix::zeros(2, 2);
        inverse[(0, 0)] = trace * (m[(1, 1)] + determinant) / denominator;
        inverse[(1, 1)] = trace * (m[(0, 0)] + determinant) / denominator;
        inverse[(0, 1)] = -trace * m[(0, 1)] / denominator;
        inverse[(1, 0)] = -trace * m[(1, 0)] / denominator;
        c.matmul_seq(&inverse)
    }

    fn central(value: impl Fn(f64) -> f64, n: usize, h: f64) -> f64 {
        match n {
            1 => stencils::Stencil::Second.first(value(h), value(-h), 0.0, 0.0, h),
            2 => (value(h) - 2.0 * value(0.0) + value(-h)) / h.powi(2),
            3 => {
                (value(2.0 * h) - 2.0 * value(h) + 2.0 * value(-h) - value(-2.0 * h))
                    / (2.0 * h.powi(3))
            }
            _ => {
                (value(2.0 * h) - 4.0 * value(h) + 6.0 * value(0.0) - 4.0 * value(-h)
                    + value(-2.0 * h))
                    / h.powi(4)
            }
        }
    }

    #[test]
    fn noncommuting_transport_retains_constant_metric_and_fits_every_coefficient() {
        let mut c = Matrix::zeros(2, 2);
        c[(0, 0)] = 0.8 * (1.0 + 1e-10);
        c[(1, 0)] = 0.6 * (1.0 + 1e-10);
        c[(0, 1)] = -0.6 * (1.0 - 2e-10);
        c[(1, 1)] = 0.8 * (1.0 - 2e-10);
        let s = overlap();
        let b = metric_transport(&c, &s, 4);
        let bt: Vec<_> = b.iter().map(Matrix::transpose).collect();
        let metric = matrix_product(&matrix_product(&bt, &s, 4), &b, 4);
        for (n, m) in metric.iter().enumerate() {
            for i in 0..2 {
                for j in 0..2 {
                    let expected = if n == 0 && i == j { 1.0 } else { 0.0 };
                    assert!((m[(i, j)] - expected).abs() < 2e-14, "metric {n} {i} {j}");
                }
            }
        }
        assert!(b[0].max_abs_diff(&dense(&c, &s, 0.0)) < 5e-16);
        let names: Vec<_> = (1..=4)
            .flat_map(|n| {
                (0..2).flat_map(move |i| {
                    (0..2).map(move |j| {
                        format!("noncommuting metric transport derivative={n} row={i} col={j}")
                    })
                })
            })
            .collect();
        let stages = [
            ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Second),
            ([0.04, 0.03, 0.02, 0.015], stencils::Stencil::Second),
            ([0.12, 0.09, 0.06, 0.045], stencils::Stencil::Second),
            ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Fourth),
        ];
        let mut failures = Vec::new();
        stencils::fit_components_staged(
            &names,
            &stages,
            |steps, stencil| {
                let mut errors = Vec::new();
                for n in 1..=4 {
                    for i in 0..2 {
                        for j in 0..2 {
                            let exact = b[n][(i, j)] * (1..=n).product::<usize>() as f64;
                            errors.push(std::array::from_fn(|k| {
                                let h = steps[k];
                                let coarse = central(|t| dense(&c, &s, t)[(i, j)], n, h);
                                let estimate = if stencil == stencils::Stencil::Second {
                                    coarse
                                } else {
                                    (4.0 * central(|t| dense(&c, &s, t)[(i, j)], n, h / 2.0)
                                        - coarse)
                                        / 3.0
                                };
                                (estimate - exact).abs()
                            }));
                        }
                    }
                }
                errors
            },
            &mut failures,
        );
        components::assert_no_failures(
            "all noncommuting metric derivatives through order four",
            &failures,
        );
    }

    #[test]
    fn transported_operator_series_fits_every_ordered_entry_through_fourth_order() {
        let c = Matrix::from_row_major(2, 2, vec![0.8, -0.6, 0.6, 0.8]);
        let s = overlap();
        // Deliberately nonsymmetric operators certify both ordered entries.
        let f: Vec<_> = [
            [-1.7, 0.31, -0.27, 0.83],
            [0.23, -0.41, 0.19, -0.37],
            [-0.29, 0.13, 0.43, 0.17],
            [0.11, -0.21, 0.37, -0.13],
            [0.17, 0.29, -0.19, 0.23],
        ]
        .into_iter()
        .map(|v| Matrix::from_row_major(2, 2, v.to_vec()))
        .collect();
        let b = metric_transport(&c, &s, 4);
        let analytic = matrix_congruence(&b, &f, 4);
        let names: Vec<_> = (1..=4)
            .flat_map(|n| {
                (0..2).flat_map(move |i| {
                    (0..2).map(move |j| {
                        format!("transported operator derivative={n} row={i} col={j}")
                    })
                })
            })
            .collect();
        let stages = [
            ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Second),
            ([0.04, 0.03, 0.02, 0.015], stencils::Stencil::Second),
            ([0.12, 0.09, 0.06, 0.045], stencils::Stencil::Second),
            ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Fourth),
        ];
        let mut failures = Vec::new();
        stencils::fit_components_staged(
            &names,
            &stages,
            |steps, stencil| {
                let mut errors = Vec::new();
                for (n, coefficient) in analytic.iter().enumerate().take(5).skip(1) {
                    for i in 0..2 {
                        for j in 0..2 {
                            let exact = coefficient[(i, j)] * (1..=n).product::<usize>() as f64;
                            errors.push(std::array::from_fn(|k| {
                                let value = |t: f64| {
                                    let bt = dense(&c, &s, t);
                                    let mut ft = Matrix::zeros(2, 2);
                                    for (power, term) in f.iter().enumerate() {
                                        add(&mut ft, term, t.powi(power as i32));
                                    }
                                    bt.transpose().matmul_seq(&ft).matmul_seq(&bt)[(i, j)]
                                };
                                let h = steps[k];
                                let coarse = central(value, n, h);
                                let estimate = if stencil == stencils::Stencil::Second {
                                    coarse
                                } else {
                                    (4.0 * central(value, n, h / 2.0) - coarse) / 3.0
                                };
                                (estimate - exact).abs()
                            }));
                        }
                    }
                }
                errors
            },
            &mut failures,
        );
        components::assert_no_failures("all ordered transported operator derivatives", &failures);
    }
}
