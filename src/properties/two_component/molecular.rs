// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::grid::BalancedGrid;
use super::matrix::{identity, validate_hermitian};
use super::{dkh2_taylor, matrix_zora_taylor, spin_scalar_operator, x2c_taylor};
use super::{finish_embedding, RelativisticSpin, TwoComponentTaylor};
use crate::basis::MolBasis;
use crate::dfpt::orbitals::{adjoint, matmul};
use crate::grid::MolecularGrid;
use crate::integrals::analytic::GaussianKernel;
use crate::integrals::analytic_basis::{AnalyticShell, IntegralBlock};
use crate::integrals::dirac::{
    balanced_cartesian_moments, balanced_electric_moments, electronic_current_exchange,
    electronic_screened_exchange, london_electric_dipoles, london_magnetic_balance,
    restricted_kinetic_balance, BalancedPairCache, DiracCoulombFields, GaussianNucleus,
    LondonCoulombTaylor, LondonCurrentTaylor,
};
use crate::integrals::dirac::{nuclear_magnetic_dipole_vertices, GaussianMagnetization};
use crate::linalg::Matrix;
use crate::nlc::Vv10Params;
use crate::periodic::occupations::OccupationScheme;
use crate::periodic::spinor_scf::{
    spin_diagonal, SpinorDensityKernel, SpinorEnergyBackend, SpinorFields, SpinorScfReport,
};
use crate::periodic::BlochMatrix;
use crate::properties::no_pair::stationary::{DiracInteraction, NoPairSpace};
use crate::response::complex::Complex;
use crate::response::krylov::ResponseSolverConfig;
use crate::response::spinor::{spinor_operator_response_with_kernel, SpinorOperatorResponse};
use crate::xc::{Family, ResolvedFunctional};
use crate::{DftError, Result};

mod core_source;
mod magnetic;
pub use core_source::DiracCoreSourceResponse;

/// The one-electron relativistic Hamiltonian and matching picture change.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TwoComponentMethod {
    X2c,
    Dkh2,
    MatrixZora,
}

impl TwoComponentMethod {
    pub fn name(self) -> &'static str {
        match self {
            Self::X2c => "x2c",
            Self::Dkh2 => "dkh2",
            Self::MatrixZora => "matrix-zora",
        }
    }
}

/// Common fixed-nucleus molecular controls in atomic units.
#[derive(Clone, Debug)]
pub struct MolecularRelativisticOptions {
    pub method: TwoComponentMethod,
    pub spin: RelativisticSpin,
    pub speed_of_light: f64,
    pub metric_cutoff: f64,
    pub xc_angular_tolerance: f64,
    /// In-core budget shared by the exact full and range-separated integrals.
    pub pair_cache_mb: usize,
    /// Clamped uniform electric perturbation in the field-free picture.
    pub electric_field: [f64; 3],
    pub electric_origin: [f64; 3],
    /// Constant empirical contribution such as the required CHG tail.
    pub dispersion_energy: Option<f64>,
}

#[derive(Clone, Copy)]
pub(crate) struct UniformMagneticBasis {
    pub field: [f64; 3],
    pub phase_order: usize,
}

pub(crate) struct NuclearMagneticFields<'a> {
    pub moments: &'a [[f64; 3]],
    pub radii: &'a [f64],
}

enum PairCache {
    Kinetic(BalancedPairCache),
    Magnetic(LondonCoulombTaylor),
}

impl PairCache {
    fn evaluate(&self, density: &BlochMatrix) -> Result<DiracCoulombFields> {
        match self {
            Self::Kinetic(cache) => cache.evaluate(density),
            Self::Magnetic(cache) => cache.fields(0, density),
        }
    }
}

/// A decoupled chart with a self-consistent positive-sector density.
/// Coulomb, full/range-separated exchange and the density descriptors
/// are evaluated on the same lifted balanced AO density.
pub struct MolecularTwoComponent<'a> {
    shells: Vec<AnalyticShell<f64>>,
    options: MolecularRelativisticOptions,
    map: TwoComponentTaylor,
    dirac_spectral_gap: Option<f64>,
    dirac_core: Option<[BlochMatrix; 2]>,
    dirac_interaction: DiracInteraction,
    map_adjoint: BlochMatrix,
    operators: [BlochMatrix; 3],
    classical_dipole: [f64; 3],
    additional_electric: Option<(BlochMatrix, f64)>,
    nuclear_repulsion: f64,
    initial: [BlochMatrix; 1],
    metric: [BlochMatrix; 1],
    functional: &'a ResolvedFunctional,
    semilocal: Option<BalancedGrid<'a>>,
    nonlocal: Option<(BalancedGrid<'a>, Vv10Params)>,
    coulomb_cache: Option<PairCache>,
    range_cache: Option<PairCache>,
    current_cache: Option<LondonCurrentTaylor>,
    magnetic: Option<UniformMagneticBasis>,
    electrons: usize,
}

struct PairFields {
    operator: BlochMatrix,
    coulomb_energies: [f64; 3],
    coulomb_defect: f64,
    current_energy: f64,
}

fn from_block(block: IntegralBlock<f64, 2>) -> Matrix {
    Matrix::from_row_major(block.shape[0], block.shape[1], block.values)
}

impl<'a> MolecularTwoComponent<'a> {
    pub fn new(
        basis: &MolBasis,
        nuclei: &[GaussianNucleus<f64>],
        electrons: usize,
        functional: &'a ResolvedFunctional,
        semilocal_grid: Option<&'a MolecularGrid>,
        vv10_grid: Option<&'a MolecularGrid>,
        options: MolecularRelativisticOptions,
    ) -> Result<Self> {
        Self::build(
            basis,
            nuclei,
            electrons,
            functional,
            semilocal_grid,
            vv10_grid,
            options,
            false,
            DiracInteraction::Coulomb,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn in_dirac_sector(
        basis: &MolBasis,
        nuclei: &[GaussianNucleus<f64>],
        electrons: usize,
        functional: &'a ResolvedFunctional,
        semilocal_grid: Option<&'a MolecularGrid>,
        vv10_grid: Option<&'a MolecularGrid>,
        options: MolecularRelativisticOptions,
        dirac_interaction: DiracInteraction,
        magnetic: Option<UniformMagneticBasis>,
        nuclear_magnetic: Option<NuclearMagneticFields<'_>>,
    ) -> Result<Self> {
        Self::build(
            basis,
            nuclei,
            electrons,
            functional,
            semilocal_grid,
            vv10_grid,
            options,
            true,
            dirac_interaction,
            magnetic,
            nuclear_magnetic,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        basis: &MolBasis,
        nuclei: &[GaussianNucleus<f64>],
        electrons: usize,
        functional: &'a ResolvedFunctional,
        semilocal_grid: Option<&'a MolecularGrid>,
        vv10_grid: Option<&'a MolecularGrid>,
        options: MolecularRelativisticOptions,
        dirac_sector: bool,
        dirac_interaction: DiracInteraction,
        magnetic: Option<UniformMagneticBasis>,
        nuclear_magnetic: Option<NuclearMagneticFields<'_>>,
    ) -> Result<Self> {
        if !basis.ecps.is_empty()
            || electrons == 0
            || electrons > 2 * basis.nao_sph
            || electrons % 2 != 0
        {
            return Err(DftError::Unsupported {
                feature: "molecular two-component relativistic reference".into(),
                missing: "an isolated all-electron, even-integer-electron insulating reference"
                    .into(),
            });
        }
        if options
            .electric_field
            .iter()
            .chain(&options.electric_origin)
            .any(|v| !v.is_finite())
            || !options.xc_angular_tolerance.is_finite()
            || options.xc_angular_tolerance <= 0.0
            || options.dispersion_energy.is_some_and(|e| !e.is_finite())
        {
            return Err(DftError::InvalidInput(
                "two-component electric and XC controls must be finite with positive XC tolerance"
                    .into(),
            ));
        }
        let speed = options.speed_of_light;
        let shells = basis
            .shells
            .iter()
            .map(|s| AnalyticShell::from_placed(s, s.center.to_array()))
            .collect::<Result<Vec<_>>>()?;
        let balanced = match magnetic {
            Some(m) => london_magnetic_balance(&shells, nuclei, speed, m.field, 0, m.phase_order)?,
            None => restricted_kinetic_balance(&shells, nuclei, speed, 0)?,
        };
        let potential = if !dirac_sector && options.method == TwoComponentMethod::Dkh2 {
            let [re, im] = balanced.nuclear_potential()?;
            Some([BlochMatrix::from_pair((from_block(re), from_block(im)))])
        } else {
            None
        };
        let mut h = [BlochMatrix::from_pair((
            from_block(balanced.hamiltonian_real),
            from_block(balanced.hamiltonian_imaginary),
        ))];
        if let Some(fields) = nuclear_magnetic {
            let uniform = magnetic.map_or([0.0; 3], |m| m.field);
            let phase = magnetic.map_or(0, |m| m.phase_order);
            for ((atom, &radius), &moments) in nuclei.iter().zip(fields.radii).zip(fields.moments) {
                if moments == [0.0; 3] {
                    continue;
                }
                let integrals = nuclear_magnetic_dipole_vertices(
                    &shells,
                    GaussianMagnetization {
                        center: atom.center,
                        rms_radius: radius,
                    },
                    speed,
                    uniform,
                    0,
                    phase,
                )?;
                for ([re, im], moment) in integrals.into_iter().zip(moments) {
                    let op = BlochMatrix::from_pair((from_block(re), from_block(im)));
                    h[0].add_scaled(&op, moment);
                }
            }
        }
        let s = [BlochMatrix::from_pair((
            from_block(balanced.overlap),
            from_block(balanced.overlap_imaginary),
        ))];
        let cutoff = options.metric_cutoff;
        let (map, dirac_spectral_gap) = if dirac_sector {
            // The variational space is the positive sector of the full Dirac
            // core. These columns are its four-component core eigenvectors.
            let space = NoPairSpace::new(&h[0], &s[0], -speed.powi(2), cutoff)?;
            let core = space.project_operator(&h[0])?;
            (
                finish_embedding(vec![space.basis().clone()], vec![core], &h, &s)?,
                Some(space.spectral_gap),
            )
        } else {
            (
                match options.method {
                    TwoComponentMethod::X2c => {
                        x2c_taylor(&h, &s, 0, -speed.powi(2), cutoff, options.spin)?
                    }
                    TwoComponentMethod::Dkh2 => dkh2_taylor(
                        &h,
                        &s,
                        &potential.expect("DKH2 nuclear operator"),
                        0,
                        -speed.powi(2),
                        cutoff,
                        options.spin,
                    )?,
                    TwoComponentMethod::MatrixZora => {
                        matrix_zora_taylor(&h, &s, 0, cutoff, options.spin)?
                    }
                },
                None,
            )
        };
        let map_adjoint = adjoint(&map.embedding()[0])?;
        let dirac_core = dirac_sector.then(|| [h[0].clone(), s[0].clone()]);
        let mut initial = [map.hamiltonian[0].clone()];
        let full_operators = match magnetic {
            Some(m) => london_electric_dipoles(
                &shells,
                speed,
                m.field,
                options.electric_origin,
                m.phase_order,
            )?,
            None => balanced_electric_moments(
                &shells,
                speed,
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                options.electric_origin,
            )?,
        };
        let spin_orbit = options.spin == RelativisticSpin::SpinOrbit;
        let mut operators = std::array::from_fn(|_| identity(initial[0].rows()));
        let mut classical_dipole = [0.0; 3];
        for (a, full) in full_operators.into_iter().enumerate() {
            let full = if spin_orbit {
                full
            } else {
                spin_scalar_operator(&full)?
            };
            operators[a] = map.transform(&[full])?.remove(0);
            initial[0].add_scaled(&operators[a], options.electric_field[a]);
            classical_dipole[a] = nuclei
                .iter()
                .map(|nucleus| nucleus.charge * (nucleus.center[a] - options.electric_origin[a]))
                .sum();
        }
        let build_grid = |grid, family| {
            let base = BalancedGrid::new(basis, grid, speed, spin_orbit, family);
            if let Some(m) = magnetic {
                base.with_uniform_magnetic_basis(basis, m.field)
            } else {
                base
            }
        };
        let semilocal = if functional.parts.is_empty() {
            None
        } else {
            let grid = semilocal_grid.ok_or_else(|| DftError::Unsupported {
                feature: "picture-changed semilocal XC".into(),
                missing: "a molecular integration grid".into(),
            })?;
            Some(build_grid(grid, functional.family))
        };
        let nonlocal = if let Some(parameters) = functional.required_vv10() {
            let grid = vv10_grid.ok_or_else(|| DftError::Unsupported {
                feature: "picture-changed self-consistent VV10".into(),
                missing: "a molecular nonlocal correlation grid".into(),
            })?;
            Some((build_grid(grid, Family::Gga), parameters))
        } else {
            None
        };
        if functional.required_pair_dispersion().is_some() && options.dispersion_energy.is_none() {
            return Err(DftError::Unsupported {
                feature: "defining dispersion correction".into(),
                missing: "the empirical molecular dispersion energy".into(),
            });
        }
        let london_current = magnetic.zip(dirac_interaction.current());
        let cache_count =
            1 + usize::from(functional.hybrid.clr != 0.0) + usize::from(london_current.is_some());
        let cache_mb = options.pair_cache_mb / cache_count;
        let current_cache = london_current
            .map(|(m, interaction)| {
                LondonCurrentTaylor::new(
                    &shells,
                    speed,
                    interaction,
                    m.field,
                    [0.0; 3],
                    0,
                    m.phase_order,
                    cache_mb,
                )
            })
            .transpose()?;
        let cache = |kernel| -> Result<Option<PairCache>> {
            if let Some(m) = magnetic {
                Ok(Some(PairCache::Magnetic(LondonCoulombTaylor::new(
                    &shells,
                    speed,
                    kernel,
                    m.field,
                    [0.0; 3],
                    0,
                    m.phase_order,
                    cache_mb,
                )?)))
            } else {
                Ok(
                    BalancedPairCache::new(&shells, speed, kernel, spin_orbit, cache_mb)?
                        .map(PairCache::Kinetic),
                )
            }
        };
        let coulomb_cache = cache(GaussianKernel::Coulomb)?;
        let range_cache = if functional.hybrid.clr != 0.0 {
            cache(GaussianKernel::LongRange {
                omega: functional.hybrid.omega,
            })?
        } else {
            None
        };
        Ok(Self {
            shells,
            options,
            metric: [identity(initial[0].rows())],
            map,
            dirac_spectral_gap,
            dirac_core,
            dirac_interaction,
            map_adjoint,
            operators,
            classical_dipole,
            additional_electric: None,
            nuclear_repulsion: balanced.nuclear_repulsion,
            initial,
            functional,
            semilocal,
            nonlocal,
            coulomb_cache,
            range_cache,
            current_cache,
            magnetic,
            electrons,
        })
    }

    pub fn transformation(&self) -> &TwoComponentTaylor {
        &self.map
    }

    pub(crate) fn dirac_spectral_gap(&self) -> Option<f64> {
        self.dirac_spectral_gap
    }

    fn current_fields(&self, density: &BlochMatrix) -> Result<Option<DiracCoulombFields>> {
        if let Some(cache) = &self.current_cache {
            return cache.fields(0, density).map(Some);
        }
        self.dirac_interaction
            .current()
            .map(|current| {
                electronic_current_exchange(
                    &self.shells,
                    density,
                    self.options.speed_of_light,
                    current,
                )
            })
            .transpose()
    }

    pub(crate) fn stationary_current_fields(
        &self,
        density: &BlochMatrix,
    ) -> Result<Option<DiracCoulombFields>> {
        if self.dirac_interaction == DiracInteraction::Coulomb {
            return Ok(None);
        }
        self.current_fields(&self.map.lift_density(&self.reference_density(density)?)?)
    }

    pub fn dipole(&self, density: &BlochMatrix) -> Result<[f64; 3]> {
        validate_hermitian(std::slice::from_ref(density), self.initial[0].rows(), 0)?;
        Ok(std::array::from_fn(|a| {
            self.classical_dipole[a] - density.trace_product(&self.operators[a])
        }))
    }

    pub(crate) fn dipole_operators(&self) -> &[BlochMatrix; 3] {
        &self.operators
    }

    /// Project traceless scalar electric modes in the same chart as the Fock.
    pub(crate) fn electric_modes(
        &self,
        powers: &[[usize; 3]],
        weights: &[Vec<f64>],
        origin: [f64; 3],
    ) -> Result<Vec<BlochMatrix>> {
        if self.magnetic.is_some() {
            return Err(DftError::Unsupported {
                feature: "electric multipoles at a four-component magnetic reference".into(),
                missing: "the corresponding Coulomb multipole operators on London-RMB spinors"
                    .into(),
            });
        }
        if weights.iter().any(|mode| mode.len() != powers.len()) {
            return Err(DftError::InvalidInput(
                "electric mode powers and weights differ".into(),
            ));
        }
        let cartesian =
            balanced_cartesian_moments(&self.shells, self.options.speed_of_light, powers, origin)?;
        let n = self.map.embedding()[0].rows();
        weights
            .iter()
            .map(|weights| {
                let mut full = BlochMatrix::zeros(n, n);
                for (operator, &weight) in cartesian.iter().zip(weights) {
                    if weight != 0.0 {
                        full.add_scaled(operator, weight);
                    }
                }
                if self.options.spin == RelativisticSpin::Scalar {
                    full = spin_scalar_operator(&full)?;
                }
                self.project(&full)
            })
            .collect()
    }

    /// Add a clamped electrostatic source to the physical Fock and energy.
    /// The field-free embedded sector and the two-electron kernel stay fixed.
    pub(crate) fn with_electric_source(
        mut self,
        operator: BlochMatrix,
        nuclear_energy: f64,
    ) -> Result<Self> {
        validate_hermitian(std::slice::from_ref(&operator), self.initial[0].rows(), 0)?;
        if !nuclear_energy.is_finite() {
            return Err(DftError::InvalidInput(
                "nonfinite nuclear electric source energy".into(),
            ));
        }
        self.initial[0].add_scaled(&operator, 1.0);
        self.additional_electric = Some((operator, nuclear_energy));
        Ok(self)
    }

    /// Stationary complex electric response with the same pair and XC
    /// functional as the SCF. Susceptibility of +r is minus polarizability.
    pub fn electric_response(
        &self,
        state: &SpinorScfReport,
        frequencies: &[Complex],
        solver: &ResponseSolverConfig,
    ) -> Result<Vec<SpinorOperatorResponse>> {
        self.operator_response(state, &self.operators, frequencies, solver)
    }

    pub(crate) fn operator_response(
        &self,
        state: &SpinorScfReport,
        operators: &[BlochMatrix],
        frequencies: &[Complex],
        solver: &ResponseSolverConfig,
    ) -> Result<Vec<SpinorOperatorResponse>> {
        let [band] = state.bands.as_slice() else {
            return Err(DftError::InvalidInput(
                "two-component molecular response needs one stationary orbital sector".into(),
            ));
        };
        if !state.converged || self.electrons >= band.energies.len() {
            return Err(DftError::Unsupported {
                feature: "two-component molecular electric response".into(),
                missing: "a converged reference with occupied and virtual spinors".into(),
            });
        }
        let kernel = self.orbital_kernel(&state.density)?;
        spinor_operator_response_with_kernel(
            &band.coefficients,
            &band.energies,
            self.electrons,
            &|density| {
                // Complexify the real-linear Hermitian density kernel.
                let opposite = adjoint(density)?;
                let mut re = density.clone();
                re.add_scaled(&opposite, 1.0);
                re.scale(0.5);
                let mut im = density.clone();
                im.add_scaled(&opposite, -1.0);
                let mut im = BlochMatrix::from_pair((im.im, im.re));
                for x in im.im.as_mut_slice() {
                    *x = -*x;
                }
                im.scale(0.5);
                let mut response = kernel.apply(&[re])?.remove(0);
                if im
                    .re
                    .as_slice()
                    .iter()
                    .chain(im.im.as_slice())
                    .any(|v| *v != 0.0)
                {
                    let other = kernel.apply(&[im])?.remove(0);
                    let mut other = BlochMatrix::from_pair((other.im, other.re));
                    for x in other.re.as_mut_slice() {
                        *x = -*x;
                    }
                    response.add_scaled(&other, 1.0);
                }
                Ok(response)
            },
            operators,
            frequencies,
            solver,
        )
    }

    fn project(&self, a: &BlochMatrix) -> Result<BlochMatrix> {
        let mut value = matmul(&self.map_adjoint, &matmul(a, &self.map.embedding()[0])?)?;
        value.make_hermitian();
        Ok(value)
    }

    fn reference_density(&self, density: &BlochMatrix) -> Result<BlochMatrix> {
        if self.options.spin == RelativisticSpin::SpinOrbit {
            return Ok(density.clone());
        }
        validate_hermitian(std::slice::from_ref(density), self.initial[0].rows(), 0)?;
        Ok(closed_shell_spin_trace(density))
    }

    fn pair(&self, density: &BlochMatrix) -> Result<PairFields> {
        let spin_orbit = self.options.spin == RelativisticSpin::SpinOrbit;
        let c = self.options.speed_of_light;
        let interaction = |cache: &Option<PairCache>, kernel| {
            if let Some(cache) = cache {
                cache.evaluate(density)
            } else {
                electronic_screened_exchange(&self.shells, density, c, kernel, spin_orbit)
            }
        };
        let full = interaction(&self.coulomb_cache, GaussianKernel::Coulomb)?;
        let mut fock = full.direct;
        let mut defect = full.hermiticity_defect;
        let weights = self.functional.hybrid;
        fock.add_scaled(&full.exchange, -weights.cx);
        let mut energy = [full.direct_energy, weights.cx * full.exchange_energy, 0.0];
        if weights.clr != 0.0 {
            let screened = interaction(
                &self.range_cache,
                GaussianKernel::LongRange {
                    omega: weights.omega,
                },
            )?;
            fock.add_scaled(&screened.exchange, -weights.clr);
            energy[2] = weights.clr * screened.exchange_energy;
            defect = defect.max(screened.hermiticity_defect);
        }
        // Add the determinant current interaction separately from the
        // Coulomb exchange fractions and the Coulomb XC functional.
        let current_energy = if let Some(current) = self.current_fields(density)? {
            fock.add_scaled(&current.direct, 1.0);
            fock.add_scaled(&current.exchange, -1.0);
            current.direct_energy + current.exchange_energy
        } else {
            0.0
        };
        Ok(PairFields {
            operator: fock,
            coulomb_energies: energy,
            coulomb_defect: defect,
            current_energy,
        })
    }

    fn full_density_kernel(
        &self,
        density: &BlochMatrix,
        direction: &BlochMatrix,
        angular: usize,
    ) -> Result<BlochMatrix> {
        let mut operator = self.pair(direction)?.operator;
        if let Some(grid) = &self.semilocal {
            let local = grid.semilocal_kernel(self.functional, density, direction, angular)?;
            operator.add_scaled(&local, 1.0);
        }
        if let Some((grid, parameters)) = &self.nonlocal {
            let (_, nonlocal) = grid.vv10(density, Some(direction), *parameters)?;
            operator.add_scaled(&nonlocal, 1.0);
        }
        Ok(operator)
    }
}

impl SpinorEnergyBackend for MolecularTwoComponent<'_> {
    fn overlap(&self) -> &[BlochMatrix] {
        &self.metric
    }
    fn initial_fock(&self) -> &[BlochMatrix] {
        &self.initial
    }
    fn electrons(&self) -> f64 {
        self.electrons as f64
    }
    fn validate_occupation(&self, occupation: OccupationScheme) -> Result<()> {
        if !matches!(occupation, OccupationScheme::Insulator) {
            return Err(DftError::Unsupported {
                feature: "molecular two-component statistical occupations".into(),
                missing: "an integer-electron insulating reference".into(),
            });
        }
        Ok(())
    }
    fn quadratic_density_energy(&self) -> bool {
        self.semilocal.is_none() && self.nonlocal.is_none()
    }
    fn evaluate(&self, density: &[BlochMatrix]) -> Result<SpinorFields> {
        let [density] = density else {
            return Err(DftError::InvalidInput(
                "two-component molecular SCF needs one trial density".into(),
            ));
        };
        let full_density = self.map.lift_density(&self.reference_density(density)?)?;
        let PairFields {
            operator: mut full_fock,
            coulomb_energies: [hartree, exchange, range_exchange],
            coulomb_defect: pair_defect,
            current_energy,
        } = self.pair(&full_density)?;
        let mut semilocal = 0.0;
        let mut nonlocal = 0.0;
        let mut parts = Vec::new();
        let mut diagnostics = None;
        if let Some(grid) = &self.semilocal {
            let (energy, operator, components, convergence) = grid.semilocal(
                self.functional,
                &full_density,
                self.options.xc_angular_tolerance,
            )?;
            semilocal = energy;
            parts = components;
            diagnostics = Some(convergence);
            full_fock.add_scaled(&operator, 1.0);
        }
        if let Some((grid, parameters)) = &self.nonlocal {
            let (energy, operator) = grid.vv10(&full_density, None, *parameters)?;
            nonlocal = energy;
            full_fock.add_scaled(&operator, 1.0);
        }
        if let Some(diagnostic) = &diagnostics {
            if !diagnostic.converged {
                return Err(DftError::LinearAlgebra(format!(
                    "two-component XC angular quadrature unresolved at order {} with energy defect {:.6e} and operator defect {:.6e} Hartree",
                    diagnostic.angular_order,
                    diagnostic.angular_energy_defect,
                    diagnostic.angular_operator_defect,
                )));
            }
        }
        let mut fock = self.initial[0].clone();
        let projected = self.project(&full_fock)?;
        fock.add_scaled(&self.reference_density(&projected)?, 1.0);
        let dipole = self.dipole(density)?;
        let mut field_energy = -dipole
            .iter()
            .zip(self.options.electric_field)
            .map(|(moment, field)| moment * field)
            .sum::<f64>();
        if let Some((operator, nuclear_energy)) = &self.additional_electric {
            field_energy += density.trace_product(operator) + nuclear_energy;
        }
        // Use the standard spinor SCF energy slots. The core combines the
        // rest-subtracted kinetic and the finite nuclear potential.
        let components = [
            self.nuclear_repulsion,
            density.trace_product(&self.map.hamiltonian[0]),
            nonlocal,
            self.options.dispersion_energy.unwrap_or(0.0),
            hartree,
            exchange,
            range_exchange,
            semilocal,
            field_energy,
            current_energy,
        ];
        Ok(SpinorFields {
            fock: vec![fock],
            energy: components.iter().sum(),
            energy_components: components,
            xc_component_energies: parts,
            nuclear_soc_energy: 0.0,
            operator_converged: true,
            coulomb_operator_defect: pair_defect,
            xc_grid_number_defect: 0.0,
            xc_diagnostics: diagnostics,
        })
    }

    fn orbital_kernel<'b>(
        &'b self,
        density: &'b [BlochMatrix],
    ) -> Result<Box<dyn SpinorDensityKernel + 'b>> {
        let [density] = density else {
            return Err(DftError::InvalidInput(
                "two-component molecular kernel needs one stationary density".into(),
            ));
        };
        let lifted = self.map.lift_density(&self.reference_density(density)?)?;
        let angular = if let Some(grid) = &self.semilocal {
            let (_, _, _, diagnostic) =
                grid.semilocal(self.functional, &lifted, self.options.xc_angular_tolerance)?;
            if !diagnostic.converged {
                return Err(DftError::Unsupported {
                    feature: "two-component molecular XC response".into(),
                    missing: "a converged spin-sphere quadrature for the stationary XC operator"
                        .into(),
                });
            }
            diagnostic.angular_order
        } else {
            0
        };
        Ok(Box::new(MolecularKernel {
            model: self,
            lifted,
            angular,
        }))
    }
}

struct MolecularKernel<'a, 'b> {
    model: &'b MolecularTwoComponent<'a>,
    lifted: BlochMatrix,
    angular: usize,
}

impl SpinorDensityKernel for MolecularKernel<'_, '_> {
    fn apply(&self, direction: &[BlochMatrix]) -> Result<Vec<BlochMatrix>> {
        let [direction] = direction else {
            return Err(DftError::InvalidInput(
                "two-component molecular kernel needs one density direction".into(),
            ));
        };
        let model = self.model;
        let direction = model
            .map
            .lift_density(&model.reference_density(direction)?)?;
        let operator = model.full_density_kernel(&self.lifted, &direction, self.angular)?;
        Ok(vec![model.reference_density(&model.project(&operator)?)?])
    }
}

fn closed_shell_spin_trace(a: &BlochMatrix) -> BlochMatrix {
    let n = a.rows() / 2;
    let mut spin_average = BlochMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            spin_average.re[(i, j)] = 0.5 * (a.re[(i, j)] + a.re[(n + i, n + j)]);
            spin_average.im[(i, j)] = 0.5 * (a.im[(i, j)] + a.im[(n + i, n + j)]);
        }
    }
    spin_diagonal(&spin_average)
}
