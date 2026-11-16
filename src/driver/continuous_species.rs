// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed-geometry, all-electron continuous nuclear-charge models.
//!
//! Element labels select the immutable AO template and quadrature only.
//! Physical charges are `Z(q) = Z_ref + B q`, independently of the fixed
//! integer electron count. Parameter derivatives use the existing Hartree/XC
//! CPKS kernel; they do not differentiate a geometry or a moving AO basis.
//! Nuclear gradients include the usual moving AO and quadrature response;
//! this is separate from the fixed-geometry parameter derivatives. Third-order
//! mixed responses are not supplied by this module.

pub mod core_jet2;
pub mod few_center;
pub mod few_center_driver;
pub mod fragment_driver;
pub mod frame;
pub mod gaussian;
pub mod local;
pub mod mixed_response;
pub mod periodic_charge;
pub mod periodic_model;
pub mod reduced;
pub mod reduction_linear;
pub mod reference_control;
pub mod rigid_pose;
pub mod rigid_pose_driver;
pub mod rigid_pose_record;
pub mod rigid_pose_response;
pub mod scalar;
pub(crate) mod static_core;
pub mod steric;

use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::integrals::{dipole_matrices, nuclear_matrix, overlap_matrix, CoulombKernel};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::response::diagnostics::{reference_diagnostics, ReferenceDiagnostics};
use crate::response::krylov::OperatorStructure;
use crate::response::{ResponseSolver, ResponseSolverConfig};
use crate::scf::{
    run_scf_in_potential, FrozenPotential, JkEngine, Reference, ScfOptions, ScfResult,
};
use crate::system::Molecule;
use crate::xc::{resolve, FunctionalSpec, ResolvedFunctional};
use crate::{DftError, Result};

/// The two semilocal model choices whose acceptance suites must be run.
/// No functional-name-dependent branches occur in the response calculation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeciesFunctional {
    Lda,
    Pbe,
}

impl SpeciesFunctional {
    pub fn name(self) -> &'static str {
        match self {
            Self::Lda => "lda",
            Self::Pbe => "pbe",
        }
    }
}

/// A column of B. Charges are measured in elementary charge units.
#[derive(Clone, Debug)]
pub struct ChargeParameter {
    pub name: String,
    pub unit: String,
    pub scale: f64,
    pub bounds: [f64; 2],
    pub charge_direction: Vec<f64>,
}

/// Affine chart with fixed parameter order. There is no cross-chart interpolation.
#[derive(Clone, Debug)]
pub struct ChargeChart {
    pub reference_charges: Vec<f64>,
    pub parameters: Vec<ChargeParameter>,
    pub conserve_total_charge: bool,
}

use static_core::IonicChargePolicy;

impl ChargeChart {
    pub fn validate(&self, atoms: usize) -> Result<()> {
        self.validate_with_policy(atoms, &IonicChargePolicy::Positive)
    }

    fn validate_with_policy(&self, atoms: usize, policy: &IonicChargePolicy) -> Result<()> {
        policy.validate_charges(&self.reference_charges, atoms)?;
        if self.parameters.is_empty() {
            return invalid(
                "charge charts require positive finite reference charges and parameters",
            );
        }
        for (i, p) in self.parameters.iter().enumerate() {
            if p.name.trim().is_empty()
                || p.unit.trim().is_empty()
                || self.parameters[..i]
                    .iter()
                    .any(|other| other.name == p.name)
                || !p.scale.is_finite()
                || p.scale <= 0.0
                || p.bounds.iter().any(|x| !x.is_finite())
                || p.bounds[0] >= p.bounds[1]
                || p.charge_direction.len() != atoms
                || p.charge_direction.iter().any(|x| !x.is_finite())
            {
                return invalid("invalid charge parameter name, units, scale, bounds or direction");
            }
            if let IonicChargePolicy::AuxiliaryZeros(mask) = policy {
                if p.charge_direction
                    .iter()
                    .zip(mask)
                    .any(|(&b, &aux)| aux && b != 0.0)
                {
                    return invalid("auxiliary-only AO centers cannot acquire ionic charge through any chart direction");
                }
            }
            let sum: f64 = p.charge_direction.iter().sum();
            let norm: f64 = p.charge_direction.iter().map(|x| x.abs()).sum();
            if self.conserve_total_charge && sum.abs() > 32.0 * f64::EPSILON * norm {
                return invalid(
                    "a charge-conserving chart requires each column sum of B to vanish",
                );
            }
        }
        Ok(())
    }

    pub fn charges(&self, q: &[f64]) -> Result<Vec<f64>> {
        self.charges_with_policy(q, &IonicChargePolicy::Positive)
    }

    fn charges_with_policy(&self, q: &[f64], policy: &IonicChargePolicy) -> Result<Vec<f64>> {
        self.validate_with_policy(self.reference_charges.len(), policy)?;
        if q.len() != self.parameters.len()
            || q.iter()
                .zip(&self.parameters)
                .any(|(&x, p)| !x.is_finite() || x < p.bounds[0] || x > p.bounds[1])
        {
            return invalid("charge coordinates must be finite and lie inside their chart bounds");
        }
        let mut z = self.reference_charges.clone();
        for (&x, p) in q.iter().zip(&self.parameters) {
            for (z, &b) in z.iter_mut().zip(&p.charge_direction) {
                *z += b * x;
            }
        }
        policy.validate_charges(&z, self.reference_charges.len())?;
        Ok(z)
    }
}

/// Content fingerprints identify the actual fixed arrays, not just input labels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeciesFingerprints {
    pub geometry: u64,
    pub basis: u64,
    pub grid: u64,
    pub chart: u64,
}

/// Shared report contract for scalar fragment reference bookkeeping.
pub const FRAGMENT_SCF_ENERGY_CONVENTION: &str = "state-common fragment_internal_energy excluded from trial energies and convergence; included only in reported total/classical energy";

/// A scoped state cannot be passed accidentally to ordinary force routines.
pub struct SpeciesState {
    scf: ScfResult,
    /// State-common fragment reference kept outside the SCF energy arithmetic.
    fragment_internal_energy: f64,
    fingerprints: SpeciesFingerprints,
    coordinates: Vec<f64>,
    nuclear_charges: Vec<f64>,
    total_charge: f64,
    electron_count: usize,
    field: Vec3,
    origin: Vec3,
    pub dipole: Vec3,
    pub diagnostics: ReferenceDiagnostics,
    /// Spin-summed AO density distance from a compatible continuation seed.
    pub continuation_density_distance: Option<f64>,
    /// Sum of squared occupied-subspace singular values, per spin.
    /// Invariant under MO signs and rotations within occupied spaces.
    pub occupied_subspace_overlap: Option<[f64; 2]>,
}

impl SpeciesState {
    pub fn coordinates(&self) -> &[f64] {
        &self.coordinates
    }
    pub fn nuclear_charges(&self) -> &[f64] {
        &self.nuclear_charges
    }
    pub fn total_charge(&self) -> f64 {
        self.total_charge
    }
    pub fn electron_count(&self) -> usize {
        self.electron_count
    }
    pub fn field(&self) -> Vec3 {
        self.field
    }
    pub fn origin(&self) -> Vec3 {
        self.origin
    }
    pub fn model_version(&self) -> &'static str {
        "continuous-species-v1-clamped-geometry"
    }
    pub fn stability_status(&self) -> &'static str {
        "not_evaluated"
    }

    pub fn energy(&self) -> f64 {
        self.scf.energy + self.fragment_internal_energy
    }

    pub fn classical_energy(&self) -> f64 {
        self.scf.e_nuc + self.fragment_internal_energy
    }

    /// Energy with the state-common eliminated-fragment reference excluded
    /// before SCF, not recovered by subtracting two large floating-point values.
    /// Reaction objectives must compare this quantity for compatible states.
    pub fn energy_without_internal_reference(&self) -> f64 {
        self.scf.energy
    }

    /// State-dependent core, nuclear, external-field and shape energy. Physical
    /// strain or polarization terms are retained; only the common constant is absent.
    pub fn classical_energy_without_internal_reference(&self) -> f64 {
        self.scf.e_nuc
    }

    pub fn fragment_internal_energy(&self) -> f64 {
        self.fragment_internal_energy
    }

    pub fn electronic_energy(&self) -> f64 {
        self.scf.e_elec
    }

    pub fn total_density(&self) -> Matrix {
        self.scf.p_total()
    }

    /// Fixed integer alpha/beta populations of this collinear determinant.
    pub fn spin_electron_counts(&self) -> [usize; 2] {
        [self.scf.n_alpha, self.scf.n_beta]
    }

    pub fn spin_density(&self) -> Matrix {
        let mut density = self.scf.density[0].clone();
        for (x, &y) in density
            .as_mut_slice()
            .iter_mut()
            .zip(self.scf.density[1].as_slice())
        {
            *x -= y;
        }
        density
    }

    pub fn fingerprints(&self) -> &SpeciesFingerprints {
        &self.fingerprints
    }

    pub fn frontier_gap(&self) -> Option<f64> {
        self.spin_frontier_gaps()
            .into_iter()
            .flatten()
            .reduce(f64::min)
    }

    pub fn spin_frontier_gaps(&self) -> [Option<f64>; 2] {
        std::array::from_fn(|spin| {
            let count = self.spin_electron_counts()[spin];
            let energies = &self.scf.mo_energy[spin];
            count
                .checked_sub(1)
                .and_then(|i| energies.get(i))
                .zip(energies.get(count))
                .map(|(&h, &l)| l - h)
        })
    }
}

/// Every tensor is analytic at clamped geometry and the recorded fixed grid.
/// `energy_hessian[a][b]`, `dipole_jacobian[i][a]`, `polarizability[i][j]`.
pub struct SpeciesSensitivity {
    pub parameter_names: Vec<String>,
    pub parameter_units: Vec<String>,
    pub energy_gradient: Vec<f64>,
    pub energy_hessian: Vec<Vec<f64>>,
    pub dipole_jacobian: [Vec<f64>; 3],
    pub polarizability: [[f64; 3]; 3],
    /// Fresh physical CPKS relative residual for every q, then x/y/z field.
    pub response_residuals: Vec<f64>,
    pub hessian_symmetry_defect: f64,
    pub polarizability_symmetry_defect: f64,
    pub mixed_derivative_defect: f64,
    pub fingerprints: SpeciesFingerprints,
}

/// Owns all fixed geometry, basis and quadrature arrays. No caller can mutate
/// cached unit nuclear-attraction matrices while retaining a model identity.
pub struct ContinuousSpeciesModel {
    molecule: Molecule,
    basis: MolBasis,
    grid: MolecularGrid,
    chart: ChargeChart,
    ionic_charge_policy: IonicChargePolicy,
    functional: ResolvedFunctional,
    options: ScfOptions,
    electron_count: usize,
    origin: Vec3,
    fingerprints: SpeciesFingerprints,
    unit_attraction: Vec<Matrix>,
    parameter_operators: Vec<Matrix>,
    dipoles: [Matrix; 3],
    overlap: Matrix,
    coulomb: JkEngine,
    pub overlap_condition_number: f64,
    local_terms: Vec<local::LocalTerm>,
    fragment_internal_energy: f64,
    static_core: Option<static_core::StaticCoreCorrection>,
}

impl ContinuousSpeciesModel {
    /// Internal scoped seam for a generated fixed-electron core model. It
    /// cannot be attached to a nuclear-charge chart until the corresponding
    /// core/charge mixed classical derivatives have been defined.
    pub(crate) fn with_static_core_correction(
        mut self,
        correction: static_core::StaticCoreCorrection,
    ) -> Result<Self> {
        correction.validate(self.basis.nao_sph, self.molecule.len())?;
        if self.static_core.is_some()
            || (correction.origin - self.origin).norm() > 1e-14
            || self
                .chart
                .parameters
                .iter()
                .any(|p| p.charge_direction.iter().any(|&b| b != 0.0))
        {
            return unsupported("a generated fixed core requires one matching-origin correction and a fixed ionic-charge chart");
        }
        let mut identity = Fingerprint(self.fingerprints.chart);
        identity.string("generated-static-core-v1");
        for &x in correction.operator.as_slice() {
            identity.float(x);
        }
        for derivatives in &correction.atom_derivative_matrices {
            for matrix in derivatives {
                for &x in matrix.as_slice() {
                    identity.float(x);
                }
            }
        }
        identity.float(correction.classical_energy);
        identity.float(correction.charge_correction);
        identity.vector(correction.permanent_dipole);
        for &v in &correction.classical_gradients {
            identity.vector(v);
        }
        for derivative in &correction.permanent_dipole_derivatives {
            for row in derivative {
                for &x in row {
                    identity.float(x);
                }
            }
        }
        for site in &correction.electrostatic_sites {
            identity.vector(site.center);
            identity.float(site.exponent);
            identity.float(site.charge);
            identity.vector(site.dipole);
        }
        self.fingerprints.chart = identity.0;
        self.static_core = Some(correction);
        Ok(self)
    }

    /// A fitted state-common internal reference. Keep it out of the SCF
    /// convergence arithmetic; only reported total/classical energies add it.
    /// The constant-free state accessor supports reaction-energy cancellation.
    pub fn with_fragment_internal_energy(mut self, energy: f64) -> Result<Self> {
        if !energy.is_finite() {
            return invalid("fragment internal energy must be finite");
        }
        if energy != self.fragment_internal_energy {
            let mut identity = Fingerprint(self.fingerprints.chart);
            identity.string("fragment-internal-energy");
            identity.float(energy);
            self.fingerprints.chart = identity.0;
            self.fragment_internal_energy = energy;
        }
        Ok(self)
    }

    /// Positive electron-number density and physical electrostatic potential
    /// at explicit Cartesian probes. Local Gaussian and steric residuals are
    /// not physical point charges and contribute only through their density.
    pub fn density_and_potential_at_points(
        &self,
        state: &SpeciesState,
        points: &[Vec3],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        self.check_state(state)?;
        if points.iter().any(|&point| {
            !finite_vector(point)
                || self
                    .molecule
                    .atoms
                    .iter()
                    .any(|a| (point - a.position).norm() < 1e-10)
        }) {
            return invalid("electrostatic probes must be finite and separated from every nucleus");
        }
        let batch = crate::grid::eval::eval_aos(&self.basis, points, false);
        let density = state.scf.p_total();
        let core_potential = self
            .static_core
            .as_ref()
            .map(|core| {
                crate::pseudo_sub::reduction_kernels::electrostatic_potential(
                    &core.electrostatic_sites,
                    points,
                )
            })
            .transpose()?;
        let mut rho = Vec::with_capacity(points.len());
        let mut potential = Vec::with_capacity(points.len());
        for (p, &point) in points.iter().enumerate() {
            let mut value = crate::linalg::AccurateSum::default();
            for i in 0..self.basis.nao_sph {
                for j in 0..self.basis.nao_sph {
                    value.add_product(density[(i, j)] * batch.phi[(p, i)], batch.phi[(p, j)]);
                }
            }
            let probe = Molecule::new(vec![crate::system::Atom {
                z: 1,
                position: point,
            }]);
            let attraction = nuclear_matrix(&self.basis, &probe, Some(&[1.0]));
            let nuclear: f64 = self
                .molecule
                .atoms
                .iter()
                .zip(&state.nuclear_charges)
                .map(|(atom, &z)| z / (point - atom.position).norm())
                .sum();
            rho.push(value.value());
            potential.push(
                nuclear
                    + density.frobenius_dot(&attraction)
                    + core_potential.as_ref().map_or(0.0, |v| v[p]),
            );
        }
        Ok((rho, potential))
    }

    /// Frequency-resolved response of the same converged physical Hamiltonian.
    /// Local potentials have no density kernel of their own; Hartree/XC is
    /// fully coupled through the existing paired response operator.
    pub fn dynamic_polarizability(
        &self,
        state: &SpeciesState,
        frequency: crate::response::Frequency,
        config: &ResponseSolverConfig,
    ) -> Result<crate::response::ResponseReport<crate::response::DynamicPolarizability>> {
        self.check_state(state)?;
        let mut options = self.options.clone();
        options.external_field = Some(state.field);
        crate::response::dynamic_polarizability(
            &self.molecule,
            &self.basis,
            &self.functional,
            Some(&self.grid),
            &state.scf,
            &options,
            frequency,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        molecule: Molecule,
        basis: MolBasis,
        grid: MolecularGrid,
        electron_count: usize,
        chart: ChargeChart,
        functional: SpeciesFunctional,
        options: ScfOptions,
        origin: Vec3,
    ) -> Result<Self> {
        Self::new_internal(
            molecule,
            basis,
            grid,
            electron_count,
            None,
            chart,
            functional,
            options,
            origin,
            IonicChargePolicy::Positive,
        )
    }

    /// Collinear UKS with an explicit fixed integer spin sector. Counts may
    /// not change along a chart. Alpha is the majority-spin convention.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_spin(
        molecule: Molecule,
        basis: MolBasis,
        grid: MolecularGrid,
        spin_electrons: [usize; 2],
        chart: ChargeChart,
        functional: SpeciesFunctional,
        options: ScfOptions,
        origin: Vec3,
    ) -> Result<Self> {
        let count = spin_electrons[0]
            .checked_add(spin_electrons[1])
            .ok_or_else(|| DftError::InvalidInput("fixed spin electron counts overflow".into()))?;
        Self::new_internal(
            molecule,
            basis,
            grid,
            count,
            Some(spin_electrons),
            chart,
            functional,
            options,
            origin,
            IonicChargePolicy::Positive,
        )
    }

    /// Scoped generated-model entry. Zero ions must be explicitly declared
    /// AO-only centers; their chart columns remain exactly zero. Element IDs
    /// still select fixed basis/grid roles and remain positive integers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_auxiliary_roles(
        molecule: Molecule,
        basis: MolBasis,
        grid: MolecularGrid,
        electron_count: usize,
        chart: ChargeChart,
        functional: SpeciesFunctional,
        options: ScfOptions,
        origin: Vec3,
        auxiliary_roles: Vec<bool>,
    ) -> Result<Self> {
        Self::new_internal(
            molecule,
            basis,
            grid,
            electron_count,
            None,
            chart,
            functional,
            options,
            origin,
            IonicChargePolicy::AuxiliaryZeros(auxiliary_roles),
        )
    }

    /// Same scoped role contract with an explicit fixed collinear spin sector.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_spin_and_auxiliary_roles(
        molecule: Molecule,
        basis: MolBasis,
        grid: MolecularGrid,
        spin_electrons: [usize; 2],
        chart: ChargeChart,
        functional: SpeciesFunctional,
        options: ScfOptions,
        origin: Vec3,
        auxiliary_roles: Vec<bool>,
    ) -> Result<Self> {
        let count = spin_electrons[0]
            .checked_add(spin_electrons[1])
            .ok_or_else(|| DftError::InvalidInput("fixed spin electron counts overflow".into()))?;
        Self::new_internal(
            molecule,
            basis,
            grid,
            count,
            Some(spin_electrons),
            chart,
            functional,
            options,
            origin,
            IonicChargePolicy::AuxiliaryZeros(auxiliary_roles),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_internal(
        mut molecule: Molecule,
        basis: MolBasis,
        grid: MolecularGrid,
        electron_count: usize,
        spin_electrons: Option<[usize; 2]>,
        chart: ChargeChart,
        functional: SpeciesFunctional,
        mut options: ScfOptions,
        origin: Vec3,
        ionic_charge_policy: IonicChargePolicy,
    ) -> Result<Self> {
        chart.validate_with_policy(molecule.len(), &ionic_charge_policy)?;
        validate_geometry(&molecule, &basis, &grid, origin)?;
        if (spin_electrons.is_none()
            && (molecule.multiplicity != 1 || matches!(options.reference, Reference::Uhf)))
            || !basis.ecps.is_empty()
            || basis.n_core_electrons != 0
            || options.electronic_temperature != 0.0
            || options.electron_offset != [0.0; 2]
            || options.occupation.is_some()
            || options.fixed_occupations.is_some()
            || options.guess_mix != 0.0
            || options.extra_hcore.is_some()
            || options.external_field.is_some()
            || options.pcm.is_some()
            || options.vv10.is_some()
            || options.skala.is_some()
            || options.asymptotic_correction.is_some()
            || options.density_fit.is_some()
            || options.kernel != CoulombKernel::Coulomb
        {
            return unsupported("continuous species require isolated all-electron RKS or explicit fixed-spin UKS with exact Coulomb, integer Aufbau occupations and no additional corrections");
        }
        let counts = spin_electrons.unwrap_or([electron_count / 2; 2]);
        if electron_count == 0
            || counts[0] < counts[1]
            || counts.iter().any(|&n| n > basis.nao_sph)
            || (spin_electrons.is_none() && electron_count % 2 != 0)
        {
            return invalid(
                "fixed electron counts must fit the AO template; RKS requires even total and UKS requires alpha >= beta",
            );
        }
        let label_charge = i64::from(molecule.total_nuclear_charge());
        let count = i64::try_from(electron_count).map_err(|_| {
            DftError::InvalidInput("electron count overflows the molecular adapter".into())
        })?;
        molecule.charge = i32::try_from(label_charge - count).map_err(|_| {
            DftError::InvalidInput("electron count overflows the molecular adapter".into())
        })?;
        // This integer charge belongs only to the existing SCF adapter. Physical
        // real charge is always calculated separately from Z(q) and N_e.
        if spin_electrons.is_some() {
            molecule.multiplicity = counts[0] - counts[1] + 1;
            options.reference = Reference::Uhf;
        } else {
            options.reference = Reference::Rhf;
        }
        options.eri_mode = options.effective_eri_mode(basis.nao_sph);
        let overlap = overlap_matrix(&basis);
        let (eigenvalues, _) = symmetric_eigen(&overlap)?;
        let min = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        let max = eigenvalues.iter().copied().fold(0.0, f64::max);
        if !min.is_finite() || min <= options.orth_threshold || !max.is_finite() {
            return invalid("continuous charts require a full-rank AO template above the orthogonalization cutoff");
        }
        let mut unit_attraction = Vec::with_capacity(molecule.len());
        for a in 0..molecule.len() {
            let mut unit = vec![0.0; molecule.len()];
            unit[a] = 1.0;
            unit_attraction.push(nuclear_matrix(&basis, &molecule, Some(&unit)));
        }
        let parameter_operators = chart
            .parameters
            .iter()
            .map(|p| linear_combination(&unit_attraction, &p.charge_direction, basis.nao_sph))
            .collect();
        let functional = resolve(&FunctionalSpec::Combined(functional.name().into()))?;
        let fingerprints = fingerprints(
            &molecule,
            &basis,
            &grid,
            &chart,
            electron_count,
            spin_electrons.is_some(),
            origin,
            &functional,
            &ionic_charge_policy,
        );
        let dipoles = dipole_matrices(&basis, origin);
        let coulomb = JkEngine::from_options(&molecule, &basis, &options)?;
        Ok(Self {
            molecule,
            basis,
            grid,
            chart,
            ionic_charge_policy,
            functional,
            options,
            electron_count,
            origin,
            fingerprints,
            unit_attraction,
            parameter_operators,
            dipoles,
            overlap,
            coulomb,
            overlap_condition_number: max / min,
            local_terms: Vec::new(),
            fragment_internal_energy: 0.0,
            static_core: None,
        })
    }

    pub fn chart(&self) -> &ChargeChart {
        &self.chart
    }

    pub fn fingerprints(&self) -> &SpeciesFingerprints {
        &self.fingerprints
    }

    /// A previous state is only an initial guess; it never changes the operator.
    /// Independent and reverse scans are obtained by omitting/reversing seeds.
    pub fn evaluate(
        &self,
        q: &[f64],
        field: Vec3,
        previous: Option<&SpeciesState>,
    ) -> Result<SpeciesState> {
        if !finite_vector(field) {
            return invalid("external field must be finite");
        }
        if let Some(previous) = previous {
            self.check_state(previous)?;
        }
        let charges = self
            .chart
            .charges_with_policy(q, &self.ionic_charge_policy)?;
        let delta: Vec<_> = charges
            .iter()
            .zip(&self.molecule.atoms)
            .map(|(&z, atom)| z - f64::from(atom.z))
            .collect();
        let mut operator = linear_combination(&self.unit_attraction, &delta, self.basis.nao_sph);
        // The common SCF field matrices use r about zero. Shift the electronic
        // operator by -F.origin S to make the recorded origin exact throughout.
        let shift = -field.dot(self.origin);
        for (v, &s) in operator
            .as_mut_slice()
            .iter_mut()
            .zip(self.overlap.as_slice())
        {
            *v += shift * s;
        }
        let mut options = self.options.clone();
        options.external_field = Some(field);
        // A large eliminated-core reference is irrelevant to the stationary
        // density and must not erase the SCF energy change through cancellation.
        let mut classical = nuclear_repulsion(&self.molecule, &charges);
        if let Some(core) = &self.static_core {
            for (value, &element) in operator
                .as_mut_slice()
                .iter_mut()
                .zip(core.operator.as_slice())
            {
                *value += element;
            }
            classical += core.classical_energy - field.dot(core.permanent_dipole);
        }
        for term in &self.local_terms {
            let (amplitude, strength) = term.amplitudes(q)?;
            for (value, &element) in operator
                .as_mut_slice()
                .iter_mut()
                .zip(term.potential.operator.as_slice())
            {
                *value += amplitude * element;
            }
            classical += strength * term.steric.value;
        }
        let mut dipole = Vec3::zero();
        if let Some(core) = &self.static_core {
            dipole += core.permanent_dipole;
        }
        for (atom, &z) in self.molecule.atoms.iter().zip(&charges) {
            let position = atom.position - self.origin;
            classical -= z * position.dot(field);
            dipole += position * z;
        }
        let scf = run_scf_in_potential(
            &self.molecule,
            &self.basis,
            &options,
            &self.functional,
            Some(&self.grid),
            previous.map(|p| &p.scf.density),
            None,
            Some(FrozenPotential {
                operator: &operator,
                coulomb: &self.coulomb,
                long_range: None,
                classical_energy: Some(classical),
            }),
        )?;
        let diagnostics = reference_diagnostics(&scf)?;
        if !scf.converged || diagnostics.orbital_residual > self.options.conv_tol_grad {
            return Err(DftError::ScfNotConverged {
                iterations: scf.iterations,
                error: diagnostics.orbital_residual,
            });
        }
        let density = scf.p_total();
        dipole.x -= density.frobenius_dot(&self.dipoles[0]);
        dipole.y -= density.frobenius_dot(&self.dipoles[1]);
        dipole.z -= density.frobenius_dot(&self.dipoles[2]);
        if !scf.energy.is_finite()
            || !(scf.energy + self.fragment_internal_energy).is_finite()
            || !(scf.e_nuc + self.fragment_internal_energy).is_finite()
            || !finite_vector(dipole)
        {
            return invalid("continuous species produced a non-finite observable");
        }
        let continuation_density_distance = previous.map(|p| {
            density
                .as_slice()
                .iter()
                .zip(p.scf.p_total().as_slice())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt()
        });
        let occupied_subspace_overlap = previous.map(|p| {
            std::array::from_fn(|s| {
                let cross = p.scf.mo_coeff[s]
                    .transpose()
                    .matmul_seq(&self.overlap)
                    .matmul_seq(&scf.mo_coeff[s]);
                let occupied = [scf.n_alpha, scf.n_beta][s];
                (0..occupied)
                    .flat_map(|i| (0..occupied).map(move |j| (i, j)))
                    .map(|(i, j)| cross[(i, j)].powi(2))
                    .sum()
            })
        });
        Ok(SpeciesState {
            scf,
            fragment_internal_energy: self.fragment_internal_energy,
            fingerprints: self.fingerprints.clone(),
            coordinates: q.to_vec(),
            total_charge: charges.iter().sum::<f64>() - self.electron_count as f64
                + self
                    .static_core
                    .as_ref()
                    .map_or(0.0, |core| core.charge_correction),
            nuclear_charges: charges,
            electron_count: self.electron_count,
            field,
            origin: self.origin,
            dipole,
            diagnostics,
            continuation_density_distance,
            occupied_subspace_overlap,
        })
    }

    /// Hellmann-Feynman gradient and fully coupled second derivatives.
    /// Energy is Eh, dipole is e a0, polarizability is a0^3. Parameter
    /// derivatives divide these units by the chart's reported parameter units.
    pub fn sensitivity(
        &self,
        state: &SpeciesState,
        config: &ResponseSolverConfig,
    ) -> Result<SpeciesSensitivity> {
        self.check_state(state)?;
        config.validate()?;
        let mut options = self.options.clone();
        options.external_field = Some(state.field);
        let solver = ResponseSolver::new(
            &self.molecule,
            &self.basis,
            &self.functional,
            Some(&self.grid),
            &state.scf,
            &options,
        )?;
        let (occupied, virtuals) = solver.ov_dims();
        let dimension = occupied[0] * virtuals[0]
            + if state.scf.unrestricted {
                occupied[1] * virtuals[1]
            } else {
                0
            };
        let mut responses = Vec::new();
        let mut response_residuals = Vec::new();
        for perturbation in self.parameter_operators.iter().chain(self.dipoles.iter()) {
            let mut rhs = vec![0.0; dimension];
            solver.project_ov([perturbation, perturbation], &mut rhs);
            // Gap positivity alone does not establish orbital stability.
            let solution = solver.solve_ov_report_with_structure(
                &rhs,
                config,
                OperatorStructure::Hermitian,
            )?;
            if solution
                .amplitudes
                .iter()
                .any(|a| a.im.abs() > config.acceptance())
            {
                return invalid(
                    "a static real charge perturbation produced complex response amplitudes",
                );
            }
            let amplitudes: Vec<_> = solution.amplitudes.iter().map(|a| a.re).collect();
            let [mut total, beta] = solver.delta_density(&amplitudes);
            for (x, &y) in total.as_mut_slice().iter_mut().zip(beta.as_slice()) {
                *x += y;
            }
            responses.push(total);
            response_residuals.push(solution.relative_residual);
        }
        let nq = self.chart.parameters.len();
        let density = state.scf.p_total();
        let mut energy_gradient = vec![0.0; nq];
        let mut energy_hessian = vec![vec![0.0; nq]; nq];
        let mut dipole_jacobian: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; nq]);
        let mut mixed_derivative_defect: f64 = 0.0;
        for (a, pa) in self.chart.parameters.iter().enumerate() {
            let mut nuclear_dipole = Vec3::zero();
            let mut classical_first = 0.0;
            for (i, atom) in self.molecule.atoms.iter().enumerate() {
                nuclear_dipole += (atom.position - self.origin) * pa.charge_direction[i];
                for j in i + 1..self.molecule.len() {
                    let r = (atom.position - self.molecule.atoms[j].position).norm();
                    classical_first += (pa.charge_direction[i] * state.nuclear_charges[j]
                        + state.nuclear_charges[i] * pa.charge_direction[j])
                        / r;
                }
            }
            energy_gradient[a] = density.frobenius_dot(&self.parameter_operators[a])
                + classical_first
                - nuclear_dipole.dot(state.field)
                + self
                    .local_terms
                    .iter()
                    .map(|term| term.spec.steric_direction[a] * term.steric.value)
                    .sum::<f64>();
            for (b, pb) in self.chart.parameters.iter().enumerate() {
                let mut classical_second = 0.0;
                for i in 0..self.molecule.len() {
                    for j in i + 1..self.molecule.len() {
                        let r = (self.molecule.atoms[i].position - self.molecule.atoms[j].position)
                            .norm();
                        classical_second += (pa.charge_direction[i] * pb.charge_direction[j]
                            + pb.charge_direction[i] * pa.charge_direction[j])
                            / r;
                    }
                }
                energy_hessian[a][b] =
                    responses[b].frobenius_dot(&self.parameter_operators[a]) + classical_second;
            }
            for i in 0..3 {
                dipole_jacobian[i][a] =
                    nuclear_dipole.get(i) - responses[a].frobenius_dot(&self.dipoles[i]);
                let other = nuclear_dipole.get(i)
                    - responses[nq + i].frobenius_dot(&self.parameter_operators[a]);
                mixed_derivative_defect =
                    mixed_derivative_defect.max((dipole_jacobian[i][a] - other).abs());
            }
        }
        let polarizability: [[f64; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| -responses[nq + j].frobenius_dot(&self.dipoles[i]))
        });
        let mut hessian_symmetry_defect: f64 = 0.0;
        let mut polarizability_symmetry_defect: f64 = 0.0;
        for a in 0..nq {
            for b in 0..nq {
                hessian_symmetry_defect = hessian_symmetry_defect
                    .max((energy_hessian[a][b] - energy_hessian[b][a]).abs());
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                polarizability_symmetry_defect = polarizability_symmetry_defect
                    .max((polarizability[i][j] - polarizability[j][i]).abs());
            }
        }
        if energy_gradient
            .iter()
            .chain(energy_hessian.iter().flatten())
            .chain(dipole_jacobian.iter().flatten())
            .chain(polarizability.iter().flatten())
            .any(|x| !x.is_finite())
        {
            return invalid("continuous species produced a non-finite response tensor");
        }
        Ok(SpeciesSensitivity {
            parameter_names: self
                .chart
                .parameters
                .iter()
                .map(|p| p.name.clone())
                .collect(),
            parameter_units: self
                .chart
                .parameters
                .iter()
                .map(|p| p.unit.clone())
                .collect(),
            energy_gradient,
            energy_hessian,
            dipole_jacobian,
            polarizability,
            response_residuals,
            hessian_symmetry_defect,
            polarizability_symmetry_defect,
            mixed_derivative_defect,
            fingerprints: self.fingerprints.clone(),
        })
    }

    /// Analytic nuclear gradient at fixed q and fixed AO/grid *templates*.
    /// AO centers and quadrature move with nuclei. Element-based partition
    /// radii stay fixed, as they do in ordinary template molecular gradients.
    pub fn nuclear_gradient(&self, state: &SpeciesState) -> Result<Vec<Vec3>> {
        self.check_state(state)?;
        let mut options = self.options.clone();
        options.external_field = Some(state.field);
        let mut gradient = crate::gradient::assemble_gradient(
            &self.molecule,
            &self.basis,
            &self.functional,
            Some(&self.grid),
            &state.scf,
            &options,
        )?;
        let density = state.scf.p_total();
        let delta: Vec<_> = state
            .nuclear_charges
            .iter()
            .zip(&self.molecule.atoms)
            .map(|(&z, atom)| z - f64::from(atom.z))
            .collect();
        // Includes both moving-AO and moving-operator-center contributions.
        let attraction = crate::integrals::deriv::nuclear_gradient(
            &self.basis,
            &self.molecule,
            &delta,
            &density,
            1.0,
        );
        // The origin shift in h is -F.origin S. Its explicit overlap derivative
        // cancels the corresponding constant shift in the metric Pulay term.
        let shifted_metric = crate::integrals::deriv::overlap_gradient(
            &self.basis,
            &density,
            self.molecule.len(),
            -state.field.dot(self.origin),
        );
        for i in 0..self.molecule.len() {
            gradient[i] += attraction[i] + shifted_metric[i] - state.field * delta[i];
            for j in 0..self.molecule.len() {
                if i == j {
                    continue;
                }
                let r = self.molecule.atoms[i].position - self.molecule.atoms[j].position;
                let norm = r.norm();
                let difference = state.nuclear_charges[i] * state.nuclear_charges[j]
                    - f64::from(self.molecule.atoms[i].z) * f64::from(self.molecule.atoms[j].z);
                gradient[i] += r * (-difference / norm.powi(3));
            }
        }
        self.local_gradient(state, &mut gradient)?;
        if let Some(core) = &self.static_core {
            let density = state.scf.p_total();
            for (atom, value) in gradient.iter_mut().enumerate() {
                let derivative = std::array::from_fn::<_, 3, _>(|axis| {
                    density.frobenius_dot(&core.atom_derivative_matrices[atom][axis])
                        + core.classical_gradients[atom].get(axis)
                        - (0..3)
                            .map(|i| {
                                state.field.get(i)
                                    * core.permanent_dipole_derivatives[atom][i][axis]
                            })
                            .sum::<f64>()
                });
                *value += Vec3::new(derivative[0], derivative[1], derivative[2]);
            }
        }
        if gradient.iter().any(|&v| !finite_vector(v)) {
            return invalid("continuous-charge nuclear gradient is not finite");
        }
        Ok(gradient)
    }

    fn check_state(&self, state: &SpeciesState) -> Result<()> {
        if state.fingerprints != self.fingerprints {
            return invalid("a state from a different fixed geometry, basis, grid or charge chart cannot be reused");
        }
        Ok(())
    }
}

/// Common-driver calculation entry. Registration in the molecular dispatcher
/// is deliberately a separate, narrow integration change.
pub fn calculate(system: &super::System, controls: &super::Controls) -> Result<super::Value> {
    use super::Value;
    if controls.contains("fragment_mode") {
        return fragment_driver::run(system, controls);
    }
    if system.is_periodic() {
        return unsupported(
            "periodic continuous-charge response requires a separate electrostatic model",
        );
    }
    let electron_count = controls.opt_usize("fixed_electron_count")?.ok_or_else(|| {
        DftError::InvalidInput(
            "continuous-species requires an explicit fixed_electron_count".into(),
        )
    })?;
    let spin_counts = controls
        .opt_usize_list("fixed_spin_electron_counts")?
        .map(|counts| -> Result<[usize; 2]> {
            let pair: [usize; 2] = counts.try_into().map_err(|_| {
                DftError::InvalidInput("fixed_spin_electron_counts requires [alpha, beta]".into())
            })?;
            if pair[0].checked_add(pair[1]) != Some(electron_count) {
                return invalid("fixed spin electron counts must sum to fixed_electron_count");
            }
            Ok(pair)
        })
        .transpose()?;
    let reference_charges = controls
        .opt_f64_list("physical_nuclear_charges")?
        .unwrap_or_else(|| system.numbers.iter().map(|&z| f64::from(z)).collect());
    let directions = controls.opt_rows("charge_directions")?.ok_or_else(|| {
        DftError::InvalidInput(
            "continuous-species requires charge_directions (parameter by atom)".into(),
        )
    })?;
    let coordinates = controls
        .opt_f64_list("charge_coordinates")?
        .ok_or_else(|| {
            DftError::InvalidInput(
                "continuous-species requires charge_coordinates in parameter order".into(),
            )
        })?;
    let bounds = controls.opt_rows("charge_bounds")?.ok_or_else(|| {
        DftError::InvalidInput(
            "continuous-species requires explicit charge_bounds per parameter".into(),
        )
    })?;
    let scales = controls
        .opt_f64_list("charge_scales")?
        .unwrap_or_else(|| vec![1.0; directions.len()]);
    let origin = controls.opt_vec3("charge_origin")?.unwrap_or(Vec3::zero());
    let conserve_total_charge = controls.bool("conserve_total_charge", true)?;
    let want_gradient = controls.bool("charge_nuclear_gradient", false)?;
    let names = match controls.take("charge_parameter_names") {
        None => (0..directions.len())
            .map(|i| format!("q{i}"))
            .collect::<Vec<_>>(),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value.as_str().map(str::to_owned).ok_or_else(|| {
                    DftError::InvalidInput("charge_parameter_names must be strings".into())
                })
            })
            .collect::<Result<Vec<_>>>()?,
        Some(_) => return invalid("charge_parameter_names must be an array of strings"),
    };
    if bounds.len() != directions.len()
        || bounds.iter().any(|r| r.len() != 2)
        || scales.len() != directions.len()
        || names.len() != directions.len()
    {
        return invalid(
            "charge direction rows, bounds, scales and names must have the same parameter count",
        );
    }
    let parameters = directions
        .into_iter()
        .enumerate()
        .map(|(i, charge_direction)| ChargeParameter {
            name: names[i].clone(),
            unit: "dimensionless".into(),
            scale: scales[i],
            bounds: [bounds[i][0], bounds[i][1]],
            charge_direction,
        })
        .collect();
    let chart = ChargeChart {
        reference_charges,
        parameters,
        conserve_total_charge,
    };
    let _ = chart.charges(&coordinates)?;
    let config = super::vibrational::response_config(controls)?;
    let mut method = super::method::MolecularMethod::from_controls(controls)?;
    let functional = match method
        .xc
        .as_deref()
        .unwrap_or("pbe")
        .to_ascii_lowercase()
        .as_str()
    {
        "lda" => SpeciesFunctional::Lda,
        "pbe" => SpeciesFunctional::Pbe,
        _ => {
            return unsupported(
                "this first continuous-charge model exposes the tested LDA and PBE choices",
            )
        }
    };
    if method.x.is_some()
        || method.c.is_some()
        || method.omega.is_some()
        || !method.xc_parameters.is_empty()
        || method.ecp.is_some()
        || method.stabilize
    {
        return unsupported("split or modified functionals, ECPs and stability following are outside the fixed continuous-charge contract");
    }
    method.xc = Some(functional.name().into());
    let mut setup = super::method::prepare_molecular(&system.numbers, &system.positions, &method)?;
    if setup.dispersion != crate::dispersion::Dispersion::None || setup.smd.is_some() {
        return unsupported(
            "empirical dispersion and solvent corrections are not continuous-charge models",
        );
    }
    let field = setup.options.external_field.take().unwrap_or(Vec3::zero());
    let grid = setup.grid.take().ok_or_else(|| {
        DftError::InvalidInput(
            "a semilocal continuous-charge model requires a molecular grid".into(),
        )
    })?;
    let local_specs = local::from_controls(controls, system.numbers.len(), coordinates.len())?;
    let fragment_internal_energy = controls.f64("fragment_internal_energy", 0.0)?;
    let dynamic_frequency = if controls.bool("charge_dynamic_response", false)? {
        Some(super::polarizability::frequency(controls)?)
    } else {
        None
    };
    let probe_points = controls
        .opt_rows("charge_probe_points")?
        .map(|rows| -> Result<Vec<Vec3>> {
            rows.into_iter()
                .map(|row| {
                    if row.len() != 3 {
                        return invalid("charge_probe_points requires Cartesian three-vectors");
                    }
                    Ok(Vec3::new(row[0], row[1], row[2]))
                })
                .collect()
        })
        .transpose()?;
    controls.finish("continuous-species")?;
    let model = ContinuousSpeciesModel::new_internal(
        setup.molecule,
        setup.basis,
        grid,
        electron_count,
        spin_counts,
        chart,
        functional,
        setup.options,
        origin,
        IonicChargePolicy::Positive,
    )?
    .with_local_potentials(local_specs)?
    .with_fragment_internal_energy(fragment_internal_energy)?;
    let state = model.evaluate(&coordinates, field, None)?;
    let response = model.sensitivity(&state, &config)?;
    let gradient = if want_gradient {
        Some(model.nuclear_gradient(&state)?)
    } else {
        None
    };
    let mut units = Value::object()
        .with("energy", "hartree")
        .with("classical_energy", "hartree")
        .with("electronic_energy", "hartree")
        .with("energy_without_internal_reference", "hartree")
        .with("classical_energy_without_internal_reference", "hartree")
        .with("fragment_internal_energy", "hartree")
        .with("dipole", "e*bohr")
        .with("polarizability", "bohr^3")
        .with("parameter_energy_gradient", "hartree")
        .with("parameter_energy_hessian", "hartree")
        .with("parameter_dipole_jacobian", "e*bohr")
        .with("gradient", "hartree/bohr")
        .with("forces", "hartree/bohr")
        .with("physical_nuclear_charges", "e")
        .with("physical_total_charge", "e")
        .with("fixed_electron_count", "integer")
        .with("fixed_spin_electron_counts", "integer")
        .with("spin_squared", "hbar^2")
        .with("local_potential_amplitudes", "hartree")
        .with("steric_strengths", "hartree")
        .with("charge_origin", "bohr")
        .with("field", "hartree/e/bohr")
        .with("charge_coordinates", "dimensionless")
        .with("charge_bounds", "dimensionless")
        .with("charge_scales", "dimensionless")
        .with("charge_directions", "e")
        .with("scf_orbital_residual", "hartree")
        .with("scf_eigen_residual", "dimensionless")
        .with("response_residuals", "dimensionless")
        .with("frontier_gap", "hartree")
        .with("ao_condition_number", "dimensionless")
        .with("hessian_symmetry_defect", "hartree")
        .with("polarizability_symmetry_defect", "bohr^3")
        .with("mixed_derivative_defect", "e*bohr");
    let dynamic = dynamic_frequency
        .map(|frequency| model.dynamic_polarizability(&state, frequency, &config))
        .transpose()?;
    if dynamic.is_some() {
        for key in ["dynamic_polarizability_real", "dynamic_polarizability_imag"] {
            units.set(key, "bohr^3");
        }
        units.set("dynamic_response_residuals", "dimensionless");
        units.set("dynamic_frequency_real", "hartree");
        units.set("dynamic_frequency_imag", "hartree");
    }
    let probes = probe_points
        .as_ref()
        .map(|points| model.density_and_potential_at_points(&state, points))
        .transpose()?;
    if probes.is_some() {
        units.set("probe_electron_density", "bohr^-3");
        units.set("probe_electrostatic_potential", "hartree/e");
        units.set("charge_probe_points", "bohr");
    }
    let amplitudes: Vec<_> = model
        .local_terms
        .iter()
        .map(|term| term.amplitudes(&coordinates))
        .collect::<Result<_>>()?;
    let fp = model.fingerprints();
    let mut report = Value::object()
        .with("model", state.model_version())
        .with("converged", true)
        .with("functional", functional.name())
        .with("basis", method.basis)
        .with("derivative_method", "analytic")
        .with("geometry_condition", "clamped_geometry")
        .with(
            "nuclear_gradient_status",
            if want_gradient {
                "analytic"
            } else {
                "not_requested"
            },
        )
        .with("polarizability_parameter_derivative_status", "unsupported")
        .with("stability_status", state.stability_status())
        .with(
            "warnings",
            vec![
                "Orbital stability has not been evaluated; a nonzero gap is not a stability proof.",
            ],
        )
        .with("energy", state.energy())
        .with("classical_energy", state.classical_energy())
        .with("electronic_energy", state.electronic_energy())
        .with(
            "energy_without_internal_reference",
            state.energy_without_internal_reference(),
        )
        .with(
            "classical_energy_without_internal_reference",
            state.classical_energy_without_internal_reference(),
        )
        .with("scf_energy_convention", FRAGMENT_SCF_ENERGY_CONVENTION)
        .with("dipole", state.dipole)
        .with("polarizability", response.polarizability)
        .with("parameter_energy_gradient", response.energy_gradient)
        .with("parameter_energy_hessian", response.energy_hessian)
        .with("parameter_dipole_jacobian", response.dipole_jacobian)
        .with("parameter_names", response.parameter_names)
        .with("parameter_units", response.parameter_units)
        .with("charge_coordinates", coordinates)
        .with("charge_bounds", bounds)
        .with("charge_scales", scales)
        .with(
            "charge_directions",
            model
                .chart
                .parameters
                .iter()
                .map(|p| p.charge_direction.clone())
                .collect::<Vec<_>>(),
        )
        .with("physical_nuclear_charges", state.nuclear_charges())
        .with("physical_total_charge", state.total_charge())
        .with("fixed_electron_count", electron_count)
        .with("fragment_internal_energy", fragment_internal_energy)
        .with(
            "local_potential_amplitudes",
            amplitudes.iter().map(|v| v.0).collect::<Vec<_>>(),
        )
        .with(
            "steric_strengths",
            amplitudes.iter().map(|v| v.1).collect::<Vec<_>>(),
        )
        .with(
            "fixed_spin_electron_counts",
            state.spin_electron_counts().to_vec(),
        )
        .with(
            "reference",
            if state.scf.unrestricted { "uks" } else { "rks" },
        )
        .with("spin_squared", state.scf.s_squared)
        .with("input_formal_charge", method.charge)
        .with("charge_origin", origin)
        .with("field", field)
        .with("scf_iterations", state.diagnostics.iterations)
        .with("scf_orbital_residual", state.diagnostics.orbital_residual)
        .with("scf_eigen_residual", state.diagnostics.eigen_residual)
        .with("response_residuals", response.response_residuals)
        .with("frontier_gap", state.frontier_gap())
        .with("ao_condition_number", model.overlap_condition_number)
        .with("hessian_symmetry_defect", response.hessian_symmetry_defect)
        .with(
            "polarizability_symmetry_defect",
            response.polarizability_symmetry_defect,
        )
        .with("mixed_derivative_defect", response.mixed_derivative_defect)
        .with("geometry_fingerprint", format!("{:016x}", fp.geometry))
        .with("basis_fingerprint", format!("{:016x}", fp.basis))
        .with("grid_fingerprint", format!("{:016x}", fp.grid))
        .with("chart_fingerprint", format!("{:016x}", fp.chart))
        .with("units", units);
    if let Some(dynamic) = dynamic {
        let frequency = dynamic.value.frequency.energy()?;
        report.set(
            "dynamic_polarizability_real",
            dynamic.value.alpha.map(|row| row.map(|z| z.re)).to_vec(),
        );
        report.set(
            "dynamic_polarizability_imag",
            dynamic.value.alpha.map(|row| row.map(|z| z.im)).to_vec(),
        );
        report.set("dynamic_response_residuals", dynamic.residuals);
        report.set("dynamic_frequency_real", frequency.re);
        report.set("dynamic_frequency_imag", frequency.im);
        report.set("dynamic_response_model", dynamic.model);
    }
    if let Some((rho, potential)) = probes {
        report.set("charge_probe_points", probe_points.unwrap());
        report.set("probe_electron_density", rho);
        report.set("probe_electrostatic_potential", potential);
    }
    if let Some(gradient) = gradient {
        let forces: Vec<_> = gradient.iter().map(|&g| g * (-1.0)).collect();
        report.set("gradient", gradient);
        report.set("forces", forces);
    }
    Ok(report)
}

fn invalid<T>(message: &str) -> Result<T> {
    Err(DftError::InvalidInput(message.into()))
}
fn unsupported<T>(message: &str) -> Result<T> {
    Err(DftError::Unsupported {
        feature: "continuous chemical-species model combination".into(),
        missing: message.into(),
    })
}
fn finite_vector(v: Vec3) -> bool {
    (0..3).all(|i| v.get(i).is_finite())
}

fn validate_geometry(
    mol: &Molecule,
    basis: &MolBasis,
    grid: &MolecularGrid,
    origin: Vec3,
) -> Result<()> {
    if !finite_vector(origin)
        || mol
            .atoms
            .iter()
            .any(|a| !(1..=118).contains(&a.z) || !finite_vector(a.position))
    {
        return invalid("fixed geometries require finite coordinates and valid element labels");
    }
    for i in 0..mol.len() {
        for j in i + 1..mol.len() {
            if (mol.atoms[i].position - mol.atoms[j].position).norm() < 1e-8 {
                return invalid("coincident nuclear centers are outside the charge chart domain");
            }
        }
    }
    if basis.shells.iter().any(|s| {
        s.atom >= mol.len()
            || s.center != mol.atoms[s.atom].position
            || s.exps.is_empty()
            || s.exps.len() != s.coefs.len()
            || s.exps.iter().any(|x| !x.is_finite() || *x <= 0.0)
            || s.coefs.iter().any(|x| !x.is_finite())
    }) {
        return invalid(
            "the AO template must be finite and centered on the fixed molecular geometry",
        );
    }
    if grid.is_empty()
        || grid.points.len() != grid.weights.len()
        || grid.raw_weights.len() != grid.len()
        || grid.atom_of.len() != grid.len()
        || grid.points.iter().any(|&p| !finite_vector(p))
        || grid
            .weights
            .iter()
            .chain(&grid.raw_weights)
            .any(|x| !x.is_finite())
        || grid.atom_of.iter().any(|&a| a >= mol.len())
    {
        return invalid("the fixed molecular quadrature is empty or inconsistent");
    }
    Ok(())
}

fn linear_combination(matrices: &[Matrix], coefficients: &[f64], dimension: usize) -> Matrix {
    let mut result = Matrix::zeros(dimension, dimension);
    for (m, &c) in matrices.iter().zip(coefficients) {
        for (v, &x) in result.as_mut_slice().iter_mut().zip(m.as_slice()) {
            *v += c * x;
        }
    }
    result
}

fn nuclear_repulsion(mol: &Molecule, charges: &[f64]) -> f64 {
    let mut energy = 0.0;
    for i in 0..mol.len() {
        for j in i + 1..mol.len() {
            energy +=
                charges[i] * charges[j] / (mol.atoms[i].position - mol.atoms[j].position).norm();
        }
    }
    energy
}

// Stable FNV-1a content identifiers. These are cache/compatibility fingerprints,
// not cryptographic hashes. Length prefixes distinguish differently shaped data.
struct Fingerprint(u64);
impl Fingerprint {
    fn new() -> Self {
        Self(0xcbf29ce484222325)
    }
    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x100000001b3);
        }
    }
    fn number(&mut self, x: usize) {
        self.bytes(&(x as u64).to_le_bytes());
    }
    fn float(&mut self, x: f64) {
        self.bytes(&x.to_bits().to_le_bytes());
    }
    fn vector(&mut self, v: Vec3) {
        for i in 0..3 {
            self.float(v.get(i));
        }
    }
    fn string(&mut self, s: &str) {
        self.number(s.len());
        self.bytes(s.as_bytes());
    }
}

#[allow(clippy::too_many_arguments)]
fn fingerprints(
    mol: &Molecule,
    basis: &MolBasis,
    grid: &MolecularGrid,
    chart: &ChargeChart,
    electrons: usize,
    unrestricted: bool,
    origin: Vec3,
    functional: &ResolvedFunctional,
    ionic_charge_policy: &IonicChargePolicy,
) -> SpeciesFingerprints {
    let mut geometry = Fingerprint::new();
    geometry.number(mol.len());
    for atom in &mol.atoms {
        geometry.number(atom.z as usize);
        geometry.vector(atom.position);
    }
    let mut ao = Fingerprint::new();
    ao.number(basis.nao_sph);
    ao.number(basis.nao_cart);
    ao.number(basis.shells.len());
    for shell in &basis.shells {
        ao.number(shell.l as usize);
        ao.number(shell.atom);
        ao.vector(shell.center);
        ao.number(shell.sph_off);
        ao.number(shell.cart_off);
        ao.number(shell.exps.len());
        for &x in shell.exps.iter().chain(&shell.coefs) {
            ao.float(x);
        }
    }
    let mut quadrature = Fingerprint::new();
    quadrature.number(grid.len());
    for i in 0..grid.len() {
        quadrature.vector(grid.points[i]);
        quadrature.float(grid.weights[i]);
        quadrature.float(grid.raw_weights[i]);
        quadrature.number(grid.atom_of[i]);
    }
    let mut parameter = Fingerprint::new();
    parameter.bytes(b"continuous-species-v1-clamped-geometry");
    parameter.bytes(&geometry.0.to_le_bytes());
    parameter.bytes(&ao.0.to_le_bytes());
    parameter.bytes(&quadrature.0.to_le_bytes());
    parameter.number(electrons);
    parameter.number(mol.multiplicity);
    parameter.number(usize::from(unrestricted));
    parameter.vector(origin);
    parameter.string(&functional.description);
    parameter.number(usize::from(chart.conserve_total_charge));
    for &z in &chart.reference_charges {
        parameter.float(z);
    }
    parameter.number(chart.parameters.len());
    for p in &chart.parameters {
        parameter.string(&p.name);
        parameter.string(&p.unit);
        parameter.float(p.scale);
        for &x in p.bounds.iter().chain(&p.charge_direction) {
            parameter.float(x);
        }
    }
    // Preserve every ordinary model fingerprint exactly. Scoped role models
    // additionally encode the complete mask, including an all-false mask.
    if let IonicChargePolicy::AuxiliaryZeros(mask) = ionic_charge_policy {
        parameter.string("scoped-zero-ionic-auxiliary-roles-v1");
        parameter.number(mask.len());
        for &aux in mask {
            parameter.number(usize::from(aux));
        }
    }
    SpeciesFingerprints {
        geometry: geometry.0,
        basis: ao.0,
        grid: quadrature.0,
        chart: parameter.0,
    }
}

#[cfg(test)]
mod auxiliary_role_tests {
    use super::*;
    use crate::basis::data;
    use crate::grid::GridOptions;
    use crate::system::Atom;

    fn chart() -> ChargeChart {
        ChargeChart {
            reference_charges: vec![1.0, 1.0, 0.0],
            parameters: vec![ChargeParameter {
                name: "redistribution".into(),
                unit: "dimensionless".into(),
                scale: 1.0,
                bounds: [-1.0, 1.0],
                charge_direction: vec![2.0, -2.0, 0.0],
            }],
            conserve_total_charge: true,
        }
    }

    #[test]
    fn strict_chart_stays_positive_and_scoped_mask_is_exact() {
        let c = chart();
        let policy = IonicChargePolicy::AuxiliaryZeros(vec![false, false, true]);
        assert!(c.validate(3).is_err());
        assert!(c.charges(&[0.0]).is_err());
        assert_eq!(
            c.charges_with_policy(&[0.1], &policy).unwrap(),
            [1.2, 0.8, 0.0]
        );
        assert!(c.charges_with_policy(&[0.5], &policy).is_err());
        assert!(c.charges_with_policy(&[f64::NAN], &policy).is_err());
        assert!(c.charges_with_policy(&[2.0], &policy).is_err());
        for mask in [
            vec![false, true],
            vec![false, false, false],
            vec![true, false, true],
        ] {
            assert!(c
                .validate_with_policy(3, &IonicChargePolicy::AuxiliaryZeros(mask))
                .is_err());
        }
        let mut wrong = c.clone();
        wrong.reference_charges[2] = 1.0;
        assert!(wrong.validate_with_policy(3, &policy).is_err());
        wrong = c.clone();
        wrong.parameters[0].charge_direction[2] = 0.1;
        assert!(wrong.validate_with_policy(3, &policy).is_err());
        wrong = c.clone();
        wrong.reference_charges[0] = -0.1;
        assert!(wrong.validate_with_policy(3, &policy).is_err());
    }

    fn inputs() -> (Molecule, MolBasis, MolecularGrid) {
        let mol = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(1.5, 0.3, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(-0.4, 1.3, 0.7),
            },
        ]);
        let basis = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
        let grid = MolecularGrid::build(
            &mol,
            &GridOptions {
                level: 0,
                weight_cutoff: 0.0,
                ..GridOptions::default()
            },
        )
        .unwrap();
        (mol, basis, grid)
    }

    #[test]
    fn ordinary_constructors_reject_zero_ions_and_scoped_counts_are_explicit() {
        let (m, b, g) = inputs();
        assert!(ContinuousSpeciesModel::new(
            m,
            b,
            g,
            2,
            chart(),
            SpeciesFunctional::Lda,
            ScfOptions::default(),
            Vec3::zero()
        )
        .is_err());
        let (m, b, g) = inputs();
        assert!(ContinuousSpeciesModel::new_with_spin(
            m,
            b,
            g,
            [1, 0],
            chart(),
            SpeciesFunctional::Lda,
            ScfOptions::default(),
            Vec3::zero()
        )
        .is_err());
        let (m, b, g) = inputs();
        let closed = ContinuousSpeciesModel::new_with_auxiliary_roles(
            m,
            b,
            g,
            2,
            chart(),
            SpeciesFunctional::Lda,
            ScfOptions::default(),
            Vec3::zero(),
            vec![false, false, true],
        )
        .unwrap();
        assert_eq!(closed.electron_count, 2);
        assert_eq!(closed.molecule.charge, 1); // label sum 3 minus declared Ne 2.
        assert_eq!(closed.molecule.n_electrons().unwrap(), 2);
        assert_eq!(
            closed
                .chart
                .charges_with_policy(&[0.0], &closed.ionic_charge_policy)
                .unwrap(),
            [1.0, 1.0, 0.0]
        );
        let (m, b, g) = inputs();
        let open = ContinuousSpeciesModel::new_with_spin_and_auxiliary_roles(
            m,
            b,
            g,
            [1, 0],
            chart(),
            SpeciesFunctional::Lda,
            ScfOptions::default(),
            Vec3::zero(),
            vec![false, false, true],
        )
        .unwrap();
        assert_eq!(open.electron_count, 1);
        assert_eq!(open.molecule.charge, 2);
        assert_eq!(open.molecule.multiplicity, 2);
        assert!(matches!(open.options.reference, Reference::Uhf));
    }

    #[test]
    fn all_positive_identity_preserves_operators_and_separates_scoped_fingerprint() {
        let mut c = chart();
        c.reference_charges[2] = 1.0;
        let (m, b, g) = inputs();
        let ordinary = ContinuousSpeciesModel::new(
            m,
            b,
            g,
            2,
            c.clone(),
            SpeciesFunctional::Lda,
            ScfOptions::default(),
            Vec3::zero(),
        )
        .unwrap();
        let (m, b, g) = inputs();
        let scoped = ContinuousSpeciesModel::new_with_auxiliary_roles(
            m,
            b,
            g,
            2,
            c,
            SpeciesFunctional::Lda,
            ScfOptions::default(),
            Vec3::zero(),
            vec![false; 3],
        )
        .unwrap();
        assert_eq!(ordinary.fingerprints.geometry, scoped.fingerprints.geometry);
        assert_eq!(ordinary.fingerprints.basis, scoped.fingerprints.basis);
        assert_eq!(ordinary.fingerprints.grid, scoped.fingerprints.grid);
        assert_ne!(ordinary.fingerprints.chart, scoped.fingerprints.chart);
        for (a, b) in ordinary.unit_attraction.iter().zip(&scoped.unit_attraction) {
            assert_eq!(a.as_slice(), b.as_slice());
        }
        for (a, b) in ordinary
            .parameter_operators
            .iter()
            .zip(&scoped.parameter_operators)
        {
            assert_eq!(a.as_slice(), b.as_slice());
        }
    }
}
