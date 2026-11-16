// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Variational molecular SCF restricted to a fixed positive Dirac sector.

use crate::dfpt::orbitals::{adjoint, generalized_orbitals, matmul};
use crate::integrals::analytic::GaussianKernel;
use crate::integrals::analytic_basis::{AnalyticShell, IntegralBlock};
use crate::integrals::dirac::{
    electronic_coulomb_exchange, electronic_current_exchange, london_magnetic_balance,
    restricted_kinetic_balance, BalancedDiracIntegrals, DiracCoulombFields,
    DiracCurrentInteraction, GaussianNucleus, LondonCoulombTaylor, LondonCurrentTaylor,
};
use crate::linalg::Matrix;
use crate::periodic::occupations::OccupationScheme;
use crate::periodic::spinor_scf::{SpinorEnergyBackend, SpinorFields};
use crate::periodic::BlochMatrix;
use crate::{DftError, Result};

/// Two-electron operator for the fixed positive-energy sector.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DiracInteraction {
    #[default]
    Coulomb,
    Gaunt,
    Breit,
}

impl DiracInteraction {
    pub(crate) fn current(self) -> Option<DiracCurrentInteraction> {
        match self {
            Self::Coulomb => None,
            Self::Gaunt => Some(DiracCurrentInteraction::Gaunt),
            Self::Breit => Some(DiracCurrentInteraction::Breit),
        }
    }
}

/// All-electron four-component Hartree-Fock. The optional current terms
/// enter both the stationary energy and variational Fock.
/// A London-RMB basis modifies every two-electron operator and regenerates
/// the nuclear-core sector at the selected magnetic field.
pub struct DiracCoulombBackend<'a> {
    shells: &'a [AnalyticShell<f64>],
    speed_of_light: f64,
    electrons: usize,
    interaction: DiracInteraction,
    overlap: [BlochMatrix; 1],
    core: [BlochMatrix; 1],
    potential: BlochMatrix,
    nuclear_repulsion: f64,
    london_coulomb: Option<LondonCoulombTaylor>,
    london_current: Option<(DiracCurrentInteraction, LondonCurrentTaylor)>,
}

impl<'a> DiracCoulombBackend<'a> {
    pub fn new(
        shells: &'a [AnalyticShell<f64>],
        nuclei: &[GaussianNucleus<f64>],
        speed_of_light: f64,
        electrons: usize,
    ) -> Result<Self> {
        let integrals = restricted_kinetic_balance(shells, nuclei, speed_of_light, 0)?;
        Self::from_integrals(shells, speed_of_light, electrons, integrals, None)
    }

    /// Variational four-component Dirac–Coulomb HF in a finite uniform field.
    /// The London plane-wave expansion order and its retained integral
    /// budget are explicit. The Gaussian nuclear charge is spin independent.
    pub fn new_london(
        shells: &'a [AnalyticShell<f64>],
        nuclei: &[GaussianNucleus<f64>],
        speed_of_light: f64,
        electrons: usize,
        magnetic_field: [f64; 3],
        phase_order: usize,
        pair_cache_mb: usize,
    ) -> Result<Self> {
        Self::new_london_interaction(
            shells,
            nuclei,
            speed_of_light,
            electrons,
            magnetic_field,
            phase_order,
            pair_cache_mb,
            DiracInteraction::Coulomb,
        )
    }

    /// Fix the field-dependent Dirac sector and retain both the Coulomb and
    /// optional current interaction data for the variational magnetic HF.
    pub fn new_london_interaction(
        shells: &'a [AnalyticShell<f64>],
        nuclei: &[GaussianNucleus<f64>],
        speed_of_light: f64,
        electrons: usize,
        magnetic_field: [f64; 3],
        phase_order: usize,
        pair_cache_mb: usize,
        interaction: DiracInteraction,
    ) -> Result<Self> {
        let cache_mb = pair_cache_mb / (1 + usize::from(interaction.current().is_some()));
        let integrals = london_magnetic_balance(
            shells,
            nuclei,
            speed_of_light,
            magnetic_field,
            0,
            phase_order,
        )?;
        let coulomb = LondonCoulombTaylor::new(
            shells,
            speed_of_light,
            GaussianKernel::Coulomb,
            magnetic_field,
            [0.0; 3],
            0,
            phase_order,
            cache_mb,
        )?;
        let mut backend =
            Self::from_integrals(shells, speed_of_light, electrons, integrals, Some(coulomb))?;
        backend.london_current = interaction
            .current()
            .map(|current| {
                let cache = LondonCurrentTaylor::new(
                    shells,
                    speed_of_light,
                    current,
                    magnetic_field,
                    [0.0; 3],
                    0,
                    phase_order,
                    cache_mb,
                )?;
                Ok::<_, DftError>((current, cache))
            })
            .transpose()?;
        backend.interaction = interaction;
        Ok(backend)
    }

    fn from_integrals(
        shells: &'a [AnalyticShell<f64>],
        speed_of_light: f64,
        electrons: usize,
        integrals: BalancedDiracIntegrals<f64>,
        london_coulomb: Option<LondonCoulombTaylor>,
    ) -> Result<Self> {
        let matrix = |block: IntegralBlock<f64, 2>| {
            Matrix::from_row_major(block.shape[0], block.shape[1], block.values)
        };
        let [vre, vim] = integrals.nuclear_potential()?;
        let potential = BlochMatrix::from_pair((matrix(vre), matrix(vim)));
        let core = BlochMatrix::from_pair((
            matrix(integrals.hamiltonian_real),
            matrix(integrals.hamiltonian_imaginary),
        ));
        if electrons == 0 || electrons > core.rows() / 2 {
            return Err(DftError::InvalidInput(
                "Dirac–Coulomb HF needs a positive integer electron count within the electronic sector".into(),
            ));
        }
        let overlap = BlochMatrix::from_pair((
            matrix(integrals.overlap),
            matrix(integrals.overlap_imaginary),
        ));
        Ok(Self {
            shells,
            speed_of_light,
            electrons,
            interaction: DiracInteraction::Coulomb,
            overlap: [overlap],
            core: [core],
            potential,
            nuclear_repulsion: integrals.nuclear_repulsion,
            london_coulomb,
            london_current: None,
        })
    }

    /// Keep the same nuclear-core projector while changing the two-electron operator.
    pub fn with_interaction(mut self, interaction: DiracInteraction) -> Self {
        self.interaction = interaction;
        self
    }

    /// Direct and signed exchange corrections on the full balanced density.
    /// A pure Coulomb reference has no current-current correction.
    pub fn current_fields(&self, density: &BlochMatrix) -> Result<Option<DiracCoulombFields>> {
        let Some(current) = self.interaction.current() else {
            return Ok(None);
        };
        if self.london_coulomb.is_some() {
            let Some((cached, integrals)) = &self.london_current else {
                return Err(DftError::Unsupported {
                    feature: "four-component London-RMB current interaction".into(),
                    missing: "a London reference constructed with new_london_interaction".into(),
                });
            };
            if current != *cached {
                return Err(DftError::InvalidInput(
                    "the selected current interaction must match the retained London reference"
                        .into(),
                ));
            }
            return integrals.fields(0, density).map(Some);
        }
        electronic_current_exchange(self.shells, density, self.speed_of_light, current).map(Some)
    }

    fn two_electron_fields(&self, density: &BlochMatrix) -> Result<DiracCoulombFields> {
        let mut fields = match &self.london_coulomb {
            Some(london) => london.fields(0, density)?,
            None => electronic_coulomb_exchange(self.shells, density, self.speed_of_light)?,
        };
        if let Some(current) = self.current_fields(density)? {
            fields.direct.add_scaled(&current.direct, 1.0);
            fields.exchange.add_scaled(&current.exchange, 1.0);
            fields.direct_energy += current.direct_energy;
            fields.exchange_energy += current.exchange_energy;
            fields.hermiticity_defect = fields
                .direct
                .hermiticity_error()
                .max(fields.exchange.hermiticity_error());
        }
        Ok(fields)
    }

    pub(crate) fn density_kernel(&self, density: &BlochMatrix) -> Result<BlochMatrix> {
        let mut fields = self.two_electron_fields(density)?;
        fields.direct.add_scaled(&fields.exchange, -1.0);
        Ok(fields.direct)
    }

    /// The molecular no-pair sector generated by the fixed nuclear core.
    pub fn electronic_space(&self, overlap_cutoff: f64) -> Result<NoPairSpace> {
        NoPairSpace::new(
            &self.core[0],
            &self.overlap[0],
            -self.speed_of_light.powi(2),
            overlap_cutoff,
        )
    }
}

impl SpinorEnergyBackend for DiracCoulombBackend<'_> {
    fn overlap(&self) -> &[BlochMatrix] {
        &self.overlap
    }
    fn initial_fock(&self) -> &[BlochMatrix] {
        &self.core
    }
    fn electrons(&self) -> f64 {
        self.electrons as f64
    }
    fn quadratic_density_energy(&self) -> bool {
        true
    }
    fn evaluate(&self, density: &[BlochMatrix]) -> Result<SpinorFields> {
        let [density] = density else {
            return Err(DftError::InvalidInput(
                "molecular four-component HF needs one four-component density".into(),
            ));
        };
        let fields = self.two_electron_fields(density)?;
        let nuclear = density.trace_product(&self.potential);
        let kinetic = density.trace_product(&self.core[0]) - nuclear;
        let components = [
            self.nuclear_repulsion,
            kinetic,
            nuclear,
            0.0,
            fields.direct_energy,
            fields.exchange_energy,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let mut fock = self.core[0].clone();
        fock.add_scaled(&fields.direct, 1.0);
        fock.add_scaled(&fields.exchange, -1.0);
        let energy = components.iter().sum::<f64>();
        if !energy.is_finite()
            || fock
                .re
                .as_slice()
                .iter()
                .chain(fock.im.as_slice())
                .any(|v| !v.is_finite())
        {
            return Err(DftError::LinearAlgebra(
                "nonfinite four-component HF energy or Fock matrix".into(),
            ));
        }
        Ok(SpinorFields {
            fock: vec![fock],
            energy,
            energy_components: components,
            xc_component_energies: Vec::new(),
            nuclear_soc_energy: 0.0,
            operator_converged: true,
            coulomb_operator_defect: fields.hermiticity_defect,
            xc_grid_number_defect: 0.0,
            xc_diagnostics: None,
        })
    }
}

/// Full-rank balanced overlap and its fixed electronic-sector embedding.
/// The generating one-electron operator is held fixed during SCF. In
/// particular, a trial density cannot change the variational domain.
pub struct NoPairSpace {
    basis: BlochMatrix,
    adjoint: BlochMatrix,
    overlap: BlochMatrix,
    projected_overlap: [BlochMatrix; 1],
    /// Separation of the highest negative and lowest electronic levels.
    pub spectral_gap: f64,
    /// Raw maximum defect of the metric of the projected basis.
    pub metric_defect: f64,
}

impl NoPairSpace {
    pub fn new(
        hamiltonian: &BlochMatrix,
        overlap: &BlochMatrix,
        separator: f64,
        overlap_cutoff: f64,
    ) -> Result<Self> {
        let n = overlap.rows();
        if n == 0 || n % 4 != 0 || !separator.is_finite() {
            return Err(DftError::InvalidInput(
                "fixed no-pair space requires a four-component basis and a finite separator".into(),
            ));
        }
        let orbitals = generalized_orbitals(hamiltonian, overlap, overlap_cutoff)?;
        let first = n / 2;
        if orbitals.energies.len() != n
            || orbitals.energies[first - 1] >= separator
            || orbitals.energies[first] <= separator
        {
            return Err(DftError::InvalidInput(
                "fixed no-pair space needs a full-rank metric and a gap separating equal Dirac sectors".into(),
            ));
        }
        let mut basis = BlochMatrix::zeros(n, first);
        for i in 0..n {
            for j in 0..first {
                basis.re[(i, j)] = orbitals.coefficients.re[(i, first + j)];
                basis.im[(i, j)] = orbitals.coefficients.im[(i, first + j)];
            }
        }
        let adjoint = adjoint(&basis)?;
        let metric = matmul(&adjoint, &matmul(overlap, &basis)?)?;
        let mut metric_defect = 0.0_f64;
        for i in 0..first {
            for j in 0..first {
                let expected = if i == j { 1.0 } else { 0.0 };
                metric_defect =
                    metric_defect.max((metric.re[(i, j)] - expected).hypot(metric.im[(i, j)]));
            }
        }
        Ok(Self {
            basis,
            adjoint,
            overlap: overlap.clone(),
            projected_overlap: [metric],
            spectral_gap: orbitals.energies[first] - orbitals.energies[first - 1],
            metric_defect,
        })
    }

    /// Fixed complex columns C in the full AO basis, with C† S C = I.
    pub fn basis(&self) -> &BlochMatrix {
        &self.basis
    }

    /// Projection of a covariant one-electron operator, C† O C.
    pub fn project_operator(&self, operator: &BlochMatrix) -> Result<BlochMatrix> {
        matmul(&self.adjoint, &matmul(operator, &self.basis)?)
    }

    /// Contravariant full-AO density, C P C†.
    pub fn lift_density(&self, density: &BlochMatrix) -> Result<BlochMatrix> {
        matmul(&self.basis, &matmul(density, &self.adjoint)?)
    }
}

/// Adapts a variational full four-component energy provider to the common
/// complex SCF solver. All energy terms and diagnostics stay with that
/// provider. Only the trial density and its conjugate Fock change basis.
pub struct NoPairBackend<'a, B> {
    space: &'a NoPairSpace,
    full: &'a B,
    initial: [BlochMatrix; 1],
}

impl<'a, B: SpinorEnergyBackend> NoPairBackend<'a, B> {
    pub fn new(space: &'a NoPairSpace, full: &'a B) -> Result<Self> {
        if full.overlap() != std::slice::from_ref(&space.overlap) || full.initial_fock().len() != 1
        {
            return Err(DftError::InvalidInput(
                "the molecular no-pair backend requires the same single balanced AO metric".into(),
            ));
        }
        Ok(Self {
            space,
            full,
            initial: [space.project_operator(&full.initial_fock()[0])?],
        })
    }
}

impl<B: SpinorEnergyBackend> SpinorEnergyBackend for NoPairBackend<'_, B> {
    fn overlap(&self) -> &[BlochMatrix] {
        &self.space.projected_overlap
    }

    fn initial_fock(&self) -> &[BlochMatrix] {
        &self.initial
    }

    fn electrons(&self) -> f64 {
        self.full.electrons()
    }

    fn validate_occupation(&self, occupation: OccupationScheme) -> Result<()> {
        self.full.validate_occupation(occupation)
    }

    fn quadratic_density_energy(&self) -> bool {
        self.full.quadratic_density_energy()
    }

    fn evaluate(&self, density: &[BlochMatrix]) -> Result<SpinorFields> {
        let [density] = density else {
            return Err(DftError::InvalidInput(
                "the molecular no-pair backend requires one trial density".into(),
            ));
        };
        let mut fields = self.full.evaluate(&[self.space.lift_density(density)?])?;
        let [fock] = fields.fock.as_mut_slice() else {
            return Err(DftError::InvalidInput(
                "the molecular no-pair provider must return one Fock operator".into(),
            ));
        };
        *fock = self.space.project_operator(fock)?;
        Ok(fields)
    }
}
