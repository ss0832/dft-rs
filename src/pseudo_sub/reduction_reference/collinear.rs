// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//! Independent fixed spin sectors over the shared frozen scalar/SCF machinery.
use super::*;

pub struct CollinearFrozenEvaluation {
    pub energy: f64,
    pub electronic_energy: f64,
    pub one_electron_energy: f64,
    pub hartree_energy: f64,
    pub xc_energy: f64,
    pub nuclear_energy: f64,
    pub reference_energy: f64,
    /// Derivatives with respect to unit-occupation alpha/beta AO densities.
    pub fock: [Matrix; 2],
    pub active_electrons: [f64; 2],
    pub frozen_electrons: [f64; 2],
    pub grid_total_electrons: f64,
}
impl scf::ScalarEvaluation for CollinearFrozenEvaluation {
    fn energy(&self) -> f64 {
        self.energy
    }
    fn focks(&self) -> Vec<&Matrix> {
        vec![&self.fock[0], &self.fock[1]]
    }
}

/// Fixed integer spin populations and metric complements. Distinct frozen
/// spin subspaces must remain separate even if their ranks happen to agree.
pub struct CollinearActiveSpaceScf<'a> {
    pub overlap: &'a Matrix,
    pub complements: [&'a Matrix; 2],
    pub frozen_orbitals: [&'a Matrix; 2],
    pub initial_hcore: [&'a Matrix; 2],
    pub active_electrons: [usize; 2],
    pub options: &'a ScfOptions,
}
pub struct CollinearFrozenScf {
    pub active_density: [Matrix; 2],
    pub active_orbitals: [Matrix; 2],
    pub orbital_energies: [Vec<f64>; 2],
    pub evaluation: CollinearFrozenEvaluation,
    pub history: Vec<FrozenIteration>,
    pub projected_orbital_residual: f64,
    pub active_idempotency_error: [f64; 2],
    pub active_frozen_overlap: [f64; 2],
    pub particle_number_error: [f64; 2],
}
impl CollinearActiveSpaceScf<'_> {
    pub fn solve(
        &self,
        initial: Option<[&Matrix; 2]>,
        mut evaluate: impl FnMut([&Matrix; 2]) -> Result<CollinearFrozenEvaluation>,
    ) -> Result<CollinearFrozenScf> {
        if self.active_electrons == [0, 0] {
            return invalid("collinear active scalar requires at least one active electron");
        }
        let channels = std::array::from_fn::<_, 2, _>(|s| scf::Channel {
            overlap: self.overlap,
            complement: self.complements[s],
            frozen: self.frozen_orbitals[s],
            core: self.initial_hcore[s],
            occupied: self.active_electrons[s],
            occupation: 1.,
        });
        let result = scf::solve(
            &channels,
            self.options,
            initial.as_ref().map(|p| &p[..]),
            |p| evaluate([&p[0], &p[1]]),
        )?;
        let mut channels = result.channels.into_iter();
        let alpha = channels.next().unwrap();
        let beta = channels.next().unwrap();
        Ok(CollinearFrozenScf {
            active_density: [alpha.density, beta.density],
            active_orbitals: [alpha.orbitals, beta.orbitals],
            orbital_energies: [alpha.energies, beta.energies],
            evaluation: result.evaluation,
            history: result.history,
            projected_orbital_residual: result.residual,
            active_idempotency_error: [alpha.idempotency, beta.idempotency],
            active_frozen_overlap: [alpha.frozen_overlap, beta.frozen_overlap],
            particle_number_error: [alpha.particle_error, beta.particle_error],
        })
    }
}
