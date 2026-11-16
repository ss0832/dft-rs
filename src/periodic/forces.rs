// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic periodic forces from the finite-k SCF Lagrangian.

use crate::error::Result;
use crate::math::Vec3;
use crate::periodic::gradient::electronic_gradient;
use crate::periodic::scf::{PeriodicScfOptions, PeriodicScfResult};
use crate::periodic::Cell;
use crate::xc::ResolvedFunctional;

#[derive(Clone, Debug)]
pub struct PeriodicForces {
    /// `-dF/dR`, where `F` is the Mermin free energy under Fermi-Dirac
    /// occupations and equals the total energy at zero temperature.
    pub total: Vec<Vec3>,
    /// Analytic electronic force, including Pulay, J, exact exchange and XC.
    pub electronic: Vec<Vec3>,
    /// Analytic nuclear-nuclear Ewald force.
    pub nuclear: Vec<Vec3>,
}

/// Analytic Mermin-free-energy forces for a converged periodic SCF result.
pub fn analytic_forces(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
) -> Result<PeriodicForces> {
    let electronic_gradient = crate::logger::timed("pbc-forces: electronic", || {
        electronic_gradient(cell, functional, result, options)
    })?;
    let electronic: Vec<Vec3> = electronic_gradient.iter().map(|value| -*value).collect();
    let nuclear = crate::logger::timed("pbc-forces: nuclear", || cell.nuclear_forces());
    let total = electronic
        .iter()
        .zip(&nuclear)
        .map(|(electronic, nuclear)| *electronic + *nuclear)
        .collect();
    Ok(PeriodicForces {
        total,
        electronic,
        nuclear,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;
    use crate::grid::GridOptions;
    use crate::math::Vec3;
    use crate::periodic::scf::run_periodic_scf;
    use crate::periodic::Lattice;
    use crate::system::{Atom, Molecule};
    use crate::xc::{resolve, FunctionalSpec};

    fn displaced(molecule: &Molecule, atom: usize, axis: usize, amount: f64) -> Molecule {
        let mut result = molecule.clone();
        match axis {
            0 => result.atoms[atom].position.x += amount,
            1 => result.atoms[atom].position.y += amount,
            _ => result.atoms[atom].position.z += amount,
        }
        result
    }

    #[test]
    #[ignore = "20-minute all-component finite-k physical validation gate; run explicitly"]
    fn finite_k_hybrid_analytic_force_matches_total_energy_difference() {
        // This single non-equilibrium H2 component exercises every analytic
        // contribution: one-electron and overlap Pulay terms, Ewald J,
        // finite-k exact exchange, GGA AO/point/weight response, and Ewald
        // nuclear repulsion.  The central difference is an independent
        // test oracle; it is not used by the force implementation.
        let molecule = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, -0.55),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 0.95),
            },
        ]);
        let lattice = Lattice::new(
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(0.0, 14.0, 0.0),
            Vec3::new(0.0, 0.0, 14.0),
        );
        let set = data::load("sto-3g").unwrap();
        let functional = resolve(&FunctionalSpec::Combined("pbe0".to_string())).unwrap();
        let options = PeriodicScfOptions {
            k_mesh: [3, 1, 1],
            max_iter: 20,
            damping: 0.0,
            exchange_cutoff: Some(6.1),
            xc_ao_cutoff: Some(6.1),
            grid_shell: 1,
            grid: GridOptions {
                radial_override: Some(9),
                angular_override: Some(7),
                prune: false,
                ..GridOptions::default()
            },
            ..PeriodicScfOptions::default()
        };
        let cell = Cell::new(molecule.clone(), lattice.clone(), &set, 6.1).unwrap();
        let scf = run_periodic_scf(&cell, &functional, &options).unwrap();
        assert!(scf.converged, "reference periodic SCF did not converge");
        let analytic = analytic_forces(&cell, &functional, &scf, &options).unwrap();
        let step = 3e-4;
        let energy = |geometry: Molecule| {
            let displaced_cell = Cell::new(geometry, lattice.clone(), &set, 6.1).unwrap();
            let displaced_scf = run_periodic_scf(&displaced_cell, &functional, &options).unwrap();
            assert!(
                displaced_scf.converged,
                "displaced periodic SCF did not converge"
            );
            displaced_scf.energy
        };
        let finite_difference = -(energy(displaced(&molecule, 1, 2, step))
            - energy(displaced(&molecule, 1, 2, -step)))
            / (2.0 * step);
        let error = (analytic.total[1].z - finite_difference).abs();
        assert!(
            error < 2e-4,
            "analytic Fz={} vs energy FD={} (error={error:e})",
            analytic.total[1].z,
            finite_difference
        );
        let sum = analytic.total[0] + analytic.total[1];
        assert!(sum.norm() < 2e-6, "translation sum rule: {sum:?}");
        for atom in 0..2 {
            assert!(
                (analytic.electronic[atom] + analytic.nuclear[atom] - analytic.total[atom]).norm()
                    < 1e-12
            );
        }
    }

    #[test]
    #[ignore = "multi-SCF periodic ECP analytic-force validation gate; run explicitly"]
    fn periodic_ecp_analytic_force_matches_total_energy_difference() {
        let molecule = Molecule::new(vec![
            Atom {
                z: 11,
                position: Vec3::zero(),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.2, -0.1, 2.6),
            },
        ]);
        let lattice = Lattice::cubic(18.0);
        let set = data::load("lanl2dz").unwrap();
        let functional = resolve(&FunctionalSpec::Combined("svwn5".to_string())).unwrap();
        let options = PeriodicScfOptions {
            max_iter: 50,
            e_tol: 2e-10,
            p_tol: 2e-9,
            conv_tol_grad: 2e-6,
            damping: 0.1,
            grid_shell: 1,
            grid: GridOptions {
                radial_override: Some(7),
                angular_override: Some(5),
                prune: false,
                ..GridOptions::default()
            },
            ..PeriodicScfOptions::default()
        };
        let cutoff = 5.0;
        let cell = Cell::new(molecule.clone(), lattice.clone(), &set, cutoff).unwrap();
        let scf = run_periodic_scf(&cell, &functional, &options).unwrap();
        assert!(scf.converged);
        let analytic = analytic_forces(&cell, &functional, &scf, &options).unwrap();
        let step = 3e-4;
        let energy = |geometry: Molecule| {
            let displaced_cell = Cell::new(geometry, lattice.clone(), &set, cutoff).unwrap();
            let displaced = run_periodic_scf(&displaced_cell, &functional, &options).unwrap();
            assert!(displaced.converged);
            displaced.energy
        };
        let finite_difference = -(energy(displaced(&molecule, 1, 2, step))
            - energy(displaced(&molecule, 1, 2, -step)))
            / (2.0 * step);
        let error = (analytic.total[1].z - finite_difference).abs();
        assert!(
            error < 3e-4,
            "periodic ECP analytic Fz={} vs total-energy FD={} (error={error:e})",
            analytic.total[1].z,
            finite_difference
        );
        let sum = analytic.total[0] + analytic.total[1];
        assert!(sum.norm() < 4e-6, "periodic ECP force sum: {sum:?}");
    }
}
