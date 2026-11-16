// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Divide-and-conquer Kohn-Sham of a periodic cell at Gamma.
//!
//! The Bloch sums at Gamma make the cell a closed problem in a real basis,
//! so the divide-and-conquer iteration of [`crate::scf::divide_conquer`]
//! applies to it unchanged: subsystems are fragments with the atoms within
//! the buffer radius by the minimum image, their blocks of the Kohn-Sham and
//! overlap matrices of the cell are diagonalized, and the density is
//! assembled with a common Fermi level. The Kohn-Sham matrices and the
//! kernel of the adjoint equation come from the integral series of
//! [`crate::periodic::path_series`] at the reference geometry, and forces
//! and stress from their first-order coefficients along a displacement or a
//! deformation of the lattice,
//!
//! ```text
//! dA/dx = E^(x)[P] + sum_s Tr[Z_s F_s^(x)] - Tr[W S^(x)].
//! ```

use super::lattice::Setup;
use crate::linalg::Matrix;
use crate::periodic::path_series::LatticePath;
use crate::periodic::scf::PeriodicScfOptions;
use crate::periodic::Cell;
use crate::response::krylov::ResponseSolverConfig;
use crate::scf::divide_conquer::{
    iterate, relaxation, subsystems_with, Build, DivideConquerOptions, Iteration, IterationInput,
    RelaxationInput,
};
use crate::scf::Reference;
use crate::xc::ResolvedFunctional;
use crate::Result;

/// Converged divide-and-conquer state of a cell.
#[derive(Clone, Debug)]
pub struct LatticeDivideConquer {
    /// Free energy `E - kT S` per cell, Hartree.
    pub energy: f64,
    /// Kohn-Sham energy without the entropy term.
    pub internal_energy: f64,
    /// `S / k_B`.
    pub entropy: f64,
    pub chemical_potentials: [f64; 2],
    pub density: [Matrix; 2],
    /// `-dA/du` on every atom, Hartree/Bohr.
    pub forces: Option<Vec<[f64; 3]>>,
    /// `dA / d eps_ab` for the deformation `1 + eps`, Hartree.
    pub strain_derivative: Option<[[f64; 3]; 3]>,
    /// Its symmetric part over the volume, Hartree/Bohr^3.
    pub stress: Option<[[f64; 3]; 3]>,
    /// Cell volume, Bohr^3.
    pub volume: f64,
    pub converged: bool,
    pub iterations: usize,
    /// Largest element of the last density change.
    pub density_residual: f64,
    /// Relative residual of the adjoint equation.
    pub relaxation_residual: Option<f64>,
    /// Fragment atoms, buffer atoms and basis functions of every subsystem.
    pub subsystem_sizes: Vec<[usize; 3]>,
    pub n_alpha: usize,
    pub n_beta: usize,
    pub unrestricted: bool,
}

/// Divide-and-conquer self-consistent field of `cell` at Gamma with the
/// partition `dc`, and on request the forces and the stress of its free
/// energy with the response of the partition.
pub fn lattice_divide_conquer(
    cell: &Cell,
    functional: &ResolvedFunctional,
    options: &PeriodicScfOptions,
    dc: &DivideConquerOptions,
    forces: bool,
    stress: bool,
    config: &ResponseSolverConfig,
) -> Result<LatticeDivideConquer> {
    config.validate()?;
    let setup = Setup::for_cell(cell, functional, options)?;
    let molecule = cell.molecule();
    let basis = cell.basis();
    let lattice = cell.lattice();
    let natoms = molecule.len();
    let nao = basis.nao_sph;
    let parts = subsystems_with(natoms, basis, dc, &|a, b| {
        lattice
            .minimum_image(molecule.atoms[a].position - molecule.atoms[b].position)
            .norm()
    })?;
    let electrons = cell.n_electrons()?;
    let (n_alpha, n_beta) = molecule.n_alpha_beta_for(electrons)?;
    let restricted = n_alpha == n_beta && options.reference != Reference::Uhf;
    let geometry = setup.geometry(&LatticePath::default(), 0)?;
    let overlap = geometry.overlap[0].clone();
    let build = |density: &[Matrix; 2]| -> Result<Build> {
        let p = [vec![density[0].clone()], vec![density[1].clone()]];
        let evaluation = geometry.evaluate(functional, &p, 0)?;
        let energy = evaluation.energy[0].clone();
        let [mut first, mut second] = evaluation.fock;
        Ok(Build {
            fock: [first.swap_remove(0), second.swap_remove(0)],
            electronic: energy.total() - energy.nuclear_repulsion - energy.nuclear_field,
            exc: energy.semilocal_xc,
            grid_electrons: None,
        })
    };
    let start = crate::scf::sad::sad_density(molecule, basis)?;
    let Iteration {
        output,
        last,
        converged,
        iterations,
        residual,
    } = iterate(&IterationInput {
        parts: &parts,
        overlap: &overlap,
        build: &build,
        start: &start,
        electrons: [n_alpha, n_beta],
        restricted,
        kt: dc.kt,
        orth_threshold: options.orth_threshold,
        max_iter: options.max_iter,
        density_tolerance: options.conv_tol_grad,
        energy_tolerance: options.e_tol,
    })?;
    let density = [output[0].density.clone(), output[1].density.clone()];
    let p0 = [vec![density[0].clone()], vec![density[1].clone()]];
    let internal_energy = geometry.evaluate(functional, &p0, 0)?.energy[0].total();
    let entropy = output[0].entropy + output[1].entropy;
    let volume = lattice.volume().abs();
    let mut out_forces = None;
    let mut strain_derivative = None;
    let mut out_stress = None;
    let mut relaxation_residual = None;
    if forces || stress {
        let kernel = |z: &[Matrix; 2]| -> Result<[Matrix; 2]> {
            let q = [vec![z[0].clone()], vec![z[1].clone()]];
            let [mut first, mut second] = geometry.evaluate_linear(functional, &p0, &q, 0)?;
            Ok([first.swap_remove(0), second.swap_remove(0)])
        };
        let (z, weighted, adjoint_residual) = relaxation(
            &RelaxationInput {
                parts: &parts,
                overlap: &overlap,
                fock: &last.fock,
                density: &density,
                chemical_potentials: [
                    Some(output[0].chemical_potential),
                    Some(output[1].chemical_potential),
                ],
                unrestricted: !restricted,
                kt: dc.kt,
                orth_threshold: options.orth_threshold,
                kernel: &kernel,
            },
            config,
        )?;
        relaxation_residual = Some(adjoint_residual);
        // First-order coefficient of the free energy along a path.
        let derivative = |line: &LatticePath| -> Result<f64> {
            let moving = setup.geometry(line, 1)?;
            let p: [Vec<Matrix>; 2] =
                std::array::from_fn(|s| vec![density[s].clone(), Matrix::zeros(nao, nao)]);
            let evaluation = moving.evaluate(functional, &p, 1)?;
            let mut value = evaluation.energy[1].total();
            for s in 0..2 {
                value += z[s].frobenius_dot(&evaluation.fock[s][1]);
            }
            Ok(value - weighted.frobenius_dot(&moving.overlap[1]))
        };
        if forces {
            let mut list = vec![[0.0; 3]; natoms];
            for (atom, force) in list.iter_mut().enumerate() {
                for (axis, value) in force.iter_mut().enumerate() {
                    let mut direction = vec![[0.0; 3]; natoms];
                    direction[atom][axis] = 1.0;
                    *value = -derivative(&LatticePath::displacement(&direction))?;
                }
            }
            out_forces = Some(list);
        }
        if stress {
            let mut first = [[0.0; 3]; 3];
            for a in 0..3 {
                for b in 0..3 {
                    let mut step = [[0.0; 3]; 3];
                    step[a][b] = 1.0;
                    first[a][b] = derivative(&LatticePath {
                        deformation: vec![step],
                        ..LatticePath::default()
                    })?;
                }
            }
            let mut symmetric = [[0.0; 3]; 3];
            for a in 0..3 {
                for b in 0..3 {
                    symmetric[a][b] = 0.5 * (first[a][b] + first[b][a]) / volume;
                }
            }
            strain_derivative = Some(first);
            out_stress = Some(symmetric);
        }
    }
    Ok(LatticeDivideConquer {
        energy: internal_energy - dc.kt * entropy,
        internal_energy,
        entropy,
        chemical_potentials: [output[0].chemical_potential, output[1].chemical_potential],
        density,
        forces: out_forces,
        strain_derivative,
        stress: out_stress,
        volume,
        converged,
        iterations,
        density_residual: residual,
        relaxation_residual,
        subsystem_sizes: parts
            .iter()
            .map(|part| {
                [
                    part.core_atoms.len(),
                    part.buffer_atoms.len(),
                    part.functions.len(),
                ]
            })
            .collect(),
        n_alpha,
        n_beta,
        unrestricted: !restricted,
    })
}
