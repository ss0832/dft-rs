// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Molecular MBD with range-separated self-consistent screening, Ambrosetti
//! et al., J. Chem. Phys. 140, 18A508 (2014), Eqs. (7)–(15), and the
//! screened van der Waals radii of Blood-Forsythe et al., Chem. Sci. 7,
//! 1631–1639 (2016).

use super::{atom_environments, coupled_energy, volume_ratios, AtomEnvironment, FreeAtom, MbdEnergy};
use crate::basis::MolBasis;
use crate::grid::{gauss_legendre, MolecularGrid};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::scaled_erf_difference;
use crate::system::Molecule;
use crate::{DftError, Result};
use std::f64::consts::PI;

/// Post-SCF MBD@rsSCS energy and stability diagnostics. The effective
/// parameters in `oscillators.atoms` are the screened ones.
#[derive(Clone, Debug)]
pub struct MbdRsScsEnergy {
    pub oscillators: MbdEnergy,
    pub unscreened_atoms: Vec<AtomEnvironment>,
    /// Smallest eigenvalue of the scaled short-range response matrix, over
    /// zero and all imaginary frequency nodes (dimensionless).
    pub minimum_screening_eigenvalue: f64,
    /// Largest residual of the three uniform-field screening equations.
    pub screening_residual: f64,
}

struct Pair {
    i: usize,
    j: usize,
    distance: f64,
    direction: [f64; 3],
    short_range_fraction: f64,
}

impl Pair {
    // The mixed Cartesian derivative of erf(r/sigma)/r. Dividing the
    // erf difference by x³ keeps the Gaussian coincidence limit finite.
    fn short_range_tensor(&self, sigma: f64) -> [[f64; 3]; 3] {
        let x = self.distance / sigma;
        let radial = scaled_erf_difference(x);
        let longitudinal = (4.0 / PI.sqrt()) * (-x * x).exp() - 3.0 * radial;
        let scale = self.short_range_fraction / sigma.powi(3);
        std::array::from_fn(|i| std::array::from_fn(|j| {
            scale * (f64::from(i == j) * radial + longitudinal * self.direction[i] * self.direction[j])
        }))
    }
}

fn pairs(molecule: &Molecule, atoms: &[AtomEnvironment], beta: f64) -> Result<Vec<Pair>> {
    let mut pairs = Vec::new();
    for (i, atom) in atoms.iter().enumerate() {
        for (j, other) in atoms[..i].iter().enumerate() {
            let displacement = molecule.atoms[i].position - molecule.atoms[j].position;
            let distance = displacement.norm();
            let radius = beta * (atom.radius + other.radius);
            if !distance.is_finite() || distance <= 0.0 || !radius.is_finite() || radius <= 0.0 {
                return Err(DftError::InvalidInput(format!("atoms {i} and {j} have invalid MBD screening pair data")));
            }
            pairs.push(Pair { i, j, distance, direction: (displacement * distance.recip()).to_array(),
                short_range_fraction: super::super::logistic_damping(distance, radius, -6.0) });
        }
    }
    Ok(pairs)
}

fn screen(pairs: &[Pair], alpha: &[f64], frequency: f64) -> Result<(Vec<f64>, f64, f64)> {
    let roots: Vec<_> = alpha.iter().map(|a| a.sqrt()).collect();
    let width: Vec<_> = alpha.iter().map(|a| ((2.0 / PI).sqrt() * a / 3.0).cbrt()).collect();
    let dimension = 3 * alpha.len();
    let mut operator = Matrix::identity(dimension);
    for pair in pairs {
        let (i,j) = (pair.i,pair.j);
        let tensor = pair.short_range_tensor(width[i].hypot(width[j]));
        for (a, row) in tensor.iter().enumerate() {
            for (b, &element) in row.iter().enumerate() {
                let value = roots[i] * element * roots[j];
                operator[(3*i+a,3*j+b)] = value;
                operator[(3*j+b,3*i+a)] = value;
            }
        }
    }
    if operator.as_slice().iter().any(|v| !v.is_finite()) {
        return Err(DftError::InvalidInput("nonfinite MBD@rsSCS short-range response operator".into()));
    }
    let (spectrum, vectors) = symmetric_eigen(&operator)?;
    let lowest = spectrum[0];
    if lowest <= 0.0 {
        return Err(DftError::InvalidInput(format!("unstable MBD@rsSCS short-range modes at {frequency:e} Hartree (minimum eigenvalue {lowest:e})")));
    }
    // Solve (I + sqrt(alpha) T_SR sqrt(alpha)) X = sqrt(alpha) E.
    // Uniform fields along the three axes give the contracted rows of the
    // screened response, avoiding storage of the full inverse.
    let rhs = Matrix::from_row_major(dimension,3,(0..dimension).flat_map(|i| {
        let a = roots[i/3];
        (0..3).map(move |j| if i%3 == j { a } else { 0.0 })
    }).collect());
    let mut projected = vectors.transpose_matmul_seq(&rhs);
    for (mode, value) in spectrum.into_iter().enumerate() {
        for axis in 0..3 {
            projected[(mode,axis)] /= value;
        }
    }
    let response = vectors.matmul_seq(&projected);
    let residual = operator.matmul_seq(&response).as_slice().iter().zip(rhs.as_slice())
        .map(|(a,b)| (a-b).abs()).fold(0.0,f64::max) / roots.iter().copied().fold(1.0,f64::max);
    let isotropic: Vec<_> = roots.iter().enumerate().map(|(i, root)| {
        (0..3).map(|axis| root * response[(3*i+axis,axis)]).sum::<f64>() / 3.0
    }).collect();
    if isotropic.iter().any(|a| !a.is_finite() || *a<=0.0) || !residual.is_finite() {
        return Err(DftError::InvalidInput(format!("invalid screened MBD polarizability at imaginary frequency {frequency:e} Hartree")));
    }
    Ok((isotropic, lowest, residual))
}

/// Molecular MBD@rsSCS from explicit volume ratios and free-atom references.
/// The imaginary-frequency quadrature maps Legendre nodes on [-1, 1] to
/// [0, infinity) with the supplied positive frequency scale (Hartree).
pub fn rsscs_energy_from_volumes(
    molecule: &Molecule,
    ratios: &[f64],
    references: &[FreeAtom],
    beta: f64,
    quadrature_nodes: usize,
    frequency_scale: f64,
) -> Result<MbdRsScsEnergy> {
    if molecule.len()!=references.len() || !beta.is_finite() || beta<=0.0
        || !(4..=256).contains(&quadrature_nodes) || !frequency_scale.is_finite() || frequency_scale<=0.0 {
        return Err(DftError::InvalidInput("MBD@rsSCS needs one free-atom reference per atom, positive finite beta and frequency scale, and 4..=256 quadrature nodes".into()));
    }
    let unscreened_atoms = atom_environments(ratios,references)?;
    let pairs = pairs(molecule,&unscreened_atoms,beta)?;
    let bare_static: Vec<_> = unscreened_atoms.iter().map(|a| a.polarizability).collect();
    let bare_frequency: Vec<_> = unscreened_atoms.iter().map(|a| (4.0/3.0) * a.c6 / a.polarizability.powi(2)).collect();
    let (screened_static,mut minimum_screening_eigenvalue,mut screening_residual) = screen(&pairs,&bare_static,0.0)?;
    let mut screened_c6 = vec![0.0;ratios.len()];
    let (nodes,weights) = gauss_legendre(quadrature_nodes);
    for (x,w) in nodes.into_iter().zip(weights) {
        let omega = frequency_scale * (1.0+x)/(1.0-x);
        let transformed_weight = w * (6.0/PI) * frequency_scale/(1.0-x).powi(2);
        let dynamic_bare: Vec<_> = bare_static.iter().zip(&bare_frequency)
            .map(|(a,f)| a / (1.0+(omega/f).powi(2))).collect();
        let (dynamic_screened,minimum,residual) = screen(&pairs,&dynamic_bare,omega)?;
        minimum_screening_eigenvalue = minimum_screening_eigenvalue.min(minimum);
        screening_residual = screening_residual.max(residual);
        for (c6,a) in screened_c6.iter_mut().zip(dynamic_screened) {
            *c6 += transformed_weight * a*a;
        }
    }
    let screened_atoms: Vec<_> = unscreened_atoms.iter().zip(screened_static).zip(screened_c6)
        .map(|((atom,polarizability),c6)| AtomEnvironment {
            volume_ratio: atom.volume_ratio,
            polarizability,
            c6,
            radius: atom.radius*(polarizability/atom.polarizability).cbrt(),
        }).collect();
    let oscillators = coupled_energy(molecule,screened_atoms,beta,"MBD@rsSCS")?;
    Ok(MbdRsScsEnergy { oscillators, unscreened_atoms, minimum_screening_eigenvalue, screening_residual })
}

/// Molecular post-SCF MBD@rsSCS energy on a supplied total AO density.
#[allow(clippy::too_many_arguments)]
pub fn rsscs_energy_on_density(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: &MolecularGrid,
    density: &Matrix,
    references: &[FreeAtom],
    beta: f64,
    quadrature_nodes: usize,
    frequency_scale: f64,
) -> Result<MbdRsScsEnergy> {
    let ratios = volume_ratios(molecule,basis,grid,density)?;
    rsscs_energy_from_volumes(molecule,&ratios,references,beta,quadrature_nodes,frequency_scale)
}
