// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Divide-and-conquer self-consistent field.
//!
//! The atoms are divided into disjoint fragments. Every fragment, together
//! with the buffer atoms within `buffer` of it, defines a subsystem whose
//! Kohn-Sham matrix and overlap are the corresponding blocks of the global
//! matrices. The local orbitals of all subsystems are occupied by one Fermi
//! function with a common chemical potential, and the density matrix is
//! assembled with the partition weights
//!
//! ```text
//! p_mn = 1    both functions on fragment atoms,
//!        1/2  one on a fragment atom, the other on a buffer atom,
//!        0    otherwise,
//! ```
//!
//! `P = sum_a p^a o [C^a f(e^a) C^a^T]`. The chemical potential fixes the
//! partition-weighted electron count. The energy is the Kohn-Sham functional
//! of this density minus `kT` times the partition-weighted entropy. No
//! global diagonalization enters. When one buffer covers the whole molecule
//! the result is the Fermi-Dirac reference of the conventional calculation.
//!
//! The free energy is not stationary in the assembled density, so its
//! gradient ([`divide_conquer_gradient`]) carries the response of the
//! density, of the entropy and of the chemical potential to the motion of
//! the subsystem blocks, through one adjoint equation. The energy-weighted
//! density `W = sum_a p^a o [C^a f e C^a^T]` alone gives the derivative only
//! in the limit of a complete buffer.

use super::fock::JkEngine;
use super::{Reference, ScfOptions, ScfResult};
use crate::basis::{n_sph, MolBasis};
use crate::error::{DftError, Result};
use crate::grid::MolecularGrid;
use crate::integrals::{kinetic_matrix, nuclear_matrix, overlap_matrix, CoulombKernel};
use crate::linalg::{solve_linear, Matrix, Orthogonalizer};
use crate::math::Vec3;
use crate::response::complex::Complex;
use crate::response::krylov::{self, LinearOperator, OperatorStructure, ResponseSolverConfig};
use crate::response::ResponseSolver;
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;

/// Partition of a molecule for the divide-and-conquer iteration.
#[derive(Clone, Debug)]
pub struct DivideConquerOptions {
    /// Fragment index of every atom; the indices are `0..n_fragments`.
    pub fragment_of: Vec<usize>,
    /// Buffer radius, Bohr.
    pub buffer: f64,
    /// `k_B T` of the common Fermi function, Hartree.
    pub kt: f64,
}

/// One subsystem: a fragment and its buffer.
#[derive(Clone, Debug)]
pub struct Subsystem {
    pub core_atoms: Vec<usize>,
    pub buffer_atoms: Vec<usize>,
    /// Global indices of the basis functions of the subsystem.
    pub functions: Vec<usize>,
    /// Whether each of them sits on a fragment atom.
    pub in_core: Vec<bool>,
}

/// Converged divide-and-conquer state.
#[derive(Clone, Debug)]
pub struct DivideConquerResult {
    /// Densities, Kohn-Sham matrices and energies. `energy` contains the
    /// entropy term; there are no global orbitals, so the orbital fields are
    /// empty.
    pub scf: ScfResult,
    /// Energy-weighted density of the gradient expression.
    pub energy_weighted: Matrix,
    pub subsystems: Vec<Subsystem>,
    /// Kohn-Sham energy without the entropy term, Hartree.
    pub internal_energy: f64,
    /// Largest element of the last density change.
    pub density_residual: f64,
}

fn add_scaled(target: &mut Matrix, source: &Matrix, scale: f64) {
    for (t, s) in target.as_mut_slice().iter_mut().zip(source.as_slice()) {
        *t += scale * s;
    }
}

/// The subsystems of `dc` for the placed basis `mb`.
pub fn subsystems(
    mol: &Molecule,
    mb: &MolBasis,
    dc: &DivideConquerOptions,
) -> Result<Vec<Subsystem>> {
    subsystems_with(mol.len(), mb, dc, &|a, b| {
        (mol.atoms[a].position - mol.atoms[b].position).norm()
    })
}

/// [`subsystems`] with the distance of two atoms supplied by the caller: the
/// minimum image in a periodic cell.
pub(crate) fn subsystems_with(
    natoms: usize,
    mb: &MolBasis,
    dc: &DivideConquerOptions,
    distance: &dyn Fn(usize, usize) -> f64,
) -> Result<Vec<Subsystem>> {
    if dc.fragment_of.len() != natoms {
        return Err(DftError::InvalidInput(
            "divide-and-conquer needs one fragment index per atom".into(),
        ));
    }
    if !(dc.buffer.is_finite() && dc.buffer >= 0.0 && dc.kt.is_finite() && dc.kt > 0.0) {
        return Err(DftError::InvalidInput(
            "divide-and-conquer needs a non-negative buffer radius and a positive smearing".into(),
        ));
    }
    let count = dc.fragment_of.iter().copied().max().map_or(0, |m| m + 1);
    let mut atom_functions: Vec<Vec<usize>> = vec![Vec::new(); natoms];
    for shell in &mb.shells {
        for k in 0..n_sph(shell.l) {
            atom_functions[shell.atom].push(shell.sph_off + k);
        }
    }
    let mut out = Vec::with_capacity(count);
    for fragment in 0..count {
        let core_atoms: Vec<usize> = (0..natoms)
            .filter(|&a| dc.fragment_of[a] == fragment)
            .collect();
        if core_atoms.is_empty() {
            return Err(DftError::InvalidInput(format!(
                "divide-and-conquer fragment {fragment} has no atom: fragment indices must be \
                 contiguous from 0"
            )));
        }
        let buffer_atoms: Vec<usize> = (0..natoms)
            .filter(|&a| dc.fragment_of[a] != fragment)
            .filter(|&a| core_atoms.iter().any(|&c| distance(a, c) <= dc.buffer))
            .collect();
        let mut functions = Vec::new();
        let mut in_core = Vec::new();
        for atom in 0..natoms {
            let core = core_atoms.contains(&atom);
            if core || buffer_atoms.contains(&atom) {
                for &function in &atom_functions[atom] {
                    functions.push(function);
                    in_core.push(core);
                }
            }
        }
        out.push(Subsystem {
            core_atoms,
            buffer_atoms,
            functions,
            in_core,
        });
    }
    Ok(out)
}

/// Kohn-Sham matrices and energy of a pair of spin densities.
pub(crate) struct Build {
    pub(crate) fock: [Matrix; 2],
    pub(crate) electronic: f64,
    pub(crate) exc: f64,
    pub(crate) grid_electrons: Option<f64>,
}

/// Density assembled from the subsystem orbitals of one Kohn-Sham matrix.
pub(crate) struct Assembled {
    pub(crate) density: Matrix,
    pub(crate) weighted: Matrix,
    pub(crate) chemical_potential: f64,
    pub(crate) entropy: f64,
}

fn occupation(energy: f64, mu: f64, kt: f64) -> f64 {
    let x = (energy - mu) / kt;
    if x > 0.0 {
        let t = (-x).exp();
        t / (1.0 + t)
    } else {
        1.0 / (1.0 + x.exp())
    }
}

/// Local orbitals of every subsystem for the global matrix `fock`, occupied
/// with a common chemical potential that holds `electrons` electrons.
fn assemble(
    fock: &Matrix,
    overlap: &Matrix,
    parts: &[Subsystem],
    solvers: &[Orthogonalizer],
    electrons: f64,
    kt: f64,
) -> Result<Assembled> {
    let nao = fock.rows;
    let mut density = Matrix::zeros(nao, nao);
    let mut weighted = Matrix::zeros(nao, nao);
    if electrons == 0.0 {
        return Ok(Assembled {
            density,
            weighted,
            chemical_potential: 0.0,
            entropy: 0.0,
        });
    }
    // Orbital energies, coefficients and partition-weighted norms.
    let mut local: Vec<(Vec<f64>, Matrix, Vec<f64>)> = Vec::with_capacity(parts.len());
    for (part, solver) in parts.iter().zip(solvers) {
        let n = part.functions.len();
        let mut block = Matrix::zeros(n, n);
        for (i, &m) in part.functions.iter().enumerate() {
            for (j, &k) in part.functions.iter().enumerate() {
                block[(i, j)] = fock[(m, k)];
            }
        }
        let (energies, c) = solver.solve_roothaan(&block)?;
        let mut norms = vec![0.0; c.cols];
        for i in 0..n {
            for j in 0..n {
                let weight = 0.5 * (f64::from(part.in_core[i]) + f64::from(part.in_core[j]));
                if weight == 0.0 {
                    continue;
                }
                let s = weight * overlap[(part.functions[i], part.functions[j])];
                if s == 0.0 {
                    continue;
                }
                for (orbital, norm) in norms.iter_mut().enumerate() {
                    *norm += s * c[(i, orbital)] * c[(j, orbital)];
                }
            }
        }
        local.push((energies, c, norms));
    }
    let count = |mu: f64| -> f64 {
        local
            .iter()
            .map(|(energies, _, norms)| {
                energies
                    .iter()
                    .zip(norms)
                    .map(|(&e, &w)| w * occupation(e, mu, kt))
                    .sum::<f64>()
            })
            .sum()
    };
    let lowest = local
        .iter()
        .flat_map(|(e, _, _)| e.iter().copied())
        .fold(f64::INFINITY, f64::min);
    let highest = local
        .iter()
        .flat_map(|(e, _, _)| e.iter().copied())
        .fold(f64::NEG_INFINITY, f64::max);
    let (mut lo, mut hi) = (lowest - 60.0 * kt - 1.0, highest + 60.0 * kt + 1.0);
    if count(hi) < electrons {
        return Err(DftError::InvalidInput(
            "the divide-and-conquer subsystems cannot hold the electrons of the molecule".into(),
        ));
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if count(mid) < electrons {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 4.0 * f64::EPSILON * mid.abs().max(1.0) {
            break;
        }
    }
    let mu = 0.5 * (lo + hi);
    let mut entropy = 0.0;
    for (part, (energies, c, norms)) in parts.iter().zip(&local) {
        let n = part.functions.len();
        let occupations: Vec<f64> = energies.iter().map(|&e| occupation(e, mu, kt)).collect();
        for (&f, &w) in occupations.iter().zip(norms) {
            if f > 0.0 && f < 1.0 {
                entropy -= w * (f * f.ln() + (1.0 - f) * (1.0 - f).ln());
            }
        }
        // C f and C f e, then the weighted products with C^T.
        let mut left = Matrix::zeros(n, c.cols);
        let mut left_energy = Matrix::zeros(n, c.cols);
        for i in 0..n {
            for (orbital, (&f, &e)) in occupations.iter().zip(energies).enumerate() {
                left[(i, orbital)] = c[(i, orbital)] * f;
                left_energy[(i, orbital)] = c[(i, orbital)] * f * e;
            }
        }
        let p = left.matmul_transpose_seq(c);
        let w = left_energy.matmul_transpose_seq(c);
        for i in 0..n {
            for j in 0..n {
                let weight = 0.5 * (f64::from(part.in_core[i]) + f64::from(part.in_core[j]));
                if weight == 0.0 {
                    continue;
                }
                density[(part.functions[i], part.functions[j])] += weight * p[(i, j)];
                weighted[(part.functions[i], part.functions[j])] += weight * w[(i, j)];
            }
        }
    }
    Ok(Assembled {
        density,
        weighted,
        chemical_potential: mu,
        entropy,
    })
}

/// What the divide-and-conquer iteration needs from the electronic problem.
pub(crate) struct IterationInput<'a> {
    pub(crate) parts: &'a [Subsystem],
    pub(crate) overlap: &'a Matrix,
    /// Kohn-Sham matrices and electronic energy of a pair of spin densities.
    pub(crate) build: &'a dyn Fn(&[Matrix; 2]) -> Result<Build>,
    /// Starting total density; it is scaled to the two spin counts.
    pub(crate) start: &'a Matrix,
    pub(crate) electrons: [usize; 2],
    pub(crate) restricted: bool,
    pub(crate) kt: f64,
    pub(crate) orth_threshold: f64,
    pub(crate) max_iter: usize,
    pub(crate) density_tolerance: f64,
    pub(crate) energy_tolerance: f64,
}

/// Last state of the divide-and-conquer iteration.
pub(crate) struct Iteration {
    pub(crate) output: [Assembled; 2],
    /// Build of the assembled densities.
    pub(crate) last: Build,
    pub(crate) converged: bool,
    pub(crate) iterations: usize,
    pub(crate) residual: f64,
}

/// The divide-and-conquer iteration: the density is assembled from the
/// subsystem orbitals of its own Kohn-Sham matrices and iterated by Pulay
/// extrapolation of its change.
pub(crate) fn iterate(input: &IterationInput) -> Result<Iteration> {
    let parts = input.parts;
    let overlap = input.overlap;
    let nao = overlap.rows;
    let [n_alpha, n_beta] = input.electrons;
    // The overlap block of a subsystem does not change during the iteration.
    let mut solvers = Vec::with_capacity(parts.len());
    for part in parts {
        let n = part.functions.len();
        let mut block = Matrix::zeros(n, n);
        for (i, &m) in part.functions.iter().enumerate() {
            for (j, &k) in part.functions.iter().enumerate() {
                block[(i, j)] = overlap[(m, k)];
            }
        }
        solvers.push(Orthogonalizer::canonical(&block, input.orth_threshold)?);
    }
    let assemble_spins = |fock: &[Matrix; 2]| -> Result<[Assembled; 2]> {
        let alpha = assemble(&fock[0], overlap, parts, &solvers, n_alpha as f64, input.kt)?;
        let beta = if input.restricted {
            Assembled {
                density: alpha.density.clone(),
                weighted: alpha.weighted.clone(),
                chemical_potential: alpha.chemical_potential,
                entropy: alpha.entropy,
            }
        } else {
            assemble(&fock[1], overlap, parts, &solvers, n_beta as f64, input.kt)?
        };
        Ok([alpha, beta])
    };
    let start_count = input.start.frobenius_dot(overlap);
    if !(start_count > 0.0) {
        return Err(DftError::InvalidInput(
            "the starting density of the divide-and-conquer iteration holds no electron".into(),
        ));
    }
    let mut trial: [Matrix; 2] = std::array::from_fn(|spin| {
        let mut p = input.start.clone();
        let electrons = if spin == 0 { n_alpha } else { n_beta } as f64;
        for value in p.as_mut_slice() {
            *value *= electrons / start_count;
        }
        p
    });
    const HISTORY: usize = 8;
    let mut history: Vec<([Matrix; 2], [Matrix; 2])> = Vec::new();
    let mut last_energy = f64::INFINITY;
    let mut converged = false;
    let mut iterations = 0;
    let mut residual = f64::INFINITY;
    let mut state = None;
    for iteration in 1..=input.max_iter {
        iterations = iteration;
        let built = (input.build)(&trial)?;
        let output = assemble_spins(&built.fock)?;
        let change: [Matrix; 2] = std::array::from_fn(|spin| {
            let mut r = output[spin].density.clone();
            add_scaled(&mut r, &trial[spin], -1.0);
            r
        });
        residual = change
            .iter()
            .flat_map(|m| m.as_slice().iter())
            .fold(0.0_f64, |m, x| m.max(x.abs()));
        let energy_change = (built.electronic - last_energy).abs();
        last_energy = built.electronic;
        if residual < input.density_tolerance && energy_change < input.energy_tolerance {
            converged = true;
            state = Some(output);
            break;
        }
        history.push((trial.clone(), change));
        if history.len() > HISTORY {
            history.remove(0);
        }
        // Pulay extrapolation of the density with its change as the error.
        let m = history.len();
        let mut weights = vec![0.0; m];
        weights[m - 1] = 1.0;
        if m > 1 {
            let mut b = Matrix::zeros(m + 1, m + 1);
            let mut rhs = vec![0.0; m + 1];
            for i in 0..m {
                for j in 0..m {
                    b[(i, j)] = history[i].1[0].frobenius_dot(&history[j].1[0])
                        + history[i].1[1].frobenius_dot(&history[j].1[1]);
                }
                b[(i, m)] = 1.0;
                b[(m, i)] = 1.0;
            }
            rhs[m] = 1.0;
            if let Ok(solution) = solve_linear(&b, &rhs) {
                if solution.iter().all(|w| w.is_finite()) {
                    weights.copy_from_slice(&solution[..m]);
                }
            }
        }
        let mut next = [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)];
        for (weight, (density, change)) in weights.iter().zip(&history) {
            for spin in 0..2 {
                add_scaled(&mut next[spin], &density[spin], *weight);
                add_scaled(&mut next[spin], &change[spin], *weight);
            }
        }
        trial = next;
        state = Some(output);
    }
    let output = state.ok_or_else(|| {
        DftError::InvalidInput("the divide-and-conquer iteration needs max_iter >= 1".into())
    })?;
    let density = [output[0].density.clone(), output[1].density.clone()];
    let last = (input.build)(&density)?;
    Ok(Iteration {
        output,
        last,
        converged,
        iterations,
        residual,
    })
}

/// Divide-and-conquer self-consistent field of `mol` with the partition
/// `dc`. The density is iterated by Pulay extrapolation of its change.
pub fn run_divide_conquer(
    mol: &Molecule,
    mb: &MolBasis,
    opts: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    dc: &DivideConquerOptions,
) -> Result<DivideConquerResult> {
    if opts.occupation.is_some() || opts.fixed_occupations.is_some() {
        return Err(DftError::Unsupported {
            feature: "fragment determinant tracking".into(),
            missing: "a shared global occupied-state constraint on the fragment optimizers".into(),
        });
    }
    if opts.pcm.is_some()
        || opts.vv10.is_some()
        || opts.skala.is_some()
        || opts.density_fit.is_some()
    {
        return Err(DftError::Unsupported {
            feature: "divide-and-conquer SCF with continuum solvation, nonlocal correlation, a \
                      learned functional or density fitting"
                .into(),
            missing: "those terms in the divide-and-conquer Kohn-Sham build".into(),
        });
    }
    let has_xc = !functional.parts.is_empty();
    if has_xc && grid.is_none() {
        return Err(DftError::InvalidInput(
            "divide-and-conquer Kohn-Sham needs the molecular grid".into(),
        ));
    }
    let parts = subsystems(mol, mb, dc)?;
    let nao = mb.nao_sph;
    let has_ecp = !mb.ecps.is_empty();
    let electrons = {
        let n = mol.n_electrons()?;
        let core = mb.n_core_electrons as usize;
        n.checked_sub(core).ok_or_else(|| {
            DftError::InvalidInput(format!(
                "ECP core electrons ({core}) exceed the total electron count ({n})"
            ))
        })?
    };
    let (n_alpha, n_beta) = mol.n_alpha_beta_for(electrons)?;
    let restricted = n_alpha == n_beta && opts.reference != Reference::Uhf;
    // Overlap and core Hamiltonian, as in the conventional iteration.
    let overlap = overlap_matrix(mb);
    let z_eff = mb.effective_charges(mol);
    let mut h_core = kinetic_matrix(mb);
    let attraction = if has_ecp {
        nuclear_matrix(mb, mol, Some(&z_eff))
    } else {
        nuclear_matrix(mb, mol, None)
    };
    add_scaled(&mut h_core, &attraction, 1.0);
    if has_ecp {
        add_scaled(&mut h_core, &crate::integrals::ecp_matrix(mb)?, 1.0);
    }
    let mut e_nuc_field = 0.0;
    if let Some(field) = opts.external_field {
        let dips = crate::integrals::dipole_matrices(mb, crate::math::Vec3::zero());
        for (axis, d) in dips.iter().enumerate() {
            add_scaled(&mut h_core, d, field.get(axis));
        }
        for (atom, z) in mol.atoms.iter().zip(&z_eff) {
            e_nuc_field -= z * atom.position.dot(field);
        }
    }
    if let Some(extra) = &opts.extra_hcore {
        add_scaled(&mut h_core, extra, 1.0);
    }
    let e_nuc = if has_ecp {
        mb.nuclear_repulsion(mol)
    } else {
        mol.nuclear_repulsion()
    } + e_nuc_field;
    let cx = functional.hybrid.cx;
    let clr = functional.hybrid.clr;
    let jk = JkEngine::from_options(mol, mb, opts)?;
    let jk_lr = if clr != 0.0 {
        Some(JkEngine::new(
            mb,
            opts.eri_mode,
            CoulombKernel::Erf {
                omega: functional.hybrid.omega,
            },
        )?)
    } else {
        None
    };
    let build = |density: &[Matrix; 2]| -> Result<Build> {
        let mut total = density[0].clone();
        add_scaled(&mut total, &density[1], 1.0);
        let (j, k) = jk.build(mb, &total, [&density[0], &density[1]], cx != 0.0)?;
        let k_lr = match &jk_lr {
            Some(engine) => Some(engine.build_k_only(mb, [&density[0], &density[1]])?),
            None => None,
        };
        let xc = match (has_xc, grid) {
            (true, Some(grid)) => Some(crate::ks::xc_contribution(
                functional,
                mb,
                grid,
                [&density[0], &density[1]],
            )?),
            _ => None,
        };
        let mut electronic = total.frobenius_dot(&h_core) + 0.5 * total.frobenius_dot(&j);
        let mut fock = [h_core.clone(), h_core.clone()];
        for spin in 0..2 {
            add_scaled(&mut fock[spin], &j, 1.0);
            if cx != 0.0 {
                add_scaled(&mut fock[spin], &k[spin], -cx);
                electronic -= 0.5 * cx * density[spin].frobenius_dot(&k[spin]);
            }
            if let Some(k_lr) = &k_lr {
                add_scaled(&mut fock[spin], &k_lr[spin], -clr);
                electronic -= 0.5 * clr * density[spin].frobenius_dot(&k_lr[spin]);
            }
            if let Some(xc) = &xc {
                add_scaled(&mut fock[spin], &xc.vxc[spin], 1.0);
            }
        }
        let exc = xc.as_ref().map_or(0.0, |x| x.exc);
        electronic += exc;
        Ok(Build {
            fock,
            electronic,
            exc,
            grid_electrons: xc.as_ref().map(|x| x.n_elec),
        })
    };
    // Superposition of atomic densities, scaled to the two spin counts.
    let start = super::sad::sad_density(mol, mb)?;
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
        orth_threshold: opts.orth_threshold,
        max_iter: opts.max_iter,
        density_tolerance: opts.conv_tol_grad,
        energy_tolerance: opts.e_tol,
    })?;
    let density = [output[0].density.clone(), output[1].density.clone()];
    let mut energy_weighted = output[0].weighted.clone();
    add_scaled(&mut energy_weighted, &output[1].weighted, 1.0);
    let entropy = output[0].entropy + output[1].entropy;
    let internal_energy = last.electronic + e_nuc;
    let sz = 0.5 * (n_alpha as f64 - n_beta as f64);
    let scf = ScfResult {
        occupation_tracking: None,
        fixed_occupation_residual: None,
        common_orbital_residual: None,
        energy: internal_energy - dc.kt * entropy,
        e_nuc,
        e_elec: last.electronic,
        mo_energy: [Vec::new(), Vec::new()],
        mo_coeff: [Matrix::zeros(nao, 0), Matrix::zeros(nao, 0)],
        density,
        fock: last.fock,
        overlap,
        h_core,
        n_alpha,
        n_beta,
        unrestricted: !restricted,
        s_squared: sz * (sz + 1.0),
        e_xc: last.exc,
        e_nlc: 0.0,
        e_pcm: 0.0,
        grid_electron_count: last.grid_electrons,
        converged,
        iterations,
        occupations: [Vec::new(), Vec::new()],
        electronic_temperature: dc.kt,
        electronic_entropy: entropy,
        chemical_potentials: [
            Some(output[0].chemical_potential),
            Some(output[1].chemical_potential),
        ],
    };
    Ok(DivideConquerResult {
        scf,
        energy_weighted,
        subsystems: parts,
        internal_energy,
        density_residual: residual,
    })
}

/// Local orbitals of one subsystem with the divided differences of the
/// occupation and entropy functions of the orbital energy.
struct LocalResponse {
    c: Matrix,
    /// Divided differences of `f`, `f e`, `s(f)` and `s(f) e`.
    tables: [Matrix; 4],
    /// `-df/de` and `-ds(f)/de`: the derivatives with respect to the
    /// chemical potential.
    shift: [Vec<f64>; 2],
    /// `s(f)`.
    entropy: Vec<f64>,
}

/// `(h_i - h_j) / (e_i - e_j)`, and the mean slope for coinciding energies.
fn divided(values: &[f64], slopes: &[f64], energies: &[f64]) -> Matrix {
    let n = energies.len();
    let mut out = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let gap = energies[i] - energies[j];
            out[(i, j)] = if gap.abs() > 1e-6 {
                (values[i] - values[j]) / gap
            } else {
                0.5 * (slopes[i] + slopes[j])
            };
        }
    }
    out
}

fn local_response(
    fock: &Matrix,
    part: &Subsystem,
    solver: &Orthogonalizer,
    mu: f64,
    kt: f64,
) -> Result<LocalResponse> {
    let n = part.functions.len();
    let mut block = Matrix::zeros(n, n);
    for (i, &m) in part.functions.iter().enumerate() {
        for (j, &k) in part.functions.iter().enumerate() {
            block[(i, j)] = fock[(m, k)];
        }
    }
    let (energies, c) = solver.solve_roothaan(&block)?;
    let count = energies.len();
    let mut values: [Vec<f64>; 4] = std::array::from_fn(|_| vec![0.0; count]);
    let mut slopes: [Vec<f64>; 4] = std::array::from_fn(|_| vec![0.0; count]);
    let mut shift = [vec![0.0; count], vec![0.0; count]];
    let mut entropy = vec![0.0; count];
    for (i, &e) in energies.iter().enumerate() {
        let f = occupation(e, mu, kt);
        let df = -f * (1.0 - f) / kt;
        let s = if f > 0.0 && f < 1.0 {
            -(f * f.ln() + (1.0 - f) * (1.0 - f).ln())
        } else {
            0.0
        };
        // ds(f)/de = s'(f) f' with s'(f) = (e - mu) / kT.
        let ds = (e - mu) / kt * df;
        values[0][i] = f;
        slopes[0][i] = df;
        values[1][i] = f * e;
        slopes[1][i] = df * e + f;
        values[2][i] = s;
        slopes[2][i] = ds;
        values[3][i] = s * e;
        slopes[3][i] = ds * e + s;
        shift[0][i] = -df;
        shift[1][i] = -ds;
        entropy[i] = s;
    }
    let tables: [Matrix; 4] = std::array::from_fn(|k| divided(&values[k], &slopes[k], &energies));
    Ok(LocalResponse {
        c,
        tables,
        shift,
        entropy,
    })
}

fn partition_weight(part: &Subsystem, i: usize, j: usize) -> f64 {
    0.5 * (f64::from(part.in_core[i]) + f64::from(part.in_core[j]))
}

/// `sum_a scatter( C [(C^T (p o Y) C) o table] C^T )`: the adjoint of the
/// first-order change of an assembled matrix function with respect to the
/// subsystem blocks.
fn adjoint(parts: &[Subsystem], locals: &[LocalResponse], y: &Matrix, table: usize) -> Matrix {
    let nao = y.rows;
    let mut out = Matrix::zeros(nao, nao);
    for (part, local) in parts.iter().zip(locals) {
        let n = part.functions.len();
        let mut block = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let weight = partition_weight(part, i, j);
                if weight != 0.0 {
                    block[(i, j)] = weight * y[(part.functions[i], part.functions[j])];
                }
            }
        }
        let mut inner = local.c.transpose_matmul_seq(&block.matmul_seq(&local.c));
        for (value, factor) in inner
            .as_mut_slice()
            .iter_mut()
            .zip(local.tables[table].as_slice())
        {
            *value *= *factor;
        }
        let image = local.c.matmul_seq(&inner).matmul_transpose_seq(&local.c);
        for i in 0..n {
            for j in 0..n {
                out[(part.functions[i], part.functions[j])] += image[(i, j)];
            }
        }
    }
    out
}

/// `sum_a scatter( p o [C diag(h) C^T] )` for `h` the chemical-potential
/// derivative of the occupation (0), of the entropy (1), or the entropy (2).
fn assembled(parts: &[Subsystem], locals: &[LocalResponse], which: usize, nao: usize) -> Matrix {
    let mut out = Matrix::zeros(nao, nao);
    for (part, local) in parts.iter().zip(locals) {
        let n = part.functions.len();
        let values: &[f64] = match which {
            0 => &local.shift[0],
            1 => &local.shift[1],
            _ => &local.entropy,
        };
        let mut left = Matrix::zeros(n, local.c.cols);
        for i in 0..n {
            for (orbital, &h) in values.iter().enumerate() {
                left[(i, orbital)] = local.c[(i, orbital)] * h;
            }
        }
        let image = left.matmul_transpose_seq(&local.c);
        for i in 0..n {
            for j in 0..n {
                let weight = partition_weight(part, i, j);
                if weight != 0.0 {
                    out[(part.functions[i], part.functions[j])] += weight * image[(i, j)];
                }
            }
        }
    }
    out
}

/// `1 - T K` on the stacked relaxation matrices of the spin channels.
struct RelaxationOperator<'a> {
    size: usize,
    image: &'a dyn Fn(&[f64]) -> Result<Vec<f64>>,
}

impl LinearOperator for RelaxationOperator<'_> {
    fn dimension(&self) -> usize {
        self.size
    }
    fn structure(&self) -> OperatorStructure {
        OperatorStructure::General
    }
    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        let real: Vec<f64> = x.iter().map(|v| v.re).collect();
        let imaginary: Vec<f64> = x.iter().map(|v| v.im).collect();
        let a = (self.image)(&real)?;
        let b = if imaginary.iter().all(|v| *v == 0.0) {
            vec![0.0; a.len()]
        } else {
            (self.image)(&imaginary)?
        };
        Ok(a.into_iter()
            .zip(b)
            .map(|(re, im)| Complex::new(re, im))
            .collect())
    }
}

/// One spin channel of the adjoint problem.
struct Channel {
    locals: Vec<LocalResponse>,
    /// Coefficient of the Kohn-Sham matrix change in the free-energy change.
    lambda: Matrix,
    /// Coefficient of the overlap change.
    theta: Matrix,
    /// Adjoint of the occupation response contracted with the overlap.
    count_adjoint: Matrix,
    /// Coefficient of the overlap change in the electron-count condition.
    count_overlap: Matrix,
    /// Derivative of the density with respect to the chemical potential.
    shift: Matrix,
    /// Its electron count; zero when no orbital is fractionally occupied.
    count: f64,
}

impl Channel {
    fn ratio(&self, numerator: f64) -> f64 {
        if self.count.abs() > 1e-12 {
            numerator / self.count
        } else {
            0.0
        }
    }
}

/// What the adjoint problem of a divide-and-conquer state needs.
pub(crate) struct RelaxationInput<'a> {
    pub(crate) parts: &'a [Subsystem],
    pub(crate) overlap: &'a Matrix,
    pub(crate) fock: &'a [Matrix; 2],
    pub(crate) density: &'a [Matrix; 2],
    pub(crate) chemical_potentials: [Option<f64>; 2],
    pub(crate) unrestricted: bool,
    pub(crate) kt: f64,
    pub(crate) orth_threshold: f64,
    /// First-order Kohn-Sham matrices of a pair of spin density changes.
    pub(crate) kernel: &'a dyn Fn(&[Matrix; 2]) -> Result<[Matrix; 2]>,
}

/// Adjoint of the divide-and-conquer free energy `A = E[P] - kT S`: the
/// relaxation matrices `Z` of both spins, minus the total coefficient of
/// the overlap change (the energy-weighted density of the fixed-density
/// terms), and the relative residual of the adjoint equation.
///
/// A change `dF`, `dS` of the Kohn-Sham and overlap blocks changes every
/// assembled matrix function `G_h = sum_a p^a o [C h(e) C^T]` by
/// `p o C [(C^T dF C) o L[h] - (C^T dS C) o L[h e]] C^T` at fixed chemical
/// potential, with the divided differences `L`; the chemical potential
/// follows from `Tr[P S] = N`. With `P = G_f` and `S = Tr[S G_s]` the change
/// of the free energy is `Tr[Lambda dF] + Tr[Theta dS]`. Because
/// `dF = F^(x) + K[dP]` with the Kohn-Sham kernel `K`, the relaxation matrix
/// `Z = Lambda + T[K[Z]]` (`T` the adjoint of the density response) gives
///
/// ```text
/// dA/dx = E^(x)[P] + sum_s Tr[Z_s F_s^(x)] + Tr[Theta_total S^(x)]
/// ```
///
/// for any parameter `x` of the integrals. For a complete buffer `Lambda`
/// vanishes and `Theta_total` is minus the energy-weighted density.
pub(crate) fn relaxation(
    input: &RelaxationInput,
    config: &ResponseSolverConfig,
) -> Result<([Matrix; 2], Matrix, f64)> {
    let parts = input.parts;
    let overlap = input.overlap;
    let nao = overlap.rows;
    let kt = input.kt;
    let spins = if input.unrestricted { 2 } else { 1 };
    let mut solvers = Vec::with_capacity(parts.len());
    for part in parts {
        let n = part.functions.len();
        let mut block = Matrix::zeros(n, n);
        for (i, &m) in part.functions.iter().enumerate() {
            for (j, &k) in part.functions.iter().enumerate() {
                block[(i, j)] = overlap[(m, k)];
            }
        }
        solvers.push(Orthogonalizer::canonical(&block, input.orth_threshold)?);
    }
    let mut channels = Vec::with_capacity(spins);
    for s in 0..spins {
        let mu = input.chemical_potentials[s].ok_or_else(|| {
            DftError::InvalidInput(
                "the divide-and-conquer gradient needs the chemical potentials of the result"
                    .into(),
            )
        })?;
        let mut locals = Vec::with_capacity(parts.len());
        for (part, solver) in parts.iter().zip(&solvers) {
            locals.push(local_response(&input.fock[s], part, solver, mu, kt)?);
        }
        let shift = assembled(parts, &locals, 0, nao);
        let shift_entropy = assembled(parts, &locals, 1, nao);
        let entropy = assembled(parts, &locals, 2, nao);
        let count = overlap.frobenius_dot(&shift);
        let count_adjoint = adjoint(parts, &locals, overlap, 0);
        // Overlap coefficient of the count condition: the adjoint of the
        // overlap response of the density, plus the density itself.
        let mut count_overlap = adjoint(parts, &locals, overlap, 1);
        for value in count_overlap.as_mut_slice() {
            *value = -*value;
        }
        add_scaled(&mut count_overlap, &input.density[s], 1.0);
        let mut channel = Channel {
            locals,
            lambda: Matrix::zeros(nao, nao),
            theta: Matrix::zeros(nao, nao),
            count_adjoint,
            count_overlap,
            shift,
            count,
        };
        let potential = channel.ratio(
            input.fock[s].frobenius_dot(&channel.shift)
                - kt * overlap.frobenius_dot(&shift_entropy),
        );
        let mut lambda = adjoint(parts, &channel.locals, &input.fock[s], 0);
        add_scaled(
            &mut lambda,
            &adjoint(parts, &channel.locals, overlap, 2),
            -kt,
        );
        add_scaled(&mut lambda, &channel.count_adjoint, -potential);
        let mut theta = adjoint(parts, &channel.locals, &input.fock[s], 1);
        for value in theta.as_mut_slice() {
            *value = -*value;
        }
        add_scaled(&mut theta, &adjoint(parts, &channel.locals, overlap, 3), kt);
        add_scaled(&mut theta, &entropy, -kt);
        add_scaled(&mut theta, &channel.count_overlap, -potential);
        channel.lambda = lambda;
        channel.theta = theta;
        channels.push(channel);
    }
    let block = nao * nao;
    let unpack = |x: &[f64]| -> [Matrix; 2] {
        let first = Matrix::from_row_major(nao, nao, x[..block].to_vec());
        let second = if spins == 2 {
            Matrix::from_row_major(nao, nao, x[block..].to_vec())
        } else {
            first.clone()
        };
        [first, second]
    };
    // T[K[Z]] of every channel.
    let transfer = |response: &[Matrix; 2]| -> Vec<Matrix> {
        channels
            .iter()
            .enumerate()
            .map(|(s, channel)| {
                let mut out = adjoint(parts, &channel.locals, &response[s], 0);
                let potential = channel.ratio(response[s].frobenius_dot(&channel.shift));
                add_scaled(&mut out, &channel.count_adjoint, -potential);
                out
            })
            .collect()
    };
    let image = |x: &[f64]| -> Result<Vec<f64>> {
        let z = unpack(x);
        let response = (input.kernel)(&z)?;
        let moved = transfer(&response);
        let mut out = x.to_vec();
        for (s, matrix) in moved.iter().enumerate() {
            for (value, m) in out[s * block..(s + 1) * block]
                .iter_mut()
                .zip(matrix.as_slice())
            {
                *value -= *m;
            }
        }
        Ok(out)
    };
    let rhs: Vec<Complex> = channels
        .iter()
        .flat_map(|channel| {
            channel
                .lambda
                .as_slice()
                .iter()
                .map(|&v| Complex::new(v, 0.0))
        })
        .collect();
    let operator = RelaxationOperator {
        size: spins * block,
        image: &image,
    };
    let solution = krylov::solve(&operator, &rhs, None, config)?;
    let amplitudes: Vec<f64> = solution.amplitudes.iter().map(|z| z.re).collect();
    let z = unpack(&amplitudes);
    let response = (input.kernel)(&z)?;
    // Minus the total overlap coefficient plays the role of the
    // energy-weighted density in the fixed-density terms.
    let mut weighted = Matrix::zeros(nao, nao);
    for (s, channel) in channels.iter().enumerate() {
        let mut theta = channel.theta.clone();
        add_scaled(
            &mut theta,
            &adjoint(parts, &channel.locals, &response[s], 1),
            -1.0,
        );
        let potential = channel.ratio(response[s].frobenius_dot(&channel.shift));
        add_scaled(&mut theta, &channel.count_overlap, -potential);
        let copies = if spins == 1 { 2.0 } else { 1.0 };
        add_scaled(&mut weighted, &theta, -copies);
    }
    Ok((z, weighted, solution.relative_residual))
}

/// Gradient of the divide-and-conquer free energy `A = E[P] - kT S`,
/// including the response of the assembled density and entropy to the
/// motion of the subsystem blocks ([`relaxation`]), and the relative
/// residual of the adjoint equation.
#[allow(clippy::too_many_arguments)]
pub fn divide_conquer_gradient(
    mol: &Molecule,
    mb: &MolBasis,
    opts: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    dc: &DivideConquerOptions,
    result: &DivideConquerResult,
    config: &ResponseSolverConfig,
) -> Result<(Vec<Vec3>, f64)> {
    let scf = &result.scf;
    let nao = mb.nao_sph;
    // Kohn-Sham kernel of the assembled densities. The engine reads the
    // orbital count of its reference only for its dimensions.
    let mut kernel_reference = scf.clone();
    kernel_reference.mo_coeff = [Matrix::identity(nao), Matrix::identity(nao)];
    let kernel = ResponseSolver::kernel_only(mol, mb, functional, grid, &kernel_reference, opts)?;
    let (z, weighted, residual) = relaxation(
        &RelaxationInput {
            parts: &result.subsystems,
            overlap: &scf.overlap,
            fock: &scf.fock,
            density: &scf.density,
            chemical_potentials: scf.chemical_potentials,
            unrestricted: scf.unrestricted,
            kt: dc.kt,
            orth_threshold: opts.orth_threshold,
            kernel: &|z: &[Matrix; 2]| kernel.fock_response(z),
        },
        config,
    )?;
    let gradient = crate::gradient::assemble_gradient_relaxed(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        [&z[0], &z[1]],
        &weighted,
    )?;
    Ok((gradient, residual))
}
