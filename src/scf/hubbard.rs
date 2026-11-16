// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed-geometry projected Hubbard corrections.

mod spinor;
pub use spinor::{SpinorHubbardFields, SpinorSiteEnergy};
pub(crate) use spinor::HubbardSpinorBackend;

use super::{fields::DensityExtension, ScfOptions, ScfResult};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

fn is_finite_symmetric(matrix: &Matrix) -> bool {
    matrix.rows == matrix.cols
        && matrix.as_slice().iter().all(|x| x.is_finite())
        && (0..matrix.rows).all(|i| {
            (0..i).all(|j| {
                let a = matrix[(i, j)];
                let b = matrix[(j, i)];
                (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
            })
        })
}

/// Interaction parameters in Hartree for one correlated subspace.
#[derive(Clone, Debug)]
pub enum Interaction {
    Dudarev {
        u_eff: f64,
    },
    Liechtenstein {
        u: f64,
        j: f64,
        /// Chemists' `(ab|cd)` integrals in row-major order in the
        /// symmetrically orthogonalized site basis.
        integrals: Vec<f64>,
    },
}

/// AO indices of one site, in the order used by the interaction tensor.
#[derive(Clone, Debug)]
pub struct Site {
    pub orbitals: Vec<usize>,
    pub interaction: Interaction,
}

/// An unordered interaction between two declared Hubbard sites.
#[derive(Clone, Copy, Debug)]
pub struct IntersiteInteraction {
    pub sites: [usize; 2],
    /// Screened intersite coupling in Hartree.
    pub v: f64,
}

struct Projector {
    covariant: Matrix,
    interaction: Interaction,
}

/// Immutable interaction and overlap projectors for one molecular geometry.
pub struct HubbardModel {
    sites: Vec<Projector>,
    pairs: Vec<IntersiteInteraction>,
    dimension: usize,
}

/// Unmixed Hubbard energy terms at the supplied electronic density.
pub struct SiteEnergy {
    pub interaction: f64,
    pub double_counting: f64,
    pub occupations: [Matrix; 2],
}

pub struct HubbardFields {
    pub energy: f64,
    pub potential: [Matrix; 2],
    pub sites: Vec<SiteEnergy>,
    pub pairs: Vec<IntersiteEnergy>,
}

/// Intersite Hubbard energy and its subtracted double counting, in Hartree.
pub struct IntersiteEnergy {
    pub pair: IntersiteInteraction,
    pub interaction: f64,
    pub double_counting: f64,
    pub correction: f64,
}

impl IntersiteInteraction {
    fn energy(self, left_number: f64, right_number: f64, bond_order: f64) -> IntersiteEnergy {
        // The symmetric ordered sum in Campo–Cococcioni Eq. (7) counts
        // this unordered pair twice, cancelling the factor of one half.
        let double_counting = self.v * left_number * right_number;
        let correction = -self.v * bond_order;
        IntersiteEnergy {
            pair: self,
            interaction: double_counting + correction,
            double_counting,
            correction,
        }
    }
}

/// A converged DFT+U reference and the correction evaluated at its density.
/// Ordinary molecular derivative routines require a different reference
/// because they do not differentiate the moving site projectors.
pub struct HubbardScfResult {
    pub(crate) scf: ScfResult,
    pub correction: HubbardFields,
}

impl HubbardScfResult {
    /// SCF energy in Hartree including Hubbard and nuclear repulsion terms.
    pub fn energy(&self) -> f64 {
        self.scf.energy
    }

    /// Alpha and beta AO density matrices of the corrected reference.
    pub fn density(&self) -> &[Matrix; 2] {
        &self.scf.density
    }
}

/// Solve the real collinear integer-occupation variational problem.
pub fn run_collinear(
    mol: &Molecule,
    basis: &MolBasis,
    options: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    sites: Vec<Site>,
) -> Result<HubbardScfResult> {
    run(mol, basis, options, functional, grid, sites, false)
}

/// Solve the common-spatial-orbital integer-occupation variational problem.
pub fn run_roks(
    mol: &Molecule,
    basis: &MolBasis,
    options: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    sites: Vec<Site>,
) -> Result<HubbardScfResult> {
    run(mol, basis, options, functional, grid, sites, true)
}

fn run(
    mol: &Molecule,
    basis: &MolBasis,
    options: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    sites: Vec<Site>,
    common_orbitals: bool,
) -> Result<HubbardScfResult> {
    let overlap = crate::integrals::overlap_matrix(basis);
    let model = HubbardModel::new(&overlap, sites, options.orth_threshold)?;
    run_with_model(mol, basis, options, functional, grid, &model, common_orbitals)
}

pub(crate) fn run_with_model(
    mol: &Molecule,
    basis: &MolBasis,
    options: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    model: &HubbardModel,
    common_orbitals: bool,
) -> Result<HubbardScfResult> {
    validate_occupations(options)?;
    let extension = Some(DensityExtension::Hubbard(model));
    let scf = if common_orbitals {
        let (result, _) =
            super::roks::run_roks_with_extension(mol, basis, options, functional, grid, extension)?;
        if !result.converged {
            return Err(DftError::ScfNotConverged {
                iterations: result.iterations,
                error: result.common_orbital_residual.unwrap_or(f64::NAN),
            });
        }
        result
    } else {
        super::run_scf_in_potential_with_completion(
            mol, basis, options, functional, grid, None, None, None, extension, true,
        )?
    };
    let correction = model.evaluate(&scf.density)?;
    Ok(HubbardScfResult { scf, correction })
}

pub(crate) fn validate_occupations(options: &ScfOptions) -> Result<()> {
    if options.occupation.is_some()
        || options.fixed_occupations.is_some()
        || options.electronic_temperature != 0.0
        || options.electron_offset != [0.0; 2]
        || options.asymptotic_correction.is_some()
    {
        return Err(DftError::Unsupported {
            feature: "DFT+U with the selected reference occupations or asymptotic potential".into(),
            missing: "the corresponding variational integer Aufbau model".into(),
        });
    }
    Ok(())
}

impl HubbardModel {
    /// Form `W = S[:,i] S[i,i]^(-1/2)` for each site. Different sites
    /// are separate subspaces and are not orthogonalized against each other.
    pub fn new(overlap: &Matrix, sites: Vec<Site>, rank_threshold: f64) -> Result<Self> {
        if !is_finite_symmetric(overlap)
            || sites.is_empty()
            || !rank_threshold.is_finite()
            || rank_threshold <= 0.0
        {
            return Err(DftError::InvalidInput(
                "Hubbard sites require a finite symmetric overlap, a positive rank threshold and at least one site".into(),
            ));
        }
        let dimension = overlap.rows;
        let mut used = vec![false; dimension];
        let mut projectors = Vec::with_capacity(sites.len());
        for Site {
            orbitals,
            interaction,
        } in sites
        {
            if orbitals.is_empty()
                || orbitals
                    .iter()
                    .any(|&i| i >= dimension || std::mem::replace(&mut used[i], true))
            {
                return Err(DftError::InvalidInput(
                    "Hubbard AO indices must be in range and used at most once".into(),
                ));
            }
            let m = orbitals.len();
            interaction.validate(m)?;
            let mut metric = Matrix::zeros(m, m);
            let mut cross = Matrix::zeros(dimension, m);
            for (b, &j) in orbitals.iter().enumerate() {
                for (a, &i) in orbitals.iter().enumerate() {
                    metric[(a, b)] = overlap[(i, j)];
                }
                for i in 0..dimension {
                    cross[(i, b)] = overlap[(i, j)];
                }
            }
            let (values, vectors) = symmetric_eigen(&metric)?;
            if values
                .iter()
                .any(|&s| !s.is_finite() || s <= rank_threshold)
            {
                return Err(DftError::InvalidInput(
                    "a Hubbard site has linearly dependent AO projectors".into(),
                ));
            }
            let mut weighted = vectors.clone();
            for row in 0..m {
                for col in 0..m {
                    weighted[(row, col)] /= values[col].sqrt();
                }
            }
            let inverse_sqrt = weighted.matmul_transpose_seq(&vectors);
            projectors.push(Projector {
                covariant: cross.matmul_seq(&inverse_sqrt),
                interaction,
            });
        }
        Ok(Self {
            sites: projectors,
            pairs: Vec::new(),
            dimension,
        })
    }

    /// Set the intersite interactions on the existing fixed projectors.
    /// The pair indices refer to the original site order and need not be sorted.
    pub fn with_intersite(mut self, mut pairs: Vec<IntersiteInteraction>) -> Result<Self> {
        for pair in &mut pairs {
            pair.sites.sort_unstable();
            if pair.sites[0] == pair.sites[1]
                || pair.sites[1] >= self.sites.len()
                || !pair.v.is_finite()
            {
                return Err(DftError::InvalidInput(
                    "each intersite Hubbard pair needs two distinct existing sites and a finite V"
                        .into(),
                ));
            }
        }
        pairs.sort_unstable_by_key(|pair| pair.sites);
        if pairs.windows(2).any(|pair| pair[0].sites == pair[1].sites) {
            return Err(DftError::InvalidInput(
                "each unordered intersite Hubbard pair may be specified once".into(),
            ));
        }
        self.pairs = pairs;
        Ok(self)
    }

    /// Evaluate the physical energy and its first AO density derivative.
    pub fn evaluate(&self, density: &[Matrix; 2]) -> Result<HubbardFields> {
        self.validate_density(density)?;
        let mut potential = self.zeros();
        let mut sites = Vec::with_capacity(self.sites.len());
        let mut energy = 0.0;
        for site in &self.sites {
            let occupations = site.occupations(density);
            let (interaction, double_counting, local) = site.interaction.evaluate(&occupations);
            energy += interaction - double_counting;
            site.lift(&local, &mut potential);
            sites.push(SiteEnergy {
                interaction,
                double_counting,
                occupations,
            });
        }
        let pairs = self.pairs.iter().map(|&pair| {
            let bond_order = self.add_intersite(pair, density, &mut potential);
            let [left, right] = pair.sites.map(|site| sites[site].occupations.iter().map(trace).sum());
            let terms = pair.energy(left, right, bond_order);
            energy += terms.correction;
            terms
        }).collect();
        Ok(HubbardFields {
            energy,
            potential,
            sites,
            pairs,
        })
    }

    /// Linearized potential of the same density energy, used in the SCF
    /// orbital Hessian at fixed geometry.
    pub fn response(&self, variation: &[Matrix; 2]) -> Result<[Matrix; 2]> {
        self.validate_density(variation)?;
        let mut response = self.zeros();
        for site in &self.sites {
            let local = site
                .interaction
                .potential(&site.occupations(variation), false);
            site.lift(&local, &mut response);
        }
        for &pair in &self.pairs {
            self.add_intersite(pair, variation, &mut response);
        }
        Ok(response)
    }

    /// Project the intersite coherence and lift its variational potential.
    fn add_intersite(
        &self,
        pair: IntersiteInteraction,
        density: &[Matrix; 2],
        potential: &mut [Matrix; 2],
    ) -> f64 {
        let [left, right] = pair.sites.map(|i| &self.sites[i].covariant);
        let mut bond_order = 0.0;
        for (p, f) in density.iter().zip(potential) {
            let occupation = left.transpose_matmul_seq(&p.matmul_seq(right));
            bond_order += occupation.frobenius_dot(&occupation);
            let lifted = left.matmul_seq(&occupation).matmul_transpose_seq(right);
            for i in 0..self.dimension {
                for j in 0..self.dimension {
                    f[(i, j)] -= pair.v * (lifted[(i, j)] + lifted[(j, i)]);
                }
            }
        }
        bond_order
    }

    fn zeros(&self) -> [Matrix; 2] {
        std::array::from_fn(|_| Matrix::zeros(self.dimension, self.dimension))
    }

    fn validate_density(&self, density: &[Matrix; 2]) -> Result<()> {
        if density
            .iter()
            .any(|p| p.rows != self.dimension || !is_finite_symmetric(p))
        {
            return Err(DftError::InvalidInput(
                "Hubbard density must contain two finite symmetric AO matrices of the reference dimension"
                    .into(),
            ));
        }
        Ok(())
    }
}

impl Projector {
    fn occupations(&self, density: &[Matrix; 2]) -> [Matrix; 2] {
        density.each_ref().map(|p| {
            self.covariant
                .transpose_matmul_seq(&p.matmul_seq(&self.covariant))
        })
    }

    fn lift(&self, local: &[Matrix; 2], output: &mut [Matrix; 2]) {
        for (v, spin) in output.iter_mut().zip(local) {
            let term = self
                .covariant
                .matmul_seq(spin)
                .matmul_transpose_seq(&self.covariant);
            for (v, t) in v.as_mut_slice().iter_mut().zip(term.as_slice()) {
                *v += t;
            }
        }
    }
}

fn eri_index(m: usize, a: usize, b: usize, c: usize, d: usize) -> usize {
    ((a * m + b) * m + c) * m + d
}

fn trace(n: &Matrix) -> f64 {
    (0..n.rows).map(|a| n[(a, a)]).sum()
}

impl Interaction {
    fn validate(&self, m: usize) -> Result<()> {
        match self {
            Self::Dudarev { u_eff } if u_eff.is_finite() => Ok(()),
            Self::Liechtenstein { u, j, integrals }
                if u.is_finite()
                    && j.is_finite()
                    && m.checked_pow(4) == Some(integrals.len())
                    && integrals.iter().all(|x| x.is_finite()) =>
            {
                for a in 0..m {
                    for b in 0..m {
                        for c in 0..m {
                            for d in 0..m {
                                let v = integrals[eri_index(m, a, b, c, d)];
                                for (a, b, c, d) in [(b, a, c, d), (a, b, d, c), (c, d, a, b)] {
                                    let w = integrals[eri_index(m, a, b, c, d)];
                                    if (v - w).abs() > 1e-12 * (1.0 + v.abs().max(w.abs())) {
                                        return Err(DftError::InvalidInput(
                                            "on-site Coulomb tensor must have real chemists' integral symmetries".into(),
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(())
            }
            _ => Err(DftError::InvalidInput(
                "Hubbard interactions require finite parameters and a complete on-site tensor where applicable".into(),
            )),
        }
    }

    fn potential(&self, n: &[Matrix; 2], include_constant: bool) -> [Matrix; 2] {
        let m = n[0].rows;
        let half = if include_constant { 0.5 } else { 0.0 };
        match self {
            Self::Dudarev { u_eff } => std::array::from_fn(|s| {
                let mut result = Matrix::zeros(m, m);
                for a in 0..m {
                    for b in 0..m {
                        result[(a, b)] = u_eff * (if a == b { half } else { 0.0 } - n[s][(a, b)]);
                    }
                }
                result
            }),
            Self::Liechtenstein { u, j, integrals } => {
                let count = [trace(&n[0]), trace(&n[1])];
                let total = count[0] + count[1];
                let mut result = Self::interaction_fock(n, integrals);
                for s in 0..2 {
                    let shift = u * (total - half) - j * (count[s] - half);
                    for a in 0..m {
                        result[s][(a, a)] -= shift;
                    }
                }
                result
            }
        }
    }

    fn interaction_fock(n: &[Matrix; 2], integrals: &[f64]) -> [Matrix; 2] {
        let m = n[0].rows;
        let mut fock = std::array::from_fn(|_| Matrix::zeros(m, m));
        // Projected direct and same-spin exchange use the same on-site ERIs.
        for a in 0..m {
            for b in 0..m {
                for c in 0..m {
                    for d in 0..m {
                        let coulomb =
                            integrals[eri_index(m, a, b, c, d)] * (n[0][(c, d)] + n[1][(c, d)]);
                        let exchange = integrals[eri_index(m, a, c, b, d)];
                        for s in 0..2 {
                            fock[s][(a, b)] += coulomb - exchange * n[s][(c, d)];
                        }
                    }
                }
            }
        }
        fock
    }

    fn evaluate(&self, n: &[Matrix; 2]) -> (f64, f64, [Matrix; 2]) {
        let count = [trace(&n[0]), trace(&n[1])];
        match self {
            Self::Dudarev { u_eff } => {
                let energy = 0.5
                    * u_eff
                    * (count[0] + count[1] - n.iter().map(|p| p.frobenius_dot(p)).sum::<f64>());
                (energy, 0.0, self.potential(n, true))
            }
            Self::Liechtenstein { u, j, integrals } => {
                let interaction = Self::interaction_fock(n, integrals);
                let energy = 0.5
                    * (n[0].frobenius_dot(&interaction[0]) + n[1].frobenius_dot(&interaction[1]));
                let total = count[0] + count[1];
                let double_count = 0.5
                    * (u * total * (total - 1.0)
                        - j * count.iter().map(|&x| x * (x - 1.0)).sum::<f64>());
                let mut potential = interaction;
                for s in 0..2 {
                    let shift = u * (total - 0.5) - j * (count[s] - 0.5);
                    for a in 0..n[s].rows {
                        potential[s][(a, a)] -= shift;
                    }
                }
                (energy, double_count, potential)
            }
        }
    }
}
