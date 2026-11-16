// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! The public periodic API: a [`Cell`] ties a molecule, a lattice and a
//! basis together and hands back the matrices a periodic SCF needs.
//!
//! # What this layer is for
//!
//! Everything below it ([`super::ewald`], [`super::integrals`],
//! [`super::planewave`]) is deliberately free-standing and gated on its
//! own identities. This module is the seam where they meet: it owns the
//! cutoffs, caches the image blocks that every k-point reuses, and
//! presents one place to ask "what is `S(k)`" without the caller having to
//! know that the answer is a lattice sum.
//!
//! # Cutoffs
//!
//! Two independent ones, for two different reasons:
//!
//! - `overlap_cutoff` bounds the real-space sums of the SHORT-range
//!   one-electron operators. Gaussian overlap dies exponentially, so this
//!   is a genuine convergence parameter with a well-defined limit.
//! - the Ewald parameters bound the CONDITIONALLY convergent Coulomb sums,
//!   where "sum further" is not merely slow but ill-defined; those live in
//!   [`super::ewald::EwaldParams`] and are chosen from the cell volume.
//!
//! Conflating the two is a classic periodic-code bug: a Coulomb sum
//! truncated like an overlap sum converges to the wrong number, silently
//! and geometry-dependently.
//!
//! # Scope
//!
//! This layer supplies one-electron matrices, nuclear electrostatics, the
//! periodic Hartree operator, and translated ECP matrices and derivatives.
//! The SCF driver builds finite-k exact exchange and LDA/GGA/meta-GGA
//! potentials on top. Dimensional source kernels and Gaussian basis/grid
//! drivers are available separately. Their nuclear/Hartree and SCF/DFPT
//! connections and periodic VV10/CHG remain tracked work.

use crate::basis::{BasisSet, MolBasis};
use crate::error::Result;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::ewald::{ewald_energy, EwaldParams};
use crate::periodic::integrals::{bloch, image_blocks, ImageBlock, Operator};
use crate::periodic::Lattice;
use crate::system::Molecule;
use std::sync::atomic::{AtomicU64, Ordering};

fn next_cell_identity() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// A periodic system: the contents of one cell, plus its lattice.
pub struct Cell {
    // These inputs are deliberately immutable outside the crate. Every
    // matrix below is a cache derived from all three, so exposing mutable
    // fields would let safe Rust silently combine a new geometry/lattice
    // with stale overlap, kinetic, ECP, and Ewald data.
    pub(crate) molecule: Molecule,
    pub(crate) lattice: Lattice,
    pub(crate) basis: MolBasis,
    /// Real-space cutoff for the short-range one-electron lattice sums.
    pub(crate) overlap_cutoff: f64,
    /// Process-local identity of this immutable cache bundle. Rebuilding a
    /// geometry or lattice creates a new identity, allowing derivative APIs
    /// to reject an SCF result produced by a different cell.
    identity: u64,
    ewald: EwaldParams,
    overlap_images: Vec<ImageBlock>,
    kinetic_images: Vec<ImageBlock>,
    /// Absolutely convergent translated semi-local ECP operator.  Empty for
    /// all-electron cells.  The same cutoff is used initially for AO and ECP
    /// images; both are explicit convergence parameters in the public plan.
    ecp_images: Vec<ImageBlock>,
    /// Built on first use rather than in the constructor: the Ewald split
    /// makes this the expensive one of the three, and a caller who only
    /// wants `S(k)` should not pay for it.
    nuclear_images: std::sync::OnceLock<Vec<ImageBlock>>,
    /// Primitive exponent that separates the compact from the diffuse part in the range-separated Hartree build; a performance parameter that does not change the result beyond the screening tolerance. The default follows the cell volume.
    hartree_compact_exponent: f64,
}

/// Nuclear derivatives of the one-electron image blocks, built once and
/// Bloch-summed at as many k points as wanted.
///
/// Kept out of [`Cell`] deliberately. These hold `natoms * 3 * nao^2`
/// numbers per lattice translation — for a cell with many images that is
/// the largest array in a periodic gradient, so whether to keep it alive is
/// the caller's decision, not a side effect of building a `Cell`.
pub struct DerivativeImages {
    pub overlap: Vec<crate::periodic::integrals::ImageDerivBlock>,
    pub kinetic: Vec<crate::periodic::integrals::ImageDerivBlock>,
    pub nuclear: Vec<crate::periodic::integrals::ImageDerivBlock>,
    pub ecp: Vec<crate::periodic::integrals::ImageDerivBlock>,
    natoms: usize,
    nao: usize,
}

impl DerivativeImages {
    /// `dS/dR` at `k`, indexed `[atom][axis]` as `(real, imaginary)`.
    pub fn overlap(&self, k: Vec3) -> Vec<[(Matrix, Matrix); 3]> {
        crate::periodic::integrals::bloch_derivative(&self.overlap, k, self.natoms, self.nao)
    }

    /// `dT/dR` at `k`.
    pub fn kinetic(&self, k: Vec3) -> Vec<[(Matrix, Matrix); 3]> {
        crate::periodic::integrals::bloch_derivative(&self.kinetic, k, self.natoms, self.nao)
    }

    /// `dV/dR` at `k`.
    pub fn nuclear(&self, k: Vec3) -> Vec<[(Matrix, Matrix); 3]> {
        crate::periodic::integrals::bloch_derivative(&self.nuclear, k, self.natoms, self.nao)
    }

    /// Derivative of the translated semi-local ECP operator at `k`.
    pub fn ecp(&self, k: Vec3) -> Vec<[(Matrix, Matrix); 3]> {
        crate::periodic::integrals::bloch_derivative(&self.ecp, k, self.natoms, self.nao)
    }

    /// `dH_core/dR = dT/dR + dV/dR + dU_ECP/dR` at `k`.
    pub fn core_hamiltonian(&self, k: Vec3) -> Result<Vec<[(Matrix, Matrix); 3]>> {
        let mut t = self.kinetic(k);
        let v = self.nuclear(k);
        let u = self.ecp(k);
        for ((ta, va), ua) in t.iter_mut().zip(v).zip(u) {
            for ax in 0..3 {
                add_into(&mut ta[ax].0, &va[ax].0);
                add_into(&mut ta[ax].1, &va[ax].1);
                add_into(&mut ta[ax].0, &ua[ax].0);
                add_into(&mut ta[ax].1, &ua[ax].1);
            }
        }
        Ok(t)
    }
}

fn add_into(dst: &mut Matrix, src: &Matrix) {
    for (o, v) in dst.as_mut_slice().iter_mut().zip(src.as_slice()) {
        *o += *v;
    }
}

impl Cell {
    /// Cached translated ECP operators used by the SCF core Hamiltonian.
    pub(crate) fn ecp_image_blocks(&self) -> &[ImageBlock] {
        &self.ecp_images
    }

    /// Build a cell and pre-compute the image blocks every k-point shares.
    ///
    /// `overlap_cutoff` in Bohr; 20 is a reasonable default for valence
    /// bases and is what [`Cell::with_default_cutoff`] uses.
    pub fn new(
        molecule: Molecule,
        lattice: Lattice,
        set: &BasisSet,
        overlap_cutoff: f64,
    ) -> Result<Self> {
        if !(overlap_cutoff.is_finite() && overlap_cutoff > 0.0) {
            return Err(crate::error::DftError::InvalidInput(
                "periodic overlap cutoff must be finite and positive".to_string(),
            ));
        }
        let vectors_are_finite = lattice
            .a
            .iter()
            .all(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite());
        if !vectors_are_finite || !lattice.volume().is_finite() || lattice.volume().abs() <= 1e-12 {
            return Err(crate::error::DftError::InvalidInput(
                "periodic lattice vectors must be finite and define a non-zero volume".to_string(),
            ));
        }
        let basis = MolBasis::build(&molecule, set)?;
        let ewald = EwaldParams::for_accuracy(&lattice, molecule.len(), 1e-12);
        let overlap_images = image_blocks(&basis, &lattice, overlap_cutoff, Operator::Overlap);
        let kinetic_images = image_blocks(&basis, &lattice, overlap_cutoff, Operator::Kinetic);
        let ecp_images = crate::periodic::integrals::ecp_image_blocks(
            &basis,
            &lattice,
            overlap_cutoff,
            overlap_cutoff,
        )?;
        let hartree_compact_exponent =
            crate::periodic::screened_coulomb::default_compact_exponent(lattice.volume());
        Ok(Self {
            molecule,
            lattice,
            basis,
            overlap_cutoff,
            identity: next_cell_identity(),
            ewald,
            overlap_images,
            kinetic_images,
            ecp_images,
            nuclear_images: std::sync::OnceLock::new(),
            hartree_compact_exponent,
        })
    }

    pub fn with_default_cutoff(
        molecule: Molecule,
        lattice: Lattice,
        set: &BasisSet,
    ) -> Result<Self> {
        Self::new(molecule, lattice, set, 20.0)
    }

    /// Immutable contents of the primitive cell.
    pub fn molecule(&self) -> &Molecule {
        &self.molecule
    }

    /// Immutable Bravais lattice used by every cached image sum.
    pub fn lattice(&self) -> &Lattice {
        &self.lattice
    }

    /// Geometry-bound AO/ECP basis used by every cached matrix.
    pub fn basis(&self) -> &MolBasis {
        &self.basis
    }

    /// Real-space cutoff used to construct the cached short-range images.
    pub fn overlap_cutoff(&self) -> f64 {
        self.overlap_cutoff
    }

    pub(crate) fn identity(&self) -> u64 {
        self.identity
    }

    /// Return a new cell for a changed geometry, rebuilding every dependent
    /// basis, image sum, Ewald parameter, and lazy cache from scratch.
    ///
    /// A basis set is required because atom identities may also have
    /// changed. The original cell remains valid and unchanged.
    pub fn rebuilt_with_molecule(&self, molecule: Molecule, set: &BasisSet) -> Result<Self> {
        Self::new(molecule, self.lattice.clone(), set, self.overlap_cutoff)
    }

    /// Number of spherical AOs per cell.
    pub fn nao(&self) -> usize {
        self.basis.nao_sph
    }

    /// Electrons per cell, with ECP cores removed.
    pub fn n_electrons(&self) -> Result<usize> {
        let n = self.molecule.n_electrons()?;
        n.checked_sub(self.basis.n_core_electrons as usize)
            .ok_or_else(|| {
                crate::error::DftError::InvalidInput(format!(
                    "charge {} removes electrons from the ECP core; only {} explicit electrons remain before subtracting {} core electrons",
                    self.molecule.charge, n, self.basis.n_core_electrons
                ))
            })
    }

    /// Overlap matrix at `k`, as `(real, imaginary)`.
    pub fn overlap(&self, k: Vec3) -> (Matrix, Matrix) {
        bloch(&self.overlap_images, k)
    }

    /// Kinetic-energy matrix at `k`, as `(real, imaginary)`.
    pub fn kinetic(&self, k: Vec3) -> (Matrix, Matrix) {
        bloch(&self.kinetic_images, k)
    }

    /// Nuclear-attraction matrix at `k`, as `(real, imaginary)`.
    ///
    /// Unlike `S` and `T` this is not a cutoff sum — see
    /// [`super::nuclear`] — and it carries the `G = 0` background shift
    /// that pairs with the one in [`Cell::nuclear_repulsion`]. Neither
    /// piece is meaningful alone; their sum is.
    pub fn nuclear(&self, k: Vec3) -> (Matrix, Matrix) {
        bloch(self.nuclear_image_blocks(), k)
    }

    /// Core Hamiltonian `T + V_ion + U_ECP` at `k`.
    pub fn core_hamiltonian(&self, k: Vec3) -> (Matrix, Matrix) {
        let (mut re, mut im) = self.kinetic(k);
        let (vre, vim) = self.nuclear(k);
        add_into(&mut re, &vre);
        add_into(&mut im, &vim);
        if !self.ecp_images.is_empty() {
            let (ure, uim) = bloch(&self.ecp_images, k);
            add_into(&mut re, &ure);
            add_into(&mut im, &uim);
        }
        (re, im)
    }

    /// Convert a Gamma-point AO density matrix into real-space `D(T)` blocks.
    ///
    /// For a one-point Gamma mesh, the inverse Bloch transform is constant:
    /// `D(T) = P(Gamma)` for every retained lattice translation. Using only
    /// a home-cell `D(0)` would instead describe a localized density and
    /// would give the wrong electron count `Tr[P S(Gamma)]` whenever basis
    /// functions overlap between cells.
    pub fn gamma_density_images(&self, density: &Matrix) -> Result<Vec<ImageBlock>> {
        let zero = Matrix::zeros(self.nao(), self.nao());
        self.density_images_from_k(
            &[Vec3::zero()],
            &[(density.clone(), zero)],
            self.overlap_cutoff,
        )
    }

    /// Inverse Bloch transform of uniformly weighted k-point density matrices.
    ///
    /// The convention matches [`crate::periodic::integrals::bloch`]:
    /// `P(k) = sum_T exp(i k.T) D(T)`, hence
    /// `D(T) = (1/Nk) sum_k exp(-i k.T) P(k)`. The current real integral
    /// engine requires a time-reversal-symmetric mesh/density, for which the
    /// imaginary part of every transformed `D(T)` cancels.
    pub fn density_images_from_k(
        &self,
        k_points: &[Vec3],
        density: &[(Matrix, Matrix)],
        cutoff: f64,
    ) -> Result<Vec<ImageBlock>> {
        if k_points.is_empty() || k_points.len() != density.len() {
            return Err(crate::error::DftError::InvalidInput(
                "k-point density needs one matrix pair per non-empty k mesh".to_string(),
            ));
        }
        if !(cutoff.is_finite() && cutoff > 0.0) {
            return Err(crate::error::DftError::InvalidInput(
                "density image cutoff must be finite and positive".to_string(),
            ));
        }
        let n = self.nao();
        for (index, (k, (re, im))) in k_points.iter().zip(density).enumerate() {
            if !k.x.is_finite() || !k.y.is_finite() || !k.z.is_finite() {
                return Err(crate::error::DftError::InvalidInput(format!(
                    "k point {index} is non-finite"
                )));
            }
            if re.rows != n || re.cols != n || im.rows != n || im.cols != n {
                return Err(crate::error::DftError::InvalidInput(format!(
                    "k-point density {index} must contain two {n}x{n} matrices"
                )));
            }
            if re
                .as_slice()
                .iter()
                .chain(im.as_slice())
                .any(|x| !x.is_finite())
            {
                return Err(crate::error::DftError::InvalidInput(format!(
                    "k-point density {index} contains a non-finite value"
                )));
            }
            let mut hermiticity = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    hermiticity = hermiticity
                        .max((re[(i, j)] - re[(j, i)]).abs())
                        .max((im[(i, j)] + im[(j, i)]).abs());
                }
            }
            if hermiticity > 1e-10 {
                return Err(crate::error::DftError::InvalidInput(format!(
                    "k-point density {index} is not Hermitian (maximum error {hermiticity:e})"
                )));
            }
        }

        let inv_nk = 1.0 / k_points.len() as f64;
        let mut blocks = Vec::new();
        for translation in self.lattice.translations_within(cutoff) {
            let mut re_t = Matrix::zeros(n, n);
            let mut im_t = Matrix::zeros(n, n);
            for (k, (re, im)) in k_points.iter().zip(density) {
                let phase = k.dot(translation);
                let (cosine, sine) = (phase.cos(), phase.sin());
                for i in 0..n {
                    for j in 0..n {
                        re_t[(i, j)] += inv_nk * (cosine * re[(i, j)] + sine * im[(i, j)]);
                        im_t[(i, j)] += inv_nk * (cosine * im[(i, j)] - sine * re[(i, j)]);
                    }
                }
            }
            let imaginary_residual = im_t
                .as_slice()
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0_f64, f64::max);
            if imaginary_residual > 1e-9 {
                return Err(crate::error::DftError::InvalidInput(format!(
                    "inverse Bloch density is complex at {translation:?} (residual {imaginary_residual:e}); use a time-reversal-symmetric k mesh"
                )));
            }
            blocks.push(ImageBlock {
                translation,
                matrix: re_t,
            });
        }
        self.validate_density(&blocks)?;
        Ok(blocks)
    }

    /// Validate a real-space periodic density and return its electron count.
    ///
    /// Every non-zero translation must have its transpose partner,
    /// `D(-T) = D(T)^T`; this is the real-space form of Hermiticity at every
    /// k point. Rejecting malformed blocks here avoids producing a plausible
    /// but non-Hermitian periodic Hartree matrix.
    pub fn electron_count(&self, density: &[ImageBlock]) -> Result<f64> {
        self.validate_density(density)?;
        Ok(crate::periodic::grid::electron_count(&self.basis, density))
    }

    /// Periodic Hartree image blocks for a validated real-space density.
    ///
    /// The real-space cutoff must cover both the Gaussian overlap reach and
    /// the screened Ewald kernel. The cell diameter is added to the latter
    /// because the cutoff applies to pair distances, not lattice-vector
    /// lengths.
    ///
    /// This is the range-separated, screened build
    /// ([`crate::periodic::screened_coulomb`]); the exhaustive supercell
    /// build that it is tested against remains available as
    /// [`Cell::reference_hartree_image_blocks`].
    pub fn hartree_image_blocks(&self, density: &[ImageBlock]) -> Result<Vec<ImageBlock>> {
        self.validate_density(density)?;
        crate::periodic::screened_coulomb::screened_hartree_image_blocks(
            &self.basis,
            &self.lattice,
            density,
            &self.hartree_screening(),
        )
    }

    /// Screening of the cell's Hartree build. Shell pairs are kept wherever
    /// their Gaussian product exceeds the screening tolerance, independently
    /// of the one-electron image cutoff, so the Hartree energy converges with
    /// its own tolerance.
    pub fn hartree_screening(&self) -> crate::periodic::screened_coulomb::HartreeScreening {
        crate::periodic::screened_coulomb::HartreeScreening {
            compact_exponent: self.hartree_compact_exponent,
            ..Default::default()
        }
    }

    /// The same cell with another Hartree split exponent (Bohr^-2).
    pub fn with_hartree_compact_exponent(mut self, value: f64) -> Result<Self> {
        if !(value.is_finite() && value > 0.0) {
            return Err(crate::error::DftError::InvalidInput(
                "hartree_compact_exponent must be finite and positive".to_string(),
            ));
        }
        self.hartree_compact_exponent = value;
        Ok(self)
    }

    /// The exhaustive supercell Hartree build: every `erfc` quartet of the
    /// image supercell within [`Cell::hartree_real_space_cutoff`]. Exact but
    /// cubic in the number of images; kept as the oracle.
    pub fn reference_hartree_image_blocks(
        &self,
        density: &[ImageBlock],
    ) -> Result<Vec<ImageBlock>> {
        self.validate_density(density)?;
        let cutoff = self.hartree_real_space_cutoff();
        Ok(crate::periodic::coulomb::hartree_image_blocks(
            &self.basis,
            &self.lattice,
            density,
            cutoff,
            self.ewald,
        ))
    }

    /// Real-space AO-image cutoff used by the Ewald Hartree build.
    ///
    /// A caller constructing `D(T)` from k-point matrices should retain
    /// density translations out to twice this value: two AO images inside
    /// the Hartree supercell can differ by that amount.
    pub fn hartree_real_space_cutoff(&self) -> f64 {
        let diameter = self
            .molecule
            .atoms
            .iter()
            .flat_map(|a| {
                self.molecule
                    .atoms
                    .iter()
                    .map(move |b| (a.position - b.position).norm())
            })
            .fold(0.0_f64, f64::max);
        self.overlap_cutoff.max(self.ewald.rcut + diameter)
    }

    /// Periodic Hartree matrix `J(k)` for a caller-supplied density.
    pub fn hartree(&self, density: &[ImageBlock], k: Vec3) -> Result<(Matrix, Matrix)> {
        let blocks = self.hartree_image_blocks(density)?;
        Ok(bloch(&blocks, k))
    }

    fn validate_density(&self, density: &[ImageBlock]) -> Result<()> {
        if density.is_empty() {
            return Err(crate::error::DftError::InvalidInput(
                "periodic density must contain at least one image block".to_string(),
            ));
        }
        let n = self.nao();
        for (ib, block) in density.iter().enumerate() {
            let t = block.translation;
            if !t.x.is_finite() || !t.y.is_finite() || !t.z.is_finite() {
                return Err(crate::error::DftError::InvalidInput(format!(
                    "periodic density block {ib} has a non-finite translation"
                )));
            }
            if block.matrix.rows != n || block.matrix.cols != n {
                return Err(crate::error::DftError::InvalidInput(format!(
                    "periodic density block {ib} is {}x{}, expected {n}x{n}",
                    block.matrix.rows, block.matrix.cols
                )));
            }
            if block.matrix.as_slice().iter().any(|x| !x.is_finite()) {
                return Err(crate::error::DftError::InvalidInput(format!(
                    "periodic density block {ib} contains a non-finite value"
                )));
            }
            if density[..ib]
                .iter()
                .any(|other| (other.translation - t).norm() < 1e-8)
            {
                return Err(crate::error::DftError::InvalidInput(format!(
                    "periodic density contains duplicate blocks at translation {t:?}"
                )));
            }
            let Some(mirror) = density
                .iter()
                .find(|other| (other.translation + t).norm() < 1e-8)
            else {
                return Err(crate::error::DftError::InvalidInput(format!(
                    "periodic density block at {t:?} has no -T transpose partner"
                )));
            };
            let mut worst = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    worst = worst.max((block.matrix[(i, j)] - mirror.matrix[(j, i)]).abs());
                }
            }
            if worst > 1e-10 {
                return Err(crate::error::DftError::InvalidInput(format!(
                    "periodic density violates D(-T)=D(T)^T at {t:?} (maximum error {worst:e})"
                )));
            }
        }
        Ok(())
    }

    /// The cached nuclear image blocks, built on first call.
    pub fn nuclear_image_blocks(&self) -> &[ImageBlock] {
        self.nuclear_images.get_or_init(|| {
            let charges = self.basis.effective_charges(&self.molecule);
            crate::periodic::nuclear::nuclear_image_blocks(
                &self.basis,
                &self.molecule,
                &charges,
                &self.lattice,
                self.overlap_cutoff,
                self.ewald,
            )
        })
    }

    /// Build the nuclear derivatives of all three one-electron operators.
    ///
    /// This is the expensive call in a periodic gradient and it allocates
    /// the largest arrays, so it is explicit: build once, then Bloch-sum at
    /// every k point through [`DerivativeImages`].
    pub fn derivative_images(&self) -> DerivativeImages {
        use crate::periodic::integrals::image_derivative_blocks;
        let charges = self.basis.effective_charges(&self.molecule);
        DerivativeImages {
            overlap: image_derivative_blocks(
                &self.basis,
                &self.lattice,
                self.overlap_cutoff,
                Operator::Overlap,
            ),
            kinetic: image_derivative_blocks(
                &self.basis,
                &self.lattice,
                self.overlap_cutoff,
                Operator::Kinetic,
            ),
            nuclear: crate::periodic::nuclear::nuclear_derivative_image_blocks(
                &self.basis,
                &self.molecule,
                &charges,
                &self.lattice,
                self.overlap_cutoff,
                self.ewald,
            ),
            ecp: crate::periodic::integrals::ecp_derivative_image_blocks(
                &self.basis,
                &self.lattice,
                self.overlap_cutoff,
                self.overlap_cutoff,
                &self.ecp_images,
            ),
            natoms: self.molecule.len(),
            nao: self.basis.nao_sph,
        }
    }

    /// Ewald parameters of the nuclear attraction and repulsion sums.
    pub(crate) fn ewald_params(&self) -> EwaldParams {
        self.ewald
    }

    /// The `G = 0` background shift carried by [`Cell::nuclear`], as a
    /// multiple of the overlap.
    ///
    /// Exposed because it is the one convention-dependent number in the
    /// one-electron Hamiltonian: it cancels against the corresponding
    /// pieces of the Hartree and nuclear-nuclear terms, so anything that
    /// compares `V` against a molecular reference has to account for it.
    pub fn background_shift(&self) -> f64 {
        let ztot: f64 = self.basis.effective_charges(&self.molecule).iter().sum();
        std::f64::consts::PI * ztot
            / (self.ewald.alpha * self.ewald.alpha * self.lattice.volume().abs())
    }

    /// How many lattice images actually contribute, per operator. Useful
    /// for judging whether `overlap_cutoff` is doing anything: if this
    /// stops growing with the cutoff, the sum has converged.
    pub fn image_counts(&self) -> (usize, usize) {
        (self.overlap_images.len(), self.kinetic_images.len())
    }

    /// Nuclear repulsion per cell, by Ewald summation over `Z_eff`.
    ///
    /// Conditionally convergent, hence Ewald rather than a cutoff — see
    /// [`super::ewald`]. Uses effective charges so an ECP basis is
    /// consistent with the electron count.
    ///
    /// **This does not reduce to the molecular value in a large cell, and
    /// should not.** Nuclei carry net positive charge, so the periodic
    /// energy is defined only alongside a neutralising background, whose
    /// `G = 0` contribution is included here. That term cancels against
    /// the matching one in the electronic energy; only the total is a
    /// physical quantity. Comparing this number alone against a molecular
    /// nuclear repulsion is a category error.
    pub fn nuclear_repulsion(&self) -> f64 {
        let charges = self.basis.effective_charges(&self.molecule);
        let pos: Vec<Vec3> = self.molecule.atoms.iter().map(|a| a.position).collect();
        ewald_energy(&self.lattice, &pos, &charges, self.ewald)
    }

    /// Net charge of the cell: nuclei minus electrons.
    pub fn net_charge(&self) -> f64 {
        let z: f64 = self.basis.effective_charges(&self.molecule).iter().sum();
        z - self.n_electrons().map(|n| n as f64).unwrap_or(0.0)
    }

    /// Charged-cell (Makov–Payne) MONOPOLE correction, applied
    /// AUTOMATICALLY when the cell carries a formal charge, zero otherwise.
    ///
    /// The neutralising background makes a charged cell's energy finite,
    /// but it cannot remove the charge's interaction with its own periodic
    /// images. That artefact falls off only as `1/L`, so for a molecular
    /// ion or a charged defect it is routinely the dominant error — larger
    /// than the basis-set incompleteness one would otherwise worry about.
    /// It is therefore applied by default, and exposing
    /// it separately keeps it visible rather than baked into a total.
    ///
    /// # Consistency with the molecular (full-DFT) treatment of charge
    ///
    /// The charge bookkeeping matches exactly:
    /// `net_charge = sum Z_eff - n_electrons`, and with
    /// `n_electrons = sum Z - q - n_core` this reduces to the `q` the
    /// caller set on the molecule, ECPs included.
    ///
    /// The ENERGY, however, is only partly comparable, and two limits are
    /// worth being explicit about:
    ///
    /// - Only the MONOPOLE term is included. The quadrupole term needs the
    ///   second radial moment of the total charge density, which for an ion
    ///   is dominated by the diffuse electronic tail rather than by nuclear
    ///   positions; there is no converged density at this layer, so
    ///   inventing one would produce a plausible wrong number. Use
    ///   [`crate::periodic::ewald::makov_payne_with_moment`] when a density
    ///   is available.
    /// - The `G = 0` divergences of the nuclear-nuclear, electron-nuclear
    ///   and electron-electron terms cancel only in combination. What is
    ///   handled here is the NUCLEAR share. When the periodic two-electron
    ///   terms land, that bookkeeping has to be done once across all three
    ///   — applying this correction on top of an independently
    ///   background-corrected electronic energy would double-count.
    pub fn charged_cell_correction(&self) -> f64 {
        crate::periodic::ewald::makov_payne_correction(&self.lattice, self.net_charge())
    }

    /// Nuclear repulsion with the charged-cell correction already applied
    /// when the cell is charged. This is what a caller normally wants.
    pub fn nuclear_energy(&self) -> f64 {
        self.nuclear_repulsion() + self.charged_cell_correction()
    }

    /// Forces on the nuclei from the periodic nuclear repulsion.
    ///
    /// This is the nuclear share only. A total force also needs the
    /// electronic terms, which do not exist yet.
    pub fn nuclear_forces(&self) -> Vec<Vec3> {
        let charges = self.basis.effective_charges(&self.molecule);
        let pos: Vec<Vec3> = self.molecule.atoms.iter().map(|a| a.position).collect();
        crate::periodic::ewald::ewald_forces(&self.lattice, &pos, &charges, self.ewald)
    }

    /// Strain derivative `dE/d(epsilon)` of the nuclear Ewald energy, and
    /// the corresponding stress `dE/d(epsilon) / V`.
    pub fn nuclear_stress(&self) -> [[f64; 3]; 3] {
        let charges = self.basis.effective_charges(&self.molecule);
        let pos: Vec<Vec3> = self.molecule.atoms.iter().map(|a| a.position).collect();
        let d = crate::periodic::ewald::ewald_stress(&self.lattice, &pos, &charges, self.ewald);
        let v = self.lattice.volume().abs();
        let mut out = [[0.0; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                out[a][b] = d[a][b] / v;
            }
        }
        out
    }

    /// Monkhorst-Pack k-point mesh for this lattice.
    pub fn k_points(&self, mesh: [usize; 3]) -> Vec<Vec3> {
        crate::periodic::integrals::monkhorst_pack(&self.lattice, mesh)
    }

    /// Fourier component `rho_mn(G)` of the AO pair density.
    pub fn plane_wave(&self, g: Vec3) -> (Matrix, Matrix) {
        crate::periodic::planewave::plane_wave_matrix(&self.basis, g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;
    use crate::linalg::symmetric_eigen;
    use crate::periodic::integrals::hermitian_as_real;
    use crate::system::Atom;

    fn h2_cell(l: f64) -> Cell {
        let mol = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 1.4),
            },
        ]);
        let set = data::load("sto-3g").unwrap();
        Cell::new(mol, Lattice::cubic(l), &set, 22.0).unwrap()
    }

    /// A cell far larger than everything in it must reproduce the MOLECULAR
    /// answers: overlap, kinetic, and the nuclear repulsion. This is the
    /// limit that connects the periodic API to the paths already verified
    /// against PySCF, and it is the check a user can repeat on their own
    /// system to convince themselves the cutoffs are adequate.
    #[test]
    fn isolated_limit_reproduces_the_molecule() {
        let cell = h2_cell(60.0);
        let (re, im) = cell.overlap(Vec3::zero());
        let s = crate::integrals::overlap_matrix(&cell.basis);
        assert!(
            re.max_abs_diff(&s) < 1e-10,
            "overlap differs from molecular"
        );
        assert!(im.as_slice().iter().all(|v| v.abs() < 1e-14));

        let (tre, _) = cell.kinetic(Vec3::zero());
        let t = crate::integrals::kinetic_matrix(&cell.basis);
        assert!(
            tre.max_abs_diff(&t) < 1e-10,
            "kinetic differs from molecular"
        );
    }

    /// The core Hamiltonian is exactly its two pieces, and the background
    /// shift is exactly its closed form with EFFECTIVE charges.
    ///
    /// `Z_eff` rather than `Z` is the load-bearing part: an ECP basis
    /// removes core electrons, and if the nuclear attraction kept bare `Z`
    /// the cell would be charged by the core count without anything saying
    /// so. The same slip cost this project a wrong ECP Hessian once.
    #[test]
    fn core_hamiltonian_is_its_pieces_and_uses_effective_charges() {
        let cell = h2_cell(20.0);
        let b = cell.lattice.reciprocal();
        let k = b[0] * 0.2 - b[2] * 0.35;
        let (hre, him) = cell.core_hamiltonian(k);
        let (tre, tim) = cell.kinetic(k);
        let (vre, vim) = cell.nuclear(k);
        for i in 0..cell.nao() {
            for j in 0..cell.nao() {
                assert!((hre[(i, j)] - tre[(i, j)] - vre[(i, j)]).abs() < 1e-14);
                assert!((him[(i, j)] - tim[(i, j)] - vim[(i, j)]).abs() < 1e-14);
            }
        }
        let ztot: f64 = cell.basis.effective_charges(&cell.molecule).iter().sum();
        assert!((ztot - 2.0).abs() < 1e-12, "H2 carries Z_eff = 2");
        let want = std::f64::consts::PI * ztot
            / (cell.ewald.alpha * cell.ewald.alpha * cell.lattice.volume().abs());
        assert!((cell.background_shift() - want).abs() < 1e-14);
    }

    /// The assembled `dH_core/dR` against a finite difference of
    /// `H_core(k)`, through the public API.
    ///
    /// The pieces are gated in their own modules; what this adds is the
    /// wiring — that `Cell` hands the derivative path the same charges,
    /// the same cutoffs and the same Ewald parameters it hands the energy
    /// path. A mismatch in any of those produces a gradient that is
    /// self-consistently wrong, which no module-level test can see.
    #[test]
    fn derivative_images_match_finite_differences_of_the_core_hamiltonian() {
        let set = data::load("sto-3g").unwrap();
        let mol = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.1, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.3, 1.5),
            },
        ]);
        let lat = Lattice::cubic(6.0);
        let cell = Cell::new(mol.clone(), lat.clone(), &set, 8.0).unwrap();
        let b = lat.reciprocal();
        let k = b[0] * 0.25 + b[1] * 0.1;
        let ana = cell.derivative_images().core_hamiltonian(k).unwrap();

        let h = 1e-4;
        let n = cell.nao();
        for a in 0..mol.len() {
            for ax in 0..3 {
                let mut mats = Vec::new();
                for s in [1.0_f64, -1.0] {
                    let mut m = mol.clone();
                    let mut pos = m.atoms[a].position;
                    match ax {
                        0 => pos.x += s * h,
                        1 => pos.y += s * h,
                        _ => pos.z += s * h,
                    }
                    m.atoms[a].position = pos;
                    let c = Cell::new(m, lat.clone(), &set, 8.0).unwrap();
                    mats.push(c.core_hamiltonian(k));
                }
                for i in 0..n {
                    for j in 0..n {
                        let fd_re = (mats[0].0[(i, j)] - mats[1].0[(i, j)]) / (2.0 * h);
                        let fd_im = (mats[0].1[(i, j)] - mats[1].1[(i, j)]) / (2.0 * h);
                        assert!(
                            (ana[a][ax].0[(i, j)] - fd_re).abs() < 2e-6,
                            "Re dH/dR[{a}][{ax}]({i},{j}): {} vs FD {fd_re}",
                            ana[a][ax].0[(i, j)]
                        );
                        assert!(
                            (ana[a][ax].1[(i, j)] - fd_im).abs() < 2e-6,
                            "Im dH/dR[{a}][{ax}]({i},{j}): {} vs FD {fd_im}",
                            ana[a][ax].1[(i, j)]
                        );
                    }
                }
            }
        }
    }

    /// The nuclear repulsion does NOT approach the molecular value in a
    /// large cell, and that is correct rather than a bug.
    ///
    /// A set of nuclei always carries net positive charge, so the periodic
    /// electrostatic energy is only defined together with a neutralising
    /// background — the `G = 0` term `-pi Q^2 / (2 alpha^2 V)`. It is a
    /// real, conventional part of the periodic energy and it does not
    /// vanish as the cell grows; it cancels instead against the matching
    /// background in the ELECTRONIC energy, which is why only the total is
    /// meaningful. Asserting a molecular limit here would be asserting a
    /// physically wrong thing, so what is checked is the sign and the fact
    /// that the gap is the background rather than an accident.
    #[test]
    fn nuclear_repulsion_carries_the_neutralising_background() {
        let cell = h2_cell(60.0);
        let got = cell.nuclear_repulsion();
        let isolated = 1.0 / 1.4;
        assert!(
            got < isolated,
            "the background must lower the energy: {got} vs isolated {isolated}"
        );
        // Two protons, so Q = 2 and the background scales as Q^2/V: in a
        // 60-Bohr cell it is tens of millihartree, far above any cutoff
        // error, and it must shrink as the cell grows.
        let bigger = h2_cell(90.0).nuclear_repulsion();
        assert!(
            (bigger - isolated).abs() < (got - isolated).abs(),
            "the background should shrink with volume: {got} then {bigger}"
        );
    }

    /// The overlap matrix must be positive definite at every k — it is a
    /// Gram matrix of linearly independent functions. A negative eigenvalue
    /// means the lattice sum has been truncated inconsistently between
    /// images, which is the failure that makes a periodic SCF diverge for
    /// reasons that look electronic.
    #[test]
    fn overlap_is_positive_definite_across_the_brillouin_zone() {
        let cell = h2_cell(6.0);
        for k in cell.k_points([3, 3, 3]) {
            let (re, im) = cell.overlap(k);
            let (vals, _) = symmetric_eigen(&hermitian_as_real(&re, &im)).unwrap();
            let lowest = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            assert!(lowest > 1e-8, "S(k) has eigenvalue {lowest} at k = {k:?}");
        }
    }

    /// Raising the cutoff must not change a converged result, and the image
    /// count must stop growing. This is how a user decides the cutoff is
    /// adequate, so it is worth having as a test rather than folklore.
    #[test]
    fn results_are_converged_with_respect_to_the_cutoff() {
        let mol = Molecule::new(vec![Atom {
            z: 1,
            position: Vec3::zero(),
        }]);
        let set = data::load("sto-3g").unwrap();
        let lat = Lattice::cubic(5.0);
        let a = Cell::new(mol.clone(), lat.clone(), &set, 18.0).unwrap();
        let b = Cell::new(mol, lat, &set, 26.0).unwrap();
        let (sa, _) = a.overlap(Vec3::new(0.2, 0.1, 0.0));
        let (sb, _) = b.overlap(Vec3::new(0.2, 0.1, 0.0));
        assert!(
            sa.max_abs_diff(&sb) < 1e-9,
            "overlap still moving with the cutoff: {:e}",
            sa.max_abs_diff(&sb)
        );
        assert!(
            b.image_counts().0 >= a.image_counts().0,
            "a larger cutoff cannot keep fewer images"
        );
    }

    /// A neutral cell gets no charged-cell correction; a charged one does,
    /// automatically. The point of wiring it to `net_charge` rather than
    /// leaving it to the caller is that forgetting it on an ion is a
    /// silent `1/L` error, usually larger than everything else being
    /// worried about.
    #[test]
    fn charged_cell_correction_applies_itself_only_when_charged() {
        let set = data::load("sto-3g").unwrap();
        let lat = Lattice::cubic(10.0);
        let neutral = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::zero(),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 1.4),
            },
        ]);
        let cell = Cell::new(neutral.clone(), lat.clone(), &set, 20.0).unwrap();
        assert!(cell.net_charge().abs() < 1e-12, "H2 should be neutral");
        assert_eq!(cell.charged_cell_correction(), 0.0);
        assert_eq!(cell.nuclear_energy(), cell.nuclear_repulsion());

        let cation = neutral.with_charge(1).with_multiplicity(2);
        let ion = Cell::new(cation, lat, &set, 20.0).unwrap();
        assert!((ion.net_charge() - 1.0).abs() < 1e-12);
        let corr = ion.charged_cell_correction();
        assert!(
            corr > 0.0,
            "the image interaction of a positive ion is repulsive: {corr}"
        );
        assert!((ion.nuclear_energy() - ion.nuclear_repulsion() - corr).abs() < 1e-14);
    }

    /// The nuclear stress trace must be `-E/V` for a Coulomb system, since
    /// `1/r` is homogeneous of degree −1 and a uniform strain scales every
    /// distance.
    #[test]
    fn nuclear_stress_trace_matches_the_energy() {
        let cell = h2_cell(5.0);
        let s = cell.nuclear_stress();
        let tr = s[0][0] + s[1][1] + s[2][2];
        let want = -cell.nuclear_repulsion() / cell.lattice.volume().abs();
        assert!(
            (tr - want).abs() < 1e-8 * want.abs().max(1e-10),
            "stress trace {tr} vs -E/V {want}"
        );
    }

    /// The nuclear repulsion must not depend on where the cell's origin
    /// happens to sit.
    #[test]
    fn nuclear_repulsion_is_translation_invariant() {
        let cell = h2_cell(5.0);
        let e0 = cell.nuclear_repulsion();
        let shifted = Molecule::new(
            cell.molecule
                .atoms
                .iter()
                .map(|a| Atom {
                    z: a.z,
                    position: a.position + Vec3::new(1.1, -0.7, 2.3),
                })
                .collect(),
        );
        let set = data::load("sto-3g").unwrap();
        let moved = Cell::new(shifted, cell.lattice.clone(), &set, 22.0).unwrap();
        let e1 = moved.nuclear_repulsion();
        assert!(
            (e1 - e0).abs() < 1e-9,
            "nuclear repulsion moved with the origin: {e0} vs {e1}"
        );
    }

    #[test]
    fn ecp_core_derivative_contains_the_translated_ecp_term() {
        let molecule = Molecule::new(vec![
            Atom {
                z: 11,
                position: Vec3::zero(),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.3, -0.2, 2.4),
            },
        ]);
        let set = data::load("lanl2dz").unwrap();
        let cell = Cell::new(molecule, Lattice::cubic(40.0), &set, 6.0).unwrap();
        let images = cell.derivative_images();
        let kinetic = images.kinetic(Vec3::zero());
        let nuclear = images.nuclear(Vec3::zero());
        let ecp = images.ecp(Vec3::zero());
        let core = images.core_hamiltonian(Vec3::zero()).unwrap();
        assert!(ecp.iter().flat_map(|atom| atom.iter()).any(|axis| axis
            .0
            .as_slice()
            .iter()
            .any(|value| value.abs() > 1e-10)));
        for atom in 0..cell.molecule.len() {
            for axis in 0..3 {
                for i in 0..cell.nao() {
                    for j in 0..cell.nao() {
                        let expected_re = kinetic[atom][axis].0[(i, j)]
                            + nuclear[atom][axis].0[(i, j)]
                            + ecp[atom][axis].0[(i, j)];
                        let expected_im = kinetic[atom][axis].1[(i, j)]
                            + nuclear[atom][axis].1[(i, j)]
                            + ecp[atom][axis].1[(i, j)];
                        assert!((core[atom][axis].0[(i, j)] - expected_re).abs() < 1e-13);
                        assert!((core[atom][axis].1[(i, j)] - expected_im).abs() < 1e-13);
                    }
                }
            }
        }
    }
}
