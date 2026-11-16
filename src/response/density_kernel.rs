// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//! Immutable semilocal AO density response without an asserted SCF solution.
//!
//! This reuses the ordinary response solver's bounded XC derivative cache and
//! J builder. A frozen scalar model supplies its actual full alpha/beta density;
//! no orbitals, eigenvalues or convergence flag are manufactured for that state.
use super::{semilocal_cache, semilocal_density_contraction};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::linalg::Matrix;
use crate::scf::{JkEngine, ScfOptions};
use crate::system::Molecule;
use crate::xc::{Family, ResolvedFunctional, XcOut};
use crate::{DftError, Result};
use std::sync::OnceLock;

/// Derivative of the LDA/GGA Hartree-plus-XC potential at explicit densities.
/// This certifies neither stationarity nor a particular electron population.
/// The owner keeps its basis, grid and reference immutable for the cache's life.
pub struct SemilocalDensityKernel<'a> {
    basis: MolBasis,
    grid: MolecularGrid,
    functional: &'a ResolvedFunctional,
    density: [Matrix; 2],
    jk: JkEngine,
    cache: Vec<OnceLock<XcOut>>,
    _allocation: crate::resources::MemoryReservation,
}

fn validate_matrices(size: usize, density: [&Matrix; 2]) -> Result<()> {
    if density.iter().any(|p| {
        p.rows != size
            || p.cols != size
            || p.as_slice().iter().any(|x| !x.is_finite())
            || (0..size).any(|i| {
                (0..i).any(|j| {
                    (p[(i, j)] - p[(j, i)]).abs()
                        > 64. * f64::EPSILON * (1. + p[(i, j)].abs().max(p[(j, i)].abs()))
                })
            })
    }) {
        return Err(DftError::InvalidInput("semilocal density kernel requires finite symmetric alpha/beta AO matrices of the declared size".into()));
    }
    Ok(())
}

impl<'a> SemilocalDensityKernel<'a> {
    pub fn new(
        molecule: &Molecule,
        basis: &MolBasis,
        grid: &MolecularGrid,
        functional: &'a ResolvedFunctional,
        density: [&Matrix; 2],
        options: &ScfOptions,
    ) -> Result<Self> {
        validate_matrices(basis.nao_sph, density)?;
        if functional.family == Family::MetaGga
            || functional.hybrid.cx != 0.
            || functional.hybrid.clr != 0.
            || functional.required_nonlocal_kind().is_some()
            || options.vv10.is_some()
            || options.skala.is_some()
            || options.pcm.is_some()
        {
            return Err(DftError::Unsupported {
                feature: "explicit-density cached semilocal response".into(),
                missing: "a local or gradient functional without exact exchange, nonlocal, learned or solvent terms".into(),
            });
        }
        let mut total = density[0].clone();
        for (a, b) in total.as_mut_slice().iter_mut().zip(density[1].as_slice()) {
            *a += b;
        }
        let mut effective = options.clone();
        effective.eri_mode = options.effective_eri_mode(basis.nao_sph);
        let jk = JkEngine::from_options(molecule, basis, &effective)?
            .with_reference_screening(&total, density);
        let budget = (options.incore_mb.saturating_mul(1024 * 1024) / 8).min(64 * 1024 * 1024);
        let (cache, allocation) = if functional.parts.is_empty() {
            (Vec::new(), Default::default())
        } else {
            semilocal_cache(grid.len(), functional.family != Family::Lda, budget)?
        };
        Ok(Self {
            basis: basis.clone(),
            grid: grid.clone(),
            functional,
            density: [density[0].clone(), density[1].clone()],
            jk,
            cache,
            _allocation: allocation,
        })
    }

    /// Full spin-resolved AO Fock derivative for a real symmetric density change.
    pub fn apply(&self, change: &[Matrix; 2]) -> Result<[Matrix; 2]> {
        validate_matrices(self.basis.nao_sph, [&change[0], &change[1]])?;
        let mut result = crate::scf::fock::coulomb_exchange_response(
            &self.basis,
            &self.jk,
            None,
            0.,
            0.,
            change,
        )?;
        if !self.functional.parts.is_empty() {
            let xc = semilocal_density_contraction(
                &self.basis,
                self.functional,
                &self.grid,
                [&self.density[0], &self.density[1]],
                &self.cache,
                change,
            )?;
            for spin in 0..2 {
                for (a, b) in result[spin]
                    .as_mut_slice()
                    .iter_mut()
                    .zip(xc[spin].as_slice())
                {
                    *a += b;
                }
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;
    use crate::grid::GridOptions;
    use crate::integrals::overlap_matrix;
    use crate::linalg::Orthogonalizer;
    use crate::math::Vec3;
    use crate::response::nuclear::geometry::GeometryTaylor;
    use crate::scf::{EriMode, Reference};
    use crate::system::Atom;
    use crate::xc::{resolve, FunctionalSpec};
    use std::time::Instant;

    #[test]
    fn cached_kernel_matches_independent_taylor_and_preserves_bounded_cache() {
        crate::resources::ResourceLimits::new(Some(1), Some(128)).unwrap().run(|| {
            let molecule = Molecule::new([
                [-1.31,0.14,0.03],[-0.22,1.18,-0.12],[0.83,-0.27,0.34],[1.56,0.85,-0.43],
            ].into_iter().map(|r| Atom {z:1,position:Vec3::new(r[0],r[1],r[2])}).collect())
                .with_charge(1).with_multiplicity(2);
            let basis = MolBasis::build(&molecule,&data::load("sto-3g")?)?;
            let grid = MolecularGrid::build(&molecule,&GridOptions {level:1,weight_cutoff:0.,..Default::default()})?;
            let options = ScfOptions {reference:Reference::Uhf,eri_mode:EriMode::InCore {limit_mb:64},incore_mb:64,..Default::default()};
            let x=Orthogonalizer::canonical(&overlap_matrix(&basis),options.orth_threshold)?.x;
            let inverse=x.matmul_transpose_seq(&x);
            let density: [Matrix;2] = std::array::from_fn(|s| {
                let mut p=inverse.clone();for v in p.as_mut_slice(){*v *= [0.36,0.21][s];}p
            });
            let change: [Matrix;2] = std::array::from_fn(|s| {
                let mut p=Matrix::zeros(4,4);for i in 0..4 {for j in 0..4 {
                    p[(i,j)]=(0.17*(i+j+1) as f64+0.11*(i*j+1+s) as f64).sin()*0.13;
                }}p
            });
            for name in ["svwn5","pbe"] {
                let functional=resolve(&FunctionalSpec::Combined(name.into()))?;
                let kernel=SemilocalDensityKernel::new(&molecule,&basis,&grid,&functional,[&density[0],&density[1]],&options)?;
                let series=GeometryTaylor::new(&molecule,&basis,Some(&grid),&functional,None,
                    &[[0.;3];4],[0.;3],[0.;3],0,64)?;
                let p=[vec![density[0].clone()],vec![density[1].clone()]];
                let dp=[vec![change[0].clone()],vec![change[1].clone()]];
                let cold=Instant::now();let expected=kernel.apply(&change)?;
                let first_seconds=cold.elapsed().as_secs_f64();
                assert!(!kernel.cache.is_empty());
                assert!(kernel.cache.iter().all(|slot|slot.get().is_some()));
                for trial in 0..5 {
                    let mut cached=None;let mut taylor=None;
                    let mut times=[0.;2];
                    for which in if trial%2==0 {[0,1]} else {[1,0]} {
                        let start=Instant::now();
                        if which==0 {cached=Some(kernel.apply(&change)?);}
                        else {taylor=Some(series.evaluate_linear(&functional,&p,&dp,0)?);}
                        times[which]=start.elapsed().as_secs_f64();
                    }
                    let cached=cached.unwrap();let taylor=taylor.unwrap();
                    let mut error=0.0_f64;
                    for spin in 0..2 {
                        assert_eq!(cached[spin].as_slice(),expected[spin].as_slice());
                        error=error.max(cached[spin].max_abs_diff(&taylor[spin][0]));
                    }
                    assert!(error<2e-10,"{name} static/Taylor kernel error {error:e}");
                    println!("DENSITY_KERNEL_PAIR {name} trial={trial} cached_seconds={:.9} taylor_seconds={:.9} first_cached_seconds={first_seconds:.9} max_error={error:.12e}",times[0],times[1]);
                }
                // Clearing a bounded cache uses the identical provider path.
                let mut streamed=kernel;
                streamed.cache.clear();
                let value=streamed.apply(&change)?;
                for spin in 0..2 {assert_eq!(value[spin].as_slice(),expected[spin].as_slice());}
            }
            Ok(())
        }).unwrap();
    }
}
