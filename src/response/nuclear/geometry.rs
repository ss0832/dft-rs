// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic moving-geometry integral and semilocal quadrature coefficients.

use super::{add, product, NuclearEnergyCoefficient};
use crate::basis::{aos::cart_to_sph, cart_components, n_sph, MolBasis, MolShell};
use crate::dual::{Scalar, Taylor};
use crate::grid::MolecularGrid;
use crate::integrals::analytic_basis::AnalyticShell;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::response::series::{evaluate_series_many_result, SeriesScalar};
use crate::skala::{SkalaInput, SkalaModel};
use crate::system::Molecule;
use crate::xc::{Family, ResolvedFunctional, XcTaylorInput};
use crate::{DftError, Result};
use rayon::prelude::*;
use std::sync::Arc;

mod integrals;
mod nonlocal;
pub(super) use nonlocal::NonlocalTaylor;

#[derive(Clone)]
pub(crate) struct GridPointTaylor {
    pub(crate) weight: Vec<f64>,
    /// [AO][value, Cartesian gradient] coefficients.
    pub(crate) aos: Vec<[Vec<f64>; 4]>,
}

/// One direct-sum particle block of a combined Gaussian integral layout.
#[derive(Clone, Copy)]
pub(crate) struct ParticleSource {
    pub block: usize,
    pub mass: f64,
    pub charge: f64,
}
pub(crate) struct GeometrySources<'a> {
    pub charges: &'a [f64],
    pub particles: &'a [ParticleSource],
    pub force_grid: bool,
    /// Basis-only centers and their displacements, after all nuclear sites.
    pub additional_centers: &'a [(Vec3, [f64; 3])],
}
#[derive(Clone)]
pub(crate) struct GeometryTaylor {
    pub overlap: Vec<Matrix>,
    pub dipoles: [Vec<Matrix>; 3],
    pub nuclear_dipole: Vec<[f64; 3]>,
    core: Vec<Matrix>,
    /// [Taylor coefficient][mu,nu,lambda,kappa], last index fastest.
    eri: Vec<Vec<f64>>,
    eri_lr: Option<Vec<Vec<f64>>>,
    /// Exchange sums of a periodic cell, where they differ from the Coulomb
    /// sums; `None` contracts `eri`.
    exchange: Option<Vec<Vec<f64>>>,
    nuclear: Vec<f64>,
    nuclear_field: Vec<f64>,
    grid: Vec<GridPointTaylor>,
    learned: Option<LearnedTaylor>,
    /// Coupling of a periodic cell to a uniform electric field.
    field: Option<LatticeField>,
    nonlocal: Option<NonlocalTaylor>,
    nonlocal_scale: f64,
    external_classical: Option<Vec<f64>>,
}
#[derive(Clone)]
pub(crate) struct Evaluation {
    pub fock: [Vec<Matrix>; 2],
    pub energy: Vec<NuclearEnergyCoefficient>,
    // A retained scalar contraction is separate from the rounded physical
    // component ledger; its cancellation remainder is not an energy term.
    retained_total_energy: Option<Vec<(f64, f64)>>,
}

impl Evaluation {
    /// Full scalar value; the separate external ledger is already included.
    pub(super) fn coefficient_total(&self, n: usize) -> f64 {
        self.retained_total_energy
            .as_ref()
            .map_or_else(|| self.energy[n].total(), |parts| parts[n].0)
    }
    pub(super) fn total_energy(&self) -> Vec<f64> {
        self.retained_total_energy.as_ref().map_or_else(
            || {
                self.energy
                    .iter()
                    .map(NuclearEnergyCoefficient::total)
                    .collect()
            },
            |values| values.iter().map(|&(high, _)| high).collect(),
        )
    }

    pub(super) fn energy_remainders(&self) -> Vec<f64> {
        self.retained_total_energy.as_ref().map_or_else(
            || vec![0.0; self.energy.len()],
            |values| values.iter().map(|&(_, low)| low).collect(),
        )
    }
}

fn inputs(mol: &Molecule, direction: &[[f64; 3]], order: usize) -> Vec<Vec<f64>> {
    mol.atoms
        .iter()
        .zip(direction)
        .flat_map(|(atom, &d)| center_input(atom.position, d, order))
        .collect()
}
fn center_input(center: Vec3, direction: [f64; 3], order: usize) -> [Vec<f64>; 3] {
    std::array::from_fn(|axis| {
        let mut v = vec![0.0; order + 1];
        v[0] = center.get(axis);
        if order > 0 {
            v[1] = direction[axis];
        }
        v
    })
}
fn centers<S: Scalar>(v: &[S]) -> Vec<[S; 3]> {
    v.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect()
}
fn displaced_shell<S: Scalar>(
    shell: &MolShell,
    molecule: &Molecule,
    position: [S; 3],
) -> Result<AnalyticShell<S>> {
    // Basis-only paths are anchored at the supplied order-zero point.
    // Their centers do not enter nuclear Coulomb or Becke weights.
    let reference = molecule.atoms.get(shell.atom).map_or_else(
        || {
            Vec3::new(
                position[0].value(),
                position[1].value(),
                position[2].value(),
            )
        },
        |atom| atom.position,
    );
    let offset = shell.center - reference;
    AnalyticShell::from_placed(
        shell,
        std::array::from_fn(|axis| position[axis] + S::from_f64(offset.get(axis))),
    )
}
fn shells<S: Scalar>(mb: &MolBasis, mol: &Molecule, c: &[[S; 3]]) -> Result<Vec<AnalyticShell<S>>> {
    mb.shells
        .iter()
        .map(|shell| displaced_shell(shell, mol, c[shell.atom]))
        .collect()
}
fn distance<S: Scalar>(a: [S; 3], b: [S; 3]) -> S {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}
fn index4(n: usize, a: usize, b: usize, c: usize, d: usize) -> usize {
    ((a * n + b) * n + c) * n + d
}

fn guard_workspace(
    operation: &'static str,
    output_bytes: usize,
    mut workspaces: Vec<usize>,
    memory_mb: usize,
) -> Result<()> {
    workspaces.sort_unstable_by(|a, b| b.cmp(a));
    let bytes = workspaces
        .into_iter()
        .take(rayon::current_num_threads())
        .try_fold(output_bytes, |a, b| a.checked_add(b))
        .ok_or_else(|| {
            DftError::InvalidInput("parallel nuclear integral workspace overflow".into())
        })?;
    let required = bytes.div_ceil(1024 * 1024);
    if required > memory_mb {
        return Err(DftError::ResourceLimit {
            operation,
            required_mb: required,
            limit_mb: memory_mb,
        });
    }
    Ok(())
}

/// Coulomb, full exchange and long-range exchange on an ordered density
/// series. Keep both density endpoints separate until final rounding.
fn contract_pair_series(
    coulomb: Option<&[Vec<f64>]>,
    exchange: Option<&[Vec<f64>]>,
    long_range: Option<&[Vec<f64>]>,
    density: &[Vec<Matrix>; 2],
    order: usize,
) -> (Vec<Matrix>, [Vec<Matrix>; 2], [Vec<Matrix>; 2]) {
    let nao = density[0][0].rows;
    let mut j = vec![Matrix::zeros(nao, nao); order + 1];
    let mut k: [Vec<Matrix>; 2] = std::array::from_fn(|_| j.clone());
    let mut lr = k.clone();
    for n in 0..=order {
        for mu in 0..nao {
            for nu in 0..nao {
                let mut js = crate::linalg::AccurateSum::default();
                let mut ks = [crate::linalg::AccurateSum::default(); 2];
                let mut ls = ks;
                for m in 0..=n {
                    for a in 0..nao {
                        for b in 0..nao {
                            let ix = index4(nao, mu, nu, a, b);
                            let kx = index4(nao, mu, a, nu, b);
                            for s in 0..2 {
                                let p = density[s][n - m][(a, b)];
                                if let Some(j) = coulomb {
                                    js.add_product(j[m][ix], p);
                                }
                                if let Some(x) = exchange {
                                    ks[s].add_product(x[m][kx], p);
                                }
                                if let Some(x) = long_range {
                                    ls[s].add_product(x[m][kx], p);
                                }
                            }
                        }
                    }
                }
                j[n][(mu, nu)] = js.value();
                for s in 0..2 {
                    k[s][n][(mu, nu)] = ks[s].value();
                    lr[s][n][(mu, nu)] = ls[s].value();
                }
            }
        }
    }
    (j, k, lr)
}

fn add_pair_fock(f: &mut Matrix, j: &Matrix, k: &Matrix, lr: &Matrix, weights: (f64, f64)) {
    for i in 0..f.rows {
        for a in 0..f.cols {
            let mut sum = crate::linalg::AccurateSum::default();
            sum.add_product(f[(i, a)], 1.0);
            sum.add_product(j[(i, a)], 1.0);
            sum.add_product(k[(i, a)], -weights.0);
            sum.add_product(lr[(i, a)], -weights.1);
            f[(i, a)] = sum.value();
        }
    }
}

impl GeometryTaylor {
    pub(super) fn reference_core(&self) -> &Matrix {
        &self.core[0]
    }
    pub(super) fn classical_coefficient(&self, n: usize) -> f64 {
        let original = self.nuclear[n] + self.nuclear_field[n];
        match &self.external_classical {
            Some(external) => original + external[n],
            None => original,
        }
    }
    pub(super) fn add_extra_core_coefficient(&mut self, n: usize, operator: &Matrix) -> Result<()> {
        if self.external_classical.is_some()
            || n >= self.core.len()
            || operator.rows != self.core[n].rows
            || operator.cols != self.core[n].cols
        {
            return Err(DftError::InvalidInput(
                "extra core attached twice or incompatible geometry order/layout".into(),
            ));
        }
        add(&mut self.core[n], operator, 1.0);
        Ok(())
    }
    pub(super) fn set_external_classical(&mut self, values: Vec<f64>) -> Result<()> {
        if self.external_classical.is_some()
            || values.len() != self.core.len()
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(DftError::InvalidInput(
                "invalid or duplicate external classical Taylor ledger".into(),
            ));
        }
        self.external_classical = Some(values);
        Ok(())
    }
    /// Preserve integral-contraction remainders and the density's low
    /// coefficients throughout the physical Fock evaluation. The nonlinear
    /// density correction is its analytic kernel action, not a displacement.
    pub(super) fn evaluate_retained(
        &self,
        functional: &ResolvedFunctional,
        p: &[super::precision::Series; 2],
        order: usize,
    ) -> Result<(Evaluation, [super::precision::Series; 2])> {
        use super::precision::{high, low, MatrixParts, Series};
        let density = std::array::from_fn(|s| high(&p[s]));
        let remainder = std::array::from_fn(|s| low(&p[s]));
        let mut evaluation = self.evaluate(functional, &density, order)?;
        let exchange = self.exchange.as_ref().unwrap_or(&self.eri);
        let (j, k, lr) = contract_pair_series(
            Some(&self.eri),
            (functional.hybrid.cx != 0.0).then_some(exchange.as_slice()),
            self.eri_lr.as_deref(),
            &density,
            order,
        );
        let tangent = self.evaluate_linear(functional, &density, &remainder, order)?;
        let (dj, dk, dlr) = contract_pair_series(
            Some(&self.eri),
            (functional.hybrid.cx != 0.0).then_some(exchange.as_slice()),
            self.eri_lr.as_deref(),
            &remainder,
            order,
        );
        let nao = p[0][0].high.rows;
        let mut exact: [Series; 2] =
            std::array::from_fn(|_| vec![MatrixParts::zeros(nao, nao); order + 1]);
        let mut linear = exact.clone();
        for s in 0..2 {
            for n in 0..=order {
                let mut rounded_pair = self.core[n].clone();
                add_pair_fock(
                    &mut rounded_pair,
                    &j[n],
                    &k[s][n],
                    &lr[s][n],
                    (functional.hybrid.cx, functional.hybrid.clr),
                );
                let mut linear_pair = Matrix::zeros(nao, nao);
                add_pair_fock(
                    &mut linear_pair,
                    &dj[n],
                    &dk[s][n],
                    &dlr[s][n],
                    (functional.hybrid.cx, functional.hybrid.clr),
                );
                for mu in 0..nao {
                    for nu in 0..nao {
                        let mut sum = crate::linalg::AccurateSum::default();
                        sum.add_product(self.core[n][(mu, nu)], 1.0);
                        for m in 0..=n {
                            for a in 0..nao {
                                for b in 0..nao {
                                    let ix = index4(nao, mu, nu, a, b);
                                    let kx = index4(nao, mu, a, nu, b);
                                    for spin in 0..2 {
                                        for value in [
                                            p[spin][n - m].high[(a, b)],
                                            p[spin][n - m].low[(a, b)],
                                        ] {
                                            sum.add_product(self.eri[m][ix], value);
                                        }
                                    }
                                    for value in [p[s][n - m].high[(a, b)], p[s][n - m].low[(a, b)]]
                                    {
                                        if functional.hybrid.cx != 0.0 {
                                            // Retain the coefficient multiplication too.
                                            let x = exchange[m][kx] * functional.hybrid.cx;
                                            sum.add_product(-x, value);
                                            sum.add_product(
                                                -exchange[m][kx].mul_add(functional.hybrid.cx, -x),
                                                value,
                                            );
                                        }
                                        if let Some(integrals) = &self.eri_lr {
                                            let x = integrals[m][kx] * functional.hybrid.clr;
                                            sum.add_product(-x, value);
                                            sum.add_product(
                                                -integrals[m][kx]
                                                    .mul_add(functional.hybrid.clr, -x),
                                                value,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        (linear[s][n].high[(mu, nu)], linear[s][n].low[(mu, nu)]) = sum.parts();
                        sum.add_product(evaluation.fock[s][n][(mu, nu)], 1.0);
                        sum.add_product(rounded_pair[(mu, nu)], -1.0);
                        sum.add_product(tangent[s][n][(mu, nu)], 1.0);
                        sum.add_product(linear_pair[(mu, nu)], -1.0);
                        (exact[s][n].high[(mu, nu)], exact[s][n].low[(mu, nu)]) = sum.parts();
                    }
                }
            }
        }
        // The low density participates in the same energy functional. Its
        // first-order contribution is enough at the retained precision;
        // omitted quadratic low*low nonlinear terms are second order in
        // machine rounding, not a physical response approximation.
        for n in 0..=order {
            let mut one = crate::linalg::AccurateSum::default();
            let mut hartree = crate::linalg::AccurateSum::default();
            let mut exchange_energy = crate::linalg::AccurateSum::default();
            let mut long_range = crate::linalg::AccurateSum::default();
            let mut other = crate::linalg::AccurateSum::default();
            for m in 0..=n {
                for s in 0..2 {
                    for i in 0..nao {
                        for a in 0..nao {
                            let q = remainder[s][m][(i, a)];
                            one.add_product(q, self.core[n - m][(i, a)]);
                            hartree.add_product(q, j[n - m][(i, a)]);
                            exchange_energy
                                .add_product(q, -functional.hybrid.cx * k[s][n - m][(i, a)]);
                            long_range
                                .add_product(q, -functional.hybrid.clr * lr[s][n - m][(i, a)]);
                            let mut pair = self.core[n - m][(i, a)] + j[n - m][(i, a)];
                            pair -= functional.hybrid.cx * k[s][n - m][(i, a)]
                                + functional.hybrid.clr * lr[s][n - m][(i, a)];
                            other.add_product(q, evaluation.fock[s][n - m][(i, a)] - pair);
                        }
                    }
                }
            }
            evaluation.energy[n].one_electron += one.value();
            evaluation.energy[n].hartree += hartree.value();
            evaluation.energy[n].exchange += exchange_energy.value();
            evaluation.energy[n].long_range_exchange += long_range.value();
            evaluation.energy[n].semilocal_xc += other.value();
        }
        // Separate explicitly composed nonlocal/Berry potential corrections
        // from the semilocal remainder in the component ledger.
        if let Some(nonlocal) = &self.nonlocal {
            let (_, v) = nonlocal.evaluate(&self.grid, &density, None, order)?;
            for n in 0..=order {
                let correction = (0..=n)
                    .map(|m| {
                        (0..2)
                            .map(|s| remainder[s][m].frobenius_dot(&v[s][n - m]))
                            .sum::<f64>()
                    })
                    .sum::<f64>()
                    * self.nonlocal_scale;
                evaluation.energy[n].nonlocal_correlation += correction;
                evaluation.energy[n].semilocal_xc -= correction;
            }
        }
        if let Some(field) = &self.field {
            let (_, v) = field.evaluate(&density, order)?;
            for n in 0..=order {
                let correction = (0..=n)
                    .map(|m| {
                        (0..2)
                            .map(|s| remainder[s][m].frobenius_dot(&v[s][n - m]))
                            .sum::<f64>()
                    })
                    .sum::<f64>();
                evaluation.energy[n].nuclear_field += correction;
                evaluation.energy[n].semilocal_xc -= correction;
            }
        }
        // E_linear = 1/2 sum_s Tr[P_s (H + F_linear,s)]. Keep
        // the high/low Cauchy contractions across both spins in one sum.
        // Adding already rounded one-electron and exchange energies loses
        // their common cancellation in large mixed fourth coefficients.
        let core = super::precision::retain(&self.core[..=order]);
        let mut totals = Vec::with_capacity(order + 1);
        for n in 0..=order {
            let mut sum = crate::linalg::AccurateSum::default();
            for s in 0..2 {
                for term in [&core, &linear[s]] {
                    let (high, low) = super::precision::trace_product_parts(&p[s], term, n);
                    sum.add_product(high, 0.5);
                    sum.add_product(low, 0.5);
                }
            }
            let e = &evaluation.energy[n];
            for value in [
                e.nuclear_repulsion,
                e.nuclear_field,
                e.semilocal_xc,
                e.nonlocal_correlation,
            ] {
                sum.add_product(value, 1.0);
            }
            if let Some(external) = &self.external_classical {
                sum.add_product(external[n], 1.0);
            }
            totals.push(sum.parts());
        }
        evaluation.retained_total_energy = Some(totals);
        evaluation.fock = std::array::from_fn(|s| high(&exact[s]));
        Ok((evaluation, exact))
    }

    /// Approximate memory of the stored integral, grid and Berry series in
    /// bytes, including nonlocal geometry (learned graph storage is separate).
    pub(crate) fn approximate_bytes(&self) -> usize {
        let matrices = |list: &[Matrix]| list.iter().map(|m| m.rows * m.cols).sum::<usize>();
        let vectors = |list: &[Vec<f64>]| list.iter().map(Vec::len).sum::<usize>();
        let mut values = matrices(&self.overlap)
            + matrices(&self.core)
            + self.dipoles.iter().map(|d| matrices(d)).sum::<usize>()
            + vectors(&self.eri)
            + self.eri_lr.as_deref().map_or(0, vectors)
            + self.exchange.as_deref().map_or(0, vectors);
        for point in &self.grid {
            values += point.weight.len() + point.aos.iter().flatten().map(Vec::len).sum::<usize>();
        }
        if let Some(field) = &self.field {
            values += field
                .polar
                .iter()
                .flatten()
                .map(|m| matrices(m))
                .sum::<usize>();
        }
        8 * values
            + self
                .nonlocal
                .as_ref()
                .map_or(0, NonlocalTaylor::approximate_bytes)
    }

    pub(super) fn with_periodic_nonlocal(mut self, term: NonlocalTaylor) -> Self {
        self.nonlocal = Some(term);
        self
    }

    /// Dedicated molecular quadrature, with both density endpoints, weights
    /// and atom-owned point motion differentiated on the SCF vacuum branch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attach_molecular_nonlocal(
        &mut self,
        molecule: &Molecule,
        basis: &MolBasis,
        grid: &MolecularGrid,
        parameters: crate::nlc::Vv10Params,
        coefficient: f64,
        direction: &[[f64; 3]],
        order: usize,
        memory_mb: usize,
    ) -> Result<()> {
        if !coefficient.is_finite()
            || !parameters.b.is_finite()
            || parameters.b <= 0.0
            || !parameters.c.is_finite()
            || parameters.c < 0.0
        {
            return Err(DftError::InvalidInput(
                "invalid molecular nonlocal nuclear energy model".into(),
            ));
        }
        let extra = grid
            .len()
            .checked_mul(4 * basis.nao_sph + 4)
            .and_then(|n| n.checked_mul(order + 1))
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| {
                DftError::InvalidInput("moving nonlocal grid storage overflow".into())
            })?;
        let base = self.approximate_bytes().saturating_sub(
            self.nonlocal
                .as_ref()
                .map_or(0, NonlocalTaylor::approximate_bytes),
        );
        let required_mb = base.saturating_add(extra).div_ceil(1024 * 1024);
        if required_mb > memory_mb {
            return Err(DftError::ResourceLimit {
                operation: "moving molecular nonlocal geometry",
                required_mb,
                limit_mb: memory_mb,
            });
        }
        self.nonlocal = Some(NonlocalTaylor::molecular(
            molecule, basis, grid, parameters, direction, order,
        )?);
        self.nonlocal_scale = coefficient;
        Ok(())
    }

    /// Scale the complete energy, potential and induced-potential provider
    /// together, as required by a distinct post-SCF target functional.
    pub(super) fn scale_periodic_nonlocal(&mut self, scale: f64) -> Result<()> {
        if !scale.is_finite() || self.nonlocal.is_none() {
            return Err(DftError::InvalidInput(
                "a finite nonlocal coefficient requires an attached periodic nonlocal provider"
                    .into(),
            ));
        }
        self.nonlocal_scale = scale;
        Ok(())
    }
    /// Full periodic Coulomb coefficients, with the same Ewald convention
    /// as the parent Hartree functional. Never substitute isolated ERIs.
    pub(crate) fn coulomb_coefficients(&self) -> &[Vec<f64>] {
        &self.eri
    }

    pub(crate) fn core_coefficients(&self) -> &[Matrix] {
        &self.core
    }

    /// Nuclear repulsion energy coefficients along the path.
    pub(crate) fn nuclear_coefficients(&self) -> &[f64] {
        &self.nuclear
    }
    pub(crate) fn nuclear_field_coefficients(&self) -> &[f64] {
        &self.nuclear_field
    }
    pub(crate) fn grid_points(&self) -> &[GridPointTaylor] {
        &self.grid
    }

    /// Full and range-separated exchange on a time-odd density. The
    /// Hartree contraction vanishes for a real antisymmetric matrix.
    pub(super) fn imaginary_exchange(
        &self,
        functional: &ResolvedFunctional,
        density: &[Matrix; 2],
    ) -> [Matrix; 2] {
        let cx = functional.hybrid.cx;
        let clr = functional.hybrid.clr;
        let n = self.overlap[0].rows;
        let mut out = std::array::from_fn(|_| Matrix::zeros(n, n));
        if cx == 0.0 && clr == 0.0 {
            return out;
        }
        let full = self.exchange.as_ref().unwrap_or(&self.eri);
        let (j, k, lr) = contract_pair_series(
            None,
            (cx != 0.0).then_some(full.as_slice()),
            self.eri_lr.as_deref(),
            &std::array::from_fn(|spin| vec![density[spin].clone()]),
            0,
        );
        for spin in 0..2 {
            add_pair_fock(&mut out[spin], &j[0], &k[spin][0], &lr[spin][0], (cx, clr));
        }
        out
    }

    pub fn new(
        mol: &Molecule,
        mb: &MolBasis,
        grid: Option<&MolecularGrid>,
        functional: &ResolvedFunctional,
        learned: Option<&Arc<SkalaModel>>,
        direction: &[[f64; 3]],
        reference_field: [f64; 3],
        field_direction: [f64; 3],
        order: usize,
        memory_mb: usize,
    ) -> Result<Self> {
        Self::with_sources(
            mol,
            mb,
            grid,
            functional,
            learned,
            direction,
            reference_field,
            field_direction,
            order,
            memory_mb,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_sources(
        mol: &Molecule,
        mb: &MolBasis,
        grid: Option<&MolecularGrid>,
        functional: &ResolvedFunctional,
        learned: Option<&Arc<SkalaModel>>,
        direction: &[[f64; 3]],
        reference_field: [f64; 3],
        field_direction: [f64; 3],
        order: usize,
        memory_mb: usize,
        sources: Option<&GeometrySources<'_>>,
    ) -> Result<Self> {
        let default_charges = mb.effective_charges(mol);
        if sources.is_some() && !mb.ecps.is_empty() {
            return Err(DftError::Unsupported {
                feature: "multicomponent moving ECP sources".into(),
                missing: "particle-charge-specific nonlocal projector couplings".into(),
            });
        }
        let charges = sources.map_or(default_charges.as_slice(), |s| s.charges);
        let additional_centers = sources.map_or(&[][..], |s| s.additional_centers);
        let sites = mol.len() + additional_centers.len();
        if charges.len() != mol.len()
            || charges.iter().any(|z| !z.is_finite())
            || additional_centers.iter().any(|(center, direction)| {
                center
                    .to_array()
                    .iter()
                    .chain(direction)
                    .any(|x| !x.is_finite())
            })
            || sources.is_some_and(|s| {
                s.particles.len() != mb.nao_sph
                    || s.particles
                        .iter()
                        .any(|p| !p.mass.is_finite() || p.mass <= 0.0 || !p.charge.is_finite())
            })
        {
            return Err(DftError::InvalidInput("moving particle sources need matching finite charges and positive masses for every AO".into()));
        }
        let width = order
            .checked_add(1)
            .ok_or_else(|| DftError::InvalidInput("nuclear Taylor order overflow".into()))?;
        let nao = mb.nao_sph;
        let size = nao.checked_pow(4).ok_or_else(|| {
            DftError::InvalidInput("nuclear Taylor integral size overflow".into())
        })?;
        let tensors = 1 + usize::from(functional.hybrid.clr != 0.0);
        let bytes = size
            .checked_mul(width)
            .and_then(|n| n.checked_mul(8 * tensors))
            .ok_or_else(|| DftError::InvalidInput("nuclear Taylor memory size overflow".into()))?;
        if bytes > memory_mb.saturating_mul(1024 * 1024) {
            return Err(DftError::Unsupported {
                feature: "nuclear Taylor integral cache exceeding the memory budget".into(),
                missing: format!(
                    "a streamed moving-integral contraction; the current cache needs {bytes} bytes"
                ),
            });
        }
        if mb
            .shells
            .iter()
            .any(|s| s.atom >= sites || s.sph_off.saturating_add(n_sph(s.l)) > nao)
        {
            return Err(DftError::InvalidInput(
                "invalid placed nuclear-response basis".into(),
            ));
        }
        for a in 0..mol.len() {
            for b in 0..a {
                if (mol.atoms[a].position - mol.atoms[b].position).norm() < 1e-12 {
                    return Err(DftError::InvalidInput(
                        "nuclear response requires distinct atomic centers".into(),
                    ));
                }
            }
        }
        let mut input = inputs(mol, direction, order);
        input.extend(
            additional_centers
                .iter()
                .flat_map(|&(center, velocity)| center_input(center, velocity, order)),
        );
        let field_offset = 3 * sites;
        for axis in 0..3 {
            let mut field = vec![0.0; width];
            field[0] = reference_field[axis];
            if order > 0 {
                field[1] = field_direction[axis];
            }
            input.push(field);
        }
        let nuclear = evaluate_series_many_result(&input, |v| {
            let c = centers(&v[..3 * mol.len()]);
            let mut e = SeriesScalar::from_f64(0.0);
            for a in 0..mol.len() {
                for b in 0..a {
                    e = e + SeriesScalar::from_f64(charges[a] * charges[b]) / distance(c[a], c[b]);
                }
            }
            let mut dipole = [SeriesScalar::from_f64(0.0); 3];
            for a in 0..mol.len() {
                for axis in 0..3 {
                    dipole[axis] = dipole[axis] + SeriesScalar::from_f64(charges[a]) * c[a][axis];
                }
            }
            let mut field_energy = SeriesScalar::from_f64(0.0);
            for axis in 0..3 {
                field_energy = field_energy - dipole[axis] * v[field_offset + axis];
            }
            Ok(vec![e, field_energy, dipole[0], dipole[1], dipole[2]])
        })?;
        let nuclear_field = nuclear[1].clone();
        let nuclear_dipole = (0..width)
            .map(|n| std::array::from_fn(|a| nuclear[2 + a][n]))
            .collect();
        let nuclear = nuclear[0].clone();
        let pairs: Vec<_> = (0..mb.shells.len())
            .flat_map(|a| (0..=a).map(move |b| (a, b)))
            .collect();
        if order <= 31 {
            let workspaces = pairs
                .iter()
                .map(|&(a, b)| {
                    integrals::pair_workspace_bytes(
                        [a, b].map(|i| mb.shells[i].l),
                        [a, b].map(|i| mb.shells[i].exps.len()),
                        mol.len(),
                        order,
                    )
                    .map(|n| n.expect("fixed pair order has a bounded scalar width"))
                })
                .collect::<Result<Vec<_>>>()?;
            let output = nao
                .checked_mul(nao)
                .and_then(|n| n.checked_mul(10 * width * 8 + 6 * std::mem::size_of::<Vec<f64>>()))
                .ok_or_else(|| {
                    DftError::InvalidInput("nuclear pair output size overflow".into())
                })?;
            guard_workspace(
                "nuclear one-electron Taylor workspace and output",
                output,
                workspaces,
                memory_mb,
            )?;
        }
        let blocks: Result<Vec<_>> = pairs
            .par_iter()
            .map(|&(a, b)| {
                integrals::one_electron(mb, mol, &input, [a, b], charges, sources, order)
            })
            .collect();
        let mut overlap = vec![Matrix::zeros(nao, nao); width];
        let mut core = overlap.clone();
        let mut dipoles: [Vec<Matrix>; 3] = std::array::from_fn(|_| overlap.clone());
        for (&(a, b), block) in pairs.iter().zip(blocks?) {
            let sa = &mb.shells[a];
            let sb = &mb.shells[b];
            let count = n_sph(sa.l) * n_sph(sb.l);
            for i in 0..n_sph(sa.l) {
                for j in 0..n_sph(sb.l) {
                    let ix = i * n_sph(sb.l) + j;
                    for n in 0..width {
                        for (matrix, coefficient) in [
                            (&mut overlap[n], block[ix][n]),
                            (&mut core[n], block[count + ix][n]),
                        ] {
                            matrix[(sa.sph_off + i, sb.sph_off + j)] = coefficient;
                            matrix[(sb.sph_off + j, sa.sph_off + i)] = coefficient;
                        }
                        for axis in 0..3 {
                            let coefficient = block[(2 + axis) * count + ix][n];
                            dipoles[axis][n][(sa.sph_off + i, sb.sph_off + j)] = coefficient;
                            dipoles[axis][n][(sb.sph_off + j, sa.sph_off + i)] = coefficient;
                        }
                    }
                }
            }
        }
        if !mb.ecps.is_empty() {
            let ecp = crate::integrals::ecp::ecp_nuclear_taylor(mb, direction, order)?;
            for (h, u) in core.iter_mut().zip(&ecp) {
                add(h, u, 1.0);
            }
        }
        let quartets: Vec<_> = (0..pairs.len())
            .flat_map(|p| (0..=p).map(move |q| (p, q)))
            .collect();
        if order <= 31 {
            let workspaces = quartets
                .iter()
                .map(|&(p, q)| {
                    let (a, b) = pairs[p];
                    let (c, d) = pairs[q];
                    integrals::workspace_bytes(
                        [a, b, c, d].map(|i| mb.shells[i].l),
                        [a, b, c, d].map(|i| mb.shells[i].exps.len()),
                        order,
                    )
                    .map(|n| n.expect("fixed integral order has a bounded scalar width"))
                })
                .collect::<Result<Vec<_>>>()?;
            let output = bytes.checked_mul(3).ok_or_else(|| {
                DftError::InvalidInput("nuclear integral memory size overflow".into())
            })?;
            guard_workspace(
                "nuclear Taylor integral workspace and output tensors",
                output,
                workspaces,
                memory_mb,
            )?;
        }
        let build_eri = |omega: Option<f64>| -> Result<Vec<Vec<f64>>> {
            let blocks: Result<Vec<_>> = quartets
                .par_iter()
                .map(|&(p, q)| {
                    let (a, b) = pairs[p];
                    let (c, d) = pairs[q];
                    integrals::repulsion(mb, mol, &input, [a, b, c, d], omega, order)
                })
                .collect();
            let mut tensor = vec![vec![0.0; size]; width];
            for (&(p, q), block) in quartets.iter().zip(blocks?) {
                let (a, b) = pairs[p];
                let (c, d) = pairs[q];
                let sh = [&mb.shells[a], &mb.shells[b], &mb.shells[c], &mb.shells[d]];
                let shape = sh.map(|s| n_sph(s.l));
                for (ix, value) in block.iter().enumerate() {
                    let mut rem = ix;
                    let mut ao = [0; 4];
                    for axis in (0..4).rev() {
                        ao[axis] = sh[axis].sph_off + rem % shape[axis];
                        rem /= shape[axis];
                    }
                    let [a, b, c, d] = ao;
                    for [i, j, k, l] in [
                        [a, b, c, d],
                        [b, a, c, d],
                        [a, b, d, c],
                        [b, a, d, c],
                        [c, d, a, b],
                        [d, c, a, b],
                        [c, d, b, a],
                        [d, c, b, a],
                    ] {
                        for n in 0..width {
                            tensor[n][index4(nao, i, j, k, l)] = value[n];
                        }
                    }
                }
            }
            Ok(tensor)
        };
        let eri = build_eri(None)?;
        let eri_lr = if functional.hybrid.clr != 0.0 {
            Some(build_eri(Some(functional.hybrid.omega))?)
        } else {
            None
        };
        let point_jets = if !functional.parts.is_empty()
            || learned.is_some()
            || sources.is_some_and(|s| s.force_grid)
        {
            moving_grid_taylor(
                mol,
                mb,
                grid.ok_or_else(|| DftError::InvalidInput("missing nuclear response grid".into()))?,
                &input,
                order,
            )?
        } else {
            Vec::new()
        };
        let learned = match (learned, grid) {
            (Some(model), Some(grid)) => Some(LearnedTaylor::new(model, mol, grid)),
            (Some(_), None) => {
                return Err(DftError::InvalidInput(
                    "missing nuclear response grid".into(),
                ))
            }
            (None, _) => None,
        };
        Ok(Self {
            overlap,
            dipoles,
            nuclear_dipole,
            core,
            eri,
            eri_lr,
            exchange: None,
            nuclear,
            nuclear_field,
            grid: point_jets,
            learned,
            field: None,
            nonlocal: None,
            nonlocal_scale: 1.0,
            external_classical: None,
        })
    }

    /// A Gamma-point cell: its lattice series take the place of the
    /// molecular integrals, and the Bloch sums of the atomic orbitals the
    /// place of the orbitals on the grid. `reduced_field` is the series of
    /// `E(t) . a_a / (2 pi)` of a uniform electric field.
    pub fn from_lattice(
        series: crate::periodic::path_series::LatticeSeries,
        reduced_field: Option<[Vec<f64>; 3]>,
        electron_charge_ratio: f64,
    ) -> Self {
        let width = series.overlap.len();
        let nao = series.overlap[0].rows;
        let zero = vec![Matrix::zeros(nao, nao); width];
        Self {
            overlap: series.overlap,
            dipoles: std::array::from_fn(|_| zero.clone()),
            nuclear_dipole: vec![[0.0; 3]; width],
            core: series.core,
            eri: series.eri,
            eri_lr: series.exchange_lr,
            exchange: series.exchange,
            nuclear: series.nuclear,
            nuclear_field: vec![0.0; width],
            grid: series
                .weights
                .into_iter()
                .zip(series.aos)
                .map(|(weight, aos)| GridPointTaylor { weight, aos })
                .collect(),
            learned: None,
            nonlocal: None,
            nonlocal_scale: 1.0,
            external_classical: None,
            field: reduced_field.map(|reduced| LatticeField {
                phase_reference: Some(std::array::from_fn(|a| {
                    series.ionic[a][0] * electron_charge_ratio
                })),
                polar: series.polar,
                ionic: series.ionic,
                reduced,
            }),
        }
    }

    /// Extend constant spatial integrals with exact zero coefficients for a
    /// field-only path. No nuclear or strain direction enters this operation.
    pub fn extend_fixed_lattice_field(mut self, order: usize) -> Result<Self> {
        if self.overlap.len() != 1 || self.learned.is_some() || self.field.is_none() {
            return Err(DftError::InvalidInput(
                "a fixed lattice field series needs zeroth-order spatial integrals and Berry matrices".into(),
            ));
        }
        let width = order.checked_add(1).ok_or_else(|| {
            DftError::InvalidInput("fixed lattice field series order overflow".into())
        })?;
        let zero = Matrix::zeros(self.overlap[0].rows, self.overlap[0].cols);
        for values in self
            .dipoles
            .iter_mut()
            .chain([&mut self.overlap, &mut self.core])
        {
            values.resize(width, zero.clone());
        }
        for values in self
            .eri_lr
            .iter_mut()
            .chain(self.exchange.iter_mut())
            .chain(std::iter::once(&mut self.eri))
        {
            values.resize(width, vec![0.0; values[0].len()]);
        }
        self.nuclear.resize(width, 0.0);
        self.nuclear_field.resize(width, 0.0);
        self.nuclear_dipole.resize(width, [0.0; 3]);
        if let Some(nonlocal) = &mut self.nonlocal {
            nonlocal.extend_fixed(width);
        }
        for point in &mut self.grid {
            point.weight.resize(width, 0.0);
            for ao in &mut point.aos {
                for component in ao {
                    component.resize(width, 0.0);
                }
            }
        }
        let field = self.field.as_mut().unwrap();
        for axis in &mut field.polar {
            for component in axis {
                component.resize(width, zero.clone());
            }
        }
        for axis in field.ionic.iter_mut().chain(field.reduced.iter_mut()) {
            axis.resize(width, 0.0);
        }
        Ok(self)
    }

    /// Reuse fixed-geometry lattice integrals when only the reduced electric
    /// field changes. The Berry matrices and ionic phases are unchanged.
    pub fn set_reduced_lattice_field(&mut self, reduced: [Vec<f64>; 3]) -> Result<()> {
        if reduced.iter().any(|v| v.len() != self.overlap.len()) {
            return Err(DftError::InvalidInput(
                "the reduced field and lattice integral series must have equal orders".into(),
            ));
        }
        let field = self.field.as_mut().ok_or_else(|| {
            DftError::InvalidInput(
                "the lattice integral series does not contain a Berry field coupling".into(),
            )
        })?;
        field.reduced = reduced;
        Ok(())
    }

    /// `sum_s Im ln det(1 + D_s dQ_a) - sum_A Z_A b_a . R_A` of the spin
    /// densities `p` at the start of the path, for the three lattice
    /// directions: minus `2 pi` times the dipole of the cell in units of its
    /// lattice vectors. `None` without a field coupling.
    pub fn berry_phases(&self, p: &[Matrix; 2]) -> Result<Option<[f64; 3]>> {
        match &self.field {
            Some(field) => Ok(Some(field.phases(p)?)),
            None => Ok(None),
        }
    }

    pub fn evaluate(
        &self,
        functional: &ResolvedFunctional,
        p: &[Vec<Matrix>; 2],
        order: usize,
    ) -> Result<Evaluation> {
        let exchange = self.exchange.as_ref().unwrap_or(&self.eri);
        let mut fock: [Vec<Matrix>; 2] = std::array::from_fn(|_| self.core[..=order].to_vec());
        let (j, k, klr) = contract_pair_series(
            Some(&self.eri),
            (functional.hybrid.cx != 0.0).then_some(exchange.as_slice()),
            self.eri_lr.as_deref(),
            p,
            order,
        );
        let mut energy = vec![NuclearEnergyCoefficient::default(); order + 1];
        for n in 0..=order {
            energy[n].nuclear_repulsion = self.nuclear[n];
            energy[n].nuclear_field = self.nuclear_field[n];
            for s in 0..2 {
                add_pair_fock(
                    &mut fock[s][n],
                    &j[n],
                    &k[s][n],
                    &klr[s][n],
                    (functional.hybrid.cx, functional.hybrid.clr),
                );
                for m in 0..=n {
                    energy[n].one_electron += p[s][m].frobenius_dot(&self.core[n - m]);
                    energy[n].hartree += 0.5 * p[s][m].frobenius_dot(&j[n - m]);
                    energy[n].exchange -=
                        0.5 * functional.hybrid.cx * p[s][m].frobenius_dot(&k[s][n - m]);
                    energy[n].long_range_exchange -=
                        0.5 * functional.hybrid.clr * p[s][m].frobenius_dot(&klr[s][n - m]);
                }
            }
        }
        let semilocal: &[GridPointTaylor] = if functional.parts.is_empty() {
            &[]
        } else {
            &self.grid
        };
        let xc: Result<Vec<_>> = semilocal
            .par_iter()
            .map(|point| point.evaluate(functional, p, order))
            .collect();
        for (e, v) in xc? {
            for n in 0..=order {
                energy[n].semilocal_xc += e[n];
                for s in 0..2 {
                    add(&mut fock[s][n], &v[s][n], 1.0);
                }
            }
        }
        if let Some(learned) = &self.learned {
            let (e, v) =
                match order {
                    0..=3 => learned.evaluate::<4>(&self.grid, p, order)?,
                    4..=7 => learned.evaluate::<8>(&self.grid, p, order)?,
                    _ => return Err(DftError::Unsupported {
                        feature:
                            "nuclear Taylor response of the learned functional above order seven"
                                .into(),
                        missing: "series arithmetic of the network beyond eight coefficients"
                            .into(),
                    }),
                };
            for n in 0..=order {
                energy[n].semilocal_xc += e[n];
                for s in 0..2 {
                    add(&mut fock[s][n], &v[s][n], 1.0);
                }
            }
        }
        if let Some(field) = &self.field {
            let (e, v) = field.evaluate(p, order)?;
            for n in 0..=order {
                energy[n].nuclear_field += e[n];
                for s in 0..2 {
                    add(&mut fock[s][n], &v[s][n], 1.0);
                }
            }
        }
        if let Some(nonlocal) = &self.nonlocal {
            let (e, v) = nonlocal.evaluate(&self.grid, p, None, order)?;
            for n in 0..=order {
                energy[n].nonlocal_correlation += self.nonlocal_scale * e[n];
                for s in 0..2 {
                    add(&mut fock[s][n], &v[s][n], self.nonlocal_scale);
                }
            }
        }
        // Only the explicit extra-core route adds this density-independent
        // ledger. The ordinary None route retains its original arithmetic.
        let retained_total_energy = self.external_classical.as_ref().map(|external| {
            energy
                .iter()
                .enumerate()
                .map(|(n, e)| {
                    let mut sum = crate::linalg::AccurateSum::default();
                    for value in [
                        e.nuclear_repulsion,
                        e.nuclear_field,
                        e.one_electron,
                        e.hartree,
                        e.exchange,
                        e.long_range_exchange,
                        e.semilocal_xc,
                        e.nonlocal_correlation,
                        external[n],
                    ] {
                        sum.add_product(value, 1.0);
                    }
                    sum.parts()
                })
                .collect()
        });
        Ok(Evaluation {
            fock,
            energy,
            retained_total_energy,
        })
    }

    /// Coefficients of the Kohn-Sham matrix change caused by the density
    /// change `q` along the density path `p`. `q` need not be symmetric: the
    /// exchange contraction keeps its index order, while the Coulomb and
    /// semilocal terms see its symmetric part.
    pub fn evaluate_linear(
        &self,
        functional: &ResolvedFunctional,
        p: &[Vec<Matrix>; 2],
        q: &[Vec<Matrix>; 2],
        order: usize,
    ) -> Result<[Vec<Matrix>; 2]> {
        if self.learned.is_some() {
            return Err(DftError::Unsupported {
                feature: "path response of the learned functional".into(),
                missing: "the first-order change of the network potential along a nuclear path"
                    .into(),
            });
        }
        let nao = self.overlap[0].rows;
        let exchange = self.exchange.as_ref().unwrap_or(&self.eri);
        let (j, k, klr) = contract_pair_series(
            Some(&self.eri),
            (functional.hybrid.cx != 0.0).then_some(exchange.as_slice()),
            self.eri_lr.as_deref(),
            q,
            order,
        );
        let mut fock: [Vec<Matrix>; 2] =
            std::array::from_fn(|_| vec![Matrix::zeros(nao, nao); order + 1]);
        for s in 0..2 {
            for n in 0..=order {
                add_pair_fock(
                    &mut fock[s][n],
                    &j[n],
                    &k[s][n],
                    &klr[s][n],
                    (functional.hybrid.cx, functional.hybrid.clr),
                );
            }
        }
        if !functional.parts.is_empty() {
            let symmetric: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
                q[s][..=order]
                    .iter()
                    .map(|m| {
                        let mut out = m.transpose();
                        add(&mut out, m, 1.0);
                        for v in out.as_mut_slice() {
                            *v *= 0.5;
                        }
                        out
                    })
                    .collect()
            });
            let zero = || -> [Vec<Matrix>; 2] {
                std::array::from_fn(|_| vec![Matrix::zeros(nao, nao); order + 1])
            };
            let total = self
                .grid
                .par_iter()
                .try_fold(zero, |mut acc, point| -> Result<[Vec<Matrix>; 2]> {
                    let v = point.linear(functional, p, &symmetric, order)?;
                    for s in 0..2 {
                        for n in 0..=order {
                            add(&mut acc[s][n], &v[s][n], 1.0);
                        }
                    }
                    Ok(acc)
                })
                .try_reduce(zero, |mut a, b| {
                    for s in 0..2 {
                        for n in 0..=order {
                            add(&mut a[s][n], &b[s][n], 1.0);
                        }
                    }
                    Ok(a)
                })?;
            for s in 0..2 {
                for n in 0..=order {
                    add(&mut fock[s][n], &total[s][n], 1.0);
                }
            }
        }
        if let Some(field) = &self.field {
            let change = field.linear(p, q, order)?;
            for s in 0..2 {
                for n in 0..=order {
                    add(&mut fock[s][n], &change[s][n], 1.0);
                }
            }
        }
        if let Some(nonlocal) = &self.nonlocal {
            let (_, v) = nonlocal.evaluate(&self.grid, p, Some(q), order)?;
            for s in 0..2 {
                for n in 0..=order {
                    add(&mut fock[s][n], &v[s][n], self.nonlocal_scale);
                }
            }
        }
        Ok(fock)
    }
}

/// Taylor coefficients of a one-electron operator between the moving shells
/// along `direction`. `antisymmetric` states how the block of an exchanged
/// shell pair follows from the computed one.
pub(super) fn operator_series(
    mol: &Molecule,
    mb: &MolBasis,
    direction: &[[f64; 3]],
    order: usize,
    antisymmetric: bool,
    operator: impl Fn(&AnalyticShell<SeriesScalar>, &AnalyticShell<SeriesScalar>) -> Result<Vec<SeriesScalar>>
        + Sync,
) -> Result<Vec<Matrix>> {
    let width = order + 1;
    let nao = mb.nao_sph;
    let input = inputs(mol, direction, order);
    let pairs: Vec<_> = (0..mb.shells.len())
        .flat_map(|a| (0..=a).map(move |b| (a, b)))
        .collect();
    let blocks: Result<Vec<_>> = pairs
        .par_iter()
        .map(|&(a, b)| {
            evaluate_series_many_result(&input, |v| {
                let sh = shells(mb, mol, &centers(v))?;
                operator(&sh[a], &sh[b])
            })
        })
        .collect();
    let sign = if antisymmetric { -1.0 } else { 1.0 };
    let mut out = vec![Matrix::zeros(nao, nao); width];
    for (&(a, b), block) in pairs.iter().zip(blocks?) {
        let sa = &mb.shells[a];
        let sb = &mb.shells[b];
        for i in 0..n_sph(sa.l) {
            for j in 0..n_sph(sb.l) {
                let ix = i * n_sph(sb.l) + j;
                for n in 0..width {
                    out[n][(sb.sph_off + j, sa.sph_off + i)] = sign * block[ix][n];
                    out[n][(sa.sph_off + i, sb.sph_off + j)] = block[ix][n];
                }
            }
        }
    }
    Ok(out)
}

pub(crate) fn becke_weight<S: Scalar>(
    c: &[[S; 3]],
    aij: &[f64],
    owner: usize,
    point: [S; 3],
    order: usize,
) -> Result<S> {
    if c.len() == 1 {
        return Ok(S::from_f64(1.0));
    }
    for (atom, &center) in c.iter().enumerate() {
        let norm2 = (point[0] - center[0]).powi(2)
            + (point[1] - center[1]).powi(2)
            + (point[2] - center[2]).powi(2);
        if norm2.value() == 0.0 {
            // The thrice-composed cubic switch is flat through order seven
            // at either endpoint. Its partition is therefore locally the
            // Kronecker value to this order despite the distance cusp.
            if order <= 7 {
                return Ok(S::from_f64(f64::from(atom == owner)));
            }
            return Err(DftError::Unsupported {
                feature: "Becke contact derivatives above order seven".into(),
                missing: "a differentiable contact-limit partition beyond the switch smoothness"
                    .into(),
            });
        }
    }
    let d: Vec<_> = c.iter().map(|&center| distance(point, center)).collect();
    let mut numerator = S::from_f64(0.0);
    let mut denominator = S::from_f64(0.0);
    for b in 0..c.len() {
        let mut cell = S::from_f64(1.0);
        for a in 0..c.len() {
            if a != b {
                let mu = (d[b] - d[a]) / distance(c[b], c[a]);
                let mut f = mu + S::from_f64(aij[b * c.len() + a]) * (S::from_f64(1.0) - mu * mu);
                for _ in 0..3 {
                    f = S::from_f64(1.5) * f - S::from_f64(0.5) * f * f * f;
                }
                cell = cell * S::from_f64(0.5) * (S::from_f64(1.0) - f);
            }
        }
        if b == owner {
            numerator = cell;
        }
        denominator = denominator + cell;
    }
    if denominator.value() <= 0.0 {
        return Err(DftError::InvalidInput(
            "nonpositive moving Becke partition denominator".into(),
        ));
    }
    Ok(numerator / denominator)
}

impl GridPointTaylor {
    /// Taylor coefficients of the spin densities (entries 0 and 1) and of
    /// the kinetic energy densities (entries 5 and 6), and of the density
    /// gradients, along the density path `p`.
    fn densities(
        &self,
        p: &[Vec<Matrix>; 2],
        order: usize,
        gga: bool,
        mgga: bool,
    ) -> ([Vec<f64>; 7], [[Vec<f64>; 3]; 2]) {
        let (mut variables, grad) = self.fields(p, order, gga, mgga);
        for s in 0..2 {
            variables[s][0] = variables[s][0].max(0.0);
            variables[5 + s][0] = variables[5 + s][0].max(0.0);
        }
        (variables, grad)
    }

    /// The same contractions without the positivity clamp of the leading
    /// coefficients, for a density change of either sign.
    fn fields(
        &self,
        p: &[Vec<Matrix>; 2],
        order: usize,
        gga: bool,
        mgga: bool,
    ) -> ([Vec<f64>; 7], [[Vec<f64>; 3]; 2]) {
        let [fields] = self.fields_many([p], order, gga, mgga);
        fields
    }

    /// Share moving AO products between a density and its independent
    /// tangent. Each contraction keeps its original accumulation order.
    fn fields_many<const N: usize>(
        &self,
        paths: [&[Vec<Matrix>; 2]; N],
        order: usize,
        gga: bool,
        mgga: bool,
    ) -> [([Vec<f64>; 7], [[Vec<f64>; 3]; 2]); N] {
        let width = order + 1;
        let nao = self.aos.len();
        let mut fields = std::array::from_fn(|_| {
            (
                std::array::from_fn(|_| vec![0.0; width]),
                std::array::from_fn(|_| std::array::from_fn(|_| vec![0.0; width])),
            )
        });
        for mu in 0..nao {
            for nu in 0..nao {
                let pair = product(&self.aos[mu][0], &self.aos[nu][0], order);
                let gp: [Vec<f64>; 3] = std::array::from_fn(|a| {
                    if gga {
                        product(&self.aos[mu][a + 1], &self.aos[nu][0], order)
                    } else {
                        Vec::new()
                    }
                });
                let gg: [Vec<f64>; 3] = std::array::from_fn(|a| {
                    if mgga {
                        product(&self.aos[mu][a + 1], &self.aos[nu][a + 1], order)
                    } else {
                        Vec::new()
                    }
                });
                for (path, (variables, grad)) in paths.iter().zip(&mut fields) {
                    for spin in 0..2 {
                        for n in 0..width {
                            for m in 0..=n {
                                let d = path[spin][m][(mu, nu)];
                                variables[spin][n] += d * pair[n - m];
                                for axis in 0..3 {
                                    if gga {
                                        grad[spin][axis][n] += 2.0 * d * gp[axis][n - m];
                                    }
                                    if mgga {
                                        variables[5 + spin][n] += 0.5 * d * gg[axis][n - m];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        fields
    }

    fn xc_variables(
        &self,
        functional: &ResolvedFunctional,
        p: &[Vec<Matrix>; 2],
        order: usize,
    ) -> ([Vec<f64>; 7], [[Vec<f64>; 3]; 2]) {
        let width = order + 1;
        let gga = functional.family != Family::Lda;
        let mgga = functional.family == Family::MetaGga;
        let (mut variables, grad) = self.densities(p, order, gga, mgga);
        if gga {
            for (ix, (s, t)) in [(0, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                for a in 0..3 {
                    let v = product(&grad[s][a], &grad[t][a], order);
                    for n in 0..width {
                        variables[2 + ix][n] += v[n];
                    }
                }
            }
        }
        (variables, grad)
    }

    /// Quadrature coefficient of the current-current term v_tau/rho at
    /// the real reference, evaluated by the same stationary XC provider.
    pub(super) fn meta_current_weight(
        &self,
        functional: &ResolvedFunctional,
        p: &[Vec<Matrix>; 2],
    ) -> Result<[f64; 2]> {
        let (variables, _) = self.xc_variables(functional, p, 0);
        let rho = [variables[0][0], variables[1][0]];
        let xc = functional.semilocal_taylor_potentials(&XcTaylorInput {
            variables,
            parameters: None,
        })?;
        let coefficients = std::array::from_fn(|spin| {
            if rho[spin] > 0.0 {
                self.weight[0] * xc.potentials[5 + spin][0] / rho[spin]
            } else {
                0.0
            }
        });
        if coefficients.iter().any(|v| !v.is_finite()) {
            return Err(DftError::InvalidInput(
                "meta-GGA current kernel has a non-finite grid coefficient".into(),
            ));
        }
        Ok(coefficients)
    }

    fn evaluate(
        &self,
        functional: &ResolvedFunctional,
        p: &[Vec<Matrix>; 2],
        order: usize,
    ) -> Result<(Vec<f64>, [Vec<Matrix>; 2])> {
        let width = order + 1;
        let nao = self.aos.len();
        let gga = functional.family != Family::Lda;
        let mgga = functional.family == Family::MetaGga;
        let (variables, grad) = self.xc_variables(functional, p, order);
        let xc = functional.semilocal_taylor_potentials(&XcTaylorInput {
            variables,
            parameters: None,
        })?;
        let energy = product(&self.weight, &xc.energy, order);
        let mut fock: [Vec<Matrix>; 2] =
            std::array::from_fn(|_| vec![Matrix::zeros(nao, nao); width]);
        for s in 0..2 {
            let b: [Vec<f64>; 3] = std::array::from_fn(|a| {
                let own = product(
                    &xc.potentials[if s == 0 { 2 } else { 4 }],
                    &grad[s][a],
                    order,
                );
                let other = product(&xc.potentials[3], &grad[1 - s][a], order);
                (0..width).map(|n| 2.0 * own[n] + other[n]).collect()
            });
            for mu in 0..nao {
                for nu in 0..nao {
                    let pair = product(&self.aos[mu][0], &self.aos[nu][0], order);
                    let mut v = product(&xc.potentials[s], &pair, order);
                    for a in 0..3 {
                        if gga {
                            let mut gp = product(&self.aos[mu][a + 1], &self.aos[nu][0], order);
                            let other = product(&self.aos[mu][0], &self.aos[nu][a + 1], order);
                            for n in 0..width {
                                gp[n] += other[n];
                            }
                            let bv = product(&b[a], &gp, order);
                            for n in 0..width {
                                v[n] += bv[n];
                            }
                        }
                        if mgga {
                            let gg = product(&self.aos[mu][a + 1], &self.aos[nu][a + 1], order);
                            let tv = product(&xc.potentials[5 + s], &gg, order);
                            for n in 0..width {
                                v[n] += 0.5 * tv[n];
                            }
                        }
                    }
                    let weighted = product(&self.weight, &v, order);
                    for n in 0..width {
                        fock[s][n][(mu, nu)] = weighted[n];
                    }
                }
            }
        }
        Ok((energy, fock))
    }
}

impl GridPointTaylor {
    /// Kohn-Sham matrix coefficients of the first-order change of the
    /// semilocal potential caused by the symmetric density change `q` on the
    /// density path `p`.
    ///
    /// An independent forward-dual direction carries `dx(t)` through the
    /// common functional provider. Subtracting two high-order potential
    /// series loses a sub-ulp density correction before it reaches the
    /// response equation; this tangent preserves it without a step size.
    fn linear(
        &self,
        functional: &ResolvedFunctional,
        p: &[Vec<Matrix>; 2],
        q: &[Vec<Matrix>; 2],
        order: usize,
    ) -> Result<[Vec<Matrix>; 2]> {
        let width = order + 1;
        let nao = self.aos.len();
        let gga = functional.family != Family::Lda;
        let mgga = functional.family == Family::MetaGga;
        let [(mut base, grad), (mut change, grad_change)] =
            self.fields_many([p, q], order, gga, mgga);
        for spin in 0..2 {
            base[spin][0] = base[spin][0].max(0.0);
            base[5 + spin][0] = base[5 + spin][0].max(0.0);
        }
        if gga {
            for (ix, (s, t)) in [(0, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                for a in 0..3 {
                    let v = product(&grad[s][a], &grad[t][a], order);
                    let left = product(&grad_change[s][a], &grad[t][a], order);
                    let right = product(&grad[s][a], &grad_change[t][a], order);
                    for n in 0..width {
                        base[2 + ix][n] += v[n];
                        change[2 + ix][n] += left[n] + right[n];
                    }
                }
            }
        }
        let tangent = functional.semilocal_taylor_directional_potentials(
            &XcTaylorInput {
                variables: base,
                parameters: None,
            },
            &change,
        )?;
        let first = tangent.directional.potentials;
        let zeroth = &tangent.value.potentials;
        let mut fock: [Vec<Matrix>; 2] =
            std::array::from_fn(|_| vec![Matrix::zeros(nao, nao); width]);
        for s in 0..2 {
            let own = if s == 0 { 2 } else { 4 };
            let b: [Vec<f64>; 3] = std::array::from_fn(|a| {
                let terms = [
                    (product(&first[own], &grad[s][a], order), 2.0),
                    (
                        product(&zeroth[own][..width], &grad_change[s][a], order),
                        2.0,
                    ),
                    (product(&first[3], &grad[1 - s][a], order), 1.0),
                    (
                        product(&zeroth[3][..width], &grad_change[1 - s][a], order),
                        1.0,
                    ),
                ];
                (0..width)
                    .map(|n| terms.iter().map(|(v, scale)| scale * v[n]).sum::<f64>())
                    .collect()
            });
            for mu in 0..nao {
                for nu in 0..nao {
                    let pair = product(&self.aos[mu][0], &self.aos[nu][0], order);
                    let mut v = product(&first[s], &pair, order);
                    for a in 0..3 {
                        if gga {
                            let mut gp = product(&self.aos[mu][a + 1], &self.aos[nu][0], order);
                            let other = product(&self.aos[mu][0], &self.aos[nu][a + 1], order);
                            for n in 0..width {
                                gp[n] += other[n];
                            }
                            let bv = product(&b[a], &gp, order);
                            for n in 0..width {
                                v[n] += bv[n];
                            }
                        }
                        if mgga {
                            let gg = product(&self.aos[mu][a + 1], &self.aos[nu][a + 1], order);
                            let tv = product(&first[5 + s], &gg, order);
                            for n in 0..width {
                                v[n] += 0.5 * tv[n];
                            }
                        }
                    }
                    let weighted = product(&self.weight, &v, order);
                    for n in 0..width {
                        fock[s][n][(mu, nu)] = weighted[n];
                    }
                }
            }
        }
        Ok(fock)
    }
}

/// Taylor coefficients of `1 / m(t)`.
fn reciprocal(m: &[f64], order: usize) -> Vec<f64> {
    let mut out = vec![0.0; order + 1];
    out[0] = 1.0 / m[0];
    for n in 1..=order {
        let sum: f64 = (1..=n).map(|k| m[k] * out[n - k]).sum();
        out[n] = -sum / m[0];
    }
    out
}

impl GeometryTaylor {
    /// Coefficients of the response of the spin-flip Kohn-Sham operator to
    /// the transition density series `t` (row: the spin that gains the
    /// electron) along the density path `p`: the exact exchange in its index
    /// order and, when requested, the non-collinear kernel of the semilocal
    /// functional, which sees the symmetric part of `t`.
    pub fn evaluate_spin_flip(
        &self,
        functional: &ResolvedFunctional,
        p: &[Vec<Matrix>; 2],
        t: &[Matrix],
        order: usize,
        noncollinear: bool,
    ) -> Result<Vec<Matrix>> {
        let nao = self.overlap[0].rows;
        let exchange = self.exchange.as_ref().unwrap_or(&self.eri);
        let cx = functional.hybrid.cx;
        let clr = functional.hybrid.clr;
        let mut out = vec![Matrix::zeros(nao, nao); order + 1];
        if cx != 0.0 || self.eri_lr.is_some() {
            for n in 0..=order {
                for m in 0..=n {
                    for mu in 0..nao {
                        for nu in 0..nao {
                            for a in 0..nao {
                                for b in 0..nao {
                                    let kx = index4(nao, mu, a, nu, b);
                                    let mut weight = -cx * exchange[m][kx];
                                    if let Some(eri) = &self.eri_lr {
                                        weight -= clr * eri[m][kx];
                                    }
                                    out[n][(mu, nu)] += weight * t[n - m][(a, b)];
                                }
                            }
                        }
                    }
                }
            }
        }
        if noncollinear && !functional.parts.is_empty() {
            let pair: [Vec<Matrix>; 2] = [
                t[..=order]
                    .iter()
                    .map(|m| {
                        let mut symmetric = m.transpose();
                        add(&mut symmetric, m, 1.0);
                        for v in symmetric.as_mut_slice() {
                            *v *= 0.5;
                        }
                        symmetric
                    })
                    .collect(),
                vec![Matrix::zeros(nao, nao); order + 1],
            ];
            let zero = || vec![Matrix::zeros(nao, nao); order + 1];
            let total = self
                .grid
                .par_iter()
                .try_fold(zero, |mut acc, point| -> Result<Vec<Matrix>> {
                    let v = point.spin_flip(functional, p, &pair, order)?;
                    for n in 0..=order {
                        add(&mut acc[n], &v[n], 1.0);
                    }
                    Ok(acc)
                })
                .try_reduce(zero, |mut a, b| {
                    for n in 0..=order {
                        add(&mut a[n], &b[n], 1.0);
                    }
                    Ok(a)
                })?;
            for n in 0..=order {
                add(&mut out[n], &total[n], 1.0);
            }
        }
        Ok(out)
    }
}

impl GridPointTaylor {
    /// Matrix coefficients of the non-collinear spin-flip kernel acting on
    /// the symmetric transition density `t[0]` (`t[1]` is zero) along the
    /// density path `p`:
    ///
    /// ```text
    /// w { a rho_T phi_m phi_n + b . [rho_T grad(phi_m phi_n) + phi_m phi_n grad rho_T]
    ///   + c [rho_T grad phi_m . grad phi_n / 2 + phi_m phi_n tau_T] },
    /// a = Dv / m - (DG . grad m + Dv_tau Dtau) / m^2,   b = DG / m,   c = Dv_tau / m,
    /// ```
    ///
    /// with every factor a series in the path parameter. Where the spin
    /// density of the reference vanishes against the density, the kernel is
    /// the series of its local limit, half the second spin-difference
    /// derivative of the energy density, from the same shifted series as the
    /// linear response of the potentials. The choice is made at the
    /// reference point of the path.
    fn spin_flip(
        &self,
        functional: &ResolvedFunctional,
        p: &[Vec<Matrix>; 2],
        t: &[Vec<Matrix>; 2],
        order: usize,
    ) -> Result<Vec<Matrix>> {
        let width = order + 1;
        let nao = self.aos.len();
        let gga = functional.family != Family::Lda;
        let mgga = functional.family == Family::MetaGga;
        let (mut base, grad) = self.densities(p, order, gga, mgga);
        let total = base[0][0] + base[1][0];
        let spin = base[0][0] - base[1][0];
        if total <= 1e-12 {
            return Ok(vec![Matrix::zeros(nao, nao); width]);
        }
        if gga {
            for (ix, (s, u)) in [(0, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                for a in 0..3 {
                    let v = product(&grad[s][a], &grad[u][a], order);
                    for n in 0..width {
                        base[2 + ix][n] += v[n];
                    }
                }
            }
        }
        let (transition, transition_grad) = self.fields(t, order, gga, mgga);
        let zero = vec![0.0; width];
        let mut vector: [Vec<f64>; 3] = std::array::from_fn(|_| zero.clone());
        let mut kinetic = zero.clone();
        let local: Vec<f64> = if spin.abs() <= 1e-6 * total {
            let padded: [Vec<f64>; 7] = std::array::from_fn(|i| {
                let mut v = base[i].clone();
                v.resize(2 * width, 0.0);
                v
            });
            let shifted: [Vec<f64>; 7] = std::array::from_fn(|i| {
                let mut v = base[i].clone();
                v.resize(2 * width, 0.0);
                match i {
                    0 => v[width] = 1.0,
                    1 => v[width] = -1.0,
                    _ => {}
                }
                v
            });
            let reference = functional.semilocal_taylor_potentials(&XcTaylorInput {
                variables: padded,
                parameters: None,
            })?;
            let moved = functional.semilocal_taylor_potentials(&XcTaylorInput {
                variables: shifted,
                parameters: None,
            })?;
            (0..width)
                .map(|k| {
                    0.5 * ((moved.potentials[0][width + k] - reference.potentials[0][width + k])
                        - (moved.potentials[1][width + k] - reference.potentials[1][width + k]))
                })
                .collect()
        } else {
            let xc = functional.semilocal_taylor_potentials(&XcTaylorInput {
                variables: base.clone(),
                parameters: None,
            })?;
            let difference = |a: &[f64], b: &[f64]| -> Vec<f64> {
                a.iter().zip(b).take(width).map(|(x, y)| x - y).collect()
            };
            let inverse = reciprocal(&difference(&base[0], &base[1]), order);
            let mut local = product(
                &difference(&xc.potentials[0], &xc.potentials[1]),
                &inverse,
                order,
            );
            if gga {
                let own: [Vec<f64>; 2] = [
                    (0..width)
                        .map(|n| 2.0 * xc.potentials[2][n] - xc.potentials[3][n])
                        .collect(),
                    (0..width)
                        .map(|n| 2.0 * xc.potentials[4][n] - xc.potentials[3][n])
                        .collect(),
                ];
                let mut projected = zero.clone();
                for (k, slot) in vector.iter_mut().enumerate() {
                    let dg = difference(
                        &product(&own[0], &grad[0][k], order),
                        &product(&own[1], &grad[1][k], order),
                    );
                    let term = product(&dg, &difference(&grad[0][k], &grad[1][k]), order);
                    for n in 0..width {
                        projected[n] += term[n];
                    }
                    *slot = product(&dg, &inverse, order);
                }
                if mgga {
                    let dv_tau = difference(&xc.potentials[5], &xc.potentials[6]);
                    let term = product(&dv_tau, &difference(&base[5], &base[6]), order);
                    for n in 0..width {
                        projected[n] += term[n];
                    }
                    kinetic = product(&dv_tau, &inverse, order);
                }
                let correction = product(&projected, &product(&inverse, &inverse, order), order);
                for n in 0..width {
                    local[n] -= correction[n];
                }
            }
            local
        };
        // Factors of phi_m phi_n, of grad(phi_m phi_n) and of
        // grad phi_m . grad phi_n.
        let rho = &transition[0];
        let mut scalar = product(&local, rho, order);
        let mut along: [Vec<f64>; 3] = std::array::from_fn(|_| zero.clone());
        let mut gradient_pair = zero.clone();
        if gga {
            for (k, slot) in along.iter_mut().enumerate() {
                let slope = product(&vector[k], &transition_grad[0][k], order);
                for n in 0..width {
                    scalar[n] += slope[n];
                }
                *slot = product(&vector[k], rho, order);
            }
        }
        if mgga {
            let term = product(&kinetic, &transition[5], order);
            for n in 0..width {
                scalar[n] += term[n];
            }
            gradient_pair = product(&kinetic, rho, order)
                .iter()
                .map(|v| 0.5 * v)
                .collect();
        }
        let mut out = vec![Matrix::zeros(nao, nao); width];
        for mu in 0..nao {
            for nu in 0..nao {
                let pair = product(&self.aos[mu][0], &self.aos[nu][0], order);
                let mut v = product(&scalar, &pair, order);
                for a in 0..3 {
                    if gga {
                        let mut gp = product(&self.aos[mu][a + 1], &self.aos[nu][0], order);
                        let other = product(&self.aos[mu][0], &self.aos[nu][a + 1], order);
                        for n in 0..width {
                            gp[n] += other[n];
                        }
                        let bv = product(&along[a], &gp, order);
                        for n in 0..width {
                            v[n] += bv[n];
                        }
                    }
                    if mgga {
                        let gg = product(&self.aos[mu][a + 1], &self.aos[nu][a + 1], order);
                        let tv = product(&gradient_pair, &gg, order);
                        for n in 0..width {
                            v[n] += tv[n];
                        }
                    }
                }
                let weighted = product(&self.weight, &v, order);
                for n in 0..width {
                    out[n][(mu, nu)] = weighted[n];
                }
            }
        }
        Ok(out)
    }
}

/// A learned functional on the moving grid: the network is evaluated on the
/// Taylor series of the fields of all grid points at once.
#[derive(Clone)]
struct LearnedTaylor {
    model: Arc<SkalaModel>,
    atom_of: Vec<usize>,
    points: Vec<[f64; 3]>,
    centers: Vec<[f64; 3]>,
    raw_weights: Vec<f64>,
}

impl LearnedTaylor {
    fn new(model: &Arc<SkalaModel>, mol: &Molecule, grid: &MolecularGrid) -> Self {
        Self {
            model: model.clone(),
            atom_of: grid.atom_of.clone(),
            points: grid.points.iter().map(|p| p.to_array()).collect(),
            centers: mol
                .atoms
                .iter()
                .map(|atom| atom.position.to_array())
                .collect(),
            raw_weights: grid.raw_weights.clone(),
        }
    }
    /// Energy coefficients and Kohn-Sham matrix coefficients of the density
    /// path `p` through `order < W`.
    fn evaluate<const W: usize>(
        &self,
        grid: &[GridPointTaylor],
        p: &[Vec<Matrix>; 2],
        order: usize,
    ) -> Result<(Vec<f64>, [Vec<Matrix>; 2])> {
        let fields: Vec<_> = grid
            .par_iter()
            .map(|point| point.densities(p, order, true, true))
            .collect();
        let series = |values: &[f64]| Taylor::<W>::from_coefficients(values);
        let rho: [Vec<Taylor<W>>; 2] =
            std::array::from_fn(|s| fields.iter().map(|(v, _)| series(&v[s])).collect());
        let tau: [Vec<Taylor<W>>; 2] =
            std::array::from_fn(|s| fields.iter().map(|(v, _)| series(&v[5 + s])).collect());
        let gradient: [Vec<[Taylor<W>; 3]>; 2] = std::array::from_fn(|s| {
            fields
                .iter()
                .map(|(_, g)| std::array::from_fn(|a| series(&g[s][a])))
                .collect()
        });
        let weights: Vec<Taylor<W>> = grid
            .iter()
            .map(|point| series(&point.weight[..=order]))
            .collect();
        let out = self.model.evaluate(
            &SkalaInput {
                atom_of: &self.atom_of,
                points: &self.points,
                centers: &self.centers,
                weights: &weights,
                atomic_weights: &self.raw_weights,
                rho: [&rho[0], &rho[1]],
                gradient: [&gradient[0], &gradient[1]],
                tau: [&tau[0], &tau[1]],
            },
            true,
        )?;
        let width = order + 1;
        let nao = grid.first().map_or(0, |point| point.aos.len());
        let zero = || -> [Vec<Matrix>; 2] {
            std::array::from_fn(|_| vec![Matrix::zeros(nao, nao); width])
        };
        let fock = grid
            .par_iter()
            .enumerate()
            .fold(zero, |mut acc, (i, point)| {
                let phi: Vec<[Taylor<W>; 4]> = point
                    .aos
                    .iter()
                    .map(|ao| std::array::from_fn(|a| series(&ao[a][..=order])))
                    .collect();
                for (s, target) in acc.iter_mut().enumerate() {
                    let a = out.d_rho[s][i];
                    let b = out.d_gradient[s][i];
                    let half = out.d_tau[s][i].scale(0.5);
                    // A_m = a phi_m / 2 + b . grad phi_m.
                    let left: Vec<Taylor<W>> = phi
                        .iter()
                        .map(|ao| {
                            (a * ao[0]).scale(0.5) + b[0] * ao[1] + b[1] * ao[2] + b[2] * ao[3]
                        })
                        .collect();
                    for mu in 0..nao {
                        for nu in 0..nao {
                            let v = left[mu] * phi[nu][0]
                                + phi[mu][0] * left[nu]
                                + half
                                    * (phi[mu][1] * phi[nu][1]
                                        + phi[mu][2] * phi[nu][2]
                                        + phi[mu][3] * phi[nu][3]);
                            for (n, matrix) in target.iter_mut().enumerate() {
                                matrix[(mu, nu)] += v.c[n];
                            }
                        }
                    }
                }
                acc
            })
            .reduce(zero, |mut a, b| {
                for s in 0..2 {
                    for n in 0..width {
                        add(&mut a[s][n], &b[s][n], 1.0);
                    }
                }
                a
            });
        Ok((out.energy.c[..width].to_vec(), fock))
    }
}

/// The same learned energy/potential Taylor provider on a fixed AO grid.
/// No moving one-/two-electron integral tensors are constructed here.
pub(crate) fn learned_fock_coefficient(
    model: &Arc<SkalaModel>,
    mol: &Molecule,
    mb: &MolBasis,
    grid: &MolecularGrid,
    density: &[[Matrix; 2]],
    order: usize,
) -> Result<[Matrix; 2]> {
    if order > 7 || density.len() <= order || grid.is_empty() {
        return Err(DftError::InvalidInput("learned fixed-grid Fock response needs a nonempty grid, complete density series and order zero through seven".into()));
    }
    let batch = crate::grid::eval::eval_aos_full(mb, &grid.points, true, false);
    let gradient = batch.grad.as_ref().expect("AO gradients requested");
    let points: Vec<_> = (0..grid.len())
        .map(|i| {
            let mut weight = vec![0.0; order + 1];
            weight[0] = grid.weights[i];
            let aos = (0..mb.nao_sph)
                .map(|mu| {
                    std::array::from_fn(|a| {
                        let mut coefficient = vec![0.0; order + 1];
                        coefficient[0] = if a == 0 {
                            batch.phi[(i, mu)]
                        } else {
                            gradient[a - 1][(i, mu)]
                        };
                        coefficient
                    })
                })
                .collect();
            GridPointTaylor { weight, aos }
        })
        .collect();
    let p: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
        density
            .iter()
            .take(order + 1)
            .map(|v| v[s].clone())
            .collect()
    });
    let provider = LearnedTaylor::new(model, mol, grid);
    let (_, fock) = if order <= 3 {
        provider.evaluate::<4>(&points, &p, order)?
    } else {
        provider.evaluate::<8>(&points, &p, order)?
    };
    Ok([fock[0][order].clone(), fock[1][order].clone()])
}

/// Coupling of a periodic cell at Gamma to a uniform electric field through
/// the single-point Berry phase,
///
/// ```text
/// -Omega E . P = sum_a (E . a_a) / (2 pi)
///                [sum_s Im ln det(1 + D_s dQ_a) - sum_A Z_A b_a . R_A],
/// ```
///
/// with `dQ_a = <m| exp(i b_a . r) - 1 |n>` between the Bloch sums. For an
/// idempotent spin density the determinant is that of `exp(i b_a . r)`
/// between the occupied orbitals, the product of the Berry-phase strings of
/// the k mesh that the cell folds. The term is a function of the density
/// matrix, so it enters the path recursion like any other part of the
/// energy; with a field that vanishes at the start of the path its order-`n`
/// Kohn-Sham matrix needs the density through order `n - 1` only.
#[derive(Clone)]
struct LatticeField {
    phase_reference: Option<[f64; 3]>,
    polar: [[Vec<Matrix>; 2]; 3],
    ionic: [Vec<f64>; 3],
    /// Series of the reduced field `E(t) . a_a / (2 pi)`.
    reduced: [Vec<f64>; 3],
}

impl LatticeField {
    fn provider(&self) -> crate::response::berry_field::BerryField {
        crate::response::berry_field::BerryField {
            phase_reference: self.phase_reference,
            polar: std::array::from_fn(|a| {
                self.polar[a][0]
                    .iter()
                    .cloned()
                    .zip(self.polar[a][1].iter().cloned())
                    .collect()
            }),
            ionic: self.ionic.clone(),
            reduced: self.reduced.clone(),
        }
    }
    fn complex_path(p: &[Vec<Matrix>; 2]) -> Vec<Vec<(Matrix, Matrix)>> {
        p.iter()
            .map(|v| {
                v.iter()
                    .map(|m| (m.clone(), Matrix::zeros(m.rows, m.cols)))
                    .collect()
            })
            .collect()
    }
    fn phases(&self, p: &[Matrix; 2]) -> Result<[f64; 3]> {
        self.provider().phases(
            &p.iter()
                .map(|m| (m.clone(), Matrix::zeros(m.rows, m.cols)))
                .collect::<Vec<_>>(),
        )
    }
    fn linear(
        &self,
        p: &[Vec<Matrix>; 2],
        q: &[Vec<Matrix>; 2],
        order: usize,
    ) -> Result<[Vec<Matrix>; 2]> {
        let value =
            self.provider()
                .linear(&Self::complex_path(p), &Self::complex_path(q), order)?;
        Ok(std::array::from_fn(|s| {
            value[s].iter().map(|m| m.0.clone()).collect()
        }))
    }
    fn evaluate(&self, p: &[Vec<Matrix>; 2], order: usize) -> Result<(Vec<f64>, [Vec<Matrix>; 2])> {
        let (energy, value) = self.provider().evaluate(&Self::complex_path(p), order)?;
        Ok((
            energy,
            std::array::from_fn(|s| value[s].iter().map(|m| m.0.clone()).collect()),
        ))
    }
}

#[cfg(test)]
mod field_convergence {
    include!("../../../tests/support/convergence.rs");
}

#[cfg(test)]
mod field_tests {
    use super::*;

    #[test]
    fn biased_berry_density_hessian_has_an_independent_four_step_fit() {
        let matrix = |values: [f64; 4]| {
            let mut m = Matrix::zeros(2, 2);
            m.as_mut_slice().copy_from_slice(&values);
            m
        };
        let field = LatticeField {
            phase_reference: None,
            polar: std::array::from_fn(|_| {
                [
                    vec![matrix([-0.2, 0.03, 0.03, -0.1])],
                    vec![matrix([0.4, 0.05, 0.05, 0.2])],
                ]
            }),
            ionic: std::array::from_fn(|_| vec![0.0]),
            reduced: [vec![0.03], vec![-0.01], vec![0.02]],
        };
        let p = [
            vec![matrix([0.5, 0.02, 0.02, 0.4])],
            vec![matrix([0.3, -0.01, -0.01, 0.2])],
        ];
        let q = [
            vec![matrix([0.1, 0.07, 0.07, -0.08])],
            vec![matrix([-0.05, 0.04, 0.04, 0.06])],
        ];
        let exact = field.linear(&p, &q, 0).unwrap();
        let steps = [0.08, 0.04, 0.02, 0.01];
        let mut errors = Vec::new();
        for h in steps {
            let shifted = |sign: f64| {
                let mut state = p.clone();
                for s in 0..2 {
                    add(&mut state[s][0], &q[s][0], sign * h);
                }
                field.evaluate(&state, 0).unwrap().1
            };
            let plus = shifted(1.0);
            let minus = shifted(-1.0);
            let error = (0..2)
                .flat_map(|s| {
                    plus[s][0]
                        .as_slice()
                        .iter()
                        .zip(minus[s][0].as_slice())
                        .zip(exact[s][0].as_slice())
                        .map(move |((p, m), exact)| ((p - m) / (2.0 * h) - exact).powi(2))
                })
                .sum::<f64>()
                .sqrt();
            errors.push(error);
        }
        let fit = field_convergence::error_power_fit(&steps, &errors);
        println!(
            "biased Berry density kernel steps={steps:?} errors={errors:?} p={:.9} R2={:.12}",
            fit.exponent, fit.r_squared
        );
        assert!((fit.exponent - 2.0).abs() < 0.01 && fit.r_squared > 0.99999);
    }
}

/// Analytic atom-owned Becke weights and real Gaussian AO values/gradients.
/// Shared by real and complex finite-field nuclear response providers.
pub(super) fn moving_grid_taylor(
    mol: &Molecule,
    mb: &MolBasis,
    grid: &MolecularGrid,
    input: &[Vec<f64>],
    order: usize,
) -> Result<Vec<GridPointTaylor>> {
    let nao = mb.nao_sph;
    let sites = mb
        .shells
        .iter()
        .map(|s| s.atom + 1)
        .fold(mol.len(), usize::max);
    if grid.points.len() != grid.raw_weights.len()
        || grid.points.len() != grid.atom_of.len()
        || grid.points.len() != grid.weights.len()
        || grid.atom_of.iter().any(|&a| a >= mol.len())
        || grid.raw_weights.iter().any(|w| !w.is_finite())
        || input.len() < 3 * sites
    {
        return Err(DftError::InvalidInput(
            "invalid atom-owned response grid".into(),
        ));
    }
    let aij = MolecularGrid::becke_aij(mol);
    (0..grid.len())
        .into_par_iter()
        .map(|i| {
            let owner = grid.atom_of[i];
            let offset = grid.points[i] - mol.atoms[owner].position;
            let coefficients = evaluate_series_many_result(input, |v| {
                let c = centers(&v[..3 * sites]);
                let sh = shells(mb, mol, &c)?;
                let point = [
                    c[owner][0] + SeriesScalar::from_f64(offset.x),
                    c[owner][1] + SeriesScalar::from_f64(offset.y),
                    c[owner][2] + SeriesScalar::from_f64(offset.z),
                ];
                let w = becke_weight(&c[..mol.len()], &aij, owner, point, order)?
                    * SeriesScalar::from_f64(grid.raw_weights[i]);
                let mut values = vec![SeriesScalar::from_f64(0.0); 1 + 4 * nao];
                values[0] = w;
                for (placed, shell) in mb.shells.iter().zip(sh) {
                    let r = std::array::from_fn::<_, 3, _>(|a| point[a] - shell.center[a]);
                    let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
                    // Contracted radial factor sum_p c_p exp(-a_p r^2) and its
                    // slope sum_p c_p (-2 a_p) exp(-a_p r^2), once per shell.
                    let mut radial = SeriesScalar::from_f64(0.0);
                    let mut slope = SeriesScalar::from_f64(0.0);
                    for (&exponent, &coef) in shell.exponents.iter().zip(&shell.coefficients) {
                        let term = coef * (-exponent * r2).exp();
                        radial = radial + term;
                        slope = slope - SeriesScalar::from_f64(2.0) * exponent * term;
                    }
                    let components = cart_components(shell.l);
                    let transform = cart_to_sph(shell.l);
                    for (cart, (lx, ly, lz)) in components.into_iter().enumerate() {
                        let angular = [lx, ly, lz];
                        let polynomial = r[0].powi(lx) * r[1].powi(ly) * r[2].powi(lz);
                        let value = radial * polynomial;
                        let gradient: [SeriesScalar; 3] = std::array::from_fn(|axis| {
                            let mut derivative = slope * r[axis] * polynomial;
                            if angular[axis] > 0 {
                                let mut lowered = SeriesScalar::from_f64(angular[axis] as f64);
                                for a in 0..3 {
                                    lowered =
                                        lowered * r[a].powi(angular[a] - i32::from(a == axis));
                                }
                                derivative = derivative + lowered * radial;
                            }
                            derivative
                        });
                        for sph in 0..n_sph(shell.l) {
                            let coefficient = transform[(cart, sph)];
                            if coefficient == 0.0 {
                                continue;
                            }
                            let t = SeriesScalar::from_f64(coefficient);
                            let ix = 1 + 4 * (placed.sph_off + sph);
                            values[ix] = values[ix] + t * value;
                            for axis in 0..3 {
                                values[ix + 1 + axis] = values[ix + 1 + axis] + t * gradient[axis];
                            }
                        }
                    }
                }
                Ok(values)
            })?;
            Ok(GridPointTaylor {
                weight: coefficients[0].clone(),
                aos: (0..nao)
                    .map(|mu| std::array::from_fn(|a| coefficients[1 + 4 * mu + a].clone()))
                    .collect(),
            })
        })
        .collect::<Result<Vec<_>>>()
}

#[cfg(test)]
mod pair_series_tests {
    use super::*;
    use crate::acceptance::{components, stencils};

    fn rational_derivative(a: f64, b: f64, n: usize, t: f64) -> f64 {
        let factorial = (1..=n).product::<usize>() as f64;
        (-1.0_f64).powi(n as i32) * factorial / (a - b)
            * (a.powi(n as i32 + 1) / (1.0 + a * t).powi(n as i32 + 1)
                - b.powi(n as i32 + 1) / (1.0 + b * t).powi(n as i32 + 1))
    }

    #[test]
    fn every_pair_contraction_derivative_matches_independent_rational_functions() {
        let tensor: Vec<Vec<f64>> = (0..3)
            .map(|kind| {
                (0..16)
                    .map(|i| 0.17 + ((i * (7 + kind * 2) + kind * 3) % 19) as f64 * 0.013)
                    .collect()
            })
            .collect();
        let rates: [f64; 3] = [0.31, 0.17, -0.23];
        let population_rates: [f64; 2] = [0.43, 0.67];
        let p0 = [
            Matrix::from_row_major(2, 2, vec![0.7, 0.13, 0.13, 0.4]),
            Matrix::from_row_major(2, 2, vec![0.5, -0.11, -0.11, 0.3]),
        ];
        let integrals: Vec<Vec<Vec<f64>>> = (0..3)
            .map(|kind| {
                (0..=4)
                    .map(|n| {
                        tensor[kind]
                            .iter()
                            .map(|v| v * (-rates[kind]).powi(n))
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let density: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
            (0..=4)
                .map(|n| {
                    Matrix::from_row_major(
                        2,
                        2,
                        p0[s]
                            .as_slice()
                            .iter()
                            .map(|v| v * (-population_rates[s]).powi(n))
                            .collect(),
                    )
                })
                .collect()
        });
        let (j, k, lr) = contract_pair_series(
            Some(&integrals[0]),
            Some(&integrals[1]),
            Some(&integrals[2]),
            &density,
            4,
        );
        let values = [&j, &k[0], &k[1], &lr[0], &lr[1]];
        // The oracle differentiates closed-form rational functions. It does
        // not use Taylor convolutions, compensated sums or tensor symmetry.
        let oracle = |channel: usize, n: usize, t: f64, i: usize, j: usize| {
            let mut value = 0.0;
            for a in 0..2 {
                for b in 0..2 {
                    if channel == 0 {
                        for s in 0..2 {
                            value += tensor[0][index4(2, i, j, a, b)]
                                * p0[s][(a, b)]
                                * rational_derivative(rates[0], population_rates[s], n, t);
                        }
                    } else {
                        let kind = if channel <= 2 { 1 } else { 2 };
                        let s = (channel - 1) % 2;
                        value += tensor[kind][index4(2, i, a, j, b)]
                            * p0[s][(a, b)]
                            * rational_derivative(rates[kind], population_rates[s], n, t);
                    }
                }
            }
            value
        };
        let mut names = Vec::new();
        for channel in 0..5 {
            for n in 1..=4 {
                for i in 0..2 {
                    for j in 0..2 {
                        names.push(format!(
                            "pair series channel={channel} derivative={n} row={i} col={j}"
                        ));
                    }
                }
            }
        }
        let stages = [
            ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
            ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
            ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Fourth),
        ];
        let mut failures = Vec::new();
        stencils::fit_components_staged(
            &names,
            &stages,
            |steps, stencil| {
                let mut errors = Vec::new();
                for (channel, series) in values.iter().enumerate() {
                    for n in 1..=4 {
                        for i in 0..2 {
                            for j in 0..2 {
                                let analytic =
                                    series[n][(i, j)] * (1..=n).product::<usize>() as f64;
                                errors.push(std::array::from_fn(|q| {
                                    let h = steps[q];
                                    let estimate = stencil.first(
                                        oracle(channel, n - 1, h, i, j),
                                        oracle(channel, n - 1, -h, i, j),
                                        oracle(channel, n - 1, 2.0 * h, i, j),
                                        oracle(channel, n - 1, -2.0 * h, i, j),
                                        h,
                                    );
                                    (estimate - analytic).abs()
                                }));
                            }
                        }
                    }
                }
                errors
            },
            &mut failures,
        );
        components::assert_no_failures("all80 Coulomb/exchange coefficient derivatives", &failures);
        let mut f = Matrix::from_row_major(2, 2, vec![0.19, -0.23, 0.31, -0.17]);
        let base = f.clone();
        add_pair_fock(&mut f, &j[4], &k[0][4], &lr[0][4], (0.37, 0.21));
        for i in 0..2 {
            for a in 0..2 {
                let expected =
                    base[(i, a)] + j[4][(i, a)] - 0.37 * k[0][4][(i, a)] - 0.21 * lr[0][4][(i, a)];
                assert!((f[(i, a)] - expected).abs() < 1e-15);
            }
        }
    }
}
#[cfg(test)]
mod density_fields_tests {
    use super::*;
    use crate::acceptance::{components, stencils};

    fn polynomial<S: Scalar>(c: &[f64], t: S) -> S {
        c.iter()
            .rev()
            .fold(S::from_f64(0.0), |v, &c| v * t + S::from_f64(c))
    }

    fn oracle<S: Scalar>(point: &GridPointTaylor, p: &[Vec<Matrix>; 2], t: S) -> [S; 10] {
        let mut out = [S::from_f64(0.0); 10];
        for spin in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    let c: Vec<_> = p[spin].iter().map(|v| v[(i, j)]).collect();
                    let density = polynomial(&c, t);
                    let a = polynomial(&point.aos[i][0], t);
                    let b = polynomial(&point.aos[j][0], t);
                    out[spin] = out[spin] + density * a * b;
                    for axis in 0..3 {
                        let da = polynomial(&point.aos[i][axis + 1], t);
                        let db = polynomial(&point.aos[j][axis + 1], t);
                        out[2 + spin] = out[2 + spin] + S::from_f64(0.5) * density * da * db;
                        out[4 + 3 * spin + axis] =
                            out[4 + 3 * spin + axis] + S::from_f64(2.0) * density * da * b;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn every_shared_density_gradient_tau_coefficient_fits_independent_polynomials() {
        let point = GridPointTaylor {
            weight: vec![0.31, 0.07, -0.03, 0.011, 0.004],
            aos: (0..2)
                .map(|i| {
                    std::array::from_fn(|a| {
                        (0..5)
                            .map(|n| {
                                let index = (11 * i + 7 * a + 3 * n + 1) as f64;
                                (0.37 * index).sin() / (n + 1) as f64 + 0.09
                            })
                            .collect()
                    })
                })
                .collect(),
        };
        let paths: [[Vec<Matrix>; 2]; 2] = std::array::from_fn(|channel| {
            std::array::from_fn(|spin| {
                (0..5)
                    .map(|n| {
                        Matrix::from_row_major(
                            2,
                            2,
                            (0..4)
                                .map(|i| {
                                    let index =
                                        (19 * channel + 11 * spin + 7 * n + 3 * i + 1) as f64;
                                    (0.23 * index).cos() / (n + 1) as f64 + 0.13
                                })
                                .collect(),
                        )
                    })
                    .collect()
            })
        });
        let fields = point.fields_many([&paths[0], &paths[1]], 4, true, true);
        for channel in 0..2 {
            let separate = point.fields(&paths[channel], 4, true, true);
            assert_eq!(fields[channel], separate);
        }
        let names: Vec<_> = (0..2)
            .flat_map(|c| {
                (1..=4).flat_map(move |n| {
                    (0..10).map(move |i| {
                        format!("shared fields channel={c} derivative={n} component={i}")
                    })
                })
            })
            .collect();
        let stages = [
            ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
            ([0.02, 0.01, 0.005, 0.0025], stencils::Stencil::Second),
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
            ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Fourth),
        ];
        let mut failures = Vec::new();
        stencils::fit_components_staged(
            &names,
            &stages,
            |steps, stencil| {
                let mut errors = Vec::new();
                for channel in 0..2 {
                    for n in 1..=4 {
                        for component in 0..10 {
                            let value = match component {
                                0..=1 => fields[channel].0[component][n],
                                2..=3 => fields[channel].0[component + 3][n],
                                _ => fields[channel].1[(component - 4) / 3][(component - 4) % 3][n],
                            } * (1..=n).product::<usize>() as f64;
                            let lower = |t| {
                                oracle(
                                    &point,
                                    &paths[channel],
                                    Taylor::<4>::from_coefficients(&[t, 1.0]),
                                )[component]
                                    .c[n - 1]
                                    * (1..n).product::<usize>() as f64
                            };
                            errors.push(std::array::from_fn(|k| {
                                let h = steps[k];
                                (stencil.first(
                                    lower(h),
                                    lower(-h),
                                    lower(2.0 * h),
                                    lower(-2.0 * h),
                                    h,
                                ) - value)
                                    .abs()
                            }));
                        }
                    }
                }
                errors
            },
            &mut failures,
        );
        components::assert_no_failures(
            "all80 shared density/gradient/tau derivative entries",
            &failures,
        );
    }
}
