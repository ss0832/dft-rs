// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//! One analytic affine assembly for restricted and unrestricted frozen spins.
use super::*;
use crate::pseudo_sub::reduction_reference::{
    CollinearActiveSpaceScf, CollinearFrozenEvaluation, CollinearFrozenScf,
};

pub struct CollinearAffineFrozenGeometry<'a> {
    pub full_reference: &'a Molecule,
    pub full_basis: &'a MolBasis,
    pub raw_reference: &'a Molecule,
    pub raw_basis: &'a MolBasis,
    pub frozen_reference: [&'a Matrix; 2],
    pub full_atom_owners: &'a [usize],
    pub functional: &'a ResolvedFunctional,
    pub options: &'a ScfOptions,
    pub grid_options: &'a GridOptions,
    pub active_electrons: [usize; 2],
    pub reference_energy: f64,
}
pub struct CollinearAffineFrozenGeometryResult {
    pub energy: f64,
    pub gradient: Vec<Vec3>,
    pub active_scf: CollinearFrozenScf,
    pub full_molecule: Molecule,
    pub chart_metric_error: [f64; 2],
    pub frozen_overlap: [f64; 2],
}

/// A restricted determinant is represented as two equal unit-spin densities
/// after solving its one constrained spatial channel. This conversion does
/// not change its SCF stopping conditions or permit spin symmetry breaking.
fn restricted_spins(value: FrozenReferenceScf) -> CollinearFrozenScf {
    let mut half = value.active_density.clone();
    for x in half.as_mut_slice() {
        *x *= 0.5;
    }
    let e = value.evaluation;
    CollinearFrozenScf {
        active_density: [half.clone(), half],
        active_orbitals: [value.active_orbitals.clone(), value.active_orbitals],
        orbital_energies: [value.orbital_energies.clone(), value.orbital_energies],
        evaluation: CollinearFrozenEvaluation {
            energy: e.energy,
            electronic_energy: e.electronic_energy,
            one_electron_energy: e.one_electron_energy,
            hartree_energy: e.hartree_energy,
            xc_energy: e.xc_energy,
            nuclear_energy: e.nuclear_energy,
            reference_energy: e.reference_energy,
            fock: [e.fock.clone(), e.fock],
            active_electrons: [e.active_electrons * 0.5; 2],
            frozen_electrons: [e.frozen_electrons * 0.5; 2],
            grid_total_electrons: e.grid_total_electrons,
        },
        history: value.history,
        projected_orbital_residual: value.projected_orbital_residual,
        active_idempotency_error: [value.active_idempotency_error * 0.25; 2],
        active_frozen_overlap: [value.active_frozen_overlap; 2],
        particle_number_error: [value.particle_number_error * 0.5; 2],
    }
}

impl CollinearAffineFrozenGeometry<'_> {
    pub fn evaluate(
        &self,
        coordinates: &[Vec3],
        initial: Option<[&Matrix; 2]>,
    ) -> Result<CollinearAffineFrozenGeometryResult> {
        self.evaluate_internal(coordinates, initial, false)
    }
    pub fn optimize(
        &self,
        options: &crate::optimizer::OptOptions,
    ) -> Result<crate::optimizer::OptResult<CollinearAffineFrozenGeometryResult>> {
        crate::optimizer::optimize_with_evaluator(self.raw_reference, options, |m| {
            let r = m.atoms.iter().map(|a| a.position).collect::<Vec<_>>();
            let value = self.evaluate(&r, None)?;
            Ok((value.energy, value.gradient.clone(), value))
        })
    }
    pub(super) fn evaluate_internal(
        &self,
        coordinates: &[Vec3],
        initial: Option<[&Matrix; 2]>,
        restricted: bool,
    ) -> Result<CollinearAffineFrozenGeometryResult> {
        if (!restricted && self.options.reference == crate::scf::Reference::Rhf)
            || (restricted && self.options.reference == crate::scf::Reference::Uhf)
        {
            return invalid(
                "affine scalar reference selector conflicts with its declared spin sector",
            );
        }
        let nr = self.raw_reference.len();
        let nf = self.full_reference.len();
        let n = self.full_basis.nao_sph;
        let m = self.raw_basis.nao_sph;
        let counts = self.full_reference.n_alpha_beta()?;
        if nr == 0
            || nf == 0
            || coordinates.len() != nr
            || self.full_atom_owners.len() != nf
            || self.full_atom_owners.iter().any(|i| *i >= nr)
            || coordinates
                .iter()
                .any(|r| r.to_array().iter().any(|x| !x.is_finite()))
            || self.frozen_reference.iter().any(|b| b.rows != n)
            || self.grid_options.weight_cutoff != 0.
            || self.raw_reference.n_alpha_beta()?
                != (self.active_electrons[0], self.active_electrons[1])
            || self.active_electrons[0].checked_add(self.frozen_reference[0].cols) != Some(counts.0)
            || self.active_electrons[1].checked_add(self.frozen_reference[1].cols) != Some(counts.1)
        {
            return invalid("affine frozen geometry requires finite fixed spin/electron/AO/owner contracts and zero grid cutoff");
        }
        let s0 = overlap_matrix(self.full_basis);
        for b in self.frozen_reference {
            if b.transpose_matmul_seq(&s0.matmul_seq(b))
                .max_abs_diff(&Matrix::identity(b.cols))
                > 1e-9
            {
                return invalid("saved frozen spin orbitals are not metric orthonormal");
            }
        }
        let mut raw = self.raw_reference.clone();
        for (a, r) in raw.atoms.iter_mut().zip(coordinates) {
            a.position = *r;
        }
        let mut full = self.full_reference.clone();
        for (a, owner) in full.atoms.iter_mut().zip(self.full_atom_owners) {
            a.position += coordinates[*owner] - self.raw_reference.atoms[*owner].position;
        }
        let rb = moved_basis(self.raw_basis, &raw)?;
        let fb = moved_basis(self.full_basis, &full)?;
        let sr = overlap_matrix(&rb);
        let sf = overlap_matrix(&fb);
        let cross = cross_basis_overlap(&fb, &rb)?;
        let gram: Vec<Matrix> = self
            .frozen_reference
            .iter()
            .map(|b| b.transpose_matmul_seq(&sf.matmul_seq(b)))
            .collect();
        let frozen = self
            .frozen_reference
            .iter()
            .zip(&gram)
            .map(|(b, g)| {
                if b.cols == 0 {
                    Ok((*b).clone())
                } else {
                    Ok(b.matmul_seq(&positive_power(g, -0.5)?))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let bath = frozen
            .iter()
            .map(|b| b.matmul_transpose_seq(b))
            .collect::<Vec<_>>();
        let charts = frozen
            .iter()
            .map(|b| OrthogonalReducedChart::new(&sf, &cross, &sr, b, 1e-10))
            .collect::<Result<Vec<_>>>()?;
        let grid = MolecularGrid::build(&full, self.grid_options)?;
        let mut total_bath = bath[0].clone();
        add(&mut total_bath, &bath[1], 1.);
        let evaluator = if restricted {
            FrozenDensityEvaluator::new(
                &full,
                &fb,
                &grid,
                self.functional,
                self.options,
                &total_bath,
            )?
        } else {
            FrozenDensityEvaluator::new_collinear(
                &full,
                &fb,
                &grid,
                self.functional,
                self.options,
                [&bath[0], &bath[1]],
            )?
        };
        let orthogonalizer = Orthogonalizer::canonical(&sr, self.options.orth_threshold)?;
        if orthogonalizer.n_dropped != 0 {
            return invalid("affine geometry loses the common raw AO rank");
        }
        let mut core = kinetic_matrix(&rb);
        add(&mut core, &nuclear_matrix(&rb, &raw, None), 1.);
        let empty = Matrix::zeros(m, 0);
        let active = if restricted {
            let action = MappedFrozenAction::new(&evaluator, &charts[0], self.reference_energy)?;
            let seed = initial.map(|p| {
                let mut total = p[0].clone();
                add(&mut total, p[1], 1.);
                total
            });
            let result = ActiveSpaceScf {
                overlap: &sr,
                complement: &orthogonalizer.x,
                frozen_orbitals: &empty,
                active_electrons: self.active_electrons.iter().sum(),
                options: self.options,
                initial_hcore: &core,
            }
            .solve(seed.as_ref(), |p| action.evaluate(p))?;
            restricted_spins(result)
        } else {
            CollinearActiveSpaceScf {
                overlap: &sr,
                complements: [&orthogonalizer.x, &orthogonalizer.x],
                frozen_orbitals: [&empty, &empty],
                initial_hcore: [&core, &core],
                active_electrons: self.active_electrons,
                options: self.options,
            }
            .solve(initial, |p| {
                let a = charts[0].lift_density(p[0])?;
                let b = charts[1].lift_density(p[1])?;
                let mut value =
                    evaluator.evaluate_collinear_with_reference([&a, &b], self.reference_energy)?;
                for (s, chart) in charts.iter().enumerate() {
                    value.fock[s] = chart.pullback_operator(&value.fock[s])?;
                }
                Ok(value)
            })?
        };
        let lifted = (0..2)
            .map(|s| charts[s].lift_density(&active.active_density[s]))
            .collect::<Result<Vec<_>>>()?;
        let total: Vec<Matrix> = lifted
            .iter()
            .zip(&bath)
            .map(|(a, b)| {
                let mut p = a.clone();
                add(&mut p, b, 1.);
                p
            })
            .collect();
        let full_evaluation = evaluator
            .evaluate_collinear_with_reference([&lifted[0], &lifted[1]], self.reference_energy)?;
        let zero = Matrix::zeros(n, n);
        let explicit = crate::gradient::assemble_gradient_density(
            &full,
            &fb,
            self.functional,
            Some(&grid),
            [&total[0], &total[1]],
            self.options,
            &zero,
        )?;
        let mut gradient = vec![Vec3::zero(); nr];
        for (g, owner) in explicit.iter().zip(self.full_atom_owners) {
            gradient[*owner] += *g;
        }
        let mut weighted = Matrix::zeros(m, m);
        for spin in 0..2 {
            for k in 0..self.active_electrons[spin] {
                for i in 0..m {
                    for j in 0..m {
                        weighted[(i, j)] += active.orbital_energies[spin][k]
                            * active.active_orbitals[spin][(i, k)]
                            * active.active_orbitals[spin][(j, k)];
                    }
                }
            }
        }
        let pulay = crate::integrals::deriv::overlap_gradient(&rb, &weighted, nr, -1.);
        for (g, p) in gradient.iter_mut().zip(pulay) {
            *g += p;
        }
        let mut union = fb.clone();
        for shell in &rb.shells {
            let mut s = shell.clone();
            s.atom += nf;
            s.sph_off += n;
            s.cart_off += fb.nao_cart;
            union.shells.push(s);
        }
        union.nao_sph += m;
        union.nao_cart += rb.nao_cart;
        union.max_l = union.max_l.max(rb.max_l);
        let derivatives = overlap_derivative_matrices(&union, nf + nr);
        for raw_atom in 0..nr {
            for axis in 0..3 {
                let mut combined = derivatives[nf + raw_atom][axis].clone();
                for (a, owner) in self.full_atom_owners.iter().enumerate() {
                    if *owner == raw_atom {
                        add(&mut combined, &derivatives[a][axis], 1.);
                    }
                }
                let ds = block(&combined, 0, 0, n, n);
                let dc = block(&combined, 0, n, n, m);
                let dr = block(&combined, n, n, m, m);
                let mut chain = AccurateSum::default();
                for spin in 0..2 {
                    let saved = self.frozen_reference[spin];
                    let b = &frozen[spin];
                    let db = if b.cols == 0 {
                        Matrix::zeros(n, 0)
                    } else {
                        let dg = saved.transpose_matmul_seq(&ds.matmul_seq(saved));
                        saved.matmul_seq(&metric_power_direction(&gram[spin], &dg, true)?)
                    };
                    let dt =
                        charts[spin].directional_transform(&sf, &cross, b, [&ds, &dc, &dr, &db])?;
                    let first = dt
                        .matmul_seq(&active.active_density[spin])
                        .matmul_transpose_seq(&charts[spin].transform);
                    let frozen_first = db.matmul_transpose_seq(b);
                    let mut dp = first.clone();
                    add(&mut dp, &first.transpose(), 1.);
                    add(&mut dp, &frozen_first, 1.);
                    add(&mut dp, &frozen_first.transpose(), 1.);
                    for (f, d) in full_evaluation.fock[spin]
                        .as_slice()
                        .iter()
                        .zip(dp.as_slice())
                    {
                        chain.add_product(*f, *d);
                    }
                }
                gradient[raw_atom] += match axis {
                    0 => Vec3::new(chain.value(), 0., 0.),
                    1 => Vec3::new(0., chain.value(), 0.),
                    _ => Vec3::new(0., 0., chain.value()),
                };
            }
        }
        if gradient
            .iter()
            .any(|g| g.to_array().iter().any(|x| !x.is_finite()))
        {
            return invalid("affine frozen geometry produced a nonfinite gradient");
        }
        Ok(CollinearAffineFrozenGeometryResult {
            energy: active.evaluation.energy,
            gradient,
            active_scf: active,
            full_molecule: full,
            chart_metric_error: [charts[0].metric_error, charts[1].metric_error],
            frozen_overlap: [charts[0].frozen_overlap, charts[1].frozen_overlap],
        })
    }
}
