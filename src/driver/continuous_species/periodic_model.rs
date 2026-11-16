// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Experimental neutral all-electron periodic energy and first charge derivatives.
//! Integer element labels, basis, cell, k mesh, quadrature and spin populations
//! stay fixed. This is not a reduced-fragment periodic model. The raw periodic
//! SCF reference is private and deliberately has no ordinary derivative plan.
use super::periodic_charge::NeutralEwaldChart;
use super::{ChargeChart, Fingerprint, SpeciesFunctional};
use crate::linalg::{Matrix, Orthogonalizer};
use crate::math::Vec3;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::features::PeriodicBoundary;
use crate::periodic::integrals::bloch;
use crate::periodic::nuclear::nuclear_image_blocks;
use crate::periodic::occupations::OccupationScheme;
use crate::periodic::scf::{
    build_fields, run_scf_iterations, validate_solver_options, PeriodicScfOptions,
    PeriodicScfResult, ScfIterationInput,
};
use crate::periodic::xc::{AoDerivativeOrder, PeriodicXcGrid};
use crate::periodic::Cell;
use crate::scf::{Guess, Reference};
use crate::xc::{resolve, FunctionalSpec, ResolvedFunctional};
use crate::{DftError, Result};

fn invalid<T>(message: impl Into<String>) -> Result<T> {
    Err(DftError::InvalidInput(message.into()))
}
fn unsupported<T>(feature: &str) -> Result<T> {
    Err(DftError::Unsupported{feature:feature.into(),missing:"this experimental periodic chart supplies only neutral fixed-cell all-electron energy and analytic first charge derivatives; additional electronic response and geometry/cell derivative contracts are not implemented".into()})
}
fn max(a: &Matrix) -> f64 {
    a.as_slice().iter().map(|x| x.abs()).fold(0., f64::max)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PeriodicSpeciesFingerprints {
    pub cell: u64,
    pub model: u64,
    pub chart: u64,
    pub grid: u64,
    pub k_mesh: u64,
    pub spin: u64,
}

#[derive(Clone, Debug)]
pub struct PeriodicSpeciesDiagnostics {
    /// Fresh metric-orthonormal commutator max over all spin/k matrices, Eh.
    pub orbital_residual: f64,
    /// Max |(S^1/2 D S^1/2)^2-S^1/2 D S^1/2|, per-spin projector.
    pub projector_defect: f64,
    pub particle_number_error: f64,
    /// Uniform-k mean Tr(D_spin S), electrons per primitive cell.
    pub spin_electron_traces: [f64; 2],
    pub hermiticity_defect: f64,
    /// Indirect fixed-band frontier gap per spin; None for empty/full channels.
    pub spin_indirect_gaps: [Option<f64>; 2],
    pub ewald_direct_energy_defect: f64,
    pub raw_ewald_matrix_asymmetry: f64,
}

pub struct PeriodicSpeciesState {
    // No Deref/AsRef/raw getter: ordinary force/DFPT routines must not consume it.
    raw: PeriodicScfResult,
    coordinates: Vec<f64>,
    charges: Vec<f64>,
    physical_total_charge: f64,
    fingerprints: PeriodicSpeciesFingerprints,
    diagnostics: PeriodicSpeciesDiagnostics,
}
impl PeriodicSpeciesState {
    pub fn energy(&self) -> f64 {
        self.raw.energy
    }
    pub fn electronic_energy(&self) -> f64 {
        self.raw.e_elec
    }
    pub fn nuclear_energy(&self) -> f64 {
        self.raw.e_nuc
    }
    pub fn coordinates(&self) -> &[f64] {
        &self.coordinates
    }
    pub fn physical_charges(&self) -> &[f64] {
        &self.charges
    }
    pub fn physical_total_charge(&self) -> f64 {
        self.physical_total_charge
    }
    pub fn spin_electron_counts(&self) -> [usize; 2] {
        [self.raw.n_alpha, self.raw.n_beta]
    }
    pub fn unrestricted(&self) -> bool {
        self.raw.unrestricted
    }
    pub fn iterations(&self) -> usize {
        self.raw.iterations
    }
    pub fn fingerprints(&self) -> &PeriodicSpeciesFingerprints {
        &self.fingerprints
    }
    pub fn diagnostics(&self) -> &PeriodicSpeciesDiagnostics {
        &self.diagnostics
    }
    pub fn energy_unit(&self) -> &'static str {
        "hartree/cell"
    }
    pub fn scope(&self) -> &'static str {
        "experimental-neutral-all-electron-periodic-energy-first-charge-derivatives"
    }
}

pub struct PeriodicSpeciesModel<'a> {
    cell: &'a Cell,
    chart: NeutralEwaldChart,
    options: PeriodicScfOptions,
    functional: ResolvedFunctional,
    spins: [usize; 2],
    restricted: bool,
    k_points: Vec<Vec3>,
    overlap: Vec<BlochMatrix>,
    h_reference: Vec<BlochMatrix>,
    parameter_operators: Vec<Vec<BlochMatrix>>,
    grid: PeriodicXcGrid,
    fingerprints: PeriodicSpeciesFingerprints,
}

impl<'a> PeriodicSpeciesModel<'a> {
    /// Explicit fixed integer spin populations never come from element labels.
    /// A shared immutable Cell supplies only fixed AO/lattice/Coulomb caches.
    pub fn new(
        cell: &'a Cell,
        chart: ChargeChart,
        spins: [usize; 2],
        functional: SpeciesFunctional,
        options: PeriodicScfOptions,
    ) -> Result<Self> {
        validate_solver_options(&options)?;
        crate::periodic::features::PeriodicCapabilities::current()
            .validate_boundary(options.boundary)?;
        if options.boundary != PeriodicBoundary::FULL_3D
            || !matches!(options.occupation, OccupationScheme::Insulator)
            || !matches!(options.guess, Guess::Core)
            || options.vv10.is_some()
            || options.skala.is_some()
            || options.pair_dispersion_from_caller
            || !cell.basis().ecps.is_empty()
            || cell.basis().n_core_electrons != 0
        {
            return unsupported("periodic continuous charges with non-3D boundaries, noninteger occupations, non-Core guesses, ECP or additional nonlocal/dispersion models");
        }
        let ne = spins[0].checked_add(spins[1]).ok_or_else(|| {
            DftError::InvalidInput("fixed periodic spin populations overflow".into())
        })?;
        if ne == 0 || spins[0] < spins[1] || spins.iter().any(|&n| n > cell.nao()) {
            return invalid(
                "fixed periodic alpha/beta populations must fit the AO template with alpha>=beta",
            );
        }
        let restricted = match options.reference {
            Reference::Auto => spins[0] == spins[1],
            Reference::Rhf => {
                if spins[0] != spins[1] {
                    return invalid("RKS periodic chart requires equal fixed spin populations");
                }
                true
            }
            Reference::Uhf => false,
        };
        let positions: Vec<_> = cell.molecule().atoms.iter().map(|a| a.position).collect();
        let labels: Vec<_> = cell.molecule().atoms.iter().map(|a| a.z).collect();
        let classical = NeutralEwaldChart::new(
            cell.lattice().clone(),
            positions,
            labels,
            ne,
            chart,
            cell.ewald_params(),
        )?;
        let functional = resolve(&FunctionalSpec::Combined(functional.name().into()))?;
        let k_points = cell.k_points(options.k_mesh);
        if k_points.is_empty() {
            return invalid("periodic chart requires a nonempty fixed k mesh");
        }
        let overlap: Vec<_> = k_points
            .iter()
            .map(|&k| BlochMatrix::from_pair(cell.overlap(k)))
            .collect();
        // Rank changes are not allowed within this fixed AO chart.
        for s in &overlap {
            let orth = Orthogonalizer::canonical(&s.real_embedding(), options.orth_threshold)?;
            if orth.n_kept != 2 * cell.nao() {
                return invalid("periodic chart refuses an overlap rank reduction");
            }
        }
        // Ordinary cached Cell operators remain unchanged. Every real-charge
        // correction is local to this model, including the q=0 reference delta.
        let mut h_reference: Vec<_> = k_points
            .iter()
            .map(|&k| BlochMatrix::from_pair(cell.core_hamiltonian(k)))
            .collect();
        let delta: Vec<_> = classical
            .chart()
            .reference_charges
            .iter()
            .zip(classical.atomic_numbers())
            .map(|(z, &label)| z - f64::from(label))
            .collect();
        let make_operator = |charges: &[f64]| {
            let blocks = nuclear_image_blocks(
                cell.basis(),
                cell.molecule(),
                charges,
                cell.lattice(),
                cell.overlap_cutoff(),
                cell.ewald_params(),
            );
            k_points
                .iter()
                .map(|&k| BlochMatrix::from_pair(bloch(&blocks, k)))
                .collect::<Vec<_>>()
        };
        if delta.iter().any(|&x| x != 0.) {
            for (h, d) in h_reference.iter_mut().zip(make_operator(&delta)) {
                h.add_scaled(&d, 1.);
            }
        }
        let parameter_operators = classical
            .chart()
            .parameters
            .iter()
            .map(|p| make_operator(&p.charge_direction))
            .collect();
        let grid = PeriodicXcGrid::build(
            cell,
            &k_points,
            &options.grid,
            options.grid_shell,
            options.xc_ao_cutoff.unwrap_or(cell.overlap_cutoff()),
            AoDerivativeOrder::for_scf(functional.family),
        )?;
        let fingerprints = make_fingerprints(
            cell,
            &classical,
            &options,
            &functional,
            &k_points,
            &grid,
            spins,
            restricted,
        );
        Ok(Self {
            cell,
            chart: classical,
            options,
            functional,
            spins,
            restricted,
            k_points,
            overlap,
            h_reference,
            parameter_operators,
            grid,
            fingerprints,
        })
    }
    pub fn chart(&self) -> &ChargeChart {
        self.chart.chart()
    }
    pub fn atomic_numbers(&self) -> &[u8] {
        self.chart.atomic_numbers()
    }
    pub fn fixed_electrons(&self) -> usize {
        self.chart.fixed_electrons()
    }
    pub fn k_points(&self) -> &[Vec3] {
        &self.k_points
    }
    pub fn grid_point_count(&self) -> usize {
        self.grid.points.len()
    }
    pub fn ao_count(&self) -> usize {
        self.cell.nao()
    }
    pub fn fingerprints(&self) -> &PeriodicSpeciesFingerprints {
        &self.fingerprints
    }
    pub fn energy_gradient_units(&self) -> Vec<String> {
        self.chart()
            .parameters
            .iter()
            .map(|p| format!("hartree/cell/({})", p.unit))
            .collect()
    }
    pub fn classical_scope(&self) -> &'static str {
        self.chart.scope()
    }

    pub fn evaluate(&self, q: &[f64]) -> Result<PeriodicSpeciesState> {
        crate::linalg::ensure_parallelism();
        let classical = self.chart.evaluate(q)?;
        // This is only a consistency check of the source convention. No
        // classical Hessian is exposed as a total electronic Hessian.
        if classical.parameter_hessian.rows != q.len()
            || classical.parameter_hessian.cols != q.len()
            || classical
                .parameter_hessian
                .as_slice()
                .iter()
                .any(|x| !x.is_finite())
        {
            return invalid("invalid internal Ewald chart Hessian");
        }
        let direct = self.chart.direct_energy(q)?;
        let ewald_defect = (direct - classical.energy).abs();
        if ewald_defect > 1e-10 * (1. + direct.abs()) {
            return invalid("cached Ewald chart and direct source convention disagree");
        }
        let mut h = self.h_reference.clone();
        for (&value, operators) in q.iter().zip(&self.parameter_operators) {
            for (hk, ha) in h.iter_mut().zip(operators) {
                hk.add_scaled(ha, value);
            }
        }
        let raw = run_scf_iterations(
            ScfIterationInput {
                k_points: self.k_points.clone(),
                overlap: self.overlap.clone(),
                h_core: h.clone(),
                guess_focks: h.clone(),
                n_alpha: self.spins[0],
                n_beta: self.spins[1],
                restricted: self.restricted,
                enforce_orbital_conjugation: true,
                e_nuc: classical.energy,
                derivative_plan: None,
            },
            &self.options,
            |density| {
                build_fields(
                    self.cell,
                    &self.functional,
                    &self.k_points,
                    density,
                    &h,
                    Some(&self.grid),
                    self.options
                        .exchange_cutoff
                        .unwrap_or(self.cell.overlap_cutoff()),
                    self.restricted,
                )
            },
        )?;
        let mut diagnostics = diagnose(&raw, &self.options)?;
        diagnostics.ewald_direct_energy_defect = ewald_defect;
        diagnostics.raw_ewald_matrix_asymmetry = classical.raw_matrix_asymmetry;
        Ok(PeriodicSpeciesState {
            raw,
            coordinates: q.to_vec(),
            charges: classical.physical_charges,
            physical_total_charge: classical.physical_total_charge,
            fingerprints: self.fingerprints.clone(),
            diagnostics,
        })
    }
    fn check_state(&self, state: &PeriodicSpeciesState) -> Result<()> {
        if state.fingerprints != self.fingerprints
            || state.raw.derivative_plan.is_some()
            || state.spin_electron_counts() != self.spins
            || state.raw.unrestricted == self.restricted
        {
            return invalid(
                "periodic state belongs to a different cell/model/chart/grid/k/spin contract",
            );
        }
        Ok(())
    }
    /// Hellmann-Feynman derivative at fixed cell, AO, grid, occupation and Ne.
    /// Every spin density is per-spin, so both channels enter once (no extra2).
    pub fn energy_gradient(&self, state: &PeriodicSpeciesState) -> Result<Vec<f64>> {
        self.check_state(state)?;
        let classical = self.chart.evaluate(&state.coordinates)?;
        let inv_nk = 1. / self.k_points.len() as f64;
        let gradient = self
            .parameter_operators
            .iter()
            .zip(&classical.parameter_gradient)
            .map(|(ha, &nuclear)| {
                nuclear
                    + inv_nk
                        * (0..2)
                            .map(|spin| {
                                state.raw.density[spin]
                                    .iter()
                                    .zip(ha)
                                    .map(|(d, h)| d.trace_product(h))
                                    .sum::<f64>()
                            })
                            .sum::<f64>()
            })
            .collect::<Vec<_>>();
        if gradient.iter().any(|x| !x.is_finite()) {
            return invalid("nonfinite periodic charge gradient");
        }
        Ok(gradient)
    }
    pub fn energy_hessian(&self, state: &PeriodicSpeciesState) -> Result<Matrix> {
        self.check_state(state)?;
        unsupported("total periodic charge Hessian")
    }
    pub fn nuclear_forces(&self, state: &PeriodicSpeciesState) -> Result<Vec<Vec3>> {
        self.check_state(state)?;
        unsupported("continuous-charge periodic nuclear forces")
    }
    pub fn stress(&self, state: &PeriodicSpeciesState) -> Result<[[f64; 3]; 3]> {
        self.check_state(state)?;
        unsupported("continuous-charge periodic cell stress")
    }
    pub fn static_polarizability(&self, state: &PeriodicSpeciesState) -> Result<[[f64; 3]; 3]> {
        self.check_state(state)?;
        unsupported("continuous-charge periodic electric response")
    }
}

fn make_fingerprints(
    cell: &Cell,
    chart: &NeutralEwaldChart,
    options: &PeriodicScfOptions,
    functional: &ResolvedFunctional,
    k: &[Vec3],
    grid: &PeriodicXcGrid,
    spins: [usize; 2],
    restricted: bool,
) -> PeriodicSpeciesFingerprints {
    let mut c = Fingerprint::new();
    c.string(&format!("{:?}", chart.chart()));
    for x in chart.charge_matrix().as_slice() {
        c.float(*x);
    }
    let mut g = Fingerprint::new();
    for (&p, &w) in grid.points.iter().zip(&grid.weights) {
        g.vector(p);
        g.float(w);
    }
    let mut km = Fingerprint::new();
    for &p in k {
        km.vector(p);
    }
    for n in options.k_mesh {
        km.number(n);
    }
    let mut spin = Fingerprint::new();
    spin.number(spins[0]);
    spin.number(spins[1]);
    spin.number(usize::from(restricted));
    let mut model = Fingerprint::new();
    model.string("periodic-fixed-charge-energy-gradient-v1");
    model.string(&format!("{options:?}|{}", functional.description));
    model.bytes(&cell.identity().to_le_bytes());
    model.bytes(&c.0.to_le_bytes());
    model.bytes(&g.0.to_le_bytes());
    model.bytes(&km.0.to_le_bytes());
    model.bytes(&spin.0.to_le_bytes());
    PeriodicSpeciesFingerprints {
        cell: cell.identity(),
        model: model.0,
        chart: c.0,
        grid: g.0,
        k_mesh: km.0,
        spin: spin.0,
    }
}

fn diagnose(
    raw: &PeriodicScfResult,
    options: &PeriodicScfOptions,
) -> Result<PeriodicSpeciesDiagnostics> {
    if !raw.converged
        || !raw.energy.is_finite()
        || raw.k_points.is_empty()
        || raw.entropy_term != 0.
        || raw.derivative_plan.is_some()
    {
        return invalid("periodic chart requires a converged integer reference with no ordinary derivative plan");
    }
    let nk = raw.k_points.len();
    let n = raw.overlap[0].rows();
    let mut d = PeriodicSpeciesDiagnostics {
        orbital_residual: 0.,
        projector_defect: 0.,
        particle_number_error: 0.,
        spin_electron_traces: [0.; 2],
        hermiticity_defect: 0.,
        spin_indirect_gaps: [None; 2],
        ewald_direct_energy_defect: 0.,
        raw_ewald_matrix_asymmetry: 0.,
    };
    for spin in 0..2 {
        let count = [raw.n_alpha, raw.n_beta][spin];
        if count > 0 && count < n {
            let occupied = raw.mo_energy[spin]
                .iter()
                .map(|e| e[count - 1])
                .fold(f64::NEG_INFINITY, f64::max);
            let virtual_min = raw.mo_energy[spin]
                .iter()
                .map(|e| e[count])
                .fold(f64::INFINITY, f64::min);
            let gap = virtual_min - occupied;
            if !gap.is_finite() || gap <= 0. {
                return unsupported("noninsulating periodic charge state");
            }
            d.spin_indirect_gaps[spin] = Some(gap);
        }
        for k in 0..nk {
            let p = &raw.density[spin][k];
            let f = &raw.fock[spin][k];
            let s = &raw.overlap[k];
            if [p, f, s].iter().any(|m| {
                m.rows() != n
                    || m.cols() != n
                    || m.re
                        .as_slice()
                        .iter()
                        .chain(m.im.as_slice())
                        .any(|x| !x.is_finite())
            }) {
                return invalid("invalid periodic spin/k electronic matrix");
            }
            d.hermiticity_defect = d
                .hermiticity_defect
                .max(p.hermiticity_error())
                .max(f.hermiticity_error())
                .max(s.hermiticity_error());
            let number = p.trace_product(s);
            d.spin_electron_traces[spin] += number / nk as f64;
            d.particle_number_error = d.particle_number_error.max((number - count as f64).abs());
            let pr = p.real_embedding();
            let fr = f.real_embedding();
            let sr = s.real_embedding();
            let orth = Orthogonalizer::canonical(&sr, options.orth_threshold)?;
            if orth.n_kept != 2 * n {
                return invalid("periodic reference changed its fixed AO rank");
            }
            let mut comm = fr.matmul_seq(&pr.matmul_seq(&sr));
            let other = sr.matmul_seq(&pr.matmul_seq(&fr));
            for (x, y) in comm.as_mut_slice().iter_mut().zip(other.as_slice()) {
                *x -= y;
            }
            d.orbital_residual = d
                .orbital_residual
                .max(max(&orth.x.transpose_matmul_seq(&comm.matmul_seq(&orth.x))));
            let sx = sr.matmul_seq(&orth.x);
            let density = sx.transpose_matmul_seq(&pr.matmul_seq(&sx));
            let mut defect = density.matmul_seq(&density);
            for (x, y) in defect.as_mut_slice().iter_mut().zip(density.as_slice()) {
                *x -= y;
            }
            d.projector_defect = d.projector_defect.max(max(&defect));
            if raw.occupations[spin][k]
                .iter()
                .enumerate()
                .any(|(band, &v)| (v - if band < count { 1. } else { 0. }).abs() > 1e-12)
            {
                return unsupported("fractional or changing periodic band occupations");
            }
        }
    }
    if d.orbital_residual > options.conv_tol_grad
        || d.particle_number_error > 1e-8
        || d.projector_defect > 1e-8
        || d.hermiticity_defect > 1e-9
    {
        return invalid(format!(
            "periodic fresh-reference diagnostics failed: {d:?}"
        ));
    }
    Ok(d)
}
