// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Magnetic Brillouin-zone quadrature with intersector Coulomb exchange.
//! A Gaussian auxiliary is subtracted at every nonzero transfer and added
//! analytically at q=0, without modifying physical denominators. Raw punctured
//! quadrature remains selectable. Both require independent k-mesh convergence.

use super::*;

/// Gaussian auxiliary correction, in Hartree, for an equal-weight magnetic
/// Bloch mesh. The combined transfers are the reciprocal mesh of a cell
/// enlarged by each mesh count: chi=2 eta/sqrt(pi)-sum_q v_q exp[-q^2/(4 eta^2)].
/// The zero-transfer limit is analytic; no physical denominator is floored.
pub fn gaussian_singularity_correction(
    geometry: &PeriodicGeometry,
    mesh: [usize; 3],
    eta: f64,
    cutoff: f64,
) -> Result<f64> {
    if geometry.dimension() != 3
        || mesh.contains(&0)
        || !eta.is_finite()
        || eta < 0.0
        || !cutoff.is_finite()
        || cutoff <= 0.0
    {
        return Err(DftError::InvalidInput(
            "Gaussian exchange auxiliary requires a 3D cell, positive mesh/cutoff \
            and finite nonnegative inverse-length width"
                .into(),
        ));
    }
    if eta == 0.0 {
        return Ok(0.0);
    }
    let a = geometry.lattice().a;
    let enlarged = PeriodicGeometry::new(
        super::super::Lattice::new(
            a[0] * mesh[0] as f64,
            a[1] * mesh[1] as f64,
            a[2] * mesh[2] as f64,
        ),
        super::super::features::PeriodicBoundary::FULL_3D,
    )?;
    let sum = enlarged
        .reciprocal_within(cutoff)?
        .iter()
        .map(|q| {
            4.0 * PI / (enlarged.measure() * q.norm2()) * (-q.norm2() / (4.0 * eta * eta)).exp()
        })
        .sum::<f64>();
    Ok(2.0 * eta / PI.sqrt() - sum)
}

/// Two-dimensional Gaussian auxiliary with the continuous normal direction
/// integrated analytically. `omega=None` selects full Coulomb; `Some(omega)`
/// includes erfc(|q_parallel|/(2 omega)) in the long-range auxiliary.
/// The open-direction mesh count must be one. Zero width selects raw omission.
pub fn slab_gaussian_singularity_correction(
    geometry: &PeriodicGeometry,
    mesh: [usize; 3],
    eta: f64,
    cutoff: f64,
    omega: Option<f64>,
) -> Result<f64> {
    if geometry.dimension() != 2
        || mesh.contains(&0)
        || (0..3).any(|a| !geometry.boundary().axes[a] && mesh[a] != 1)
        || !eta.is_finite()
        || eta < 0.0
        || !cutoff.is_finite()
        || cutoff <= 0.0
        || omega.is_some_and(|w| !w.is_finite() || w < 0.0)
    {
        return Err(DftError::InvalidInput("slab exchange auxiliary requires a 2D physical mesh, positive cutoff and finite nonnegative widths".into()));
    }
    if eta == 0.0 || omega == Some(0.0) {
        return Ok(0.0);
    }
    let a = geometry.lattice().a;
    let enlarged = PeriodicGeometry::new(
        super::super::Lattice::new(
            a[0] * mesh[0] as f64,
            a[1] * mesh[1] as f64,
            a[2] * mesh[2] as f64,
        ),
        geometry.boundary(),
    )?;
    let integral = omega.map_or(PI.sqrt() * eta, |w| {
        2.0 * eta / PI.sqrt() * (w / eta).atan()
    });
    let sum = enlarged
        .reciprocal_within(cutoff)?
        .iter()
        .map(|q| {
            let factor = omega.map_or(1.0, |w| libm::erfc(q.norm() / (2.0 * w)));
            2.0 * PI / (enlarged.measure() * q.norm())
                * (-q.norm2() / (4.0 * eta * eta)).exp()
                * factor
        })
        .sum::<f64>();
    Ok(integral - sum)
}

struct TransferMode {
    vertex: BlochMatrix,
    coulomb: f64,
    long_range: f64,
    fine: bool,
    coarse: bool,
}
/// Continuum magnetic Bloch SCF on an equal-weight Cartesian k mesh.
/// The one-sector finite-cell calculation remains an explicit separate model.
pub struct MagneticBlochBackend<'a> {
    pub cell: MagneticCell,
    pub k_points: Vec<Vec3>,
    pub image_operator_defect: f64,
    pub crystal: bool,
    sectors: Vec<MagneticDftBackend<'a>>,
    overlap: Vec<BlochMatrix>,
    initial: Vec<BlochMatrix>,
    transfers: Vec<Vec<Vec<TransferMode>>>,
    /// Exact contraction for a one-orbital spatial space. Four spin blocks
    /// share the same scalar exchange integral; do not repeat its Fourier
    /// sum at every SCF iteration. This changes no kernel or denominator.
    scalar_exchange: Vec<Vec<[[f64; 2]; 2]>>,
    scalar_hartree: Vec<Vec<[Complex; 2]>>,
    grid: Option<Mutex<xc::Cache>>,
    corrections: [[f64; 2]; 2],
}

impl<'a> MagneticBlochBackend<'a> {
    /// Finite-cell Berry dipole of the electric-biased spinor reference.
    pub fn electric_dipole(&self, density: &[BlochMatrix]) -> Result<Option<[f64; 3]>> {
        if self.crystal {
            return Ok(None);
        }
        self.sectors[0].electric_dipole(density)
    }
    /// Relaxed nuclear energy on the explicit finite-Gamma magnetic model.
    /// Intersector crystal response retains its independent provider contract.
    pub fn nuclear_energy_taylor(
        &self,
        reference: &crate::periodic::spinor_scf::SpinorScfReport,
        direction: &[[f64; 3]],
        order: usize,
        config: &crate::response::ResponseSolverConfig,
    ) -> Result<MagneticNuclearTaylor> {
        if self.crystal || self.sectors.len() != 1 {
            return Err(DftError::Unsupported {
                feature: "magnetic crystal nuclear response".into(),
                missing: "the intersector moving-vertex and magnetic-BZ nuclear response provider"
                    .into(),
            });
        }
        self.sectors[0].nuclear_energy_taylor(reference, direction, order, config)
    }

    pub fn new(
        mol: &Molecule,
        basis: &MolBasis,
        cell: MagneticCell,
        functional: &'a ResolvedFunctional,
        options: MagneticScfOptions,
        spin_operator: Option<BlochMatrix>,
        mesh: [usize; 3],
        crystal: bool,
    ) -> Result<Self> {
        if crystal && options.electric_field.iter().any(|x| *x != 0.0) {
            return Err(DftError::Unsupported {
                feature: "finite electric bias on magnetic crystal integration".into(),
                missing: "the coupled occupied-link Berry strings across magnetic Bloch sectors"
                    .into(),
            });
        }
        if mesh.contains(&0)
            || (!crystal && mesh != [1; 3])
            || cell.bloch_wave != Vec3::zero()
            || !options.exchange_auxiliary_width.is_finite()
            || options.exchange_auxiliary_width < 0.0
        {
            return Err(DftError::InvalidInput(
                "magnetic crystal meshes must be positive; the finite torus has one \
                Gamma sector and the supplied cell must have zero Bloch wave"
                    .into(),
            ));
        }
        let k_points = cell.geometry.k_points(mesh)?;
        let mut sectors = k_points
            .iter()
            .map(|&k| {
                MagneticDftBackend::new(
                    mol,
                    basis,
                    cell.with_bloch_wave(k)?,
                    functional,
                    options.clone(),
                    spin_operator.clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        // A shared charge-density quadrature requires exactly matching normal
        // nodes in every sector, including its separate coarse rule.
        if crystal && cell.geometry.dimension() == 2 {
            loop {
                let order = sectors
                    .iter()
                    .map(|s| s.options.open_quadrature_order)
                    .max()
                    .unwrap();
                if sectors
                    .iter()
                    .all(|s| s.options.open_quadrature_order == order)
                {
                    break;
                }
                for sector in &mut sectors {
                    if sector.options.open_quadrature_order < order {
                        let mut history = sector.open_refinement_history.clone();
                        let mut refined = options.clone();
                        refined.open_quadrature_order = order;
                        *sector = MagneticDftBackend::new(
                            mol,
                            basis,
                            sector.cell.clone(),
                            functional,
                            refined,
                            spin_operator.clone(),
                        )?;
                        history.extend_from_slice(&sector.open_refinement_history);
                        sector.open_refinement_history = history;
                    }
                }
            }
        }
        let overlap = sectors.iter().map(|s| s.overlap[0].clone()).collect();
        let initial = sectors.iter().map(|s| s.initial[0].clone()).collect();
        let image_operator_defect = sectors
            .iter()
            .map(|s| s.image_operator_defect)
            .fold(0.0_f64, f64::max);
        let transfers = if crystal && (functional.hybrid.cx != 0.0 || functional.hybrid.clr != 0.0)
        {
            k_points
                .par_iter()
                .map(|&left| {
                    k_points
                        .iter()
                        .map(|&right| {
                            if cell.geometry.dimension() == 2 {
                                let mut result = Vec::new();
                                for (coarse, factor, order) in [
                                    (false, 1.0, sectors[0].options.open_quadrature_order),
                                    (true, 0.8, sectors[0].options.open_quadrature_order / 2),
                                ] {
                                    for m in super::slab::transfer_modes(
                                        &cell,
                                        basis,
                                        functional,
                                        &[],
                                        options.reciprocal_cutoff * factor,
                                        options.basis_cutoff,
                                        order,
                                        left,
                                        right,
                                    )? {
                                        result.push(TransferMode {
                                            vertex: m.density,
                                            coulomb: m.coulomb,
                                            long_range: m.long_range,
                                            fine: !coarse,
                                            coarse,
                                        });
                                    }
                                }
                                return Ok(result);
                            }
                            let shift = left - right;
                            let mut waves = cell
                                .geometry
                                .reciprocal_within(options.reciprocal_cutoff + shift.norm())?;
                            waves.push(Vec3::zero());
                            waves
                                .into_iter()
                                .map(|g| g + shift)
                                .filter(|q| {
                                    q.norm2() > 0.0 && q.norm() <= options.reciprocal_cutoff
                                })
                                .map(|q| {
                                    let coulomb = 4.0 * PI / (cell.geometry.measure() * q.norm2());
                                    let long_range = if functional.hybrid.omega == 0.0 {
                                        0.0
                                    } else {
                                        coulomb
                                            * (-q.norm2() / (4.0 * functional.hybrid.omega.powi(2)))
                                                .exp()
                                    };
                                    Ok(TransferMode {
                                        vertex: cell.fourier_transfer(
                                            basis,
                                            options.basis_cutoff,
                                            left,
                                            right,
                                            q,
                                        )?,
                                        coulomb,
                                        long_range,
                                        fine: true,
                                        coarse: q.norm() <= 0.8 * options.reciprocal_cutoff,
                                    })
                                })
                                .collect::<Result<Vec<_>>>()
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let scalar_exchange = if basis.nao_sph == 1 {
            transfers
                .iter()
                .map(|pairs| {
                    pairs
                        .iter()
                        .map(|modes| {
                            let mut sums = [[0.0; 2]; 2];
                            for mode in modes {
                                let square = z(&mode.vertex, 0, 0).abs().powi(2);
                                if mode.fine {
                                    sums[0][0] += mode.coulomb * square;
                                    sums[0][1] += mode.long_range * square;
                                }
                                if mode.coarse {
                                    sums[1][0] += mode.coulomb * square;
                                    sums[1][1] += mode.long_range * square;
                                }
                            }
                            sums
                        })
                        .collect()
                })
                .collect()
        } else {
            Vec::new()
        };
        let scalar_hartree = if basis.nao_sph == 1 && crystal {
            sectors
                .iter()
                .map(|left| {
                    sectors
                        .iter()
                        .map(|right| {
                            let mut sums = [Complex::ZERO; 2];
                            for coarse in [false, true] {
                                for (a, b) in Self::hartree_modes(left, coarse)
                                    .iter()
                                    .zip(Self::hartree_modes(right, coarse))
                                {
                                    if coarse
                                        && left.coarse_modes.is_none()
                                        && a.wave.norm() > 0.8 * options.reciprocal_cutoff
                                    {
                                        continue;
                                    }
                                    sums[usize::from(coarse)] += z(&a.density, 0, 0)
                                        * z(&b.density, 0, 0).conj()
                                        * a.coulomb;
                                }
                            }
                            sums
                        })
                        .collect()
                })
                .collect()
        } else {
            Vec::new()
        };
        let grid = if crystal && !functional.parts.is_empty() {
            for sector in &mut sectors {
                sector.grid = None;
            }
            xc::Cache::new(functional, options.grid_order, mol)
        } else {
            None
        };
        let mut corrections = [[0.0; 2]; 2];
        if crystal && (functional.hybrid.cx != 0.0 || functional.hybrid.clr != 0.0) {
            let eta = options.exchange_auxiliary_width;
            let omega = functional.hybrid.omega;
            let eta_lr = if eta == 0.0 || omega == 0.0 {
                0.0
            } else {
                eta.min(omega) / (1.0 + (eta.min(omega) / eta.max(omega)).powi(2)).sqrt()
            };
            for (i, factor) in [1.0, 0.8].into_iter().enumerate() {
                if cell.geometry.dimension() == 2 {
                    corrections[i] = [
                        slab_gaussian_singularity_correction(
                            &cell.geometry,
                            mesh,
                            eta,
                            options.reciprocal_cutoff * factor,
                            None,
                        )?,
                        slab_gaussian_singularity_correction(
                            &cell.geometry,
                            mesh,
                            eta,
                            options.reciprocal_cutoff * factor,
                            Some(omega),
                        )?,
                    ];
                    continue;
                }
                corrections[i] = [
                    gaussian_singularity_correction(
                        &cell.geometry,
                        mesh,
                        eta,
                        options.reciprocal_cutoff * factor,
                    )?,
                    gaussian_singularity_correction(
                        &cell.geometry,
                        mesh,
                        eta_lr,
                        options.reciprocal_cutoff * factor,
                    )?,
                ];
            }
        }
        Ok(Self {
            cell,
            k_points,
            image_operator_defect,
            crystal,
            sectors,
            overlap,
            initial,
            transfers,
            scalar_exchange,
            scalar_hartree,
            grid,
            corrections,
        })
    }
    fn hartree_modes<'s>(sector: &'s MagneticDftBackend<'_>, coarse: bool) -> &'s [FourierMode] {
        if coarse {
            sector.coarse_modes.as_deref().unwrap_or(&sector.modes)
        } else {
            &sector.modes
        }
    }
    pub fn open_quadrature_order(&self) -> usize {
        self.sectors
            .iter()
            .map(|s| s.open_quadrature_order())
            .max()
            .unwrap_or(0)
    }
    pub fn open_refinement_history(&self) -> Vec<[f64; 4]> {
        self.sectors
            .iter()
            .enumerate()
            .flat_map(|(k, s)| {
                s.open_refinement_history()
                    .iter()
                    .map(move |r| [k as f64, r[0], r[1], r[2]])
            })
            .collect()
    }
    pub fn reciprocal_modes(&self) -> usize {
        self.sectors[0].reciprocal_modes()
    }
    pub fn exchange_transfer_modes(&self) -> usize {
        self.transfers
            .iter()
            .flatten()
            .flatten()
            .filter(|m| m.fine)
            .count()
    }
    pub fn exchange_singularity_correction(&self) -> [f64; 2] {
        self.corrections[0]
    }
    pub fn exchange_auxiliary_width(&self) -> f64 {
        self.sectors[0].options.exchange_auxiliary_width
    }
    pub fn grid_order(&self) -> usize {
        self.grid.as_ref().map_or_else(
            || self.sectors[0].grid_order(),
            |c| c.lock().expect("magnetic Bloch grid cache poisoned").order,
        )
    }
    pub fn grid_refinement_history(&self) -> Vec<[f64; 4]> {
        self.grid.as_ref().map_or_else(
            || self.sectors[0].grid_refinement_history(),
            |c| {
                c.lock()
                    .expect("magnetic Bloch grid cache poisoned")
                    .measurements
                    .clone()
            },
        )
    }
    pub fn grid_diagnostics(&self, density: &[BlochMatrix]) -> Result<[f64; 3]> {
        if !self.crystal {
            return self.sectors[0].grid_diagnostics(&density[0]);
        }
        if let Some(grid) = &self.grid {
            if let Some(spatial) = grid
                .lock()
                .map_err(|_| DftError::InvalidInput("magnetic Bloch grid cache poisoned".into()))?
                .retained_spatial(density)
            {
                return Ok(spatial);
            }
        }
        Ok(self.local_density(density)?.spatial)
    }
    pub fn set_external_spin_operator(&mut self, operator: BlochMatrix) -> Result<()> {
        for s in &mut self.sectors {
            s.set_external_spin_operator(operator.clone())?;
        }
        self.initial = self.sectors.iter().map(|s| s.initial[0].clone()).collect();
        Ok(())
    }
    pub fn sample_particle_current(
        &self,
        density: &[BlochMatrix],
        points: &[Vec3],
    ) -> Result<MagneticCurrentSample> {
        self.validate_density(density)?;
        let mut samples = self
            .sectors
            .iter()
            .zip(density)
            .map(|(s, p)| s.sample_particle_current(p, points))
            .collect::<Result<Vec<_>>>()?;
        let mut out = samples.remove(0);
        for s in samples {
            for g in 0..points.len() {
                out.number_density[g] += s.number_density[g];
                for axis in 0..3 {
                    out.spin_density[g][axis] += s.spin_density[g][axis];
                    out.kinetic_current[g][axis] += s.kinetic_current[g][axis];
                    out.spin_magnetization_current[g][axis] +=
                        s.spin_magnetization_current[g][axis];
                    out.nuclear_soc_current[g][axis] += s.nuclear_soc_current[g][axis];
                    out.particle_current[g][axis] += s.particle_current[g][axis];
                }
            }
        }
        let weight = 1.0 / self.sectors.len() as f64;
        for g in 0..points.len() {
            out.number_density[g] *= weight;
            for axis in 0..3 {
                out.spin_density[g][axis] *= weight;
                out.kinetic_current[g][axis] *= weight;
                out.spin_magnetization_current[g][axis] *= weight;
                out.nuclear_soc_current[g][axis] *= weight;
                out.particle_current[g][axis] *= weight;
            }
        }
        Ok(out)
    }
    fn validate_density(&self, density: &[BlochMatrix]) -> Result<()> {
        let n = self.overlap[0].rows();
        if density.len() != self.sectors.len()
            || density.iter().any(|p| {
                !valid(p, n) || p.hermiticity_error() > self.sectors[0].options.operator_tolerance
            })
        {
            return Err(DftError::InvalidInput(
                "provide a finite Hermitian full spinor density at each magnetic k point".into(),
            ));
        }
        Ok(())
    }
    fn local_density(&self, density: &[BlochMatrix]) -> Result<xc::Evaluation> {
        let Some(grid) = &self.grid else {
            return Ok(xc::Evaluation::default());
        };
        let sector = &self.sectors[0];
        grid.lock()
            .map_err(|_| DftError::InvalidInput("magnetic Bloch grid cache poisoned".into()))?
            .evaluate(
                &self.sectors.iter().map(|s| &s.cell).collect::<Vec<_>>(),
                &sector.basis,
                sector.functional,
                &sector.options,
                density,
                &self.overlap,
            )
    }
    fn interactions(&self, density: &[BlochMatrix], coarse: bool) -> Result<Vec<[BlochMatrix; 3]>> {
        let n = density[0].rows() / 2;
        let nk = density.len() as f64;
        let charge = density
            .iter()
            .map(|p| Ok(pauli_components(p)?[0].clone()))
            .collect::<Result<Vec<_>>>()?;
        let mut out = (0..density.len())
            .map(|_| std::array::from_fn(|_| BlochMatrix::zeros(2 * n, 2 * n)))
            .collect::<Vec<_>>();
        if !self.scalar_hartree.is_empty() {
            for (i, row) in self.scalar_hartree.iter().enumerate() {
                let value = row
                    .iter()
                    .zip(&charge)
                    .map(|(c, p)| c[usize::from(coarse)] * (p.re[(0, 0)] / nk))
                    .fold(Complex::ZERO, |a, b| a + b);
                put(&mut out[i][0], 0, 0, value);
                put(&mut out[i][0], 1, 1, value);
            }
        } else {
            let modes = Self::hartree_modes(&self.sectors[0], coarse);
            for m in 0..modes.len() {
                let mode = &modes[m];
                if coarse
                    && self.sectors[0].coarse_modes.is_none()
                    && mode.wave.norm() > 0.8 * self.sectors[0].options.reciprocal_cutoff
                {
                    continue;
                }
                let mut rho = Complex::ZERO;
                for (s, p) in self.sectors.iter().zip(&charge) {
                    for a in 0..n {
                        for b in 0..n {
                            rho += z(p, a, b)
                                * z(&Self::hartree_modes(s, coarse)[m].density, a, b).conj()
                                / nk;
                        }
                    }
                }
                for (i, s) in self.sectors.iter().enumerate() {
                    add(
                        &mut out[i][0],
                        &spin_diagonal(&Self::hartree_modes(s, coarse)[m].density),
                        rho * mode.coulomb,
                    );
                }
            }
        }
        for (i, pairs) in self.transfers.iter().enumerate() {
            for (j, modes) in pairs.iter().enumerate() {
                if !self.scalar_exchange.is_empty() {
                    let sums = self.scalar_exchange[i][j][usize::from(coarse)];
                    out[i][1].add_scaled(&density[j], sums[0] / nk);
                    out[i][2].add_scaled(&density[j], sums[1] / nk);
                    continue;
                }
                for m in modes {
                    if (coarse && !m.coarse) || (!coarse && !m.fine) {
                        continue;
                    }
                    let r = spin_diagonal(&m.vertex);
                    let kp = matmul(&matmul(&r, &density[j])?, &adjoint(&r)?)?;
                    out[i][1].add_scaled(&kp, m.coulomb / nk);
                    out[i][2].add_scaled(&kp, m.long_range / nk);
                }
            }
        }
        let correction = self.corrections[usize::from(coarse)];
        for (i, p) in density.iter().enumerate() {
            if correction.iter().any(|x| *x != 0.0) {
                let limit = matmul(&matmul(&self.overlap[i], p)?, &self.overlap[i])?;
                out[i][1].add_scaled(&limit, correction[0]);
                out[i][2].add_scaled(&limit, correction[1]);
            }
        }
        Ok(out)
    }
}
impl SpinorEnergyBackend for MagneticBlochBackend<'_> {
    fn validate_occupation(
        &self,
        occupation: super::super::occupations::OccupationScheme,
    ) -> Result<()> {
        for sector in &self.sectors {
            sector.validate_occupation(occupation)?;
        }
        Ok(())
    }
    fn overlap(&self) -> &[BlochMatrix] {
        &self.overlap
    }
    fn initial_fock(&self) -> &[BlochMatrix] {
        &self.initial
    }
    fn electrons(&self) -> f64 {
        self.sectors[0].electrons
    }
    fn evaluate(&self, density: &[BlochMatrix]) -> Result<SpinorFields> {
        self.validate_density(density)?;
        if !self.crystal {
            return self.sectors[0].evaluate(density);
        }
        let fine = self.interactions(density, false)?;
        let coarse = self.interactions(density, true)?;
        let xc = self.local_density(density)?;
        let mut components = [0.0; 10];
        components[0] = self.sectors[0].source_energy;
        components[7] = xc.energy;
        let mut fock = Vec::new();
        let mut nuclear_soc_energy = 0.0;
        let mut defect = 0.0_f64;
        let nk = density.len() as f64;
        for (i, (p, s)) in density.iter().zip(&self.sectors).enumerate() {
            let cx = s.functional.hybrid.cx;
            let clr = s.functional.hybrid.clr;
            let mut f = s.initial[0].clone();
            f.add_scaled(&fine[i][0], 1.0);
            f.add_scaled(&fine[i][1], -cx);
            f.add_scaled(&fine[i][2], -clr);
            if let Some(v) = xc.potential.get(i) {
                f.add_scaled(v, 1.0);
            }
            defect = defect
                .max(s.source_defect)
                .max(gap(&s.nuclear, &s.nuclear_coarse))
                .max(gap(&s.soc, &s.soc_coarse))
                .max(gap(&fine[i][0], &coarse[i][0]))
                .max(cx.abs() * gap(&fine[i][1], &coarse[i][1]))
                .max(clr.abs() * gap(&fine[i][2], &coarse[i][2]));
            let charge = pauli_components(p)?[0].clone();
            components[1] += charge.trace_product(&s.kinetic) / nk;
            components[2] += charge.trace_product(&s.nuclear) / nk;
            components[4] += 0.5 * p.trace_product(&fine[i][0]) / nk;
            components[5] += -0.5 * cx * p.trace_product(&fine[i][1]) / nk;
            components[6] += -0.5 * clr * p.trace_product(&fine[i][2]) / nk;
            nuclear_soc_energy += p.trace_product(&s.soc) / nk;
            components[9] += p.trace_product(&s.spin) / nk;
            if !valid(&f, 2 * s.kinetic.rows())
                || f.hermiticity_error() > s.options.operator_tolerance
            {
                return Err(DftError::InvalidInput(
                    "magnetic Bloch Fock violates finite Hermiticity before projection".into(),
                ));
            }
            fock.push(f);
        }
        components[9] += nuclear_soc_energy;
        Ok(SpinorFields {
            fock,
            energy: components.iter().sum(),
            energy_components: components,
            xc_component_energies: xc.parts,
            nuclear_soc_energy,
            operator_converged: defect <= self.sectors[0].options.operator_tolerance
                && self.image_operator_defect <= self.sectors[0].options.operator_tolerance
                && xc.angular.as_ref().is_none_or(|d| d.converged)
                && xc
                    .spatial
                    .iter()
                    .all(|d| *d <= self.sectors[0].options.operator_tolerance),
            coulomb_operator_defect: defect,
            xc_grid_number_defect: xc.spatial[2],
            xc_diagnostics: xc.angular,
        })
    }
}
