// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Experimental reference-generated EPC quadrature with a fixed laboratory
//! motion law. Generation resolves a single-centre s+p proton wave node.
//! During every SCF, orbital response and nuclear derivative the points and
//! volume weights are immutable: their density and geometry derivatives vanish.
//! The calculated energy is the complete finite-quadrature EPC17 energy.

use super::correlation::{CorrelationModel, Epc17};
use super::{MulticomponentOptions, MulticomponentResult, QuantumParticle};
use crate::basis::MolBasis;
use crate::grid::{eval::eval_aos, MolecularGrid};
use crate::integrals::overlap_matrix;
use crate::math::Vec3;
use crate::resources::MemoryReservation;
use crate::response::ResponseSolverConfig;
use crate::scf::ScfOptions;
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

/// Orders are per radial topology panel and per polar nodal panel. The
/// finite radius and chart displacement are explicit numerical conditions,
/// not an assertion of continuum convergence or chemical accuracy.
#[derive(Clone, Copy, Debug)]
pub struct FrozenEpcGridOptions {
    pub radial_order: usize,
    pub polar_order: usize,
    pub azimuthal_points: usize,
    pub radius_bohr: f64,
    pub max_displacement_bohr: f64,
}

impl Default for FrozenEpcGridOptions {
    fn default() -> Self {
        Self {
            radial_order: 96,
            polar_order: 48,
            azimuthal_points: 48,
            radius_bohr: 4.0,
            max_displacement_bohr: 0.08,
        }
    }
}

impl FrozenEpcGridOptions {
    fn validate(self) -> Result<()> {
        if !(8..=128).contains(&self.radial_order)
            || !(8..=128).contains(&self.polar_order)
            || !(8..=128).contains(&self.azimuthal_points)
            || !self.radius_bohr.is_finite()
            || !(2.0..=12.0).contains(&self.radius_bohr)
            || !self.max_displacement_bohr.is_finite()
            || self.max_displacement_bohr <= 0.0
            || self.max_displacement_bohr > 0.25
        {
            return Err(DftError::InvalidInput(
                "invalid frozen EPC quadrature orders, radius or chart displacement".into(),
            ));
        }
        Ok(())
    }
}

/// An immutable, unpartitioned volume quadrature. It must not be sent through
/// the ordinary atom-owned Becke nuclear-weight response. Its reference and
/// finite radius define an explicit local numerical chart.
pub struct FrozenEpcGrid {
    grid: MolecularGrid,
    reference: Molecule,
    owner: usize,
    options: FrozenEpcGridOptions,
    epc: Epc17,
    particle_template: QuantumParticle,
    _reservation: MemoryReservation,
}

/// The immutable context ties a stationary state to the exact grid, provider,
/// basis, field and molecular inputs used to produce it. No public constructor
/// can attach a state from another integration condition.
pub struct FrozenEpcState<'a> {
    quadrature: &'a FrozenEpcGrid,
    molecule: &'a Molecule,
    basis: &'a MolBasis,
    options: &'a ScfOptions,
    functional: &'a ResolvedFunctional,
    particles: &'a [QuantumParticle],
    correlation: CorrelationModel,
    state: MulticomponentResult,
}

pub(super) fn legendre(n: usize) -> Result<Vec<(f64, f64)>> {
    let mut values = Vec::with_capacity(n);
    for k in 0..n {
        let mut x = (std::f64::consts::PI * (k as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut converged = false;
        for _ in 0..64 {
            let (p, derivative) = legendre_value(n, x);
            let next = x - p / derivative;
            if (next - x).abs() < 2e-16 {
                x = next;
                converged = true;
                break;
            }
            x = next;
        }
        if !converged {
            return Err(DftError::InvalidInput(
                "Gauss-Legendre root did not converge".into(),
            ));
        }
        let (_, derivative) = legendre_value(n, x);
        values.push((
            (x + 1.0) / 2.0,
            1.0 / ((1.0 - x * x) * derivative * derivative),
        ));
    }
    Ok(values)
}

fn legendre_value(n: usize, x: f64) -> (f64, f64) {
    let (mut a, mut b) = (1.0, x);
    for j in 2..=n {
        let c = ((2 * j - 1) as f64 * x * b - (j - 1) as f64 * a) / j as f64;
        a = b;
        b = c;
    }
    (b, n as f64 * (x * b - a) / (x * x - 1.0))
}

fn wave_parts(basis: &MolBasis, center: Vec3, c: &[f64], radius: f64) -> (f64, Vec3) {
    let mut points = Vec::with_capacity(6);
    for axis in 0..3 {
        for sign in [-1.0, 1.0] {
            let mut v = center.to_array();
            v[axis] += sign * radius;
            points.push(Vec3::new(v[0], v[1], v[2]));
        }
    }
    let ao = eval_aos(basis, &points, false).phi;
    let wave: Vec<f64> = (0..6)
        .map(|g| (0..c.len()).map(|i| ao[(g, i)] * c[i]).sum())
        .collect();
    (
        (wave[0] + wave[1]) / 2.0,
        Vec3::new(
            (wave[1] - wave[0]) / 2.0,
            (wave[3] - wave[2]) / 2.0,
            (wave[5] - wave[4]) / 2.0,
        ),
    )
}

impl FrozenEpcGrid {
    /// Generate once from a normalized reference proton orbital. Generation
    /// parameters are subsequently held fixed, including across displaced
    /// nuclear calculations. Rebuilding here at every displacement would
    /// define a different derivative and is not this interface's contract.
    pub fn from_reference(
        molecule: &Molecule,
        particle: &QuantumParticle,
        coefficients: &[f64],
        epc: Epc17,
        options: FrozenEpcGridOptions,
    ) -> Result<Self> {
        options.validate()?;
        let owner = particle.replaced_atom.ok_or_else(|| {
            DftError::InvalidInput(
                "frozen EPC grid requires an attached proton basis centre".into(),
            )
        })?;
        if owner >= molecule.len()
            || molecule.multiplicity != 1
            || molecule.atoms[owner].z != 1
            || particle.charge != 1.0
            || particle.correlation_channel != Some(0)
            || particle.basis.max_l > 1
            || coefficients.len() != particle.basis.nao_sph
            || coefficients.iter().any(|v| !v.is_finite())
            || !particle.basis.ecps.is_empty()
        {
            return Err(DftError::Unsupported { feature: "frozen EPC node-resolving reference".into(),
                missing: "one attached proton with a single-centre s+p basis and a singlet molecular chart".into() });
        }
        // Validate the provider before a potentially large allocation.
        super::correlation::InterspeciesCorrelation::evaluate(&epc, 1.0, 1.0)?;
        let center = molecule.atoms[owner].position;
        if particle
            .basis
            .shells
            .iter()
            .any(|s| (s.center - center).norm() > 1e-12)
        {
            return Err(DftError::InvalidInput(
                "frozen EPC reference basis must have one matching centre".into(),
            ));
        }
        let overlap = overlap_matrix(&particle.basis);
        let norm: f64 = (0..coefficients.len())
            .flat_map(|i| (0..coefficients.len()).map(move |j| (i, j)))
            .map(|(i, j)| coefficients[i] * overlap[(i, j)] * coefficients[j])
            .sum();
        if !norm.is_finite() || (norm - 1.0).abs() > 1e-8 {
            return Err(DftError::InvalidInput(
                "reference proton orbital must be overlap-normalized".into(),
            ));
        }
        let radial_function = |r: f64| {
            let (a, b) = wave_parts(&particle.basis, center, coefficients, r);
            let size = a.abs() + b.norm();
            if size == 0.0 {
                0.0
            } else {
                (a.abs() - b.norm()) / size
            }
        };
        let mut radial = vec![0.0];
        let mut left = 0.0;
        let mut fl = radial_function(left);
        for k in 1..=1024 {
            let right = options.radius_bohr * k as f64 / 1024.0;
            let fr = radial_function(right);
            if fl * fr < 0.0 {
                let (mut lo, mut hi, mut flo) = (left, right, fl);
                for _ in 0..64 {
                    let mid = (lo + hi) / 2.0;
                    let fm = radial_function(mid);
                    if flo * fm > 0.0 {
                        lo = mid;
                        flo = fm;
                    } else {
                        hi = mid;
                    }
                }
                radial.push((lo + hi) / 2.0);
            } else if fr == 0.0 && right < options.radius_bohr {
                radial.push(right);
            }
            left = right;
            fl = fr;
        }
        radial.push(options.radius_bohr);
        if radial.len() > 10 || radial.windows(2).any(|r| r[1] <= r[0]) {
            return Err(DftError::Unsupported {
                feature: "frozen EPC radial topology".into(),
                missing: "an isolated, bounded simple-node partition".into(),
            });
        }
        let count = (radial.len() - 1)
            * options.radial_order
            * 2
            * options.polar_order
            * options.azimuthal_points;
        if count > 3_000_000 {
            return Err(DftError::ResourceLimit {
                operation: "frozen EPC quadrature",
                required_mb: count * 48 / 1024 / 1024 + 1,
                limit_mb: 3_000_000 * 48 / 1024 / 1024,
            });
        }
        let reservation = crate::resources::reserve(count * 48, "frozen EPC quadrature")?;
        let mut grid = MolecularGrid {
            points: Vec::with_capacity(count),
            weights: Vec::with_capacity(count),
            raw_weights: Vec::with_capacity(count),
            atom_of: Vec::with_capacity(count),
        };
        let rr = legendre(options.radial_order)?;
        let mr = legendre(options.polar_order)?;
        for panel in radial.windows(2) {
            for &(t, rw) in &rr {
                let r = panel[0] + (panel[1] - panel[0]) * t;
                let dr = (panel[1] - panel[0]) * rw;
                let (a, b) = wave_parts(&particle.basis, center, coefficients, r);
                let bn = b.norm();
                let z = if bn > 0.0 {
                    b / bn
                } else {
                    Vec3::new(0.0, 0.0, 1.0)
                };
                let least = (0..3)
                    .min_by(|&i, &j| z.get(i).abs().total_cmp(&z.get(j).abs()))
                    .unwrap();
                let mut seed = [0.0; 3];
                seed[least] = 1.0;
                let x = z.cross(Vec3::new(seed[0], seed[1], seed[2])).normalized();
                let y = z.cross(x);
                let node = if bn > 0.0 { -a / bn } else { 2.0 };
                let panels = if node > -1.0 && node < 1.0 {
                    vec![(-1.0, node), (node, 1.0)]
                } else {
                    vec![(-1.0, 1.0)]
                };
                for (lo, hi) in panels {
                    for &(v, mw) in &mr {
                        let mu = lo + (hi - lo) * v;
                        let sin = (1.0 - mu * mu).sqrt();
                        for k in 0..options.azimuthal_points {
                            let phi = std::f64::consts::TAU * (k as f64 + 0.5)
                                / options.azimuthal_points as f64;
                            let point = center
                                + (x * (sin * phi.cos()) + y * (sin * phi.sin()) + z * mu) * r;
                            let w = dr * r * r * (hi - lo) * mw * std::f64::consts::TAU
                                / options.azimuthal_points as f64;
                            grid.points.push(point);
                            grid.weights.push(w);
                            grid.raw_weights.push(w);
                            grid.atom_of.push(owner);
                        }
                    }
                }
            }
        }
        Ok(Self {
            grid,
            reference: molecule.clone(),
            owner,
            options,
            epc,
            particle_template: particle.clone(),
            _reservation: reservation,
        })
    }

    /// Exact immutable points and volume weights of this numerical condition.
    pub fn grid(&self) -> &MolecularGrid {
        &self.grid
    }

    #[allow(clippy::too_many_arguments)]
    pub fn solve<'a>(
        &'a self,
        molecule: &'a Molecule,
        basis: &'a MolBasis,
        options: &'a ScfOptions,
        functional: &'a ResolvedFunctional,
        particles: &'a [QuantumParticle],
        coupled: &MulticomponentOptions,
    ) -> Result<FrozenEpcState<'a>> {
        if !functional.parts.is_empty()
            || functional.hybrid.cx != 1.0
            || functional.hybrid.clr != 0.0
            || functional.required_nonlocal_term().is_some()
            || options.skala.is_some()
            || options.vv10.is_some()
            || options.density_fit.is_some()
            || !basis.ecps.is_empty()
            || molecule.multiplicity != 1
            || options.electronic_temperature != 0.0
            || options.reference == crate::scf::Reference::Uhf
            || options.occupation.is_some()
            || options.fixed_occupations.is_some()
            || options.electron_offset != [0.0; 2]
            || particles.len() != 1
            || particles[0].replaced_atom != Some(self.owner)
            || particles[0].charge != 1.0
            || particles[0].correlation_channel != Some(0)
        {
            return Err(DftError::Unsupported {
                feature: "frozen EPC quadrature calculation".into(),
                missing:
                    "nonthermal closed-shell HF, one attached proton and analytic Coulomb integrals"
                        .into(),
            });
        }
        let q = &particles[0];
        let template = &self.particle_template;
        let owner_position = molecule
            .atoms
            .get(self.owner)
            .ok_or_else(|| DftError::InvalidInput("frozen EPC geometry owner is absent".into()))?
            .position;
        if q.mass != template.mass
            || q.basis.nao_sph != template.basis.nao_sph
            || q.basis.shells.len() != template.basis.shells.len()
            || q.basis
                .shells
                .iter()
                .zip(&template.basis.shells)
                .any(|(a, b)| {
                    a.l != b.l
                        || a.exps != b.exps
                        || a.coefs != b.coefs
                        || a.sph_off != b.sph_off
                        || (a.center - owner_position).norm() > 1e-12
                })
        {
            return Err(DftError::InvalidInput(
                "quantum basis or mass differs from the frozen EPC reference chart".into(),
            ));
        }
        if molecule.len() != self.reference.len()
            || molecule.charge != self.reference.charge
            || molecule
                .atoms
                .iter()
                .zip(&self.reference.atoms)
                .any(|(a, b)| {
                    a.z != b.z
                        || (a.position - b.position).norm() > self.options.max_displacement_bohr
                })
        {
            return Err(DftError::InvalidInput(
                "molecule is outside the frozen EPC geometry/charge chart".into(),
            ));
        }
        let correlation = CorrelationModel::single(Box::new(self.epc));
        let state = super::solve(
            molecule,
            basis,
            options,
            functional,
            Some(&self.grid),
            particles,
            &correlation,
            coupled,
        )?;
        Ok(FrozenEpcState {
            quadrature: self,
            molecule,
            basis,
            options,
            functional,
            particles,
            correlation,
            state,
        })
    }
}

impl FrozenEpcState<'_> {
    pub fn result(&self) -> &MulticomponentResult {
        &self.state
    }

    /// Complete analytic geometry response of the same fixed-point energy.
    /// No derivative of the reference-generation algorithm is included or
    /// implied: regenerating the mesh defines another numerical condition.
    pub fn directional(
        &self,
        direction: &[[f64; 3]],
        config: &ResponseSolverConfig,
    ) -> Result<super::derivatives::CoupledGeometryDerivative> {
        super::derivatives::directional_fixed_grid(
            self.molecule,
            self.basis,
            self.options,
            self.functional,
            &self.quadrature.grid,
            self.particles,
            &self.correlation,
            &self.state,
            direction,
            config,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quadrature_moments_and_invalid_orders() -> Result<()> {
        for n in [8, 32, 64, 96, 128] {
            let r = legendre(n)?;
            for p in 0..=6 {
                let integral: f64 = r.iter().map(|(x, w)| w * x.powi(p)).sum();
                assert!((integral - 1.0 / f64::from(p + 1)).abs() < 2e-14);
            }
        }
        assert!(FrozenEpcGridOptions {
            radial_order: 0,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(FrozenEpcGridOptions {
            radius_bohr: f64::NAN,
            ..Default::default()
        }
        .validate()
        .is_err());
        Ok(())
    }
}
