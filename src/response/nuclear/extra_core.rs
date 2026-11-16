// SPDX-License-Identifier: Apache-2.0
//! Scoped density-independent extra-core Taylor source. Coefficients are
//! factorial normalized: v(t)=v[0]+t*v[1]+t^2*v[2]. No displaced SCF is used.
use super::{geometry::GeometryTaylor, MolecularDirection, MolecularTaylor};
use crate::basis::MolBasis;
use crate::grid::MolecularGrid;
use crate::linalg::{AccurateSum, Matrix};
use crate::response::{ResponseReport, ResponseSolverConfig};
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

fn invalid<T>(message: &str) -> Result<T> {
    Err(DftError::InvalidInput(message.into()))
}
fn unsupported<T>(feature: &str) -> Result<T> {
    Err(DftError::Unsupported {
        feature: feature.into(),
        missing: "complete real common-spin density-independent analytic core coefficients through order two".into(),
    })
}

/// The producer and consumer must name the same physical chart, AO/projector
/// layout, grid recipe, and coordinate convention. These are not ordinary
/// derivative-plan flags and do not authorize a missing source coefficient.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExtraCoreIdentity {
    pub chart: u64,
    pub operator_layout: u64,
    pub grid_recipe: u64,
    pub coordinate_map: u64,
}

/// IndependentRoles means straight paths of all physical/auxiliary/grid sites.
/// LiftedPrimary means J and curvature have ALREADY been included by the
/// analytic producer. It must never receive a second geometry pullback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExtraCoreCoordinates {
    IndependentRoles,
    LiftedPrimary,
}

/// Both operator and interaction already contain the complete fixed-field
/// physical-minus-labelled and origin corrections. This token is not an
/// instruction to apply either correction again. Electric response is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExtraCoreFieldConvention {
    ProvidedCompleteDeltaFromOrdinaryOriginZero,
}

/// A complete scalar ledger relative to the ordinary labelled-nuclear energy.
/// interaction ALREADY includes all field-coupled scalar corrections but not E0. E0 is constant
/// on this path, never relabelled as nuclear repulsion. permanent_dipole is the
/// ionic/core correction RELATIVE TO ORDINARY ORIGIN ZERO, including the
/// labelled-nuclear origin shift when supplied by the origin-zero pair kernel.
/// It is not another Gaussian source and is not the full molecular dipole.
#[derive(Clone, Debug)]
pub(crate) struct ExternalClassicalTaylor {
    pub interaction: Vec<f64>,
    pub internal_energy_reference: f64,
    pub permanent_dipole: Vec<[f64; 3]>,
}

/// Source bytes are private after validation. Binding includes exact molecular,
/// AO, quadrature, spin, reference density/Fock and combined path data.
#[derive(Clone, Debug)]
pub(crate) struct ExtraCoreTaylor {
    identity: ExtraCoreIdentity,
    functional_description: String,
    binding: u64,
    fingerprint: u64,
    coordinates: ExtraCoreCoordinates,
    operator: [Vec<Matrix>; 2],
    classical: ExternalClassicalTaylor,
    origin: [f64; 3],
    field_convention: ExtraCoreFieldConvention,
    order: usize,
}

/// The standard component ledger remains unchanged; its additional scalar
/// coefficients are returned here explicitly and are already INCLUDED once in
/// response.value.energy_coefficients. They must not be added to it again.
pub(crate) struct ExtraCoreTaylorReport {
    pub response: ResponseReport<MolecularTaylor>,
    pub external_classical_coefficients: Vec<f64>,
    pub internal_energy_reference: f64,
    pub source_fingerprint: u64,
    pub coordinates: ExtraCoreCoordinates,
}

struct Fingerprint(u64);
impl Fingerprint {
    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
    fn number(&mut self, n: usize) {
        self.bytes(&(n as u64).to_le_bytes());
    }
    fn real(&mut self, x: f64) {
        self.bytes(&x.to_bits().to_le_bytes());
    }
    fn matrix(&mut self, m: &Matrix) {
        self.number(m.rows);
        self.number(m.cols);
        for &v in m.as_slice() {
            self.real(v);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn binding(
    mol: &Molecule,
    basis: &MolBasis,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    options: &ScfOptions,
    direction: &MolecularDirection,
    q_direction: &[f64],
    identity: &ExtraCoreIdentity,
) -> u64 {
    let mut h = Fingerprint(0xcbf29ce484222325);
    // Debug serialization is an in-process structural guard, not a persisted
    // model format. No pointer/address or invented state fingerprint is used.
    h.bytes(format!("{mol:?}|{basis:?}|{identity:?}").as_bytes());
    h.number(reference.n_alpha);
    h.number(reference.n_beta);
    h.number(usize::from(reference.unrestricted));
    h.real(reference.energy);
    h.real(reference.e_nuc);
    for matrix in [
        &reference.overlap,
        &reference.h_core,
        &reference.fock[0],
        &reference.fock[1],
        &reference.density[0],
        &reference.density[1],
    ] {
        h.matrix(matrix);
    }
    h.number(usize::from(grid.is_some()));
    if let Some(g) = grid {
        h.number(g.len());
        for (((p, &w), &raw), &atom) in g
            .points
            .iter()
            .zip(&g.weights)
            .zip(&g.raw_weights)
            .zip(&g.atom_of)
        {
            for v in [p.x, p.y, p.z, w, raw] {
                h.real(v);
            }
            h.number(atom);
        }
    }
    for &v in direction
        .nuclear
        .iter()
        .flatten()
        .chain(&direction.electric)
        .chain(q_direction)
    {
        h.real(v);
    }
    let field = options.external_field.map_or([0.; 3], |v| [v.x, v.y, v.z]);
    for v in field {
        h.real(v);
    }
    if let Some(extra) = &options.extra_hcore {
        h.matrix(extra);
    }
    h.0
}

impl ExtraCoreTaylor {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        mol: &Molecule,
        basis: &MolBasis,
        functional: &ResolvedFunctional,
        grid: Option<&MolecularGrid>,
        reference: &ScfResult,
        options: &ScfOptions,
        direction: &MolecularDirection,
        q_direction: &[f64],
        identity: ExtraCoreIdentity,
        coordinates: ExtraCoreCoordinates,
        operator: [Vec<Matrix>; 2],
        classical: ExternalClassicalTaylor,
        origin: [f64; 3],
        field_convention: ExtraCoreFieldConvention,
        order: usize,
    ) -> Result<Self> {
        if order > 2 {
            return unsupported("extra-core source above second order");
        }
        if direction.electric != [0.; 3] {
            return unsupported("extra-core electric response without complete field jets");
        }
        if coordinates == ExtraCoreCoordinates::LiftedPrimary {
            return unsupported(
                "lifted-primary extra core without matching curved integral/grid series",
            );
        }
        if basis.nao_sph != reference.overlap.rows
            || reference.overlap.rows != reference.overlap.cols
            || !reference.converged
            || !reference.energy.is_finite()
            || !reference.e_nuc.is_finite()
            || [
                &reference.h_core,
                &reference.fock[0],
                &reference.fock[1],
                &reference.density[0],
                &reference.density[1],
            ]
            .iter()
            .any(|m| {
                m.rows != basis.nao_sph
                    || m.cols != basis.nao_sph
                    || m.as_slice().iter().any(|v| !v.is_finite())
            })
        {
            return invalid("extra-core requires a finite converged correctly-sized reference");
        }
        if !basis.ecps.is_empty()
            || basis.n_core_electrons != 0
            || reference.fixed_occupation_residual.is_some()
        {
            return unsupported("extra-core ECP or prescribed fractional-population reference");
        }
        let width = order + 1;
        if direction.nuclear.len() != mol.len()
            || direction
                .nuclear
                .iter()
                .flatten()
                .chain(&direction.electric)
                .chain(q_direction)
                .chain(&origin)
                .any(|v| !v.is_finite())
            || operator.iter().any(|v| v.len() != width)
            || classical.interaction.len() != width
            || classical.permanent_dipole.len() != width
            || !classical.internal_energy_reference.is_finite()
            || classical
                .interaction
                .iter()
                .chain(classical.permanent_dipole.iter().flatten())
                .any(|v| !v.is_finite())
        {
            return invalid(
                "extra-core source requires complete finite factorial-normalized coefficients",
            );
        }
        if let Some(g) = grid {
            if g.points.len() != g.weights.len()
                || g.points.len() != g.raw_weights.len()
                || g.points.len() != g.atom_of.len()
                || g.atom_of.iter().any(|&a| a >= mol.len())
                || g.points
                    .iter()
                    .any(|p| [p.x, p.y, p.z].iter().any(|v| !v.is_finite()))
                || g.weights
                    .iter()
                    .chain(&g.raw_weights)
                    .any(|v| !v.is_finite())
            {
                return invalid("extra-core source quadrature binding is malformed");
            }
        }
        for matrices in &operator {
            for m in matrices {
                if m.rows != basis.nao_sph
                    || m.cols != basis.nao_sph
                    || m.as_slice().iter().any(|v| !v.is_finite())
                {
                    return invalid("extra-core operator dimensions or values are invalid");
                }
                for i in 0..m.rows {
                    for j in 0..i {
                        if (m[(i, j)] - m[(j, i)]).abs() > 1e-12 {
                            return invalid("extra-core operator must be real symmetric");
                        }
                    }
                }
            }
        }
        for n in 0..width {
            if operator[0][n].max_abs_diff(&operator[1][n]) != 0.0 {
                return unsupported("spin-dependent extra-core source");
            }
        }
        let binding = binding(
            mol,
            basis,
            grid,
            reference,
            options,
            direction,
            q_direction,
            &identity,
        );
        let mut h = Fingerprint(binding);
        h.bytes(functional.description.as_bytes());
        h.number(order);
        h.number(match coordinates {
            ExtraCoreCoordinates::IndependentRoles => 0,
            ExtraCoreCoordinates::LiftedPrimary => 1,
        });
        for &v in &origin {
            h.real(v);
        }
        for m in &operator[0] {
            h.matrix(m);
        }
        for &v in &classical.interaction {
            h.real(v);
        }
        for &v in classical.permanent_dipole.iter().flatten() {
            h.real(v);
        }
        h.real(classical.internal_energy_reference);
        Ok(Self {
            identity,
            functional_description: functional.description.clone(),
            binding,
            fingerprint: h.0,
            coordinates,
            operator,
            classical,
            origin,
            field_convention,
            order,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn validate_binding(
        &self,
        mol: &Molecule,
        basis: &MolBasis,
        functional: &ResolvedFunctional,
        grid: Option<&MolecularGrid>,
        reference: &ScfResult,
        options: &ScfOptions,
        direction: &MolecularDirection,
        q_direction: &[f64],
        identity: &ExtraCoreIdentity,
        order: usize,
    ) -> Result<()> {
        if self.order != order
            || &self.identity != identity
            || self.functional_description != functional.description
            || self.binding
                != binding(
                    mol,
                    basis,
                    grid,
                    reference,
                    options,
                    direction,
                    q_direction,
                    identity,
                )
        {
            return invalid("extra-core source/reference/chart/path binding mismatch");
        }
        Ok(())
    }

    /// Source energy jets already contain the origin shift. Only the dipole
    /// OBSERVABLE gets r-origin*S here; it does not feed a second field term
    /// into the core. Permanent correction is supplied relative to that origin.
    pub(super) fn attach(
        &self,
        geometry: &mut GeometryTaylor,
        _mol: &Molecule,
        _options: &ScfOptions,
        direction: &MolecularDirection,
    ) -> Result<Vec<f64>> {
        if self.field_convention
            != ExtraCoreFieldConvention::ProvidedCompleteDeltaFromOrdinaryOriginZero
            || direction.electric != [0.; 3]
        {
            return unsupported("unproved extra-core field convention");
        }
        let width = self.order + 1;
        let mut scalar = self.classical.interaction.clone();
        scalar[0] += self.classical.internal_energy_reference;
        for n in 0..width {
            geometry.add_extra_core_coefficient(n, &self.operator[0][n])?;
            for axis in 0..3 {
                geometry.nuclear_dipole[n][axis] += self.classical.permanent_dipole[n][axis];
                super::add(
                    &mut geometry.dipoles[axis][n],
                    &geometry.overlap[n],
                    -self.origin[axis],
                );
            }
        }
        geometry.set_external_classical(scalar.clone())?;
        Ok(scalar)
    }

    /// Bound construction uses exactly the supplied direction, grid and AO
    /// layout. Unbound prebuilt geometry cannot masquerade as this source.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn geometry(
        &self,
        mol: &Molecule,
        basis: &MolBasis,
        functional: &ResolvedFunctional,
        grid: Option<&MolecularGrid>,
        reference: &ScfResult,
        options: &ScfOptions,
        direction: &MolecularDirection,
        q_direction: &[f64],
        identity: &ExtraCoreIdentity,
    ) -> Result<ExtraCoreGeometry> {
        self.validate_binding(
            mol,
            basis,
            functional,
            grid,
            reference,
            options,
            direction,
            q_direction,
            identity,
            self.order,
        )?;
        let reference_field = options.external_field.map_or([0.; 3], |v| [v.x, v.y, v.z]);
        let geometry = GeometryTaylor::new(
            mol,
            basis,
            grid,
            functional,
            options.skala.as_ref(),
            &direction.nuclear,
            reference_field,
            direction.electric,
            self.order,
            options.incore_mb,
        )?;
        Ok(ExtraCoreGeometry {
            geometry,
            binding: self.binding,
            source: self.fingerprint,
        })
    }
}

/// Only this module can bind a prebuilt geometry to a validated source.
pub(crate) struct ExtraCoreGeometry {
    geometry: GeometryTaylor,
    binding: u64,
    source: u64,
}

/// Prebuilt geometry uses the same moving metric/grid/ERI machinery. The
/// provided source is checked before any response or reference polishing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn molecular_taylor_with_extra_core(
    mol: &Molecule,
    basis: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    options: &ScfOptions,
    direction: &MolecularDirection,
    q_direction: &[f64],
    identity: &ExtraCoreIdentity,
    source: &ExtraCoreTaylor,
    order: usize,
    config: &ResponseSolverConfig,
    geometry: ExtraCoreGeometry,
) -> Result<ExtraCoreTaylorReport> {
    source.validate_binding(
        mol,
        basis,
        functional,
        grid,
        reference,
        options,
        direction,
        q_direction,
        identity,
        order,
    )?;
    if geometry.binding != source.binding || geometry.source != source.fingerprint {
        return invalid("prebuilt geometry belongs to another extra-core source");
    }
    let (mut response, _) = super::molecular_taylor_reference(
        mol,
        basis,
        functional,
        grid,
        reference,
        options,
        direction,
        order,
        config,
        Some(geometry.geometry),
        Some(source),
    )?;
    response.model = "static fixed-chart density-independent analytic extra-core KS/HF; moving basis/grid and coupled orbital response";
    let mut external = source.classical.interaction.clone();
    external[0] += source.classical.internal_energy_reference;
    Ok(ExtraCoreTaylorReport {
        response,
        external_classical_coefficients: external,
        internal_energy_reference: source.classical.internal_energy_reference,
        source_fingerprint: source.fingerprint,
        coordinates: source.coordinates,
    })
}

pub(super) fn accurate_total(high: f64, low: f64) -> f64 {
    let mut sum = AccurateSum::default();
    sum.add_product(high, 1.);
    sum.add_product(low, 1.);
    sum.value()
}

/// Convert actual directional derivatives (not pair mixed lanes) to the
/// factorial-normalized convention used by GeometryTaylor. For a combined
/// direction d=(Rdot,qdot), second is D_d D_d h, obtained analytically by
/// polarization or the pair kernel with BOTH legs equal to d.
pub(crate) fn second_order_operator_coefficients(
    value: Matrix,
    first: Matrix,
    second: Matrix,
) -> Result<Vec<Matrix>> {
    if [first.rows, second.rows] != [value.rows; 2]
        || [first.cols, second.cols] != [value.cols; 2]
        || [&value, &first, &second]
            .iter()
            .any(|m| m.rows != m.cols || m.as_slice().iter().any(|v| !v.is_finite()))
    {
        return invalid("invalid actual derivative matrices for extra-core Taylor conversion");
    }
    let mut factorial_second = second;
    for v in factorial_second.as_mut_slice() {
        *v *= 0.5;
    }
    Ok(vec![value, first, factorial_second])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_second_derivative_is_halved_exactly_once() {
        let mut a = Matrix::zeros(1, 1);
        a[(0, 0)] = 3.;
        let mut da = Matrix::zeros(1, 1);
        da[(0, 0)] = -7.;
        let mut ddaa = Matrix::zeros(1, 1);
        ddaa[(0, 0)] = 10.;
        let c = second_order_operator_coefficients(a, da, ddaa).unwrap();
        assert_eq!(c[0][(0, 0)], 3.);
        assert_eq!(c[1][(0, 0)], -7.);
        assert_eq!(c[2][(0, 0)], 5.);
        let t = 0.125;
        assert_eq!(
            c[0][(0, 0)] + t * c[1][(0, 0)] + t * t * c[2][(0, 0)],
            3. - 7. * t + 5. * t * t
        );
    }
    #[test]
    fn malformed_derivative_shape_is_rejected() {
        assert!(second_order_operator_coefficients(
            Matrix::zeros(1, 1),
            Matrix::zeros(2, 2),
            Matrix::zeros(1, 1)
        )
        .is_err());
    }
}

#[cfg(test)]
#[path = "extra_core_context_tests.rs"]
mod component_oracles;
