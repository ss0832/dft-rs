// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Analytic two-direction jets of the saved static one-body/core energy.
//!
//! This is not an SCF Hessian. Kinetic, overlap/Pulay, ERI, moving quadrature,
//! active Hartree/XC and orbital response are supplied by the response driver.
//! Geometry directions are independent full-role coordinates. A rigid lifting
//! map and its second-order connection term are applied by the caller once.

use super::static_core::StaticCoreParameters;
use crate::basis::{aos, MolBasis};
use crate::dual::{DualN, Scalar};
use crate::integrals::analytic::{GaussianKernel, Primitive};
use crate::integrals::analytic_basis::{self, AnalyticNucleus, AnalyticShell};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::pseudo_sub::frame::{
    build_local_frame_with_connectivity, frame_directional_second, frame_jacobian,
    FrameConnectivity,
};
use crate::pseudo_sub::SiteSymmetry;
use crate::system::Molecule;
use crate::{DftError, Result};
use std::f64::consts::PI;

/// Lanes extracted below are value, derivative a, derivative b, derivative ab.
pub type PairScalar = DualN<DualN<f64, 1>, 1>;

pub fn seed(value: f64, a: f64, b: f64, ab: f64) -> PairScalar {
    PairScalar {
        re: DualN { re: value, im: [b] },
        im: [DualN { re: a, im: [ab] }],
    }
}

pub fn lanes(value: PairScalar) -> [f64; 4] {
    [
        value.re.re,
        value.im[0].re,
        value.re.im[0],
        value.im[0].im[0],
    ]
}

fn invalid<T>(message: &str) -> Result<T> {
    Err(DftError::InvalidInput(message.into()))
}

fn finite(value: PairScalar) -> bool {
    lanes(value).into_iter().all(f64::is_finite)
}

fn plain(position: [PairScalar; 3]) -> Vec3 {
    Vec3::new(
        position[0].value(),
        position[1].value(),
        position[2].value(),
    )
}

/// Positions and the orientation use the same guarded attachment frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeometryJetSpace {
    /// Entries are partials in independent full-role Cartesian coordinates.
    /// Apply the rigid-map Hessian connection in the final response pullback.
    IndependentRoles,
    /// The supplied coordinate jets already include derivatives of a declared
    /// lifting map. Do not apply a second lifting Hessian or force pullback.
    LiftedPrimary { lifting_fingerprint: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeometryOrientation {
    AttachmentFrame,
    /// An explicitly differentiated proper rotation, independent of the AO
    /// center coordinates. All orthogonality derivative lanes are checked.
    RigidPose,
    /// Deliberately frame-free spherical sources. The identity orientation is
    /// fixed in laboratory coordinates, not an inferred molecular body frame.
    LaboratoryFixed,
}

pub struct PairGeometry {
    positions: Vec<[PairScalar; 3]>,
    orientation: [[PairScalar; 3]; 3],
    own_ion: usize,
    space: GeometryJetSpace,
    fingerprint: u64,
    orientation_convention: GeometryOrientation,
}

impl PairGeometry {
    /// Independent retained-center and orientation jets. Columns of `rotation`
    /// are body axes in laboratory coordinates. This constructor includes any
    /// chart curvature already present in those jets; callers must not apply
    /// a second Euler/frame Hessian connection afterward.
    pub fn rigid_pose(
        molecule: &Molecule,
        own_ion: usize,
        positions: Vec<[PairScalar; 3]>,
        rotation: [[PairScalar; 3]; 3],
        coordinate_map: u64,
    ) -> Result<Self> {
        if molecule.is_empty()
            || own_ion >= molecule.len()
            || positions.len() != molecule.len()
            || coordinate_map == 0
            || positions
                .iter()
                .flatten()
                .chain(rotation.iter().flatten())
                .any(|&v| !finite(v))
            || molecule
                .atoms
                .iter()
                .zip(&positions)
                .any(|(a, &p)| (a.position - plain(p)).norm() > 1e-12)
        {
            return invalid("rigid pose jets require matching finite centers, a source role and a nonzero coordinate-map identity");
        }
        for i in 0..3 {
            for j in 0..3 {
                let terms: [PairScalar; 3] =
                    std::array::from_fn(|k| rotation[i][k] * rotation[j][k]);
                let dot = terms
                    .into_iter()
                    .fold(PairScalar::from_f64(0.), |s, v| s + v);
                let expected = if i == j { 1. } else { 0. };
                for (lane, value) in lanes(dot).into_iter().enumerate() {
                    let scale = terms
                        .iter()
                        .map(|&v| lanes(v)[lane].abs())
                        .sum::<f64>()
                        .max(1.);
                    let target = if lane == 0 { expected } else { 0. };
                    if (value - target).abs() > 512. * f64::EPSILON * scale {
                        return invalid("rigid pose rotation must preserve orthonormality in every derivative lane");
                    }
                }
            }
        }
        let u = rotation.map(plain);
        if (u[0].cross(u[1]).dot(u[2]) - 1.).abs() > 512. * f64::EPSILON {
            return invalid("rigid pose rotation must be proper and right handed");
        }
        let mut identity = super::Fingerprint::new();
        identity.string("static-core-rigid-pose-pair-v1");
        identity.number(own_ion);
        identity.bytes(&coordinate_map.to_le_bytes());
        for &v in positions.iter().flatten().chain(rotation.iter().flatten()) {
            for value in lanes(v) {
                identity.float(value);
            }
        }
        Ok(Self {
            positions,
            orientation: rotation,
            own_ion,
            space: GeometryJetSpace::LiftedPrimary {
                lifting_fingerprint: coordinate_map,
            },
            fingerprint: identity.0,
            orientation_convention: GeometryOrientation::RigidPose,
        })
    }

    /// Anchors are [host X, fragment G, retained reference]. The mixed position
    /// direction allows an already differentiated geometry map, but no lifting
    /// pullback is performed here. Independent Cartesian R derivatives use zero.
    pub fn new(
        molecule: &Molecule,
        anchors: [usize; 3],
        a: &[Vec3],
        b: &[Vec3],
        ab: &[Vec3],
        space: GeometryJetSpace,
    ) -> Result<Self> {
        let n = molecule.len();
        if n == 0
            || a.len() != n
            || b.len() != n
            || ab.len() != n
            || anchors.iter().any(|&i| i >= n)
            || (0..3).any(|i| (0..i).any(|j| anchors[i] == anchors[j]))
        {
            return invalid(
                "core geometry jets need matching positions, directions and three distinct anchors",
            );
        }
        if space == GeometryJetSpace::IndependentRoles
            && ab
                .iter()
                .any(|v| v.to_array().into_iter().any(|x| x != 0.0))
        {
            return invalid(
                "independent role-coordinate jets cannot already include a lifting curvature",
            );
        }
        let frame = build_local_frame_with_connectivity(
            molecule,
            anchors[1],
            FrameConnectivity {
                x_atom: anchors[0],
                reference_candidates: &anchors[2..],
            },
            SiteSymmetry::None,
        )?
        .ok_or_else(|| {
            DftError::InvalidInput("an anisotropic core needs a complete frame".into())
        })?;
        let points = anchors.map(|i| molecule.atoms[i].position);
        let da = std::array::from_fn(|i| a[anchors[i / 3]].get(i % 3));
        let db = std::array::from_fn(|i| b[anchors[i / 3]].get(i % 3));
        let dab: [f64; 9] = std::array::from_fn(|i| ab[anchors[i / 3]].get(i % 3));
        let raw = frame_directional_second(points[0], points[1], points[2], da, db);
        let jacobian = frame_jacobian(points[0], points[1], points[2]);
        let orientation = std::array::from_fn(|column| {
            std::array::from_fn(|axis| {
                let v = raw[column][axis];
                let connection: f64 = (0..9)
                    .map(|i| jacobian[(3 * column + axis, i)] * dab[i])
                    .sum();
                seed(frame.u.col[column].get(axis), v[1], v[2], v[3] + connection)
            })
        });
        let positions: Vec<_> = molecule
            .atoms
            .iter()
            .enumerate()
            .map(|(i, atom)| {
                std::array::from_fn(|axis| {
                    seed(
                        atom.position.get(axis),
                        a[i].get(axis),
                        b[i].get(axis),
                        ab[i].get(axis),
                    )
                })
            })
            .collect();
        if positions
            .iter()
            .flatten()
            .chain(orientation.iter().flatten())
            .any(|&x| !finite(x))
        {
            return invalid("core geometry jet values and all derivative lanes must be finite");
        }
        let mut identity = super::Fingerprint::new();
        identity.string("static-core-geometry-pair-v1");
        match space {
            GeometryJetSpace::IndependentRoles => identity.string("independent-role-coordinates"),
            GeometryJetSpace::LiftedPrimary {
                lifting_fingerprint,
            } => {
                identity.string("already-lifted-primary-coordinates");
                identity.bytes(&lifting_fingerprint.to_le_bytes());
            }
        }
        for &anchor in &anchors {
            identity.number(anchor);
        }
        for point in &positions {
            for &component in point {
                for value in lanes(component) {
                    identity.float(value);
                }
            }
        }
        Ok(Self {
            positions,
            orientation,
            own_ion: anchors[1],
            space,
            fingerprint: identity.0,
            orientation_convention: GeometryOrientation::AttachmentFrame,
        })
    }

    /// Explicit frame-free constructor for point ions and spherical local
    /// sources, including atoms and two-atom systems. This is never a fallback
    /// after an attachment frame degenerates. Body-fixed dipoles/projectors
    /// require `new`; the evaluator rejects them for this geometry convention.
    pub fn laboratory(
        molecule: &Molecule,
        own_ion: usize,
        a: &[Vec3],
        b: &[Vec3],
        ab: &[Vec3],
        space: GeometryJetSpace,
    ) -> Result<Self> {
        let n = molecule.len();
        if n == 0 || own_ion >= n || a.len() != n || b.len() != n || ab.len() != n {
            return invalid(
                "laboratory geometry jets need matching nonempty roles and a valid source role",
            );
        }
        if space == GeometryJetSpace::IndependentRoles
            && ab.iter().any(|v| v.to_array().into_iter().any(|x| x != 0.))
        {
            return invalid(
                "independent role-coordinate jets cannot already include a lifting curvature",
            );
        }
        let positions: Vec<_> = molecule
            .atoms
            .iter()
            .enumerate()
            .map(|(i, atom)| {
                std::array::from_fn(|axis| {
                    seed(
                        atom.position.get(axis),
                        a[i].get(axis),
                        b[i].get(axis),
                        ab[i].get(axis),
                    )
                })
            })
            .collect();
        if positions.iter().flatten().any(|&x| !finite(x)) {
            return invalid(
                "laboratory geometry jet values and all derivative lanes must be finite",
            );
        }
        let mut identity = super::Fingerprint::new();
        identity.string("static-core-laboratory-geometry-pair-v1");
        identity.number(own_ion);
        match space {
            GeometryJetSpace::IndependentRoles => identity.string("independent-role-coordinates"),
            GeometryJetSpace::LiftedPrimary {
                lifting_fingerprint,
            } => {
                identity.string("already-lifted-primary-coordinates");
                identity.bytes(&lifting_fingerprint.to_le_bytes());
            }
        }
        for point in &positions {
            for &component in point {
                for value in lanes(component) {
                    identity.float(value);
                }
            }
        }
        Ok(Self {
            positions,
            orientation: std::array::from_fn(|column| {
                std::array::from_fn(|axis| seed(if column == axis { 1. } else { 0. }, 0., 0., 0.))
            }),
            own_ion,
            space,
            fingerprint: identity.0,
            orientation_convention: GeometryOrientation::LaboratoryFixed,
        })
    }

    pub fn positions(&self) -> &[[PairScalar; 3]] {
        &self.positions
    }
    pub fn orientation(&self) -> &[[PairScalar; 3]; 3] {
        &self.orientation
    }
    pub fn space(&self) -> GeometryJetSpace {
        self.space
    }
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }
    pub fn orientation_convention(&self) -> GeometryOrientation {
        self.orientation_convention
    }
}

/// All amplitudes can carry the same affine q directions. No coefficient is
/// inferred from an element label or renormalized to repair charge conservation.
pub struct PairCoefficients {
    pub ionic_charges: Vec<PairScalar>,
    pub gaussian_charges: [PairScalar; 3],
    pub body_dipoles: [[PairScalar; 3]; 3],
    pub local_amplitudes: [PairScalar; 3],
    pub nonlocal_upper: [PairScalar; 36],
    pub internal_energy: PairScalar,
}

impl PairCoefficients {
    pub fn constant(
        ionic_charges: &[f64],
        parameters: &StaticCoreParameters,
        local_amplitudes: [f64; 3],
        internal_energy: f64,
    ) -> Result<Self> {
        parameters.validate()?;
        let c = PairScalar::from_f64;
        Ok(Self {
            ionic_charges: ionic_charges.iter().map(|&x| c(x)).collect(),
            gaussian_charges: parameters.charge_coefficients.map(c),
            body_dipoles: parameters.body_dipoles.map(|p| p.to_array().map(c)),
            local_amplitudes: local_amplitudes.map(c),
            nonlocal_upper: parameters.nonlocal_coefficients.map(c),
            internal_energy: c(internal_energy),
        })
    }

    fn validate(&self, atoms: usize) -> Result<()> {
        if self.ionic_charges.len() != atoms
            || self
                .ionic_charges
                .iter()
                .chain(&self.gaussian_charges)
                .chain(self.body_dipoles.iter().flatten())
                .chain(&self.local_amplitudes)
                .chain(&self.nonlocal_upper)
                .chain(std::iter::once(&self.internal_energy))
                .any(|&x| !finite(x))
            || self.ionic_charges.iter().any(|x| x.value() < 0.0)
        {
            return invalid("core coefficient jets require finite matching ionic and fixed-size amplitude arrays");
        }
        if self
            .ionic_charges
            .iter()
            .any(|&x| x.value() == 0.0 && lanes(x)[1..].iter().any(|&v| v != 0.0))
        {
            return invalid(
                "an auxiliary zero-charge role must stay absent in every derivative lane",
            );
        }
        for lane in 0..4 {
            let sum: f64 = self.gaussian_charges.iter().map(|&x| lanes(x)[lane]).sum();
            let scale: f64 = self
                .gaussian_charges
                .iter()
                .map(|&x| lanes(x)[lane].abs())
                .sum();
            if sum.abs() > 64.0 * f64::EPSILON * scale.max(1.0) {
                return invalid(
                    "every Gaussian correction derivative lane must preserve neutral charge",
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub enum Projectors<'a> {
    /// No nonlocal projector action. All D coefficient lanes must be exactly
    /// zero; this is an explicit operator choice, not a discarded correction.
    None,
    Primitive {
        s_exponents: [f64; 2],
        p_exponents: [f64; 2],
    },
    /// Exact normalized C1s,C2s,Cp0,Cp1,Cp2,H1s,H2s,H3s order.
    ContractedMe8 {
        basis: &'a MolBasis,
        full_role_sites: [usize; 4],
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointIonScope {
    /// Same correction as the saved static provider, plus its three local
    /// Gaussian amplitudes and separately saved E0. Physical point ions belong
    /// to the base Hamiltonian/classical energy and are not repeated.
    CorrectionOnly,
    /// Add the physical-minus-labelled point-ion operator, classical energy
    /// and field dipole. For a base ordinary nuclear derivative engine only.
    DifferenceFromLabelledPointIons,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseFieldConvention {
    /// The base electronic field operator is F.r and its point-ion coupling
    /// is about zero. Include -F.origin S and the matching ionic constant.
    OriginZero,
    /// Both base electronic and point-ion field terms already use the supplied
    /// origin. Do not add the electronic shift or ionic constant a second time.
    AlreadyShifted,
}

pub struct CorePairJet {
    pub operator: [Matrix; 4],
    pub classical_energy: [f64; 4],
    /// Minus the field derivative of the returned classical correction. In the
    /// OriginZero convention this includes the base point-ion origin constant;
    /// it is not a second physical charge distribution to add to the model.
    pub permanent_dipole: [[f64; 4]; 3],
    pub gaussian_charge_correction: [f64; 4],
    pub geometry_space: GeometryJetSpace,
    /// Geometry/coordinate-space identity only. A response context must also
    /// bind the basis, operator dictionary, coefficients, spin and field; this
    /// is not a complete Hamiltonian cache key by itself.
    pub geometry_fingerprint: u64,
    pub base_field_convention: BaseFieldConvention,
    pub point_ion_scope: PointIonScope,
    pub origin: Vec3,
    /// None means a field direction was explicitly seeded. Such a leaf jet is
    /// not a complete electronic field-response provider and must be rejected
    /// by a context restricted to clamped-field nuclear/parameter derivatives.
    pub fixed_external_field: Option<Vec3>,
}

pub struct ElectronicOriginPair {
    /// [Cartesian position component][value,Da,Db,Dab]. These are position
    /// matrices added to the ordinary r operator, not charge-weighted dipoles.
    pub position_matrices: [[Matrix; 4]; 3],
    pub geometry_space: GeometryJetSpace,
    pub geometry_fingerprint: u64,
}

/// Observable correction required in addition to the energy h jets. A response
/// implementation using origin-zero position matrices must add -origin_i*S(R)
/// before forming the electronic dipole. This function does not add energy,
/// nuclear moments or another geometry pullback. Lanes are derivatives, not
/// factorial-normalized Taylor coefficients.
pub fn electronic_position_origin_correction_pair(
    basis: &MolBasis,
    geometry: &PairGeometry,
    origin: Vec3,
    convention: BaseFieldConvention,
) -> Result<ElectronicOriginPair> {
    if basis.nao_sph == 0
        || !basis.ecps.is_empty()
        || basis.n_core_electrons != 0
        || origin.to_array().iter().any(|x| !x.is_finite())
        || basis.shells.iter().any(|s| {
            s.atom >= geometry.positions.len() || s.sph_off + 2 * s.l as usize + 1 > basis.nao_sph
        })
    {
        return invalid("electronic origin correction requires matching finite role coordinates");
    }
    let n = basis.nao_sph;
    for shell in &basis.shells {
        if (shell.center - plain(geometry.positions[shell.atom])).norm() > 1e-12 {
            return invalid("origin correction AO centers must match the supplied geometry");
        }
    }
    let mut position_matrices =
        std::array::from_fn(|_| std::array::from_fn(|_| Matrix::zeros(n, n)));
    if convention == BaseFieldConvention::OriginZero {
        for a in &basis.shells {
            for b in &basis.shells {
                let aa = AnalyticShell::from_placed(a, geometry.positions[a.atom])?;
                let bb = AnalyticShell::from_placed(b, geometry.positions[b.atom])?;
                let block = analytic_basis::overlap(&aa, &bb)?;
                for i in 0..block.shape[0] {
                    for j in 0..block.shape[1] {
                        let v = lanes(block.values[i * block.shape[1] + j]);
                        for axis in 0..3 {
                            for lane in 0..4 {
                                position_matrices[axis][lane][(a.sph_off + i, b.sph_off + j)] =
                                    -origin.get(axis) * v[lane];
                            }
                        }
                    }
                }
            }
        }
    }
    if position_matrices
        .iter()
        .flatten()
        .flat_map(|m| m.as_slice())
        .any(|x| !x.is_finite())
    {
        return invalid("electronic position-origin jet overflowed");
    }
    Ok(ElectronicOriginPair {
        position_matrices,
        geometry_space: geometry.space,
        geometry_fingerprint: geometry.fingerprint,
    })
}

fn lifted<S: Scalar>(shell: &AnalyticShell<S>) -> AnalyticShell<DualN<S, 3>> {
    AnalyticShell {
        l: shell.l,
        center: shell.center.map(DualN::constant),
        exponents: shell
            .exponents
            .iter()
            .copied()
            .map(DualN::constant)
            .collect(),
        coefficients: shell
            .coefficients
            .iter()
            .copied()
            .map(DualN::constant)
            .collect(),
    }
}

fn radial<S: Scalar>(r: [S; 3], exponent: f64) -> S {
    let r2 = r.into_iter().fold(S::from_f64(0.0), |sum, x| sum + x * x);
    let t = r2.scale(exponent);
    if t.value() < 1e-6 {
        let mut sum = S::from_f64(1.0);
        let mut term = sum;
        for k in 1..=12 {
            term = term * (-t).scale(1.0 / k as f64);
            sum = sum + term.scale(1.0 / (2 * k + 1) as f64);
        }
        sum.scale(2.0 * (exponent / PI).sqrt())
    } else {
        let distance = r2.sqrt();
        distance.scale(exponent.sqrt()).erf() / distance
    }
}

fn charge_potential(
    center: [PairScalar; 3],
    point: [PairScalar; 3],
    exponent: f64,
    charge: PairScalar,
    dipole: [PairScalar; 3],
) -> PairScalar {
    let shifted = std::array::from_fn(|axis| {
        DualN::<PairScalar, 3>::constant(point[axis]) - DualN::variable(center[axis], axis)
    });
    let phi = radial(shifted, exponent);
    (0..3).fold(phi.re * charge, |sum, axis| {
        sum + phi.im[axis] * dipole[axis]
    })
}

/// A Gaussian monopole/dipole distribution with differentiated center and
/// laboratory dipole. Widths are immutable; normalization matches charge_pair.
pub struct GaussianDistributionPair {
    pub center: [PairScalar; 3],
    pub exponent: f64,
    pub charge: PairScalar,
    pub dipole: [PairScalar; 3],
}

/// One cross-distribution interaction, including center, rotating-dipole and
/// mixed second derivatives. There is no self energy, pair duplication or
/// implicit half factor. Inner AD lanes differentiate the Gaussian source
/// distribution; the outer pair lanes differentiate the retained pose.
pub fn gaussian_distribution_pair(
    a: &GaussianDistributionPair,
    b: &GaussianDistributionPair,
) -> Result<[f64; 4]> {
    for source in [a, b] {
        if !source.exponent.is_finite()
            || source.exponent <= 0.
            || !finite(source.charge)
            || source
                .center
                .iter()
                .chain(&source.dipole)
                .any(|&v| !finite(v))
        {
            return invalid(
                "Gaussian distribution pair requires finite jets and positive fixed widths",
            );
        }
    }
    type Inner = DualN<PairScalar, 3>;
    type Outer = DualN<Inner, 3>;
    let r = std::array::from_fn(|axis| {
        Outer::variable(Inner::constant(b.center[axis]), axis)
            - Outer::constant(Inner::variable(a.center[axis], axis))
    });
    // This form avoids overflow in the product of two valid finite widths.
    let eta = if a.exponent <= b.exponent {
        a.exponent / (1. + a.exponent / b.exponent)
    } else {
        b.exponent / (1. + b.exponent / a.exponent)
    };
    let value = radial(r, eta);
    let mut energy = value.re.re * a.charge * b.charge;
    for i in 0..3 {
        energy = energy
            + value.re.im[i] * a.dipole[i] * b.charge
            + value.im[i].re * b.dipole[i] * a.charge;
        for j in 0..3 {
            energy = energy + value.im[j].im[i] * a.dipole[i] * b.dipole[j];
        }
    }
    if !finite(energy) {
        return invalid("Gaussian distribution pair jet overflowed");
    }
    Ok(lanes(energy))
}

fn binomial(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |v, i| v * (n - i) as f64 / (i + 1) as f64)
}

/// Exact Cartesian triple-Gaussian integral, contracted/spherical transformed by
/// the existing analytic_basis machinery. The local Gaussian is unnormalized,
/// matching gaussian::gaussian_potential and its Hartree amplitude convention.
fn local_primitive<S: Scalar>(a: Primitive<S>, b: Primitive<S>, center: [S; 3], eta: f64) -> S {
    let p = a.exponent + b.exponent + S::from_f64(eta);
    let product: [S; 3] = std::array::from_fn(|axis| {
        (a.exponent * a.center[axis] + b.exponent * b.center[axis] + center[axis].scale(eta)) / p
    });
    let exponent = (0..3).fold(S::from_f64(0.0), |sum, axis| {
        let ab = a.center[axis] - b.center[axis];
        let ac = a.center[axis] - center[axis];
        let bc = b.center[axis] - center[axis];
        sum + (a.exponent * b.exponent * ab * ab
            + (a.exponent * ac * ac + b.exponent * bc * bc).scale(eta))
            / p
    });
    let mut result = (-exponent).exp();
    for axis in 0..3 {
        let la = a.angular[axis];
        let lb = b.angular[axis];
        let mut value = S::from_f64(0.0);
        for i in 0..=la {
            for j in 0..=lb {
                if (i + j) % 2 != 0 {
                    continue;
                }
                let k = (i + j) / 2;
                let df = (1..=k).fold(1.0, |v, l| v * (2 * l - 1) as f64);
                value = value
                    + (product[axis] - a.center[axis]).powi((la - i) as i32)
                        * (product[axis] - b.center[axis]).powi((lb - j) as i32)
                        / (p.scale(2.0)).powi(k as i32)
                        * S::from_f64(binomial(la, i) * binomial(lb, j) * df);
            }
        }
        result = result * value * (S::from_f64(PI) / p).sqrt();
    }
    result
}

pub struct LocalGaussianPair {
    /// Complete amplitude-weighted operator lanes [value,Da,Db,Dab].
    pub operator: [Matrix; 4],
    pub geometry_space: GeometryJetSpace,
    pub geometry_fingerprint: u64,
    pub fixed_exponent: f64,
}

/// One arbitrary spherical local source, amplitude*exp(-eta*|r-center|^2).
/// Center jets may be supplied from Atom, Fixed or Attached center rules and
/// must use the SAME coordinate space as geometry. Their chain-rule curvature
/// is supplied explicitly; this helper neither freezes nor reconstructs it.
/// No steric scalar energy or new model dictionary is inferred here. This is
/// an operator building block; the saved few-center recipe still has 3 slots.
pub fn local_gaussian_pair(
    basis: &MolBasis,
    geometry: &PairGeometry,
    center: [PairScalar; 3],
    exponent: f64,
    amplitude: PairScalar,
) -> Result<LocalGaussianPair> {
    let n = basis.nao_sph;
    if n == 0
        || !basis.ecps.is_empty()
        || basis.n_core_electrons != 0
        || !exponent.is_finite()
        || exponent <= 0.
        || center.iter().any(|&x| !finite(x))
        || !finite(amplitude)
        || basis
            .shells
            .iter()
            .any(|s| s.atom >= geometry.positions.len() || s.sph_off + 2 * s.l as usize + 1 > n)
    {
        return invalid("spherical local jet requires finite center/amplitude lanes, a fixed positive exponent and matching all-electron basis");
    }
    for shell in &basis.shells {
        if (shell.center - plain(geometry.positions[shell.atom])).norm() > 1e-12 {
            return invalid("spherical local jet AO centers must match the supplied geometry");
        }
    }
    let mut operator = std::array::from_fn(|_| Matrix::zeros(n, n));
    for a in &basis.shells {
        for b in &basis.shells {
            let aa = AnalyticShell::from_placed(a, geometry.positions[a.atom])?;
            let bb = AnalyticShell::from_placed(b, geometry.positions[b.atom])?;
            let [block] = analytic_basis::contracted_components([&aa, &bb], |[pa, pb]| {
                Ok([local_primitive(pa, pb, center, exponent)])
            })?;
            for i in 0..block.shape[0] {
                for j in 0..block.shape[1] {
                    let value = lanes(block.values[i * block.shape[1] + j] * amplitude);
                    for lane in 0..4 {
                        operator[lane][(a.sph_off + i, b.sph_off + j)] = value[lane];
                    }
                }
            }
        }
    }
    if operator
        .iter()
        .flat_map(|m| m.as_slice())
        .any(|x| !x.is_finite())
    {
        return invalid("spherical local second-order jet overflowed");
    }
    Ok(LocalGaussianPair {
        operator,
        geometry_space: geometry.space,
        geometry_fingerprint: geometry.fingerprint,
        fixed_exponent: exponent,
    })
}

fn rotation(orientation: [[PairScalar; 3]; 3], primitive: bool) -> Vec<PairScalar> {
    let c = PairScalar::from_f64;
    let mut r = vec![c(0.0); 64];
    for i in 0..8 {
        r[8 * i + i] = c(1.0);
    }
    let t = aos::cart_to_sph(1);
    for offset in if primitive { vec![2, 5] } else { vec![2] } {
        for i in 0..3 {
            for j in 0..3 {
                r[8 * (offset + i) + offset + j] = (0..3)
                    .flat_map(|a| (0..3).map(move |b| (a, b)))
                    .fold(c(0.0), |sum, (a, b)| {
                        sum + orientation[b][a].scale(t[(a, i)] * t[(b, j)])
                    });
            }
        }
    }
    r
}

fn add_block(
    target: &mut [PairScalar],
    n: usize,
    a: usize,
    b: usize,
    shape: [usize; 2],
    values: &[PairScalar],
) {
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            target[(a + i) * n + b + j] = target[(a + i) * n + b + j] + values[i * shape[1] + j];
        }
    }
}

/// Evaluate both directions analytically in the same scalar energy. The caller
/// seeds either Cartesian coordinates or affine q coefficients (or one of each).
/// Auxiliary zero-charge sites remain distinct from physical ions and AO sites.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_pair(
    molecule: &Molecule,
    basis: &MolBasis,
    geometry: &PairGeometry,
    coefficients: &PairCoefficients,
    charge_exponents: [f64; 3],
    local_exponents: [f64; 3],
    projectors: Projectors<'_>,
    origin: Vec3,
    field: [PairScalar; 3],
    point_scope: PointIonScope,
    base_field_convention: BaseFieldConvention,
) -> Result<CorePairJet> {
    let atoms = molecule.len();
    let n = basis.nao_sph;
    let c = PairScalar::from_f64;
    coefficients.validate(atoms)?;
    if geometry.orientation_convention == GeometryOrientation::LaboratoryFixed
        && (coefficients
            .body_dipoles
            .iter()
            .flatten()
            .any(|&x| lanes(x).into_iter().any(|v| v != 0.))
            || !matches!(projectors, Projectors::None))
    {
        return invalid(
            "laboratory spherical geometry cannot supply body-fixed dipoles or projectors",
        );
    }
    if matches!(projectors, Projectors::None)
        && coefficients
            .nonlocal_upper
            .iter()
            .any(|&x| lanes(x).into_iter().any(|v| v != 0.))
    {
        return invalid("Projectors::None requires every nonlocal coefficient derivative lane to be exactly zero");
    }
    if n == 0
        || geometry.positions.len() != atoms
        || basis
            .shells
            .iter()
            .any(|s| s.atom >= atoms || s.sph_off + 2 * s.l as usize + 1 > n)
        || geometry.own_ion >= atoms
        || !basis.ecps.is_empty()
        || basis.n_core_electrons != 0
        || charge_exponents
            .iter()
            .chain(&local_exponents)
            .any(|x| !x.is_finite() || *x <= 0.0)
        || field.iter().any(|&x| !finite(x))
        || origin.to_array().iter().any(|x| !x.is_finite())
    {
        return invalid("static core jet basis, role dimensions, widths or field are incompatible");
    }
    if molecule
        .atoms
        .iter()
        .zip(&geometry.positions)
        .any(|(atom, &position)| (atom.position - plain(position)).norm() > 1e-12)
    {
        return invalid("core geometry jets and labelled molecule coordinates must match");
    }
    for shell in &basis.shells {
        if (shell.center - plain(geometry.positions[shell.atom])).norm() > 1e-12 {
            return invalid("core jet basis centers must match the supplied labelled geometry");
        }
    }
    let own = geometry.own_ion;
    let center = geometry.positions[own];
    let dipoles: [[PairScalar; 3]; 3] = std::array::from_fn(|s| {
        std::array::from_fn(|world| {
            (0..3).fold(c(0.0), |sum, local| {
                sum + geometry.orientation[local][world] * coefficients.body_dipoles[s][local]
            })
        })
    });
    let _operator_workspace = crate::resources::reserve(
        crate::resources::checked_bytes(
            &[n, n, std::mem::size_of::<PairScalar>()],
            "static-core pair operator workspace",
        )?,
        "static-core pair operator workspace",
    )?;
    let mut values = vec![c(0.0); n * n];
    for a in &basis.shells {
        for b in &basis.shells {
            let aa = AnalyticShell::from_placed(a, geometry.positions[a.atom])?;
            let bb = AnalyticShell::from_placed(b, geometry.positions[b.atom])?;
            for s in 0..3 {
                let inner_center = std::array::from_fn(|axis| {
                    DualN::<PairScalar, 3>::variable(center[axis], axis)
                });
                let block = analytic_basis::attraction(
                    &lifted(&aa),
                    &lifted(&bb),
                    &[AnalyticNucleus {
                        center: inner_center,
                        charge: DualN::constant(c(1.0)),
                    }],
                    GaussianKernel::LongRange {
                        omega: DualN::constant(c(charge_exponents[s].sqrt())),
                    },
                    // Two outer pose derivatives plus the inner source-center
                    // derivative defining a dipole require the third Boys jet.
                    // Order two preserves values/first derivatives but omits a
                    // term in every moving-center dipole Hessian block.
                    3,
                )?;
                let charge_values: Vec<_> = block
                    .values
                    .iter()
                    .map(|v| {
                        (0..3).fold(v.re * coefficients.gaussian_charges[s], |sum, axis| {
                            sum + v.im[axis] * dipoles[s][axis]
                        })
                    })
                    .collect();
                add_block(
                    &mut values,
                    n,
                    a.sph_off,
                    b.sph_off,
                    block.shape,
                    &charge_values,
                );
                let [local] = analytic_basis::contracted_components([&aa, &bb], |[pa, pb]| {
                    Ok([local_primitive(pa, pb, center, local_exponents[s])])
                })?;
                let local_values: Vec<_> = local
                    .values
                    .iter()
                    .map(|&v| v * coefficients.local_amplitudes[s])
                    .collect();
                add_block(
                    &mut values,
                    n,
                    a.sph_off,
                    b.sph_off,
                    local.shape,
                    &local_values,
                );
            }
            if matches!(point_scope, PointIonScope::DifferenceFromLabelledPointIons) {
                let nuclei: Vec<_> = molecule
                    .atoms
                    .iter()
                    .enumerate()
                    .map(|(i, atom)| AnalyticNucleus {
                        center: geometry.positions[i],
                        charge: coefficients.ionic_charges[i] - c(f64::from(atom.z)),
                    })
                    .collect();
                let block =
                    analytic_basis::attraction(&aa, &bb, &nuclei, GaussianKernel::Coulomb, 2)?;
                add_block(
                    &mut values,
                    n,
                    a.sph_off,
                    b.sph_off,
                    block.shape,
                    &block.values,
                );
            }
            if base_field_convention == BaseFieldConvention::OriginZero {
                let shift = -(0..3).fold(c(0.0), |sum, axis| {
                    sum + field[axis].scale(origin.get(axis))
                });
                let overlap = analytic_basis::overlap(&aa, &bb)?;
                let shifted: Vec<_> = overlap.values.iter().map(|&v| v * shift).collect();
                add_block(
                    &mut values,
                    n,
                    a.sph_off,
                    b.sph_off,
                    overlap.shape,
                    &shifted,
                );
            }
        }
    }
    let (shells, primitive): (Vec<(usize, AnalyticShell<PairScalar>)>, bool) = match projectors {
        Projectors::None => (Vec::new(), false),
        Projectors::Primitive {
            s_exponents,
            p_exponents,
        } => {
            if s_exponents
                .iter()
                .chain(&p_exponents)
                .any(|x| !x.is_finite() || *x <= 0.0)
            {
                return invalid("projector exponents must be positive and fixed");
            }
            let shells = [
                (0, 0, s_exponents[0]),
                (1, 0, s_exponents[1]),
                (2, 1, p_exponents[0]),
                (5, 1, p_exponents[1]),
            ]
            .into_iter()
            .map(|(off, l, e)| {
                Ok((
                    off,
                    AnalyticShell::normalized(l, center, vec![c(e)], vec![c(1.0)])?,
                ))
            })
            .collect::<Result<_>>()?;
            (shells, true)
        }
        Projectors::ContractedMe8 {
            basis: right,
            full_role_sites: sites,
        } => {
            if right.nao_sph != 8
                || right.shells.len() != 6
                || sites.iter().any(|&i| i >= atoms)
                || (0..4).any(|i| (0..i).any(|j| sites[i] == sites[j]))
                || right
                    .shells
                    .iter()
                    .zip([0, 0, 1, 0, 0, 0])
                    .any(|(s, l)| s.l != l)
                || right
                    .shells
                    .iter()
                    .zip([0, 1, 2, 5, 6, 7])
                    .any(|(s, off)| s.sph_off != off)
                || right.shells.iter().any(|s| s.atom >= 4)
                || right
                    .shells
                    .iter()
                    .zip([0, 0, 0, 1, 2, 3])
                    .any(|(s, atom)| s.atom != atom)
                || !right.ecps.is_empty()
                || right.n_core_electrons != 0
                || molecule.atoms[sites[0]].z != 6
                || sites[1..].iter().any(|&i| molecule.atoms[i].z != 1)
            {
                return invalid("contracted Me8 shell order, labels or fixed role sites changed");
            }
            let mut shells = Vec::new();
            for shell in &right.shells {
                if (shell.center - plain(geometry.positions[sites[shell.atom]])).norm() > 1e-12 {
                    return invalid(
                        "contracted projector centers must match their explicit full roles",
                    );
                }
                shells.push((
                    shell.sph_off,
                    AnalyticShell::from_placed(shell, geometry.positions[sites[shell.atom]])?,
                ));
            }
            (shells, false)
        }
    };
    if !shells.is_empty() {
        let _projector_workspace = crate::resources::reserve(
            crate::resources::checked_bytes(
                &[2, n, 8, std::mem::size_of::<PairScalar>()],
                "static-core pair projector workspace",
            )?,
            "static-core pair projector workspace",
        )?;
        let mut overlaps = vec![c(0.0); n * 8];
        for a in &basis.shells {
            let aa = AnalyticShell::from_placed(a, geometry.positions[a.atom])?;
            for (off, beta) in &shells {
                let block = analytic_basis::overlap(&aa, beta)?;
                for i in 0..block.shape[0] {
                    for j in 0..block.shape[1] {
                        overlaps[(a.sph_off + i) * 8 + *off + j] =
                            block.values[i * block.shape[1] + j];
                    }
                }
            }
        }
        let rotation = rotation(geometry.orientation, primitive);
        let body: Vec<_> = (0..n * 8)
            .map(|index| {
                (0..8).fold(c(0.0), |sum, k| {
                    sum + overlaps[(index / 8) * 8 + k] * rotation[k * 8 + index % 8]
                })
            })
            .collect();
        let mut d = vec![c(0.0); 64];
        let mut entry = 0;
        for u in 0..8 {
            for v in u..8 {
                d[8 * u + v] = coefficients.nonlocal_upper[entry];
                d[8 * v + u] = coefficients.nonlocal_upper[entry];
                entry += 1;
            }
        }
        for i in 0..n {
            for j in 0..n {
                values[i * n + j] = values[i * n + j]
                    + (0..8).fold(c(0.0), |sum, u| {
                        sum + (0..8).fold(c(0.0), |inner, v| {
                            inner + body[8 * i + u] * d[8 * u + v] * body[8 * j + v]
                        })
                    });
            }
        }
    }
    let mut classical = coefficients.internal_energy;
    let mut moment = [c(0.0); 3];
    let mut total = c(0.0);
    for s in 0..3 {
        total = total + coefficients.gaussian_charges[s];
        for axis in 0..3 {
            moment[axis] = moment[axis]
                + (center[axis] - c(origin.get(axis))) * coefficients.gaussian_charges[s]
                + dipoles[s][axis];
        }
        for i in 0..atoms {
            if i != own {
                classical = classical
                    + coefficients.ionic_charges[i]
                        * charge_potential(
                            center,
                            geometry.positions[i],
                            charge_exponents[s],
                            coefficients.gaussian_charges[s],
                            dipoles[s],
                        );
            }
        }
    }
    if matches!(point_scope, PointIonScope::DifferenceFromLabelledPointIons) {
        for i in 0..atoms {
            let delta = coefficients.ionic_charges[i] - c(f64::from(molecule.atoms[i].z));
            for axis in 0..3 {
                moment[axis] =
                    moment[axis] + delta * (geometry.positions[i][axis] - c(origin.get(axis)));
            }
            for j in 0..i {
                let r2 = (0..3).fold(c(0.0), |sum, axis| {
                    let r = geometry.positions[i][axis] - geometry.positions[j][axis];
                    sum + r * r
                });
                if r2.value() <= 0.0 {
                    return invalid("distinct point-ion roles must not coincide");
                }
                classical = classical
                    + (coefficients.ionic_charges[i] * coefficients.ionic_charges[j]
                        - c(f64::from(molecule.atoms[i].z) * f64::from(molecule.atoms[j].z)))
                        / r2.sqrt();
            }
        }
    }
    if base_field_convention == BaseFieldConvention::OriginZero {
        let base_point_charge =
            if matches!(point_scope, PointIonScope::DifferenceFromLabelledPointIons) {
                c(molecule.atoms.iter().map(|atom| f64::from(atom.z)).sum())
            } else {
                coefficients
                    .ionic_charges
                    .iter()
                    .copied()
                    .fold(c(0.0), |sum, z| sum + z)
            };
        // The returned moment is minus the field derivative of this classical
        // correction relative to the declared base convention. This constant
        // combines with the electronic overlap shift; it is not a new source.
        for axis in 0..3 {
            moment[axis] = moment[axis] - base_point_charge.scale(origin.get(axis));
        }
    }
    for axis in 0..3 {
        classical = classical - field[axis] * moment[axis];
    }
    if values.iter().any(|&x| !finite(x))
        || !finite(classical)
        || moment.iter().any(|&x| !finite(x))
    {
        return invalid("core second-order jet overflowed");
    }
    let mut operator = std::array::from_fn(|_| Matrix::zeros(n, n));
    for i in 0..n {
        for j in 0..n {
            for lane in 0..4 {
                operator[lane][(i, j)] = lanes(values[i * n + j])[lane];
            }
        }
    }
    Ok(CorePairJet {
        operator,
        classical_energy: lanes(classical),
        permanent_dipole: moment.map(lanes),
        gaussian_charge_correction: lanes(total),
        geometry_space: geometry.space,
        geometry_fingerprint: geometry.fingerprint,
        base_field_convention,
        point_ion_scope: point_scope,
        origin,
        fixed_external_field: if field
            .iter()
            .all(|&x| lanes(x)[1..].iter().all(|&v| v == 0.0))
        {
            Some(plain(field))
        } else {
            None
        },
    })
}

#[cfg(test)]
mod m1_field_value_oracle {
    use super::super::local::{LocalCenter, LocalGaussianSpec};
    use super::super::{ChargeChart, ChargeParameter, ContinuousSpeciesModel, SpeciesFunctional};
    use super::*;
    use crate::basis::data;
    use crate::grid::{GridOptions, MolecularGrid};
    use crate::integrals::{dipole_matrices, kinetic_matrix, nuclear_matrix};
    use crate::pseudo_sub::reduction_geometry::BodyFrameKinematics;
    use crate::scf::{EriMode, Guess, ScfOptions};
    use crate::system::Atom;

    #[test]
    fn nonzero_origin_extra_jets_reconstruct_actual_m1_hcore_and_classical_energy() {
        let points = [
            (6, [1.35, 0.17, -0.12]),
            (1, [2.07, 1.98, 0.31]),
            (1, [2.18, -0.92, 1.45]),
            (1, [2.01, -0.84, -1.67]),
            (6, [-1.46, -0.14, 0.23]),
            (1, [-2.13, 1.69, 0.68]),
            (1, [-2.28, -1.57, 1.31]),
            (1, [-2.19, -0.43, -1.65]),
        ];
        let molecule = Molecule::new(
            points
                .into_iter()
                .map(|(z, p)| Atom {
                    z,
                    position: Vec3::new(p[0], p[1], p[2]),
                })
                .collect(),
        );
        let mut basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
        basis.shells.retain(|s| [0, 4, 5, 6, 7].contains(&s.atom));
        let mut sph = 0;
        let mut cart = 0;
        for shell in &mut basis.shells {
            shell.sph_off = sph;
            shell.cart_off = cart;
            sph += 2 * shell.l as usize + 1;
            cart += (shell.l as usize + 1) * (shell.l as usize + 2) / 2;
        }
        basis.nao_sph = sph;
        basis.nao_cart = cart;
        assert_eq!(sph, 13);
        let z = vec![Vec3::zero(); 8];
        let geometry = PairGeometry::new(
            &molecule,
            [4, 0, 5],
            &z,
            &z,
            &z,
            GeometryJetSpace::IndependentRoles,
        )
        .unwrap();
        let parameters = StaticCoreParameters {
            charge_exponents: [0.3, 0.9, 2.4],
            charge_coefficients: [0.17, -0.23, 0.06],
            body_dipoles: [
                Vec3::new(0.07, -0.04, 0.02),
                Vec3::new(-0.05, 0.03, 0.06),
                Vec3::new(0.02, 0.01, -0.08),
            ],
            projector_s_exponents: [0.35, 1.3],
            projector_p_exponents: [0.25, 1.1],
            nonlocal_coefficients: std::array::from_fn(|i| {
                (0.031 + 0.0007 * i as f64) * if i % 3 == 0 { -1.0 } else { 1.0 }
            }),
        };
        let ionic = [1., 0., 0., 0., 6., 1., 1., 1.];
        let local = [-0.12, 0.04, 0.09];
        let widths = [0.43, 1.1, 2.7];
        let origin = Vec3::new(0.11, -0.07, 0.19);
        let field = Vec3::new(0.0013, -0.0017, 0.0009);
        let mut orientation_jacobian = Matrix::zeros(9, 24);
        let local_jacobian = frame_jacobian(
            molecule.atoms[4].position,
            molecule.atoms[0].position,
            molecule.atoms[5].position,
        );
        for row in 0..9 {
            for (slot, atom) in [4, 0, 5].into_iter().enumerate() {
                for axis in 0..3 {
                    orientation_jacobian[(row, 3 * atom + axis)] =
                        local_jacobian[(row, 3 * slot + axis)];
                }
            }
        }
        let mut center_jacobian = Matrix::zeros(3, 24);
        for axis in 0..3 {
            center_jacobian[(axis, axis)] = 1.;
        }
        let frame = BodyFrameKinematics {
            center: molecule.atoms[0].position,
            orientation: crate::math::Mat3::from_columns(
                plain(geometry.orientation[0]),
                plain(geometry.orientation[1]),
                plain(geometry.orientation[2]),
            ),
            orientation_jacobian,
            center_jacobian,
        };
        let auxiliary: Vec<_> = (0..8).map(|i| [1, 2, 3].contains(&i)).collect();
        let correction = super::super::static_core::static_core_correction_with_auxiliary_roles(
            &basis,
            &molecule
                .atoms
                .iter()
                .map(|a| a.position)
                .collect::<Vec<_>>(),
            &ionic,
            0,
            &frame,
            &parameters,
            origin,
            &auxiliary,
        )
        .unwrap();
        let chart = ChargeChart {
            reference_charges: ionic.to_vec(),
            conserve_total_charge: true,
            parameters: vec![ChargeParameter {
                name: "fixed-value-oracle".into(),
                unit: "dimensionless".into(),
                scale: 1.,
                bounds: [-1., 1.],
                charge_direction: vec![0.; 8],
            }],
        };
        let grid = MolecularGrid::build(
            &molecule,
            &GridOptions {
                level: 1,
                weight_cutoff: 0.,
                ..GridOptions::default()
            },
        )
        .unwrap();
        let options = ScfOptions {
            guess: Guess::Core,
            max_iter: 300,
            e_tol: 1e-13,
            p_tol: 1e-12,
            conv_tol_grad: 1e-11,
            eri_mode: EriMode::InCore { limit_mb: 64 },
            ..ScfOptions::default()
        };
        let terms = (0..3)
            .map(|i| LocalGaussianSpec {
                center: LocalCenter::Atom(0),
                exponent: widths[i],
                reference_amplitude: local[i],
                amplitude_direction: vec![0.],
                reference_steric_strength: 0.,
                steric_direction: vec![0.],
                environment_exponents: vec![1.; 8],
                environment_hardness: vec![0.; 8],
                excluded_atoms: vec![true; 8],
            })
            .collect();
        let model = ContinuousSpeciesModel::new_with_auxiliary_roles(
            molecule.clone(),
            basis.clone(),
            grid,
            10,
            chart,
            SpeciesFunctional::Pbe,
            options,
            origin,
            auxiliary,
        )
        .unwrap()
        .with_local_potentials(terms)
        .unwrap()
        .with_static_core_correction(correction)
        .unwrap()
        .with_fragment_internal_energy(-38.2)
        .unwrap();
        let state = model.evaluate(&[0.], field, None).unwrap();
        let coefficients = PairCoefficients::constant(&ionic, &parameters, local, -38.2).unwrap();
        let extra = evaluate_pair(
            &molecule,
            &basis,
            &geometry,
            &coefficients,
            parameters.charge_exponents,
            widths,
            Projectors::Primitive {
                s_exponents: parameters.projector_s_exponents,
                p_exponents: parameters.projector_p_exponents,
            },
            origin,
            field.to_array().map(PairScalar::from_f64),
            PointIonScope::DifferenceFromLabelledPointIons,
            BaseFieldConvention::OriginZero,
        )
        .unwrap();
        let mut ordinary = kinetic_matrix(&basis);
        let nuclear = nuclear_matrix(&basis, &molecule, None);
        let dipoles = dipole_matrices(&basis, Vec3::zero());
        for i in 0..sph {
            for j in 0..sph {
                ordinary[(i, j)] += nuclear[(i, j)]
                    + (0..3)
                        .map(|axis| field.get(axis) * dipoles[axis][(i, j)])
                        .sum::<f64>();
                assert!(
                    (ordinary[(i, j)] + extra.operator[0][(i, j)] - state.scf.h_core[(i, j)]).abs()
                        < 2e-10
                );
            }
        }
        let ordinary_classical = basis.nuclear_repulsion(&molecule)
            - molecule
                .atoms
                .iter()
                .map(|a| f64::from(a.z) * a.position.dot(field))
                .sum::<f64>();
        assert!(
            (ordinary_classical + extra.classical_energy[0] - state.classical_energy()).abs()
                < 2e-10
        );
        assert!(
            (state.energy() - state.electronic_energy() - state.classical_energy()).abs() < 1e-11
        );
    }
}

#[cfg(test)]
mod moving_dipole_regression {
    use super::*;
    use crate::basis::data;
    use crate::pseudo_sub::reduction_kernels::{charge_operator, GaussianCharge};
    use crate::system::Atom;

    fn configuration(q: &[f64; 6]) -> (Molecule, MolBasis) {
        let molecule = Molecule::new(
            (0..2)
                .map(|a| Atom {
                    z: 1,
                    position: Vec3::new(q[3 * a], q[3 * a + 1], q[3 * a + 2]),
                })
                .collect(),
        );
        let basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
        (molecule, basis)
    }

    fn density() -> Matrix {
        let mut p = Matrix::zeros(2, 2);
        p[(0, 0)] = 0.17;
        p[(1, 1)] = 0.23;
        p[(0, 1)] = -0.08;
        p[(1, 0)] = -0.08;
        p
    }

    // Independent retained f64 value/first-derivative kernels. The source
    // follows atom zero, so both its AO and source-center terms are required.
    fn independent(q: &[f64; 6]) -> (f64, Vec<f64>) {
        let (molecule, basis) = configuration(q);
        let operator = charge_operator(
            &basis,
            2,
            &[GaussianCharge {
                center: molecule.atoms[0].position,
                exponent: 0.7,
                charge: 0.,
                dipole: Vec3::new(0.13, -0.09, 0.07),
            }],
        )
        .unwrap();
        let p = density();
        let gradient = (0..6)
            .map(|i| {
                let mut value = p.frobenius_dot(&operator.atom_derivatives[i / 3][i % 3]);
                if i / 3 == 0 {
                    value += p.frobenius_dot(&operator.center_derivatives[0][i % 3]);
                }
                value
            })
            .collect();
        (p.frobenius_dot(&operator.operator), gradient)
    }

    fn electronic_jet(q: &[f64; 6], a: &[f64; 6], b: &[f64; 6]) -> [f64; 4] {
        let (molecule, basis) = configuration(q);
        let positions = (0..2)
            .map(|atom| {
                std::array::from_fn(|k| seed(q[3 * atom + k], a[3 * atom + k], b[3 * atom + k], 0.))
            })
            .collect();
        let rotation = std::array::from_fn(|i| {
            std::array::from_fn(|j| seed(f64::from(u8::from(i == j)), 0., 0., 0.))
        });
        let geometry = PairGeometry::rigid_pose(&molecule, 0, positions, rotation, 1).unwrap();
        let parameters = StaticCoreParameters {
            charge_exponents: [0.7, 1.1, 1.9],
            charge_coefficients: [0.; 3],
            body_dipoles: [Vec3::new(0.13, -0.09, 0.07), Vec3::zero(), Vec3::zero()],
            projector_s_exponents: [0.4, 1.3],
            projector_p_exponents: [0.3, 1.2],
            nonlocal_coefficients: [0.; 36],
        };
        let coefficients = PairCoefficients::constant(&[1., 1.], &parameters, [0.; 3], 0.).unwrap();
        let result = evaluate_pair(
            &molecule,
            &basis,
            &geometry,
            &coefficients,
            parameters.charge_exponents,
            [0.4, 0.9, 1.7],
            Projectors::None,
            Vec3::zero(),
            [PairScalar::from_f64(0.); 3],
            PointIonScope::CorrectionOnly,
            BaseFieldConvention::AlreadyShifted,
        )
        .unwrap();
        std::array::from_fn(|lane| density().frobenius_dot(&result.operator[lane]))
    }

    #[test]
    fn moving_gaussian_dipole_requires_the_third_inner_boys_lane() {
        let q = [-0.7, 0.23, -0.15, 1.1, -0.31, 0.42];
        let zero = [0.; 6];
        let steps = [0.04, 0.02, 0.01, 0.005];
        let (value, _) = independent(&q);
        assert!((value - electronic_jet(&q, &zero, &zero)[0]).abs() < 1e-12);
        let mut failures = Vec::new();
        for j in 0..6 {
            let mut direction = zero;
            direction[j] = 1.;
            let first = electronic_jet(&q, &direction, &zero)[1];
            let mut energy_errors = [0.; 4];
            let mut hessian_errors = [[0.; 4]; 6];
            for (k, &h) in steps.iter().enumerate() {
                let mut plus = q;
                let mut minus = q;
                plus[j] += h;
                minus[j] -= h;
                let (ep, gp) = independent(&plus);
                let (em, gm) = independent(&minus);
                energy_errors[k] = ((ep - em) / (2. * h) - first).abs();
                for i in 0..6 {
                    let mut other = zero;
                    other[i] = 1.;
                    let analytic = electronic_jet(&q, &other, &direction)[3];
                    hessian_errors[i][k] = ((gp[i] - gm[i]) / (2. * h) - analytic).abs();
                }
            }
            let label = format!("moving dipole gradient {j}");
            if !crate::acceptance::components::component_fit_of_order(
                &label,
                &steps,
                &energy_errors,
                2.,
            ) {
                failures.push(label);
            }
            for (i, errors) in hessian_errors.iter().enumerate() {
                let label = format!("moving dipole Hessian {i},{j}");
                if !crate::acceptance::components::component_fit_of_order(
                    &label, &steps, errors, 2.,
                ) {
                    failures.push(label);
                }
            }
        }
        assert!(
            failures.is_empty(),
            "Unresolved dipole derivative components: {failures:?}"
        );
    }
}
