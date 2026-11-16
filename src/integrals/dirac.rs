// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Finite-Gaussian-nucleus Dirac integrals with kinetic or magnetic balance.
//!
//! The small basis is (sigma.p)/(2c) times each large-component spatial
//! Gaussian. The Hamiltonian excludes the electron rest energy. Its blocks
//! are [V,T;T,W/(4c^2)-T], with W=(sigma.p) V (sigma.p), and the overlap is
//! diag(S,T/(2c^2)). The complete spin-orbit part of W is retained. Nested
//! center duals differentiate the two basis endpoints independently; nuclear
//! positions remain independent of those inner seeds. Outer Scalar jets
//! retain moving centers, nuclear sizes, charges and basis parameters.
//! In a magnetic field p is replaced by pi=p+A in both the small basis and
//! the Hamiltonian. The resulting kinetic and overlap spinor blocks are
//! complex and include every vector-potential cross term and quadratic term.

use super::analytic::{boys_jet, GaussianKernel};
use super::analytic_basis::{self, AnalyticNucleus, AnalyticShell, IntegralBlock};
use crate::dual::{DualN, Scalar, Taylor};
use crate::linalg::Matrix;
use crate::periodic::BlochMatrix;
use crate::{DftError, Result};
use rayon::prelude::*;
use std::f64::consts::PI;

mod current;
pub use current::{electronic_current_exchange, DiracCurrentInteraction};
mod coupled;
pub use coupled::{electron_quantum_coulomb, BalancedCoulombFields, BalancedCoulombInput};
mod electronic;
pub use electronic::{electronic_coulomb_exchange, DiracCoulombFields};
pub(crate) use electronic::{electronic_screened_exchange, BalancedPairCache};
mod london;
mod london_eri;
pub use london_eri::{LondonCoulombTaylor, LondonCurrentTaylor};
mod nuclear_magnetic;
pub use nuclear_magnetic::{nuclear_magnetic_dipole_vertices, GaussianMagnetization};

/// Gaussian charge density Z*(omega^2/pi)^(3/2)*exp(-omega^2*r^2),
/// with omega=sqrt(3/2)/rms_radius. Positions and RMS radius are Bohr.
#[derive(Clone, Copy, Debug)]
pub struct GaussianNucleus<S: Scalar> {
    pub center: [S; 3],
    pub charge: S,
    pub rms_radius: S,
}

/// Row-major matrices in [large up, large down, small up, small down]
/// spin-major spherical AO order. Every matrix entry retains its Scalar.
#[derive(Clone, Debug)]
pub struct BalancedDiracIntegrals<S: Scalar> {
    pub spatial_overlap: IntegralBlock<S, 2>,
    pub spatial_overlap_imaginary: IntegralBlock<S, 2>,
    /// Unphased spatial kinetic matrix, retained as a reference diagnostic.
    pub kinetic: IntegralBlock<S, 2>,
    pub potential: IntegralBlock<S, 2>,
    pub potential_imaginary: IntegralBlock<S, 2>,
    /// W/(4c^2) in the two-component small-basis block.
    pub small_potential_real: IntegralBlock<S, 2>,
    pub small_potential_imaginary: IntegralBlock<S, 2>,
    pub hamiltonian_real: IntegralBlock<S, 2>,
    pub hamiltonian_imaginary: IntegralBlock<S, 2>,
    pub overlap: IntegralBlock<S, 2>,
    pub overlap_imaginary: IntegralBlock<S, 2>,
    /// (sigma.pi)^2/2 in the spinor large-basis space, Hartree.
    pub balanced_kinetic_real: IntegralBlock<S, 2>,
    pub balanced_kinetic_imaginary: IntegralBlock<S, 2>,
    /// Interaction of distinct Gaussian nuclear charge distributions;
    /// individual nuclear self-energies are excluded.
    pub nuclear_repulsion: S,
}

impl<S: Scalar> BalancedDiracIntegrals<S> {
    /// Balanced electron–nuclear operator as real/imaginary parts.
    pub fn nuclear_potential(&self) -> Result<[IntegralBlock<S, 2>; 2]> {
        let n = self.potential.shape[0];
        let mut re = block(4 * n, 4 * n)?;
        let mut im = block(4 * n, 4 * n)?;
        for spin in 0..2 {
            for i in 0..n {
                for j in 0..n {
                    re.values[(spin * n + i) * 4 * n + spin * n + j] =
                        self.potential.values[i * n + j];
                    im.values[(spin * n + i) * 4 * n + spin * n + j] =
                        self.potential_imaginary.values[i * n + j];
                }
            }
        }
        for i in 0..2 * n {
            let row = (2 * n + i) * 4 * n + 2 * n;
            let source = i * 2 * n;
            re.values[row..row + 2 * n]
                .copy_from_slice(&self.small_potential_real.values[source..source + 2 * n]);
            im.values[row..row + 2 * n]
                .copy_from_slice(&self.small_potential_imaginary.values[source..source + 2 * n]);
        }
        Ok([re, im])
    }
}

type CenterJet<S> = DualN<DualN<S, 3>, 3>;

fn constant<S: Scalar>(value: S) -> CenterJet<S> {
    DualN::constant(DualN::constant(value))
}

fn endpoint<S: Scalar>(shell: &AnalyticShell<S>, first: bool) -> AnalyticShell<CenterJet<S>> {
    AnalyticShell {
        l: shell.l,
        center: std::array::from_fn(|i| {
            if first {
                DualN::constant(DualN::variable(shell.center[i], i))
            } else {
                DualN::variable(DualN::constant(shell.center[i]), i)
            }
        }),
        exponents: shell.exponents.iter().copied().map(constant).collect(),
        coefficients: shell.coefficients.iter().copied().map(constant).collect(),
    }
}

fn kinetic_scalar_channels<S: Scalar>(value: CenterJet<S>, scale: S) -> [S; 4] {
    let m = |i: usize, j: usize| value.im[j].im[i] * scale;
    let d = m(0, 0) + m(1, 1) + m(2, 2);
    [d, m(1, 2) - m(2, 1), m(2, 0) - m(0, 2), m(0, 1) - m(1, 0)]
}

fn kinetic_scalar_product<S: Scalar>(value: CenterJet<S>, scale: S) -> [(S, S); 4] {
    let [d, x, y, z] = kinetic_scalar_channels(value, scale);
    [(d, z), (y, x), (-y, x), (d, -z)]
}

pub(crate) fn shell_offsets<S: Scalar>(shells: &[AnalyticShell<S>]) -> Result<Vec<usize>> {
    let mut offsets = vec![0_usize];
    for shell in shells {
        shell.validate()?;
        let next = offsets
            .last()
            .unwrap()
            .checked_add(crate::basis::n_sph(shell.l))
            .ok_or_else(|| {
                DftError::InvalidInput("balanced Coulomb AO dimension overflow".into())
            })?;
        offsets.push(next);
    }
    Ok(offsets)
}

fn constant_shell<S: Scalar>(shell: &AnalyticShell<S>) -> AnalyticShell<CenterJet<S>> {
    AnalyticShell {
        l: shell.l,
        center: shell.center.map(constant),
        exponents: shell.exponents.iter().copied().map(constant).collect(),
        coefficients: shell.coefficients.iter().copied().map(constant).collect(),
    }
}

fn block<S: Scalar>(rows: usize, cols: usize) -> Result<IntegralBlock<S, 2>> {
    let size = rows.checked_mul(cols).ok_or_else(|| {
        DftError::InvalidInput("balanced Dirac integral allocation overflow".into())
    })?;
    let mut values = Vec::new();
    values.try_reserve_exact(size).map_err(|_| {
        DftError::InvalidInput("balanced Dirac integral exceeds available allocation".into())
    })?;
    values.resize(size, S::from_f64(0.0));
    Ok(IntegralBlock {
        shape: [rows, cols],
        values,
    })
}

const POWERS: [[usize; 3]; 10] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [2, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [0, 2, 0],
    [0, 1, 1],
    [0, 0, 2],
];

/// Cartesian moment matrices in the four-component kinetic-balance basis.
/// A scalar polynomial M(r) connects large-large and small-small blocks.
/// The latter contains (sigma.p) M(r) (sigma.p) / (4 c^2). The endpoint jets
/// act on the two basis functions while the polynomial and its origin stay fixed.
pub(crate) fn balanced_electric_moments<const N: usize>(
    shells: &[AnalyticShell<f64>],
    speed_of_light: f64,
    powers: [[usize; 3]; N],
    origin: [f64; 3],
) -> Result<[BlochMatrix; N]> {
    let mut matrices =
        balanced_cartesian_moments(shells, speed_of_light, &powers, origin)?.into_iter();
    Ok(std::array::from_fn(|_| {
        matrices
            .next()
            .expect("one balanced matrix per Cartesian power")
    }))
}

pub(crate) fn balanced_cartesian_moments(
    shells: &[AnalyticShell<f64>],
    speed_of_light: f64,
    powers: &[[usize; 3]],
    origin: [f64; 3],
) -> Result<Vec<BlochMatrix>> {
    if shells.is_empty() || !speed_of_light.is_finite() || speed_of_light <= 0.0 {
        return Err(DftError::InvalidInput(
            "balanced moments need nonempty shells and positive finite speed of light".into(),
        ));
    }
    let offsets = shell_offsets(shells)?;
    let n = *offsets.last().unwrap();
    let dimension = n
        .checked_mul(4)
        .ok_or_else(|| DftError::InvalidInput("balanced moment AO dimension overflow".into()))?;
    let scale = 0.25 / speed_of_light.powi(2);
    let left: Vec<_> = shells.iter().map(|shell| endpoint(shell, true)).collect();
    let right: Vec<_> = shells.iter().map(|shell| endpoint(shell, false)).collect();
    let mut matrices: Vec<_> = powers
        .iter()
        .map(|_| BlochMatrix::zeros(dimension, dimension))
        .collect();
    for (a, bra) in left.iter().enumerate() {
        for (b, ket) in right.iter().enumerate() {
            for (&power, matrix) in powers.iter().zip(&mut matrices) {
                let moments = analytic_basis::moment(bra, ket, power, origin.map(constant))?;
                let [rows, columns] = moments.shape;
                for row in 0..rows {
                    for column in 0..columns {
                        let (i, j) = (offsets[a] + row, offsets[b] + column);
                        let value = moments.values[row * columns + column];
                        for spin in 0..2 {
                            matrix.re[(spin * n + i, spin * n + j)] = value.re.re;
                        }
                        for (pair, (re, im)) in
                            kinetic_scalar_product(value, scale).into_iter().enumerate()
                        {
                            let p = ((2 + pair / 2) * n + i, (2 + pair % 2) * n + j);
                            matrix.re[p] = re;
                            matrix.im[p] = im;
                        }
                    }
                }
            }
        }
    }
    Ok(matrices)
}

/// Electric dipole matrices on both components of a finite-field London-RMB
/// basis. The global magnetic gauge phase cancels within each pair.
pub(crate) fn london_electric_dipoles(
    shells: &[AnalyticShell<f64>],
    speed_of_light: f64,
    field: [f64; 3],
    origin: [f64; 3],
    phase_order: usize,
) -> Result<[BlochMatrix; 3]> {
    let offsets = shell_offsets(shells)?;
    let n = *offsets.last().unwrap();
    let dimension = 4 * n;
    let scale = (0.5 / speed_of_light).powi(2);
    let left: Vec<_> = shells.iter().map(|shell| endpoint(shell, true)).collect();
    let right: Vec<_> = shells.iter().map(|shell| endpoint(shell, false)).collect();
    let mut moments = std::array::from_fn(|_| BlochMatrix::zeros(dimension, dimension));
    for (a, bra) in left.iter().enumerate() {
        for (b, ket) in right.iter().enumerate() {
            let [rows, cols] = [offsets[a + 1] - offsets[a], offsets[b + 1] - offsets[b]];
            let (dipoles, origins) = london::dipole_moments(
                bra,
                ket,
                [shells[a].center, shells[b].center],
                origin,
                field,
                phase_order,
            )?;
            for (matrix, integrals) in moments.iter_mut().zip(&dipoles) {
                for ia in 0..rows {
                    for ib in 0..cols {
                        let (i, j) = (offsets[a] + ia, offsets[b] + ib);
                        let index = ia * cols + ib;
                        let [re, im] = integrals[0][index].map(|x| x.re.re);
                        for s in 0..2 {
                            let p = (s * n + i, s * n + j);
                            matrix.re[p] = re;
                            matrix.im[p] = im;
                        }
                        for (k, (re, im)) in pauli_product(integrals, index, field, origins)
                            .into_iter()
                            .enumerate()
                        {
                            let p = ((2 + k / 2) * n + i, (2 + k % 2) * n + j);
                            matrix.re[p] = re * scale;
                            matrix.im[p] = im * scale;
                        }
                    }
                }
            }
        }
    }
    Ok(moments)
}

fn potential_moment<S: Scalar>(
    a: &AnalyticShell<CenterJet<S>>,
    b: &AnalyticShell<CenterJet<S>>,
    nuclei: &[GaussianNucleus<S>],
    order: usize,
    power: [usize; 3],
    origin: [S; 3],
) -> Result<Vec<CenterJet<S>>> {
    let size = (2 * a.l as usize + 1) * (2 * b.l as usize + 1);
    let mut total = vec![constant(S::from_f64(0.0)); size];
    for nucleus in nuclei {
        let v = analytic_basis::attraction_moment(
            a,
            b,
            &[AnalyticNucleus {
                center: nucleus.center.map(constant),
                charge: constant(nucleus.charge),
            }],
            GaussianKernel::LongRange {
                omega: constant(S::from_f64(1.5_f64.sqrt()) / nucleus.rms_radius),
            },
            order + 2,
            power,
            origin.map(constant),
        )?;
        for (sum, value) in total.iter_mut().zip(v.values) {
            *sum = *sum + value;
        }
    }
    Ok(total)
}

#[derive(Clone, Copy)]
enum MagneticGauge<S: Scalar> {
    Common([S; 3]),
    London { phase_order: usize },
}

type PairMoments<S> = [Vec<[CenterJet<S>; 2]>; POWERS.len()];

struct MagneticMoments<S: Scalar> {
    overlap: PairMoments<S>,
    potential: PairMoments<S>,
    /// Constant parts of the left and right vector potentials in the
    /// coordinates in which the moments were evaluated.
    offsets: [[S; 3]; 2],
}

/// Contract (sigma.pi) V (sigma.pi) using independent endpoint derivatives.
/// The vector potentials have a shared linear part and independent offsets.
fn pauli_product<S: Scalar>(
    moments: &PairMoments<S>,
    index: usize,
    field: [S; 3],
    [left, right]: [[S; 3]; 2],
) -> [(S, S); 4] {
    let zero = S::from_f64(0.0);
    let [bx, by, bz] = field.map(|x| x.scale(0.5));
    let a = [[zero, -bz, by], [bz, zero, -bx], [-by, bx, zero]];
    let second = [[4, 5, 6], [5, 7, 8], [6, 8, 9]];
    let mut re = [[zero; 3]; 3];
    let mut im = re;
    for i in 0..3 {
        for j in 0..3 {
            let [real, imaginary] = moments[0][index];
            re[i][j] = real.im[j].im[i] + left[i] * right[j] * real.re.re;
            im[i][j] = imaginary.im[j].im[i] + left[i] * right[j] * imaginary.re.re;
            re[i][j] = re[i][j] - left[i] * imaginary.im[j].re + right[j] * imaginary.re.im[i];
            im[i][j] = im[i][j] + left[i] * real.im[j].re - right[j] * real.re.im[i];
            for l in 0..3 {
                let [real, imaginary] = moments[1 + l][index];
                let cross = left[i] * a[j][l] + a[i][l] * right[j];
                re[i][j] = re[i][j] + cross * real.re.re - a[i][l] * imaginary.im[j].re
                    + a[j][l] * imaginary.re.im[i];
                im[i][j] = im[i][j] + cross * imaginary.re.re + a[i][l] * real.im[j].re
                    - a[j][l] * real.re.im[i];
                for m in 0..3 {
                    let [real, imaginary] = moments[second[l][m]][index];
                    let weight = a[i][l] * a[j][m];
                    re[i][j] = re[i][j] + weight * real.re.re;
                    im[i][j] = im[i][j] + weight * imaginary.re.re;
                }
            }
        }
    }
    pauli_tensor(re, im)
}

fn pauli_tensor<S: Scalar>(re: [[S; 3]; 3], im: [[S; 3]; 3]) -> [(S, S); 4] {
    let dr = re[0][0] + re[1][1] + re[2][2];
    let di = im[0][0] + im[1][1] + im[2][2];
    let qr = [
        re[1][2] - re[2][1],
        re[2][0] - re[0][2],
        re[0][1] - re[1][0],
    ];
    let qi = [
        im[1][2] - im[2][1],
        im[2][0] - im[0][2],
        im[0][1] - im[1][0],
    ];
    [
        (dr - qi[2], di + qr[2]),
        (-qi[0] + qr[1], qr[0] + qi[1]),
        (-qi[0] - qr[1], qr[0] - qi[1]),
        (dr + qi[2], di - qr[2]),
    ]
}

/// Generate actual balanced four-component matrices at zero magnetic field.
/// The finite-nucleus model requires explicit positive RMS radii: a missing
/// isotope radius is never replaced with a point nucleus or a guessed value.
/// `order` includes every outer AD level (the two internal basis derivatives
/// are added here). No finite difference or same-triangle copy is used.
pub fn restricted_kinetic_balance<S: Scalar + Send + Sync>(
    shells: &[AnalyticShell<S>],
    nuclei: &[GaussianNucleus<S>],
    speed_of_light: S,
    order: usize,
) -> Result<BalancedDiracIntegrals<S>> {
    balanced_integrals(shells, nuclei, speed_of_light, None, order)
}

/// Restricted magnetic balance, chi_S=(sigma.(p+A))*chi_L/(2c), with
/// A=B cross (r-origin)/2. This common-origin finite-basis model retains
/// orbital, spin Zeeman, diamagnetic and small-component basis/metric terms.
/// It does not infer London phases or an interacting Dirac reference.
pub fn restricted_magnetic_balance<S: Scalar + Send + Sync>(
    shells: &[AnalyticShell<S>],
    nuclei: &[GaussianNucleus<S>],
    speed_of_light: S,
    magnetic_field: [S; 3],
    gauge_origin: [S; 3],
    order: usize,
) -> Result<BalancedDiracIntegrals<S>> {
    if magnetic_field
        .iter()
        .chain(&gauge_origin)
        .any(|x| !x.value().is_finite())
    {
        return Err(DftError::InvalidInput(
            "magnetic balance needs finite field and gauge origin".into(),
        ));
    }
    balanced_integrals(
        shells,
        nuclei,
        speed_of_light,
        Some((magnetic_field, MagneticGauge::Common(gauge_origin))),
        order,
    )
}

/// Restricted magnetic balance with a London phase on each large and small
/// component. The uniform field is in atomic units and the reference is
/// defined by molecular coordinates. A shared shift of the global gauge
/// multiplies every basis function by the same phase, which cancels in all
/// integrals. For a zero-field path a phase order at least equal to the
/// derivative order gives exact Taylor coefficients. For a finite field,
/// use successive phase orders to converge the plane-wave expansion.
pub fn london_magnetic_balance<S: Scalar + Send + Sync>(
    shells: &[AnalyticShell<S>],
    nuclei: &[GaussianNucleus<S>],
    speed_of_light: S,
    magnetic_field: [S; 3],
    order: usize,
    phase_order: usize,
) -> Result<BalancedDiracIntegrals<S>> {
    if magnetic_field.iter().any(|x| !x.value().is_finite())
        || phase_order < order
        || phase_order > 8
    {
        return Err(DftError::InvalidInput(
            "London magnetic balance needs a finite field and phase order from the derivative order through eight".into(),
        ));
    }
    balanced_integrals(
        shells,
        nuclei,
        speed_of_light,
        Some((magnetic_field, MagneticGauge::London { phase_order })),
        order,
    )
}

fn balanced_integrals<S: Scalar + Send + Sync>(
    shells: &[AnalyticShell<S>],
    nuclei: &[GaussianNucleus<S>],
    speed_of_light: S,
    magnetic: Option<([S; 3], MagneticGauge<S>)>,
    order: usize,
) -> Result<BalancedDiracIntegrals<S>> {
    if shells.is_empty()
        || order > 32
        || !speed_of_light.value().is_finite()
        || speed_of_light.value() <= 0.0
        || nuclei.iter().any(|n| {
            !n.charge.value().is_finite()
                || n.charge.value() <= 0.0
                || !n.rms_radius.value().is_finite()
                || n.rms_radius.value() <= 0.0
                || n.center.iter().any(|x| !x.value().is_finite())
        })
    {
        return Err(DftError::InvalidInput("balanced Dirac integrals need nonempty shells, positive finite c, positive finite supplied nuclear charges/radii, finite centers and outer AD order at most 32".into()));
    }
    let mut offsets = vec![0_usize];
    for shell in shells {
        shell.validate()?;
        offsets.push(
            offsets
                .last()
                .unwrap()
                .checked_add(2 * shell.l as usize + 1)
                .ok_or_else(|| {
                    DftError::InvalidInput("balanced basis dimension overflow".into())
                })?,
        );
    }
    let n = *offsets.last().unwrap();
    let two = n
        .checked_mul(2)
        .ok_or_else(|| DftError::InvalidInput("balanced basis dimension overflow".into()))?;
    let four = two
        .checked_mul(2)
        .ok_or_else(|| DftError::InvalidInput("balanced basis dimension overflow".into()))?;
    let count = shells
        .len()
        .checked_mul(shells.len())
        .ok_or_else(|| DftError::InvalidInput("balanced shell-pair count overflow".into()))?;
    let pairs = (0..count)
        .into_par_iter()
        .map(|index| {
            let a = &shells[index / shells.len()];
            let b = &shells[index % shells.len()];
            let overlap = analytic_basis::overlap(a, b)?;
            let kinetic = analytic_basis::kinetic(a, b)?;
            let ad_a = endpoint(a, true);
            let ad_b = endpoint(b, false);
            let zero = S::from_f64(0.0);
            let potential = potential_moment(&ad_a, &ad_b, nuclei, order, [0; 3], [zero; 3])?;
            let moments = match magnetic {
                None => None,
                Some((field, MagneticGauge::London { phase_order })) => Some(london::moments(
                    &ad_a,
                    &ad_b,
                    [a.center, b.center],
                    nuclei,
                    field,
                    order,
                    phase_order,
                )?),
                Some((_, MagneticGauge::Common(origin))) => {
                    let mut sm = std::array::from_fn(|_| Vec::new());
                    let mut vm = std::array::from_fn(|_| Vec::new());
                    let pair = |v: Vec<_>| v.into_iter().map(|x| [x, constant(zero)]).collect();
                    for (k, &power) in POWERS.iter().enumerate() {
                        sm[k] = pair(
                            analytic_basis::moment(&ad_a, &ad_b, power, origin.map(constant))?
                                .values,
                        );
                        vm[k] = pair(if k == 0 {
                            potential.clone()
                        } else {
                            potential_moment(&ad_a, &ad_b, nuclei, order, power, origin)?
                        });
                    }
                    Some(MagneticMoments {
                        overlap: sm,
                        potential: vm,
                        offsets: [[zero; 3]; 2],
                    })
                }
            };
            Ok((overlap, kinetic, potential, moments))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut s = block(n, n)?;
    let mut si = block(n, n)?;
    let mut t = block(n, n)?;
    let mut v = block(n, n)?;
    let mut vi = block(n, n)?;
    let mut wr = block(two, two)?;
    let mut wi = block(two, two)?;
    let mut hr = block(four, four)?;
    let mut hi = block(four, four)?;
    let mut metric = block(four, four)?;
    let mut metric_im = block(four, four)?;
    let mut tr = block(two, two)?;
    let mut ti = block(two, two)?;
    let inv4c2 = S::from_f64(0.25) / (speed_of_light * speed_of_light);
    for (pair, (sp, tp, vp, moments)) in pairs.into_iter().enumerate() {
        let (a, b) = (pair / shells.len(), pair % shells.len());
        for ia in 0..sp.shape[0] {
            for ib in 0..sp.shape[1] {
                let local = ia * sp.shape[1] + ib;
                let (i, j) = (offsets[a] + ia, offsets[b] + ib);
                let vv = vp[local];
                s.values[i * n + j] = sp.values[local];
                t.values[i * n + j] = tp.values[local];
                v.values[i * n + j] = vv.re.re;
                let mut potential_spin = kinetic_scalar_product(vv, inv4c2);
                let mut kinetic_spin = [
                    (tp.values[local], S::from_f64(0.0)),
                    (S::from_f64(0.0), S::from_f64(0.0)),
                    (S::from_f64(0.0), S::from_f64(0.0)),
                    (tp.values[local], S::from_f64(0.0)),
                ];
                if let Some(moments) = &moments {
                    let b = magnetic.unwrap().0;
                    let w = pauli_product(&moments.potential, local, b, moments.offsets);
                    let tb = pauli_product(&moments.overlap, local, b, moments.offsets);
                    let [sr, sp] = moments.overlap[0][local].map(|val| val.re.re);
                    let [vr, vp] = moments.potential[0][local].map(|val| val.re.re);
                    (s.values[i * n + j], si.values[i * n + j]) = (sr, sp);
                    (v.values[i * n + j], vi.values[i * n + j]) = (vr, vp);
                    for k in 0..4 {
                        potential_spin[k] = (w[k].0 * inv4c2, w[k].1 * inv4c2);
                        kinetic_spin[k] = (tb[k].0.scale(0.5), tb[k].1.scale(0.5));
                    }
                }
                for (k, (r, im)) in potential_spin.into_iter().enumerate() {
                    let (row, col) = (k / 2 * n + i, k % 2 * n + j);
                    wr.values[row * two + col] = r;
                    wi.values[row * two + col] = im;
                    hr.values[(two + row) * four + two + col] = r;
                    hi.values[(two + row) * four + two + col] = im;
                    let (kr, ki) = kinetic_spin[k];
                    tr.values[row * two + col] = kr;
                    ti.values[row * two + col] = ki;
                    let ss = (two + row) * four + two + col;
                    metric.values[ss] = kr * inv4c2.scale(2.0);
                    metric_im.values[ss] = ki * inv4c2.scale(2.0);
                    hr.values[row * four + two + col] = kr;
                    hi.values[row * four + two + col] = ki;
                    hr.values[(two + row) * four + col] = kr;
                    hi.values[(two + row) * four + col] = ki;
                    hr.values[ss] = hr.values[ss] - kr;
                    hi.values[ss] = hi.values[ss] - ki;
                }
                for spin in 0..2 {
                    let (l, r) = (spin * n + i, spin * n + j);
                    metric.values[l * four + r] = s.values[i * n + j];
                    metric_im.values[l * four + r] = si.values[i * n + j];
                    hr.values[l * four + r] = v.values[i * n + j];
                    hi.values[l * four + r] = vi.values[i * n + j];
                }
            }
        }
    }
    let mut nuclear_repulsion = S::from_f64(0.0);
    for (i, a) in nuclei.iter().enumerate() {
        for b in &nuclei[..i] {
            let width2 =
                S::from_f64(1.5) / (a.rms_radius * a.rms_radius + b.rms_radius * b.rms_radius);
            let r2 = (0..3).fold(S::from_f64(0.0), |sum, k| {
                let d = a.center[k] - b.center[k];
                sum + d * d
            });
            nuclear_repulsion = nuclear_repulsion
                + a.charge
                    * b.charge
                    * width2.sqrt().scale(2.0 / PI.sqrt())
                    * boys_jet(0, width2 * r2, order)?;
        }
    }
    Ok(BalancedDiracIntegrals {
        spatial_overlap: s,
        spatial_overlap_imaginary: si,
        kinetic: t,
        potential: v,
        potential_imaginary: vi,
        small_potential_real: wr,
        small_potential_imaginary: wi,
        hamiltonian_real: hr,
        hamiltonian_imaginary: hi,
        overlap: metric,
        overlap_imaginary: metric_im,
        balanced_kinetic_real: tr,
        balanced_kinetic_imaginary: ti,
        nuclear_repulsion,
    })
}

/// Convert factorial-normalized scalar matrix jets to the common complex
/// response matrices. Missing imaginary entries mean exact zero.
pub fn matrix_coefficients<const W: usize>(
    real: &IntegralBlock<Taylor<W>, 2>,
    imaginary: Option<&IntegralBlock<Taylor<W>, 2>>,
    order: usize,
) -> Result<Vec<BlochMatrix>> {
    let [rows, cols] = real.shape;
    let size = rows
        .checked_mul(cols)
        .ok_or_else(|| DftError::InvalidInput("Dirac coefficient shape overflow".into()))?;
    if order >= W
        || real.values.len() != size
        || imaginary.is_some_and(|im| im.shape != real.shape || im.values.len() != size)
    {
        return Err(DftError::InvalidInput(
            "invalid Dirac Taylor matrix shape or coefficient order".into(),
        ));
    }
    (0..=order)
        .map(|k| {
            let re: Vec<_> = real.values.iter().map(|x| x.c[k]).collect();
            let im: Vec<_> = imaginary.map_or_else(
                || vec![0.0; size],
                |m| m.values.iter().map(|x| x.c[k]).collect(),
            );
            if re.iter().chain(&im).any(|x| !x.is_finite()) {
                return Err(DftError::InvalidInput(
                    "non-finite balanced Dirac integral Taylor coefficient".into(),
                ));
            }
            Ok(BlochMatrix {
                re: Matrix::from_row_major(rows, cols, re),
                im: Matrix::from_row_major(rows, cols, im),
            })
        })
        .collect()
}
