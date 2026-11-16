// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Moving bare one-electron spin-orbit and mixed magnetic operators. Nuclear
//! motion, the Coulomb centre, both AO centres and an optional moving gauge
//! origin share one Taylor axis. Independent auxiliary axes differentiate the
//! Coulomb centre and ket Gaussian; finite differences are not used here.
//! This is operator infrastructure, not a complete g or SOC D response.

use crate::basis::{n_sph, MolBasis};
use crate::dual::{DualN, Scalar, Taylor};
use crate::integrals::analytic::GaussianKernel;
use crate::integrals::analytic_basis::{
    angular_momentum, attraction, attraction_moment, AnalyticNucleus, AnalyticShell,
};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::system::Molecule;
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct SpinOrbitOperatorPath {
    /// Real antisymmetric <mu|((r-O) x grad)_a|nu>, without -i.
    pub angular_momentum: [Vec<Matrix>; 3],
    /// Real antisymmetric matrices for <mu|((r-C) x grad)_a/r_C^3|nu>.
    /// The physical imaginary factor and SOC charge/prefactor are separate.
    pub paramagnetic: [Vec<Matrix>; 3],
    /// [magnetic-field axis][nuclear magnetic-moment axis][Taylor degree].
    /// Same convention as the static shielding mixed operator.
    pub mixed: [[Vec<Matrix>; 3]; 3],
    pub paramagnetic_antisymmetry_residuals: Vec<f64>,
    pub angular_antisymmetry_residuals: Vec<f64>,
    pub mixed_hermiticity_residuals: Vec<f64>,
}

fn curl(axis: usize) -> [(usize, usize, f64); 2] {
    match axis {
        0 => [(1, 2, 1.0), (2, 1, -1.0)],
        1 => [(2, 0, 1.0), (0, 2, -1.0)],
        _ => [(0, 1, 1.0), (1, 0, -1.0)],
    }
}

fn lift<const W: usize>(
    shell: &AnalyticShell<Taylor<W>>,
    ket: bool,
) -> AnalyticShell<DualN<DualN<Taylor<W>, 3>, 3>> {
    type Inner<const V: usize> = DualN<Taylor<V>, 3>;
    let constant = |value| DualN::constant(Inner::constant(value));
    AnalyticShell {
        l: shell.l,
        center: std::array::from_fn(|axis| {
            if ket {
                DualN::variable(Inner::constant(shell.center[axis]), axis)
            } else {
                constant(shell.center[axis])
            }
        }),
        exponents: shell.exponents.iter().copied().map(constant).collect(),
        coefficients: shell.coefficients.iter().copied().map(constant).collect(),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn spin_orbit_operator_path(
    molecule: &Molecule,
    basis: &MolBasis,
    nucleus: usize,
    direction: &[[f64; 3]],
    origin: Vec3,
    origin_direction: [f64; 3],
    london: bool,
    order: usize,
) -> Result<SpinOrbitOperatorPath> {
    match order {
        0 => build::<1>(
            molecule,
            basis,
            nucleus,
            direction,
            origin,
            origin_direction,
            london,
            order,
        ),
        1 => build::<2>(
            molecule,
            basis,
            nucleus,
            direction,
            origin,
            origin_direction,
            london,
            order,
        ),
        2 => build::<3>(
            molecule,
            basis,
            nucleus,
            direction,
            origin,
            origin_direction,
            london,
            order,
        ),
        _ => Err(DftError::Unsupported {
            feature: "moving spin-orbit operator order above two".into(),
            missing: "higher-order bare spin-orbit operator acceptance".into(),
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn build<const W: usize>(
    molecule: &Molecule,
    basis: &MolBasis,
    nucleus: usize,
    direction: &[[f64; 3]],
    origin: Vec3,
    origin_direction: [f64; 3],
    london: bool,
    order: usize,
) -> Result<SpinOrbitOperatorPath> {
    if !basis.ecps.is_empty() {
        return Err(DftError::Unsupported {
            feature: "ECP moving spin-orbit operators".into(),
            missing: "effective-core spin-orbit and projector nuclear derivatives".into(),
        });
    }
    let n = basis.nao_sph;
    if n == 0
        || nucleus >= molecule.len()
        || direction.len() != molecule.len()
        || direction
            .iter()
            .flatten()
            .chain(&origin_direction)
            .any(|v| !v.is_finite())
        || origin.to_array().iter().any(|v| !v.is_finite())
        || basis.shells.iter().any(|shell| {
            shell.atom >= direction.len()
                || shell
                    .sph_off
                    .checked_add(n_sph(shell.l))
                    .is_none_or(|end| end > n)
        })
    {
        return Err(DftError::InvalidInput(
            "moving spin-orbit operators require finite complete AO/nuclear/gauge paths".into(),
        ));
    }
    type Q<const V: usize> = Taylor<V>;
    type Inner<const V: usize> = DualN<Q<V>, 3>;
    type Outer<const V: usize> = DualN<Inner<V>, 3>;
    let point: [Q<W>; 3] = std::array::from_fn(|axis| {
        Q::from_coefficients(&[
            molecule.atoms[nucleus].position.get(axis),
            direction[nucleus][axis],
        ])
    });
    let gauge: [Q<W>; 3] = std::array::from_fn(|axis| {
        Q::from_coefficients(&[origin.get(axis), origin_direction[axis]])
    });
    let charge = AnalyticNucleus {
        center: std::array::from_fn(|axis| Outer::constant(Inner::variable(point[axis], axis))),
        charge: Outer::from_f64(1.0),
    };
    let gauge_probe = gauge.map(|value| Outer::constant(Inner::constant(value)));
    let shells = basis
        .shells
        .iter()
        .map(|shell| {
            AnalyticShell::from_placed(
                shell,
                std::array::from_fn(|axis| {
                    Q::<W>::from_coefficients(&[
                        shell.center.get(axis),
                        direction[shell.atom][axis],
                    ])
                }),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let max_shell = basis
        .shells
        .iter()
        .map(|shell| n_sph(shell.l))
        .max()
        .ok_or_else(|| DftError::InvalidInput("empty spin-orbit shell layout".into()))?;
    let bytes = crate::resources::checked_bytes(
        &[4, max_shell, max_shell, std::mem::size_of::<Outer<W>>()],
        "moving spin-orbit blocks",
    )?;
    let _workspace = crate::resources::reserve(bytes, "moving spin-orbit blocks")?;
    let mut paramagnetic = std::array::from_fn(|_| vec![Matrix::zeros(n, n); order + 1]);
    let mut orbital = std::array::from_fn(|_| vec![Matrix::zeros(n, n); order + 1]);
    let mut mixed =
        std::array::from_fn(|_| std::array::from_fn(|_| vec![Matrix::zeros(n, n); order + 1]));
    for (ia, a) in shells.iter().enumerate() {
        for (ib, b) in shells.iter().enumerate() {
            let bra = lift(a, false);
            let ket = lift(b, true);
            let angular = (0..3)
                .map(|axis| angular_momentum(a, b, axis, gauge))
                .collect::<Result<Vec<_>>>()?;
            let base = attraction(&bra, &ket, &[charge], GaussianKernel::Coulomb, order + 2)?;
            let moments = (0..3)
                .map(|axis| {
                    let mut power = [0; 3];
                    power[axis] = 1;
                    attraction_moment(
                        &bra,
                        &ket,
                        &[charge],
                        GaussianKernel::Coulomb,
                        order + 2,
                        power,
                        gauge_probe,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let relative: [Q<W>; 3] = std::array::from_fn(|axis| a.center[axis] - b.center[axis]);
            let ket_origin: [Q<W>; 3] = std::array::from_fn(|axis| b.center[axis] - gauge[axis]);
            for i in 0..base.shape[0] {
                for j in 0..base.shape[1] {
                    let local = i * base.shape[1] + j;
                    let value = base.values[local];
                    let field: [Q<W>; 3] = std::array::from_fn(|axis| -value.re.im[axis]);
                    let q: [[Q<W>; 3]; 3] = std::array::from_fn(|axis| {
                        std::array::from_fn(|moment| -moments[moment].values[local].re.im[axis])
                    });
                    let spin: [Q<W>; 3] = std::array::from_fn(|axis| {
                        let mut result = Q::from_f64(0.0);
                        for (c, d, sign) in curl(axis) {
                            result = result + value.im[d].im[c].scale(sign);
                        }
                        result
                    });
                    let y: [[Q<W>; 3]; 3] = std::array::from_fn(|moment| {
                        std::array::from_fn(|axis| {
                            let mut result = Q::from_f64(0.0);
                            if london {
                                for (c, d, sign) in curl(axis) {
                                    result = result
                                        + moments[moment].values[local].im[d].im[c].scale(sign);
                                }
                            }
                            result
                        })
                    });
                    let mu = basis.shells[ia].sph_off + i;
                    let nu = basis.shells[ib].sph_off + j;
                    for axis in 0..3 {
                        for degree in 0..=order {
                            paramagnetic[axis][degree][(mu, nu)] = spin[axis].c[degree];
                            orbital[axis][degree][(mu, nu)] = angular[axis].values[local].c[degree];
                        }
                        let (c, d, _) = curl(axis)[0];
                        for component in 0..3 {
                            let mut result = -q[axis][component];
                            if axis == component {
                                result = result + q[0][0] + q[1][1] + q[2][2];
                            }
                            if london {
                                result = result + relative[c] * y[d][component]
                                    - relative[d] * y[c][component];
                                let mut extra = -ket_origin[component] * field[axis];
                                if axis == component {
                                    for k in 0..3 {
                                        extra = extra + ket_origin[k] * field[k];
                                    }
                                }
                                result = result - extra;
                            }
                            result = result.scale(0.5);
                            for degree in 0..=order {
                                mixed[axis][component][degree][(mu, nu)] = result.c[degree];
                            }
                        }
                    }
                }
            }
        }
    }
    let mut antisymmetry = vec![0.0_f64; order + 1];
    let mut hermiticity = antisymmetry.clone();
    let mut orbital_antisymmetry = antisymmetry.clone();
    for degree in 0..=order {
        for i in 0..n {
            for j in 0..n {
                for axis in 0..3 {
                    let a = paramagnetic[axis][degree][(i, j)];
                    let b = paramagnetic[axis][degree][(j, i)];
                    if !a.is_finite() || !b.is_finite() {
                        return Err(DftError::InvalidInput(
                            "nonfinite moving spin-orbit operator".into(),
                        ));
                    }
                    antisymmetry[degree] = antisymmetry[degree].max((a + b).abs());
                    let a = orbital[axis][degree][(i, j)];
                    let b = orbital[axis][degree][(j, i)];
                    if !a.is_finite() || !b.is_finite() {
                        return Err(DftError::InvalidInput(
                            "nonfinite moving angular operator".into(),
                        ));
                    }
                    orbital_antisymmetry[degree] = orbital_antisymmetry[degree].max((a + b).abs());
                    for component in 0..3 {
                        let a = mixed[axis][component][degree][(i, j)];
                        let b = mixed[axis][component][degree][(j, i)];
                        if !a.is_finite() || !b.is_finite() {
                            return Err(DftError::InvalidInput(
                                "nonfinite moving mixed magnetic operator".into(),
                            ));
                        }
                        hermiticity[degree] = hermiticity[degree].max((a - b).abs());
                    }
                }
            }
        }
    }
    Ok(SpinOrbitOperatorPath {
        angular_momentum: orbital,
        paramagnetic,
        mixed,
        paramagnetic_antisymmetry_residuals: antisymmetry,
        angular_antisymmetry_residuals: orbital_antisymmetry,
        mixed_hermiticity_residuals: hermiticity,
    })
}
