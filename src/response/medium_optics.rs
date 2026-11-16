// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Electric constitutive response and Maxwell scattering by a homogeneous slab.

use super::complex::Complex;
use crate::linalg::{solve_linear_matrix, Matrix};
use crate::periodic::BlochMatrix;
use crate::{DftError, Result};
use std::array::from_fn;
use std::f64::consts::PI;

type Square<const N: usize> = [[Complex; N]; N];
const I: Complex = Complex::new(0.0, 1.0);

fn identity<const N: usize>() -> Square<N> {
    from_fn(|i| from_fn(|j| if i == j { Complex::ONE } else { Complex::ZERO }))
}

fn add<const N: usize>(a: &Square<N>, b: &Square<N>, scale: f64) -> Square<N> {
    from_fn(|i| from_fn(|j| a[i][j] + b[i][j] * scale))
}

fn product<const N: usize>(a: &Square<N>, b: &Square<N>) -> Square<N> {
    from_fn(|i| from_fn(|j| (0..N).fold(Complex::ZERO, |sum, k| sum + a[i][k] * b[k][j])))
}

fn solve<const N: usize, const M: usize>(
    a: &Square<N>,
    b: &[[Complex; M]; N],
) -> Result<[[Complex; M]; N]> {
    let real = Matrix::from_row_major(N, N, a.iter().flatten().map(|z| z.re).collect());
    let imag = Matrix::from_row_major(N, N, a.iter().flatten().map(|z| z.im).collect());
    let lhs = BlochMatrix::from_pair((real, imag)).real_embedding();
    let rhs = Matrix::from_row_major(
        2 * N,
        M,
        b.iter()
            .flatten()
            .map(|z| z.re)
            .chain(b.iter().flatten().map(|z| z.im))
            .collect(),
    );
    let x = solve_linear_matrix(&lhs, &rhs)?;
    Ok(from_fn(|i| {
        from_fn(|j| Complex::new(x[(i, j)], x[(i + N, j)]))
    }))
}

fn max_norm<const N: usize>(a: &Square<N>) -> f64 {
    a.iter().flatten().map(|z| z.abs()).fold(0.0, f64::max)
}

/// Local electric field model for a molecular medium.
#[derive(Clone, Copy, Debug)]
pub enum LocalField {
    /// Each molecule sees the macroscopic electric field.
    Dilute,
    /// Spherical cavity with the self-consistent field E + 4 pi P / 3.
    Lorentz,
}

/// Relative permittivity and the raw residual of the polarization equation.
#[derive(Clone, Copy, Debug)]
pub struct MolecularDielectric {
    pub permittivity: [[Complex; 3]; 3],
    pub local_field_residual: f64,
}

fn principal_sqrt(z: Complex) -> Complex {
    if z.abs() == 0.0 {
        return Complex::ZERO;
    }
    if z.re >= 0.0 {
        let re = (0.5 * z.abs() + 0.5 * z.re).sqrt();
        Complex::new(re, 0.5 * z.im / re)
    } else {
        let im = (0.5 * z.abs() - 0.5 * z.re).sqrt().copysign(z.im);
        Complex::new(0.5 * z.im / im, im)
    }
}

impl MolecularDielectric {
    /// Two normal-axis bulk refractive indices after eliminating the normal
    /// electric field. At a passive propagating mode the index has positive
    /// real part and nonnegative attenuation for the exp(-i omega t) convention.
    pub fn normal_indices(&self) -> Result<[Complex; 2]> {
        let d = generator(&self.permittivity, 0.0)?;
        let mean = (d[3][0] - d[2][1]) * 0.5;
        let half_difference = (d[3][0] + d[2][1]) * 0.5;
        let gap = principal_sqrt(half_difference * half_difference - d[3][1] * d[2][0]);
        let indices = [principal_sqrt(mean + gap), principal_sqrt(mean - gap)];
        if indices.iter().any(|z| !z.is_finite()) {
            return Err(DftError::LinearAlgebra(
                "slab normal indices exceed finite arithmetic".into(),
            ));
        }
        Ok(indices)
    }
}

/// Convert a molecular electric dipole polarizability and a molecular number
/// density from atomic units to a dimensionless relative permittivity.
/// The tensor is expressed in the laboratory frame.
pub fn molecular_dielectric(
    polarizability: [[Complex; 3]; 3],
    number_density: f64,
    local_field: LocalField,
) -> Result<MolecularDielectric> {
    if !number_density.is_finite()
        || number_density < 0.0
        || polarizability.iter().flatten().any(|z| !z.is_finite())
    {
        return Err(DftError::InvalidInput(
            "molecular optics needs finite polarizability and nonnegative number density".into(),
        ));
    }
    let unit = identity();
    let source = polarizability.map(|row| row.map(|z| z * (4.0 * PI * number_density)));
    if source.iter().flatten().any(|z| !z.is_finite()) {
        return Err(DftError::InvalidInput(
            "molecular polarization exceeds finite arithmetic".into(),
        ));
    }
    let equation = match local_field {
        LocalField::Dilute => unit,
        LocalField::Lorentz => add(&unit, &source, -1.0 / 3.0),
    };
    let polarization = match local_field {
        LocalField::Dilute => source,
        LocalField::Lorentz => solve(&equation, &source)?,
    };
    Ok(MolecularDielectric {
        permittivity: add(&unit, &polarization, 1.0),
        local_field_residual: max_norm(&add(&product(&equation, &polarization), &source, -1.0)),
    })
}

/// Dimensionless boundary data. The optical thickness is the vacuum
/// wavenumber times slab thickness. The incidence plane is xz.
#[derive(Clone, Copy, Debug)]
pub struct SlabBoundary {
    pub incident_index: f64,
    pub exit_index: f64,
    pub incidence_angle: f64,
    pub optical_thickness: f64,
}

impl SlabBoundary {
    pub(crate) fn validate(&self) -> Result<()> {
        if [self.incident_index, self.exit_index]
            .into_iter()
            .any(|n| !n.is_finite() || n <= 0.0)
            || !self.incidence_angle.is_finite()
            || self.incidence_angle.abs() >= PI / 2.0
            || !self.optical_thickness.is_finite()
            || self.optical_thickness < 0.0
        {
            return Err(DftError::InvalidInput("dielectric slab needs positive exterior indices, nongrazing incidence and finite nonnegative optical thickness".into()));
        }
        Ok(())
    }
}

/// Reflected and transmitted Jones matrices and normal fluxes for incoming
/// s and p waves. Jones amplitudes refer to the two slab interfaces.
#[derive(Clone, Copy, Debug)]
pub struct SlabScattering {
    pub reflection: [[Complex; 2]; 2],
    pub transmission: [[Complex; 2]; 2],
    pub reflectance: [f64; 2],
    pub transmittance: [f64; 2],
    pub absorptance: [f64; 2],
    pub flux_balance_residual: [f64; 2],
    pub boundary_residual: f64,
}

/// Field-odd Kerr angles, in radians, for s and p incident light.
#[derive(Clone, Copy, Debug)]
pub struct KerrAngles {
    pub rotation: [Option<f64>; 2],
    pub ellipticity: [Option<f64>; 2],
}

/// Form the field-odd rotation and ellipticity from reflected Jones matrices
/// at opposite magnetic biases. The transverse axes are co- and cross-
/// polarized with respect to each incident state. A vanishing reflected
/// field has no ellipse and a circular ellipse has no orientation.
pub fn magnetic_kerr(positive: [[Complex; 2]; 2], negative: [[Complex; 2]; 2]) -> KerrAngles {
    let stokes = |m: &Square<2>, j: usize| {
        let co = m[j][j];
        let cross = m[1 - j][j];
        let scale = co.abs().max(cross.abs());
        if !scale.is_finite() || scale == 0.0 {
            return None;
        }
        let co = co / scale;
        let cross = cross / scale;
        let mixed = co.conj() * cross * 2.0;
        let co_power = co.abs().powi(2);
        let cross_power = cross.abs().powi(2);
        Some([
            co_power - cross_power,
            mixed.re,
            mixed.im / (co_power + cross_power),
        ])
    };
    let paired = [0, 1].map(|j| stokes(&positive, j).zip(stokes(&negative, j)));
    let rotation = paired.map(|pair| {
        let ([x, y, _], [u, v, _]) = pair?;
        if (x == 0.0 && y == 0.0) || (u == 0.0 && v == 0.0) {
            return None;
        }
        Some(0.25 * (y * u - x * v).atan2(x * u + y * v))
    });
    let ellipticity = paired.map(|pair| {
        let ([_, _, a], [_, _, b]) = pair?;
        Some(0.25 * (a.clamp(-1.0, 1.0).asin() - b.clamp(-1.0, 1.0).asin()))
    });
    KerrAngles {
        rotation,
        ellipticity,
    }
}

#[derive(Clone, Copy)]
struct Scattering {
    left_reflection: Square<2>,
    right_reflection: Square<2>,
    forward: Square<2>,
    backward: Square<2>,
    residual: f64,
}

fn block(a: &Square<4>, row: usize, col: usize) -> Square<2> {
    from_fn(|i| from_fn(|j| a[2 * row + i][2 * col + j]))
}

impl Scattering {
    fn from_scattering(s: &Square<4>, residual: f64) -> Self {
        Self {
            left_reflection: block(s, 0, 0),
            right_reflection: block(s, 1, 1),
            forward: block(s, 1, 0),
            backward: block(s, 0, 1),
            residual,
        }
    }

    fn from_transfer(t: &Square<4>) -> Result<Self> {
        let d = block(t, 1, 1);
        let unit = identity();
        let backward = solve(&d, &unit)?;
        let left_reflection = product(&backward, &block(t, 1, 0)).map(|row| row.map(|z| -z));
        let b = block(t, 0, 1);
        Ok(Self {
            left_reflection,
            right_reflection: product(&b, &backward),
            forward: add(&block(t, 0, 0), &product(&b, &left_reflection), 1.0),
            backward,
            residual: max_norm(&add(&product(&d, &backward), &unit, -1.0)),
        })
    }

    fn cascade(self, next: Self) -> Result<Self> {
        let connection = add(
            &identity(),
            &product(&self.right_reflection, &next.left_reflection),
            -1.0,
        );
        let backward_source = product(&self.right_reflection, &next.backward);
        let sources = from_fn(|i| {
            [
                self.forward[i][0],
                self.forward[i][1],
                backward_source[i][0],
                backward_source[i][1],
            ]
        });
        let solution = solve(&connection, &sources)?;
        let forward = solution.map(|row| [row[0], row[1]]);
        let backward = solution.map(|row| [row[2], row[3]]);
        let left_exit = product(&self.backward, &next.left_reflection);
        Ok(Self {
            left_reflection: add(&self.left_reflection, &product(&left_exit, &forward), 1.0),
            right_reflection: add(
                &next.right_reflection,
                &product(&next.forward, &backward),
                1.0,
            ),
            forward: product(&next.forward, &forward),
            backward: add(
                &product(&self.backward, &next.backward),
                &product(&left_exit, &backward),
                1.0,
            ),
            residual: self
                .residual
                .max(next.residual)
                .max(max_norm(&add(
                    &product(&connection, &forward),
                    &self.forward,
                    -1.0,
                )))
                .max(max_norm(&add(
                    &product(&connection, &backward),
                    &backward_source,
                    -1.0,
                ))),
        })
    }
}

fn generator(epsilon: &Square<3>, p: f64) -> Result<Square<4>> {
    let [x, y, z] = *epsilon;
    if z[2].abs() == 0.0 {
        return Err(DftError::LinearAlgebra(
            "zero normal permittivity in the slab Maxwell equations".into(),
        ));
    }
    let zero = Complex::ZERO;
    let ez = [-z[0] / z[2], -z[1] / z[2], zero, Complex::from(-p) / z[2]];
    let mut ex = ez.map(|z| z * p);
    ex[3] += Complex::ONE;
    let hx = [-y[0], Complex::from(p * p) - y[1], zero, zero];
    let hy = [x[0], x[1], zero, zero];
    Ok([
        ex,
        [zero, zero, -Complex::ONE, zero],
        from_fn(|j| hx[j] - y[2] * ez[j]),
        from_fn(|j| hy[j] + x[2] * ez[j]),
    ])
}

/// Columns are forward s, forward p, backward s, backward p.
fn exterior(index: f64, q: Complex) -> Square<4> {
    let z = Complex::ZERO;
    let n = Complex::from(index);
    [
        [z, q / index, z, q / index],
        [Complex::ONE, z, Complex::ONE, z],
        [-q, z, q, z],
        [z, n, z, -n],
    ]
}

fn flux(field: [Complex; 4]) -> f64 {
    let [ex, ey, hx, hy] = field;
    (ex * hy.conj() - ey * hx.conj()).re
}

/// Propagate tangential E/H through a homogeneous, nonmagnetic dielectric
/// with arbitrary electric anisotropy. Exterior media are lossless isotropic.
pub fn dielectric_slab(
    epsilon: [[Complex; 3]; 3],
    boundary: SlabBoundary,
) -> Result<SlabScattering> {
    boundary.validate()?;
    let SlabBoundary {
        incident_index,
        exit_index,
        incidence_angle,
        optical_thickness,
    } = boundary;
    if epsilon.iter().flatten().any(|z| !z.is_finite()) {
        return Err(DftError::InvalidInput(
            "dielectric slab needs finite permittivity".into(),
        ));
    }
    let p = incident_index * incidence_angle.sin();
    let q_left = incident_index * incidence_angle.cos();
    let right_normal_squared = (exit_index - p) * (exit_index + p);
    if !right_normal_squared.is_finite() || q_left == 0.0 {
        return Err(DftError::InvalidInput(
            "exterior wave numbers exceed finite arithmetic".into(),
        ));
    }
    let q_right = principal_sqrt(Complex::from(right_normal_squared));
    let left = exterior(incident_index, Complex::from(q_left));
    let right = exterior(exit_index, q_right);
    let interface_matrix = from_fn(|i| {
        from_fn(|j| {
            if j < 2 {
                left[i][j + 2]
            } else {
                -right[i][j - 2]
            }
        })
    });
    let interface_rhs = from_fn(|i| from_fn(|j| if j < 2 { -left[i][j] } else { right[i][j] }));
    let interface = solve(&interface_matrix, &interface_rhs)?;
    let boundary_residual = max_norm(&add(
        &product(&interface_matrix, &interface),
        &interface_rhs,
        -1.0,
    ));
    let mut scattering = Scattering::from_scattering(&interface, boundary_residual);
    if optical_thickness > 0.0 {
        let d = generator(&epsilon, p)?;
        let a = d.map(|row| row.map(|z| I * z * optical_thickness));
        let bound = a
            .iter()
            .map(|row| row.iter().map(|z| z.abs()).sum::<f64>())
            .fold(0.0, f64::max);
        if !bound.is_finite() || a.iter().flatten().any(|z| !z.is_finite()) {
            return Err(DftError::InvalidInput(
                "slab propagation exceeds finite arithmetic".into(),
            ));
        }
        let doublings = if bound > 0.25 {
            (bound.log2() + 2.0).ceil() as i32
        } else {
            0
        };
        let scale = 2.0_f64.powi(-doublings);
        let a = a.map(|row| row.map(|z| z * scale));
        let mut transfer = identity();
        let mut term = transfer;
        for k in 1..=16 {
            term = product(&term, &a).map(|row| row.map(|z| z / f64::from(k)));
            transfer = add(&transfer, &term, 1.0);
        }
        let mut slab = Scattering::from_transfer(&solve(&left, &product(&transfer, &left))?)?;
        // Doubling scattering matrices avoids the growing transfer-matrix
        // exponentials of evanescent and absorbed waves in a thick layer.
        for _ in 0..doublings {
            slab = slab.cascade(slab)?;
        }
        scattering = slab.cascade(scattering)?;
    }
    let reflection = scattering.left_reflection;
    let transmission = scattering.forward;
    let reflectance = from_fn(|j| (0..2).map(|i| reflection[i][j].abs().powi(2)).sum::<f64>());
    let transmittance = from_fn(|j| {
        (q_right.re / q_left)
            * (0..2)
                .map(|i| transmission[i][j].abs().powi(2))
                .sum::<f64>()
    });
    let absorptance = from_fn(|j| {
        let incoming_and_reflected =
            from_fn(|i| left[i][j] + left[i][2] * reflection[0][j] + left[i][3] * reflection[1][j]);
        let transmitted =
            from_fn(|i| right[i][0] * transmission[0][j] + right[i][1] * transmission[1][j]);
        (flux(incoming_and_reflected) - flux(transmitted)) / q_left
    });
    let flux_balance_residual =
        from_fn(|j| (1.0 - reflectance[j] - transmittance[j] - absorptance[j]).abs());
    if reflectance
        .into_iter()
        .chain(transmittance)
        .chain(absorptance)
        .any(|x| !x.is_finite())
    {
        return Err(DftError::LinearAlgebra(
            "slab scattering exceeds finite arithmetic".into(),
        ));
    }
    Ok(SlabScattering {
        reflection,
        transmission,
        reflectance,
        transmittance,
        absorptance,
        flux_balance_residual,
        boundary_residual: scattering.residual,
    })
}

#[cfg(test)]
mod tests;
