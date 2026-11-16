// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Small 3-vector / 3×3-matrix algebra.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// [erf(x) − 2 x exp(−x²)/sqrt(pi)] / x³ for nonnegative x, including x=0.
/// The Gaussian moment series avoids subtraction of near-equal terms.
pub(crate) fn scaled_erf_difference(x: f64) -> f64 {
    let x2 = x * x;
    let root_pi = std::f64::consts::PI.sqrt();
    if x2 < 0.25 {
        let mut moment = 1.0 / 3.0;
        let mut power = 1.0;
        for order in 1..13 {
            power *= -x2 / f64::from(order);
            moment += power / f64::from(2 * order + 3);
        }
        4.0 * moment / root_pi
    } else {
        (libm::erf(x) - 2.0 * x * (-x2).exp() / root_pi) / x.powi(3)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    #[inline]
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
    #[inline]
    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }
    #[inline]
    pub fn norm2(self) -> f64 {
        self.dot(self)
    }
    #[inline]
    pub fn norm(self) -> f64 {
        self.norm2().sqrt()
    }
    #[inline]
    pub fn normalized(self) -> Self {
        let n = self.norm();
        if n <= f64::EPSILON {
            Self::zero()
        } else {
            self / n
        }
    }
    #[inline]
    pub fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
    #[inline]
    pub fn get(self, i: usize) -> f64 {
        match i {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}
impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}
impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}
impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}
impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}
impl Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}
impl Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Vec3 {
        rhs * self
    }
}
impl Div<f64> for Vec3 {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

/// Column-major 3×3 matrix (three column vectors).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3 {
    pub col: [Vec3; 3],
}

impl Mat3 {
    #[inline]
    pub const fn from_columns(a: Vec3, b: Vec3, c: Vec3) -> Self {
        Self { col: [a, b, c] }
    }
    #[inline]
    pub fn zero() -> Self {
        Self::from_columns(Vec3::zero(), Vec3::zero(), Vec3::zero())
    }
    #[inline]
    pub fn mul_vec(self, v: Vec3) -> Vec3 {
        self.col[0] * v.x + self.col[1] * v.y + self.col[2] * v.z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3_basic_algebra() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(-2.0, 0.5, 4.0);
        assert!((a.dot(b) - (-2.0 + 1.0 + 12.0)).abs() < 1e-15);
        let c = a.cross(b);
        // Orthogonality of the cross product.
        assert!(c.dot(a).abs() < 1e-12);
        assert!(c.dot(b).abs() < 1e-12);
        assert!((a.norm2() - 14.0).abs() < 1e-15);
        assert!(((a + b) - Vec3::new(-1.0, 2.5, 7.0)).norm() < 1e-15);
        assert!(((a * 2.0) - Vec3::new(2.0, 4.0, 6.0)).norm() < 1e-15);
    }

    #[test]
    fn mat3_mul_vec() {
        // Identity columns.
        let m = Mat3::from_columns(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let v = Vec3::new(3.0, -1.0, 2.0);
        assert!((m.mul_vec(v) - v).norm() < 1e-15);
    }
}
