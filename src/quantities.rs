// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Physical units and meanings at calculation boundaries. Numerical kernels
//! retain their existing atomic-unit storage; extracting it is explicit.
//! Wrappers do not infer whether a caller mislabeled a raw input number.
//!
//! ```compile_fail
//! use dft_rs::quantities::{Angstrom, Bohr, Coordinates};
//! fn atomic_input(_: Coordinates<Bohr>) {}
//! atomic_input(Coordinates::<Angstrom>::new(vec![[0.0; 3]]).unwrap());
//! ```
//! ```compile_fail
//! use dft_rs::quantities::{EnergyGradient, Force};
//! fn use_force(_: Force) {}
//! use_force(EnergyGradient::new(vec![[1.0; 3]]).unwrap());
//! ```
//! ```compile_fail
//! use dft_rs::quantities::{Energy, Hartree, PerCell, PerMolecule};
//! fn molecular_energy(_: Energy<Hartree, PerMolecule>) {}
//! molecular_energy(Energy::<Hartree, PerCell>::new(-1.0).unwrap());
//! ```

use crate::{constants, DftError, Matrix, Result};
use std::marker::PhantomData;

mod sealed {
    pub trait Unit {}
}

macro_rules! markers {
    ($($name:ident),+ $(,)?) => {$(
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name;
        impl sealed::Unit for $name {}
    )+};
}
markers!(
    Bohr,
    Angstrom,
    Hartree,
    ElectronVolt,
    PerMolecule,
    PerCell,
    HartreePerBohr,
    EvPerAngstrom,
    HartreePerBohr2,
    EvPerAngstrom2,
    HartreePerBohr3,
    Gigapascal
);

pub trait LengthUnit: sealed::Unit {}
impl LengthUnit for Bohr {}
impl LengthUnit for Angstrom {}
pub trait EnergyUnit: sealed::Unit {}
impl EnergyUnit for Hartree {}
impl EnergyUnit for ElectronVolt {}
pub trait EnergyScope: sealed::Unit {}
impl EnergyScope for PerMolecule {}
impl EnergyScope for PerCell {}
pub trait GradientUnit: sealed::Unit {}
impl GradientUnit for HartreePerBohr {}
impl GradientUnit for EvPerAngstrom {}
pub trait HessianUnit: sealed::Unit {}
impl HessianUnit for HartreePerBohr2 {}
impl HessianUnit for EvPerAngstrom2 {}
pub trait StressUnit: sealed::Unit {}
impl StressUnit for HartreePerBohr3 {}
impl StressUnit for Gigapascal {}

fn finite(values: impl Iterator<Item = f64>) -> Result<()> {
    if values.into_iter().all(f64::is_finite) {
        Ok(())
    } else {
        Err(DftError::InvalidInput(
            "physical quantity contains a nonfinite value".into(),
        ))
    }
}

/// Cartesian positions in a declared length unit. Conversion consumes and
/// scales the existing allocation, without copying the coordinate array.
#[repr(transparent)]
#[derive(Clone, Debug, PartialEq)]
pub struct Coordinates<U: LengthUnit> {
    values: Vec<[f64; 3]>,
    unit: PhantomData<U>,
}
impl<U: LengthUnit> Coordinates<U> {
    pub fn new(values: Vec<[f64; 3]>) -> Result<Self> {
        finite(values.iter().flatten().copied())?;
        Ok(Self {
            values,
            unit: PhantomData,
        })
    }
    pub fn as_slice(&self) -> &[[f64; 3]] {
        &self.values
    }
    pub fn into_values(self) -> Vec<[f64; 3]> {
        self.values
    }
    fn scaled<V: LengthUnit>(mut self, scale: f64) -> Result<Coordinates<V>> {
        for v in self.values.iter_mut().flatten() {
            *v *= scale;
        }
        Coordinates::new(self.values)
    }
}
impl Coordinates<Angstrom> {
    pub fn into_bohr(self) -> Result<Coordinates<Bohr>> {
        self.scaled(constants::ANGSTROM_TO_BOHR)
    }
}
impl Coordinates<Bohr> {
    pub fn into_angstrom(self) -> Result<Coordinates<Angstrom>> {
        self.scaled(constants::BOHR_TO_ANGSTROM)
    }
}

/// One position; useful for parsers without allocating a one-element vector.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Position<U: LengthUnit> {
    values: [f64; 3],
    unit: PhantomData<U>,
}
impl<U: LengthUnit> Position<U> {
    pub fn new(values: [f64; 3]) -> Result<Self> {
        finite(values.into_iter())?;
        Ok(Self {
            values,
            unit: PhantomData,
        })
    }
    pub fn into_array(self) -> [f64; 3] {
        self.values
    }
}
impl Position<Angstrom> {
    pub fn into_bohr(self) -> Result<Position<Bohr>> {
        Position::new(self.values.map(|v| v * constants::ANGSTROM_TO_BOHR))
    }
}

/// An energy per molecule or per periodic cell. The normalization is not
/// changed by converting Hartree to eV. Molar conversion is a distinct type.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Energy<U: EnergyUnit, S: EnergyScope> {
    value: f64,
    meaning: PhantomData<(U, S)>,
}
impl<U: EnergyUnit, S: EnergyScope> Energy<U, S> {
    pub fn new(value: f64) -> Result<Self> {
        finite(std::iter::once(value))?;
        Ok(Self {
            value,
            meaning: PhantomData,
        })
    }
    pub fn value(self) -> f64 {
        self.value
    }
}
impl<S: EnergyScope> Energy<Hartree, S> {
    pub fn into_ev(self) -> Result<Energy<ElectronVolt, S>> {
        Energy::new(self.value * constants::HARTREE_TO_EV)
    }
    /// One mole of the declared entities (molecules or cells).
    pub fn into_kcal_per_mol(self) -> Result<MolarEnergy<S>> {
        let value = self.value * constants::HARTREE_TO_KCAL_MOL;
        finite(std::iter::once(value))?;
        Ok(MolarEnergy {
            value,
            scope: PhantomData,
        })
    }
}
impl<S: EnergyScope> Energy<ElectronVolt, S> {
    pub fn into_hartree(self) -> Result<Energy<Hartree, S>> {
        Energy::new(self.value * constants::EV_TO_HARTREE)
    }
}

/// kcal per mole of molecules or cells; not interchangeable with energy.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MolarEnergy<S: EnergyScope> {
    value: f64,
    scope: PhantomData<S>,
}
impl<S: EnergyScope> MolarEnergy<S> {
    pub fn kcal_per_mol(self) -> f64 {
        self.value
    }
    pub fn into_hartree_per_entity(self) -> Result<Energy<Hartree, S>> {
        Energy::new(self.value * constants::KCAL_MOL_TO_HARTREE)
    }
}

macro_rules! vector_quantity {
    ($name:ident) => {
        #[repr(transparent)]
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name<U: GradientUnit = HartreePerBohr> {
            values: Vec<[f64; 3]>,
            unit: PhantomData<U>,
        }
        impl<U: GradientUnit> $name<U> {
            pub fn new(values: Vec<[f64; 3]>) -> Result<Self> {
                finite(values.iter().flatten().copied())?;
                Ok(Self {
                    values,
                    unit: PhantomData,
                })
            }
            pub fn as_slice(&self) -> &[[f64; 3]] {
                &self.values
            }
            pub fn into_values(self) -> Vec<[f64; 3]> {
                self.values
            }
        }
        impl $name<HartreePerBohr> {
            pub fn into_ev_per_angstrom(mut self) -> Result<$name<EvPerAngstrom>> {
                for v in self.values.iter_mut().flatten() {
                    *v *= constants::HARTREE_TO_EV / constants::BOHR_TO_ANGSTROM;
                }
                $name::new(self.values)
            }
        }
    };
}
vector_quantity!(EnergyGradient);
vector_quantity!(Force);
impl<U: GradientUnit> EnergyGradient<U> {
    /// Force is the negative energy gradient, with exactly the same unit.
    pub fn into_force(mut self) -> Force<U> {
        for v in self.values.iter_mut().flatten() {
            *v = -*v;
        }
        Force {
            values: self.values,
            unit: PhantomData,
        }
    }
}
impl<U: GradientUnit> Force<U> {
    pub fn into_energy_gradient(mut self) -> EnergyGradient<U> {
        for v in self.values.iter_mut().flatten() {
            *v = -*v;
        }
        EnergyGradient {
            values: self.values,
            unit: PhantomData,
        }
    }
}

/// Unweighted Cartesian energy Hessian, in atom-major x/y/z ordering.
/// This is deliberately a different type from stress or a mass-weighted matrix.
#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct CartesianHessian<U: HessianUnit = HartreePerBohr2> {
    matrix: Matrix,
    unit: PhantomData<U>,
}
impl<U: HessianUnit> CartesianHessian<U> {
    pub fn new(matrix: Matrix) -> Result<Self> {
        if matrix.rows != matrix.cols || matrix.rows % 3 != 0 {
            return Err(DftError::InvalidInput(
                "Cartesian Hessian must be square with dimension 3N".into(),
            ));
        }
        finite(matrix.as_slice().iter().copied())?;
        Ok(Self {
            matrix,
            unit: PhantomData,
        })
    }
    pub fn as_matrix(&self) -> &Matrix {
        &self.matrix
    }
}
impl CartesianHessian<HartreePerBohr2> {
    pub fn into_ev_per_angstrom2(mut self) -> Result<CartesianHessian<EvPerAngstrom2>> {
        for v in self.matrix.as_mut_slice() {
            *v *= constants::HARTREE_TO_EV / constants::BOHR_TO_ANGSTROM.powi(2);
        }
        CartesianHessian::new(self.matrix)
    }
}

/// Three-dimensional tensile-positive Cauchy stress, distinct from the
/// unnormalized strain derivative dE/depsilon. Low-dimensional surface and
/// line stresses require their own conventions and are not this type.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stress<U: StressUnit = HartreePerBohr3> {
    values: [[f64; 3]; 3],
    unit: PhantomData<U>,
}
impl<U: StressUnit> Stress<U> {
    pub fn new(values: [[f64; 3]; 3]) -> Result<Self> {
        finite(values.iter().flatten().copied())?;
        Ok(Self {
            values,
            unit: PhantomData,
        })
    }
    pub fn as_array(&self) -> &[[f64; 3]; 3] {
        &self.values
    }
}
impl Stress<HartreePerBohr3> {
    pub fn into_gigapascal(mut self) -> Result<Stress<Gigapascal>> {
        for v in self.values.iter_mut().flatten() {
            *v *= constants::HARTREE_PER_BOHR3_TO_GPA;
        }
        Stress::new(self.values)
    }
}
