// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic scalar/vector-potential vertices in independent (k+q,k) spaces.
//! Fields use exp(i*q.r-i*z*t). The scalar input is electron potential energy,
//! so E=i*q*V+i*z*A and B=i*q cross A in reduced atomic units.

use super::planewave::{plane_wave_current_shell_pair, plane_wave_shell_pair};
use super::{BlochMatrix, PeriodicGeometry};
use crate::basis::{n_sph, MolBasis};
use crate::math::Vec3;
use crate::response::{complex::Complex, Frequency};
use crate::system::Molecule;
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct PeriodicFieldOptions {
    pub k_mesh: [usize; 3],
    pub basis_cutoff: f64,
    pub k_offset: Vec3,
    /// Cartesian wave vector in inverse Bohr, including open-direction variation.
    pub wave: Vec3,
    pub frequency: Frequency,
    pub scalar_potential: Complex,
    pub vector_potential: [Complex; 3],
}
impl Default for PeriodicFieldOptions {
    fn default() -> Self {
        Self {
            k_mesh: [1; 3],
            basis_cutoff: 12.0,
            k_offset: Vec3::zero(),
            wave: Vec3::zero(),
            frequency: Frequency::Static,
            scalar_potential: Complex::ZERO,
            vector_potential: [Complex::ZERO; 3],
        }
    }
}
pub struct PeriodicFieldReport {
    pub k_points: Vec<Vec3>,
    pub target_k_points: Vec<Vec3>,
    pub second_target_k_points: Vec<Vec3>,
    pub wave: Vec3,
    pub frequency: Complex,
    pub density_vertex: Vec<BlochMatrix>,
    /// Anticommutator current, Cartesian axis then ket k.
    pub current_vertex: [Vec<BlochMatrix>; 3],
    /// dH/dlambda at zero field for simultaneous input scalar/vector amplitudes.
    pub first_order: Vec<BlochMatrix>,
    /// d2H/dlambda2: (A.A) times density vertex at 2q, without complex conjugation.
    pub second_order: Vec<BlochMatrix>,
    /// Explicit diamagnetic mixed vertex delta_ab * density_vertex(2q).
    pub contact_vertex: Vec<BlochMatrix>,
    pub electric_field: [Complex; 3],
    pub magnetic_field: [Complex; 3],
    pub transversality_defect: f64,
    pub formal_charge: i32,
    pub model: &'static str,
    pub units: &'static str,
    pub axis_convention: &'static str,
    pub warnings: Vec<String>,
}

fn add(target: &mut BlochMatrix, source: &BlochMatrix, scale: Complex) {
    for i in 0..target.rows() {
        for j in 0..target.cols() {
            let z = Complex::new(source.re[(i, j)], source.im[(i, j)]) * scale;
            target.re[(i, j)] += z.re;
            target.im[(i, j)] += z.im;
        }
    }
}
/// Exact contracted Gaussian image sums; no integration grid, SCF or finite field.
/// The first and second orders are actual minimal-coupling Hamiltonian
/// derivatives. Screening of these sources is a separate response calculation.
pub fn periodic_field_operators(
    mol: &Molecule,
    basis: &MolBasis,
    geometry: &PeriodicGeometry,
    options: &PeriodicFieldOptions,
) -> Result<PeriodicFieldReport> {
    options.frequency.validate()?;
    if geometry.dimension() == 0
        || mol.is_empty()
        || basis.nao_sph == 0
        || !options.basis_cutoff.is_finite()
        || options.basis_cutoff <= 0.0
        || !options.wave.norm().is_finite()
        || !options.k_offset.norm().is_finite()
        || !options.scalar_potential.is_finite()
        || options.vector_potential.iter().any(|z| !z.is_finite())
    {
        return Err(DftError::InvalidInput("periodic field operators require 1D/2D/3D geometry, atoms, AOs and finite field/cutoff controls".into()));
    }
    if mol.atoms.iter().any(|a| !a.position.norm().is_finite())
        || basis.shells.is_empty()
        || basis.shells.iter().any(|s| {
            s.atom >= mol.len()
                || s.l > crate::basis::aos::MAX_L
                || !s.center.norm().is_finite()
                || s.sph_off + n_sph(s.l) > basis.nao_sph
                || s.exps.is_empty()
                || s.exps.len() != s.coefs.len()
                || s.exps.iter().any(|x| !x.is_finite() || *x <= 0.0)
                || s.coefs.iter().any(|x| !x.is_finite())
        })
    {
        return Err(DftError::InvalidInput(
            "invalid contracted field basis".into(),
        ));
    }
    let ks = geometry
        .k_points(options.k_mesh)?
        .into_iter()
        .map(|k| k + options.k_offset)
        .collect::<Vec<_>>();
    let q = options.wave;
    let (density, current) = vertices(basis, geometry, &ks, q, options.basis_cutoff, true)?;
    let (contact, _) = vertices(basis, geometry, &ks, q * 2.0, options.basis_cutoff, false)?;
    let mut first = vec![BlochMatrix::zeros(basis.nao_sph, basis.nao_sph); ks.len()];
    let mut second = first.clone();
    let aa = options
        .vector_potential
        .iter()
        .map(|&a| a * a)
        .fold(Complex::ZERO, |a, b| a + b);
    for k in 0..ks.len() {
        add(&mut first[k], &density[k], options.scalar_potential);
        for a in 0..3 {
            add(&mut first[k], &current[a][k], options.vector_potential[a]);
        }
        add(&mut second[k], &contact[k], aa);
    }
    let z = options.frequency.energy()?;
    let iu = Complex::new(0.0, 1.0);
    let electric = std::array::from_fn(|a| {
        iu * (options.scalar_potential * q.get(a) + z * options.vector_potential[a])
    });
    let av = options.vector_potential;
    let magnetic = [
        iu * (av[2] * q.y - av[1] * q.z),
        iu * (av[0] * q.z - av[2] * q.x),
        iu * (av[1] * q.x - av[0] * q.y),
    ];
    let defect = magnetic
        .iter()
        .enumerate()
        .map(|(a, &b)| b * q.get(a))
        .fold(Complex::ZERO, |a, b| a + b)
        .abs();
    Ok(PeriodicFieldReport {
        target_k_points:ks.iter().map(|&k|k+q).collect(),
        second_target_k_points:ks.iter().map(|&k|k+q*2.0).collect(),
        k_points:ks,wave:q,frequency:z,density_vertex:density,current_vertex:current,
        first_order:first,second_order:second,contact_vertex:contact,
        electric_field:electric,magnetic_field:magnetic,transversality_defect:defect,
        formal_charge:mol.charge,
        model:"Analytic dimensional minimal-coupling Fourier Hamiltonian vertices; independent shifted Bloch spaces",
        units:"Hartree Hamiltonian derivatives for reduced atomic potentials; q in inverse Bohr",
        axis_convention:"ket k, target spherical AO row, ket spherical AO column; current axis x,y,z; second order connects k+2q to k",
        warnings:vec![
            "A real field includes the conjugate -q,-conjugate-frequency mode; one q block is not Hermitianized.".into(),
            "At q=0 static periodic scalar/vector potentials do not represent \
                uniform electric or magnetic fields; Berry electric response and \
                flux-compatible magnetic cells are separate drivers.".into(),
            "These are bare analytic field vertices, not screened response, finite-field SCF or a susceptibility.".into(),
        ],
    })
}
pub(crate) fn vertices(
    basis: &MolBasis,
    geometry: &PeriodicGeometry,
    ks: &[Vec3],
    wave: Vec3,
    cutoff: f64,
    want_current: bool,
) -> Result<(Vec<BlochMatrix>, [Vec<BlochMatrix>; 3])> {
    let blank = || vec![BlochMatrix::zeros(basis.nao_sph, basis.nao_sph); ks.len()];
    let mut density = blank();
    let mut current = std::array::from_fn(|_| blank());
    for left in &basis.shells {
        for right in &basis.shells {
            for index in geometry.nearby_translation_indices(right.center - left.center, cutoff)? {
                let shift = geometry.lattice().translation(index);
                let mut shifted = right.clone();
                shifted.center = right.center + shift;
                let (nr, ni) = plane_wave_shell_pair(left, &shifted, wave);
                let momentum = if want_current {
                    Some(std::array::from_fn::<_, 3, _>(|axis| {
                        plane_wave_current_shell_pair(left, &shifted, wave, axis)
                    }))
                } else {
                    None
                };
                for (k, &kv) in ks.iter().enumerate() {
                    let angle = kv.dot(shift);
                    let phase = Complex::new(angle.cos(), angle.sin());
                    for i in 0..nr.rows {
                        for j in 0..nr.cols {
                            let row = left.sph_off + i;
                            let col = right.sph_off + j;
                            let n = Complex::new(nr[(i, j)], ni[(i, j)]) * phase;
                            density[k].re[(row, col)] += n.re;
                            density[k].im[(row, col)] += n.im;
                            if let Some(p) = &momentum {
                                for axis in 0..3 {
                                    let v =
                                        Complex::new(p[axis].0[(i, j)], p[axis].1[(i, j)]) * phase;
                                    current[axis][k].re[(row, col)] += v.re;
                                    current[axis][k].im[(row, col)] += v.im;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((density, current))
}
