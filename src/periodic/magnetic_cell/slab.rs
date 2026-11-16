// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Magnetic slab Coulomb operators: discrete in-plane waves, continuous
//! open-normal Fourier integration, including the nonconstant G_parallel=0
//! Green function. No artificial periodicity or physical denominator floor.

use super::*;
use crate::periodic::ewald::EwaldParams;

fn modes(
    cell: &MagneticCell,
    basis: &MolBasis,
    functional: &ResolvedFunctional,
    sources: &[ElectrostaticSource],
    cutoff: f64,
    image_cutoff: f64,
    order: usize,
) -> Result<Vec<FourierMode>> {
    transfer_modes(
        cell,
        basis,
        functional,
        sources,
        cutoff,
        image_cutoff,
        order,
        cell.bloch_wave,
        cell.bloch_wave,
    )
}

/// Continuous-normal Coulomb vertices in distinct (left,right) Bloch spaces.
/// Only the exactly diagonal in-plane zero transfer uses the finite Green
/// zero mode; all nonzero transfers keep their physical denominator.
pub(super) fn transfer_modes(
    cell: &MagneticCell,
    basis: &MolBasis,
    functional: &ResolvedFunctional,
    sources: &[ElectrostaticSource],
    cutoff: f64,
    image_cutoff: f64,
    order: usize,
    left: Vec3,
    right: Vec3,
) -> Result<Vec<FourierMode>> {
    let axes: Vec<_> = (0..3)
        .filter(|&a| cell.geometry.boundary().axes[a])
        .collect();
    let normal = cell.geometry.lattice().a[axes[0]]
        .cross(cell.geometry.lattice().a[axes[1]])
        .normalized();
    let area = cell.geometry.measure();
    let eta = 0.5_f64;
    let (nodes, weights) = gauss_legendre(order);
    let shift = left - right;
    let mut planes = cell.geometry.reciprocal_within(cutoff + shift.norm())?;
    planes.push(Vec3::zero());
    let per_plane = planes
        .par_iter()
        .map(|&reciprocal| -> Result<Vec<FourierMode>> {
            let g = reciprocal + shift;
            if g.norm() >= cutoff {
                return Ok(Vec::new());
            }
            let extent = (cutoff * cutoff - g.norm2()).max(0.0).sqrt();
            if extent == 0.0 {
                return Ok(Vec::new());
            }
            let zero = g.norm2() == 0.0;
            // qz=scale*tan(theta) resolves small in-plane wave numbers.
            // For nonzero G, dqz/(G^2+qz^2)=dtheta/|G| exactly.
            // The zero-plane scale is an integration coordinate, not a
            // modification of any physical Coulomb denominator.
            let scale = if zero { 1.0 } else { g.norm() };
            let angle = (extent / scale).atan();
            let mut result = Vec::with_capacity(order + usize::from(zero));
            let mut subtraction = 0.0;
            let mut lr_subtraction = 0.0;
            for (&node, &weight) in nodes.iter().zip(&weights) {
                let theta = angle * node;
                let qz = scale * theta.tan();
                let wave = g + normal * qz;
                // The Gauss rule has even order and never samples the singular point.
                let jacobian = angle * scale / theta.cos().powi(2);
                let coulomb = 2.0 * jacobian * weight / (area * wave.norm2());
                let lr = if functional.hybrid.omega == 0.0 {
                    0.0
                } else {
                    coulomb * (-wave.norm2() / (4.0 * functional.hybrid.omega.powi(2))).exp()
                };
                let nuclear = sources
                    .iter()
                    .map(|s| {
                        phase(-wave.dot(s.position))
                            * (-coulomb
                                * s.charge
                                * (-0.25 * wave.norm2() * s.width * s.width).exp())
                    })
                    .fold(Complex::ZERO, |a, b| a + b);
                if zero {
                    subtraction += coulomb * (-qz * qz / (4.0 * eta * eta)).exp();
                    // The range-separated zero mode is referenced to its value at
                    // equal open coordinates, retaining the exact omega=0 limit.
                    lr_subtraction += lr;
                }
                result.push(FourierMode {
                    wave,
                    density: cell.fourier_transfer(basis, image_cutoff, left, right, wave)?,
                    coulomb,
                    long_range: lr,
                    nuclear,
                });
            }
            if zero {
                // Integral of (exp[-qz^2/(4 eta^2)]-1)/qz^2 over the real line.
                // This restores G0(z)=-2 pi |z|/area analytically.
                let coulomb = -subtraction - 2.0 * PI.sqrt() / (area * eta);
                let nuclear =
                    Complex::from(-coulomb * sources.iter().map(|s| s.charge).sum::<f64>());
                result.push(FourierMode {
                    wave: Vec3::zero(),
                    density: cell.fourier_transfer(
                        basis,
                        image_cutoff,
                        left,
                        right,
                        Vec3::zero(),
                    )?,
                    coulomb,
                    long_range: -lr_subtraction,
                    nuclear,
                });
            }
            Ok(result)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(per_plane.into_iter().flatten().collect())
}

fn one_body(
    cell: &MagneticCell,
    basis: &MolBasis,
    options: &MagneticScfOptions,
    modes: &[FourierMode],
) -> Result<(BlochMatrix, BlochMatrix)> {
    let n = basis.nao_sph;
    let mut nuclear = BlochMatrix::zeros(n, n);
    let mut pauli = std::array::from_fn(|_| BlochMatrix::zeros(n, n));
    let c = 137.035_999_084_f64;
    for mode in modes {
        add(&mut nuclear, &mode.density, mode.nuclear);
        if options.nuclear_soc_scale != 0.0 {
            let p = [
                cell.momentum(basis, options.basis_cutoff, mode.wave, 0)?,
                cell.momentum(basis, options.basis_cutoff, mode.wave, 1)?,
                cell.momentum(basis, options.basis_cutoff, mode.wave, 2)?,
            ];
            for axis in 0..3 {
                let x = (axis + 1) % 3;
                let y = (axis + 2) % 3;
                let coefficient = I * mode.nuclear * (options.nuclear_soc_scale / (2.0 * c * c));
                add(&mut pauli[axis + 1], &p[y], coefficient * mode.wave.get(x));
                add(
                    &mut pauli[axis + 1],
                    &p[x],
                    coefficient * (-mode.wave.get(y)),
                );
            }
        }
    }
    Ok((nuclear, from_pauli(&pauli)?))
}

pub(super) fn new_backend<'a>(
    mol: &Molecule,
    basis: &MolBasis,
    cell: MagneticCell,
    functional: &'a ResolvedFunctional,
    mut options: MagneticScfOptions,
    spin_operator: Option<BlochMatrix>,
    sources: Vec<ElectrostaticSource>,
) -> Result<MagneticDftBackend<'a>> {
    if matches!(
        options.electrostatics.compensation,
        ChargeCompensation::UniformBackground
    ) {
        return Err(DftError::InvalidInput(
            "a charged magnetic slab requires a finite transverse Gaussian \
            countercharge; uniform background requires 3D periodicity"
                .into(),
        ));
    }
    let order = options.open_quadrature_order;
    if !(32..=512).contains(&order)
        || order % 4 != 0
        || !(16..=1024).contains(&options.electrostatics.quadrature_order)
        || !options.electrostatics.tolerance.is_finite()
        || options.electrostatics.tolerance <= 0.0
    {
        return Err(DftError::InvalidInput(
            "magnetic slab requires open Fourier order 32..512 divisible by four and \
            valid dimensional electrostatic quadrature controls"
                .into(),
        ));
    }
    let alpha = options
        .electrostatics
        .alpha
        .unwrap_or(PI.sqrt() / cell.geometry.measure().sqrt());
    let width = sources.iter().map(|s| s.width).fold(0.0_f64, f64::max);
    let margin = (-options.electrostatics.tolerance.min(1e-12).ln()).sqrt();
    let params = EwaldParams {
        alpha,
        rcut: options
            .electrostatics
            .real_cutoff
            .unwrap_or(margin * width.max(alpha.recip())),
        gcut: options
            .electrostatics
            .reciprocal_cutoff
            .unwrap_or(2.0 * alpha * margin),
    };
    let coarse_green = DimensionalCoulomb::new(
        cell.geometry.clone(),
        params,
        options.electrostatics.quadrature_order,
    )?;
    let source_green = DimensionalCoulomb::new(
        cell.geometry.clone(),
        params,
        2 * options.electrostatics.quadrature_order,
    )?;
    let coarse_source = coarse_green.evaluate(&sources)?;
    let source = source_green.evaluate(&sources)?;
    let source_defect = (source.0.total() - coarse_source.0.total()).abs().max(
        source
            .1
            .iter()
            .zip(&coarse_source.1)
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0_f64, f64::max),
    );
    let s = cell.fourier(basis, options.basis_cutoff, Vec3::zero())?;
    let kinetic = cell.kinetic(basis, options.basis_cutoff)?;
    let image_operator_defect = gap(
        &s,
        &cell.fourier(basis, 0.8 * options.basis_cutoff, Vec3::zero())?,
    )
    .max(gap(
        &kinetic,
        &cell.kinetic(basis, 0.8 * options.basis_cutoff)?,
    ));
    let mut history = Vec::new();
    let mut actual_order = order;
    let (modes, coarse_modes, nuclear, soc, nuclear_coarse, soc_coarse) = loop {
        let modes = modes(
            &cell,
            basis,
            functional,
            &sources,
            options.reciprocal_cutoff,
            options.basis_cutoff,
            actual_order,
        )?;
        let coarse_modes = self::modes(
            &cell,
            basis,
            functional,
            &sources,
            0.8 * options.reciprocal_cutoff,
            options.basis_cutoff,
            actual_order / 2,
        )?;
        let (nuclear, soc) = one_body(&cell, basis, &options, &modes)?;
        let (nuclear_coarse, soc_coarse) = one_body(&cell, basis, &options, &coarse_modes)?;
        let defects = [gap(&nuclear, &nuclear_coarse), gap(&soc, &soc_coarse)];
        history.push([actual_order as f64, defects[0], defects[1]]);
        if defects.iter().all(|d| *d <= options.operator_tolerance) || actual_order == 512 {
            break (
                modes,
                coarse_modes,
                nuclear,
                soc,
                nuclear_coarse,
                soc_coarse,
            );
        }
        actual_order = (2 * actual_order).min(512);
    };
    options.open_quadrature_order = actual_order;
    let zeeman = from_pauli(&std::array::from_fn(|a| {
        let mut v = s.clone();
        v.scale(if a == 0 { 0.0 } else { cell.field.get(a - 1) });
        v
    }))?;
    let mut spin =
        spin_operator.unwrap_or_else(|| BlochMatrix::zeros(2 * basis.nao_sph, 2 * basis.nao_sph));
    spin.add_scaled(&zeeman, 1.0);
    let mut core = kinetic.clone();
    core.add_scaled(&nuclear, 1.0);
    let mut initial = spin_diagonal(&core);
    initial.add_scaled(&spin, 1.0);
    initial.add_scaled(&soc, 1.0);
    for matrix in [&s, &kinetic, &nuclear, &soc, &initial] {
        if !valid(matrix, matrix.rows()) || matrix.hermiticity_error() > options.operator_tolerance
        {
            return Err(DftError::InvalidInput(format!(
                "magnetic slab operator violates finite Hermiticity before projection: {:.6e}",
                matrix.hermiticity_error()
            )));
        }
    }
    let grid = xc::Cache::new(functional, options.grid_order, mol);
    let electrons = mol.atoms.iter().map(|a| a.z as f64).sum::<f64>() - mol.charge as f64;
    Ok(MagneticDftBackend {
        cell,
        overlap: vec![spin_diagonal(&s)],
        initial: vec![initial],
        kinetic,
        nuclear,
        nuclear_coarse,
        soc,
        soc_coarse,
        zeeman,
        spin,
        modes,
        coarse_modes: Some(coarse_modes),
        open_refinement_history: history,
        source_energy: source.0.total(),
        sources,
        source_green,
        electric: None,
        electric_coarse: None,
        electric_background: Vec3::zero(),
        source_defect,
        image_operator_defect,
        functional,
        basis: basis.clone(),
        grid,
        electrons,
        options,
    })
}
