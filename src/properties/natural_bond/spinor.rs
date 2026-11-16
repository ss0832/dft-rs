// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Exact reduction of a globally collinear spinor reference to its two
//! real spatial Lewis-search channels.

use crate::linalg::Matrix;
use crate::periodic::{spinor_scf::pauli_components, BlochMatrix};
use crate::{DftError, Result};

/// Real density and Fock blocks in the detected common spin frame.
#[derive(Clone, Debug)]
pub struct CollinearSpinorChannels {
    pub density: [Matrix; 2],
    pub fock: [Matrix; 2],
    pub axis: [f64; 3],
    pub density_defect: f64,
    pub fock_defect: f64,
}

fn largest_spin(components: &[BlochMatrix; 4]) -> [f64; 3] {
    let mut largest = [0.0; 3];
    let mut largest_norm = 0.0;
    for i in 0..components[0].re.as_slice().len() {
        let spin: [f64; 3] = std::array::from_fn(|axis| components[axis + 1].re.as_slice()[i]);
        let norm = spin.iter().map(|x| x * x).sum::<f64>();
        if norm > largest_norm {
            largest = spin;
            largest_norm = norm;
        }
    }
    largest
}

fn reduce(components: &[BlochMatrix; 4], axis: [f64; 3]) -> ([Matrix; 2], f64, f64) {
    let mut defect = 0.0_f64;
    let mut scale = 1.0_f64;
    let mut blocks = [components[0].re.clone(), components[0].re.clone()];
    for i in 0..blocks[0].as_slice().len() {
        let spin: [f64; 3] = std::array::from_fn(|a| components[a + 1].re.as_slice()[i]);
        let longitudinal: f64 = (0..3).map(|a| axis[a] * spin[a]).sum();
        for a in 0..3 {
            defect = defect.max((spin[a] - axis[a] * longitudinal).abs());
            scale = scale.max(spin[a].abs());
        }
        for component in components {
            defect = defect.max(component.im.as_slice()[i].abs());
        }
        let charge = components[0].re.as_slice()[i];
        scale = scale.max(charge.abs());
        blocks[0].as_mut_slice()[i] = 0.5 * (charge + longitudinal);
        blocks[1].as_mut_slice()[i] = 0.5 * (charge - longitudinal);
    }
    (blocks, defect, scale)
}

/// Rotate a common spin axis into the quantization axis. The density and
/// Fock operator must both be real after this single spin rotation.
/// The preferred direction fixes the sign of otherwise equivalent labels.
pub fn collinear_channels(
    density: &BlochMatrix,
    fock: &BlochMatrix,
    preferred_direction: [f64; 3],
) -> Result<CollinearSpinorChannels> {
    let density = pauli_components(density)?;
    let fock = pauli_components(fock)?;
    if density[0].rows() != fock[0].rows() {
        return Err(DftError::InvalidInput(
            "spinor natural-bond density and Fock matrix dimensions differ".into(),
        ));
    }
    let length = |v: [f64; 3]| v[0].hypot(v[1]).hypot(v[2]);
    let preferred_norm = length(preferred_direction);
    if !preferred_norm.is_finite() || preferred_norm == 0.0 {
        return Err(DftError::InvalidInput(
            "spinor natural-bond spin axis must be finite and nonzero".into(),
        ));
    }
    let mut axis = largest_spin(&density);
    if length(axis) < 1e-12 {
        axis = largest_spin(&fock);
    }
    if length(axis) < 1e-12 {
        axis = preferred_direction;
    }
    let norm = length(axis);
    let sign = if (0..3)
        .map(|a| axis[a] * preferred_direction[a])
        .sum::<f64>()
        < 0.0
    {
        -1.0
    } else {
        1.0
    };
    let axis = axis.map(|component| sign * component / norm);
    let (density, density_defect, density_scale) = reduce(&density, axis);
    let (fock, fock_defect, fock_scale) = reduce(&fock, axis);
    if density_defect > 1e-9 * density_scale || fock_defect > 1e-9 * fock_scale {
        return Err(DftError::Unsupported {
            feature: "natural-bond analysis of complex spatial or noncollinear spinor channels".into(),
            missing: format!("complex Lewis and hybrid orbital search (density defect {density_defect:.6e}, Fock defect {fock_defect:.6e} Hartree)"),
        });
    }
    Ok(CollinearSpinorChannels {
        density,
        fock,
        axis,
        density_defect,
        fock_defect,
    })
}
