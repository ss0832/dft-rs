// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Stationary complex-density contractions of shared integral derivatives.
//! The quadratic provider has all four complex Pauli exchange components.
//! A strictly real collinear projector uses the shared AO/Becke gradient.

use super::{add_source_gradient, assemble_gradient_density, one_particle_gradient};
use crate::basis::MolBasis;
use crate::dfpt::orbitals::exact_spin_blocks;
use crate::grid::MolecularGrid;
use crate::integrals::{deriv, schwarz_matrix, CoulombKernel};
use crate::linalg::{AccurateSum, Matrix};
use crate::periodic::spinor_scf::{pauli_components, SpinorScfReport};
use crate::scf::ScfOptions;
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::Result;

pub(crate) fn stationary_gradient(
    molecule: &Molecule,
    basis: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    reference: &SpinorScfReport,
    options: &ScfOptions,
) -> Result<Option<Vec<f64>>> {
    if options.skala.is_some()
        || options.pcm.is_some()
        || options.density_fit.is_some()
        || functional.required_vv10().is_some() && options.vv10.is_none()
        || !matches!(options.kernel, CoulombKernel::Coulomb)
        || reference.density.len() != 1
    {
        return Ok(None);
    }
    let density = &reference.density[0];
    let quadratic = functional.parts.is_empty() && options.vv10.is_none();
    let collinear = if quadratic {
        None
    } else if density.im.as_slice().iter().all(|&v| v == 0.0) {
        exact_spin_blocks(density)
    } else {
        None
    };
    if !quadratic && collinear.is_none() {
        return Ok(None);
    }
    let n = basis.nao_sph;
    let coefficients = &reference.bands[0].coefficients;
    let mut weighted = Matrix::zeros(n, n);
    // At Mermin stationarity, changes in occupations cancel against the
    // entropy derivative at fixed total electron number. Only the explicit
    // geometry derivative and W = C diag(f epsilon) C^H remain.
    for i in 0..n {
        for j in 0..n {
            let mut value = AccurateSum::default();
            for spin in 0..2 {
                for (k, occupation) in reference.occupations[0].iter().enumerate() {
                    let energy = occupation * reference.bands[0].energies[k];
                    value.add_product(
                        energy * coefficients.re[(spin * n + i, k)],
                        coefficients.re[(spin * n + j, k)],
                    );
                    value.add_product(
                        energy * coefficients.im[(spin * n + i, k)],
                        coefficients.im[(spin * n + j, k)],
                    );
                }
            }
            weighted[(i, j)] = value.value();
        }
    }
    if let Some([alpha, beta]) = collinear {
        // The spin-current and transverse descriptors vanish identically.
        // Under nuclear motion at fixed AO coefficients, the covariant XC
        // reduces to the native collinear AO and co-moving quadrature term.
        let gradient = assemble_gradient_density(
            molecule,
            basis,
            functional,
            grid,
            [&alpha.re, &beta.re],
            options,
            &weighted,
        )?;
        return Ok(Some(
            gradient.into_iter().flat_map(|g| g.to_array()).collect(),
        ));
    }
    let components = pauli_components(density)?;
    let charges = basis.effective_charges(molecule);
    let total = &components[0].re;
    let mut gradient = one_particle_gradient(molecule, basis, &charges, total, &weighted);
    if !basis.ecps.is_empty() {
        let core = crate::integrals::ecp::ecp_gradient(basis, molecule.len(), total);
        for (value, contribution) in gradient.iter_mut().zip(core) {
            *value += contribution;
        }
    }
    for (kernel, coulomb, exchange) in [
        (CoulombKernel::Coulomb, 1.0, functional.hybrid.cx),
        (
            CoulombKernel::Erf {
                omega: functional.hybrid.omega,
            },
            0.0,
            functional.hybrid.clr,
        ),
    ] {
        if coulomb == 0.0 && exchange == 0.0 {
            continue;
        }
        // Ex = -cx/4 sum_a Tr[P_a K(P_a)], including complex orbital
        // coherences. Both real and imaginary Pauli parts enter with cx/2
        // in the common -1/2 weighted-real-channel convention.
        let channels: Vec<_> = components
            .iter()
            .flat_map(|p| [(&p.re, 0.5 * exchange), (&p.im, 0.5 * exchange)])
            .collect();
        let schwarz = schwarz_matrix(basis, kernel);
        let contribution = deriv::eri_gradient_channels(
            basis,
            molecule.len(),
            total,
            &channels,
            kernel,
            coulomb,
            &schwarz,
            1e-12,
        );
        for (g, value) in gradient.iter_mut().zip(contribution) {
            *g += value;
        }
    }
    add_source_gradient(
        molecule,
        basis,
        &charges,
        total,
        options.external_field,
        &mut gradient,
    );
    Ok(Some(
        gradient.into_iter().flat_map(|g| g.to_array()).collect(),
    ))
}
