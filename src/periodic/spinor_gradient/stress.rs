// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Field-free stationary dimensional complex-spinor affine stress in the same gauge.
use super::stationary::{
    accepted_reference, add_exchange_geometry, density_images, exchange_geometry_terms,
    FieldFreeFrame,
};
use crate::math::Vec3;
use crate::periodic::coulomb_boundary::PeriodicCoulombStrain;
use crate::periodic::integrals::Operator;
use crate::periodic::spinor_scf::{
    pauli_components, SpinorDftBackend, SpinorScfConfig, SpinorScfReport,
};
use crate::periodic::spinor_xc::{SpinorXcGridContext, SpinorXcGridTangent};
use crate::periodic::{BlochMatrix, PeriodicGeometry};
use crate::{DftError, Result};

pub struct DimensionalStress {
    /// Unprojected dE/d epsilon_ab, Hartree; all nine components retained.
    pub strain_derivative: [[f64; 3]; 3],
    /// dE/d epsilon_ab divided by the periodic length, area or volume.
    pub stress: [[f64; 3]; 3],
    pub pressure: f64,
    pub kinetic: [[f64; 3]; 3],
    pub electrostatic: [[f64; 3]; 3],
    pub ecp: [[f64; 3]; 3],
    pub full_exchange: [[f64; 3]; 3],
    pub long_range_exchange: [[f64; 3]; 3],
    pub semilocal_xc: [[f64; 3]; 3],
    pub nonlocal_correlation: Option<[[f64; 3]; 3]>,
    pub pulay: [[f64; 3]; 3],
    pub external_field: Option<[[f64; 3]; 3]>,
    pub external_field_energy_defect: Option<f64>,
    pub fresh_orbital_residual: f64,
    pub fresh_occupation_consistency: f64,
    pub fresh_number_error: f64,
    pub reference_energy_defect: f64,
    pub reference_fock_defect: f64,
    pub xc_reference_energy_defect: f64,
    pub coulomb_reference_energy_defect: f64,
    pub coulomb_quadrature_defect: f64,
}
pub type SpinorStress = DimensionalStress;

pub(crate) fn evaluate(
    frame: FieldFreeFrame<'_>,
    backend: &SpinorDftBackend<'_>,
    reference: &SpinorScfReport,
    config: &SpinorScfConfig,
) -> Result<SpinorStress> {
    let accepted = accepted_reference(&frame, backend, reference, config)?;
    let pc = reference
        .density
        .iter()
        .map(pauli_components)
        .collect::<Result<Vec<_>>>()?;
    let charge: Vec<_> = pc.iter().map(|p| p[0].clone()).collect();
    let points = frame.geometry.k_points(frame.options.scf.k_mesh)?;
    let one = one_body_affine(&frame, &points, &charge, &accepted.weighted)?;
    let coulomb = frame.coulomb.affine_strain(&charge)?;
    let mut exchange = exchange_geometry_terms(frame.molecule.len());
    if frame.functional.hybrid.cx != 0.0 || frame.functional.hybrid.clr != 0.0 {
        for component in 0..4 {
            let density: Vec<_> = pc.iter().map(|p| p[component].clone()).collect();
            add_exchange_geometry(&frame, &points, &density, -0.25, &mut exchange)?;
        }
    }
    let (semilocal_xc, xc_defect) = xc_strain(
        &frame,
        &pc,
        accepted.xc_energy,
        accepted.angular_order,
        config.energy_tolerance,
    )?;
    let vv10 = frame.vv10_geometry(&points, &charge, accepted.nonlocal_energy, true)?;
    let mut stress = assemble_stress(
        frame.geometry,
        one,
        coulomb,
        frame.ecp_geometry(&points, &charge)?.strain,
        exchange.map(|term| term.strain_derivative),
        semilocal_xc,
        StressQuality {
            orbital: accepted.orbital,
            consistency: accepted.consistency,
            number_error: accepted.number_error,
            energy_defect: accepted.energy_defect,
            fock_defect: accepted.fock_defect,
            xc_defect: xc_defect.max(vv10.as_ref().map_or(0.0, |term| term.energy_defect)),
        },
    )?;
    if let Some(vv10) = vv10 {
        let strain = vv10.strain.expect("VV10 strain requested");
        stress.add_affine_derivative(frame.geometry, strain)?;
        stress.nonlocal_correlation = Some(strain);
    }
    Ok(stress)
}

pub(crate) fn one_body_affine(
    frame: &FieldFreeFrame<'_>,
    points: &[crate::math::Vec3],
    charge: &[BlochMatrix],
    weighted: &[BlochMatrix],
) -> Result<[[[f64; 3]; 3]; 2]> {
    let mut one = [[[0.0; 3]; 3]; 2];
    for (index, operator, density, factor) in [
        (0, Operator::Kinetic, charge, 1.0),
        (1, Operator::Overlap, weighted, -1.0),
    ] {
        let (images, _) =
            density_images(frame.geometry, points, density, frame.options.basis_cutoff)?;
        one[index] = crate::periodic::strain::dimensional_two_centre_strain(
            frame.basis,
            frame.geometry,
            frame.options.basis_cutoff,
            operator,
            &images,
        )?;
        for row in &mut one[index] {
            for v in row {
                *v *= factor;
            }
        }
    }
    Ok(one)
}

#[derive(Clone, Copy)]
pub(crate) struct StressQuality {
    pub orbital: f64,
    pub consistency: f64,
    pub number_error: f64,
    pub energy_defect: f64,
    pub fock_defect: f64,
    pub xc_defect: f64,
}

pub(crate) fn assemble_stress(
    geometry: &PeriodicGeometry,
    one: [[[f64; 3]; 3]; 2],
    coulomb: PeriodicCoulombStrain,
    ecp: [[f64; 3]; 3],
    exchange: [[[f64; 3]; 3]; 2],
    semilocal_xc: [[f64; 3]; 3],
    quality: StressQuality,
) -> Result<SpinorStress> {
    if !coulomb.converged {
        return Err(DftError::InvalidInput(
            "dimensional stress requires the converged same-gauge Coulomb strain provider".into(),
        ));
    }
    let electrostatic = std::array::from_fn(|a| std::array::from_fn(|b| coulomb.total[3 * a + b]));
    let [kinetic, pulay] = one;
    let [full_exchange, long_range_exchange] = exchange;
    let derivative: [[f64; 3]; 3] = std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            kinetic[a][b]
                + electrostatic[a][b]
                + ecp[a][b]
                + full_exchange[a][b]
                + long_range_exchange[a][b]
                + semilocal_xc[a][b]
                + pulay[a][b]
        })
    });
    let (stress, pressure) = normalized_stress(geometry, derivative)?;
    Ok(SpinorStress {
        strain_derivative: derivative,
        stress,
        pressure,
        kinetic,
        electrostatic,
        ecp,
        full_exchange,
        long_range_exchange,
        semilocal_xc,
        nonlocal_correlation: None,
        pulay,
        external_field: None,
        external_field_energy_defect: None,
        fresh_orbital_residual: quality.orbital,
        fresh_occupation_consistency: quality.consistency,
        fresh_number_error: quality.number_error,
        reference_energy_defect: quality.energy_defect,
        reference_fock_defect: quality.fock_defect,
        xc_reference_energy_defect: quality.xc_defect,
        coulomb_reference_energy_defect: coulomb.reference_energy_defect,
        coulomb_quadrature_defect: coulomb.quadrature_defect,
    })
}

type StressTensor = [[f64; 3]; 3];

fn normalized_stress(
    geometry: &PeriodicGeometry,
    derivative: StressTensor,
) -> Result<(StressTensor, f64)> {
    let measure = geometry.measure();
    let stress = derivative.map(|row| row.map(|value| value / measure));
    let dimension = geometry.dimension();
    let pressure = if dimension == 3 {
        -(stress[0][0] + stress[1][1] + stress[2][2]) / 3.0
    } else {
        let mut trace = 0.0;
        for b in 0..3 {
            let axis =
                crate::math::Vec3::new(f64::from(b == 0), f64::from(b == 1), f64::from(b == 2));
            let projected = geometry.parallel(axis);
            for (a, row) in stress.iter().enumerate() {
                trace += row[b] * projected.get(a);
            }
        }
        -trace / dimension as f64
    };
    if derivative.iter().flatten().any(|v| !v.is_finite()) || !pressure.is_finite() {
        return Err(DftError::InvalidInput(
            "dimensional strain exceeds finite numerical range".into(),
        ));
    }
    Ok((stress, pressure))
}

impl DimensionalStress {
    pub(crate) fn include_external_field(
        &mut self,
        geometry: &PeriodicGeometry,
        strain: StressTensor,
        energy_defect: f64,
    ) -> Result<()> {
        self.add_affine_derivative(geometry, strain)?;
        self.external_field = Some(strain);
        self.external_field_energy_defect = Some(energy_defect);
        Ok(())
    }

    pub(crate) fn add_affine_derivative(
        &mut self,
        geometry: &PeriodicGeometry,
        strain: StressTensor,
    ) -> Result<()> {
        let mut derivative = self.strain_derivative;
        for (total, extra) in derivative.iter_mut().flatten().zip(strain.iter().flatten()) {
            *total += extra;
        }
        (self.stress, self.pressure) = normalized_stress(geometry, derivative)?;
        self.strain_derivative = derivative;
        Ok(())
    }
}

fn xc_strain(
    frame: &FieldFreeFrame<'_>,
    density: &[[BlochMatrix; 4]],
    expected: f64,
    order: usize,
    tolerance: f64,
) -> Result<([[f64; 3]; 3], f64)> {
    let Some(grid) = frame.grid else {
        return Ok(([[0.0; 3]; 3], 0.0));
    };
    let (ng, n, nk) = (grid.points.len(), frame.basis.nao_sph, grid.k_points.len());
    let bytes =
        crate::resources::checked_bytes(&[ng, n, nk, 16, 48], "spinor XC strain geometry buffers")?;
    let _memory = crate::resources::reserve(bytes, "spinor XC strain geometry buffers")?;
    let cutoff = frame
        .options
        .scf
        .xc_ao_cutoff
        .unwrap_or(frame.options.basis_cutoff);
    let aos = crate::periodic::xc::build_dimensional_bloch_aos(
        frame.basis,
        frame.geometry,
        &grid.k_points,
        &grid.points,
        cutoff,
        crate::periodic::xc::AoDerivativeOrder::Hessian,
    )?;
    let moments = crate::periodic::xc::dimensional_bloch_ao_moments(
        frame.basis,
        frame.geometry,
        &grid.k_points,
        &grid.points,
        cutoff,
        true,
    )?;
    let multiply_minus_i = |g: &BlochMatrix| {
        let mut p = g.clone();
        for ((r, i), (gr, gi)) in
            p.re.as_mut_slice()
                .iter_mut()
                .zip(p.im.as_mut_slice())
                .zip(g.re.as_slice().iter().zip(g.im.as_slice()))
        {
            *r = *gi;
            *i = -*gr;
        }
        p
    };
    let momentum: Vec<[BlochMatrix; 3]> = aos
        .iter()
        .map(|a| {
            std::array::from_fn(|d| {
                multiply_minus_i(&a.grad.as_ref().expect("strain gradients")[d])
            })
        })
        .collect();
    let orbitals: Vec<_> = aos.iter().map(|a| &a.phi).collect();
    let momenta: Vec<_> = momentum.iter().collect();
    let context = SpinorXcGridContext::new(
        frame.functional,
        &grid.weights,
        &orbitals,
        &momenta,
        density,
        order,
    )?;
    let defect = (context.energy() - expected).abs();
    if defect > tolerance {
        return Err(DftError::InvalidInput(
            "spinor strain grid does not reproduce the actual same-frame XC energy".into(),
        ));
    }
    let mut virials = Vec::with_capacity(ng);
    let _frame_memory = grid
        .angular_frame
        .as_ref()
        .map(|_| {
            crate::resources::checked_bytes(
                &[ng, 9, std::mem::size_of::<Vec3>()],
                "angular grid strains",
            )
            .and_then(|bytes| crate::resources::reserve(bytes, "angular grid strains"))
        })
        .transpose()?;
    let grid_motion = grid.angular_frame.as_ref().map(|angular_frame| {
        grid.points
            .iter()
            .zip(&grid.atom_of)
            .map(|(&point, &owner)| {
                angular_frame.offset_strain(point - frame.molecule.atoms[owner].position)
            })
            .collect::<Vec<_>>()
    });
    let stationary_offsets = [[Vec3::zero(); 3]; 3];
    for g in 0..ng {
        virials.push(crate::periodic::grid::dimensional_weight_virial(
            frame.molecule,
            frame.geometry,
            grid.points[g],
            grid.atom_of[g],
            frame.options.grid_cutoff,
            grid_motion
                .as_ref()
                .map_or(&stationary_offsets, |motion| &motion[g]),
        )?);
    }
    let mut result = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let mut phi: Vec<_> = (0..nk).map(|_| BlochMatrix::zeros(ng, n)).collect();
            let mut dp: Vec<[BlochMatrix; 3]> = (0..nk)
                .map(|_| std::array::from_fn(|_| BlochMatrix::zeros(ng, n)))
                .collect();
            for k in 0..nk {
                for g in 0..ng {
                    // Remove the stretching of the radial/angular offset in
                    // the AO moments. In bulk, add the rotation of that offset
                    // with the cell's quadrature frame.
                    let offset = grid.points[g] - frame.molecule.atoms[grid.atom_of[g]].position;
                    let gradients = aos[k].grad.as_ref().expect("strain gradient");
                    let hessians = aos[k].hess.as_ref().expect("strain Hessian");
                    let motion = grid_motion.as_ref().map(|motion| motion[g][a][b]);
                    for mu in 0..n {
                        phi[k].re[(g, mu)] = moments[k].grad_u[a][b].re[(g, mu)]
                            - offset.get(b) * gradients[a].re[(g, mu)];
                        phi[k].im[(g, mu)] = moments[k].grad_u[a][b].im[(g, mu)]
                            - offset.get(b) * gradients[a].im[(g, mu)];
                        for d in 0..3 {
                            let component = crate::grid::eval::second_index(a, d);
                            let h = &hessians[component];
                            let moment =
                                &moments[k].hess_u.as_ref().expect("strain Hessian moments")
                                    [component][b];
                            dp[k][d].re[(g, mu)] =
                                moment.im[(g, mu)] - offset.get(b) * h.im[(g, mu)];
                            dp[k][d].im[(g, mu)] =
                                -moment.re[(g, mu)] + offset.get(b) * h.re[(g, mu)];
                        }
                        if let Some(motion) = motion {
                            for (t, g_t) in gradients.iter().enumerate() {
                                let value = motion.get(t);
                                phi[k].re[(g, mu)] += value * g_t.re[(g, mu)];
                                phi[k].im[(g, mu)] += value * g_t.im[(g, mu)];
                                for (d, dp_d) in dp[k].iter_mut().enumerate() {
                                    let h = &hessians[crate::grid::eval::second_index(t, d)];
                                    dp_d.re[(g, mu)] += value * h.im[(g, mu)];
                                    dp_d.im[(g, mu)] -= value * h.re[(g, mu)];
                                }
                            }
                        }
                    }
                }
            }
            let weights: Vec<_> = (0..ng)
                .map(|g| grid.raw_weights[g] * virials[g][a][b])
                .collect();
            let p: Vec<_> = phi.iter().collect();
            let m: Vec<_> = dp.iter().collect();
            result[a][b] = context.differential(SpinorXcGridTangent {
                weights: &weights,
                orbitals: &p,
                momentum: &m,
            })?;
        }
    }
    Ok((result, defect))
}
