// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic AO, point-motion and periodic-Becke-weight response of `Exc`.

use crate::basis::MolBasis;
use crate::error::Result;
use crate::grid::eval::second_index as hessian_index;
use crate::math::Vec3;
use crate::periodic::scf::PeriodicScfOptions;
#[cfg(test)]
use crate::periodic::scf::PeriodicScfResult;
use crate::periodic::stationary_density::StationaryDensity;
use crate::periodic::xc::{AoDerivativeOrder, PeriodicXcGrid};
use crate::periodic::BlochMatrix;
use crate::periodic::{Cell, PeriodicGeometry};
use crate::system::Molecule;
use crate::xc::{Family, GridVars, ResolvedFunctional, XcOut};

#[inline]
fn mul(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    (ar * br - ai * bi, ar * bi + ai * br)
}

#[inline]
fn real_mul(ar: f64, ai: f64, br: f64, bi: f64) -> f64 {
    ar * br - ai * bi
}

/// `x_mu = sum_nu P_mu,nu Phi_nu*` and its gradient analogues at one point.
fn density_rows(
    density: &crate::periodic::BlochMatrix,
    aos: &super::xc::BlochAos,
    point: usize,
) -> (Vec<(f64, f64)>, [Vec<(f64, f64)>; 3]) {
    let nao = density.rows();
    let mut x = vec![(0.0, 0.0); nao];
    let mut y: [Vec<(f64, f64)>; 3] = std::array::from_fn(|_| vec![(0.0, 0.0); nao]);
    for mu in 0..nao {
        for nu in 0..nao {
            let pr = density.re[(mu, nu)];
            let pi = density.im[(mu, nu)];
            let (xr, xi) = mul(pr, pi, aos.phi.re[(point, nu)], -aos.phi.im[(point, nu)]);
            x[mu].0 += xr;
            x[mu].1 += xi;
            if let Some(grad) = &aos.grad {
                for axis in 0..3 {
                    let (yr, yi) = mul(
                        pr,
                        pi,
                        grad[axis].re[(point, nu)],
                        -grad[axis].im[(point, nu)],
                    );
                    y[axis][mu].0 += yr;
                    y[axis][mu].1 += yi;
                }
            }
        }
    }
    (x, y)
}

pub(crate) fn xc_gradient(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &StationaryDensity<'_>,
    options: &PeriodicScfOptions,
) -> Result<Vec<Vec3>> {
    Ok(xc_gradient_and_virial(cell, functional, result, options, false)?.0)
}

/// Analytic strain derivative of the periodic `Exc` quadrature at the SCF
/// density: rigidly moving Bloch AOs, points riding their owner atoms and
/// the periodic Becke weights, all strained with the lattice.
#[cfg(test)]
pub(crate) fn xc_strain_derivative(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
) -> Result<[[f64; 3]; 3]> {
    Ok(xc_gradient_and_virial(
        cell,
        functional,
        &StationaryDensity::from_scf(result),
        options,
        true,
    )?
    .1
    .expect("virial was requested"))
}

pub(crate) fn xc_strain_derivative_from_density(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &StationaryDensity<'_>,
    options: &PeriodicScfOptions,
) -> Result<[[f64; 3]; 3]> {
    Ok(
        xc_gradient_and_virial(cell, functional, result, options, true)?
            .1
            .expect("virial was requested"),
    )
}

fn xc_gradient_and_virial(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &StationaryDensity<'_>,
    options: &PeriodicScfOptions,
    want_virial: bool,
) -> Result<(Vec<Vec3>, Option<[[f64; 3]; 3]>)> {
    let grid = PeriodicXcGrid::build(
        cell,
        result.k_points,
        &options.grid,
        options.grid_shell,
        options.xc_ao_cutoff.unwrap_or(cell.overlap_cutoff),
        // Density/XC evaluation needs values and gradients globally; force
        // Hessians are evaluated in bounded point batches below.
        AoDerivativeOrder::Gradient,
    )?;
    let grid = match &options.skala {
        Some(model) => grid.with_learned(model.clone(), cell),
        None => grid,
    };
    let grid = match options.vv10 {
        Some(settings) => grid.with_vv10(cell, settings)?,
        None => grid,
    };
    let (gradient, virial, _) = collinear_xc_geometry(
        XcGeometry::Bulk(cell),
        &grid,
        functional,
        result.k_points,
        result.density,
        options.xc_ao_cutoff.unwrap_or(cell.overlap_cutoff),
        want_virial,
    )?;
    Ok((gradient, virial))
}

pub(crate) enum XcGeometry<'a> {
    Bulk(&'a Cell),
    Dimensional {
        molecule: &'a Molecule,
        basis: &'a MolBasis,
        geometry: &'a PeriodicGeometry,
        grid_cutoff: f64,
    },
}

#[allow(clippy::type_complexity)]
pub(crate) fn collinear_xc_geometry(
    geometry: XcGeometry<'_>,
    grid: &PeriodicXcGrid,
    functional: &ResolvedFunctional,
    k_points: &[Vec3],
    density: &[Vec<BlochMatrix>; 2],
    ao_cutoff: f64,
    want_virial: bool,
) -> Result<(Vec<Vec3>, Option<[[f64; 3]; 3]>, f64)> {
    let (molecule, basis) = match geometry {
        XcGeometry::Bulk(cell) => (&cell.molecule, &cell.basis),
        XcGeometry::Dimensional {
            molecule, basis, ..
        } => (molecule, basis),
    };
    let natoms = molecule.len();
    let nao = basis.nao_sph;
    let nk = k_points.len();
    let inv_nk = 1.0 / nk as f64;
    let npts = grid.len();
    let semilocal_gga = functional.family != Family::Lda;
    let gga = semilocal_gga || grid.vv10.is_some();
    let meta = functional.family == Family::MetaGga;
    let ao_atom = basis.sph_ao_atoms();

    let mut rho = [vec![0.0; npts], vec![0.0; npts]];
    let mut drho: [[Vec<f64>; 3]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| vec![0.0; npts]));
    let mut tau = [vec![0.0; npts], vec![0.0; npts]];
    for spin in 0..2 {
        for (ik, aos) in grid.aos.iter().enumerate() {
            let grad = aos.grad.as_ref().expect("derivative grid has gradients");
            for point in 0..npts {
                let (x, y) = density_rows(&density[spin][ik], aos, point);
                for mu in 0..nao {
                    rho[spin][point] += inv_nk
                        * real_mul(
                            aos.phi.re[(point, mu)],
                            aos.phi.im[(point, mu)],
                            x[mu].0,
                            x[mu].1,
                        );
                    for axis in 0..3 {
                        drho[spin][axis][point] += 2.0
                            * inv_nk
                            * real_mul(
                                grad[axis].re[(point, mu)],
                                grad[axis].im[(point, mu)],
                                x[mu].0,
                                x[mu].1,
                            );
                        if meta {
                            tau[spin][point] += 0.5
                                * inv_nk
                                * real_mul(
                                    grad[axis].re[(point, mu)],
                                    grad[axis].im[(point, mu)],
                                    y[axis][mu].0,
                                    y[axis][mu].1,
                                );
                        }
                    }
                }
            }
        }
        for value in &mut rho[spin] {
            if *value < 0.0 && *value > -1e-12 {
                *value = 0.0;
            }
        }
    }

    let sigma_store;
    let vars = if semilocal_gga {
        let mut sigma = [vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]];
        for point in 0..npts {
            for axis in 0..3 {
                sigma[0][point] += drho[0][axis][point] * drho[0][axis][point];
                sigma[1][point] += drho[0][axis][point] * drho[1][axis][point];
                sigma[2][point] += drho[1][axis][point] * drho[1][axis][point];
            }
        }
        sigma_store = sigma;
        GridVars {
            rho: [&rho[0], &rho[1]],
            sigma: Some([&sigma_store[0], &sigma_store[1], &sigma_store[2]]),
            tau: meta.then_some([&tau[0], &tau[1]]),
        }
    } else {
        GridVars {
            rho: [&rho[0], &rho[1]],
            sigma: None,
            tau: None,
        }
    };
    let semilocal = !functional.parts.is_empty();
    let mut out = XcOut::default();
    if semilocal {
        functional.eval(&vars, 1, &mut out)?;
    }
    // Energy density and derivatives of the energy with respect to the
    // densities at the grid points; the derivatives include the quadrature
    // weights.
    let mut energy_density = vec![0.0; npts];
    let mut d_rho = [vec![0.0; npts], vec![0.0; npts]];
    let mut d_gradient = [vec![[0.0_f64; 3]; npts], vec![[0.0_f64; 3]; npts]];
    let mut d_tau = [vec![0.0; npts], vec![0.0; npts]];
    if semilocal {
        for point in 0..npts {
            let weight = grid.weights[point];
            energy_density[point] = out.e[point];
            for spin in 0..2 {
                let other = 1 - spin;
                d_rho[spin][point] = weight * out.vrho[spin][point];
                if semilocal_gga {
                    let (own, cross) = if spin == 0 {
                        (out.vsigma[0][point], out.vsigma[1][point])
                    } else {
                        (out.vsigma[2][point], out.vsigma[1][point])
                    };
                    for axis in 0..3 {
                        d_gradient[spin][point][axis] = weight
                            * (2.0 * own * drho[spin][axis][point]
                                + cross * drho[other][axis][point]);
                    }
                }
                if meta {
                    d_tau[spin][point] = weight * out.vtau[spin][point];
                }
            }
        }
    }
    if let Some(learned) = &grid.learned {
        let result = learned.evaluate(grid, &rho, &drho, &tau)?;
        for point in 0..npts {
            energy_density[point] += result.energy_density[point];
            for spin in 0..2 {
                d_rho[spin][point] += result.d_rho[spin][point];
                d_tau[spin][point] += result.d_tau[spin][point];
                for axis in 0..3 {
                    d_gradient[spin][point][axis] += result.d_gradient[spin][point][axis];
                }
            }
        }
    }
    let (nonlocal_energy, nonlocal_geometry) = if let Some((geometry, settings)) = &grid.vv10 {
        let total: Vec<_> = (0..npts).map(|i| rho[0][i] + rho[1][i]).collect();
        let gradients: Vec<[f64; 3]> = (0..npts)
            .map(|i| std::array::from_fn(|a| drho[0][a][i] + drho[1][a][i]))
            .collect();
        let sigma: Vec<_> = gradients
            .iter()
            .map(|g| g.iter().map(|v| v * v).sum())
            .collect();
        let (fields, geometry) = super::nonlocal::vv10_geometry_fields(
            geometry,
            &grid.points,
            &grid.weights,
            &total,
            &sigma,
            *settings,
        )?;
        for spin in 0..2 {
            for (point, gradient) in gradients.iter().enumerate() {
                d_rho[spin][point] += fields.d_rho[point];
                for (axis, value) in gradient.iter().enumerate() {
                    d_gradient[spin][point][axis] += 2.0 * fields.d_sigma[point] * value;
                }
            }
        }
        (fields.energy, Some(geometry))
    } else {
        (0.0, None)
    };
    let mut gradient = vec![Vec3::zero(); natoms];
    let mut moments = None;
    let mut moment_start = 0;
    let mut force_aos = None;
    let mut force_start = 0;
    let mut virial = nonlocal_geometry
        .as_ref()
        .map_or([[0.0; 3]; 3], |v| v.image_virial);

    for point in 0..npts {
        if gga && point % 64 == 0 {
            force_start = point;
            let end = (point + 64).min(npts);
            let points = &grid.points[point..end];
            force_aos = Some(match geometry {
                XcGeometry::Bulk(cell) => super::xc::build_bloch_aos_on_points(
                    cell,
                    k_points,
                    points,
                    ao_cutoff,
                    AoDerivativeOrder::Hessian,
                ),
                XcGeometry::Dimensional { geometry, .. } => super::xc::build_dimensional_bloch_aos(
                    basis,
                    geometry,
                    k_points,
                    points,
                    ao_cutoff,
                    AoDerivativeOrder::Hessian,
                )?,
            });
        }
        let ao_point = if gga { point - force_start } else { point };
        if want_virial && point % 64 == 0 {
            moment_start = point;
            let end = (point + 64).min(npts);
            let points = &grid.points[point..end];
            moments = Some(match geometry {
                XcGeometry::Bulk(cell) => {
                    super::xc::bloch_ao_moments(cell, k_points, points, ao_cutoff, gga)
                }
                XcGeometry::Dimensional { geometry, .. } => {
                    super::xc::dimensional_bloch_ao_moments(
                        basis, geometry, k_points, points, ao_cutoff, gga,
                    )?
                }
            });
        }
        let mut rho_hessian = [[[0.0; 3]; 3]; 2];
        let mut tau_gradient = [[0.0; 3]; 2];
        for spin in 0..2 {
            let b: [f64; 3] = d_gradient[spin][point];
            let b_field = |axis: usize| b[axis];
            for (ik, aos) in force_aos.as_ref().unwrap_or(&grid.aos).iter().enumerate() {
                let grad = aos.grad.as_ref().expect("gradients");
                let (x, y) = density_rows(&density[spin][ik], aos, ao_point);
                for mu in 0..nao {
                    let atom = ao_atom[mu];
                    for axis in 0..3 {
                        let mut orbital = d_rho[spin][point]
                            * real_mul(
                                grad[axis].re[(ao_point, mu)],
                                grad[axis].im[(ao_point, mu)],
                                x[mu].0,
                                x[mu].1,
                            );
                        if gga {
                            let hess = aos
                                .hess
                                .as_ref()
                                .expect("GGA/meta-GGA force grid has AO Hessians");
                            for spatial in 0..3 {
                                let hi = hessian_index(axis, spatial);
                                orbital += b[spatial]
                                    * (real_mul(
                                        hess[hi].re[(ao_point, mu)],
                                        hess[hi].im[(ao_point, mu)],
                                        x[mu].0,
                                        x[mu].1,
                                    ) + real_mul(
                                        grad[axis].re[(ao_point, mu)],
                                        grad[axis].im[(ao_point, mu)],
                                        y[spatial][mu].0,
                                        y[spatial][mu].1,
                                    ));
                            }
                        }
                        let mut value = -2.0 * inv_nk * orbital;
                        if meta {
                            let hess = aos
                                .hess
                                .as_ref()
                                .expect("meta-GGA force grid has AO Hessians");
                            let mut kinetic = 0.0;
                            for spatial in 0..3 {
                                let hi = hessian_index(axis, spatial);
                                kinetic += real_mul(
                                    hess[hi].re[(ao_point, mu)],
                                    hess[hi].im[(ao_point, mu)],
                                    y[spatial][mu].0,
                                    y[spatial][mu].1,
                                );
                            }
                            value -= inv_nk * d_tau[spin][point] * kinetic;
                        }
                        match axis {
                            0 => gradient[atom].x += value,
                            1 => gradient[atom].y += value,
                            _ => gradient[atom].z += value,
                        }
                    }

                    // Strain of the rigid AO images: the gradient formula
                    // with d_a phi -> d_a phi u_b, referred to the point.
                    if let Some(moments) = moments.as_ref() {
                        let moment = &moments[ik];
                        for axis in 0..3 {
                            for column in 0..3 {
                                let gu = &moment.grad_u[axis][column];
                                let mut orbital = d_rho[spin][point]
                                    * real_mul(
                                        gu.re[(point - moment_start, mu)],
                                        gu.im[(point - moment_start, mu)],
                                        x[mu].0,
                                        x[mu].1,
                                    );
                                let mut kinetic = 0.0;
                                if gga {
                                    let hess_u =
                                        moment.hess_u.as_ref().expect("GGA moments carry Hessians");
                                    for spatial in 0..3 {
                                        let hu = &hess_u[hessian_index(axis, spatial)][column];
                                        orbital += b_field(spatial)
                                            * (real_mul(
                                                hu.re[(point - moment_start, mu)],
                                                hu.im[(point - moment_start, mu)],
                                                x[mu].0,
                                                x[mu].1,
                                            ) + real_mul(
                                                gu.re[(point - moment_start, mu)],
                                                gu.im[(point - moment_start, mu)],
                                                y[spatial][mu].0,
                                                y[spatial][mu].1,
                                            ));
                                        if meta {
                                            kinetic += real_mul(
                                                hu.re[(point - moment_start, mu)],
                                                hu.im[(point - moment_start, mu)],
                                                y[spatial][mu].0,
                                                y[spatial][mu].1,
                                            );
                                        }
                                    }
                                }
                                let mut value = -2.0 * inv_nk * orbital;
                                if meta {
                                    value -= inv_nk * d_tau[spin][point] * kinetic;
                                }
                                virial[axis][column] -= value;
                            }
                        }
                    }

                    // Spatial second derivatives for point motion.
                    if gga {
                        let hess = aos
                            .hess
                            .as_ref()
                            .expect("GGA/meta-GGA force grid has AO Hessians");
                        for u in 0..3 {
                            for v in 0..3 {
                                let hi = hessian_index(u, v);
                                rho_hessian[spin][u][v] += 2.0
                                    * inv_nk
                                    * (real_mul(
                                        hess[hi].re[(ao_point, mu)],
                                        hess[hi].im[(ao_point, mu)],
                                        x[mu].0,
                                        x[mu].1,
                                    ) + real_mul(
                                        grad[u].re[(ao_point, mu)],
                                        grad[u].im[(ao_point, mu)],
                                        y[v][mu].0,
                                        y[v][mu].1,
                                    ));
                                if meta {
                                    tau_gradient[spin][u] += inv_nk
                                        * real_mul(
                                            hess[hi].re[(ao_point, mu)],
                                            hess[hi].im[(ao_point, mu)],
                                            y[v][mu].0,
                                            y[v][mu].1,
                                        );
                                }
                            }
                        }
                    }
                }
            }
        }

        // Periodic Becke fixed-point response, folded from images to their
        // parent atoms. The grid point itself rides its owner atom.
        let owner = grid.atom_of[point];
        let position = grid.points[point];
        let (_, weight_gradient) = match geometry {
            XcGeometry::Bulk(cell) => crate::periodic::grid::periodic_weight_gradient(
                molecule,
                &cell.lattice,
                position,
                owner,
                grid.becke_shell,
            ),
            XcGeometry::Dimensional {
                geometry,
                grid_cutoff,
                ..
            } => crate::periodic::grid::dimensional_weight_gradient(
                molecule,
                geometry,
                position,
                owner,
                grid_cutoff,
            )?,
        };
        let weight_energy = grid.raw_weights[point]
            * (energy_density[point]
                + nonlocal_geometry
                    .as_ref()
                    .map_or(0.0, |v| v.d_weight[point]));
        let mut sum_weight_gradient = Vec3::zero();
        for atom in 0..natoms {
            gradient[atom] += weight_gradient[atom] * weight_energy;
            sum_weight_gradient += weight_gradient[atom];
        }
        gradient[owner] -= sum_weight_gradient * weight_energy;
        let offset = position - molecule.atoms[owner].position;
        let offset_strain = grid.angular_frame.as_ref().map_or_else(
            || [[Vec3::zero(); 3]; 3],
            |frame| frame.offset_strain(offset),
        );
        if want_virial {
            let weight_virial = match geometry {
                XcGeometry::Bulk(cell) => crate::periodic::grid::periodic_weight_virial(
                    molecule,
                    &cell.lattice,
                    position,
                    owner,
                    grid.becke_shell,
                ),
                XcGeometry::Dimensional {
                    geometry,
                    grid_cutoff,
                    ..
                } => crate::periodic::grid::dimensional_weight_virial(
                    molecule,
                    geometry,
                    position,
                    owner,
                    grid_cutoff,
                    &offset_strain,
                )?,
            };
            for a in 0..3 {
                for c in 0..3 {
                    virial[a][c] += weight_energy * weight_virial[a][c];
                }
            }
        }

        // Derivative with respect to the position of the point at fixed
        // orbitals, quadrature weight included.
        let mut spatial = [0.0_f64; 3];
        for spin in 0..2 {
            for u in 0..3 {
                let mut value = d_rho[spin][point] * drho[spin][u][point];
                if gga {
                    for v in 0..3 {
                        value += d_gradient[spin][point][v] * rho_hessian[spin][u][v];
                    }
                }
                if meta {
                    value += d_tau[spin][point] * tau_gradient[spin][u];
                }
                spatial[u] += value;
            }
        }
        let spatial_energy_gradient = Vec3::new(spatial[0], spatial[1], spatial[2]);
        gradient[owner] += spatial_energy_gradient;
        if let Some(nonlocal) = &nonlocal_geometry {
            let [x, y, z] = nonlocal.d_position[point];
            gradient[owner] += Vec3::new(x, y, z);
            if want_virial {
                let center = molecule.atoms[owner].position;
                let point_gradient = Vec3::new(x, y, z);
                // The pair term sees both the owner motion and the rotation
                // of the quadrature offset. The lattice-image term is separate.
                for (a, row) in virial.iter_mut().enumerate() {
                    for (b, value) in row.iter_mut().enumerate() {
                        *value += point_gradient.get(a) * center.get(b)
                            + point_gradient.dot(offset_strain[a][b]);
                    }
                }
            }
        }
        if want_virial {
            // The AO moments already include fully affine point motion.
            // Remove the stretching of the offset and add its rotation.
            for a in 0..3 {
                for c in 0..3 {
                    virial[a][c] += spatial_energy_gradient.dot(offset_strain[a][c])
                        - spatial_energy_gradient.get(a) * offset.get(c);
                }
            }
        }
    }
    let energy = energy_density
        .iter()
        .zip(&grid.weights)
        .map(|(e, w)| e * w)
        .sum::<f64>()
        + nonlocal_energy;
    Ok((gradient, want_virial.then_some(virial), energy))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;
    use crate::grid::GridOptions;
    use crate::linalg::Matrix;
    use crate::periodic::bloch::BlochMatrix;
    use crate::periodic::scf::PeriodicScfResult;
    use crate::periodic::xc::{xc_contribution, AoDerivativeOrder, PeriodicXcGrid};
    use crate::periodic::Lattice;
    use crate::system::{Atom, Molecule};
    use crate::xc::{resolve, FunctionalSpec};

    fn displaced(molecule: &Molecule, atom: usize, axis: usize, amount: f64) -> Molecule {
        let mut result = molecule.clone();
        match axis {
            0 => result.atoms[atom].position.x += amount,
            1 => result.atoms[atom].position.y += amount,
            _ => result.atoms[atom].position.z += amount,
        }
        result
    }

    fn fixed_density_result(cell: &Cell, k_points: Vec<Vec3>) -> PeriodicScfResult {
        let n = cell.nao();
        let re = Matrix::from_row_major(n, n, vec![0.43, 0.07, 0.07, 0.36]);
        let density: [Vec<BlochMatrix>; 2] = std::array::from_fn(|_| {
            k_points
                .iter()
                .map(|_| BlochMatrix {
                    re: re.clone(),
                    im: Matrix::zeros(n, n),
                })
                .collect()
        });
        let zero_matrices = || {
            k_points
                .iter()
                .map(|_| BlochMatrix::zeros(n, n))
                .collect::<Vec<_>>()
        };
        let result_k_points = k_points.clone();
        PeriodicScfResult {
            orbital_conjugation_constraint: true,
            energy: 0.0,
            free_energy: 0.0,
            entropy_term: 0.0,
            e_nuc: 0.0,
            e_elec: 0.0,
            e_xc: 0.0,
            e_nlc: 0.0,
            k_points: result_k_points,
            mo_energy: std::array::from_fn(|_| Vec::new()),
            occupations: std::array::from_fn(|_| Vec::new()),
            chemical_potential: [None, None],
            density,
            energy_weighted_density: std::array::from_fn(|_| zero_matrices()),
            fock: std::array::from_fn(|_| zero_matrices()),
            overlap: zero_matrices(),
            h_core: zero_matrices(),
            n_alpha: 1,
            n_beta: 1,
            unrestricted: false,
            grid_electron_count: None,
            converged: true,
            iterations: 0,
            orbital_residual: None,
            derivative_plan: None,
        }
    }

    #[test]
    fn periodic_lda_xc_gradient_does_not_require_ao_hessians() {
        let molecule = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, -0.6),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 0.8),
            },
        ]);
        let lattice = Lattice::new(
            Vec3::new(8.0, 0.0, 0.0),
            Vec3::new(0.0, 12.0, 0.0),
            Vec3::new(0.0, 0.0, 12.0),
        );
        let set = data::load("sto-3g").unwrap();
        let cell = Cell::new(molecule, lattice, &set, 4.0).unwrap();
        let options = PeriodicScfOptions {
            k_mesh: [1, 1, 1],
            xc_ao_cutoff: Some(4.0),
            grid_shell: 1,
            grid: GridOptions {
                radial_override: Some(3),
                angular_override: Some(3),
                prune: false,
                weight_cutoff: 0.0,
                ..GridOptions::default()
            },
            ..PeriodicScfOptions::default()
        };
        let functional = resolve(&FunctionalSpec::Combined("svwn5".to_string())).unwrap();
        assert_eq!(functional.family, Family::Lda);
        assert_eq!(
            AoDerivativeOrder::for_forces(functional.family),
            AoDerivativeOrder::Gradient
        );
        let result = fixed_density_result(&cell, cell.k_points(options.k_mesh));
        let gradient = xc_gradient(
            &cell,
            &functional,
            &StationaryDensity::from_scf(&result),
            &options,
        )
        .unwrap();
        assert!(gradient
            .iter()
            .all(|value| value.x.is_finite() && value.y.is_finite() && value.z.is_finite()));
    }

    #[test]
    fn periodic_r2scan_xc_gradient_matches_fixed_density_energy_difference() {
        // A fixed-density derivative isolates the complete semilocal
        // skeleton: translated Bloch AO centres, moving atom-centred points,
        // periodic Becke weights, rho/sigma, and the meta-GGA tau channel.
        // No SCF response or finite-difference force is used by production.
        let molecule = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, -0.55),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 0.95),
            },
        ]);
        let lattice = Lattice::new(
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(0.0, 12.0, 0.0),
            Vec3::new(0.0, 0.0, 12.0),
        );
        let set = data::load("sto-3g").unwrap();
        let functional = resolve(&FunctionalSpec::Combined("r2scan".to_string())).unwrap();
        let options = PeriodicScfOptions {
            k_mesh: [3, 1, 1],
            xc_ao_cutoff: Some(6.1),
            grid_shell: 1,
            grid: GridOptions {
                radial_override: Some(9),
                angular_override: Some(7),
                prune: false,
                weight_cutoff: 0.0,
                ..GridOptions::default()
            },
            ..PeriodicScfOptions::default()
        };
        let cell = Cell::new(molecule.clone(), lattice.clone(), &set, 6.1).unwrap();
        let result = fixed_density_result(&cell, cell.k_points(options.k_mesh));
        let analytic = xc_gradient(
            &cell,
            &functional,
            &StationaryDensity::from_scf(&result),
            &options,
        )
        .unwrap();
        let energy = |geometry: Molecule| {
            let displaced_cell = Cell::new(geometry, lattice.clone(), &set, 6.1).unwrap();
            let grid = PeriodicXcGrid::build(
                &displaced_cell,
                &result.k_points,
                &options.grid,
                options.grid_shell,
                options.xc_ao_cutoff.unwrap(),
                AoDerivativeOrder::for_scf(functional.family),
            )
            .unwrap();
            xc_contribution(&functional, &grid, &result.density)
                .unwrap()
                .exc
        };
        let step = 2e-4;
        let finite_difference = (energy(displaced(&molecule, 1, 2, step))
            - energy(displaced(&molecule, 1, 2, -step)))
            / (2.0 * step);
        let error = (analytic[1].z - finite_difference).abs();
        assert!(
            error < 3e-5,
            "r2SCAN periodic dExc/dR: analytic={} vs FD={} (error={error:e})",
            analytic[1].z,
            finite_difference
        );
        let sum = analytic[0] + analytic[1];
        assert!(sum.norm() < 2e-8, "XC translation sum rule: {sum:?}");
    }
}
