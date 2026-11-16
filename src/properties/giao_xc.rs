// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! London-phase derivative of semilocal XC matrix elements.
//!
//! The magnetic first-order density is antisymmetric, so `d rho/dB = 0`
//! and no density/gradient semilocal XC kernel enters the magnetic CPHF. The
//! AO matrix element still changes because both basis functions carry London
//! phases. LDA and GGA therefore need only the explicit contraction. A
//! meta-GGA additionally uses the gauge-covariant Dobson variable
//! `tau_tilde_s = tau_s - |j_p,s|^2/(2 rho_s)`: its conventional-tau London
//! term is explicit, while the current-current Hessian remains in CPHF even
//! at a real, current-free ground state.

use crate::basis::{n_sph, MolBasis};
use crate::error::{DftError, Result};
use crate::grid::eval::eval_aos;
use crate::grid::MolecularGrid;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::xc::{Family, GridVars, ResolvedFunctional, XcOut};

const BATCH: usize = 256;

/// `dVxc_sigma/dB_a = i * out[spin][a]` for a semilocal functional.
///
/// This is kept private to the properties implementation. Low-level tests
/// live beside it; the public contract remains the shielding API.
#[cfg(test)]
pub(super) fn giao_semilocal_xc_fock_derivative(
    functional: &ResolvedFunctional,
    mb: &MolBasis,
    grid: &MolecularGrid,
    density: [&Matrix; 2],
    origin: Vec3,
) -> Result<[[Matrix; 3]; 2]> {
    giao_density_xc_fock_derivative(functional, mb, grid, density, origin, None)
}

/// Weighted partials of any differentiable grid-density functional, including
/// nonlocal learned models. No pointwise XC-kernel assumption is made.
pub(crate) struct WeightedDensityPartials {
    density: [Vec<f64>; 2],
    rho: [Vec<f64>; 2],
    gradient: [Vec<[f64; 3]>; 2],
    tau: [Vec<f64>; 2],
}

impl WeightedDensityPartials {
    pub(crate) fn learned(
        model: &crate::skala::SkalaModel,
        molecule: &crate::system::Molecule,
        basis: &MolBasis,
        grid: &MolecularGrid,
        density: [&Matrix; 2],
    ) -> Result<Self> {
        let ranges = crate::skala::batch_ranges(grid.len());
        let (rho, gradient, tau) = crate::skala::grid_densities(basis, grid, density, &ranges);
        let (points, centers) = crate::skala::grid_geometry(molecule, grid);
        let out = model.evaluate(
            &crate::skala::SkalaInput {
                atom_of: &grid.atom_of,
                points: &points,
                centers: &centers,
                weights: &grid.weights,
                atomic_weights: &grid.raw_weights,
                rho: [&rho[0], &rho[1]],
                gradient: [&gradient[0], &gradient[1]],
                tau: [&tau[0], &tau[1]],
            },
            true,
        )?;
        Ok(Self {
            density: rho,
            rho: out.d_rho,
            gradient: out.d_gradient,
            tau: out.d_tau,
        })
    }
}

/// London basis derivative from density, gradient and kinetic-density
/// partials. Supplied partials already contain the integration weights.
pub(super) fn giao_density_xc_fock_derivative(
    functional: &ResolvedFunctional,
    mb: &MolBasis,
    grid: &MolecularGrid,
    density: [&Matrix; 2],
    origin: Vec3,
    partials: Option<&WeightedDensityPartials>,
) -> Result<[[Matrix; 3]; 2]> {
    if let Some(fields) = partials.filter(|_| !functional.parts.is_empty()) {
        // SCF adds independent functional providers. Preserve that same
        // additive energy/Fock model when both providers are selected.
        let mut result =
            giao_density_xc_fock_derivative(functional, mb, grid, density, origin, None)?;
        let extra = giao_density_xc_fock_derivative(
            &ResolvedFunctional::hartree_fock(),
            mb,
            grid,
            density,
            origin,
            Some(fields),
        )?;
        for spin in 0..2 {
            for axis in 0..3 {
                for (a, b) in result[spin][axis]
                    .as_mut_slice()
                    .iter_mut()
                    .zip(extra[spin][axis].as_slice())
                {
                    *a += *b;
                }
            }
        }
        return Ok(result);
    }
    let gga = partials.is_some() || functional.family != Family::Lda;
    let meta = partials.is_some() || functional.family == Family::MetaGga;
    let nao = mb.nao_sph;
    let zeros = || std::array::from_fn(|_| Matrix::zeros(nao, nao));
    if functional.parts.is_empty() && partials.is_none() {
        return Ok([zeros(), zeros()]);
    }
    if grid.is_empty() {
        return Err(DftError::InvalidInput(
            "GIAO XC derivative requires a non-empty molecular grid".into(),
        ));
    }
    if grid.points.len() != grid.weights.len() {
        return Err(DftError::InvalidInput(
            "GIAO XC grid point/weight lengths do not match".into(),
        ));
    }
    if density.iter().any(|p| p.rows != nao || p.cols != nao) {
        return Err(DftError::InvalidInput(
            "GIAO XC density dimensions do not match the AO basis".into(),
        ));
    }

    let mut centres = vec![Vec3::zero(); nao];
    for sh in &mb.shells {
        for local in 0..n_sph(sh.l) {
            centres[sh.sph_off + local] = sh.center;
        }
    }
    let mut moments: [[Matrix; 3]; 2] = [zeros(), zeros()];
    let mut tau_phase: [[Matrix; 3]; 2] = [zeros(), zeros()];
    for start in (0..grid.len()).step_by(BATCH) {
        let end = (start + BATCH).min(grid.len());
        let points = &grid.points[start..end];
        let supplied_weights = vec![1.0; end - start];
        let weights = if partials.is_some() {
            &supplied_weights[..]
        } else {
            &grid.weights[start..end]
        };
        let batch = eval_aos(mb, points, gga);
        let phi = &batch.phi;
        let npts = points.len();
        let (vrho, b, vtau) = if let Some(fields) = partials {
            let vrho = std::array::from_fn(|s| fields.rho[s][start..end].to_vec());
            let b: [[Vec<f64>; 3]; 2] = std::array::from_fn(|s| {
                std::array::from_fn(|d| {
                    fields.gradient[s][start..end]
                        .iter()
                        .map(|v| v[d])
                        .collect()
                })
            });
            let vtau = std::array::from_fn(|s| fields.tau[s][start..end].to_vec());
            (vrho, b, vtau)
        } else {
            let mut rho = [vec![0.0; npts], vec![0.0; npts]];
            let mut grad: [[Vec<f64>; 3]; 2] = Default::default();
            let mut tau = [Vec::new(), Vec::new()];
            if gga {
                for spin in &mut grad {
                    for axis in spin {
                        *axis = vec![0.0; npts];
                    }
                }
                if meta {
                    tau = [vec![0.0; npts], vec![0.0; npts]];
                }
            }
            for spin in 0..2 {
                let phip = phi.matmul_seq(density[spin]);
                for g in 0..npts {
                    let mut value = 0.0;
                    for mu in 0..nao {
                        value += phip[(g, mu)] * phi[(g, mu)];
                    }
                    rho[spin][g] = value.max(0.0);
                }
                if gga {
                    let ao_grad = batch.grad.as_ref().expect("gradients requested");
                    for axis in 0..3 {
                        let grad_p = if meta {
                            Some(ao_grad[axis].matmul_seq(density[spin]))
                        } else {
                            None
                        };
                        for g in 0..npts {
                            let mut value = 0.0;
                            for mu in 0..nao {
                                value += phip[(g, mu)] * ao_grad[axis][(g, mu)];
                            }
                            grad[spin][axis][g] = 2.0 * value;
                            if let Some(gp) = &grad_p {
                                let mut value = 0.0;
                                for mu in 0..nao {
                                    value += gp[(g, mu)] * ao_grad[axis][(g, mu)];
                                }
                                tau[spin][g] += 0.5 * value;
                            }
                        }
                    }
                }
            }
            let sigma_store;
            let vars = if gga {
                let mut saa = vec![0.0; npts];
                let mut sab = vec![0.0; npts];
                let mut sbb = vec![0.0; npts];
                for g in 0..npts {
                    let ga = [grad[0][0][g], grad[0][1][g], grad[0][2][g]];
                    let gb = [grad[1][0][g], grad[1][1][g], grad[1][2][g]];
                    saa[g] = ga[0] * ga[0] + ga[1] * ga[1] + ga[2] * ga[2];
                    sab[g] = ga[0] * gb[0] + ga[1] * gb[1] + ga[2] * gb[2];
                    sbb[g] = gb[0] * gb[0] + gb[1] * gb[1] + gb[2] * gb[2];
                }
                sigma_store = [saa, sab, sbb];
                GridVars {
                    rho: [&rho[0], &rho[1]],
                    sigma: Some([&sigma_store[0], &sigma_store[1], &sigma_store[2]]),
                    tau: if meta { Some([&tau[0], &tau[1]]) } else { None },
                }
            } else {
                GridVars {
                    rho: [&rho[0], &rho[1]],
                    sigma: None,
                    tau: None,
                }
            };
            let mut xc = XcOut::default();
            functional.eval(&vars, 1, &mut xc)?;
            let mut b: [[Vec<f64>; 3]; 2] = Default::default();
            if gga {
                for spin in 0..2 {
                    let other = 1 - spin;
                    let vs_own = if spin == 0 {
                        &xc.vsigma[0]
                    } else {
                        &xc.vsigma[2]
                    };
                    let vs_cross = &xc.vsigma[1];
                    for axis in 0..3 {
                        b[spin][axis] = (0..npts)
                            .map(|g| {
                                2.0 * vs_own[g] * grad[spin][axis][g]
                                    + vs_cross[g] * grad[other][axis][g]
                            })
                            .collect();
                    }
                }
            }
            (xc.vrho, b, xc.vtau)
        };
        for spin in 0..2 {
            for axis in 0..3 {
                let mut scaled = phi.clone();
                for g in 0..npts {
                    let r_axis = (points[g] - origin).get(axis);
                    let factor = weights[g]
                        * (r_axis * vrho[spin][g] + if gga { b[spin][axis][g] } else { 0.0 });
                    for mu in 0..nao {
                        scaled[(g, mu)] *= factor;
                    }
                }
                let mut part = phi.transpose_matmul_seq(&scaled);
                if gga {
                    let ao_grad = batch.grad.as_ref().expect("gradients requested");
                    let mut a = Matrix::zeros(npts, nao);
                    for g in 0..npts {
                        let r_axis = (points[g] - origin).get(axis);
                        for mu in 0..nao {
                            let b_dot_grad = (0..3)
                                .map(|j| b[spin][j][g] * ao_grad[j][(g, mu)])
                                .sum::<f64>();
                            a[(g, mu)] = weights[g] * r_axis * b_dot_grad;
                        }
                    }
                    let cross = a.transpose_matmul_seq(phi);
                    for m in 0..nao {
                        for n in 0..nao {
                            part[(m, n)] += cross[(m, n)] + cross[(n, m)];
                        }
                    }
                    if meta {
                        // M_j^tau = 1/2 int r_j vtau grad(phi_m).grad(phi_n).
                        // The final centre-difference cross product supplies
                        // the remaining London value-phase factor.
                        for derivative_axis in 0..3 {
                            let mut scaled_grad = ao_grad[derivative_axis].clone();
                            for g in 0..npts {
                                let factor = 0.5
                                    * weights[g]
                                    * (points[g] - origin).get(axis)
                                    * vtau[spin][g];
                                for mu in 0..nao {
                                    scaled_grad[(g, mu)] *= factor;
                                }
                            }
                            let tau_part =
                                ao_grad[derivative_axis].transpose_matmul_seq(&scaled_grad);
                            for (dst, src) in
                                part.as_mut_slice().iter_mut().zip(tau_part.as_slice())
                            {
                                *dst += *src;
                            }
                        }
                    }
                }
                for (dst, src) in moments[spin][axis]
                    .as_mut_slice()
                    .iter_mut()
                    .zip(part.as_slice())
                {
                    *dst += *src;
                }
            }
        }
        if meta {
            let ao_grad = batch.grad.as_ref().expect("gradients requested");
            for spin in 0..2 {
                // C_j[m,n] = int w vtau phi_m partial_j phi_n. The
                // conventional-tau gradient-phase term is
                // 1/2 [t_m.C_mn - t_n.C_nm].
                let mut c: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(nao, nao));
                for j in 0..3 {
                    let mut scaled_grad = ao_grad[j].clone();
                    for g in 0..npts {
                        let factor = weights[g] * vtau[spin][g];
                        for n in 0..nao {
                            scaled_grad[(g, n)] *= factor;
                        }
                    }
                    c[j] = phi.transpose_matmul_seq(&scaled_grad);
                }
                for comp in 0..3 {
                    let e = match comp {
                        0 => Vec3::new(1.0, 0.0, 0.0),
                        1 => Vec3::new(0.0, 1.0, 0.0),
                        _ => Vec3::new(0.0, 0.0, 1.0),
                    };
                    for m in 0..nao {
                        let t_m = e.cross(centres[m] - origin) * 0.5;
                        for n in 0..nao {
                            let t_n = e.cross(centres[n] - origin) * 0.5;
                            let mut a_mn = 0.0;
                            let mut a_nm = 0.0;
                            for j in 0..3 {
                                a_mn += t_m.get(j) * c[j][(m, n)];
                                a_nm += t_n.get(j) * c[j][(n, m)];
                            }
                            tau_phase[spin][comp][(m, n)] += 0.5 * (a_mn - a_nm);
                        }
                    }
                }
            }
        }
    }
    let mut out = [zeros(), zeros()];
    for spin in 0..2 {
        let moment = &moments[spin];
        for m in 0..nao {
            for n in 0..nao {
                let d = centres[m] - centres[n];
                out[spin][0][(m, n)] = 0.5 * (d.y * moment[2][(m, n)] - d.z * moment[1][(m, n)]);
                out[spin][1][(m, n)] = 0.5 * (d.z * moment[0][(m, n)] - d.x * moment[2][(m, n)]);
                out[spin][2][(m, n)] = 0.5 * (d.x * moment[1][(m, n)] - d.y * moment[0][(m, n)]);
                for comp in 0..3 {
                    out[spin][comp][(m, n)] += tau_phase[spin][comp][(m, n)];
                }
            }
        }
    }
    Ok(out)
}

/// Cached `w v_tau/rho` coefficients for the meta-GGA current-current
/// magnetic Hessian. Building this once keeps each CPHF matrix-vector product
/// to bounded AO/gradient batches rather than reevaluating the functional.
pub(crate) struct MetaCurrentKernel {
    coefficient: [Vec<f64>; 2],
}

impl MetaCurrentKernel {
    /// Complete current kernel of the same semilocal/global learned model.
    /// Keeping the assembly here gives ordinary and post-SCF response the
    /// identical tau-|j|^2/(2 rho) zero-field Hessian.
    pub(crate) fn for_reference(
        molecule: &crate::system::Molecule,
        basis: &MolBasis,
        functional: &ResolvedFunctional,
        grid: Option<&MolecularGrid>,
        reference: &crate::scf::ScfResult,
        options: &crate::scf::ScfOptions,
    ) -> Result<Option<Self>> {
        let Some(grid) = grid else {
            if functional.family == Family::MetaGga || options.skala.is_some() {
                return Err(DftError::InvalidInput(
                    "the current response kernel requires its molecular grid".into(),
                ));
            }
            return Ok(None);
        };
        let density = [&reference.density[0], &reference.density[1]];
        let mut kernel = Self::build(functional, basis, grid, density)?;
        if let Some(model) = &options.skala {
            let fields = WeightedDensityPartials::learned(model, molecule, basis, grid, density)?;
            let extra = Self::from_partials(&fields);
            if let Some(kernel) = &mut kernel {
                kernel.add(&extra);
            } else {
                kernel = Some(extra);
            }
        }
        Ok(kernel)
    }
    pub(crate) fn add(&mut self, other: &Self) {
        for spin in 0..2 {
            assert_eq!(self.coefficient[spin].len(), other.coefficient[spin].len());
            for (a, b) in self.coefficient[spin]
                .iter_mut()
                .zip(&other.coefficient[spin])
            {
                *a += *b;
            }
        }
    }

    pub(crate) fn from_partials(fields: &WeightedDensityPartials) -> Self {
        Self {
            coefficient: std::array::from_fn(|s| {
                fields.tau[s]
                    .iter()
                    .zip(&fields.density[s])
                    .map(|(&v, &rho)| if rho > 0.0 { v / rho } else { 0.0 })
                    .collect()
            }),
        }
    }

    pub(crate) fn build(
        functional: &ResolvedFunctional,
        mb: &MolBasis,
        grid: &MolecularGrid,
        ground_density: [&Matrix; 2],
    ) -> Result<Option<Self>> {
        if functional.family != Family::MetaGga {
            return Ok(None);
        }
        let nao = mb.nao_sph;
        if ground_density
            .iter()
            .any(|p| p.rows != nao || p.cols != nao)
            || grid.points.len() != grid.weights.len()
            || grid.is_empty()
        {
            return Err(DftError::InvalidInput(
                "meta-GGA current kernel requires matching densities and a consistent grid".into(),
            ));
        }
        let mut coefficient = [vec![0.0; grid.len()], vec![0.0; grid.len()]];
        for start in (0..grid.len()).step_by(BATCH) {
            let end = (start + BATCH).min(grid.len());
            let batch = eval_aos(mb, &grid.points[start..end], true);
            let phi = &batch.phi;
            let ao_grad = batch.grad.as_ref().expect("gradients requested");
            let npts = end - start;
            let mut rho = [vec![0.0; npts], vec![0.0; npts]];
            let mut grad: [[Vec<f64>; 3]; 2] = Default::default();
            let mut tau = [vec![0.0; npts], vec![0.0; npts]];
            for spin in 0..2 {
                for axis in &mut grad[spin] {
                    *axis = vec![0.0; npts];
                }
                let phip = phi.matmul_seq(ground_density[spin]);
                for g in 0..npts {
                    rho[spin][g] = (0..nao)
                        .map(|mu| phip[(g, mu)] * phi[(g, mu)])
                        .sum::<f64>()
                        .max(0.0);
                }
                for axis in 0..3 {
                    let grad_p = ao_grad[axis].matmul_seq(ground_density[spin]);
                    for g in 0..npts {
                        grad[spin][axis][g] = 2.0
                            * (0..nao)
                                .map(|mu| phip[(g, mu)] * ao_grad[axis][(g, mu)])
                                .sum::<f64>();
                        tau[spin][g] += 0.5
                            * (0..nao)
                                .map(|mu| grad_p[(g, mu)] * ao_grad[axis][(g, mu)])
                                .sum::<f64>();
                    }
                }
            }
            let mut sigma = [vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]];
            for g in 0..npts {
                let ga = Vec3::new(grad[0][0][g], grad[0][1][g], grad[0][2][g]);
                let gb = Vec3::new(grad[1][0][g], grad[1][1][g], grad[1][2][g]);
                sigma[0][g] = ga.dot(ga);
                sigma[1][g] = ga.dot(gb);
                sigma[2][g] = gb.dot(gb);
            }
            let vars = GridVars {
                rho: [&rho[0], &rho[1]],
                sigma: Some([&sigma[0], &sigma[1], &sigma[2]]),
                tau: Some([&tau[0], &tau[1]]),
            };
            let mut xc = XcOut::default();
            functional.eval(&vars, 1, &mut xc)?;
            for spin in 0..2 {
                for g in 0..npts {
                    coefficient[spin][start + g] =
                        if rho[0][g] + rho[1][g] >= 1e-13 && rho[spin][g] > 0.0 {
                            grid.weights[start + g] * xc.vtau[spin][g] / rho[spin][g]
                        } else {
                            0.0
                        };
                }
            }
        }
        Ok(Some(Self { coefficient }))
    }

    /// `K_j[D]` for one spin's real antisymmetric density `dP/dx = iD`.
    pub(crate) fn response_fock(
        &self,
        mb: &MolBasis,
        grid: &MolecularGrid,
        density_hat: &Matrix,
        spin: usize,
    ) -> Result<Matrix> {
        let nao = mb.nao_sph;
        if spin >= 2 || density_hat.rows != nao || density_hat.cols != nao {
            return Err(DftError::InvalidInput(
                "meta-GGA current response density dimensions do not match the AO basis".into(),
            ));
        }
        let mut result = Matrix::zeros(nao, nao);
        let density_hat_t = density_hat.transpose();
        for start in (0..grid.len()).step_by(BATCH) {
            let end = (start + BATCH).min(grid.len());
            let batch = eval_aos(mb, &grid.points[start..end], true);
            Self::accumulate_response(
                &mut result,
                &batch.phi,
                batch.grad.as_ref().expect("gradients requested"),
                &density_hat_t,
                &self.coefficient[spin][start..end],
            );
        }
        Ok(result)
    }

    /// Add the current response from one batch of real AO collocation data.
    /// The coefficients are the quadrature-weighted v_tau/rho of one spin.
    pub(crate) fn accumulate_response(
        result: &mut Matrix,
        phi: &Matrix,
        gradients: &[Matrix; 3],
        density_hat_t: &Matrix,
        coefficients: &[f64],
    ) {
        let nao = phi.cols;
        // Row g, column mu is sum_n phi_n D_{mu n}; using D^T makes
        // the current contraction exactly grad(phi)^T D phi.
        let dphi = phi.matmul_seq(density_hat_t);
        let mut weighted = Matrix::zeros(phi.rows, nao);
        for (g, &coefficient) in coefficients.iter().enumerate() {
            let current = gradients.each_ref().map(|gradient| {
                (0..nao)
                    .map(|mu| gradient[(g, mu)] * dphi[(g, mu)])
                    .sum::<f64>()
            });
            for nu in 0..nao {
                weighted[(g, nu)] = coefficient
                    * (0..3)
                        .map(|axis| current[axis] * gradients[axis][(g, nu)])
                        .sum::<f64>();
            }
        }
        let c = phi.transpose_matmul_seq(&weighted);
        for m in 0..nao {
            for n in 0..nao {
                result[(m, n)] += 0.5 * (c[(m, n)] - c[(n, m)]);
            }
        }
    }

    /// Frozen-density current carried directly by London AO phases.
    pub(super) fn basis_fock(
        &self,
        mb: &MolBasis,
        grid: &MolecularGrid,
        density: &Matrix,
        origin: Vec3,
        field_axis: usize,
        spin: usize,
    ) -> Result<Matrix> {
        let nao = mb.nao_sph;
        if spin >= 2 || density.rows != nao || density.cols != nao {
            return Err(DftError::InvalidInput(
                "meta-GGA basis-current density dimensions do not match the AO basis".into(),
            ));
        }
        let mut centres = vec![Vec3::zero(); nao];
        for shell in &mb.shells {
            for local in 0..n_sph(shell.l) {
                centres[shell.sph_off + local] = shell.center;
            }
        }
        let e = match field_axis {
            0 => Vec3::new(1.0, 0.0, 0.0),
            1 => Vec3::new(0.0, 1.0, 0.0),
            2 => Vec3::new(0.0, 0.0, 1.0),
            _ => {
                return Err(DftError::InvalidInput(
                    "magnetic field axis must be 0, 1, or 2".into(),
                ))
            }
        };
        let mut result = Matrix::zeros(nao, nao);
        for start in (0..grid.len()).step_by(BATCH) {
            let end = (start + BATCH).min(grid.len());
            let batch = eval_aos(mb, &grid.points[start..end], true);
            let phi = &batch.phi;
            let ao_grad = batch.grad.as_ref().expect("gradients requested");
            let mut a = Matrix::zeros(end - start, nao);
            for local in 0..end - start {
                let r = grid.points[start + local] - origin;
                let mut qphi = vec![0.0; nao];
                let mut q = vec![0.0; nao];
                for mu in 0..nao {
                    q[mu] = 0.5 * e.dot((centres[mu] - origin).cross(r));
                    qphi[mu] = q[mu] * phi[(local, mu)];
                }
                for axis in 0..3 {
                    let mut q_p_grad = 0.0;
                    let mut phi_p_qgrad = 0.0;
                    let mut phi_p_tphi = 0.0;
                    for m in 0..nao {
                        for n in 0..nao {
                            let p = density[(m, n)];
                            q_p_grad += qphi[m] * p * ao_grad[axis][(local, n)];
                            phi_p_qgrad += phi[(local, m)] * p * q[n] * ao_grad[axis][(local, n)];
                            phi_p_tphi += phi[(local, m)]
                                * p
                                * (e.cross(centres[n] - origin) * 0.5).get(axis)
                                * phi[(local, n)];
                        }
                    }
                    let current = q_p_grad - phi_p_qgrad - phi_p_tphi;
                    for nu in 0..nao {
                        a[(local, nu)] += self.coefficient[spin][start + local]
                            * current
                            * ao_grad[axis][(local, nu)];
                    }
                }
            }
            let c = phi.transpose_matmul_seq(&a);
            for m in 0..nao {
                for n in 0..nao {
                    result[(m, n)] += 0.5 * (c[(m, n)] - c[(n, m)]);
                }
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::data;
    use crate::grid::GridOptions;
    use crate::scf::{run_scf, ScfOptions};
    use crate::system::{Atom, Molecule};
    use crate::xc::{resolve, FunctionalSpec};

    fn water() -> Molecule {
        Molecule::new(vec![
            Atom {
                z: 8,
                position: Vec3::new(0.0, 0.0, 0.2217),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 1.4309, -0.8867),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, -1.4309, -0.8867),
            },
        ])
    }

    fn meta_energy_with_current(
        functional: &ResolvedFunctional,
        mb: &MolBasis,
        grid: &MolecularGrid,
        density: [&Matrix; 2],
        current_density: &Matrix,
    ) -> f64 {
        let nao = mb.nao_sph;
        let mut energy = 0.0;
        for start in (0..grid.len()).step_by(BATCH) {
            let end = (start + BATCH).min(grid.len());
            let batch = eval_aos(mb, &grid.points[start..end], true);
            let phi = &batch.phi;
            let ao_grad = batch.grad.as_ref().unwrap();
            let npts = end - start;
            let mut rho = [vec![0.0; npts], vec![0.0; npts]];
            let mut grad: [[Vec<f64>; 3]; 2] = Default::default();
            let mut tau = [vec![0.0; npts], vec![0.0; npts]];
            for spin in 0..2 {
                for axis in &mut grad[spin] {
                    *axis = vec![0.0; npts];
                }
                let phip = phi.matmul_seq(density[spin]);
                for g in 0..npts {
                    rho[spin][g] = (0..nao)
                        .map(|mu| phi[(g, mu)] * phip[(g, mu)])
                        .sum::<f64>()
                        .max(0.0);
                }
                for axis in 0..3 {
                    let grad_p = ao_grad[axis].matmul_seq(density[spin]);
                    for g in 0..npts {
                        grad[spin][axis][g] = 2.0
                            * (0..nao)
                                .map(|mu| ao_grad[axis][(g, mu)] * phip[(g, mu)])
                                .sum::<f64>();
                        tau[spin][g] += 0.5
                            * (0..nao)
                                .map(|mu| ao_grad[axis][(g, mu)] * grad_p[(g, mu)])
                                .sum::<f64>();
                    }
                }
            }
            let dphi = phi.matmul_seq(&current_density.transpose());
            for g in 0..npts {
                if rho[0][g] + rho[1][g] >= 1e-13 && rho[0][g] > 0.0 {
                    let j2 = (0..3)
                        .map(|axis| {
                            let j = (0..nao)
                                .map(|mu| ao_grad[axis][(g, mu)] * dphi[(g, mu)])
                                .sum::<f64>();
                            j * j
                        })
                        .sum::<f64>();
                    tau[0][g] -= 0.5 * j2 / rho[0][g];
                }
            }
            let mut sigma = [vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]];
            for g in 0..npts {
                let ga = Vec3::new(grad[0][0][g], grad[0][1][g], grad[0][2][g]);
                let gb = Vec3::new(grad[1][0][g], grad[1][1][g], grad[1][2][g]);
                sigma[0][g] = ga.dot(ga);
                sigma[1][g] = ga.dot(gb);
                sigma[2][g] = gb.dot(gb);
            }
            let vars = GridVars {
                rho: [&rho[0], &rho[1]],
                sigma: Some([&sigma[0], &sigma[1], &sigma[2]]),
                tau: Some([&tau[0], &tau[1]]),
            };
            let mut xc = XcOut::default();
            functional.eval(&vars, 0, &mut xc).unwrap();
            for g in 0..npts {
                energy += grid.weights[start + g] * xc.e[g];
            }
        }
        energy
    }

    #[test]
    fn meta_current_kernel_matches_mixed_energy_finite_difference() {
        let mol = water();
        let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
        let functional = resolve(&FunctionalSpec::Combined("tpss".into())).unwrap();
        let grid = MolecularGrid::build(
            &mol,
            &GridOptions {
                level: 2,
                ..GridOptions::default()
            },
        )
        .unwrap();
        let opts = ScfOptions {
            e_tol: 1e-11,
            p_tol: 1e-10,
            ..ScfOptions::default()
        };
        let scf = run_scf(&mol, &mb, &opts, &functional, Some(&grid)).unwrap();
        let mut dx = Matrix::zeros(mb.nao_sph, mb.nao_sph);
        let mut dy = Matrix::zeros(mb.nao_sph, mb.nao_sph);
        for m in 0..mb.nao_sph {
            for n in m + 1..mb.nao_sph {
                let x = 0.03 * ((m + 2 * n + 1) as f64).sin();
                let y = 0.02 * ((2 * m + n + 3) as f64).cos();
                dx[(m, n)] = x;
                dx[(n, m)] = -x;
                dy[(m, n)] = y;
                dy[(n, m)] = -y;
            }
        }
        let kernel =
            MetaCurrentKernel::build(&functional, &mb, &grid, [&scf.density[0], &scf.density[1]])
                .unwrap()
                .unwrap();
        let kx = kernel.response_fock(&mb, &grid, &dx, 0).unwrap();
        let mut analytic = 0.0;
        for m in 0..mb.nao_sph {
            for n in 0..mb.nao_sph {
                analytic -= dy[(m, n)] * kx[(n, m)];
            }
        }
        let h = 2e-3;
        let energy = |sx: f64, sy: f64| {
            let mut d = Matrix::zeros(mb.nao_sph, mb.nao_sph);
            for i in 0..d.as_slice().len() {
                d.as_mut_slice()[i] = h * (sx * dx.as_slice()[i] + sy * dy.as_slice()[i]);
            }
            meta_energy_with_current(
                &functional,
                &mb,
                &grid,
                [&scf.density[0], &scf.density[1]],
                &d,
            )
        };
        let fd = (energy(1.0, 1.0) - energy(1.0, -1.0) - energy(-1.0, 1.0) + energy(-1.0, -1.0))
            / (4.0 * h * h);
        assert!(analytic.is_finite() && fd.is_finite());
        assert!(analytic.abs() > 1e-8, "current kernel gate is vacuous");
        assert!(
            (analytic - fd).abs() < 2e-7 * analytic.abs().max(fd.abs()).max(1.0),
            "meta current kernel: analytic {analytic:.12e}, mixed FD {fd:.12e}"
        );
    }

    /// The two explicit meta-GGA pieces are gated independently of the
    /// production moment/cross-product assembly:
    ///
    /// 1. construct the conjugated bra and ket London AO values AND gradients
    ///    at B=+/-h, then finite-difference the complete weak semilocal Fock;
    /// 2. construct the frozen-density London paramagnetic current directly
    ///    and finite-difference `-1/2 int (v_tau/rho) |j_B + j_D|^2` in B and
    ///    an arbitrary imaginary antisymmetric density direction D.
    ///
    /// The density is deliberately spin-asymmetric and the molecule/grid have
    /// no useful Cartesian symmetry.  Thus the conventional-tau value-phase,
    /// its gradient-phase companion, and the basis-current term are all
    /// separately observable rather than passing as numerical zeros.
    #[test]
    fn meta_london_tau_and_basis_current_match_independent_complex_finite_differences() {
        type Complex = (f64, f64);
        let cmul =
            |a: Complex, b: Complex| -> Complex { (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0) };
        let cscale = |a: Complex, s: f64| -> Complex { (s * a.0, s * a.1) };
        let cadd = |a: Complex, b: Complex| -> Complex { (a.0 + b.0, a.1 + b.1) };

        let mol = Molecule::new(vec![
            Atom {
                z: 8,
                position: Vec3::new(0.16, -0.24, 0.31),
            },
            Atom {
                z: 1,
                position: Vec3::new(1.41, 0.38, -0.62),
            },
            Atom {
                z: 1,
                position: Vec3::new(-0.72, 1.22, 0.47),
            },
        ]);
        let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
        let functional = resolve(&FunctionalSpec::Combined("tpss".into())).unwrap();
        let points = vec![
            Vec3::new(0.21, -0.11, 0.37),
            Vec3::new(0.83, 0.27, -0.19),
            Vec3::new(-0.44, 0.68, 0.53),
            Vec3::new(1.17, -0.36, 0.42),
            Vec3::new(-0.91, 1.09, -0.28),
            Vec3::new(0.35, 1.42, 0.76),
            Vec3::new(-0.18, -0.73, -0.61),
            Vec3::new(1.64, 0.71, -0.84),
            Vec3::new(-1.13, 0.32, 0.91),
            Vec3::new(0.58, -1.21, 0.14),
        ];
        let weights = vec![0.11, 0.08, 0.13, 0.07, 0.09, 0.12, 0.06, 0.10, 0.075, 0.095];
        let grid = MolecularGrid {
            points,
            raw_weights: weights.clone(),
            weights,
            atom_of: vec![0; 10],
        };
        let nao = mb.nao_sph;
        let npts = grid.len();

        // Two-rank positive semidefinite densities stay inside the physical
        // tau domain without landing on the one-orbital branch point.
        let psd = |spin: usize| {
            let v: Vec<f64> = (0..nao)
                .map(|i| 0.13 + 0.027 * ((i + 2 * spin) % nao) as f64)
                .collect();
            let u: Vec<f64> = (0..nao)
                .map(|i| 0.09 + 0.021 * ((3 * i + spin + 1) % nao) as f64)
                .collect();
            let mut p = Matrix::zeros(nao, nao);
            let (av, au) = if spin == 0 {
                (0.71, 0.23)
            } else {
                (0.39, 0.31)
            };
            for m in 0..nao {
                for n in 0..nao {
                    p[(m, n)] = av * v[m] * v[n] + au * u[m] * u[n];
                }
            }
            p
        };
        let density = [psd(0), psd(1)];
        let origin = Vec3::new(0.47, -0.31, 0.82);
        let mut centres = vec![Vec3::zero(); nao];
        for shell in &mb.shells {
            for local in 0..n_sph(shell.l) {
                centres[shell.sph_off + local] = shell.center;
            }
        }

        let batch = eval_aos(&mb, &grid.points, true);
        let phi = &batch.phi;
        let ao_grad = batch.grad.as_ref().unwrap();
        let mut rho = [vec![0.0; npts], vec![0.0; npts]];
        let mut grad: [[Vec<f64>; 3]; 2] = Default::default();
        let mut tau = [vec![0.0; npts], vec![0.0; npts]];
        for spin in 0..2 {
            for axis in &mut grad[spin] {
                *axis = vec![0.0; npts];
            }
            let phip = phi.matmul_seq(&density[spin]);
            for g in 0..npts {
                rho[spin][g] = (0..nao)
                    .map(|mu| phip[(g, mu)] * phi[(g, mu)])
                    .sum::<f64>()
                    .max(0.0);
            }
            for axis in 0..3 {
                let gp = ao_grad[axis].matmul_seq(&density[spin]);
                for g in 0..npts {
                    grad[spin][axis][g] = 2.0
                        * (0..nao)
                            .map(|mu| phip[(g, mu)] * ao_grad[axis][(g, mu)])
                            .sum::<f64>();
                    tau[spin][g] += 0.5
                        * (0..nao)
                            .map(|mu| gp[(g, mu)] * ao_grad[axis][(g, mu)])
                            .sum::<f64>();
                }
            }
        }
        let mut sigma = [vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]];
        for g in 0..npts {
            for axis in 0..3 {
                sigma[0][g] += grad[0][axis][g] * grad[0][axis][g];
                sigma[1][g] += grad[0][axis][g] * grad[1][axis][g];
                sigma[2][g] += grad[1][axis][g] * grad[1][axis][g];
            }
        }
        let vars = GridVars {
            rho: [&rho[0], &rho[1]],
            sigma: Some([&sigma[0], &sigma[1], &sigma[2]]),
            tau: Some([&tau[0], &tau[1]]),
        };
        let mut xc = XcOut::default();
        functional.eval(&vars, 1, &mut xc).unwrap();
        let mut b: [[Vec<f64>; 3]; 2] = Default::default();
        for spin in 0..2 {
            let other = 1 - spin;
            let own = if spin == 0 {
                &xc.vsigma[0]
            } else {
                &xc.vsigma[2]
            };
            for axis in 0..3 {
                b[spin][axis] = (0..npts)
                    .map(|g| {
                        2.0 * own[g] * grad[spin][axis][g] + xc.vsigma[1][g] * grad[other][axis][g]
                    })
                    .collect();
            }
        }

        let production = giao_semilocal_xc_fock_derivative(
            &functional,
            &mb,
            &grid,
            [&density[0], &density[1]],
            origin,
        )
        .unwrap();
        let fock_imag = |spin: usize, field_axis: usize, m: usize, n: usize, field: f64| {
            let e = match field_axis {
                0 => Vec3::new(1.0, 0.0, 0.0),
                1 => Vec3::new(0.0, 1.0, 0.0),
                _ => Vec3::new(0.0, 0.0, 1.0),
            };
            let mut base = (0.0, 0.0);
            let mut tau_value_phase = (0.0, 0.0);
            let mut tau_full = (0.0, 0.0);
            for g in 0..npts {
                let r = grid.points[g] - origin;
                let q_m = 0.5 * e.dot((centres[m] - origin).cross(r));
                let q_n = 0.5 * e.dot((centres[n] - origin).cross(r));
                let t_m = e.cross(centres[m] - origin) * 0.5;
                let t_n = e.cross(centres[n] - origin) * 0.5;
                let phase_m = ((field * q_m).cos(), (field * q_m).sin());
                let phase_n = ((field * q_n).cos(), -(field * q_n).sin());
                let value_m = cscale(phase_m, phi[(g, m)]);
                let value_n = cscale(phase_n, phi[(g, n)]);
                let value_product = cmul(value_m, value_n);
                let mut base_point = cscale(value_product, xc.vrho[spin][g]);
                let mut tau_value_point = (0.0, 0.0);
                let mut tau_full_point = (0.0, 0.0);
                for derivative_axis in 0..3 {
                    let gm = ao_grad[derivative_axis][(g, m)];
                    let gn = ao_grad[derivative_axis][(g, n)];
                    let grad_m_value = cscale(phase_m, gm);
                    let grad_n_value = cscale(phase_n, gn);
                    let grad_m_full = cmul(
                        phase_m,
                        (gm, field * t_m.get(derivative_axis) * phi[(g, m)]),
                    );
                    let grad_n_full = cmul(
                        phase_n,
                        (gn, -field * t_n.get(derivative_axis) * phi[(g, n)]),
                    );
                    let gga_term = cadd(cmul(grad_m_full, value_n), cmul(value_m, grad_n_full));
                    base_point = cadd(base_point, cscale(gga_term, b[spin][derivative_axis][g]));
                    tau_value_point = cadd(
                        tau_value_point,
                        cscale(cmul(grad_m_value, grad_n_value), 0.5 * xc.vtau[spin][g]),
                    );
                    tau_full_point = cadd(
                        tau_full_point,
                        cscale(cmul(grad_m_full, grad_n_full), 0.5 * xc.vtau[spin][g]),
                    );
                }
                let weight = grid.weights[g];
                base = cadd(base, cscale(base_point, weight));
                tau_value_phase = cadd(tau_value_phase, cscale(tau_value_point, weight));
                tau_full = cadd(tau_full, cscale(tau_full_point, weight));
            }
            (base.1, tau_value_phase.1, tau_full.1)
        };
        let derivative = |spin: usize, axis: usize, m: usize, n: usize, h: f64| {
            let plus = fock_imag(spin, axis, m, n, h);
            let minus = fock_imag(spin, axis, m, n, -h);
            (
                (plus.0 - minus.0) / (2.0 * h),
                (plus.1 - minus.1) / (2.0 * h),
                (plus.2 - minus.2) / (2.0 * h),
            )
        };
        let h = 2e-4;
        let mut full_worst = 0.0_f64;
        let mut full_scale = 0.0_f64;
        let mut tau_value_scale = 0.0_f64;
        let mut tau_gradient_scale = 0.0_f64;
        let mut full_asym = 0.0_f64;
        let mut tau_asym = 0.0_f64;
        let mut direct_full: [[Matrix; 3]; 2] =
            std::array::from_fn(|_| std::array::from_fn(|_| Matrix::zeros(nao, nao)));
        let mut direct_tau: [[Matrix; 3]; 2] =
            std::array::from_fn(|_| std::array::from_fn(|_| Matrix::zeros(nao, nao)));
        let mut direct_tau_gradient: [[Matrix; 3]; 2] =
            std::array::from_fn(|_| std::array::from_fn(|_| Matrix::zeros(nao, nao)));
        for spin in 0..2 {
            for axis in 0..3 {
                for m in 0..nao {
                    for n in 0..nao {
                        let coarse = derivative(spin, axis, m, n, h);
                        let fine = derivative(spin, axis, m, n, 0.5 * h);
                        let extrapolate = |coarse: f64, fine: f64| (4.0 * fine - coarse) / 3.0;
                        let base_fd = extrapolate(coarse.0, fine.0);
                        let tau_value_fd = extrapolate(coarse.1, fine.1);
                        let tau_full_fd = extrapolate(coarse.2, fine.2);
                        let tau_gradient_fd = tau_full_fd - tau_value_fd;
                        let full_fd = base_fd + tau_full_fd;
                        direct_full[spin][axis][(m, n)] = full_fd;
                        direct_tau[spin][axis][(m, n)] = tau_full_fd;
                        direct_tau_gradient[spin][axis][(m, n)] = tau_gradient_fd;
                        let got = production[spin][axis][(m, n)];
                        assert!(
                            got.is_finite()
                                && full_fd.is_finite()
                                && tau_value_fd.is_finite()
                                && tau_gradient_fd.is_finite(),
                            "non-finite meta London Fock derivative"
                        );
                        full_worst = full_worst.max((got - full_fd).abs());
                        full_scale = full_scale.max(got.abs()).max(full_fd.abs());
                        tau_value_scale = tau_value_scale.max(tau_value_fd.abs());
                        tau_gradient_scale = tau_gradient_scale.max(tau_gradient_fd.abs());
                    }
                }
            }
        }
        for spin in 0..2 {
            for axis in 0..3 {
                for m in 0..nao {
                    for n in 0..nao {
                        full_asym = full_asym.max(
                            (direct_full[spin][axis][(m, n)] + direct_full[spin][axis][(n, m)])
                                .abs(),
                        );
                        tau_asym = tau_asym
                            .max(
                                (direct_tau[spin][axis][(m, n)] + direct_tau[spin][axis][(n, m)])
                                    .abs(),
                            )
                            .max(
                                (direct_tau_gradient[spin][axis][(m, n)]
                                    + direct_tau_gradient[spin][axis][(n, m)])
                                    .abs(),
                            );
                    }
                }
            }
        }
        assert!(
            tau_value_scale > 1e-8,
            "conventional-tau value-phase gate is vacuous"
        );
        assert!(
            tau_gradient_scale > 1e-8,
            "conventional-tau gradient-phase gate is vacuous"
        );
        assert!(
            full_worst < 2e-8 * full_scale.max(1.0),
            "meta London Fock derivative differs from complex FD by {full_worst:e} (scale {full_scale:e})"
        );
        assert!(
            full_asym < 2e-10 * full_scale.max(1.0)
                && tau_asym < 2e-10 * tau_value_scale.max(tau_gradient_scale).max(1.0),
            "meta London FD is not antisymmetric: full {full_asym:e}, tau {tau_asym:e}"
        );

        // Independently vary B in the London basis current and an imaginary
        // antisymmetric density direction.  This mixed energy derivative
        // must equal -Tr[D V_basis^B].
        let kernel = MetaCurrentKernel::build(&functional, &mb, &grid, [&density[0], &density[1]])
            .unwrap()
            .unwrap();
        let field_axis = 1;
        let basis_fock = kernel
            .basis_fock(&mb, &grid, &density[0], origin, field_axis, 0)
            .unwrap();
        let mut d = Matrix::zeros(nao, nao);
        for m in 0..nao {
            for n in m + 1..nao {
                let value = 0.025 * ((2 * m + 3 * n + 1) as f64).sin();
                d[(m, n)] = value;
                d[(n, m)] = -value;
            }
        }
        let e = Vec3::new(0.0, 1.0, 0.0);
        let london_current = |field: f64| {
            let mut current = [vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]];
            for g in 0..npts {
                let r = grid.points[g] - origin;
                for m in 0..nao {
                    let q_m = 0.5 * e.dot((centres[m] - origin).cross(r));
                    let t_m = e.cross(centres[m] - origin) * 0.5;
                    let phase_m = ((field * q_m).cos(), (field * q_m).sin());
                    let value_m = cscale(phase_m, phi[(g, m)]);
                    for n in 0..nao {
                        let q_n = 0.5 * e.dot((centres[n] - origin).cross(r));
                        let t_n = e.cross(centres[n] - origin) * 0.5;
                        let phase_n = ((field * q_n).cos(), -(field * q_n).sin());
                        let value_n = cscale(phase_n, phi[(g, n)]);
                        let p = density[0][(m, n)];
                        for axis in 0..3 {
                            let grad_m = cmul(
                                phase_m,
                                (ao_grad[axis][(g, m)], field * t_m.get(axis) * phi[(g, m)]),
                            );
                            let grad_n = cmul(
                                phase_n,
                                (ao_grad[axis][(g, n)], -field * t_n.get(axis) * phi[(g, n)]),
                            );
                            let left = cmul(value_m, grad_n);
                            let right = cmul(grad_m, value_n);
                            current[axis][g] += 0.5 * p * (left.1 - right.1);
                        }
                    }
                }
            }
            current
        };
        let mut response_current = [vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]];
        for g in 0..npts {
            for axis in 0..3 {
                response_current[axis][g] = (0..nao)
                    .flat_map(|m| (0..nao).map(move |n| (m, n)))
                    .map(|(m, n)| d[(m, n)] * ao_grad[axis][(g, m)] * phi[(g, n)])
                    .sum();
            }
        }
        let mixed_energy = |field_step: f64, density_step: f64| {
            let energy = |field_sign: f64, density_sign: f64| {
                let jb = london_current(field_sign * field_step);
                let mut value = 0.0;
                for g in 0..npts {
                    for axis in 0..3 {
                        let j =
                            jb[axis][g] + density_sign * density_step * response_current[axis][g];
                        value -= 0.5 * kernel.coefficient[0][g] * j * j;
                    }
                }
                value
            };
            (energy(1.0, 1.0) - energy(1.0, -1.0) - energy(-1.0, 1.0) + energy(-1.0, -1.0))
                / (4.0 * field_step * density_step)
        };
        let coarse = mixed_energy(2e-4, 1e-3);
        let fine = mixed_energy(1e-4, 1e-3);
        let mixed_fd = (4.0 * fine - coarse) / 3.0;
        let mut analytic = 0.0;
        let mut basis_scale = 0.0_f64;
        let mut basis_asym = 0.0_f64;
        for m in 0..nao {
            for n in 0..nao {
                analytic -= d[(m, n)] * basis_fock[(n, m)];
                basis_scale = basis_scale.max(basis_fock[(m, n)].abs());
                basis_asym = basis_asym.max((basis_fock[(m, n)] + basis_fock[(n, m)]).abs());
            }
        }
        assert!(
            analytic.is_finite() && mixed_fd.is_finite() && basis_scale.is_finite(),
            "non-finite meta basis-current gate"
        );
        assert!(
            basis_scale > 1e-8 && analytic.abs() > 1e-9,
            "basis-current gate is vacuous"
        );
        assert!(
            (analytic - mixed_fd).abs() < 2e-8 * analytic.abs().max(mixed_fd.abs()).max(1.0),
            "meta basis current: analytic {analytic:.12e}, mixed FD {mixed_fd:.12e}"
        );
        assert!(
            basis_asym < 2e-12 * basis_scale.max(1.0),
            "meta basis-current Fock is not antisymmetric: {basis_asym:e}"
        );
    }

    #[test]
    fn lda_giao_xc_rejects_empty_and_malformed_grids() {
        let mol = water();
        let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
        let functional = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
        let p = Matrix::zeros(mb.nao_sph, mb.nao_sph);
        let empty = MolecularGrid {
            points: Vec::new(),
            weights: Vec::new(),
            raw_weights: Vec::new(),
            atom_of: Vec::new(),
        };
        assert!(giao_semilocal_xc_fock_derivative(
            &functional,
            &mb,
            &empty,
            [&p, &p],
            Vec3::zero(),
        )
        .is_err());
        let malformed = MolecularGrid {
            points: vec![Vec3::zero()],
            weights: Vec::new(),
            raw_weights: vec![1.0],
            atom_of: vec![0],
        };
        assert!(giao_semilocal_xc_fock_derivative(
            &functional,
            &mb,
            &malformed,
            [&p, &p],
            Vec3::zero(),
        )
        .is_err());
    }

    #[test]
    fn lda_london_phase_derivative_matches_independent_finite_difference() {
        let mol = water();
        let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
        let functional = resolve(&FunctionalSpec::Combined("svwn5".into())).unwrap();
        let gopts = GridOptions {
            level: 2,
            ..GridOptions::default()
        };
        let grid = MolecularGrid::build(&mol, &gopts).unwrap();
        let opts = ScfOptions {
            e_tol: 1e-11,
            p_tol: 1e-10,
            ..ScfOptions::default()
        };
        let scf = run_scf(&mol, &mb, &opts, &functional, Some(&grid)).unwrap();
        let origin = Vec3::new(0.47, -0.31, 0.82);
        let got = giao_semilocal_xc_fock_derivative(
            &functional,
            &mb,
            &grid,
            [&scf.density[0], &scf.density[1]],
            origin,
        )
        .unwrap();

        // Freeze v_rho at B=0, then differentiate the complex London phase
        // directly through sin(theta). This does not reuse the analytic
        // moment/cross-product assembly above.
        let batch = eval_aos(&mb, &grid.points, false);
        let phi = &batch.phi;
        let npts = grid.len();
        let mut rho = [vec![0.0; npts], vec![0.0; npts]];
        for spin in 0..2 {
            let phip = phi.matmul_seq(&scf.density[spin]);
            for g in 0..npts {
                for mu in 0..mb.nao_sph {
                    rho[spin][g] += phip[(g, mu)] * phi[(g, mu)];
                }
                rho[spin][g] = rho[spin][g].max(0.0);
            }
        }
        let vars = GridVars {
            rho: [&rho[0], &rho[1]],
            sigma: None,
            tau: None,
        };
        let mut xc = XcOut::default();
        functional.eval(&vars, 1, &mut xc).unwrap();
        let mut centres = vec![Vec3::zero(); mb.nao_sph];
        for sh in &mb.shells {
            for local in 0..n_sph(sh.l) {
                centres[sh.sph_off + local] = sh.center;
            }
        }
        let h = 1e-4;
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        let mut asym = 0.0_f64;
        let mut got_scale = 0.0_f64;
        for spin in 0..2 {
            for axis in 0..3 {
                for m in 0..mb.nao_sph {
                    for n in 0..mb.nao_sph {
                        let fd_at = |step: f64| {
                            let mut plus = 0.0;
                            let mut minus = 0.0;
                            for g in 0..npts {
                                let r = grid.points[g] - origin;
                                // Construct the conjugated bra and ket phases
                                // separately. The reference never forms the
                                // production path's centre difference.
                                let bra = 0.5 * step * (centres[m] - origin).cross(r).get(axis);
                                let ket = -0.5 * step * (centres[n] - origin).cross(r).get(axis);
                                let bra_minus = -bra;
                                let ket_minus = -ket;
                                let prefactor =
                                    grid.weights[g] * xc.vrho[spin][g] * phi[(g, m)] * phi[(g, n)];
                                plus += prefactor * (bra + ket).sin();
                                minus += prefactor * (bra_minus + ket_minus).sin();
                            }
                            (plus - minus) / (2.0 * step)
                        };
                        let coarse = fd_at(h);
                        let fine = fd_at(0.5 * h);
                        let fd = (4.0 * fine - coarse) / 3.0;
                        let diff = (got[spin][axis][(m, n)] - fd).abs();
                        assert!(diff.is_finite(), "non-finite LDA GIAO FD difference");
                        worst = worst.max(diff);
                        scale = scale.max(fd.abs());
                        asym = asym.max((got[spin][axis][(m, n)] + got[spin][axis][(n, m)]).abs());
                        got_scale = got_scale.max(got[spin][axis][(m, n)].abs());
                    }
                }
            }
        }
        assert!(scale > 1e-7, "LDA GIAO XC derivative is a numerical zero");
        assert!(
            worst < 2e-9 * scale.max(1.0),
            "LDA GIAO XC derivative differs from London-phase FD by {worst:e} (scale {scale:e})"
        );
        assert!(
            asym < 1e-12 * got_scale.max(1.0),
            "LDA GIAO XC derivative is not antisymmetric: {asym:e}"
        );

        // Exact gauge-origin covariance, independently pinned against the
        // ordinary LDA Vxc matrix assembled by the shared KS implementation.
        let delta = Vec3::new(-0.28, 0.43, 0.19);
        let shifted = giao_semilocal_xc_fock_derivative(
            &functional,
            &mb,
            &grid,
            [&scf.density[0], &scf.density[1]],
            origin + delta,
        )
        .unwrap();
        let v0 =
            crate::ks::xc_contribution(&functional, &mb, &grid, [&scf.density[0], &scf.density[1]])
                .unwrap()
                .vxc;
        let mut shift_worst = 0.0_f64;
        let mut shift_scale = 0.0_f64;
        for spin in 0..2 {
            for axis in 0..3 {
                for m in 0..mb.nao_sph {
                    let gamma_m = 0.5 * (centres[m] - origin).cross(delta).get(axis);
                    for n in 0..mb.nao_sph {
                        let gamma_n = 0.5 * (centres[n] - origin).cross(delta).get(axis);
                        let want = got[spin][axis][(m, n)] + (gamma_n - gamma_m) * v0[spin][(m, n)];
                        let diff = (shifted[spin][axis][(m, n)] - want).abs();
                        assert!(diff.is_finite(), "non-finite LDA XC origin-shift gap");
                        shift_worst = shift_worst.max(diff);
                        shift_scale = shift_scale.max(shifted[spin][axis][(m, n)].abs());
                    }
                }
            }
        }
        assert!(
            shift_worst < 2e-10 * shift_scale.max(1.0),
            "LDA GIAO XC origin-shift identity gap {shift_worst:e} (scale {shift_scale:e})"
        );
    }

    #[test]
    fn gga_london_phase_and_phase_gradient_match_independent_finite_difference() {
        let mol = water();
        let mb = MolBasis::build(&mol, &data::load("sto-3g").unwrap()).unwrap();
        let functional = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
        let grid = MolecularGrid::build(
            &mol,
            &GridOptions {
                level: 2,
                ..GridOptions::default()
            },
        )
        .unwrap();
        let nao = mb.nao_sph;

        // Unequal, positive-semidefinite spin densities make own-spin and
        // cross-spin sigma coefficients independently observable. They need
        // not be a self-consistent state: this is a local contraction gate.
        let va: Vec<f64> = (0..nao).map(|i| 0.17 + 0.031 * i as f64).collect();
        let vb: Vec<f64> = (0..nao)
            .map(|i| 0.11 + 0.023 * ((3 * i + 1) % nao) as f64)
            .collect();
        let outer = |v: &[f64], scale: f64| {
            let mut p = Matrix::zeros(nao, nao);
            for m in 0..nao {
                for n in 0..nao {
                    p[(m, n)] = scale * v[m] * v[n];
                }
            }
            p
        };
        let pa = outer(&va, 0.73);
        let pb = outer(&vb, 0.41);
        let origin = Vec3::new(0.47, -0.31, 0.82);
        let got =
            giao_semilocal_xc_fock_derivative(&functional, &mb, &grid, [&pa, &pb], origin).unwrap();

        let batch = eval_aos(&mb, &grid.points, true);
        let phi = &batch.phi;
        let ao_grad = batch.grad.as_ref().unwrap();
        let npts = grid.len();
        let mut rho = [vec![0.0; npts], vec![0.0; npts]];
        let mut grad: [[Vec<f64>; 3]; 2] = Default::default();
        for spin in &mut grad {
            for axis in spin {
                *axis = vec![0.0; npts];
            }
        }
        for (spin, p) in [&pa, &pb].into_iter().enumerate() {
            let phip = phi.matmul_seq(p);
            for g in 0..npts {
                for mu in 0..nao {
                    rho[spin][g] += phip[(g, mu)] * phi[(g, mu)];
                }
                rho[spin][g] = rho[spin][g].max(0.0);
            }
            for axis in 0..3 {
                for g in 0..npts {
                    for mu in 0..nao {
                        grad[spin][axis][g] += 2.0 * phip[(g, mu)] * ao_grad[axis][(g, mu)];
                    }
                }
            }
        }
        let mut sigma = [vec![0.0; npts], vec![0.0; npts], vec![0.0; npts]];
        for g in 0..npts {
            for axis in 0..3 {
                sigma[0][g] += grad[0][axis][g] * grad[0][axis][g];
                sigma[1][g] += grad[0][axis][g] * grad[1][axis][g];
                sigma[2][g] += grad[1][axis][g] * grad[1][axis][g];
            }
        }
        let vars = GridVars {
            rho: [&rho[0], &rho[1]],
            sigma: Some([&sigma[0], &sigma[1], &sigma[2]]),
            tau: None,
        };
        let mut xc = XcOut::default();
        functional.eval(&vars, 1, &mut xc).unwrap();
        // Build b_s = d e / d(grad rho_s) independently from the production
        // vsigma contraction. Perturbing the density gradients and evaluating
        // the order-zero energy density catches wrong own-spin or cross-spin
        // coefficients in the analytic assembly above.
        let mut b: [[Vec<f64>; 3]; 2] = Default::default();
        let grad_step = 1e-4;
        for spin in 0..2 {
            for axis in 0..3 {
                let point_steps: Vec<f64> = (0..npts)
                    .map(|g| {
                        grad_step * grad[0][axis][g].abs().max(grad[1][axis][g].abs()).max(1e-8)
                    })
                    .collect();
                let energy_at = |factor: f64| {
                    let mut perturbed = sigma.clone();
                    for g in 0..npts {
                        let delta = factor * point_steps[g];
                        let ga = grad[0][axis][g];
                        let gb = grad[1][axis][g];
                        if spin == 0 {
                            perturbed[0][g] += 2.0 * delta * ga + delta * delta;
                            perturbed[1][g] += delta * gb;
                        } else {
                            perturbed[1][g] += delta * ga;
                            perturbed[2][g] += 2.0 * delta * gb + delta * delta;
                        }
                    }
                    let perturbed_vars = GridVars {
                        rho: [&rho[0], &rho[1]],
                        sigma: Some([&perturbed[0], &perturbed[1], &perturbed[2]]),
                        tau: None,
                    };
                    let mut out = XcOut::default();
                    functional.eval(&perturbed_vars, 0, &mut out).unwrap();
                    out.e
                };
                let plus = energy_at(1.0);
                let minus = energy_at(-1.0);
                let plus_fine = energy_at(0.5);
                let minus_fine = energy_at(-0.5);
                b[spin][axis] = plus
                    .iter()
                    .zip(&minus)
                    .zip(&plus_fine)
                    .zip(&minus_fine)
                    .zip(&point_steps)
                    .map(|((((ep, em), ep_fine), em_fine), step)| {
                        let coarse = (ep - em) / (2.0 * step);
                        let fine = (ep_fine - em_fine) / step;
                        (4.0 * fine - coarse) / 3.0
                    })
                    .collect();
            }
        }
        let mut centres = vec![Vec3::zero(); nao];
        for sh in &mb.shells {
            for local in 0..n_sph(sh.l) {
                centres[sh.sph_off + local] = sh.center;
            }
        }

        let complex_mul =
            |a: (f64, f64), z: (f64, f64)| (a.0 * z.0 - a.1 * z.1, a.0 * z.1 + a.1 * z.0);
        let fd_at = |spin: usize,
                     axis: usize,
                     m: usize,
                     n: usize,
                     step: f64,
                     include_phase_gradient: bool| {
            let integrate = |field: f64| {
                let mut imag = 0.0;
                let field_axis = match axis {
                    0 => Vec3::new(field, 0.0, 0.0),
                    1 => Vec3::new(0.0, field, 0.0),
                    _ => Vec3::new(0.0, 0.0, field),
                };
                for g in 0..npts {
                    let r = grid.points[g] - origin;
                    let dm = centres[m] - origin;
                    let dn = centres[n] - origin;
                    let theta_m = 0.5 * field_axis.dot(dm.cross(r));
                    let theta_n = -0.5 * field_axis.dot(dn.cross(r));
                    let phase_m = (theta_m.cos(), theta_m.sin());
                    let phase_n = (theta_n.cos(), theta_n.sin());
                    let value_m = (phi[(g, m)] * phase_m.0, phi[(g, m)] * phase_m.1);
                    let value_n = (phi[(g, n)] * phase_n.0, phi[(g, n)] * phase_n.1);
                    let product = complex_mul(value_m, value_n);
                    let mut integrand_im = xc.vrho[spin][g] * product.1;
                    let grad_theta_m = if include_phase_gradient {
                        field_axis.cross(dm) * 0.5
                    } else {
                        Vec3::zero()
                    };
                    let grad_theta_n = if include_phase_gradient {
                        field_axis.cross(dn) * -0.5
                    } else {
                        Vec3::zero()
                    };
                    for j in 0..3 {
                        let gm = ao_grad[j][(g, m)];
                        let gn = ao_grad[j][(g, n)];
                        let phase_grad_m = grad_theta_m.get(j) * phi[(g, m)];
                        let phase_grad_n = grad_theta_n.get(j) * phi[(g, n)];
                        let grad_m = (
                            phase_m.0 * gm - phase_m.1 * phase_grad_m,
                            phase_m.1 * gm + phase_m.0 * phase_grad_m,
                        );
                        let grad_n = (
                            phase_n.0 * gn - phase_n.1 * phase_grad_n,
                            phase_n.1 * gn + phase_n.0 * phase_grad_n,
                        );
                        integrand_im += b[spin][j][g]
                            * (complex_mul(grad_m, value_n).1 + complex_mul(value_m, grad_n).1);
                    }
                    imag += grid.weights[g] * integrand_im;
                }
                imag
            };
            (integrate(step) - integrate(-step)) / (2.0 * step)
        };

        let h = 1e-4;
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        let mut phase_gradient_scale = 0.0_f64;
        let mut spin_gap = 0.0_f64;
        let mut asym = 0.0_f64;
        for spin in 0..2 {
            for axis in 0..3 {
                for m in 0..nao {
                    for n in 0..nao {
                        let coarse = fd_at(spin, axis, m, n, h, true);
                        let fine = fd_at(spin, axis, m, n, 0.5 * h, true);
                        let fd = (4.0 * fine - coarse) / 3.0;
                        let moment_only = fd_at(spin, axis, m, n, 0.5 * h, false);
                        assert!(moment_only.is_finite(), "non-finite GGA moment-only FD");
                        let diff = (got[spin][axis][(m, n)] - fd).abs();
                        assert!(diff.is_finite(), "non-finite GGA London-phase FD gap");
                        worst = worst.max(diff);
                        scale = scale.max(fd.abs());
                        let phase_gradient = (fd - moment_only).abs();
                        assert!(
                            phase_gradient.is_finite(),
                            "non-finite GGA phase-gradient FD"
                        );
                        phase_gradient_scale = phase_gradient_scale.max(phase_gradient);
                        spin_gap =
                            spin_gap.max((got[0][axis][(m, n)] - got[1][axis][(m, n)]).abs());
                        asym = asym.max((got[spin][axis][(m, n)] + got[spin][axis][(n, m)]).abs());
                    }
                }
            }
        }
        assert!(scale > 1e-6, "GGA GIAO XC derivative is a numerical zero");
        assert!(
            phase_gradient_scale > 1e-7,
            "GGA phase-gradient term is not observable in the FD gate"
        );
        assert!(
            spin_gap > 1e-7,
            "unequal spin channels produced equal derivatives"
        );
        assert!(
            worst < 2e-8 * scale.max(1.0),
            "GGA GIAO XC derivative differs from independent complex FD by {worst:e} (scale {scale:e})"
        );
        assert!(
            asym < 2e-12 * scale.max(1.0),
            "GGA GIAO XC derivative is not antisymmetric: {asym:e}"
        );

        let swapped =
            giao_semilocal_xc_fock_derivative(&functional, &mb, &grid, [&pb, &pa], origin).unwrap();
        let mut swap_gap = 0.0_f64;
        for spin in 0..2 {
            for axis in 0..3 {
                for m in 0..nao {
                    for n in 0..nao {
                        let diff =
                            (got[spin][axis][(m, n)] - swapped[1 - spin][axis][(m, n)]).abs();
                        assert!(diff.is_finite(), "non-finite GGA spin-swap gap");
                        swap_gap = swap_gap.max(diff);
                    }
                }
            }
        }
        assert!(
            swap_gap < 1e-10,
            "GGA spin-swap covariance gap {swap_gap:e}"
        );

        let delta = Vec3::new(-0.28, 0.43, 0.19);
        let shifted =
            giao_semilocal_xc_fock_derivative(&functional, &mb, &grid, [&pa, &pb], origin + delta)
                .unwrap();
        let v0 = crate::ks::xc_contribution(&functional, &mb, &grid, [&pa, &pb])
            .unwrap()
            .vxc;
        let mut shift_worst = 0.0_f64;
        for spin in 0..2 {
            for axis in 0..3 {
                for m in 0..nao {
                    let gamma_m = 0.5 * (centres[m] - origin).cross(delta).get(axis);
                    for n in 0..nao {
                        let gamma_n = 0.5 * (centres[n] - origin).cross(delta).get(axis);
                        let want = got[spin][axis][(m, n)] + (gamma_n - gamma_m) * v0[spin][(m, n)];
                        let diff = (shifted[spin][axis][(m, n)] - want).abs();
                        assert!(diff.is_finite(), "non-finite GGA XC origin-shift gap");
                        shift_worst = shift_worst.max(diff);
                    }
                }
            }
        }
        assert!(
            shift_worst < 5e-9,
            "GGA XC origin-shift identity gap {shift_worst:e}"
        );
    }
}

#[cfg(test)]
#[path = "giao_learned_tests.rs"]
mod learned_tests;
