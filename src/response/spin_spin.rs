// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Indirect nuclear spin-spin coupling of a closed-shell molecule.
//!
//! With the nuclear magnetic moments `m_N` as perturbations (atomic units,
//! `mu_0 / 4 pi = alpha^2`), the reduced coupling tensor
//! `K_MN,ab = d2E / (dm_M,a dm_N,b)` has four parts:
//!
//! ```text
//! diamagnetic spin-orbit   alpha^4 <[delta_ab r_M.r_N - r_N,a r_M,b] / (r_M^3 r_N^3)>
//! paramagnetic spin-orbit  response to  -i alpha^2 m_N . (r_N x grad) / r_N^3
//! Fermi contact            response to  (g_e/2)(8 pi alpha^2 / 3) delta(r_N) m_N . s
//! spin dipole              response to  (g_e/2) alpha^2 [3 (s.r_N)(m_N.r_N) - r_N^2 s.m_N] / r_N^5
//! ```
//!
//! The diamagnetic term is an expectation value, integrated on the molecular
//! grid. The paramagnetic term is an imaginary singlet perturbation, solved
//! with the magnetic Hessian. The last two are real triplet perturbations:
//! for a closed shell the three spin components are equivalent, and the
//! component along z is the response of the alpha and beta densities to
//! `+A/2` and `-A/2`, solved with the spin-resolved orbital Hessian. The
//! second derivative of the nuclear attraction integral with respect to the
//! position of the nucleus is the principal value of the dipolar operator
//! minus `(4 pi / 3) delta_ka delta(r_N)`, which gives both operators from
//! one integral and the basis functions at the nucleus.

use super::krylov::OperatorStructure;
use super::magnetic_taylor::{imaginary_density, solve_imaginary_with_current};
use super::{ResponseSolver, ResponseSolverConfig};
use crate::basis::{n_sph, MolBasis};
use crate::constants::{ELECTRON_G_FACTOR, FINE_STRUCTURE};
use crate::dual::{DualN, Scalar};
use crate::grid::eval::{eval_aos, eval_aos_full};
use crate::grid::{GridOptions, MolecularGrid};
use crate::integrals::analytic::GaussianKernel;
use crate::integrals::analytic_basis::{self as jets, AnalyticNucleus, AnalyticShell};
use crate::linalg::Matrix;
use crate::properties::nmr::paramagnetic_operator;
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

mod spinor;
pub use spinor::{
    spin_direction_constraints, spinor_nuclear_magnetic_sources, spinor_spin_spin_couplings,
    total_spin_constraints, SpinorSpinSpinCouplings,
};

/// Reduced coupling tensors of every requested pair of nuclei.
#[derive(Clone, Debug)]
pub struct SpinSpinCouplings {
    /// Pairs `(M, N)` in the supplied `nuclei` order. The first tensor
    /// index belongs to M and the second to N; reversing a pair transposes
    /// its tensor. The atom indices themselves need not satisfy M < N.
    pub pairs: Vec<(usize, usize)>,
    /// `K[a][b]` per pair and term, atomic units.
    pub diamagnetic: Vec<[[f64; 3]; 3]>,
    pub paramagnetic: Vec<[[f64; 3]; 3]>,
    /// Fermi contact with Fermi contact.
    pub fermi_contact: Vec<[[f64; 3]; 3]>,
    /// Spin dipole with spin dipole and its cross terms with Fermi contact.
    pub spin_dipole: Vec<[[f64; 3]; 3]>,
    pub total: Vec<[[f64; 3]; 3]>,
    /// Largest relative residual of the response solves.
    pub residual: f64,
}

/// Contributions of the orbitals of an orthonormal occupied set to every
/// coupling tensor, `[pair][orbital][a][b]`, atomic units.
#[derive(Clone, Debug)]
pub struct OrbitalCouplings {
    pub diamagnetic: Vec<Vec<[[f64; 3]; 3]>>,
    pub paramagnetic: Vec<Vec<[[f64; 3]; 3]>>,
    pub fermi_contact: Vec<Vec<[[f64; 3]; 3]>>,
    pub spin_dipole: Vec<Vec<[[f64; 3]; 3]>>,
    /// Largest `|sum over orbitals - total|` over pairs, terms and components.
    pub decomposition_residual: f64,
    /// Largest `|T^T T - I|` of the occupied-space rotation.
    pub rotation_residual: f64,
}

fn add(target: &mut Matrix, source: &Matrix, weight: f64) {
    for (a, b) in target.as_mut_slice().iter_mut().zip(source.as_slice()) {
        *a += weight * b;
    }
}

/// `d2 / (dC_k dC_a) <m| 1 / |r - C| |n>` at `center`, as `[3 k + a]`.
pub(crate) fn field_gradient_operator(mb: &MolBasis, center: [f64; 3]) -> Result<Vec<Matrix>> {
    type Inner = DualN<f64, 3>;
    type Outer = DualN<Inner, 3>;
    let nao = mb.nao_sph;
    let position: [Outer; 3] =
        std::array::from_fn(|k| Outer::variable(Inner::variable(center[k], k), k));
    // Charge -1 turns the attraction integral into +<1 / |r - C|>.
    let nucleus = [AnalyticNucleus {
        center: position,
        charge: <Outer as Scalar>::from_f64(-1.0),
    }];
    let shells: Vec<AnalyticShell<Outer>> = mb
        .shells
        .iter()
        .map(|shell| {
            let c = shell.center.to_array();
            AnalyticShell::from_placed(shell, c.map(<Outer as Scalar>::from_f64))
        })
        .collect::<Result<_>>()?;
    let mut out = vec![Matrix::zeros(nao, nao); 9];
    for (a, sa) in mb.shells.iter().enumerate() {
        for (b, sb) in mb.shells.iter().enumerate() {
            let block =
                jets::attraction(&shells[a], &shells[b], &nucleus, GaussianKernel::Coulomb, 2)?;
            let nb = n_sph(sb.l);
            for i in 0..n_sph(sa.l) {
                for j in 0..nb {
                    let value = block.values[i * nb + j];
                    for k in 0..3 {
                        for c in 0..3 {
                            out[3 * k + c][(sa.sph_off + i, sb.sph_off + j)] = value.im[k].im[c];
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Operators of one nucleus.
pub struct NuclearMagneticSite {
    pub atom: usize,
    /// `phi_m(R) phi_n(R)`.
    pub contact: Matrix,
    /// Fermi-contact plus spin-dipole operator `[3 k + a]` for the spin
    /// component `k` and the moment component `a`.
    pub triplet: Vec<Matrix>,
    /// `<(r_N x grad)_a / r_N^3>`, antisymmetric.
    pub orbital: [Matrix; 3],
}

/// Analytic all-electron nuclear magnetic operators in the supplied AO basis.
/// Contact is unscaled delta(r-R); triplet includes (g_e/2) alpha^2 and
/// orbital has its factor -i alpha^2 removed. Components are [3*k+a],
/// with electronic spin k and nuclear moment a. No AO phase is assumed.
pub fn nuclear_magnetic_sites(
    mol: &Molecule,
    mb: &MolBasis,
    nuclei: &[usize],
    fermi_contact_only: bool,
) -> Result<Vec<NuclearMagneticSite>> {
    if !mb.ecps.is_empty() {
        return Err(DftError::Unsupported {
            feature: "spin-spin coupling with effective core potentials".into(),
            missing: "the core region of the contact and dipolar operators".into(),
        });
    }
    if nuclei.iter().any(|&atom| atom >= mol.len()) {
        return Err(DftError::InvalidInput(
            "spin-spin coupling nuclei are atom indices of the molecule".into(),
        ));
    }
    let mut selected = nuclei.to_vec();
    selected.sort_unstable();
    if selected.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(DftError::InvalidInput(
            "spin-spin coupling nuclei must be distinct atom indices".into(),
        ));
    }
    let nao = mb.nao_sph;
    let alpha2 = FINE_STRUCTURE * FINE_STRUCTURE;
    let spin_factor = 0.5 * ELECTRON_G_FACTOR;
    let points: Vec<crate::math::Vec3> = nuclei.iter().map(|&a| mol.atoms[a].position).collect();
    let at_nuclei = eval_aos(mb, &points, false);
    let mut sites = Vec::with_capacity(nuclei.len());
    for (index, &atom) in nuclei.iter().enumerate() {
        let mut contact = Matrix::zeros(nao, nao);
        for m in 0..nao {
            for n in 0..nao {
                contact[(m, n)] = at_nuclei.phi[(index, m)] * at_nuclei.phi[(index, n)];
            }
        }
        let center = mol.atoms[atom].position;
        let mut triplet = Vec::new();
        if !fermi_contact_only {
            // (g_e / 2) alpha^2 [d_k d_a <1/r> + 4 pi delta_ka delta(r)]:
            // Fermi contact (8 pi / 3) plus the principal-value dipolar
            // operator, whose contact part is (4 pi / 3).
            triplet = field_gradient_operator(mb, center.to_array())?;
            for k in 0..3 {
                for a in 0..3 {
                    let operator = &mut triplet[3 * k + a];
                    for value in operator.as_mut_slice() {
                        *value *= spin_factor * alpha2;
                    }
                    if k == a {
                        add(
                            operator,
                            &contact,
                            spin_factor * alpha2 * 4.0 * std::f64::consts::PI,
                        );
                    }
                }
            }
        }
        sites.push(NuclearMagneticSite {
            atom,
            contact,
            triplet,
            orbital: paramagnetic_operator(mb, center),
        });
    }
    Ok(sites)
}

/// DSO kernel without alpha^4, with axes belonging to centres M and N.
/// A quadrature node exactly at a nucleus has no pointwise value.
pub(crate) fn diamagnetic_spin_orbit_kernel(
    point: [f64; 3],
    rm: [f64; 3],
    rn: [f64; 3],
) -> Option<[[f64; 3]; 3]> {
    let dm: [f64; 3] = std::array::from_fn(|a| point[a] - rm[a]);
    let dn: [f64; 3] = std::array::from_fn(|a| point[a] - rn[a]);
    let lm = dm.iter().map(|v| v * v).sum::<f64>().sqrt();
    let ln = dn.iter().map(|v| v * v).sum::<f64>().sqrt();
    if lm == 0.0 || ln == 0.0 {
        return None;
    }
    let dot = (0..3).map(|a| dm[a] * dn[a]).sum::<f64>();
    Some(std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            ((if a == b { dot } else { 0.0 }) - dn[a] * dm[b]) / (lm.powi(3) * ln.powi(3))
        })
    }))
}

/// Reduced coupling tensors between the atoms `nuclei` of a restricted
/// closed-shell reference. `fermi_contact_only` skips every other term.
/// Atom indices must be distinct. Pairs follow the input order, with
/// `K_MN[a][b] = d2E / (dm_M,a dm_N,b)`; no within-pair symmetrization.
#[allow(clippy::too_many_arguments)]
pub fn spin_spin_couplings(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    nuclei: &[usize],
    fermi_contact_only: bool,
    config: &ResponseSolverConfig,
) -> Result<SpinSpinCouplings> {
    Ok(couplings(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        nuclei,
        fermi_contact_only,
        config,
        None,
    )?
    .0)
}

/// The couplings and the contributions of the orbitals `orbitals` (AO
/// coefficients of an orthonormal set spanning the occupied space, for
/// example natural localized molecular orbitals): the occupied rotation
/// `T = C_o^T S L` carries the response amplitudes of the canonical
/// orbitals to the set, and every term is a sum over its orbitals, the
/// diamagnetic one with the orbital densities `2 |l_k|^2`.
#[allow(clippy::too_many_arguments)]
pub fn spin_spin_couplings_by_orbital(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    nuclei: &[usize],
    config: &ResponseSolverConfig,
    orbitals: &Matrix,
) -> Result<(SpinSpinCouplings, OrbitalCouplings)> {
    let (total, parts) = couplings(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        nuclei,
        false,
        config,
        Some(orbitals),
    )?;
    Ok((total, parts.expect("orbital contributions were requested")))
}

#[allow(clippy::too_many_arguments)]
fn couplings(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    nuclei: &[usize],
    fermi_contact_only: bool,
    config: &ResponseSolverConfig,
    orbitals: Option<&Matrix>,
) -> Result<(SpinSpinCouplings, Option<OrbitalCouplings>)> {
    config.validate()?;
    if scf.unrestricted || scf.n_alpha != scf.n_beta || !scf.converged {
        return Err(DftError::Unsupported {
            feature: "spin-spin coupling of an open-shell reference".into(),
            missing: "the open-shell coupling of the singlet and triplet responses; use a \
                      converged restricted closed-shell reference"
                .into(),
        });
    }
    let sites = nuclear_magnetic_sites(mol, mb, nuclei, fermi_contact_only)?;
    let nao = mb.nao_sph;
    let alpha2 = FINE_STRUCTURE * FINE_STRUCTURE;
    let spin_factor = 0.5 * ELECTRON_G_FACTOR;
    // Spin-resolved orbital Hessian of the same closed-shell orbitals for the
    // triplet perturbations.
    let mut open = scf.clone();
    open.unrestricted = true;
    let spin_solver = ResponseSolver::new(mol, mb, functional, grid, &open, opts)?;
    let solver = ResponseSolver::new(mol, mb, functional, grid, scf, opts)?;
    let mut residual = 0.0_f64;
    // Spin density response `dP_alpha - dP_beta` to `+A/2`, `-A/2`.
    let mut triplet_response = |operator: &Matrix| -> Result<(Matrix, Vec<f64>)> {
        let mut up = operator.clone();
        for value in up.as_mut_slice() {
            *value *= 0.5;
        }
        let mut down = up.clone();
        for value in down.as_mut_slice() {
            *value = -*value;
        }
        let mut source = vec![0.0; spin_solver.u_len()];
        spin_solver.project_ov([&up, &down], &mut source);
        if source.iter().all(|v| *v == 0.0) {
            return Ok((Matrix::zeros(nao, nao), vec![0.0; spin_solver.u_len()]));
        }
        let solution = spin_solver.solve_ov_report_with_structure(
            &source,
            config,
            OperatorStructure::Hermitian,
        )?;
        residual = residual.max(solution.relative_residual);
        let u: Vec<f64> = solution.amplitudes.iter().map(|z| z.re).collect();
        let density = spin_solver.delta_density(&u);
        let mut spin = density[0].clone();
        add(&mut spin, &density[1], -1.0);
        Ok((spin, u))
    };
    let contact_scale = spin_factor * alpha2 * 8.0 * std::f64::consts::PI / 3.0;
    let mut contact_response = Vec::with_capacity(sites.len());
    let mut contact_amplitudes = Vec::with_capacity(sites.len());
    let mut full_response: Vec<Vec<Matrix>> = Vec::with_capacity(sites.len());
    let mut full_amplitudes: Vec<Vec<Vec<f64>>> = Vec::with_capacity(sites.len());
    for site in &sites {
        let mut scaled = site.contact.clone();
        for value in scaled.as_mut_slice() {
            *value *= contact_scale;
        }
        let (density, amplitudes) = triplet_response(&scaled)?;
        contact_response.push(density);
        contact_amplitudes.push(amplitudes);
        let mut responses = Vec::new();
        let mut amplitudes = Vec::new();
        for operator in &site.triplet {
            let (density, u) = triplet_response(operator)?;
            responses.push(density);
            amplitudes.push(u);
        }
        full_response.push(responses);
        full_amplitudes.push(amplitudes);
    }
    // Imaginary singlet response to the orbital operators.
    let mut orbital_response: Vec<Vec<Matrix>> = Vec::with_capacity(sites.len());
    let mut orbital_amplitudes: Vec<Vec<Vec<f64>>> = Vec::with_capacity(sites.len());
    let current = crate::properties::giao_xc::MetaCurrentKernel::for_reference(
        mol, mb, functional, grid, scf, opts,
    )?;
    let mut imaginary_residual = 0.0_f64;
    if !fermi_contact_only {
        for site in &sites {
            let mut responses = Vec::with_capacity(3);
            let mut amplitudes = Vec::with_capacity(3);
            for a in 0..3 {
                let mut source_matrix = site.orbital[a].clone();
                for value in source_matrix.as_mut_slice() {
                    *value *= -alpha2;
                }
                let mut source = vec![0.0; solver.u_len()];
                solver.project_ov([&source_matrix, &source_matrix], &mut source);
                let (y, relative) =
                    solve_imaginary_with_current(&solver, &source, config, current.as_ref())?;
                imaginary_residual = imaginary_residual.max(relative);
                let density = imaginary_density(&solver, &y);
                let mut total = density[0].clone();
                add(&mut total, &density[1], 1.0);
                responses.push(total);
                amplitudes.push(y);
            }
            orbital_response.push(responses);
            orbital_amplitudes.push(amplitudes);
        }
    }
    // Density on the grid for the diamagnetic term.
    let built;
    let quadrature = match grid {
        Some(grid) => grid,
        None => {
            built = MolecularGrid::build(mol, &GridOptions::default())?;
            &built
        }
    };
    let mut density = vec![0.0; quadrature.len()];
    if !fermi_contact_only {
        let total = scf.p_total();
        for start in (0..quadrature.len()).step_by(256) {
            let end = (start + 256).min(quadrature.len());
            let batch = eval_aos_full(mb, &quadrature.points[start..end], false, false);
            let x = batch.phi.matmul(&total);
            for i in 0..end - start {
                let mut value = 0.0;
                for m in 0..nao {
                    value += x[(i, m)] * batch.phi[(i, m)];
                }
                density[start + i] = value;
            }
        }
    }
    let mut out = SpinSpinCouplings {
        pairs: Vec::new(),
        diamagnetic: Vec::new(),
        paramagnetic: Vec::new(),
        fermi_contact: Vec::new(),
        spin_dipole: Vec::new(),
        total: Vec::new(),
        residual: 0.0,
    };
    for m in 0..sites.len() {
        for n in m + 1..sites.len() {
            let mut diamagnetic = [[0.0; 3]; 3];
            let mut paramagnetic = [[0.0; 3]; 3];
            let mut fermi = [[0.0; 3]; 3];
            let mut dipolar = [[0.0; 3]; 3];
            // Fermi contact with Fermi contact: isotropic.
            let mut scaled = sites[n].contact.clone();
            for value in scaled.as_mut_slice() {
                *value *= contact_scale;
            }
            let contact = 0.5 * contact_response[m].frobenius_dot(&scaled);
            for a in 0..3 {
                fermi[a][a] = contact;
            }
            if !fermi_contact_only {
                for a in 0..3 {
                    for b in 0..3 {
                        // Triplet part summed over the spin component k.
                        let mut triplet = 0.0;
                        for k in 0..3 {
                            triplet += 0.5
                                * full_response[m][3 * k + a]
                                    .frobenius_dot(&sites[n].triplet[3 * k + b]);
                        }
                        dipolar[a][b] = triplet - fermi[a][b];
                        // Tr[(i I)(-i alpha^2 O)] = -alpha^2 sum I_mn O_mn.
                        paramagnetic[a][b] =
                            -alpha2 * orbital_response[m][a].frobenius_dot(&sites[n].orbital[b]);
                    }
                }
                let rm = mol.atoms[sites[m].atom].position.to_array();
                let rn = mol.atoms[sites[n].atom].position.to_array();
                for (i, point) in quadrature.points.iter().enumerate() {
                    let Some(kernel) = diamagnetic_spin_orbit_kernel(point.to_array(), rm, rn)
                    else {
                        continue;
                    };
                    let weight = alpha2 * alpha2 * quadrature.weights[i] * density[i];
                    for a in 0..3 {
                        for b in 0..3 {
                            diamagnetic[a][b] += weight * kernel[a][b];
                        }
                    }
                }
            }
            let mut total = [[0.0; 3]; 3];
            for a in 0..3 {
                for b in 0..3 {
                    total[a][b] =
                        diamagnetic[a][b] + paramagnetic[a][b] + fermi[a][b] + dipolar[a][b];
                }
            }
            out.pairs.push((sites[m].atom, sites[n].atom));
            out.diamagnetic.push(diamagnetic);
            out.paramagnetic.push(paramagnetic);
            out.fermi_contact.push(fermi);
            out.spin_dipole.push(dipolar);
            out.total.push(total);
        }
    }
    out.residual = residual.max(imaginary_residual);
    let Some(l) = orbitals else {
        return Ok((out, None));
    };
    // Orbital contributions.
    let c = &scf.mo_coeff[0];
    let no = scf.n_alpha;
    let nv = c.cols - no;
    let count = l.cols;
    let mut co = Matrix::zeros(nao, no);
    let mut cv = Matrix::zeros(nao, nv);
    for m in 0..nao {
        for i in 0..no {
            co[(m, i)] = c[(m, i)];
        }
        for a in 0..nv {
            cv[(m, a)] = c[(m, no + a)];
        }
    }
    if l.rows != nao || count != no {
        return Err(DftError::InvalidInput(format!(
            "the orbital decomposition needs {no} orthonormal occupied orbitals, not {count}"
        )));
    }
    let t = co.transpose_matmul(&scf.overlap.matmul(l));
    let rotation_residual = t.transpose_matmul(&t).max_abs_diff(&Matrix::identity(no));
    if rotation_residual > 1e-6 {
        return Err(DftError::InvalidInput(format!(
            "the orbitals of the decomposition do not span the occupied space (|T^T T - I| = \
             {rotation_residual:e})"
        )));
    }
    let column = |k: usize| -> Vec<f64> { (0..nao).map(|m| l[(m, k)]).collect() };
    // AO vector C_v (U t_k) of the amplitudes u[offset + a no + i].
    let response_vector = |u: &[f64], offset: usize, k: usize| -> Vec<f64> {
        let w: Vec<f64> = (0..nv)
            .map(|a| (0..no).map(|i| u[offset + a * no + i] * t[(i, k)]).sum())
            .collect();
        (0..nao)
            .map(|m| (0..nv).map(|a| cv[(m, a)] * w[a]).sum())
            .collect()
    };
    let outer = |x: &[f64], y: &[f64], sign: f64, scale: f64, target: &mut Matrix| {
        for (m, xm) in x.iter().enumerate() {
            for (n, yn) in y.iter().enumerate() {
                target[(m, n)] += scale * xm * yn;
                target[(n, m)] += scale * sign * xm * yn;
            }
        }
    };
    // Spin density of orbital k: alpha minus beta of (u l^T + l u^T).
    let triplet_part = |u: &[f64], k: usize| -> Matrix {
        let lk = column(k);
        let mut d = Matrix::zeros(nao, nao);
        outer(&response_vector(u, 0, k), &lk, 1.0, 1.0, &mut d);
        outer(&response_vector(u, no * nv, k), &lk, 1.0, -1.0, &mut d);
        d
    };
    // Imaginary density of orbital k, both spins: 2 (y l^T - l y^T).
    let singlet_part = |y: &[f64], k: usize| -> Matrix {
        let mut d = Matrix::zeros(nao, nao);
        outer(&response_vector(y, 0, k), &column(k), -1.0, 2.0, &mut d);
        d
    };
    // Orbital densities 2 |l_k|^2 on the grid.
    let mut orbital_density = vec![vec![0.0; quadrature.len()]; count];
    if !fermi_contact_only {
        for start in (0..quadrature.len()).step_by(256) {
            let end = (start + 256).min(quadrature.len());
            let batch = eval_aos_full(mb, &quadrature.points[start..end], false, false);
            let values = batch.phi.matmul(l);
            for (k, density) in orbital_density.iter_mut().enumerate() {
                for i in 0..end - start {
                    density[start + i] = 2.0 * values[(i, k)] * values[(i, k)];
                }
            }
        }
    }
    let mut parts = OrbitalCouplings {
        diamagnetic: Vec::new(),
        paramagnetic: Vec::new(),
        fermi_contact: Vec::new(),
        spin_dipole: Vec::new(),
        decomposition_residual: 0.0,
        rotation_residual,
    };
    let mut pair = 0;
    for m in 0..sites.len() {
        for n in m + 1..sites.len() {
            let mut scaled = sites[n].contact.clone();
            for value in scaled.as_mut_slice() {
                *value *= contact_scale;
            }
            let mut dia = vec![[[0.0; 3]; 3]; count];
            let mut para = vec![[[0.0; 3]; 3]; count];
            let mut fermi = vec![[[0.0; 3]; 3]; count];
            let mut dipolar = vec![[[0.0; 3]; 3]; count];
            for k in 0..count {
                let contact = 0.5 * triplet_part(&contact_amplitudes[m], k).frobenius_dot(&scaled);
                for a in 0..3 {
                    fermi[k][a][a] = contact;
                }
                if fermi_contact_only {
                    continue;
                }
                let triplets: Vec<Matrix> = full_amplitudes[m]
                    .iter()
                    .map(|u| triplet_part(u, k))
                    .collect();
                let singlets: Vec<Matrix> = orbital_amplitudes[m]
                    .iter()
                    .map(|y| singlet_part(y, k))
                    .collect();
                for a in 0..3 {
                    for b in 0..3 {
                        let mut triplet = 0.0;
                        for s in 0..3 {
                            triplet += 0.5
                                * triplets[3 * s + a].frobenius_dot(&sites[n].triplet[3 * s + b]);
                        }
                        dipolar[k][a][b] = triplet - fermi[k][a][b];
                        para[k][a][b] = -alpha2 * singlets[a].frobenius_dot(&sites[n].orbital[b]);
                    }
                }
            }
            if !fermi_contact_only {
                let rm = mol.atoms[sites[m].atom].position.to_array();
                let rn = mol.atoms[sites[n].atom].position.to_array();
                for (i, point) in quadrature.points.iter().enumerate() {
                    let Some(kernel) = diamagnetic_spin_orbit_kernel(point.to_array(), rm, rn)
                    else {
                        continue;
                    };
                    let base = alpha2 * alpha2 * quadrature.weights[i];
                    for (tensor, density) in dia.iter_mut().zip(&orbital_density) {
                        for a in 0..3 {
                            for b in 0..3 {
                                tensor[a][b] += base * density[i] * kernel[a][b];
                            }
                        }
                    }
                }
            }
            let totals = [
                (&dia, &out.diamagnetic[pair]),
                (&para, &out.paramagnetic[pair]),
                (&fermi, &out.fermi_contact[pair]),
                (&dipolar, &out.spin_dipole[pair]),
            ];
            for (orbital_parts, total) in totals {
                for a in 0..3 {
                    for b in 0..3 {
                        let sum: f64 = orbital_parts.iter().map(|x| x[a][b]).sum();
                        parts.decomposition_residual =
                            parts.decomposition_residual.max((sum - total[a][b]).abs());
                    }
                }
            }
            parts.diamagnetic.push(dia);
            parts.paramagnetic.push(para);
            parts.fermi_contact.push(fermi);
            parts.spin_dipole.push(dipolar);
            pair += 1;
        }
    }
    Ok((out, Some(parts)))
}
