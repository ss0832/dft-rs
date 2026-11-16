// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic stress of the finite-k periodic SCF energy.
//!
//! The strain derivative `dE/d(epsilon_ab)` is taken with the atoms and the
//! lattice strained together at fixed fractional coordinates and at fixed
//! fractional k points. At a converged SCF the energy is stationary in the
//! density, so only the explicit dependence at fixed density images `D(T)`
//! enters, together with the Pulay term `-sum_T W(T) . dS(T)/d(epsilon)` of
//! the moving non-orthogonal basis. Each energy term contributes:
//!
//! * kinetic and overlap: centre virial of every image pair;
//! * nuclear attraction: screened real-space charges, reciprocal half
//!   (centres, wave vectors, `1/V` and `1/G^2` kernels) and the overlap
//!   background of the Ewald split;
//! * Hartree: the range-separated screened build of the SCF, differentiated
//!   in the same pieces;
//! * exact exchange: centre virial of the translated quartets;
//! * exchange-correlation: rigidly moving Bloch AOs, grid points riding
//!   their owner atoms and the periodic Becke weights;
//! * nuclear repulsion: the Ewald strain derivative, plus the
//!   charged-cell monopole correction of a charged cell.
//!
//! The stress is `sigma_ab = (1/V) dE/d(epsilon_ab)` in Hartree/Bohr^3; a
//! negative trace means the energy decreases under expansion at this
//! reference (positive pressure is `-tr(sigma)/3`).

use crate::error::Result;
use crate::integrals::CoulombKernel;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::integrals::{contract_ecp_geometry, EcpBoundary, Operator};
use crate::periodic::scf::{PeriodicScfOptions, PeriodicScfResult};
use crate::periodic::stationary_density::StationaryDensity;
use crate::periodic::strain::{add_tensor, nuclear_attraction_strain, two_centre_strain, Tensor};
use crate::periodic::Cell;
use crate::xc::ResolvedFunctional;

/// Analytic strain derivative and stress of a converged periodic SCF.
#[derive(Clone, Debug)]
pub struct PeriodicStress {
    /// `dE/d(epsilon_ab)` per cell (Hartree), `[a][b]`.
    pub strain_derivative: Tensor,
    /// Symmetric stress `sigma = (1/V) (dE/d(epsilon) + dE/d(epsilon)^T)/2`
    /// (Hartree/Bohr^3).
    pub stress: Tensor,
    /// `max |dE/d(epsilon_ab) - dE/d(epsilon_ba)|` (Hartree): a rigid
    /// rotation leaves the energy unchanged, so this measures the residual
    /// of rotational invariance of the truncated sums.
    pub rotation_residual: f64,
    /// Cell volume (Bohr^3).
    pub volume: f64,
    /// Strain derivative of each energy term (Hartree per cell).
    pub terms: Vec<(&'static str, Tensor)>,
}

impl PeriodicStress {
    pub(crate) fn from_terms(volume: f64, terms: Vec<(&'static str, Tensor)>) -> Self {
        let mut strain_derivative = [[0.0; 3]; 3];
        for (_, tensor) in &terms {
            add_tensor(&mut strain_derivative, tensor);
        }
        let mut stress = [[0.0; 3]; 3];
        let mut rotation_residual = 0.0_f64;
        for a in 0..3 {
            for b in 0..3 {
                stress[a][b] = 0.5 * (strain_derivative[a][b] + strain_derivative[b][a]) / volume;
                rotation_residual = rotation_residual
                    .max((strain_derivative[a][b] - strain_derivative[b][a]).abs());
            }
        }
        Self {
            strain_derivative,
            stress,
            rotation_residual,
            volume,
            terms,
        }
    }
}

fn k_images(
    cell: &Cell,
    k_points: &[crate::math::Vec3],
    matrices: &[BlochMatrix],
    cutoff: f64,
) -> Result<Vec<crate::periodic::integrals::ImageBlock>> {
    cell.density_images_from_k(
        k_points,
        &matrices
            .iter()
            .map(|p| (p.re.clone(), p.im.clone()))
            .collect::<Vec<_>>(),
        cutoff,
    )
}

fn scaled(tensor: Tensor, factor: f64) -> Tensor {
    let mut out = tensor;
    for row in &mut out {
        for value in row {
            *value *= factor;
        }
    }
    out
}

/// Analytic strain derivative and stress of a converged finite-k SCF.
pub fn analytic_stress(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &PeriodicScfResult,
    options: &PeriodicScfOptions,
) -> Result<PeriodicStress> {
    result.validate_derivative_plan(cell, functional, options)?;
    analytic_stress_from_density(
        cell,
        functional,
        &StationaryDensity::from_scf(result),
        options,
    )
}

pub(crate) fn analytic_stress_from_density(
    cell: &Cell,
    functional: &ResolvedFunctional,
    result: &StationaryDensity<'_>,
    options: &PeriodicScfOptions,
) -> Result<PeriodicStress> {
    let nk = result.k_points.len();
    let total = result.total_density();
    let weighted: Vec<BlochMatrix> = (0..nk)
        .map(|ik| {
            let mut sum = BlochMatrix::zeros(cell.nao(), cell.nao());
            for spin in 0..2 {
                sum.add_scaled(&result.energy_weighted_density[spin][ik], 1.0);
            }
            sum
        })
        .collect();
    let exchange_cutoff = options.exchange_cutoff.unwrap_or(cell.overlap_cutoff);
    let interaction_cutoff = cell.hartree_real_space_cutoff().max(exchange_cutoff);
    let total_images = k_images(cell, result.k_points, &total, 2.0 * interaction_cutoff)?;
    let weighted_images = k_images(cell, result.k_points, &weighted, 2.0 * interaction_cutoff)?;

    let mut terms: Vec<(&'static str, Tensor)> = Vec::new();
    if !cell.ecp_image_blocks().is_empty() {
        let ecp = contract_ecp_geometry(
            &cell.basis,
            EcpBoundary::Bulk(&cell.lattice),
            cell.overlap_cutoff,
            cell.molecule.len(),
            cell.ecp_image_blocks(),
            result.k_points,
            &total,
        )?;
        terms.push(("ECP", ecp.strain));
    }
    terms.push((
        "kinetic",
        crate::logger::timed("stress: kinetic", || {
            two_centre_strain(
                &cell.basis,
                &cell.lattice,
                cell.overlap_cutoff,
                Operator::Kinetic,
                &total_images,
            )
        }),
    ));
    terms.push((
        "overlap (Pulay)",
        crate::logger::timed("stress: overlap", || {
            scaled(
                two_centre_strain(
                    &cell.basis,
                    &cell.lattice,
                    cell.overlap_cutoff,
                    Operator::Overlap,
                    &weighted_images,
                ),
                -1.0,
            )
        }),
    ));
    let charges = cell.basis.effective_charges(&cell.molecule);
    terms.push((
        "nuclear attraction",
        crate::logger::timed("stress: nuclear attraction", || {
            nuclear_attraction_strain(
                &cell.basis,
                &cell.molecule,
                &charges,
                &cell.lattice,
                cell.overlap_cutoff,
                cell.ewald_params(),
                &total_images,
            )
        }),
    ));
    terms.push((
        "Hartree",
        crate::logger::timed("stress: Hartree", || {
            crate::periodic::screened_coulomb::screened_hartree_strain(
                &cell.basis,
                &cell.lattice,
                &total_images,
                &cell.hartree_screening(),
            )
        })?
        .strain_derivative,
    ));
    if functional.hybrid.cx != 0.0 || functional.hybrid.clr != 0.0 {
        let exchange_spins = if result.distinct_spin_channels { 2 } else { 1 };
        let restricted_factor = if result.distinct_spin_channels {
            1.0
        } else {
            2.0
        };
        let mut exchange = [[0.0; 3]; 3];
        for spin in 0..exchange_spins {
            let spin_images = crate::periodic::exchange::minimum_image_blocks(
                cell,
                result.k_points,
                k_images(
                    cell,
                    result.k_points,
                    &result.density[spin],
                    2.0 * exchange_cutoff,
                )?,
            );
            for (coefficient, kernel) in [
                (functional.hybrid.cx, CoulombKernel::Coulomb),
                (
                    functional.hybrid.clr,
                    CoulombKernel::Erf {
                        omega: functional.hybrid.omega,
                    },
                ),
            ] {
                if coefficient == 0.0 {
                    continue;
                }
                let (_, virial) = crate::periodic::gradient::real_two_electron_gradient(
                    cell,
                    &spin_images,
                    &spin_images,
                    exchange_cutoff,
                    kernel,
                    -0.5 * restricted_factor * coefficient,
                );
                add_tensor(&mut exchange, &virial);
            }
        }
        terms.push(("exact exchange", exchange));
    }
    if !functional.parts.is_empty() || options.skala.is_some() || options.vv10.is_some() {
        terms.push((
            "exchange-correlation",
            crate::logger::timed("stress: exchange-correlation", || {
                crate::periodic::xc_gradient::xc_strain_derivative_from_density(
                    cell, functional, result, options,
                )
            })?,
        ));
    }
    let positions: Vec<crate::math::Vec3> =
        cell.molecule.atoms.iter().map(|a| a.position).collect();
    let mut nuclear = crate::periodic::ewald::ewald_stress(
        &cell.lattice,
        &positions,
        &charges,
        cell.ewald_params(),
    );
    add_tensor(
        &mut nuclear,
        &crate::periodic::ewald::monopole_strain(&cell.lattice, cell.net_charge()),
    );
    terms.push(("nuclear repulsion", nuclear));

    Ok(PeriodicStress::from_terms(
        cell.lattice.volume().abs(),
        terms,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::MolBasis;
    use crate::grid::GridOptions;
    use crate::linalg::Matrix;
    use crate::math::Vec3;
    use crate::periodic::ewald::EwaldParams;
    use crate::periodic::integrals::{image_blocks, ImageBlock};
    use crate::periodic::xc::{xc_contribution, AoDerivativeOrder, PeriodicXcGrid};
    use crate::periodic::Lattice;
    use crate::system::{Atom, Molecule};
    use crate::xc::{resolve, FunctionalSpec};

    /// One s and one p shell per hydrogen, each contracted from a compact
    /// and a diffuse primitive, so that every branch of the split Hartree
    /// build and every angular case of the pair derivatives is exercised.
    const BASIS: &str = "H 0\nS 2 1.00\n 2.4 0.6\n 0.55 0.5\nP 2 1.00\n 1.6 0.6\n 0.5 0.5\n****\n";
    const STEPS: [f64; 4] = [4e-3, 2e-3, 1e-3, 5e-4];
    /// A diagonal, an off-diagonal shear and a component with the opposite
    /// index order (strains are not symmetrized).
    const COMPONENTS: [(usize, usize); 3] = [(0, 0), (0, 1), (2, 1)];
    const CUTOFF: f64 = 11.0;

    fn deform(v: Vec3, eps: &Tensor) -> Vec3 {
        Vec3::new(
            v.x + eps[0][0] * v.x + eps[0][1] * v.y + eps[0][2] * v.z,
            v.y + eps[1][0] * v.x + eps[1][1] * v.y + eps[1][2] * v.z,
            v.z + eps[2][0] * v.x + eps[2][1] * v.y + eps[2][2] * v.z,
        )
    }

    fn strain(a: usize, b: usize, h: f64) -> Tensor {
        let mut e = [[0.0; 3]; 3];
        e[a][b] = h;
        e
    }

    fn set() -> crate::basis::BasisSet {
        crate::basis::g94::parse("strain-test", BASIS).unwrap()
    }

    fn system(eps: &Tensor) -> (Molecule, Lattice, MolBasis) {
        system_with_lattice_scale(eps, 1.0)
    }

    fn system_with_lattice_scale(eps: &Tensor, scale: f64) -> (Molecule, Lattice, MolBasis) {
        let base = [
            Vec3::new(4.2, 0.0, 0.0) * scale,
            Vec3::new(0.3, 4.6, 0.0) * scale,
            Vec3::new(0.2, -0.25, 4.9) * scale,
        ];
        let lattice = Lattice::new(
            deform(base[0], eps),
            deform(base[1], eps),
            deform(base[2], eps),
        );
        let molecule = Molecule::new(
            [Vec3::new(0.1, 0.2, -0.3), Vec3::new(1.2, 0.9, 0.8)]
                .iter()
                .map(|p| Atom {
                    z: 1,
                    position: deform(*p, eps),
                })
                .collect(),
        );
        let basis = MolBasis::build(&molecule, &set()).unwrap();
        (molecule, lattice, basis)
    }

    /// Density images on integer translations of `lattice`, with
    /// `D(-n) = D(n)^T` and a decay with `|n|`. Strain moves the
    /// translations and keeps the matrices.
    fn images(lattice: &Lattice, nao: usize) -> Vec<ImageBlock> {
        let entry = |n: [i64; 3], p: usize, q: usize| -> f64 {
            let s = (n[0] * 7 + n[1] * 13 + n[2] * 29) as f64 + (p * 3 + q * 5) as f64 * 0.37;
            (s * 1.37).sin()
        };
        let mut out = Vec::new();
        for i in -2..=2i64 {
            for j in -2..=2i64 {
                for k in -2..=2i64 {
                    let r2 = (i * i + j * j + k * k) as f64;
                    if r2 > 6.0 {
                        continue;
                    }
                    let decay = (-0.7 * r2).exp();
                    let mut matrix = Matrix::zeros(nao, nao);
                    for p in 0..nao {
                        for q in 0..nao {
                            matrix[(p, q)] =
                                decay * 0.25 * (entry([i, j, k], p, q) + entry([-i, -j, -k], q, p));
                            if r2 == 0.0 && p == q {
                                matrix[(p, q)] += 0.6;
                            }
                        }
                    }
                    out.push(ImageBlock {
                        translation: lattice.translation([i, j, k]),
                        matrix,
                    });
                }
            }
        }
        out
    }

    fn contract_images(density: &[ImageBlock], blocks: &[ImageBlock]) -> f64 {
        blocks
            .iter()
            .filter_map(|b| {
                crate::periodic::strain::find_block(density, b.translation)
                    .map(|d| d.frobenius_dot(&b.matrix))
            })
            .sum()
    }

    /// Fit `log|err| = log C + p log h`.
    fn power_fit(steps: &[f64], errors: &[f64]) -> (f64, f64) {
        let (x, y): (Vec<f64>, Vec<f64>) = steps
            .iter()
            .zip(errors)
            .map(|(h, e)| (h.ln(), e.ln()))
            .unzip();
        let n = x.len() as f64;
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
        let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
        let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
        (sxy / sxx, sxy * sxy / (sxx * syy))
    }

    /// Central differences of `energy(eps)` for every component and step,
    /// fitted against the analytic tensor.
    fn check(label: &str, analytic: &Tensor, energy: impl Fn(&Tensor) -> f64) {
        check_with_steps(label, analytic, &STEPS, energy)
    }

    fn check_with_steps(
        label: &str,
        analytic: &Tensor,
        steps: &[f64],
        energy: impl Fn(&Tensor) -> f64,
    ) {
        for &(a, b) in &COMPONENTS {
            let exact = analytic[a][b];
            let errors: Vec<f64> = steps
                .iter()
                .map(|&h| {
                    ((energy(&strain(a, b, h)) - energy(&strain(a, b, -h))) / (2.0 * h) - exact)
                        .abs()
                })
                .collect();
            let (p, r2) = power_fit(steps, &errors);
            println!(
                "{label} dE/de[{a}][{b}] = {exact:.12e}: steps {steps:?} errors {errors:?} p = {p:.6} R2 = {r2:.8}"
            );
            assert!(
                errors.iter().all(|e| e.is_finite() && *e > 0.0),
                "{label}: {errors:?}"
            );
            assert!(
                (p - 2.0).abs() < 0.05 && r2 > 0.9999,
                "{label} [{a}][{b}]: p = {p}, R2 = {r2}"
            );
            assert!(
                errors[3] < 1e-5 * exact.abs().max(1.0),
                "{label} [{a}][{b}]: {errors:?}"
            );
        }
    }

    #[test]
    fn kinetic_and_overlap_strain_pass_four_step_fits() {
        let (_, lattice, basis) = system(&[[0.0; 3]; 3]);
        let d = images(&lattice, basis.nao_sph);
        for (label, op) in [
            ("kinetic", Operator::Kinetic),
            ("overlap", Operator::Overlap),
        ] {
            let analytic = two_centre_strain(&basis, &lattice, CUTOFF, op, &d);
            check(label, &analytic, |eps| {
                let (_, lattice, basis) = system(eps);
                contract_images(
                    &images(&lattice, basis.nao_sph),
                    &image_blocks(&basis, &lattice, CUTOFF, op),
                )
            });
        }
    }

    #[test]
    fn nuclear_attraction_strain_passes_four_step_fits() {
        let (molecule, lattice, basis) = system(&[[0.0; 3]; 3]);
        let d = images(&lattice, basis.nao_sph);
        let charges = vec![1.0, 1.0];
        // One Ewald split for every strain: each piece is differentiated at
        // fixed alpha, and the oracle holds it fixed too.
        let p = EwaldParams::for_accuracy(&lattice, 2, 1e-12);
        let analytic =
            nuclear_attraction_strain(&basis, &molecule, &charges, &lattice, CUTOFF, p, &d);
        check("nuclear attraction", &analytic, |eps| {
            let (molecule, lattice, basis) = system(eps);
            let blocks = crate::periodic::nuclear::nuclear_image_blocks(
                &basis, &molecule, &charges, &lattice, CUTOFF, p,
            );
            contract_images(&images(&lattice, basis.nao_sph), &blocks)
        });
    }

    #[test]
    fn screened_hartree_strain_passes_four_step_fits() {
        use crate::periodic::screened_coulomb::{
            screened_hartree_image_blocks, screened_hartree_strain, HartreeScreening,
        };
        // Splits on both sides of the primitive exponents.
        for compact_exponent in [1.0, 3.0] {
            let screening = HartreeScreening {
                compact_exponent,
                tolerance: 1e-15,
                max_translation: None,
            };
            let (_, lattice, basis) = system(&[[0.0; 3]; 3]);
            let d = images(&lattice, basis.nao_sph);
            let strained = screened_hartree_strain(&basis, &lattice, &d, &screening).unwrap();
            let blocks = screened_hartree_image_blocks(&basis, &lattice, &d, &screening).unwrap();
            let energy = 0.5 * contract_images(&d, &blocks);
            assert!(
                (energy - strained.energy).abs() < 1e-10,
                "{energy} vs {}",
                strained.energy
            );
            check(
                &format!("Hartree (split {compact_exponent})"),
                &strained.strain_derivative,
                |eps| {
                    let (_, lattice, basis) = system(eps);
                    let d = images(&lattice, basis.nao_sph);
                    0.5 * contract_images(
                        &d,
                        &screened_hartree_image_blocks(&basis, &lattice, &d, &screening).unwrap(),
                    )
                },
            );
        }
    }

    #[test]
    fn exact_exchange_strain_passes_four_step_fits() {
        let cutoff = 5.0;
        let (molecule, lattice, _) = system(&[[0.0; 3]; 3]);
        let cell = Cell::new(molecule, lattice.clone(), &set(), cutoff).unwrap();
        let d = images(&lattice, cell.nao());
        let (_, analytic) = crate::periodic::gradient::real_two_electron_gradient(
            &cell,
            &d,
            &d,
            cutoff,
            CoulombKernel::Coulomb,
            -0.5,
        );
        check("exact exchange", &analytic, |eps| {
            let (_, lattice, basis) = system(eps);
            let d = images(&lattice, basis.nao_sph);
            let blocks = crate::periodic::exchange::exchange_image_blocks(
                &basis,
                &lattice,
                &d,
                cutoff,
                CoulombKernel::Coulomb,
            );
            -0.5 * contract_images(&d, &blocks)
        });
    }

    fn fixed_density_result(cell: &Cell, k_points: Vec<Vec3>) -> PeriodicScfResult {
        let n = cell.nao();
        // Diagonally dominant, hence positive definite: rho >= 0 everywhere.
        let mut re = Matrix::identity(n);
        for p in 0..n {
            re[(p, p)] = 0.32 + 0.03 * p as f64;
            for q in 0..n {
                if p != q {
                    re[(p, q)] = 0.012 * ((p * 5 + q * 5) as f64 * 0.7).cos();
                }
            }
        }
        let density: [Vec<BlochMatrix>; 2] = std::array::from_fn(|_| {
            k_points
                .iter()
                .map(|_| BlochMatrix {
                    re: re.clone(),
                    im: Matrix::zeros(n, n),
                })
                .collect()
        });
        let zero = || {
            k_points
                .iter()
                .map(|_| BlochMatrix::zeros(n, n))
                .collect::<Vec<_>>()
        };
        PeriodicScfResult {
            orbital_conjugation_constraint: true,
            energy: 0.0,
            free_energy: 0.0,
            entropy_term: 0.0,
            e_nuc: 0.0,
            e_elec: 0.0,
            e_xc: 0.0,
            e_nlc: 0.0,
            k_points: k_points.clone(),
            mo_energy: std::array::from_fn(|_| Vec::new()),
            occupations: std::array::from_fn(|_| Vec::new()),
            chemical_potential: [None, None],
            density,
            energy_weighted_density: std::array::from_fn(|_| zero()),
            fock: std::array::from_fn(|_| zero()),
            overlap: zero(),
            h_core: zero(),
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

    /// The periodic Becke denominator counts the atom images inside a sphere
    /// around each point whose radius scales with `(V/N)^(1/3)`, and points
    /// and AO images are kept inside fixed radii. Under strain, images and
    /// points cross these spheres and change the quadrature
    /// discontinuously. Recorded oracle failures (SVWN5, dE/de[0][0]):
    /// in the compact 4.2-4.9 Bohr cell the steps 4e-3..5e-4 gave errors
    /// 1.76e-4, 8.4e-5, 1.47e-4, 1.97e-4 (p = -0.13, R2 = 0.095, an energy
    /// noise near 1e-7 Hartree); dilated by 1.6 they gave 1.9e-5, 7.9e-6,
    /// 5.2e-6, 5.8e-6 (p = 0.58, R2 = 0.77), and steps 1.6e-2..2.5e-4 changed
    /// sign. Dilated by 2.2, the errors for 1.6e-2, 8e-3, 4e-3 are 1.94e-5,
    /// 4.67e-6, 1.14e-6 (ratios 4.16, 4.08) and below 2e-3 they reach the
    /// 1e-10 Hartree discontinuity floor. With one Becke shell the images
    /// that cross the sphere also bias the central difference by a
    /// strain-independent staircase slope: on 6.4e-2..8e-3 the shear
    /// dE/de[0][1] errors 8.14e-5, 2.0e-5, 4.49e-6, 9.17e-7 have ratios
    /// 4.07, 4.46, 4.89 (p = 2.157, R2 = 0.99925), an offset near 5e-7.
    /// Three Becke shells move the crossing images beyond any point that
    /// carries density; the acceptance fit uses them with the 2.2 dilation
    /// on 6.4e-2..8e-3, where every component and class has ratios near 4.
    #[test]
    fn exchange_correlation_strain_passes_four_step_fits() {
        let options = PeriodicScfOptions {
            k_mesh: [2, 1, 1],
            xc_ao_cutoff: Some(9.0),
            grid_shell: 3,
            grid: GridOptions {
                radial_override: Some(12),
                angular_override: Some(26),
                prune: false,
                weight_cutoff: 0.0,
                ..GridOptions::default()
            },
            ..PeriodicScfOptions::default()
        };
        let scale = 2.2;
        let steps = [6.4e-2, 3.2e-2, 1.6e-2, 8e-3];
        for name in ["svwn5", "pbe", "r2scan"] {
            let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
            let (molecule, lattice, _) = system_with_lattice_scale(&[[0.0; 3]; 3], scale);
            let cell = Cell::new(molecule, lattice, &set(), 9.0).unwrap();
            let result = fixed_density_result(&cell, cell.k_points(options.k_mesh));
            let analytic = crate::periodic::xc_gradient::xc_strain_derivative(
                &cell,
                &functional,
                &result,
                &options,
            )
            .unwrap();
            check_with_steps(&format!("XC {name}"), &analytic, &steps, |eps| {
                let (molecule, lattice, _) = system_with_lattice_scale(eps, scale);
                let strained = Cell::new(molecule, lattice, &set(), 9.0).unwrap();
                let grid = PeriodicXcGrid::build(
                    &strained,
                    &strained.k_points(options.k_mesh),
                    &options.grid,
                    options.grid_shell,
                    9.0,
                    AoDerivativeOrder::for_scf(functional.family),
                )
                .unwrap();
                xc_contribution(&functional, &grid, &result.density)
                    .unwrap()
                    .exc
            });
        }
    }
}
