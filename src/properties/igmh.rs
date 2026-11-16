// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Molecular IGMH scalar fields from a spin-summed AO density.

use crate::basis::MolBasis;
use crate::dispersion::hirshfeld::Promolecule;
use crate::grid::eval::{eval_aos_full, second_index};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::system::Molecule;
use crate::{DftError, Result};

/// Atomic vector convention for subtracting the norms of summed gradients.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GradientConvention {
    /// w_A grad(rho) - rho grad(w_A), used in the published maps and
    /// explicitly adopted in the authors' erratum.
    #[default]
    Published,
    /// The spatial derivative of rho_A = w_A rho by the product rule.
    Hirshfeld,
}

/// Point values in atomic units. Per-atom fields follow input atom order.
#[derive(Clone, Debug)]
pub struct IgmhPoint {
    pub density: f64,
    pub density_gradient: Vec3,
    /// xx, xy, xz, yy, yz, zz.
    pub density_hessian: [f64; 6],
    pub signed_density: f64,
    pub delta_g: f64,
    pub delta_g_inter: f64,
    pub delta_g_intra: f64,
    pub atom_densities: Vec<f64>,
    pub atom_vectors: Vec<Vec3>,
}

/// Point fields and maximum absolute partition closure defects.
#[derive(Clone, Debug)]
pub struct IgmhAnalysis {
    pub points: Vec<IgmhPoint>,
    pub density_residual: f64,
    pub gradient_residual: f64,
    /// Points where both the molecular field and the neutral proatoms vanish.
    pub vacuum_points: usize,
}

/// Validate the model and assign compact group indices. An absent assignment
/// treats all atoms as a single fragment. Supplied integer labels can be sparse.
pub(crate) fn validate_partition(
    molecule: &Molecule,
    basis: &MolBasis,
    fragments: Option<&[usize]>,
) -> Result<Vec<usize>> {
    if !basis.ecps.is_empty() || molecule.atoms.iter().any(|atom| atom.z == 0) {
        return Err(DftError::Unsupported {
            feature: "molecular IGMH with ECPs or ghost centres".into(),
            missing: "an all-electron atomic density reconstruction".into(),
        });
    }
    let Some(fragments) = fragments else {
        return Ok(vec![0; molecule.len()]);
    };
    if fragments.len() != molecule.len() {
        return Err(DftError::InvalidInput(
            "IGMH fragments must supply one nonnegative label per atom".into(),
        ));
    }
    let mut labels = fragments.to_vec();
    labels.sort_unstable();
    labels.dedup();
    Ok(fragments
        .iter()
        .map(|label| labels.binary_search(label).expect("input label"))
        .collect())
}

fn product_at(left: &Matrix, right: &Matrix, point: usize) -> f64 {
    (0..left.cols)
        .map(|mu| left[(point, mu)] * right[(point, mu)])
        .sum()
}

fn sum_vectors(vectors: &[Vec3]) -> Vec3 {
    vectors.iter().fold(Vec3::zero(), |sum, &value| sum + value)
}

/// Sample the isolated, all-electron molecular fields at explicit Bohr points.
/// The input is the symmetric real part of the spin-summed AO density.
/// Neutral proatoms use spherical HF SAD in the same molecular basis.
pub fn analyze(
    molecule: &Molecule,
    basis: &MolBasis,
    density: &Matrix,
    points: &[Vec3],
    fragments: Option<&[usize]>,
    convention: GradientConvention,
    atom_details: bool,
) -> Result<IgmhAnalysis> {
    let groups = validate_partition(molecule, basis, fragments)?;
    if molecule.is_empty()
        || points.is_empty()
        || points.iter().any(|r| !r.norm2().is_finite())
        || density.rows != basis.nao_sph
        || density.cols != basis.nao_sph
        || density.as_slice().iter().any(|x| !x.is_finite())
    {
        return Err(DftError::InvalidInput(
            "IGMH requires a finite AO density and at least one finite Cartesian point".into(),
        ));
    }
    // Only the symmetric real AO charge density contributes to local scalar fields.
    let mut p = density.clone();
    for u in 0..p.rows {
        for v in 0..u {
            let value = 0.5 * density[(u, v)] + 0.5 * density[(v, u)];
            p[(u, v)] = value;
            p[(v, u)] = value;
        }
    }
    let promolecule = Promolecule::new(molecule, basis)?;
    let mut result = IgmhAnalysis {
        points: Vec::with_capacity(points.len()),
        density_residual: 0.0,
        gradient_residual: 0.0,
        vacuum_points: 0,
    };
    let weight_sign = match convention {
        GradientConvention::Published => -1.0,
        GradientConvention::Hirshfeld => 1.0,
    };
    let group_count = groups.iter().copied().max().unwrap_or(0) + 1;
    let mut atom_densities = vec![0.0; molecule.len()];
    let mut atom_vectors = vec![Vec3::zero(); molecule.len()];
    let mut group_vectors = vec![Vec3::zero(); group_count];
    let mut h = Matrix::zeros(3, 3);
    const BATCH: usize = 256;
    for batch in points.chunks(BATCH) {
        let ao = eval_aos_full(basis, batch, true, true);
        let grad = ao.grad.as_ref().expect("requested AO gradients");
        let hess = ao.hess.as_ref().expect("requested AO Hessians");
        let phip = ao.phi.matmul_seq(&p);
        let gradp = grad.each_ref().map(|d| d.matmul_seq(&p));
        let (proatoms, dproatoms) = promolecule.collocate_gradients(&ao.phi, grad);
        for i in 0..batch.len() {
            let rho = product_at(&ao.phi, &phip, i);
            let [gx, gy, gz] = grad.each_ref().map(|d| 2.0 * product_at(d, &phip, i));
            let drho = Vec3::new(gx, gy, gz);
            let proatom_gradient =
                |a: usize| Vec3::new(dproatoms[0][a][i], dproatoms[1][a][i], dproatoms[2][a][i]);
            let promol: f64 = proatoms.iter().map(|atom| atom[i]).sum();
            let dpromol =
                (0..molecule.len()).fold(Vec3::zero(), |sum, a| sum + proatom_gradient(a));
            let mut packed = [0.0; 6];
            for u in 0..3 {
                for v in u..3 {
                    let index = second_index(u, v);
                    let value = 2.0
                        * (product_at(&hess[index], &phip, i) + product_at(&grad[u], &gradp[v], i));
                    packed[index] = value;
                    h[(u, v)] = value;
                    h[(v, u)] = value;
                }
            }
            if !(rho.is_finite() && drho.norm2().is_finite())
                || packed.iter().any(|x| !x.is_finite())
            {
                return Err(DftError::InvalidInput(
                    "IGMH molecular density field overflowed".into(),
                ));
            }
            if promol > 0.0 {
                group_vectors.fill(Vec3::zero());
                for a in 0..molecule.len() {
                    let w = proatoms[a][i] / promol;
                    // The quotient rule in this form avoids squaring a very
                    // small promolecular density in the molecular tail.
                    let density_times_dw = (proatom_gradient(a) - dpromol * w) / promol * rho;
                    atom_densities[a] = w * rho;
                    atom_vectors[a] = drho * w + density_times_dw * weight_sign;
                    group_vectors[groups[a]] += atom_vectors[a];
                }
            } else if promol == 0.0 && rho == 0.0 && drho == Vec3::zero() && packed == [0.0; 6] {
                atom_densities.fill(0.0);
                atom_vectors.fill(Vec3::zero());
                group_vectors.fill(Vec3::zero());
                result.vacuum_points += 1;
            } else {
                return Err(DftError::InvalidInput(
                    "IGMH neutral proatom density vanished where the molecular field is nonzero"
                        .into(),
                ));
            }
            let gradient_sum = sum_vectors(&atom_vectors);
            let atom_norms: f64 = atom_vectors.iter().map(|v| v.norm()).sum();
            let group_norms: f64 = group_vectors.iter().map(|v| v.norm()).sum();
            let total_norm = gradient_sum.norm();
            if !atom_norms.is_finite() || !group_norms.is_finite() || !total_norm.is_finite() {
                return Err(DftError::InvalidInput(
                    "IGMH partition gradients overflowed".into(),
                ));
            }
            result.density_residual = result
                .density_residual
                .max((atom_densities.iter().sum::<f64>() - rho).abs());
            result.gradient_residual = result.gradient_residual.max((gradient_sum - drho).norm());
            let scale = packed.iter().fold(0.0_f64, |scale, x| scale.max(x.abs()));
            let middle_eigenvalue = if scale == 0.0 {
                0.0
            } else {
                for x in h.as_mut_slice() {
                    *x /= scale;
                }
                symmetric_eigen(&h)?.0[1]
            };
            let signed_density = if middle_eigenvalue < 0.0 {
                -rho
            } else if middle_eigenvalue > 0.0 {
                rho
            } else {
                0.0
            };
            result.points.push(IgmhPoint {
                density: rho,
                density_gradient: drho,
                density_hessian: packed,
                signed_density,
                // Each difference has a nonnegative triangle-inequality limit.
                delta_g: (atom_norms - total_norm).max(0.0),
                delta_g_inter: (group_norms - total_norm).max(0.0),
                delta_g_intra: (atom_norms - group_norms).max(0.0),
                atom_densities: if atom_details {
                    atom_densities.clone()
                } else {
                    Vec::new()
                },
                atom_vectors: if atom_details {
                    atom_vectors.clone()
                } else {
                    Vec::new()
                },
            });
        }
    }
    Ok(result)
}
