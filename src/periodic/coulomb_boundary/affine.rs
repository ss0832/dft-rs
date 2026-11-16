// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Same-gauge affine Coulomb derivative at fixed reduced k and AO density.
use super::*;
use crate::periodic::gaussian_coulomb::contract_integral_lattice;

pub struct PeriodicCoulombStrain {
    /// dE/d epsilon_ab in Hartree, indexed [3*a+b].
    pub source: [f64; 9],
    pub electron_nuclear: [f64; 9],
    pub hartree: [f64; 9],
    pub total: [f64; 9],
    pub quadrature_defect: f64,
    pub reference_energy_defect: f64,
    pub converged: bool,
}

struct Contributions {
    strain: [[[f64; 3]; 9]; 3],
    energy: [f64; 3],
}

fn affine_integral(
    kernel: &DimensionalCoulomb,
    gs: &[CartesianGaussian],
    source: Option<&GaussianNucleus>,
    source_moves: bool,
) -> Result<([f64; 3], [[f64; 3]; 9])> {
    let order = 1 + gs.iter().flat_map(|g| g.angular).sum::<usize>();
    let tensor = prepare_integral_tensor(kernel, gs, source, order)?;
    let (values, gradients) = contract_integral_tensor(&tensor, gs, source.is_some())?;
    let mut strain = contract_integral_lattice(kernel, gs, source, order)?;
    for (i, g) in gradients.iter().enumerate() {
        let center = if i < gs.len() {
            gs[i].center
        } else if source_moves {
            source
                .expect("nuclear derivative requires its source")
                .center
        } else {
            Vec3::zero()
        };
        for a in 0..3 {
            for b in 0..3 {
                // Only the sum of Green components has the existing centre
                // derivative. Keep its virial separately in the real slot;
                // raw lattice components themselves retain their exact split.
                strain[3 * a + b][0] += g.get(a) * center.get(b);
            }
        }
    }
    Ok((values, strain))
}

fn add_strain(target: &mut [[f64; 3]; 9], value: &[[f64; 3]; 9], weight: f64) {
    for (row, source) in target.iter_mut().zip(value) {
        for (a, b) in row.iter_mut().zip(source) {
            *a += weight * b;
        }
    }
}

impl DimensionalCoulombBackend {
    fn strain_at(&self, density: &[BlochMatrix], refinement: usize) -> Result<Contributions> {
        let kernel = &self.kernels[refinement];
        let natoms = self.centres
            - usize::from(matches!(
                self.compensation,
                ChargeCompensation::GaussianCountercharge { .. }
            ));
        let (pairs, _) = self.density_pairs(density)?;
        let mut result = Contributions {
            strain: [[[0.0; 3]; 9]; 3],
            energy: [0.0; 3],
        };
        for (a, owner_a) in &self.sources {
            for (b, owner_b) in &self.sources {
                let r = a.center - b.center;
                let width = a.width.hypot(b.width);
                let own = width == 0.0 && std::ptr::eq(a, b);
                let (value, gradient) = kernel.pair(r, width, own)?;
                let mut cell = if width > 0.0 {
                    crate::periodic::gaussian_coulomb::lattice_green::fixed_displacement_cell_response(
                        kernel, &kernel.reciprocal, r, width, 0,
                    )?.components([0; 3])?
                } else {
                    super::point_lattice::fixed_point_cell_response(
                        kernel,
                        &kernel.reciprocal,
                        r,
                        own,
                    )?
                };
                let moving = (if *owner_a < natoms {
                    a.center
                } else {
                    Vec3::zero()
                }) - (if *owner_b < natoms {
                    b.center
                } else {
                    Vec3::zero()
                });
                for i in 0..3 {
                    for j in 0..3 {
                        cell[3 * i + j][0] += gradient.get(i) * moving.get(j);
                    }
                }
                let weight = 0.5 * a.charge * b.charge;
                result.energy[0] += weight * value;
                add_strain(&mut result.strain[0], &cell, weight);
            }
        }
        let angular = |v: (i32, i32, i32)| [v.0 as usize, v.1 as usize, v.2 as usize];
        for (a, b, pa) in &pairs {
            let ca = cart_components(a.l);
            let cb = cart_components(b.l);
            for (source, owner) in &self.sources {
                for (&ea, &wa) in a.exps.iter().zip(&a.coefs) {
                    for (&eb, &wb) in b.exps.iter().zip(&b.coefs) {
                        let mut gs = [
                            CartesianGaussian {
                                center: a.center,
                                exponent: ea,
                                angular: [0; 3],
                            },
                            CartesianGaussian {
                                center: b.center,
                                exponent: eb,
                                angular: [0; 3],
                            },
                        ];
                        for (i, &ia) in ca.iter().enumerate() {
                            for (j, &ib) in cb.iter().enumerate() {
                                let weight = -source.charge * wa * wb * pa.re[(i, j)];
                                if weight == 0.0 {
                                    continue;
                                }
                                gs[0].angular = angular(ia);
                                gs[1].angular = angular(ib);
                                let (values, response) =
                                    affine_integral(kernel, &gs, Some(source), *owner < natoms)?;
                                result.energy[1] += weight * values.iter().sum::<f64>();
                                add_strain(&mut result.strain[1], &response, weight);
                            }
                        }
                    }
                }
            }
            for (c, d, pb) in &pairs {
                let cc = cart_components(c.l);
                let cd = cart_components(d.l);
                for (&ea, &wa) in a.exps.iter().zip(&a.coefs) {
                    for (&eb, &wb) in b.exps.iter().zip(&b.coefs) {
                        for (&ec, &wc) in c.exps.iter().zip(&c.coefs) {
                            for (&ed, &wd) in d.exps.iter().zip(&d.coefs) {
                                let mut gs = [
                                    CartesianGaussian {
                                        center: a.center,
                                        exponent: ea,
                                        angular: [0; 3],
                                    },
                                    CartesianGaussian {
                                        center: b.center,
                                        exponent: eb,
                                        angular: [0; 3],
                                    },
                                    CartesianGaussian {
                                        center: c.center,
                                        exponent: ec,
                                        angular: [0; 3],
                                    },
                                    CartesianGaussian {
                                        center: d.center,
                                        exponent: ed,
                                        angular: [0; 3],
                                    },
                                ];
                                for (i, &ia) in ca.iter().enumerate() {
                                    for (j, &ib) in cb.iter().enumerate() {
                                        gs[0].angular = angular(ia);
                                        gs[1].angular = angular(ib);
                                        for (k, &ic) in cc.iter().enumerate() {
                                            for (l, &id) in cd.iter().enumerate() {
                                                // Same complex trace as the energy backend; imaginary
                                                // inverse-Bloch density coefficients are not dropped.
                                                let coefficient = pa.re[(i, j)] * pb.re[(k, l)]
                                                    + pa.im[(i, j)] * pb.im[(k, l)];
                                                let weight = 0.5 * wa * wb * wc * wd * coefficient;
                                                if weight == 0.0 {
                                                    continue;
                                                }
                                                gs[2].angular = angular(ic);
                                                gs[3].angular = angular(id);
                                                let (values, response) =
                                                    affine_integral(kernel, &gs, None, false)?;
                                                result.energy[2] +=
                                                    weight * values.iter().sum::<f64>();
                                                add_strain(
                                                    &mut result.strain[2],
                                                    &response,
                                                    weight,
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(result)
    }

    /// Differentiate the actual dimensional Coulomb energy, with physical
    /// atomic centres and lattice images affine, and reduced k fixed.
    /// Fixed Cartesian countercharge centres remain fixed. Widths, Ewald
    /// splitting and cutoff controls are held fixed; converge the truncations
    /// independently. No SCF stationarity or Pulay derivative is inferred.
    pub fn affine_strain(&self, density: &[BlochMatrix]) -> Result<PeriodicCoulombStrain> {
        for pair in &self.pairs {
            let (a, b) = self.pair_shells(pair);
            if (a.center - b.center).norm() == self.basis_cutoff {
                return Err(DftError::InvalidInput(
                    "Coulomb strain is undefined at an exact hard AO-pair cutoff".into(),
                ));
            }
        }
        let reference = self.evaluate(Some(density))?;
        let coarse = self.strain_at(density, 0)?;
        let fine = self.strain_at(density, 1)?;
        let old = [
            reference.source_energy,
            reference
                .electron_nuclear_components
                .expect("density supplied")
                .iter()
                .sum(),
            reference
                .hartree_energy_components
                .expect("density supplied")
                .iter()
                .sum(),
        ];
        let energy_defect = fine
            .energy
            .iter()
            .zip(old)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let defect = coarse
            .strain
            .iter()
            .flatten()
            .flatten()
            .zip(fine.strain.iter().flatten().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        if fine
            .strain
            .iter()
            .flatten()
            .flatten()
            .any(|v| !v.is_finite())
            || !energy_defect.is_finite()
            || !defect.is_finite()
        {
            return Err(DftError::InvalidInput(
                "Coulomb strain exceeds finite numerical range".into(),
            ));
        }
        let total = std::array::from_fn(|c| {
            fine.strain
                .iter()
                .map(|part| part[c].iter().sum::<f64>())
                .sum()
        });
        let [source, electron_nuclear, hartree] = fine
            .strain
            .map(|term| term.map(|parts| parts.iter().sum::<f64>()));
        Ok(PeriodicCoulombStrain {
            source,
            electron_nuclear,
            hartree,
            total,
            quadrature_defect: defect,
            reference_energy_defect: energy_defect,
            converged: reference.converged
                && defect <= self.tolerance
                && energy_defect <= self.tolerance,
        })
    }
}

#[cfg(test)]
mod lower_dimensional_tests;
