// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Relaxed nuclear energy response on continuum finite magnetic cells.
//! Geometry-dependent Fourier vertices and source terms feed the shared
//! complex stationary metric/projector response; differences are not used.
use super::*;
use crate::dual::{Scalar, Taylor};
use crate::integrals::london::ComplexMatrix as Cm;
use crate::response::field_path::{
    adjoint, cadd, complex_trace, stationary_complex_energy_path, ComplexPathBackend,
    ComplexPathReference,
};
use crate::response::ResponseSolverConfig;

/// Energy Taylor coefficients and raw relaxed magnetic nuclear diagnostics.
#[derive(Clone, Debug)]
pub struct MagneticNuclearTaylor {
    pub energy: Vec<f64>,
    /// Density equations actually solved; energy uses the variational 2n+1 rule.
    pub solved_response_order: usize,
    pub response_residuals: Vec<f64>,
    pub reference_defect: f64,
    pub reference_energy_defect: f64,
    pub projector_defects: Vec<f64>,
    pub stationarity_defects: Vec<f64>,
}
fn cm(m: &BlochMatrix) -> Cm {
    Cm {
        re: m.re.clone(),
        im: m.im.clone(),
    }
}
fn bm(m: &Cm) -> BlochMatrix {
    BlochMatrix {
        re: m.re.clone(),
        im: m.im.clone(),
    }
}
fn spin(m: &Cm) -> Cm {
    cm(&spin_diagonal(&bm(m)))
}
fn add_complex(target: &mut Cm, source: &Cm, weight: Complex) {
    for i in 0..target.re.rows {
        for j in 0..target.re.cols {
            target.re[(i, j)] += weight.re * source.re[(i, j)] - weight.im * source.im[(i, j)];
            target.im[(i, j)] += weight.re * source.im[(i, j)] + weight.im * source.re[(i, j)];
        }
    }
}
struct Vertex {
    value: Vec<Cm>,
    coulomb: f64,
    exchange: f64,
}
struct Geometry {
    overlap: Vec<Cm>,
    core: Vec<Cm>,
    source: Vec<f64>,
    vertices: Vec<Vertex>,
    electric: Option<crate::response::berry_field::BerryField>,
    density: Cm,
}
impl Geometry {
    fn interaction(&self, density: &[Cm], order: usize) -> Vec<Cm> {
        let size = self.overlap[0].re.rows;
        let mut out = vec![Cm::zeros(size, size); order + 1];
        for vertex in &self.vertices {
            let conjugates: Vec<_> = vertex.value.iter().map(adjoint).collect();
            let mut charge = vec![Complex::ZERO; order + 1];
            for n in 0..=order {
                for k in 0..=n {
                    let (r, i) = complex_trace(&density[k], &conjugates[n - k]);
                    charge[n] += Complex::new(r, i);
                }
            }
            for n in 0..=order {
                for k in 0..=n {
                    add_complex(
                        &mut out[n],
                        &vertex.value[k],
                        charge[n - k] * vertex.coulomb,
                    );
                    if vertex.exchange != 0.0 {
                        for l in 0..=n - k {
                            let product = vertex.value[k]
                                .matmul_seq(&density[l])
                                .matmul_seq(&conjugates[n - k - l]);
                            cadd(&mut out[n], &product, -vertex.exchange);
                        }
                    }
                }
            }
        }
        out
    }
}
impl ComplexPathBackend for Geometry {
    fn overlap(&self) -> &[Cm] {
        &self.overlap
    }
    fn fock_coefficient(&self, p: &[Vec<Cm>; 2], order: usize) -> Result<[Cm; 2]> {
        let mut f = self.core[order].clone();
        cadd(&mut f, &self.interaction(&p[0], order)[order], 1.0);
        if let Some(field) = &self.electric {
            let path = vec![p[0].iter().map(|m| (m.re.clone(), m.im.clone())).collect()];
            let (_, v) = field.evaluate(&path, order)?;
            cadd(
                &mut f,
                &Cm {
                    re: v[0][order].0.clone(),
                    im: v[0][order].1.clone(),
                },
                1.0,
            );
        }
        Ok([f.clone(), f])
    }
    fn linear_kernel(&self, p: &[Cm; 2]) -> Result<[Cm; 2]> {
        let mut v = self.interaction(&[p[0].clone()], 0).remove(0);
        if let Some(field) = &self.electric {
            let base = vec![vec![(self.density.re.clone(), self.density.im.clone())]];
            let direction = vec![vec![(p[0].re.clone(), p[0].im.clone())]];
            let kernel = field.linear(&base, &direction, 0)?;
            cadd(
                &mut v,
                &Cm {
                    re: kernel[0][0].0.clone(),
                    im: kernel[0][0].1.clone(),
                },
                1.0,
            );
        }
        Ok([v.clone(), v])
    }
    fn energy(&self, p: &[Vec<Cm>; 2], order: usize) -> Result<Vec<f64>> {
        let interaction = self.interaction(&p[0], order);
        let electric = if let Some(field) = &self.electric {
            let path = vec![p[0].iter().map(|m| (m.re.clone(), m.im.clone())).collect()];
            field.evaluate(&path, order)?.0
        } else {
            vec![0.0; order + 1]
        };
        Ok((0..=order)
            .map(|n| {
                self.source[n]
                    + electric[n]
                    + (0..=n)
                        .map(|k| {
                            p[0][k].real_trace_product(&self.core[n - k])
                                + 0.5 * p[0][k].real_trace_product(&interaction[n - k])
                        })
                        .sum::<f64>()
            })
            .collect())
    }
    fn dipole(&self, _: &[Vec<Cm>; 2], order: usize) -> Vec<[f64; 3]> {
        vec![[0.0; 3]; order + 1]
    }
}

impl MagneticDftBackend<'_> {
    fn validate_nuclear_reference(
        &self,
        reference: &crate::periodic::spinor_scf::SpinorScfReport,
    ) -> Result<usize> {
        let n = 2 * self.basis.nao_sph;
        if !reference.converged
            || reference.density.len() != 1
            || reference.fields.fock.len() != 1
            || reference.bands.len() != 1
            || reference.occupations.len() != 1
            || !reference.fields.energy.is_finite()
            || !valid(&reference.density[0], n)
            || !valid(&reference.fields.fock[0], n)
        {
            return Err(DftError::InvalidInput("magnetic nuclear response requires one finite converged spinor reference of matching AO size".into()));
        }
        let band = &reference.bands[0];
        let occupations = &reference.occupations[0];
        if band.coefficients.rows() != n
            || band.coefficients.cols() != band.energies.len()
            || band.energies.len() != occupations.len()
            || band.energies.iter().any(|e| !e.is_finite())
            || band
                .coefficients
                .re
                .as_slice()
                .iter()
                .chain(band.coefficients.im.as_slice())
                .any(|x| !x.is_finite())
            || occupations.iter().any(|f| *f != 0.0 && *f != 1.0)
        {
            return Err(DftError::InvalidInput(
                "magnetic nuclear reference needs finite matching orbitals and integer occupations"
                    .into(),
            ));
        }
        let occupied = occupations.iter().take_while(|f| **f == 1.0).count();
        if occupations[occupied..].iter().any(|f| *f != 0.0)
            || (occupied as f64 - self.electrons).abs() > 1e-10
        {
            return Err(DftError::InvalidInput("magnetic nuclear reference must occupy the lowest contiguous spinor subspace at the backend electron number".into()));
        }
        Ok(occupied)
    }
    fn nuclear_geometry(
        &self,
        direction: &[[f64; 3]],
        order: usize,
        density: &BlochMatrix,
    ) -> Result<Geometry> {
        let fixed_countercharge = usize::from(matches!(
            self.options.electrostatics.compensation,
            ChargeCompensation::GaussianCountercharge { .. }
        ));
        if !matches!(self.cell.geometry.dimension(), 2 | 3)
            || !self.functional.parts.is_empty()
            || self.sources.len() != direction.len() + fixed_countercharge
            || order > 4
        {
            return Err(DftError::Unsupported {
                feature: "moving magnetic nuclear geometry".into(),
                missing:
                    "matching atomic directions and a two/three-dimensional exchange-only reference; moving semilocal geometry providers remain required"
                        .into(),
            });
        }
        let mut overlap = self
            .cell
            .nuclear_operator_taylor(
                &self.basis,
                self.options.basis_cutoff,
                Vec3::zero(),
                MagneticPairOperator::Fourier,
                direction,
                order,
            )?
            .iter()
            .map(cm)
            .collect::<Vec<_>>();
        let kinetic = self
            .cell
            .nuclear_operator_taylor(
                &self.basis,
                self.options.basis_cutoff,
                Vec3::zero(),
                MagneticPairOperator::Kinetic,
                direction,
                order,
            )?
            .iter()
            .map(cm)
            .collect::<Vec<_>>();
        let source_positions: Vec<[Taylor<5>; 3]> = self
            .sources
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let d = direction.get(i).copied().unwrap_or([0.0; 3]);
                std::array::from_fn(|a| Taylor::from_coefficients(&[s.position.get(a), d[a]]))
            })
            .collect();
        let source = self
            .source_green
            .energy_components_scalar(&self.sources, &source_positions, |x| {
                let (s, c) = x.sin_cos();
                (c, s)
            })?
            .total();
        let spatial = self.basis.nao_sph;
        let mut core: Vec<_> = kinetic.iter().map(spin).collect();
        let extra = {
            let mut m = self.spin.clone();
            m.add_scaled(&self.zeeman, -1.0);
            cm(&m)
        };
        cadd(&mut core[0], &extra, 1.0);
        // Zeeman S(t) and generated nuclear SOC retain their moving metric.
        let mut pauli: Vec<[BlochMatrix; 4]> = (0..=order)
            .map(|_| std::array::from_fn(|_| BlochMatrix::zeros(spatial, spatial)))
            .collect();
        for n in 0..=order {
            for a in 0..3 {
                pauli[n][a + 1] = bm(&overlap[n]);
                pauli[n][a + 1].scale(self.cell.field.get(a));
            }
        }
        // Parallel construction, deterministic collection and the original
        // sequential contraction order preserve every physical coefficient.
        let terms = self
            .modes
            .par_iter()
            .map(|mode| -> Result<_> {
                let orbital = self
                    .cell
                    .nuclear_operator_taylor(
                        &self.basis,
                        self.options.basis_cutoff,
                        mode.wave,
                        MagneticPairOperator::Fourier,
                        direction,
                        order,
                    )?
                    .iter()
                    .map(cm)
                    .collect::<Vec<_>>();
                let mut nuclear = (Taylor::<5>::from_f64(0.0), Taylor::<5>::from_f64(0.0));
                for (r, source) in source_positions.iter().zip(&self.sources) {
                    let angle = -(r[0].scale(mode.wave.x)
                        + r[1].scale(mode.wave.y)
                        + r[2].scale(mode.wave.z));
                    let (s, c) = angle.sin_cos();
                    let weight = -mode.coulomb
                        * source.charge
                        * (-0.25 * mode.wave.norm2() * source.width * source.width).exp();
                    nuclear.0 = nuclear.0 + c.scale(weight);
                    nuclear.1 = nuclear.1 + s.scale(weight);
                }
                let momenta = if self.options.nuclear_soc_scale != 0.0 {
                    Some(
                        (0..3)
                            .map(|a| {
                                self.cell
                                    .nuclear_operator_taylor(
                                        &self.basis,
                                        self.options.basis_cutoff,
                                        mode.wave,
                                        MagneticPairOperator::Momentum(a),
                                        direction,
                                        order,
                                    )
                                    .map(|v| v.iter().map(cm).collect::<Vec<_>>())
                            })
                            .collect::<Result<Vec<_>>>()?,
                    )
                } else {
                    None
                };
                Ok((orbital, nuclear, momenta))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut vertices = Vec::with_capacity(self.modes.len());
        for (mode, (orbital, nuclear, momenta)) in self.modes.iter().zip(terms) {
            for n in 0..=order {
                for k in 0..=n {
                    let coefficient = Complex::new(nuclear.0.c[k], nuclear.1.c[k]);
                    add_complex(&mut core[n], &spin(&orbital[n - k]), coefficient);
                    if let Some(momenta) = &momenta {
                        for a in 0..3 {
                            let (x, y) = ((a + 1) % 3, (a + 2) % 3);
                            let factor = I
                                * coefficient
                                * (self.options.nuclear_soc_scale
                                    / (2.0 * 137.035_999_084_f64.powi(2)));
                            add(
                                &mut pauli[n][a + 1],
                                &bm(&momenta[y][n - k]),
                                factor * mode.wave.get(x),
                            );
                            add(
                                &mut pauli[n][a + 1],
                                &bm(&momenta[x][n - k]),
                                factor * -mode.wave.get(y),
                            );
                        }
                    }
                }
            }
            vertices.push(Vertex {
                value: orbital.iter().map(spin).collect(),
                coulomb: mode.coulomb,
                exchange: self.functional.hybrid.cx * mode.coulomb
                    + self.functional.hybrid.clr * mode.long_range,
            });
        }
        for n in 0..=order {
            cadd(&mut core[n], &cm(&from_pauli(&pauli[n])?), 1.0);
        }
        overlap = overlap.iter().map(spin).collect();
        Ok(Geometry {
            overlap,
            core,
            source: source.c[..=order].to_vec(),
            vertices,
            electric: super::electric::geometry(
                &self.cell,
                &self.basis,
                self.options.basis_cutoff,
                &self.sources,
                self.electric_background,
                self.options.electric_field,
                self.electrons,
                direction,
                order,
            )?,
            density: cm(density),
        })
    }

    /// Direct Hellmann-Feynman/Pulay energy slope of the stationary spinor
    /// density. This contracts integral derivatives at fixed AO density and
    /// subtracts Tr[P F P S']; it does not solve any density response equation.
    pub fn nuclear_energy_slope(
        &self,
        reference: &crate::periodic::spinor_scf::SpinorScfReport,
        direction: &[[f64; 3]],
    ) -> Result<f64> {
        self.validate_nuclear_reference(reference)?;
        let geometry = self.nuclear_geometry(direction, 1, &reference.density[0])?;
        let density = cm(&reference.density[0]);
        let size = density.re.rows;
        let paths = [
            vec![density.clone(), Cm::zeros(size, size)],
            vec![Cm::zeros(size, size); 2],
        ];
        let weighted = density
            .matmul_seq(&cm(&reference.fields.fock[0]))
            .matmul_seq(&density);
        Ok(geometry.energy(&paths, 1)?[1] - weighted.real_trace_product(&geometry.overlap[1]))
    }

    /// Relaxed nuclear energy on a finite Gamma magnetic bulk cell or slab.
    /// All atomic images move together at fixed cell/flux; a supplied Gaussian
    /// countercharge stays fixed. Point/Gaussian nuclei, full/range exchange
    /// and generated nuclear SOC share the integer spinor response. Semilocal,
    /// nonlocal and magnetic-BZ geometries require their matching providers.
    pub fn nuclear_energy_taylor(
        &self,
        reference: &crate::periodic::spinor_scf::SpinorScfReport,
        direction: &[[f64; 3]],
        order: usize,
        config: &ResponseSolverConfig,
    ) -> Result<MagneticNuclearTaylor> {
        config.validate()?;
        let nocc = self.validate_nuclear_reference(reference)?;
        if !matches!(self.cell.geometry.dimension(), 2 | 3) || !self.functional.parts.is_empty() {
            return Err(DftError::Unsupported {feature:"relaxed finite-flux nuclear response".into(),
                missing:"the moving semilocal/nonlocal or intersector providers for this magnetic reference".into()});
        }
        if order == 0
            || order > 4
            || !reference.converged
            || reference.density.len() != 1
            || reference.occupations.len() != 1
            || reference.bands.len() != 1
            || direction.iter().flatten().any(|v| !v.is_finite())
            || reference.occupations[0]
                .iter()
                .any(|f| *f != 0.0 && *f != 1.0)
        {
            return Err(DftError::InvalidInput("magnetic nuclear response requires a converged integer spinor reference and finite atom directions of order one through four".into()));
        }
        let geometry = self.nuclear_geometry(direction, order, &reference.density[0])?;
        let size = geometry.overlap[0].re.rows;
        let orbitals = &reference.bands[0];
        let state = ComplexPathReference {
            coefficients: [cm(&orbitals.coefficients), cm(&orbitals.coefficients)],
            energies: [orbitals.energies.clone(), orbitals.energies.clone()],
            occupations: [nocc, 0],
            density: [cm(&reference.density[0]), Cm::zeros(size, size)],
            fock: [cm(&reference.fields.fock[0]), cm(&reference.fields.fock[0])],
            unrestricted: true,
        };
        let result = stationary_complex_energy_path(&geometry, &state, order, config)?;
        let energy_defect = (result.energy[0] - reference.fields.energy).abs();
        if result.reference_defect > config.acceptance()
            || energy_defect > config.acceptance() * reference.fields.energy.abs().max(1.0)
        {
            return Err(DftError::InvalidInput(format!(
                "magnetic nuclear SCF/response mismatch: Fock {:.16e}, energy {energy_defect:.16e}",
                result.reference_defect
            )));
        }
        Ok(MagneticNuclearTaylor {
            energy: result.energy,
            solved_response_order: result.solved_order,
            response_residuals: result.residuals,
            reference_defect: result.reference_defect,
            reference_energy_defect: energy_defect,
            projector_defects: result.projector_defects,
            stationarity_defects: result.stationarity_defects,
        })
    }
}
