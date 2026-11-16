// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Electronic magnetic dipole and quadrupole in a source-free linear field.

use super::method::{
    run_collinear_property_reference, run_molecular_reference, CollinearPropertyReference,
    MolecularSetup,
};
use super::molecular::{moment_origin, setup, units};
use super::{Controls, System, Value};
use crate::basis::{n_sph, MolBasis};
use crate::integrals::analytic_basis::{moment_gradient, AnalyticShell};
use crate::integrals::one_electron::moment_matrices;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::spinor_scf::{run_spinor_scf, spin_diagonal};
use crate::periodic::BlochMatrix;
use crate::response::spinor::{MolecularSpinorBackend, MolecularSpinorFunctionalModel};
use crate::{DftError, Result};

type Tensor = [[f64; 3]; 3];
type Power = [usize; 3];
type Polynomial = Vec<(Power, f64)>;
type Potential = [Polynomial; 3];

fn unit(axis: usize) -> Power {
    let mut p = [0; 3];
    p[axis] = 1;
    p
}

fn power_add(a: Power, b: Power) -> Power {
    std::array::from_fn(|i| a[i] + b[i])
}

fn vector_potential(field: [f64; 3], gradient: Tensor) -> Potential {
    // Poincare gauge centred at the reported origin:
    // A(r) = B0 x r / 2 + (G r) x r / 3.
    std::array::from_fn(|i| {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let mut terms = vec![(unit(k), 0.5 * field[j]), (unit(j), -0.5 * field[k])];
        for axis in 0..3 {
            terms.push((power_add(unit(k), unit(axis)), gradient[j][axis] / 3.0));
            terms.push((power_add(unit(j), unit(axis)), -gradient[k][axis] / 3.0));
        }
        terms.retain(|(_, c)| *c != 0.0);
        terms
    })
}

fn add_scaled(out: &mut Matrix, matrix: &Matrix, scale: f64) {
    for (a, &b) in out.as_mut_slice().iter_mut().zip(matrix.as_slice()) {
        *a += scale * b;
    }
}

fn orbital_operator(
    basis: &MolBasis,
    origin: Vec3,
    perturbation: &Potential,
    bias: &Potential,
    quadratic_factor: f64,
) -> Result<BlochMatrix> {
    let mut op = BlochMatrix::zeros(basis.nao_sph, basis.nao_sph);
    let shells = basis
        .shells
        .iter()
        .map(|s| AnalyticShell::from_placed(s, s.center.to_array()))
        .collect::<Result<Vec<_>>>()?;
    for (j, b) in basis.shells.iter().enumerate() {
        for (i, a) in basis.shells.iter().enumerate().take(j + 1) {
            for (axis, terms) in perturbation.iter().enumerate() {
                for &(power, coefficient) in terms {
                    let block =
                        moment_gradient(&shells[i], &shells[j], power, origin.to_array(), axis)?;
                    for p in 0..n_sph(a.l) {
                        for q in 0..n_sph(b.l) {
                            op.im[(a.sph_off + p, b.sph_off + q)] -=
                                coefficient * block.values[p * block.shape[1] + q];
                        }
                    }
                }
            }
        }
    }
    for j in 0..op.rows() {
        op.im[(j, j)] = 0.0;
        for i in 0..j {
            op.im[(j, i)] = -op.im[(i, j)];
        }
    }
    let mut products: std::collections::BTreeMap<Power, f64> = Default::default();
    for (a, b) in perturbation.iter().zip(bias) {
        for &(pa, ca) in a {
            for &(pb, cb) in b {
                *products.entry(power_add(pa, pb)).or_default() += quadratic_factor * ca * cb;
            }
        }
    }
    for degree in 2..=4 {
        if products
            .iter()
            .any(|(power, scale)| *scale != 0.0 && power.iter().sum::<usize>() == degree)
        {
            for ((i, j, k), matrix) in moment_matrices(basis, origin, degree as u8) {
                if let Some(&scale) = products.get(&[i as usize, j as usize, k as usize]) {
                    add_scaled(&mut op.re, &matrix, scale);
                }
            }
        }
    }
    Ok(spin_diagonal(&op))
}

fn spin_operator(
    overlap: &Matrix,
    dipoles: &[Matrix; 3],
    field: [f64; 3],
    gradient: Tensor,
) -> BlochMatrix {
    let n = overlap.rows;
    let components: [Matrix; 3] = std::array::from_fn(|i| {
        let mut matrix = Matrix::zeros(n, n);
        add_scaled(&mut matrix, overlap, 0.5 * field[i]);
        for (j, d) in dipoles.iter().enumerate() {
            add_scaled(&mut matrix, d, 0.5 * gradient[i][j]);
        }
        matrix
    });
    let mut op = BlochMatrix::zeros(2 * n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            op.re[(i, j)] = components[2][(i, j)];
            op.re[(n + i, n + j)] = -components[2][(i, j)];
            op.re[(i, n + j)] = components[0][(i, j)];
            op.re[(n + i, j)] = components[0][(i, j)];
            op.im[(i, n + j)] = -components[1][(i, j)];
            op.im[(n + i, j)] = components[1][(i, j)];
        }
    }
    op
}

fn tensor_modes() -> [Tensor; 5] {
    let a = 2.0_f64.sqrt().recip();
    let b = 6.0_f64.sqrt().recip();
    [
        [[a, 0., 0.], [0., -a, 0.], [0., 0., 0.]],
        [[b, 0., 0.], [0., b, 0.], [0., 0., -2. * b]],
        [[0., a, 0.], [a, 0., 0.], [0., 0., 0.]],
        [[0., 0., a], [0., 0., 0.], [a, 0., 0.]],
        [[0., 0., 0.], [0., 0., a], [0., a, 0.]],
    ]
}

fn tensor(modes: &[Tensor; 5], components: &[f64; 5]) -> Tensor {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| modes.iter().zip(components).map(|(m, c)| c * m[i][j]).sum())
    })
}

fn zero_field_density(
    setup: &MolecularSetup,
    roks: bool,
) -> Result<(BlochMatrix, f64, f64, usize, &'static str)> {
    let reference = run_collinear_property_reference(
        setup,
        if roks {
            CollinearPropertyReference::Roks
        } else {
            CollinearPropertyReference::Auto
        },
    )?;
    let scf = &reference.scf;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: scf.common_orbital_residual.unwrap_or(f64::INFINITY),
        });
    }
    let diagnostic = crate::response::diagnostics::reference_diagnostics(scf)?;
    let n = setup.basis.nao_sph;
    let mut p = BlochMatrix::zeros(2 * n, 2 * n);
    for s in 0..2 {
        for i in 0..n {
            for j in 0..n {
                p.re[(s * n + i, s * n + j)] = scf.density[s][(i, j)];
            }
        }
    }
    Ok((
        p,
        reference.total_energy(),
        scf.common_orbital_residual
            .unwrap_or(diagnostic.orbital_residual),
        scf.iterations,
        if roks {
            "roks"
        } else if scf.unrestricted {
            "uks"
        } else {
            "rks"
        },
    ))
}

pub(super) fn run(system: &System, controls: &Controls) -> Result<Value> {
    let feature = "magnetic-quadrupole";
    let reference = controls.str("reference", "auto")?.to_ascii_lowercase();
    let gks = matches!(reference.as_str(), "gks" | "ghf");
    let roks = matches!(reference.as_str(), "roks" | "rohf");
    if super::double_hybrid::requested(controls)? {
        return Err(DftError::Unsupported {
            feature: feature.into(),
            missing: "a variational PT2 magnetic-gradient density".into(),
        });
    }
    let setup = setup(system, controls)?;
    let (origin, origin_kind) = moment_origin(controls, &setup)?;
    let field = controls
        .opt_vec3("magnetic_field")?
        .map_or([0.0; 3], Vec3::to_array);
    let mut gradient = [[0.0; 3]; 3];
    if let Some(rows) = controls.opt_rows("magnetic_field_gradient")? {
        if rows.len() != 3 || rows.iter().any(|r| r.len() != 3) {
            return Err(DftError::InvalidInput(
                "magnetic_field_gradient needs a Cartesian 3 by 3 tensor".into(),
            ));
        }
        for (row, input) in gradient.iter_mut().zip(rows) {
            row.copy_from_slice(&input);
        }
        let trace: f64 = (0..3).map(|i| gradient[i][i]).sum();
        if trace.abs() > 1e-12
            || (0..3).any(|i| (0..i).any(|j| (gradient[i][j] - gradient[j][i]).abs() > 1e-12))
        {
            return Err(DftError::InvalidInput(
                "source-free magnetic_field_gradient must be symmetric and traceless".into(),
            ));
        }
    }
    if !gks && (field != [0.0; 3] || gradient != [[0.0; 3]; 3]) {
        return Err(DftError::Unsupported {
            feature: "magnetic quadrupole at nonzero magnetic bias".into(),
            missing: "complex orbital and transverse spin self-consistency; select reference=gks"
                .into(),
        });
    }
    let axis = if gks {
        controls
            .opt_vec3("spin_axis")?
            .map_or([0., 0., 1.], Vec3::to_array)
    } else {
        [0., 0., 1.]
    };
    let xc_tolerance = if gks {
        controls.f64("spinor_xc_tolerance", 1e-10)?
    } else {
        1e-10
    };
    controls.finish(feature)?;
    if !setup.basis.ecps.is_empty()
        || setup.options.electronic_temperature != 0.0
        || setup.options.electron_offset != [0.0; 2]
        || setup.options.asymptotic_correction.is_some()
        || setup.options.occupation.is_some()
        || setup.options.fixed_occupations.is_some()
        || setup.method.stabilize
    {
        return Err(DftError::Unsupported {
            feature: feature.into(),
            missing: "an all-electron integer aufbau reference and its magnetic-gradient operator"
                .into(),
        });
    }
    let overlap = crate::integrals::overlap_matrix(&setup.basis);
    let dipoles = crate::integrals::dipole_matrices(&setup.basis, origin);
    let bias = vector_potential(field, gradient);
    let (density, energy, stationarity, iterations, kind, spinor_components) = if gks {
        let seed = run_molecular_reference(&setup)?;
        let mut operator = orbital_operator(&setup.basis, origin, &bias, &bias, 0.5)?;
        let spin = spin_operator(&overlap, &dipoles, field, gradient);
        add_scaled(&mut operator.re, &spin.re, 1.0);
        add_scaled(&mut operator.im, &spin.im, 1.0);
        let backend = MolecularSpinorBackend::new(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &seed.scf,
            &setup.options,
            operator,
            axis,
            MolecularSpinorFunctionalModel::SpinCovariant,
            xc_tolerance,
        )?;
        let state = run_spinor_scf(&backend, &setup.spinor_scf_config())?;
        if !state.converged {
            return Err(DftError::ScfNotConverged {
                iterations: state.iterations,
                error: state.orbital_residual,
            });
        }
        (
            state
                .density
                .into_iter()
                .next()
                .expect("molecular Gamma density"),
            state.fields.energy + seed.e_dispersion + seed.e_smd_cds,
            state.orbital_residual,
            state.iterations,
            "gks",
            Some(state.fields.energy_components),
        )
    } else {
        let (density, energy, stationarity, iterations, kind) = zero_field_density(&setup, roks)?;
        (density, energy, stationarity, iterations, kind, None)
    };
    let modes = tensor_modes();
    let mut orbital = [0.0; 5];
    let mut spin = [0.0; 5];
    for (i, mode) in modes.iter().copied().enumerate() {
        let direction = vector_potential([0.0; 3], mode);
        if gks {
            orbital[i] = -density.trace_product(&orbital_operator(
                &setup.basis,
                origin,
                &direction,
                &bias,
                1.0,
            )?);
        }
        spin[i] = -density.trace_product(&spin_operator(&overlap, &dipoles, [0.0; 3], mode));
    }
    let components = std::array::from_fn(|i| orbital[i] + spin[i]);
    let mut dipole = [0.0; 3];
    for (i, dipole) in dipole.iter_mut().enumerate() {
        let mut field_mode = [0.0; 3];
        field_mode[i] = 1.0;
        *dipole = -density.trace_product(&spin_operator(
            &overlap,
            &dipoles,
            field_mode,
            [[0.0; 3]; 3],
        ));
        if gks {
            let direction = vector_potential(field_mode, [[0.0; 3]; 3]);
            *dipole -= density.trace_product(&orbital_operator(
                &setup.basis,
                origin,
                &direction,
                &bias,
                1.0,
            )?);
        }
    }
    Ok(Value::object()
        .with("model","electronic magnetic quadrupole Q=-dE/dG in the orthonormal symmetric-traceless basis for B(r)=B0+G(r-origin); common Poincare gauge, minimal orbital coupling, g=2 electronic Pauli spin and current-invariant meta-GGA kinetic density")
        .with("energy",energy).with("converged",true)
        .with("reference_type",kind).with("requested_reference",reference)
        .with("functional",setup.functional.description.as_str())
        .with("spinor_energy_components",spinor_components)
        .with("scf_iterations",iterations).with("scf_orbital_residual",stationarity)
        .with("origin",origin).with("origin_kind",origin_kind)
        .with("magnetic_field",field).with("magnetic_field_gradient",gradient)
        .with("magnetic_dipole",dipole).with("magnetic_quadrupole",tensor(&modes,&components))
        .with("orbital_quadrupole",tensor(&modes,&orbital)).with("spin_quadrupole",tensor(&modes,&spin))
        .with("magnetic_quadrupole_modes",modes).with("magnetic_quadrupole_components",components)
        .with("warnings",Vec::<String>::new())
        .with("units",units(&[
            ("energy","hartree"),("scf_orbital_residual","hartree"),("origin","bohr"),
            ("spinor_energy_components","hartree"),
            ("magnetic_field","atomic units of magnetic field"),
            ("magnetic_field_gradient","atomic units of magnetic field gradient"),
            ("magnetic_dipole","atomic units of magnetic dipole"),
            ("magnetic_quadrupole","atomic units of magnetic quadrupole"),
            ("orbital_quadrupole","atomic units of magnetic quadrupole"),
            ("spin_quadrupole","atomic units of magnetic quadrupole"),
            ("magnetic_quadrupole_modes","dimensionless"),
            ("magnetic_quadrupole_components","atomic units of magnetic quadrupole"),
        ])))
}
