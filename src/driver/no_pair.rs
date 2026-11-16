// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Four-component sector projection on supplied or generated balanced H/S.

use super::{Controls, System, Value};
use crate::basis::MolBasis;
use crate::dual::{Scalar, Taylor};
use crate::integrals::analytic_basis::AnalyticShell;
use crate::integrals::dirac::{
    london_magnetic_balance, matrix_coefficients, restricted_kinetic_balance,
    restricted_magnetic_balance, GaussianNucleus,
};
use crate::linalg::Matrix;
use crate::periodic::spinor_scf::{run_spinor_scf, SpinorScfConfig};
use crate::periodic::BlochMatrix;
use crate::properties::no_pair::stationary::{
    DiracCoulombBackend, DiracInteraction, NoPairBackend,
};
use crate::properties::no_pair::{no_pair_operator_taylor, no_pair_projector_taylor};
use crate::system::Molecule;
use crate::{DftError, Result};

pub(super) struct FiniteGeometry {
    pub(super) molecule: Molecule,
    pub(super) basis: MolBasis,
    pub(super) basis_name: String,
    pub(super) radii: Vec<f64>,
    pub(super) c: f64,
}

pub(super) fn finite_geometry(
    system: &System,
    controls: &Controls,
    charge: i32,
) -> Result<FiniteGeometry> {
    if system.is_periodic() {
        return Err(DftError::InvalidInput(
            "the generated molecular balanced Dirac operator requires an isolated system".into(),
        ));
    }
    let molecule = super::method::molecule_from(&system.numbers, &system.positions, charge, 1)?;
    let basis_name = controls.str("basis", "sto-3g")?;
    let basis = MolBasis::build(&molecule, &super::method::build_basis(&basis_name, None)?)?;
    if !basis.ecps.is_empty() {
        return Err(DftError::InvalidInput(
            "finite-nucleus all-electron Dirac integrals require a basis without an ECP".into(),
        ));
    }
    let radii = controls.opt_f64_list("nuclear_rms_radii")?.ok_or_else(|| {
        DftError::InvalidInput(
            "nuclear_rms_radii in Bohr must be supplied for every finite Gaussian nucleus".into(),
        )
    })?;
    if radii.len() != molecule.atoms.len() || radii.iter().any(|r| !r.is_finite() || *r <= 0.0) {
        return Err(DftError::InvalidInput(
            "nuclear_rms_radii requires one positive finite Bohr radius per atom".into(),
        ));
    }
    let c = controls.f64("speed_of_light", 137.035999084)?;
    Ok(FiniteGeometry {
        molecule,
        basis,
        basis_name,
        radii,
        c,
    })
}

pub(super) fn electronic_options(controls: &Controls) -> Result<(i32, SpinorScfConfig)> {
    let charge = i32::try_from(controls.opt_i64("charge")?.unwrap_or(0))
        .map_err(|_| DftError::InvalidInput("charge exceeds the signed molecular range".into()))?;
    let mut config = SpinorScfConfig::default();
    config.max_iter = controls.usize("max_iter", config.max_iter)?;
    config.energy_tolerance = controls.f64("conv_tol", config.energy_tolerance)?;
    config.density_tolerance = controls.f64("conv_tol_dp", config.density_tolerance)?;
    config.orbital_tolerance = controls.f64("conv_tol_grad", config.orbital_tolerance)?;
    config.overlap_cutoff = controls.f64("metric_eigenvalue_cutoff", 1e-12)?;
    config.damping = controls.f64("damping", config.damping)?;
    config.diis_size = controls.usize("diis_size", config.diis_size)?;
    Ok((charge, config))
}

fn generated(
    system: &System,
    controls: &Controls,
    order: usize,
    provider: &str,
) -> Result<(Vec<BlochMatrix>, Vec<BlochMatrix>, f64, Value)> {
    let FiniteGeometry {
        molecule,
        basis,
        basis_name,
        radii,
        c,
    } = finite_geometry(system, controls, 0)?;
    let count = molecule.atoms.len();
    let dradii = controls
        .opt_f64_list("nuclear_radius_direction")?
        .unwrap_or_else(|| vec![0.0; count]);
    let displacements = controls
        .opt_rows("nuclear_displacements")?
        .unwrap_or_else(|| vec![vec![0.0; 3]; count]);
    if dradii.len() != count
        || displacements.len() != count
        || displacements.iter().any(|r| r.len() != 3)
    {
        return Err(DftError::InvalidInput("nuclear_rms_radii/nuclear_radius_direction require one entry per atom, and nuclear_displacements one Cartesian row per atom".into()));
    }
    let dc = controls.f64("speed_of_light_direction", 0.0)?;
    let seed = |x: f64, dx: f64| Taylor::<5>::from_coefficients(&[x, dx]);
    let nuclei: Vec<_> = molecule
        .atoms
        .iter()
        .enumerate()
        .map(|(i, atom)| GaussianNucleus {
            center: std::array::from_fn(|k| seed(system.positions[i][k], displacements[i][k])),
            charge: Taylor::from_f64(atom.z as f64),
            rms_radius: seed(radii[i], dradii[i]),
        })
        .collect();
    let shells = basis
        .shells
        .iter()
        .map(|shell| AnalyticShell::from_placed(shell, nuclei[shell.atom].center))
        .collect::<Result<Vec<_>>>()?;
    let mut phase_order = None;
    let integrals = if provider != "rkb" {
        let field = controls
            .opt_vec3("magnetic_field")?
            .unwrap_or(crate::math::Vec3::zero())
            .to_array();
        let direction = controls
            .opt_vec3("magnetic_direction")?
            .unwrap_or(crate::math::Vec3::zero())
            .to_array();
        let origin = controls
            .opt_vec3("gauge_origin")?
            .unwrap_or(crate::math::Vec3::zero())
            .to_array();
        let magnetic_field = std::array::from_fn(|k| seed(field[k], direction[k]));
        if provider == "london-rmb" {
            if origin.iter().any(|v| !v.is_finite()) {
                return Err(DftError::InvalidInput("gauge_origin must be finite".into()));
            }
            let fallback = if field == [0.0; 3] {
                order
            } else {
                order.max(6)
            };
            let degree = controls.usize("magnetic_phase_order", fallback)?;
            phase_order = Some(degree);
            london_magnetic_balance(&shells, &nuclei, seed(c, dc), magnetic_field, order, degree)?
        } else {
            restricted_magnetic_balance(
                &shells,
                &nuclei,
                seed(c, dc),
                magnetic_field,
                origin.map(Taylor::from_f64),
                order,
            )?
        }
    } else {
        restricted_kinetic_balance(&shells, &nuclei, seed(c, dc), order)?
    };
    let h = matrix_coefficients(
        &integrals.hamiltonian_real,
        Some(&integrals.hamiltonian_imaginary),
        order,
    )?;
    let s = matrix_coefficients(
        &integrals.overlap,
        Some(&integrals.overlap_imaginary),
        order,
    )?;
    let tb = matrix_coefficients(
        &integrals.balanced_kinetic_real,
        Some(&integrals.balanced_kinetic_imaginary),
        order,
    )?;
    let v = matrix_coefficients(
        &integrals.potential,
        Some(&integrals.potential_imaginary),
        order,
    )?;
    let t = matrix_coefficients(&integrals.kinetic, None, order)?;
    let w = matrix_coefficients(
        &integrals.small_potential_real,
        Some(&integrals.small_potential_imaginary),
        order,
    )?;
    let nuclear: Vec<_> = integrals.nuclear_repulsion.c[..=order].to_vec();
    if nuclear.iter().any(|x| !x.is_finite()) {
        return Err(DftError::InvalidInput(
            "non-finite finite-nucleus interaction coefficient".into(),
        ));
    }
    let mut details = Value::object()
        .with("basis", basis_name)
        .with("balance", match provider {
            "london-rmb" => "London orbitals and restricted magnetic balance with a local vector potential at each Gaussian centre",
            "rmb" => "restricted magnetic balance in a common-origin Gaussian basis; field-dependent small basis and complex metric",
            _ => "restricted kinetic balance at zero magnetic field",
        })
        .with("speed_of_light", c)
        .with("nuclear_rms_radii", radii)
        .with("nuclear_repulsion_coefficients", nuclear)
        .with(
            "dirac_hamiltonian_coefficients_real",
            h.iter().map(|a| a.re.clone()).collect::<Vec<_>>(),
        )
        .with(
            "dirac_hamiltonian_coefficients_imaginary",
            h.iter().map(|a| a.im.clone()).collect::<Vec<_>>(),
        )
        .with(
            "dirac_overlap_coefficients_real",
            s.iter().map(|a| a.re.clone()).collect::<Vec<_>>(),
        )
        .with("dirac_overlap_coefficients_imaginary",s.iter().map(|a|a.im.clone()).collect::<Vec<_>>())
        .with("balanced_kinetic_coefficients_real",tb.iter().map(|a|a.re.clone()).collect::<Vec<_>>())
        .with("balanced_kinetic_coefficients_imaginary",tb.iter().map(|a|a.im.clone()).collect::<Vec<_>>())
        .with(
            "kinetic_coefficients",
            t.iter().map(|a| a.re.clone()).collect::<Vec<_>>(),
        )
        .with(
            "large_potential_coefficients",
            v.iter().map(|a| a.re.clone()).collect::<Vec<_>>(),
        )
        .with(
            "small_potential_coefficients_real",
            w.iter().map(|a| a.re.clone()).collect::<Vec<_>>(),
        )
        .with(
            "small_potential_coefficients_imaginary",
            w.iter().map(|a| a.im.clone()).collect::<Vec<_>>(),
        )
        .with(
            "hamiltonian_hermiticity_defects",
            h.iter()
                .map(BlochMatrix::hermiticity_error)
                .collect::<Vec<_>>(),
        )
        .with(
            "overlap_hermiticity_defects",
            s.iter()
                .map(BlochMatrix::hermiticity_error)
                .collect::<Vec<_>>(),
        );
    if let Some(phase_order) = phase_order {
        details.set("magnetic_phase_order", phase_order);
        details.set(
            "large_potential_coefficients_imaginary",
            v.iter().map(|a| a.im.clone()).collect::<Vec<_>>(),
        );
    }
    Ok((h, s, -c * c, details))
}

pub(super) fn electronic_interaction(
    controls: &Controls,
) -> Result<(DiracInteraction, &'static str, &'static str)> {
    match controls
        .str("dirac_interaction", "coulomb")?
        .to_ascii_lowercase()
        .as_str()
    {
        "coulomb" => Ok((DiracInteraction::Coulomb, "coulomb", "Dirac–Coulomb")),
        "gaunt" => Ok((DiracInteraction::Gaunt, "gaunt", "Dirac–Coulomb–Gaunt")),
        "breit" => Ok((DiracInteraction::Breit, "breit", "Dirac–Coulomb–Breit")),
        _ => Err(DftError::InvalidInput(
            "dirac_interaction must be coulomb, gaunt or breit".into(),
        )),
    }
}

pub(super) fn dirac_coulomb_scf(system: &System, controls: &Controls) -> Result<Value> {
    let id = "dirac-coulomb-scf";
    let (charge, config) = electronic_options(controls)?;
    let (interaction, interaction_name, hamiltonian) = electronic_interaction(controls)?;
    let FiniteGeometry {
        molecule,
        basis,
        basis_name,
        radii,
        c,
    } = finite_geometry(system, controls, charge)?;
    let integral_mode = controls.str("dirac_integrals", "rkb")?;
    let magnetic = match integral_mode.as_str() {
        "rkb" => None,
        "london-rmb" => {
            let field = controls
                .opt_vec3("magnetic_field")?
                .unwrap_or_default()
                .to_array();
            let origin = controls
                .opt_vec3("gauge_origin")?
                .unwrap_or_default()
                .to_array();
            if origin.iter().any(|v| !v.is_finite()) {
                return Err(DftError::InvalidInput("gauge_origin must be finite".into()));
            }
            let default_degree = if field == [0.0; 3] { 0 } else { 6 };
            let degree = controls.usize("magnetic_phase_order", default_degree)?;
            let budget = controls.usize("incore_mb", 256)?;
            Some((field, degree, budget))
        }
        _ => {
            return Err(DftError::InvalidInput(
                "the Dirac–Coulomb SCF dirac_integrals selector must be rkb or london-rmb".into(),
            ));
        }
    };
    controls.finish(id)?;
    let nuclei: Vec<_> = molecule
        .atoms
        .iter()
        .zip(&radii)
        .map(|(atom, &rms_radius)| GaussianNucleus {
            center: atom.position.to_array(),
            charge: f64::from(atom.z),
            rms_radius,
        })
        .collect();
    let shells = basis
        .shells
        .iter()
        .map(|s| AnalyticShell::from_placed(s, s.center.to_array()))
        .collect::<Result<Vec<_>>>()?;
    let electrons = molecule.n_electrons()?;
    let backend = match magnetic {
        Some((field, degree, budget)) => DiracCoulombBackend::new_london_interaction(
            &shells,
            &nuclei,
            c,
            electrons,
            field,
            degree,
            budget,
            interaction,
        )?,
        None => {
            DiracCoulombBackend::new(&shells, &nuclei, c, electrons)?.with_interaction(interaction)
        }
    };
    let space = backend.electronic_space(config.overlap_cutoff)?;
    let result = run_spinor_scf(&NoPairBackend::new(&space, &backend)?, &config)?;
    let terms = result.fields.energy_components;
    let labels = [
        ("energy", "hartree"),
        ("energy_components", "hartree"),
        ("scf_orbital_residual", "hartree"),
        ("energy_change", "hartree"),
        ("density_change", "dimensionless"),
        ("metric_defect", "dimensionless"),
        ("occupation_consistency", "dimensionless"),
        ("particle_number_error", "electrons"),
        ("coulomb_hermiticity_defect", "hartree"),
        ("spectral_gap", "hartree"),
        ("orbital_energies", "hartree"),
        ("occupations", "electrons per spinor"),
        ("n_ao", "integer"),
        ("electronic_dimension", "integer"),
        ("speed_of_light", "atomic units of velocity"),
        ("nuclear_rms_radii", "bohr"),
        ("scf_iterations", "integer"),
    ];
    let mut units = Value::object();
    for (key, label) in labels {
        units.set(key, label);
    }
    // The Fock uses the full selected interaction. Separate its signed
    // current contributions at the final density for the energy report.
    let mut components = Value::object()
        .with("classical_nuclear_repulsion", terms[0])
        .with("kinetic_rest_subtracted", terms[1])
        .with("electron_nuclear", terms[2])
        .with("hartree", terms[4])
        .with("exchange", terms[5]);
    let mut report = Value::object();
    if let Some((field, degree, _)) = magnetic {
        report.set("magnetic_field", field);
        report.set("magnetic_phase_order", degree);
        report.set("dirac_integrals", "london-rmb");
        units.set("magnetic_field", "atomic units of magnetic field");
        units.set("magnetic_phase_order", "integer");
    }
    if interaction == DiracInteraction::Coulomb {
        report.set(
            "coulomb_hermiticity_defect",
            result.fields.coulomb_operator_defect,
        );
    } else {
        let density = space.lift_density(&result.density[0])?;
        let current = backend.current_fields(&density)?.ok_or_else(|| {
            DftError::InvalidInput("selected Dirac current interaction has no field report".into())
        })?;
        components.set("hartree", terms[4] - current.direct_energy);
        components.set("exchange", terms[5] - current.exchange_energy);
        components.set("current_direct", current.direct_energy);
        components.set("current_exchange", current.exchange_energy);
        report.set("current_hermiticity_defect", current.hermiticity_defect);
        report.set(
            "two_electron_hermiticity_defect",
            result.fields.coulomb_operator_defect,
        );
        units.set("current_hermiticity_defect", "hartree");
        units.set("two_electron_hermiticity_defect", "hartree");
    }
    let balance = if magnetic.is_some() {
        "London orbitals and restricted magnetic balance, a nuclear-core positive sector formed at the selected field"
    } else {
        "restricted kinetic balance, fixed nuclear-core positive-energy sector"
    };
    Ok(report
        .with("model", format!("all-electron finite-Gaussian-nucleus no-pair {hamiltonian} HF with {balance} and electron rest energy subtracted"))
        .with("dirac_interaction", interaction_name)
        .with("energy", result.fields.energy)
        .with("energy_components", components)
        .with("n_ao", basis.nao_sph)
        .with("basis", basis_name)
        .with("electronic_dimension", space.basis().cols())
        .with("scf_iterations", result.iterations)
        .with("scf_orbital_residual", result.orbital_residual)
        .with("energy_change", result.energy_change)
        .with("density_change", result.density_change)
        .with("occupation_consistency", result.occupation_consistency)
        .with("metric_defect", space.metric_defect.max(result.metric_defect))
        .with("particle_number_error", result.particle_number_error)
        .with("spectral_gap", space.spectral_gap)
        .with("orbital_energies", result.bands[0].energies.clone())
        .with("occupations", result.occupations[0].clone())
        .with("speed_of_light", c)
        .with("nuclear_rms_radii", radii)
        .with("converged", result.converged)
        .with("warnings", Vec::<String>::new())
        .with("units", units))
}

pub(super) fn operator_coefficients(
    controls: &Controls,
    dimension: usize,
    order: usize,
) -> Result<Option<Vec<BlochMatrix>>> {
    let parse = |key: &str| -> Result<Option<Vec<Matrix>>> {
        let Some(value) = controls.take(key) else {
            return Ok(None);
        };
        let Value::Array(coefficients) = value else {
            return Err(DftError::InvalidInput(format!(
                "{key} must be a coefficient-major array of square matrices"
            )));
        };
        if coefficients.is_empty() || coefficients.len() > order + 1 {
            return Err(DftError::InvalidInput(format!(
                "{key} requires one through response_order+1 factorial-normalized coefficients"
            )));
        }
        let mut matrices = Vec::with_capacity(coefficients.len());
        for coefficient in coefficients {
            let Value::Array(rows) = coefficient else {
                return Err(DftError::InvalidInput(format!(
                    "{key} coefficients must be square matrices"
                )));
            };
            if rows.len() != dimension {
                return Err(DftError::InvalidInput(format!(
                    "{key} coefficients must match the Dirac dimension"
                )));
            }
            let mut matrix = Matrix::zeros(dimension, dimension);
            for (i, row) in rows.iter().enumerate() {
                let Value::Array(entries) = row else {
                    return Err(DftError::InvalidInput(format!(
                        "{key} matrix rows must be numeric arrays"
                    )));
                };
                if entries.len() != dimension {
                    return Err(DftError::InvalidInput(format!(
                        "{key} coefficients must match the Dirac dimension"
                    )));
                }
                for (j, value) in entries.iter().enumerate() {
                    matrix[(i, j)] = value.as_f64().filter(|x| x.is_finite()).ok_or_else(|| {
                        DftError::InvalidInput(format!("{key} coefficients must be finite"))
                    })?;
                }
            }
            matrices.push(matrix);
        }
        Ok(Some(matrices))
    };
    let real = parse("dirac_operator_real")?;
    let imaginary = parse("dirac_operator_imaginary")?;
    if real.is_none() && imaginary.is_none() {
        return Ok(None);
    }
    let count = real
        .as_ref()
        .map_or(0, Vec::len)
        .max(imaginary.as_ref().map_or(0, Vec::len));
    Ok(Some(
        (0..count)
            .map(|k| BlochMatrix {
                re: real
                    .as_ref()
                    .and_then(|a| a.get(k))
                    .cloned()
                    .unwrap_or_else(|| Matrix::zeros(dimension, dimension)),
                im: imaginary
                    .as_ref()
                    .and_then(|a| a.get(k))
                    .cloned()
                    .unwrap_or_else(|| Matrix::zeros(dimension, dimension)),
            })
            .collect(),
    ))
}

pub(super) fn run(system: &System, controls: &Controls) -> Result<Value> {
    let order = controls.usize("response_order", 0)?;
    if order > 4 {
        return Err(DftError::InvalidInput(
            "no-pair response_order must be zero through four".into(),
        ));
    }
    let provider = controls.str("dirac_integrals", "supplied")?;
    let (hamiltonian, overlap, default_separator, details) = match provider.as_str() {
        "rkb" | "rmb" | "london-rmb" => generated(system, controls, order, &provider)?,
        "supplied" => {
            let hreal = controls
                .opt_rows("dirac_hamiltonian_real")?
                .ok_or_else(|| {
                    DftError::InvalidInput("dirac_hamiltonian_real is required".into())
                })?;
            let n = hreal.len();
            let matrix = |key: &str, rows: Vec<Vec<f64>>| -> Result<Matrix> {
                if rows.len() != n || rows.iter().any(|row| row.len() != n) {
                    return Err(DftError::InvalidInput(format!(
                        "{key} must have the same square shape as the four-component Hamiltonian"
                    )));
                }
                Ok(Matrix::from_row_major(
                    n,
                    n,
                    rows.into_iter().flatten().collect(),
                ))
            };
            let optional = |key: &str, default: Matrix| -> Result<Matrix> {
                match controls.opt_rows(key)? {
                    Some(rows) => matrix(key, rows),
                    None => Ok(default),
                }
            };
            let h = BlochMatrix {
                re: matrix("dirac_hamiltonian_real", hreal)?,
                im: optional("dirac_hamiltonian_imaginary", Matrix::zeros(n, n))?,
            };
            let s = BlochMatrix {
                re: optional("dirac_overlap_real", Matrix::identity(n))?,
                im: optional("dirac_overlap_imaginary", Matrix::zeros(n, n))?,
            };
            let dh = BlochMatrix {
                re: optional("dirac_hamiltonian_direction_real", Matrix::zeros(n, n))?,
                im: optional("dirac_hamiltonian_direction_imaginary", Matrix::zeros(n, n))?,
            };
            let ds = BlochMatrix {
                re: optional("dirac_overlap_direction_real", Matrix::zeros(n, n))?,
                im: optional("dirac_overlap_direction_imaginary", Matrix::zeros(n, n))?,
            };
            let (hamiltonian, overlap) = if order == 0 {
                (vec![h], vec![s])
            } else {
                (vec![h, dh], vec![s, ds])
            };
            (hamiltonian, overlap, f64::NAN, Value::object())
        }
        _ => {
            return Err(DftError::InvalidInput(
                "dirac_integrals must be supplied, rkb, rmb or london-rmb".into(),
            ))
        }
    };
    let separator = controls
        .opt_f64("spectrum_separator")?
        .or_else(|| default_separator.is_finite().then_some(default_separator))
        .ok_or_else(|| {
            DftError::InvalidInput(
                "spectrum_separator is required inside the electronic/negative-energy gap".into(),
            )
        })?;
    let cutoff = controls.f64("metric_eigenvalue_cutoff", 1e-12)?;
    let operator = operator_coefficients(controls, hamiltonian[0].rows(), order)?;
    controls.finish("no-pair-projector")?;
    let (result, projected_operator) = if let Some(operator) = operator {
        let result =
            no_pair_operator_taylor(&hamiltonian, &overlap, &operator, order, separator, cutoff)?;
        (result.sector, Some(result.operator))
    } else {
        (
            no_pair_projector_taylor(&hamiltonian, &overlap, order, separator, cutoff)?,
            None,
        )
    };
    let converged = result
        .projector_defects
        .iter()
        .chain(&result.metric_defects)
        .chain(&result.relative_commutator_defects)
        .chain(&result.number_defects)
        .chain(&result.complex_embedding_defects)
        .chain(&result.relative_hamiltonian_embedding_defects)
        .all(|x| x.is_finite() && *x <= 1e-10);
    let warnings = if converged {
        Vec::<String>::new()
    } else {
        vec!["electronic-sector projector response did not meet residual tolerance 1e-10; inspect unprojected raw diagnostics".into()]
    };
    let model = if provider == "london-rmb" {
        "one-electron finite-Gaussian-nucleus Dirac no-pair projector with London orbitals, restricted magnetic balance and a moving nuclear-core positive-energy sector"
    } else if provider == "rmb" {
        "four-component finite-Gaussian-nucleus restricted-magnetic-balance no-pair operator; common-origin uniform field, full one-electron orbital/spin/SOC and field-dependent small basis/complex metric; no London basis or interacting Dirac self-consistent reference"
    } else if provider == "rkb" {
        "four-component no-pair spectral projector on finite Gaussian nuclei and restricted-kinetic-balance molecular integrals; full one-electron spin-orbit operator and analytic basis/metric/nuclear response; no magnetic-balance or interacting Dirac self-consistent state"
    } else {
        "four-component no-pair spectral projector on supplied balanced H/S; analytic moving-overlap and negative-energy-sector response; no generated finite-nucleus integrals or Dirac self-consistent state"
    };
    let convention = if provider == "london-rmb" {
        "Factorial-normalized coefficients along dimensionless t, including nuclear, speed-of-light and magnetic directions. The large basis phase is exp[-i (B cross (R_A-gauge_origin)) dot r/2]. Restricted magnetic balance applies sigma.(p+A)/(2c) to the phased large basis. The global gauge phase cancels from integrals. The zero-field path is exact in the reported order; at finite field converge the local phase expansion with magnetic_phase_order. The rest energy is removed. Contravariant P obeys P S P=P."
    } else if provider == "rmb" {
        "Factorial-normalized coefficients in dimensionless t with nuclear positions, radii, c and B following their supplied linear directions. A=B cross (r-gauge_origin)/2; chi_S=(sigma.(p+A))*chi_L/(2c). AO order [large up,large down,small up,small down]; rest energy removed; complex contravariant P obeys P S P=P. Both metric parts and all balance terms are differentiated without Hermitian or endpoint triangle copying."
    } else if provider == "rkb" {
        "Factorial-normalized coefficients along dimensionless t: nuclear positions, RMS radii and c follow their supplied linear directions. AO order [large up,large down,small up,small down]; H excludes c^2 rest energy. Small basis=(sigma.p)/(2c) times large basis. Contravariant P obeys P S P=P; covariant projected operator=S P H P S. No Hermitian projection or endpoint triangle copy."
    } else {
        "H(t)=H+t*dH; S(t)=S+t*dS; t dimensionless; factorial-normalized coefficients. Contravariant P obeys P S P=P, covariant projected operator is S P H P S. Provider four-component order is preserved; no spin or Hermitian projection."
    };
    let mut report = Value::object()
        .with("model", model)
        .with("dirac_integrals", provider.as_str())
        .with("converged", converged)
        .with("warnings", warnings)
        .with("response_order", order)
        .with("electronic_dimension", result.electronic_dimension)
        .with("spectral_gap", result.spectral_gap)
        .with("spectrum_separator", separator)
        .with("metric_eigenvalue_cutoff", cutoff)
        .with(
            "projector_coefficients_real",
            result
                .projector
                .iter()
                .map(|a| a.re.clone())
                .collect::<Vec<_>>(),
        )
        .with(
            "projector_coefficients_imaginary",
            result
                .projector
                .iter()
                .map(|a| a.im.clone())
                .collect::<Vec<_>>(),
        )
        .with(
            "projected_hamiltonian_coefficients_real",
            result
                .projected_hamiltonian
                .iter()
                .map(|a| a.re.clone())
                .collect::<Vec<_>>(),
        )
        .with(
            "projected_hamiltonian_coefficients_imaginary",
            result
                .projected_hamiltonian
                .iter()
                .map(|a| a.im.clone())
                .collect::<Vec<_>>(),
        )
        .with("projector_defects", result.projector_defects)
        .with("metric_defects", result.metric_defects)
        .with("commutator_defects", result.commutator_defects)
        .with(
            "relative_commutator_defects",
            result.relative_commutator_defects,
        )
        .with("number_defects", result.number_defects)
        .with(
            "complex_embedding_defects",
            result.complex_embedding_defects,
        )
        .with(
            "hamiltonian_embedding_defects",
            result.hamiltonian_embedding_defects,
        )
        .with(
            "relative_hamiltonian_embedding_defects",
            result.relative_hamiltonian_embedding_defects,
        )
        .with("coefficient_convention", convention);
    if let Value::Object(entries) = details {
        for (key, value) in entries {
            report.set(&key, value);
        }
    }
    let mut units = Value::object();
    for key in [
        "spectral_gap",
        "spectrum_separator",
        "projected_hamiltonian_coefficients_real",
        "projected_hamiltonian_coefficients_imaginary",
        "commutator_defects",
        "hamiltonian_embedding_defects",
        "nuclear_repulsion_coefficients",
        "dirac_hamiltonian_coefficients_real",
        "dirac_hamiltonian_coefficients_imaginary",
        "kinetic_coefficients",
        "balanced_kinetic_coefficients_real",
        "balanced_kinetic_coefficients_imaginary",
        "large_potential_coefficients",
        "small_potential_coefficients_real",
        "small_potential_coefficients_imaginary",
        "hamiltonian_hermiticity_defects",
    ] {
        units.set(key, "hartree");
    }
    for key in [
        "metric_eigenvalue_cutoff",
        "projector_coefficients_real",
        "projector_coefficients_imaginary",
        "projector_defects",
        "metric_defects",
        "relative_commutator_defects",
        "number_defects",
        "complex_embedding_defects",
        "relative_hamiltonian_embedding_defects",
        "dirac_overlap_coefficients_real",
        "dirac_overlap_coefficients_imaginary",
        "overlap_hermiticity_defects",
    ] {
        units.set(key, "dimensionless");
    }
    units.set("nuclear_rms_radii", "bohr");
    units.set("speed_of_light", "atomic units of velocity");
    if provider == "london-rmb" {
        units.set("magnetic_phase_order", "integer");
        units.set("large_potential_coefficients_imaginary", "hartree");
    }
    if let Some(operator) = projected_operator {
        report.set(
            "projected_operator_coefficients_real",
            operator.iter().map(|a| a.re.clone()).collect::<Vec<_>>(),
        );
        report.set(
            "projected_operator_coefficients_imaginary",
            operator.iter().map(|a| a.im.clone()).collect::<Vec<_>>(),
        );
        report.set("projected_operator_model", "bare no-pair projection of a supplied dimensionless covariant one-electron operator; Q=S P O P S; factorial-normalized coefficients retain electronic/negative-sector and moving-metric response; no X2C decoupling, interacting DKS, two-electron picture change or conserved-current construction");
        units.set("projected_operator_coefficients_real", "dimensionless");
        units.set("projected_operator_coefficients_imaginary", "dimensionless");
    }
    report.set("units", units);
    Ok(report)
}

#[cfg(test)]
mod stationary_route_tests {
    use super::*;

    #[test]
    fn dirac_coulomb_molecular_input_and_report_units() {
        let system = System::molecule(vec![2, 1], vec![[0.12, -0.27, 0.09], [1.41, 0.31, -0.46]]);
        let input = Value::object()
            .with("basis", "sto-3g")
            .with("charge", 1_i64)
            .with("speed_of_light", 7.4)
            .with("nuclear_rms_radii", [0.12, 0.081])
            .with("conv_tol", 2e-13)
            .with("conv_tol_dp", 1e-11)
            .with("conv_tol_grad", 1e-11);
        let mut report =
            dirac_coulomb_scf(&system, &Controls::from_value(&input).unwrap()).unwrap();
        assert_eq!(report.get("converged").and_then(Value::as_bool), Some(true));
        let problems = crate::driver::units::complete(&mut report);
        assert!(problems.is_empty(), "{problems:?}");
        println!("Molecular Dirac Coulomb SCF {}", report.to_json());
    }
}

// Supplied O is dimensionless on the common driver. Coefficients are indexed
// [Taylor order][AO row][AO column], including explicit zeroth order.
