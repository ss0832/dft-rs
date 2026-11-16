// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Band-structure post-processing of a converged finite-k reference: total
//! and Mulliken-projected densities of states, and the Fermi-level
//! thermodynamics of the rigid bands (bare Pauli spin susceptibility and
//! electronic heat capacity).

use super::controls::Controls;
use super::molecular::units;
use super::periodic::{
    band_energies, prepare_periodic, reference_report, run_reference, PeriodicMethod, PeriodicSetup,
};
use super::value::Value;
use super::System;
use crate::basis::MolBasis;
use crate::constants::{
    AVOGADRO_PER_MOL, BOHR_TO_METER, BOLTZMANN_J_PER_K, FINE_STRUCTURE, HARTREE_TO_EV,
};
use crate::math::Vec3;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::fermi_thermodynamics::collinear_fermi_thermodynamics;
use crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
use crate::periodic::scf::{reference_fock_at, PeriodicScfResult};
use crate::{DftError, Result};

const ANGULAR_LABELS: [&str; 7] = ["s", "p", "d", "f", "g", "h", "i"];

/// The k points of a band-structure sum with their Fock and overlap
/// matrices: the SCF mesh itself, or a denser Monkhorst-Pack mesh on which
/// the converged Fock operator is reconstructed.
struct Sampling {
    k_points: Vec<Vec3>,
    /// `None` on the SCF mesh, where the reference already holds them.
    reconstructed: Option<([Vec<BlochMatrix>; 2], Vec<BlochMatrix>)>,
}

impl Sampling {
    fn new(
        setup: &PeriodicSetup,
        scf: &PeriodicScfResult,
        mesh: Option<Vec<usize>>,
        control: &str,
    ) -> Result<Self> {
        let Some(mesh) = mesh else {
            return Ok(Self {
                k_points: scf.k_points.clone(),
                reconstructed: None,
            });
        };
        if mesh.len() != 3 || mesh.contains(&0) {
            return Err(DftError::InvalidInput(format!(
                "{control} takes three positive integers"
            )));
        }
        let k_points = setup.cell.k_points([mesh[0], mesh[1], mesh[2]]);
        let focks = reference_fock_at(
            &setup.cell,
            &setup.functional,
            scf,
            &setup.options,
            &k_points,
        )?;
        let overlaps = k_points
            .iter()
            .map(|k| BlochMatrix::from_pair(setup.cell.overlap(*k)))
            .collect();
        Ok(Self {
            k_points,
            reconstructed: Some((focks, overlaps)),
        })
    }

    /// Band energies of one spin channel, `[k][band]`.
    fn energies(&self, scf: &PeriodicScfResult, spin: usize) -> Result<Vec<Vec<f64>>> {
        match &self.reconstructed {
            None => Ok(scf.mo_energy[spin].clone()),
            Some((focks, overlaps)) => focks[spin]
                .iter()
                .zip(overlaps)
                .map(|(fock, overlap)| band_energies(fock, overlap))
                .collect(),
        }
    }

    /// Fock and overlap matrix of one spin channel at one k point.
    fn matrices<'a>(
        &'a self,
        scf: &'a PeriodicScfResult,
        spin: usize,
        ik: usize,
    ) -> (&'a BlochMatrix, &'a BlochMatrix) {
        match &self.reconstructed {
            None => (&scf.fock[spin][ik], &scf.overlap[ik]),
            Some((focks, overlaps)) => (&focks[spin][ik], &overlaps[ik]),
        }
    }
}

/// Mulliken projection groups of a basis: one per atom and angular momentum
/// present on it, in shell order, and the group index of every spherical
/// basis function.
fn projection_groups(basis: &MolBasis) -> (Vec<(usize, u8)>, Vec<usize>) {
    let mut groups: Vec<(usize, u8)> = Vec::new();
    let mut of_ao = vec![0usize; basis.nao_sph];
    for shell in &basis.shells {
        let key = (shell.atom, shell.l);
        let index = match groups.iter().position(|g| *g == key) {
            Some(index) => index,
            None => {
                groups.push(key);
                groups.len() - 1
            }
        };
        for m in 0..crate::basis::n_sph(shell.l) {
            of_ao[shell.sph_off + m] = index;
        }
    }
    (groups, of_ao)
}

/// Band energies and the Mulliken weight of every projection group in every
/// band, `w = sum_{mu in group} Re[conj(C_mu) (S C)_mu]`. The weights of one
/// band sum to one because the bands are normalized in the overlap metric.
fn projected_bands(
    fock: &BlochMatrix,
    overlap: &BlochMatrix,
    of_ao: &[usize],
    groups: usize,
) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let bands = crate::dfpt::orbitals::generalized_orbitals(fock, overlap, 1e-7)?;
    let weights = projection_weights(&bands.coefficients, overlap, of_ao, groups)?;
    Ok((bands.energies, weights))
}

fn projection_weights(
    c: &BlochMatrix,
    overlap: &BlochMatrix,
    of_ao: &[usize],
    groups: usize,
) -> Result<Vec<Vec<f64>>> {
    if of_ao.is_empty() || c.rows() % of_ao.len() != 0 {
        return Err(DftError::InvalidInput(
            "DOS coefficients do not match the AO projection groups".into(),
        ));
    }
    let sc = crate::dfpt::orbitals::matmul(overlap, c)?;
    let mut weights = vec![vec![0.0; groups]; c.cols()];
    for (band, row) in weights.iter_mut().enumerate() {
        for mu in 0..c.rows() {
            row[of_ao[mu % of_ao.len()]] +=
                c.re[(mu, band)] * sc.re[(mu, band)] + c.im[(mu, band)] * sc.im[(mu, band)];
        }
    }
    Ok(weights)
}

pub(super) struct DosOptions {
    sigma: f64,
    points: usize,
    window: Option<[f64; 2]>,
    mesh: Option<[usize; 3]>,
    projected: bool,
}

impl DosOptions {
    pub(super) fn parse(controls: &Controls) -> Result<Self> {
        let sigma = controls.opt_f64("sigma")?.ok_or_else(|| {
            DftError::InvalidInput("dos needs sigma, the Gaussian broadening in Hartree".into())
        })?;
        let points = controls.usize("points", 1000)?;
        let window = controls.opt_f64_list("energy_window")?;
        let mesh = controls.opt_usize_list("dos_mesh")?;
        let projected = controls.bool("projected", false)?;
        if !sigma.is_finite() || sigma <= 0.0 || points < 2 {
            return Err(DftError::InvalidInput(
                "dos needs finite sigma > 0 and points >= 2".into(),
            ));
        }
        let window = match window.as_deref() {
            Some([lo, hi]) if lo.is_finite() && hi.is_finite() && lo < hi => Some([*lo, *hi]),
            Some(_) => {
                return Err(DftError::InvalidInput(
                    "energy_window takes [minimum, maximum] in Hartree".into(),
                ))
            }
            None => None,
        };
        let mesh = match mesh.as_deref() {
            Some([x, y, z]) if [*x, *y, *z].iter().all(|&n| n > 0) => Some([*x, *y, *z]),
            Some(_) => {
                return Err(DftError::InvalidInput(
                    "dos_mesh takes three positive integers".into(),
                ))
            }
            None => None,
        };
        Ok(Self {
            sigma,
            points,
            window,
            mesh,
            projected,
        })
    }

    pub(super) fn require_reference_mesh(&self, mesh: [usize; 3]) -> Result<()> {
        if self.mesh.is_some_and(|selected| selected != mesh) {
            return Err(DftError::Unsupported {
                feature: "frozen DOS resampling for this periodic reference".into(),
                missing: "the matching operator and occupation subspaces on the requested mesh; set k_mesh for a self-consistent DOS mesh".into(),
            });
        }
        Ok(())
    }
}

#[derive(Default)]
struct DosBands {
    energies: Vec<Vec<Vec<f64>>>,
    weights: Vec<Vec<Vec<Vec<f64>>>>,
}

/// Gaussian-broadened density of states; with `projected`, also its Mulliken
/// decomposition by atom and angular momentum.
pub(super) fn dos(system: &System, controls: &Controls) -> Result<Value> {
    let mut method = PeriodicMethod::from_controls(controls)?;
    if let Some(reference) = controls.opt_str("reference")? {
        match reference.to_ascii_lowercase().as_str() {
            "auto" => {}
            "rks" if method.multiplicity == 1 && !method.unrestricted => {}
            "rks" => {
                return Err(DftError::InvalidInput(
                    "periodic RKS needs multiplicity=1 and unrestricted=false".into(),
                ))
            }
            "uks" => method.unrestricted = true,
            _ => {
                return Err(DftError::InvalidInput(format!(
                    "unknown DOS reference {reference}"
                )))
            }
        }
    }
    let options = DosOptions::parse(controls)?;
    controls.finish("dos")?;
    let setup = prepare_periodic(system, &method)?;
    let scf = run_reference(&setup)?;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: scf.orbital_residual.unwrap_or(f64::INFINITY),
        });
    }
    let sampled = Sampling::new(
        &setup,
        &scf,
        options.mesh.map(|mesh| mesh.to_vec()),
        "dos_mesh",
    )?;
    let spins = if scf.unrestricted { 2 } else { 1 };
    let (groups, of_ao) = projection_groups(setup.cell.basis());
    let mut data = DosBands::default();
    for spin in 0..spins {
        if options.projected {
            let mut channel_energies = Vec::new();
            let mut channel_weights = Vec::new();
            for ik in 0..sampled.k_points.len() {
                let (fock, overlap) = sampled.matrices(&scf, spin, ik);
                let (e, w) = projected_bands(fock, overlap, &of_ao, groups.len())?;
                channel_energies.push(e);
                channel_weights.push(w);
            }
            data.energies.push(channel_energies);
            data.weights.push(channel_weights);
        } else {
            data.energies.push(sampled.energies(&scf, spin)?);
        }
    }
    let report = reference_report(
        &setup,
        &scf,
        "Gaussian-broadened density of states of the converged periodic Fock operator",
    )
    .with("chemical_potential", scf.chemical_potential.to_vec())
    .with(
        "reference_type",
        if scf.unrestricted { "uks" } else { "rks" },
    );
    dos_report(
        report,
        setup.cell.basis(),
        setup.cell.molecule().len(),
        sampled.k_points.len(),
        if scf.unrestricted { 1.0 } else { 2.0 },
        &options,
        data,
    )
}

pub(super) fn stationary_mesh_dos(
    mut report: Value,
    basis: &MolBasis,
    natoms: usize,
    overlaps: &[BlochMatrix],
    bands: &[Vec<crate::dfpt::orbitals::BandOrbitals>],
    spin_factor: f64,
    options: &DosOptions,
) -> Result<Value> {
    let (groups, of_ao) = projection_groups(basis);
    let mut data = DosBands::default();
    for channel in bands {
        if channel.len() != overlaps.len() {
            return Err(DftError::InvalidInput(
                "DOS mesh has inconsistent band and metric lengths".into(),
            ));
        }
        data.energies
            .push(channel.iter().map(|band| band.energies.clone()).collect());
        if options.projected {
            data.weights.push(
                channel
                    .iter()
                    .zip(overlaps)
                    .map(|(band, s)| {
                        projection_weights(&band.coefficients, s, &of_ao, groups.len())
                    })
                    .collect::<Result<_>>()?,
            );
        }
    }
    let (eigen, metric) = bands
        .iter()
        .flatten()
        .fold((0.0_f64, 0.0_f64), |(e, s), band| {
            (e.max(band.eigen_residual), s.max(band.metric_defect))
        });
    report.set("dos_eigen_residual", eigen);
    report.set("dos_metric_defect", metric);
    dos_report(
        report,
        basis,
        natoms,
        overlaps.len(),
        spin_factor,
        options,
        data,
    )
}

pub(super) fn dos_reference(
    model: &str,
    reference: &str,
    functional: &crate::xc::ResolvedFunctional,
    energy: f64,
    iterations: usize,
    residual: f64,
) -> Value {
    Value::object()
        .with("model", model)
        .with("reference_type", reference)
        .with("functional", functional.description.as_str())
        .with("converged", true)
        .with("energy", energy)
        .with("scf_iterations", iterations)
        .with("orbital_residual", residual)
        .with(
            "units",
            units(&[
                ("energy", "hartree per cell"),
                ("chemical_potential", "hartree"),
                ("scf_iterations", "dimensionless"),
                ("orbital_residual", "hartree"),
                ("dimension", "dimensionless"),
                ("k_mesh", "dimensionless"),
            ]),
        )
}

/// Separate spin spectra with the shared closed/open/virtual projectors
/// preserved. Rotations inside one block leave both reference densities
/// unchanged. Off-block Fock elements are not used as band eigenvalues.
pub(super) fn roks_semicanonical_bands(
    coefficients: &[BlochMatrix],
    fock: &[Vec<BlochMatrix>; 2],
    [alpha, beta]: [usize; 2],
) -> Result<[Vec<crate::dfpt::orbitals::BandOrbitals>; 2]> {
    use crate::dfpt::orbitals::{adjoint, generalized_orbitals, matmul, BandOrbitals};
    use crate::linalg::Matrix;
    let [up, down] = std::array::from_fn(|spin| {
        coefficients
            .iter()
            .zip(&fock[spin])
            .map(|(c, f)| {
                let fm = matmul(&adjoint(c)?, &matmul(f, c)?)?;
                let mut energies = Vec::with_capacity(c.cols());
                let mut orbitals = BlochMatrix::zeros(c.rows(), c.cols());
                let mut eigen_residual = 0.0_f64;
                let mut metric_defect = 0.0_f64;
                for (start, end) in [(0, beta), (beta, alpha), (alpha, c.cols())] {
                    let rank = end - start;
                    if rank == 0 {
                        continue;
                    }
                    let mut block = BlochMatrix::zeros(rank, rank);
                    let mut subspace = BlochMatrix::zeros(c.rows(), rank);
                    for i in 0..rank {
                        for j in 0..rank {
                            block.re[(i, j)] = fm.re[(start + i, start + j)];
                            block.im[(i, j)] = fm.im[(start + i, start + j)];
                        }
                        for row in 0..c.rows() {
                            subspace.re[(row, i)] = c.re[(row, start + i)];
                            subspace.im[(row, i)] = c.im[(row, start + i)];
                        }
                    }
                    block.make_hermitian();
                    let identity =
                        BlochMatrix::from_pair((Matrix::identity(rank), Matrix::zeros(rank, rank)));
                    let rotated = generalized_orbitals(&block, &identity, 1e-7)?;
                    let ao = matmul(&subspace, &rotated.coefficients)?;
                    for row in 0..c.rows() {
                        for col in 0..rank {
                            orbitals.re[(row, start + col)] = ao.re[(row, col)];
                            orbitals.im[(row, start + col)] = ao.im[(row, col)];
                        }
                    }
                    energies.extend(rotated.energies);
                    eigen_residual = eigen_residual.max(rotated.eigen_residual);
                    metric_defect = metric_defect.max(rotated.metric_defect);
                }
                Ok(BandOrbitals {
                    energies,
                    coefficients: orbitals,
                    eigen_residual,
                    metric_defect,
                })
            })
            .collect::<Result<Vec<_>>>()
    });
    Ok([up?, down?])
}

fn dos_report(
    mut report: Value,
    basis: &MolBasis,
    natoms: usize,
    k_count: usize,
    spin_factor: f64,
    options: &DosOptions,
    data: DosBands,
) -> Result<Value> {
    let sigma = options.sigma;
    let points = options.points;
    let projected = options.projected;
    let (groups, _) = projection_groups(basis);
    let energies = data.energies;
    let weights = data.weights;
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for &energy in energies.iter().flatten().flatten() {
        if !energy.is_finite() {
            return Err(DftError::InvalidInput(
                "DOS needs finite converged band energies".into(),
            ));
        }
        minimum = minimum.min(energy);
        maximum = maximum.max(energy);
    }
    if !minimum.is_finite() || k_count == 0 {
        return Err(DftError::InvalidInput(
            "DOS needs a non-empty band mesh".into(),
        ));
    }
    let [lo, hi] = options
        .window
        .unwrap_or([minimum - 5.0 * sigma, maximum + 5.0 * sigma]);
    let de = (hi - lo) / (points - 1) as f64;
    if !de.is_finite() || de <= 0.0 {
        return Err(DftError::InvalidInput(
            "DOS energy grid cannot be represented".into(),
        ));
    }
    let grid: Vec<f64> = (0..points).map(|i| lo + de * i as f64).collect();
    let nk = k_count as f64;
    let norm = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());
    if !norm.is_finite() || norm <= 0.0 {
        return Err(DftError::InvalidInput(
            "DOS Gaussian normalization cannot be represented".into(),
        ));
    }
    let state_weight = spin_factor / nk;
    let width = sigma * std::f64::consts::SQRT_2;
    let mut channels = Vec::new();
    for (spin, spin_energies) in energies.iter().enumerate() {
        let mut density = vec![0.0; points];
        let mut shells = vec![vec![0.0; points]; if projected { groups.len() } else { 0 }];
        let mut window_count = 0.0;
        let mut state_count = 0.0;
        for (ik, per_k) in spin_energies.iter().enumerate() {
            for (band, &e) in per_k.iter().enumerate() {
                state_count += state_weight;
                window_count += 0.5
                    * state_weight
                    * (libm::erf((hi - e) / width) - libm::erf((lo - e) / width));
                let begin = grid.partition_point(|&x| (x - e) / sigma <= -9.0);
                let end = grid.partition_point(|&x| (x - e) / sigma < 9.0);
                for i in begin..end {
                    let t = (grid[i] - e) / sigma;
                    let g = spin_factor * norm * (-0.5 * t * t).exp() / nk;
                    density[i] += g;
                    for (group, curve) in shells.iter_mut().enumerate() {
                        curve[i] += g * weights[spin][ik][band][group];
                    }
                }
            }
        }
        let mut integrated = vec![0.0; points];
        for i in 1..points {
            integrated[i] = integrated[i - 1] + 0.5 * de * (density[i] + density[i - 1]);
        }
        let mut channel = Value::object()
            .with("spin", spin)
            .with("state_count", state_count)
            .with("window_state_count", window_count)
            .with(
                "integration_residual",
                integrated[points - 1] - window_count,
            );
        if projected {
            let mut atoms = vec![vec![0.0; points]; natoms];
            let mut residual = 0.0_f64;
            for i in 0..points {
                let mut sum = 0.0;
                for (group, curve) in shells.iter().enumerate() {
                    atoms[groups[group].0][i] += curve[i];
                    sum += curve[i];
                }
                residual = residual.max((sum - density[i]).abs());
            }
            channel.set("projected_shells", shells);
            channel.set("projected_atoms", atoms);
            channel.set("projection_sum_residual", residual);
        }
        channel.set("dos", density);
        channel.set("integrated", integrated);
        channels.push(channel);
    }
    report.set("sigma", sigma);
    report.set("energies", grid);
    report.set("sampling_k_points", k_count);
    report.set("channels", channels);
    if projected {
        report.set(
            "projection_groups",
            groups
                .iter()
                .map(|(atom, l)| {
                    Value::object().with("atom", *atom).with("l", *l).with(
                        "label",
                        ANGULAR_LABELS.get(*l as usize).copied().unwrap_or("?"),
                    )
                })
                .collect::<Vec<Value>>(),
        );
    }
    let mut labels = match report.get("units") {
        Some(Value::Object(labels)) => labels.clone(),
        _ => Vec::new(),
    };
    for (key, unit) in [
        ("energies", "hartree"),
        (
            "channels.dos",
            "states/hartree per cell (both spins for a restricted reference)",
        ),
        ("channels.integrated", "states per cell"),
        (
            "channels.projected_shells",
            "states/hartree per cell, [group][point], Mulliken weights of projection_groups",
        ),
        (
            "channels.projected_atoms",
            "states/hartree per cell, [atom][point]",
        ),
        (
            "channels.projection_sum_residual",
            "states/hartree per cell",
        ),
        ("channels.state_count", "states per cell"),
        ("channels.window_state_count", "states per cell"),
        ("channels.integration_residual", "states per cell"),
        ("dos_eigen_residual", "dimensionless"),
        ("dos_metric_defect", "dimensionless"),
        ("sigma", "hartree"),
    ] {
        labels.push((key.into(), Value::Str(unit.into())));
    }
    report.set("units", Value::Object(labels));
    Ok(report)
}

/// Bare Pauli spin susceptibility, Fermi-level density of states and
/// electronic heat capacity of the rigid Kohn-Sham bands.
pub(super) fn pauli_spin_susceptibility(system: &System, controls: &Controls) -> Result<Value> {
    let method = PeriodicMethod::from_controls(controls)?;
    let temperatures = controls.opt_f64_list("temperatures_k")?;
    let fermi_mesh = controls.opt_usize_list("fermi_k_mesh")?;
    if controls.bool("exchange_enhanced", false)? {
        return enhanced_spin_susceptibility(system, controls, method, temperatures, fermi_mesh);
    }
    let g = controls.f64("spin_g_factor", 2.0)?;
    controls.finish("pauli-spin-susceptibility")?;
    let temperatures = match temperatures {
        Some(list) => list,
        None if method.electronic_temperature > 0.0 => vec![method.electronic_temperature],
        None => {
            return Err(DftError::InvalidInput(
                "pauli-spin-susceptibility needs temperatures_k (Kelvin) for a \
                 zero-temperature reference"
                    .into(),
            ))
        }
    };
    if temperatures.is_empty()
        || temperatures.iter().any(|t| !t.is_finite() || !(*t > 0.0))
        || !g.is_finite()
        || g <= 0.0
    {
        return Err(DftError::InvalidInput(
            "temperatures_k and spin_g_factor must be positive and finite".into(),
        ));
    }
    let setup = prepare_periodic(system, &method)?;
    let scf = run_reference(&setup)?;
    let sampled = Sampling::new(&setup, &scf, fermi_mesh, "fermi_k_mesh")?;
    let energies: [Vec<f64>; 2] = [
        sampled.energies(&scf, 0)?.into_iter().flatten().collect(),
        sampled.energies(&scf, 1)?.into_iter().flatten().collect(),
    ];
    let weight = 1.0 / sampled.k_points.len() as f64;
    let electrons = (scf.n_alpha + scf.n_beta) as f64;
    let volume = setup.cell.lattice().volume().abs();
    let alpha2 = FINE_STRUCTURE * FINE_STRUCTURE;
    let bohr3_to_cm3 = (BOHR_TO_METER * 100.0).powi(3);
    let gas_constant = AVOGADRO_PER_MOL * BOLTZMANN_J_PER_K;
    let pi = std::f64::consts::PI;
    let mut rows = Vec::new();
    for &t in &temperatures {
        let kt = t * BOLTZMANN_HARTREE_PER_K;
        let spin =
            collinear_fermi_thermodynamics([&energies[0], &energies[1]], weight, electrons, kt, g)?;
        let r = spin.thermal;
        // mu_0 = 4 pi alpha^2 in atomic units. The Gaussian volume
        // susceptibility is the SI value divided by 4 pi.
        let volume_si = 4.0 * pi * alpha2 * spin.susceptibility / volume;
        let volume_cgs = volume_si / (4.0 * pi);
        let molar_cgs = volume_cgs * volume * bohr3_to_cm3 * AVOGADRO_PER_MOL;
        let heat_capacity_molar = r.heat_capacity * gas_constant;
        let sommerfeld_dos = pi * pi / 3.0 * BOLTZMANN_HARTREE_PER_K * r.dos * gas_constant;
        let wilson = if r.heat_capacity > 0.0 {
            Some(pi * pi / 3.0 * kt * spin.spin_constrained_dos / r.heat_capacity)
        } else {
            None
        };
        rows.push(
            Value::object()
                .with("temperature_k", t)
                .with("chemical_potential", r.chemical_potential)
                .with("spin_chemical_potentials", [r.chemical_potential; 2])
                .with("spin_electrons", spin.spin_electrons)
                .with("spin_dos_at_fermi_level", spin.spin_dos)
                .with("fixed_number_spin_dos", spin.spin_constrained_dos)
                .with("rigid_band_spin_magnetization", spin.magnetization)
                .with(
                    "chemical_potential_response",
                    spin.chemical_potential_response,
                )
                .with("delta_spin_numbers", spin.delta_spin_numbers)
                .with(
                    "raw_charge_constraint_residual",
                    spin.charge_constraint_residual,
                )
                .with("thermal_number_residual", (r.electrons - electrons).abs())
                .with("electrons", r.electrons)
                .with("dos_at_fermi_level", r.dos)
                .with("dos_at_fermi_level_per_ev", r.dos / HARTREE_TO_EV)
                .with("pauli_susceptibility_per_cell", spin.susceptibility)
                .with("pauli_susceptibility_volume_si", volume_si)
                .with("pauli_susceptibility_volume_cgs", volume_cgs)
                .with("pauli_susceptibility_molar_cgs", molar_cgs)
                .with("heat_capacity", r.heat_capacity)
                .with("heat_capacity_j_mol_k", heat_capacity_molar)
                .with(
                    "sommerfeld_coefficient_mj_mol_k2",
                    1000.0 * heat_capacity_molar / t,
                )
                .with("sommerfeld_from_dos_mj_mol_k2", 1000.0 * sommerfeld_dos)
                .with("wilson_ratio", wilson)
                .with("entropy", r.entropy)
                .with("dmu_dkt", r.dmu_dkt)
                .with("band_energy", r.band_energy),
        );
    }
    Ok(reference_report(
        &setup,
        &scf,
        "bare fixed-total-number longitudinal spin susceptibility and electronic thermodynamics of rigid Kohn-Sham spin bands reoccupied with one common chemical potential and zero spin reservoir bias; orbital reference and its separate spin constraints are frozen, no self-consistent exchange enhancement",
    )
    .with("exchange_enhanced", false)
    .with("spin_g_factor", g)
    .with("thermodynamic_spin_bias", 0.0)
    .with("orbital_reference_spin_chemical_potentials", scf.chemical_potential.to_vec())
    .with("sampling_k_points", sampled.k_points.len())
    .with("volume", volume)
    .with("rows", rows)
    .with(
        "units",
        units(&[
            ("rows.temperature_k", "K"),
            ("rows.chemical_potential", "hartree"),
            ("spin_g_factor", "dimensionless"),
            ("thermodynamic_spin_bias", "hartree, common reservoir has zero spin bias"),
            ("orbital_reference_spin_chemical_potentials", "hartree, frozen orbital reference alpha and beta"),
            ("rows.spin_chemical_potentials", "hartree, common reservoir alpha and beta"),
            ("rows.spin_electrons", "electrons per cell, alpha and beta"),
            ("rows.spin_dos_at_fermi_level", "states/hartree per cell, alpha and beta"),
            ("rows.fixed_number_spin_dos", "states/hartree per cell, fixed-number longitudinal response"),
            ("rows.rigid_band_spin_magnetization", "atomic units of spin moment per cell"),
            ("rows.chemical_potential_response", "hartree, per atomic magnetic field; null when indeterminate"),
            ("rows.delta_spin_numbers", "atomic units of electron count per magnetic field, alpha and beta"),
            ("rows.raw_charge_constraint_residual", "atomic units of electron count per magnetic field"),
            ("rows.thermal_number_residual", "electrons per cell"),
            ("rows.electrons", "electrons per cell"),
            (
                "rows.dos_at_fermi_level",
                "states/hartree per cell, both spins, thermally averaged",
            ),
            ("rows.dos_at_fermi_level_per_ev", "states per eV per cell"),
            (
                "rows.pauli_susceptibility_per_cell",
                "atomic units of spin moment per magnetic field per cell",
            ),
            (
                "rows.pauli_susceptibility_volume_si",
                "dimensionless SI volume susceptibility",
            ),
            (
                "rows.pauli_susceptibility_volume_cgs",
                "dimensionless Gaussian volume susceptibility (emu/cm^3)",
            ),
            (
                "rows.pauli_susceptibility_molar_cgs",
                "cm^3 per mole of cells (emu/mol)",
            ),
            ("rows.heat_capacity", "k_B per cell, fixed electron count"),
            ("rows.heat_capacity_j_mol_k", "J/(mol K), per mole of cells"),
            (
                "rows.sommerfeld_coefficient_mj_mol_k2",
                "mJ/(mol K^2): heat capacity over temperature",
            ),
            (
                "rows.sommerfeld_from_dos_mj_mol_k2",
                "mJ/(mol K^2): (pi^2/3) k_B^2 N(E_F)",
            ),
            (
                "rows.wilson_ratio",
                "dimensionless: (pi^2/3) kT times the fixed-number spin DOS over the heat capacity in k_B",
            ),
            ("rows.entropy", "k_B per cell"),
            ("rows.dmu_dkt", "dimensionless, fixed electron count"),
            (
                "rows.band_energy",
                "hartree per cell: sum of occupied band energies",
            ),
            ("volume", "bohr^3"),
        ]),
    ))
}

/// Each requested enhanced-response temperature has its own stationary
/// thermal density. A dense Fermi mesh becomes the SCF/response mesh, rather
/// than applying a self-consistent kernel to unrelated frozen occupations.
fn enhanced_spin_susceptibility(
    system: &System,
    controls: &Controls,
    method: PeriodicMethod,
    temperatures: Option<Vec<f64>>,
    fermi_mesh: Option<Vec<usize>>,
) -> Result<Value> {
    use crate::periodic::spin_susceptibility::longitudinal_spin_susceptibility;
    let config = super::vibrational::response_config(controls)?;
    // A polarized spin perturbation can redistribute charge locally even
    // when its cell integral vanishes, so the Hartree sum must be resolved.
    let default_cutoff = if method.unrestricted || method.multiplicity != 1 {
        16.0
    } else {
        6.0
    };
    let cutoff = controls.f64("response_reciprocal_cutoff", default_cutoff)?;
    let g = controls.f64("spin_g_factor", 2.0)?;
    controls.finish("pauli-spin-susceptibility")?;
    let temperatures = temperatures.unwrap_or_else(|| vec![method.electronic_temperature]);
    if temperatures.is_empty()
        || temperatures.iter().any(|t| !t.is_finite() || *t <= 0.0)
        || !g.is_finite()
        || g <= 0.0
        || !cutoff.is_finite()
        || cutoff <= 0.0
    {
        return Err(DftError::InvalidInput("enhanced spin response needs positive finite temperatures, spin_g_factor and response_reciprocal_cutoff".into()));
    }
    if fermi_mesh
        .as_ref()
        .is_some_and(|m| m.len() != 3 || m.contains(&0))
    {
        return Err(DftError::InvalidInput(
            "fermi_k_mesh takes three positive integers".into(),
        ));
    }
    let mut rows = Vec::new();
    let mut report = None;
    let mut all_converged = true;
    let mut warnings = Vec::new();
    for t in temperatures {
        let mut thermal_method = method.clone();
        thermal_method.electronic_temperature = t;
        if let Some(mesh) = &fermi_mesh {
            thermal_method.k_mesh = [mesh[0], mesh[1], mesh[2]];
        }
        let setup = prepare_periodic(system, &thermal_method)?;
        let scf = run_reference(&setup)?;
        let result = longitudinal_spin_susceptibility(
            &setup.cell,
            &setup.functional,
            &scf,
            &setup.options,
            &config,
            cutoff,
            g,
        )?;
        let volume = setup.cell.lattice().volume().abs();
        let kt = t * BOLTZMANN_HARTREE_PER_K;
        let weight = 1.0 / scf.k_points.len() as f64;
        let mut spin_dos = [0.0; 2];
        let mut spin_numbers = [0.0; 2];
        for spin in 0..2 {
            spin_numbers[spin] = weight * scf.occupations[spin].iter().flatten().sum::<f64>();
            if let Some(mu) = scf.chemical_potential[spin] {
                for &energy in scf.mo_energy[spin].iter().flatten() {
                    let (_, variance, _) =
                        crate::periodic::fermi_thermodynamics::level((energy - mu) / kt);
                    spin_dos[spin] += weight * variance / kt;
                }
            }
        }
        let dos = spin_dos[0] + spin_dos[1];
        let bare_oracle = if dos > 0.0 {
            (g / 4.0).powi(2) * 4.0 * spin_dos[0] * (spin_dos[1] / dos)
        } else {
            0.0
        };
        let value = &result.value;
        let volume_si =
            4.0 * std::f64::consts::PI * FINE_STRUCTURE.powi(2) * value.susceptibility / volume;
        rows.push(
            Value::object()
                .with("temperature_k", t)
                .with(
                    "chemical_potential",
                    scf.chemical_potential[0]
                        .zip(scf.chemical_potential[1])
                        .map(|(a, b)| 0.5 * (a + b)),
                )
                .with("spin_chemical_potentials", scf.chemical_potential.to_vec())
                .with("spin_dos_at_fermi_level", spin_dos.to_vec())
                .with("electrons", spin_numbers.iter().sum::<f64>())
                .with("dos_at_fermi_level", dos)
                .with("pauli_susceptibility_per_cell", value.bare_susceptibility)
                .with(
                    "enhanced_spin_susceptibility_per_cell",
                    value.susceptibility,
                )
                .with("enhanced_spin_susceptibility_volume_si", volume_si)
                .with("enhancement_factor", value.enhancement_factor)
                .with(
                    "bare_dos_consistency_residual",
                    (value.bare_susceptibility - bare_oracle).abs(),
                )
                .with("delta_spin_numbers", value.delta_numbers)
                .with(
                    "chemical_potential_response",
                    value.chemical_potential_response,
                )
                .with(
                    "bare_chemical_potential_response",
                    value.bare_chemical_potential_response,
                )
                .with("charge_compressibility", value.charge_compressibility)
                .with(
                    "bare_charge_compressibility",
                    value.bare_charge_compressibility,
                )
                .with("reference_spin_bias", value.reference_spin_bias)
                .with(
                    "unconstrained_delta_spin_numbers",
                    value.unconstrained_delta_numbers,
                )
                .with(
                    "bare_charge_constraint_residual",
                    value.bare_charge_constraint_residual,
                )
                .with(
                    "constrained_equation_residual",
                    value.constrained_equation_residual,
                )
                .with(
                    "raw_charge_constraint_residual",
                    value.charge_constraint_residual,
                )
                .with(
                    "raw_spin_density_cancellation_residual",
                    value.spin_density_cancellation_residual,
                )
                .with(
                    "raw_density_hermitian_residual",
                    value.density_hermitian_residual,
                )
                .with(
                    "nonnegative_susceptibility",
                    value.nonnegative_susceptibility,
                )
                .with("response_converged", result.converged)
                .with("response_residuals", result.residuals)
                .with("response_iterations", result.iterations)
                .with("reference_eigen_residual", result.reference.eigen_residual)
                .with(
                    "reference_orbital_residual",
                    result.reference.orbital_residual,
                )
                .with("reference_energy", scf.energy)
                .with("reference_free_energy", scf.free_energy)
                .with("reference_iterations", scf.iterations)
                .with("sampling_k_points", scf.k_points.len()),
        );
        all_converged &= result.converged && scf.converged;
        warnings.extend(result.warnings);
        if report.is_none() {
            report = Some(reference_report(&setup, &scf,
                "static exchange-enhanced spin-only longitudinal response of collinear finite-temperature Kohn-Sham references; full semilocal spin XC kernel; each temperature and sampling mesh has a separate self-consistent reference; common chemical-potential response fixes total particle number at fixed baseline spin bias, spin populations can exchange particles")
                .with("volume", volume));
        }
    }
    let mut report = report.expect("non-empty validated temperature list");
    report.set("converged", all_converged);
    report.set("exchange_enhanced", true);
    report.set("spin_g_factor", g);
    report.set("response_reciprocal_cutoff", cutoff);
    report.set("rows", rows);
    report.set("warnings", warnings);
    report.set("spin_convention", "M=-g(N_alpha-N_beta)/4; F_alpha^B=+g S/4 and F_beta^B=-g S/4; response per atomic magnetic field; no orbital or SOC contribution");
    report.set(
        "units",
        units(&[
            ("volume", "bohr^3"),
            ("spin_g_factor", "dimensionless"),
            ("response_reciprocal_cutoff", "1/bohr"),
            ("rows.temperature_k", "K"),
            ("rows.chemical_potential", "hartree"),
            ("rows.spin_chemical_potentials", "hartree, alpha and beta"),
            ("rows.spin_dos_at_fermi_level", "states/hartree per cell, alpha and beta"),
            ("rows.reference_spin_bias", "hartree, (mu_alpha-mu_beta)/2 held fixed"),
            ("rows.chemical_potential_response", "hartree, per atomic magnetic field"),
            ("rows.bare_chemical_potential_response", "hartree, per atomic magnetic field; null when indeterminate"),
            ("rows.charge_compressibility", "electrons/hartree per cell"),
            ("rows.bare_charge_compressibility", "electrons/hartree per cell"),
            ("rows.unconstrained_delta_spin_numbers", "atomic units of electron count per magnetic field, alpha and beta before the common charge multiplier"),
            ("rows.bare_charge_constraint_residual", "atomic units of electron count per magnetic field"),
            ("rows.constrained_equation_residual", "dimensionless relative L2; absolute L2 for an exactly zero source"),
            ("rows.electrons", "electrons per cell"),
            (
                "rows.dos_at_fermi_level",
                "states/hartree per cell, both spins",
            ),
            (
                "rows.pauli_susceptibility_per_cell",
                "atomic units of spin moment per magnetic field per cell",
            ),
            (
                "rows.enhanced_spin_susceptibility_per_cell",
                "atomic units of spin moment per magnetic field per cell",
            ),
            (
                "rows.enhanced_spin_susceptibility_volume_si",
                "dimensionless SI volume susceptibility",
            ),
            (
                "rows.enhancement_factor",
                "dimensionless; null when the bare response vanishes",
            ),
            (
                "rows.bare_dos_consistency_residual",
                "atomic units of spin moment per magnetic field per cell",
            ),
            (
                "rows.delta_spin_numbers",
                "atomic units of electron count per magnetic field, alpha and beta",
            ),
            (
                "rows.raw_charge_constraint_residual",
                "atomic units of electron count per magnetic field",
            ),
            (
                "rows.raw_spin_density_cancellation_residual",
                "atomic units of AO density per magnetic field",
            ),
            (
                "rows.raw_density_hermitian_residual",
                "atomic units of AO density per magnetic field",
            ),
            ("rows.response_residuals", "dimensionless"),
            ("rows.reference_eigen_residual", "hartree"),
            ("rows.reference_orbital_residual", "hartree"),
            ("rows.reference_energy", "hartree"),
            ("rows.reference_free_energy", "hartree"),
        ]),
    );
    Ok(report)
}

#[cfg(test)]
mod dos_tests {
    use super::roks_semicanonical_bands;
    use crate::dfpt::orbitals::band_density;
    use crate::linalg::Matrix;
    use crate::periodic::BlochMatrix;

    #[test]
    fn roks_semicanonical_dos_retains_spin_projectors_and_uses_only_block_eigenvalues() {
        let n = 5;
        let coefficients = [BlochMatrix::from_pair((
            Matrix::identity(n),
            Matrix::zeros(n, n),
        ))];
        let mut fock = BlochMatrix::zeros(n, n);
        for (i, energy) in [-2.0, -2.0, 0.35, 2.0, 2.0].into_iter().enumerate() {
            fock.re[(i, i)] = energy;
        }
        for (i, j, re, im) in [
            (0, 1, 0.3, 0.4),
            (3, 4, 0.15, -0.2),
            (1, 2, 3.7, 1.9),
            (0, 3, -4.2, 2.4),
        ] {
            fock.re[(i, j)] = re;
            fock.re[(j, i)] = re;
            fock.im[(i, j)] = im;
            fock.im[(j, i)] = -im;
        }
        let mut beta = fock.clone();
        beta.scale(0.5);
        let spectra =
            roks_semicanonical_bands(&coefficients, &[vec![fock], vec![beta]], [3, 2]).unwrap();
        for (spin, occupied) in [3, 2].into_iter().enumerate() {
            let band = &spectra[spin][0];
            let scale = if spin == 0 { 1.0 } else { 0.5 };
            for (actual, expected) in band.energies.iter().zip([-2.5, -1.5, 0.35, 1.75, 2.25]) {
                assert!((actual - scale * expected).abs() < 1e-13);
            }
            let weights: Vec<_> = (0..n).map(|i| f64::from(i < occupied)).collect();
            let density = band_density(band, &weights).unwrap();
            for (i, weight) in weights.into_iter().enumerate() {
                for j in 0..n {
                    assert!((density.re[(i, j)] - f64::from(i == j) * weight).abs() < 1e-13);
                    assert!(density.im[(i, j)].abs() < 1e-13);
                }
            }
        }
    }
}
