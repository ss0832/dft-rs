// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Natural bond orbital analysis on the common driver: natural atomic
//! orbitals and natural population analysis, natural electron
//! configurations, Wiberg bond indices, the natural Lewis structure with its
//! hybrids, natural localized molecular orbitals and the second-order
//! donor-acceptor energies. Open shells (different spin densities) are
//! analysed per spin with at most one electron per orbital
//! (`nbo_spin = auto`, the default); `total` and `separate` force one
//! treatment.
//!
//! `nbo_density` selects the analysed density: `scf` (the converged
//! reference), `casscf` (the state-averaged active-space density of the
//! `casscf` controls, or the state `nbo_cas_state`) and `pt2_relaxed` (the
//! relaxed density of a double hybrid or spin-component PT2); `auto`, the
//! default, takes `pt2_relaxed` for double-hybrid requests, `casscf` when
//! `cas` is given and `scf` otherwise. The second-order energies of a
//! correlated density use the Hartree-Fock operator of that density,
//! `F_s = h + J[D] - K[D_s]`.

use super::controls::Controls;
use super::method::{
    run_collinear_property_reference, run_molecular_reference, CollinearPropertyReference,
};
use super::molecular::{reference_report, setup, units};
use super::value::Value;
use super::System;
use crate::integrals::eri::EriTensor;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::properties::natural_bond::{
    ecp_core_electrons, natural_bond_from_density, shell_layout, NaturalBondAnalysis, NboChannel,
    NboOptions, NboSpin,
};
use crate::{DftError, Result};

fn channel_value(channel: &NboChannel, nao_coefficients: &Matrix) -> Value {
    let nbos: Vec<Value> = channel
        .nbos
        .iter()
        .map(|nbo| {
            let hybrids: Vec<Value> = nbo
                .hybrids
                .iter()
                .map(|h| {
                    Value::object()
                        .with("atom", h.atom)
                        .with("coefficient", h.coefficient)
                        .with("composition", h.composition.clone())
                })
                .collect();
            Value::object()
                .with("label", nbo.label.as_str())
                .with("type", nbo.kind.name())
                .with("lewis", nbo.kind.is_lewis())
                .with("atoms", nbo.atoms.clone())
                .with("serial", nbo.serial)
                .with("occupancy", nbo.occupancy)
                .with("energy", nbo.energy)
                .with("hybrids", hybrids)
        })
        .collect();
    let trials: Vec<Value> = channel
        .trials
        .iter()
        .map(|t| {
            Value::object()
                .with("threshold", t.threshold)
                .with("lewis_count", t.lewis_count)
                .with("non_lewis_occupancy", t.non_lewis_occupancy)
                .with("dependent", t.dependent)
        })
        .collect();
    let perturbation: Vec<Value> = channel
        .perturbation
        .iter()
        .map(|p| {
            Value::object()
                .with("donor", p.donor)
                .with("acceptor", p.acceptor)
                .with("donor_label", channel.nbos[p.donor].label.as_str())
                .with("acceptor_label", channel.nbos[p.acceptor].label.as_str())
                .with("stabilization", p.stabilization)
                .with("fock_element", p.fock_element)
                .with("energy_gap", p.energy_gap)
        })
        .collect();
    let nlmos: Vec<Value> = channel
        .nlmos
        .iter()
        .map(|m| {
            Value::object()
                .with("parent", m.parent)
                .with("parent_label", channel.nbos[m.parent].label.as_str())
                .with("occupancy", m.occupancy)
                .with("parent_percentage", m.parent_percentage)
                .with("atomic_percentages", m.atomic_percentages.clone())
        })
        .collect();
    Value::object()
        .with("spin", channel.spin.as_str())
        .with("maximum_occupancy", channel.maximum_occupancy)
        .with("electrons", channel.electrons)
        .with("expected_lewis_count", channel.expected_lewis_count)
        .with("lewis_threshold", channel.lewis_threshold)
        .with("trials", trials)
        .with("lewis_count", channel.lewis_count)
        .with("lewis_occupancy", channel.lewis_occupancy)
        .with("non_lewis_occupancy", channel.non_lewis_occupancy)
        .with("nbos", nbos)
        .with(
            "nbo_coefficients",
            nao_coefficients.matmul(&channel.coefficients),
        )
        .with("orthonormality_residual", channel.orthonormality_residual)
        .with("perturbation", perturbation)
        .with("perturbation_total", channel.perturbation_total)
        .with("nonpositive_gaps", channel.nonpositive_gaps)
        .with("nlmos", nlmos)
        .with(
            "nlmo_coefficients",
            nao_coefficients.matmul(&channel.nlmo_coefficients),
        )
        .with("nlmo_residual", channel.nlmo_residual)
        .with("nlmo_sweeps", channel.nlmo_sweeps)
}

/// Unit labels of the analysis entries.
const NBO_UNITS: &[(&str, &str)] = &[
    ("nao_occupancies", "electrons"),
    ("nao_spin_densities", "electrons (alpha minus beta)"),
    ("nao_coefficients", "dimensionless, AO rows and NAO columns"),
    ("nao_orthonormality_residual", "dimensionless"),
    ("natural_charges", "e"),
    ("natural_populations", "electrons"),
    ("natural_spin_populations", "electrons (alpha minus beta)"),
    ("core_populations", "electrons"),
    ("valence_populations", "electrons"),
    ("rydberg_populations", "electrons"),
    ("natural_subshells.occupancy", "electrons"),
    ("wiberg_bond_indices", "dimensionless"),
    ("wiberg_atom_totals", "dimensionless"),
    ("density_dipole", "e*bohr, about the coordinate origin"),
    ("density_electron_count", "electrons"),
    ("channels.maximum_occupancy", "electrons"),
    ("channels.electrons", "electrons"),
    ("channels.lewis_threshold", "electrons"),
    ("channels.trials.threshold", "electrons"),
    ("channels.trials.non_lewis_occupancy", "electrons"),
    ("channels.lewis_occupancy", "electrons"),
    ("channels.non_lewis_occupancy", "electrons"),
    ("channels.nbos.occupancy", "electrons"),
    ("channels.nbos.energy", "hartree"),
    (
        "channels.nbos.hybrids.coefficient",
        "dimensionless polarization coefficient",
    ),
    (
        "channels.nbos.hybrids.composition",
        "percent s, p, d, f, ... character",
    ),
    (
        "channels.nbo_coefficients",
        "dimensionless, AO rows and NBO columns",
    ),
    ("channels.orthonormality_residual", "dimensionless"),
    ("channels.perturbation.stabilization", "hartree"),
    ("channels.perturbation.fock_element", "hartree"),
    ("channels.perturbation.energy_gap", "hartree"),
    ("channels.perturbation_total", "hartree"),
    ("channels.nlmos.occupancy", "electrons"),
    ("channels.nlmos.parent_percentage", "percent"),
    ("channels.nlmos.atomic_percentages", "percent per atom"),
    (
        "channels.nlmo_coefficients",
        "dimensionless, AO rows and NLMO columns",
    ),
    ("channels.nlmo_residual", "electrons"),
];

const NRT_UNITS: &[(&str, &str)] = &[
    ("channels.resonance.structures.weight", "dimensionless"),
    (
        "channels.resonance.structures.deviation",
        "dimensionless, ||D - Gamma||/||D||",
    ),
    ("channels.resonance.structures.formal_charges", "e"),
    (
        "channels.resonance.bond_orders",
        "dimensionless, bond units of the channel",
    ),
    ("channels.resonance.valencies", "dimensionless"),
    ("channels.resonance.fractional_deviation", "dimensionless"),
    ("channels.resonance.reference_deviation", "dimensionless"),
    ("channels.resonance.optimality_residual", "dimensionless"),
    (
        "resonance_bond_orders",
        "dimensionless, electron-pair bonds",
    ),
];

/// The analysis entries of a report.
fn with_analysis(
    report: Value,
    analysis: &NaturalBondAnalysis,
    density_dipole: [f64; 3],
    electrons: f64,
) -> Value {
    let nao = &analysis.nao;
    let subshells: Vec<Value> = nao
        .configurations
        .iter()
        .map(|list| {
            Value::Array(
                list.iter()
                    .map(|s| {
                        Value::object()
                            .with("n", s.n)
                            .with("l", s.l)
                            .with("type", s.kind.name())
                            .with("occupancy", s.occupancy)
                    })
                    .collect(),
            )
        })
        .collect();
    let channels: Vec<Value> = analysis
        .channels
        .iter()
        .map(|c| channel_value(c, &nao.nao_coefficients))
        .collect();
    let mode = if analysis.channels.len() == 2 {
        "separate"
    } else {
        "total"
    };
    report
        .with("open_shell", analysis.open_shell)
        .with("nbo_spin", mode)
        .with(
            "nao_labels",
            nao.labels.iter().map(|l| l.text()).collect::<Vec<_>>(),
        )
        .with(
            "nao_atoms",
            nao.labels.iter().map(|l| l.atom).collect::<Vec<_>>(),
        )
        .with("nao_l", nao.labels.iter().map(|l| l.l).collect::<Vec<_>>())
        .with(
            "nao_types",
            nao.labels.iter().map(|l| l.kind.name()).collect::<Vec<_>>(),
        )
        .with("nao_occupancies", nao.occupations.clone())
        .with("nao_spin_densities", nao.spin_densities.clone())
        .with("nao_coefficients", nao.nao_coefficients.clone())
        .with("nao_orthonormality_residual", nao.orthonormality_residual)
        .with("natural_charges", nao.atomic_charges.clone())
        .with("natural_populations", nao.atomic_populations.clone())
        .with("natural_spin_populations", nao.spin_populations.clone())
        .with("core_populations", nao.core_populations.clone())
        .with("valence_populations", nao.valence_populations.clone())
        .with("rydberg_populations", nao.rydberg_populations.clone())
        .with(
            "natural_electron_configurations",
            nao.configuration_text.clone(),
        )
        .with("natural_subshells", subshells)
        .with("wiberg_bond_indices", nao.wiberg_bond_indices.clone())
        .with("wiberg_atom_totals", nao.wiberg_atom_totals.clone())
        .with("density_dipole", density_dipole)
        .with("density_electron_count", electrons)
        .with("channels", channels)
}

/// Natural resonance analysis of every channel: `resonance` entries of the
/// channels and, for two spin channels, the spin-averaged NRT bond orders.
fn with_resonance(
    report: Value,
    analysis: &NaturalBondAnalysis,
    numbers: &[u8],
    fock: &[Matrix; 2],
    max_structures: usize,
) -> Result<Value> {
    use crate::properties::natural_bond::resonance::natural_resonance;
    let nao = &analysis.nao;
    let c = &nao.nao_coefficients;
    let to_nao = |f: &Matrix| c.transpose_matmul(&f.matmul(c));
    let natoms = numbers.len();
    let mut values = Vec::new();
    let mut summed = Matrix::zeros(natoms, natoms);
    let channels = report
        .get("channels")
        .and_then(Value::as_array)
        .map(<[Value]>::to_vec)
        .unwrap_or_default();
    for (index, (channel, value)) in analysis.channels.iter().zip(channels).enumerate() {
        let (density, f, q) = if analysis.channels.len() == 2 {
            (&nao.spin_density_nao[index], to_nao(&fock[index]), 1.0)
        } else {
            let mut average = to_nao(&fock[0]);
            for (x, y) in average
                .as_mut_slice()
                .iter_mut()
                .zip(to_nao(&fock[1]).as_slice())
            {
                *x = 0.5 * (*x + y);
            }
            (&nao.density_nao, average, 2.0)
        };
        let r = natural_resonance(nao, numbers, density, Some(&f), q, channel, max_structures)?;
        for (x, y) in summed
            .as_mut_slice()
            .iter_mut()
            .zip(r.bond_orders.as_slice())
        {
            *x += 0.5 * q * y;
        }
        let structures: Vec<Value> = r
            .structures
            .iter()
            .map(|s| {
                Value::object()
                    .with("weight", s.weight)
                    .with("deviation", s.deviation)
                    .with("lone_pairs", s.lone_pairs.clone())
                    .with(
                        "bonds",
                        s.bonds
                            .iter()
                            .map(|&(a, b, o)| {
                                Value::object().with("atoms", vec![a, b]).with("order", o)
                            })
                            .collect::<Vec<_>>(),
                    )
                    .with("formal_charges", s.formal_charges.clone())
            })
            .collect();
        values.push(
            value.with(
                "resonance",
                Value::object()
                    .with("structures", structures)
                    .with("bond_orders", r.bond_orders.clone())
                    .with("valencies", r.valencies.clone())
                    .with("fractional_deviation", r.fractional_deviation)
                    .with("reference_deviation", r.reference_deviation)
                    .with("enumerated", r.enumerated)
                    .with("evaluated", r.evaluated)
                    .with("skipped", r.skipped)
                    .with("truncated", r.truncated)
                    .with("optimality_residual", r.optimality_residual)
                    .with("degenerate_directions", r.degenerate_directions),
            ),
        );
    }
    Ok(report
        .with("channels", values)
        .with("resonance_bond_orders", summed))
}

/// Dipole of the density `p` (AO, spin-summed) and of the effective nuclear
/// charges about the coordinate origin, and its electron count.
fn density_dipole(
    setup: &super::method::MolecularSetup,
    p: &Matrix,
    overlap: &Matrix,
) -> ([f64; 3], f64) {
    let r = crate::integrals::one_electron::dipole_matrices(&setup.basis, Vec3::zero());
    let charges = setup.basis.effective_charges(&setup.molecule);
    let dipole = std::array::from_fn(|a| {
        let nuclear: f64 = setup
            .molecule
            .atoms
            .iter()
            .zip(&charges)
            .map(|(atom, z)| z * atom.position.get(a))
            .sum();
        nuclear - p.frobenius_dot(&r[a])
    });
    (dipole, p.frobenius_dot(overlap))
}

/// `F_s = h + J[D] - K[D_s]` of a correlated density.
fn density_fock(h: &Matrix, eri: &EriTensor, density: &[Matrix; 2]) -> [Matrix; 2] {
    let mut total = density[0].clone();
    for (x, y) in total.as_mut_slice().iter_mut().zip(density[1].as_slice()) {
        *x += y;
    }
    let (j, _) = crate::mcscf::casscf::coulomb_exchange(eri, &total);
    std::array::from_fn(|s| {
        let (_, k) = crate::mcscf::casscf::coulomb_exchange(eri, &density[s]);
        let mut f = h.clone();
        for ((x, a), b) in f
            .as_mut_slice()
            .iter_mut()
            .zip(j.as_slice())
            .zip(k.as_slice())
        {
            *x += a - b;
        }
        f
    })
}

fn analyse(
    setup: &super::method::MolecularSetup,
    overlap: &Matrix,
    density: &[Matrix; 2],
    fock: &[Matrix; 2],
    spin: NboSpin,
    options: &NboOptions,
) -> Result<NaturalBondAnalysis> {
    let numbers: Vec<u8> = setup.molecule.atoms.iter().map(|a| a.z).collect();
    natural_bond_from_density(
        overlap,
        [&density[0], &density[1]],
        Some([&fock[0], &fock[1]]),
        &shell_layout(&setup.basis),
        &numbers,
        &ecp_core_electrons(&setup.basis, numbers.len()),
        &setup.basis.effective_charges(&setup.molecule),
        spin,
        options,
    )
}

fn spin_summed(density: &[Matrix; 2]) -> Matrix {
    let mut p = density[0].clone();
    for (x, y) in p.as_mut_slice().iter_mut().zip(density[1].as_slice()) {
        *x += y;
    }
    p
}

pub(super) struct SpinorRequest {
    spin: NboSpin,
    options: NboOptions,
    resonance: bool,
    max_structures: usize,
    spin_density: Option<SpinDensityRequest>,
    natural_orbitals: bool,
}

const NATURAL_ORBITAL_UNITS: &[(&str, &str)] = &[
    ("natural_orbitals.occupations", "electrons"),
    ("natural_orbitals.ao_coefficients", "dimensionless"),
    ("natural_orbitals.trace_residual", "electrons"),
    ("natural_orbitals.reconstruction_residual", "electrons"),
    ("natural_orbitals.occupation_bound_defect", "electrons"),
    ("natural_orbitals.effectively_unpaired_weights", "electrons"),
    (
        "natural_orbitals.effectively_unpaired_electrons",
        "electrons",
    ),
];

const CORRESPONDING_ORBITAL_UNITS: &[(&str, &str)] = &[
    ("corresponding_orbitals.paired_overlap", "dimensionless"),
    (
        "corresponding_orbitals.paired_alpha_coefficients",
        "dimensionless",
    ),
    (
        "corresponding_orbitals.paired_beta_coefficients",
        "dimensionless",
    ),
    (
        "corresponding_orbitals.unpaired_alpha_coefficients",
        "dimensionless",
    ),
    (
        "corresponding_orbitals.unpaired_beta_coefficients",
        "dimensionless",
    ),
    (
        "corresponding_orbitals.cross_overlap_residual",
        "dimensionless",
    ),
    (
        "corresponding_orbitals.spin_squared",
        "dimensionless, hbar^2",
    ),
];

fn with_natural_orbitals(report: Value, analysis: &NaturalBondAnalysis) -> Result<Value> {
    use crate::properties::natural_bond::spectral::{
        effectively_unpaired_weights, NaturalDensityOrbitals,
    };
    let natural = NaturalDensityOrbitals::from_charge_density(&analysis.nao)?;
    let unpaired = effectively_unpaired_weights(&natural.weights);
    let bound_defect = natural
        .weights
        .iter()
        .fold(0.0_f64, |defect, &n| defect.max(-n).max(n - 2.0));
    Ok(report.with(
        "natural_orbitals",
        Value::object()
            .with("occupations", natural.weights)
            .with("ao_coefficients", natural.ao_coefficients)
            .with("trace_residual", natural.trace_residual)
            .with("reconstruction_residual", natural.reconstruction_residual)
            .with("occupation_bound_defect", bound_defect)
            .with(
                "effectively_unpaired_electrons",
                unpaired.iter().sum::<f64>(),
            )
            .with("effectively_unpaired_weights", unpaired),
    ))
}

struct SpinDensityRequest {
    points: Vec<Vec3>,
}

impl SpinDensityRequest {
    const UNITS: &'static [(&'static str, &'static str)] = &[
        ("natural_spin_density.weights", "electrons"),
        ("natural_spin_density.ao_coefficients", "dimensionless"),
        ("natural_spin_density.trace_residual", "electrons"),
        ("natural_spin_density.reconstruction_residual", "electrons"),
        ("natural_spin_density.points", "bohr"),
        ("natural_spin_density.values", "electrons/bohr^3"),
    ];

    fn read(controls: &Controls) -> Result<Option<Self>> {
        let enabled = controls.bool("nbo_spin_density", false)?;
        let points = controls.opt_rows("nbo_spin_density_points")?;
        if !enabled && points.is_none() {
            return Ok(None);
        }
        let points = points
            .unwrap_or_default()
            .into_iter()
            .map(|r| match r.as_slice() {
                [x, y, z] if [x, y, z].iter().all(|v| v.is_finite()) => Ok(Vec3::new(*x, *y, *z)),
                _ => Err(DftError::InvalidInput(
                    "nbo_spin_density_points must contain finite Cartesian triples".into(),
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(Self { points }))
    }

    fn add_report(
        &self,
        report: Value,
        basis: &crate::basis::MolBasis,
        analysis: &NaturalBondAnalysis,
    ) -> Result<Value> {
        let spin =
            crate::properties::natural_bond::spectral::NaturalDensityOrbitals::from_spin_density(
                &analysis.nao,
            )?;
        let values = spin.values_at(basis, &self.points)?;
        Ok(report.with(
            "natural_spin_density",
            Value::object()
                .with("weights", spin.weights)
                .with("ao_coefficients", spin.ao_coefficients)
                .with("trace_residual", spin.trace_residual)
                .with("reconstruction_residual", spin.reconstruction_residual)
                .with(
                    "points",
                    self.points.iter().map(|p| p.to_array()).collect::<Vec<_>>(),
                )
                .with("values", values),
        ))
    }
}

impl SpinorRequest {
    pub(super) fn new(controls: &Controls, correlated: bool) -> Result<Self> {
        let source = controls.str("nbo_density", "auto")?;
        if correlated || controls.contains("cas") || !matches!(source.as_str(), "auto" | "scf") {
            return Err(DftError::Unsupported {
                feature: "correlated spinor natural-bond density".into(),
                missing: "a complex correlated density with matching Fock operators".into(),
            });
        }
        let (spin, options) = read_options(controls)?;
        Ok(Self {
            spin,
            options,
            resonance: controls.bool("nbo_resonance", false)?,
            max_structures: controls.usize("nbo_resonance_structures", 20)?,
            spin_density: SpinDensityRequest::read(controls)?,
            natural_orbitals: controls.bool("nbo_natural_orbitals", false)?,
        })
    }

    pub(super) fn report(
        &self,
        report: Value,
        setup: &super::method::MolecularSetup,
        overlap: &Matrix,
        density: &crate::periodic::BlochMatrix,
        fock: &crate::periodic::BlochMatrix,
        axis: [f64; 3],
    ) -> Result<Value> {
        let projected =
            crate::properties::natural_bond::spinor::collinear_channels(density, fock, axis)?;
        let analysis = analyse(
            setup,
            overlap,
            &projected.density,
            &projected.fock,
            self.spin,
            &self.options,
        )?;
        let (dipole, electrons) = density_dipole(setup, &spin_summed(&projected.density), overlap);
        let mut report = with_analysis(report, &analysis, dipole, electrons)
            .with("density_source", "scf")
            .with("nbo_spin_axis", projected.axis)
            .with("nbo_spinor_density_defect", projected.density_defect)
            .with("nbo_spinor_fock_defect", projected.fock_defect);
        if self.resonance {
            let numbers: Vec<_> = setup.molecule.atoms.iter().map(|a| a.z).collect();
            report = with_resonance(
                report,
                &analysis,
                &numbers,
                &projected.fock,
                self.max_structures,
            )?;
        }
        if let Some(spin_density) = &self.spin_density {
            report = spin_density.add_report(report, &setup.basis, &analysis)?;
        }
        if self.natural_orbitals {
            report = with_natural_orbitals(report, &analysis)?;
        }
        // Keep the source's SCF diagnostics while replacing its spinor
        // orbital channel by the natural bond analysis channel.
        let mut report_units = report.get("units").cloned().unwrap_or_else(Value::object);
        if let Value::Object(labels) = &mut report_units {
            labels.retain(|(name, _)| !name.starts_with("channels."));
        }
        for &(name, unit) in NBO_UNITS.iter().chain(
            [
                ("nbo_spin_axis", "dimensionless"),
                ("nbo_spinor_density_defect", "dimensionless"),
                ("nbo_spinor_fock_defect", "hartree"),
            ]
            .iter(),
        ) {
            report_units.set(name, unit);
        }
        if self.resonance {
            for &(name, unit) in NRT_UNITS {
                report_units.set(name, unit);
            }
        }
        if self.spin_density.is_some() {
            for &(name, unit) in SpinDensityRequest::UNITS {
                report_units.set(name, unit);
            }
        }
        if self.natural_orbitals {
            for &(name, unit) in NATURAL_ORBITAL_UNITS {
                report_units.set(name, unit);
            }
        }
        Ok(report.with("units", report_units))
    }
}

pub(super) fn natural_bond(system: &System, controls: &Controls) -> Result<Value> {
    let (spin, options) = read_options(controls)?;
    let requested = controls.str("nbo_density", "auto")?;
    let source = match requested.as_str() {
        "auto" if super::double_hybrid::requested(controls)? => "pt2_relaxed",
        "auto" if controls.contains("cas") => "casscf",
        "auto" => "scf",
        "scf" | "casscf" | "pt2_relaxed" => requested.as_str(),
        other => {
            return Err(DftError::InvalidInput(format!(
                "nbo_density is auto, scf, casscf or pt2_relaxed, not `{other}`"
            )))
        }
    };
    let state = controls.opt_usize("nbo_cas_state")?;
    if state.is_some() && source != "casscf" {
        return Err(DftError::InvalidInput(
            "nbo_cas_state selects a state of the casscf density".into(),
        ));
    }
    let resonance = controls.bool("nbo_resonance", false)?;
    let max_structures = controls.usize("nbo_resonance_structures", 20)?;
    let spin_density = SpinDensityRequest::read(controls)?;
    let natural_orbitals = controls.bool("nbo_natural_orbitals", false)?;
    let corresponding = controls.bool("nbo_corresponding_orbitals", false)?;
    if corresponding && source != "scf" {
        return Err(DftError::Unsupported {
            feature: "corresponding orbitals for a correlated density".into(),
            missing: "integer occupied determinant frames for both spin channels".into(),
        });
    }
    let mut labels: Vec<(&str, &str)> =
        vec![("energy", "hartree"), ("energy_components", "hartree")];
    labels.extend_from_slice(NBO_UNITS);
    if resonance {
        labels.extend_from_slice(NRT_UNITS);
    }
    if spin_density.is_some() {
        labels.extend_from_slice(SpinDensityRequest::UNITS);
    }
    if natural_orbitals {
        labels.extend_from_slice(NATURAL_ORBITAL_UNITS);
    }
    if corresponding {
        labels.extend_from_slice(CORRESPONDING_ORBITAL_UNITS);
    }
    let numbers: Vec<u8> = system.numbers.clone();
    let finish = |mut report: Value,
                  analysis: &NaturalBondAnalysis,
                  fock: &[Matrix; 2],
                  basis: &crate::basis::MolBasis|
     -> Result<Value> {
        if resonance {
            report = with_resonance(report, analysis, &numbers, fock, max_structures)?;
        }
        if let Some(spin_density) = &spin_density {
            report = spin_density.add_report(report, basis, analysis)?;
        }
        if natural_orbitals {
            report = with_natural_orbitals(report, analysis)?;
        }
        Ok(report)
    };
    match source {
        "scf" => {
            let kind = CollinearPropertyReference::from_controls(controls)?;
            let setup = setup(system, controls)?;
            controls.finish("natural-bond")?;
            let reference = run_collinear_property_reference(&setup, kind)?;
            let scf = &reference.scf;
            let analysis = analyse(
                &setup,
                &scf.overlap,
                &scf.density,
                &scf.fock,
                spin,
                &options,
            )?;
            let (dipole, electrons) =
                density_dipole(&setup, &spin_summed(&scf.density), &scf.overlap);
            let report = reference_report(
                &setup,
                &reference,
                "natural bond orbital analysis of the SCF density: natural atomic orbitals \
                 (occupancy-weighted orthogonalization of the natural minimal basis), natural \
                 population analysis, natural Lewis structure with orthogonal hybrids, natural \
                 localized molecular orbitals and second-order donor-acceptor energies of the \
                 Fock matrix in the NBO basis",
            )
            .with("density_source", "scf")
            .with("reference_type", kind.name());
            let mut report = finish(
                with_analysis(report, &analysis, dipole, electrons),
                &analysis,
                &scf.fock,
                &setup.basis,
            )?;
            if corresponding {
                let uco =
                    crate::properties::natural_bond::corresponding::corresponding_orbitals(scf)?;
                let [alpha, beta] = uco.paired_coefficients;
                let [unpaired_alpha, unpaired_beta] = uco.unpaired_coefficients;
                report.set(
                    "corresponding_orbitals",
                    Value::object()
                        .with("paired_overlap", uco.paired_overlap)
                        .with("paired_alpha_coefficients", alpha)
                        .with("paired_beta_coefficients", beta)
                        .with("unpaired_alpha_coefficients", unpaired_alpha)
                        .with("unpaired_beta_coefficients", unpaired_beta)
                        .with("cross_overlap_residual", uco.cross_overlap_residual)
                        .with("spin_squared", uco.spin_squared),
                );
            }
            Ok(report.with("units", units(&labels)))
        }
        "pt2_relaxed" => {
            let prepared = super::double_hybrid::prepare(system, controls, "natural-bond")?;
            let setup = &prepared.setup;
            let scf = &prepared.reference.scf;
            let mut solver = crate::response::ResponseSolver::new(
                &setup.molecule,
                &setup.basis,
                &setup.functional,
                setup.grid.as_ref(),
                scf,
                &setup.options,
            )?;
            if let Some(config) = &prepared.config {
                solver.tol = config.tolerance;
                solver.max_iter = config.max_iterations;
            }
            let delta = super::correlated_electric::target_difference(
                setup,
                &prepared.reference,
                prepared.target.as_ref(),
                prepared.target_vv10,
            )?;
            let density = crate::properties::pt2_density::relaxed_density(
                scf,
                &prepared.problem,
                &prepared.pt2.amplitudes,
                prepared.coefficients,
                &prepared.eri,
                &solver,
                &delta,
            )?;
            let fock = density_fock(&scf.h_core, &prepared.eri, &density.relaxed);
            let analysis = analyse(setup, &scf.overlap, &density.relaxed, &fock, spin, &options)?;
            let (dipole, electrons) =
                density_dipole(setup, &spin_summed(&density.relaxed), &scf.overlap);
            let report = reference_report(
                setup,
                &prepared.reference,
                "natural bond orbital analysis of the relaxed spin-component PT2 (double-hybrid) \
                 density: reference density, unrelaxed PT2 occupied/virtual blocks and the \
                 orbital relaxation of the parent CPKS multipliers; second-order energies with \
                 the Hartree-Fock operator of the relaxed density",
            )
            .with("density_source", "pt2_relaxed")
            .with("double_hybrid_model", prepared.model.as_str())
            .with("pt2_correlation", prepared.pt2.correlation)
            .with("pt2_same_spin_coefficient", prepared.coefficients.same_spin)
            .with(
                "pt2_opposite_spin_coefficient",
                prepared.coefficients.opposite_spin,
            )
            .with("relaxed_density_gradient_norm", density.gradient_norm)
            .with("unrelaxed_density_trace", density.unrelaxed_trace.to_vec());
            labels.push(("pt2_correlation", "hartree"));
            labels.push(("pt2_same_spin_coefficient", "dimensionless"));
            labels.push(("pt2_opposite_spin_coefficient", "dimensionless"));
            labels.push(("relaxed_density_gradient_norm", "hartree"));
            labels.push(("unrelaxed_density_trace", "electrons"));
            let report = finish(
                with_analysis(report, &analysis, dipole, electrons),
                &analysis,
                &fock,
                &setup.basis,
            )?;
            Ok(report.with("units", units(&labels)))
        }
        _ => super::mcpdft::compute(
            system,
            controls,
            "natural-bond",
            |setup, problem, computed| {
                let point = &computed.result.point;
                let spec = &computed.spec;
                let space = &problem.space;
                let roots = point.states.vectors.len();
                let weights: Vec<f64> = match state {
                    Some(k) if k < roots => {
                        (0..roots).map(|j| f64::from(u8::from(j == k))).collect()
                    }
                    Some(k) => {
                        return Err(DftError::InvalidInput(format!(
                            "nbo_cas_state {k} is not one of the {roots} computed states"
                        )))
                    }
                    None => spec.weights.clone(),
                };
                let n = setup.basis.nao_sph;
                let (nc, na) = (spec.n_core, spec.n_active);
                let mut gamma = [Matrix::zeros(na, na), Matrix::zeros(na, na)];
                for (vector, weight) in point.states.vectors.iter().zip(&weights) {
                    if *weight == 0.0 {
                        continue;
                    }
                    let spin_rdm = space.rdm1_spin(vector, vector)?;
                    for (g, r) in gamma.iter_mut().zip(&spin_rdm) {
                        for (x, y) in g.as_mut_slice().iter_mut().zip(r.as_slice()) {
                            *x += weight * y;
                        }
                    }
                }
                let c = &point.orbitals;
                let density: [Matrix; 2] = std::array::from_fn(|s| {
                    let mut p = Matrix::zeros(n, n);
                    for mu in 0..n {
                        for nu in 0..n {
                            let mut value = 0.0;
                            for k in 0..nc {
                                value += c[(mu, k)] * c[(nu, k)];
                            }
                            for t in 0..na {
                                for u in 0..na {
                                    value += c[(mu, nc + t)] * gamma[s][(t, u)] * c[(nu, nc + u)];
                                }
                            }
                            p[(mu, nu)] = value;
                        }
                    }
                    p
                });
                let overlap = crate::integrals::one_electron::overlap_matrix(&setup.basis);
                let fock = density_fock(problem.operators.h_core, problem.operators.eri, &density);
                let analysis = analyse(setup, &overlap, &density, &fock, spin, &options)?;
                let (dipole, electrons) = density_dipole(setup, &spin_summed(&density), &overlap);
                let report = Value::object()
                    .with(
                        "model",
                        "natural bond orbital analysis of the complete-active-space SCF density \
                     (closed-shell core plus the spin-resolved active one-particle density, \
                     state-averaged with the state weights or of one state); second-order \
                     energies with the Hartree-Fock operator of the analysed density",
                    )
                    .with("density_source", "casscf")
                    .with("basis", setup.method.basis.as_str())
                    .with("energy", point.average_energy)
                    .with("cas_energies", point.energies.clone())
                    .with("cas_state_weights", weights.clone())
                    .with("orbitals_converged", computed.result.converged)
                    .with("orbital_gradient", point.gradient_norm());
                labels.push(("cas_energies", "hartree"));
                labels.push(("cas_state_weights", "dimensionless"));
                labels.push(("orbital_gradient", "hartree"));
                let report = finish(
                    with_analysis(report, &analysis, dipole, electrons),
                    &analysis,
                    &fock,
                    &setup.basis,
                )?;
                Ok(report.with("units", units(&labels)))
            },
        ),
    }
}

/// Spin treatment and Lewis-search controls of the analysis.
pub(super) fn read_options(controls: &Controls) -> Result<(NboSpin, NboOptions)> {
    let spin = NboSpin::parse(&controls.str("nbo_spin", "auto")?)?;
    let defaults = NboOptions::default();
    Ok((
        spin,
        NboOptions {
            lewis_threshold: controls.opt_f64("nbo_lewis_threshold")?,
            three_center: controls.bool("nbo_three_center", false)?,
            perturbation_threshold: controls.f64(
                "nbo_perturbation_threshold",
                defaults.perturbation_threshold,
            )?,
        },
    ))
}

/// Cell offset from `from` to `to` on the mesh, wrapped to `(-n/2, n/2]`.
fn offset(from: [usize; 3], to: [usize; 3], mesh: [usize; 3]) -> [i64; 3] {
    std::array::from_fn(|axis| {
        let n = mesh[axis] as i64;
        let mut d = (to[axis] as i64 - from[axis] as i64).rem_euclid(n);
        if 2 * d > n {
            d -= n;
        }
        d
    })
}

/// Anchor (supercell atom) and key of the translation class of an orbital
/// on the supercell atoms `atoms`: among the atoms with the lowest index in
/// the cell, the one giving the smallest sorted list of (atom in cell, cell
/// offset).
fn canonical(
    atoms: &[usize],
    natoms: usize,
    cells: &[[usize; 3]],
    mesh: [usize; 3],
) -> (usize, Vec<(usize, [i64; 3])>) {
    let lowest = atoms.iter().map(|g| g % natoms).min().unwrap_or(0);
    let mut best: Option<(Vec<(usize, [i64; 3])>, usize)> = None;
    for &g in atoms.iter().filter(|g| *g % natoms == lowest) {
        let anchor = cells[g / natoms];
        let mut key: Vec<(usize, [i64; 3])> = atoms
            .iter()
            .map(|h| (h % natoms, offset(anchor, cells[h / natoms], mesh)))
            .collect();
        key.sort();
        if best.as_ref().is_none_or(|(k, _)| key < *k) {
            best = Some((key, g));
        }
    }
    let (key, anchor) = best.expect("an orbital has at least one atom");
    (anchor, key)
}

fn periodic_label(numbers: &[u8], atom: usize, shift: [i64; 3]) -> String {
    let symbol = crate::system::z_to_symbol(numbers[atom]).unwrap_or("X");
    if shift == [0; 3] {
        format!("{symbol}{}", atom + 1)
    } else {
        format!(
            "{symbol}{}[{},{},{}]",
            atom + 1,
            shift[0],
            shift[1],
            shift[2]
        )
    }
}

/// Report entries of the supercell analysis per cell of the crystal.
pub(super) fn periodic_report(
    report: Value,
    result: &crate::properties::natural_bond::periodic::PeriodicNaturalBond,
    numbers: &[u8],
    periodic_axes: [bool; 3],
) -> Value {
    let nao = &result.analysis.nao;
    let natoms = result.atoms_per_cell;
    let cells = result.cells.len();
    let home = |values: &[f64]| values[..natoms].to_vec();
    let mut bonds = Vec::new();
    for a in 0..natoms {
        for g in 0..cells * natoms {
            let index = nao.wiberg_bond_indices[(a, g)];
            if g != a && index > 0.01 {
                let shift = offset([0; 3], result.cells[g / natoms], result.mesh);
                bonds.push(
                    Value::object()
                        .with("atom", a)
                        .with("partner", g % natoms)
                        .with("cell_offset", shift.to_vec())
                        .with("label", periodic_label(numbers, g % natoms, shift))
                        .with("index", index),
                );
            }
        }
    }
    let channels: Vec<Value> = result
        .analysis
        .channels
        .iter()
        .map(|channel| {
            type Key = Vec<(usize, [i64; 3])>;
            // Translation class, anchor cell and atom names of every NBO.
            let classes: Vec<((String, Key), [usize; 3], Vec<String>)> = channel
                .nbos
                .iter()
                .map(|nbo| {
                    let (anchor, key) = canonical(&nbo.atoms, natoms, &result.cells, result.mesh);
                    let anchor_cell = result.cells[anchor / natoms];
                    let names = nbo
                        .atoms
                        .iter()
                        .map(|&g| {
                            periodic_label(
                                numbers,
                                g % natoms,
                                offset(anchor_cell, result.cells[g / natoms], result.mesh),
                            )
                        })
                        .collect();
                    ((nbo.kind.name().to_string(), key), anchor_cell, names)
                })
                .collect();
            let keep: Vec<bool> = classes.iter().map(|(_, cell, _)| *cell == [0; 3]).collect();
            // Serials among the orbitals of the home cell; every image takes
            // the label of the home orbital of its class closest in
            // occupancy and energy.
            let mut labels = vec![String::new(); channel.nbos.len()];
            let mut seen: Vec<(String, Key)> = Vec::new();
            for (j, (class, _, names)) in classes.iter().enumerate() {
                if keep[j] {
                    let serial = seen.iter().filter(|c| *c == class).count() + 1;
                    seen.push(class.clone());
                    labels[j] = format!("{}({serial}) {}", class.0, names.join("-"));
                }
            }
            for j in 0..channel.nbos.len() {
                if keep[j] {
                    continue;
                }
                let distance = |i: usize| -> f64 {
                    let (a, b) = (&channel.nbos[j], &channel.nbos[i]);
                    (a.occupancy - b.occupancy).abs()
                        + (a.energy.unwrap_or(0.0) - b.energy.unwrap_or(0.0)).abs()
                };
                let partner = (0..channel.nbos.len())
                    .filter(|&i| keep[i] && classes[i].0 == classes[j].0)
                    .min_by(|&x, &y| distance(x).total_cmp(&distance(y)));
                labels[j] = match partner {
                    Some(i) => labels[i].clone(),
                    None => format!("{} {}", classes[j].0 .0, classes[j].2.join("-")),
                };
            }
            let nbos: Vec<Value> = channel
                .nbos
                .iter()
                .zip(&labels)
                .zip(&keep)
                .filter(|(_, kept)| **kept)
                .map(|((nbo, label), _)| {
                    Value::object()
                        .with("label", label.as_str())
                        .with("type", nbo.kind.name())
                        .with("lewis", nbo.kind.is_lewis())
                        .with("occupancy", nbo.occupancy)
                        .with("energy", nbo.energy)
                        .with(
                            "hybrids",
                            nbo.hybrids
                                .iter()
                                .map(|h| {
                                    Value::object()
                                        .with("atom", h.atom % natoms)
                                        .with("coefficient", h.coefficient)
                                        .with("composition", h.composition.clone())
                                })
                                .collect::<Vec<_>>(),
                        )
                })
                .collect();
            let perturbation: Vec<Value> = channel
                .perturbation
                .iter()
                .filter(|p| keep[p.donor])
                .map(|p| {
                    Value::object()
                        .with("donor_label", labels[p.donor].as_str())
                        .with("acceptor_label", labels[p.acceptor].as_str())
                        .with("stabilization", p.stabilization)
                        .with("fock_element", p.fock_element)
                        .with("energy_gap", p.energy_gap)
                })
                .collect();
            let per_cell = 1.0 / cells as f64;
            Value::object()
                .with("spin", channel.spin.as_str())
                .with("maximum_occupancy", channel.maximum_occupancy)
                .with("electrons_per_cell", channel.electrons * per_cell)
                .with("lewis_threshold", channel.lewis_threshold)
                .with(
                    "lewis_occupancy_per_cell",
                    channel.lewis_occupancy * per_cell,
                )
                .with(
                    "non_lewis_occupancy_per_cell",
                    channel.non_lewis_occupancy * per_cell,
                )
                .with("nbos", nbos)
                .with("perturbation", perturbation)
                .with(
                    "perturbation_total_per_cell",
                    channel.perturbation_total.map(|t| t * per_cell),
                )
                .with("orthonormality_residual", channel.orthonormality_residual)
                .with("nlmo_residual", channel.nlmo_residual)
        })
        .collect();
    let warnings: Vec<String> = (0..3)
        .filter(|&axis| periodic_axes[axis] && result.mesh[axis] < 3)
        .map(|axis| {
            format!(
                "WARNING [natural-bond] axis {axis} has {} k points: an atom and its image in \
                 the next cell can share supercell orbitals",
                result.mesh[axis]
            )
        })
        .collect();
    report
        .with("supercell_mesh", result.mesh.to_vec())
        .with("cells_in_supercell", cells)
        .with("atoms_per_cell", natoms)
        .with("natural_charges", home(&nao.atomic_charges))
        .with("natural_populations", home(&nao.atomic_populations))
        .with("natural_spin_populations", home(&nao.spin_populations))
        .with("core_populations", home(&nao.core_populations))
        .with("valence_populations", home(&nao.valence_populations))
        .with("rydberg_populations", home(&nao.rydberg_populations))
        .with(
            "natural_electron_configurations",
            nao.configuration_text[..natoms].to_vec(),
        )
        .with("wiberg_bonds", bonds)
        .with("wiberg_atom_totals", home(&nao.wiberg_atom_totals))
        .with("translation_defect", result.translation_defect)
        .with("unfolding_imaginary", result.unfolding_imaginary)
        .with("nao_orthonormality_residual", nao.orthonormality_residual)
        .with("open_shell", result.analysis.open_shell)
        .with("channels", channels)
        .with("nbo_warnings", warnings)
        .with(
            "units",
            units(&[
                ("energy", "hartree per cell"),
                ("energy_components", "hartree per cell"),
                ("natural_charges", "e"),
                ("natural_populations", "electrons"),
                ("natural_spin_populations", "electrons (alpha minus beta)"),
                ("core_populations", "electrons"),
                ("valence_populations", "electrons"),
                ("rydberg_populations", "electrons"),
                ("wiberg_bonds.index", "dimensionless"),
                ("wiberg_atom_totals", "dimensionless"),
                ("translation_defect", "e"),
                (
                    "unfolding_imaginary",
                    "dimensionless, largest imaginary AO matrix element",
                ),
                ("nao_orthonormality_residual", "dimensionless"),
                ("channels.maximum_occupancy", "electrons"),
                ("channels.electrons_per_cell", "electrons"),
                ("channels.lewis_threshold", "electrons"),
                ("channels.lewis_occupancy_per_cell", "electrons"),
                ("channels.non_lewis_occupancy_per_cell", "electrons"),
                ("channels.nbos.occupancy", "electrons"),
                ("channels.nbos.energy", "hartree"),
                (
                    "channels.nbos.hybrids.coefficient",
                    "dimensionless polarization coefficient",
                ),
                (
                    "channels.nbos.hybrids.composition",
                    "percent s, p, d, f, ... character",
                ),
                ("channels.perturbation.stabilization", "hartree"),
                ("channels.perturbation.fock_element", "hartree"),
                ("channels.perturbation.energy_gap", "hartree"),
                ("channels.perturbation_total_per_cell", "hartree"),
                ("channels.orthonormality_residual", "dimensionless"),
                ("channels.nlmo_residual", "electrons"),
            ]),
        )
}

/// NLMOs (AO coefficients) of every spin and their labels; a spin-summed
/// analysis gives both spins the same orbitals.
fn spin_nlmos(analysis: &NaturalBondAnalysis) -> ([Matrix; 2], [Vec<String>; 2]) {
    let c = &analysis.nao.nao_coefficients;
    let of = |channel: &NboChannel| -> (Matrix, Vec<String>) {
        (
            c.matmul(&channel.nlmo_coefficients),
            channel
                .nlmos
                .iter()
                .map(|m| channel.nbos[m.parent].label.clone())
                .collect(),
        )
    };
    if analysis.channels.len() == 2 {
        let (a, la) = of(&analysis.channels[0]);
        let (b, lb) = of(&analysis.channels[1]);
        ([a, b], [la, lb])
    } else {
        let (a, la) = of(&analysis.channels[0]);
        ([a.clone(), a], [la.clone(), la])
    }
}

fn isotropic(t: &[[f64; 3]; 3]) -> f64 {
    (t[0][0] + t[1][1] + t[2][2]) / 3.0
}

/// Natural chemical shielding analysis: NMR shielding as a sum over the
/// natural localized molecular orbitals of the natural Lewis structure.
pub(super) fn natural_shielding(system: &System, controls: &Controls) -> Result<Value> {
    use crate::properties::magnetic::{orbital_shielding, Gauge};
    let (spin, options) = read_options(controls)?;
    let config = super::vibrational::response_config(controls)?;
    let gauge_name = controls.str("gauge", "giao")?;
    let gauge = Gauge::from_name(&gauge_name)?;
    let origin = controls
        .opt_vec3("gauge_origin")?
        .unwrap_or_else(Vec3::zero);
    let selected = controls.opt_usize_list("nuclei")?;
    let setup = setup(system, controls)?;
    controls.finish("natural-shielding")?;
    let reference = run_molecular_reference(&setup)?;
    let scf = &reference.scf;
    let analysis = analyse(
        &setup,
        &scf.overlap,
        &scf.density,
        &scf.fock,
        spin,
        &options,
    )?;
    let (orbitals, labels) = spin_nlmos(&analysis);
    let ncs = orbital_shielding(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        scf,
        &setup.options,
        gauge,
        origin,
        &config,
        [&orbitals[0], &orbitals[1]],
    )?;
    let nuclei = selected.unwrap_or_else(|| (0..setup.molecule.len()).collect());
    if nuclei.iter().any(|&n| n >= setup.molecule.len()) {
        return Err(DftError::InvalidInput(
            "natural-shielding nuclei are atom indices of the molecule".into(),
        ));
    }
    let separate = analysis.channels.len() == 2;
    let entries: Vec<Value> = nuclei
        .iter()
        .map(|&n| {
            let mut parts: Vec<(String, &'static str, [[f64; 3]; 3], [[f64; 3]; 3])> = Vec::new();
            if separate {
                for (s, name) in ["alpha", "beta"].into_iter().enumerate() {
                    for (k, label) in labels[s].iter().enumerate() {
                        parts.push((
                            label.clone(),
                            name,
                            ncs.diamagnetic[s][n][k],
                            ncs.paramagnetic[s][n][k],
                        ));
                    }
                }
            } else {
                for (k, label) in labels[0].iter().enumerate() {
                    let sum = |x: &[Vec<Vec<[[f64; 3]; 3]>>; 2]| -> [[f64; 3]; 3] {
                        std::array::from_fn(|i| {
                            std::array::from_fn(|j| x[0][n][k][i][j] + x[1][n][k][i][j])
                        })
                    };
                    parts.push((
                        label.clone(),
                        "total",
                        sum(&ncs.diamagnetic),
                        sum(&ncs.paramagnetic),
                    ));
                }
            }
            parts.sort_by(|x, y| {
                let size = |p: &(String, &str, [[f64; 3]; 3], [[f64; 3]; 3])| {
                    (isotropic(&p.2) + isotropic(&p.3)).abs()
                };
                size(y).total_cmp(&size(x))
            });
            let contributions: Vec<Value> = parts
                .iter()
                .map(|(label, spin, dia, para)| {
                    let total: [[f64; 3]; 3] =
                        std::array::from_fn(|i| std::array::from_fn(|j| dia[i][j] + para[i][j]));
                    Value::object()
                        .with("orbital", label.as_str())
                        .with("spin", *spin)
                        .with("isotropic", isotropic(&total))
                        .with("diamagnetic_isotropic", isotropic(dia))
                        .with("paramagnetic_isotropic", isotropic(para))
                        .with("tensor", total)
                })
                .collect();
            let total = &ncs.totals[n];
            Value::object()
                .with("atom", n)
                .with("isotropic", total.isotropic())
                .with("tensor", total.tensor)
                .with("contributions", contributions)
        })
        .collect();
    Ok(reference_report(
        &setup,
        &reference,
        "natural chemical shielding analysis: orbital NMR shielding of the coupled-perturbed \
         magnetic response decomposed over the natural localized molecular orbitals of the \
         natural Lewis structure (diamagnetic part from the orbital densities, paramagnetic \
         part from the first-order orbitals rotated with them)",
    )
    .with("gauge", gauge_name)
    .with("gauge_origin", origin)
    .with("shieldings", entries)
    .with("decomposition_residual", ncs.decomposition_residual)
    .with("rotation_residual", ncs.rotation_residual)
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            ("energy_components", "hartree"),
            ("gauge_origin", "bohr"),
            ("shieldings.isotropic", "ppm"),
            (
                "shieldings.tensor",
                "ppm, [moment component][field component]",
            ),
            ("shieldings.contributions.isotropic", "ppm"),
            ("shieldings.contributions.diamagnetic_isotropic", "ppm"),
            ("shieldings.contributions.paramagnetic_isotropic", "ppm"),
            ("shieldings.contributions.tensor", "ppm"),
            ("decomposition_residual", "ppm"),
            ("rotation_residual", "dimensionless"),
        ]),
    ))
}

/// Natural J-coupling analysis: indirect spin-spin couplings of a
/// closed-shell molecule as sums over its natural localized molecular
/// orbitals.
pub(super) fn natural_coupling(system: &System, controls: &Controls) -> Result<Value> {
    use crate::response::spin_spin::spin_spin_couplings_by_orbital;
    let (_, options) = read_options(controls)?;
    let config = super::vibrational::response_config(controls)?;
    let selected = controls.opt_usize_list("nuclei")?;
    let nuclear_g = controls.opt_f64_list("nuclear_g")?;
    let setup = setup(system, controls)?;
    controls.finish("natural-coupling")?;
    let natoms = setup.molecule.len();
    let nuclei = selected.unwrap_or_else(|| (0..natoms).collect());
    if nuclei.len() < 2 || nuclear_g.as_ref().is_some_and(|g| g.len() != natoms) {
        return Err(DftError::InvalidInput(
            "natural-coupling needs at least two nuclei, and nuclear_g holds one g factor per \
             atom"
                .into(),
        ));
    }
    let reference = run_molecular_reference(&setup)?;
    let scf = &reference.scf;
    let analysis = analyse(
        &setup,
        &scf.overlap,
        &scf.density,
        &scf.fock,
        NboSpin::Total,
        &options,
    )?;
    let (orbitals, labels) = spin_nlmos(&analysis);
    let (total, parts) = spin_spin_couplings_by_orbital(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        scf,
        &setup.options,
        &nuclei,
        &config,
        &orbitals[0],
    )?;
    let magneton = 0.5 / crate::constants::PROTON_ELECTRON_MASS_RATIO;
    let hertz = |m: usize, n: usize| -> Option<f64> {
        nuclear_g
            .as_ref()
            .map(|g| g[m] * g[n] * magneton * magneton * super::nmr::HARTREE_IN_HERTZ)
    };
    let pairs: Vec<Value> = total
        .pairs
        .iter()
        .enumerate()
        .map(|(p, &(m, n))| {
            let factor = hertz(m, n);
            let mut contributions: Vec<(usize, [f64; 4])> = (0..labels[0].len())
                .map(|k| {
                    (
                        k,
                        [
                            isotropic(&parts.fermi_contact[p][k]),
                            isotropic(&parts.spin_dipole[p][k]),
                            isotropic(&parts.paramagnetic[p][k]),
                            isotropic(&parts.diamagnetic[p][k]),
                        ],
                    )
                })
                .collect();
            contributions.sort_by(|x, y| {
                let size = |v: &[f64; 4]| v.iter().sum::<f64>().abs();
                size(&y.1).total_cmp(&size(&x.1))
            });
            let entries: Vec<Value> = contributions
                .iter()
                .map(|(k, v)| {
                    let sum: f64 = v.iter().sum();
                    Value::object()
                        .with("orbital", labels[0][*k].as_str())
                        .with("reduced_isotropic", sum)
                        .with("fermi_contact", v[0])
                        .with("spin_dipole", v[1])
                        .with("paramagnetic", v[2])
                        .with("diamagnetic", v[3])
                        .with("coupling_hz", factor.map(|f| f * sum))
                })
                .collect();
            let reduced = isotropic(&total.total[p]);
            Value::object()
                .with("atoms", vec![m, n])
                .with("reduced_isotropic", reduced)
                .with("fermi_contact", isotropic(&total.fermi_contact[p]))
                .with("spin_dipole", isotropic(&total.spin_dipole[p]))
                .with("paramagnetic", isotropic(&total.paramagnetic[p]))
                .with("diamagnetic", isotropic(&total.diamagnetic[p]))
                .with("coupling_hz", factor.map(|f| f * reduced))
                .with("contributions", entries)
        })
        .collect();
    Ok(reference_report(
        &setup,
        &reference,
        "natural J-coupling analysis: indirect spin-spin coupling of a closed-shell reference \
         (Fermi contact, spin dipole, paramagnetic and diamagnetic spin-orbit terms) decomposed \
         over the natural localized molecular orbitals of the natural Lewis structure",
    )
    .with("couplings", pairs)
    .with("decomposition_residual", parts.decomposition_residual)
    .with("rotation_residual", parts.rotation_residual)
    .with("response_residual", total.residual)
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            ("energy_components", "hartree"),
            (
                "couplings.reduced_isotropic",
                "atomic units, reduced coupling K",
            ),
            ("couplings.fermi_contact", "atomic units"),
            ("couplings.spin_dipole", "atomic units"),
            ("couplings.paramagnetic", "atomic units"),
            ("couplings.diamagnetic", "atomic units"),
            ("couplings.coupling_hz", "Hz, isotropic J"),
            ("couplings.contributions.reduced_isotropic", "atomic units"),
            ("couplings.contributions.fermi_contact", "atomic units"),
            ("couplings.contributions.spin_dipole", "atomic units"),
            ("couplings.contributions.paramagnetic", "atomic units"),
            ("couplings.contributions.diamagnetic", "atomic units"),
            ("couplings.contributions.coupling_hz", "Hz"),
            ("decomposition_residual", "atomic units"),
            ("rotation_residual", "dimensionless"),
            ("response_residual", "dimensionless"),
        ]),
    ))
}

/// Energy of spin densities with the functional of `setup`, the one-electron
/// operator `h`, the classical Coulomb and exact exchange of the densities
/// and the nuclear repulsion `repulsion`.
fn energy_of(
    setup: &super::method::MolecularSetup,
    engine: &crate::scf::JkEngine,
    long_range: Option<&crate::scf::JkEngine>,
    h: &Matrix,
    repulsion: f64,
    density: [&Matrix; 2],
) -> Result<f64> {
    let p = spin_summed(&[density[0].clone(), density[1].clone()]);
    let (j, k) = engine.build(&setup.basis, &p, density, true)?;
    let hybrid = setup.functional.hybrid;
    let mut energy = p.frobenius_dot(h) + 0.5 * p.frobenius_dot(&j) + repulsion
        - 0.5 * hybrid.cx * (density[0].frobenius_dot(&k[0]) + density[1].frobenius_dot(&k[1]));
    if let Some(engine) = long_range {
        let k = engine.build_k_only(&setup.basis, density)?;
        energy -=
            0.5 * hybrid.clr * (density[0].frobenius_dot(&k[0]) + density[1].frobenius_dot(&k[1]));
    }
    if let Some(grid) = &setup.grid {
        energy += crate::ks::xc_contribution(&setup.functional, &setup.basis, grid, density)?.exc;
    }
    Ok(energy)
}

/// Natural energy decomposition of the interaction of molecular fragments:
/// charge transfer, electrostatic, exchange-correlation and deformation
/// energies from the natural bond orbitals of the complex.
pub(super) fn natural_energy_decomposition(system: &System, controls: &Controls) -> Result<Value> {
    let (spin, options) = read_options(controls)?;
    let fragment_of = controls.opt_usize_list("fragments")?.ok_or_else(|| {
        DftError::InvalidInput(
            "natural-energy-decomposition needs fragments, the fragment index of every atom".into(),
        )
    })?;
    let setup = setup(system, controls)?;
    controls.finish("natural-energy-decomposition")?;
    let natoms = setup.molecule.len();
    let count = fragment_of.iter().max().map_or(0, |m| m + 1);
    if fragment_of.len() != natoms || count < 2 || (0..count).any(|f| !fragment_of.contains(&f)) {
        return Err(DftError::InvalidInput(
            "fragments holds one fragment index per atom, numbering at least two fragments \
             0, 1, ... without gaps"
                .into(),
        ));
    }
    if !setup.basis.ecps.is_empty()
        || setup.options.pcm.is_some()
        || setup.options.vv10.is_some()
        || setup.functional.required_nonlocal_term().is_some()
        || setup.options.fixed_occupations.is_some()
    {
        return Err(DftError::Unsupported {
            feature: "natural energy decomposition with effective core potentials, a continuum, \
                      nonlocal correlation or prescribed populations"
                .into(),
            missing: "fragment energy functionals of these terms".into(),
        });
    }
    let reference = run_molecular_reference(&setup)?;
    let scf = &reference.scf;
    let analysis = analyse(
        &setup,
        &scf.overlap,
        &scf.density,
        &scf.fock,
        spin,
        &options,
    )?;
    let nao = &analysis.nao;
    let t = &nao.nao_coefficients;
    let n = setup.basis.nao_sph;
    // Localized density: inter-fragment NBO Fock couplings deleted, every
    // fragment occupying its Lewis count.
    let mut deformed: Vec<[Matrix; 2]> = (0..count)
        .map(|_| [Matrix::zeros(n, n), Matrix::zeros(n, n)])
        .collect();
    let mut electrons = vec![[0usize; 2]; count];
    let spins_of_channel: Vec<Vec<usize>> = if analysis.channels.len() == 2 {
        vec![vec![0], vec![1]]
    } else {
        vec![vec![0, 1]]
    };
    for (channel, spins) in analysis.channels.iter().zip(&spins_of_channel) {
        let b = t.matmul(&channel.coefficients);
        let fock_ao = if spins.len() == 2 {
            let mut f = scf.fock[0].clone();
            for (x, y) in f.as_mut_slice().iter_mut().zip(scf.fock[1].as_slice()) {
                *x = 0.5 * (*x + y);
            }
            f
        } else {
            scf.fock[spins[0]].clone()
        };
        let f = b.transpose_matmul(&fock_ao.matmul(&b));
        let mut owner = Vec::with_capacity(channel.nbos.len());
        for nbo in &channel.nbos {
            let fragment = fragment_of[nbo.atoms[0]];
            if nbo.atoms.iter().any(|&a| fragment_of[a] != fragment) {
                return Err(DftError::InvalidInput(format!(
                    "{} joins two fragments; the natural energy decomposition needs fragments \
                     without shared Lewis or antibonding orbitals",
                    nbo.label
                )));
            }
            owner.push(fragment);
        }
        for (x, (density, count_x)) in deformed.iter_mut().zip(electrons.iter_mut()).enumerate() {
            let index: Vec<usize> = (0..owner.len()).filter(|&j| owner[j] == x).collect();
            let occupied = channel
                .nbos
                .iter()
                .zip(&owner)
                .filter(|(nbo, &o)| o == x && nbo.kind.is_lewis())
                .count();
            let mut block = Matrix::zeros(index.len(), index.len());
            for (i, &p) in index.iter().enumerate() {
                for (j, &q) in index.iter().enumerate() {
                    block[(i, j)] = f[(p, q)];
                }
            }
            let (_, vectors) = crate::linalg::symmetric_eigen(&block)?;
            let mut orbitals = Matrix::zeros(n, occupied);
            for k in 0..occupied {
                for m in 0..n {
                    orbitals[(m, k)] = index
                        .iter()
                        .enumerate()
                        .map(|(i, &p)| b[(m, p)] * vectors[(i, k)])
                        .sum();
                }
            }
            let p = orbitals.matmul_transpose_seq(&orbitals);
            for &s in spins {
                density[s] = p.clone();
                count_x[s] = occupied;
            }
        }
    }
    // Operators.
    let kinetic = crate::integrals::one_electron::kinetic_matrix(&setup.basis);
    let charges: Vec<f64> = setup.molecule.atoms.iter().map(|a| a.z as f64).collect();
    let attraction = |x: Option<usize>| -> Matrix {
        let selected: Vec<f64> = charges
            .iter()
            .zip(&fragment_of)
            .map(|(&z, &f)| if x.is_none_or(|x| f == x) { z } else { 0.0 })
            .collect();
        crate::integrals::one_electron::nuclear_matrix(
            &setup.basis,
            &setup.molecule,
            Some(&selected),
        )
    };
    let repulsion = |x: Option<usize>, y: Option<usize>| -> f64 {
        let atoms = &setup.molecule.atoms;
        let mut sum = 0.0;
        for a in 0..natoms {
            for b in 0..natoms {
                let (fa, fb) = (fragment_of[a], fragment_of[b]);
                let ok = match (x, y) {
                    (None, None) => a < b,
                    (Some(x), None) => fa == x && fb == x && a < b,
                    (Some(x), Some(y)) => fa == x && fb == y,
                    (None, Some(_)) => false,
                };
                if ok {
                    sum += charges[a] * charges[b] / (atoms[a].position - atoms[b].position).norm();
                }
            }
        }
        sum
    };
    let engine = crate::scf::JkEngine::from_options(&setup.molecule, &setup.basis, &setup.options)?;
    let long_range = if setup.functional.hybrid.clr != 0.0 {
        Some(crate::scf::JkEngine::new(
            &setup.basis,
            setup.options.eri_mode,
            crate::integrals::CoulombKernel::Erf {
                omega: setup.functional.hybrid.omega,
            },
        )?)
    } else {
        None
    };
    let one = |x: Option<usize>| -> Matrix {
        let mut h = kinetic.clone();
        for (a, b) in h.as_mut_slice().iter_mut().zip(attraction(x).as_slice()) {
            *a += b;
        }
        h
    };
    let h_full = one(None);
    let full_energy = |density: [&Matrix; 2]| {
        energy_of(
            &setup,
            &engine,
            long_range.as_ref(),
            &h_full,
            repulsion(None, None),
            density,
        )
    };
    let e_scf = full_energy([&scf.density[0], &scf.density[1]])?;
    let functional_residual = (e_scf - (scf.e_elec + scf.e_nuc)).abs();
    let local: [Matrix; 2] = std::array::from_fn(|s| {
        let mut p = Matrix::zeros(n, n);
        for d in &deformed {
            for (x, y) in p.as_mut_slice().iter_mut().zip(d[s].as_slice()) {
                *x += y;
            }
        }
        p
    });
    let e_local = full_energy([&local[0], &local[1]])?;
    let idempotency = (0..2)
        .map(|s| {
            local[s]
                .matmul(&scf.overlap.matmul(&local[s]))
                .max_abs_diff(&local[s])
        })
        .fold(0.0_f64, f64::max);
    // Fragments.
    let mut fragment_values = Vec::with_capacity(count);
    let mut deformation_total = 0.0;
    let mut isolated_total = 0.0;
    let mut deformed_total = 0.0;
    for x in 0..count {
        let h = one(Some(x));
        let nn = repulsion(Some(x), None);
        let e_def = energy_of(
            &setup,
            &engine,
            long_range.as_ref(),
            &h,
            nn,
            [&deformed[x][0], &deformed[x][1]],
        )?;
        let nuclear: f64 = (0..natoms)
            .filter(|&a| fragment_of[a] == x)
            .map(|a| charges[a])
            .sum();
        let [na, nb] = electrons[x];
        let charge = nuclear.round() as i32 - (na + nb) as i32;
        let multiplicity = na.abs_diff(nb) + 1;
        let mut atoms = setup.molecule.atoms.clone();
        for (atom, &f) in atoms.iter_mut().zip(&fragment_of) {
            if f != x {
                atom.z = 0;
            }
        }
        let ghost = crate::system::Molecule::new(atoms)
            .with_charge(charge)
            .with_multiplicity(multiplicity);
        let isolated = crate::scf::run_scf(
            &ghost,
            &setup.basis,
            &setup.options,
            &setup.functional,
            setup.grid.as_ref(),
        )?;
        let e_iso = energy_of(
            &setup,
            &engine,
            long_range.as_ref(),
            &h,
            nn,
            [&isolated.density[0], &isolated.density[1]],
        )?;
        deformation_total += e_def - e_iso;
        isolated_total += e_iso;
        deformed_total += e_def;
        fragment_values.push(
            Value::object()
                .with("fragment", x)
                .with(
                    "atoms",
                    (0..natoms)
                        .filter(|&a| fragment_of[a] == x)
                        .collect::<Vec<_>>(),
                )
                .with("charge", charge as i64)
                .with("multiplicity", multiplicity as i64)
                .with("electrons", (na + nb) as i64)
                .with("isolated_energy", e_iso)
                .with("isolated_converged", isolated.converged)
                .with(
                    "isolated_functional_residual",
                    (e_iso - (isolated.e_elec + isolated.e_nuc)).abs(),
                )
                .with("deformed_energy", e_def)
                .with("deformation", e_def - e_iso),
        );
    }
    // Electrostatics of the deformed fragments.
    let mut electrostatic = 0.0;
    for x in 0..count {
        for y in x + 1..count {
            let px = spin_summed(&deformed[x]);
            let py = spin_summed(&deformed[y]);
            let (j, _) =
                engine.build(&setup.basis, &py, [&deformed[y][0], &deformed[y][1]], false)?;
            electrostatic += px.frobenius_dot(&j)
                + px.frobenius_dot(&attraction(Some(y)))
                + py.frobenius_dot(&attraction(Some(x)))
                + repulsion(Some(x), Some(y));
        }
    }
    let charge_transfer = e_scf - e_local;
    let exchange = e_local - deformed_total - electrostatic;
    let interaction = e_scf - isolated_total;
    let identity =
        (interaction - (charge_transfer + electrostatic + exchange + deformation_total)).abs();
    Ok(reference_report(
        &setup,
        &reference,
        "natural energy decomposition: charge transfer from the deletion of the inter-fragment \
         Fock couplings in the natural bond orbital basis (fragments occupying their Lewis \
         counts), electrostatics of the deformed fragment densities, exchange-correlation \
         remainder of the localized state and deformation against the isolated fragments in \
         the basis of the complex",
    )
    .with("fragments", fragment_values)
    .with("interaction_energy", interaction)
    .with("charge_transfer", charge_transfer)
    .with("electrostatic", electrostatic)
    .with("exchange_correlation", exchange)
    .with("deformation", deformation_total)
    .with("localized_energy", e_local)
    .with("decomposition_residual", identity)
    .with("energy_functional_residual", functional_residual)
    .with("localized_idempotency_residual", idempotency)
    .with(
        "units",
        units(&[
            ("energy", "hartree"),
            ("energy_components", "hartree"),
            ("fragments.isolated_energy", "hartree"),
            ("fragments.isolated_functional_residual", "hartree"),
            ("fragments.deformed_energy", "hartree"),
            ("fragments.deformation", "hartree"),
            ("interaction_energy", "hartree"),
            ("charge_transfer", "hartree"),
            ("electrostatic", "hartree"),
            ("exchange_correlation", "hartree"),
            ("deformation", "hartree"),
            ("localized_energy", "hartree"),
            ("decomposition_residual", "hartree"),
            ("energy_functional_residual", "hartree"),
            ("localized_idempotency_residual", "dimensionless"),
        ]),
    ))
}
