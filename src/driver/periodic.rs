// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Three-dimensional finite-k periodic calculations on the common driver:
//! the periodic energy with analytic forces and stress, fixed-cell and
//! variable-cell optimization, the electronic structure of the k mesh, band
//! structures along a path and the density of states.
//!
//! The reference is the finite-k Kohn-Sham/Hartree-Fock SCF of
//! [`crate::periodic::scf::run_periodic_scf`]. Lattice rows and positions are
//! Bohr; k points are reported both as Cartesian vectors (1/Bohr) and in
//! reduced coordinates of the reciprocal lattice.

use super::controls::Controls;
use super::molecular::units;
use super::value::Value;
use super::System;
use crate::grid::GridOptions;
use crate::linalg::{Matrix, Orthogonalizer};
use crate::math::Vec3;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::dispersion::{cell_dispersion, PeriodicDispersion, PeriodicDispersionResult};
use crate::periodic::occupations::OccupationScheme;
use crate::periodic::scf::{run_periodic_scf, PeriodicScfOptions, PeriodicScfResult};
use crate::periodic::{Cell, Lattice};
use crate::xc::{resolve, FunctionalSpec, ResolvedFunctional, WeightedFunctionalTerm};
use crate::{DftError, Result};

/// Reference controls of a periodic calculation.
#[derive(Clone, Debug)]
pub struct PeriodicMethod {
    pub vv10: Option<crate::periodic::nonlocal::PeriodicVv10Options>,
    pub vv10_cutoff: f64,
    pub xc: String,
    pub xc_terms: Option<Vec<WeightedFunctionalTerm>>,
    pub basis: String,
    pub ecp: Option<String>,
    pub charge: i32,
    pub multiplicity: usize,
    pub unrestricted: bool,
    pub k_mesh: [usize; 3],
    pub overlap_cutoff: f64,
    pub exchange_cutoff: Option<f64>,
    /// Independent localized XC AO point/shell sum radius (Bohr). None retains overlap_cutoff.
    pub xc_ao_cutoff: Option<f64>,
    pub grid_level: usize,
    /// Named grid (`fine`, `ultrafine`, `superfine`) replacing the level.
    pub grid_preset: Option<crate::grid::GridPreset>,
    /// Angular rule family (`lebedev`, default, or `product`).
    pub grid_angular_scheme: crate::grid::AngularScheme,
    /// Reduce angular resolution on selected radial shells.
    pub grid_prune: bool,
    /// Partitioned quadrature weights below this (Bohr^3) are dropped.
    pub grid_weight_cutoff: f64,
    pub grid_shell: i64,
    pub max_iter: usize,
    pub conv_tol: f64,
    pub conv_tol_grad: Option<f64>,
    pub conv_tol_dp: f64,
    /// Kelvin; positive values give mesh-global Fermi-Dirac occupations.
    pub electronic_temperature: f64,
    pub level_shift: f64,
    pub damping: f64,
    pub diis_size: usize,
    pub adiis: bool,
    pub adiis_until_error: f64,
    pub guess: String,
    /// Model file of the learned functional selected by `xc = skala`.
    pub skala_model: Option<String>,
    /// Empirical dispersion (`none` or periodic D3); see
    /// [`crate::periodic::dispersion`].
    pub dispersion: crate::dispersion::Dispersion,
    d1: Option<(Vec<crate::dispersion::d1::AtomParameters>, f64)>,
    /// Radius of the D3 coordination-number sum (Bohr).
    pub cn_cutoff: f64,
    /// Radius of the D3 pair sum (Bohr).
    pub disp_cutoff: f64,
    /// Adds the three-body (Axilrod-Teller-Muto) term to a D3 model.
    pub three_body: bool,
    /// Largest pair distance inside a triple of the three-body term (Bohr).
    pub abc_cutoff: f64,
    /// Split exponent of the range-separated Hartree build (Bohr^-2); a performance control.
    pub hartree_compact_exponent: Option<f64>,
}

impl PeriodicMethod {
    pub(super) fn apply_collinear_reference(&mut self, controls: &Controls) -> Result<()> {
        if let Some(reference) = controls.opt_str("reference")? {
            match reference.to_ascii_lowercase().as_str() {
                "auto" => {}
                "rks" if self.multiplicity == 1 && !self.unrestricted => {}
                "rks" => {
                    return Err(DftError::InvalidInput(
                        "periodic RKS requires multiplicity=1 and unrestricted=false".into(),
                    ));
                }
                "uks" => self.unrestricted = true,
                _ => {
                    return Err(DftError::InvalidInput(format!(
                        "unknown periodic energy reference `{reference}`"
                    )));
                }
            }
        }
        Ok(())
    }

    fn resolve_functional(&self) -> Result<ResolvedFunctional> {
        match &self.xc_terms {
            Some(terms) => ResolvedFunctional::weighted(terms),
            None => resolve(&FunctionalSpec::Combined(self.xc.clone())),
        }
    }

    fn resolve_vv10(&self) -> Option<crate::periodic::nonlocal::PeriodicVv10Options> {
        self.vv10.or_else(|| {
            if self.xc_terms.is_some() {
                return None;
            }
            crate::nlc::Vv10Params::for_functional(&self.xc).map(|parameters| {
                crate::periodic::nonlocal::PeriodicVv10Options {
                    parameters,
                    cutoff: self.vv10_cutoff,
                }
            })
        })
    }

    /// Reference controls of a calculation that does not support
    /// dispersion: a periodic D3 request is refused, never ignored.
    pub fn from_controls(controls: &Controls) -> Result<Self> {
        let method = Self::from_controls_with_dispersion(controls)?;
        if method.dispersion != crate::dispersion::Dispersion::None || method.d1.is_some() {
            return Err(DftError::Unsupported {
                feature: format!("dispersion `{}` in this periodic calculation", method.dispersion_name()),
                missing: "dispersion second derivatives (and its contribution to this feature); periodic D3 is available \
                          for periodic-energy, stress, opt-fixed-cell and cell-opt"
                    .into(),
            });
        }
        Ok(method)
    }

    /// The periodic D3 model of the requested dispersion, with the published
    /// parameters of the functional.
    pub fn dispersion_model(
        &self,
    ) -> Result<Option<crate::periodic::dispersion::PeriodicDispersion>> {
        if self.d1.is_some() {
            return Err(DftError::Unsupported {
                feature: "periodic D1 without primitive-cell atom references".into(),
                missing: "binding the D1 inputs to the primitive-cell composition".into(),
            });
        }
        let model = crate::periodic::dispersion::PeriodicDispersion::from_model(
            &self.dispersion,
            &self.xc,
            self.cn_cutoff,
            self.disp_cutoff,
        )?;
        match model {
            Some(model) if self.three_body => Ok(Some(model.with_three_body(
                crate::periodic::dispersion::ThreeBody {
                    cutoff: self.abc_cutoff,
                    ..Default::default()
                },
            )?)),
            other => Ok(other),
        }
    }

    /// Resolve explicit atom inputs against the primitive-cell composition.
    pub fn dispersion_model_for(&self, numbers: &[u8]) -> Result<Option<PeriodicDispersion>> {
        match &self.d1 {
            Some((atoms, scale)) => Ok(Some(PeriodicDispersion::d1(
                numbers,
                atoms,
                *scale,
                self.disp_cutoff,
            )?)),
            None => self.dispersion_model(),
        }
    }

    fn dispersion_name(&self) -> String {
        if self.d1.is_some() {
            "D1".into()
        } else {
            format!("{:?}", self.dispersion)
        }
    }

    /// Reference controls for periodic empirical dispersion
    /// (`dispersion` or `disp`: none, d1, d2, d3, d3bj, chg).
    pub fn from_controls_with_dispersion(controls: &Controls) -> Result<Self> {
        let model = match controls.opt_str("dispersion")? {
            Some(model) if controls.contains("disp") => {
                return Err(DftError::InvalidInput(format!(
                    "give dispersion or its alias disp, not both (dispersion={model})"
                )))
            }
            Some(model) => model,
            None => controls.str("disp", "none")?,
        };
        let model = model.trim().to_ascii_lowercase();
        let d1 = (model == "d1")
            .then(|| super::density_dispersion::d1_inputs(controls))
            .transpose()?;
        let dispersion = match model.as_str() {
            "none" | "auto" | "d1" => crate::dispersion::Dispersion::None,
            other => crate::dispersion::Dispersion::parse(other).ok_or_else(|| {
                DftError::InvalidInput(format!(
                    "unknown dispersion model `{model}` (none, d1, d2, d3, d3bj, chg)"
                ))
            })?,
        };
        let cn_cutoff =
            controls.f64("cn_cutoff", crate::periodic::dispersion::DEFAULT_CN_CUTOFF)?;
        let disp_cutoff = controls.f64(
            "disp_cutoff",
            crate::periodic::dispersion::DEFAULT_DISP_CUTOFF,
        )?;
        let includes_atm = matches!(
            dispersion,
            crate::dispersion::Dispersion::D3ZeroAbc | crate::dispersion::Dispersion::D3BjAbc
        );
        let three_body = controls.bool("three_body", includes_atm)?;
        if includes_atm && !three_body {
            return Err(DftError::InvalidInput(
                "an ATM dispersion alias requires three_body=true; select d3 or d3bj for the pair-only model".into(),
            ));
        }
        let abc_cutoff = controls.f64(
            "abc_cutoff",
            crate::periodic::dispersion::DEFAULT_ABC_CUTOFF,
        )?;
        let k = controls
            .opt_usize_list("k_mesh")?
            .unwrap_or_else(|| vec![1, 1, 1]);
        if k.len() != 3 || k.contains(&0) {
            return Err(DftError::InvalidInput(
                "k_mesh takes three positive integers".into(),
            ));
        }
        let charge = controls.opt_i64("charge")?.unwrap_or(0);
        let xc_terms = super::method::xc_terms(controls)?;
        let xc = controls.opt_str("xc")?;
        if xc_terms.is_some() {
            if xc.is_some() {
                return Err(DftError::InvalidInput(
                    "xc_terms cannot be combined with xc, x or c".into(),
                ));
            }
            if dispersion != crate::dispersion::Dispersion::None || d1.is_some() {
                return Err(DftError::Unsupported {
                    feature: "periodic weighted XC with empirical dispersion".into(),
                    missing: "dispersion parameters defined for the composed functional".into(),
                });
            }
        }
        let xc = xc.unwrap_or_else(|| "pbe".into());
        let default_nlc =
            crate::nlc::Vv10Params::for_functional(&xc).unwrap_or(crate::nlc::Vv10Params::ORIGINAL);
        let nlc_b = controls.opt_f64("vv10_b")?;
        let nlc_c = controls.opt_f64("vv10_c")?;
        let nlc_enabled = controls.bool("vv10", false)?;
        let nlc_cutoff = controls.f64("vv10_cutoff", 40.0)?;
        let nlc = (nlc_enabled || nlc_b.is_some() || nlc_c.is_some()).then(|| {
            crate::periodic::nonlocal::PeriodicVv10Options {
                parameters: crate::nlc::Vv10Params {
                    b: nlc_b.unwrap_or(default_nlc.b),
                    c: nlc_c.unwrap_or(default_nlc.c),
                },
                cutoff: nlc_cutoff,
            }
        });
        if let Some(settings) = nlc {
            settings.validate()?;
        }
        let mut method = Self {
            vv10: nlc,
            vv10_cutoff: nlc_cutoff,
            xc,
            xc_terms,
            basis: controls.str("basis", "sto-3g")?,
            ecp: controls.opt_str("ecp")?,
            charge: i32::try_from(charge)
                .map_err(|_| DftError::InvalidInput("charge out of range".into()))?,
            multiplicity: controls.usize("multiplicity", 1)?,
            unrestricted: controls.bool("unrestricted", false)?,
            k_mesh: [k[0], k[1], k[2]],
            overlap_cutoff: controls.f64("overlap_cutoff", 20.0)?,
            exchange_cutoff: controls.opt_f64("exchange_cutoff")?,
            xc_ao_cutoff: {
                let value = controls.opt_f64("xc_ao_cutoff")?;
                if value.is_some_and(|x| !x.is_finite() || x <= 0.0) {
                    return Err(DftError::InvalidInput(
                        "xc_ao_cutoff must be finite and positive (Bohr)".into(),
                    ));
                }
                value
            },
            hartree_compact_exponent: controls.opt_f64("hartree_compact_exponent")?,
            grid_level: controls.usize("grid_level", 3)?,
            grid_preset: controls
                .opt_str("grid_preset")?
                .as_deref()
                .map(crate::grid::GridPreset::parse)
                .transpose()?
                .flatten(),
            grid_angular_scheme: controls
                .opt_str("grid_angular_scheme")?
                .as_deref()
                .map(crate::grid::AngularScheme::parse)
                .transpose()?
                .unwrap_or_default(),
            grid_prune: controls.bool("grid_prune", true)?,
            grid_weight_cutoff: {
                let cutoff = controls.f64("grid_weight_cutoff", 1e-14)?;
                if !cutoff.is_finite() || cutoff < 0.0 {
                    return Err(DftError::InvalidInput(
                        "grid_weight_cutoff must be finite and non-negative".into(),
                    ));
                }
                cutoff
            },
            grid_shell: controls.opt_i64("grid_shell")?.unwrap_or(2),
            max_iter: controls.usize("max_iter", 100)?,
            conv_tol: controls.f64("conv_tol", 1e-9)?,
            conv_tol_grad: controls.opt_f64("conv_tol_grad")?,
            conv_tol_dp: controls.f64("conv_tol_dp", 1e-8)?,
            electronic_temperature: controls.f64("electronic_temperature", 0.0)?,
            level_shift: controls.f64("level_shift", 0.0)?,
            damping: controls.f64("damping", 0.25)?,
            diis_size: controls.usize("diis_size", 8)?,
            adiis: !controls.bool("no_adiis", false)?,
            adiis_until_error: controls.f64("adiis_until_error", 0.1)?,
            guess: controls.str("guess", "core")?,
            skala_model: controls.opt_str("skala_model")?,
            dispersion,
            d1,
            cn_cutoff,
            disp_cutoff,
            three_body,
            abc_cutoff,
        };
        // The defining pair correction follows the functional through every
        // periodic feature that carries a geometry-only dispersion model.
        if method.xc_terms.is_none() {
            let (_, builtin) = crate::xc::built_in_dispersion(&method.xc);
            if let Some((term, required)) =
                builtin.and_then(|t| t.pair_dispersion().map(|d| (t, d)))
            {
                if method.d1.is_some()
                    || (method.dispersion != crate::dispersion::Dispersion::None
                        && method.dispersion != required)
                    || (term == crate::xc::RequiredNonlocalTerm::D3Zero
                        && (method.three_body || method.vv10.is_some()))
                {
                    return Err(DftError::InvalidInput(format!(
                        "`{}` includes its defining {} pair correction. Select the published pair model without additional dispersion terms.",
                        method.xc, term.name()
                    )));
                }
                method.dispersion = required;
            }
        }
        if method.three_body
            && !matches!(
                method.dispersion,
                crate::dispersion::Dispersion::D3Zero
                    | crate::dispersion::Dispersion::D3Bj
                    | crate::dispersion::Dispersion::D3ZeroAbc
                    | crate::dispersion::Dispersion::D3BjAbc
            )
        {
            return Err(DftError::InvalidInput(
                "three_body needs a D3 dispersion model (dispersion = d3 or d3bj)".into(),
            ));
        }
        if method.multiplicity == 0
            || !(method.overlap_cutoff > 0.0)
            || method.grid_level > 9
            || method.grid_shell < 1
            || method.max_iter == 0
            || !(method.electronic_temperature >= 0.0)
        {
            return Err(DftError::InvalidInput(
                "periodic controls need multiplicity >= 1, positive overlap_cutoff, grid_level 0..9, \
                 grid_shell >= 1, max_iter >= 1 and a non-negative electronic_temperature (Kelvin)"
                    .into(),
            ));
        }
        Ok(method)
    }
}

/// A prepared three-dimensional periodic reference.
pub struct PeriodicSetup {
    pub method: PeriodicMethod,
    pub cell: Cell,
    pub functional: ResolvedFunctional,
    pub options: PeriodicScfOptions,
}

fn lattice_rows_of(system: &System) -> Result<Lattice> {
    let rows = system.cell.ok_or_else(|| {
        DftError::InvalidInput("a periodic calculation needs a cell (lattice rows, Bohr)".into())
    })?;
    Ok(Lattice::new(
        Vec3::new(rows[0][0], rows[0][1], rows[0][2]),
        Vec3::new(rows[1][0], rows[1][1], rows[1][2]),
        Vec3::new(rows[2][0], rows[2][1], rows[2][2]),
    ))
}

/// Geometry of the physical periodic subspace. Unused open-axis vectors may
/// be zero or dependent; only the periodic vectors define the lattice rank.
pub fn dimensional_geometry_of(system: &System) -> Result<crate::periodic::PeriodicGeometry> {
    crate::periodic::PeriodicGeometry::new(
        lattice_rows_of(system)?,
        crate::periodic::features::PeriodicBoundary::new(system.pbc),
    )
}

pub fn lattice_of(system: &System) -> Result<Lattice> {
    let lattice = lattice_rows_of(system)?;
    if !(lattice.volume().is_finite() && lattice.volume().abs() > 1e-12) {
        return Err(DftError::InvalidInput(
            "cell vectors must define a finite non-zero volume".into(),
        ));
    }
    Ok(lattice)
}

pub fn prepare_periodic(system: &System, method: &PeriodicMethod) -> Result<PeriodicSetup> {
    if system.pbc != [true; 3] {
        return Err(DftError::Unsupported {
            feature: "finite-k periodic driver calculations with open axes".into(),
            missing: "the 1D/2D route is `dimensional-scf`; this driver uses the 3D finite-k SCF"
                .into(),
        });
    }
    let lattice = lattice_of(system)?;
    let molecule = super::method::molecule_from(
        &system.numbers,
        &system.positions,
        method.charge,
        method.multiplicity,
    )?;
    let set = super::method::build_basis(&method.basis, method.ecp.as_deref())?;
    // `xc = skala` selects the learned functional of a model file instead of
    // a registered functional.
    let learned = method.xc.eq_ignore_ascii_case("skala");
    if learned != method.skala_model.is_some() {
        return Err(DftError::InvalidInput(
            "xc = skala and skala_model = <model file> are given together".into(),
        ));
    }
    let functional = if learned {
        crate::xc::ResolvedFunctional::model_based("Skala learned functional (model file)")
    } else {
        method.resolve_functional()?
    };
    let adds_pair_dispersion = match functional.required_nonlocal_kind() {
        Some(crate::xc::RequiredNonlocalTerm::D3Zero) => {
            method.dispersion == crate::dispersion::Dispersion::D3Zero
                && !method.three_body
                && method.vv10.is_none()
                && method.d1.is_none()
        }
        _ => method.dispersion == crate::dispersion::Dispersion::Chg,
    };
    let vv10 = method.resolve_vv10();
    if let Some(term) = functional
        .required_nonlocal_kind()
        .filter(|term| !term.is_supplied(vv10.is_some(), adds_pair_dispersion))
    {
        return Err(DftError::Unsupported {
            feature: format!("periodic SCF for `{}`", method.xc),
            missing: format!(
                "a periodic implementation of its defining {} nonlocal term",
                term.name()
            ),
        });
    }
    let guess = match method.guess.to_ascii_lowercase().as_str() {
        "sad" => crate::scf::Guess::Sad,
        "gwh" => crate::scf::Guess::Gwh,
        "core" => crate::scf::Guess::Core,
        other => {
            return Err(DftError::InvalidInput(format!(
                "unknown periodic guess `{other}` (sad, gwh, core)"
            )))
        }
    };
    let options = PeriodicScfOptions {
        vv10,
        k_mesh: method.k_mesh,
        occupation: if method.electronic_temperature == 0.0 {
            OccupationScheme::Insulator
        } else {
            OccupationScheme::FermiDirac {
                temperature_kelvin: method.electronic_temperature,
            }
        },
        reference: if method.unrestricted {
            crate::scf::Reference::Uhf
        } else {
            crate::scf::Reference::Auto
        },
        guess,
        use_adiis: method.adiis,
        adiis_until_error: method.adiis_until_error,
        max_iter: method.max_iter,
        e_tol: method.conv_tol,
        p_tol: method.conv_tol_dp,
        conv_tol_grad: method
            .conv_tol_grad
            .unwrap_or_else(|| method.conv_tol.sqrt()),
        level_shift: method.level_shift,
        damping: method.damping,
        diis_size: method.diis_size,
        exchange_cutoff: method.exchange_cutoff,
        xc_ao_cutoff: method.xc_ao_cutoff,
        // The non-local layers of a learned functional sum over the
        // unpartitioned atomic grids, so no point is dropped for its small
        // partition weight.
        grid: GridOptions {
            level: method.grid_level,
            preset: method.grid_preset,
            scheme: method.grid_angular_scheme,
            prune: method.grid_prune,
            weight_cutoff: if learned {
                0.0
            } else {
                method.grid_weight_cutoff
            },
            ..GridOptions::default()
        },
        grid_shell: method.grid_shell,
        pair_dispersion_from_caller: adds_pair_dispersion,
        skala: match &method.skala_model {
            Some(path) => Some(std::sync::Arc::new(crate::skala::SkalaModel::load(
                std::path::Path::new(path),
            )?)),
            None => None,
        },
        ..PeriodicScfOptions::default()
    };
    let mut cell = Cell::new(molecule, lattice, &set, method.overlap_cutoff)?;
    if let Some(value) = method.hartree_compact_exponent {
        cell = cell.with_hartree_compact_exponent(value)?;
    }
    Ok(PeriodicSetup {
        method: method.clone(),
        cell,
        functional,
        options,
    })
}

pub(super) fn run(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    let dedicated_spinor = matches!(id, "periodic-spinor-scf" | "periodic-magnetic-scf");
    if !dedicated_spinor
        && controls.peek_str("reference")?.is_some_and(|name| {
            name.eq_ignore_ascii_case("gks") || name.eq_ignore_ascii_case("ghf")
        })
    {
        if matches!(
            id,
            "periodic-energy"
                | "stress"
                | "dos"
                | "periodic-field-scf"
                | "orbital-susceptibility-q"
        ) {
            // A generalized field SCF uses the same complex reference as
            // periodic-spinor-scf. Its field controls leave nuclear SOC opt-in.
            return periodic_field_scf_impl(system, controls, true, Some(id));
        }
        return Err(DftError::Unsupported{feature:format!("periodic GKS/GHF {id}"),missing:"the corresponding generalized reference/property response; no collinear substitution is available".into()});
    }
    if !dedicated_spinor
        && controls.peek_str("reference")?.is_some_and(|name| {
            name.eq_ignore_ascii_case("roks") || name.eq_ignore_ascii_case("rohf")
        })
    {
        return super::periodic_roks::run(id, system, controls);
    }
    if !matches!(id, "phonon-bands" | "phonon-thermo") && super::double_hybrid::requested(controls)?
    {
        return super::correlated_periodic::run(id, system, controls);
    }
    let fourier_field = [
        "field_wave_indices",
        "field_vector_real",
        "field_vector_imag",
        "field_scalar",
        "field_amplitude",
    ]
    .iter()
    .any(|key| controls.contains(key));
    if (system.pbc != [true; 3] && matches!(id, "periodic-energy" | "stress" | "dos"))
        || (fourier_field && matches!(id, "periodic-energy" | "stress"))
    {
        return periodic_field_scf_impl(system, controls, false, Some(id));
    }
    // The learned functional has an energy, Kohn-Sham matrices, forces, a
    // stress, a response kernel and second derivatives with respect to
    // displacements and strain in periodic cells.
    const LEARNED: &[&str] = &[
        "periodic-energy",
        "stress",
        "opt-fixed-cell",
        "cell-opt",
        "equation-of-state",
        "electronic-structure",
        "bands",
        "dos",
        "phonons",
        "phonon-bands",
        "phonon-thermo",
        "electron-phonon",
        "quasi-harmonic",
        "elastic",
        "elastic-fd",
        "elastic-moduli",
        "christoffel",
        "material-card",
        "dielectric-tensor",
        "born-charges",
        "crystal-ir",
        "static-dielectric",
        "piezoelectric-tensor",
    ];
    // The gate follows the presence of a model file, not a functional name.
    if controls.contains("skala_model") && !LEARNED.contains(&id) {
        return Err(DftError::Unsupported {
            feature: format!(
                "`{id}` with a learned functional (Skala model file) in a periodic cell"
            ),
            missing: "the terms of the learned functional for this periodic calculation".into(),
        });
    }
    match id {
        "periodic-energy" => periodic_energy(system, controls, false),
        "stress" => periodic_energy(system, controls, true),
        "periodic-spin-response" => longitudinal_spin_response_feature(system, controls),
        "opt-fixed-cell" => opt_fixed_cell(system, controls),
        "cell-opt" => cell_opt(system, controls),
        "phonons" => phonons(system, controls),
        "phonon-bands" => phonon_bands(system, controls),
        "phonon-thermo" => phonon_thermo(system, controls),
        "electron-phonon" => electron_phonon_feature(system, controls),
        "quasi-harmonic" => quasi_harmonic_feature(system, controls),
        "elastic" => elastic(system, controls),
        "elastic-fd" => elastic_fd(system, controls),
        "elastic-moduli" => elastic_moduli_feature(system, controls),
        "christoffel" => christoffel_feature(system, controls),
        "material-card" => material_card(system, controls),
        "equation-of-state" => equation_of_state(system, controls),
        "dielectric-tensor" => dielectric_feature(system, controls, false),
        "born-charges" => dielectric_feature(system, controls, true),
        "piezoelectric-tensor" => piezoelectric_feature(system, controls),
        "periodic-natural-bond" => periodic_natural_bond(system, controls),
        "berry-polarization" => berry_polarization_feature(system, controls),
        "crystal-ir" => polar_feature(system, controls, "crystal-ir"),
        "static-dielectric" => polar_feature(system, controls, "static-dielectric"),
        "electronic-structure" => electronic_structure(system, controls),
        "bands" => bands(system, controls),
        "dos" => super::band_properties::dos(system, controls),
        "pauli-spin-susceptibility" => {
            super::band_properties::pauli_spin_susceptibility(system, controls)
        }
        "effective-mass" => effective_mass(system, controls),
        "boltzmann-transport" | "hall-transport" => boltzmann_transport(system, controls, id),
        "coherent-transport" => coherent_transport(system, controls),
        "quantum-hall" => quantum_hall(system, controls),
        "wannier-flow" => wannier_flow_feature(system, controls),
        "periodic-cubic-fc-along" | "periodic-quartic-fc-along" => {
            force_constants_along(id, system, controls)
        }
        "periodic-cubic-fc" | "periodic-quartic-fc" => force_constant_tensor(id, system, controls),
        "toec" | "foec" => high_order_elastic(id, system, controls),
        "mode-gruneisen" => mode_gruneisen(system, controls),
        "periodic-gamma" => gamma_field_response(system, controls),
        "periodic-beta" | "periodic-raman" | "electrostriction" | "piezo-higher" => {
            field_response(id, system, controls)
        }
        "periodic-efield" => finite_field(system, controls),
        "dielectric-q" => dielectric_q(system, controls),
        "orbital-susceptibility-q" => orbital_susceptibility_q(system, controls),
        "periodic-dc" | "dc-stress" => periodic_dc(id, system, controls),
        "periodic-nmr-shielding" => periodic_nmr(system, controls),
        "berry-dipole" => berry_dipole(system, controls),
        "lab-frame-spin-spiral" | "dmi" => spin_spiral(id, system, controls),
        "periodic-field-scf" => periodic_field_scf(system, controls, false),
        "periodic-spinor-scf" => periodic_field_scf(system, controls, true),
        "periodic-magnetic-scf" => periodic_magnetic_scf(system, controls),
        other => Err(DftError::Unsupported {
            feature: format!("periodic calculation `{other}`"),
            missing: "a common-driver implementation".into(),
        }),
    }
}

fn periodic_spinor_reference(controls: &Controls, feature: &str) -> Result<Option<String>> {
    let reference = controls
        .opt_str("reference")?
        .map(|name| name.to_ascii_lowercase());
    match reference.as_deref() {
        None | Some("gks" | "ghf") => Ok(reference),
        Some(name @ ("auto" | "rks" | "rhf" | "uks" | "uhf" | "roks" | "rohf")) => {
            Err(DftError::Unsupported {
                feature: format!("{feature} with reference={name}"),
                missing: "the requested reference sector; this entry solves an unconstrained complex two-component reference".into(),
            })
        }
        Some(name) => Err(DftError::InvalidInput(format!("unknown reference `{name}`"))),
    }
}

fn validate_periodic_spinor_functional(
    reference: Option<&str>,
    functional: &ResolvedFunctional,
) -> Result<()> {
    if reference == Some("ghf")
        && (!functional.parts.is_empty()
            || functional.hybrid != crate::xc::HybridParams::HF
            || functional.required_nonlocal_term().is_some())
    {
        return Err(DftError::Unsupported {
            feature: "periodic GHF with a non-HF energy provider".into(),
            missing: "an HF-consistent reference; use reference=gks for a supported generalized density functional".into(),
        });
    }
    Ok(())
}

fn periodic_magnetic_scf(system: &System, controls: &Controls) -> Result<Value> {
    use crate::periodic::electrostatics::ChargeCompensation;
    use crate::periodic::magnetic_cell::{MagneticBlochBackend, MagneticCell, MagneticScfOptions};
    use crate::periodic::spinor_scf::{run_spinor_scf, SpinorEnergyBackend, SpinorScfConfig};
    let spinor_reference = periodic_spinor_reference(controls, "periodic-magnetic-scf")?;
    if super::double_hybrid::requested(controls)? {
        return Err(DftError::Unsupported {
            feature: "double-hybrid uniform-flux magnetic SCF".into(),
            missing: "the correlated magnetic energy and PT2 response provider".into(),
        });
    }
    let method = PeriodicMethod::from_controls(controls)?;
    if method.grid_preset.is_some()
        || method.grid_angular_scheme != crate::grid::AngularScheme::Lebedev
        || !method.grid_prune
        || method.grid_weight_cutoff != 1e-14
    {
        return Err(DftError::Unsupported {
            feature: "atomic-grid preset, angular scheme or weight pruning on the magnetic-torus grid".into(),
            missing: "a mapping to the London quadrature; use magnetic_grid_order and magnetic_open_order, with grid_cutoff for open tails".into(),
        });
    }
    let electric_field = controls
        .opt_vec3("field")?
        .unwrap_or_else(Vec3::zero)
        .to_array();
    if electric_field.iter().any(|x| *x != 0.0) && method.electronic_temperature != 0.0 {
        return Err(DftError::Unsupported {
            feature: "finite electric Berry bias with metallic occupations".into(),
            missing:
                "an insulating integer-occupation magnetic reference for the occupied determinant"
                    .into(),
        });
    }
    let nuclear = super::periodic_magnetic_nuclear::Request::parse(controls, system.numbers.len())?;
    if method.dispersion != crate::dispersion::Dispersion::None
        || method.d1.is_some()
        || method.vv10.is_some()
    {
        return Err(DftError::Unsupported {feature:"dispersion/nonlocal terms on the continuum magnetic reference".into(),
            missing:"their energy and moving-source response providers; requested terms must not be silently omitted".into()});
    }
    let magnetic_model = controls.str("magnetic_model", "finite_gamma")?;
    let crystal = match magnetic_model.as_str() {
        "finite_gamma" => false,
        "crystal_bz" => true,
        _ => {
            return Err(DftError::InvalidInput(
                "magnetic_model must be finite_gamma or crystal_bz".into(),
            ))
        }
    };
    let field = controls.opt_f64_list("magnetic_field")?.ok_or_else(|| {
        DftError::InvalidInput("magnetic_field requires three atomic-unit components".into())
    })?;
    let origin = controls
        .opt_f64_list("magnetic_origin")?
        .unwrap_or_else(|| vec![0.0; 3]);
    let reciprocal_cutoff = controls.f64("magnetic_reciprocal_cutoff", 32.0)?;
    let exchange_auxiliary_width = controls.f64("magnetic_exchange_auxiliary_width", 0.5)?;
    let operator_tolerance = controls.f64("magnetic_operator_tolerance", 1e-10)?;
    let grid_order = controls.usize("magnetic_grid_order", 16 + 4 * method.grid_level)?;
    let open_quadrature_order = controls.usize("magnetic_open_order", 64)?;
    let nuclear_soc_scale = controls.f64("nuclear_soc_scale", 1.0)?;
    let nuclear_widths = controls.opt_f64_list("nuclear_widths")?.unwrap_or_default();
    let grid_cutoff = controls.f64("grid_cutoff", 12.0)?;
    let compensation = controls.str("charge_compensation", "none")?;
    let center = controls.opt_f64_list("countercharge_center")?;
    let width = controls.opt_f64("countercharge_width")?;
    let alpha = controls.opt_f64("ewald_alpha")?;
    let real = controls.opt_f64("ewald_real_cutoff")?;
    let reciprocal = controls.opt_f64("ewald_reciprocal_cutoff")?;
    let order = controls.usize("electrostatic_order", 96)?;
    let spin_real = controls.opt_f64_list("spinor_operator_real")?;
    let spin_imag = controls.opt_f64_list("spinor_operator_imag")?;
    let spinor_amplitude = controls.f64("spinor_amplitude", 1.0)?;
    let exchange_parameters = controls.opt_f64_list("exchange_coefficients")?;
    let sample_points = controls.opt_f64_list("current_points")?.unwrap_or_default();
    controls.finish("periodic-magnetic-scf")?;
    if system.pbc.iter().filter(|v| **v).count() < 2
        || field.len() != 3
        || origin.len() != 3
        || (!crystal && method.k_mesh != [1; 3])
    {
        return Err(DftError::InvalidInput(
            "uniform-flux SCF requires a physical 2D slab or 3D magnetic cell and \
            three magnetic field/origin components; finite_gamma has k_mesh=1,1,1, \
            crystal_bz integrates magnetic Bloch sectors; wires use \
            periodic-spinor-scf with uniform_magnetic_field"
                .into(),
        ));
    }
    if method.multiplicity != 1
        || method.unrestricted
        || controls.contains("spin")
        || method.level_shift != 0.0
        || !method.guess.eq_ignore_ascii_case("core")
        || controls.contains("adiis_until_error")
        || controls.contains("grid_shell")
        || method
            .exchange_cutoff
            .is_some_and(|r| r != method.overlap_cutoff)
    {
        return Err(DftError::InvalidInput(
            "magnetic spinor SCF fixes total number with a core guess, zero level \
            shift and CDIIS; separate spin constraints, ADIIS, grid_shell and \
            distinct exchange image cutoffs are inapplicable to this magnetic-torus \
            model"
                .into(),
        ));
    }
    let geometry = dimensional_geometry_of(system)?;
    let cell = MagneticCell::new(
        geometry,
        Vec3::new(field[0], field[1], field[2]),
        Vec3::new(origin[0], origin[1], origin[2]),
    )?;
    let mol = super::method::molecule_from(
        &system.numbers,
        &system.positions,
        method.charge,
        method.multiplicity,
    )?;
    let set = super::method::build_basis(&method.basis, method.ecp.as_deref())?;
    let basis = crate::basis::MolBasis::build(&mol, &set)?;
    let mut functional = method.resolve_functional()?;
    if nuclear.is_some() && (crystal || !functional.parts.is_empty()) {
        return Err(DftError::Unsupported {
            feature: "relaxed finite-flux nuclear response".into(),
            missing: "the moving semilocal or intersector providers for this magnetic reference"
                .into(),
        });
    }
    if let Some(parameters) = exchange_parameters {
        if parameters.len() != 3 || parameters.iter().any(|v| !v.is_finite()) || parameters[2] < 0.0
        {
            return Err(DftError::InvalidInput(
                "exchange_coefficients requires finite [full coefficient, long-range \
                coefficient, nonnegative omega]"
                    .into(),
            ));
        }
        functional.hybrid = crate::xc::HybridParams {
            cx: parameters[0],
            clr: parameters[1],
            omega: parameters[2],
        };
        functional.description = format!(
            "{} [explicit exchange coefficients {:?}; semilocal weights retained]",
            functional.description, parameters
        );
    }
    validate_periodic_spinor_functional(spinor_reference.as_deref(), &functional)?;
    if sample_points.len() % 3 != 0 {
        return Err(DftError::InvalidInput(
            "current_points requires flattened Cartesian triples in Bohr".into(),
        ));
    }
    let sample_points: Vec<_> = sample_points
        .chunks_exact(3)
        .map(|r| Vec3::new(r[0], r[1], r[2]))
        .collect();
    let mut options = MagneticScfOptions {
        electric_field,
        basis_cutoff: method.overlap_cutoff,
        reciprocal_cutoff,
        nuclear_soc_scale,
        nuclear_widths,
        operator_tolerance,
        grid_order,
        grid_cutoff,
        exchange_auxiliary_width,
        open_quadrature_order,
        ..MagneticScfOptions::default()
    };
    options.electrostatics.alpha = alpha;
    options.electrostatics.real_cutoff = real;
    options.electrostatics.reciprocal_cutoff = reciprocal;
    options.electrostatics.quadrature_order = order;
    options.electrostatics.compensation = match compensation.as_str() {
        "none" if center.is_none() && width.is_none() => ChargeCompensation::RequireNeutral,
        "uniform" if center.is_none() && width.is_none() => ChargeCompensation::UniformBackground,
        "gaussian" => {
            let c = center.ok_or_else(|| {
                DftError::InvalidInput("Gaussian compensation requires countercharge_center".into())
            })?;
            if c.len() != 3 {
                return Err(DftError::InvalidInput(
                    "countercharge_center requires three Bohr coordinates".into(),
                ));
            }
            ChargeCompensation::GaussianCountercharge {
                center: Vec3::new(c[0], c[1], c[2]),
                width: width.ok_or_else(|| {
                    DftError::InvalidInput(
                        "Gaussian compensation requires countercharge_width".into(),
                    )
                })?,
            }
        }
        _ => {
            return Err(DftError::InvalidInput(
                "charge_compensation must be none, uniform or gaussian with its center/width"
                    .into(),
            ))
        }
    };
    let spin_operator = match (spin_real, spin_imag) {
        (None, None) => None,
        (Some(r), Some(i)) => {
            let n = 2 * basis.nao_sph;
            if r.len() != n * n || i.len() != n * n {
                return Err(DftError::InvalidInput(format!(
                    "magnetic spinor operators require {} flattened elements per \
                    real/imaginary part",
                    n * n
                )));
            }
            let mut h = BlochMatrix::zeros(n, n);
            h.re.as_mut_slice().copy_from_slice(&r);
            h.im.as_mut_slice().copy_from_slice(&i);
            Some(h)
        }
        _ => {
            return Err(DftError::InvalidInput(
                "supply both spinor_operator_real and spinor_operator_imag".into(),
            ))
        }
    };
    if spin_operator.is_none() && controls.contains("spinor_amplitude") {
        return Err(DftError::InvalidInput(
            "spinor_amplitude requires an additional spinor operator; the quantized \
            magnetic field remains fixed"
                .into(),
        ));
    }
    let generator = spin_operator.clone();
    let applied = spin_operator.map(|mut h| {
        h.scale(spinor_amplitude);
        h
    });
    let backend = MagneticBlochBackend::new(
        &mol,
        &basis,
        cell,
        &functional,
        options,
        applied,
        method.k_mesh,
        crystal,
    )?;
    let defaults = SpinorScfConfig::default();
    let config = SpinorScfConfig {
        max_iter: if controls.contains("max_iter") {
            method.max_iter
        } else {
            defaults.max_iter
        },
        energy_tolerance: if controls.contains("conv_tol") {
            method.conv_tol
        } else {
            defaults.energy_tolerance
        },
        density_tolerance: if controls.contains("conv_tol_dp") {
            method.conv_tol_dp
        } else {
            defaults.density_tolerance
        },
        orbital_tolerance: method.conv_tol_grad.unwrap_or(defaults.orbital_tolerance),
        damping: method.damping,
        diis_size: method.diis_size,
        occupation: if method.electronic_temperature == 0.0 {
            OccupationScheme::Insulator
        } else {
            OccupationScheme::FermiDirac {
                temperature_kelvin: method.electronic_temperature,
            }
        },
        ..defaults
    };
    let report = run_spinor_scf(&backend, &config)?;
    let electric_dipole = backend.electric_dipole(&report.density)?;
    let nuclear_report = nuclear
        .map(|request| request.run(&backend, &report))
        .transpose()?;
    let grid_diagnostics = backend.grid_diagnostics(&report.density)?;
    let current = backend.sample_particle_current(&report.density, &sample_points)?;
    let spinor_energy_derivative = generator.as_ref().map_or(0.0, |h| {
        report
            .density
            .iter()
            .map(|p| p.trace_product(h) / report.density.len() as f64)
            .sum()
    });
    let mut magnetization = vec![0.0; 3];
    for (p, s) in report.density.iter().zip(backend.overlap()) {
        let pc = crate::periodic::spinor_scf::pauli_components(p)?;
        let sc = crate::periodic::spinor_scf::pauli_components(s)?;
        for a in 0..3 {
            magnetization[a] += 0.5 * pc[a + 1].trace_product(&sc[0]) / report.density.len() as f64;
        }
    }
    let matrices = |mesh: &[BlochMatrix], imag: bool| -> Vec<Vec<Vec<f64>>> {
        mesh.iter()
            .map(|p| {
                let a = if imag { &p.im } else { &p.re };
                (0..a.rows)
                    .map(|i| (0..a.cols).map(|j| a[(i, j)]).collect())
                    .collect()
            })
            .collect()
    };
    let mut components = Value::object();
    for (name, value) in [
        "source",
        "kinetic",
        "electron_nuclear",
        "ecp",
        "hartree",
        "full_exchange",
        "long_range_exchange",
        "semilocal",
        "orbital_field",
        "spin_soc",
    ]
    .iter()
    .zip(report.fields.energy_components)
    {
        components = components.with(name, value);
    }
    let model = if backend.cell.geometry.dimension() == 2 {
        if crystal {
            "2D magnetic crystal with equal-weight in-plane Bloch quadrature; \
                intersector exchange with continuous open-normal Coulomb integration \
                and a two-dimensional full/range Gaussian auxiliary; averaged charge \
                and spin-covariant semilocal XC; continuum London kinetic energy, Zeeman and \
                nuclear Pauli SOC; finite transverse charge compensation; k-mesh \
                convergence is independent of SCF and quadrature residuals; \
                core/two-electron SOC and moving semilocal responses remain separate"
        } else {
            "2D finite magnetic slab with one Gamma sector; continuum London orbital \
                coupling and open-normal Coulomb Fourier integral, including the \
                nonconstant in-plane zero mode; full/range exchange, spin-covariant \
                semilocal XC, uniform Zeeman and Pauli nuclear kinetic-momentum SOC; finite \
                transverse compensation for charged cells; crystal integration is a \
                separate model; core/two-electron SOC and moving semilocal responses remain separate"
        }
    } else if crystal {
        "3D magnetic crystal with equal-weight magnetic Bloch Brillouin-zone \
            quadrature; continuum London kinetic and intersector full/range Coulomb \
            exchange; Gaussian auxiliary q=0 exchange integration, or raw punctured \
            quadrature at zero auxiliary width; k-mesh convergence is independent \
            of SCF/integral residuals; spin-covariant semilocal XC; open-axis uniform \
            flux, core/two-electron SOC and moving semilocal responses remain separate"
    } else {
        "3D finite magnetic torus with one Gamma sector; continuum London \
            kinetic energy and zero-average periodic Coulomb and semilocal XC \
            energy; full complex spin coherence; uniform spin Zeeman and \
            one-electron Pauli nuclear SOC including kinetic-momentum contact; \
            infinite-crystal convergence, open-axis uniform flux, core/two-electron \
            SOC and moving semilocal responses remain separate"
    };
    let xc_diagnostics = spinor_xc_diagnostics(&report.fields);
    let mut output=Value::object()
        .with("model", model)
        .with("reference_type", spinor_reference.as_deref().unwrap_or("gks"))
        .with("magnetic_model", magnetic_model)
        .with("k_mesh", method.k_mesh.to_vec())
        .with(
            "k_points",
            backend
                .k_points
                .iter()
                .map(|k| k.to_array().to_vec())
                .collect::<Vec<_>>(),
        )
        .with("exchange_transfer_modes", backend.exchange_transfer_modes())
        .with(
            "magnetic_exchange_auxiliary_width",
            backend.exchange_auxiliary_width(),
        )
        .with(
            "exchange_singularity_correction",
            backend.exchange_singularity_correction().to_vec(),
        )
        .with(
            "k_integration_convention",
            if crystal {
                "Equal weights; q=G+k_left-k_right, with continuous normal integration \
            for slabs. The dimension-specific full/range Gaussian auxiliary \
            integrates the in-plane or bulk q=0 exchange limit; width zero selects \
            raw omission. Physical nonzero denominators are unchanged. Compare \
            refined meshes and auxiliary widths independently of SCF/backend \
            convergence."
            } else if backend.cell.geometry.dimension() == 2 {
                "One magnetic Gamma sector of the finite open-normal slab Green \
            operator; no in-plane exchange singularity correction."
            } else {
                "One magnetic Gamma sector of the finite zero-average-Coulomb torus; no \
            exchange singularity correction."
            },
        )
        .with("functional", functional.description.as_str())
        .with(
            "converged",
            report.converged && report.fields.operator_converged,
        )
        .with(
            "exchange_coefficients",
            vec![
                functional.hybrid.cx,
                functional.hybrid.clr,
                functional.hybrid.omega,
            ],
        )
        .with("scf_converged", report.converged)
        .with("backend_converged", report.fields.operator_converged)
        .with("scf_iterations", report.iterations)
        .with("energy", report.fields.energy)
        .with("reference_field",electric_field.to_vec())
        .with("electric_field_model",if electric_dipole.is_some() {
            "static single-point Berry electric enthalpy on a 3D finite magnetic cell; integer full-spinor occupied subspace; determinant phase continued near the electron-number-weighted nuclear charge centroid; both Cartesian fields and quantized flux fixed; uniform compensation centred at half the lattice-vector sum, explicit Gaussian compensation at its supplied centroid"
        } else {"zero external electric bias"})
        .with("free_energy", report.free_energy)
        .with("entropy_term", report.entropy_term)
        .with("energy_components", components)
        .with("xc_component_energies", report.fields.xc_component_energies)
        .with("nuclear_soc_energy", report.fields.nuclear_soc_energy)
        .with("magnetic_field", field)
        .with("magnetic_origin", origin)
        .with("flux_quanta", backend.cell.flux_quanta.to_vec())
        .with("flux_defect", backend.cell.flux_defect)
        .with("reciprocal_modes", backend.reciprocal_modes())
        .with("magnetic_reciprocal_cutoff", reciprocal_cutoff)
        .with("magnetic_operator_tolerance", operator_tolerance)
        .with("dimension", backend.cell.geometry.dimension())
        .with("magnetic_open_order", backend.open_quadrature_order())
        .with(
            "magnetic_open_initial_order",
            if backend.cell.geometry.dimension() == 2 {
                open_quadrature_order
            } else {
                0
            },
        )
        .with(
            "open_quadrature_refinement_history",
            backend
                .open_refinement_history()
                .iter()
                .map(|r| r.to_vec())
                .collect::<Vec<_>>(),
        )
        .with(
            "coulomb_boundary_convention",
            if backend.cell.geometry.dimension() == 2 {
                "Discrete in-plane reciprocal waves and continuous open-normal Fourier \
            integral; G_parallel=0 is -2 pi |z-z'|/area. Full Coulomb zero mode is \
            restored by analytic Gaussian subtraction; long-range zero mode is \
            referenced to equal open coordinates, preserving omega=0."
            } else {
                "3D zero-average periodic Coulomb"
            },
        )
        .with("image_operator_defect", backend.image_operator_defect)
        .with(
            "coulomb_operator_defect",
            report.fields.coulomb_operator_defect,
        )
        .with("xc_grid_number_defect", report.fields.xc_grid_number_defect)
        .with("magnetic_grid_order", backend.grid_order())
        .with("xc_grid_energy_defect", grid_diagnostics[0])
        .with("xc_grid_operator_defect", grid_diagnostics[1])
        .with("xc_diagnostics", xc_diagnostics)
        .with("magnetic_grid_initial_order", grid_order)
        .with(
            "xc_grid_refinement_history",
            backend
                .grid_refinement_history()
                .into_iter()
                .map(|r| r.to_vec())
                .collect::<Vec<_>>(),
        )
        .with("orbital_residual", report.orbital_residual)
        .with("occupation_consistency", report.occupation_consistency)
        .with("particle_number_error", report.particle_number_error)
        .with("metric_defect", report.metric_defect)
        .with("density_change", report.density_change)
        .with("energy_change", report.energy_change)
        .with("density_real", matrices(&report.density, false))
        .with("density_imag", matrices(&report.density, true))
        .with("fock_real", matrices(&report.fields.fock, false))
        .with("fock_imag", matrices(&report.fields.fock, true))
        .with("magnetization", magnetization)
        .with("occupations", report.occupations)
        .with(
            "chemical_potential",
            report.chemical_potential.map_or(Value::Null, Value::Float),
        )
        .with("orbital_conjugation_constraint", false)
        .with("separate_spin_population_constraint", false)
        .with(
            "spin_coefficient_conjugation_defect",
            crate::periodic::spinor_scf::spinful_time_reversal_defect(
                &report.density,
                method.k_mesh,
            )?,
        )
        .with(
            "external_field_breaks_time_reversal",
            [
                backend.cell.field.x,
                backend.cell.field.y,
                backend.cell.field.z,
            ]
            .iter()
            .any(|v| *v != 0.0),
        )
        .with(
            "time_reversal_convention",
            "Physical time reversal maps B to -B and London orbitals phi(B) to \
            phi(-B). The coefficient defect measures spin conjugation in the fixed \
            London frame.",
        )
        .with("spinor_amplitude", spinor_amplitude)
        .with("spinor_energy_derivative", spinor_energy_derivative)
        .with(
            "amplitude_perturbation",
            "additional supplied Hermitian London AO operator only, at fixed \
            magnetic flux and nuclear SOC scale",
        )
        .with(
            "current_points",
            sample_points
                .iter()
                .map(|r| vec![r.x, r.y, r.z])
                .collect::<Vec<_>>(),
        )
        .with("number_density", current.number_density)
        .with(
            "spin_density",
            current
                .spin_density
                .iter()
                .map(|v| v.to_vec())
                .collect::<Vec<_>>(),
        )
        .with(
            "source_potential_gradient",
            current
                .source_potential_gradient
                .iter()
                .map(|v| v.to_vec())
                .collect::<Vec<_>>(),
        )
        .with(
            "kinetic_particle_current_density",
            current
                .kinetic_current
                .iter()
                .map(|v| v.to_vec())
                .collect::<Vec<_>>(),
        )
        .with(
            "spin_magnetization_particle_current_density",
            current
                .spin_magnetization_current
                .iter()
                .map(|v| v.to_vec())
                .collect::<Vec<_>>(),
        )
        .with(
            "nuclear_soc_particle_current_density",
            current
                .nuclear_soc_current
                .iter()
                .map(|v| v.to_vec())
                .collect::<Vec<_>>(),
        )
        .with(
            "pauli_particle_current_density",
            current
                .particle_current
                .iter()
                .map(|v| v.to_vec())
                .collect::<Vec<_>>(),
        )
        .with(
            "pauli_electric_current_density",
            current
                .particle_current
                .iter()
                .map(|v| v.iter().map(|x| -x).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
        )
        .with(
            "current_model",
            "kinetic p+A, Zeeman spin-magnetization curl and nuclear Pauli anomalous \
            velocity; source gradient in the Ewald gauge; additional nonlocal AO \
            operators require a supplied current vertex",
        )
        .with("nuclear_soc_scale", nuclear_soc_scale)
        .with("pbc", system.pbc.to_vec())
        .with(
            "axis_convention",
            "matrices indexed by magnetic k sector, then spin-major AO (up AOs then \
            down AOs); flux faces (a1,a2),(a2,a3),(a3,a1), unused physical faces \
            zero; occupation <=1 per spinor; xc_grid_refinement_history rows [fine \
            tensor order, energy defect, Fock defect, number defect]; \
            open_quadrature_refinement_history rows [sector index, fine normal \
            order, nuclear operator defect, nuclear SOC defect]; all failed \
            refinements retained",
        )
        .with(
            "units",
            units(&[
                ("k_points", "1/bohr, Cartesian magnetic Bloch waves"),
                ("magnetic_exchange_auxiliary_width", "1/bohr"),
                (
                    "exchange_singularity_correction",
                    "Hartree, [full,long-range] Gaussian auxiliary kernel corrections",
                ),
                ("energy", "Hartree per magnetic cell"),
                ("reference_field", "hartree/e/bohr"),
                ("electric_dipole", "e*bohr"),
                ("free_energy", "Hartree per magnetic cell"),
                ("entropy_term", "Hartree per magnetic cell"),
                ("energy_components", "Hartree per magnetic cell"),
                ("xc_component_energies", "Hartree per magnetic cell"),
                ("nuclear_soc_energy", "Hartree per magnetic cell"),
                (
                    "magnetic_field",
                    "atomic magnetic field in (p+A)^2/2, A=B cross (r-origin)/2",
                ),
                ("magnetic_origin", "Bohr"),
                ("flux_quanta", "magnetic flux divided by 2 pi"),
                ("flux_defect", "magnetic flux divided by 2 pi"),
                ("magnetic_reciprocal_cutoff", "1/bohr"),
                (
                    "magnetic_open_order",
                    "dimensionless fine continuous-normal quadrature order, zero in 3D",
                ),
                (
                    "magnetic_open_initial_order",
                    "dimensionless initial normal quadrature order",
                ),
                (
                    "open_quadrature_refinement_history",
                    "[dimensionless, dimensionless, hartree, hartree] columns: sector, order, nuclear defect, nuclear SOC defect",
                ),
                ("orbital_residual", "Hartree, canonical overlap metric"),
                ("coulomb_operator_defect", "Hartree"),
                (
                    "image_operator_defect",
                    "atomic units, maximum of the overlap and kinetic differences",
                ),
                ("xc_grid_energy_defect", "Hartree per magnetic cell"),
                ("xc_grid_operator_defect", "Hartree"),
                ("particle_number_error", "electrons"),
                ("xc_grid_number_defect", "electrons"),
                ("xc_diagnostics.angular_order", "dimensionless"),
                ("xc_diagnostics.angular_energy_defect", "Hartree per magnetic cell"),
                ("xc_diagnostics.angular_operator_defect", "Hartree"),
                ("xc_diagnostics.hermiticity_defect", "Hartree"),
                ("xc_diagnostics.component_energy_defect", "Hartree per magnetic cell"),
                ("xc_diagnostics.angular_refinement_history", "[dimensionless, hartree, hartree]"),
                (
                    "xc_grid_refinement_history",
                    "[dimensionless, hartree, hartree, electrons] columns: order, energy defect per magnetic cell, operator defect, number defect",
                ),
                ("magnetization", "Pauli spin polarization"),
                (
                    "spin_coefficient_conjugation_defect",
                    "dimensionless London AO density coefficients",
                ),
                (
                    "exchange_coefficients",
                    "[dimensionless, dimensionless, 1/bohr] full fraction, long-range fraction, range-separation parameter",
                ),
                ("spinor_amplitude", "dimensionless"),
                (
                    "spinor_energy_derivative",
                    "Hartree per magnetic cell per dimensionless additional-operator \
            amplitude",
                ),
                ("current_points", "Bohr"),
                ("number_density", "electrons/bohr^3"),
                ("spin_density", "1/bohr^3, Pauli spin polarization"),
                ("source_potential_gradient", "Hartree/Bohr"),
                (
                    "kinetic_particle_current_density",
                    "electrons*hartree/hbar/bohr^2",
                ),
                (
                    "spin_magnetization_particle_current_density",
                    "electrons*hartree/hbar/bohr^2",
                ),
                (
                    "nuclear_soc_particle_current_density",
                    "electrons*hartree/hbar/bohr^2",
                ),
                (
                    "pauli_particle_current_density",
                    "electrons*hartree/hbar/bohr^2",
                ),
                (
                    "pauli_electric_current_density",
                    "e*hartree/hbar/bohr^2, electron charge=-1",
                ),
            ]),
        )
        .with(
            "warnings",
            vec![if backend.cell.geometry.dimension() == 2 {
                "The slab retains the nonconstant in-plane zero mode and continuous open \
            normal direction; the long-range zero mode is referenced to equal open \
            coordinates. Converge normal integration, in-plane k mesh, \
            reciprocal/source/basis/grid controls independently. Kinetic energy \
            contains orbital magnetic terms; nuclear_soc_energy is included in \
            spin_soc."
                    .to_string()
            } else {
                "The 3D finite torus omits G=0 in Hartree and exchange; crystal exchange \
            additionally uses its selected auxiliary correction. Independently \
            converge k mesh, reciprocal, source-Ewald, basis-image and cell sizes. \
            Kinetic energy includes orbital magnetic terms; nuclear_soc_energy is \
            already included in spin_soc."
                    .to_string()
            }],
        );
    if let Some(dipole) = electric_dipole {
        output.set("electric_dipole", dipole.to_vec());
    }
    if let Some(nuclear) = nuclear_report {
        super::periodic_magnetic_nuclear::Request::attach(&mut output, nuclear);
    }
    Ok(output)
}

fn spinor_xc_diagnostics(fields: &crate::periodic::spinor_scf::SpinorFields) -> Value {
    fields.xc_diagnostics.as_ref().map_or(Value::Null, |d| {
        Value::object()
            .with("angular_order", d.angular_order)
            .with("angular_energy_defect", d.angular_energy_defect)
            .with("angular_operator_defect", d.angular_operator_defect)
            .with(
                "angular_refinement_history",
                d.angular_refinement_history.clone(),
            )
            .with("hermiticity_defect", d.hermiticity_defect)
            .with("component_energy_defect", d.component_energy_defect)
            .with("converged", d.converged)
    })
}

fn periodic_field_scf(system: &System, controls: &Controls, spinor: bool) -> Result<Value> {
    periodic_field_scf_impl(system, controls, spinor, None)
}
pub(super) fn periodic_field_scf_impl(
    system: &System,
    controls: &Controls,
    spinor: bool,
    ordinary_feature: Option<&str>,
) -> Result<Value> {
    let ordinary = ordinary_feature.is_some();
    let dos = (ordinary_feature == Some("dos"))
        .then(|| super::band_properties::DosOptions::parse(controls))
        .transpose()?;
    let stress_feature = ordinary_feature == Some("stress");
    let with_forces = controls.bool("forces", stress_feature)?;
    let with_stress = controls.bool("stress", false)? || stress_feature;
    use crate::periodic::electrostatics::ChargeCompensation;
    use crate::periodic::magnetic_scf::StaticPeriodicField;
    use crate::periodic::scf_boundary::{
        run_dimensional_field_scf, run_dimensional_scf, run_dimensional_scf_with_geometry,
        DimensionalScfBackend, DimensionalScfOptions, GeometryDerivatives,
    };
    use crate::response::complex::Complex;
    if super::double_hybrid::requested(controls)? {
        return Err(DftError::Unsupported {
            feature: "correlated periodic field/spinor reference".into(),
            missing:
                "the requested PT2 energy and response; an SCF parent is not the correlated recipe"
                    .into(),
        });
    }
    let spinor_reference = if spinor {
        periodic_spinor_reference(controls, "periodic-spinor-scf")?
    } else {
        None
    };
    let mut method = if ordinary && (spinor || system.pbc != [true; 3]) {
        PeriodicMethod::from_controls_with_dispersion(controls)?
    } else {
        PeriodicMethod::from_controls(controls)?
    };
    let orbital_request = if ordinary_feature == Some("orbital-susceptibility-q") {
        if system.pbc != [true; 3] || method.electronic_temperature != 0.0 {
            return Err(DftError::Unsupported {
                feature: "periodic spinor orbital susceptibility".into(),
                missing:
                    "the reduced-dimensional or finite-temperature transverse current response"
                        .into(),
            });
        }
        let mesh = method.k_mesh;
        let waves = commensurate_waves("orbital-susceptibility-q", controls, mesh)?;
        let response = super::vibrational::response_config(controls)?;
        let (cell, super_method) = supercell_system(system, &method, mesh)?;
        method = super_method;
        Some((cell, mesh, waves, response))
    } else {
        None
    };
    let system = orbital_request
        .as_ref()
        .map_or(system, |request| &request.0);
    if let Some(options) = &dos {
        options.require_reference_mesh(method.k_mesh)?;
        if with_forces || with_stress {
            return Err(DftError::InvalidInput(
                "dos accepts a stationary reference without geometry derivatives".into(),
            ));
        }
    }
    if ordinary && method.hartree_compact_exponent.is_some() {
        return Err(DftError::Unsupported {
            feature: "compact Hartree model in dimensional periodic SCF".into(),
            missing: "the compact Hartree auxiliary model for these boundary conditions".into(),
        });
    }
    let mut common_reference = None;
    if ordinary && !spinor {
        if let Some(reference) = controls.opt_str("reference")? {
            let reference = reference.trim().to_ascii_lowercase();
            match reference.as_str() {
                "auto" => {}
                "rks" if method.multiplicity == 1 && !method.unrestricted => {}
                "rks" => {
                    return Err(DftError::InvalidInput(
                        "periodic RKS requires multiplicity=1 and unrestricted=false".into(),
                    ))
                }
                "uks" => method.unrestricted = true,
                "roks" | "rohf" => {
                    super::periodic_roks::validate_method(controls, &method)?;
                    common_reference = Some(reference);
                }
                _ => {
                    return Err(DftError::InvalidInput(format!(
                        "unknown periodic energy reference `{reference}`"
                    )))
                }
            }
        }
    }
    let uniform = controls.opt_f64_list("uniform_magnetic_field")?;
    let origin = controls.opt_f64_list("magnetic_origin")?;
    let indices = controls
        .opt_f64_list("field_wave_indices")?
        .unwrap_or_else(|| {
            if uniform.is_some() {
                vec![0.0; 3]
            } else {
                let axis = system.pbc.iter().position(|active| *active).unwrap_or(0);
                (0..3).map(|i| f64::from(i == axis)).collect()
            }
        });
    let ar = controls
        .opt_f64_list("field_vector_real")?
        .unwrap_or_else(|| vec![0.0; 3]);
    let ai = controls
        .opt_f64_list("field_vector_imag")?
        .unwrap_or_else(|| vec![0.0; 3]);
    let scalar = controls
        .opt_f64_list("field_scalar")?
        .unwrap_or_else(|| vec![0.0; 2]);
    let amplitude = controls.f64("field_amplitude", 1.0)?;
    let grid_cutoff = controls.f64("grid_cutoff", 12.0)?;
    let operator_tolerance = controls.f64("operator_tolerance", 1e-11)?;
    let grid_tolerance = controls.f64("grid_tolerance", 1e-5)?;
    if !operator_tolerance.is_finite()
        || operator_tolerance <= 0.0
        || !grid_tolerance.is_finite()
        || grid_tolerance <= 0.0
    {
        return Err(DftError::InvalidInput(
            "operator_tolerance and grid_tolerance must be finite and positive".into(),
        ));
    }
    let compensation = controls.str("charge_compensation", "none")?;
    let center = controls.opt_f64_list("countercharge_center")?;
    let width = controls.opt_f64("countercharge_width")?;
    let alpha = controls.opt_f64("ewald_alpha")?;
    let real = controls.opt_f64("ewald_real_cutoff")?;
    let reciprocal = controls.opt_f64("ewald_reciprocal_cutoff")?;
    let order = controls.usize("electrostatic_order", 96)?;
    let nuclear_widths = controls.opt_f64_list("nuclear_widths")?.unwrap_or_default();
    let spin_real = if spinor {
        controls
            .opt_f64_list("spin_field_real")?
            .unwrap_or_else(|| vec![0.0; 3])
    } else {
        vec![0.0; 3]
    };
    let spin_imag = if spinor {
        controls
            .opt_f64_list("spin_field_imag")?
            .unwrap_or_else(|| vec![0.0; 3])
    } else {
        vec![0.0; 3]
    };
    let soc_scale = if spinor {
        controls.f64("soc_scale", 1.0)?
    } else {
        0.0
    };
    let nuclear_soc_scale = if spinor {
        controls.f64("nuclear_soc_scale", if ordinary { 0.0 } else { 1.0 })?
    } else {
        0.0
    };
    let soc_real = if spinor {
        controls.opt_f64_list("spinor_operator_real")?
    } else {
        None
    };
    let soc_imag = if spinor {
        controls.opt_f64_list("spinor_operator_imag")?
    } else {
        None
    };
    if spinor
        && (spin_real.len() != 3
            || spin_imag.len() != 3
            || soc_real.is_some() != soc_imag.is_some())
    {
        return Err(DftError::InvalidInput("spin fields require three components and explicit operators require both real/imaginary arrays".into()));
    }
    let vv10 = method.resolve_vv10();
    if orbital_request.is_some()
        && (with_forces
            || with_stress
            || uniform.is_some()
            || origin.is_some()
            || ar.iter().chain(&ai).chain(&scalar).any(|&v| v != 0.0)
            || nuclear_soc_scale != 0.0
            || soc_real.is_some()
            || compensation == "gaussian")
    {
        return Err(DftError::Unsupported {
            feature: "periodic spinor orbital susceptibility with additional fields or derivatives".into(),
            missing: "the response with a biased orbital gauge, nonlocal spin-orbit operator, Gaussian countercharge, or moving geometry".into(),
        });
    }
    if ordinary && method.skala_model.is_some() {
        let feature = if spinor {
            "periodic GKS nonlocal/learned provider"
        } else {
            "dimensional periodic nonlocal/learned provider"
        };
        return Err(DftError::Unsupported {
            feature: feature.into(),
            missing: "the matching energy and geometry-response provider".into(),
        });
    }
    if (with_forces || with_stress)
        && (uniform.is_some()
            || nuclear_soc_scale != 0.
            || soc_real.is_some()
            || (spinor
                && ar
                    .iter()
                    .chain(&ai)
                    .chain(&scalar)
                    .chain(&spin_real)
                    .chain(&spin_imag)
                    .any(|v| *v != 0.)))
    {
        let feature = if with_stress {
            "dimensional affine stress with orbital/spin/SOC operators"
        } else {
            "dimensional forces with orbital/spin/SOC operators"
        };
        return Err(DftError::Unsupported {
            feature: feature.into(),
            missing: "the matching moving gauge and nonlocal operator derivatives".into(),
        });
    }
    controls.finish(ordinary_feature.unwrap_or(if spinor {
        "periodic-spinor-scf"
    } else {
        "periodic-field-scf"
    }))?;
    if indices.len() != 3
        || ar.len() != 3
        || ai.len() != 3
        || scalar.len() != 2
        || indices.iter().any(|x| *x != x.round())
        || system.pbc == [false; 3]
    {
        return Err(DftError::InvalidInput(
            "periodic field SCF requires 1D/2D/3D boundaries, three integer wave \
            indices, three real/imaginary vector components and a scalar \
            real/imaginary pair"
                .into(),
        ));
    }
    let geometry = dimensional_geometry_of(system)?;
    let uniform = if let Some(field) = uniform {
        let origin = origin.unwrap_or_else(|| vec![0.0; 3]);
        if field.len() != 3
            || origin.len() != 3
            || ar
                .iter()
                .chain(&ai)
                .chain(&scalar)
                .chain(&indices)
                .any(|v| *v != 0.0)
        {
            return Err(DftError::InvalidInput(
                "uniform_magnetic_field and magnetic_origin require three Cartesian \
                components; the uniform open-field model cannot ignore additional \
                Fourier orbital field controls"
                    .into(),
            ));
        }
        let f = crate::periodic::uniform_open::UniformOpenField {
            magnetic_field: Vec3::new(field[0], field[1], field[2]),
            origin: Vec3::new(origin[0], origin[1], origin[2]),
            amplitude,
        };
        f.gauge_matrix(&geometry)?;
        Some(f)
    } else {
        if origin.is_some() {
            return Err(DftError::InvalidInput(
                "magnetic_origin requires uniform_magnetic_field".into(),
            ));
        }
        None
    };
    if (0..3).any(|a| !system.pbc[a] && indices[a] != 0.0) {
        return Err(DftError::InvalidInput(
            "field_wave_indices on open axes must be zero; reciprocal lattice \
            indices label only physical periodic axes"
                .into(),
        ));
    }
    let mol = super::method::molecule_from(
        &system.numbers,
        &system.positions,
        method.charge,
        method.multiplicity,
    )?;
    let set = super::method::build_basis(&method.basis, method.ecp.as_deref())?;
    let basis = crate::basis::MolBasis::build(&mol, &set)?;
    let functional = method.resolve_functional()?;
    validate_periodic_spinor_functional(spinor_reference.as_deref(), &functional)?;
    let mut options = DimensionalScfOptions::for_boundary(geometry.boundary());
    options.scf.vv10 = vv10;
    options.basis_cutoff = method.overlap_cutoff;
    options.grid_cutoff = grid_cutoff;
    options.operator_tolerance = operator_tolerance;
    options.grid_tolerance = grid_tolerance;
    options.nuclear_widths = nuclear_widths;
    options.scf.k_mesh = method.k_mesh;
    options.scf.occupation = if method.electronic_temperature == 0.0 {
        OccupationScheme::Insulator
    } else {
        OccupationScheme::FermiDirac {
            temperature_kelvin: method.electronic_temperature,
        }
    };
    options.scf.reference = if method.unrestricted {
        crate::scf::Reference::Uhf
    } else {
        crate::scf::Reference::Auto
    };
    options.scf.guess = match method.guess.to_ascii_lowercase().as_str() {
        "core" => crate::scf::Guess::Core,
        "gwh" => crate::scf::Guess::Gwh,
        "sad" => crate::scf::Guess::Sad,
        other => {
            return Err(DftError::InvalidInput(format!(
                "unknown periodic guess {other}"
            )))
        }
    };
    options.scf.max_iter = method.max_iter;
    options.scf.e_tol = method.conv_tol;
    options.scf.p_tol = method.conv_tol_dp;
    options.scf.conv_tol_grad = method
        .conv_tol_grad
        .unwrap_or_else(|| method.conv_tol.sqrt());
    options.scf.grid.level = method.grid_level;
    options.scf.grid.preset = method.grid_preset;
    options.scf.grid.scheme = method.grid_angular_scheme;
    options.scf.grid.prune = method.grid_prune;
    options.scf.grid.weight_cutoff = method.grid_weight_cutoff;
    options.scf.grid_shell = method.grid_shell;
    options.scf.exchange_cutoff = method.exchange_cutoff;
    options.scf.xc_ao_cutoff = method.xc_ao_cutoff;
    options.scf.use_adiis = method.adiis;
    options.scf.adiis_until_error = method.adiis_until_error;
    options.scf.diis_size = method.diis_size;
    options.scf.level_shift = method.level_shift;
    options.scf.damping = method.damping;
    options.electrostatics.alpha = alpha;
    options.electrostatics.real_cutoff = real;
    options.electrostatics.reciprocal_cutoff = reciprocal;
    options.electrostatics.quadrature_order = order;
    options.electrostatics.compensation = match compensation.as_str() {
        "none" if center.is_none() && width.is_none() => ChargeCompensation::RequireNeutral,
        "uniform" if center.is_none() && width.is_none() => ChargeCompensation::UniformBackground,
        "gaussian" => {
            let c = center.ok_or_else(|| {
                DftError::InvalidInput("Gaussian compensation requires countercharge_center".into())
            })?;
            if c.len() != 3 {
                return Err(DftError::InvalidInput(
                    "countercharge_center requires three Bohr coordinates".into(),
                ));
            }
            ChargeCompensation::GaussianCountercharge {
                center: Vec3::new(c[0], c[1], c[2]),
                width: width.ok_or_else(|| {
                    DftError::InvalidInput(
                        "Gaussian compensation requires countercharge_width".into(),
                    )
                })?,
            }
        }
        _ => {
            return Err(DftError::InvalidInput(
                "charge_compensation must be none, uniform or gaussian with its \
            center/width; unrelated compensation controls cannot be ignored"
                    .into(),
            ))
        }
    };
    let b = geometry.reciprocal();
    let field = StaticPeriodicField {
        wave: b[0] * indices[0] + b[1] * indices[1] + b[2] * indices[2],
        vector_potential: std::array::from_fn(|a| Complex::new(ar[a], ai[a])),
        scalar_potential: Complex::new(scalar[0], scalar[1]),
        amplitude,
    };
    let field_free = uniform.is_none() && ar.iter().chain(&ai).chain(&scalar).all(|v| *v == 0.0);
    let dispersion_model = method.dispersion_model_for(&system.numbers)?;
    let spin_perturbed = nuclear_soc_scale != 0.0
        || soc_real.is_some()
        || spin_real.iter().chain(&spin_imag).any(|v| *v != 0.0);
    if dispersion_model.is_some() && (!field_free || spin_perturbed) {
        return Err(DftError::Unsupported {
            feature: "dimensional periodic dispersion with orbital/spin/SOC operators".into(),
            missing: "acceptance of the combined field and dispersion geometry response".into(),
        });
    }
    // Pair terms depend only on the geometry; the KS equations use the electronic energy.
    let dispersion = dispersion_model
        .as_ref()
        .map(|model| {
            let positions: Vec<_> = mol.atoms.iter().map(|atom| atom.position).collect();
            crate::periodic::dispersion::dimensional_dispersion(
                &system.numbers,
                &positions,
                &geometry,
                model,
            )
        })
        .transpose()?;
    options.scf.pair_dispersion_from_caller = dispersion_model.as_ref().is_some_and(|model| {
        functional
            .required_nonlocal_kind()
            .is_some_and(|term| model.supplies(term))
            || matches!(
                model.damping,
                crate::periodic::dispersion::D3Damping::Chg { .. }
            )
    });
    if (!field_free || spin_perturbed) && options.scf.vv10.is_some() {
        return Err(DftError::Unsupported {
            feature: "dimensional periodic VV10 under an orbital field".into(),
            missing: "a validated nonlocal provider for the requested field sector".into(),
        });
    }
    if let Some(reference) = common_reference {
        if !field_free {
            return Err(DftError::Unsupported {
                feature: "dimensional Gamma ROKS in an orbital field".into(),
                missing: "the constrained common-orbital field reference".into(),
            });
        }
        let backend = DimensionalScfBackend::new(&mol, &basis, &geometry, &functional, &options)?;
        let mut report = super::periodic_roks::dimensional_report(
            &backend,
            &options,
            &geometry,
            &reference,
            &functional,
            [with_forces, with_stress],
            dos.as_ref(),
            dispersion.as_ref(),
        )?;
        report_dispersion(&mut report, &method, dispersion.as_ref());
        return Ok(report);
    }
    if spinor {
        if spin_real.len() != 3
            || spin_imag.len() != 3
            || (soc_real.is_some() != soc_imag.is_some())
        {
            return Err(DftError::InvalidInput(
                "spin fields require three real/imaginary components; supply both \
                spinor_operator_real and spinor_operator_imag"
                    .into(),
            ));
        }
        if (controls.contains("mult") && method.multiplicity != 1)
            || controls.contains("spin")
            || method.unrestricted
            || method.level_shift != 0.0
            || !method.guess.eq_ignore_ascii_case("core")
            || controls.contains("adiis_until_error")
            || (controls.contains("no_adiis") && method.adiis)
        {
            return Err(DftError::InvalidInput(
                "spinor SCF fixes total particle number, not separate spin populations; \
                use the core guess and CDIIS with zero level shift"
                    .into(),
            ));
        }
        use crate::periodic::spinor_scf::{
            periodic_pauli_operators, run_spinor_scf, SpinorDftBackend, SpinorEnergyBackend,
            SpinorScfConfig,
        };
        let points = geometry.k_points(options.scf.k_mesh)?;
        // The orbital periodic magnetic field also enters the spin Zeeman term.
        let aq = field.vector_potential;
        let q = field.wave;
        let zeeman = std::array::from_fn(|a| {
            let x = (a + 1) % 3;
            let y = (a + 2) % 3;
            Complex::new(spin_real[a], spin_imag[a])
                + Complex::new(0.0, 1.0) * (aq[y] * q.get(x) - aq[x] * q.get(y))
        });
        let mut first = periodic_pauli_operators(
            &basis,
            &geometry,
            &points,
            options.basis_cutoff,
            q,
            field.scalar_potential,
            zeeman,
            soc_scale,
        )?;
        let second = crate::periodic::spinor_scf::periodic_pauli_contact(
            &basis,
            &geometry,
            &points,
            options.basis_cutoff,
            &field,
            soc_scale,
        )?;
        if let (Some(real), Some(imag)) = (soc_real, soc_imag) {
            let n = 2 * basis.nao_sph;
            let count = points.len() * n * n;
            if real.len() != count || imag.len() != count {
                return Err(DftError::InvalidInput(format!(
                    "general spinor operators require {count} flattened real and imaginary \
                    elements in [k][spin-major AO][spin-major AO] order"
                )));
            }
            for ik in 0..points.len() {
                for i in 0..n {
                    for j in 0..n {
                        let index = (ik * n + i) * n + j;
                        first[ik].re[(i, j)] += real[index];
                        first[ik].im[(i, j)] += imag[index];
                    }
                }
            }
        }
        let mut operator = first.clone();
        let mut derivative = first.clone();
        for ik in 0..points.len() {
            operator[ik].scale(amplitude);
            operator[ik].add_scaled(&second[ik], 0.5 * amplitude * amplitude);
            derivative[ik].add_scaled(&second[ik], amplitude);
        }
        let mut backend = SpinorDftBackend::new(
            &mol,
            &basis,
            &geometry,
            &functional,
            &options,
            if uniform.is_some()
                || ((ordinary || with_forces || with_stress)
                    && ar.iter().chain(&ai).chain(&scalar).all(|v| *v == 0.))
            {
                None
            } else {
                Some(&field)
            },
            operator,
        )?;
        if let Some(f) = &uniform {
            backend = backend.with_uniform_open_field(f)?;
        }
        let backend = backend.with_nuclear_soc(nuclear_soc_scale)?;
        let defaults = SpinorScfConfig::default();
        let config = SpinorScfConfig {
            max_iter: if controls.contains("max_iter") {
                method.max_iter
            } else {
                defaults.max_iter
            },
            energy_tolerance: if controls.contains("conv_tol") {
                method.conv_tol
            } else {
                defaults.energy_tolerance
            },
            density_tolerance: if controls.contains("conv_tol_dp") {
                method.conv_tol_dp
            } else {
                defaults.density_tolerance
            },
            orbital_tolerance: method.conv_tol_grad.unwrap_or(defaults.orbital_tolerance),
            damping: method.damping,
            diis_size: method.diis_size,
            occupation: options.scf.occupation,
            ..defaults
        };
        let report = run_spinor_scf(&backend, &config)?;
        if let Some((_, mesh, waves, response)) = &orbital_request {
            let replicas = mesh.iter().product();
            let points = backend.orbital_susceptibility_q(&report, replicas, waves, response)?;
            let volume = geometry.lattice().volume().abs() / replicas as f64;
            let mut value = spinor_orbital_susceptibility_report(
                &report,
                &points,
                *mesh,
                volume,
                &method.basis,
                &functional,
                spinor_reference.as_deref().unwrap_or("gks"),
            );
            report_dispersion(&mut value, &method, dispersion.as_ref());
            return Ok(value);
        }
        let nonlocal_energy = report.fields.periodic_vv10_energy(vv10.is_some());
        if let Some(dos) = &dos {
            if !report.converged {
                return Err(DftError::ScfNotConverged {
                    iterations: report.iterations,
                    error: report.orbital_residual,
                });
            }
            let mut output = super::band_properties::dos_reference(
                "Gaussian-broadened DOS of the stationary spinor Fock operator",
                spinor_reference.as_deref().unwrap_or("gks"),
                &functional,
                report.fields.energy,
                report.iterations,
                report.orbital_residual,
            )
            .with("dimension", geometry.dimension())
            .with("k_mesh", method.k_mesh)
            .with("pbc", system.pbc)
            .with("chemical_potential", report.chemical_potential);
            append_spinor_vv10(&mut output, vv10.zip(nonlocal_energy));
            report_dispersion(&mut output, &method, dispersion.as_ref());
            return super::band_properties::stationary_mesh_dos(
                output,
                &basis,
                mol.len(),
                backend.overlap(),
                std::slice::from_ref(&report.bands),
                1.0,
                dos,
            );
        }
        let nuclear_gradient = if with_forces {
            Some(backend.nuclear_gradient(&report, &config)?)
        } else {
            None
        };
        let lattice_stress = if with_stress {
            Some(backend.stress(&report, &config)?)
        } else {
            None
        };
        let orbital = if let Some(f) = &uniform {
            f.operators(&basis, &geometry, &points, options.basis_cutoff)?
        } else {
            field.operators(&basis, &geometry, &points, options.basis_cutoff)?
        };
        let energy_derivative = backend.amplitude_derivative(
            &report.density,
            &orbital.first_derivative,
            &derivative,
        )?;
        let matrices = |mesh: &[BlochMatrix], imag: bool| -> Vec<Vec<Vec<f64>>> {
            mesh.iter()
                .map(|p| {
                    let a = if imag { &p.im } else { &p.re };
                    (0..a.rows)
                        .map(|i| (0..a.cols).map(|j| a[(i, j)]).collect())
                        .collect()
                })
                .collect()
        };
        let magnetization = report
            .density
            .iter()
            .zip(backend.overlap())
            .map(|(p, s)| {
                let pc = crate::periodic::spinor_scf::pauli_components(p)?;
                let sc = crate::periodic::spinor_scf::pauli_components(s)?;
                Ok(std::array::from_fn::<_, 3, _>(|a| {
                    pc[a + 1].trace_product(&sc[0]) * 0.5 / points.len() as f64
                }))
            })
            .collect::<Result<Vec<_>>>()?
            .iter()
            .fold([0.0; 3], |mut total, row| {
                for a in 0..3 {
                    total[a] += row[a];
                }
                total
            });
        let mut components = Value::object();
        for (name, value) in [
            "source",
            "kinetic",
            "electron_nuclear",
            "ecp",
            "hartree",
            "full_exchange",
            "long_range_exchange",
            "semilocal",
            "orbital_field",
            "spin_soc",
        ]
        .iter()
        .zip(report.fields.energy_components)
        {
            components = components.with(name, value);
        }
        if let Some(nonlocal) = nonlocal_energy {
            components.set("semilocal", report.fields.energy_components[7] - nonlocal);
            components.set("nonlocal_correlation", nonlocal);
        }
        let xc_diagnostics = spinor_xc_diagnostics(&report.fields);
        let band_energies: Vec<Vec<f64>> =
            report.bands.iter().map(|b| b.energies.clone()).collect();
        // Reuse the ordinary gap summary only for an actual integer occupied
        // prefix of the same size at every k. Thermal/metallic or empty/full
        // manifolds retain their raw bands, without a fictitious insulating gap.
        let occupied: Vec<usize> = report
            .occupations
            .iter()
            .map(|row| row.iter().take_while(|&&f| f == 1.0).count())
            .collect();
        let integer_manifold = method.electronic_temperature == 0.0
            && !occupied.is_empty()
            && occupied.iter().all(|&n| n == occupied[0])
            && report
                .occupations
                .iter()
                .zip(&occupied)
                .all(|(row, &n)| row[n..].iter().all(|&f| f == 0.0));
        let band_edges = if integer_manifold {
            gap_summary(&band_energies, occupied[0])
        } else {
            Value::Null
        };
        let mut output=Value::object()
            .with("reference_type", spinor_reference.as_deref().unwrap_or("gks"))
            .with("band_energies", band_energies)
            .with("band_edges", band_edges)
            .with("band_eigen_residuals", report.bands.iter().map(|b| b.eigen_residual).collect::<Vec<_>>())
            .with("band_metric_defects", report.bands.iter().map(|b| b.metric_defect).collect::<Vec<_>>())
            .with("k_points", points.iter().map(|k| k.to_array()).collect::<Vec<_>>())
            .with("band_sampling", "SCF mesh, one occupation per spinor; no extra factor of two. Arbitrary-path interpolation and DOS are separate calculations.")
            .with(
                "model",
                if uniform.is_some() {
                    "Continuum uniform-field two-component Bloch SCF for wires and in-plane \
            slab fields; periodic linear gauge, full orbital minimal coupling, \
            sigma.B/2 and Pauli nuclear grad(V) cross (p+A); compensated \
            open-boundary Coulomb, semilocal spinor XC and full/range exchange; \
            common gauge-origin AO phase; two-electron/core SOC and relativistic \
            corrections are separate models"
                } else {
                    "variational two-component complex Bloch SCF; semilocal spinor XC and \
            full/range exchange; general Hermitian spin/SOC operators; one-electron \
            Pauli nuclear SOC in the dimensional Coulomb gauge, \
            prescribed-potential Pauli SOC with kinetic-momentum contact and \
            periodic Zeeman field; two-electron/core SOC, scalar relativistic terms \
            and uniform-flux magnetic translations are separate tracked calculations"
                },
            )
            .with(
                "xc_convention",
                if functional.family == crate::xc::Family::Lda {
                    "rotationally invariant local spin-density eigenvalues"
                } else {
                    "global spin projection of f+D_spin f with the exact collinear limit; \
                all spin-density gradients; meta-GGA tau-j^2/(2 rho) in the covariant \
                momentum frame; angular convergence of the actual energy and Fock; \
                spatial grid electron count measured separately"
                },
            )
            .with("xc_diagnostics", xc_diagnostics)
            .with(
                "uniform_magnetic_field",
                uniform.as_ref().map_or(Value::Null, |f| {
                    Value::Array(f.magnetic_field.to_array().map(Value::Float).to_vec())
                }),
            )
            .with(
                "magnetic_origin",
                uniform.as_ref().map_or(Value::Null, |f| {
                    Value::Array(f.origin.to_array().map(Value::Float).to_vec())
                }),
            )
            .with(
                "magnetic_gauge_convention",
                if uniform.is_some() {
                    "A=lambda M(r-origin); common AO phase exp(i lambda (M origin).r); \
                matrices evaluated in the origin-zero gauge frame; curl A=lambda B; \
                unaltered open-axis electrostatics"
                } else {
                    "real periodic Fourier gauge"
                },
            )
            .with(
                "raw_field_hermitian_residual",
                orbital.raw_hermitian_residual,
            )
            .with("functional", functional.description.as_str())
            .with(
                "converged",
                crate::periodic::spinor_scf::acceptance::reference_converged(
                    report.converged,
                    report.fields.operator_converged,
                    report.fields.xc_diagnostics.as_ref(),
                )
                    && backend.nuclear_soc_diagnostics()[0] <= options.operator_tolerance,
            )
            .with("scf_converged", report.converged)
            .with("backend_converged", report.fields.operator_converged)
            .with("scf_iterations", report.iterations)
            .with("energy", report.fields.energy)
            .with("free_energy", report.free_energy)
            .with("entropy_term", report.entropy_term)
            .with("energy_components", components)
            .with(
                "xc_component_energies",
                report.fields.xc_component_energies.clone(),
            )
            .with("nuclear_soc_energy", report.fields.nuclear_soc_energy)
            .with("nuclear_soc_scale", nuclear_soc_scale)
            .with("soc_scale", soc_scale)
            .with(
                "nuclear_soc_quadrature_defect",
                backend.nuclear_soc_diagnostics()[0],
            )
            .with(
                "nuclear_soc_hermitian_defect",
                backend.nuclear_soc_diagnostics()[1],
            )
            .with(
                "coulomb_operator_defect",
                report.fields.coulomb_operator_defect,
            )
            .with("xc_grid_number_defect", report.fields.xc_grid_number_defect)
            .with("field_energy_derivative", energy_derivative)
            .with("field_amplitude", amplitude)
            .with("dimension", geometry.dimension())
            .with("pbc", system.pbc.to_vec())
            .with("density_real", matrices(&report.density, false))
            .with("density_imag", matrices(&report.density, true))
            .with("fock_real", matrices(&report.fields.fock, false))
            .with("fock_imag", matrices(&report.fields.fock, true))
            .with("occupations", report.occupations)
            .with(
                "chemical_potential",
                report.chemical_potential.map_or(Value::Null, Value::Float),
            )
            .with("magnetization", magnetization.to_vec())
            .with("orbital_residual", report.orbital_residual)
            .with("particle_number_error", report.particle_number_error)
            .with("occupation_consistency", report.occupation_consistency)
            .with("metric_defect", report.metric_defect)
            .with("density_change", report.density_change)
            .with("energy_change", report.energy_change)
            .with("orbital_conjugation_constraint", false)
            .with("separate_spin_population_constraint", false)
            .with(
                "spinful_density_residual",
                crate::periodic::spinor_scf::spinful_time_reversal_defect(
                    &report.density,
                    options.scf.k_mesh,
                )?,
            )
            .with(
                "spinful_fock_residual",
                crate::periodic::spinor_scf::spinful_time_reversal_defect(
                    &report.fields.fock,
                    options.scf.k_mesh,
                )?,
            )
            .with(
                "axis_convention",
                "k, spin-major AO (up AOs then down AOs); complex parts separate; one \
                chemical potential and occupation <=1 per spinor",
            )
            .with(
                "units",
                units(&[
                    ("energy", "Hartree per cell"),
                    ("band_energies", "hartree, [k][spinor band]"),
                    ("band_eigen_residuals", "dimensionless, normalized generalized eigen residual per k"),
                    ("band_metric_defects", "dimensionless"),
                    ("k_points", "1/bohr"),
                    ("band_edges.valence_maximum", "hartree"),
                    ("band_edges.conduction_minimum", "hartree"),
                    ("band_edges.gap", "hartree"),
                    ("band_edges.direct_gap", "hartree"),
                    ("free_energy", "Hartree per cell"),
                    (
                        "field_energy_derivative",
                        "Hartree per cell per dimensionless amplitude",
                    ),
                    (
                        "magnetization",
                        "electron spin polarization, Pauli expectation",
                    ),
                    (
                        "spin_field_real",
                        "atomic magnetic field Fourier coefficient",
                    ),
                    (
                        "uniform_magnetic_field",
                        "reduced atomic magnetic field; applied curl A is field_amplitude times \
                this vector",
                    ),
                    ("magnetic_origin", "Bohr"),
                    ("spinor_operator_real", "Hartree at unit amplitude"),
                    ("xc_grid_number_defect", "electrons"),
                    (
                        "xc_diagnostics.angular_order",
                        "dimensionless global spin projection order",
                    ),
                    ("xc_diagnostics.angular_energy_defect", "Hartree per cell"),
                    ("xc_diagnostics.angular_operator_defect", "Hartree"),
                    ("xc_diagnostics.hermiticity_defect", "Hartree"),
                    ("xc_diagnostics.component_energy_defect", "Hartree per cell"),
                    (
                        "xc_diagnostics.angular_refinement_history",
                        "[dimensionless, hartree, hartree] columns: order, energy difference per cell, Fock difference",
                    ),
                    ("orbital_residual", "Hartree, canonical overlap metric"),
                    ("particle_number_error", "electrons"),
                    ("coulomb_operator_defect", "Hartree"),
                    (
                        "raw_field_hermitian_residual",
                        "Hartree, unprojected",
                    ),
                    (
                        "xc_component_energies",
                        "Hartree per magnetic cell",
                    ),
                    ("nuclear_soc_energy", "Hartree per magnetic cell"),
                    ("field_amplitude", "dimensionless"),
                ]),
            );
        append_dimensional_geometry(
            &mut output,
            &geometry,
            GeometryDerivatives {
                gradient: nuclear_gradient,
                stress: lattice_stress,
            },
            dispersion.as_ref(),
            "complex-spinor",
        )?;
        append_spinor_vv10(&mut output, vv10.zip(nonlocal_energy));
        report_dispersion(&mut output, &method, dispersion.as_ref());
        return Ok(output);
    }
    let mut derivatives = None;
    let report = if with_forces || with_stress {
        let (report, geometry_derivatives) = run_dimensional_scf_with_geometry(
            &mol,
            &basis,
            &geometry,
            &functional,
            &options,
            (!field_free).then_some(&field),
            [with_forces, with_stress],
        )?;
        derivatives = Some(geometry_derivatives);
        report
    } else if ordinary && field_free {
        run_dimensional_scf(&mol, &basis, &geometry, &functional, &options)?
    } else if let Some(f) = &uniform {
        crate::periodic::scf_boundary::run_dimensional_uniform_scf(
            &mol,
            &basis,
            &geometry,
            &functional,
            &options,
            f,
        )?
    } else {
        run_dimensional_field_scf(&mol, &basis, &geometry, &functional, &options, &field)?
    };
    let r = &report.reference;
    if let Some(dos) = &dos {
        let residual = report.reference_diagnostics.orbital_residual;
        if !report.converged {
            return Err(DftError::ScfNotConverged {
                iterations: r.iterations,
                error: residual,
            });
        }
        let bands = r
            .fock
            .iter()
            .take(if r.unrestricted { 2 } else { 1 })
            .map(|spin| {
                spin.iter()
                    .zip(&r.overlap)
                    .map(|(f, s)| {
                        crate::dfpt::orbitals::generalized_orbitals(
                            f,
                            s,
                            options.scf.orth_threshold,
                        )
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let mut output = super::band_properties::dos_reference(
            "Gaussian-broadened DOS of the stationary dimensional collinear Fock operator",
            if r.unrestricted { "uks" } else { "rks" },
            &functional,
            r.energy,
            r.iterations,
            residual,
        )
        .with("dimension", geometry.dimension())
        .with("k_mesh", method.k_mesh)
        .with("pbc", system.pbc)
        .with("chemical_potential", r.chemical_potential.to_vec());
        report_dispersion(&mut output, &method, dispersion.as_ref());
        return super::band_properties::stationary_mesh_dos(
            output,
            &basis,
            mol.len(),
            &r.overlap,
            &bands,
            if r.unrestricted { 1.0 } else { 2.0 },
            dos,
        );
    }
    let (derivative, field_hermiticity) = if ordinary && field_free {
        (0.0, 0.0)
    } else {
        let operator = if let Some(f) = &uniform {
            f.operators(&basis, &geometry, &r.k_points, options.basis_cutoff)?
        } else {
            field.operators(&basis, &geometry, &r.k_points, options.basis_cutoff)?
        };
        let derivative = r
            .total_density()
            .iter()
            .zip(&operator.first_derivative)
            .map(|(p, h)| p.trace_product(h))
            .sum::<f64>()
            / r.k_points.len() as f64;
        (derivative, operator.raw_hermitian_residual)
    };
    let tr = r.time_reversal_diagnostics(method.k_mesh)?;
    let matrices = |spin: &[Vec<BlochMatrix>; 2], imaginary: bool| -> Vec<Vec<Vec<Vec<f64>>>> {
        spin.iter()
            .map(|mesh| {
                mesh.iter()
                    .map(|m| {
                        let a = if imaginary { &m.im } else { &m.re };
                        (0..a.rows)
                            .map(|i| (0..a.cols).map(|j| a[(i, j)]).collect())
                            .collect()
                    })
                    .collect()
            })
            .collect()
    };
    let mut output = Value::object()
        .with("model", report.model)
        .with("functional", functional.description.as_str())
        .with("reference_type", if r.unrestricted { "uks" } else { "rks" })
        .with("unrestricted", r.unrestricted)
        .with(
            "k_points",
            r.k_points.iter().map(|k| k.to_array()).collect::<Vec<_>>(),
        )
        .with(
            "xc_component_energies",
            &report.semilocal_component_energies,
        )
        .with(
            "xc_component_labels",
            report.semilocal_component_labels.clone(),
        )
        .with(
            "uniform_magnetic_field",
            uniform.as_ref().map_or(Value::Null, |f| {
                Value::Array(f.magnetic_field.to_array().map(Value::Float).to_vec())
            }),
        )
        .with(
            "magnetic_origin",
            uniform.as_ref().map_or(Value::Null, |f| {
                Value::Array(f.origin.to_array().map(Value::Float).to_vec())
            }),
        )
        .with("converged", report.converged)
        .with("scf_converged", r.converged)
        .with("scf_iterations", r.iterations)
        .with("energy", r.energy)
        .with("free_energy", r.free_energy)
        .with("entropy_term", r.entropy_term)
        .with("external_field_energy", report.external_field_energy)
        .with("field_energy_derivative", derivative)
        .with(
            "energy_components",
            Value::object()
                .with("source", r.e_nuc)
                .with("kinetic", report.kinetic_energy)
                .with("electron_nuclear", report.electron_nuclear_energy)
                .with("ecp", report.ecp_energy)
                .with("hartree", report.hartree_energy)
                .with("full_exchange", report.full_exchange_energy)
                .with("long_range_exchange", report.long_range_exchange_energy)
                .with("external_field", report.external_field_energy)
                .with("semilocal", r.e_xc - r.e_nlc)
                .with("nonlocal_correlation", r.e_nlc),
        )
        .with("field_amplitude", amplitude)
        .with("field_wave_indices", indices)
        .with("wave_cartesian", field.wave.to_array().to_vec())
        .with("pbc", system.pbc.to_vec())
        .with("dimension", geometry.dimension())
        .with("n_alpha", r.n_alpha)
        .with("n_beta", r.n_beta)
        .with(
            "orbital_conjugation_constraint",
            r.orbital_conjugation_constraint,
        )
        .with(
            "orbital_density_residual",
            tr.orbital_density_residual.to_vec(),
        )
        .with("orbital_fock_residual", tr.orbital_fock_residual.to_vec())
        .with("spinful_density_residual", tr.spinful_density_residual)
        .with("spinful_fock_residual", tr.spinful_fock_residual)
        .with(
            "orbital_residual",
            report.reference_diagnostics.orbital_residual,
        )
        .with("occupation_consistency", report.occupation_consistency)
        .with(
            "particle_number_error",
            report.reference_diagnostics.particle_number_error,
        )
        .with("component_energy_defect", report.component_energy_defect)
        .with("final_fock_change", report.final_fock_change)
        .with("coulomb_operator_defect", report.coulomb_operator_defect)
        .with("grid_number_defect", report.grid_number_defect)
        .with("raw_field_hermitian_residual", field_hermiticity)
        .with("density_real", matrices(&r.density, false))
        .with("density_imag", matrices(&r.density, true))
        .with("fock_real", matrices(&r.fock, false))
        .with("fock_imag", matrices(&r.fock, true))
        .with("warnings", report.warnings)
        .with(
            "units",
            units(&[
                ("energy", "Hartree per cell"),
                ("free_energy", "Hartree per cell"),
                ("k_points", "1/bohr"),
                ("xc_component_energies", "hartree per cell"),
                ("external_field_energy", "Hartree per cell"),
                (
                    "field_energy_derivative",
                    "Hartree per dimensionless field amplitude per cell; variational \
                Hellmann-Feynman derivative of E at zero T or F at fixed T and particle \
                number",
                ),
                (
                    "field_vector_real/imag",
                    "reduced vector potential in (p+A)^2/2 atomic units",
                ),
                ("wave_cartesian", "1/bohr"),
                ("fock_real/imag", "Hartree"),
                (
                    "density_real/imag",
                    "dimensionless contravariant AO density",
                ),
                (
                    "orbital_density_residual",
                    "dimensionless raw AO conjugation defect",
                ),
                ("orbital_fock_residual", "Hartree raw AO conjugation defect"),
                (
                    "field_scalar",
                    "hartree, Fourier coefficient of the electron potential energy",
                ),
                ("uniform_magnetic_field", "reduced atomic magnetic field"),
                ("magnetic_origin", "Bohr"),
                ("field_amplitude", "dimensionless"),
                ("raw_field_hermitian_residual", "Hartree, unprojected"),
                ("orbital_residual", "Hartree, canonical overlap metric"),
                ("particle_number_error", "electrons"),
                ("coulomb_operator_defect", "Hartree"),
            ]),
        )
        .with(
            "field_convention",
            if uniform.is_some() {
                "A=lambda M(r-origin); common AO phase exp(i lambda (M origin).r); full \
            continuum orbital field in the periodic gauge; no spin Zeeman or SOC in \
            the collinear route"
            } else {
                "A(r)=lambda[A_q exp(i q.r)+A_q* exp(-i q.r)]; V likewise; includes all \
            A^2 terms; periodic magnetic field B=curl A has zero net flux; no spin \
            Zeeman or SOC; uniform flux uses a distinct magnetic-translation cell"
            },
        );
    if let Some(settings) = options.scf.vv10 {
        append_vv10_metadata(&mut output, settings);
    }
    if let Some(derivatives) = derivatives {
        append_dimensional_geometry(
            &mut output,
            &geometry,
            derivatives,
            dispersion.as_ref(),
            "collinear",
        )?;
    }
    report_dispersion(&mut output, &method, dispersion.as_ref());
    Ok(output)
}

pub(super) fn append_dimensional_stress(
    output: &mut Value,
    geometry: &crate::periodic::PeriodicGeometry,
    s: &crate::periodic::spinor_gradient::DimensionalStress,
    sector: &str,
) {
    let dimension = geometry.dimension();
    let (stress_unit, measure_unit) = match dimension {
        1 => ("hartree/bohr", "bohr"),
        2 => ("hartree/bohr^2", "bohr^2"),
        _ => ("hartree/bohr^3", "bohr^3"),
    };
    output.set("stress", s.stress);
    output.set(
        "stress_voigt",
        [
            s.stress[0][0],
            s.stress[1][1],
            s.stress[2][2],
            s.stress[1][2],
            s.stress[0][2],
            s.stress[0][1],
        ],
    );
    output.set(
        "stress_voigt_convention",
        "xx,yy,zz,yz,xz,xy; upper off-diagonal entries of the unprojected tensor",
    );
    output.set(
        "stress_antisymmetric_residual",
        (0..3)
            .flat_map(|a| (0..3).map(move |b| (s.stress[a][b] - s.stress[b][a]).abs()))
            .fold(0.0, f64::max),
    );

    output.set("pressure", s.pressure);
    output.set("stress_periodic_measure", geometry.measure());
    output.set(
        "stress_pressure_convention",
        "minus the trace in the active periodic subspace divided by its dimension",
    );
    output.set("strain_derivative", s.strain_derivative);
    let field_model = if s.external_field.is_some() {
        "commensurate Fourier field"
    } else {
        "field-free"
    };
    output.set("stress_method", format!("analytic {field_model} same-gauge {sector} affine derivative with fixed reduced k and all nine Cartesian components, divided by periodic length, area or volume"));
    let mut components = Value::object()
        .with("kinetic", s.kinetic)
        .with("electrostatic", s.electrostatic)
        .with("ecp", s.ecp)
        .with("full_exchange", s.full_exchange)
        .with("long_range_exchange", s.long_range_exchange)
        .with("semilocal_xc", s.semilocal_xc)
        .with("pulay", s.pulay);
    let mut diagnostics = Value::object()
        .with("orbital_residual", s.fresh_orbital_residual)
        .with("occupation_consistency", s.fresh_occupation_consistency)
        .with("number_error", s.fresh_number_error)
        .with("reference_energy_defect", s.reference_energy_defect)
        .with("reference_fock_defect", s.reference_fock_defect)
        .with("xc_reference_energy_defect", s.xc_reference_energy_defect)
        .with(
            "coulomb_reference_energy_defect",
            s.coulomb_reference_energy_defect,
        )
        .with("coulomb_quadrature_defect", s.coulomb_quadrature_defect);
    let mut labels = match output.get("units") {
        Some(Value::Object(labels)) => labels.clone(),
        _ => Vec::new(),
    };
    for (key, unit) in [
        ("stress_voigt", stress_unit),
        ("stress_antisymmetric_residual", stress_unit),
        ("stress", stress_unit),
        ("pressure", stress_unit),
        ("stress_periodic_measure", measure_unit),
        ("strain_derivative", "hartree"),
    ] {
        labels.push((key.into(), Value::Str(unit.into())));
    }
    if dimension == 3 {
        let gpa = crate::constants::HARTREE_PER_BOHR3_TO_GPA;
        output.set("stress_gpa", s.stress.map(|r| r.map(|v| v * gpa)));
        output.set("pressure_gpa", s.pressure * gpa);
        labels.push(("stress_gpa".into(), Value::Str("GPa".into())));
        labels.push(("pressure_gpa".into(), Value::Str("GPa".into())));
    }
    for key in [
        "kinetic",
        "electrostatic",
        "ecp",
        "full_exchange",
        "long_range_exchange",
        "semilocal_xc",
        "pulay",
    ] {
        labels.push((
            format!("strain_components.{key}"),
            Value::Str("hartree".into()),
        ));
    }
    for (key, unit) in [
        ("orbital_residual", "hartree"),
        ("occupation_consistency", "dimensionless"),
        ("number_error", "electrons"),
        ("reference_energy_defect", "hartree"),
        ("reference_fock_defect", "hartree"),
        ("xc_reference_energy_defect", "hartree"),
        ("coulomb_reference_energy_defect", "hartree"),
        ("coulomb_quadrature_defect", "hartree"),
    ] {
        labels.push((format!("stress_diagnostics.{key}"), Value::Str(unit.into())));
    }
    if let Some(nonlocal) = s.nonlocal_correlation {
        components.set("nonlocal_correlation", nonlocal);
        labels.push((
            "strain_components.nonlocal_correlation".into(),
            Value::Str("hartree per cell".into()),
        ));
    }
    if let Some(field) = s.external_field {
        output.set(
            "stress_field_convention",
            "fixed reduced wave indices and fixed Cartesian Fourier amplitudes",
        );
        components.set("external_field", field);
        diagnostics.set(
            "external_field_energy_defect",
            s.external_field_energy_defect,
        );
        for key in [
            "strain_components.external_field",
            "stress_diagnostics.external_field_energy_defect",
        ] {
            labels.push((key.into(), Value::Str("hartree".into())));
        }
    }
    output.set("strain_components", components);
    output.set("stress_diagnostics", diagnostics);
    output.set("units", Value::Object(labels));
}

pub(super) fn append_dimensional_geometry(
    report: &mut Value,
    geometry: &crate::periodic::PeriodicGeometry,
    derivatives: crate::periodic::scf_boundary::GeometryDerivatives,
    dispersion: Option<&PeriodicDispersionResult>,
    sector: &str,
) -> Result<()> {
    if let Some(mut stress) = derivatives.stress {
        if let Some(correction) = dispersion {
            stress.add_affine_derivative(geometry, correction.strain_derivative)?;
        }
        append_dimensional_stress(report, geometry, &stress, sector);
        if let Some(correction) = dispersion {
            if let Some(mut components) = report.get("strain_components").cloned() {
                components.set("dispersion", correction.strain_derivative);
                report.set("strain_components", components);
            }
            if let Some(mut labels) = report.get("units").cloned() {
                labels.set("strain_components.dispersion", "hartree per cell");
                report.set("units", labels);
            }
        }
    }
    if let Some(mut gradient) = derivatives.gradient {
        if let Some(correction) = dispersion {
            correction.add_to_gradient(&mut gradient.total);
        }
        append_dimensional_gradient(report, &gradient, sector);
        if let Some(correction) = dispersion {
            if let Some(mut components) = report.get("gradient_components").cloned() {
                let gradient: Vec<_> = correction.forces.iter().map(|f| (-*f).to_array()).collect();
                components.set("dispersion", gradient);
                report.set("gradient_components", components);
            }
            if let Some(mut labels) = report.get("units").cloned() {
                labels.set("gradient_components.dispersion", "hartree/bohr");
                report.set("units", labels);
            }
        }
    }
    Ok(())
}

pub(super) fn append_dimensional_gradient(
    output: &mut Value,
    g: &crate::periodic::spinor_gradient::DimensionalNuclearGradient,
    sector: &str,
) {
    let vectors = |v: &[Vec3]| v.iter().map(|x| x.to_array()).collect::<Vec<_>>();
    output.set("gradient", vectors(&g.total));
    output.set(
        "forces",
        g.total
            .iter()
            .map(|v| (-1. * *v).to_array())
            .collect::<Vec<_>>(),
    );
    output.set("gradient_method", format!("analytic same-frame {sector} kinetic, dimensional electrostatics, scalar ECP, complex exchange, moving XC AO/points/weights and Pulay"));
    let mut components = Value::object()
        .with("kinetic", vectors(&g.kinetic))
        .with("electrostatic", vectors(&g.electrostatic))
        .with("ecp", vectors(&g.ecp))
        .with("full_exchange", vectors(&g.full_exchange))
        .with("long_range_exchange", vectors(&g.long_range_exchange))
        .with("semilocal_xc", vectors(&g.semilocal_xc))
        .with("pulay", vectors(&g.pulay));
    output.set("environment_gradient", vectors(&g.environment_gradient));
    let mut diagnostics = Value::object()
        .with("orbital_residual", g.fresh_orbital_residual)
        .with("occupation_consistency", g.fresh_occupation_consistency)
        .with("number_error", g.fresh_number_error)
        .with("reference_energy_defect", g.reference_energy_defect)
        .with("reference_fock_defect", g.reference_fock_defect)
        .with("xc_reference_energy_defect", g.xc_reference_energy_defect);
    let mut labels = match output.get("units") {
        Some(Value::Object(labels)) => labels.clone(),
        _ => Vec::new(),
    };
    for key in [
        "gradient",
        "forces",
        "gradient_components.kinetic",
        "gradient_components.electrostatic",
        "gradient_components.ecp",
        "gradient_components.full_exchange",
        "gradient_components.long_range_exchange",
        "gradient_components.semilocal_xc",
        "gradient_components.pulay",
        "environment_gradient",
    ] {
        labels.push((key.into(), Value::Str("hartree/bohr".into())));
    }
    for (key, unit) in [
        ("orbital_residual", "hartree"),
        ("occupation_consistency", "dimensionless"),
        ("number_error", "electrons"),
        ("reference_energy_defect", "hartree"),
        ("reference_fock_defect", "hartree"),
        ("xc_reference_energy_defect", "hartree"),
    ] {
        labels.push((
            format!("gradient_diagnostics.{key}"),
            Value::Str(unit.into()),
        ));
    }
    if let Some(nonlocal) = &g.nonlocal_correlation {
        components.set("nonlocal_correlation", vectors(nonlocal));
        labels.push((
            "gradient_components.nonlocal_correlation".into(),
            Value::Str("hartree/bohr".into()),
        ));
    }
    if let Some(field) = &g.external_field {
        components.set("external_field", vectors(field));
        diagnostics.set(
            "external_field_energy_defect",
            g.external_field_energy_defect,
        );
        labels.push((
            "gradient_components.external_field".into(),
            Value::Str("hartree/bohr".into()),
        ));
        labels.push((
            "gradient_diagnostics.external_field_energy_defect".into(),
            Value::Str("hartree".into()),
        ));
    }
    output.set("gradient_components", components);
    output.set("gradient_diagnostics", diagnostics);
    output.set("units", Value::Object(labels));
}

fn reduced(lattice: &Lattice, k: Vec3) -> [f64; 3] {
    // k = sum_i f_i b_i with a_i . b_j = 2 pi delta_ij, so f_i = a_i . k / (2 pi).
    let two_pi = 2.0 * std::f64::consts::PI;
    [
        lattice.a[0].dot(k) / two_pi,
        lattice.a[1].dot(k) / two_pi,
        lattice.a[2].dot(k) / two_pi,
    ]
}

fn number_integration_diagnostics(
    nominal: f64,
    metric: f64,
    grid: Option<f64>,
) -> (Value, Option<String>) {
    // Catch gross inconsistencies without treating this advisory threshold
    // as a material-convergence gate or changing the SCF definition.
    let advisory_relative_error = 0.01;
    let relative = grid.map(|value| (value - nominal).abs() / nominal.abs().max(1.0));
    let large = relative.is_some_and(|error| error > advisory_relative_error);
    let warning = if large {
        Some(format!(
            "WARNING [periodic integration] the XC-grid electron integral {:.12} differs from the occupied electron number {nominal:.12} by more than 1%; converge the basis-image, AO, quadrature and interaction supports before interpreting properties. This advisory does not change SCF convergence or certify accuracy when absent",
            grid.expect("a grid discrepancy requires an integral"),
        ))
    } else {
        None
    };
    let report = Value::object()
        .with("occupied_electrons", nominal)
        .with("overlap_metric_electrons", metric)
        .with("overlap_metric_number_error", metric - nominal)
        .with("grid_electrons", grid)
        .with("grid_number_error", grid.map(|value| value - nominal))
        .with("relative_grid_number_error", relative)
        .with("advisory_relative_error", advisory_relative_error)
        .with("large_grid_discrepancy", relative.map(|_| large))
        .with("scope", "particle counts; separate from basis, image, grid and k convergence; no density rescaling or adaptive cutoff change");
    (report, warning)
}

pub(super) fn reference_report(
    setup: &PeriodicSetup,
    scf: &PeriodicScfResult,
    model: &str,
) -> Value {
    let mut warnings: Vec<String> = Vec::new();
    if !scf.converged {
        warnings.push("WARNING [periodic] the reference SCF did not converge".into());
    }
    let metric_number: f64 = scf
        .density
        .iter()
        .flat_map(|spin| spin.iter().zip(&scf.overlap))
        .map(|(density, overlap)| density.trace_product(overlap) / scf.k_points.len() as f64)
        .sum();
    let (number_diagnostics, number_warning) = number_integration_diagnostics(
        (scf.n_alpha + scf.n_beta) as f64,
        metric_number,
        scf.grid_electron_count,
    );
    if let Some(warning) = number_warning {
        warnings.push(warning);
    }
    let lattice = setup.cell.lattice();
    let symmetry = match scf.time_reversal_diagnostics(setup.options.k_mesh) {
        Ok(r) => Value::object()
            .with(
                "orbital_density_residual",
                r.orbital_density_residual.to_vec(),
            )
            .with("orbital_fock_residual", r.orbital_fock_residual.to_vec())
            .with("spinful_density_residual", r.spinful_density_residual)
            .with("spinful_fock_residual", r.spinful_fock_residual)
            .with(
                "spin_population_difference",
                scf.n_alpha as f64 - scf.n_beta as f64,
            )
            .with(
                "reference_space",
                "zero-field collinear; each spin has orbital conjugation symmetry; \
                physical spin-1/2 time reversal exchanges alpha and beta",
            )
            .with("orbital_current_order_allowed", false)
            .with("spin_orbit_coupling", false)
            .with("external_orbital_field", false)
            .with(
                "residual_convention",
                "raw maximum AO entry defects: P_s(k)-P_s(-k)* within each spin; \
                P_s(k)-P_opposite(-k)* for spinful time reversal; likewise F; density \
                dimensionless and Fock Hartree; no symmetry projection in this \
                diagnostic",
            ),
        Err(error) => Value::object().with("diagnostic_error", error.to_string()),
    };
    let mut report = Value::object()
        .with("model", model)
        .with("reference_symmetry", symmetry)
        .with("functional", setup.functional.description.as_str())
        .with("basis", setup.method.basis.as_str())
        .with("particle_number_diagnostics", number_diagnostics)
        .with("converged", scf.converged)
        .with("scf_iterations", scf.iterations)
        .with(
            "reference_type",
            if scf.unrestricted { "uks" } else { "rks" },
        )
        .with("unrestricted", scf.unrestricted)
        .with("n_alpha", scf.n_alpha)
        .with("n_beta", scf.n_beta)
        .with("k_mesh", setup.method.k_mesh.to_vec())
        .with(
            "k_points",
            scf.k_points
                .iter()
                .map(|k| k.to_array())
                .collect::<Vec<_>>(),
        )
        .with(
            "k_points_reduced",
            scf.k_points
                .iter()
                .map(|k| reduced(lattice, *k))
                .collect::<Vec<_>>(),
        )
        .with("volume", lattice.volume().abs())
        .with("energy", scf.energy)
        .with(
            "energy_components",
            Value::object()
                .with("electronic", scf.e_elec)
                .with("nuclear_repulsion", scf.e_nuc)
                .with("exchange_correlation", scf.e_xc)
                .with("semilocal_exchange_correlation", scf.e_xc - scf.e_nlc)
                .with("nonlocal_correlation", scf.e_nlc)
                .with("entropy_term", scf.entropy_term),
        )
        .with(
            "electronic_temperature_kelvin",
            setup.method.electronic_temperature,
        )
        .with("warnings", warnings);
    if let Some(settings) = setup.options.vv10 {
        append_vv10_metadata(&mut report, settings);
    }
    report
}

pub(super) fn append_vv10_metadata(
    report: &mut Value,
    settings: crate::periodic::nonlocal::PeriodicVv10Options,
) {
    report.set(
        "nonlocal_correlation_model",
        "periodic VV10 direct pair sum with C-infinity distance/vacuum switches; cutoff must be converged",
    );
    report.set("vv10_b", settings.parameters.b);
    report.set("vv10_c", settings.parameters.c);
    report.set("vv10_cutoff", settings.cutoff);
}

fn append_spinor_vv10(
    report: &mut Value,
    term: Option<(crate::periodic::nonlocal::PeriodicVv10Options, f64)>,
) {
    if let Some((settings, energy)) = term {
        append_vv10_metadata(report, settings);
        report.set("nonlocal_correlation_energy", energy);
        if let Some(mut labels) = report.get("units").cloned() {
            labels.set("nonlocal_correlation_energy", "hartree per cell");
            report.set("units", labels);
        }
    }
}

/// Add the geometry-only correction to reported energies and record the model.
pub(super) fn report_dispersion(
    report: &mut Value,
    method: &PeriodicMethod,
    dispersion: Option<&PeriodicDispersionResult>,
) {
    if let Some(dispersion) = dispersion {
        include_dispersion(report, method, dispersion.energy);
        if let Some(mut labels) = report.get("units").cloned() {
            for key in ["disp_cutoff", "cn_cutoff", "abc_cutoff"] {
                labels.set(key, "bohr");
            }
            report.set("units", labels);
        }
    }
}

pub(super) fn include_dispersion(report: &mut Value, method: &PeriodicMethod, energy: f64) {
    for key in ["energy", "free_energy"] {
        if let Some(value) = report.get(key).and_then(Value::as_f64) {
            report.set(key, value + energy);
        }
    }
    if let Some(mut components) = report.get("energy_components").cloned() {
        components.set("dispersion", energy);
        report.set("energy_components", components);
    }
    report.set("dispersion_energy", energy);
    // Every caller reports per-cell energies unless it labels the entry
    // itself afterwards (the k-mesh supercell routes); the strict unit check
    // refused `phonons` with a dispersion model for the missing label.
    if let Some(mut labels) = report.get("units").cloned() {
        if labels.get("dispersion_energy").is_none() {
            labels.set("dispersion_energy", "hartree per cell");
            report.set("units", labels);
        }
    }
    report.set("dispersion_model", method.dispersion_name());
    report.set("disp_cutoff", method.disp_cutoff);
    report.set("cn_cutoff", method.cn_cutoff);
    report.set("three_body", method.three_body);
    if method.three_body {
        report.set("abc_cutoff", method.abc_cutoff);
    }
}

/// Hessian of the geometry-only dispersion energy per cell.
fn dispersion_hessian(
    cell: &Cell,
    model: &crate::periodic::dispersion::PeriodicDispersion,
) -> Result<crate::linalg::Matrix> {
    let numbers: Vec<_> = cell.molecule().atoms.iter().map(|a| a.z).collect();
    let positions: Vec<_> = cell.molecule().atoms.iter().map(|a| a.position).collect();
    crate::periodic::dispersion::periodic_dispersion_hessian(
        &numbers,
        &positions,
        cell.lattice(),
        model,
    )
}

/// Add dispersion to a conventional Gamma result before taking normal
/// modes so raw, symmetrized and skeleton matrices stay consistent.
fn add_gamma_dispersion(
    cell: &Cell,
    dispersion: &Option<crate::periodic::dispersion::PeriodicDispersion>,
    force_constants: &mut crate::periodic::phonon::GammaForceConstants,
) -> Result<Option<f64>> {
    let Some(model) = dispersion else {
        return Ok(None);
    };
    force_constants.add_geometry_hessian(&dispersion_hessian(cell, model)?);
    Ok(cell_dispersion(cell, dispersion)?.map(|result| result.energy))
}

/// Add dispersion to a lattice Taylor Hessian (which has no separate
/// raw and skeleton views) and return its energy.
fn add_dispersion_force_constants(
    cell: &Cell,
    dispersion: &Option<PeriodicDispersion>,
    hessian: &mut crate::linalg::Matrix,
) -> Result<f64> {
    let Some(model) = dispersion else {
        return Ok(0.0);
    };
    let correction = dispersion_hessian(cell, model)?;
    for (total, extra) in hessian.as_mut_slice().iter_mut().zip(correction.as_slice()) {
        *total += extra;
    }
    Ok(cell_dispersion(cell, dispersion)?.map_or(0.0, |result| result.energy))
}

pub(super) fn run_reference(setup: &PeriodicSetup) -> Result<PeriodicScfResult> {
    run_periodic_scf(&setup.cell, &setup.functional, &setup.options)
}

/// Stress fields of a report: the symmetric stress, its Voigt form, the
/// pressure, the strain derivative and its terms.
pub(super) fn stress_fields(report: &mut Value, stress: &crate::periodic::stress::PeriodicStress) {
    let s = stress.stress;
    let pressure = -(s[0][0] + s[1][1] + s[2][2]) / 3.0;
    let gpa = crate::constants::HARTREE_PER_BOHR3_TO_GPA;
    report.set("stress", s);
    report.set(
        "stress_voigt",
        [s[0][0], s[1][1], s[2][2], s[1][2], s[0][2], s[0][1]],
    );
    report.set("stress_gpa", s.map(|row| row.map(|v| v * gpa)));
    report.set("pressure", pressure);
    report.set("pressure_gpa", pressure * gpa);
    report.set("strain_derivative", stress.strain_derivative);
    report.set("rotation_residual", stress.rotation_residual);
    let mut terms = Value::object();
    for (name, tensor) in &stress.terms {
        terms.set(name, *tensor);
    }
    report.set("strain_derivative_terms", terms);
}

pub(super) const STRESS_UNITS: [(&str, &str); 7] = [
    (
        "stress",
        "hartree/bohr^3 (sigma = (1/V) dE/d(epsilon); Voigt xx, yy, zz, yz, xz, xy)",
    ),
    ("stress_gpa", "GPa"),
    ("stress_voigt", "hartree/bohr^3 (xx, yy, zz, yz, xz, xy)"),
    ("pressure", "hartree/bohr^3 (-tr(sigma)/3)"),
    ("pressure_gpa", "GPa"),
    ("strain_derivative", "hartree per cell"),
    ("rotation_residual", "hartree per cell"),
];

fn periodic_energy(system: &System, controls: &Controls, stress_feature: bool) -> Result<Value> {
    let mut method = PeriodicMethod::from_controls_with_dispersion(controls)?;
    method.apply_collinear_reference(controls)?;
    let dispersion_model = method.dispersion_model_for(&system.numbers)?;
    let with_forces = controls.bool("forces", stress_feature)?;
    let with_stress = controls.bool("stress", false)? || stress_feature;
    controls.finish(if stress_feature {
        "stress"
    } else {
        "periodic-energy"
    })?;
    let setup = prepare_periodic(system, &method)?;
    let scf = run_reference(&setup)?;
    let dispersion = cell_dispersion(&setup.cell, &dispersion_model)?;
    let mut report = reference_report(
        &setup,
        &scf,
        "finite-k periodic Kohn-Sham/Hartree-Fock energy per cell (Mermin free \
            energy at finite temperature)",
    )
    .with("free_energy", scf.free_energy);
    report.set("orbital_residual", scf.orbital_residual);
    let mut unit_pairs = vec![
        ("energy", "hartree per cell"),
        ("k_points", "1/bohr"),
        ("orbital_residual", "hartree"),
        ("volume", "bohr^3"),
    ];
    if let Some(d) = &dispersion {
        include_dispersion(&mut report, &method, d.energy);
        unit_pairs.push(("dispersion_energy", "hartree per cell"));
    }
    if with_forces {
        let forces = crate::periodic::forces::analytic_forces(
            &setup.cell,
            &setup.functional,
            &scf,
            &setup.options,
        )?;
        let mut total = forces.total.clone();
        if let Some(d) = &dispersion {
            for (t, f) in total.iter_mut().zip(&d.forces) {
                *t += *f;
            }
        }
        let mut sum = Vec3::zero();
        for f in &total {
            sum += *f;
        }
        report.set(
            "forces",
            total.iter().map(|f| f.to_array()).collect::<Vec<_>>(),
        );
        report.set(
            "gradient",
            total
                .iter()
                .map(|f| (*f * -1.0).to_array())
                .collect::<Vec<_>>(),
        );
        report.set("translation_residual", sum.norm());
        report.set("provenance", "analytic");
        unit_pairs.push(("forces", "hartree/bohr"));
        unit_pairs.push(("gradient", "hartree/bohr"));
    }
    if with_stress {
        let mut stress = crate::periodic::stress::analytic_stress(
            &setup.cell,
            &setup.functional,
            &scf,
            &setup.options,
        )?;
        if let Some(d) = &dispersion {
            crate::periodic::dispersion::add_to_stress(&mut stress, &d.strain_derivative);
        }
        stress_fields(&mut report, &stress);
        report.set("stress_provenance", "analytic");
        unit_pairs.extend(STRESS_UNITS);
    }
    report.set("units", units(&unit_pairs));
    Ok(report)
}

fn trajectory_frame(step: usize, objective: f64, grad_max: f64) -> Value {
    Value::object()
        .with("step", step)
        .with("objective", objective)
        .with("grad_max", grad_max)
}

/// Gamma-point harmonic phonons from the analytic coupled-perturbed force
/// constants: frequencies of the mass-weighted Hessian and the Cartesian
/// displacement patterns of its modes.
fn phonons(system: &System, controls: &Controls) -> Result<Value> {
    // The dispersion models of the lattice sum are accepted: their force
    // constants are added below.
    let mut method = PeriodicMethod::from_controls_with_dispersion(controls)?;
    method.apply_collinear_reference(controls)?;
    let config = super::vibrational::response_config(controls)?;
    let field = controls.opt_vec3("field")?;
    controls.finish("phonons")?;
    if field.is_some() {
        return lattice_taylor_phonons(system, &method, field, &config);
    }
    let setup = prepare_periodic(system, &method)?;
    // A variational VV10 term has moving-grid and induced-density
    // contributions in the stationary lattice Taylor implementation.
    if setup.options.vv10.is_some() {
        return lattice_taylor_phonons(system, &method, None, &config);
    }
    let scf = run_reference(&setup)?;
    let mut fc = crate::periodic::phonon::gamma_force_constants(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
        &config,
    )?;
    let dispersion_energy = add_gamma_dispersion(
        &setup.cell,
        &method.dispersion_model_for(&system.numbers)?,
        &mut fc,
    )?;
    let molecule = setup.cell.molecule();
    let natoms = molecule.len();
    let dim = 3 * natoms;
    let numbers: Vec<_> = molecule.atoms.iter().map(|a| a.z).collect();
    let (frequencies, modes) = crate::periodic::phonon::gamma_modes(&fc.hessian, &numbers)?;
    let rows: Vec<Vec<f64>> = (0..dim)
        .map(|x| (0..dim).map(|y| fc.hessian[(x, y)]).collect())
        .collect();
    let raw_rows: Vec<Vec<f64>> = (0..dim)
        .map(|x| (0..dim).map(|y| fc.raw[(x, y)]).collect())
        .collect();
    let mut report = reference_report(
        &setup,
        &scf,
        "Gamma-point harmonic phonons from analytic coupled-perturbed force \
            constants (all lattice images of an atom move together)",
    )
    .with("orbital_residual", scf.orbital_residual)
    .with("q", vec![0.0, 0.0, 0.0])
    .with("force_constants", rows)
    .with("raw_force_constants", raw_rows)
    .with("frequencies_cm1", frequencies)
    .with("normal_modes_cartesian", modes)
    .with("raw_asymmetry", fc.raw_asymmetry)
    .with("acoustic_residual", fc.acoustic_residual)
    .with(
        "response_solves",
        fc.solves
            .iter()
            .map(|(iterations, residual)| {
                Value::object()
                    .with("iterations", *iterations)
                    .with("relative_residual", *residual)
            })
            .collect::<Vec<_>>(),
    )
    .with("provenance", "analytic")
    .with(
        "units",
        units(&[
            ("energy", "hartree per cell"),
            ("orbital_residual", "hartree"),
            ("force_constants", "hartree/bohr^2"),
            (
                "raw_force_constants",
                "hartree/bohr^2, before symmetrization: column y is the derivative of the analytic gradient with respect to displacement y",
            ),
            ("frequencies_cm1", "cm^-1 (negative = imaginary)"),
            (
                "normal_modes_cartesian",
                "atomic units, electron-mass-weighted displacement pattern",
            ),
            ("raw_asymmetry", "hartree/bohr^2"),
            ("acoustic_residual", "hartree/bohr^2"),
        ]),
    );
    if let Some(energy) = dispersion_energy {
        include_dispersion(&mut report, &method, energy);
    }
    Ok(report)
}

/// Harmonic nuclear response of the lattice Taylor stationary energy,
/// with a Berry enthalpy term when a Cartesian electric bias is supplied.
fn lattice_taylor_phonons(
    system: &System,
    method: &PeriodicMethod,
    field: Option<Vec3>,
    config: &crate::response::ResponseSolverConfig,
) -> Result<Value> {
    let dispersion_model = method.dispersion_model_for(&system.numbers)?;
    let (setup, scf, replicas) = gamma_supercell("phonons", system, method)?;
    let result = crate::response::nuclear::lattice_electronic_force_constant_tensor(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
        replicas,
        2,
        field.map(Vec3::to_array),
        config,
        dispersion_model.as_ref(),
    )?;
    let n = result.dimension;
    let mut hessian = crate::linalg::Matrix::zeros(n, n);
    hessian.as_mut_slice().copy_from_slice(&result.tensor);
    let dispersion_energy = if dispersion_model.is_some() {
        // This Hessian is folded to primitive Gamma displacements. Dispersion
        // must use that primitive cell, not a truncated supercell Hessian.
        let primitive = prepare_periodic(system, method)?;
        add_dispersion_force_constants(&primitive.cell, &dispersion_model, &mut hessian)?
    } else {
        0.0
    };
    let numbers: Vec<_> = system.numbers.clone();
    let (frequencies, modes) = crate::periodic::phonon::gamma_modes(&hessian, &numbers)?;
    let acoustic = (0..n)
        .flat_map(|x| (0..3).map(move |axis| (x, axis)))
        .map(|(x, axis)| {
            (0..numbers.len())
                .map(|atom| hessian[(x, 3 * atom + axis)])
                .sum::<f64>()
                .abs()
        })
        .fold(0.0_f64, f64::max);
    let mut report = reference_report(&setup, &scf,
        if field.is_some() {
            "Gamma nuclear Hessian of the self-consistent finite-field Berry electric enthalpy; fixed Cartesian electric field, moving basis and metric; intrinsic permutation symmetry"
        } else {
            "Gamma nuclear Hessian of the stationary zero-field lattice Taylor energy with variational nonlocal correlation; moving basis, grid and metric"
        })
        .with("q", vec![0.0;3])
        .with("supercell", method.k_mesh.to_vec())
        .with("force_constants", nested(hessian.as_slice(), n, 1.0))
        .with("frequencies_cm1", frequencies)
        .with("normal_modes_cartesian", modes)
        .with("acoustic_residual", acoustic)
        .with("response_residuals", result.response_residuals.clone())
        .with("path_evaluations", result.paths)
        .with("force_constant_symmetry", "intrinsic permutation; independent ordered asymmetry not evaluated")
        .with("provenance", "analytic");
    let mut labels = vec![
        (
            "energy",
            "hartree per supercell of the k mesh, zero-field reference",
        ),
        ("force_constants", "hartree/bohr^2 per primitive cell"),
        ("frequencies_cm1", "cm^-1 (negative = imaginary)"),
        (
            "normal_modes_cartesian",
            "atomic units, electron-mass-weighted displacement pattern",
        ),
        ("acoustic_residual", "hartree/bohr^2"),
        ("response_residuals", "dimensionless"),
    ];
    labels.extend(SERIES_REFERENCE_UNITS);
    report.set("units", units(&labels));
    let mut reference = result.reference;
    reference.energy += dispersion_energy;
    report = series_reference_fields(report, &reference);
    if let Some(field) = field {
        report.set("reference_field", field.to_array());
        report.set("field_energy", reference.energy);
        let mut units = report.get("units").cloned().expect("biased phonon units");
        units.set("reference_field", "hartree/e/bohr");
        units.set("field_energy", "hartree per primitive cell");
        report.set("units", units);
    }
    if dispersion_model.is_some() {
        include_dispersion(&mut report, method, dispersion_energy * replicas as f64);
        let mut labels = report.get("units").cloned().expect("lattice phonon units");
        labels.set("dispersion_energy", "hartree per supercell of the k mesh");
        labels.set(
            "dispersion_energy_per_primitive_cell",
            "hartree per primitive cell",
        );
        for key in ["disp_cutoff", "cn_cutoff", "abc_cutoff"] {
            labels.set(key, "bohr");
        }
        report.set("dispersion_energy_per_primitive_cell", dispersion_energy);
        report.set("units", labels);
    }
    Ok(report)
}

/// Analytic Gamma-point force constants of an `N1 x N2 x N3` supercell with
/// the k mesh divided accordingly (the same Brillouin-zone sampling), folded
/// onto the primitive cell.
struct SupercellPhonons {
    setup: PeriodicSetup,
    scf: PeriodicScfResult,
    model: crate::periodic::phonon_bands::ForceConstantModel,
    force_constants: crate::periodic::phonon::GammaForceConstants,
    supercell: [usize; 3],
    /// Dispersion energy of the supercell (Hartree); zero without a model.
    dispersion_energy: f64,
}

/// The supercell of `system` with the k mesh divided accordingly (the same
/// Brillouin-zone sampling).
pub(super) fn supercell_system(
    system: &System,
    method: &PeriodicMethod,
    supercell: [usize; 3],
) -> Result<(System, PeriodicMethod)> {
    use crate::periodic::phonon_bands::build_supercell;
    if supercell.contains(&0) {
        return Err(DftError::InvalidInput(
            "`supercell` needs three positive repetitions".into(),
        ));
    }
    let cells = supercell
        .iter()
        .try_fold(1usize, |n, k| n.checked_mul(*k))
        .ok_or_else(|| DftError::InvalidInput("supercell replica count overflow".into()))?;
    let mut super_method = method.clone();
    let replicas = i32::try_from(cells)
        .map_err(|_| DftError::InvalidInput("supercell charge multiplier out of range".into()))?;
    super_method.charge = method
        .charge
        .checked_mul(replicas)
        .ok_or_else(|| DftError::InvalidInput("supercell charge out of range".into()))?;
    super_method.multiplicity = method
        .multiplicity
        .checked_sub(1)
        .and_then(|spin| spin.checked_mul(cells))
        .and_then(|spin| spin.checked_add(1))
        .ok_or_else(|| DftError::InvalidInput("supercell spin multiplicity out of range".into()))?;
    for i in 0..3 {
        if method.k_mesh[i] % supercell[i] != 0 {
            return Err(DftError::InvalidInput(format!(
                "k_mesh {:?} must be divisible by supercell {supercell:?} so that the \
                    supercell samples the same Brillouin zone",
                method.k_mesh
            )));
        }
        super_method.k_mesh[i] = method.k_mesh[i] / supercell[i];
    }
    let lattice = lattice_of(system)?;
    let positions: Vec<Vec3> = system
        .positions
        .iter()
        .map(|p| Vec3::new(p[0], p[1], p[2]))
        .collect();
    let (big, super_positions) = build_supercell(&lattice, &positions, supercell);
    let numbers: Vec<u8> = (0..cells)
        .flat_map(|_| system.numbers.iter().copied())
        .collect();
    let super_system = System {
        numbers,
        positions: super_positions.iter().map(|p| p.to_array()).collect(),
        cell: Some(big.a.map(|v| v.to_array())),
        pbc: system.pbc,
    };
    Ok((super_system, super_method))
}

fn supercell_phonons(
    system: &System,
    method: &PeriodicMethod,
    supercell: [usize; 3],
    config: &crate::response::krylov::ResponseSolverConfig,
) -> Result<SupercellPhonons> {
    let (super_system, super_method) = supercell_system(system, method, supercell)?;
    let setup = prepare_periodic(&super_system, &super_method)?;
    let scf = run_reference(&setup)?;
    let mut force_constants = crate::periodic::phonon::gamma_force_constants(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
        config,
    )?;
    let dispersion_energy = add_dispersion_force_constants(
        &setup.cell,
        &setup.method.dispersion_model_for(&system.numbers)?,
        &mut force_constants.hessian,
    )?;
    let model = fold_supercell_phonons(system, supercell, &force_constants.hessian)?;
    Ok(SupercellPhonons {
        setup,
        scf,
        model,
        force_constants,
        supercell,
        dispersion_energy,
    })
}

fn fold_supercell_phonons(
    system: &System,
    supercell: [usize; 3],
    hessian: &Matrix,
) -> Result<crate::periodic::phonon_bands::ForceConstantModel> {
    let positions: Vec<_> = system
        .positions
        .iter()
        .map(|p| Vec3::new(p[0], p[1], p[2]))
        .collect();
    let masses: Vec<_> = system
        .numbers
        .iter()
        .map(|z| crate::constants::atomic_mass_au(*z))
        .collect();
    crate::periodic::phonon_bands::ForceConstantModel::from_supercell_hessian(
        &lattice_of(system)?,
        &positions,
        &masses,
        supercell,
        hessian,
    )
}

/// Both spectra use the same mass weighting and force-constant interpolation.
/// The correlated provider changes the Hessian, including the relaxed parent
/// orbitals and PT2 amplitudes, before the spectrum is formed.
fn phonon_spectrum(
    feature: &str,
    system: &System,
    controls: &Controls,
    method: &PeriodicMethod,
    supercell: [usize; 3],
    config: &crate::response::ResponseSolverConfig,
    description: &str,
) -> Result<(crate::periodic::phonon_bands::ForceConstantModel, Value)> {
    if super::double_hybrid::requested(controls)? {
        let (mut report, hessian) = super::correlated_periodic::supercell_force_constants(
            feature, system, controls, supercell,
        )?;
        let model = fold_supercell_phonons(system, supercell, &hessian)?;
        report.set("model", format!("{description}. The force constants include the full periodic double-hybrid orbital and PT2 amplitude response"));
        report.set("acoustic_residual", model.acoustic_residual);
        report.set("force_constant_asymmetry", Value::Null);
        report.set(
            "force_constant_symmetry",
            "intrinsic permutation symmetry of analytic energy derivatives",
        );
        report.set(
            "supercell_force_constants",
            nested(hessian.as_slice(), hessian.rows, 1.0),
        );
        report.set(
            "provenance",
            "analytic double-hybrid supercell force constants",
        );
        Ok((model, report))
    } else {
        controls.finish(feature)?;
        let sp = supercell_phonons(system, method, supercell, config)?;
        let report = phonon_common_report(&sp, description);
        Ok((sp.model, report))
    }
}

fn phonon_spectrum_units(report: &Value, pairs: &[(&str, &str)]) -> Value {
    let mut labels = report.get("units").cloned().unwrap_or_else(Value::object);
    for (key, label) in pairs {
        labels.set(key, *label);
    }
    labels
}

fn supercell_control(controls: &Controls) -> Result<[usize; 3]> {
    let list = controls.opt_usize_list("supercell")?.ok_or_else(|| {
        DftError::InvalidInput(
            "phonon dispersion needs supercell: three repetitions, e.g. [2,2,2]".into(),
        )
    })?;
    if list.len() != 3 {
        return Err(DftError::InvalidInput(
            "`supercell` must have three entries".into(),
        ));
    }
    Ok([list[0], list[1], list[2]])
}

fn mesh_points(lattice: &Lattice, mesh: [usize; 3]) -> Vec<Vec3> {
    // Gamma-centred uniform mesh in reduced coordinates.
    let mut out = Vec::with_capacity(mesh.iter().product());
    for i in 0..mesh[0] {
        for j in 0..mesh[1] {
            for k in 0..mesh[2] {
                let f = [
                    i as f64 / mesh[0] as f64,
                    j as f64 / mesh[1] as f64,
                    k as f64 / mesh[2] as f64,
                ];
                out.push(cartesian(lattice, f));
            }
        }
    }
    out
}

fn phonon_common_report(sp: &SupercellPhonons, model: &str) -> Value {
    reference_report(&sp.setup, &sp.scf, model)
        .with("supercell", sp.supercell.to_vec())
        .with("supercell_k_mesh", sp.setup.method.k_mesh.to_vec())
        .with("acoustic_residual", sp.model.acoustic_residual)
        .with("force_constant_asymmetry", sp.force_constants.raw_asymmetry)
        .with(
            "supercell_force_constants",
            (0..sp.force_constants.hessian.rows)
                .map(|x| {
                    (0..sp.force_constants.hessian.cols)
                        .map(|y| sp.force_constants.hessian[(x, y)])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        )
        .with(
            "response_solves",
            sp.force_constants
                .solves
                .iter()
                .map(|(iterations, residual)| {
                    Value::object()
                        .with("iterations", *iterations)
                        .with("relative_residual", *residual)
                })
                .collect::<Vec<_>>(),
        )
        .with("provenance", "analytic supercell force constants")
}

/// Electron-phonon coupling, the Eliashberg function and McMillan/Allen-Dynes
/// critical temperatures from the Gamma displacement responses of a
/// supercell (all commensurate wave vectors).
fn electron_phonon_feature(system: &System, controls: &Controls) -> Result<Value> {
    use crate::periodic::electron_phonon::screening::{
        screened_pair_coulomb, CoulombScreeningOptions, GammaHead,
    };

    let method = PeriodicMethod::from_controls(controls)?;
    let config = super::vibrational::response_config(controls)?;
    let supercell = match controls.opt_usize_list("supercell")? {
        None => [1, 1, 1],
        Some(l) if l.len() == 3 => [l[0], l[1], l[2]],
        Some(_) => {
            return Err(DftError::InvalidInput(
                "`supercell` must have three entries".into(),
            ))
        }
    };
    let kt = method.electronic_temperature * crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
    let electron_smearing = match controls.opt_f64("electron_smearing")? {
        Some(v) => v,
        None if kt > 0.0 => kt,
        None => {
            return Err(DftError::InvalidInput(
                "`electron_smearing` (Hartree) is required without an electronic temperature"
                    .into(),
            ))
        }
    };
    let cm = crate::constants::HARTREE_TO_CM1;
    let phonon_smearing = controls.f64("phonon_smearing_cm1", 10.0)? / cm;
    let mode_cutoff = controls.f64("mode_cutoff_cm1", 5.0)? / cm;
    let mu_star = controls.f64("mu_star", 0.1)?;
    let points = controls.usize("a2f_points", 400)?;
    let eliashberg_cutoff = controls.opt_f64("eliashberg_cutoff_cm1")?.map(|v| v / cm);
    let eliashberg_t_min = controls.f64("eliashberg_t_min_k", 0.01)?
        * crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
    if let Some(c) = eliashberg_cutoff {
        if !(c > 0.0) {
            return Err(DftError::InvalidInput(
                "`eliashberg_cutoff_cm1` must be positive".into(),
            ));
        }
    }
    if !(eliashberg_t_min > 0.0) {
        return Err(DftError::InvalidInput(
            "`eliashberg_t_min_k` must be positive".into(),
        ));
    }
    let gap_temperatures = controls.opt_f64_list("gap_temperatures_k")?;
    if gap_temperatures
        .as_ref()
        .is_some_and(|l| l.iter().any(|t| !(*t > 0.0)))
    {
        return Err(DftError::InvalidInput(
            "`gap_temperatures_k` must be positive".into(),
        ));
    }
    let pade_points = controls.usize("pade_points", 64)?;
    let scdft_pairing = controls.bool("scdft_pairing", false)?;
    let screening_model = if scdft_pairing {
        controls.str("scdft_screening", "supplied")?
    } else {
        String::new()
    };
    let pair_rows = if screening_model == "supplied" {
        let rows = controls.opt_rows("screened_pair_coulomb")?.ok_or_else(|| {
            DftError::InvalidInput(
                "scdft_pairing requires screened_pair_coulomb on flattened k/bands or an explicit generated scdft_screening model".into(),
            )
        })?;
        if rows.iter().any(|row| row.len() != rows.len()) {
            return Err(DftError::InvalidInput(
                "screened_pair_coulomb must be square on flattened k/bands".into(),
            ));
        }
        Some(rows.into_iter().flatten().collect())
    } else {
        None
    };
    let screening = match screening_model.as_str() {
        "" if !scdft_pairing => None,
        "supplied" => None,
        "rpa" | "rpa-neutral" => {
            if controls.contains("screened_pair_coulomb") {
                return Err(DftError::InvalidInput(
                    "generated scdft_screening and supplied screened_pair_coulomb are mutually exclusive".into(),
                ));
            }
            Some(CoulombScreeningOptions {
                reciprocal_cutoff: controls.f64("response_reciprocal_cutoff", 6.0)?,
                gamma_head: if screening_model == "rpa" {
                    GammaHead::Thermal
                } else {
                    GammaHead::Neutral
                },
                tolerance: config.acceptance(),
            })
        }
        _ => {
            return Err(DftError::InvalidInput(
                "scdft_screening must be supplied, rpa, or rpa-neutral".into(),
            ))
        }
    };
    if scdft_pairing && kt <= 0.0 {
        return Err(DftError::InvalidInput(
            "SCDFT phonon kernel requires positive electronic_temperature".into(),
        ));
    }
    if pade_points == 0 {
        return Err(DftError::InvalidInput(
            "`pade_points` must be at least one".into(),
        ));
    }
    controls.finish("electron-phonon")?;
    let (super_system, super_method) = supercell_system(system, &method, supercell)?;
    let setup = prepare_periodic(&super_system, &super_method)?;
    let scf = run_reference(&setup)?;
    let generated_screening = screening
        .map(|screening| {
            screened_pair_coulomb(
                &setup.cell,
                &setup.functional,
                &scf,
                &setup.options,
                &screening,
            )
        })
        .transpose()?;
    let options = crate::periodic::electron_phonon::ElectronPhononOptions {
        electron_smearing,
        phonon_smearing,
        mode_cutoff,
    };
    let ep = crate::periodic::electron_phonon::electron_phonon(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
        &config,
        &options,
    )?;
    let top = ep.frequencies.iter().fold(0.0_f64, |m, w| m.max(*w)) + 5.0 * phonon_smearing;
    let grid: Vec<f64> = (0..points)
        .map(|i| top * i as f64 / (points - 1).max(1) as f64)
        .collect();
    let a2f = crate::periodic::electron_phonon::eliashberg_function(
        &ep.frequencies,
        &ep.mode_lambda,
        &grid,
        phonon_smearing,
    );
    let kelvin = 1.0 / crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
    let ad = crate::periodic::electron_phonon::allen_dynes(
        ep.lambda,
        mu_star,
        ep.omega_log,
        ep.omega_2,
        true,
    );
    let mc = crate::periodic::electron_phonon::allen_dynes(
        ep.lambda,
        mu_star,
        ep.omega_log,
        ep.omega_2,
        false,
    );
    // The Matsubara cutoff defaults to ten times the highest included mode;
    // without an included mode there is no spectrum to solve for.
    let top_mode = ep
        .frequencies
        .iter()
        .filter(|w| **w > mode_cutoff)
        .fold(0.0_f64, |m, w| m.max(*w));
    let cutoff = eliashberg_cutoff.unwrap_or(10.0 * top_mode);
    let eliashberg = if top_mode > 0.0 {
        crate::periodic::electron_phonon::eliashberg_tc(
            &ep.frequencies,
            &ep.mode_lambda,
            mu_star,
            cutoff,
            eliashberg_t_min,
        )?
    } else {
        None
    };
    // Nonlinear gap at the requested temperatures; its real-axis value is the
    // Pade continuation at omega = 0.1 Delta(i w_0) with eta = 1e-8 times the
    // highest included mode.
    let mut gaps = Vec::new();
    if top_mode > 0.0 {
        for t_k in gap_temperatures.iter().flatten() {
            let temperature = t_k * crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
            let gap = crate::periodic::eliashberg::eliashberg_gap(
                &ep.frequencies,
                &ep.mode_lambda,
                mu_star,
                cutoff,
                temperature,
                &crate::periodic::eliashberg::GapOptions::default(),
            )?;
            let superconducting = gap.delta.iter().any(|d| *d != 0.0);
            let delta_real = if superconducting {
                let omega = [0.1 * gap.delta[0]];
                let value = crate::periodic::eliashberg::real_axis_gap(
                    &gap,
                    pade_points,
                    &omega,
                    1e-8 * top_mode,
                )?;
                Value::Float(value[0].re)
            } else {
                Value::Null
            };
            gaps.push(
                Value::object()
                    .with("temperature_k", *t_k)
                    .with("converged", gap.converged)
                    .with("iterations", gap.iterations)
                    .with("residual", gap.residual)
                    .with("delta_iw0_hartree", gap.delta[0])
                    .with("z_iw0", gap.z[0])
                    .with("delta_real_hartree", delta_real),
            );
        }
    }
    let mut report = reference_report(
        &setup,
        &scf,
        "electron-phonon coupling from the Gamma displacement responses of a \
            supercell (commensurate wave vectors), isotropic Eliashberg function, \
            McMillan/Allen-Dynes critical temperatures and the critical temperature \
            and nonlinear gap of the isotropic imaginary-axis Eliashberg equations \
            with a Pade continuation to the real axis",
    )
    .with("supercell", supercell.to_vec())
    .with("lambda", ep.lambda)
    .with("omega_log_k", ep.omega_log * kelvin)
    .with("omega_2_k", ep.omega_2 * kelvin)
    .with("fermi_dos", ep.fermi_dos.to_vec())
    .with(
        "frequencies_cm1",
        ep.frequencies.iter().map(|w| w * cm).collect::<Vec<_>>(),
    )
    .with("mode_lambda", ep.mode_lambda.clone())
    .with("excluded_modes", ep.excluded_modes)
    .with(
        "a2f_frequencies_cm1",
        grid.iter().map(|w| w * cm).collect::<Vec<_>>(),
    )
    .with("a2f", a2f)
    .with("mu_star", mu_star)
    .with("electron_smearing", electron_smearing)
    .with("eigenvalue_derivatives", ep.eigenvalue_derivatives.clone())
    .with("acoustic_residual", ep.force_constants.acoustic_residual)
    .with("eliashberg_cutoff_cm1", cutoff * cm)
    .with("eliashberg_gaps", gaps)
    .with("pade_points", pade_points)
    .with(
        "provenance",
        "analytic displacement responses; McMillan/Allen-Dynes are \
            semi-empirical formulas; the Eliashberg Tc is the linearized isotropic \
            gap equation and the gaps are its nonlinear form, both with a \
            semi-empirical mu_star; the Pade continuation is ill-conditioned for \
            many points",
    );
    report.set(
        "tc_allen_dynes_k",
        ad.map_or(Value::Null, |t| Value::Float(t * kelvin)),
    );
    report.set(
        "tc_mcmillan_k",
        mc.map_or(Value::Null, |t| Value::Float(t * kelvin)),
    );
    report.set(
        "tc_eliashberg_k",
        eliashberg.map_or(Value::Null, |t| Value::Float(t * kelvin)),
    );
    report.set(
        "units",
        units(&[
            ("omega_log_k", "K"),
            ("omega_2_k", "K"),
            ("fermi_dos", "states/hartree per spin per supercell"),
            ("frequencies_cm1", "cm^-1"),
            ("a2f", "dimensionless, on a2f_frequencies_cm1"),
            ("electron_smearing", "hartree (Gaussian)"),
            (
                "eigenvalue_derivatives",
                "hartree/bohr, [spin][k][band][3 atom + axis]",
            ),
            ("tc_allen_dynes_k", "K (with f1 f2)"),
            ("tc_mcmillan_k", "K (omega_log prefactor)"),
            (
                "tc_eliashberg_k",
                "K (isotropic Migdal-Eliashberg, imaginary axis; null when no solution \
                above eliashberg_t_min_k)",
            ),
            ("eliashberg_cutoff_cm1", "cm^-1 (Matsubara energy cutoff)"),
            (
                "eliashberg_gaps",
                "list of objects: temperature_k K; delta_iw0_hartree and \
                delta_real_hartree as energies in atomic units; z_iw0 dimensionless; delta_real_hartree \
                null in the normal state",
            ),
        ]),
    );
    let screened = pair_rows.or_else(|| {
        generated_screening
            .as_ref()
            .map(|result| result.kernel.as_slice().to_vec())
    });
    if let Some(screened) = screened {
        let (input, weights, selected, spin_defect) =
            ep.pairing_input(screened, kt, mode_cutoff, config.acceptance())?;
        let n = input.energies.len();
        let matrices = |v: &[f64]| v.chunks(n).map(|r| r.to_vec()).collect::<Vec<_>>();
        let rule = crate::properties::pairing::Lm2005::default();
        let kernel = rule.evaluate(&input)?;
        let mass = rule.mass(&input, &weights)?;
        let inputs = Value::object()
            .with("band_energies", input.energies)
            .with("band_weights", weights)
            .with("chemical_potential", input.chemical_potential)
            .with("electronic_temperature", method.electronic_temperature)
            .with("phonon_energies", input.phonon_energies)
            .with("pair_couplings_squared", input.coupling_squared)
            .with("screened_pair_coulomb", matrices(&input.screened_coulomb));
        report.set("scdft_inputs", inputs);
        report.set("scdft_pairing_kernel", matrices(&kernel.total));
        report.set("scdft_phonon_pairing_kernel", matrices(&kernel.phonon));
        report.set("scdft_mass_renormalization", mass);
        report.set("scdft_selected_modes", selected);
        report.set("scdft_spin_defect", spin_defect);
        report.set("scdft_screening", screening_model);
        report.set("scdft_electron_number", (scf.n_alpha + scf.n_beta) as f64);
        report.set("scdft_adapter_model","normal bands and Pulay-complete DFPT Gamma-supercell phonon vertices, without Fermi-surface filtering; k/band flattening and one band weight 1/Nk. The phonon kernel uses the Gamma-node contribution with no renormalization of missing q nodes. Nk=1 is the full commensurate supercell phonon model; Nk>1 omits phonon transfers between distinct supercell k points. Generated static RPA Coulomb screening uses the complete k-point difference mesh. Spin-degenerate matching frames are required; normal-state vertex response and arbitrary inter-k/q interpolation remain separate.");
        let mut labels = report.get("units").unwrap().clone();
        for (key, label) in [
            ("scdft_inputs.band_energies", "hartree"),
            ("scdft_inputs.band_weights", "dimensionless"),
            ("scdft_inputs.chemical_potential", "hartree"),
            ("scdft_inputs.electronic_temperature", "K"),
            ("scdft_inputs.phonon_energies", "hartree"),
            ("scdft_inputs.pair_couplings_squared", "hartree^2"),
            ("scdft_inputs.screened_pair_coulomb", "hartree"),
            ("scdft_pairing_kernel", "hartree"),
            ("scdft_phonon_pairing_kernel", "hartree"),
            ("scdft_mass_renormalization", "dimensionless"),
            ("scdft_spin_defect", "hartree"),
            ("scdft_electron_number", "dimensionless"),
        ] {
            labels.set(key, label);
        }
        if let (Some(options), Some(screening)) = (screening, generated_screening) {
            report.set(
                "scdft_screening_diagnostics",
                Value::object()
                    .with("reciprocal_cutoff", options.reciprocal_cutoff)
                    .with("transfers", screening.transfers.iter().map(|q| q.to_array()).collect::<Vec<_>>())
                    .with("reciprocal_modes", screening.reciprocal_modes)
                    .with("scaled_inverse_residuals", screening.scaled_inverse_residuals)
                    .with("inverse_residuals", screening.inverse_residuals)
                    .with("inverse_hermiticity_defects", screening.inverse_hermiticity_defects)
                    .with("gamma_head", screening.gamma_head)
                    .with("gamma_compressibility", screening.gamma_compressibility)
                    .with("gamma_ensemble", if options.gamma_head == GammaHead::Thermal {
                        "finite-temperature static long-wavelength limit at fixed chemical potential"
                    } else {
                        "fixed-number neutral cell with the constant Coulomb mode removed"
                    })
                    .with("fermi_level_crossing", screening.fermi_level_crossing)
                    .with("occupation_defect", screening.occupation_defect)
                    .with("electron_number_residual", screening.electron_number_residual)
                    .with("spin_fock_defect", screening.spin_fock_defect)
                    .with("time_reversal_fock_defect", screening.time_reversal_fock_defect)
                    .with("time_reversal_density_defect", screening.time_reversal_density_defect)
                    .with("pair_imaginary_defect", screening.pair_imaginary_defect)
                    .with("pair_reciprocity_defect", screening.pair_reciprocity_defect),
            );
            for (key, label) in [
                ("reciprocal_cutoff", "1/bohr"),
                ("transfers", "1/bohr"),
                ("reciprocal_modes", "integer"),
                ("scaled_inverse_residuals", "dimensionless"),
                ("inverse_residuals", "dimensionless"),
                ("inverse_hermiticity_defects", "dimensionless"),
                ("gamma_head", "hartree"),
                ("gamma_compressibility", "bohr^-3/hartree"),
                ("occupation_defect", "dimensionless"),
                ("electron_number_residual", "dimensionless"),
                ("spin_fock_defect", "hartree"),
                ("time_reversal_fock_defect", "hartree"),
                ("time_reversal_density_defect", "dimensionless"),
                ("pair_imaginary_defect", "hartree"),
                ("pair_reciprocity_defect", "hartree"),
            ] {
                labels.set(&format!("scdft_screening_diagnostics.{key}"), label);
            }
        }
        report.set("units", labels);
    }
    Ok(report)
}

/// Phonon dispersion along a reduced q path from analytic supercell force
/// constants, with an optional density of states on a uniform mesh.
fn phonon_bands(system: &System, controls: &Controls) -> Result<Value> {
    let mut method = PeriodicMethod::from_controls_with_dispersion(controls)?;
    method.apply_collinear_reference(controls)?;
    let config = super::vibrational::response_config(controls)?;
    let supercell = supercell_control(controls)?;
    let vertices = controls.opt_rows("path")?.ok_or_else(|| {
        DftError::InvalidInput(
            "phonon-bands needs path: reduced q vertices, e.g. [[0,0,0],[0.5,0,0]]".into(),
        )
    })?;
    let points = controls.usize("points", 20)?;
    let labels = controls.opt_str("labels")?;
    let dos_mesh = controls.opt_usize_list("dos_mesh")?;
    let dos_sigma = controls.f64("dos_sigma", 5.0)?;
    let dos_points = controls.usize("dos_points", 400)?;
    if let Some(mesh) = &dos_mesh {
        if mesh.len() != 3
            || mesh.contains(&0)
            || mesh
                .iter()
                .try_fold(1usize, |n, k| n.checked_mul(*k))
                .is_none()
            || dos_sigma <= 0.0
            || dos_points < 2
        {
            return Err(DftError::InvalidInput(
                "phonon DOS requires three positive mesh entries, a positive Gaussian width, and at least two grid points".into(),
            ));
        }
    }
    let wavevector_direction = controls.opt_vec3("wavevector_direction")?;
    let lattice_flexoelectric = controls.bool("lattice_flexoelectric", false)?;
    let supplied_born = controls.opt_rows("born_charge_rows")?;
    let wavevector_order = controls.opt_usize("response_order")?;
    let phonon_phase = controls.str("phonon_phase", "cell")?;
    let phase = match phonon_phase.as_str() {
        "cell" => crate::periodic::phonon_bands::PhononPhase::Cell,
        "atomic" => crate::periodic::phonon_bands::PhononPhase::Atomic,
        _ => {
            return Err(DftError::InvalidInput(
                "phonon_phase must be cell or atomic".into(),
            ))
        }
    };
    if wavevector_order.is_some_and(|n| n > 4)
        || (wavevector_direction.is_none()
            && (wavevector_order.is_some() || phonon_phase != "cell"))
    {
        return Err(DftError::InvalidInput("phonon wavevector response needs wavevector_direction and response_order zero through four".into()));
    }
    if (lattice_flexoelectric
        && (wavevector_direction.is_none()
            || wavevector_order.unwrap_or(2) < 2
            || method.charge != 0))
        || (supplied_born.is_some() && !lattice_flexoelectric)
    {
        return Err(DftError::InvalidInput("lattice_flexoelectric needs a neutral cell, wavevector_direction and order two through four; born_charge_rows requires this option".into()));
    }
    if lattice_flexoelectric
        && supplied_born.is_none()
        && super::double_hybrid::requested(controls)?
    {
        return Err(DftError::Unsupported {
            feature: "phonon-bands correlated lattice flexoelectricity".into(),
            missing: "the correlated Born-charge connection; supply matching born_charge_rows"
                .into(),
        });
    }
    let (model, report) = phonon_spectrum(
        "phonon-bands",
        system,
        controls,
        &method,
        supercell,
        &config,
        "phonon dispersion from analytic Gamma-point force constants of a supercell (exact at commensurate q, interpolated elsewhere)",
    )?;
    let lattice = lattice_of(system)?;
    let reduced_path = path_points(&vertices, points)?;
    let path: Vec<Vec3> = reduced_path
        .iter()
        .map(|f| cartesian(&lattice, *f))
        .collect();
    let mut frequencies = Vec::with_capacity(path.len());
    let mut hermiticity = 0.0_f64;
    for q in &path {
        let (f, defect) = model.frequencies(*q)?;
        hermiticity = hermiticity.max(defect);
        frequencies.push(f);
    }
    let mut distance = vec![0.0; path.len()];
    for i in 1..path.len() {
        distance[i] = distance[i - 1] + (path[i] - path[i - 1]).norm();
    }
    let mut report = report
        .with(
            "q_path",
            path.iter().map(|q| q.to_array()).collect::<Vec<_>>(),
        )
        .with("q_path_reduced", reduced_path.clone())
        .with("path_distance", distance)
        .with("frequencies_cm1", frequencies)
        .with("dynamical_matrix_hermiticity", hermiticity);
    if let Some(labels) = labels {
        report.set("labels", labels);
    }
    if let Some(mesh) = dos_mesh {
        let qs = mesh_points(&lattice, [mesh[0], mesh[1], mesh[2]]);
        let mesh_frequencies: Vec<Vec<f64>> = qs
            .iter()
            .map(|q| model.frequencies(*q).map(|r| r.0))
            .collect::<Result<_>>()?;
        let top = mesh_frequencies
            .iter()
            .flatten()
            .fold(0.0_f64, |m, f| m.max(*f))
            + 5.0 * dos_sigma;
        let bottom = mesh_frequencies
            .iter()
            .flatten()
            .fold(0.0_f64, |m, f| m.min(*f))
            - 5.0 * dos_sigma;
        let grid: Vec<f64> = (0..dos_points)
            .map(|i| bottom + (top - bottom) * i as f64 / (dos_points - 1).max(1) as f64)
            .collect();
        let dos = crate::periodic::phonon_bands::phonon_dos(&mesh_frequencies, &grid, dos_sigma);
        report.set("dos_mesh", mesh.clone());
        report.set("dos_frequencies_cm1", grid);
        report.set("dos", dos);
    }
    report.set(
        "units",
        phonon_spectrum_units(
            &report,
            &[
                ("energy", "hartree per supercell"),
                ("q_path", "1/bohr"),
                ("path_distance", "1/bohr"),
                ("frequencies_cm1", "cm^-1 (negative = imaginary)"),
                ("dos_frequencies_cm1", "cm^-1"),
                ("dos", "states per cm^-1 per primitive cell (3N modes)"),
                ("acoustic_residual", "hartree/bohr^2"),
                ("force_constant_asymmetry", "hartree/bohr^2"),
                ("supercell_force_constants", "hartree/bohr^2"),
                ("dynamical_matrix_hermiticity", "hartree/bohr^2/m_e"),
            ],
        ),
    );
    if let Some(direction) = wavevector_direction {
        let order = wavevector_order.unwrap_or(2);
        let rows = |series: &[Matrix]| -> Vec<Vec<Vec<f64>>> {
            series
                .iter()
                .map(|a| {
                    (0..a.rows)
                        .map(|i| (0..a.cols).map(|j| a[(i, j)]).collect())
                        .collect()
                })
                .collect()
        };
        let mut dynamical_real = Vec::new();
        let mut dynamical_imaginary = Vec::new();
        let mut force_real = Vec::new();
        let mut force_imaginary = Vec::new();
        let mut defects = Vec::new();
        for q in &path {
            let series = model.fourier_taylor(*q, direction, order, phase, true)?;
            dynamical_real.push(rows(&series.real));
            dynamical_imaginary.push(rows(&series.imaginary));
            defects.push(series.hermiticity_defects);
            let series = model.fourier_taylor(*q, direction, order, phase, false)?;
            force_real.push(rows(&series.real));
            force_imaginary.push(rows(&series.imaginary));
        }
        report.set("wavevector_direction", direction);
        report.set("wavevector_response_order", order);
        report.set("phonon_phase", phonon_phase);
        report.set("dynamical_q_coefficients_real", dynamical_real);
        report.set("dynamical_q_coefficients_imaginary", dynamical_imaginary);
        report.set("force_constant_q_coefficients_real", force_real);
        report.set("force_constant_q_coefficients_imaginary", force_imaginary);
        report.set("dynamical_q_hermiticity_defects", defects);
        report.set("wavevector_coefficient_convention","q(t)=q+t*wavevector_direction; dimensionless t, factorial-normalized coefficients in [path point][order][row][column]; atomic phase uses R+tau_final-tau_initial, cell phase uses R; no acoustic or Hermitian projection; this is a force-constant moment component, not the complete flexoelectric polarization");
        let mut labels = report.get("units").unwrap().clone();
        labels.set("wavevector_direction", "1/bohr");
        for key in [
            "dynamical_q_coefficients_real",
            "dynamical_q_coefficients_imaginary",
            "dynamical_q_hermiticity_defects",
        ] {
            labels.set(key, "hartree/bohr^2/m_e");
        }
        for key in [
            "force_constant_q_coefficients_real",
            "force_constant_q_coefficients_imaginary",
        ] {
            labels.set(key, "hartree/bohr^2");
        }
        report.set("units", labels);
        if lattice_flexoelectric {
            let order = wavevector_order.unwrap_or(2);
            let direction = direction / direction.norm();
            let relaxation = model.long_wave_relaxation(direction, order, &config)?;
            let born = if let Some(rows) = supplied_born {
                if rows.len() != 3 * system.numbers.len() || rows.iter().any(|r| r.len() != 3) {
                    return Err(DftError::InvalidInput("born_charge_rows needs 3N rows of three displacement-axis charges, ordered atom then polarization".into()));
                }
                rows.chunks(3)
                    .map(|r| std::array::from_fn(|a| [r[a][0], r[a][1], r[a][2]]))
                    .collect::<Vec<_>>()
            } else {
                let primitive = prepare_periodic(system, &method)?;
                let reference = run_reference(&primitive)?;
                crate::periodic::dielectric::dielectric_tensors_with_force_constants(
                    &primitive.cell,
                    &primitive.functional,
                    &reference,
                    &primitive.options,
                    &config,
                    true,
                    false,
                )?
                .born_charges
            };
            let (pr, pi) = relaxation.born_polarization(&born, lattice.volume().abs())?;
            let rows = (0..=order)
                .map(|n| {
                    Value::object()
                        .with("order", n)
                        .with("displacement_real", relaxation.real[n].clone())
                        .with("displacement_imaginary", relaxation.imaginary[n].clone())
                        .with("polarization_real", pr[n].clone())
                        .with("polarization_imaginary", pi[n].clone())
                        .with(
                            "relative_residuals",
                            relaxation.relative_residuals[n].clone(),
                        )
                        .with(
                            "center_of_mass_defects",
                            relaxation.center_of_mass_defects[n].clone(),
                        )
                        .with(
                            "acoustic_reaction_real",
                            relaxation.acoustic_reactions_real[n].clone(),
                        )
                        .with(
                            "acoustic_reaction_imaginary",
                            relaxation.acoustic_reactions_imaginary[n].clone(),
                        )
                })
                .collect::<Vec<_>>();
            let mut mu = pr[2].clone();
            for x in mu.as_mut_slice() {
                *x = -*x;
            }
            report.set("long_wave_relaxation", rows);
            report.set("long_wave_direction", direction);
            report.set("long_wave_born_charges", born);
            report.set("born_lattice_flexoelectric", mu);
            report.set(
                "long_wave_uniform_force_defect",
                relaxation.uniform_force_defect,
            );
            report.set(
                "long_wave_converged",
                relaxation.uniform_force_defect <= config.acceptance(),
            );
            report.set("long_wave_model","atomic-phase constrained internal relaxation at fixed mass center; q=kappa*unit_direction, kappa in inverse Bohr; displacement per imposed acoustic displacement. born_lattice_flexoelectric=-P_2 is the Born-charge lattice contribution to type-I dP_a/d(partial_gamma partial_delta u_beta) contracted along unit_direction twice. Finite supercell IFC model; clamped-ion polarization, dynamic multipoles, metric/current terms and bulk/surface completion remain separate.");
            let mut units = report.get("units").unwrap().clone();
            for (key, label) in [
                ("long_wave_direction", "dimensionless"),
                ("long_wave_born_charges", "e"),
                ("born_lattice_flexoelectric", "e/bohr"),
                ("long_wave_uniform_force_defect", "hartree/bohr^2"),
                ("long_wave_relaxation.displacement_real", "bohr^order"),
                ("long_wave_relaxation.displacement_imaginary", "bohr^order"),
                ("long_wave_relaxation.polarization_real", "e*bohr^(order-3)"),
                (
                    "long_wave_relaxation.polarization_imaginary",
                    "e*bohr^(order-3)",
                ),
                ("long_wave_relaxation.center_of_mass_defects", "bohr^order"),
                ("long_wave_relaxation.relative_residuals", "dimensionless"),
                (
                    "long_wave_relaxation.acoustic_reaction_real",
                    "hartree*bohr^(order-2)",
                ),
                (
                    "long_wave_relaxation.acoustic_reaction_imaginary",
                    "hartree*bohr^(order-2)",
                ),
            ] {
                units.set(key, label);
            }
            report.set("units", units);
        }
    }
    Ok(report)
}

/// Harmonic lattice thermodynamics per primitive cell from analytic
/// supercell force constants on a uniform q mesh.
fn phonon_thermo(system: &System, controls: &Controls) -> Result<Value> {
    let mut method = PeriodicMethod::from_controls_with_dispersion(controls)?;
    method.apply_collinear_reference(controls)?;
    let config = super::vibrational::response_config(controls)?;
    let supercell = supercell_control(controls)?;
    let mesh = controls.opt_usize_list("q_mesh")?.ok_or_else(|| {
        DftError::InvalidInput("phonon-thermo needs q_mesh: three positive entries".into())
    })?;
    if mesh.len() != 3
        || mesh.contains(&0)
        || mesh
            .iter()
            .try_fold(1usize, |n, k| n.checked_mul(*k))
            .is_none()
    {
        return Err(DftError::InvalidInput(
            "`q_mesh` must have three positive entries".into(),
        ));
    }
    let temperatures = controls
        .opt_f64_list("temperatures")?
        .unwrap_or_else(|| vec![0.0, 100.0, 200.0, 298.15, 400.0, 600.0]);
    if temperatures.is_empty() || temperatures.iter().any(|t| *t < 0.0) {
        return Err(DftError::InvalidInput(
            "`temperatures` needs at least one non-negative value (K)".into(),
        ));
    }
    let cutoff = controls.f64("mode_cutoff_cm1", 1.0)?;
    if cutoff < 0.0 {
        return Err(DftError::InvalidInput(
            "`mode_cutoff_cm1` must be non-negative".into(),
        ));
    }
    let (model, report) = phonon_spectrum(
        "phonon-thermo",
        system,
        controls,
        &method,
        supercell,
        &config,
        "harmonic lattice thermodynamics per primitive cell from analytic supercell force constants on a uniform q mesh",
    )?;
    let lattice = lattice_of(system)?;
    let qs = mesh_points(&lattice, [mesh[0], mesh[1], mesh[2]]);
    let frequencies: Vec<Vec<f64>> = qs
        .iter()
        .map(|q| model.frequencies(*q).map(|r| r.0))
        .collect::<Result<_>>()?;
    let imaginary = frequencies
        .iter()
        .flatten()
        .filter(|f| **f < -cutoff)
        .count();
    let (thermo, excluded) =
        crate::periodic::phonon_bands::harmonic_thermo(&frequencies, &temperatures, cutoff);
    let kj = crate::constants::HARTREE_TO_KJ_MOL;
    let report = report
        .with("q_mesh", mesh.clone())
        .with("mesh_frequencies_cm1", frequencies)
        .with("excluded_modes", excluded)
        .with("imaginary_modes", imaginary)
        .with("temperatures", temperatures.clone())
        .with(
            "zero_point_energy",
            thermo.first().map_or(0.0, |t| t.zero_point_energy),
        )
        .with(
            "internal_energy",
            thermo.iter().map(|t| t.internal_energy).collect::<Vec<_>>(),
        )
        .with(
            "free_energy",
            thermo.iter().map(|t| t.free_energy).collect::<Vec<_>>(),
        )
        .with(
            "entropy",
            thermo.iter().map(|t| t.entropy).collect::<Vec<_>>(),
        )
        .with(
            "heat_capacity",
            thermo.iter().map(|t| t.heat_capacity).collect::<Vec<_>>(),
        )
        .with(
            "free_energy_kj_mol",
            thermo
                .iter()
                .map(|t| t.free_energy * kj)
                .collect::<Vec<_>>(),
        )
        .with(
            "entropy_j_mol_k",
            thermo
                .iter()
                .map(|t| t.entropy * kj * 1000.0)
                .collect::<Vec<_>>(),
        )
        .with(
            "heat_capacity_j_mol_k",
            thermo
                .iter()
                .map(|t| t.heat_capacity * kj * 1000.0)
                .collect::<Vec<_>>(),
        );
    let labels = phonon_spectrum_units(
        &report,
        &[
            ("energy", "hartree per supercell"),
            ("zero_point_energy", "hartree per primitive cell"),
            (
                "internal_energy",
                "hartree per primitive cell (zero point included)",
            ),
            (
                "free_energy",
                "hartree per primitive cell (zero point included)",
            ),
            ("entropy", "hartree/K per primitive cell"),
            (
                "heat_capacity",
                "hartree/K per primitive cell (constant volume)",
            ),
            ("free_energy_kj_mol", "kJ per mol of primitive cells"),
            ("entropy_j_mol_k", "J/(mol K) per mol of primitive cells"),
            (
                "heat_capacity_j_mol_k",
                "J/(mol K) per mol of primitive cells",
            ),
            ("temperatures", "K"),
            ("mesh_frequencies_cm1", "cm^-1 (negative = imaginary)"),
            ("supercell_force_constants", "hartree/bohr^2"),
        ],
    );
    Ok(report.with("units", labels))
}

/// Quasi-harmonic thermodynamics from analytic supercell phonons on a grid
/// of isotropically scaled cells.
fn quasi_harmonic_feature(system: &System, controls: &Controls) -> Result<Value> {
    let method = PeriodicMethod::from_controls_with_dispersion(controls)?;
    let config = super::vibrational::response_config(controls)?;
    let supercell = supercell_control(controls)?;
    let mesh = controls.opt_usize_list("q_mesh")?.ok_or_else(|| {
        DftError::InvalidInput("quasi-harmonic needs q_mesh: three positive entries".into())
    })?;
    if mesh.len() != 3 || mesh.contains(&0) {
        return Err(DftError::InvalidInput(
            "`q_mesh` must have three positive entries".into(),
        ));
    }
    let scales = controls
        .opt_f64_list("scales")?
        .unwrap_or_else(|| vec![0.98, 0.99, 1.0, 1.01, 1.02]);
    if scales.len() < 4 || scales.iter().any(|s| *s <= 0.0) {
        return Err(DftError::InvalidInput(
            "`scales` needs at least four positive linear scale factors".into(),
        ));
    }
    let temperatures = controls
        .opt_f64_list("temperatures")?
        .unwrap_or_else(|| vec![0.0, 100.0, 200.0, 298.15, 400.0, 600.0]);
    if temperatures.iter().any(|t| *t < 0.0) {
        return Err(DftError::InvalidInput(
            "`temperatures` must be non-negative (K)".into(),
        ));
    }
    let cutoff = controls.f64("mode_cutoff_cm1", 1.0)?;
    let pressure_gpa = controls.f64("pressure_gpa", 0.0)?;
    if !pressure_gpa.is_finite() {
        return Err(DftError::InvalidInput(
            "`pressure_gpa` must be finite (GPa)".into(),
        ));
    }
    controls.finish("quasi-harmonic")?;
    let rows = system.cell.ok_or_else(|| {
        DftError::InvalidInput("a periodic calculation needs a cell (lattice rows, Bohr)".into())
    })?;
    let cells = supercell.iter().product::<usize>() as f64;
    let mut volumes = Vec::new();
    let mut energies = Vec::new();
    let mut free = Vec::new();
    let mut entropy = Vec::new();
    let mut capacity = Vec::new();
    let mut imaginary = Vec::new();
    let mut dynamical: Vec<Vec<(crate::linalg::Matrix, crate::linalg::Matrix)>> = Vec::new();
    let mut reference: Option<SupercellPhonons> = None;
    for &scale in &scales {
        let scaled = System {
            numbers: system.numbers.clone(),
            positions: system
                .positions
                .iter()
                .map(|p| p.map(|v| v * scale))
                .collect(),
            cell: Some(rows.map(|r| r.map(|v| v * scale))),
            pbc: system.pbc,
        };
        let sp = supercell_phonons(&scaled, &method, supercell, &config)?;
        let lattice = lattice_of(&scaled)?;
        let qs = mesh_points(&lattice, [mesh[0], mesh[1], mesh[2]]);
        dynamical.push(qs.iter().map(|q| sp.model.dynamical_matrix(*q)).collect());
        let frequencies: Vec<Vec<f64>> = qs
            .iter()
            .map(|q| sp.model.frequencies(*q).map(|r| r.0))
            .collect::<Result<_>>()?;
        imaginary.push(
            frequencies
                .iter()
                .flatten()
                .filter(|f| **f < -cutoff)
                .count(),
        );
        let (thermo, _) =
            crate::periodic::phonon_bands::harmonic_thermo(&frequencies, &temperatures, cutoff);
        volumes.push(lattice.volume().abs());
        energies.push((sp.scf.energy + sp.dispersion_energy) / cells);
        free.push(thermo.iter().map(|t| t.free_energy).collect::<Vec<_>>());
        entropy.push(thermo.iter().map(|t| t.entropy).collect::<Vec<_>>());
        capacity.push(thermo.iter().map(|t| t.heat_capacity).collect::<Vec<_>>());
        if (scale - 1.0).abs() < 1e-12 || reference.is_none() {
            reference = Some(sp);
        }
    }
    let gpa = crate::constants::HARTREE_PER_BOHR3_TO_GPA;
    let states = crate::periodic::quasi_harmonic::quasi_harmonic_at_pressure(
        &volumes,
        &energies,
        &temperatures,
        &free,
        &entropy,
        &capacity,
        pressure_gpa / gpa,
    )?;
    let kj = crate::constants::HARTREE_TO_KJ_MOL;
    let sp = reference.expect("at least four volumes");
    // The static curve need not have a minimum on a grid chosen for a pressure.
    let static_fit = crate::periodic::eos::fit_birch_murnaghan_isotherm(&volumes, &energies)?;
    let finite = |value: f64| Some(value).filter(|v| v.is_finite());
    let unit_volume = volumes[scales
        .iter()
        .position(|s| (*s - 1.0).abs() < 1e-12)
        .unwrap_or(0)];
    // Mode Grueneisen parameters at that volume from the volume dependence
    // of the dynamical matrices, and their heat-capacity-weighted average.
    let kb = crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
    let mut reduced_q = Vec::with_capacity(mesh.iter().product());
    for i in 0..mesh[0] {
        for j in 0..mesh[1] {
            for k in 0..mesh[2] {
                reduced_q.push([
                    i as f64 / mesh[0] as f64,
                    j as f64 / mesh[1] as f64,
                    k as f64 / mesh[2] as f64,
                ]);
            }
        }
    }
    let mut mode_gamma: Vec<Vec<Option<f64>>> = Vec::with_capacity(reduced_q.len());
    let mut mode_frequency: Vec<Vec<f64>> = Vec::with_capacity(reduced_q.len());
    let mut average = vec![(0.0_f64, 0.0_f64); temperatures.len()];
    for iq in 0..reduced_q.len() {
        let per_volume: Vec<(crate::linalg::Matrix, crate::linalg::Matrix)> =
            dynamical.iter().map(|d| d[iq].clone()).collect();
        let modes = crate::periodic::quasi_harmonic::eigenvalue_volume_derivatives(
            &volumes,
            &per_volume,
            unit_volume,
        )?;
        let mut gammas = Vec::with_capacity(modes.len());
        let mut wavenumbers = Vec::with_capacity(modes.len());
        for (lambda, slope) in modes {
            let wavenumber =
                lambda.signum() * lambda.abs().sqrt() * crate::constants::HARTREE_TO_CM1;
            wavenumbers.push(wavenumber);
            if wavenumber > cutoff {
                let gamma = -slope / (2.0 * lambda);
                gammas.push(Some(gamma));
                let omega = lambda.sqrt();
                for (entry, &temperature) in average.iter_mut().zip(&temperatures) {
                    if temperature > 0.0 {
                        let x = omega / (kb * temperature);
                        let em = (-x).exp();
                        let c = kb * x * x * em / ((1.0 - em) * (1.0 - em));
                        entry.0 += gamma * c;
                        entry.1 += c;
                    }
                }
            } else {
                gammas.push(None);
            }
        }
        mode_gamma.push(gammas);
        mode_frequency.push(wavenumbers);
    }
    let mode_average: Vec<f64> = average
        .iter()
        .map(|(numerator, denominator)| {
            if *denominator > 0.0 {
                numerator / denominator
            } else {
                0.0
            }
        })
        .collect();
    let mut report = phonon_common_report(
        &sp,
        "quasi-harmonic lattice thermodynamics: Birch-Murnaghan fits of E(V) + \
        F_vib(V, T) over isotropically scaled cells with analytic supercell \
        phonons on a uniform q mesh (fitted volume-grid model); mode \
        Grueneisen parameters from least-squares quadratics of the dynamical \
        matrices in ln V",
    )
    .with("scales", scales.clone())
    .with("volumes", volumes.clone())
    .with("static_energies", energies.clone())
    .with("imaginary_modes_per_volume", imaginary.clone())
    .with("static_v0", finite(static_fit.v0))
    .with("static_b0_gpa", finite(static_fit.b0 * gpa))
    .with("static_b0_prime", finite(static_fit.b0_prime))
    .with("temperatures", temperatures.clone())
    .with("pressure_gpa", pressure_gpa)
    .with("vibrational_free_energies", free.clone())
    .with(
        "gibbs_energy",
        states.iter().map(|s| s.gibbs_energy).collect::<Vec<_>>(),
    )
    .with("mode_gruneisen_volume", unit_volume)
    .with("mode_gruneisen_q_reduced", reduced_q)
    .with("mode_gruneisen_frequencies_cm1", mode_frequency)
    .with("mode_gruneisen", mode_gamma)
    .with("gruneisen_mode_average", mode_average)
    .with(
        "volume",
        states.iter().map(|s| s.volume).collect::<Vec<_>>(),
    )
    .with(
        "linear_scale",
        states
            .iter()
            .map(|s| (s.volume / unit_volume).cbrt())
            .collect::<Vec<_>>(),
    )
    .with(
        "free_energy",
        states.iter().map(|s| s.free_energy).collect::<Vec<_>>(),
    )
    .with(
        "bulk_modulus_gpa",
        states
            .iter()
            .map(|s| s.bulk_modulus * gpa)
            .collect::<Vec<_>>(),
    )
    .with(
        "thermal_expansion",
        states
            .iter()
            .map(|s| s.thermal_expansion)
            .collect::<Vec<_>>(),
    )
    .with(
        "entropy_j_mol_k",
        states
            .iter()
            .map(|s| s.entropy * kj * 1000.0)
            .collect::<Vec<_>>(),
    )
    .with(
        "heat_capacity_v_j_mol_k",
        states
            .iter()
            .map(|s| s.heat_capacity_v * kj * 1000.0)
            .collect::<Vec<_>>(),
    )
    .with(
        "heat_capacity_p_j_mol_k",
        states
            .iter()
            .map(|s| s.heat_capacity_p * kj * 1000.0)
            .collect::<Vec<_>>(),
    )
    .with(
        "gruneisen",
        states.iter().map(|s| s.gruneisen).collect::<Vec<_>>(),
    )
    .with(
        "fit_residual_rms",
        states.iter().map(|s| s.residual_rms).collect::<Vec<_>>(),
    )
    .with(
        "provenance",
        "fitted quasi-harmonic model on analytic supercell force constants",
    );
    if imaginary.iter().any(|n| *n > 0) {
        report.set(
            "warnings",
            vec![
                "WARNING [quasi-harmonic] imaginary modes were excluded at some volumes"
                    .to_string(),
            ],
        );
    }
    report.set(
        "units",
        units(&[
            ("volumes", "bohr^3 per primitive cell"),
            ("static_energies", "hartree per primitive cell"),
            ("volume", "bohr^3 per primitive cell at each temperature"),
            ("linear_scale", "relative to scale 1 (or the first scale)"),
            ("free_energy", "hartree per primitive cell"),
            ("bulk_modulus_gpa", "GPa (isothermal)"),
            ("thermal_expansion", "1/K (volumetric)"),
            ("entropy_j_mol_k", "J/(mol K) per mol of primitive cells"),
            ("heat_capacity_v_j_mol_k", "J/(mol K)"),
            ("heat_capacity_p_j_mol_k", "J/(mol K)"),
            ("temperatures", "K"),
            ("pressure_gpa", "GPa, external pressure of every state"),
            ("gibbs_energy", "hartree per primitive cell, F + p V"),
            (
                "vibrational_free_energies",
                "hartree per primitive cell, [volume][temperature]",
            ),
            ("mode_gruneisen_volume", "bohr^3 per primitive cell"),
            ("mode_gruneisen_q_reduced", "reduced coordinates of the q mesh"),
            (
                "mode_gruneisen_frequencies_cm1",
                "cm^-1, [q][mode] of the fitted dynamical matrices at mode_gruneisen_volume",
            ),
            (
                "mode_gruneisen",
                "dimensionless, [q][mode] at mode_gruneisen_volume; null below mode_cutoff_cm1",
            ),
            (
                "gruneisen_mode_average",
                "dimensionless, heat-capacity-weighted mode average per temperature at mode_gruneisen_volume",
            ),
            ("fit_residual_rms", "hartree per primitive cell"),
        ]),
    );
    Ok(report)
}

/// Clamped- and relaxed-ion elastic constants from the analytic strain,
/// internal-strain and force-constant responses.
fn elastic(system: &System, controls: &Controls) -> Result<Value> {
    let mut method = PeriodicMethod::from_controls_with_dispersion(controls)?;
    method.apply_collinear_reference(controls)?;
    let config = super::vibrational::response_config(controls)?;
    controls.finish("elastic")?;
    let setup = prepare_periodic(system, &method)?;
    let scf = run_reference(&setup)?;
    let dispersion = method.dispersion_model_for(&system.numbers)?;
    let ec = crate::periodic::elastic::elastic_constants_with_dispersion(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
        &config,
        dispersion.as_ref(),
    )?;
    let natoms = setup.cell.molecule().len();
    let dim = 3 * natoms;
    let gpa = crate::constants::HARTREE_PER_BOHR3_TO_GPA;
    let to_gpa = |m: &[[f64; 6]; 6]| m.map(|row| row.map(|v| v * gpa));
    let rows = |data: &[f64], cols: usize| -> Vec<Vec<f64>> {
        data.chunks(cols).map(|c| c.to_vec()).collect()
    };
    let born = {
        let mut m = crate::linalg::Matrix::zeros(6, 6);
        for i in 0..6 {
            for j in 0..6 {
                m[(i, j)] = 0.5 * (ec.relaxed_ion[i][j] + ec.relaxed_ion[j][i]) * gpa;
            }
        }
        crate::linalg::symmetric_eigen(&m)?.0
    };
    let solves = |list: &[(usize, f64)]| {
        list.iter()
            .map(|(iterations, residual)| {
                Value::object()
                    .with("iterations", *iterations)
                    .with("relative_residual", *residual)
            })
            .collect::<Vec<_>>()
    };
    let s = ec.stress;
    let mut report = reference_report(
        &setup,
        &scf,
        "second-order elastic constants from analytic coupled-perturbed strain, \
            internal-strain and Gamma force-constant responses (Lagrangian strain; \
            relaxed ions minimize the energy at each strain)",
    )
    .with("orbital_residual", scf.orbital_residual)
    .with("voigt_order", "xx, yy, zz, yz, xz, xy")
    .with("stress", s)
    .with("stress_gpa", s.map(|row| row.map(|v| v * gpa)))
    .with("strain_derivative", ec.strain_derivative)
    .with("strain_hessian", rows(&ec.strain_hessian, 9))
    .with("internal_strain", rows(&ec.internal_strain, dim))
    .with("internal_relaxation", rows(&ec.internal_relaxation, 9))
    .with(
        "force_constants",
        (0..dim)
            .map(|x| {
                (0..dim)
                    .map(|y| ec.force_constants.hessian[(x, y)])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    )
    .with("clamped_ion", ec.clamped_ion)
    .with("relaxed_ion", ec.relaxed_ion)
    .with("clamped_ion_gpa", to_gpa(&ec.clamped_ion))
    .with("relaxed_ion_gpa", to_gpa(&ec.relaxed_ion))
    .with("clamped_ion_birch_gpa", to_gpa(&ec.clamped_ion_birch))
    .with("relaxed_ion_birch_gpa", to_gpa(&ec.relaxed_ion_birch))
    .with("relaxed_ion_eigenvalues_gpa", born)
    .with("raw_asymmetry", ec.raw_asymmetry)
    .with("internal_strain_asymmetry", ec.internal_strain_asymmetry)
    .with("rotation_residual", ec.rotation_residual)
    .with("force_constant_asymmetry", ec.force_constants.raw_asymmetry)
    .with("acoustic_residual", ec.force_constants.acoustic_residual)
    .with(
        "smallest_optical_eigenvalue",
        ec.smallest_optical_eigenvalue
            .map_or(Value::Null, Value::Float),
    )
    .with("strain_solves", solves(&ec.strain_solves))
    .with("displacement_solves", solves(&ec.force_constants.solves))
    .with("provenance", "analytic")
    .with(
        "units",
        units(&[
            ("energy", "hartree per cell"),
            ("orbital_residual", "hartree"),
            ("stress", "hartree/bohr^3 (sym(dE/d(epsilon))/V)"),
            ("stress_gpa", "GPa"),
            ("strain_derivative", "hartree per cell"),
            (
                "strain_hessian",
                "hartree per cell, [3a+b][3c+d] for x -> (1+epsilon) x",
            ),
            (
                "internal_strain",
                "hartree/bohr per cell, [3a+b][3 atom + c]",
            ),
            (
                "internal_relaxation",
                "bohr per unit strain, [3 atom + c][3a+b]",
            ),
            ("force_constants", "hartree/bohr^2"),
            ("clamped_ion", "hartree/bohr^3 (Voigt 6x6)"),
            ("relaxed_ion", "hartree/bohr^3 (Voigt 6x6)"),
            ("clamped_ion_gpa", "GPa"),
            ("relaxed_ion_gpa", "GPa"),
            ("clamped_ion_birch_gpa", "GPa (stress-strain coefficients)"),
            ("relaxed_ion_birch_gpa", "GPa (stress-strain coefficients)"),
            (
                "relaxed_ion_eigenvalues_gpa",
                "GPa (Born stability: all positive)",
            ),
            ("raw_asymmetry", "hartree per cell"),
            ("internal_strain_asymmetry", "hartree/bohr per cell"),
            ("rotation_residual", "hartree per cell"),
            ("smallest_optical_eigenvalue", "hartree/bohr^2"),
        ]),
    );
    if let Some(energy) = ec.dispersion_energy {
        include_dispersion(&mut report, &method, energy);
    }
    Ok(report)
}

fn response_solve_records(solves: &[(usize, f64)]) -> Vec<Value> {
    solves
        .iter()
        .map(|(iterations, residual)| {
            Value::object()
                .with("iterations", *iterations)
                .with("relative_residual", *residual)
        })
        .collect()
}

/// Polar Gamma-point lattice dynamics: transverse modes, mode effective
/// charges, infrared intensities, the static dielectric tensor and the
/// longitudinal frequencies of the non-analytic term along given directions.
fn polar_feature(system: &System, controls: &Controls, feature: &str) -> Result<Value> {
    let mut method = PeriodicMethod::from_controls_with_dispersion(controls)?;
    method.apply_collinear_reference(controls)?;
    let config = super::vibrational::response_config(controls)?;
    let directions: Vec<[f64; 3]> = controls
        .opt_rows("directions")?
        .unwrap_or_else(|| {
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ]
        })
        .iter()
        .map(|r| {
            if r.len() == 3 {
                Ok([r[0], r[1], r[2]])
            } else {
                Err(DftError::InvalidInput(
                    "`directions` must be Cartesian 3-vectors".into(),
                ))
            }
        })
        .collect::<Result<_>>()?;
    let cutoff = controls.f64("mode_cutoff_cm1", 1.0)?;
    controls.finish(feature)?;
    let dispersion = method.dispersion_model_for(&system.numbers)?;
    let setup = prepare_periodic(system, &method)?;
    let scf = run_reference(&setup)?;
    let mut d = crate::periodic::dielectric::dielectric_tensors(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
        &config,
        true,
    )?;
    let force_constants = d.force_constants.as_mut().ok_or_else(|| {
        DftError::LinearAlgebra("polar response did not build Gamma force constants".into())
    })?;
    let dispersion_energy = add_gamma_dispersion(&setup.cell, &dispersion, force_constants)?;
    let acoustic_residual = force_constants.acoustic_residual;
    let raw_asymmetry = force_constants.raw_asymmetry;
    let masses: Vec<f64> = setup
        .cell
        .molecule()
        .atoms
        .iter()
        .map(|a| crate::constants::atomic_mass_au(a.z))
        .collect();
    let volume = setup.cell.lattice().volume().abs();
    let polar = crate::periodic::dielectric::polar_modes(&d, &masses, volume, &directions, cutoff)?;
    let mut report = reference_report(
        &setup,
        &scf,
        "polar Gamma-point lattice dynamics from DFPT Born charges, the \
            electronic dielectric tensor and analytic force constants: transverse \
            modes, infrared intensities, the static dielectric tensor and LO \
            frequencies of the non-analytic term",
    )
    .with("epsilon_infinity", d.epsilon_infinity)
    .with("epsilon_static", polar.epsilon_static)
    .with("born_charges", d.born_charges.clone())
    .with("born_sum_rule", d.born_sum_rule)
    .with("frequencies_cm1", polar.frequencies_cm1.clone())
    .with("mode_effective_charges", polar.mode_charges.clone())
    .with("ir_intensities_km_mol", polar.ir_intensities_km_mol.clone())
    .with("lo_directions", directions.clone())
    .with("lo_frequencies_cm1", polar.lo_frequencies_cm1.clone())
    .with("excluded_modes", polar.excluded_modes)
    .with("acoustic_residual", acoustic_residual)
    .with("raw_asymmetry", raw_asymmetry)
    .with("fock_image_residual", d.fock_image_residual)
    .with("field_solves", response_solve_records(&d.field_solves))
    .with(
        "displacement_solves",
        response_solve_records(&d.displacement_solves),
    )
    .with("provenance", "analytic")
    .with(
        "units",
        units(&[
            ("epsilon_infinity", "dimensionless"),
            ("acoustic_residual", "hartree/bohr^2"),
            ("raw_asymmetry", "hartree/bohr^2"),
            ("fock_image_residual", "hartree"),
            ("epsilon_static", "dimensionless (relaxed ions)"),
            ("born_charges", "elementary charges"),
            ("born_sum_rule", "elementary charges (sum over atoms)"),
            (
                "frequencies_cm1",
                "cm^-1 (transverse; negative = imaginary)",
            ),
            ("mode_effective_charges", "e/sqrt(amu)"),
            ("ir_intensities_km_mol", "km/mol (974.8801 |Z~|^2)"),
            (
                "lo_frequencies_cm1",
                "cm^-1 per direction, with the non-analytic term",
            ),
        ]),
    );
    if let Some(energy) = dispersion_energy {
        include_dispersion(&mut report, &method, energy);
    }
    Ok(report)
}

/// Natural bond orbital analysis of a periodic reference on the
/// Born-von Karman supercell of its k mesh (3D finite-k SCF, or the
/// dimensional SCF for open axes).
fn periodic_natural_bond(system: &System, controls: &Controls) -> Result<Value> {
    use crate::properties::natural_bond::periodic::periodic_natural_bond as analyse;
    let (spin, options) = super::natural_bond::read_options(controls)?;
    let method = PeriodicMethod::from_controls(controls)?;
    let rows = system.cell.ok_or_else(|| {
        DftError::InvalidInput("periodic-natural-bond needs a cell (lattice rows, Bohr)".into())
    })?;
    let lattice = rows.map(|r| Vec3::new(r[0], r[1], r[2]));
    let model = "natural bond orbital analysis of the Born-von Karman supercell of the k mesh: \
        real supercell overlap, spin densities and Fock matrices unfolded from the Bloch \
        matrices; natural atomic orbitals, natural populations, natural Lewis structure, \
        hybrids and second-order donor-acceptor energies per cell, bonds to periodic images \
        labelled by cell offsets";
    if system.pbc == [true; 3] {
        controls.finish("periodic-natural-bond")?;
        let setup = prepare_periodic(system, &method)?;
        let scf = run_reference(&setup)?;
        let result = analyse(
            &setup.cell.molecule,
            &setup.cell.basis,
            lattice,
            setup.options.k_mesh,
            &scf,
            spin,
            &options,
        )?;
        let report = reference_report(&setup, &scf, model);
        Ok(super::natural_bond::periodic_report(
            report,
            &result,
            &system.numbers,
            system.pbc,
        ))
    } else {
        let dimensional = dimensional_scf_options(controls, &method, system.pbc)?;
        controls.finish("periodic-natural-bond")?;
        let geometry = dimensional_geometry_of(system)?;
        let mol = super::method::molecule_from(
            &system.numbers,
            &system.positions,
            method.charge,
            method.multiplicity,
        )?;
        let set = super::method::build_basis(&method.basis, method.ecp.as_deref())?;
        let basis = crate::basis::MolBasis::build(&mol, &set)?;
        let functional = resolve(&FunctionalSpec::Combined(method.xc.clone()))?;
        let scf = crate::periodic::scf_boundary::run_dimensional_scf(
            &mol,
            &basis,
            &geometry,
            &functional,
            &dimensional,
        )?
        .reference;
        let result = analyse(
            &mol,
            &basis,
            lattice,
            dimensional.scf.k_mesh,
            &scf,
            spin,
            &options,
        )?;
        let report = Value::object()
            .with("model", model)
            .with("functional", functional.description.as_str())
            .with("energy", scf.energy)
            .with("converged", scf.converged)
            .with("scf_iterations", scf.iterations)
            .with("pbc", system.pbc.to_vec());
        Ok(super::natural_bond::periodic_report(
            report,
            &result,
            &system.numbers,
            system.pbc,
        ))
    }
}

/// Proper piezoelectric tensor of an insulating reference: the clamped-ion
/// strain derivative of the polarization, and (`relaxed_ion`, default true)
/// the relaxed-ion tensor with the Born charges and the internal relaxation.
fn piezoelectric_feature(system: &System, controls: &Controls) -> Result<Value> {
    let mut method = PeriodicMethod::from_controls_with_dispersion(controls)?;
    method.apply_collinear_reference(controls)?;
    let config = super::vibrational::response_config(controls)?;
    let relax_ions = controls.bool("relaxed_ion", true)?;
    controls.finish("piezoelectric-tensor")?;
    let dispersion = method.dispersion_model_for(&system.numbers)?;
    let setup = prepare_periodic(system, &method)?;
    let scf = run_reference(&setup)?;
    let p = crate::periodic::dielectric::piezoelectric_tensor_with_dispersion(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
        &config,
        relax_ions,
        dispersion.as_ref(),
    )?;
    const PAIRS: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)];
    // 1 e/bohr^2 in C/m^2.
    let si = crate::constants::ELEMENTARY_CHARGE_C
        / (crate::constants::BOHR_TO_METER * crate::constants::BOHR_TO_METER);
    let voigt = |tensor: &[[f64; 9]; 3], scale: f64| -> Vec<Vec<f64>> {
        tensor
            .iter()
            .map(|row| {
                PAIRS
                    .iter()
                    .map(|&(a, b)| 0.5 * scale * (row[3 * a + b] + row[3 * b + a]))
                    .collect()
            })
            .collect()
    };
    let mut asymmetry = 0.0_f64;
    for row in p.clamped_ion.iter().chain(p.relaxed_ion.iter().flatten()) {
        for a in 0..3 {
            for b in 0..3 {
                asymmetry = asymmetry.max((row[3 * a + b] - row[3 * b + a]).abs());
            }
        }
    }
    let mut report = reference_report(
        &setup,
        &scf,
        "proper piezoelectric tensor from the strain response of the Berry connection (clamped \
         ions) and the Born charges with the internal relaxation (relaxed ions)",
    )
    .with("clamped_ion", voigt(&p.clamped_ion, 1.0))
    .with("clamped_ion_si", voigt(&p.clamped_ion, si))
    .with("clamped_ion_full", p.clamped_ion)
    .with("strain_index_asymmetry", asymmetry)
    .with("born_charges", p.born_charges.clone())
    .with("fock_image_residual", p.fock_image_residual)
    .with("strain_solves", response_solve_records(&p.strain_solves))
    .with("provenance", "analytic");
    if let Some(relaxed) = &p.relaxed_ion {
        report.set("relaxed_ion", voigt(relaxed, 1.0));
        report.set("relaxed_ion_si", voigt(relaxed, si));
        report.set("relaxed_ion_full", *relaxed);
    }
    report.set(
        "units",
        units(&[
            (
                "clamped_ion",
                "e/bohr^2, [polarization axis][Voigt strain xx, yy, zz, yz, xz, xy], proper tensor",
            ),
            ("relaxed_ion", "e/bohr^2"),
            ("clamped_ion_si", "C/m^2"),
            ("relaxed_ion_si", "C/m^2"),
            (
                "clamped_ion_full",
                "e/bohr^2, [polarization axis][3a + b] for the strain component eps_ab",
            ),
            ("relaxed_ion_full", "e/bohr^2"),
            ("strain_index_asymmetry", "e/bohr^2"),
            (
                "born_charges",
                "elementary charges, [atom][a][b] = V dP_a/du_b",
            ),
            ("fock_image_residual", "hartree"),
        ]),
    );
    if let Some(energy) = p.dispersion_energy {
        include_dispersion(&mut report, &method, energy);
    } else if let Some(correction) = cell_dispersion(&setup.cell, &dispersion)? {
        include_dispersion(&mut report, &method, correction.energy);
    }
    Ok(report)
}

/// Static electronic dielectric tensor, and Born effective charges when
/// `born`, by density-functional perturbation theory at fixed macroscopic
/// field.
fn dielectric_feature(system: &System, controls: &Controls, born: bool) -> Result<Value> {
    let feature = if born {
        "born-charges"
    } else {
        "dielectric-tensor"
    };
    let mut method = PeriodicMethod::from_controls_with_dispersion(controls)?;
    method.apply_collinear_reference(controls)?;
    let config = super::vibrational::response_config(controls)?;
    controls.finish(feature)?;
    let dispersion = method.dispersion_model_for(&system.numbers)?;
    let setup = prepare_periodic(system, &method)?;
    let scf = run_reference(&setup)?;
    let d = crate::periodic::dielectric::dielectric_tensors_with_force_constants(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
        &config,
        born,
        false,
    )?;
    let mut report = reference_report(
        &setup,
        &scf,
        if born {
            "Born effective charges and the electronic dielectric tensor from \
                density-functional perturbation theory (covariant k derivative, fixed \
                macroscopic field)"
        } else {
            "electronic (clamped-ion) dielectric tensor from density-functional \
                perturbation theory (covariant k derivative, fixed macroscopic field)"
        },
    )
    .with("epsilon_infinity", d.epsilon_infinity)
    .with("susceptibility", d.susceptibility)
    .with("raw_asymmetry", d.raw_asymmetry)
    .with("fock_image_residual", d.fock_image_residual)
    .with("field_solves", response_solve_records(&d.field_solves))
    .with("provenance", "analytic");
    if born {
        report.set("born_charges", d.born_charges.clone());
        report.set("born_sum_rule", d.born_sum_rule);
        report.set("imaginary_residual", d.imaginary_residual);
        report.set(
            "displacement_solves",
            response_solve_records(&d.displacement_solves),
        );
    }
    report.set(
        "units",
        units(&[
            ("epsilon_infinity", "dimensionless"),
            ("susceptibility", "dimensionless (Gaussian, dP/dE)"),
            ("raw_asymmetry", "dimensionless"),
            (
                "born_charges",
                "elementary charges, [atom][a][b] = V dP_a/du_b",
            ),
            ("born_sum_rule", "elementary charges (sum over atoms)"),
            ("fock_image_residual", "hartree"),
        ]),
    );
    if let Some(correction) = cell_dispersion(&setup.cell, &dispersion)? {
        include_dispersion(&mut report, &method, correction.energy);
    }
    Ok(report)
}

/// Third-order Birch-Murnaghan equation of state from SCF energies of
/// isotropically scaled cells (ions at fixed fractional coordinates or
/// relaxed in every cell), with the analytic pressure of every cell as an
/// independent check of the fitted `p = -dE/dV`.
fn equation_of_state(system: &System, controls: &Controls) -> Result<Value> {
    use crate::periodic::optimizer::{optimize_fixed_cell, PeriodicOptOptions};
    let method = PeriodicMethod::from_controls(controls)?;
    let scales = controls
        .opt_f64_list("scales")?
        .unwrap_or_else(|| vec![0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03]);
    if scales.len() < 4 || scales.iter().any(|s| *s <= 0.0) {
        return Err(DftError::InvalidInput(
            "`scales` needs at least four positive linear scale factors".into(),
        ));
    }
    let relax = controls.bool("relax_ions", false)?;
    let defaults = PeriodicOptOptions::default();
    let opt = PeriodicOptOptions {
        max_steps: controls.usize("max_steps", defaults.max_steps)?,
        gtol: controls.f64("gtol", 1e-5)?,
        step_max: controls.f64("step_max", defaults.step_max)?,
        history: controls.usize("history", defaults.history)?,
        trajectory: None,
        dispersion: None,
    };
    controls.finish("equation-of-state")?;
    let rows = system.cell.ok_or_else(|| {
        DftError::InvalidInput("a periodic calculation needs a cell (lattice rows, Bohr)".into())
    })?;
    let set = super::method::build_basis(&method.basis, method.ecp.as_deref())?;
    let mut volumes = Vec::with_capacity(scales.len());
    let mut energies = Vec::with_capacity(scales.len());
    let mut pressures = Vec::with_capacity(scales.len());
    let mut converged = true;
    let mut reference: Option<(PeriodicSetup, PeriodicScfResult)> = None;
    for &scale in &scales {
        let scaled = System {
            numbers: system.numbers.clone(),
            positions: system
                .positions
                .iter()
                .map(|p| p.map(|v| v * scale))
                .collect(),
            cell: Some(rows.map(|r| r.map(|v| v * scale))),
            pbc: system.pbc,
        };
        let setup = prepare_periodic(&scaled, &method)?;
        let (volume, stress, scf) = if relax {
            let r = optimize_fixed_cell(
                setup.cell.molecule(),
                setup.cell.lattice(),
                &set,
                method.overlap_cutoff,
                &setup.functional,
                &setup.options,
                &opt,
            )?;
            converged &= r.converged;
            let stress = crate::periodic::stress::analytic_stress(
                &r.cell,
                &setup.functional,
                &r.scf,
                &setup.options,
            )?;
            (r.cell.lattice().volume().abs(), stress, r.scf)
        } else {
            let scf = run_reference(&setup)?;
            let stress = crate::periodic::stress::analytic_stress(
                &setup.cell,
                &setup.functional,
                &scf,
                &setup.options,
            )?;
            (setup.cell.lattice().volume().abs(), stress, scf)
        };
        converged &= scf.converged;
        volumes.push(volume);
        energies.push(scf.energy);
        pressures.push(-(stress.stress[0][0] + stress.stress[1][1] + stress.stress[2][2]) / 3.0);
        if (scale - 1.0).abs() < 1e-12 || reference.is_none() {
            reference = Some((setup, scf));
        }
    }
    let fit = crate::periodic::eos::fit_birch_murnaghan(&volumes, &energies)?;
    let fitted_pressure: Vec<f64> = volumes.iter().map(|v| fit.pressure(*v)).collect();
    let pressure_gap = pressures
        .iter()
        .zip(&fitted_pressure)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let gpa = crate::constants::HARTREE_PER_BOHR3_TO_GPA;
    let angstrom3 = crate::constants::BOHR_TO_ANGSTROM.powi(3);
    let (setup, scf) = reference.expect("at least four volumes");
    let mut report = reference_report(
        &setup,
        &scf,
        if relax {
            "third-order Birch-Murnaghan equation of state of isotropically scaled \
                cells, ions relaxed in every cell"
        } else {
            "third-order Birch-Murnaghan equation of state of isotropically scaled \
                cells at fixed fractional coordinates"
        },
    )
    .with("scales", scales.clone())
    .with("volumes", volumes.clone())
    .with("energies", energies.clone())
    .with(
        "analytic_pressures_gpa",
        pressures.iter().map(|p| p * gpa).collect::<Vec<_>>(),
    )
    .with(
        "fitted_pressures_gpa",
        fitted_pressure.iter().map(|p| p * gpa).collect::<Vec<_>>(),
    )
    .with("pressure_fit_gap_gpa", pressure_gap * gpa)
    .with(
        "fitted_bulk_moduli_gpa",
        volumes
            .iter()
            .map(|v| fit.bulk_modulus(*v) * gpa)
            .collect::<Vec<_>>(),
    )
    .with("v0", fit.v0)
    .with("v0_angstrom3", fit.v0 * angstrom3)
    .with("e0", fit.e0)
    .with("b0_gpa", fit.b0 * gpa)
    .with("b0_prime", fit.b0_prime)
    .with(
        "equilibrium_scale",
        (fit.v0
            / volumes[scales
                .iter()
                .position(|s| (*s - 1.0).abs() < 1e-12)
                .unwrap_or(0)])
        .cbrt(),
    )
    .with("residual_rms", fit.residual_rms)
    .with("all_converged", converged)
    .with("relax_ions", relax)
    .with(
        "provenance",
        "least-squares Birch-Murnaghan fit of SCF energies; pressures from the \
        analytic stress",
    );
    if !converged {
        report.set(
            "warnings",
            vec![
                "WARNING [equation-of-state] an SCF or ionic relaxation did not converge"
                    .to_string(),
            ],
        );
    }
    report.set(
        "units",
        units(&[
            ("volumes", "bohr^3 per cell"),
            ("energies", "hartree per cell"),
            ("v0", "bohr^3 per cell"),
            ("v0_angstrom3", "angstrom^3 per cell"),
            ("e0", "hartree per cell"),
            ("b0_gpa", "GPa"),
            ("analytic_pressures_gpa", "GPa (-tr(sigma)/3)"),
            ("fitted_pressures_gpa", "GPa (-dE/dV of the fit)"),
            (
                "fitted_bulk_moduli_gpa",
                "GPa (V d2E/dV2 of the fit at every volume)",
            ),
            ("pressure_fit_gap_gpa", "GPa"),
            ("residual_rms", "hartree per cell"),
            (
                "equilibrium_scale",
                "linear scale of the input cell at V0 (relative to scale 1, or the first scale)",
            ),
        ]),
    );
    Ok(report)
}

/// Continuum material card: the analytic elastic tensor with its compliance,
/// orthotropic engineering constants (optionally in rotated material axes),
/// Hill moduli, wave speeds, density, the reference stress, conventions and
/// residuals, as JSON or CSV.
fn material_card(system: &System, controls: &Controls) -> Result<Value> {
    let rotation = controls.opt_rows("rotation")?;
    let path = controls.opt_str("card_path")?;
    let format = controls.opt_str("card_format")?;
    let (setup, scf, ec, tensor, selection) =
        selected_elastic_tensor(system, controls, "material-card", &|_| Ok(()))?;
    let rotation: Option<[[f64; 3]; 3]> = match rotation {
        None => None,
        Some(rows) if rows.len() == 3 && rows.iter().all(|r| r.len() == 3) => {
            let r: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| rows[i][j]));
            let mut worst = 0.0_f64;
            for i in 0..3 {
                for j in 0..3 {
                    let dot: f64 = (0..3).map(|k| r[i][k] * r[j][k]).sum();
                    worst = worst.max((dot - if i == j { 1.0 } else { 0.0 }).abs());
                }
            }
            if worst > 1e-8 {
                return Err(DftError::InvalidInput(
                    "`rotation` must be orthonormal (rows are the material axes)".into(),
                ));
            }
            Some(r)
        }
        Some(_) => {
            return Err(DftError::InvalidInput(
                "`rotation` must be a 3x3 matrix".into(),
            ))
        }
    };
    let axes_tensor = match &rotation {
        Some(r) => crate::geomech::rotate_stiffness(&tensor, r),
        None => tensor,
    };
    let compliance = crate::geomech::compliance(&axes_tensor)?;
    let (young, poisson, shear) = crate::geomech::engineering_constants(&compliance);
    let moduli = crate::periodic::elastic::elastic_moduli(&axes_tensor)?;
    let density = cell_density(&setup.cell);
    let vp = ((moduli.hill_bulk + 4.0 * moduli.hill_shear / 3.0) * 1e9 / density).sqrt();
    let vs = (moduli.hill_shear * 1e9 / density).sqrt();
    let gpa = crate::constants::HARTREE_PER_BOHR3_TO_GPA;
    let stress = ec.stress.map(|row| row.map(|v| v * gpa));
    let mut csv = String::from("quantity,i,j,value,unit\n");
    for i in 0..6 {
        for j in 0..6 {
            csv.push_str(&format!(
                "C,{},{},{:.10e},GPa\n",
                i + 1,
                j + 1,
                axes_tensor[i][j]
            ));
        }
    }
    for i in 0..6 {
        for j in 0..6 {
            csv.push_str(&format!(
                "S,{},{},{:.10e},1/GPa\n",
                i + 1,
                j + 1,
                compliance[i][j]
            ));
        }
    }
    for i in 0..3 {
        csv.push_str(&format!("E,{},,{:.10e},GPa\n", i + 1, young[i]));
        csv.push_str(&format!("G,{},,{:.10e},GPa\n", [23, 13, 12][i], shear[i]));
        for j in 0..3 {
            if i != j {
                csv.push_str(&format!(
                    "nu,{},{},{:.10e},1\n",
                    i + 1,
                    j + 1,
                    poisson[i][j]
                ));
            }
        }
    }
    for (name, value, unit) in [
        ("K_hill", moduli.hill_bulk, "GPa"),
        ("G_hill", moduli.hill_shear, "GPa"),
        ("E_hill", moduli.young, "GPa"),
        ("nu_hill", moduli.poisson, "1"),
        ("density", density, "kg/m^3"),
        ("vp", vp, "m/s"),
        ("vs", vs, "m/s"),
    ] {
        csv.push_str(&format!("{name},,,{value:.10e},{unit}\n"));
    }
    for i in 0..3 {
        for j in 0..3 {
            csv.push_str(&format!(
                "stress,{},{},{:.10e},GPa\n",
                i + 1,
                j + 1,
                stress[i][j]
            ));
        }
    }
    let mut report = reference_report(
        &setup,
        &scf,
        "continuum material card from the analytic elastic tensor: stiffness, \
            compliance, orthotropic engineering constants, Hill moduli, wave speeds \
            and density",
    )
    .with("selection", selection)
    .with(
        "conventions",
        "Voigt order xx, yy, zz, yz, xz, xy; engineering shear strains in the \
            compliance; isothermal static-lattice constants at the reference \
            stress; rotation rows are the material axes in Cartesian coordinates",
    )
    .with("stiffness_gpa", axes_tensor)
    .with("compliance_per_gpa", compliance)
    .with("young_gpa", young.to_vec())
    .with("poisson_ratios", poisson)
    .with("shear_moduli_gpa", shear.to_vec())
    .with("hill_bulk_gpa", moduli.hill_bulk)
    .with("hill_shear_gpa", moduli.hill_shear)
    .with("density_kg_m3", density)
    .with("vp_m_s", vp)
    .with("vs_m_s", vs)
    .with("reference_stress_gpa", stress)
    .with("tensor_eigenvalues_gpa", moduli.eigenvalues.clone())
    .with("raw_asymmetry", ec.raw_asymmetry)
    .with("internal_strain_asymmetry", ec.internal_strain_asymmetry)
    .with("rotation_residual", ec.rotation_residual)
    .with("csv", csv.as_str())
    .with("provenance", "analytic elastic tensor");
    if let Some(r) = rotation {
        report.set("rotation", r);
    }
    if let Some(path) = path {
        let kind = format.unwrap_or_else(|| {
            if path.ends_with(".csv") {
                "csv".into()
            } else {
                "json".into()
            }
        });
        let text = match kind.as_str() {
            "csv" => csv.clone(),
            "json" => report.to_json_pretty(),
            other => {
                return Err(DftError::InvalidInput(format!(
                    "`card_format` must be json or csv, got {other}"
                )))
            }
        };
        std::fs::write(&path, text)
            .map_err(|e| DftError::InvalidInput(format!("cannot write {path}: {e}")))?;
        report.set("card_path", path.as_str());
    }
    report.set(
        "units",
        units(&[
            ("stiffness_gpa", "GPa (Voigt)"),
            ("compliance_per_gpa", "1/GPa (engineering shear strains)"),
            ("young_gpa", "GPa (E1, E2, E3)"),
            ("poisson_ratios", "nu_ij = -S_ji/S_ii"),
            ("shear_moduli_gpa", "GPa (G23, G13, G12)"),
            ("density_kg_m3", "kg/m^3"),
            ("vp_m_s", "m/s"),
            ("vs_m_s", "m/s"),
            ("reference_stress_gpa", "GPa"),
            ("raw_asymmetry", "hartree per cell"),
            ("internal_strain_asymmetry", "hartree/bohr per cell"),
            ("rotation_residual", "hartree per cell"),
        ]),
    );
    Ok(report)
}

/// Mass density of a cell (kg/m^3) from isotope-averaged atomic masses.
fn cell_density(cell: &Cell) -> f64 {
    let mass_kg: f64 = cell
        .molecule()
        .atoms
        .iter()
        .map(|a| crate::constants::atomic_mass_au(a.z) * crate::constants::ELECTRON_MASS_KG)
        .sum();
    mass_kg / (cell.lattice().volume().abs() * crate::constants::BOHR_TO_METER.powi(3))
}

/// The analytic elastic tensor selected by the `ions` (relaxed, clamped) and
/// `coefficients` (birch, lagrangian) controls, in GPa, with the reference.
fn selected_elastic_tensor(
    system: &System,
    controls: &Controls,
    feature: &str,
    extra: &dyn Fn(&Controls) -> Result<()>,
) -> Result<(
    PeriodicSetup,
    PeriodicScfResult,
    crate::periodic::elastic::ElasticConstants,
    [[f64; 6]; 6],
    String,
)> {
    let method = PeriodicMethod::from_controls(controls)?;
    let config = super::vibrational::response_config(controls)?;
    let ions = controls.str("ions", "relaxed")?;
    let coefficients = controls.str("coefficients", "birch")?;
    extra(controls)?;
    controls.finish(feature)?;
    let setup = prepare_periodic(system, &method)?;
    let scf = run_reference(&setup)?;
    let ec = crate::periodic::elastic::elastic_constants(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
        &config,
    )?;
    let tensor = match (ions.as_str(), coefficients.as_str()) {
        ("relaxed", "birch") => ec.relaxed_ion_birch,
        ("relaxed", "lagrangian") => ec.relaxed_ion,
        ("clamped", "birch") => ec.clamped_ion_birch,
        ("clamped", "lagrangian") => ec.clamped_ion,
        _ => {
            return Err(DftError::InvalidInput(format!(
                "`ions` must be relaxed or clamped and `coefficients` birch or \
                    lagrangian (got {ions}, {coefficients})"
            )))
        }
    };
    let gpa = crate::constants::HARTREE_PER_BOHR3_TO_GPA;
    Ok((
        setup,
        scf,
        ec,
        tensor.map(|row| row.map(|v| v * gpa)),
        format!("{ions} ions, {coefficients} coefficients"),
    ))
}

/// Polycrystalline Voigt, Reuss and Hill moduli, isotropic wave speeds and
/// the Debye temperature from the analytic elastic tensor.
fn elastic_moduli_feature(system: &System, controls: &Controls) -> Result<Value> {
    let (setup, scf, ec, tensor, selection) =
        selected_elastic_tensor(system, controls, "elastic-moduli", &|_| Ok(()))?;
    let m = crate::periodic::elastic::elastic_moduli(&tensor)?;
    let density = cell_density(&setup.cell);
    let vp = ((m.hill_bulk + 4.0 * m.hill_shear / 3.0) * 1e9 / density).sqrt();
    let vs = (m.hill_shear * 1e9 / density).sqrt();
    let mean = (((2.0 / vs.powi(3)) + 1.0 / vp.powi(3)) / 3.0).powf(-1.0 / 3.0);
    let number_density = setup.cell.molecule().len() as f64
        / (setup.cell.lattice().volume().abs() * crate::constants::BOHR_TO_METER.powi(3));
    let debye = 1.054571817e-34 / 1.380649e-23
        * (6.0 * std::f64::consts::PI.powi(2) * number_density).cbrt()
        * mean;
    let report = reference_report(
        &setup,
        &scf,
        "polycrystalline Voigt, Reuss and Hill elastic moduli of the analytic \
            elastic tensor, isotropic wave speeds and Debye temperature",
    )
    .with("selection", selection)
    .with("elastic_tensor_gpa", tensor)
    .with("compliance_per_gpa", m.compliance)
    .with("tensor_eigenvalues_gpa", m.eigenvalues.clone())
    .with("born_stable", m.eigenvalues.iter().all(|v| *v > 0.0))
    .with("voigt_bulk_gpa", m.voigt_bulk)
    .with("reuss_bulk_gpa", m.reuss_bulk)
    .with("hill_bulk_gpa", m.hill_bulk)
    .with("voigt_shear_gpa", m.voigt_shear)
    .with("reuss_shear_gpa", m.reuss_shear)
    .with("hill_shear_gpa", m.hill_shear)
    .with("young_gpa", m.young)
    .with("poisson_ratio", m.poisson)
    .with("universal_anisotropy", m.universal_anisotropy)
    .with("density_kg_m3", density)
    .with("p_wave_velocity_m_s", vp)
    .with("s_wave_velocity_m_s", vs)
    .with("debye_temperature_k", debye)
    .with(
        "stress_gpa",
        ec.stress
            .map(|row| row.map(|v| v * crate::constants::HARTREE_PER_BOHR3_TO_GPA)),
    )
    .with("provenance", "analytic elastic tensor")
    .with(
        "units",
        units(&[
            ("elastic_tensor_gpa", "GPa (Voigt xx, yy, zz, yz, xz, xy)"),
            ("compliance_per_gpa", "1/GPa (engineering shear strains)"),
            ("density_kg_m3", "kg/m^3"),
            ("p_wave_velocity_m_s", "m/s (isotropic Hill aggregate)"),
            ("s_wave_velocity_m_s", "m/s (isotropic Hill aggregate)"),
            (
                "debye_temperature_k",
                "K (from the mean Hill sound velocity)",
            ),
        ]),
    );
    Ok(report)
}

/// Christoffel phase velocities and polarizations of the analytic elastic
/// tensor along given directions.
fn christoffel_feature(system: &System, controls: &Controls) -> Result<Value> {
    let directions: std::cell::RefCell<Vec<[f64; 3]>> = std::cell::RefCell::new(Vec::new());
    let (setup, scf, _, tensor, selection) =
        selected_elastic_tensor(system, controls, "christoffel", &|c| {
            let rows = c.opt_rows("directions")?.unwrap_or_else(|| {
                vec![
                    vec![1.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 1.0],
                    vec![1.0, 1.0, 0.0],
                    vec![1.0, 0.0, 1.0],
                    vec![0.0, 1.0, 1.0],
                    vec![1.0, 1.0, 1.0],
                ]
            });
            if rows.iter().any(|r| r.len() != 3) {
                return Err(DftError::InvalidInput(
                    "`directions` must be Cartesian 3-vectors".into(),
                ));
            }
            *directions.borrow_mut() = rows.iter().map(|r| [r[0], r[1], r[2]]).collect();
            Ok(())
        })?;
    let density = cell_density(&setup.cell);
    let mut velocities = Vec::new();
    let mut polarizations = Vec::new();
    for d in directions.borrow().iter() {
        let (v, p) = crate::periodic::elastic::christoffel(&tensor, density, *d)?;
        velocities.push(v.to_vec());
        polarizations.push(p);
    }
    let report = reference_report(
        &setup,
        &scf,
        "elastic plane-wave phase velocities from the Christoffel equation of \
            the analytic elastic tensor",
    )
    .with("selection", selection)
    .with("elastic_tensor_gpa", tensor)
    .with("density_kg_m3", density)
    .with("directions", directions.borrow().clone())
    .with("velocities_m_s", velocities)
    .with("polarizations", polarizations)
    .with("provenance", "analytic elastic tensor")
    .with(
        "units",
        units(&[
            ("elastic_tensor_gpa", "GPa"),
            ("density_kg_m3", "kg/m^3"),
            ("directions", "Cartesian (normalized internally)"),
            (
                "velocities_m_s",
                "m/s per direction, ascending (negative = unstable)",
            ),
            (
                "polarizations",
                "unit polarization vectors as columns, per direction",
            ),
        ]),
    );
    Ok(report)
}

/// Validation route for the elastic constants: central differences of the
/// analytic strain derivative over the six symmetric strains, with the ions
/// clamped (fixed fractional coordinates) or relaxed in every strained cell.
/// For a symmetric strain direction `E` the difference measures the
/// composed strain, `(W E) + S E`, so the Lagrangian constants are
/// `sym[FD - S E - E tau] / V`.
fn elastic_fd(system: &System, controls: &Controls) -> Result<Value> {
    use crate::periodic::optimizer::{optimize_fixed_cell, PeriodicOptOptions};
    let method = PeriodicMethod::from_controls(controls)?;
    let steps = controls
        .opt_f64_list("steps")?
        .unwrap_or_else(|| vec![0.01, 0.005]);
    if steps.is_empty() || steps.iter().any(|h| *h <= 0.0) {
        return Err(DftError::InvalidInput(
            "`steps` must hold positive strain steps".into(),
        ));
    }
    let relax = controls.bool("relax_ions", false)?;
    let defaults = PeriodicOptOptions::default();
    let opt = PeriodicOptOptions {
        max_steps: controls.usize("max_steps", defaults.max_steps)?,
        gtol: controls.f64("gtol", 1e-6)?,
        step_max: controls.f64("step_max", defaults.step_max)?,
        history: controls.usize("history", defaults.history)?,
        trajectory: None,
        dispersion: None,
    };
    controls.finish("elastic-fd")?;
    let setup = prepare_periodic(system, &method)?;
    let scf = run_reference(&setup)?;
    let reference = crate::periodic::stress::analytic_stress(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
    )?;
    let reference_forces = crate::periodic::forces::analytic_forces(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
    )?;
    let reference_force = reference_forces
        .total
        .iter()
        .map(|f| f.norm())
        .fold(0.0_f64, f64::max);
    let set = super::method::build_basis(&method.basis, method.ecp.as_deref())?;
    let rows = system.cell.ok_or_else(|| {
        DftError::InvalidInput("a periodic calculation needs a cell (lattice rows, Bohr)".into())
    })?;
    let mut residual_force = 0.0_f64;
    let mut all_converged = scf.converged;
    let mut strained_derivative = |e: &[[f64; 3]; 3]| -> Result<[[f64; 3]; 3]> {
        let map = |v: [f64; 3]| -> [f64; 3] {
            std::array::from_fn(|a| v[a] + (0..3).map(|b| e[a][b] * v[b]).sum::<f64>())
        };
        let strained = System {
            numbers: system.numbers.clone(),
            positions: system.positions.iter().map(|p| map(*p)).collect(),
            cell: Some(rows.map(map)),
            pbc: system.pbc,
        };
        let s = prepare_periodic(&strained, &method)?;
        if relax {
            let r = optimize_fixed_cell(
                s.cell.molecule(),
                s.cell.lattice(),
                &set,
                method.overlap_cutoff,
                &s.functional,
                &s.options,
                &opt,
            )?;
            let forces = crate::periodic::forces::analytic_forces(
                &r.cell,
                &s.functional,
                &r.scf,
                &s.options,
            )?;
            residual_force = forces
                .total
                .iter()
                .map(|f| f.norm())
                .fold(residual_force, f64::max);
            all_converged &= r.converged && r.scf.converged;
            Ok(crate::periodic::stress::analytic_stress(
                &r.cell,
                &s.functional,
                &r.scf,
                &s.options,
            )?
            .strain_derivative)
        } else {
            let scf = run_periodic_scf(&s.cell, &s.functional, &s.options)?;
            all_converged &= scf.converged;
            Ok(
                crate::periodic::stress::analytic_stress(&s.cell, &s.functional, &scf, &s.options)?
                    .strain_derivative,
            )
        }
    };
    let s0 = reference.strain_derivative;
    let tau: [[f64; 3]; 3] =
        std::array::from_fn(|a| std::array::from_fn(|b| 0.5 * (s0[a][b] + s0[b][a])));
    let volume = setup.cell.lattice().volume().abs();
    let gpa = crate::constants::HARTREE_PER_BOHR3_TO_GPA;
    let voigt = crate::periodic::elastic::VOIGT;
    let mut tensors: Vec<[[f64; 6]; 6]> = Vec::with_capacity(steps.len());
    for &h in &steps {
        let mut c = [[0.0; 6]; 6];
        for (j, &(cc, dd)) in voigt.iter().enumerate() {
            let mut direction = [[0.0; 3]; 3];
            let weight = if cc == dd { 1.0 } else { 0.5 };
            direction[cc][dd] = weight;
            direction[dd][cc] = weight;
            let scaled = |f: f64| direction.map(|row| row.map(|v| v * f));
            let plus = strained_derivative(&scaled(h))?;
            let minus = strained_derivative(&scaled(-h))?;
            let column: [[f64; 3]; 3] = std::array::from_fn(|a| {
                std::array::from_fn(|b| {
                    let fd = (plus[a][b] - minus[a][b]) / (2.0 * h);
                    let composed: f64 = (0..3).map(|d| s0[a][d] * direction[d][b]).sum();
                    let initial: f64 = (0..3).map(|d| direction[a][d] * tau[d][b]).sum();
                    fd - composed - initial
                })
            });
            for (i, &(a, b)) in voigt.iter().enumerate() {
                c[i][j] = 0.5 * (column[a][b] + column[b][a]) / volume;
            }
        }
        tensors.push(c);
    }
    let to_gpa = |m: &[[f64; 6]; 6]| m.map(|row| row.map(|v| v * gpa));
    let mut report = reference_report(
        &setup,
        &scf,
        if relax {
            "validation only: Lagrangian elastic constants from central differences \
                of the analytic stress, ions relaxed in every strained cell"
        } else {
            "validation only: Lagrangian elastic constants from central differences \
                of the analytic stress at fixed fractional coordinates"
        },
    )
    .with("voigt_order", "xx, yy, zz, yz, xz, xy")
    .with("relax_ions", relax)
    .with("steps", steps.clone())
    .with(
        "lagrangian_gpa",
        tensors.iter().map(to_gpa).collect::<Vec<_>>(),
    )
    .with("strain_derivative", s0)
    .with("reference_max_force", reference_force)
    .with("all_converged", all_converged)
    .with(
        "provenance",
        "finite differences of the analytic stress (validation only)",
    );
    if steps.len() >= 2 {
        let n = steps.len();
        let r2 = (steps[n - 2] / steps[n - 1]).powi(2);
        let richardson: [[f64; 6]; 6] = std::array::from_fn(|i| {
            std::array::from_fn(|j| (r2 * tensors[n - 1][i][j] - tensors[n - 2][i][j]) / (r2 - 1.0))
        });
        report.set("richardson_gpa", to_gpa(&richardson));
    }
    if relax {
        report.set("max_residual_force", residual_force);
    }
    let mut warnings: Vec<String> = Vec::new();
    if !all_converged {
        warnings.push("WARNING [elastic-fd] an SCF or ionic relaxation did not converge".into());
    }
    if relax && reference_force > opt.gtol {
        warnings.push(format!(
            "WARNING [elastic-fd] the reference forces reach {reference_force:e} \
                hartree/bohr; relaxed-ion constants assume a relaxed reference"
        ));
    }
    report.set("warnings", warnings);
    report.set(
        "units",
        units(&[
            ("lagrangian_gpa", "GPa per step (Voigt 6x6)"),
            ("richardson_gpa", "GPa"),
            ("strain_derivative", "hartree per cell"),
            ("reference_max_force", "hartree/bohr"),
            ("max_residual_force", "hartree/bohr"),
        ]),
    );
    Ok(report)
}

/// Fixed-cell relaxation of the atoms with analytic finite-k forces.
fn opt_fixed_cell(system: &System, controls: &Controls) -> Result<Value> {
    use crate::periodic::optimizer::{optimize_fixed_cell, PeriodicOptOptions};
    let method = PeriodicMethod::from_controls_with_dispersion(controls)?;
    let dispersion_model = method.dispersion_model_for(&system.numbers)?;
    let defaults = PeriodicOptOptions::default();
    let options = PeriodicOptOptions {
        dispersion: dispersion_model,
        max_steps: controls.usize("max_steps", defaults.max_steps)?,
        gtol: controls.f64("gtol", defaults.gtol)?,
        step_max: controls.f64("step_max", defaults.step_max)?,
        history: controls.usize("history", defaults.history)?,
        trajectory: None,
    };
    controls.finish("opt-fixed-cell")?;
    let setup = prepare_periodic(system, &method)?;
    let set = super::method::build_basis(&method.basis, method.ecp.as_deref())?;
    let result = optimize_fixed_cell(
        setup.cell.molecule(),
        setup.cell.lattice(),
        &set,
        method.overlap_cutoff,
        &setup.functional,
        &setup.options,
        &options,
    )?;
    let mut forces = crate::periodic::forces::analytic_forces(
        &result.cell,
        &setup.functional,
        &result.scf,
        &setup.options,
    )?;
    let dispersion = cell_dispersion(&result.cell, &options.dispersion)?;
    if let Some(d) = &dispersion {
        for (t, f) in forces.total.iter_mut().zip(&d.forces) {
            *t += *f;
        }
    }
    let PeriodicSetup {
        functional,
        options: scf_options,
        ..
    } = setup;
    let final_setup = PeriodicSetup {
        method: method.clone(),
        cell: result.cell,
        functional,
        options: scf_options,
    };
    let mut report = reference_report(
        &final_setup,
        &result.scf,
        "fixed-cell relaxation of the atoms with analytic finite-k forces \
            (objective: energy, Mermin free energy at finite temperature)",
    )
    .with("free_energy", result.scf.free_energy)
    .with("objective", result.objective)
    .with("optimization_converged", result.converged)
    .with("iterations", result.iterations)
    .with("numbers", system.numbers.clone())
    .with(
        "positions",
        result
            .molecule
            .atoms
            .iter()
            .map(|a| a.position.to_array())
            .collect::<Vec<_>>(),
    )
    .with(
        "forces",
        forces
            .total
            .iter()
            .map(|f| f.to_array())
            .collect::<Vec<_>>(),
    )
    .with(
        "trajectory",
        result
            .trajectory
            .iter()
            .map(|frame| trajectory_frame(frame.step, frame.objective, frame.grad_max))
            .collect::<Vec<_>>(),
    );
    if let Some(d) = &dispersion {
        include_dispersion(&mut report, &method, d.energy);
    }
    if !result.converged {
        report.set(
            "warnings",
            vec![format!(
                "WARNING [opt-fixed-cell] not converged after {} steps (gtol {} hartree/bohr)",
                result.iterations, options.gtol
            )],
        );
    }
    report.set(
        "units",
        units(&[
            ("energy", "hartree per cell"),
            ("positions", "bohr"),
            ("forces", "hartree/bohr"),
            ("volume", "bohr^3"),
        ]),
    );
    Ok(report)
}

/// Variable-cell relaxation of atoms and lattice with analytic forces and
/// stress at an external hydrostatic pressure.
fn cell_opt(system: &System, controls: &Controls) -> Result<Value> {
    use crate::periodic::optimizer::{optimize_cell, CellOptOptions};
    let method = PeriodicMethod::from_controls_with_dispersion(controls)?;
    let dispersion_model = method.dispersion_model_for(&system.numbers)?;
    let defaults = CellOptOptions::default();
    let gpa = crate::constants::HARTREE_PER_BOHR3_TO_GPA;
    let options = CellOptOptions {
        dispersion: dispersion_model,
        max_steps: controls.usize("max_steps", defaults.max_steps)?,
        gtol: controls.f64("gtol", defaults.gtol)?,
        stress_tol: controls.f64("stress_tol", defaults.stress_tol)?,
        pressure: controls.f64("pressure_gpa", 0.0)? / gpa,
        step_max: controls.f64("step_max", defaults.step_max)?,
        history: controls.usize("history", defaults.history)?,
    };
    controls.finish("cell-opt")?;
    let setup = prepare_periodic(system, &method)?;
    let set = super::method::build_basis(&method.basis, method.ecp.as_deref())?;
    let result = optimize_cell(
        setup.cell.molecule(),
        setup.cell.lattice(),
        &set,
        method.overlap_cutoff,
        &setup.functional,
        &setup.options,
        &options,
    )?;
    let final_dispersion = cell_dispersion(&result.cell, &options.dispersion)?;
    let PeriodicSetup {
        functional,
        options: scf_options,
        ..
    } = setup;
    let final_setup = PeriodicSetup {
        method: method.clone(),
        cell: result.cell,
        functional,
        options: scf_options,
    };
    let mut report = reference_report(
        &final_setup,
        &result.scf,
        "variable-cell relaxation with analytic finite-k forces and stress \
            (objective: enthalpy E + pV, Mermin free energy at finite temperature)",
    )
    .with("free_energy", result.scf.free_energy)
    .with("enthalpy", result.enthalpy)
    .with("external_pressure", options.pressure)
    .with("external_pressure_gpa", options.pressure * gpa)
    .with("optimization_converged", result.converged)
    .with("iterations", result.iterations)
    .with("numbers", system.numbers.clone())
    .with("cell", result.lattice.a.map(|row| row.to_array()))
    .with(
        "positions",
        result
            .molecule
            .atoms
            .iter()
            .map(|a| a.position.to_array())
            .collect::<Vec<_>>(),
    )
    .with(
        "forces",
        result
            .forces
            .iter()
            .map(|f| f.to_array())
            .collect::<Vec<_>>(),
    )
    .with(
        "trajectory",
        result
            .trajectory
            .iter()
            .map(|frame| {
                Value::object()
                    .with("step", frame.step)
                    .with("enthalpy", frame.enthalpy)
                    .with("grad_max", frame.grad_max)
                    .with("stress_max", frame.stress_max)
                    .with("volume", frame.lattice.volume().abs())
            })
            .collect::<Vec<_>>(),
    );
    stress_fields(&mut report, &result.stress);
    if let Some(d) = &final_dispersion {
        include_dispersion(&mut report, &method, d.energy);
    }
    if !result.converged {
        report.set(
            "warnings",
            vec![format!(
                "WARNING [cell-opt] not converged after {} steps (gtol {} hartree/bohr, \
                    stress_tol {} hartree/bohr^3)",
                result.iterations, options.gtol, options.stress_tol
            )],
        );
    }
    let mut unit_pairs = vec![
        ("energy", "hartree per cell"),
        ("enthalpy", "hartree per cell"),
        ("external_pressure", "hartree/bohr^3"),
        ("cell", "bohr (lattice rows)"),
        ("positions", "bohr"),
        ("forces", "hartree/bohr"),
        ("volume", "bohr^3"),
    ];
    unit_pairs.extend(STRESS_UNITS);
    report.set("units", units(&unit_pairs));
    Ok(report)
}

/// Band energies of `fock` with the metric `overlap`, ascending.
pub fn band_energies(fock: &BlochMatrix, overlap: &BlochMatrix) -> Result<Vec<f64>> {
    let orth = Orthogonalizer::canonical(&overlap.real_embedding(), 1e-7)?;
    let (energies, _) = orth.solve_roothaan(&fock.real_embedding())?;
    Ok(energies
        .chunks_exact(2)
        .map(|p| 0.5 * (p[0] + p[1]))
        .collect())
}

fn gap_summary(energies: &[Vec<f64>], electrons_per_k: usize) -> Value {
    // Integer insulating occupation: the lowest `electrons_per_k` bands per
    // spin channel are filled at every k point.
    if electrons_per_k == 0 || energies.iter().any(|e| e.len() <= electrons_per_k) {
        return Value::Null;
    }
    let vbm = energies
        .iter()
        .enumerate()
        .map(|(k, e)| (e[electrons_per_k - 1], k))
        .fold((f64::NEG_INFINITY, 0), |a, b| if b.0 > a.0 { b } else { a });
    let cbm = energies
        .iter()
        .enumerate()
        .map(|(k, e)| (e[electrons_per_k], k))
        .fold((f64::INFINITY, 0), |a, b| if b.0 < a.0 { b } else { a });
    let direct = energies
        .iter()
        .map(|e| e[electrons_per_k] - e[electrons_per_k - 1])
        .fold(f64::INFINITY, f64::min);
    Value::object()
        .with("valence_maximum", vbm.0)
        .with("valence_maximum_k", vbm.1)
        .with("conduction_minimum", cbm.0)
        .with("conduction_minimum_k", cbm.1)
        .with("gap", cbm.0 - vbm.0)
        .with("direct_gap", direct)
        .with("indirect", vbm.1 != cbm.1)
}

/// Canonical complex orbitals of the actual converged Fock. The same
/// overlap cutoff and shared complex solver are used as in periodic DFPT;
/// occupations stay those of the reference, including thermal occupations.
fn orbital_export(scf: &PeriodicScfResult, spin: usize, cutoff: f64) -> Result<Value> {
    use crate::dfpt::orbitals::{band_density, generalized_orbitals};
    let orbitals = scf.fock[spin]
        .iter()
        .zip(&scf.overlap)
        .map(|(f, s)| generalized_orbitals(f, s, cutoff))
        .collect::<Result<Vec<_>>>()?;
    let mut energy_residual = Vec::new();
    let mut density_residual = Vec::new();
    for (k, orbital) in orbitals.iter().enumerate() {
        if orbital.energies.len() != scf.mo_energy[spin][k].len() {
            return Err(DftError::LinearAlgebra(
                "orbital export and reference have different overlap ranks".into(),
            ));
        }
        energy_residual.push(
            orbital
                .energies
                .iter()
                .zip(&scf.mo_energy[spin][k])
                .fold(0.0_f64, |r, (a, b)| r.max((a - b).abs())),
        );
        let density = band_density(orbital, &scf.occupations[spin][k])?;
        density_residual.push(
            density
                .re
                .max_abs_diff(&scf.density[spin][k].re)
                .max(density.im.max_abs_diff(&scf.density[spin][k].im)),
        );
    }
    Ok(Value::object()
        .with("coefficients_real", orbitals.iter().map(|o| o.coefficients.re.clone()).collect::<Vec<_>>())
        .with("coefficients_imaginary", orbitals.iter().map(|o| o.coefficients.im.clone()).collect::<Vec<_>>())
        .with("overlap_real", scf.overlap.iter().map(|m| m.re.clone()).collect::<Vec<_>>())
        .with("overlap_imaginary", scf.overlap.iter().map(|m| m.im.clone()).collect::<Vec<_>>())
        .with("fock_real", scf.fock[spin].iter().map(|m| m.re.clone()).collect::<Vec<_>>())
        .with("fock_imaginary", scf.fock[spin].iter().map(|m| m.im.clone()).collect::<Vec<_>>())
        .with("density_real", scf.density[spin].iter().map(|m| m.re.clone()).collect::<Vec<_>>())
        .with("density_imaginary", scf.density[spin].iter().map(|m| m.im.clone()).collect::<Vec<_>>())
        .with("metric_defect", orbitals.iter().map(|o| o.metric_defect).collect::<Vec<_>>())
        .with("eigen_residual", orbitals.iter().map(|o| o.eigen_residual).collect::<Vec<_>>())
        .with("band_energy_residual", energy_residual)
        .with("density_reconstruction_residual", density_residual)
        .with("convention", "psi_nk(r)=sum_mu C_mu,n(k) sum_T exp(i k.T) chi_mu(r-T); spherical AO rows in basis order, ascending band columns, no time-reversal phase fixing; phases and degenerate-subspace rotations are arbitrary; P_spin=C f_spin C^dagger, total density=sum_channels spin_weight*P_spin; coefficients of the SCF parent, without a post-SCF correlated density"))
}

fn electronic_structure(system: &System, controls: &Controls) -> Result<Value> {
    let method = PeriodicMethod::from_controls(controls)?;
    let coefficients = controls.bool("coefficients", false)?;
    controls.finish("electronic-structure")?;
    let setup = prepare_periodic(system, &method)?;
    let scf = run_reference(&setup)?;
    let nk = scf.k_points.len();
    let spins = if scf.unrestricted { 2 } else { 1 };
    // Mulliken populations: sum_k w_k Re Tr_A[P(k) S(k)] for each spin.
    let owner = setup.cell.basis().sph_ao_atoms();
    let natoms = setup.cell.molecule().len();
    let mut populations = vec![vec![0.0; natoms]; 2];
    for spin in 0..2 {
        for ik in 0..nk {
            let p = &scf.density[spin][ik];
            let s = &scf.overlap[ik];
            for mu in 0..setup.cell.nao() {
                let mut diag = 0.0;
                for nu in 0..setup.cell.nao() {
                    // Re[(P S)_mu mu] = sum_nu Re P_mu nu S_nu mu - Im P_mu nu Im S_nu mu.
                    diag += p.re[(mu, nu)] * s.re[(nu, mu)] - p.im[(mu, nu)] * s.im[(nu, mu)];
                }
                populations[spin][owner[mu]] += diag / nk as f64;
            }
        }
    }
    let charges = setup.cell.basis().effective_charges(setup.cell.molecule());
    let mulliken: Vec<f64> = (0..natoms)
        .map(|a| charges[a] - populations[0][a] - populations[1][a])
        .collect();
    let spin_populations: Vec<f64> = (0..natoms)
        .map(|a| populations[0][a] - populations[1][a])
        .collect();
    let electrons: f64 = populations.iter().flatten().sum();
    let expected = (scf.n_alpha + scf.n_beta) as f64;
    let mut channels = Vec::new();
    for spin in 0..spins {
        let filled = if spin == 0 { scf.n_alpha } else { scf.n_beta };
        let mut channel = Value::object()
            .with(
                "spin",
                if scf.unrestricted {
                    if spin == 0 {
                        "alpha"
                    } else {
                        "beta"
                    }
                } else {
                    "restricted"
                },
            )
            .with("band_energies", scf.mo_energy[spin].clone())
            .with("occupations", scf.occupations[spin].clone())
            .with("chemical_potential", scf.chemical_potential[spin])
            .with(
                "band_edges",
                if setup.method.electronic_temperature == 0.0 {
                    gap_summary(&scf.mo_energy[spin], filled)
                } else {
                    Value::Null
                },
            );
        if coefficients {
            channel.set(
                "orbitals",
                orbital_export(&scf, spin, setup.options.orth_threshold)?,
            );
            channel.set("spin_weight", if scf.unrestricted { 1 } else { 2 });
        }
        channels.push(channel);
    }
    Ok(reference_report(
        &setup,
        &scf,
        "finite-k electronic structure of the converged periodic reference",
    )
    .with("channels", channels)
    .with("mulliken", mulliken)
    .with("mulliken_spin", spin_populations)
    .with("electron_count_residual", electrons - expected)
    .with(
        "units",
        units(&[
            ("energy", "hartree per cell"),
            (
                "channels.orbitals.coefficients_real",
                "dimensionless, [k][AO][band]",
            ),
            (
                "channels.orbitals.coefficients_imaginary",
                "dimensionless, [k][AO][band]",
            ),
            (
                "channels.orbitals.overlap_real",
                "dimensionless, [k][AO][AO]",
            ),
            (
                "channels.orbitals.overlap_imaginary",
                "dimensionless, [k][AO][AO]",
            ),
            ("channels.orbitals.fock_real", "hartree, [k][AO][AO]"),
            ("channels.orbitals.fock_imaginary", "hartree, [k][AO][AO]"),
            (
                "channels.orbitals.density_real",
                "dimensionless, per spin [k][AO][AO]",
            ),
            (
                "channels.orbitals.density_imaginary",
                "dimensionless, per spin [k][AO][AO]",
            ),
            (
                "channels.orbitals.metric_defect",
                "dimensionless, maximum entry per k",
            ),
            (
                "channels.orbitals.eigen_residual",
                "dimensionless, relative Frobenius norm per k",
            ),
            (
                "channels.orbitals.band_energy_residual",
                "hartree, maximum band defect per k",
            ),
            (
                "channels.orbitals.density_reconstruction_residual",
                "dimensionless, maximum AO entry per k",
            ),
            ("channels.band_energies", "hartree, [k][band]"),
            (
                "channels.occupations",
                "electrons per band and spin, [k][band]",
            ),
            ("mulliken", "e"),
            ("k_points", "1/bohr"),
        ]),
    ))
}

/// Fractional k points along straight segments between `vertices`.
fn path_points(vertices: &[Vec<f64>], points: usize) -> Result<Vec<[f64; 3]>> {
    if vertices.len() < 2 || vertices.iter().any(|v| v.len() != 3) || points == 0 {
        return Err(DftError::InvalidInput(
            "path needs at least two reduced k vertices of three numbers and points >= 1".into(),
        ));
    }
    let mut out = Vec::new();
    for w in vertices.windows(2) {
        for i in 0..points {
            let t = i as f64 / points as f64;
            out.push(std::array::from_fn(|a| w[0][a] + t * (w[1][a] - w[0][a])));
        }
    }
    let last = vertices.last().unwrap();
    out.push([last[0], last[1], last[2]]);
    Ok(out)
}

fn cartesian(lattice: &Lattice, f: [f64; 3]) -> Vec3 {
    let b = lattice.reciprocal();
    b[0] * f[0] + b[1] * f[1] + b[2] * f[2]
}

/// Reference energy of a channel: the SCF chemical potential, or for an
/// insulating reference the midpoint between the highest occupied and the
/// lowest empty band of the SCF mesh.
fn reference_level(scf: &PeriodicScfResult, spin: usize) -> f64 {
    if let Some(mu) = scf.chemical_potential[spin] {
        return mu;
    }
    let mut homo = f64::NEG_INFINITY;
    let mut lumo = f64::INFINITY;
    for (energies, occupations) in scf.mo_energy[spin].iter().zip(&scf.occupations[spin]) {
        for (e, f) in energies.iter().zip(occupations) {
            if *f > 0.5 {
                homo = homo.max(*e);
            } else {
                lumo = lumo.min(*e);
            }
        }
    }
    match (homo.is_finite(), lumo.is_finite()) {
        (true, true) => 0.5 * (homo + lumo),
        (true, false) => homo,
        (false, true) => lumo,
        _ => 0.0,
    }
}

/// Velocity, inverse effective-mass tensor and the degenerate branches of
/// one band at one k point from analytic k derivatives of the converged
/// Kohn-Sham operator.
fn effective_mass(system: &System, controls: &Controls) -> Result<Value> {
    let method = PeriodicMethod::from_controls(controls)?;
    let k_reduced = controls.opt_f64_list("k_point")?.ok_or_else(|| {
        DftError::InvalidInput(
            "effective-mass needs k_point: reduced coordinates, e.g. [0, 0, 0]".into(),
        )
    })?;
    if k_reduced.len() != 3 {
        return Err(DftError::InvalidInput(
            "k_point takes three reduced coordinates".into(),
        ));
    }
    let band = controls.opt_usize("band")?.ok_or_else(|| {
        DftError::InvalidInput("effective-mass needs band: the zero-based band index".into())
    })?;
    let direction = match controls.opt_f64_list("direction")? {
        None => None,
        Some(d) if d.len() == 3 => Some(Vec3::new(d[0], d[1], d[2])),
        Some(_) => {
            return Err(DftError::InvalidInput(
                "direction takes a Cartesian vector of three numbers".into(),
            ))
        }
    };
    let spin = controls.usize("spin", 0)?;
    let tolerance = controls.f64("degeneracy_tolerance", 1e-6)?;
    controls.finish("effective-mass")?;
    if !(tolerance > 0.0) {
        return Err(DftError::InvalidInput(
            "degeneracy_tolerance must be positive (Hartree)".into(),
        ));
    }
    let setup = prepare_periodic(system, &method)?;
    let scf = run_reference(&setup)?;
    let model = crate::periodic::band_transport::BandModel::new(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
    )?;
    let k = cartesian(
        setup.cell.lattice(),
        [k_reduced[0], k_reduced[1], k_reduced[2]],
    );
    let d = model.derivatives(spin, k, band, direction, tolerance)?;
    let degenerate = d.cluster.len() > 1;
    if degenerate && direction.is_none() {
        return Err(DftError::InvalidInput(format!(
            "band {band} is degenerate with bands {:?} within {tolerance:e} Hartree: \
                give a direction for the branch masses",
            d.cluster
        )));
    }
    let mut report = reference_report(
        &setup,
        &scf,
        "band velocity and inverse effective-mass tensor from analytic k \
            derivatives of the converged Kohn-Sham operator (non-degenerate \
            perturbation theory; degenerate perturbation theory along a direction \
            for clusters)",
    )
    .with("k_point_reduced", k_reduced.clone())
    .with("k_point", k.to_array())
    .with("band", band)
    .with("spin", spin)
    .with("energy", d.energies[band])
    .with("reference_level", reference_level(&scf, spin))
    .with("cluster", d.cluster.clone())
    .with("degenerate", degenerate)
    .with("fock_image_residual", model.fock_image_residual);
    if let (Some(v), Some(m)) = (d.velocity, d.inverse_mass) {
        let rows: Vec<Vec<f64>> = m.iter().map(|r| r.to_vec()).collect();
        let mut matrix = crate::linalg::Matrix::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                matrix[(i, j)] = m[i][j];
            }
        }
        let (values, vectors) = crate::linalg::symmetric_eigen(&matrix)?;
        let masses: Vec<Value> = values
            .iter()
            .map(|x| {
                if *x != 0.0 {
                    Value::Float(1.0 / x)
                } else {
                    Value::Null
                }
            })
            .collect();
        let axes: Vec<Vec<f64>> = (0..3)
            .map(|c| (0..3).map(|r| vectors[(r, c)]).collect())
            .collect();
        report = report
            .with("velocity", v.to_vec())
            .with(
                "velocity_m_s",
                v.iter().map(|x| x * velocity_si()).collect::<Vec<_>>(),
            )
            .with("inverse_mass_tensor", rows)
            .with("inverse_mass_eigenvalues", values.clone())
            .with("principal_axes", axes);
        report.set("effective_masses", Value::Array(masses));
    }
    if let Some(u) = direction {
        let unit = u * (1.0 / u.norm());
        report = report
            .with("direction", unit.to_array())
            .with(
                "branch_velocities",
                d.branches.iter().map(|b| b.velocity).collect::<Vec<_>>(),
            )
            .with(
                "branch_second_derivatives",
                d.branches
                    .iter()
                    .map(|b| b.second_derivative)
                    .collect::<Vec<_>>(),
            );
        report.set(
            "branch_effective_masses",
            Value::Array(
                d.branches
                    .iter()
                    .map(|b| {
                        if b.second_derivative != 0.0 {
                            Value::Float(1.0 / b.second_derivative)
                        } else {
                            Value::Null
                        }
                    })
                    .collect(),
            ),
        );
    }
    report.set(
        "units",
        units(&[
            ("energy", "hartree"),
            ("velocity", "hartree*bohr/hbar (atomic units of velocity)"),
            ("velocity_m_s", "m/s"),
            ("inverse_mass_tensor", "1/m_e (d2e/dk_a dk_b)"),
            (
                "effective_masses",
                "m_e (inverse eigenvalues; negative for hole-like curvature)",
            ),
            (
                "branch_second_derivatives",
                "1/m_e along the unit direction",
            ),
            ("k_point", "1/bohr"),
        ]),
    );
    Ok(report)
}

/// Atomic unit of velocity `a0 E_h / hbar` in m/s.
fn velocity_si() -> f64 {
    crate::constants::BOHR_TO_METER * crate::constants::HARTREE_TO_JOULE
        / crate::constants::HBAR_J_S
}

/// Scalar-relaxation-time, rigid-band Boltzmann transport on a dense k
/// mesh of the converged Kohn-Sham operator.
fn boltzmann_transport(system: &System, controls: &Controls, feature: &str) -> Result<Value> {
    use crate::periodic::band_transport::{
        conductivity_factor, frequency_transport, hall_coefficient, hall_coefficient_factor,
        hall_conductivity_factor, hall_integrals, seebeck_factor, thermal_conductivity_factor,
        transport_coefficients, transport_integrals, BandModel, HallTensor,
    };
    use crate::periodic::scattering;
    let with_hall = feature == "hall-transport";
    let method = PeriodicMethod::from_controls(controls)?;
    let mesh = controls
        .opt_usize_list("transport_k_mesh")?
        .ok_or_else(|| {
            DftError::InvalidInput(format!(
                "{feature} needs transport_k_mesh: three positive integers"
            ))
        })?;
    if mesh.len() != 3 || mesh.contains(&0) {
        return Err(DftError::InvalidInput(
            "transport_k_mesh takes three positive integers".into(),
        ));
    }
    let temperatures = controls
        .opt_f64_list("transport_temperatures_k")?
        .ok_or_else(|| {
            DftError::InvalidInput(format!("{feature} needs transport_temperatures_k"))
        })?;
    let shifts = controls.opt_f64_list("transport_mu_shifts")?;
    let concentrations = controls.opt_f64_list("carrier_concentrations_cm3")?;
    let window = controls.f64("delta_window", 20.0)?;
    let tolerance = controls.f64("degeneracy_tolerance", 1e-6)?;
    let onsager_tolerance = if with_hall {
        controls.f64("hall_onsager_tolerance", 1e-10)?
    } else {
        1e-10
    };
    let frequencies = controls.opt_f64_list("transport_frequencies_hartree")?;
    let relaxation_time_fs = controls.opt_f64("relaxation_time_fs")?;
    let scattering_rows = controls.opt_rows("scattering_table")?;
    controls.finish(feature)?;
    if (scattering_rows.is_some() && relaxation_time_fs.is_some())
        || (scattering_rows.is_none() && frequencies.is_some() != relaxation_time_fs.is_some())
        || frequencies.as_ref().is_some_and(|f| f.is_empty())
        || relaxation_time_fs.is_some_and(|t| t <= 0.0)
    {
        return Err(DftError::InvalidInput(
            "give both non-empty transport_frequencies_hartree and positive \
            relaxation_time_fs, or neither; alternatively give scattering_table, \
            optionally with frequencies, without relaxation_time_fs"
                .into(),
        ));
    }
    let atomic_time_s = crate::constants::HBAR_J_S / crate::constants::HARTREE_TO_JOULE;
    let tau = relaxation_time_fs.map(|t| t * 1e-15 / atomic_time_s);
    let scattering_table = scattering_rows
        .as_ref()
        .map(|rows| {
            let knots = rows
                .iter()
                .map(|row| {
                    if row.len() != 2 {
                        return Err(DftError::InvalidInput(
                            "scattering_table rows must be [energy_hartree, lifetime_fs]".into(),
                        ));
                    }
                    Ok([row[0], row[1] * 1e-15 / atomic_time_s])
                })
                .collect::<Result<Vec<_>>>()?;
            scattering::RelaxationTimeTable::new(knots)
        })
        .transpose()?;
    let lifetime = |e| match &scattering_table {
        Some(table) => table.lifetime(e),
        None => {
            tau.ok_or_else(|| DftError::InvalidInput("frequency transport needs a lifetime".into()))
        }
    };
    let time_factor = if scattering_table.is_some() {
        atomic_time_s
    } else {
        1.0
    };
    let conductivity_key = if scattering_table.is_some() {
        "conductivity"
    } else {
        "conductivity_over_tau"
    };
    let thermal_key = if scattering_table.is_some() {
        "electronic_thermal_conductivity"
    } else {
        "electronic_thermal_conductivity_over_tau"
    };
    let hall_key = if scattering_table.is_some() {
        "hall_conductivity_derivative"
    } else {
        "hall_conductivity_derivative_over_tau_squared"
    };
    if onsager_tolerance <= 0.0 {
        return Err(DftError::InvalidInput(
            "hall_onsager_tolerance must be positive".into(),
        ));
    }
    if temperatures.is_empty() || temperatures.iter().any(|t| !(*t > 0.0)) {
        return Err(DftError::InvalidInput(
            "transport_temperatures_k must be positive".into(),
        ));
    }
    if shifts.is_some() == concentrations.is_some() {
        return Err(DftError::InvalidInput(
            "give exactly one of transport_mu_shifts (Hartree) and carrier_concentrations_cm3"
                .into(),
        ));
    }
    if shifts
        .as_ref()
        .or(concentrations.as_ref())
        .is_some_and(|v| v.is_empty())
    {
        return Err(DftError::InvalidInput(
            "transport chemical-potential shifts or carrier concentrations must not be empty"
                .into(),
        ));
    }
    if !(window > 0.0 && tolerance > 0.0) {
        return Err(DftError::InvalidInput(
            "delta_window and degeneracy_tolerance must be positive".into(),
        ));
    }
    let setup = prepare_periodic(system, &method)?;
    let scf = run_reference(&setup)?;
    let model = BandModel::new(&setup.cell, &setup.functional, &scf, &setup.options)?;
    let k_points = setup.cell.k_points([mesh[0], mesh[1], mesh[2]]);
    let volume = setup.cell.lattice().volume().abs();
    let g = if model.spins == 1 { 2.0 } else { 1.0 };
    let electrons = (scf.n_alpha + scf.n_beta) as f64;
    let bohr3_to_cm3 = (crate::constants::BOHR_TO_METER * 100.0).powi(3);
    let kelvin = crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
    let max_kt = temperatures.iter().fold(0.0_f64, |m, t| m.max(*t)) * kelvin;
    // Energies of every state first: the window of included velocities.
    let mut energies: Vec<f64> = Vec::new();
    for spin in 0..model.spins {
        for k in &k_points {
            energies.extend(model.bands(spin, *k)?.energies);
        }
    }
    let reference = reference_level(&scf, 0);
    let counted = |mu: f64, kt: f64| -> f64 {
        energies
            .iter()
            .map(|e| {
                let x = (e - mu) / kt;
                if x > 0.0 {
                    let t = (-x).exp();
                    t / (1.0 + t)
                } else {
                    1.0 / (1.0 + x.exp())
                }
            })
            .sum::<f64>()
            * g
            / k_points.len() as f64
    };
    // Chemical potentials per temperature.
    let mut targets: Vec<(f64, f64, Option<f64>)> = Vec::new();
    for t in &temperatures {
        let kt = t * kelvin;
        match (&shifts, &concentrations) {
            (Some(list), _) => {
                for s in list {
                    targets.push((*t, reference + s, None));
                }
            }
            (_, Some(list)) => {
                for c in list {
                    // Electrons per cell beyond neutrality: n (1/cm^3) V.
                    let extra = c * bohr3_to_cm3 * volume;
                    let goal = electrons + extra;
                    let (mut lo, mut hi) = (
                        energies.iter().fold(f64::INFINITY, |m, e| m.min(*e)) - 1.0,
                        energies.iter().fold(f64::NEG_INFINITY, |m, e| m.max(*e)) + 1.0,
                    );
                    if !(counted(lo, kt) <= goal && counted(hi, kt) >= goal) {
                        return Err(DftError::InvalidInput(format!(
                            "carrier concentration {c:e} cm^-3 is outside the bands of the basis"
                        )));
                    }
                    for _ in 0..200 {
                        let mid = 0.5 * (lo + hi);
                        if counted(mid, kt) < goal {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    targets.push((*t, 0.5 * (lo + hi), Some(*c)));
                }
            }
            _ => unreachable!(),
        }
    }
    let (mu_min, mu_max) = targets
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), (_, mu, _)| {
            (a.min(*mu), b.max(*mu))
        });
    let include = |e: f64| e >= mu_min - window * max_kt && e <= mu_max + window * max_kt;
    let mut states = Vec::new();
    let mut hall_states = Vec::new();
    for spin in 0..model.spins {
        for k in &k_points {
            states.extend(model.transport_states(spin, *k, tolerance, include)?);
            if with_hall {
                // Test the actual union of Fermi windows: an irrelevant
                // degeneracy between distant requested chemical potentials
                // must not invalidate an isolated-band Hall calculation.
                hall_states.extend(model.hall_states(spin, *k, tolerance, |e| {
                    targets
                        .iter()
                        .any(|(t, mu, _)| (e - mu).abs() <= window * t * kelvin)
                })?);
            }
        }
    }
    let mut rows = Vec::new();
    let mut hall_warnings: Vec<String> = Vec::new();
    for (t, mu, concentration) in &targets {
        let kt = t * kelvin;
        let l = if scattering_table.is_some() {
            scattering::transport_integrals(
                &states,
                volume,
                k_points.len(),
                g,
                *mu,
                kt,
                window,
                &lifetime,
            )?
        } else {
            transport_integrals(&states, volume, k_points.len(), g, *mu, kt, window)?
        };
        let c = transport_coefficients(&l, kt);
        let doping = l.electron_density * volume - electrons;
        let tensor = |m: [[f64; 3]; 3], scale: f64| -> Vec<Vec<f64>> {
            m.iter()
                .map(|r| r.iter().map(|v| v * scale).collect())
                .collect()
        };
        let mut row = Value::object()
            .with("temperature_k", *t)
            .with("chemical_potential", *mu)
            .with("mu_shift", mu - reference)
            .with("doping_electrons_per_cell", doping)
            .with(
                "carrier_concentration_cm3",
                doping / (volume * bohr3_to_cm3),
            )
            .with(
                conductivity_key,
                tensor(c.conductivity_over_tau, conductivity_factor() * time_factor),
            );
        row.set("requested_concentration_cm3", *concentration);
        row.set("seebeck", c.seebeck.map(|s| tensor(s, seebeck_factor())));
        row.set(
            thermal_key,
            c.thermal_conductivity_over_tau
                .map(|s| tensor(s, thermal_conductivity_factor() * time_factor)),
        );
        let hall = if with_hall {
            Some(if scattering_table.is_some() {
                scattering::hall_integrals(
                    &hall_states,
                    volume,
                    k_points.len(),
                    g,
                    *mu,
                    kt,
                    window,
                    &lifetime,
                )?
                .quadrature()
            } else {
                hall_integrals(&hall_states, volume, k_points.len(), g, *mu, kt, window)?
            })
        } else {
            None
        };
        if let Some(hall) = &hall {
            let tensor3 = |h: HallTensor, scale: f64| -> Vec<Vec<Vec<f64>>> {
                h.iter()
                    .map(|plane| {
                        plane
                            .iter()
                            .map(|r| r.iter().map(|v| v * scale).collect())
                            .collect()
                    })
                    .collect()
            };
            row.set(
                hall_key,
                tensor3(
                    hall.conductivity_derivative_over_tau_squared,
                    hall_conductivity_factor() * time_factor.powi(2),
                ),
            );
            row.set(
                "hall_coefficient",
                hall_coefficient(&l, hall).map(|h| tensor3(h, hall_coefficient_factor())),
            );
            row.set("raw_hall_onsager_residual", hall.raw_onsager_residual);
            row.set(
                "relative_hall_onsager_residual",
                hall.relative_onsager_residual,
            );
            row.set(
                "hall_onsager_satisfied",
                hall.relative_onsager_residual
                    .map(|r| r <= onsager_tolerance),
            );
            if let Some(r) = hall
                .relative_onsager_residual
                .filter(|r| *r > onsager_tolerance)
            {
                hall_warnings.push(format!(
                    "WARNING [hall-transport] relative Onsager defect {r:.6e} exceeds \
                    {onsager_tolerance:.6e} at T={t:.6e} K, mu={mu:.6e} Hartree; refine \
                    transport_k_mesh and check delta_window; no symmetry projection was \
                    applied"
                ));
            }
        }
        if let Some(frequencies) = &frequencies {
            let mut dynamic = Vec::new();
            for omega in frequencies {
                let thermal = scattering::thermoelectric_frequency_transport(
                    &states,
                    with_hall.then_some(hall_states.as_slice()),
                    volume,
                    k_points.len(),
                    g,
                    *mu,
                    kt,
                    window,
                    *omega,
                    &lifetime,
                )?;
                let ac = if scattering_table.is_some() {
                    thermal.electric.clone()
                } else {
                    frequency_transport(&l, hall.as_ref(), *omega, tau.unwrap())?
                };
                let pair = |z: crate::response::complex::Complex, scale: f64| {
                    vec![z.re * scale, z.im * scale]
                };
                let conductivity: Vec<Vec<Vec<f64>>> = ac
                    .conductivity
                    .iter()
                    .map(|r| {
                        r.iter()
                            .map(|z| pair(*z, conductivity_factor() * atomic_time_s))
                            .collect()
                    })
                    .collect();
                let mut entry = Value::object()
                    .with("angular_frequency_hartree", *omega)
                    .with("conductivity", conductivity);
                let complex_tensor = |m: [[crate::response::complex::Complex; 3]; 3],
                                      scale: f64|
                 -> Vec<Vec<Vec<f64>>> {
                    m.iter()
                        .map(|row| row.iter().map(|z| pair(*z, scale)).collect())
                        .collect()
                };
                entry.set(
                    "seebeck",
                    thermal.seebeck.map(|m| complex_tensor(m, seebeck_factor())),
                );
                entry.set(
                    "peltier",
                    thermal.peltier.map(|m| {
                        complex_tensor(
                            m,
                            crate::constants::HARTREE_TO_JOULE
                                / crate::constants::ELEMENTARY_CHARGE_C,
                        )
                    }),
                );
                entry.set(
                    "electronic_thermal_conductivity",
                    thermal
                        .thermal_conductivity
                        .map(|m| complex_tensor(m, thermal_conductivity_factor() * atomic_time_s)),
                );
                entry.set(
                    "raw_thermoelectric_inverse_residual",
                    thermal.inverse_residual,
                );
                entry.set("raw_kelvin_residual", thermal.kelvin_residual);
                if let Some(h) = ac.hall_conductivity_derivative {
                    let values: Vec<Vec<Vec<Vec<f64>>>> = h
                        .iter()
                        .map(|p| {
                            p.iter()
                                .map(|r| {
                                    r.iter()
                                        .map(|z| {
                                            pair(
                                                *z,
                                                hall_conductivity_factor() * atomic_time_s.powi(2),
                                            )
                                        })
                                        .collect()
                                })
                                .collect()
                        })
                        .collect();
                    entry.set("hall_conductivity_derivative", values);
                }
                dynamic.push(entry);
            }
            row.set("frequency_results", dynamic);
        }
        rows.push(row);
    }
    let mut report = reference_report(
        &setup,
        &scf,
        "constant-relaxation-time, rigid-band Boltzmann transport from analytic \
            band velocities of the converged Kohn-Sham operator on a dense k mesh",
    )
    .with("transport_k_mesh", mesh.clone())
    .with("reference_level", reference)
    .with("delta_window", window)
    .with("fock_image_residual", model.fock_image_residual)
    .with("results", rows);
    report.set(
        "units",
        units(&[
            ("chemical_potential", "hartree"),
            (
                "mu_shift",
                "hartree relative to reference_level (SCF chemical potential or mid-gap)",
            ),
            (
                "carrier_concentration_cm3",
                "electrons per cm^3 beyond neutrality (negative: holes)",
            ),
            (
                if scattering_table.is_some() {
                    "results.conductivity"
                } else {
                    "results.conductivity_over_tau"
                },
                if scattering_table.is_some() {
                    "S/m"
                } else {
                    "S/(m s)"
                },
            ),
            ("results.seebeck", "V/K (null when sigma is singular)"),
            (
                if scattering_table.is_some() {
                    "results.electronic_thermal_conductivity"
                } else {
                    "results.electronic_thermal_conductivity_over_tau"
                },
                if scattering_table.is_some() {
                    "W/(m K)"
                } else {
                    "W/(m K s)"
                },
            ),
        ]),
    );
    if with_hall {
        report.set(
            "model",
            "ordinary weak-field Hall and constant-relaxation-time rigid-band \
            transport from analytic isolated-band velocities and curvatures; \
            electron charge is negative; interband coherence, Berry curvature and \
            field-induced changes of the reference are outside this model",
        );
        report.set(
            "hall_axes",
            vec![
                "current (or electric field for hall_coefficient): x,y,z",
                "electric field (or current for hall_coefficient): x,y,z",
                "magnetic induction: x,y,z",
            ],
        );
        report.set(
            "hall_convention",
            "sigma_ab(B)=sigma_ab(0)+tau^2 sum_c B_c H_abc+O(B^2); E_a=rho_ab j_b; \
            hall_coefficient_abc=d rho_ab/d B_c; electron parabolic band \
            R_yxz=-1/(n e)",
        );
        report.set("hall_onsager_tolerance", onsager_tolerance);
        let mut warnings = report
            .get("warnings")
            .and_then(Value::as_array)
            .unwrap()
            .to_vec();
        warnings.extend(hall_warnings.into_iter().map(Value::Str));
        report.set("warnings", warnings);
        let mut u = report.get("units").unwrap().clone();
        u.set(
            &format!("results.{hall_key}"),
            if scattering_table.is_some() {
                "S/(m T)"
            } else {
                "S/(m T s^2)"
            },
        );
        u.set(
            "results.hall_coefficient",
            "m^3/C = ohm m/T (null when zero-field conductivity is singular)",
        );
        u.set(
            "results.raw_hall_onsager_residual",
            if scattering_table.is_some() { "atomic units of d sigma/dB, including lifetime squared, before any symmetry projection" }
            else { "atomic units of d sigma/dB / tau^2, before any symmetry projection" },
        );
        u.set(
            "results.relative_hall_onsager_residual",
            "dimensionless; null for an identically zero Hall tensor",
        );
        u.set(
            "hall_onsager_tolerance",
            "dimensionless; diagnostic only, does not modify the Hall tensor",
        );
        report.set("units", u);
    }
    if let Some(t) = relaxation_time_fs {
        report.set("relaxation_time_fs", t);
    }
    if frequencies.is_some() {
        report.set(
            "frequency_convention",
            "exp(-i omega t); signed angular frequency in Hartree/hbar; causal \
            intraband scalar relaxation-time kinetic equation; interband and Berry terms \
            require the coherent model",
        );
        report.set("complex_axes", "last axis [real, imaginary]");
        let mut u = report.get("units").unwrap().clone();
        u.set(
            "results.frequency_results.angular_frequency_hartree",
            "Hartree/hbar (angular frequency)",
        );
        u.set(
            "results.frequency_results.conductivity",
            "S/m, shape [3,3,2]",
        );
        u.set(
            "results.frequency_results.seebeck",
            "V/K, shape [3,3,2]; null for singular conductivity",
        );
        u.set(
            "results.frequency_results.peltier",
            "V, shape [3,3,2]; heat current per electric current; null for singular conductivity",
        );
        u.set(
            "results.frequency_results.electronic_thermal_conductivity",
            "W/(m K), shape [3,3,2]; open circuit; null for singular conductivity",
        );
        u.set(
            "results.frequency_results.raw_thermoelectric_inverse_residual",
            "dimensionless; normalized constitutive inverse residual before projection",
        );
        u.set(
            "results.frequency_results.raw_kelvin_residual",
            "hartree/e; Peltier minus temperature times transposed Seebeck before projection",
        );
        report.set("thermal_frequency_model", "retarded scalar relaxation-time response to an imposed local-equilibrium temperature gradient; electric open-circuit heat conduction; excludes energy-conserving hydrodynamic collision corrections and magnetic heat-current vertices");
        if with_hall {
            u.set(
                "results.frequency_results.hall_conductivity_derivative",
                "S/(m T), shape [3,3,3,2]",
            );
        }
        if relaxation_time_fs.is_some() {
            u.set("relaxation_time_fs", "fs");
        }
        report.set("units", u);
    }
    if let Some(rows) = scattering_rows {
        report.set("scattering_table", rows);
        report.set("scattering_convention", "rows [absolute Kohn-Sham energy in Hartree, lifetime in fs]; log-linear positive lifetime interpolation, constant endpoint tails; shift table energies together with the band energy zero; scalar energy-dependent intraband relaxation, no microscopic scattering-rate calculation");
        report.set("model", if with_hall {
            "energy-dependent scalar relaxation-time ordinary weak-field Hall and rigid-band thermoelectric transport from analytic isolated-band velocities and curvatures; lifetime squared inside the Hall sum; electron charge is negative; interband coherence, Berry terms and field-induced reference changes require separate models"
        } else {
            "energy-dependent scalar relaxation-time rigid-band thermoelectric transport from analytic band velocities; lifetime inside each transport moment; degenerate-cluster velocity products are retained"
        });
        if with_hall {
            report.set("hall_convention", "sigma_ab(B)=sigma_ab(0)+sum_c B_c H_abc+O(B^2), with tau(E)^2 inside H; E_a=rho_ab j_b; hall_coefficient_abc=d rho_ab/d B_c=-K0^-1 H_c K0^-1; electron isotropic parabolic band R_yxz is negative");
        }
        let mut u = report.get("units").unwrap().clone();
        u.set(
            "scattering_table",
            "[hartree, fs] columns: absolute band energy, lifetime",
        );
        report.set("units", u);
    }
    Ok(report)
}

/// Every-band retarded conductivity with physical covariant AO currents.
fn coherent_transport(system: &System, controls: &Controls) -> Result<Value> {
    use crate::periodic::coherent_transport::{conductivity, conductivity_factor, CoherentModel};
    let method = PeriodicMethod::from_controls(controls)?;
    let mesh = controls
        .opt_usize_list("transport_k_mesh")?
        .ok_or_else(|| {
            DftError::InvalidInput("coherent-transport requires transport_k_mesh".into())
        })?;
    let frequencies = controls
        .opt_f64_list("transport_frequencies_hartree")?
        .ok_or_else(|| {
            DftError::InvalidInput(
                "coherent-transport requires transport_frequencies_hartree".into(),
            )
        })?;
    let temperature = controls.f64("transport_temperature_k", 300.0)?;
    let eta = controls.f64("scattering_hartree", 0.001)?;
    let shift = controls.f64("transport_mu_shift", 0.0)?;
    controls.finish("coherent-transport")?;
    if mesh.len() != 3
        || mesh.contains(&0)
        || frequencies.is_empty()
        || temperature <= 0.0
        || eta <= 0.0
    {
        return Err(DftError::InvalidInput(
            "coherent transport needs a positive three-dimensional mesh, non-empty \
            frequencies, positive temperature and scattering rate"
                .into(),
        ));
    }
    let setup = prepare_periodic(system, &method)?;
    let scf = run_reference(&setup)?;
    let model = CoherentModel::new(&setup.cell, &setup.functional, &scf, &setup.options)?;
    let points = setup.cell.k_points([mesh[0], mesh[1], mesh[2]]);
    let mut currents = Vec::new();
    for spin in 0..model.bands.spins {
        for k in &points {
            currents.push(model.point(spin, *k)?);
        }
    }
    let reference = reference_level(&scf, 0);
    let mu = reference + shift;
    let kt = temperature * crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
    let g = if model.bands.spins == 1 { 2.0 } else { 1.0 };
    let scale = conductivity_factor(3)?;
    let values = |m: [[crate::response::complex::Complex; 3]; 3]| -> Vec<Vec<Vec<f64>>> {
        m.iter()
            .map(|r| r.iter().map(|z| vec![scale * z.re, scale * z.im]).collect())
            .collect()
    };
    let mut rows = Vec::new();
    for frequency in frequencies {
        let c = conductivity(
            &currents,
            setup.cell.lattice().volume().abs(),
            points.len(),
            g,
            mu,
            kt,
            frequency,
            eta,
        )?;
        rows.push(
            Value::object()
                .with("angular_frequency_hartree", frequency)
                .with("conductivity", values(c.conductivity))
                .with("intraband_conductivity", values(c.intraband))
                .with("interband_conductivity", values(c.interband))
                .with("antisymmetric_conductivity", values(c.antisymmetric))
                .with(
                    "raw_current_hermitian_residual",
                    c.current_hermitian_residual,
                )
                .with("raw_equation_residual", c.equation_residual),
        );
    }
    Ok(reference_report(
        &setup,
        &scf,
        "covariant independent-particle Kubo-Greenwood conductivity of the \
        frozen self-consistent operator; every intra/interband pair including \
        exact degenerate multiplets; finite physical scattering; no induced \
        Hartree/XC vertex or field-induced reference change",
    )
    .with("transport_k_mesh", mesh)
    .with("temperature_k", temperature)
    .with("scattering_hartree", eta)
    .with("chemical_potential", mu)
    .with("reference_level", reference)
    .with("fock_image_residual", model.bands.fock_image_residual)
    .with(
        "frequency_convention",
        "exp(-i omega t); signed angular frequency in Hartree/hbar; poles below the real axis",
    )
    .with(
        "axes",
        "[current x,y,z, electric field x,y,z, real/imaginary]",
    )
    .with("results", rows)
    .with(
        "units",
        units(&[
            ("results.conductivity", "S/m, shape [3,3,2]"),
            ("results.intraband_conductivity", "S/m"),
            ("results.interband_conductivity", "S/m"),
            (
                "results.antisymmetric_conductivity",
                "S/m; includes intrinsic Hall within this independent-particle model",
            ),
            ("results.angular_frequency_hartree", "Hartree/hbar"),
            ("scattering_hartree", "Hartree"),
            ("chemical_potential", "Hartree"),
            ("temperature_k", "K"),
            ("fock_image_residual", "Hartree"),
            (
                "results.raw_current_hermitian_residual",
                "hartree*bohr/hbar, before projection",
            ),
            (
                "results.raw_equation_residual",
                "atomic units; re-evaluated retarded pair equation",
            ),
        ]),
    ))
}

/// Integer quantum Hall invariant of a gapped occupied Bloch slice.
/// Berry-phase polarization of an insulating reference on a string mesh.
fn berry_polarization_feature(system: &System, controls: &Controls) -> Result<Value> {
    let mut method = PeriodicMethod::from_controls(controls)?;
    method.apply_collinear_reference(controls)?;
    let mesh = match controls.opt_usize_list("berry_mesh")? {
        None => method.k_mesh,
        Some(m) if m.len() == 3 && !m.contains(&0) => [m[0], m[1], m[2]],
        Some(_) => {
            return Err(DftError::InvalidInput(
                "berry_mesh takes three positive integers".into(),
            ))
        }
    };
    controls.finish("berry-polarization")?;
    let setup = prepare_periodic(system, &method)?;
    let scf = run_reference(&setup)?;
    let model = crate::periodic::band_transport::BandModel::new(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
    )?;
    let b = crate::periodic::berry_phase::berry_polarization(&setup.cell, &model, &scf, mesh)?;
    let lattice = setup.cell.lattice();
    let volume = lattice.volume().abs();
    let quanta: Vec<Vec<f64>> = lattice
        .a
        .iter()
        .map(|a| a.to_array().iter().map(|x| x / volume).collect())
        .collect();
    let si = crate::constants::ELEMENTARY_CHARGE_C
        / (crate::constants::BOHR_TO_METER * crate::constants::BOHR_TO_METER);
    let mut report = reference_report(
        &setup,
        &scf,
        "Berry-phase polarization of the occupied manifold on reciprocal strings with \
         plane-wave cross-k AO overlaps, plus the ionic term; defined modulo the \
         polarization quanta, so only changes along an insulating path are physical",
    )
    .with("berry_mesh", mesh.to_vec())
    .with(
        "berry_phases",
        b.phases.iter().map(|p| p.to_vec()).collect::<Vec<_>>(),
    )
    .with("electronic_reduced", b.electronic_reduced.to_vec())
    .with("ionic_reduced", b.ionic_reduced.to_vec())
    .with("total_reduced", b.total_reduced.to_vec())
    .with("polarization", b.polarization.to_vec())
    .with(
        "polarization_c_m2",
        b.polarization.iter().map(|p| p * si).collect::<Vec<_>>(),
    )
    .with("polarization_quanta", quanta)
    .with("string_spread", b.string_spread.to_vec())
    .with(
        "min_link_log_abs_determinant",
        b.min_link_log_abs_determinant,
    )
    .with("fock_image_residual", model.fock_image_residual);
    report.set(
        "min_gap",
        if b.min_gap.is_finite() {
            Some(b.min_gap)
        } else {
            None
        },
    );
    report.set(
        "units",
        units(&[
            (
                "berry_phases",
                "radians, [spin][reciprocal axis], average over the perpendicular mesh",
            ),
            (
                "electronic_reduced",
                "electrons along the lattice vectors, modulo the spin degeneracy",
            ),
            (
                "ionic_reduced",
                "elementary charges along the lattice vectors",
            ),
            (
                "total_reduced",
                "elementary charges along the lattice vectors, modulo one",
            ),
            (
                "polarization",
                "e/bohr^2, Cartesian, modulo polarization_quanta",
            ),
            ("polarization_c_m2", "C/m^2"),
            (
                "polarization_quanta",
                "e/bohr^2, one vector a_i/V per lattice vector",
            ),
            ("min_gap", "hartree (null when the basis has no empty band)"),
        ]),
    );
    Ok(report)
}

fn quantum_hall(system: &System, controls: &Controls) -> Result<Value> {
    use crate::periodic::coherent_transport::{quantum_conductance, quantum_hall_slice};
    let method = PeriodicMethod::from_controls(controls)?;
    let mesh = controls
        .opt_usize_list("topology_mesh")?
        .unwrap_or_else(|| vec![8, 8]);
    let axes = controls
        .opt_usize_list("topology_plane_axes")?
        .unwrap_or_else(|| vec![0, 1]);
    let fixed = controls.f64("topology_fixed_k", 0.0)?;
    controls.finish("quantum-hall")?;
    if mesh.len() != 2
        || mesh.iter().any(|n| *n < 2)
        || axes.len() != 2
        || axes.iter().any(|a| *a >= 3)
        || axes[0] == axes[1]
        || method.electronic_temperature != 0.0
    {
        return Err(DftError::InvalidInput(
            "quantum Hall requires two distinct lattice axes 0..2, a topology_mesh \
            >=2 and a zero-temperature insulating reference"
                .into(),
        ));
    }
    let setup = prepare_periodic(system, &method)?;
    let scf = run_reference(&setup)?;
    let model = crate::periodic::band_transport::BandModel::new(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
    )?;
    let occupied = [scf.n_alpha, scf.n_beta];
    let g = if model.spins == 1 { 2.0 } else { 1.0 };
    let mut rows = Vec::new();
    let mut total = 0.0;
    for spin in 0..model.spins {
        if occupied[spin] == 0 {
            rows.push(
                Value::object()
                    .with("spin", spin)
                    .with("occupied_rank", 0_usize)
                    .with("chern_number", 0.0)
                    .with("slice_hall_conductance", 0.0),
            );
            continue;
        }
        let (r, gap) = quantum_hall_slice(
            &setup.cell,
            &model,
            spin,
            occupied[spin],
            [mesh[0], mesh[1]],
            [axes[0], axes[1]],
            fixed,
        )?;
        total += g * r.chern_number;
        rows.push(
            Value::object()
                .with("spin", spin)
                .with("occupied_rank", occupied[spin])
                .with("chern_number", r.chern_number)
                .with("integer_residual", r.integer_residual)
                .with("sampled_min_gap", gap)
                .with("max_plaquette_phase", r.max_plaquette_phase)
                .with(
                    "min_link_log_abs_determinant",
                    r.min_link_log_abs_determinant,
                )
                .with("plaquette_phases", r.plaquette_phases)
                .with(
                    "slice_hall_conductance",
                    quantum_conductance(r.chern_number),
                ),
        );
    }
    let mut report = reference_report(
        &setup,
        &scf,
        "non-Abelian occupied-subspace lattice-gauge Chern invariant and integer \
        quantum Hall conductance of a gapped two-dimensional slice of the \
        converged Bloch reference; exact plane-wave AO cross-k overlaps and \
        reciprocal-boundary sewing; no integer rounding",
    );
    let mut warnings = report
        .get("warnings")
        .and_then(Value::as_array)
        .unwrap()
        .to_vec();
    warnings.push(Value::Str(
        "Verify the spectral gap and Chern invariant on refined topology meshes. \
        A 3D slice conductance is not a bulk conductivity. Spin polarization \
        can break physical time reversal in this collinear zero-field SCF, \
        while each spin retains orbital conjugation symmetry. Nonzero intrinsic \
        charge quantum Hall states require a reference with broken orbital \
        conjugation symmetry; SOC, flux-compatible orbital fields or \
        spontaneous orbital-current order require additional ground-state \
        operators."
            .into(),
    ));
    report.set("warnings", warnings);
    Ok(report
        .with("topology_mesh", mesh)
        .with("topology_plane_axes", axes)
        .with("topology_fixed_k", fixed)
        .with("chern_number", total)
        .with("slice_hall_conductance", quantum_conductance(total))
        .with("spin_weight", g)
        .with("fock_image_residual", model.fock_image_residual)
        .with("results", rows)
        .with(
            "hall_convention",
            "ordered plane axes; sigma_12=C e^2/h with C=sum principal \
            link-plaquette phases/(2 pi); current response uses exp(-i omega t)",
        )
        .with(
            "units",
            units(&[
                (
                    "slice_hall_conductance",
                    "S; two-dimensional slice, includes spin weight",
                ),
                ("chern_number", "dimensionless, no integer rounding"),
                ("results.sampled_min_gap", "Hartree"),
                ("results.integer_residual", "dimensionless"),
                ("results.plaquette_phases", "radians; x fastest"),
                ("results.max_plaquette_phase", "radians"),
                (
                    "results.min_link_log_abs_determinant",
                    "dimensionless logarithm",
                ),
                (
                    "topology_fixed_k",
                    "reduced reciprocal coordinate of the axis outside the plane",
                ),
                ("fock_image_residual", "Hartree"),
            ]),
        ))
}

/// Cubic or quartic force constants of a cell contracted with Gamma-point
/// displacement patterns on their leading indices: the derivative of the
/// Gamma-point force constants along `probe`, or their second derivative
/// along `probe` and `second_probe`. The k mesh is sampled at the Gamma point
/// of its supercell, in which every replica of an atom moves with it.
fn force_constants_along(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    let quartic = id == "periodic-quartic-fc-along";
    let method = PeriodicMethod::from_controls_with_dispersion(controls)?;
    let dispersion_model = method.dispersion_model_for(&system.numbers)?;
    let config = super::vibrational::response_config(controls)?;
    let field = controls.opt_vec3("field")?;
    let probe = controls.opt_f64_list("probe")?.ok_or_else(|| {
        DftError::InvalidInput(format!(
            "{id} needs probe: one displacement value per Cartesian coordinate (3*atom + axis) \
             of the cell"
        ))
    })?;
    let second = if quartic {
        controls.opt_f64_list("second_probe")?
    } else {
        None
    };
    controls.finish(id)?;
    if method.electronic_temperature != 0.0 {
        return Err(DftError::Unsupported {
            feature: format!("{id} of a Fermi-Dirac reference"),
            missing: "the occupation response along the displacement path".into(),
        });
    }
    let (super_system, super_method) = supercell_system(system, &method, method.k_mesh)?;
    let replicas: usize = method.k_mesh.iter().product();
    let setup = prepare_periodic(&super_system, &super_method)?;
    let scf = run_reference(&setup)?;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    let mut probes = vec![probe.clone()];
    if quartic {
        probes.push(second.unwrap_or_else(|| probe.clone()));
    }
    let mut result = crate::response::nuclear::lattice_electronic_force_constants_along(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
        replicas,
        &probes,
        field.map(Vec3::to_array),
        &config,
        dispersion_model.as_ref(),
    )?;
    let dispersion = dispersion_model
        .as_ref()
        .map(|model| {
            let positions: Vec<_> = system
                .positions
                .iter()
                .map(|p| Vec3::new(p[0], p[1], p[2]))
                .collect();
            crate::periodic::dispersion::periodic_dispersion_force_constants_along(
                &system.numbers,
                &positions,
                &lattice_rows_of(system)?,
                model,
                &probes,
            )
        })
        .transpose()?;
    if let Some(extra) = &dispersion {
        for (total, term) in result
            .force_constants
            .matrix
            .as_mut_slice()
            .iter_mut()
            .zip(extra.matrix.as_slice())
        {
            *total += *term;
        }
        result.reference_energy += extra.energy;
    }
    let fc = &result.force_constants;
    let n = fc.matrix.rows;
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|x| (0..n).map(|y| fc.matrix[(x, y)]).collect())
        .collect();
    let mut report = reference_report(
        &setup,
        &scf,
        if quartic {
            "quartic force constants of the cell contracted with two Gamma-point displacement \
             patterns, from responses through second order of the Gamma-point supercell of the \
             k mesh"
        } else {
            "cubic force constants of the cell contracted with one Gamma-point displacement \
             pattern, from first-order responses (2n + 1 rule) of the Gamma-point supercell of \
             the k mesh"
        },
    )
    .with("supercell", method.k_mesh.to_vec())
    .with("probe", probe)
    .with("contracted_force_constants", rows)
    .with("path_evaluations", fc.paths)
    .with("response_residuals",fc.response_residuals.clone())
    .with("accumulation_roundoff", fc.accumulation_roundoff)
    .with("response_converged", fc.converged)
    .with("series_reference_energy", result.reference_energy)
    .with(
        "series_reference_energy_difference",
        result.reference_energy_difference,
    )
    .with("series_reference_residual", result.reference_residual)
    .with("series_reference_iterations", result.reference_iterations)
    .with("provenance", "analytic")
    .with(
        "units",
        units(&[
            ("energy", "hartree per supercell of the k mesh"),
            ("supercell", "integer"),
            ("probe", "dimensionless"),
            (
                "contracted_force_constants",
                if quartic {
                    "hartree/bohr^4 per unit of both patterns and per primitive cell, rows and columns 3*atom + axis"
                } else {
                    "hartree/bohr^3 per unit pattern and per primitive cell, rows and columns 3*atom + axis"
                },
            ),
            (
                "accumulation_roundoff",
                if quartic {
                    "hartree/bohr^4"
                } else {
                    "hartree/bohr^3"
                },
            ),
            ("series_reference_energy", "hartree per primitive cell"),
            (
                "series_reference_energy_difference",
                "hartree per primitive cell, reference of the integral series minus the self-consistent field",
            ),
            ("series_reference_residual", "hartree"),
            ("response_residuals","dimensionless"),
        ]),
    );
    if let Some(Value::Array(existing)) = report.get("warnings").cloned() {
        let mut all = existing;
        all.extend(fc.warnings.iter().cloned().map(Value::Str));
        report.set("warnings", Value::Array(all));
    }
    if let Some(field) = field {
        report.set("reference_field", field.to_array());
        report.set("field_energy", result.reference_energy);
        report.set(
            "nuclear_energy_model",
            "stationary Berry electric enthalpy at fixed Cartesian electric field",
        );
        let mut labels = report.get("units").cloned().expect("force-constant units");
        labels.set("reference_field", "hartree/e/bohr");
        labels.set("field_energy", "hartree per primitive cell");
        report.set("units", labels);
    }
    if let Some(extra) = dispersion {
        include_dispersion(&mut report, &method, extra.energy * replicas as f64);
        report.set("dispersion_energy_per_primitive_cell", extra.energy);
        report.set(
            "dispersion_force_constants",
            nested(extra.matrix.as_slice(), n, 1.0),
        );
        report.set("dispersion_acoustic_residual", extra.acoustic_residual);
        let mut labels = report.get("units").cloned().expect("force-constant units");
        labels.set("dispersion_energy", "hartree per supercell of the k mesh");
        labels.set(
            "dispersion_energy_per_primitive_cell",
            "hartree per primitive cell",
        );
        for key in ["disp_cutoff", "cn_cutoff", "abc_cutoff"] {
            labels.set(key, "bohr");
        }
        labels.set(
            "dispersion_force_constants",
            if quartic {
                "hartree/bohr^4 per primitive cell"
            } else {
                "hartree/bohr^3 per primitive cell"
            },
        );
        labels.set(
            "dispersion_acoustic_residual",
            if quartic {
                "hartree/bohr^4 per primitive cell"
            } else {
                "hartree/bohr^3 per primitive cell"
            },
        );
        report.set("units", labels);
    }
    Ok(report)
}

/// The Gamma-point supercell of the k mesh with its converged reference,
/// and the number of primitive cells in it.
fn gamma_supercell(
    id: &str,
    system: &System,
    method: &PeriodicMethod,
) -> Result<(PeriodicSetup, PeriodicScfResult, usize)> {
    if method.electronic_temperature != 0.0 {
        return Err(DftError::Unsupported {
            feature: format!("{id} of a Fermi-Dirac reference"),
            missing: "the occupation response along the path of the cell".into(),
        });
    }
    let (super_system, super_method) = supercell_system(system, method, method.k_mesh)?;
    let replicas: usize = method.k_mesh.iter().product();
    let setup = prepare_periodic(&super_system, &super_method)?;
    let scf = run_reference(&setup)?;
    if !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    Ok((setup, scf, replicas))
}

/// A tensor with `n` values per index as nested arrays, scaled.
fn nested(values: &[f64], n: usize, scale: f64) -> Value {
    if values.len() <= n {
        Value::Array(values.iter().map(|v| Value::Float(v * scale)).collect())
    } else {
        Value::Array(
            values
                .chunks(values.len() / n)
                .map(|chunk| nested(chunk, n, scale))
                .collect(),
        )
    }
}

fn series_reference_fields(
    report: Value,
    reference: &crate::response::nuclear::SeriesReference,
) -> Value {
    report
        .with("series_reference_energy", reference.energy)
        .with(
            "series_reference_energy_difference",
            reference.energy_difference,
        )
        .with("series_reference_residual", reference.residual)
        .with("series_reference_iterations", reference.iterations)
}

fn include_lattice_series_dispersion(
    report: &mut Value,
    method: &PeriodicMethod,
    replicas: usize,
    energy: Option<f64>,
) {
    if let Some(energy) = energy {
        include_dispersion(report, method, energy * replicas as f64);
        report.set("dispersion_energy_per_primitive_cell", energy);
        if let Some(mut labels) = report.get("units").cloned() {
            labels.set("dispersion_energy", "hartree per supercell of the k mesh");
            labels.set(
                "dispersion_energy_per_primitive_cell",
                "hartree per primitive cell",
            );
            for key in ["disp_cutoff", "cn_cutoff", "abc_cutoff"] {
                labels.set(key, "bohr");
            }
            report.set("units", labels);
        }
    }
}

const SERIES_REFERENCE_UNITS: [(&str, &str); 3] = [
    ("series_reference_energy", "hartree per primitive cell"),
    (
        "series_reference_energy_difference",
        "hartree per primitive cell, reference of the integral series minus the self-consistent field",
    ),
    ("series_reference_residual", "hartree"),
];

/// Complete cubic or quartic force constants of a cell for Gamma-point
/// displacements.
fn force_constant_tensor(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    let order = if id == "periodic-quartic-fc" { 4 } else { 3 };
    let method = PeriodicMethod::from_controls_with_dispersion(controls)?;
    let dispersion_model = method.dispersion_model_for(&system.numbers)?;
    let config = super::vibrational::response_config(controls)?;
    let field = controls.opt_vec3("field")?;
    controls.finish(id)?;
    let (setup, scf, replicas) = gamma_supercell(id, system, &method)?;
    let mut result = crate::response::nuclear::lattice_electronic_force_constant_tensor(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
        replicas,
        order,
        field.map(Vec3::to_array),
        &config,
        dispersion_model.as_ref(),
    )?;
    let dispersion = dispersion_model
        .as_ref()
        .map(|model| {
            let positions: Vec<_> = system
                .positions
                .iter()
                .map(|p| Vec3::new(p[0], p[1], p[2]))
                .collect();
            crate::periodic::dispersion::periodic_dispersion_force_constant_tensor(
                &system.numbers,
                &positions,
                &lattice_rows_of(system)?,
                model,
                order,
            )
        })
        .transpose()?;
    if let Some(extra) = &dispersion {
        for (total, term) in result.tensor.iter_mut().zip(&extra.tensor) {
            *total += *term;
        }
        result.reference.energy += extra.energy;
    }
    let unit = if order == 3 {
        "hartree/bohr^3 per primitive cell, indices 3*atom + axis"
    } else {
        "hartree/bohr^4 per primitive cell, indices 3*atom + axis"
    };
    let mut list = vec![
        ("energy", "hartree per supercell of the k mesh"),
        ("supercell", "integer"),
        ("force_constant_tensor", unit),
        ("response_residuals", "dimensionless"),
    ];
    list.extend(SERIES_REFERENCE_UNITS);
    let report = reference_report(
        &setup,
        &scf,
        if order == 3 {
            "cubic force constants of the cell for Gamma-point displacements, from first-order \
             responses (2n + 1 rule) of the Gamma-point supercell of the k mesh"
        } else {
            "quartic force constants of the cell for Gamma-point displacements, from responses \
             through second order of the Gamma-point supercell of the k mesh"
        },
    )
    .with("supercell", method.k_mesh.to_vec())
    .with("tensor_order", result.order)
    .with("dimension", result.dimension)
    .with(
        "force_constant_tensor",
        nested(&result.tensor, result.dimension, 1.0),
    )
    .with("path_evaluations", result.paths)
    .with("response_residuals", result.response_residuals.clone())
    .with("provenance", "analytic")
    .with("units", units(&list));
    let mut report = series_reference_fields(report, &result.reference);
    if let Some(field) = field {
        report.set("reference_field", field.to_array());
        report.set("field_energy", result.reference.energy);
        report.set(
            "nuclear_energy_model",
            "stationary Berry electric enthalpy at fixed Cartesian electric field",
        );
        let mut labels = report.get("units").cloned().expect("force-constant units");
        labels.set("reference_field", "hartree/e/bohr");
        labels.set("field_energy", "hartree per primitive cell");
        report.set("units", labels);
    }
    if let Some(extra) = dispersion {
        include_dispersion(&mut report, &method, extra.energy * replicas as f64);
        report.set("dispersion_energy_per_primitive_cell", extra.energy);
        report.set(
            "dispersion_force_constant_tensor",
            nested(&extra.tensor, result.dimension, 1.0),
        );
        report.set(
            "dispersion_permutation_residual",
            extra.permutation_residual,
        );
        report.set("dispersion_acoustic_residual", extra.acoustic_residual);
        let mut labels = report.get("units").cloned().expect("force-constant units");
        labels.set("dispersion_energy", "hartree per supercell of the k mesh");
        labels.set(
            "dispersion_energy_per_primitive_cell",
            "hartree per primitive cell",
        );
        for key in ["disp_cutoff", "cn_cutoff", "abc_cutoff"] {
            labels.set(key, "bohr");
        }
        for key in [
            "dispersion_force_constant_tensor",
            "dispersion_permutation_residual",
            "dispersion_acoustic_residual",
        ] {
            labels.set(key, unit);
        }
        report.set("units", labels);
    }
    Ok(report)
}

/// Third-order (`toec`) or fourth-order (`foec`) Lagrangian elastic
/// constants at clamped and relaxed ions.
fn high_order_elastic(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    let order = if id == "foec" { 4 } else { 3 };
    let method = PeriodicMethod::from_controls_with_dispersion(controls)?;
    let dispersion = method.dispersion_model_for(&system.numbers)?;
    let field = controls.opt_vec3("field")?;
    let config = super::vibrational::response_config(controls)?;
    controls.finish(id)?;
    let (setup, scf, replicas) = gamma_supercell(id, system, &method)?;
    let result = crate::response::nuclear::lattice_elastic_constants_at_field(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
        replicas,
        order,
        &config,
        dispersion.as_ref(),
        field.map(Vec3::to_array),
    )?;
    let gpa = crate::constants::HARTREE_PER_BOHR3_TO_GPA;
    let relaxation = &result.internal_relaxation;
    let rows: Vec<Vec<f64>> = (0..relaxation.rows)
        .map(|x| (0..relaxation.cols).map(|y| relaxation[(x, y)]).collect())
        .collect();
    let mut list = vec![
        ("energy", "hartree per supercell of the k mesh"),
        ("supercell", "integer"),
        ("volume", "bohr^3 per primitive cell"),
        (
            "elastic_constants_clamped",
            "hartree/bohr^3, Voigt 6x6, Lagrangian",
        ),
        (
            "elastic_constants_relaxed",
            "hartree/bohr^3, Voigt 6x6, Lagrangian",
        ),
        (
            "high_order_clamped",
            "hartree/bohr^3, Voigt indices, Brugger constants at clamped ions",
        ),
        (
            "high_order_relaxed",
            "hartree/bohr^3, Voigt indices, Brugger constants with relaxed ions",
        ),
        ("high_order_clamped_gpa", "GPa"),
        ("high_order_relaxed_gpa", "GPa"),
        (
            "internal_relaxation",
            "bohr per unit Lagrangian strain, rows 3*atom + axis, columns Voigt",
        ),
        (
            "second_internal_relaxation",
            "bohr per squared unit Lagrangian strain, rows 6*I + J of the Voigt pair, columns 3*atom + axis (fourth order only)",
        ),
        ("smallest_optical_eigenvalue", "hartree/bohr^2"),
        ("response_residuals", "dimensionless"),
    ];
    list.extend(SERIES_REFERENCE_UNITS);
    if field.is_some() {
        list.extend([
            ("reference_field", "hartree/e/bohr"),
            (
                "field_energy",
                "hartree per primitive cell, stationary Berry electric enthalpy",
            ),
        ]);
    }
    let mut report = reference_report(
        &setup,
        &scf,
        if order == 3 {
            "third-order Lagrangian elastic constants at clamped and relaxed ions, from \
             first-order responses along Lagrangian strain paths of the Gamma-point supercell \
             of the k mesh"
        } else {
            "fourth-order Lagrangian elastic constants at clamped and relaxed ions, from \
             responses through second order along Lagrangian strain paths of the Gamma-point \
             supercell of the k mesh"
        },
    )
    .with("supercell", method.k_mesh.to_vec())
    .with("elastic_order", result.order)
    .with("voigt_order", "xx, yy, zz, yz, xz, xy")
    .with("volume", result.volume)
    .with(
        "elastic_constants_clamped",
        nested(&result.second_clamped, 6, 1.0),
    )
    .with(
        "elastic_constants_relaxed",
        nested(&result.second_relaxed, 6, 1.0),
    )
    .with("high_order_clamped", nested(&result.clamped, 6, 1.0))
    .with("high_order_relaxed", nested(&result.relaxed, 6, 1.0))
    .with("high_order_clamped_gpa", nested(&result.clamped, 6, gpa))
    .with("high_order_relaxed_gpa", nested(&result.relaxed, 6, gpa))
    .with("internal_relaxation", rows)
    .with(
        "smallest_optical_eigenvalue",
        result.smallest_optical_eigenvalue,
    )
    .with("path_evaluations", result.paths)
    .with("response_residuals", result.response_residuals)
    .with("provenance", "analytic")
    .with("units", units(&list));
    if let Some(second) = &result.second_internal_relaxation {
        let width = second.len() / 36;
        report.set("second_internal_relaxation", nested(second, width, 1.0));
    }
    if let Some(field) = field {
        report.set("reference_field", field.to_array());
        report.set("field_energy", result.reference.energy);
        report.set(
            "strain_energy_model",
            "stationary Berry electric enthalpy at fixed Cartesian electric field",
        );
    }
    include_lattice_series_dispersion(
        &mut report,
        &method,
        replicas,
        result.dispersion_energy_per_primitive_cell,
    );
    Ok(series_reference_fields(report, &result.reference))
}

/// Mode Grueneisen parameters of the Gamma-point phonons from the third
/// derivatives of the energy.
fn mode_gruneisen(system: &System, controls: &Controls) -> Result<Value> {
    let method = PeriodicMethod::from_controls_with_dispersion(controls)?;
    let dispersion = method.dispersion_model_for(&system.numbers)?;
    let config = super::vibrational::response_config(controls)?;
    let shear = controls.bool("gruneisen_shear", false)?;
    let order = controls.usize("gruneisen_order", 1)?;
    if order > 2 {
        return Err(DftError::Unsupported {
            feature: format!("mode Gruneisen derivative order {order}"),
            missing: "a lattice mode response above second order".into(),
        });
    }
    if order == 2 {
        if shear {
            return Err(DftError::Unsupported {
                feature: "second-order shear mode Gruneisen response".into(),
                missing: "the shear response; the second-order route supports isotropic log volume"
                    .into(),
            });
        }
        let relaxed = controls.bool("relaxed_ion", false)?;
        let tolerance = controls.f64("gruneisen_force_tolerance", 1e-5)?;
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(DftError::InvalidInput(
                "gruneisen_force_tolerance must be finite and positive".into(),
            ));
        }
        controls.finish("mode-gruneisen")?;
        let (setup, scf, replicas) = gamma_supercell("mode-gruneisen", system, &method)?;
        let result = crate::response::nuclear::lattice_volume_gruneisen_with_dispersion(
            &setup.cell,
            &setup.functional,
            &scf,
            &setup.options,
            replicas,
            relaxed,
            tolerance,
            &config,
            dispersion.as_ref(),
        )?;
        let branches = &result.spectrum.branches;
        let frequencies: Vec<f64> = branches
            .iter()
            .map(|b| {
                b.squared_frequency.signum()
                    * b.squared_frequency.abs().sqrt()
                    * crate::constants::HARTREE_TO_CM1
            })
            .collect();
        let first: Vec<Option<f64>> = branches.iter().map(|b| b.first_gruneisen).collect();
        let second: Vec<Option<f64>> = branches.iter().map(|b| b.second_log_frequency).collect();
        let mut list = vec![
            ("energy", "hartree per supercell of the k mesh"),
            ("frequencies_cm1", "cm^-1; negative means imaginary"),
            ("mode_gruneisen", "dimensionless, -d ln(omega)/d ln(V); null for acoustic or unstable modes"),
            ("second_log_frequency", "dimensionless, d2 ln(omega)/d ln(V)^2 = -d gamma/d ln(V); null for acoustic or unstable modes"),
            ("force_constants", "hartree/bohr^2, material displacements"),
            ("dynamical_jets", "hartree/bohr^2/electron_mass; D, dD/dln(V), d2D/dln(V)^2"),
            ("dynamical_symmetry_defects", "hartree/bohr^2/electron_mass"),
            ("reference_eigenvalues", "hartree/bohr^2/electron_mass"),
            ("modes", "dimensionless, orthonormal mass-weighted branch vectors"),
            ("response_residuals", "dimensionless, fresh relative Krylov residuals"),
            ("gruneisen_force_tolerance", "hartree/bohr"),
            ("internal_first", "bohr, du/ds for isotropic Lagrangian strain eta=sI"),
            ("internal_second", "bohr, d2u/ds2 for isotropic Lagrangian strain eta=sI"),
            ("ionic_stationarity_residual", "hartree/bohr"),
            ("pressure", "hartree/bohr^3, reference pressure from analytic isotropic strain derivative"),
        ];
        list.extend(SERIES_REFERENCE_UNITS);
        let mut report = reference_report(&setup, &scf,
            "Gamma-point first and second log-volume mode response from analytic third/fourth energy derivatives; physical displacement and internal-relaxation chain rules; degenerate-subspace spectral perturbation")
            .with("gruneisen_order", order)
            .with("supercell", method.k_mesh)
            .with("relaxed_ion", relaxed)
            .with("gruneisen_force_tolerance", tolerance)
            .with("frequencies_cm1", frequencies)
            .with("mode_gruneisen", first)
            .with("second_log_frequency", second)
            .with("modes", &result.spectrum.modes)
            .with("acoustic_modes", &result.acoustic)
            .with("force_constants", &result.force_constants)
            .with("dynamical_jets", &result.dynamical_jets)
            .with("dynamical_symmetry_defects", result.spectrum.symmetry_defects)
            .with("reference_eigenvalues", &result.spectrum.reference_eigenvalues)
            .with("internal_first", &result.internal_first)
            .with("internal_second", &result.internal_second)
            .with("ionic_stationarity_residual", result.ionic_stationarity_residual)
            .with("pressure", result.pressure)
            .with("path_evaluations", result.paths)
            .with("response_residuals", &result.response_residuals)
            .with("provenance", "analytic")
            .with("units", units(&list));
        include_lattice_series_dispersion(
            &mut report,
            &method,
            replicas,
            result.dispersion_energy_per_primitive_cell,
        );
        return Ok(series_reference_fields(report, &result.reference));
    }
    if order != 1 {
        return Err(DftError::InvalidInput(
            "gruneisen_order must be one or two".into(),
        ));
    }
    controls.finish("mode-gruneisen")?;
    let (setup, scf, replicas) = gamma_supercell("mode-gruneisen", system, &method)?;
    let result = crate::response::nuclear::lattice_mode_gruneisen_with_dispersion(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
        replicas,
        shear,
        &config,
        dispersion.as_ref(),
    )?;
    let n = result.force_constants.rows;
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|x| (0..n).map(|y| result.force_constants[(x, y)]).collect())
        .collect();
    let frequencies: Vec<f64> = result
        .frequencies
        .iter()
        .map(|w| w * crate::constants::HARTREE_TO_CM1)
        .collect();
    let strain: Vec<Vec<f64>> = result.strain.iter().map(|row| row.to_vec()).collect();
    let mut list =
        vec![
        ("energy", "hartree per supercell of the k mesh"),
        ("supercell", "integer"),
        ("frequencies_cm1", "cm^-1 (negative = imaginary)"),
        (
            "mode_gruneisen_strain",
            "dimensionless, -d ln(omega)/d eta per mode and Voigt strain (xx, yy, zz, yz, xz, xy)",
        ),
        (
            "mode_gruneisen",
            "dimensionless, -d ln(omega)/d ln(V) per mode",
        ),
        ("force_constants", "hartree/bohr^2"),
        ("response_residuals", "dimensionless, fresh relative Krylov residuals"),
    ];
    list.extend(SERIES_REFERENCE_UNITS);
    let mut report = reference_report(
        &setup,
        &scf,
        "mode Grueneisen parameters of the Gamma-point phonons from the strain derivative of \
         the force constants with first-order relaxation of the atoms (third derivatives of \
         the energy by the 2n + 1 rule)",
    )
    .with("supercell", method.k_mesh.to_vec())
    .with("frequencies_cm1", frequencies)
    .with("mode_gruneisen", result.volume.clone())
    .with("mode_gruneisen_strain", strain)
    .with("acoustic_modes", result.acoustic.clone())
    .with("gruneisen_shear", result.shear)
    .with("force_constants", rows)
    .with("path_evaluations", result.paths)
    .with("response_residuals", &result.response_residuals)
    .with("provenance", "analytic")
    .with("units", units(&list));
    include_lattice_series_dispersion(
        &mut report,
        &method,
        replicas,
        result.dispersion_energy_per_primitive_cell,
    );
    Ok(series_reference_fields(report, &result.reference))
}

/// Fourth electric-enthalpy derivative from the shared Berry-phase series.
fn gamma_field_response(system: &System, controls: &Controls) -> Result<Value> {
    let relaxed = controls.bool("relaxed_ion", false)?;
    let use_symmetry = controls.bool("use_symmetry", true)?;
    let field = controls
        .opt_vec3("field")?
        .unwrap_or_else(crate::math::Vec3::zero);
    let method = PeriodicMethod::from_controls(controls)?;
    let config = super::vibrational::response_config(controls)?;
    controls.finish("periodic-gamma")?;
    if relaxed {
        return Err(DftError::Unsupported {
            feature: "relaxed-ion periodic third-order electric susceptibility".into(),
            missing:
                "second-order internal-coordinate relaxation in the fourth enthalpy derivative"
                    .into(),
        });
    }
    let (setup, scf, replicas) = gamma_supercell("periodic-gamma", system, &method)?;
    let result = crate::response::nuclear::lattice_electric_tensor_at_field(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
        replicas,
        3,
        if use_symmetry {
            crate::response::nuclear::ElectricTensorSymmetry::IntrinsicPermutation
        } else {
            crate::response::nuclear::ElectricTensorSymmetry::Unprojected
        },
        [field.x, field.y, field.z],
        &config,
    )?;
    let mut labels = vec![
        ("energy", "hartree per supercell of the k mesh"),
        ("supercell", "integer"),
        ("volume", "bohr^3 per primitive cell"),
        ("reference_field", "hartree/e/bohr"),
        (
            "tensor",
            "e^2*bohr^2/hartree^2, Gaussian susceptibility per squared atomic electric field",
        ),
        (
            "third_susceptibility",
            "e^2*bohr^2/hartree^2, d3P_a/(dE_b dE_c dE_d), Gaussian convention",
        ),
        (
            "chi3",
            "e^2*bohr^2/hartree^2, third_susceptibility divided by six",
        ),
        ("raw_permutation_residual", "e^2*bohr^2/hartree^2"),
        ("tensor_shape", "integer"),
        ("polarization_derivative_order", "integer"),
        ("response_order", "integer"),
        ("response_residuals", "dimensionless"),
        ("path_evaluations", "integer"),
        ("independent_components", "integer"),
    ];
    labels.extend(SERIES_REFERENCE_UNITS);
    let report = reference_report(&setup, &scf,
        "static clamped-ion third-order electric susceptibility from the fourth derivative of the single-point Berry-phase electric enthalpy; Gamma supercell sampling of the k mesh, second-order orbital response and the 2n+1 rule; no finite differences")
        .with("supercell", method.k_mesh.to_vec())
        .with("volume", result.volume)
        .with("reference_field", result.reference_field)
        .with("tensor", result.tensor.clone())
        .with("third_susceptibility", result.tensor.clone())
        .with("chi3", result.tensor.iter().map(|v| v / 6.0).collect::<Vec<_>>())
        .with("tensor_shape", result.shape)
        .with("polarization_derivative_order", result.order)
        .with("response_order", 2usize)
        .with("raw_permutation_residual", result.permutation_residual.map(Value::Float).unwrap_or(Value::Null))
        .with("use_symmetry", use_symmetry)
        .with("independent_components", result.independent_components)
        .with("response_residuals", result.response_residuals)
        .with("path_evaluations", result.paths)
        .with("tensor_convention", "flat row-major [polarization axis][field axis][field axis][field axis]; actual static derivatives, chi3=third_susceptibility/6; intrinsic permutation expansion when use_symmetry=true; raw permutation defect measured only without expansion")
        .with("provenance", "analytic")
        .with("units", units(&labels));
    Ok(series_reference_fields(report, &result.reference))
}

/// Clamped-ion response of a cell to a uniform electric field through third
/// order: susceptibility and, by feature, the second-order susceptibility
/// (`periodic-beta`), Born charges and the displacement derivative of the
/// susceptibility (`periodic-raman`), its strain derivative
/// (`electrostriction`), or the piezoelectric tensor and its strain
/// derivative (`piezo-higher`).
fn field_response(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    let relaxed = controls.bool("relaxed_ion", false)?;
    let field = controls
        .opt_vec3("field")?
        .unwrap_or_else(crate::math::Vec3::zero);
    let method = PeriodicMethod::from_controls(controls)?;
    let config = super::vibrational::response_config(controls)?;
    controls.finish(id)?;
    let (setup, scf, replicas) = gamma_supercell(id, system, &method)?;
    let request = crate::response::nuclear::FieldRequest {
        second_susceptibility: id == "periodic-beta",
        raman: id == "periodic-raman",
        strain_susceptibility: id == "electrostriction",
        piezoelectric: id == "piezo-higher",
        relaxed,
    };
    let result = crate::response::nuclear::lattice_field_response_at_field(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
        replicas,
        &request,
        [field.x, field.y, field.z],
        &config,
    )?;
    let four_pi = 4.0 * std::f64::consts::PI;
    let dielectric: [[f64; 3]; 3] = std::array::from_fn(|a| {
        std::array::from_fn(|b| f64::from(u8::from(a == b)) + four_pi * result.susceptibility[a][b])
    });
    let mut list = vec![
        ("energy", "hartree per supercell of the k mesh"),
        ("supercell", "integer"),
        ("volume", "bohr^3 per primitive cell"),
        ("reference_field", "hartree/e/bohr"),
        ("response_residuals", "dimensionless"),
        ("susceptibility", "dimensionless, dP_a/dE_b at clamped ions"),
        ("dielectric_tensor", "dimensionless, 1 + 4 pi chi"),
        (
            "second_susceptibility",
            "e*bohr/hartree, d2P_a/(dE_b dE_c); chi2 is half of it",
        ),
        (
            "born_charges",
            "e, Omega dP_a/du_k, rows 3*atom + axis, columns field axis",
        ),
        (
            "raman_tensor",
            "bohr^-1, d chi_ab/du_k as [a][b][3*atom + axis]",
        ),
        (
            "strain_susceptibility",
            "dimensionless, -(1/Omega) d3F/(dE_a dE_b d eta_I) at fixed reduced field, as [a][b][Voigt]",
        ),
        (
            "piezoelectric_tensor",
            "e/bohr^2, proper clamped-ion tensor -(1/Omega) d2F/(dE_a d eta_I), as [a][Voigt]",
        ),
        (
            "second_piezoelectric",
            "e/bohr^2, -(1/Omega) d3F/(dE_a d eta_I d eta_J), as [a][Voigt][Voigt]",
        ),
        (
            "susceptibility_strain_derivative",
            "dimensionless, d chi_ab/d eta_I at fixed reduced field with the volume factor, as [a][b][Voigt]",
        ),
        (
            "relaxed_susceptibility",
            "dimensionless, static dP_a/dE_b with relaxed ions",
        ),
        ("static_dielectric_tensor", "dimensionless, 1 + 4 pi chi with relaxed ions"),
        (
            "field_relaxation",
            "bohr^2*e/hartree, du_k/dE_a, rows 3*atom + axis, columns field axis",
        ),
        (
            "strain_relaxation",
            "bohr, du_k/d eta_I, rows 3*atom + axis, columns Voigt",
        ),
        ("smallest_optical_eigenvalue", "hartree/bohr^2"),
        (
            "relaxed_second_susceptibility",
            "e*bohr/hartree, d2P_a/(dE_b dE_c) with relaxed ions",
        ),
        (
            "relaxed_strain_susceptibility",
            "dimensionless, -(1/Omega) d3F/(dE_a dE_b d eta_I) with relaxed ions, as [a][b][Voigt]",
        ),
        (
            "relaxed_susceptibility_strain_derivative",
            "dimensionless, d chi_ab/d eta_I with relaxed ions and the volume factor, as [a][b][Voigt]",
        ),
        (
            "relaxed_piezoelectric_tensor",
            "e/bohr^2, proper relaxed-ion tensor, as [a][Voigt]",
        ),
        (
            "relaxed_second_piezoelectric",
            "e/bohr^2, strain derivative of the relaxed-ion tensor, as [a][Voigt][Voigt]",
        ),
    ];
    list.extend(SERIES_REFERENCE_UNITS);
    let mut report = reference_report(
        &setup,
        &scf,
        "clamped-ion response of the cell to a uniform electric field: derivatives of \
         E - Omega E.P with the single-point Berry phase of the Gamma-point supercell of the \
         k mesh, from first-order responses along combined paths (2n + 1 rule)",
    )
    .with("supercell", method.k_mesh.to_vec())
    .with("volume", result.volume)
    .with("reference_field", result.reference_field)
    .with("voigt_order", "xx, yy, zz, yz, xz, xy")
    .with("susceptibility", result.susceptibility)
    .with("dielectric_tensor", dielectric)
    .with("path_evaluations", result.paths)
    .with("response_residuals", result.response_residuals)
    .with("provenance", "analytic");
    if let Some(tensor) = &result.second_susceptibility {
        report.set("second_susceptibility", nested(tensor, 3, 1.0));
    }
    if let Some(born) = &result.born_charges {
        let rows: Vec<Vec<f64>> = (0..born.rows)
            .map(|k| (0..3).map(|a| born[(k, a)]).collect())
            .collect();
        report.set("born_charges", rows);
    }
    if let Some(tensor) = &result.raman {
        let n = tensor.len() / 9;
        let blocks: Vec<Value> = tensor
            .chunks(3 * n)
            .map(|row| nested(row, 3, 1.0))
            .collect();
        report.set("raman_tensor", Value::Array(blocks));
    }
    // d chi / d eta_I = -(1/Omega) d3F - chi d ln(Omega) / d eta_I: the volume
    // changes with the three normal strains.
    let with_volume = |tensor: &[f64], chi: &[[f64; 3]; 3]| -> Value {
        let full: Vec<f64> = tensor
            .iter()
            .enumerate()
            .map(|(flat, value)| {
                let (pair, voigt) = (flat / 6, flat % 6);
                if voigt < 3 {
                    value - chi[pair / 3][pair % 3]
                } else {
                    *value
                }
            })
            .collect();
        Value::Array(full.chunks(18).map(|row| nested(row, 3, 1.0)).collect())
    };
    if let Some(tensor) = &result.strain_susceptibility {
        let blocks: Vec<Value> = tensor.chunks(18).map(|row| nested(row, 3, 1.0)).collect();
        report.set("strain_susceptibility", Value::Array(blocks));
        report.set(
            "susceptibility_strain_derivative",
            with_volume(tensor, &result.susceptibility),
        );
    }
    if let Some(tensor) = &result.piezoelectric {
        report.set("piezoelectric_tensor", *tensor);
    }
    if let Some(tensor) = &result.second_piezoelectric {
        let blocks: Vec<Value> = tensor.chunks(36).map(|row| nested(row, 6, 1.0)).collect();
        report.set("second_piezoelectric", Value::Array(blocks));
    }
    if let Some(ions) = &result.relaxed {
        let rows = |m: &crate::linalg::Matrix| -> Vec<Vec<f64>> {
            (0..m.rows)
                .map(|r| (0..m.cols).map(|c| m[(r, c)]).collect())
                .collect()
        };
        let static_dielectric: [[f64; 3]; 3] = std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                f64::from(u8::from(a == b)) + four_pi * ions.susceptibility[a][b]
            })
        });
        report.set("relaxed_susceptibility", ions.susceptibility);
        report.set("static_dielectric_tensor", static_dielectric);
        report.set("field_relaxation", rows(&ions.field_relaxation));
        report.set("strain_relaxation", rows(&ions.strain_relaxation));
        report.set(
            "smallest_optical_eigenvalue",
            ions.smallest_optical_eigenvalue,
        );
        if let Some(tensor) = &ions.second_susceptibility {
            report.set("relaxed_second_susceptibility", nested(tensor, 3, 1.0));
        }
        if let Some(tensor) = &ions.strain_susceptibility {
            let blocks: Vec<Value> = tensor.chunks(18).map(|row| nested(row, 3, 1.0)).collect();
            report.set("relaxed_strain_susceptibility", Value::Array(blocks));
            report.set(
                "relaxed_susceptibility_strain_derivative",
                with_volume(tensor, &ions.susceptibility),
            );
        }
        if let Some(tensor) = &ions.piezoelectric {
            report.set("relaxed_piezoelectric_tensor", *tensor);
        }
        if let Some(tensor) = &ions.second_piezoelectric {
            let blocks: Vec<Value> = tensor.chunks(36).map(|row| nested(row, 6, 1.0)).collect();
            report.set("relaxed_second_piezoelectric", Value::Array(blocks));
        }
    }
    report.set("units", units(&list));
    Ok(series_reference_fields(report, &result.reference))
}

/// Self-consistent field of a cell in a finite uniform electric field, with
/// the polarization of the single-point Berry phase and analytic forces.
fn finite_field(system: &System, controls: &Controls) -> Result<Value> {
    let method = PeriodicMethod::from_controls_with_dispersion(controls)?;
    let dispersion_model = method.dispersion_model_for(&system.numbers)?;
    let field = controls.opt_vec3("field")?.ok_or_else(|| {
        DftError::InvalidInput(
            "periodic-efield needs field: the uniform electric field in atomic units".into(),
        )
    })?;
    let forces = controls.bool("forces", false)?;
    controls.finish("periodic-efield")?;
    let (setup, scf, replicas) = gamma_supercell("periodic-efield", system, &method)?;
    let mut result = crate::response::nuclear::lattice_electronic_finite_field(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
        replicas,
        [field.x, field.y, field.z],
        forces,
        dispersion_model.as_ref(),
    )?;
    let dispersion = dispersion_model
        .as_ref()
        .map(|model| {
            let positions: Vec<_> = system
                .positions
                .iter()
                .map(|p| Vec3::new(p[0], p[1], p[2]))
                .collect();
            crate::periodic::dispersion::periodic_d3(
                &system.numbers,
                &positions,
                &lattice_rows_of(system)?,
                model,
            )
        })
        .transpose()?;
    if let Some(extra) = &dispersion {
        // Geometry-only dispersion is independent of the Berry polarization.
        // Add its primitive-cell enthalpy and forces to the stationary result.
        result.energy += extra.energy;
        if let Some(forces) = &mut result.forces {
            for (total, correction) in forces.iter_mut().zip(&extra.forces) {
                for (component, extra) in total.iter_mut().zip(correction.to_array()) {
                    *component += extra;
                }
            }
        }
    }
    let polarization = result.dipole.map(|v| v / result.volume);
    let mut report = reference_report(
        &setup,
        &scf,
        "self-consistent field of the cell in a finite uniform electric field: stationary \
         point of E - Omega E.P with the single-point Berry phase of the Gamma-point supercell \
         of the k mesh, started from the zero-field reference",
    )
    .with("supercell", method.k_mesh.to_vec())
    .with("field", result.field)
    .with("field_energy", result.energy)
    .with("dipole", result.dipole)
    .with("polarization", polarization)
    .with("volume", result.volume)
    .with("series_reference_residual", result.residual)
    .with("series_reference_iterations", result.iterations)
    .with("provenance", "analytic");
    let mut list = vec![
        (
            "energy",
            "hartree per supercell of the k mesh, zero-field reference",
        ),
        ("supercell", "integer"),
        ("field", "hartree/e/bohr"),
        (
            "field_energy",
            "hartree per primitive cell, E_KS - Omega E.P in the field",
        ),
        (
            "dipole",
            "e*bohr per primitive cell, modulo the lattice vectors",
        ),
        ("polarization", "e/bohr^2, modulo the polarization quantum"),
        ("volume", "bohr^3 per primitive cell"),
        ("series_reference_residual", "hartree"),
    ];
    if let Some(values) = &result.forces {
        report.set("forces", values.clone());
        list.push(("forces", "hartree/bohr"));
    }
    if let Some(extra) = &dispersion {
        include_dispersion(&mut report, &method, extra.energy * replicas as f64);
        report.set("dispersion_energy_per_primitive_cell", extra.energy);
        list.extend([
            ("dispersion_energy", "hartree per supercell of the k mesh"),
            (
                "dispersion_energy_per_primitive_cell",
                "hartree per primitive cell",
            ),
            ("disp_cutoff", "bohr"),
            ("cn_cutoff", "bohr"),
            ("abc_cutoff", "bohr"),
        ]);
    }
    report.set("units", units(&list));
    Ok(report)
}

/// The wave vectors of `q_points` (reduced coordinates of the reciprocal
/// lattice of the cell, three numbers each) as integer multiples of the
/// reciprocal basis vectors of the Gamma-point supercell of the k mesh.
fn commensurate_waves(id: &str, controls: &Controls, mesh: [usize; 3]) -> Result<Vec<[i64; 3]>> {
    let list = controls.opt_f64_list("q_points")?.ok_or_else(|| {
        DftError::InvalidInput(format!(
            "{id} needs q_points: reduced wave vectors, three numbers each, commensurate with \
             the k mesh"
        ))
    })?;
    if list.is_empty() || list.len() % 3 != 0 {
        return Err(DftError::InvalidInput(
            "q_points holds three reduced coordinates per wave vector".into(),
        ));
    }
    let mut waves = Vec::with_capacity(list.len() / 3);
    for q in list.chunks_exact(3) {
        let mut wave = [0_i64; 3];
        for axis in 0..3 {
            let scaled = q[axis] * mesh[axis] as f64;
            let nearest = scaled.round();
            if !scaled.is_finite() || (scaled - nearest).abs() > 1e-8 {
                return Err(DftError::InvalidInput(format!(
                    "the wave vector {q:?} is not commensurate with the k mesh {mesh:?}"
                )));
            }
            wave[axis] = nearest as i64;
        }
        if wave == [0, 0, 0] {
            return Err(DftError::InvalidInput(
                "q_points must differ from zero".into(),
            ));
        }
        waves.push(wave);
    }
    Ok(waves)
}

/// Self-consistent density response and inverse dielectric function at
/// wave vectors commensurate with the k mesh.
fn longitudinal_spin_response_feature(system: &System, controls: &Controls) -> Result<Value> {
    use crate::dfpt::density::PeriodicResponseOptions;
    use crate::dfpt::spin::periodic_longitudinal_spin_response;
    use crate::response::complex::Complex;
    let method = PeriodicMethod::from_controls(controls)?;
    let request = PeriodicResponseOptions {
        q: controls.opt_vec3("response_q")?.unwrap_or_else(Vec3::zero),
        frequency: super::polarizability::frequency(controls)?,
        reciprocal_cutoff: controls.f64("response_reciprocal_cutoff", 6.0)?,
        solver: super::vibrational::response_config(controls)?,
        fixed_number: false,
    };
    let field = controls
        .opt_f64_list("spin_probe_field")?
        .unwrap_or_else(|| vec![1.0, 0.0]);
    if field.len() != 2 {
        return Err(DftError::InvalidInput(
            "spin_probe_field requires [real, imaginary] in magnetic atomic units".into(),
        ));
    }
    let g = controls.f64("spin_g_factor", 2.0)?;
    controls.finish("periodic-spin-response")?;
    let setup = prepare_periodic(system, &method)?;
    let scf = run_periodic_scf(&setup.cell, &setup.functional, &setup.options)?;
    let response = periodic_longitudinal_spin_response(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
        &request,
        Complex::new(field[0], field[1]),
        g,
    )?;
    let densities = response
        .value
        .density_response
        .density
        .iter()
        .map(|spin| {
            spin.iter()
                .map(|p| Value::object().with("real", &p.re).with("imag", &p.im))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Ok(reference_report(&setup, &scf, response.model)
        .with("q", request.q)
        .with("frequency", request.frequency.energy()?)
        .with("spin_probe_field", response.value.magnetic_field)
        .with("spin_g_factor", g)
        .with("spin_numbers", response.value.spin_numbers)
        .with("total_number", response.value.total_number)
        .with("magnetic_moment", response.value.magnetic_moment)
        .with("spin_susceptibility", response.value.susceptibility)
        .with("density", densities)
        .with(
            "occupation_consistency_defect",
            response
                .value
                .density_response
                .occupation_consistency_defect,
        )
        .with(
            "response_temperature_hartree",
            response.value.density_response.temperature_hartree,
        )
        .with(
            "response_reciprocal_cutoff",
            response.value.density_response.reciprocal_cutoff,
        )
        .with(
            "response_fixed_number",
            response.value.density_response.fixed_number,
        )
        .with("response_residuals", response.residuals)
        .with("response_iterations", response.iterations)
        .with("axis_convention", response.axis_convention)
        .with("converged", scf.converged && response.converged)
        .with("warnings", response.warnings)
        .with(
            "units",
            units(&[
                ("energy", "hartree per primitive cell"),
                ("volume", "bohr^3"),
                ("q", "1/bohr"),
                ("frequency", "hartree"),
                ("spin_probe_field", "atomic units of magnetic field"),
                ("spin_g_factor", "dimensionless"),
                ("spin_numbers", "electrons per cell"),
                ("total_number", "electrons per cell"),
                (
                    "magnetic_moment",
                    "atomic units of magnetic moment per cell",
                ),
                (
                    "spin_susceptibility",
                    "atomic units of magnetic moment per cell per atomic magnetic field",
                ),
                ("density", "dimensionless"),
                ("occupation_consistency_defect", "dimensionless"),
                ("response_temperature_hartree", "hartree"),
                ("response_reciprocal_cutoff", "1/bohr"),
                ("response_residuals", "dimensionless"),
            ]),
        ))
}

fn dielectric_q(system: &System, controls: &Controls) -> Result<Value> {
    let method = PeriodicMethod::from_controls(controls)?;
    let config = super::vibrational::response_config(controls)?;
    let waves = commensurate_waves("dielectric-q", controls, method.k_mesh)?;
    controls.finish("dielectric-q")?;
    let (setup, scf, replicas) = gamma_supercell("dielectric-q", system, &method)?;
    let result = crate::response::nuclear::lattice_density_response(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
        replicas,
        &waves,
        &config,
    )?;
    let rows: Vec<Value> = result
        .points
        .iter()
        .map(|point| {
            let reduced: Vec<f64> = (0..3)
                .map(|axis| point.wave[axis] as f64 / method.k_mesh[axis] as f64)
                .collect();
            Value::object()
                .with("q_reduced", reduced)
                .with("q", point.vector)
                .with("chi", point.chi)
                .with("chi_cosine", point.cosine)
                .with("chi_sine", point.sine)
                .with("inverse_dielectric", point.inverse_dielectric)
                .with("dielectric", 1.0 / point.inverse_dielectric)
        })
        .collect();
    let mut list = vec![
        ("energy", "hartree per supercell of the k mesh"),
        ("supercell", "integer"),
        ("volume", "bohr^3 per primitive cell"),
        (
            "points",
            "per wave vector: q in bohr^-1; chi, chi_cosine, chi_sine in electrons/(hartree*bohr^3); inverse_dielectric = 1 + 4 pi chi/q^2 and dielectric dimensionless",
        ),
    ];
    list.extend(SERIES_REFERENCE_UNITS);
    let report = reference_report(
        &setup,
        &scf,
        "self-consistent density response chi(q, q) and inverse dielectric function at wave \
         vectors commensurate with the k mesh: second derivatives of the energy in the \
         amplitudes of cos(q.r) and sin(q.r) in the Gamma-point supercell",
    )
    .with("supercell", method.k_mesh.to_vec())
    .with("volume", result.volume)
    .with("points", rows)
    .with("path_evaluations", result.paths)
    .with("provenance", "analytic")
    .with("units", units(&list));
    Ok(series_reference_fields(report, &result.reference))
}

/// Orbital magnetic susceptibility at wave vectors commensurate with the
/// k mesh.
fn orbital_susceptibility_q(system: &System, controls: &Controls) -> Result<Value> {
    let mut method = PeriodicMethod::from_controls(controls)?;
    method.apply_collinear_reference(controls)?;
    let config = super::vibrational::response_config(controls)?;
    let waves = commensurate_waves("orbital-susceptibility-q", controls, method.k_mesh)?;
    controls.finish("orbital-susceptibility-q")?;
    let (setup, scf, replicas) = gamma_supercell("orbital-susceptibility-q", system, &method)?;
    let result = crate::response::nuclear::lattice_orbital_susceptibility(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
        replicas,
        &waves,
        &config,
    )?;
    let rows: Vec<_> = result
        .points
        .iter()
        .map(|point| orbital_q_row(point, method.k_mesh))
        .collect();
    let mut list = orbital_q_units();
    list.extend(SERIES_REFERENCE_UNITS);
    let report = reference_report(
        &setup,
        &scf,
        "orbital magnetic susceptibility chi(q) at wave vectors commensurate with the k mesh: \
         paramagnetic response of the Gamma-point supercell to transverse vector potentials, \
         with the diamagnetic term replaced through the sum rule of the uniform potential",
    )
    .with("supercell", method.k_mesh.to_vec())
    .with("volume", result.volume)
    .with("points", rows)
    .with("provenance", "analytic")
    .with("units", units(&list));
    Ok(series_reference_fields(report, &result.reference))
}

fn orbital_q_row(
    point: &crate::response::nuclear::OrbitalSusceptibility,
    mesh: [usize; 3],
) -> Value {
    let alpha2 = crate::constants::FINE_STRUCTURE * crate::constants::FINE_STRUCTURE;
    let reduced: [f64; 3] = std::array::from_fn(|axis| point.wave[axis] as f64 / mesh[axis] as f64);
    Value::object()
        .with("q_reduced", reduced)
        .with("q", point.vector)
        .with("polarization", point.polarization)
        .with("field_direction", point.field)
        .with("chi", point.chi)
        .with("chi_cgs", point.chi.map(|v| v * alpha2))
        .with(
            "chi_si",
            point.chi.map(|v| v * alpha2 * 4.0 * std::f64::consts::PI),
        )
        .with("paramagnetic", point.paramagnetic)
        .with("uniform", point.uniform)
        .with("response_residuals", point.response_residuals)
}

fn orbital_q_units() -> Vec<(&'static str, &'static str)> {
    vec![
        ("energy", "hartree per supercell of the k mesh"),
        ("supercell", "integer"),
        ("volume", "bohr^3 per primitive cell"),
        ("points.q_reduced", "dimensionless, reduced wave vector"),
        ("points.q", "bohr^-1"),
        ("points.polarization", "dimensionless, transverse polarization"),
        ("points.field_direction", "dimensionless, direction of the magnetic field"),
        (
            "points.chi",
            "atomic units of the volume susceptibility (E = -chi V B^2/2), per wave vector and transverse polarization",
        ),
        ("points.chi_cgs", "dimensionless, alpha^2 chi"),
        ("points.chi_si", "dimensionless, 4 pi alpha^2 chi"),
        (
            "points.paramagnetic",
            "atomic units of energy per unit squared vector potential",
        ),
        (
            "points.uniform",
            "atomic units of energy per unit squared vector potential",
        ),
        (
            "points.response_residuals",
            "dimensionless; polarization by cosine, sine and uniform response",
        ),
    ]
}

fn spinor_orbital_susceptibility_report(
    reference: &crate::periodic::spinor_scf::SpinorScfReport,
    points: &[crate::periodic::spinor_scf::SpinorOrbitalSusceptibilityPoint],
    mesh: [usize; 3],
    volume: f64,
    basis: &str,
    functional: &crate::xc::ResolvedFunctional,
    reference_type: &str,
) -> Value {
    let rows: Vec<_> = points
        .iter()
        .map(|point| {
            orbital_q_row(&point.orbital, mesh)
                .with("chi_transverse", point.chi_transverse)
                .with("paramagnetic_transverse", point.paramagnetic_transverse)
                .with("uniform_transverse", point.uniform_transverse)
                .with("response_iterations", point.response_iterations)
                .with("imaginary_defect", point.imaginary_defect)
                .with("reciprocity_defect", point.reciprocity_defect)
        })
        .collect();
    let mut list = orbital_q_units();
    list.extend([
        ("orbital_residual", "hartree"),
        ("occupation_consistency", "dimensionless"),
        ("particle_number_error", "electrons per supercell"),
        ("coulomb_operator_defect", "hartree"),
        ("xc_grid_number_defect", "electrons per supercell"),
        (
            "points.chi_transverse",
            "atomic units of volume susceptibility, in the two magnetic field axes",
        ),
        (
            "points.paramagnetic_transverse",
            "atomic units of energy per primitive cell per squared vector potential",
        ),
        (
            "points.uniform_transverse",
            "atomic units of energy per primitive cell per squared vector potential",
        ),
        (
            "points.response_iterations",
            "integer; polarization by cosine, sine and uniform response",
        ),
        (
            "points.imaginary_defect",
            "atomic units of unscaled energy second derivative",
        ),
        (
            "points.reciprocity_defect",
            "atomic units of unscaled energy second derivative",
        ),
    ]);
    Value::object()
        .with("model", "phase-averaged orbital susceptibility of an insulating Gamma-supercell spinor reference with the uniform-potential sum rule")
        .with("reference_type", reference_type)
        .with("functional", functional.description.as_str())
        .with("basis", basis)
        .with("energy", reference.fields.energy)
        .with("converged", reference.converged)
        .with("scf_iterations", reference.iterations)
        .with("orbital_residual", reference.orbital_residual)
        .with("occupation_consistency", reference.occupation_consistency)
        .with("particle_number_error", reference.particle_number_error)
        .with("coulomb_operator_defect", reference.fields.coulomb_operator_defect)
        .with("xc_grid_number_defect", reference.fields.xc_grid_number_defect)
        .with("supercell", mesh)
        .with("volume", volume)
        .with("points", rows)
        .with("provenance", "analytic")
        .with("units", units(&list))
}
/// Divide-and-conquer Kohn-Sham of a cell at Gamma with a minimum-image
/// partition, with forces and, for `dc-stress` or `stress`, the stress of
/// its free energy.
fn periodic_dc(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    let method = PeriodicMethod::from_controls(controls)?;
    let config = super::vibrational::response_config(controls)?;
    let fragments = controls.opt_usize_list("dc_fragments")?;
    let buffer = controls.f64("dc_buffer", 8.0)?;
    let smearing = controls.f64("dc_smearing", 1.0e-3)?;
    let forces = controls.bool("forces", false)?;
    let stress = controls.bool("stress", id == "dc-stress")?;
    controls.finish(id)?;
    if method.k_mesh != [1, 1, 1] {
        return Err(DftError::InvalidInput(format!(
            "{id} samples the cell at the Gamma point: use k_mesh 1,1,1 and a supercell"
        )));
    }
    let setup = prepare_periodic(system, &method)?;
    let natoms = setup.cell.molecule().len();
    // The temperature of the method, when it has one, is the Fermi function
    // of the subsystems.
    let kt = if method.electronic_temperature > 0.0 {
        method.electronic_temperature * crate::constants::BOLTZMANN_J_PER_K
            / crate::constants::HARTREE_TO_JOULE
    } else {
        smearing
    };
    let options = crate::scf::divide_conquer::DivideConquerOptions {
        fragment_of: fragments.unwrap_or_else(|| (0..natoms).collect()),
        buffer,
        kt,
    };
    let result = crate::response::nuclear::lattice_divide_conquer(
        &setup.cell,
        &setup.functional,
        &setup.options,
        &options,
        forces,
        stress,
        &config,
    )?;
    if !result.converged {
        return Err(DftError::ScfNotConverged {
            iterations: result.iterations,
            error: result.density_residual,
        });
    }
    let mut report = Value::object()
        .with(
            "model",
            "divide-and-conquer Kohn-Sham of the cell at Gamma: subsystem orbitals with a \
             common Fermi level, minimum-image buffers and partition-weighted density \
             assembly, on the functional of the integral series of the cell",
        )
        .with("reference_type", "divide-and-conquer")
        .with("functional", setup.functional.description.as_str())
        .with("basis", setup.method.basis.as_str())
        .with("converged", result.converged)
        .with("scf_iterations", result.iterations)
        .with("unrestricted", result.unrestricted)
        .with("n_alpha", result.n_alpha)
        .with("n_beta", result.n_beta)
        .with("k_mesh", method.k_mesh.to_vec())
        .with("volume", result.volume)
        .with("energy", result.energy)
        .with("internal_energy", result.internal_energy)
        .with("electronic_entropy", result.entropy)
        .with("chemical_potentials", result.chemical_potentials)
        .with("dc_buffer", buffer)
        .with("dc_smearing", kt)
        .with("fragment_of", options.fragment_of.clone())
        .with("subsystem_sizes", result.subsystem_sizes.clone())
        .with("density_residual", result.density_residual)
        .with("provenance", "analytic");
    let mut list = vec![
        (
            "energy",
            "hartree per cell (free energy: internal energy minus kT S)",
        ),
        ("internal_energy", "hartree per cell"),
        ("electronic_entropy", "dimensionless (S / k_B)"),
        ("chemical_potentials", "hartree"),
        ("volume", "bohr^3"),
        ("dc_buffer", "bohr"),
        ("dc_smearing", "hartree"),
        ("density_residual", "dimensionless"),
    ];
    if let Some(residual) = result.relaxation_residual {
        report.set("relaxation_residual", residual);
        list.push(("relaxation_residual", "dimensionless"));
    }
    if let Some(values) = &result.forces {
        report.set("forces", values.clone());
        list.push(("forces", "hartree/bohr"));
    }
    if let (Some(first), Some(symmetric)) = (&result.strain_derivative, &result.stress) {
        let gpa = crate::constants::HARTREE_PER_BOHR3_TO_GPA;
        report.set("strain_derivative", *first);
        report.set("stress", *symmetric);
        report.set("stress_gpa", symmetric.map(|row| row.map(|v| v * gpa)));
        list.push(("strain_derivative", "hartree, dA/d(epsilon)"));
        list.push(("stress", "hartree/bohr^3"));
        list.push(("stress_gpa", "GPa"));
    }
    report.set("units", units(&list));
    Ok(report)
}

/// Nuclear magnetic shielding in a periodic cell at wave vectors
/// commensurate with the k mesh.
fn periodic_nmr(system: &System, controls: &Controls) -> Result<Value> {
    let method = PeriodicMethod::from_controls(controls)?;
    let config = super::vibrational::response_config(controls)?;
    let waves = commensurate_waves("periodic-nmr-shielding", controls, method.k_mesh)?;
    let nuclei = controls.opt_usize_list("nuclei")?;
    controls.finish("periodic-nmr-shielding")?;
    let atoms = nuclei.unwrap_or_else(|| (0..system.numbers.len()).collect());
    let (setup, scf, replicas) = gamma_supercell("periodic-nmr-shielding", system, &method)?;
    let result = crate::response::nuclear::lattice_nmr_shielding(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
        replicas,
        &atoms,
        &waves,
        &config,
    )?;
    let ppm = 1.0e6;
    let rows: Vec<Value> = result
        .nuclei
        .iter()
        .map(|nucleus| {
            let points: Vec<Value> = nucleus
                .points
                .iter()
                .map(|point| {
                    let reduced: Vec<f64> = (0..3)
                        .map(|axis| point.wave[axis] as f64 / method.k_mesh[axis] as f64)
                        .collect();
                    Value::object()
                        .with("q_reduced", reduced)
                        .with("q", point.vector)
                        .with("polarization", point.polarization)
                        .with("field_direction", point.field)
                        .with(
                            "shielding_ppm",
                            point.shielding.map(|row| row.map(|v| v * ppm)),
                        )
                        .with(
                            "paramagnetic_ppm",
                            point.paramagnetic.map(|row| row.map(|v| v * ppm)),
                        )
                        .with(
                            "diamagnetic_ppm",
                            point.diamagnetic.map(|row| row.map(|v| v * ppm)),
                        )
                })
                .collect();
            Value::object()
                .with("atom", nucleus.atom)
                .with("element", system.numbers[nucleus.atom] as usize)
                .with("points", points)
                .with(
                    "shielding_tensor_ppm",
                    nucleus.tensor.map(|t| t.map(|row| row.map(|v| v * ppm))),
                )
                .with("isotropic_ppm", nucleus.isotropic.map(|v| v * ppm))
        })
        .collect();
    let mut list = vec![
        ("energy", "hartree per supercell of the k mesh"),
        ("supercell", "integer"),
        (
            "nuclei.points.q_reduced",
            "dimensionless, reduced wave vector",
        ),
        ("nuclei.points.q", "bohr^-1"),
        (
            "nuclei.points.polarization",
            "dimensionless, transverse polarization",
        ),
        (
            "nuclei.points.field_direction",
            "dimensionless, direction of the induced field",
        ),
        (
            "nuclei.points.shielding_ppm",
            "ppm, [polarization][component of the induced field]",
        ),
        ("nuclei.points.paramagnetic_ppm", "ppm, paramagnetic part"),
        ("nuclei.points.diamagnetic_ppm", "ppm, diamagnetic part"),
        (
            "nuclei.shielding_tensor_ppm",
            "ppm, [induced][applied], without the bulk magnetization (G = 0) term",
        ),
        (
            "nuclei.isotropic_ppm",
            "ppm, without the bulk magnetization (G = 0) term",
        ),
    ];
    list.extend(SERIES_REFERENCE_UNITS);
    let report = reference_report(
        &setup,
        &scf,
        "nuclear magnetic shielding in the cell at wave vectors commensurate with the k mesh: \
         mixed second derivative of the energy in a transverse vector potential wave centred \
         on the nucleus and in the Ewald lattice of nuclear moments, without the bulk \
         magnetization term",
    )
    .with("supercell", method.k_mesh.to_vec())
    .with("nuclei", rows)
    .with("provenance", "analytic")
    .with("units", units(&list));
    Ok(series_reference_fields(report, &result.reference))
}

/// Berry curvature dipole of the Fermi surface at a temperature and a
/// chemical potential or doping level.
fn berry_dipole(system: &System, controls: &Controls) -> Result<Value> {
    use crate::periodic::coherent_transport::{
        berry_curvature_dipole, doped_chemical_potential, CoherentModel,
    };
    let method = PeriodicMethod::from_controls(controls)?;
    let mesh = controls
        .opt_usize_list("transport_k_mesh")?
        .ok_or_else(|| DftError::InvalidInput("berry-dipole requires transport_k_mesh".into()))?;
    let temperature = controls.f64("transport_temperature_k", 300.0)?;
    let shift = controls.opt_f64("transport_mu_shift")?;
    let doping = controls.opt_f64("berry_dipole_doping")?;
    controls.finish("berry-dipole")?;
    if mesh.len() != 3 || mesh.contains(&0) || !(temperature > 0.0) {
        return Err(DftError::InvalidInput(
            "berry-dipole needs a positive three-dimensional mesh and a positive temperature"
                .into(),
        ));
    }
    if shift.is_some() && doping.is_some() {
        return Err(DftError::InvalidInput(
            "berry-dipole takes transport_mu_shift or berry_dipole_doping, not both".into(),
        ));
    }
    let setup = prepare_periodic(system, &method)?;
    let scf = run_reference(&setup)?;
    let model = CoherentModel::new(&setup.cell, &setup.functional, &scf, &setup.options)?;
    let points = setup.cell.k_points([mesh[0], mesh[1], mesh[2]]);
    let mut currents = Vec::new();
    for spin in 0..model.bands.spins {
        for k in &points {
            currents.push(model.point(spin, *k)?);
        }
    }
    let kt = temperature * crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
    let g = if model.bands.spins == 1 { 2.0 } else { 1.0 };
    let reference = reference_level(&scf, 0);
    let mu = match doping {
        Some(extra) => doped_chemical_potential(
            &currents,
            points.len(),
            g,
            (scf.n_alpha + scf.n_beta) as f64 + extra,
            kt,
        )?,
        None => reference + shift.unwrap_or(0.0),
    };
    let result = berry_curvature_dipole(
        &currents,
        setup.cell.lattice().volume().abs(),
        points.len(),
        g,
        mu,
        kt,
    )?;
    let mut report = reference_report(
        &setup,
        &scf,
        "Berry curvature dipole of the Fermi surface: -df/de weighted trace of the band \
         velocity and the Berry curvature of every degenerate level, from the physical \
         velocity matrices of the converged bands on the transport mesh",
    )
    .with("transport_k_mesh", mesh)
    .with("temperature_k", temperature)
    .with("chemical_potential", mu)
    .with("reference_level", reference)
    .with("electrons_per_cell", result.electrons)
    .with("berry_curvature_dipole", result.dipole)
    .with("fermi_weight", result.fermi_weight)
    .with(
        "raw_current_hermitian_residual",
        result.current_hermitian_residual,
    )
    .with("provenance", "analytic");
    if let Some(extra) = doping {
        report.set("berry_dipole_doping", extra);
    }
    report.set(
        "units",
        units(&[
            ("energy", "hartree per cell"),
            ("temperature_k", "kelvin"),
            ("chemical_potential", "hartree"),
            ("reference_level", "hartree"),
            ("electrons_per_cell", "electrons"),
            (
                "berry_curvature_dipole",
                "dimensionless, D_ab as [a][b] with a the derivative direction and b the curvature component",
            ),
            ("fermi_weight", "1/hartree/bohr^3"),
            ("berry_dipole_doping", "electrons per cell"),
            ("raw_current_hermitian_residual", "hartree*bohr"),
        ]),
    );
    Ok(report)
}

/// Options of the dimensional Coulomb backend from the periodic reference,
/// grid and charge controls. It reads its controls, so it runs before
/// `Controls::finish`.
fn dimensional_scf_options(
    controls: &Controls,
    method: &PeriodicMethod,
    pbc: [bool; 3],
) -> Result<crate::periodic::scf_boundary::DimensionalScfOptions> {
    use crate::periodic::electrostatics::ChargeCompensation;
    use crate::periodic::scf_boundary::DimensionalScfOptions;
    let compensation = controls.str("charge_compensation", "none")?;
    let center = controls.opt_f64_list("countercharge_center")?;
    let width = controls.opt_f64("countercharge_width")?;
    let mut options =
        DimensionalScfOptions::for_boundary(crate::periodic::features::PeriodicBoundary::new(pbc));
    options.basis_cutoff = method.overlap_cutoff;
    options.grid_cutoff = controls.f64("grid_cutoff", 12.0)?;
    options.nuclear_widths = controls.opt_f64_list("nuclear_widths")?.unwrap_or_default();
    options.scf.k_mesh = method.k_mesh;
    options.scf.occupation = if method.electronic_temperature == 0.0 {
        OccupationScheme::Insulator
    } else {
        OccupationScheme::FermiDirac {
            temperature_kelvin: method.electronic_temperature,
        }
    };
    options.scf.reference = if method.unrestricted {
        crate::scf::Reference::Uhf
    } else {
        crate::scf::Reference::Auto
    };
    options.scf.guess = match method.guess.to_ascii_lowercase().as_str() {
        "core" => crate::scf::Guess::Core,
        "gwh" => crate::scf::Guess::Gwh,
        "sad" => crate::scf::Guess::Sad,
        other => {
            return Err(DftError::InvalidInput(format!(
                "unknown periodic guess {other}"
            )))
        }
    };
    options.scf.max_iter = method.max_iter;
    options.scf.e_tol = method.conv_tol;
    options.scf.p_tol = method.conv_tol_dp;
    options.scf.conv_tol_grad = method
        .conv_tol_grad
        .unwrap_or_else(|| method.conv_tol.sqrt());
    options.scf.grid.level = method.grid_level;
    options.scf.grid.preset = method.grid_preset;
    options.scf.grid.scheme = method.grid_angular_scheme;
    options.scf.grid.prune = method.grid_prune;
    options.scf.grid.weight_cutoff = method.grid_weight_cutoff;
    options.scf.grid_shell = method.grid_shell;
    options.scf.exchange_cutoff = method.exchange_cutoff;
    options.scf.use_adiis = method.adiis;
    options.scf.adiis_until_error = method.adiis_until_error;
    options.scf.diis_size = method.diis_size;
    options.scf.level_shift = method.level_shift;
    options.scf.damping = method.damping;
    options.electrostatics.alpha = controls.opt_f64("ewald_alpha")?;
    options.electrostatics.real_cutoff = controls.opt_f64("ewald_real_cutoff")?;
    options.electrostatics.reciprocal_cutoff = controls.opt_f64("ewald_reciprocal_cutoff")?;
    options.electrostatics.quadrature_order = controls.usize("electrostatic_order", 96)?;
    options.electrostatics.compensation = match compensation.as_str() {
        "none" if center.is_none() && width.is_none() => ChargeCompensation::RequireNeutral,
        "uniform" if center.is_none() && width.is_none() => ChargeCompensation::UniformBackground,
        "gaussian" => {
            let c = center.ok_or_else(|| {
                DftError::InvalidInput("Gaussian compensation requires countercharge_center".into())
            })?;
            if c.len() != 3 {
                return Err(DftError::InvalidInput(
                    "countercharge_center requires three Bohr coordinates".into(),
                ));
            }
            ChargeCompensation::GaussianCountercharge {
                center: Vec3::new(c[0], c[1], c[2]),
                width: width.ok_or_else(|| {
                    DftError::InvalidInput(
                        "Gaussian compensation requires countercharge_width".into(),
                    )
                })?,
            }
        }
        _ => {
            return Err(DftError::InvalidInput(
                "charge_compensation must be none, uniform or gaussian with its \
                 center/width; unrelated compensation controls cannot be ignored"
                    .into(),
            ))
        }
    };
    Ok(options)
}

/// The smallest supercell that holds a spiral of reduced wave vector `q`:
/// the replicated structure, its k mesh, the repetitions and the integer
/// indices of the wave vector in the reciprocal lattice of the supercell.
fn spiral_supercell(
    system: &System,
    mesh: [usize; 3],
    q: &[f64],
    limit: usize,
) -> Result<(System, [usize; 3], [usize; 3], [f64; 3])> {
    let rows = system.cell.ok_or_else(|| {
        DftError::InvalidInput("a spin spiral needs a cell (lattice rows, Bohr)".into())
    })?;
    let mut cells = [1usize; 3];
    let mut indices = [0.0; 3];
    let mut super_mesh = mesh;
    for a in 0..3 {
        if !system.pbc[a] {
            if q[a] != 0.0 {
                return Err(DftError::InvalidInput(
                    "spiral_q on open axes must be zero; reduced components label only \
                     physical periodic axes"
                        .into(),
                ));
            }
            continue;
        }
        let n = (1..=limit)
            .find(|n| {
                let cycles = q[a] * *n as f64;
                (cycles - cycles.round()).abs() <= 1e-9
            })
            .ok_or_else(|| {
                DftError::InvalidInput(format!(
                    "spiral_q component {} is not a multiple of 1/n for any n up to \
                     spiral_max_supercell = {limit}",
                    q[a]
                ))
            })?;
        if mesh[a] % n != 0 {
            return Err(DftError::InvalidInput(format!(
                "k_mesh {mesh:?} must be divisible by the {n} cells of the spiral along axis \
                 {a} so that the supercell samples the same Brillouin zone"
            )));
        }
        cells[a] = n;
        indices[a] = (q[a] * n as f64).round();
        super_mesh[a] = mesh[a] / n;
    }
    let mut numbers = Vec::new();
    let mut positions = Vec::new();
    for i in 0..cells[0] {
        for j in 0..cells[1] {
            for k in 0..cells[2] {
                let shift: [f64; 3] = std::array::from_fn(|c| {
                    i as f64 * rows[0][c] + j as f64 * rows[1][c] + k as f64 * rows[2][c]
                });
                for (z, p) in system.numbers.iter().zip(&system.positions) {
                    numbers.push(*z);
                    positions.push(std::array::from_fn(|c| p[c] + shift[c]));
                }
            }
        }
    }
    let cell = std::array::from_fn(|a| rows[a].map(|v| v * cells[a] as f64));
    Ok((
        System {
            numbers,
            positions,
            cell: Some(cell),
            pbc: system.pbc,
        },
        super_mesh,
        cells,
        indices,
    ))
}

/// Energy of a flat spin spiral in the supercell that holds it and, for
/// `dmi`, the part of its spin-orbit energy that is odd in the chirality.
fn spin_spiral(id: &str, system: &System, controls: &Controls) -> Result<Value> {
    use crate::periodic::spinor_scf::{
        from_pauli, pauli_components, periodic_pauli_operators, run_spinor_scf,
        BiasedSpinorBackend, SpinorDftBackend, SpinorEnergyBackend, SpinorScfConfig,
        SpinorScfReport,
    };
    use crate::response::complex::Complex;
    let dmi = id == "dmi";
    let method = PeriodicMethod::from_controls(controls)?;
    let q = controls.opt_f64_list("spiral_q")?.ok_or_else(|| {
        DftError::InvalidInput(format!(
            "{id} requires spiral_q, the wave vector of the spiral in reduced coordinates of \
             the reciprocal lattice"
        ))
    })?;
    let axis = controls
        .opt_vec3("spiral_axis")?
        .unwrap_or_else(|| Vec3::new(0.0, 0.0, 1.0));
    let seed = controls.f64("spiral_seed_field", 0.05)?;
    let limit = controls.usize("spiral_max_supercell", 12)?;
    let soc_scale = controls.f64("nuclear_soc_scale", if dmi { 1.0 } else { 0.0 })?;
    let self_consistent = dmi && controls.bool("dmi_self_consistent", false)?;
    let mut options = dimensional_scf_options(controls, &method, system.pbc)?;
    controls.finish(id)?;
    if q.len() != 3
        || q.iter().any(|v| !v.is_finite())
        || !(axis.norm().is_finite() && axis.norm() > 0.0)
        || !(seed.is_finite() && seed != 0.0)
        || !soc_scale.is_finite()
        || limit == 0
        || system.pbc == [false; 3]
    {
        return Err(DftError::InvalidInput(format!(
            "{id} requires at least one periodic axis, three finite reduced components of \
             spiral_q, a non-zero spiral_axis, a non-zero finite spiral_seed_field and a \
             positive spiral_max_supercell"
        )));
    }
    if method.multiplicity != 1
        || controls.contains("spin")
        || method.unrestricted
        || method.level_shift != 0.0
        || !method.guess.eq_ignore_ascii_case("core")
        || controls.contains("adiis_until_error")
        || (controls.contains("no_adiis") && method.adiis)
    {
        return Err(DftError::InvalidInput(
            "spinor SCF fixes total particle number, not separate spin populations; \
             use the core guess and CDIIS with zero level shift"
                .into(),
        ));
    }
    let (super_system, super_mesh, cells, indices) =
        spiral_supercell(system, method.k_mesh, &q, limit)?;
    let replicas: usize = cells.iter().product();
    options.scf.k_mesh = super_mesh;
    let geometry = dimensional_geometry_of(&super_system)?;
    let mol = super::method::molecule_from(
        &super_system.numbers,
        &super_system.positions,
        method.charge * replicas as i32,
        method.multiplicity,
    )?;
    let set = super::method::build_basis(&method.basis, method.ecp.as_deref())?;
    let basis = crate::basis::MolBasis::build(&mol, &set)?;
    let functional = resolve(&FunctionalSpec::Combined(method.xc.clone()))?;
    let points = geometry.k_points(options.scf.k_mesh)?;
    let b = geometry.reciprocal();
    let wave = b[0] * indices[0] + b[1] * indices[1] + b[2] * indices[2];
    // Right-handed frame (e1, e2, n): along the wave vector the seed field
    // turns from e1 towards e2 for the direct chirality.
    let n = axis.normalized();
    let trial = if n.get(0).abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let e1 = (trial - n * trial.dot(n)).normalized();
    let e2 = n.cross(e1);
    // sigma.B/2 of B = e1 cos(q.r) + chirality e2 sin(q.r).
    let texture = |chirality: f64| -> Result<Vec<BlochMatrix>> {
        let field =
            std::array::from_fn(|a| Complex::new(0.5 * e1.get(a), -0.5 * chirality * e2.get(a)));
        periodic_pauli_operators(
            &basis,
            &geometry,
            &points,
            options.basis_cutoff,
            wave,
            Complex::ZERO,
            field,
            0.0,
        )
    };
    let direct = texture(1.0)?;
    let mirrored = texture(-1.0)?;
    let defaults = SpinorScfConfig::default();
    let config = SpinorScfConfig {
        max_iter: if controls.contains("max_iter") {
            method.max_iter
        } else {
            defaults.max_iter
        },
        energy_tolerance: if controls.contains("conv_tol") {
            method.conv_tol
        } else {
            defaults.energy_tolerance
        },
        density_tolerance: if controls.contains("conv_tol_dp") {
            method.conv_tol_dp
        } else {
            defaults.density_tolerance
        },
        orbital_tolerance: method.conv_tol_grad.unwrap_or(defaults.orbital_tolerance),
        damping: method.damping,
        diis_size: method.diis_size,
        occupation: options.scf.occupation,
        ..defaults
    };
    let size = 2 * basis.nao_sph;
    let zero = vec![BlochMatrix::zeros(size, size); points.len()];
    let base = SpinorDftBackend::new(
        &mol,
        &basis,
        &geometry,
        &functional,
        &options,
        None,
        zero.clone(),
    )?;
    let coupled = if soc_scale != 0.0 {
        Some(
            SpinorDftBackend::new(&mol, &basis, &geometry, &functional, &options, None, zero)?
                .with_nuclear_soc(soc_scale)?,
        )
    } else {
        None
    };
    let scale = 1.0 / replicas as f64;
    let cells_k = (points.len() * replicas) as f64;
    // Integrated spin density along a texture, per cell of the structure.
    let projection = |density: &[BlochMatrix], operator: &[BlochMatrix]| -> f64 {
        2.0 * density
            .iter()
            .zip(operator)
            .map(|(p, v)| p.trace_product(v))
            .sum::<f64>()
            / cells_k
    };
    let polarization = |density: &[BlochMatrix]| -> Result<Vec<f64>> {
        let mut total = vec![0.0; 3];
        for (p, s) in density.iter().zip(base.overlap()) {
            let pc = pauli_components(p)?;
            let sc = pauli_components(s)?;
            for (a, value) in total.iter_mut().enumerate() {
                *value += 0.5 * pc[a + 1].trace_product(&sc[0]) / cells_k;
            }
        }
        Ok(total)
    };
    let settled = |report: SpinorScfReport| -> Result<SpinorScfReport> {
        if report.converged {
            Ok(report)
        } else {
            Err(DftError::ScfNotConverged {
                iterations: report.iterations,
                error: report.orbital_residual,
            })
        }
    };
    let mut bias = direct.clone();
    for v in &mut bias {
        v.scale(seed);
    }
    let seeded = settled(run_spinor_scf(
        &BiasedSpinorBackend::new(&base, Some(bias), None)?,
        &config,
    )?)?;
    let free = settled(run_spinor_scf(
        &BiasedSpinorBackend::new(&base, None, Some(seeded.fields.fock.clone()))?,
        &config,
    )?)?;
    let mut report = Value::object()
        .with(
            "model",
            "Flat spin spiral of wave vector q in the smallest supercell that holds it: \
             variational two-component Bloch SCF seeded by the Zeeman field sigma.B/2 of \
             B = h [e1 cos(q.r) + e2 sin(q.r)], then converged again without the seed from \
             the seeded state; no generalized Bloch condition, so the one-electron Pauli \
             nuclear spin-orbit term can be included; energies per cell of the structure",
        )
        .with("functional", functional.description.as_str())
        .with("spiral_q", q.clone())
        .with("spiral_wave_vector", wave)
        .with("spiral_supercell", cells.to_vec())
        .with("spiral_axis", n)
        .with("spiral_frame", vec![e1, e2, n])
        .with("spiral_seed_field", seed)
        .with("nuclear_soc_scale", soc_scale)
        .with("dimension", geometry.dimension())
        .with("pbc", system.pbc.to_vec())
        .with("seeded_energy", seeded.fields.energy * scale)
        .with("seed_energy", seeded.fields.energy_components[9] * scale)
        .with(
            "seeded_spin_projection",
            projection(&seeded.density, &direct),
        )
        .with("seeded_scf_iterations", seeded.iterations)
        .with(
            "units",
            units(&[
                ("energy", "hartree per cell"),
                ("free_energy", "hartree per cell"),
                ("entropy_term", "hartree per cell"),
                ("seeded_energy", "hartree per cell"),
                ("seed_energy", "hartree per cell"),
                ("nuclear_soc_energy", "hartree per cell"),
                ("nuclear_soc_energies", "hartree per cell"),
                ("dmi_energy", "hartree per cell"),
                ("soc_even_energy", "hartree per cell"),
                ("dmi_per_wave_number", "hartree*bohr"),
                ("chirality_rotation_energy_defect", "hartree per cell"),
                ("self_consistent_energies", "hartree per cell"),
                ("dmi_energy_self_consistent", "hartree per cell"),
                ("dmi_per_wave_number_self_consistent", "hartree*bohr"),
                ("self_consistent_orbital_residual", "hartree"),
                ("spiral_q", "reduced coordinates of the reciprocal lattice"),
                ("spiral_wave_vector", "1/bohr"),
                ("spiral_axis", "dimensionless"),
                ("spiral_frame", "dimensionless"),
                ("spiral_seed_field", "atomic units of magnetic field"),
                ("nuclear_soc_scale", "dimensionless"),
                (
                    "seeded_spin_projection",
                    "electron spin polarization, Pauli expectation",
                ),
                (
                    "spiral_spin_projection",
                    "electron spin polarization, Pauli expectation",
                ),
                (
                    "opposite_spin_projection",
                    "electron spin polarization, Pauli expectation",
                ),
                (
                    "spin_projections",
                    "electron spin polarization, Pauli expectation",
                ),
                (
                    "self_consistent_spin_projections",
                    "electron spin polarization, Pauli expectation",
                ),
                (
                    "magnetization",
                    "electron spin polarization, Pauli expectation",
                ),
                ("chemical_potential", "hartree"),
                ("orbital_residual", "hartree"),
                ("density_change", "dimensionless"),
                ("energy_change", "hartree"),
                ("particle_number_error", "electrons"),
                ("nuclear_soc_quadrature_defect", "hartree"),
                ("nuclear_soc_hermitian_defect", "hartree"),
            ]),
        );
    if !dmi {
        let last = match &coupled {
            Some(backend) => {
                let start = backend.evaluate(&free.density)?.fock;
                settled(run_spinor_scf(
                    &BiasedSpinorBackend::new(backend, None, Some(start))?,
                    &config,
                )?)?
            }
            None => free,
        };
        report.set("energy", last.fields.energy * scale);
        report.set("free_energy", last.free_energy * scale);
        report.set("entropy_term", last.entropy_term * scale);
        report.set("nuclear_soc_energy", last.fields.nuclear_soc_energy * scale);
        report.set("spiral_spin_projection", projection(&last.density, &direct));
        report.set(
            "opposite_spin_projection",
            projection(&last.density, &mirrored),
        );
        report.set("magnetization", polarization(&last.density)?);
        report.set(
            "chemical_potential",
            last.chemical_potential.map_or(Value::Null, Value::Float),
        );
        report.set("converged", last.converged);
        report.set("backend_converged", last.fields.operator_converged);
        report.set("scf_iterations", last.iterations);
        report.set("orbital_residual", last.orbital_residual);
        report.set("density_change", last.density_change);
        report.set("energy_change", last.energy_change);
        report.set("particle_number_error", last.particle_number_error);
        return Ok(report);
    }
    let coupled = coupled.ok_or_else(|| {
        DftError::InvalidInput("dmi requires a non-zero nuclear_soc_scale".into())
    })?;
    // Without spin-orbit coupling the opposite chirality is the rotation of
    // every spin by pi about e1, so no second SCF is needed for it.
    let reversed = free
        .density
        .iter()
        .map(|p| {
            let c = pauli_components(p)?;
            let mut turned: [BlochMatrix; 4] = std::array::from_fn(|m| {
                if m == 0 {
                    c[0].clone()
                } else {
                    BlochMatrix::zeros(c[0].rows(), c[0].cols())
                }
            });
            for a in 0..3 {
                for (other, part) in c[1..].iter().enumerate() {
                    let delta = if a == other { 1.0 } else { 0.0 };
                    turned[a + 1].add_scaled(part, 2.0 * e1.get(a) * e1.get(other) - delta);
                }
            }
            from_pauli(&turned)
        })
        .collect::<Result<Vec<_>>>()?;
    let rotation_defect = (base.evaluate(&reversed)?.energy - free.fields.energy).abs() * scale;
    let pair = [
        coupled.evaluate(&free.density)?,
        coupled.evaluate(&reversed)?,
    ];
    let first = [
        pair[0].nuclear_soc_energy * scale,
        pair[1].nuclear_soc_energy * scale,
    ];
    let length = wave.norm();
    let per_wave = |energy: f64| {
        if length > 0.0 {
            Value::Float(energy / length)
        } else {
            Value::Null
        }
    };
    let odd = 0.5 * (first[0] - first[1]);
    report.set("energy", free.fields.energy * scale);
    report.set("nuclear_soc_energies", first.to_vec());
    report.set("dmi_energy", odd);
    report.set("soc_even_energy", 0.5 * (first[0] + first[1]));
    report.set("dmi_per_wave_number", per_wave(odd));
    report.set("chirality_rotation_energy_defect", rotation_defect);
    report.set(
        "spin_projections",
        vec![
            projection(&free.density, &direct),
            projection(&reversed, &mirrored),
        ],
    );
    report.set(
        "chirality_convention",
        "dmi_energy = [E(direct) - E(mirrored)]/2 of the nuclear spin-orbit energy; direct: \
         the spin texture turns counterclockwise about spiral_axis along the wave vector; \
         first order in the spin-orbit scale on the states converged without it",
    );
    report.set("converged", free.converged);
    report.set(
        "backend_converged",
        free.fields.operator_converged && pair.iter().all(|f| f.operator_converged),
    );
    report.set("scf_iterations", free.iterations);
    report.set("orbital_residual", free.orbital_residual);
    report.set("density_change", free.density_change);
    report.set("energy_change", free.energy_change);
    report.set("particle_number_error", free.particle_number_error);
    report.set(
        "nuclear_soc_quadrature_defect",
        coupled.nuclear_soc_diagnostics()[0],
    );
    report.set(
        "nuclear_soc_hermitian_defect",
        coupled.nuclear_soc_diagnostics()[1],
    );
    if self_consistent {
        let mut energies = vec![0.0; 2];
        let mut projections = vec![0.0; 2];
        let mut iterations = vec![0usize; 2];
        let mut residual = 0.0_f64;
        for (side, (fields, operator)) in pair.iter().zip([&direct, &mirrored]).enumerate() {
            let state = settled(run_spinor_scf(
                &BiasedSpinorBackend::new(&coupled, None, Some(fields.fock.clone()))?,
                &config,
            )?)?;
            energies[side] = state.fields.energy * scale;
            projections[side] = projection(&state.density, operator);
            iterations[side] = state.iterations;
            residual = residual.max(state.orbital_residual);
        }
        let full = 0.5 * (energies[0] - energies[1]);
        report.set("self_consistent_energies", energies);
        report.set("dmi_energy_self_consistent", full);
        report.set("dmi_per_wave_number_self_consistent", per_wave(full));
        report.set("self_consistent_spin_projections", projections);
        report.set("self_consistent_scf_iterations", iterations);
        report.set("self_consistent_orbital_residual", residual);
    }
    Ok(report)
}

/// Hybrid Wannier charge centres and their flow, per spin channel.
fn wannier_flow_feature(system: &System, controls: &Controls) -> Result<Value> {
    use crate::periodic::coherent_transport::wannier_flow;
    let method = PeriodicMethod::from_controls(controls)?;
    let axes = controls
        .opt_usize_list("topology_plane_axes")?
        .unwrap_or_else(|| vec![0, 1]);
    let fixed = controls.f64("topology_fixed_k", 0.0)?;
    let loop_divisions = controls.usize("wannier_loop_divisions", 12)?;
    let flow_points = controls.usize("wannier_flow_points", 12)?;
    controls.finish("wannier-flow")?;
    if axes.len() != 2
        || axes.iter().any(|a| *a >= 3)
        || axes[0] == axes[1]
        || loop_divisions < 2
        || flow_points < 2
        || method.electronic_temperature != 0.0
    {
        return Err(DftError::InvalidInput(
            "the Wannier flow requires two distinct lattice axes 0..2 (loop axis, flow axis), \
             at least two loop divisions and flow points and a zero-temperature insulating \
             reference"
                .into(),
        ));
    }
    let setup = prepare_periodic(system, &method)?;
    let scf = run_reference(&setup)?;
    let model = crate::periodic::band_transport::BandModel::new(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
    )?;
    let occupied = [scf.n_alpha, scf.n_beta];
    let weight = if model.spins == 1 { 2.0 } else { 1.0 };
    let mut rows = Vec::new();
    let mut winding = 0.0;
    for spin in 0..model.spins {
        if occupied[spin] == 0 {
            continue;
        }
        let result = wannier_flow(
            &setup.cell,
            &model,
            spin,
            occupied[spin],
            loop_divisions,
            flow_points,
            [axes[0], axes[1]],
            fixed,
        )?;
        winding += weight * result.winding;
        rows.push(
            Value::object()
                .with("spin", spin)
                .with("occupied_rank", occupied[spin])
                .with("flow", result.flow)
                .with("centres", result.centres)
                .with("total", result.total)
                .with("winding", result.winding)
                .with("sampled_min_gap", result.sampled_min_gap)
                .with(
                    "min_link_log_abs_determinant",
                    result.min_link_log_abs_determinant,
                ),
        );
    }
    let report = reference_report(
        &setup,
        &scf,
        "hybrid Wannier charge centres of the occupied multiplet: minus the eigenphases over \
         2 pi of the product of the occupied-subspace links around the loop axis as a function of the transverse \
         wave vector; the winding of their sum is the Chern number of the slice",
    );
    Ok(report
        .with("topology_plane_axes", axes)
        .with("topology_fixed_k", fixed)
        .with("wannier_loop_divisions", loop_divisions)
        .with("wannier_flow_points", flow_points)
        .with("winding", winding)
        .with("spin_weight", weight)
        .with("fock_image_residual", model.fock_image_residual)
        .with("results", rows)
        .with(
            "units",
            units(&[
                (
                    "results.flow",
                    "reduced reciprocal coordinate along the flow axis",
                ),
                (
                    "results.centres",
                    "units of the lattice vector of the loop axis, [flow point][band]",
                ),
                ("results.total", "units of the lattice vector, continuous"),
                ("results.winding", "dimensionless, no integer rounding"),
                ("results.sampled_min_gap", "Hartree"),
                (
                    "results.min_link_log_abs_determinant",
                    "dimensionless logarithm",
                ),
                ("winding", "dimensionless, includes the spin weight"),
                (
                    "topology_fixed_k",
                    "reduced reciprocal coordinate of the axis outside the plane",
                ),
                ("fock_image_residual", "Hartree"),
            ]),
        ))
}

fn bands(system: &System, controls: &Controls) -> Result<Value> {
    let method = PeriodicMethod::from_controls(controls)?;
    let vertices = controls.opt_rows("path")?.ok_or_else(|| {
        DftError::InvalidInput(
            "bands needs path: reduced k vertices, e.g. [[0,0,0],[0.5,0,0.5]]".into(),
        )
    })?;
    let points = controls.usize("points", 20)?;
    let labels = controls.opt_str("labels")?;
    controls.finish("bands")?;
    let setup = prepare_periodic(system, &method)?;
    let scf = run_reference(&setup)?;
    let lattice = setup.cell.lattice();
    let reduced_path = path_points(&vertices, points)?;
    let path: Vec<Vec3> = reduced_path
        .iter()
        .map(|f| cartesian(lattice, *f))
        .collect();
    let focks = crate::periodic::scf::reference_fock_at(
        &setup.cell,
        &setup.functional,
        &scf,
        &setup.options,
        &path,
    )?;
    let spins = if scf.unrestricted { 2 } else { 1 };
    let mut channels = Vec::new();
    for spin in 0..spins {
        let mut energies = Vec::with_capacity(path.len());
        for (ik, k) in path.iter().enumerate() {
            let overlap = BlochMatrix::from_pair(setup.cell.overlap(*k));
            energies.push(band_energies(&focks[spin][ik], &overlap)?);
        }
        channels.push(
            Value::object()
                .with("spin", spin)
                .with("band_energies", energies),
        );
    }
    let mut distance = vec![0.0; path.len()];
    for i in 1..path.len() {
        distance[i] = distance[i - 1] + (path[i] - path[i - 1]).norm();
    }
    Ok(reference_report(
        &setup,
        &scf,
        "band structure of the converged periodic Fock operator along a reduced k path",
    )
    .with("path_vertices", vertices.clone())
    .with("labels", labels)
    .with(
        "k_path",
        path.iter().map(|k| k.to_array()).collect::<Vec<_>>(),
    )
    .with("k_path_reduced", reduced_path.to_vec())
    .with("path_distance", distance)
    .with("channels", channels)
    .with("chemical_potential", scf.chemical_potential.to_vec())
    .with(
        "units",
        units(&[
            ("channels.band_energies", "hartree, [k][band]"),
            ("k_path", "1/bohr"),
            ("path_distance", "1/bohr"),
        ]),
    ))
}

#[cfg(test)]
mod supercell_tests {
    use super::*;

    #[test]
    fn replicas_preserve_charge_and_collinear_spin_populations() {
        let system = System {
            numbers: vec![3],
            positions: vec![[0.; 3]],
            cell: Some([[9., 0., 0.], [0., 9., 0.], [0., 0., 9.]]),
            pbc: [true; 3],
        };
        for (charge, multiplicity, expected_charge, expected_multiplicity) in
            [(1, 1, 2, 1), (0, 2, 0, 3)]
        {
            let controls = Controls::from_value(
                &Value::object()
                    .with("xc", "hf")
                    .with("charge", charge)
                    .with("multiplicity", multiplicity)
                    .with("k_mesh", vec![2usize, 1, 1]),
            )
            .unwrap();
            let method = PeriodicMethod::from_controls(&controls).unwrap();
            let (replicated, method) = supercell_system(&system, &method, [2, 1, 1]).unwrap();
            assert_eq!(replicated.numbers, vec![3, 3]);
            assert_eq!(method.charge, expected_charge);
            assert_eq!(method.multiplicity, expected_multiplicity);
            assert_eq!(method.k_mesh, [1; 3]);
        }
    }
}

#[cfg(test)]
mod xc_ao_cutoff_contract {
    use super::PeriodicMethod;
    use crate::driver::{Controls, Value};
    #[test]
    fn optional_xc_ao_radius_preserves_default_and_accepts_positive_lengths() {
        let empty = Controls::from_value(&Value::object()).unwrap();
        assert_eq!(
            PeriodicMethod::from_controls(&empty).unwrap().xc_ao_cutoff,
            None
        );
        let explicit = Controls::from_value(&Value::object().with("xc_ao_cutoff", 16.0)).unwrap();
        assert_eq!(
            PeriodicMethod::from_controls(&explicit)
                .unwrap()
                .xc_ao_cutoff,
            Some(16.0)
        );
    }
    #[test]
    fn invalid_xc_ao_radii_fail_before_any_basis_or_reference_is_built() {
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let controls =
                Controls::from_value(&Value::object().with("xc_ao_cutoff", value)).unwrap();
            assert!(matches!(
                PeriodicMethod::from_controls(&controls),
                Err(crate::DftError::InvalidInput(_))
            ));
        }
    }
}
