// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Electronic-structure method selection shared by every surface.
//!
//! [`MolecularMethod`] collects the reference controls (functional, basis,
//! charge, multiplicity, SCF, grid, dispersion, field and continuum
//! solvation) from [`Controls`], and [`prepare_molecular`] builds the
//! molecule, basis, functional, grid and SCF options from them. The grammar of
//! every control is the one the CLI and Python have always used: a solvent
//! spec takes an optional `smd:`/`cosmo:`/`ief:`/`ssvpe:` prefix, and
//! functionals with a built-in dispersion term refuse a second one.

use super::controls::Controls;
use super::value::Value;
use crate::basis::{BasisSet, MolBasis};
use crate::dispersion::Dispersion;
use crate::grid::{GridOptions, MolecularGrid};
use crate::math::Vec3;
use crate::scf::{run_scf, ScfOptions, ScfResult};
use crate::system::{Atom, Molecule};
use crate::xc::{resolve, FunctionalSpec, ResolvedFunctional, WeightedFunctionalTerm};
use crate::{DftError, Result};

/// Reference controls of a molecular calculation. Positions are Bohr.
#[derive(Clone, Debug)]
pub struct MolecularMethod {
    pub xc: Option<String>,
    pub x: Option<String>,
    pub c: Option<String>,
    pub xc_terms: Option<Vec<WeightedFunctionalTerm>>,
    pub omega: Option<f64>,
    /// Named functional parameter overrides (`xc_parameters`).
    pub xc_parameters: Vec<(String, f64)>,
    pub basis: String,
    pub ecp: Option<String>,
    pub charge: i32,
    pub multiplicity: usize,
    pub unrestricted: bool,
    pub occupation: Option<crate::scf::occupations::DeterminantOccupation>,
    pub fixed_occupations: Option<crate::scf::occupations::FixedOccupations>,
    pub guess_mix: Option<f64>,
    pub grid_level: usize,
    /// Named grid (`fine`, `ultrafine`, `superfine`) replacing the level.
    pub grid_preset: Option<crate::grid::GridPreset>,
    /// Angular rule family (`lebedev`, default, or `product`).
    pub grid_angular_scheme: crate::grid::AngularScheme,
    /// Quadrature points whose partitioned weight is below this (Bohr^3)
    /// are dropped; zero keeps every point, so the energy is a smooth
    /// function of the nuclear positions.
    pub grid_weight_cutoff: f64,
    pub dispersion: String,
    /// Explicit atomic references and scale for the geometry-only D1 model.
    pub d1_parameters: Option<(Vec<crate::dispersion::d1::AtomParameters>, f64)>,
    pub density_fit: Option<String>,
    pub max_iter: usize,
    pub conv_tol: f64,
    pub conv_tol_grad: Option<f64>,
    pub conv_tol_dp: f64,
    pub incore_mb: usize,
    pub vv10_grid_level: usize,
    /// Model file of the learned functional selected by `xc = skala`.
    pub skala_model: Option<String>,
    pub field: Option<Vec3>,
    pub solvent: Option<String>,
    pub eps: Option<f64>,
    pub guess: Option<String>,
    pub level_shift: Option<f64>,
    pub damping: Option<f64>,
    pub diis_size: Option<usize>,
    pub adiis: bool,
    pub soscf: bool,
    pub orbital_optimizer: crate::scf::OrbitalOptimizer,
    /// Follow internal orbital instabilities before the requested property.
    pub stabilize: bool,
    /// Electronic temperature in Kelvin; positive values give Fermi-Dirac
    /// occupations and a Mermin free-energy reference.
    pub electronic_temperature: f64,
    /// Fractional alpha/beta electron additions (need a positive electronic
    /// temperature).
    pub electron_offset: [f64; 2],
}

/// `xc_parameters`: an object of functional parameter names and values.
fn xc_parameters(controls: &Controls) -> Result<Vec<(String, f64)>> {
    match controls.take("xc_parameters") {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Object(pairs)) => pairs
            .iter()
            .map(|(name, value)| match value.as_f64() {
                Some(v) if v.is_finite() => Ok((name.clone(), v)),
                _ => Err(DftError::InvalidInput(format!(
                    "xc_parameters `{name}` must be a finite number"
                ))),
            })
            .collect(),
        Some(other) => Err(DftError::InvalidInput(format!(
            "xc_parameters must be an object of parameter names and numbers, got {}",
            other.to_json()
        ))),
    }
}

pub(super) fn xc_terms(controls: &Controls) -> Result<Option<Vec<WeightedFunctionalTerm>>> {
    let Some(value) = controls.take("xc_terms") else {
        return Ok(None);
    };
    let entries = value.as_array().ok_or_else(|| {
        DftError::InvalidInput("xc_terms must be an array of weighted functional terms".into())
    })?;
    entries
        .iter()
        .map(|entry| {
            let term = Controls::from_value(entry)?;
            let weight = term.opt_f64("weight")?.ok_or_else(|| {
                DftError::InvalidInput("each xc_terms entry requires weight".into())
            })?;
            let functional = match (term.opt_str("xc")?, term.opt_str("x")?, term.opt_str("c")?) {
                (Some(name), None, None) => FunctionalSpec::Combined(name),
                (None, Some(x), Some(c)) => FunctionalSpec::Split { x, c },
                _ => {
                    return Err(DftError::InvalidInput(
                        "each xc_terms entry requires either xc or both x and c".into(),
                    ))
                }
            };
            term.finish("xc_terms")?;
            Ok(WeightedFunctionalTerm { weight, functional })
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

impl Default for MolecularMethod {
    fn default() -> Self {
        Self {
            xc: None,
            x: None,
            c: None,
            xc_terms: None,
            omega: None,
            xc_parameters: Vec::new(),
            basis: "sto-3g".into(),
            ecp: None,
            charge: 0,
            multiplicity: 1,
            unrestricted: false,
            occupation: None,
            fixed_occupations: None,
            guess_mix: None,
            grid_level: 3,
            grid_preset: None,
            grid_angular_scheme: crate::grid::AngularScheme::Lebedev,
            grid_weight_cutoff: 1e-14,
            dispersion: "auto".into(),
            d1_parameters: None,
            density_fit: None,
            max_iter: 150,
            conv_tol: 1e-9,
            conv_tol_grad: None,
            conv_tol_dp: 1e-8,
            incore_mb: 2048,
            vv10_grid_level: 3,
            skala_model: None,
            field: None,
            solvent: None,
            eps: None,
            guess: None,
            level_shift: None,
            damping: None,
            diis_size: None,
            adiis: true,
            soscf: true,
            orbital_optimizer: crate::scf::OrbitalOptimizer::Auto,
            stabilize: false,
            electronic_temperature: 0.0,
            electron_offset: [0.0; 2],
        }
    }
}

/// Names of every method control, for help text and documentation.
pub const MOLECULAR_METHOD_CONTROLS: &[&str] = &[
    "xc",
    "x",
    "c",
    "xc_terms",
    "omega",
    "xc_parameters",
    "basis",
    "ecp",
    "charge",
    "multiplicity",
    "unrestricted",
    "occupation_mode",
    "occupied_alpha",
    "occupied_beta",
    "occupations_alpha",
    "occupations_beta",
    "guess_mix",
    "grid_level",
    "grid_preset",
    "grid_angular_scheme",
    "grid_weight_cutoff",
    "dispersion",
    "dispersion_free_c6",
    "dispersion_free_radii",
    "dispersion_energy_scale",
    "density_fit",
    "max_iter",
    "conv_tol",
    "conv_tol_grad",
    "conv_tol_dp",
    "incore_mb",
    "vv10_grid_level",
    "skala_model",
    "field",
    "reference_field",
    "solvent",
    "eps",
    "guess",
    "level_shift",
    "damping",
    "diis_size",
    "no_adiis",
    "no_soscf",
    "scf_optimizer",
    "stabilize",
    "electronic_temperature",
    "electron_offset",
];

impl MolecularMethod {
    /// Read the method controls (and only those) from `controls`.
    pub fn from_controls(controls: &Controls) -> Result<Self> {
        let d = Self::default();
        if controls.contains("multiplicity") && controls.contains("mult") {
            return Err(DftError::InvalidInput(
                "give multiplicity or its alias mult, not both".into(),
            ));
        }
        let multiplicity = match controls.opt_usize("multiplicity")? {
            Some(m) => Some(m),
            None => controls.opt_usize("mult")?,
        };
        let occupied = [
            controls.opt_usize_list("occupied_alpha")?,
            controls.opt_usize_list("occupied_beta")?,
        ];
        let occupation_mode = controls
            .str("occupation_mode", "aufbau")?
            .to_ascii_lowercase();
        let occupation = match occupation_mode.as_str() {
            "aufbau" | "fixed" if occupied.iter().all(Option::is_none) => None,
            "mom" | "imom" => {
                let mut indices: [Vec<usize>; 2] = std::array::from_fn(|_| Vec::new());
                for spin in 0..2 {
                    let values = occupied[spin].as_ref().ok_or_else(|| DftError::InvalidInput("MOM/IMOM needs complete occupied_alpha and occupied_beta lists (zero-based initial reference columns)".into()))?;
                    for &index in values {
                        if indices[spin].contains(&index) {
                            return Err(DftError::InvalidInput(
                                "occupied orbital indices cannot be repeated within a spin".into(),
                            ));
                        }
                        indices[spin].push(index);
                    }
                }
                Some(crate::scf::occupations::DeterminantOccupation {
                    occupied: indices,
                    reference: if occupation_mode == "mom" {
                        crate::scf::occupations::OverlapReference::Previous
                    } else {
                        crate::scf::occupations::OverlapReference::Initial
                    },
                })
            }
            _ => return Err(DftError::InvalidInput(
                "occupation_mode is aufbau, fixed, mom or imom; explicit occupied lists require mom/imom"
                    .into(),
            )),
        };
        let populations = [
            controls.opt_f64_list("occupations_alpha")?,
            controls.opt_f64_list("occupations_beta")?,
        ];
        let fixed_occupations = if occupation_mode == "fixed" {
            Some(crate::scf::occupations::FixedOccupations {
                populations: [
                    populations[0].clone().ok_or_else(|| {
                        DftError::InvalidInput(
                            "fixed mode needs occupations_alpha and occupations_beta".into(),
                        )
                    })?,
                    populations[1].clone().ok_or_else(|| {
                        DftError::InvalidInput(
                            "fixed mode needs occupations_alpha and occupations_beta".into(),
                        )
                    })?,
                ],
            })
        } else {
            if populations.iter().any(Option::is_some) {
                return Err(DftError::InvalidInput(
                    "explicit orbital populations require occupation_mode=fixed".into(),
                ));
            }
            None
        };
        if controls.contains("field") && controls.contains("reference_field") {
            return Err(DftError::InvalidInput(
                "use either field or reference_field for the reference electric field".into(),
            ));
        }
        let field = if controls.contains("reference_field") {
            controls.opt_vec3("reference_field")?
        } else {
            controls.opt_vec3("field")?
        };
        let dispersion = match controls.opt_str("dispersion")? {
            Some(model) if controls.contains("disp") => {
                return Err(DftError::InvalidInput(format!(
                    "give dispersion or its alias disp, not both (dispersion={model})"
                )))
            }
            Some(model) => model,
            None => controls.str("disp", &d.dispersion)?,
        };
        let d1_parameters = dispersion
            .trim()
            .eq_ignore_ascii_case("d1")
            .then(|| super::density_dispersion::d1_inputs(controls))
            .transpose()?;
        let method = Self {
            xc: controls.opt_str("xc")?,
            x: controls.opt_str("x")?,
            c: controls.opt_str("c")?,
            xc_terms: xc_terms(controls)?,
            omega: controls.opt_f64("omega")?,
            xc_parameters: xc_parameters(controls)?,
            basis: controls.str("basis", &d.basis)?,
            ecp: controls.opt_str("ecp")?,
            charge: controls
                .opt_i64("charge")?
                .map(|c| {
                    i32::try_from(c)
                        .map_err(|_| DftError::InvalidInput("charge out of range".into()))
                })
                .transpose()?
                .unwrap_or(0),
            multiplicity: multiplicity.unwrap_or(1),
            unrestricted: controls.bool("unrestricted", false)?,
            occupation,
            fixed_occupations,
            guess_mix: controls.opt_f64("guess_mix")?,
            grid_level: controls.usize("grid_level", d.grid_level)?,
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
            grid_weight_cutoff: {
                let cutoff = controls.f64("grid_weight_cutoff", 1e-14)?;
                if !cutoff.is_finite() || cutoff < 0.0 {
                    return Err(DftError::InvalidInput(
                        "grid_weight_cutoff must be finite and non-negative".into(),
                    ));
                }
                cutoff
            },
            dispersion,
            d1_parameters,
            density_fit: controls.opt_str("density_fit")?,
            max_iter: controls.usize("max_iter", d.max_iter)?,
            conv_tol: controls.f64("conv_tol", d.conv_tol)?,
            conv_tol_grad: controls.opt_f64("conv_tol_grad")?,
            conv_tol_dp: controls.f64("conv_tol_dp", d.conv_tol_dp)?,
            incore_mb: controls.usize("incore_mb", d.incore_mb)?,
            vv10_grid_level: controls.usize("vv10_grid_level", d.vv10_grid_level)?,
            skala_model: controls.opt_str("skala_model")?,
            field,
            solvent: controls.opt_str("solvent")?,
            eps: controls.opt_f64("eps")?,
            guess: controls.opt_str("guess")?,
            level_shift: controls.opt_f64("level_shift")?,
            damping: controls.opt_f64("damping")?,
            diis_size: controls.opt_usize("diis_size")?,
            adiis: !controls.bool("no_adiis", false)?,
            soscf: !controls.bool("no_soscf", false)?,
            orbital_optimizer: match controls.str("scf_optimizer", "auto")?.as_str() {
                "auto" => crate::scf::OrbitalOptimizer::Auto,
                "trah" => crate::scf::OrbitalOptimizer::Trah,
                "legacy" => crate::scf::OrbitalOptimizer::Legacy,
                other => {
                    return Err(DftError::InvalidInput(format!(
                        "scf_optimizer must be auto, trah or legacy; got {other}"
                    )))
                }
            },
            stabilize: controls.bool("stabilize", false)?,
            electronic_temperature: controls.f64("electronic_temperature", 0.0)?,
            electron_offset: match controls.opt_f64_list("electron_offset")? {
                None => [0.0; 2],
                Some(v) if v.len() == 2 => [v[0], v[1]],
                Some(_) => {
                    return Err(DftError::InvalidInput(
                        "electron_offset takes two numbers (alpha, beta)".into(),
                    ))
                }
            },
        };
        Ok(method)
    }

    pub fn functional_spec(&self) -> Result<FunctionalSpec> {
        if self.xc_terms.is_some() {
            return Err(DftError::InvalidInput(
                "weighted XC uses xc_terms as its functional specification".into(),
            ));
        }
        match (&self.xc, &self.x, &self.c) {
            (Some(name), None, None) => Ok(FunctionalSpec::Combined(name.clone())),
            (None, Some(x), Some(c)) => Ok(FunctionalSpec::Split {
                x: x.clone(),
                c: c.clone(),
            }),
            (None, None, None) => Ok(FunctionalSpec::Combined("hf".into())),
            _ => Err(DftError::InvalidInput(
                "give either xc, or both x and c".into(),
            )),
        }
    }

    /// The name used for functional-specific dispersion parameters.
    pub fn dispersion_functional_name(&self) -> String {
        self.xc
            .clone()
            .or_else(|| {
                let (x, c) = self.x.as_ref().zip(self.c.as_ref())?;
                (crate::xc::normalize(x) == crate::xc::normalize(c)
                    && crate::xc::built_in_dispersion(x).1.is_some())
                .then(|| x.clone())
            })
            .unwrap_or_else(|| "hf".into())
    }
}

/// SCF options and the empirical dispersion model for a functional name.
#[allow(clippy::too_many_arguments)]
pub fn scf_options(
    xc: &str,
    dispersion: &str,
    density_fit: Option<&str>,
    max_iter: usize,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    conv_tol_dp: f64,
    incore_mb: usize,
    vv10_grid_level: usize,
) -> Result<(ScfOptions, Dispersion)> {
    if max_iter == 0 {
        return Err(DftError::InvalidInput("max_iter must be at least 1".into()));
    }
    for (name, value) in [("conv_tol", conv_tol), ("conv_tol_dp", conv_tol_dp)] {
        if !(value.is_finite() && value > 0.0) {
            return Err(DftError::InvalidInput(format!(
                "{name} must be finite and positive"
            )));
        }
    }
    if let Some(value) = conv_tol_grad {
        if !(value.is_finite() && value > 0.0) {
            return Err(DftError::InvalidInput(
                "conv_tol_grad must be finite and positive".into(),
            ));
        }
    }
    if incore_mb == 0 {
        return Err(DftError::InvalidInput(
            "incore_mb must be at least 1".into(),
        ));
    }
    if vv10_grid_level > 9 {
        return Err(DftError::InvalidInput(
            "vv10_grid_level must be between 0 and 9".into(),
        ));
    }
    let (vv10, empirical) = resolve_molecular_dispersion(xc, dispersion)?;
    let opts = ScfOptions {
        max_iter,
        e_tol: conv_tol,
        conv_tol_grad: conv_tol_grad.unwrap_or_else(|| conv_tol.sqrt()),
        p_tol: conv_tol_dp,
        incore_mb,
        density_fit: density_fit.map(str::to_owned),
        vv10_grid_level,
        vv10,
        ..ScfOptions::default()
    };
    Ok((opts, empirical))
}

/// Resolve geometry and density dispersion consistently across molecular entry points.
pub(crate) fn resolve_molecular_dispersion(
    xc: &str,
    dispersion: &str,
) -> Result<(Option<crate::nlc::Vv10Params>, Dispersion)> {
    let (builtin_vv10, builtin) = crate::xc::built_in_dispersion(xc);
    let requested = dispersion.trim().to_ascii_lowercase();
    if let Some(term) = builtin {
        let allow_disabled = term != crate::xc::RequiredNonlocalTerm::D3Zero;
        if requested != "auto" && !(allow_disabled && requested == "none") {
            let choices = if allow_disabled {
                "use dispersion='auto', or 'none' only for debugging"
            } else {
                "use dispersion='auto'"
            };
            return Err(DftError::InvalidInput(format!(
                "`{xc}` includes {} by definition; {choices}",
                term.name()
            )));
        }
    }
    match requested.as_str() {
        "auto" => Ok((
            builtin_vv10,
            builtin
                .and_then(crate::xc::RequiredNonlocalTerm::pair_dispersion)
                .unwrap_or_default(),
        )),
        "vv10" | "vv10-scf" => Ok((Some(crate::nlc::Vv10Params::ORIGINAL), Dispersion::None)),
        "vv10-post" => Err(DftError::InvalidInput(
            "this route uses variational self-consistent VV10; use dispersion='vv10'".into(),
        )),
        other => Dispersion::parse(other)
            .map(|model| (None, model))
            .ok_or_else(|| {
                DftError::InvalidInput(format!("unknown dispersion model `{dispersion}`"))
            }),
    }
}

/// Resolve a solvent spec: an optional `smd:`/`cosmo:`/`ief:`/`ssvpe:`
/// prefix selects the continuum model (C-PCM by default); the remaining name
/// is a bundled solvent, or `custom` with an explicit `eps`. Returns the PCM
/// options and whether the SMD non-electrostatic term applies.
pub fn resolve_solvent(
    solvent: Option<&str>,
    eps: Option<f64>,
) -> Result<(Option<crate::pcm::PcmOptions>, bool)> {
    let mut smd = false;
    let mut p = crate::pcm::PcmOptions::default();
    let mut want_pcm = false;
    if let Some(spec) = solvent {
        want_pcm = true;
        let mut name = spec;
        if let Some(rest) = name.strip_prefix("smd:") {
            smd = true;
            name = rest;
        } else if let Some(rest) = name.strip_prefix("cosmo:") {
            p.model = crate::pcm::PcmModel::Cosmo;
            name = rest;
        } else if let Some(rest) = name.strip_prefix("ief:") {
            p.model = crate::pcm::PcmModel::IefPcm;
            name = rest;
        } else if let Some(rest) = name.strip_prefix("ssvpe:") {
            p.model = crate::pcm::PcmModel::Ssvpe;
            name = rest;
        }
        // SMD electrostatics uses its own intrinsic, unscaled Coulomb radii.
        if smd {
            p.smd_radii = true;
            p.vdw_scale = 1.0;
        }
        if name.eq_ignore_ascii_case("custom") {
            if eps.is_none() {
                return Err(DftError::InvalidInput(
                    "solvent 'custom' needs eps; a custom solvent without bundled \
                     descriptors gets electrostatics only (no SMD non-electrostatic term)"
                        .into(),
                ));
            }
        } else {
            match crate::pcm::solvent_by_name(name) {
                Some(s) => {
                    p.eps = s.eps;
                    p.descriptors = Some(s);
                }
                None => {
                    return Err(DftError::InvalidInput(format!(
                        "unknown solvent `{name}` ({} bundled)",
                        crate::pcm::solvent_names().len()
                    )));
                }
            }
        }
    }
    if let Some(e) = eps {
        p.eps = e;
        want_pcm = true;
    }
    Ok((if want_pcm { Some(p) } else { None }, smd))
}

pub fn build_basis(basis: &str, ecp: Option<&str>) -> Result<BasisSet> {
    let mut set = crate::basis::mixed::load_spec(basis)?;
    if let Some(spec) = ecp {
        crate::basis::mixed::apply_ecp_spec(&mut set, spec)?;
    }
    Ok(set)
}

pub fn molecule_from(
    numbers: &[u8],
    positions: &[[f64; 3]],
    charge: i32,
    multiplicity: usize,
) -> Result<Molecule> {
    if numbers.is_empty() {
        return Err(DftError::InvalidInput(
            "at least one atom is required".into(),
        ));
    }
    if numbers.len() != positions.len() {
        return Err(DftError::InvalidInput(
            "numbers and positions must have the same length".into(),
        ));
    }
    if multiplicity == 0 {
        return Err(DftError::InvalidInput(
            "multiplicity must be at least 1".into(),
        ));
    }
    for (i, (&z, p)) in numbers.iter().zip(positions).enumerate() {
        if !(1..=118).contains(&z) {
            return Err(DftError::InvalidInput(format!(
                "atom {i} has invalid atomic number {z}"
            )));
        }
        if p.iter().any(|x| !x.is_finite()) {
            return Err(DftError::InvalidInput(format!(
                "atom {i} has a non-finite Cartesian coordinate"
            )));
        }
    }
    Ok(Molecule::new(
        numbers
            .iter()
            .zip(positions)
            .map(|(&z, p)| Atom {
                z,
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    )
    .with_charge(charge)
    .with_multiplicity(multiplicity))
}

/// Everything needed to run or differentiate a molecular reference.
pub struct MolecularSetup {
    pub method: MolecularMethod,
    pub molecule: Molecule,
    pub basis_set: BasisSet,
    pub basis: MolBasis,
    pub functional: ResolvedFunctional,
    pub grid_options: GridOptions,
    pub grid: Option<MolecularGrid>,
    pub options: ScfOptions,
    pub dispersion: Dispersion,
    /// The SMD non-electrostatic solvent, when SMD was requested.
    pub smd: Option<crate::pcm::SmdSolvent>,
}

impl MolecularSetup {
    /// Reuse the XC grid, or build the requested quadrature for a property
    /// that needs numerical integration even on a grid-free HF reference.
    pub fn integration_grid(&self) -> Result<std::borrow::Cow<'_, MolecularGrid>> {
        match &self.grid {
            Some(grid) => Ok(std::borrow::Cow::Borrowed(grid)),
            None => Ok(std::borrow::Cow::Owned(MolecularGrid::build(
                &self.molecule,
                &self.grid_options,
            )?)),
        }
    }

    /// Project the resolved electronic SCF settings into the complex spinor solver.
    /// Algorithm-specific collinear controls, such as ADIIS and level shifting,
    /// still belong to the collinear seed; they are not spinor solver options.
    pub fn spinor_scf_config(&self) -> crate::periodic::spinor_scf::SpinorScfConfig {
        use crate::periodic::occupations::{OccupationScheme, BOLTZMANN_HARTREE_PER_K};
        crate::periodic::spinor_scf::SpinorScfConfig {
            max_iter: self.options.max_iter,
            orbital_optimizer: if self.options.use_soscf {
                self.options.orbital_optimizer
            } else {
                crate::scf::OrbitalOptimizer::Legacy
            },
            energy_tolerance: self.options.e_tol,
            density_tolerance: self.options.p_tol,
            orbital_tolerance: self.options.conv_tol_grad,
            overlap_cutoff: self.options.orth_threshold,
            damping: self.options.damping,
            diis_size: self.options.diis_size,
            occupation: if self.options.electronic_temperature > 0.0 {
                OccupationScheme::FermiDirac {
                    temperature_kelvin: self.options.electronic_temperature
                        / BOLTZMANN_HARTREE_PER_K,
                }
            } else {
                OccupationScheme::Insulator
            },
        }
    }

    /// Preserve the historical integer-SOC solver defaults, while applying
    /// explicitly requested common settings to the nested SCF as well.
    /// Presence matters even when the value equals a collinear default.
    /// The SOC reference constructor separately rejects smeared occupations.
    pub fn nested_soc_scf_config(
        &self,
        controls: &Controls,
    ) -> crate::periodic::spinor_scf::SpinorScfConfig {
        let resolved = self.spinor_scf_config();
        let mut config = crate::periodic::spinor_scf::SpinorScfConfig::default();
        if controls.contains("max_iter") {
            config.max_iter = resolved.max_iter;
        }
        if controls.contains("conv_tol") {
            config.energy_tolerance = resolved.energy_tolerance;
        }
        if controls.contains("conv_tol_dp") {
            config.density_tolerance = resolved.density_tolerance;
        }
        if controls.contains("conv_tol_grad") {
            config.orbital_tolerance = resolved.orbital_tolerance;
        }
        if controls.contains("damping") {
            config.damping = resolved.damping;
        }
        if controls.contains("diis_size") {
            config.diis_size = resolved.diis_size;
        }
        config
    }
}

pub fn prepare_molecular(
    numbers: &[u8],
    positions: &[[f64; 3]],
    method: &MolecularMethod,
) -> Result<MolecularSetup> {
    if method.grid_level > 9 {
        return Err(DftError::InvalidInput(
            "grid_level must be between 0 and 9".into(),
        ));
    }
    let molecule = molecule_from(numbers, positions, method.charge, method.multiplicity)?;
    let basis_set = build_basis(&method.basis, method.ecp.as_deref())?;
    let basis = MolBasis::build(&molecule, &basis_set)?;
    // `xc = skala` selects the learned functional of a model file instead of
    // a registered semilocal functional.
    let learned = method
        .xc
        .as_deref()
        .is_some_and(|xc| xc.eq_ignore_ascii_case("skala"));
    if learned != method.skala_model.is_some() {
        return Err(DftError::InvalidInput(
            "xc = skala and skala_model = <model file> are given together".into(),
        ));
    }
    let mut functional = if let Some(terms) = &method.xc_terms {
        if method.xc.is_some() || method.x.is_some() || method.c.is_some() {
            return Err(DftError::InvalidInput(
                "xc_terms cannot be combined with xc, x or c".into(),
            ));
        }
        if !matches!(
            method.dispersion.trim().to_ascii_lowercase().as_str(),
            "auto" | "none" | "d1"
        ) || method
            .solvent
            .as_deref()
            .is_some_and(|s| s.to_ascii_lowercase().starts_with("smd:"))
        {
            return Err(DftError::Unsupported {
                feature: "weighted XC composition".into(),
                missing: "functional-specific dispersion or SMD parameters for the composition"
                    .into(),
            });
        }
        ResolvedFunctional::weighted(terms)?
    } else if learned {
        ResolvedFunctional::model_based("Skala learned functional (model file)")
    } else {
        resolve(&method.functional_spec()?)?
    };
    if let Some(omega) = method.omega {
        functional = functional.with_omega(omega)?;
    }
    if !method.xc_parameters.is_empty() {
        functional = functional.with_parameters(&method.xc_parameters)?;
    }
    // The non-local layers of a learned functional sum over the unpartitioned
    // atomic grids, so a point dropped for its small partition weight would
    // make the energy discontinuous in the nuclear positions.
    let grid_options = GridOptions {
        level: method.grid_level,
        preset: method.grid_preset,
        scheme: method.grid_angular_scheme,
        weight_cutoff: if learned {
            0.0
        } else {
            method.grid_weight_cutoff
        },
        ..GridOptions::default()
    };
    let grid = if functional.parts.is_empty() && !learned {
        None
    } else {
        Some(MolecularGrid::build(&molecule, &grid_options)?)
    };
    let d1 = if method.dispersion.trim().eq_ignore_ascii_case("d1") {
        let (atoms, scale) = method.d1_parameters.as_ref().ok_or_else(|| {
            DftError::InvalidInput("D1 requires atomic references and a positive scale".into())
        })?;
        if atoms.len() != numbers.len() || numbers.contains(&0) {
            return Err(DftError::InvalidInput(
                "D1 requires one atomic reference per nucleus and no ghost centres".into(),
            ));
        }
        crate::dispersion::d1::validate_parameters(atoms, *scale)?;
        if let Some(term) = functional.required_nonlocal_term() {
            return Err(DftError::InvalidInput(format!(
                "D1 cannot be combined with a functional that already requires {term}"
            )));
        }
        Some(Dispersion::D1 {
            atoms: atoms.clone(),
            s6: *scale,
        })
    } else {
        None
    };
    let (mut options, empirical) = scf_options(
        &method.dispersion_functional_name(),
        if d1.is_some() {
            "none"
        } else {
            &method.dispersion
        },
        method.density_fit.as_deref(),
        method.max_iter,
        method.conv_tol,
        method.conv_tol_grad,
        method.conv_tol_dp,
        method.incore_mb,
        method.vv10_grid_level,
    )?;
    let dispersion = d1.unwrap_or(empirical);
    options.external_field = method.field;
    options.vv10_grid_weight_cutoff = method.grid_weight_cutoff;
    let (pcm, smd) = resolve_solvent(method.solvent.as_deref(), method.eps)?;
    let smd = if smd {
        Some(match pcm.as_ref().and_then(|o| o.descriptors) {
            Some(d)
                if !method
                    .solvent
                    .as_deref()
                    .unwrap_or("")
                    .eq_ignore_ascii_case("smd:water") =>
            {
                crate::pcm::SmdSolvent::NonAqueous(d)
            }
            _ => crate::pcm::SmdSolvent::Water,
        })
    } else {
        None
    };
    options.pcm = pcm;
    if method.unrestricted {
        options.reference = crate::scf::Reference::Uhf;
    }
    if let Some(v) = method.guess_mix {
        options.guess_mix = v;
    }
    if let Some(g) = &method.guess {
        options.guess = match g.to_ascii_lowercase().as_str() {
            "sad" => crate::scf::guess::Guess::Sad,
            "gwh" => crate::scf::guess::Guess::Gwh,
            "core" => crate::scf::guess::Guess::Core,
            other => {
                return Err(DftError::InvalidInput(format!(
                    "unknown guess `{other}` (sad|gwh|core)"
                )))
            }
        };
    }
    if let Some(v) = method.level_shift {
        options.level_shift = v;
    }
    if let Some(v) = method.damping {
        options.damping = v;
    }
    if let Some(v) = method.diis_size {
        options.diis_size = v;
    }
    options.use_adiis = method.adiis;
    options.use_soscf = method.soscf;
    options.orbital_optimizer = method.orbital_optimizer;
    if let Some(path) = &method.skala_model {
        let model = crate::skala::SkalaModel::load(std::path::Path::new(path))?;
        options.skala = Some(std::sync::Arc::new(model));
    }
    if !(method.electronic_temperature >= 0.0) {
        return Err(DftError::InvalidInput(
            "electronic_temperature must be non-negative (Kelvin)".into(),
        ));
    }
    options.electronic_temperature =
        method.electronic_temperature * crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
    options.electron_offset = method.electron_offset;
    options.occupation = method.occupation.clone();
    options.fixed_occupations = method.fixed_occupations.clone();
    if method.fixed_occupations.is_some()
        && (method.stabilize
            || method.occupation.is_some()
            || method.electronic_temperature != 0.0
            || method.electron_offset != [0.0; 2])
    {
        return Err(DftError::InvalidInput("fixed populations cannot be combined with ground-state stability descent, Fermi smearing, electron offsets or integer tracking".into()));
    }
    if method.occupation.is_some()
        && (method.stabilize
            || method.electronic_temperature != 0.0
            || method.electron_offset != [0.0; 2])
    {
        return Err(DftError::InvalidInput("integer MOM/IMOM cannot be combined with ground-state stability descent, Fermi smearing or fractional electron offsets".into()));
    }
    if method.stabilize && method.electronic_temperature > 0.0 {
        return Err(DftError::Unsupported {
            feature: "stability following of a Fermi-smeared reference".into(),
            missing: "the occupation-response terms of the orbital Hessian".into(),
        });
    }
    crate::logger::log(&format!(
        "method basis={} functional={} grid_level={}",
        method.basis, functional.description, method.grid_level
    ));
    Ok(MolecularSetup {
        method: method.clone(),
        molecule,
        basis_set,
        basis,
        functional,
        grid_options,
        grid,
        options,
        dispersion,
        smd,
    })
}

/// A converged (and, on request, stability-followed) molecular reference
/// with the energy terms that sit outside the SCF.
pub struct MolecularReference {
    pub scf: ScfResult,
    pub e_dispersion: f64,
    pub e_smd_cds: f64,
    pub following: Option<crate::response::StabilizeReport>,
}

impl MolecularReference {
    /// Total energy including empirical dispersion and the SMD CDS term.
    pub fn total_energy(&self) -> f64 {
        self.scf.energy + self.e_dispersion + self.e_smd_cds
    }
}

/// Collinear constraints for static density properties; derivative kernels
/// have separate compatibility requirements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollinearPropertyReference {
    Auto,
    Rks,
    Uks,
    Roks,
}

impl CollinearPropertyReference {
    /// Read the shared property selector, including the corresponding HF aliases.
    pub fn from_controls(controls: &Controls) -> Result<Self> {
        match controls
            .str("reference", "auto")?
            .to_ascii_lowercase()
            .as_str()
        {
            "auto" => Ok(Self::Auto),
            "rks" | "rhf" => Ok(Self::Rks),
            "uks" | "uhf" => Ok(Self::Uks),
            "roks" | "rohf" => Ok(Self::Roks),
            other => Err(DftError::InvalidInput(format!(
                "unknown collinear property reference `{other}` (auto, rks, uks, roks)"
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Rks => "rks",
            Self::Uks => "uks",
            Self::Roks => "roks",
        }
    }
}

pub fn run_molecular_reference(setup: &MolecularSetup) -> Result<MolecularReference> {
    run_collinear_property_reference(setup, CollinearPropertyReference::Auto)
}

/// Same method and additive terms, retaining a trial state for the checked
/// typestate transition. Ordinary public property references remain strict.
pub(crate) fn iterate_molecular_reference(setup: &MolecularSetup) -> Result<MolecularReference> {
    run_collinear_property_reference_inner(setup, CollinearPropertyReference::Auto, true)
}

/// Reuse the selected SCF engine and the common additive energy terms.
/// Functional providers are unchanged; no registered-name property dispatch.
pub fn run_collinear_property_reference(
    setup: &MolecularSetup,
    kind: CollinearPropertyReference,
) -> Result<MolecularReference> {
    run_collinear_property_reference_inner(setup, kind, false)
}

fn run_collinear_property_reference_inner(
    setup: &MolecularSetup,
    kind: CollinearPropertyReference,
    retain_trial: bool,
) -> Result<MolecularReference> {
    if kind == CollinearPropertyReference::Roks && (setup.smd.is_some() || setup.method.stabilize) {
        return Err(DftError::Unsupported {
            feature: "ROKS static property with SMD or unrestricted stability following".into(),
            missing: "the corresponding constrained open-shell solver connection".into(),
        });
    }
    let mut options = setup.options.clone();
    options.reference = match kind {
        CollinearPropertyReference::Auto | CollinearPropertyReference::Roks => options.reference,
        CollinearPropertyReference::Rks => crate::scf::Reference::Rhf,
        CollinearPropertyReference::Uks => crate::scf::Reference::Uhf,
    };
    let mut scf = if kind == CollinearPropertyReference::Roks {
        crate::scf::roks::run_roks(
            &setup.molecule,
            &setup.basis,
            &options,
            &setup.functional,
            setup.grid.as_ref(),
        )?
    } else if retain_trial {
        crate::scf::run_scf_trial(
            &setup.molecule,
            &setup.basis,
            &options,
            &setup.functional,
            setup.grid.as_ref(),
        )?
    } else {
        run_scf(
            &setup.molecule,
            &setup.basis,
            &options,
            &setup.functional,
            setup.grid.as_ref(),
        )?
    };
    if kind == CollinearPropertyReference::Roks && !scf.converged {
        return Err(DftError::ScfNotConverged {
            iterations: scf.iterations,
            error: f64::NAN,
        });
    }
    let following = if setup.method.stabilize && scf.converged {
        let report = crate::response::stabilize(
            &setup.molecule,
            &setup.basis,
            &setup.functional,
            setup.grid.as_ref(),
            &scf,
            &options,
            &crate::response::StabilizeOptions::default(),
        )?;
        scf = report.final_state.clone();
        Some(report)
    } else {
        None
    };
    finish_molecular_reference(setup, scf, following)
}

pub(super) fn finish_molecular_reference(
    setup: &MolecularSetup,
    scf: ScfResult,
    following: Option<crate::response::StabilizeReport>,
) -> Result<MolecularReference> {
    let e_dispersion = crate::dispersion::dispersion_energy(
        &setup.molecule,
        &setup.dispersion,
        &setup.method.dispersion_functional_name(),
    )?;
    let e_smd_cds = match setup.smd {
        Some(solvent) => crate::pcm::smd_cds_energy(&setup.molecule, solvent, 2048)?,
        None => 0.0,
    };
    Ok(MolecularReference {
        scf,
        e_dispersion,
        e_smd_cds,
        following,
    })
}
