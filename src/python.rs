// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Python bindings (feature `python`, pyo3/abi3).
//!
//! Explicit entry points cover molecular SCF, Hessians and spectra, geometry
//! walks, response properties, SAPT, and periodic calculations. They return
//! plain Python dictionaries and lists; callers may convert lists with
//! `np.array(...)`. Avoiding a direct NumPy dependency keeps the native
//! extension usable without NumPy and preserves the small dependency graph.
//!
//! Units are ATOMIC UNITS throughout (Bohr, Hartree); the bundled ASE
//! calculator (`python/dft_rs/ase_calculator.py`) does the eV/Angstrom
//! conversions.

use pyo3::exceptions::{PyNotImplementedError, PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::basis::{data, MolBasis};
use crate::error::DftError;
use crate::gradient::assemble_gradient;
use crate::grid::{GridOptions, MolecularGrid};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::scf::{run_scf, ScfOptions};
use crate::system::{Atom, Molecule};
use crate::xc::{resolve, FunctionalSpec};

fn err(e: DftError) -> PyErr {
    let message = e.to_string();
    match e {
        DftError::Io(_) => PyOSError::new_err(message),
        DftError::Parse { .. }
        | DftError::InvalidInput(_)
        | DftError::MissingElement { .. }
        | DftError::UnknownBasis(_)
        | DftError::FunctionalResolution(_) => PyValueError::new_err(message),
        DftError::DerivOrderUnsupported { .. } | DftError::Unsupported { .. } => {
            PyNotImplementedError::new_err(message)
        }
        DftError::LinearAlgebra(_)
        | DftError::ResourceLimit { .. }
        | DftError::ScfNotConverged { .. }
        | DftError::ResponseNotConverged { .. } => PyRuntimeError::new_err(message),
    }
}

fn validate_geometry(
    numbers: &[u8],
    positions: &[[f64; 3]],
    multiplicity: usize,
) -> crate::error::Result<()> {
    if numbers.is_empty() {
        return Err(DftError::InvalidInput(
            "at least one atom is required".to_string(),
        ));
    }
    if numbers.len() != positions.len() {
        return Err(DftError::InvalidInput(
            "numbers and positions must have the same length".to_string(),
        ));
    }
    if multiplicity == 0 {
        return Err(DftError::InvalidInput(
            "multiplicity must be at least 1".to_string(),
        ));
    }
    for (i, (&z, p)) in numbers.iter().zip(positions).enumerate() {
        if !(1..=118).contains(&z) {
            return Err(DftError::InvalidInput(format!(
                "atom {i} has invalid atomic number {z}; Z=0 cannot identify which ghost basis to place"
            )));
        }
        if p.iter().any(|x| !x.is_finite()) {
            return Err(DftError::InvalidInput(format!(
                "atom {i} has a non-finite Cartesian coordinate"
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn scf_options(
    xc: &str,
    dispersion: &str,
    density_fit: Option<&str>,
    max_iter: usize,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    conv_tol_dp: f64,
    incore_mb: usize,
    vv10_grid_level: usize,
) -> crate::error::Result<(ScfOptions, crate::dispersion::Dispersion)> {
    if max_iter == 0 {
        return Err(DftError::InvalidInput(
            "max_iter must be at least 1".to_string(),
        ));
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
                "conv_tol_grad must be finite and positive".to_string(),
            ));
        }
    }
    if incore_mb == 0 {
        return Err(DftError::InvalidInput(
            "incore_mb must be at least 1".to_string(),
        ));
    }
    if vv10_grid_level > 9 {
        return Err(DftError::InvalidInput(
            "vv10_grid_level must be between 0 and 9".to_string(),
        ));
    }

    let (vv10, empirical) = crate::driver::method::resolve_molecular_dispersion(xc, dispersion)?;
    let opts = ScfOptions {
        max_iter,
        e_tol: conv_tol,
        conv_tol_grad: conv_tol_grad.unwrap_or_else(|| conv_tol.sqrt()),
        p_tol: conv_tol_dp,
        incore_mb: crate::resources::component_limit_mb(incore_mb),
        density_fit: density_fit.map(str::to_owned),
        vv10_grid_level,
        vv10,
        ..ScfOptions::default()
    };
    Ok((opts, empirical))
}

/// Resolve a `solvent` spec the same way the CLI's `--solvent` does: an
/// optional `smd:`/`cosmo:`/`ief:`/`ssvpe:` prefix selects the continuum
/// model (electrostatics defaults to C-PCM), the remaining name is looked
/// up in the bundled 179-solvent table, and `custom` defers entirely to
/// `eps`. This is the SAME grammar the CLI parses (`src/bin/dft_rs.rs`),
/// deliberately not reimplemented with different rules, so a spec that
/// works on the command line works here unchanged.
///
/// Returns `(pcm_options, smd_requested)`; `smd_requested` tells the caller
/// whether to add the non-electrostatic `smd_cds_energy` term on top of the
/// SCF total, exactly as the CLI does.
fn resolve_solvent(
    solvent: Option<&str>,
    eps: Option<f64>,
) -> crate::error::Result<(Option<crate::pcm::PcmOptions>, bool)> {
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
        // SMD's electrostatics uses its own intrinsic Coulomb radii,
        // unscaled -- not the scaled Bondi set plain C-PCM uses.
        if smd {
            p.smd_radii = true;
            p.vdw_scale = 1.0;
        }
        if name.eq_ignore_ascii_case("custom") {
            if eps.is_none() {
                return Err(DftError::InvalidInput(
                    "solvent='custom' needs eps=...; the Python API does not expose the \
                     per-descriptor overrides (refractive_index/alpha/beta/gamma) the CLI's \
                     `--solvent-alpha` etc. take, so a custom solvent gets electrostatics only \
                     (no SMD non-electrostatic term) unless it names a bundled solvent instead"
                        .to_string(),
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

fn mat_rows(m: &Matrix) -> Vec<Vec<f64>> {
    (0..m.rows)
        .map(|r| (0..m.cols).map(|c| m[(r, c)]).collect())
        .collect()
}

fn molecule_positions(molecule: &Molecule) -> Vec<Vec<f64>> {
    molecule
        .atoms
        .iter()
        .map(|atom| vec![atom.position.x, atom.position.y, atom.position.z])
        .collect()
}

fn bloch_rows<'py>(
    py: Python<'py>,
    matrices: &[crate::periodic::BlochMatrix],
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for matrix in matrices {
        let entry = PyDict::new(py);
        entry.set_item("real", mat_rows(&matrix.re))?;
        entry.set_item("imag", mat_rows(&matrix.im))?;
        out.append(entry)?;
    }
    Ok(out)
}

/// Run an SCF (and optionally the analytic gradient) for a molecule given
/// as atomic numbers + positions in BOHR. Returns a dict with energies,
/// convergence data, the dipole moment, and (on request) the gradient and
/// the full wavefunction (MO energies/coefficients, density, overlap).
#[pyfunction]
#[pyo3(signature = (numbers, positions, basis="sto-3g", ecp=None, xc="hf", charge=0,
    multiplicity=1, grid_level=3, gradient=false, wavefunction=false, nbo=false,
    max_iter=150, dispersion="auto", density_fit=None, conv_tol=1e-9,
    conv_tol_grad=None, conv_tol_dp=1e-8, incore_mb=2048, vv10_grid_level=3,
    field=None, solvent=None, eps=None, pseudo_sub=None, unrestricted=false, stabilize=false,
    grid_preset=None, grid_angular_scheme=None, grid_weight_cutoff=None))]
#[allow(clippy::too_many_arguments)]
fn calculate<'py>(
    py: Python<'py>,
    numbers: Vec<u8>,
    positions: Vec<[f64; 3]>,
    basis: &str,
    ecp: Option<&str>,
    xc: &str,
    charge: i32,
    multiplicity: usize,
    grid_level: usize,
    gradient: bool,
    wavefunction: bool,
    nbo: bool,
    max_iter: usize,
    dispersion: &str,
    density_fit: Option<&str>,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    conv_tol_dp: f64,
    incore_mb: usize,
    vv10_grid_level: usize,
    field: Option<[f64; 3]>,
    solvent: Option<&str>,
    eps: Option<f64>,
    pseudo_sub: Option<&str>,
    unrestricted: bool,
    stabilize: bool,
    grid_preset: Option<&str>,
    grid_angular_scheme: Option<&str>,
    grid_weight_cutoff: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    if stabilize && gradient && density_fit.is_some() {
        return Err(PyNotImplementedError::new_err(
            "stabilized-reference RI-J forces need the analytic fitted-integral \
            gradient; the finite-difference fallback does not follow the reference",
        ));
    }
    validate_geometry(&numbers, &positions, multiplicity).map_err(err)?;
    let grid_options = GridOptions::from_controls(
        grid_level,
        grid_preset,
        grid_angular_scheme,
        grid_weight_cutoff,
    )
    .map_err(err)?;
    if crate::xc::double_hybrid::DoubleHybridFunctional::named(xc).is_some() {
        if wavefunction || nbo || pseudo_sub.is_some() {
            return Err(PyNotImplementedError::new_err(
                "the double-hybrid calculation entry does not yet supply correlated densities or wavefunctions"));
        }
        let controls = crate::driver::Controls::from_value(
            &crate::driver::Value::object()
                .with("xc", xc)
                .with("basis", basis)
                .with("ecp", ecp)
                .with("charge", charge)
                .with("multiplicity", multiplicity)
                .with("grid_level", grid_level)
                .with("grid_preset", grid_preset)
                .with("grid_angular_scheme", grid_angular_scheme)
                .with("grid_weight_cutoff", grid_weight_cutoff)
                .with("max_iter", max_iter)
                .with("dispersion", dispersion)
                .with("density_fit", density_fit)
                .with("conv_tol", conv_tol)
                .with("conv_tol_grad", conv_tol_grad)
                .with("conv_tol_dp", conv_tol_dp)
                .with("incore_mb", incore_mb)
                .with("vv10_grid_level", vv10_grid_level)
                .with("field", field)
                .with("solvent", solvent)
                .with("eps", eps)
                .with("unrestricted", unrestricted)
                .with("stabilize", stabilize),
        )
        .map_err(err)?;
        let system = crate::driver::System::molecule(numbers, positions);
        let report = py
            .allow_threads(move || {
                crate::driver::run(
                    if gradient { "gradient" } else { "energy" },
                    &system,
                    &controls,
                )
            })
            .map_err(err)?;
        return value_to_py(py, &report)?
            .downcast_into::<PyDict>()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()));
    }
    let mol = Molecule::new(
        numbers
            .iter()
            .zip(&positions)
            .map(|(&z, p)| Atom {
                z,
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    )
    .with_charge(charge)
    .with_multiplicity(multiplicity);
    if let Some(spec) = pseudo_sub {
        let _table = crate::pseudo_sub::build_named_site_table(spec, &mol).map_err(err)?;
        return Err(PyNotImplementedError::new_err(
            "pseudo-substituent calculation has no PseudoSiteTable Hamiltonian/gradient consumer",
        ));
    }
    let set = build_basis(basis, ecp).map_err(err)?;
    let mb = MolBasis::build(&mol, &set).map_err(err)?;
    let functional = resolve(&FunctionalSpec::Combined(xc.to_string())).map_err(err)?;
    let grid = if functional.parts.is_empty() {
        None
    } else {
        Some(MolecularGrid::build(&mol, &grid_options).map_err(err)?)
    };
    let (mut opts, empirical_dispersion) = scf_options(
        xc,
        dispersion,
        density_fit,
        max_iter,
        conv_tol,
        conv_tol_grad,
        conv_tol_dp,
        incore_mb,
        vv10_grid_level,
    )
    .map_err(err)?;
    opts.vv10_grid_weight_cutoff = grid_options.weight_cutoff;
    // A uniform external field is just +E.r on the core Hamiltonian and the
    // matching -Z_A R_A.E on the nuclear energy (`ScfOptions::external_field`);
    // everything downstream (energy, gradient, dipole) already threads it
    // through, so plugging it in here is all this entry point needs to do.
    opts.external_field = field.map(|e| Vec3::new(e[0], e[1], e[2]));
    let (pcm, smd) = resolve_solvent(solvent, eps).map_err(err)?;
    opts.pcm = pcm;
    if unrestricted {
        opts.reference = crate::scf::Reference::Uhf;
    }
    let mut scf = run_scf(&mol, &mb, &opts, &functional, grid.as_ref()).map_err(err)?;
    let following = if stabilize {
        let report = crate::response::stabilize(
            &mol,
            &mb,
            &functional,
            grid.as_ref(),
            &scf,
            &opts,
            &crate::response::StabilizeOptions::default(),
        )
        .map_err(err)?;
        scf = report.final_state.clone();
        Some(report)
    } else {
        None
    };
    let e_dispersion =
        crate::dispersion::dispersion_energy(&mol, &empirical_dispersion, xc).map_err(err)?;
    // SMD's non-electrostatic (cavity/dispersion/structural) term sits on
    // top of the SCF total rather than inside it, exactly as the CLI adds
    // it (`src/bin/dft_rs.rs`): the electrostatic reaction field is already
    // folded into `scf.energy` via `ScfOptions::pcm`.
    let e_smd_cds = if smd {
        let sv = match opts.pcm.as_ref().and_then(|o| o.descriptors) {
            Some(d) if !solvent.unwrap_or("").eq_ignore_ascii_case("smd:water") => {
                crate::pcm::SmdSolvent::NonAqueous(d)
            }
            _ => crate::pcm::SmdSolvent::Water,
        };
        crate::pcm::smd_cds_energy(&mol, sv, 2048).map_err(err)?
    } else {
        0.0
    };

    let out = PyDict::new(py);
    out.set_item("energy", scf.energy + e_dispersion + e_smd_cds)?;
    if let Some(report) = &following {
        out.set_item("stabilized", report.stabilized)?;
        out.set_item("stability", following_dict(py, report)?)?;
    }
    out.set_item("e_nuc", scf.e_nuc)?;
    out.set_item("e_elec", scf.e_elec)?;
    out.set_item("e_xc", scf.e_xc)?;
    out.set_item("e_nlc", scf.e_nlc)?;
    out.set_item("e_dispersion", e_dispersion)?;
    out.set_item("e_pcm", scf.e_pcm)?;
    out.set_item("e_smd_cds", e_smd_cds)?;
    out.set_item("converged", scf.converged)?;
    out.set_item("iterations", scf.iterations)?;
    out.set_item("s_squared", scf.s_squared)?;
    out.set_item("n_alpha", scf.n_alpha)?;
    out.set_item("n_beta", scf.n_beta)?;
    out.set_item("unrestricted", scf.unrestricted)?;
    out.set_item("n_core_electrons", mb.n_core_electrons)?;
    let mu = crate::properties::dipole_moment(&mol, &mb, &scf);
    out.set_item("dipole", vec![mu.x, mu.y, mu.z])?;
    out.set_item(
        "mulliken",
        crate::properties::mulliken_charges(&mol, &mb, &scf),
    )?;
    if nbo {
        use crate::properties::natural_bond::{natural_bond_analysis, NboOptions, NboSpin};
        let analysis =
            natural_bond_analysis(&mol, &mb, &scf, NboSpin::Auto, &NboOptions::default())
                .map_err(err)?;
        let nao = &analysis.nao;
        let values = PyDict::new(py);
        values.set_item(
            "method",
            "natural population analysis and natural Lewis structure (the natural-bond \
             feature reports hybrids, NLMOs and donor-acceptor energies)",
        )?;
        values.set_item("occupations", nao.occupations.clone())?;
        values.set_item("orbital_atoms", nao.orbital_atoms.clone())?;
        values.set_item(
            "nao_labels",
            nao.labels.iter().map(|l| l.text()).collect::<Vec<_>>(),
        )?;
        values.set_item("atomic_populations", nao.atomic_populations.clone())?;
        values.set_item("atomic_charges", nao.atomic_charges.clone())?;
        values.set_item("spin_populations", nao.spin_populations.clone())?;
        values.set_item(
            "natural_electron_configurations",
            nao.configuration_text.clone(),
        )?;
        values.set_item("nao_coefficients", mat_rows(&nao.nao_coefficients))?;
        values.set_item("density_nao", mat_rows(&nao.density_nao))?;
        values.set_item("wiberg_bond_indices", mat_rows(&nao.wiberg_bond_indices))?;
        let lewis: Vec<Vec<String>> = analysis
            .channels
            .iter()
            .map(|c| {
                c.nbos
                    .iter()
                    .filter(|b| b.kind.is_lewis())
                    .map(|b| b.label.clone())
                    .collect()
            })
            .collect();
        values.set_item("lewis_structures", lewis)?;
        values.set_item(
            "non_lewis_occupancies",
            analysis
                .channels
                .iter()
                .map(|c| c.non_lewis_occupancy)
                .collect::<Vec<_>>(),
        )?;
        out.set_item("nbo", values)?;
    }
    if gradient {
        let mut g = if opts.density_fit.is_some() {
            crate::gradient::numerical_gradient(
                &mol,
                &set,
                &opts,
                &functional,
                if functional.parts.is_empty() {
                    None
                } else {
                    Some(&grid_options)
                },
                1e-4,
            )
            .map_err(err)?
        } else {
            assemble_gradient(&mol, &mb, &functional, grid.as_ref(), &scf, &opts).map_err(err)?
        };
        if empirical_dispersion != crate::dispersion::Dispersion::None {
            let correction =
                crate::dispersion::dispersion_gradient(&mol, &empirical_dispersion, xc)
                    .map_err(err)?;
            for (total, delta) in g.iter_mut().zip(correction) {
                *total += delta;
            }
        }
        let rows: Vec<Vec<f64>> = g.iter().map(|v| vec![v.x, v.y, v.z]).collect();
        out.set_item("gradient", rows)?;
    }
    if wavefunction {
        let wf = PyDict::new(py);
        wf.set_item("nao", mb.nao_sph)?;
        wf.set_item(
            "mo_energy",
            vec![scf.mo_energy[0].clone(), scf.mo_energy[1].clone()],
        )?;
        wf.set_item(
            "mo_coeff",
            vec![mat_rows(&scf.mo_coeff[0]), mat_rows(&scf.mo_coeff[1])],
        )?;
        wf.set_item(
            "density",
            vec![mat_rows(&scf.density[0]), mat_rows(&scf.density[1])],
        )?;
        let s = crate::integrals::overlap_matrix(&mb);
        wf.set_item("overlap", mat_rows(&s))?;
        out.set_item("wavefunction", wf)?;
    }
    Ok(out)
}

/// Resolve the `basis` spec (a bundled name, a `.g94` path, or a
/// per-element spec) and apply the independent `ecp` spec on top.
///
/// The two are separate because a core potential and an orbital basis are
/// separate choices: `basis="def2-tzvp", ecp="Ni=lanl2dz"` is a legitimate
/// and otherwise inexpressible combination.
fn build_basis(basis: &str, ecp: Option<&str>) -> crate::error::Result<crate::basis::BasisSet> {
    let mut set = crate::basis::mixed::load_spec(basis)?;
    if let Some(spec) = ecp {
        crate::basis::mixed::apply_ecp_spec(&mut set, spec)?;
    }
    Ok(set)
}

/// Shared setup for the derivative entry points.
#[allow(clippy::type_complexity)]
fn prepare(
    numbers: &[u8],
    positions: &[[f64; 3]],
    basis: &str,
    ecp: Option<&str>,
    xc: &str,
    charge: i32,
    multiplicity: usize,
    grid_level: usize,
    max_iter: usize,
    dispersion: &str,
    density_fit: Option<&str>,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    conv_tol_dp: f64,
    incore_mb: usize,
    vv10_grid_level: usize,
) -> PyResult<(
    Molecule,
    MolBasis,
    crate::xc::ResolvedFunctional,
    Option<MolecularGrid>,
    ScfOptions,
    crate::dispersion::Dispersion,
)> {
    validate_geometry(numbers, positions, multiplicity).map_err(err)?;
    if grid_level > 9 {
        return Err(PyValueError::new_err("grid_level must be between 0 and 9"));
    }
    let mol = Molecule::new(
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
    .with_multiplicity(multiplicity);
    let set = build_basis(basis, ecp).map_err(err)?;
    let mb = MolBasis::build(&mol, &set).map_err(err)?;
    let functional = resolve(&FunctionalSpec::Combined(xc.to_string())).map_err(err)?;
    let grid = if functional.parts.is_empty() {
        None
    } else {
        Some(
            MolecularGrid::build(
                &mol,
                &GridOptions {
                    level: grid_level,
                    ..GridOptions::default()
                },
            )
            .map_err(err)?,
        )
    };
    let (opts, empirical_dispersion) = scf_options(
        xc,
        dispersion,
        density_fit,
        max_iter,
        conv_tol,
        conv_tol_grad,
        conv_tol_dp,
        incore_mb,
        vv10_grid_level,
    )
    .map_err(err)?;
    Ok((mol, mb, functional, grid, opts, empirical_dispersion))
}

/// Cartesian Hessian (Hartree/Bohr^2). Analytic where the method supports
/// it, central differences of the analytic gradient otherwise — the choice
/// is reported in `method` so the caller is never guessing.
#[pyfunction]
#[pyo3(signature = (numbers, positions, basis="sto-3g", ecp=None, xc="hf", charge=0,
    multiplicity=1, grid_level=3, max_iter=150, dispersion="auto",
    density_fit=None, conv_tol=1e-9, conv_tol_grad=None, conv_tol_dp=1e-8,
    incore_mb=2048, vv10_grid_level=3))]
#[allow(clippy::too_many_arguments)]
fn hessian<'py>(
    py: Python<'py>,
    numbers: Vec<u8>,
    positions: Vec<[f64; 3]>,
    basis: &str,
    ecp: Option<&str>,
    xc: &str,
    charge: i32,
    multiplicity: usize,
    grid_level: usize,
    max_iter: usize,
    dispersion: &str,
    density_fit: Option<&str>,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    conv_tol_dp: f64,
    incore_mb: usize,
    vv10_grid_level: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if crate::xc::double_hybrid::DoubleHybridFunctional::named(xc).is_some() {
        let controls = crate::driver::Controls::from_value(
            &crate::driver::Value::object()
                .with("xc", xc)
                .with("basis", basis)
                .with("ecp", ecp)
                .with("charge", charge)
                .with("multiplicity", multiplicity)
                .with("grid_level", grid_level)
                .with("max_iter", max_iter)
                .with("dispersion", dispersion)
                .with("density_fit", density_fit)
                .with("conv_tol", conv_tol)
                .with("conv_tol_grad", conv_tol_grad)
                .with("conv_tol_dp", conv_tol_dp)
                .with("incore_mb", incore_mb)
                .with("vv10_grid_level", vv10_grid_level),
        )
        .map_err(err)?;
        let system = crate::driver::System::molecule(numbers, positions);
        let mut report = py
            .allow_threads(move || crate::driver::run("hessian", &system, &controls))
            .map_err(err)?;
        report.set("method", "analytic");
        return value_to_py(py, &report)?
            .downcast_into::<PyDict>()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()));
    }
    let (mol, mb, functional, grid, opts, empirical_dispersion) = prepare(
        &numbers,
        &positions,
        basis,
        ecp,
        xc,
        charge,
        multiplicity,
        grid_level,
        max_iter,
        dispersion,
        density_fit,
        conv_tol,
        conv_tol_grad,
        conv_tol_dp,
        incore_mb,
        vv10_grid_level,
    )?;
    let scf = run_scf(&mol, &mb, &opts, &functional, grid.as_ref()).map_err(err)?;
    let out = PyDict::new(py);
    let set = build_basis(basis, ecp).map_err(err)?;
    let gopts = GridOptions {
        level: grid_level,
        ..GridOptions::default()
    };
    let (res, method) = crate::hessian::hessian_with_fallback(
        &mol,
        &set,
        &mb,
        &opts,
        &functional,
        xc,
        if functional.parts.is_empty() {
            None
        } else {
            Some(&gopts)
        },
        grid.as_ref(),
        &scf,
        &empirical_dispersion,
        5e-3,
    )
    .map_err(err)?;
    out.set_item("hessian", mat_rows(&res.hessian))?;
    out.set_item("method", method.as_str())?;
    let e_dispersion =
        crate::dispersion::dispersion_energy(&mol, &empirical_dispersion, xc).map_err(err)?;
    out.set_item("energy", scf.energy + e_dispersion)?;
    out.set_item("e_nlc", scf.e_nlc)?;
    out.set_item("e_dispersion", e_dispersion)?;
    Ok(out)
}

/// Harmonic frequencies (cm^-1) and IR intensities (km/mol), Eckart
/// projected.
#[pyfunction]
#[pyo3(signature = (numbers, positions, basis="sto-3g", ecp=None, xc="hf", charge=0,
    multiplicity=1, grid_level=3, max_iter=150, dispersion="auto",
    density_fit=None, conv_tol=1e-9, conv_tol_grad=None, conv_tol_dp=1e-8,
    incore_mb=2048, vv10_grid_level=3, temperature=298.15,
    pressure=101325.0, symmetry_number=1))]
#[allow(clippy::too_many_arguments)]
fn vibrations<'py>(
    py: Python<'py>,
    numbers: Vec<u8>,
    positions: Vec<[f64; 3]>,
    basis: &str,
    ecp: Option<&str>,
    xc: &str,
    charge: i32,
    multiplicity: usize,
    grid_level: usize,
    max_iter: usize,
    dispersion: &str,
    density_fit: Option<&str>,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    conv_tol_dp: f64,
    incore_mb: usize,
    vv10_grid_level: usize,
    temperature: f64,
    pressure: f64,
    symmetry_number: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let (mol, mb, functional, grid, opts, empirical_dispersion) = prepare(
        &numbers,
        &positions,
        basis,
        ecp,
        xc,
        charge,
        multiplicity,
        grid_level,
        max_iter,
        dispersion,
        density_fit,
        conv_tol,
        conv_tol_grad,
        conv_tol_dp,
        incore_mb,
        vv10_grid_level,
    )?;
    let scf = run_scf(&mol, &mb, &opts, &functional, grid.as_ref()).map_err(err)?;
    let set = build_basis(basis, ecp).map_err(err)?;
    let gopts = GridOptions {
        level: grid_level,
        ..GridOptions::default()
    };
    let (res, method) = crate::hessian::hessian_with_fallback(
        &mol,
        &set,
        &mb,
        &opts,
        &functional,
        xc,
        if functional.parts.is_empty() {
            None
        } else {
            Some(&gopts)
        },
        grid.as_ref(),
        &scf,
        &empirical_dispersion,
        5e-3,
    )
    .map_err(err)?;
    let vib = crate::properties::vibrational::vibrational_analysis(
        &mol,
        &res.hessian,
        Some(&res.dipole_derivatives),
    )
    .map_err(err)?;
    let out = PyDict::new(py);
    out.set_item("frequencies", vib.frequencies_cm.clone())?;
    out.set_item("ir_intensities", vib.ir_intensities.clone())?;
    out.set_item("n_imaginary", vib.n_imaginary)?;
    out.set_item("n_projected", vib.n_projected)?;
    out.set_item("method", method.as_str())?;
    if vib.n_imaginary == 0 {
        let e_dispersion =
            crate::dispersion::dispersion_energy(&mol, &empirical_dispersion, xc).map_err(err)?;
        let thermo = crate::properties::thermochemistry::thermochemistry(
            &mol,
            &vib.frequencies_cm,
            scf.energy + e_dispersion,
            crate::properties::thermochemistry::ThermochemistryOptions {
                temperature_kelvin: temperature,
                pressure_pascal: pressure,
                symmetry_number,
                electronic_degeneracy: multiplicity,
            },
        )
        .map_err(err)?;
        let t = PyDict::new(py);
        t.set_item("temperature", thermo.temperature_kelvin)?;
        t.set_item("pressure", thermo.pressure_pascal)?;
        t.set_item("symmetry_number", thermo.symmetry_number)?;
        t.set_item("electronic_degeneracy", thermo.electronic_degeneracy)?;
        t.set_item("is_linear", thermo.is_linear)?;
        t.set_item("zero_point_energy", thermo.zero_point_energy)?;
        t.set_item(
            "vibrational_thermal_energy",
            thermo.vibrational_thermal_energy,
        )?;
        t.set_item("translational_energy", thermo.translational_energy)?;
        t.set_item("rotational_energy", thermo.rotational_energy)?;
        t.set_item(
            "internal_energy_correction",
            thermo.internal_energy_correction,
        )?;
        t.set_item("enthalpy_correction", thermo.enthalpy_correction)?;
        t.set_item("entropy_translational", thermo.entropy_translational)?;
        t.set_item("entropy_rotational", thermo.entropy_rotational)?;
        t.set_item("entropy_vibrational", thermo.entropy_vibrational)?;
        t.set_item("entropy_electronic", thermo.entropy_electronic)?;
        t.set_item("entropy", thermo.entropy)?;
        t.set_item("gibbs_correction", thermo.gibbs_correction)?;
        t.set_item("internal_energy", thermo.internal_energy)?;
        t.set_item("enthalpy", thermo.enthalpy)?;
        t.set_item("gibbs_free_energy", thermo.gibbs_free_energy)?;
        out.set_item("thermochemistry", t)?;
    } else {
        out.set_item("thermochemistry", py.None())?;
    }
    Ok(out)
}

/// Harmonic frequencies, IR intensities, and Raman activities.  The normal
/// modes come from the same analytic-first complete-Hessian path as
/// `vibrations`; Raman activities are finite differences of analytic CPKS
/// polarizabilities along those mass-weighted modes.
#[pyfunction]
#[pyo3(signature = (numbers, positions, basis="sto-3g", ecp=None, xc="hf", charge=0,
    multiplicity=1, grid_level=3, max_iter=150, dispersion="auto",
    density_fit=None, conv_tol=1e-9, conv_tol_grad=None, conv_tol_dp=1e-8,
    incore_mb=2048, vv10_grid_level=3, hessian_step=5e-3, raman_step=5e-3))]
#[allow(clippy::too_many_arguments)]
fn raman<'py>(
    py: Python<'py>,
    numbers: Vec<u8>,
    positions: Vec<[f64; 3]>,
    basis: &str,
    ecp: Option<&str>,
    xc: &str,
    charge: i32,
    multiplicity: usize,
    grid_level: usize,
    max_iter: usize,
    dispersion: &str,
    density_fit: Option<&str>,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    conv_tol_dp: f64,
    incore_mb: usize,
    vv10_grid_level: usize,
    hessian_step: f64,
    raman_step: f64,
) -> PyResult<Bound<'py, PyDict>> {
    for (name, value) in [("hessian_step", hessian_step), ("raman_step", raman_step)] {
        if !(value.is_finite() && value > 0.0) {
            return Err(PyValueError::new_err(format!(
                "{name} must be finite and positive"
            )));
        }
    }
    let (mol, mb, functional, grid, opts, empirical_dispersion) = prepare(
        &numbers,
        &positions,
        basis,
        ecp,
        xc,
        charge,
        multiplicity,
        grid_level,
        max_iter,
        dispersion,
        density_fit,
        conv_tol,
        conv_tol_grad,
        conv_tol_dp,
        incore_mb,
        vv10_grid_level,
    )?;
    let set = build_basis(basis, ecp).map_err(err)?;
    let gopts = GridOptions {
        level: grid_level,
        ..GridOptions::default()
    };
    let grid_opts = if functional.parts.is_empty() {
        None
    } else {
        Some(&gopts)
    };
    let scf = run_scf(&mol, &mb, &opts, &functional, grid.as_ref()).map_err(err)?;
    let (hessian, method) = crate::hessian::hessian_with_fallback(
        &mol,
        &set,
        &mb,
        &opts,
        &functional,
        xc,
        grid_opts,
        grid.as_ref(),
        &scf,
        &empirical_dispersion,
        hessian_step,
    )
    .map_err(err)?;
    let vib = crate::properties::vibrational::vibrational_analysis(
        &mol,
        &hessian.hessian,
        Some(&hessian.dipole_derivatives),
    )
    .map_err(err)?;
    let activities = crate::response::polarizability::raman_activities(
        &mol,
        &set,
        &opts,
        &functional,
        grid_opts,
        &vib.modes,
        raman_step,
    )
    .map_err(err)?;
    let out = PyDict::new(py);
    out.set_item("frequencies", vib.frequencies_cm)?;
    out.set_item("ir_intensities", vib.ir_intensities)?;
    out.set_item("raman_activities", activities)?;
    out.set_item("n_imaginary", vib.n_imaginary)?;
    out.set_item("n_projected", vib.n_projected)?;
    out.set_item("hessian_method", method.as_str())?;
    Ok(out)
}

/// Molecular L-BFGS geometry optimization in atomic units.
#[pyfunction]
#[pyo3(signature = (numbers, positions, basis="sto-3g", ecp=None, xc="hf", charge=0,
    multiplicity=1, grid_level=3, max_iter=150, dispersion="auto",
    density_fit=None, conv_tol=1e-9, conv_tol_grad=None, conv_tol_dp=1e-8,
    incore_mb=2048, vv10_grid_level=3, max_steps=200, gtol=3e-4,
    step_max=0.3, history=8, trajectory=None))]
#[allow(clippy::too_many_arguments)]
fn optimize<'py>(
    py: Python<'py>,
    numbers: Vec<u8>,
    positions: Vec<[f64; 3]>,
    basis: &str,
    ecp: Option<&str>,
    xc: &str,
    charge: i32,
    multiplicity: usize,
    grid_level: usize,
    max_iter: usize,
    dispersion: &str,
    density_fit: Option<&str>,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    conv_tol_dp: f64,
    incore_mb: usize,
    vv10_grid_level: usize,
    max_steps: usize,
    gtol: f64,
    step_max: f64,
    history: usize,
    trajectory: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    if max_steps == 0
        || history == 0
        || !(gtol.is_finite() && gtol > 0.0)
        || !(step_max.is_finite() && step_max > 0.0)
    {
        return Err(PyValueError::new_err(
            "max_steps/history must be nonzero and gtol/step_max finite and positive",
        ));
    }
    let (mol, _mb, functional, _grid, opts, empirical_dispersion) = prepare(
        &numbers,
        &positions,
        basis,
        ecp,
        xc,
        charge,
        multiplicity,
        grid_level,
        max_iter,
        dispersion,
        density_fit,
        conv_tol,
        conv_tol_grad,
        conv_tol_dp,
        incore_mb,
        vv10_grid_level,
    )?;
    let set = build_basis(basis, ecp).map_err(err)?;
    let gopts = GridOptions {
        level: grid_level,
        ..GridOptions::default()
    };
    let result = crate::optimizer::optimize(
        &mol,
        &set,
        &opts,
        &functional,
        xc,
        if functional.parts.is_empty() {
            None
        } else {
            Some(&gopts)
        },
        &empirical_dispersion,
        &crate::optimizer::OptOptions {
            max_steps,
            gtol,
            step_max,
            history,
            trajectory: trajectory.map(std::path::PathBuf::from),
        },
    )
    .map_err(err)?;
    let out = PyDict::new(py);
    out.set_item("positions", molecule_positions(&result.molecule))?;
    out.set_item("energy", result.energy)?;
    out.set_item("converged", result.converged)?;
    out.set_item("iterations", result.iterations)?;
    let frames = PyList::empty(py);
    for frame in &result.trajectory {
        let item = PyDict::new(py);
        item.set_item("step", frame.step)?;
        item.set_item("positions", molecule_positions(&frame.molecule))?;
        item.set_item("energy", frame.energy)?;
        item.set_item("grad_max", frame.grad_max)?;
        frames.append(item)?;
    }
    out.set_item("trajectory", frames)?;
    Ok(out)
}

/// Molecular transition-state search using partitioned RFO.
#[pyfunction]
#[pyo3(signature = (numbers, positions, basis="sto-3g", ecp=None, xc="hf", charge=0,
    multiplicity=1, grid_level=3, max_iter=150, dispersion="auto",
    density_fit=None, conv_tol=1e-9, conv_tol_grad=None, conv_tol_dp=1e-8,
    incore_mb=2048, vv10_grid_level=3, method="prfo", max_steps=50, gtol=3e-4,
    step_max=0.2, hessian_step=5e-3, trajectory=None))]
#[allow(clippy::too_many_arguments)]
fn saddle<'py>(
    py: Python<'py>,
    numbers: Vec<u8>,
    positions: Vec<[f64; 3]>,
    basis: &str,
    ecp: Option<&str>,
    xc: &str,
    charge: i32,
    multiplicity: usize,
    grid_level: usize,
    max_iter: usize,
    dispersion: &str,
    density_fit: Option<&str>,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    conv_tol_dp: f64,
    incore_mb: usize,
    vv10_grid_level: usize,
    method: &str,
    max_steps: usize,
    gtol: f64,
    step_max: f64,
    hessian_step: f64,
    trajectory: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    if max_steps == 0
        || !(gtol.is_finite() && gtol > 0.0)
        || !(step_max.is_finite() && step_max > 0.0)
        || !(hessian_step.is_finite() && hessian_step > 0.0)
    {
        return Err(PyValueError::new_err(
            "max_steps must be nonzero and gtol/step_max/hessian_step finite and positive",
        ));
    }
    let (mol, _mb, functional, _grid, opts, empirical_dispersion) = prepare(
        &numbers,
        &positions,
        basis,
        ecp,
        xc,
        charge,
        multiplicity,
        grid_level,
        max_iter,
        dispersion,
        density_fit,
        conv_tol,
        conv_tol_grad,
        conv_tol_dp,
        incore_mb,
        vv10_grid_level,
    )?;
    let set = build_basis(basis, ecp).map_err(err)?;
    let gopts = GridOptions {
        level: grid_level,
        ..GridOptions::default()
    };
    let saddle_method = match method.to_ascii_lowercase().as_str() {
        "prfo" | "p-rfo" => crate::saddle::SaddleMethod::Prfo,
        "rsirfo" | "rs-i-rfo" => crate::saddle::SaddleMethod::RsIrfo,
        _ => {
            return Err(PyValueError::new_err(
                "saddle method must be 'prfo' or 'rsirfo'",
            ))
        }
    };
    let result = crate::saddle::optimize_saddle(
        &mol,
        &set,
        &opts,
        &functional,
        xc,
        if functional.parts.is_empty() {
            None
        } else {
            Some(&gopts)
        },
        &empirical_dispersion,
        &crate::saddle::SaddleOptions {
            method: saddle_method,
            max_steps,
            gtol,
            step_max,
            hessian_step,
            trajectory: trajectory.map(std::path::PathBuf::from),
        },
    )
    .map_err(err)?;
    let out = PyDict::new(py);
    out.set_item("positions", molecule_positions(&result.molecule))?;
    out.set_item("method", result.method.as_str())?;
    out.set_item("energy", result.energy)?;
    out.set_item("converged", result.converged)?;
    out.set_item("iterations", result.iterations)?;
    out.set_item("n_negative", result.n_negative)?;
    let frames = PyList::empty(py);
    for frame in &result.trajectory {
        let item = PyDict::new(py);
        item.set_item("step", frame.step)?;
        item.set_item("positions", molecule_positions(&frame.molecule))?;
        item.set_item("energy", frame.energy)?;
        item.set_item("grad_max", frame.grad_max)?;
        frames.append(item)?;
    }
    out.set_item("trajectory", frames)?;
    Ok(out)
}

/// Static dipole polarizability (a.u., 3x3) from electric-field CPKS.
///
/// `alpha[x][y] = -dmu_x/dE_y` at the converged SCF density. The gate on the
/// Rust side is the conjugacy identity `dE/dE_ax = -mu_ax`, which also makes
/// the tensor symmetric to numerical precision — worth checking on the
/// Python side too, since a transcription bug in the binding (e.g. swapping
/// rows and columns) would not otherwise be visible from Python.
#[pyfunction]
#[pyo3(signature = (numbers, positions, basis="sto-3g", ecp=None, xc="hf", charge=0,
    multiplicity=1, grid_level=3, max_iter=150, dispersion="auto",
    density_fit=None, conv_tol=1e-9, conv_tol_grad=None, conv_tol_dp=1e-8,
    incore_mb=2048, vv10_grid_level=3))]
#[allow(clippy::too_many_arguments)]
fn polarizability<'py>(
    py: Python<'py>,
    numbers: Vec<u8>,
    positions: Vec<[f64; 3]>,
    basis: &str,
    ecp: Option<&str>,
    xc: &str,
    charge: i32,
    multiplicity: usize,
    grid_level: usize,
    max_iter: usize,
    dispersion: &str,
    density_fit: Option<&str>,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    conv_tol_dp: f64,
    incore_mb: usize,
    vv10_grid_level: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let (mol, mb, functional, grid, opts, _empirical_dispersion) = prepare(
        &numbers,
        &positions,
        basis,
        ecp,
        xc,
        charge,
        multiplicity,
        grid_level,
        max_iter,
        dispersion,
        density_fit,
        conv_tol,
        conv_tol_grad,
        conv_tol_dp,
        incore_mb,
        vv10_grid_level,
    )?;
    let scf = run_scf(&mol, &mb, &opts, &functional, grid.as_ref()).map_err(err)?;
    let alpha = crate::response::polarizability::polarizability(
        &mol,
        &mb,
        &functional,
        grid.as_ref(),
        &scf,
        &opts,
    )
    .map_err(err)?;
    let out = PyDict::new(py);
    out.set_item(
        "alpha",
        alpha.iter().map(|row| row.to_vec()).collect::<Vec<_>>(),
    )?;
    out.set_item("isotropic", (alpha[0][0] + alpha[1][1] + alpha[2][2]) / 3.0)?;
    out.set_item("energy", scf.energy)?;
    Ok(out)
}

fn response_metadata<T>(
    py: Python<'_>,
    out: &Bound<'_, PyDict>,
    report: &crate::response::ResponseReport<T>,
) -> PyResult<()> {
    out.set_item("shape", &report.shape)?;
    out.set_item("functional", &report.functional)?;
    out.set_item("units", report.units)?;
    out.set_item("model", report.model)?;
    out.set_item("axis_convention", report.axis_convention)?;
    out.set_item("converged", report.converged)?;
    out.set_item("residuals", &report.residuals)?;
    out.set_item("iterations", &report.iterations)?;
    out.set_item("warnings", &report.warnings)?;
    let reference = PyDict::new(py);
    reference.set_item("converged", report.reference.converged)?;
    reference.set_item("iterations", report.reference.iterations)?;
    reference.set_item("orbital_residual", report.reference.orbital_residual)?;
    reference.set_item("eigen_residual", report.reference.eigen_residual)?;
    reference.set_item(
        "orthonormality_defect",
        report.reference.orthonormality_defect,
    )?;
    reference.set_item("projector_defect", report.reference.projector_defect)?;
    reference.set_item(
        "particle_number_error",
        report.reference.particle_number_error,
    )?;
    out.set_item("reference", reference)
}

fn following_dict<'py>(
    py: Python<'py>,
    report: &crate::response::StabilizeReport,
) -> PyResult<Bound<'py, PyDict>> {
    let out = stability_dict(py, &report.final_analysis)?;
    out.set_item("initial", stability_dict(py, &report.initial)?)?;
    out.set_item("initial_energy", report.initial_energy)?;
    out.set_item("energy", report.final_state.energy)?;
    out.set_item("stabilized", report.stabilized)?;
    out.set_item("following_converged", report.converged)?;
    out.set_item("warnings", &report.warnings)?;
    let descents = PyList::empty(py);
    for d in &report.descents {
        let entry = PyDict::new(py);
        entry.set_item("channel", d.channel.id())?;
        entry.set_item("eigenvalue", d.eigenvalue)?;
        entry.set_item("theta", d.theta)?;
        entry.set_item("energy_before", d.energy_before)?;
        entry.set_item("energy_after", d.energy_after)?;
        descents.append(entry)?;
    }
    out.set_item("descents", descents)?;
    let rows = |m: &Matrix| {
        (0..m.rows)
            .map(|i| (0..m.cols).map(|j| m[(i, j)]).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };
    out.set_item(
        "density",
        report
            .final_state
            .density
            .iter()
            .map(rows)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "mo_coeff",
        report
            .final_state
            .mo_coeff
            .iter()
            .map(rows)
            .collect::<Vec<_>>(),
    )?;
    out.set_item("mo_energy", &report.final_state.mo_energy)?;
    Ok(out)
}

fn stability_dict<'py>(
    py: Python<'py>,
    report: &crate::response::ResponseReport<crate::response::MolecularStability>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    response_metadata(py, &out, report)?;
    out.set_item("stable", report.value.stable)?;
    out.set_item("unrestricted", report.value.unrestricted)?;
    out.set_item("classification", report.value.classification)?;
    out.set_item("zero_tolerance", report.value.zero_tolerance)?;
    out.set_item("curvature_factor", report.value.curvature_factor)?;
    out.set_item(
        "provenance",
        if report.converged {
            "analytic"
        } else {
            "unconverged"
        },
    )?;
    let blocks = PyList::empty(py);
    for block in &report.value.blocks {
        let entry = PyDict::new(py);
        entry.set_item("name", block.name)?;
        entry.set_item("channel", block.name)?;
        entry.set_item("dimension", block.dimension)?;
        entry.set_item("classification", block.classification)?;
        entry.set_item("verdict", block.classification)?;
        entry.set_item("resolution", block.resolution)?;
        entry.set_item("converged", block.converged)?;
        entry.set_item("matvec_count", block.matvec_count)?;
        entry.set_item(
            "eigenvalues",
            block.modes.iter().map(|m| m.eigenvalue).collect::<Vec<_>>(),
        )?;
        entry.set_item("lowest", block.modes.first().map(|m| m.eigenvalue))?;
        entry.set_item("eigenvector", &block.eigenvector)?;
        entry.set_item("n_occupied", block.n_occupied)?;
        entry.set_item("n_virtual", block.n_virtual)?;
        entry.set_item("warnings", &block.warnings)?;
        entry.set_item(
            "iterations",
            block.modes.iter().map(|m| m.iterations).max().unwrap_or(0),
        )?;
        entry.set_item(
            "max_residual",
            block
                .modes
                .iter()
                .map(|m| m.absolute_residual)
                .fold(0.0_f64, f64::max),
        )?;
        entry.set_item("hermiticity_defect", block.hermiticity_defect)?;
        entry.set_item(
            "provenance",
            if block.converged {
                "analytic"
            } else {
                "unconverged"
            },
        )?;
        let dominant = PyList::empty(py);
        for d in &block.dominant {
            let e = PyDict::new(py);
            e.set_item("from", d.from)?;
            e.set_item("to", d.to)?;
            e.set_item("amplitude", d.amplitude)?;
            e.set_item("channel", d.spin)?;
            dominant.append(e)?;
        }
        entry.set_item("dominant", dominant)?;
        let modes = PyList::empty(py);
        for mode in &block.modes {
            let m = PyDict::new(py);
            m.set_item("eigenvalue", mode.eigenvalue)?;
            m.set_item("energy_curvature", mode.energy_curvature)?;
            m.set_item("amplitudes", &mode.amplitudes)?;
            m.set_item("absolute_residual", mode.absolute_residual)?;
            m.set_item("relative_residual", mode.relative_residual)?;
            m.set_item("iterations", mode.iterations)?;
            modes.append(m)?;
        }
        entry.set_item("modes", modes)?;
        blocks.append(entry)?;
    }
    out.set_item("blocks", &blocks)?;
    out.set_item("channels", blocks)?;
    Ok(out)
}

/// Frequency-dependent dipole polarizability, atomic units throughout.
#[pyfunction]
#[pyo3(signature = (numbers, positions, omega=0.0, damping=0.0, imaginary=false,
    basis="sto-3g", ecp=None, xc="hf", charge=0, multiplicity=1, grid_level=3,
    max_iter=150, dispersion="auto", density_fit=None, conv_tol=1e-9,
    conv_tol_grad=None, conv_tol_dp=1e-8, incore_mb=2048, vv10_grid_level=3,
    response_tol=1e-10, response_max_iter=500, solvent=None, eps=None,
    input_frequencies=None, process=None, frequencies=None, tensor_order=None, stability_roots=None, stability_channels=None,
    stability_resolution=None, stability_max_subspace=60, stability_block_buffer=3,
    stability_verify_lowest=true, stability_verify_steps=24, follow_unstable=false,
    max_descents=3, step_angle=0.3, min_energy_drop=1e-8, unrestricted=false))]
#[allow(clippy::too_many_arguments)]
fn dynamic_polarizability<'py>(
    py: Python<'py>,
    numbers: Vec<u8>,
    positions: Vec<[f64; 3]>,
    omega: f64,
    damping: f64,
    imaginary: bool,
    basis: &str,
    ecp: Option<&str>,
    xc: &str,
    charge: i32,
    multiplicity: usize,
    grid_level: usize,
    max_iter: usize,
    dispersion: &str,
    density_fit: Option<&str>,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    conv_tol_dp: f64,
    incore_mb: usize,
    vv10_grid_level: usize,
    response_tol: f64,
    response_max_iter: usize,
    solvent: Option<&str>,
    eps: Option<f64>,
    input_frequencies: Option<Vec<[f64; 2]>>,
    process: Option<&str>,
    frequencies: Option<Vec<f64>>,
    tensor_order: Option<usize>,
    stability_roots: Option<usize>,
    stability_channels: Option<Vec<String>>,
    stability_resolution: Option<f64>,
    stability_max_subspace: usize,
    stability_block_buffer: usize,
    stability_verify_lowest: bool,
    stability_verify_steps: usize,
    follow_unstable: bool,
    max_descents: usize,
    step_angle: f64,
    min_energy_drop: f64,
    unrestricted: bool,
) -> PyResult<Bound<'py, PyDict>> {
    use crate::response::{Frequency, ResponseSolverConfig};
    if let Some(roots) = stability_roots {
        if roots == 0
            || input_frequencies.is_some()
            || process.is_some()
            || frequencies.is_some()
            || tensor_order.is_some()
            || omega != 0.0
            || damping != 0.0
            || imaginary
        {
            return Err(PyValueError::new_err(
                "stability needs positive roots and no electric-frequency controls",
            ));
        }
    }
    let analysis = crate::response::StabilityOptions {
        n_roots: stability_roots.unwrap_or(3),
        tolerance: response_tol,
        max_iterations: response_max_iter,
        max_subspace: stability_max_subspace,
        block_buffer: stability_block_buffer,
        resolution: stability_resolution,
        channels: stability_channels
            .unwrap_or_default()
            .iter()
            .map(|c| crate::response::StabilityChannel::from_id(c).map_err(err))
            .collect::<PyResult<Vec<_>>>()?,
        verify_lowest: stability_verify_lowest,
        verify_steps: stability_verify_steps,
    };
    let following = crate::response::StabilizeOptions {
        analysis,
        max_descents,
        step_angle,
        min_energy_drop,
    };
    if stability_roots.is_some() {
        following.validate().map_err(err)?;
    } else if follow_unstable
        || !following.analysis.channels.is_empty()
        || stability_resolution.is_some()
        || stability_max_subspace != 60
        || stability_block_buffer != 3
        || !stability_verify_lowest
        || stability_verify_steps != 24
        || max_descents != 3
        || step_angle != 0.3
        || min_energy_drop != 1e-8
    {
        return Err(PyValueError::new_err(
            "stability-only controls require the stability entry point",
        ));
    }
    if imaginary && damping != 0.0 {
        return Err(PyValueError::new_err(
            "damping is not an imaginary-axis parameter",
        ));
    }
    let frequency = if imaginary {
        Frequency::Imaginary { frequency: omega }
    } else {
        Frequency::Retarded { omega, damping }
    };
    frequency.validate().map_err(err)?;
    if input_frequencies.is_some() && (omega != 0.0 || damping != 0.0) {
        return Err(PyValueError::new_err(
            "explicit input frequencies cannot be combined with single-frequency defaults",
        ));
    }
    let selected_process = process
        .map(|s| s.parse::<crate::response::ElectricProcess>().map_err(err))
        .transpose()?;
    if selected_process.is_some() && (input_frequencies.is_some() || imaginary || omega != 0.0) {
        return Err(PyValueError::new_err(
            "electric process presets require real optical frequencies and cannot be \
            combined with explicit input pairs or omega",
        ));
    }
    if selected_process.is_none() && frequencies.is_some() {
        return Err(PyValueError::new_err(
            "frequencies requires an electric process preset",
        ));
    }
    let tensor_frequencies = if let Some(process) = selected_process {
        Some(
            process
                .inputs(
                    frequencies.as_deref().ok_or_else(|| {
                        PyValueError::new_err("electric process requires frequencies")
                    })?,
                    damping,
                )
                .map_err(err)?,
        )
    } else {
        input_frequencies
            .map(|inputs| {
                if !(1..=3).contains(&inputs.len()) {
                    return Err(PyValueError::new_err(
                        "input_frequencies requires one, two or three [omega, damping] pairs",
                    ));
                }
                inputs
                    .iter()
                    .map(|&[omega, damping]| {
                        if imaginary && damping != 0.0 {
                            return Err(PyValueError::new_err(
                                "imaginary-axis inputs require zero damping",
                            ));
                        }
                        let f = if imaginary {
                            Frequency::Imaginary { frequency: omega }
                        } else {
                            Frequency::Retarded { omega, damping }
                        };
                        f.validate().map_err(err)?;
                        Ok(f)
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?
    };
    if let Some(order) = tensor_order {
        if !(1..=3).contains(&order) || tensor_frequencies.as_ref().map(Vec::len) != Some(order) {
            return Err(PyValueError::new_err(
                "electric process/input leg count does not match requested tensor order",
            ));
        }
    }
    let config = ResponseSolverConfig {
        tolerance: response_tol,
        max_iterations: response_max_iter,
        ..Default::default()
    };
    config.validate().map_err(err)?;
    let (mol, mb, functional, grid, mut opts, _) = prepare(
        &numbers,
        &positions,
        basis,
        ecp,
        xc,
        charge,
        multiplicity,
        grid_level,
        max_iter,
        dispersion,
        density_fit,
        conv_tol,
        conv_tol_grad,
        conv_tol_dp,
        incore_mb,
        vv10_grid_level,
    )?;
    opts.pcm = resolve_solvent(solvent, eps).map_err(err)?.0;
    if unrestricted {
        opts.reference = crate::scf::Reference::Uhf;
    }
    let scf = run_scf(&mol, &mb, &opts, &functional, grid.as_ref()).map_err(err)?;
    if stability_roots.is_some() {
        if follow_unstable {
            let report = crate::response::stabilize(
                &mol,
                &mb,
                &functional,
                grid.as_ref(),
                &scf,
                &opts,
                &following,
            )
            .map_err(err)?;
            return following_dict(py, &report);
        }
        let report = crate::response::stability_from_state(
            &mol,
            &mb,
            &functional,
            grid.as_ref(),
            &scf,
            &opts,
            &following.analysis,
        )
        .map_err(err)?;
        let out = stability_dict(py, &report)?;
        out.set_item("energy", scf.energy)?;
        out.set_item("stabilized", false)?;
        out.set_item("descents", PyList::empty(py))?;
        return Ok(out);
    }
    if let Some(frequencies) = tensor_frequencies {
        let mut report = crate::response::dynamic_electric_tensor(
            &mol,
            &mb,
            &functional,
            grid.as_ref(),
            &scf,
            &opts,
            &frequencies,
            &config,
        )
        .map_err(err)?;
        report.value.process = selected_process;
        let out = PyDict::new(py);
        response_metadata(py, &out, &report)?;
        let value = report.value;
        out.set_item("process", value.process.map(|p| p.name()))?;
        out.set_item(
            "tensor_real",
            value.tensor.iter().map(|z| z.re).collect::<Vec<_>>(),
        )?;
        out.set_item(
            "tensor_imag",
            value.tensor.iter().map(|z| z.im).collect::<Vec<_>>(),
        )?;
        out.set_item(
            "input_frequencies",
            frequencies
                .iter()
                .map(|f| {
                    let z = f.energy().expect("validated frequency");
                    [z.re, z.im]
                })
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            "output_frequency",
            [value.output_frequency.re, value.output_frequency.im],
        )?;
        out.set_item("projector_defect", value.projector_defect)?;
        out.set_item("equation_defect", value.equation_defect)?;
        out.set_item("particle_number_defect", value.particle_number_defect)?;
        out.set_item(
            "intrinsic_permutation_defect",
            value.intrinsic_permutation_defect,
        )?;
        out.set_item(
            "intrinsic_permutation_comparisons",
            value.intrinsic_permutation_comparisons,
        )?;
        out.set_item("subset_solves", value.subset_solves)?;
        out.set_item(
            "nonlinear_kernel_evaluations",
            value.nonlinear_kernel_evaluations,
        )?;
        out.set_item("energy", scf.energy)?;
        out.set_item("frequency_units", "hartree")?;
        return Ok(out);
    }
    let report = crate::response::dynamic_polarizability(
        &mol,
        &mb,
        &functional,
        grid.as_ref(),
        &scf,
        &opts,
        frequency,
        &config,
    )
    .map_err(err)?;
    let out = PyDict::new(py);
    response_metadata(py, &out, &report)?;
    out.set_item(
        "alpha_real",
        report
            .value
            .alpha
            .iter()
            .map(|row| row.iter().map(|v| v.re).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "alpha_imag",
        report
            .value
            .alpha
            .iter()
            .map(|row| row.iter().map(|v| v.im).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    )?;
    out.set_item("isotropic_real", report.value.isotropic.re)?;
    out.set_item("isotropic_imag", report.value.isotropic.im)?;
    out.set_item("reciprocity_defect", report.value.reciprocity_defect)?;
    out.set_item("omega", omega)?;
    out.set_item("damping", damping)?;
    out.set_item("imaginary", imaginary)?;
    out.set_item("energy", scf.energy)?;
    out.set_item("units", report.units)?;
    out.set_item("frequency_units", "hartree")?;
    out.set_item("model", report.model)?;
    out.set_item("axis_convention", report.axis_convention)?;
    out.set_item("converged", report.converged)?;
    out.set_item("residuals", report.residuals)?;
    out.set_item("iterations", report.iterations)?;
    out.set_item("warnings", report.warnings)?;
    Ok(out)
}

/// Analytic nonlinear molecular dipole response, factorial-normalized.
#[pyfunction]
#[pyo3(signature = (numbers, positions, direction, order=3,
    basis="sto-3g", ecp=None, xc="hf", charge=0, multiplicity=1, grid_level=3,
    max_iter=150, dispersion="auto", density_fit=None, conv_tol=1e-9,
    conv_tol_grad=None, conv_tol_dp=1e-8, incore_mb=2048, vv10_grid_level=3,
    response_tol=1e-10, response_max_iter=500, solvent=None, eps=None,
    reference_field=None, mixed_directions=None, tensor_order=None, tensor_assembly="projector"))]
#[allow(clippy::too_many_arguments)]
fn electric_field_taylor<'py>(
    py: Python<'py>,
    numbers: Vec<u8>,
    positions: Vec<[f64; 3]>,
    direction: [f64; 3],
    order: usize,
    basis: &str,
    ecp: Option<&str>,
    xc: &str,
    charge: i32,
    multiplicity: usize,
    grid_level: usize,
    max_iter: usize,
    dispersion: &str,
    density_fit: Option<&str>,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    conv_tol_dp: f64,
    incore_mb: usize,
    vv10_grid_level: usize,
    response_tol: f64,
    response_max_iter: usize,
    solvent: Option<&str>,
    eps: Option<f64>,
    reference_field: Option<[f64; 3]>,
    mixed_directions: Option<Vec<[f64; 3]>>,
    tensor_order: Option<usize>,
    tensor_assembly: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let config = crate::response::ResponseSolverConfig {
        tolerance: response_tol,
        max_iterations: response_max_iter,
        ..Default::default()
    };
    config.validate().map_err(err)?;
    if !matches!(tensor_assembly, "projector" | "variational")
        || (tensor_assembly == "variational" && tensor_order != Some(2))
    {
        return Err(PyValueError::new_err(
            "tensor_assembly must be projector, or variational with tensor_order=2",
        ));
    }
    if tensor_order.is_some_and(|n| !(2..=3).contains(&n))
        || (tensor_order.is_some()
            && (mixed_directions.is_some() || order != 0 || direction != [0.0; 3]))
    {
        return Err(PyValueError::new_err("tensor requests require tensor_order=2 or 3, order=0, direction=(0,0,0), and no mixed_directions"));
    }
    if direction.iter().any(|x| !x.is_finite())
        || reference_field.is_some_and(|v| v.iter().any(|x| !x.is_finite()))
    {
        return Err(PyValueError::new_err(
            "electric field and direction must be finite",
        ));
    }
    let (mol, mb, functional, grid, mut opts, _) = prepare(
        &numbers,
        &positions,
        basis,
        ecp,
        xc,
        charge,
        multiplicity,
        grid_level,
        max_iter,
        dispersion,
        density_fit,
        conv_tol,
        conv_tol_grad,
        conv_tol_dp,
        incore_mb,
        vv10_grid_level,
    )?;
    opts.pcm = resolve_solvent(solvent, eps).map_err(err)?.0;
    opts.external_field = reference_field.map(|f| crate::math::Vec3::new(f[0], f[1], f[2]));
    let scf = run_scf(&mol, &mb, &opts, &functional, grid.as_ref()).map_err(err)?;
    if let Some(n) = tensor_order {
        let report = if tensor_assembly == "variational" {
            crate::response::static_beta_variational(
                &mol,
                &mb,
                &functional,
                grid.as_ref(),
                &scf,
                &opts,
                &config,
            )
        } else {
            crate::response::static_electric_tensor(
                &mol,
                &mb,
                &functional,
                grid.as_ref(),
                &scf,
                &opts,
                n,
                &config,
            )
        }
        .map_err(err)?;
        let out = PyDict::new(py);
        response_metadata(py, &out, &report)?;
        let value = &report.value;
        out.set_item("order", value.order)?;
        out.set_item("tensor", &value.tensor)?;
        out.set_item("accumulation_roundoff", &value.accumulation_roundoff)?;
        out.set_item(
            "intrinsic_permutation_defect",
            value.intrinsic_permutation_defect,
        )?;
        out.set_item(
            "intrinsic_permutation_comparisons",
            value.intrinsic_permutation_comparisons,
        )?;
        out.set_item("energy_permutation_defect", value.energy_permutation_defect)?;
        out.set_item(
            "energy_permutation_comparisons",
            value.energy_permutation_comparisons,
        )?;
        out.set_item("frobenius_norm", value.frobenius_norm)?;
        out.set_item("beta_vector", value.beta_vector)?;
        out.set_item("beta_vector_norm", value.beta_vector_norm)?;
        out.set_item("gamma_isotropic", value.gamma_isotropic)?;
        out.set_item("ordered_field_evaluations", value.ordered_field_evaluations)?;
        out.set_item("directional_evaluations", value.directional_evaluations)?;
        out.set_item("projector_defect", value.projector_defect)?;
        out.set_item("stationarity_defect", value.stationarity_defect)?;
        out.set_item("particle_number_defect", value.particle_number_defect)?;
        out.set_item("reference_field", reference_field.unwrap_or([0.0; 3]))?;
        out.set_item("energy", scf.energy)?;
        return Ok(out);
    }
    if let Some(directions) = mixed_directions {
        if direction != [0.0; 3] || order != 0 {
            return Err(PyValueError::new_err("mixed field requests use order=0 and direction=(0,0,0); the derivative order is the direction count"));
        }
        let report = crate::response::mixed_electric_field_response(
            &mol,
            &mb,
            &functional,
            grid.as_ref(),
            &scf,
            &opts,
            &directions,
            &config,
        )
        .map_err(err)?;
        let out = PyDict::new(py);
        response_metadata(py, &out, &report)?;
        out.set_item("dipole_derivative", report.value.dipole_derivative)?;
        out.set_item("directions", report.value.directions)?;
        out.set_item("accumulation_roundoff", report.value.accumulation_roundoff)?;
        out.set_item(
            "directional_evaluations",
            report.value.directional_evaluations,
        )?;
        out.set_item("reference_field", reference_field.unwrap_or([0.0; 3]))?;
        out.set_item("energy", scf.energy)?;
        out.set_item("units", report.units)?;
        out.set_item("model", report.model)?;
        out.set_item("axis_convention", report.axis_convention)?;
        out.set_item("converged", report.converged)?;
        out.set_item("residuals", report.residuals)?;
        out.set_item("iterations", report.iterations)?;
        out.set_item("warnings", report.warnings)?;
        return Ok(out);
    }
    let report = crate::response::electric_field_taylor(
        &mol,
        &mb,
        &functional,
        grid.as_ref(),
        &scf,
        &opts,
        direction,
        order,
        &config,
    )
    .map_err(err)?;
    let out = PyDict::new(py);
    response_metadata(py, &out, &report)?;
    out.set_item("dipole_coefficients", report.value.dipole_coefficients)?;
    out.set_item("direction", direction)?;
    out.set_item("reference_field", reference_field.unwrap_or([0.0; 3]))?;
    out.set_item("projector_defects", report.value.projector_defects)?;
    out.set_item("stationarity_defects", report.value.stationarity_defects)?;
    out.set_item(
        "particle_number_derivatives",
        report.value.particle_number_derivatives,
    )?;
    out.set_item("energy", scf.energy)?;
    out.set_item("units", report.units)?;
    out.set_item("model", report.model)?;
    out.set_item("axis_convention", report.axis_convention)?;
    out.set_item("converged", report.converged)?;
    out.set_item("residuals", report.residuals)?;
    out.set_item("iterations", report.iterations)?;
    out.set_item("warnings", report.warnings)?;
    Ok(out)
}

/// Analytic moving-geometry energy derivatives, including cubic and quartic.
#[pyfunction]
#[pyo3(signature = (numbers, positions, direction, order=4,
    basis="sto-3g", ecp=None, xc="hf", charge=0, multiplicity=1, grid_level=3,
    max_iter=150, dispersion="none", density_fit=None, conv_tol=1e-9,
    conv_tol_grad=None, conv_tol_dp=1e-8, incore_mb=2048, vv10_grid_level=3,
    response_tol=1e-10, response_max_iter=500, mixed_directions=None,
    field_direction=None, reference_field=None, mixed_field_directions=None, electric_order=None,
    force_constant_order=None, force_coordinates=None, orbital_gradients=false,
    frontier_gap=false, orbital_coordinates=None, degeneracy_tolerance=1e-8, unrestricted=false))]
#[allow(clippy::too_many_arguments)]
fn nuclear_taylor<'py>(
    py: Python<'py>,
    numbers: Vec<u8>,
    positions: Vec<[f64; 3]>,
    direction: Vec<[f64; 3]>,
    order: usize,
    basis: &str,
    ecp: Option<&str>,
    xc: &str,
    charge: i32,
    multiplicity: usize,
    grid_level: usize,
    max_iter: usize,
    dispersion: &str,
    density_fit: Option<&str>,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    conv_tol_dp: f64,
    incore_mb: usize,
    vv10_grid_level: usize,
    response_tol: f64,
    response_max_iter: usize,
    mixed_directions: Option<Vec<Vec<[f64; 3]>>>,
    field_direction: Option<[f64; 3]>,
    reference_field: Option<[f64; 3]>,
    mixed_field_directions: Option<Vec<[f64; 3]>>,
    electric_order: Option<usize>,
    force_constant_order: Option<usize>,
    force_coordinates: Option<Vec<usize>>,
    orbital_gradients: bool,
    frontier_gap: bool,
    orbital_coordinates: Option<Vec<usize>>,
    degeneracy_tolerance: f64,
    unrestricted: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let config = crate::response::ResponseSolverConfig {
        tolerance: response_tol,
        max_iterations: response_max_iter,
        ..Default::default()
    };
    config.validate().map_err(err)?;
    if order == usize::MAX
        || direction.len() != numbers.len()
        || direction.iter().flatten().any(|x| !x.is_finite())
    {
        return Err(PyValueError::new_err(
            "nuclear direction must contain one finite Cartesian displacement per atom",
        ));
    }
    if let Some(directions) = &mixed_directions {
        if order != 0
            || direction.iter().flatten().any(|&x| x != 0.0)
            || directions.is_empty()
            || directions
                .iter()
                .any(|d| d.len() != numbers.len() || d.iter().flatten().any(|x| !x.is_finite()))
        {
            return Err(PyValueError::new_err("mixed nuclear response requires finite per-atom directions, order=0 and a zero placeholder direction"));
        }
    }
    if field_direction
        .iter()
        .chain(reference_field.iter())
        .flatten()
        .any(|x| !x.is_finite())
    {
        return Err(PyValueError::new_err(
            "electric fields must be finite Cartesian vectors",
        ));
    }
    if let Some(fields) = &mixed_field_directions {
        if fields.iter().flatten().any(|x| !x.is_finite())
            || mixed_directions.as_ref().map(|d| d.len()) != Some(fields.len())
        {
            return Err(PyValueError::new_err(
                "mixed nuclear and electric direction counts must agree and be finite",
            ));
        }
    }
    if (mixed_directions.is_some() && field_direction.is_some())
        || electric_order.is_some_and(|n| {
            !(1..=3).contains(&n)
                || order != 0
                || mixed_directions.is_some()
                || field_direction.is_some()
        })
    {
        return Err(PyValueError::new_err(
            "spectroscopy requires electric_order=1..3, order=0 and one nuclear \
            direction; incompatible mixed controls are rejected",
        ));
    }
    if force_coordinates.is_some() && force_constant_order.is_none()
        || force_constant_order.is_some_and(|n| {
            !(3..=4).contains(&n)
                || order != 0
                || direction.iter().flatten().any(|&x| x != 0.0)
                || field_direction.is_some()
                || mixed_field_directions.is_some()
                || electric_order.is_some()
                || (force_coordinates.is_some() && mixed_directions.is_some())
        })
    {
        return Err(PyValueError::new_err(
            "force constants require order=3 or 4 and either displacement patterns \
            or Cartesian coordinates; mixed electric/spectroscopy controls are \
            incompatible",
        ));
    }
    if let Some(coordinates) = &force_coordinates {
        let mut sorted = coordinates.clone();
        sorted.sort_unstable();
        if sorted.is_empty()
            || sorted.iter().any(|&i| i >= 3 * numbers.len())
            || sorted.windows(2).any(|w| w[0] == w[1])
        {
            return Err(PyValueError::new_err(
                "force coordinates must be unique nonempty Cartesian IDs 3*atom+axis",
            ));
        }
    }
    let orbital_options = crate::response::OrbitalGradientOptions {
        coordinates: orbital_coordinates,
        degeneracy_tolerance,
    };
    if orbital_gradients {
        orbital_options.validate(numbers.len()).map_err(err)?;
        if order != 0
            || direction.iter().flatten().any(|&v| v != 0.0)
            || mixed_directions.is_some()
            || field_direction.is_some()
            || mixed_field_directions.is_some()
            || electric_order.is_some()
            || force_constant_order.is_some()
            || force_coordinates.is_some()
        {
            return Err(PyValueError::new_err(
                "orbital gradients use a zero placeholder direction/order and their own \
                Cartesian coordinates; mixed/Taylor/spectroscopy controls are \
                incompatible",
            ));
        }
    } else if frontier_gap || orbital_options.coordinates.is_some() || degeneracy_tolerance != 1e-8
    {
        return Err(PyValueError::new_err(
            "orbital controls require orbital_gradients",
        ));
    }
    let (mol, mb, functional, grid, mut opts, empirical) = prepare(
        &numbers,
        &positions,
        basis,
        ecp,
        xc,
        charge,
        multiplicity,
        grid_level,
        max_iter,
        dispersion,
        density_fit,
        conv_tol,
        conv_tol_grad,
        conv_tol_dp,
        incore_mb,
        vv10_grid_level,
    )?;
    if empirical != crate::dispersion::Dispersion::None
        || !mb.ecps.is_empty()
        || opts.vv10.is_some()
        || functional.required_nonlocal_kind().is_some()
        || opts.density_fit.is_some()
    {
        return Err(PyValueError::new_err(
            "nuclear Taylor response currently requires all-electron exact-integral \
            HF/semilocal/hybrid/RSH without empirical or nonlocal corrections",
        ));
    }
    opts.external_field = reference_field.map(|v| Vec3::new(v[0], v[1], v[2]));
    if unrestricted {
        opts.reference = crate::scf::Reference::Uhf;
    }
    let scf = run_scf(&mol, &mb, &opts, &functional, grid.as_ref()).map_err(err)?;
    let out = PyDict::new(py);
    if orbital_gradients {
        if frontier_gap {
            let report = crate::response::frontier_gap_gradient(
                &mol,
                &mb,
                &functional,
                grid.as_ref(),
                &scf,
                &opts,
                &orbital_options,
                &config,
            )
            .map_err(err)?;
            response_metadata(py, &out, &report)?;
            orbital_gradient_items(py, &out, &report.value.orbitals)?;
            let mut gaps = Vec::new();
            for gap in &report.value.gaps {
                if let Some(gap) = gap {
                    let item = PyDict::new(py);
                    item.set_item("spin", gap.spin)?;
                    item.set_item("homo", gap.homo)?;
                    item.set_item("lumo", gap.lumo)?;
                    item.set_item("gap", gap.gap)?;
                    item.set_item("gradient", &gap.gradient)?;
                    gaps.push(Some(item));
                } else {
                    gaps.push(None);
                }
            }
            out.set_item("gaps", gaps)?;
        } else {
            let report = crate::response::orbital_energy_gradients(
                &mol,
                &mb,
                &functional,
                grid.as_ref(),
                &scf,
                &opts,
                &orbital_options,
                &config,
            )
            .map_err(err)?;
            response_metadata(py, &out, &report)?;
            orbital_gradient_items(py, &out, &report.value)?;
        }
        out.set_item("energy", scf.energy)?;
        return Ok(out);
    }
    if let Some(rank) = force_constant_order {
        let report = if let Some(patterns) = mixed_directions {
            crate::response::force_constants::analytic_force_constants(
                &mol,
                &mb,
                &functional,
                grid.as_ref(),
                &scf,
                &opts,
                &patterns,
                rank,
                &config,
            )
        } else if let Some(coordinates) = force_coordinates {
            crate::response::force_constants::cartesian_force_constant_block(
                &mol,
                &mb,
                &functional,
                grid.as_ref(),
                &scf,
                &opts,
                &coordinates,
                rank,
                &config,
            )
        } else {
            crate::response::force_constants::cartesian_force_constants(
                &mol,
                &mb,
                &functional,
                grid.as_ref(),
                &scf,
                &opts,
                rank,
                &config,
            )
        }
        .map_err(err)?;
        response_metadata(py, &out, &report)?;
        let v = report.value;
        out.set_item("order", v.order)?;
        out.set_item("patterns", v.patterns)?;
        out.set_item("tensor", v.tensor)?;
        out.set_item("accumulation_roundoff", v.accumulation_roundoff)?;
        out.set_item("permutation_defect", v.permutation_defect)?;
        out.set_item("permutation_comparisons", v.permutation_comparisons)?;
        out.set_item("translation_residual", v.translation_residual)?;
        out.set_item("translation_comparisons", v.translation_comparisons)?;
        out.set_item("independent_components", v.independent_components)?;
        out.set_item("directional_evaluations", v.directional_evaluations)?;
        out.set_item("energy", scf.energy)?;
        return Ok(out);
    }
    if let Some(rank) = electric_order {
        let report = crate::response::nuclear::static_spectroscopic_derivative(
            &mol,
            &mb,
            &functional,
            grid.as_ref(),
            &scf,
            &opts,
            &direction,
            rank,
            &config,
        )
        .map_err(err)?;
        response_metadata(py, &out, &report)?;
        out.set_item("nuclear_direction", report.value.nuclear_direction)?;
        out.set_item("electric_order", rank)?;
        out.set_item("tensor", report.value.tensor)?;
        out.set_item("accumulation_roundoff", report.value.accumulation_roundoff)?;
        out.set_item("permutation_defect", report.value.permutation_defect)?;
        out.set_item(
            "permutation_comparisons",
            report.value.permutation_comparisons,
        )?;
        out.set_item(
            "directional_evaluations",
            report.value.directional_evaluations,
        )?;
        out.set_item("energy", scf.energy)?;
        return Ok(out);
    }
    if let Some(directions) = mixed_directions {
        if let Some(fields) = mixed_field_directions {
            let paths: Vec<_> = directions
                .into_iter()
                .zip(fields)
                .map(
                    |(nuclear, electric)| crate::response::nuclear::MolecularDirection {
                        nuclear,
                        electric,
                    },
                )
                .collect();
            let report = crate::response::nuclear::mixed_molecular_response(
                &mol,
                &mb,
                &functional,
                grid.as_ref(),
                &scf,
                &opts,
                &paths,
                &config,
            )
            .map_err(err)?;
            response_metadata(py, &out, &report)?;
            out.set_item("energy_derivative", report.value.energy_derivative)?;
            out.set_item("dipole_derivative", report.value.dipole_derivative)?;
            out.set_item(
                "nuclear_directions",
                report
                    .value
                    .directions
                    .iter()
                    .map(|d| &d.nuclear)
                    .collect::<Vec<_>>(),
            )?;
            out.set_item(
                "electric_directions",
                report
                    .value
                    .directions
                    .iter()
                    .map(|d| d.electric)
                    .collect::<Vec<_>>(),
            )?;
            out.set_item(
                "accumulation_roundoff",
                report.value.energy_accumulation_roundoff,
            )?;
            out.set_item(
                "dipole_accumulation_roundoff",
                report.value.dipole_accumulation_roundoff,
            )?;
            out.set_item(
                "directional_evaluations",
                report.value.directional_evaluations,
            )?;
            out.set_item("energy", scf.energy)?;
            return Ok(out);
        }
        let report = crate::response::nuclear::mixed_nuclear_response(
            &mol,
            &mb,
            &functional,
            grid.as_ref(),
            &scf,
            &opts,
            &directions,
            &config,
        )
        .map_err(err)?;
        response_metadata(py, &out, &report)?;
        out.set_item("energy_derivative", report.value.energy_derivative)?;
        out.set_item("directions", report.value.directions)?;
        out.set_item("accumulation_roundoff", report.value.accumulation_roundoff)?;
        out.set_item(
            "directional_evaluations",
            report.value.directional_evaluations,
        )?;
        out.set_item("energy", scf.energy)?;
        return Ok(out);
    }
    let report = crate::response::nuclear::molecular_taylor(
        &mol,
        &mb,
        &functional,
        grid.as_ref(),
        &scf,
        &opts,
        &crate::response::nuclear::MolecularDirection {
            nuclear: direction,
            electric: field_direction.unwrap_or([0.0; 3]),
        },
        order,
        &config,
    )
    .map_err(err)?;
    response_metadata(py, &out, &report)?;
    let v = &report.value;
    out.set_item("direction", &v.direction)?;
    out.set_item("field_direction", v.field_direction)?;
    out.set_item("dipole_coefficients", &v.dipole_coefficients)?;
    out.set_item(
        "dipole_units",
        "atomic units; factorial-normalized along the supplied path",
    )?;
    out.set_item("energy_coefficients", &v.energy_coefficients)?;
    let components = PyDict::new(py);
    components.set_item(
        "nuclear_field",
        v.energy_components
            .iter()
            .map(|e| e.nuclear_field)
            .collect::<Vec<_>>(),
    )?;
    components.set_item(
        "nuclear_repulsion",
        v.energy_components
            .iter()
            .map(|e| e.nuclear_repulsion)
            .collect::<Vec<_>>(),
    )?;
    components.set_item(
        "one_electron",
        v.energy_components
            .iter()
            .map(|e| e.one_electron)
            .collect::<Vec<_>>(),
    )?;
    components.set_item(
        "hartree",
        v.energy_components
            .iter()
            .map(|e| e.hartree)
            .collect::<Vec<_>>(),
    )?;
    components.set_item(
        "exchange",
        v.energy_components
            .iter()
            .map(|e| e.exchange)
            .collect::<Vec<_>>(),
    )?;
    components.set_item(
        "long_range_exchange",
        v.energy_components
            .iter()
            .map(|e| e.long_range_exchange)
            .collect::<Vec<_>>(),
    )?;
    components.set_item(
        "semilocal_xc",
        v.energy_components
            .iter()
            .map(|e| e.semilocal_xc)
            .collect::<Vec<_>>(),
    )?;
    out.set_item("energy_components", components)?;
    out.set_item(
        "density_coefficients",
        v.density_coefficients
            .iter()
            .map(|p| p.iter().map(mat_rows).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "overlap_coefficients",
        v.overlap_coefficients
            .iter()
            .map(mat_rows)
            .collect::<Vec<_>>(),
    )?;
    out.set_item("projector_defects", &v.projector_defects)?;
    out.set_item("stationarity_defects", &v.stationarity_defects)?;
    out.set_item("metric_defects", &v.metric_defects)?;
    out.set_item(
        "particle_number_coefficients",
        &v.particle_number_coefficients,
    )?;
    out.set_item("reference_energy_error", v.reference_energy_error)?;
    out.set_item("reference_fock_error", v.reference_fock_error)?;
    out.set_item(
        "orthonormal_fock_coefficients",
        v.orthonormal_fock_coefficients
            .iter()
            .map(|f| f.iter().map(mat_rows).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    )?;
    out.set_item("energy", scf.energy)?;
    Ok(out)
}

fn orbital_gradient_items<'py>(
    py: Python<'py>,
    out: &Bound<'py, PyDict>,
    v: &crate::response::OrbitalEnergyGradients,
) -> PyResult<()> {
    out.set_item("coordinates", &v.coordinates)?;
    out.set_item("orbital_energies", &v.orbital_energies)?;
    out.set_item("occupations", v.occupations)?;
    out.set_item("gradients", &v.gradients)?;
    out.set_item("degeneracy_tolerance", v.degeneracy_tolerance)?;
    out.set_item("translation_residual", v.translation_residual)?;
    out.set_item("hermiticity_defects", &v.hermiticity_defects)?;
    out.set_item("projector_defects", &v.projector_defects)?;
    out.set_item("stationarity_defects", &v.stationarity_defects)?;
    out.set_item("metric_defects", &v.metric_defects)?;
    out.set_item(
        "particle_number_coefficients",
        &v.particle_number_coefficients,
    )?;
    out.set_item("reference_energy_errors", &v.reference_energy_errors)?;
    out.set_item("reference_fock_errors", &v.reference_fock_errors)?;
    let mut clusters = Vec::new();
    for cluster in &v.clusters {
        let item = PyDict::new(py);
        item.set_item("spin", cluster.spin)?;
        item.set_item("orbitals", &cluster.orbitals)?;
        item.set_item("energies", &cluster.energies)?;
        item.set_item("energy_spread", cluster.energy_spread)?;
        item.set_item(
            "derivative_matrices",
            cluster
                .derivative_matrices
                .iter()
                .map(mat_rows)
                .collect::<Vec<_>>(),
        )?;
        item.set_item("commutator_defect", cluster.commutator_defect)?;
        clusters.push(item);
    }
    out.set_item("clusters", clusters)?;
    Ok(())
}

/// Analytic rectilinear molecular anharmonic frequencies and IR/static Raman.
#[pyfunction]
#[pyo3(signature = (numbers, positions, basis="sto-3g", ecp=None, xc="hf", charge=0,
    multiplicity=1, grid_level=3, max_iter=150, dispersion="none", density_fit=None,
    conv_tol=1e-9, conv_tol_grad=None, conv_tol_dp=1e-8, incore_mb=2048, vv10_grid_level=3,
    response_tol=1e-10, response_max_iter=500, mode_ids=None, include_raman=false,
    equilibrium_tol=1e-4, resonance_tol=1e-8, resonance_coupling_factor=1.0,
    max_basis_states=1024, max_polyad_states=256, method="retained-polyad"))]
#[allow(clippy::too_many_arguments)]
fn anharmonic<'py>(
    py: Python<'py>,
    numbers: Vec<u8>,
    positions: Vec<[f64; 3]>,
    basis: &str,
    ecp: Option<&str>,
    xc: &str,
    charge: i32,
    multiplicity: usize,
    grid_level: usize,
    max_iter: usize,
    dispersion: &str,
    density_fit: Option<&str>,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    conv_tol_dp: f64,
    incore_mb: usize,
    vv10_grid_level: usize,
    response_tol: f64,
    response_max_iter: usize,
    mode_ids: Option<Vec<usize>>,
    include_raman: bool,
    equilibrium_tol: f64,
    resonance_tol: f64,
    resonance_coupling_factor: f64,
    max_basis_states: usize,
    max_polyad_states: usize,
    method: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let config = crate::response::anharmonic::AnharmonicConfig {
        method: crate::response::anharmonic::AnharmonicMethod::parse(method).map_err(err)?,
        equilibrium_tolerance: equilibrium_tol,
        resonance_tolerance: resonance_tol,
        resonance_coupling_factor,
        max_basis_states,
        max_polyad_states,
        tolerance: response_tol,
        memory_mb: incore_mb.min(512),
    };
    config.validate().map_err(err)?;
    let response_config = crate::response::ResponseSolverConfig {
        tolerance: response_tol,
        max_iterations: response_max_iter,
        ..Default::default()
    };
    response_config.validate().map_err(err)?;
    let (mol, mb, functional, grid, opts, empirical) = prepare(
        &numbers,
        &positions,
        basis,
        ecp,
        xc,
        charge,
        multiplicity,
        grid_level,
        max_iter,
        dispersion,
        density_fit,
        conv_tol,
        conv_tol_grad,
        conv_tol_dp,
        incore_mb,
        vv10_grid_level,
    )?;
    if empirical != crate::dispersion::Dispersion::None
        || !mb.ecps.is_empty()
        || opts.vv10.is_some()
        || functional.required_nonlocal_kind().is_some()
        || opts.density_fit.is_some()
    {
        return Err(PyValueError::new_err(
            "molecular anharmonic response currently requires all-electron \
            exact-integral models without nonlocal or empirical corrections",
        ));
    }
    let scf = run_scf(&mol, &mb, &opts, &functional, grid.as_ref()).map_err(err)?;
    let report = crate::response::anharmonic::molecular::molecular_anharmonic(
        &mol,
        &mb,
        &functional,
        grid.as_ref(),
        &scf,
        &opts,
        mode_ids.as_deref(),
        include_raman,
        &config,
        &response_config,
    )
    .map_err(err)?;
    let out = PyDict::new(py);
    response_metadata(py, &out, &report)?;
    let v = &report.value;
    let s = &v.spectrum;
    out.set_item("anharmonic_method", s.method.as_str())?;
    out.set_item("energy_order", 2)?;
    out.set_item("energy_treatment", s.method.energy_treatment())?;
    out.set_item("transition_moment_order", 2)?;
    out.set_item("intensity_treatment", s.method.intensity_treatment())?;
    out.set_item("resonance_treatment", s.method.resonance_treatment())?;
    out.set_item("eigensystem_kind", s.method.eigensystem_kind())?;
    out.set_item(
        "convergence_scope",
        "numerical diagnostics; perturbation validity is separate",
    )?;
    out.set_item("selected_mode_ids", &v.selected_mode_ids)?;
    out.set_item("harmonic_frequencies_cm", &v.harmonic_frequencies_cm)?;
    out.set_item("patterns", &v.patterns)?;
    out.set_item("n_projected", v.n_projected)?;
    out.set_item("maximum_force", v.maximum_force)?;
    out.set_item("force_permutation_defects", v.force_permutation_defects)?;
    out.set_item(
        "force_permutation_comparisons",
        v.force_permutation_comparisons,
    )?;
    out.set_item(
        "polarizability_symmetry_defect",
        v.polarizability_symmetry_defect,
    )?;
    out.set_item(
        "polarizability_symmetry_comparisons",
        v.polarizability_symmetry_comparisons,
    )?;
    out.set_item("cubic", &v.polynomial.cubic)?;
    out.set_item("quartic", &v.polynomial.quartic)?;
    if let Some(rotation) = &v.polynomial.rotation {
        let metadata = PyDict::new(py);
        metadata.set_item(
            "class",
            match rotation.class {
                crate::response::anharmonic::rotation::RotorClass::Linear => "linear",
                crate::response::anharmonic::rotation::RotorClass::Nonlinear => "nonlinear",
            },
        )?;
        metadata.set_item("inverse_inertia", rotation.inverse_inertia)?;
        metadata.set_item("principal_moments", rotation.principal_moments)?;
        metadata.set_item("coriolis", &rotation.coriolis)?;
        metadata.set_item("watson_pseudopotential", rotation.watson_pseudopotential)?;
        metadata.set_item("eckart_defect", rotation.eckart_defect)?;
        metadata.set_item("mode_metric_defect", rotation.mode_metric_defect)?;
        out.set_item("rotation", metadata)?;
    }
    let serialize =
        |o: &crate::response::anharmonic::ObservablePolynomial| -> PyResult<Bound<'py, PyDict>> {
            let d = PyDict::new(py);
            d.set_item("components", o.components)?;
            d.set_item("first", &o.first)?;
            d.set_item("second", &o.second)?;
            d.set_item("third", &o.third)?;
            Ok(d)
        };
    out.set_item("dipole_derivatives", serialize(&v.polynomial.dipole)?)?;
    if let Some(alpha) = &v.polynomial.polarizability {
        out.set_item("polarizability_derivatives", serialize(alpha)?)?;
    } else {
        out.set_item("polarizability_derivatives", py.None())?;
    }
    let mut bands = Vec::new();
    for b in &s.bands {
        let d = PyDict::new(py);
        d.set_item("dominant_occupation", &b.dominant_occupation)?;
        d.set_item("character", b.character)?;
        d.set_item("dominant_weight", b.dominant_weight)?;
        d.set_item("frequency_cm", b.frequency_cm)?;
        d.set_item("harmonic_frequency_cm", b.harmonic_frequency_cm)?;
        d.set_item("transition_dipole", b.transition_dipole)?;
        d.set_item("double_harmonic_dipole", b.double_harmonic_dipole)?;
        d.set_item("mechanical_dipole", b.mechanical_dipole)?;
        d.set_item("electrical_dipole", b.electrical_dipole)?;
        d.set_item("mixed_dipole", b.mixed_dipole)?;
        d.set_item("ir_intensity", b.ir_intensity)?;
        d.set_item(
            "double_harmonic_ir_intensity",
            b.double_harmonic_ir_intensity,
        )?;
        d.set_item("transition_polarizability", b.transition_polarizability)?;
        d.set_item("raman_activity", b.raman_activity)?;
        d.set_item(
            "double_harmonic_raman_activity",
            b.double_harmonic_raman_activity,
        )?;
        bands.push(d);
    }
    out.set_item("bands", bands)?;
    out.set_item("zero_point_energy", s.zero_point_energy)?;
    out.set_item("harmonic_zero_point_energy", s.harmonic_zero_point_energy)?;
    out.set_item("polyad_occupations", &s.polyad_occupations)?;
    out.set_item("polyad_energies", &s.polyad_energies)?;
    out.set_item("polyad_vectors", mat_rows(&s.polyad_vectors))?;
    out.set_item("effective_hamiltonian", mat_rows(&s.effective_hamiltonian))?;
    out.set_item("basis_states", s.basis_states)?;
    out.set_item("added_resonant_states", s.added_resonant_states)?;
    out.set_item(
        "minimum_external_denominator",
        s.minimum_external_denominator,
    )?;
    out.set_item("exterior_mixing_norm", s.exterior_mixing_norm)?;
    out.set_item("hermiticity_defect", s.hermiticity_defect)?;
    out.set_item("eigen_residual", s.eigen_residual)?;
    out.set_item("energy", scf.energy)?;
    Ok(out)
}

/// Collinear orbital NMR shielding tensors in ppm.
#[pyfunction]
#[pyo3(signature = (numbers, positions, basis="sto-3g", ecp=None, xc="hf", charge=0,
    multiplicity=1, grid_level=3, gauge="giao", origin=None, max_iter=150,
    density_fit=None, conv_tol=1e-9, conv_tol_grad=None, conv_tol_dp=1e-8,
    incore_mb=2048, vv10_grid_level=3, response_tol=1e-10, response_max_iter=500,
    response_restart=40, unrestricted=false, density_response=false))]
#[allow(clippy::too_many_arguments)]
fn nmr<'py>(
    py: Python<'py>,
    numbers: Vec<u8>,
    positions: Vec<[f64; 3]>,
    basis: &str,
    ecp: Option<&str>,
    xc: &str,
    charge: i32,
    multiplicity: usize,
    grid_level: usize,
    gauge: &str,
    origin: Option<[f64; 3]>,
    max_iter: usize,
    density_fit: Option<&str>,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    conv_tol_dp: f64,
    incore_mb: usize,
    vv10_grid_level: usize,
    response_tol: f64,
    response_max_iter: usize,
    response_restart: usize,
    unrestricted: bool,
    density_response: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let (mol, mb, functional, grid, mut opts, _dispersion) = prepare(
        &numbers,
        &positions,
        basis,
        ecp,
        xc,
        charge,
        multiplicity,
        grid_level,
        max_iter,
        "none",
        density_fit,
        conv_tol,
        conv_tol_grad,
        conv_tol_dp,
        incore_mb,
        vv10_grid_level,
    )?;
    let gauge = crate::properties::magnetic::Gauge::from_name(gauge).map_err(err)?;
    let config = crate::response::ResponseSolverConfig {
        tolerance: response_tol,
        max_iterations: response_max_iter,
        gmres_restart: response_restart,
        ..Default::default()
    };
    config.validate().map_err(err)?;
    if unrestricted {
        opts.reference = crate::scf::Reference::Uhf;
    }
    let origin = origin.unwrap_or([0.0; 3]);
    if origin.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("NMR origin must be finite"));
    }
    let scf = run_scf(&mol, &mb, &opts, &functional, grid.as_ref()).map_err(err)?;
    let o = Vec3::new(origin[0], origin[1], origin[2]);
    let rows = |m: &Matrix| {
        (0..m.rows)
            .map(|i| (0..m.cols).map(|j| m[(i, j)]).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };
    if density_response {
        let report = crate::properties::magnetic::magnetic_density_response(
            &mol,
            &mb,
            &functional,
            grid.as_ref(),
            &scf,
            &opts,
            gauge,
            o,
            &config,
        )
        .map_err(err)?;
        let out = PyDict::new(py);
        response_metadata(py, &out, &report)?;
        out.set_item(
            "density_hat",
            report
                .value
                .density_hat
                .iter()
                .map(|s| s.iter().map(rows).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            "overlap_hat",
            report
                .value
                .overlap_hat
                .iter()
                .map(rows)
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            "explicit_fock_hat",
            report
                .value
                .explicit_fock_hat
                .iter()
                .map(|s| s.iter().map(rows).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            "reference_density",
            scf.density.iter().map(rows).collect::<Vec<_>>(),
        )?;
        out.set_item(
            "reference_fock",
            scf.fock.iter().map(rows).collect::<Vec<_>>(),
        )?;
        out.set_item("overlap", rows(&scf.overlap))?;
        out.set_item(
            "mo_coeff",
            scf.mo_coeff.iter().map(rows).collect::<Vec<_>>(),
        )?;
        out.set_item("mo_energy", &scf.mo_energy)?;
        out.set_item("occupied", report.value.occupied)?;
        out.set_item("projector_defect", report.value.projector_defect)?;
        out.set_item(
            "particle_number_defect",
            report.value.particle_number_defect,
        )?;
        out.set_item("antisymmetry_defect", report.value.antisymmetry_defect)?;
        out.set_item("solver_methods", report.value.solver_methods)?;
        out.set_item("gauge", format!("{gauge:?}"))?;
        out.set_item("origin", origin.to_vec())?;
        out.set_item("origin_units", "bohr")?;
        out.set_item("energy", scf.energy)?;
        out.set_item("unrestricted", scf.unrestricted)?;
        return Ok(out);
    }
    let report = crate::properties::magnetic::molecular_nmr_response(
        &mol,
        &mb,
        &functional,
        grid.as_ref(),
        &scf,
        &opts,
        gauge,
        o,
        &config,
    )
    .map_err(err)?;
    let entries = PyList::empty(py);
    for (index, shielding) in report.value.iter().enumerate() {
        let entry = PyDict::new(py);
        entry.set_item("atom", index)?;
        entry.set_item("atomic_number", numbers[index])?;
        entry.set_item("tensor", shielding.tensor.map(|row| row.to_vec()).to_vec())?;
        entry.set_item(
            "diamagnetic",
            shielding.diamagnetic.map(|row| row.to_vec()).to_vec(),
        )?;
        entry.set_item(
            "paramagnetic",
            shielding.paramagnetic.map(|row| row.to_vec()).to_vec(),
        )?;
        entry.set_item("isotropic", shielding.isotropic())?;
        let (dia_iso, para_iso) = shielding.isotropic_parts();
        entry.set_item("isotropic_diamagnetic", dia_iso)?;
        entry.set_item("isotropic_paramagnetic", para_iso)?;
        entries.append(entry)?;
    }
    let out = PyDict::new(py);
    response_metadata(py, &out, &report)?;
    out.set_item("unrestricted", scf.unrestricted)?;
    out.set_item("origin_units", "bohr")?;
    out.set_item("energy", scf.energy)?;
    out.set_item("gauge", format!("{gauge:?}"))?;
    out.set_item("origin", origin.to_vec())?;
    out.set_item("shieldings", entries)?;
    Ok(out)
}

/// TDA or full-RPA excitation energies, oscillator strengths, and length-
/// and velocity-gauge rotatory strengths (ECD).
///
/// `method`: `"tda"` (Tamm-Dancoff, default) or `"rpa"` (full TDDFT/TDHF via
/// the Casida A/B form). `spin`: `"singlet"` / `"triplet"` for a restricted
/// reference, `"spin_resolved"` for an unrestricted one, or `"auto"` (picks
/// singlet/spin_resolved from `scf.unrestricted`, matching the CLI default).
///
/// Rotatory strengths are in atomic units; multiply by 471.4438 for the
/// customary 1e-40 cgs unit the CLI prints. The length gauge is
/// origin-dependent in a finite basis, the velocity gauge is not — the gap
/// between them on an achiral molecule (where both must vanish) is the
/// basis-set error bar, not noise.
#[pyfunction]
#[pyo3(signature = (numbers, positions, basis="sto-3g", ecp=None, xc="hf", charge=0,
    multiplicity=1, grid_level=3, nstates=5, method="tda", spin="auto", max_iter=150,
    dispersion="auto", density_fit=None, conv_tol=1e-9, conv_tol_grad=None,
    conv_tol_dp=1e-8, incore_mb=2048, vv10_grid_level=3))]
#[allow(clippy::too_many_arguments)]
fn tddft<'py>(
    py: Python<'py>,
    numbers: Vec<u8>,
    positions: Vec<[f64; 3]>,
    basis: &str,
    ecp: Option<&str>,
    xc: &str,
    charge: i32,
    multiplicity: usize,
    grid_level: usize,
    nstates: usize,
    method: &str,
    spin: &str,
    max_iter: usize,
    dispersion: &str,
    density_fit: Option<&str>,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    conv_tol_dp: f64,
    incore_mb: usize,
    vv10_grid_level: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use crate::response::tddft::{TdaSolver, TdaSpin};
    let (mol, mb, functional, grid, opts, _empirical_dispersion) = prepare(
        &numbers,
        &positions,
        basis,
        ecp,
        xc,
        charge,
        multiplicity,
        grid_level,
        max_iter,
        dispersion,
        density_fit,
        conv_tol,
        conv_tol_grad,
        conv_tol_dp,
        incore_mb,
        vv10_grid_level,
    )?;
    let scf = run_scf(&mol, &mb, &opts, &functional, grid.as_ref()).map_err(err)?;
    let mode = match spin {
        "singlet" => TdaSpin::Singlet,
        "triplet" => TdaSpin::Triplet,
        "spin_resolved" | "spinresolved" => TdaSpin::SpinResolved,
        "auto" => {
            if scf.unrestricted {
                TdaSpin::SpinResolved
            } else {
                TdaSpin::Singlet
            }
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown spin `{other}` (singlet|triplet|spin_resolved|auto)"
            )))
        }
    };
    let solver =
        TdaSolver::new(&mol, &mb, &functional, grid.as_ref(), &scf, &opts, mode).map_err(err)?;
    let states = match method {
        "tda" => solver.solve(nstates).map_err(err)?,
        "rpa" => solver.solve_rpa(nstates).map_err(err)?,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown method `{other}` (tda|rpa)"
            )))
        }
    };
    let out_states = PyList::empty(py);
    for st in &states {
        let d = PyDict::new(py);
        d.set_item("energy", st.energy)?;
        d.set_item("energy_ev", st.energy * crate::constants::HARTREE_TO_EV)?;
        d.set_item("oscillator_strength", st.oscillator_strength)?;
        d.set_item("transition_dipole", st.transition_dipole.to_vec())?;
        d.set_item("rotatory_strength_length", st.rotatory_strength)?;
        d.set_item("rotatory_strength_velocity", st.rotatory_strength_velocity)?;
        d.set_item(
            "transition_angular_momentum",
            st.transition_angular_momentum.to_vec(),
        )?;
        d.set_item("transition_momentum", st.transition_momentum.to_vec())?;
        d.set_item("tamm_dancoff", st.tamm_dancoff)?;
        out_states.append(d)?;
    }
    let out = PyDict::new(py);
    out.set_item("ground_state_energy", scf.energy)?;
    out.set_item("states", out_states)?;
    out.set_item("method", method)?;
    out.set_item("spin", format!("{mode:?}"))?;
    Ok(out)
}

/// SAPT0 (HF monomers) / DFT-SAPT (KS monomers) interaction-energy
/// decomposition of two separately specified monomers, every term reported
/// individually.
///
/// Second order in the intermolecular operator, zeroth order in intramonomer
/// correlation; exchange terms are Heitler-London (not S^2-truncated), and
/// induction is always the coupled/response value. Monomer-centred basis,
/// counterpoise-consistent throughout (`src/sapt.rs` has the full formulation).
///
/// Raises `ValueError` (the Rust side's `DftError::InvalidInput`) rather
/// than returning an incomplete decomposition when: either monomer uses an
/// ECP basis (consistent effective nuclear charges and semi-local ECP
/// intermolecular operators are not implemented), or a monomer's
/// first-order orbital rotation leaves the small-rotation regime the
/// second-order expansion assumes (a nearly singular orbital Hessian,
/// typically a partially-filled degenerate shell whose symmetry-broken SCF
/// solution is a saddle point).
#[pyfunction]
#[pyo3(signature = (numbers_a, positions_a, numbers_b, positions_b, basis="sto-3g", ecp=None,
    xc="hf", charge_a=0, multiplicity_a=1, charge_b=0, multiplicity_b=1, grid_level=3,
    max_iter=150, density_fit=None, conv_tol=1e-9, conv_tol_grad=None, conv_tol_dp=1e-8,
    incore_mb=2048))]
#[allow(clippy::too_many_arguments)]
fn sapt<'py>(
    py: Python<'py>,
    numbers_a: Vec<u8>,
    positions_a: Vec<[f64; 3]>,
    numbers_b: Vec<u8>,
    positions_b: Vec<[f64; 3]>,
    basis: &str,
    ecp: Option<&str>,
    xc: &str,
    charge_a: i32,
    multiplicity_a: usize,
    charge_b: i32,
    multiplicity_b: usize,
    grid_level: usize,
    max_iter: usize,
    density_fit: Option<&str>,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    conv_tol_dp: f64,
    incore_mb: usize,
) -> PyResult<Bound<'py, PyDict>> {
    validate_geometry(&numbers_a, &positions_a, multiplicity_a).map_err(err)?;
    validate_geometry(&numbers_b, &positions_b, multiplicity_b).map_err(err)?;
    if grid_level > 9 {
        return Err(PyValueError::new_err("grid_level must be between 0 and 9"));
    }
    // Mirrors `scf_options`'s validation, minus dispersion/VV10: SAPT takes
    // neither (`src/bin/dft_rs.rs`'s `"sapt"` arm builds `ScfOptions`
    // directly rather than through the shared dispersion/VV10 resolver, so
    // this binding follows the same, narrower path rather than accepting
    // controls the CLI itself refuses for SAPT).
    if max_iter == 0 {
        return Err(PyValueError::new_err("max_iter must be at least 1"));
    }
    for (name, value) in [("conv_tol", conv_tol), ("conv_tol_dp", conv_tol_dp)] {
        if !(value.is_finite() && value > 0.0) {
            return Err(PyValueError::new_err(format!(
                "{name} must be finite and positive"
            )));
        }
    }
    if let Some(value) = conv_tol_grad {
        if !(value.is_finite() && value > 0.0) {
            return Err(PyValueError::new_err(
                "conv_tol_grad must be finite and positive",
            ));
        }
    }
    if incore_mb == 0 {
        return Err(PyValueError::new_err("incore_mb must be at least 1"));
    }
    let mol_a = Molecule::new(
        numbers_a
            .iter()
            .zip(&positions_a)
            .map(|(&z, p)| Atom {
                z,
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    )
    .with_charge(charge_a)
    .with_multiplicity(multiplicity_a);
    let mol_b = Molecule::new(
        numbers_b
            .iter()
            .zip(&positions_b)
            .map(|(&z, p)| Atom {
                z,
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    )
    .with_charge(charge_b)
    .with_multiplicity(multiplicity_b);
    let set = build_basis(basis, ecp).map_err(err)?;
    let functional = resolve(&FunctionalSpec::Combined(xc.to_string())).map_err(err)?;
    let opts = ScfOptions {
        max_iter,
        e_tol: conv_tol,
        conv_tol_grad: conv_tol_grad.unwrap_or_else(|| conv_tol.sqrt()),
        p_tol: conv_tol_dp,
        incore_mb,
        density_fit: density_fit.map(str::to_owned),
        ..ScfOptions::default()
    };
    let gopts = GridOptions {
        level: grid_level,
        ..GridOptions::default()
    };
    let res = crate::sapt::sapt_interaction(&mol_a, &mol_b, &set, &functional, &opts, Some(&gopts))
        .map_err(err)?;
    let out = PyDict::new(py);
    out.set_item("e_monomer_a", res.e_monomer_a)?;
    out.set_item("e_monomer_b", res.e_monomer_b)?;
    out.set_item("e_elst", res.e_elst)?;
    out.set_item("e_exch", res.e_exch)?;
    out.set_item("e_ind_a", res.e_ind_a)?;
    out.set_item("e_ind_b", res.e_ind_b)?;
    out.set_item("e_exch_ind", res.e_exch_ind)?;
    out.set_item("e_disp_unc", res.e_disp_unc)?;
    out.set_item("e_disp_sos_uncoupled", res.e_disp_sos_uncoupled)?;
    out.set_item("e_disp_coupled", res.e_disp_coupled)?;
    out.set_item("e_exch_disp", res.e_exch_disp)?;
    out.set_item("e_int_supermolecular", res.e_int_supermolecular)?;
    out.set_item("delta", res.delta)?;
    out.set_item("interaction_total", res.interaction_total())?;
    out.set_item(
        "interaction_total_uncoupled_calibration",
        res.interaction_total_uncoupled_calibration(),
    )?;
    Ok(out)
}

/// List the bundled basis-set names.
#[pyfunction]
fn basis_sets() -> Vec<String> {
    data::bundled_names()
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// The functional registry: one dict per canonical name with its table
/// (`exchange`, `correlation`, `combined`, `double-hybrid`), family,
/// exact-exchange fractions, range separation (1/Bohr), required nonlocal
/// correlation and, for double hybrids, the PT2 spin coefficients.
#[pyfunction]
fn functionals(py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
    let out = PyList::empty(py);
    for r in crate::xc::catalogue::catalogue().map_err(err)? {
        let entry = PyDict::new(py);
        entry.set_item("name", r.name)?;
        entry.set_item("class", r.class.as_str())?;
        entry.set_item("family", crate::xc::catalogue::family_name(r.family))?;
        entry.set_item("description", r.description)?;
        entry.set_item("exact_exchange", r.exact_exchange)?;
        entry.set_item("long_range_exchange", r.long_range_exchange)?;
        entry.set_item("omega", r.omega)?;
        entry.set_item("nonlocal", r.nonlocal)?;
        entry.set_item("pt2_same_spin", r.pt2.map(|p| p.0))?;
        entry.set_item("pt2_opposite_spin", r.pt2.map(|p| p.1))?;
        entry.set_item("distinct_target", r.distinct_target)?;
        entry.set_item("dispersion", r.dispersion)?;
        out.append(entry)?;
    }
    Ok(out)
}

/// Periodic one-electron matrices, nuclear energy and stress for a cell.
///
/// `cell` is the 3x3 lattice in BOHR, rows being the lattice vectors.
/// `positions` are Cartesian, in Bohr, and need not be inside the cell.
/// `k_points` are Cartesian reciprocal vectors (1/Bohr); omit for Gamma.
/// `derivatives=True` adds the nuclear derivatives of every one-electron
/// matrix; they cost considerably more than the matrices themselves, which
/// is why they are opt-in.
///
/// The nuclear attraction carries the `G = 0` background shift that pairs
/// with the one in `nuclear_repulsion`; `background_shift` reports it, since
/// neither term is meaningful without the other.
///
/// The charged-cell (Makov-Payne) correction is applied automatically
/// whenever `charge` is non-zero — its interaction with its own periodic
/// images falls off only as 1/L, so on an ion it is routinely the largest
/// error in the calculation. It is also returned separately so it is
/// visible rather than folded silently into a total.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, basis="sto-3g", ecp=None, charge=0,
    multiplicity=1, k_points=None, overlap_cutoff=20.0, derivatives=false, density=None))]
#[allow(clippy::too_many_arguments)]
fn periodic<'py>(
    py: Python<'py>,
    numbers: Vec<u8>,
    positions: Vec<[f64; 3]>,
    cell: [[f64; 3]; 3],
    basis: &str,
    ecp: Option<&str>,
    charge: i32,
    multiplicity: usize,
    k_points: Option<Vec<[f64; 3]>>,
    overlap_cutoff: f64,
    derivatives: bool,
    density: Option<Vec<Vec<f64>>>,
) -> PyResult<Bound<'py, PyDict>> {
    let want_derivatives = derivatives;
    use crate::periodic::{Cell, Lattice};
    validate_geometry(&numbers, &positions, multiplicity).map_err(err)?;
    if cell.iter().flatten().any(|x| !x.is_finite()) {
        return Err(PyValueError::new_err("cell vectors must be finite"));
    }
    if !(overlap_cutoff.is_finite() && overlap_cutoff > 0.0) {
        return Err(PyValueError::new_err(
            "overlap_cutoff must be finite and positive",
        ));
    }
    let mol = Molecule::new(
        numbers
            .iter()
            .zip(&positions)
            .map(|(&z, p)| Atom {
                z,
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    )
    .with_charge(charge)
    .with_multiplicity(multiplicity);
    let lat = Lattice::new(
        Vec3::new(cell[0][0], cell[0][1], cell[0][2]),
        Vec3::new(cell[1][0], cell[1][1], cell[1][2]),
        Vec3::new(cell[2][0], cell[2][1], cell[2][2]),
    );
    if !(lat.volume().is_finite() && lat.volume().abs() > 1e-12) {
        return Err(PyValueError::new_err(
            "cell vectors must define a finite non-zero volume",
        ));
    }
    let set = build_basis(basis, ecp).map_err(err)?;
    let c = Cell::new(mol, lat, &set, overlap_cutoff).map_err(err)?;
    let n_electrons = c.n_electrons().map_err(err)?;
    c.molecule.n_alpha_beta_for(n_electrons).map_err(err)?;

    // Python exposes the common Gamma-point density matrix. It must be
    // inverse-Bloch-transformed over every retained lattice translation;
    // a lone D(0) block would instead describe a localized density. The
    // Rust API also accepts an explicitly supplied general D(T) list.
    let density_blocks = if let Some(rows) = density {
        let n = c.nao();
        if rows.len() != n || rows.iter().any(|row| row.len() != n) {
            return Err(PyValueError::new_err(format!(
                "density must be a finite {n}x{n} home-cell matrix"
            )));
        }
        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        if flat.iter().any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err(
                "density must contain only finite values",
            ));
        }
        let gamma = Matrix::from_row_major(n, n, flat);
        Some(c.gamma_density_images(&gamma).map_err(err)?)
    } else {
        None
    };
    let density_electron_count = density_blocks
        .as_deref()
        .map(|d| c.electron_count(d).map_err(err))
        .transpose()?;
    let hartree_blocks = density_blocks
        .as_deref()
        .map(|d| c.hartree_image_blocks(d).map_err(err))
        .transpose()?;

    let ks: Vec<Vec3> = match k_points {
        Some(list) => {
            if list.is_empty() || list.iter().flatten().any(|x| !x.is_finite()) {
                return Err(PyValueError::new_err(
                    "k_points must be a non-empty list of finite vectors",
                ));
            }
            list.iter().map(|k| Vec3::new(k[0], k[1], k[2])).collect()
        }
        None => vec![Vec3::zero()],
    };
    let overlaps = PyList::empty(py);
    let kinetics = PyList::empty(py);
    let nuclears = PyList::empty(py);
    let cores = PyList::empty(py);
    let hartrees = PyList::empty(py);
    for k in &ks {
        let (sre, sim) = c.overlap(*k);
        let (tre, tim) = c.kinetic(*k);
        let (vre, vim) = c.nuclear(*k);
        let (hre, him) = c.core_hamiltonian(*k);
        for (list, (re, im)) in [
            (&overlaps, (sre, sim)),
            (&kinetics, (tre, tim)),
            (&nuclears, (vre, vim)),
            (&cores, (hre, him)),
        ] {
            let d = PyDict::new(py);
            d.set_item("real", mat_rows(&re))?;
            d.set_item("imag", mat_rows(&im))?;
            list.append(d)?;
        }
        if let Some(blocks) = hartree_blocks.as_deref() {
            let (re, im) = crate::periodic::integrals::bloch(blocks, *k);
            let d = PyDict::new(py);
            d.set_item("real", mat_rows(&re))?;
            d.set_item("imag", mat_rows(&im))?;
            hartrees.append(d)?;
        }
    }

    // Nuclear derivatives are the expensive part, so they are opt-in.
    let derivatives = if want_derivatives {
        let images = c.derivative_images();
        let per_k = PyList::empty(py);
        for k in &ks {
            let entry = PyDict::new(py);
            for (key, blocks) in [
                ("overlap", images.overlap(*k)),
                ("kinetic", images.kinetic(*k)),
                ("nuclear", images.nuclear(*k)),
                ("ecp", images.ecp(*k)),
                (
                    "core_hamiltonian",
                    images.core_hamiltonian(*k).map_err(err)?,
                ),
            ] {
                let atoms = PyList::empty(py);
                for per_atom in &blocks {
                    let axes = PyList::empty(py);
                    for (re, im) in per_atom.iter() {
                        let d = PyDict::new(py);
                        d.set_item("real", mat_rows(re))?;
                        d.set_item("imag", mat_rows(im))?;
                        axes.append(d)?;
                    }
                    atoms.append(axes)?;
                }
                entry.set_item(key, atoms)?;
            }
            per_k.append(entry)?;
        }
        Some(per_k)
    } else {
        None
    };

    let out = PyDict::new(py);
    out.set_item("nao", c.nao())?;
    out.set_item("n_electrons", n_electrons)?;
    out.set_item("volume", c.lattice.volume().abs())?;
    out.set_item("net_charge", c.net_charge())?;
    out.set_item("nuclear_repulsion", c.nuclear_repulsion())?;
    out.set_item("charged_cell_correction", c.charged_cell_correction())?;
    out.set_item("nuclear_energy", c.nuclear_energy())?;
    let stress = c.nuclear_stress();
    out.set_item(
        "nuclear_stress",
        stress.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
    )?;
    let (n_s, n_t) = c.image_counts();
    out.set_item("image_counts", (n_s, n_t))?;
    out.set_item(
        "k_points",
        ks.iter().map(|k| vec![k.x, k.y, k.z]).collect::<Vec<_>>(),
    )?;
    out.set_item("overlap", overlaps)?;
    out.set_item("kinetic", kinetics)?;
    out.set_item("nuclear", nuclears)?;
    out.set_item("core_hamiltonian", cores)?;
    out.set_item("background_shift", c.background_shift())?;
    if let Some(count) = density_electron_count {
        out.set_item("density_electron_count", count)?;
        out.set_item("hartree", hartrees)?;
    }
    if let Some(d) = derivatives {
        out.set_item("derivatives", d)?;
    }
    Ok(out)
}

/// Periodic HF/KS-DFT total energy and optional analytic forces.
///
/// All matrices are solved independently on the requested uniform k mesh;
/// the real-space density is only an intermediate for J and K. Positions,
/// lattice vectors and cutoffs are in Bohr.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, basis="sto-3g", ecp=None, xc="pbe",
    charge=0, multiplicity=1, pbc=[true, true, true], k_mesh=[1, 1, 1], overlap_cutoff=20.0,
    exchange_cutoff=None, grid_level=3, grid_shell=2, max_iter=100,
    conv_tol=1e-9, conv_tol_grad=None, conv_tol_dp=1e-8, forces=false,
    wavefunction=false, electronic_temperature=0.0, level_shift=0.0,
    damping=0.25, diis_size=8, use_adiis=true, adiis_until_error=0.1,
    guess="core", relax=false,
    opt_max_steps=200, opt_gtol=3e-4, opt_step_max=0.3, opt_history=8,
    trajectory=None, response_q=None, response_omega=0.0, response_damping=0.0,
    response_imaginary=false, response_fixed_number=true, response_reciprocal_cutoff=6.0,
    response_modes=None, response_density_mode=0, response_tol=1e-10,
    response_max_iter=500, response_restart=40, response_thermal=false,
    response_delta_temperature=1.0, response_delta_numbers=[0.0,0.0],
    response_delta_mu=[0.0,0.0], response_scalar_potential=None,
    response_vector_potential=None, grid_preset=None, grid_angular_scheme=None,
    grid_weight_cutoff=None))]
#[allow(clippy::too_many_arguments)]
fn periodic_scf<'py>(
    py: Python<'py>,
    numbers: Vec<u8>,
    positions: Vec<[f64; 3]>,
    cell: [[f64; 3]; 3],
    basis: &str,
    ecp: Option<&str>,
    xc: &str,
    charge: i32,
    multiplicity: usize,
    pbc: [bool; 3],
    k_mesh: [usize; 3],
    overlap_cutoff: f64,
    exchange_cutoff: Option<f64>,
    grid_level: usize,
    grid_shell: i64,
    max_iter: usize,
    conv_tol: f64,
    conv_tol_grad: Option<f64>,
    conv_tol_dp: f64,
    forces: bool,
    wavefunction: bool,
    electronic_temperature: f64,
    level_shift: f64,
    damping: f64,
    diis_size: usize,
    use_adiis: bool,
    adiis_until_error: f64,
    guess: &str,
    relax: bool,
    opt_max_steps: usize,
    opt_gtol: f64,
    opt_step_max: f64,
    opt_history: usize,
    trajectory: Option<&str>,
    response_q: Option<[f64; 3]>,
    response_omega: f64,
    response_damping: f64,
    response_imaginary: bool,
    response_fixed_number: bool,
    response_reciprocal_cutoff: f64,
    response_modes: Option<Vec<usize>>,
    response_density_mode: usize,
    response_tol: f64,
    response_max_iter: usize,
    response_restart: usize,
    response_thermal: bool,
    response_delta_temperature: f64,
    response_delta_numbers: [f64; 2],
    response_delta_mu: [f64; 2],
    response_scalar_potential: Option<[f64; 2]>,
    response_vector_potential: Option<[[f64; 2]; 3]>,
    grid_preset: Option<&str>,
    grid_angular_scheme: Option<&str>,
    grid_weight_cutoff: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    use crate::periodic::features::{PeriodicBoundary, PeriodicCapabilities, PeriodicEcpPlan};
    use crate::periodic::forces::analytic_forces;
    use crate::periodic::occupations::OccupationScheme;
    use crate::periodic::scf::{run_periodic_scf, PeriodicScfOptions};
    use crate::periodic::{Cell, Lattice};

    let field_request = response_scalar_potential.is_some() || response_vector_potential.is_some();
    if field_request
        && (response_q.is_none()
            || response_thermal
            || response_modes.is_some()
            || response_density_mode != 0)
    {
        return Err(PyValueError::new_err("field response needs response_q and cannot be combined with dielectric, thermal or source-mode controls"));
    }
    let field_scalar = response_scalar_potential.unwrap_or([0.0; 2]);
    let field_vector = response_vector_potential.unwrap_or([[0.0; 2]; 3]);
    if field_scalar
        .iter()
        .chain(field_vector.iter().flatten())
        .any(|v| !v.is_finite())
    {
        return Err(PyValueError::new_err(
            "field potential amplitudes must be finite",
        ));
    }
    if response_thermal {
        if response_q != Some([0.0; 3])
            || response_omega != 0.0
            || response_damping != 0.0
            || response_modes.is_some()
            || response_density_mode != 0
            || electronic_temperature <= 0.0
            || !response_delta_temperature.is_finite()
            || response_delta_numbers
                .iter()
                .chain(&response_delta_mu)
                .any(|x| !x.is_finite())
            || (response_fixed_number && response_delta_mu != [0.0; 2])
            || (!response_fixed_number && response_delta_numbers != [0.0; 2])
        {
            return Err(PyValueError::new_err(
                "thermal response requires q=0, zero frequency, finite positive \
                reference temperature, and sources compatible with its ensemble",
            ));
        }
    } else if response_delta_temperature != 1.0
        || response_delta_numbers != [0.0; 2]
        || response_delta_mu != [0.0; 2]
    {
        return Err(PyValueError::new_err(
            "thermal source controls require response_thermal",
        ));
    }

    let response_request = if let Some(q) = response_q {
        if q.iter().any(|v| !v.is_finite())
            || !response_reciprocal_cutoff.is_finite()
            || response_reciprocal_cutoff <= 0.0
            || (response_imaginary && response_damping != 0.0)
        {
            return Err(PyValueError::new_err(
                "invalid response q, reciprocal cutoff, or imaginary-frequency damping",
            ));
        }
        if relax || forces || trajectory.is_some() || wavefunction {
            return Err(PyValueError::new_err("periodic response cannot be combined with relax, forces, trajectory, or wavefunction export"));
        }
        if response_modes
            .as_ref()
            .is_some_and(|m| m.is_empty() || m.iter().enumerate().any(|(i, x)| m[..i].contains(x)))
        {
            return Err(PyValueError::new_err(
                "dielectric modes must be nonempty and unique",
            ));
        }
        let frequency = if response_imaginary {
            crate::response::Frequency::Imaginary {
                frequency: response_omega,
            }
        } else {
            crate::response::Frequency::Retarded {
                omega: response_omega,
                damping: response_damping,
            }
        };
        frequency.validate().map_err(err)?;
        let solver = crate::response::ResponseSolverConfig {
            tolerance: response_tol,
            max_iterations: response_max_iter,
            gmres_restart: response_restart,
            ..Default::default()
        };
        solver.validate().map_err(err)?;
        Some(crate::dfpt::density::PeriodicResponseOptions {
            q: Vec3::new(q[0], q[1], q[2]),
            frequency,
            fixed_number: response_fixed_number,
            reciprocal_cutoff: response_reciprocal_cutoff,
            solver,
        })
    } else {
        if response_omega != 0.0
            || response_damping != 0.0
            || response_imaginary
            || !response_fixed_number
            || response_reciprocal_cutoff != 6.0
            || response_modes.is_some()
            || response_density_mode != 0
            || response_tol != 1e-10
            || response_max_iter != 500
            || response_restart != 40
            || response_thermal
        {
            return Err(PyValueError::new_err(
                "response controls require response_q",
            ));
        }
        None
    };

    validate_geometry(&numbers, &positions, multiplicity).map_err(err)?;
    if cell.iter().flatten().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("cell vectors must be finite"));
    }
    if grid_level > 9 {
        return Err(PyValueError::new_err("grid_level must be between 0 and 9"));
    }
    if grid_shell < 1 {
        return Err(PyValueError::new_err("grid_shell must be at least 1"));
    }
    if k_mesh.contains(&0) {
        return Err(PyValueError::new_err(
            "every k_mesh dimension must be at least 1",
        ));
    }
    if !(electronic_temperature.is_finite() && electronic_temperature >= 0.0) {
        return Err(PyValueError::new_err(
            "electronic_temperature must be finite and non-negative",
        ));
    }
    if !(adiis_until_error.is_finite() && adiis_until_error >= 0.0) {
        return Err(PyValueError::new_err(
            "adiis_until_error must be finite and non-negative",
        ));
    }
    if opt_max_steps == 0
        || opt_history == 0
        || !(opt_gtol.is_finite() && opt_gtol > 0.0)
        || !(opt_step_max.is_finite() && opt_step_max > 0.0)
    {
        return Err(PyValueError::new_err(
            "periodic optimization needs opt_max_steps/opt_history > 0 and finite positive opt_gtol/opt_step_max",
        ));
    }
    if !relax
        && (opt_max_steps != 200
            || opt_gtol != 3e-4
            || opt_step_max != 0.3
            || opt_history != 8
            || trajectory.is_some())
    {
        return Err(PyValueError::new_err(
            "opt_* and trajectory require relax=True",
        ));
    }
    let boundary = PeriodicBoundary::new(pbc);
    boundary.validate_k_mesh(k_mesh).map_err(err)?;
    PeriodicCapabilities::current()
        .validate_boundary(boundary)
        .map_err(err)?;
    let molecule = Molecule::new(
        numbers
            .iter()
            .zip(&positions)
            .map(|(&z, p)| Atom {
                z,
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    )
    .with_charge(charge)
    .with_multiplicity(multiplicity);
    let lattice = Lattice::new(
        Vec3::new(cell[0][0], cell[0][1], cell[0][2]),
        Vec3::new(cell[1][0], cell[1][1], cell[1][2]),
        Vec3::new(cell[2][0], cell[2][1], cell[2][2]),
    );
    if !(lattice.volume().is_finite() && lattice.volume().abs() > 1e-12) {
        return Err(PyValueError::new_err(
            "cell vectors must define a finite non-zero volume",
        ));
    }
    let set = build_basis(basis, ecp).map_err(err)?;
    let functional = resolve(&FunctionalSpec::Combined(xc.to_string())).map_err(err)?;
    if let Some(term) = functional.required_nonlocal_term() {
        return Err(PyNotImplementedError::new_err(format!(
            "periodic SCF for `{xc}` is unavailable because its defining {term} \
                nonlocal term is not yet periodic; use a semilocal/hybrid functional or \
                the molecular API"
        )));
    }
    let guess = match guess.to_ascii_lowercase().as_str() {
        "sad" => crate::scf::Guess::Sad,
        "gwh" => crate::scf::Guess::Gwh,
        "core" => crate::scf::Guess::Core,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown periodic SCF guess `{other}`; expected sad, gwh, or core"
            )))
        }
    };
    let options = PeriodicScfOptions {
        k_mesh,
        boundary,
        occupation: if electronic_temperature == 0.0 {
            OccupationScheme::Insulator
        } else {
            OccupationScheme::FermiDirac {
                temperature_kelvin: electronic_temperature,
            }
        },
        guess,
        use_adiis,
        adiis_until_error,
        max_iter,
        e_tol: conv_tol,
        p_tol: conv_tol_dp,
        conv_tol_grad: conv_tol_grad.unwrap_or_else(|| conv_tol.sqrt()),
        level_shift,
        damping,
        diis_size,
        exchange_cutoff,
        grid: GridOptions::from_controls(
            grid_level,
            grid_preset,
            grid_angular_scheme,
            grid_weight_cutoff,
        )
        .map_err(err)?,
        grid_shell,
        ..PeriodicScfOptions::default()
    };
    let (periodic_cell, result, optimization) = if relax {
        let optimized = crate::periodic::optimizer::optimize_fixed_cell(
            &molecule,
            &lattice,
            &set,
            overlap_cutoff,
            &functional,
            &options,
            &crate::periodic::optimizer::PeriodicOptOptions {
                max_steps: opt_max_steps,
                gtol: opt_gtol,
                step_max: opt_step_max,
                history: opt_history,
                trajectory: trajectory.map(std::path::PathBuf::from),
                dispersion: None,
            },
        )
        .map_err(err)?;
        let positions = optimized
            .molecule
            .atoms
            .iter()
            .map(|atom| vec![atom.position.x, atom.position.y, atom.position.z])
            .collect::<Vec<_>>();
        let summary = (
            optimized.converged,
            optimized.iterations,
            optimized.objective,
            optimized.trajectory.len(),
            positions,
        );
        (optimized.cell, optimized.scf, Some(summary))
    } else {
        let periodic_cell =
            Cell::new(molecule.clone(), lattice.clone(), &set, overlap_cutoff).map_err(err)?;
        let result = run_periodic_scf(&periodic_cell, &functional, &options).map_err(err)?;
        (periodic_cell, result, None)
    };
    let out = PyDict::new(py);
    out.set_item("energy", result.energy)?;
    out.set_item("free_energy", result.free_energy)?;
    out.set_item("mermin_free_energy", result.free_energy)?;
    out.set_item("free_energy_kind", "mermin; electronic A = E - T_e S_e")?;
    out.set_item("entropy_term", result.entropy_term)?;
    out.set_item("electronic_temperature", electronic_temperature)?;
    out.set_item("e_elec", result.e_elec)?;
    out.set_item("e_nuc", result.e_nuc)?;
    out.set_item("e_xc", result.e_xc)?;
    out.set_item("converged", result.converged)?;
    out.set_item("iterations", result.iterations)?;
    out.set_item("n_alpha", result.n_alpha)?;
    out.set_item("n_beta", result.n_beta)?;
    out.set_item("unrestricted", result.unrestricted)?;
    out.set_item("occupations", result.occupations.clone())?;
    out.set_item("chemical_potential", result.chemical_potential)?;
    if let Some((converged, iterations, objective, frame_count, positions)) = optimization {
        out.set_item("optimization_converged", converged)?;
        out.set_item("optimization_iterations", iterations)?;
        out.set_item("optimization_objective", objective)?;
        out.set_item("optimization_frames", frame_count)?;
        out.set_item("optimized_positions", positions)?;
        out.set_item("optimization_cell", "fixed")?;
    }
    if let Some(plan) = PeriodicEcpPlan::from_cell(&periodic_cell) {
        let ecp_plan = PyDict::new(py);
        ecp_plan.set_item("centers_per_cell", plan.centers_per_cell)?;
        ecp_plan.set_item("core_electrons_per_cell", plan.core_electrons_per_cell)?;
        ecp_plan.set_item("max_projector_l", plan.max_projector_l)?;
        ecp_plan.set_item("image_cutoff", plan.image_cutoff)?;
        ecp_plan.set_item("analytic_forces", true)?;
        out.set_item("periodic_ecp", ecp_plan)?;
    }
    out.set_item(
        "k_points",
        result
            .k_points
            .iter()
            .map(|k| vec![k.x, k.y, k.z])
            .collect::<Vec<_>>(),
    )?;
    if let Some(count) = result.grid_electron_count {
        out.set_item("grid_electron_count", count)?;
    }
    if wavefunction {
        let wf = PyDict::new(py);
        let densities = PyList::empty(py);
        let focks = PyList::empty(py);
        for spin in 0..2 {
            densities.append(bloch_rows(py, &result.density[spin])?)?;
            focks.append(bloch_rows(py, &result.fock[spin])?)?;
        }
        wf.set_item("mo_energy", result.mo_energy.clone())?;
        wf.set_item("density", densities)?;
        wf.set_item("fock", focks)?;
        wf.set_item("overlap", bloch_rows(py, &result.overlap)?)?;
        wf.set_item("h_core", bloch_rows(py, &result.h_core)?)?;
        out.set_item("wavefunction", wf)?;
    }
    if forces {
        let values =
            analytic_forces(&periodic_cell, &functional, &result, &options).map_err(err)?;
        let rows = |vectors: &[Vec3]| {
            vectors
                .iter()
                .map(|v| vec![v.x, v.y, v.z])
                .collect::<Vec<_>>()
        };
        out.set_item("forces", rows(&values.total))?;
        out.set_item("electronic_forces", rows(&values.electronic))?;
        out.set_item("nuclear_forces", rows(&values.nuclear))?;
        out.set_item("force_method", "analytic")?;
        out.set_item(
            "force_energy",
            if electronic_temperature > 0.0 {
                "mermin_free_energy"
            } else {
                "energy"
            },
        )?;
    }
    if let Some(request) = response_request {
        let solver = if field_request {
            None
        } else {
            Some(
                crate::dfpt::density::PeriodicDensitySolver::new(
                    &periodic_cell,
                    &functional,
                    &result,
                    &options,
                    &request,
                )
                .map_err(err)?,
            )
        };
        let response = PyDict::new(py);
        response.set_item("q", [request.q.x, request.q.y, request.q.z])?;
        response.set_item("q_units", "bohr^-1")?;
        response.set_item(
            "frequency_real",
            request.frequency.energy().map_err(err)?.re,
        )?;
        response.set_item(
            "frequency_imag",
            request.frequency.energy().map_err(err)?.im,
        )?;
        response.set_item("frequency_units", "hartree")?;
        response.set_item("fixed_number", request.fixed_number)?;
        response.set_item(
            "ensemble",
            if request.fixed_number {
                "fixed spin populations"
            } else {
                "fixed spin chemical potentials"
            },
        )?;
        response.set_item("reciprocal_cutoff", request.reciprocal_cutoff)?;
        response.set_item(
            "temperature_hartree",
            electronic_temperature * crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K,
        )?;
        if field_request {
            use crate::dfpt::field::{periodic_field_response, PeriodicFieldResponseOptions};
            use crate::response::complex::Complex;
            let report = periodic_field_response(
                &periodic_cell,
                &functional,
                &result,
                &options,
                &PeriodicFieldResponseOptions {
                    response: request.clone(),
                    scalar_potential: Complex::new(field_scalar[0], field_scalar[1]),
                    vector_potential: field_vector.map(|a| Complex::new(a[0], a[1])),
                },
            )
            .map_err(err)?;
            response_metadata(py, &response, &report)?;
            for (key, values) in [
                ("vector_potential", report.value.vector_potential),
                ("electric_field", report.value.electric_field),
                ("magnetic_field", report.value.magnetic_field),
                (
                    "paramagnetic_number_current",
                    report.value.paramagnetic_number_current,
                ),
                (
                    "diamagnetic_number_current",
                    report.value.diamagnetic_number_current,
                ),
                ("number_current", report.value.number_current),
                ("charge_current", report.value.charge_current),
            ] {
                response.set_item(format!("{key}_real"), values.map(|v| v.re))?;
                response.set_item(format!("{key}_imag"), values.map(|v| v.im))?;
            }
            for (key, value) in [
                ("scalar_potential", report.value.scalar_potential),
                ("number_density", report.value.number_density),
                ("charge_density", report.value.charge_density),
            ] {
                response.set_item(format!("{key}_real"), value.re)?;
                response.set_item(format!("{key}_imag"), value.im)?;
            }
            response.set_item("continuity_defect", report.value.continuity_defect)?;
            response.set_item("reference_electrons", report.value.reference_electrons)?;
            response.set_item(
                "occupation_consistency_defect",
                report.value.density_response.occupation_consistency_defect,
            )?;
            for (key, points) in [
                ("k_points", &report.value.k_points),
                ("target_k_points", &report.value.target_k_points),
                (
                    "reciprocal_waves",
                    &report.value.density_response.reciprocal_waves,
                ),
            ] {
                response.set_item(
                    key,
                    points.iter().map(|p| [p.x, p.y, p.z]).collect::<Vec<_>>(),
                )?;
            }
            let densities = PyList::empty(py);
            for spin in 0..2 {
                densities.append(bloch_rows(
                    py,
                    &report.value.density_response.density[spin],
                )?)?;
            }
            response.set_item("density", densities)?;
            response.set_item(
                "fourier_density_real",
                report
                    .value
                    .density_response
                    .fourier_density
                    .iter()
                    .map(|v| v.re)
                    .collect::<Vec<_>>(),
            )?;
            response.set_item(
                "fourier_density_imag",
                report
                    .value
                    .density_response
                    .fourier_density
                    .iter()
                    .map(|v| v.im)
                    .collect::<Vec<_>>(),
            )?;
        } else if response_thermal {
            use crate::dfpt::density::ThermalSource;
            use crate::dfpt::thermodynamic::Ensemble;
            let source = ThermalSource {
                delta_temperature_hartree: response_delta_temperature
                    * crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K,
                ensembles: std::array::from_fn(|spin| {
                    if request.fixed_number {
                        Ensemble::FixedNumber {
                            delta_number: response_delta_numbers[spin],
                        }
                    } else {
                        Ensemble::FixedChemicalPotential {
                            delta_mu: response_delta_mu[spin],
                        }
                    }
                }),
            };
            let report = solver
                .as_ref()
                .expect("density solver")
                .solve_thermal(source)
                .map_err(err)?;
            response_metadata(py, &response, &report)?;
            let densities = PyList::empty(py);
            for spin in 0..2 {
                densities.append(bloch_rows(py, &report.value.density[spin])?)?;
            }
            response.set_item("density", densities)?;
            response.set_item(
                "fourier_density_real",
                report
                    .value
                    .fourier_density
                    .iter()
                    .map(|z| z.re)
                    .collect::<Vec<_>>(),
            )?;
            response.set_item(
                "fourier_density_imag",
                report
                    .value
                    .fourier_density
                    .iter()
                    .map(|z| z.im)
                    .collect::<Vec<_>>(),
            )?;
            response.set_item(
                "reciprocal_waves",
                report
                    .value
                    .reciprocal_waves
                    .iter()
                    .map(|g| [g.x, g.y, g.z])
                    .collect::<Vec<_>>(),
            )?;
            response.set_item("delta_temperature_kelvin", response_delta_temperature)?;
            response.set_item(
                "delta_temperature_hartree",
                source.delta_temperature_hartree,
            )?;
            response.set_item(
                "delta_chemical_potentials",
                report.value.delta_chemical_potentials,
            )?;
            response.set_item("delta_numbers", report.value.delta_numbers)?;
            response.set_item(
                "occupation_equation_defect",
                report.value.occupation_equation_defect,
            )?;
            response.set_item(
                "number_constraint_defects",
                report.value.number_constraint_defects,
            )?;
            response.set_item(
                "occupation_consistency_defect",
                report.value.occupation_consistency_defect,
            )?;
            response.set_item("delta_entropy", report.value.delta_entropy)?;
            response.set_item("delta_internal_energy", report.value.delta_internal_energy)?;
            response.set_item("delta_free_energy", report.value.delta_free_energy)?;
            response.set_item("delta_mermin_free_energy", report.value.delta_free_energy)?;
        } else if let Some(modes) = response_modes {
            let report = solver
                .as_ref()
                .expect("density solver")
                .dielectric(&modes)
                .map_err(err)?;
            response_metadata(py, &response, &report)?;
            response.set_item("units", report.units)?;
            response.set_item("model", report.model)?;
            response.set_item("axis_convention", report.axis_convention)?;
            response.set_item("converged", report.converged)?;
            response.set_item("residuals", report.residuals)?;
            response.set_item("iterations", report.iterations)?;
            response.set_item("warnings", report.warnings)?;
            response.set_item(
                "screened_chi_real",
                report
                    .value
                    .screened_chi
                    .iter()
                    .map(|row| row.iter().map(|z| z.re).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
            )?;
            response.set_item(
                "screened_chi_imag",
                report
                    .value
                    .screened_chi
                    .iter()
                    .map(|row| row.iter().map(|z| z.im).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
            )?;
            response.set_item(
                "inverse_epsilon_real",
                report
                    .value
                    .inverse_epsilon
                    .iter()
                    .map(|row| row.iter().map(|z| z.re).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
            )?;
            response.set_item(
                "inverse_epsilon_imag",
                report
                    .value
                    .inverse_epsilon
                    .iter()
                    .map(|row| row.iter().map(|z| z.im).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
            )?;
            response.set_item(
                "reciprocal_waves",
                report
                    .value
                    .reciprocal_waves
                    .iter()
                    .map(|g| vec![g.x, g.y, g.z])
                    .collect::<Vec<_>>(),
            )?;
        } else {
            let solver = solver.as_ref().expect("density solver");
            let report = solver
                .solve(
                    &solver
                        .plane_wave_source(response_density_mode)
                        .map_err(err)?,
                )
                .map_err(err)?;
            response_metadata(py, &response, &report)?;
            response.set_item("units", report.units)?;
            response.set_item("model", report.model)?;
            response.set_item("axis_convention", report.axis_convention)?;
            response.set_item("converged", report.converged)?;
            response.set_item("residuals", report.residuals)?;
            response.set_item("iterations", report.iterations)?;
            response.set_item("warnings", report.warnings)?;
            let densities = PyList::empty(py);
            for spin in 0..2 {
                densities.append(bloch_rows(py, &report.value.density[spin])?)?;
            }
            response.set_item("density", densities)?;
            response.set_item(
                "fourier_density_real",
                report
                    .value
                    .fourier_density
                    .iter()
                    .map(|z| z.re)
                    .collect::<Vec<_>>(),
            )?;
            response.set_item(
                "fourier_density_imag",
                report
                    .value
                    .fourier_density
                    .iter()
                    .map(|z| z.im)
                    .collect::<Vec<_>>(),
            )?;
            response.set_item(
                "reciprocal_waves",
                report
                    .value
                    .reciprocal_waves
                    .iter()
                    .map(|g| vec![g.x, g.y, g.z])
                    .collect::<Vec<_>>(),
            )?;
            response.set_item(
                "occupation_consistency_defect",
                report.value.occupation_consistency_defect,
            )?;
            response.set_item("temperature_hartree", report.value.temperature_hartree)?;
        }
        response.set_item("energy", result.energy)?;
        response.set_item("free_energy", result.free_energy)?;
        response.set_item("mermin_free_energy", result.free_energy)?;
        response.set_item("free_energy_kind", "mermin; electronic A = E - T_e S_e")?;
        out.set_item("response", response)?;
    }
    Ok(out)
}

/// Electronic Gaussian basis and quadrature in a periodic subspace.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, basis="sto-3g", pbc=[true,true,true],
    k_mesh=[1,1,1], basis_cutoff=12.0, grid_level=None, grid_cutoff=12.0, grid_derivatives=false))]
#[allow(clippy::too_many_arguments)]
fn periodic_basis<'py>(
    py: Python<'py>,
    numbers: Vec<u8>,
    positions: Vec<[f64; 3]>,
    cell: [[f64; 3]; 3],
    basis: &str,
    pbc: [bool; 3],
    k_mesh: [usize; 3],
    basis_cutoff: f64,
    grid_level: Option<usize>,
    grid_cutoff: f64,
    grid_derivatives: bool,
) -> PyResult<Bound<'py, PyDict>> {
    validate_geometry(&numbers, &positions, 1).map_err(err)?;
    if grid_level.is_some_and(|level| level > 9) {
        return Err(PyValueError::new_err("grid_level must be 0..=9"));
    }
    let vector = |v: [f64; 3]| Vec3::new(v[0], v[1], v[2]);
    let mol = Molecule::new(
        numbers
            .into_iter()
            .zip(positions)
            .map(|(z, r)| Atom {
                z,
                position: vector(r),
            })
            .collect(),
    );
    let mb =
        MolBasis::build(&mol, &crate::basis::mixed::load_spec(basis).map_err(err)?).map_err(err)?;
    let geometry = crate::periodic::PeriodicGeometry::new(
        crate::periodic::Lattice::new(vector(cell[0]), vector(cell[1]), vector(cell[2])),
        crate::periodic::features::PeriodicBoundary::new(pbc),
    )
    .map_err(err)?;
    let report = crate::periodic::periodic_basis(
        &mol,
        &mb,
        &geometry,
        &crate::periodic::PeriodicBasisOptions {
            k_mesh,
            basis_cutoff,
            grid_cutoff,
            grid_derivatives,
            grid: grid_level.map(|level| GridOptions {
                level,
                prune: false,
                weight_cutoff: 0.0,
                ..Default::default()
            }),
        },
    )
    .map_err(err)?;
    let out = PyDict::new(py);
    let matrix = |m: &Matrix| {
        m.as_slice()
            .chunks(m.cols)
            .map(|row| row.to_vec())
            .collect::<Vec<_>>()
    };
    let array = |v: Vec3| [v.x, v.y, v.z];
    out.set_item("dimension", geometry.dimension())?;
    out.set_item("pbc", pbc)?;
    out.set_item("measure", geometry.measure())?;
    out.set_item("nao", mb.nao_sph)?;
    out.set_item("shape", [report.k_points.len(), mb.nao_sph, mb.nao_sph])?;
    out.set_item(
        "k_points",
        report
            .k_points
            .iter()
            .copied()
            .map(array)
            .collect::<Vec<_>>(),
    )?;
    out.set_item("reciprocal_vectors", geometry.reciprocal().map(array))?;
    for (label, operators, derivatives) in [
        ("overlap", &report.overlap, &report.overlap_derivatives),
        ("kinetic", &report.kinetic, &report.kinetic_derivatives),
    ] {
        out.set_item(
            format!("{label}_real"),
            operators.iter().map(|m| matrix(&m.re)).collect::<Vec<_>>(),
        )?;
        out.set_item(
            format!("{label}_imag"),
            operators.iter().map(|m| matrix(&m.im)).collect::<Vec<_>>(),
        )?;
        out.set_item(
            format!("{label}_derivatives_real"),
            derivatives
                .iter()
                .map(|atoms| {
                    atoms
                        .iter()
                        .map(|axes| axes.iter().map(|m| matrix(&m.re)).collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            format!("{label}_derivatives_imag"),
            derivatives
                .iter()
                .map(|atoms| {
                    atoms
                        .iter()
                        .map(|axes| axes.iter().map(|m| matrix(&m.im)).collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        )?;
    }
    out.set_item(
        "minimum_overlap_eigenvalue",
        report.minimum_overlap_eigenvalue,
    )?;
    out.set_item("hermiticity_defect", report.hermiticity_defect)?;
    out.set_item("translation_defect", report.translation_defect)?;
    out.set_item("basis_cutoff", basis_cutoff)?;
    out.set_item("grid_cutoff", grid_cutoff)?;
    out.set_item("grid_overlap_defect", report.grid_overlap_defect)?;
    if let Some(grid) = report.grid {
        let item = PyDict::new(py);
        item.set_item(
            "points",
            grid.points.into_iter().map(array).collect::<Vec<_>>(),
        )?;
        item.set_item("weights", grid.weights)?;
        item.set_item("raw_weights", grid.raw_weights)?;
        item.set_item("owners", grid.atom_of)?;
        item.set_item(
            "moving_weight_derivatives",
            report.moving_weight_derivatives.map(|points| {
                points
                    .into_iter()
                    .map(|atoms| atoms.into_iter().map(array).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            }),
        )?;
        out.set_item("grid", item)?;
    } else {
        out.set_item("grid", py.None())?;
    }
    out.set_item("model", report.model)?;
    out.set_item("units", report.units)?;
    out.set_item("warnings", report.warnings)?;
    Ok(out)
}

/// Self-consistent dimensional collinear DFT with explicit charge compensation.
#[pyfunction]
#[pyo3(signature=(numbers,positions,cell,basis="sto-3g",xc="pbe",charge=0,multiplicity=1,
    pbc=[true,true,true],k_mesh=[1,1,1],basis_cutoff=12.0,grid_cutoff=12.0,grid_level=3,
    compensation="neutral",countercharge_center=None,countercharge_width=None,alpha=None,
    real_cutoff=None,reciprocal_cutoff=None,quadrature_order=96,tolerance=1e-11,
    operator_tolerance=1e-11,grid_tolerance=1e-5,nuclear_widths=None,temperature=0.0,
    unrestricted=false,max_iter=100,conv_tol=1e-9,conv_tol_dp=1e-8,conv_tol_grad=None,
    guess="core",exchange_cutoff=None,use_adiis=true,adiis_until_error=0.1,
    damping=0.25,level_shift=0.0,diis_size=8,field_transfer=None,field_scalar=None,field_vector=None,
    response_omega=0.0,response_damping=0.0,response_imaginary=false,response_fixed_number=true,
    response_tol=1e-10,response_max_iter=500,response_restart=40,
    grid_preset=None,grid_angular_scheme=None,grid_weight_cutoff=None))]
#[allow(clippy::too_many_arguments)]
fn dimensional_scf<'py>(
    py: Python<'py>,
    numbers: Vec<u8>,
    positions: Vec<[f64; 3]>,
    cell: [[f64; 3]; 3],
    basis: &str,
    xc: &str,
    charge: i32,
    multiplicity: usize,
    pbc: [bool; 3],
    k_mesh: [usize; 3],
    basis_cutoff: f64,
    grid_cutoff: f64,
    grid_level: usize,
    compensation: &str,
    countercharge_center: Option<[f64; 3]>,
    countercharge_width: Option<f64>,
    alpha: Option<f64>,
    real_cutoff: Option<f64>,
    reciprocal_cutoff: Option<f64>,
    quadrature_order: usize,
    tolerance: f64,
    operator_tolerance: f64,
    grid_tolerance: f64,
    nuclear_widths: Option<Vec<f64>>,
    temperature: f64,
    unrestricted: bool,
    max_iter: usize,
    conv_tol: f64,
    conv_tol_dp: f64,
    conv_tol_grad: Option<f64>,
    guess: &str,
    exchange_cutoff: Option<f64>,
    use_adiis: bool,
    adiis_until_error: f64,
    damping: f64,
    level_shift: f64,
    diis_size: usize,
    field_transfer: Option<[f64; 3]>,
    field_scalar: Option<[f64; 2]>,
    field_vector: Option<[[f64; 2]; 3]>,
    response_omega: f64,
    response_damping: f64,
    response_imaginary: bool,
    response_fixed_number: bool,
    response_tol: f64,
    response_max_iter: usize,
    response_restart: usize,
    grid_preset: Option<&str>,
    grid_angular_scheme: Option<&str>,
    grid_weight_cutoff: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    use crate::periodic::electrostatics::{ChargeCompensation, ElectrostaticOptions};
    use crate::periodic::scf_boundary::{run_dimensional_scf, DimensionalScfOptions};
    validate_geometry(&numbers, &positions, multiplicity).map_err(err)?;
    let vector = |v: [f64; 3]| Vec3::new(v[0], v[1], v[2]);
    let mol = Molecule::new(
        numbers
            .into_iter()
            .zip(positions)
            .map(|(z, r)| Atom {
                z,
                position: vector(r),
            })
            .collect(),
    )
    .with_charge(charge)
    .with_multiplicity(multiplicity);
    let mb =
        MolBasis::build(&mol, &crate::basis::mixed::load_spec(basis).map_err(err)?).map_err(err)?;
    let boundary = crate::periodic::features::PeriodicBoundary::new(pbc);
    let geometry = crate::periodic::PeriodicGeometry::new(
        crate::periodic::Lattice::new(vector(cell[0]), vector(cell[1]), vector(cell[2])),
        boundary,
    )
    .map_err(err)?;
    let functional = resolve(&FunctionalSpec::Combined(xc.into())).map_err(err)?;
    let compensation=match compensation {
        "neutral" if countercharge_center.is_none() && countercharge_width.is_none()=>ChargeCompensation::RequireNeutral,
        "uniform" if countercharge_center.is_none() && countercharge_width.is_none()=>ChargeCompensation::UniformBackground,
        "gaussian"=>ChargeCompensation::GaussianCountercharge{
            center:vector(countercharge_center.ok_or_else(||PyValueError::new_err("Gaussian compensation requires countercharge_center"))?),
            width:countercharge_width.ok_or_else(||PyValueError::new_err("Gaussian compensation requires countercharge_width"))?},
        _=>return Err(PyValueError::new_err("compensation must be neutral, uniform or gaussian; centre/width belong only to gaussian")),
    };
    let mut options = DimensionalScfOptions::for_boundary(boundary);
    options.basis_cutoff = basis_cutoff;
    options.grid_cutoff = grid_cutoff;
    options.operator_tolerance = operator_tolerance;
    options.grid_tolerance = grid_tolerance;
    options.nuclear_widths = nuclear_widths.unwrap_or_default();
    options.electrostatics = ElectrostaticOptions {
        compensation,
        alpha,
        real_cutoff,
        reciprocal_cutoff,
        quadrature_order,
        tolerance,
    };
    options.scf.k_mesh = k_mesh;
    options.scf.grid = GridOptions::from_controls(
        grid_level,
        grid_preset,
        grid_angular_scheme,
        grid_weight_cutoff,
    )
    .map_err(err)?;
    options.scf.occupation = if temperature == 0.0 {
        crate::periodic::occupations::OccupationScheme::Insulator
    } else {
        crate::periodic::occupations::OccupationScheme::FermiDirac {
            temperature_kelvin: temperature,
        }
    };
    options.scf.reference = if unrestricted {
        crate::scf::Reference::Uhf
    } else {
        crate::scf::Reference::Auto
    };
    options.scf.guess = match guess {
        "core" => crate::scf::Guess::Core,
        "gwh" => crate::scf::Guess::Gwh,
        "sad" => crate::scf::Guess::Sad,
        _ => return Err(PyValueError::new_err("guess must be core, gwh or sad")),
    };
    options.scf.max_iter = max_iter;
    options.scf.e_tol = conv_tol;
    options.scf.p_tol = conv_tol_dp;
    options.scf.conv_tol_grad = conv_tol_grad.unwrap_or_else(|| conv_tol.sqrt());
    options.scf.exchange_cutoff = exchange_cutoff;
    options.scf.use_adiis = use_adiis;
    options.scf.adiis_until_error = adiis_until_error;
    options.scf.damping = damping;
    options.scf.level_shift = level_shift;
    options.scf.diis_size = diis_size;

    use crate::dfpt::field_dimensional::{
        dimensional_field_response, DimensionalFieldResponseOptions,
    };
    let request = if let Some(transfer_fractional) = field_transfer {
        if response_imaginary && response_damping != 0.0 {
            return Err(PyValueError::new_err(
                "imaginary frequency requires zero damping",
            ));
        }
        let frequency = if response_imaginary {
            crate::response::Frequency::Imaginary {
                frequency: response_omega,
            }
        } else {
            crate::response::Frequency::Retarded {
                omega: response_omega,
                damping: response_damping,
            }
        };
        let request = DimensionalFieldResponseOptions {
            transfer_fractional,
            frequency,
            fixed_number: response_fixed_number,
            scalar_potential: field_scalar
                .map(|z| crate::response::complex::Complex::new(z[0], z[1]))
                .unwrap_or_default(),
            vector_potential: field_vector
                .unwrap_or([[0.0; 2]; 3])
                .map(|z| crate::response::complex::Complex::new(z[0], z[1])),
            solver: crate::response::krylov::ResponseSolverConfig {
                tolerance: response_tol,
                max_iterations: response_max_iter,
                gmres_restart: response_restart,
                ..Default::default()
            },
        };
        request.frequency.validate().map_err(err)?;
        request.solver.validate().map_err(err)?;
        crate::periodic::quasiperiodic::PeriodicTransfer::new(&geometry, transfer_fractional)
            .map_err(err)?;
        if !request.scalar_potential.is_finite()
            || request.vector_potential.iter().any(|z| !z.is_finite())
        {
            return Err(PyValueError::new_err("field potentials must be finite"));
        }
        Some(request)
    } else {
        if field_scalar.is_some()
            || field_vector.is_some()
            || response_omega != 0.0
            || response_damping != 0.0
            || response_imaginary
            || !response_fixed_number
            || response_tol != 1e-10
            || response_max_iter != 500
            || response_restart != 40
        {
            return Err(PyValueError::new_err(
                "dimensional response controls require field_transfer",
            ));
        }
        None
    };
    let report = run_dimensional_scf(&mol, &mb, &geometry, &functional, &options).map_err(err)?;
    let s = &report.reference;
    let out = PyDict::new(py);
    for (name, value) in [
        ("energy", s.energy),
        ("free_energy", s.free_energy),
        ("mermin_free_energy", s.free_energy),
        ("entropy_term", s.entropy_term),
        ("source_energy", s.e_nuc),
        ("electronic_energy", s.e_elec),
        ("semilocal_energy", s.e_xc),
        ("kinetic_energy", report.kinetic_energy),
        ("electron_nuclear_energy", report.electron_nuclear_energy),
        ("ecp_energy", report.ecp_energy),
        ("external_field_energy", report.external_field_energy),
        ("hartree_energy", report.hartree_energy),
        ("full_exchange_energy", report.full_exchange_energy),
        (
            "long_range_exchange_energy",
            report.long_range_exchange_energy,
        ),
        ("component_energy_defect", report.component_energy_defect),
        ("final_fock_change", report.final_fock_change),
        ("occupation_consistency", report.occupation_consistency),
        ("coulomb_operator_defect", report.coulomb_operator_defect),
        (
            "coulomb_derivative_defect",
            report.coulomb_derivative_defect,
        ),
        ("coulomb_source_defect", report.coulomb_source_defect),
        ("grid_number_defect", report.grid_number_defect),
    ] {
        out.set_item(name, value)?;
    }
    let d = &report.reference_diagnostics;
    for (name, value) in [
        ("orbital_residual", d.orbital_residual),
        ("eigen_residual", d.eigen_residual),
        ("orthonormality_defect", d.orthonormality_defect),
        ("projector_defect", d.projector_defect),
        ("particle_number_error", d.particle_number_error),
    ] {
        out.set_item(name, value)?;
    }
    out.set_item("dimension", geometry.dimension())?;
    out.set_item("measure", geometry.measure())?;
    out.set_item("pbc", pbc)?;
    out.set_item("shape", [2, s.k_points.len(), mb.nao_sph, mb.nao_sph])?;
    out.set_item("formal_charge", charge)?;
    out.set_item("compensating_charge", report.compensating_charge)?;
    out.set_item("n_alpha", s.n_alpha)?;
    out.set_item("n_beta", s.n_beta)?;
    out.set_item("unrestricted", s.unrestricted)?;
    out.set_item(
        "k_points",
        s.k_points
            .iter()
            .map(|k| [k.x, k.y, k.z])
            .collect::<Vec<_>>(),
    )?;
    out.set_item("occupations", &s.occupations)?;
    out.set_item("mo_energy", &s.mo_energy)?;
    out.set_item("chemical_potential", s.chemical_potential)?;
    out.set_item("temperature", temperature)?;
    out.set_item("grid_electron_count", s.grid_electron_count)?;
    out.set_item("iterations", s.iterations)?;
    out.set_item("scf_converged", s.converged)?;
    out.set_item("converged", report.converged)?;
    out.set_item(
        "semilocal_component_energies",
        &report.semilocal_component_energies,
    )?;
    out.set_item(
        "semilocal_component_labels",
        &report.semilocal_component_labels,
    )?;
    for imaginary in [false, true] {
        let suffix = if imaginary { "imag" } else { "real" };
        for (name, matrices) in [
            ("density", &s.density),
            ("fock", &s.fock),
            ("energy_weighted_density", &s.energy_weighted_density),
        ] {
            out.set_item(
                format!("{name}_{suffix}"),
                matrices
                    .iter()
                    .map(|spin| {
                        spin.iter()
                            .map(|m| mat_rows(if imaginary { &m.im } else { &m.re }))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>(),
            )?;
        }
        for (name, matrices) in [("overlap", &s.overlap), ("h_core", &s.h_core)] {
            out.set_item(
                format!("{name}_{suffix}"),
                matrices
                    .iter()
                    .map(|m| mat_rows(if imaginary { &m.im } else { &m.re }))
                    .collect::<Vec<_>>(),
            )?;
        }
    }

    if let Some(request) = request {
        let report = dimensional_field_response(
            &mol,
            &mb,
            &geometry,
            &functional,
            &report,
            &options,
            &request,
        )
        .map_err(err)?;
        let response = PyDict::new(py);
        let value = &report.value;
        response.set_item("model", report.model)?;
        response.set_item("units", report.units)?;
        response.set_item("axis_convention", report.axis_convention)?;
        response.set_item("functional", &report.functional)?;
        response.set_item("converged", report.converged)?;
        response.set_item("shape", report.shape)?;
        response.set_item("residuals", report.residuals)?;
        response.set_item("iterations", report.iterations)?;
        response.set_item("warnings", report.warnings)?;
        response.set_item("transfer_fractional", value.transfer_fractional)?;
        response.set_item(
            "reduced_transfer_fractional",
            value.reduced_transfer_fractional,
        )?;
        response.set_item("wave", [value.wave.x, value.wave.y, value.wave.z])?;
        response.set_item(
            "reduced_transfer",
            [
                value.reduced_transfer.x,
                value.reduced_transfer.y,
                value.reduced_transfer.z,
            ],
        )?;
        let z = value.frequency.energy().map_err(err)?;
        response.set_item("frequency", [z.re, z.im])?;
        for (name, points) in [
            ("k_points", &value.k_points),
            ("target_k_points", &value.target_k_points),
        ] {
            response.set_item(
                name,
                points.iter().map(|k| [k.x, k.y, k.z]).collect::<Vec<_>>(),
            )?;
        }
        for (name, values) in [
            ("vector_potential", value.vector_potential),
            ("electric_field", value.electric_field),
            ("magnetic_field", value.magnetic_field),
            (
                "paramagnetic_number_current",
                value.paramagnetic_number_current,
            ),
            (
                "diamagnetic_number_current",
                value.diamagnetic_number_current,
            ),
            ("number_current", value.number_current),
            ("charge_current", value.charge_current),
        ] {
            response.set_item(format!("{name}_real"), values.map(|z| z.re))?;
            response.set_item(format!("{name}_imag"), values.map(|z| z.im))?;
        }
        for (name, z) in [
            ("scalar_potential", value.scalar_potential),
            ("number_density", value.number_density),
            ("charge_density", value.charge_density),
        ] {
            response.set_item(format!("{name}_real"), z.re)?;
            response.set_item(format!("{name}_imag"), z.im)?;
        }
        for (name, v) in [
            ("reference_electrons", value.reference_electrons),
            ("temperature_hartree", value.temperature_hartree),
            (
                "occupation_consistency_defect",
                value.occupation_consistency_defect,
            ),
            ("hartree_quadrature_defect", value.hartree_quadrature_defect),
            ("continuity_defect", value.continuity_defect),
            ("reference_operator_defect", value.reference_operator_defect),
            ("orbital_residual", report.reference.orbital_residual),
            ("eigen_residual", report.reference.eigen_residual),
            (
                "orthonormality_defect",
                report.reference.orthonormality_defect,
            ),
            ("projector_defect", report.reference.projector_defect),
            (
                "particle_number_error",
                report.reference.particle_number_error,
            ),
            ("reference_energy", s.energy),
            ("reference_free_energy", s.free_energy),
            ("reference_mermin_free_energy", s.free_energy),
        ] {
            response.set_item(name, v)?;
        }
        response.set_item("fixed_number", value.fixed_number)?;
        response.set_item("dimension", geometry.dimension())?;
        response.set_item("pbc", pbc)?;
        response.set_item("formal_charge", charge)?;
        for imaginary in [false, true] {
            let suffix = if imaginary { "imag" } else { "real" };
            response.set_item(
                format!("density_{suffix}"),
                value
                    .density
                    .iter()
                    .map(|spin| {
                        spin.iter()
                            .map(|m| mat_rows(if imaginary { &m.im } else { &m.re }))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>(),
            )?;
        }
        out.set_item("field_response", response)?;
    }
    out.set_item("model", report.model)?;
    out.set_item("units", report.units)?;
    out.set_item("warnings", report.warnings)?;
    out.set_item("functional", functional.description)?;
    out.set_item("basis_cutoff", basis_cutoff)?;
    out.set_item("grid_cutoff", grid_cutoff)?;
    out.set_item("grid_level", grid_level)?;
    Ok(out)
}

/// Dimensional semilocal energy/components and spin/Bloch potentials.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, density_real, basis="sto-3g", xc="pbe", charge=0,
    pbc=[true,true,true], k_mesh=[1,1,1], density_imag=None, basis_cutoff=12.0,
    grid_cutoff=12.0, grid_level=3, operator_tolerance=1e-11, grid_tolerance=1e-5))]
#[allow(clippy::too_many_arguments)]
fn periodic_xc_operators<'py>(
    py: Python<'py>,
    numbers: Vec<u8>,
    positions: Vec<[f64; 3]>,
    cell: [[f64; 3]; 3],
    density_real: Vec<Vec<Vec<Vec<f64>>>>,
    basis: &str,
    xc: &str,
    charge: i32,
    pbc: [bool; 3],
    k_mesh: [usize; 3],
    density_imag: Option<Vec<Vec<Vec<Vec<f64>>>>>,
    basis_cutoff: f64,
    grid_cutoff: f64,
    grid_level: usize,
    operator_tolerance: f64,
    grid_tolerance: f64,
) -> PyResult<Bound<'py, PyDict>> {
    use crate::periodic::xc_boundary::{periodic_xc_operators, PeriodicXcOptions};
    use crate::periodic::{BlochMatrix, Lattice, PeriodicGeometry};
    validate_geometry(&numbers, &positions, 1).map_err(err)?;
    let vector = |v: [f64; 3]| Vec3::new(v[0], v[1], v[2]);
    let mol = Molecule::new(
        numbers
            .into_iter()
            .zip(positions)
            .map(|(z, r)| Atom {
                z,
                position: vector(r),
            })
            .collect(),
    )
    .with_charge(charge);
    let mb =
        MolBasis::build(&mol, &crate::basis::mixed::load_spec(basis).map_err(err)?).map_err(err)?;
    let geometry = PeriodicGeometry::new(
        Lattice::new(vector(cell[0]), vector(cell[1]), vector(cell[2])),
        crate::periodic::features::PeriodicBoundary::new(pbc),
    )
    .map_err(err)?;
    let nk = geometry.k_points(k_mesh).map_err(err)?.len();
    let functional =
        crate::xc::resolve(&crate::xc::FunctionalSpec::Combined(xc.into())).map_err(err)?;
    let imaginary =
        density_imag.unwrap_or_else(|| vec![vec![vec![vec![0.0; mb.nao_sph]; mb.nao_sph]; nk]; 2]);
    let valid = |spin: &Vec<Vec<Vec<Vec<f64>>>>| {
        spin.len() == 2
            && spin.iter().all(|k| {
                k.len() == nk
                    && k.iter().all(|m| {
                        m.len() == mb.nao_sph
                            && m.iter().all(|row| {
                                row.len() == mb.nao_sph && row.iter().all(|v| v.is_finite())
                            })
                    })
            })
    };
    if !valid(&density_real) || !valid(&imaginary) {
        return Err(PyValueError::new_err(
            "XC densities must be finite [spin=2,k,AO,AO] arrays matching the mesh and basis",
        ));
    }
    let convert = |spin: usize| -> Vec<BlochMatrix> {
        (0..nk)
            .map(|k| BlochMatrix {
                re: Matrix::from_row_major(
                    mb.nao_sph,
                    mb.nao_sph,
                    density_real[spin][k].iter().flatten().copied().collect(),
                ),
                im: Matrix::from_row_major(
                    mb.nao_sph,
                    mb.nao_sph,
                    imaginary[spin][k].iter().flatten().copied().collect(),
                ),
            })
            .collect()
    };
    let report = periodic_xc_operators(
        &mol,
        &mb,
        &geometry,
        &functional,
        &[convert(0), convert(1)],
        &PeriodicXcOptions {
            vv10: None,
            k_mesh,
            basis_cutoff,
            grid_cutoff,
            grid: GridOptions {
                level: grid_level,
                ..Default::default()
            },
            operator_tolerance,
            grid_tolerance,
        },
    )
    .map_err(err)?;
    let out = PyDict::new(py);
    out.set_item("shape", [2, nk, mb.nao_sph, mb.nao_sph])?;
    out.set_item("dimension", geometry.dimension())?;
    out.set_item("measure", geometry.measure())?;
    out.set_item("pbc", pbc)?;
    out.set_item("formal_charge", charge)?;
    out.set_item(
        "k_points",
        report
            .k_points
            .iter()
            .map(|v| [v.x, v.y, v.z])
            .collect::<Vec<_>>(),
    )?;
    out.set_item("energy", report.energy)?;
    out.set_item("component_energies", report.component_energies)?;
    out.set_item("component_labels", report.component_labels)?;
    for imaginary in [false, true] {
        out.set_item(
            if imaginary {
                "potential_imag"
            } else {
                "potential_real"
            },
            report
                .potential
                .iter()
                .map(|spin| {
                    spin.iter()
                        .map(|m| mat_rows(if imaginary { &m.im } else { &m.re }))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        )?;
    }
    for (name, value) in [
        ("electron_count", report.electron_count),
        ("grid_electron_count", report.grid_electron_count),
        ("expected_electrons", report.expected_electrons),
        ("charge_residual", report.charge_residual),
        ("grid_number_defect", report.grid_number_defect),
        ("hermiticity_defect", report.hermiticity_defect),
        ("component_energy_defect", report.component_energy_defect),
    ] {
        out.set_item(name, value)?;
    }
    out.set_item("grid_points", report.grid_points)?;
    out.set_item("functional", &functional.description)?;
    out.set_item("family", format!("{:?}", functional.family))?;
    out.set_item(
        "exact_exchange_coefficients",
        [report.hybrid.cx, report.hybrid.clr, report.hybrid.omega],
    )?;
    out.set_item("required_nonlocal", report.required_nonlocal)?;
    out.set_item("basis_cutoff", basis_cutoff)?;
    out.set_item("grid_cutoff", grid_cutoff)?;
    out.set_item("grid_level", grid_level)?;
    out.set_item("operator_tolerance", operator_tolerance)?;
    out.set_item("grid_tolerance", grid_tolerance)?;
    out.set_item("converged", report.converged)?;
    out.set_item("model", report.model)?;
    out.set_item("units", report.units)?;
    out.set_item("warnings", report.warnings)?;
    Ok(out)
}

/// Analytic scalar/current/contact field vertices in independent Bloch spaces.
#[pyfunction]
#[pyo3(signature=(numbers,positions,cell,basis="sto-3g",charge=0,pbc=[true,true,true],
    k_mesh=[1,1,1],basis_cutoff=12.0,wave=[0.0,0.0,0.0],k_offset=[0.0,0.0,0.0],
    frequency=[0.0,0.0],scalar_potential=[0.0,0.0],vector_potential=None))]
#[allow(clippy::too_many_arguments)]
fn periodic_field_operators<'py>(
    py: Python<'py>,
    numbers: Vec<u8>,
    positions: Vec<[f64; 3]>,
    cell: [[f64; 3]; 3],
    basis: &str,
    charge: i32,
    pbc: [bool; 3],
    k_mesh: [usize; 3],
    basis_cutoff: f64,
    wave: [f64; 3],
    k_offset: [f64; 3],
    frequency: [f64; 2],
    scalar_potential: [f64; 2],
    vector_potential: Option<[[f64; 2]; 3]>,
) -> PyResult<Bound<'py, PyDict>> {
    use crate::periodic::field::{periodic_field_operators, PeriodicFieldOptions};
    use crate::response::{complex::Complex, Frequency};
    if numbers.is_empty()
        || numbers.len() != positions.len()
        || numbers.iter().any(|&z| z == 0 || z > 118)
        || positions.iter().flatten().any(|x| !x.is_finite())
    {
        return Err(PyValueError::new_err(
            "field vertices require matching atomic numbers and finite Bohr positions",
        ));
    }
    let vector = |v: [f64; 3]| Vec3::new(v[0], v[1], v[2]);
    let c = |v: [f64; 2]| Complex::new(v[0], v[1]);
    let mol = Molecule::new(
        numbers
            .iter()
            .zip(&positions)
            .map(|(&z, &p)| Atom {
                z,
                position: vector(p),
            })
            .collect(),
    )
    .with_charge(charge);
    let mb = MolBasis::build(&mol, &build_basis(basis, None).map_err(err)?).map_err(err)?;
    let geometry = crate::periodic::PeriodicGeometry::new(
        crate::periodic::Lattice::new(vector(cell[0]), vector(cell[1]), vector(cell[2])),
        crate::periodic::features::PeriodicBoundary::new(pbc),
    )
    .map_err(err)?;
    let report = periodic_field_operators(
        &mol,
        &mb,
        &geometry,
        &PeriodicFieldOptions {
            k_mesh,
            basis_cutoff,
            wave: vector(wave),
            k_offset: vector(k_offset),
            frequency: Frequency::Retarded {
                omega: frequency[0],
                damping: frequency[1],
            },
            scalar_potential: c(scalar_potential),
            vector_potential: vector_potential.unwrap_or([[0.0; 2]; 3]).map(c),
        },
    )
    .map_err(err)?;
    let out = PyDict::new(py);
    let rows = |m: &Matrix| {
        (0..m.rows)
            .map(|i| (0..m.cols).map(|j| m[(i, j)]).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };
    for (name, matrices) in [
        ("density_vertex", &report.density_vertex),
        ("first_order", &report.first_order),
        ("second_order", &report.second_order),
        ("contact_vertex", &report.contact_vertex),
    ] {
        out.set_item(
            format!("{name}_real"),
            matrices.iter().map(|m| rows(&m.re)).collect::<Vec<_>>(),
        )?;
        out.set_item(
            format!("{name}_imag"),
            matrices.iter().map(|m| rows(&m.im)).collect::<Vec<_>>(),
        )?;
    }
    out.set_item(
        "current_vertex_real",
        report
            .current_vertex
            .iter()
            .map(|s| s.iter().map(|m| rows(&m.re)).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "current_vertex_imag",
        report
            .current_vertex
            .iter()
            .map(|s| s.iter().map(|m| rows(&m.im)).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    )?;
    for (name, ks) in [
        ("k_points", &report.k_points),
        ("target_k_points", &report.target_k_points),
        ("second_target_k_points", &report.second_target_k_points),
    ] {
        out.set_item(name, ks.iter().map(|k| [k.x, k.y, k.z]).collect::<Vec<_>>())?;
    }
    out.set_item(
        "electric_field",
        report.electric_field.map(|z| [z.re, z.im]),
    )?;
    out.set_item(
        "magnetic_field",
        report.magnetic_field.map(|z| [z.re, z.im]),
    )?;
    out.set_item("frequency", [report.frequency.re, report.frequency.im])?;
    out.set_item("wave", wave)?;
    out.set_item("formal_charge", report.formal_charge)?;
    out.set_item("dimension", geometry.dimension())?;
    out.set_item("pbc", pbc)?;
    out.set_item("basis_cutoff", basis_cutoff)?;
    out.set_item("shape", [report.k_points.len(), mb.nao_sph, mb.nao_sph])?;
    out.set_item("transversality_defect", report.transversality_defect)?;
    out.set_item("units", report.units)?;
    out.set_item("model", report.model)?;
    out.set_item("axis_convention", report.axis_convention)?;
    out.set_item("warnings", report.warnings)?;
    Ok(out)
}

/// Contracted spherical AO nuclear/Hartree operators in a periodic subspace.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, basis="sto-3g", charge=0, pbc=[true,true,true],
    k_mesh=[1,1,1], basis_cutoff=12.0, density_real=None, density_imag=None,
    nuclear_widths=None, compensation="neutral", countercharge_center=None, countercharge_width=None,
    alpha=None, real_cutoff=None, reciprocal_cutoff=None, quadrature_order=96, tolerance=1e-11))]
#[allow(clippy::too_many_arguments)]
fn periodic_coulomb_operators<'py>(
    py: Python<'py>,
    numbers: Vec<u8>,
    positions: Vec<[f64; 3]>,
    cell: [[f64; 3]; 3],
    basis: &str,
    charge: i32,
    pbc: [bool; 3],
    k_mesh: [usize; 3],
    basis_cutoff: f64,
    density_real: Option<Vec<Vec<Vec<f64>>>>,
    density_imag: Option<Vec<Vec<Vec<f64>>>>,
    nuclear_widths: Option<Vec<f64>>,
    compensation: &str,
    countercharge_center: Option<[f64; 3]>,
    countercharge_width: Option<f64>,
    alpha: Option<f64>,
    real_cutoff: Option<f64>,
    reciprocal_cutoff: Option<f64>,
    quadrature_order: usize,
    tolerance: f64,
) -> PyResult<Bound<'py, PyDict>> {
    use crate::periodic::coulomb_boundary::{periodic_coulomb_operators, PeriodicCoulombOptions};
    use crate::periodic::electrostatics::{ChargeCompensation, ElectrostaticOptions};
    use crate::periodic::BlochMatrix;
    validate_geometry(&numbers, &positions, 1).map_err(err)?;
    let vector = |v: [f64; 3]| Vec3::new(v[0], v[1], v[2]);
    let mol = Molecule::new(
        numbers
            .into_iter()
            .zip(positions)
            .map(|(z, r)| Atom {
                z,
                position: vector(r),
            })
            .collect(),
    )
    .with_charge(charge);
    let mb =
        MolBasis::build(&mol, &crate::basis::mixed::load_spec(basis).map_err(err)?).map_err(err)?;
    let geometry = crate::periodic::PeriodicGeometry::new(
        crate::periodic::Lattice::new(vector(cell[0]), vector(cell[1]), vector(cell[2])),
        crate::periodic::features::PeriodicBoundary::new(pbc),
    )
    .map_err(err)?;
    let compensation_model=match compensation {
        "neutral" if countercharge_center.is_none() && countercharge_width.is_none()=>ChargeCompensation::RequireNeutral,
        "uniform" if countercharge_center.is_none() && countercharge_width.is_none()=>ChargeCompensation::UniformBackground,
        "gaussian"=>ChargeCompensation::GaussianCountercharge{
            center:vector(countercharge_center.ok_or_else(||PyValueError::new_err("Gaussian compensation requires countercharge_center"))?),
            width:countercharge_width.ok_or_else(||PyValueError::new_err("Gaussian compensation requires countercharge_width"))?},
        _=>return Err(PyValueError::new_err("compensation must be neutral, uniform or gaussian; centre/width belong only to gaussian")),
    };
    if density_real.is_none() && density_imag.is_some() {
        return Err(PyValueError::new_err("density_imag requires density_real"));
    }
    let density = if let Some(real) = density_real {
        let imaginary = density_imag
            .unwrap_or_else(|| vec![vec![vec![0.0; mb.nao_sph]; mb.nao_sph]; real.len()]);
        let valid = |matrices: &Vec<Vec<Vec<f64>>>| {
            matrices.len() == geometry.k_points(k_mesh).map_or(0, |k| k.len())
                && matrices.iter().all(|m| {
                    m.len() == mb.nao_sph
                        && m.iter()
                            .all(|row| row.len() == mb.nao_sph && row.iter().all(|v| v.is_finite()))
                })
        };
        if !valid(&real) || !valid(&imaginary) {
            return Err(PyValueError::new_err(
                "density arrays must be finite [k,AO,AO] matrices matching the mesh and basis",
            ));
        }
        Some(
            real.into_iter()
                .zip(imaginary)
                .map(|(re, im)| BlochMatrix {
                    re: Matrix::from_row_major(
                        mb.nao_sph,
                        mb.nao_sph,
                        re.into_iter().flatten().collect(),
                    ),
                    im: Matrix::from_row_major(
                        mb.nao_sph,
                        mb.nao_sph,
                        im.into_iter().flatten().collect(),
                    ),
                })
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let report = periodic_coulomb_operators(
        &mol,
        &mb,
        &geometry,
        density.as_deref(),
        &PeriodicCoulombOptions {
            k_mesh,
            basis_cutoff,
            nuclear_widths: nuclear_widths.unwrap_or_default(),
            electrostatics: ElectrostaticOptions {
                compensation: compensation_model,
                alpha,
                real_cutoff,
                reciprocal_cutoff,
                quadrature_order,
                tolerance,
            },
        },
    )
    .map_err(err)?;
    let out = PyDict::new(py);
    let array = |v: Vec3| [v.x, v.y, v.z];
    out.set_item("dimension", geometry.dimension())?;
    out.set_item("measure", geometry.measure())?;
    out.set_item("pbc", pbc)?;
    out.set_item("nao", mb.nao_sph)?;
    out.set_item("shape", [report.k_points.len(), mb.nao_sph, mb.nao_sph])?;
    out.set_item(
        "k_points",
        report
            .k_points
            .iter()
            .copied()
            .map(array)
            .collect::<Vec<_>>(),
    )?;
    out.set_item("centre_count", report.centre_count)?;
    out.set_item(
        "component_labels",
        ["real_space", "reciprocal_nonzero", "reciprocal_zero"],
    )?;
    for (name, operators, parts, derivatives) in [
        (
            "nuclear",
            Some(&report.nuclear),
            Some(&report.nuclear_components),
            Some(&report.nuclear_derivatives),
        ),
        (
            "hartree",
            report.hartree.as_ref(),
            report.hartree_components.as_ref(),
            report.hartree_derivatives.as_ref(),
        ),
    ] {
        for imaginary in [false, true] {
            let suffix = if imaginary { "imag" } else { "real" };
            let select = |m: &BlochMatrix| {
                if imaginary {
                    mat_rows(&m.im)
                } else {
                    mat_rows(&m.re)
                }
            };
            out.set_item(
                format!("{name}_{suffix}"),
                operators.map(|m| m.iter().map(&select).collect::<Vec<_>>()),
            )?;
            out.set_item(
                format!("{name}_components_{suffix}"),
                parts.map(|p| {
                    p.iter()
                        .map(|p| p.iter().map(&select).collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                }),
            )?;
            out.set_item(
                format!("{name}_derivatives_{suffix}"),
                derivatives.map(|d| {
                    d.iter()
                        .map(|centres| {
                            centres
                                .iter()
                                .map(|axes| axes.iter().map(&select).collect::<Vec<_>>())
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                }),
            )?;
        }
    }
    let components = PyDict::new(py);
    components.set_item("real_space", report.source_energy_components.real_space)?;
    components.set_item(
        "reciprocal_nonzero",
        report.source_energy_components.reciprocal_nonzero,
    )?;
    components.set_item(
        "reciprocal_zero",
        report.source_energy_components.reciprocal_zero,
    )?;
    components.set_item("point_self", report.source_energy_components.point_self)?;
    out.set_item("source_energy_components", components)?;
    out.set_item("source_energy", report.source_energy)?;
    out.set_item(
        "source_gradients",
        report
            .source_gradients
            .iter()
            .copied()
            .map(array)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "electron_nuclear_components",
        report.electron_nuclear_components,
    )?;
    out.set_item(
        "hartree_energy_components",
        report.hartree_energy_components,
    )?;
    out.set_item("total_coulomb_energy", report.total_coulomb_energy)?;
    out.set_item(
        "energy_gradients",
        report
            .energy_gradients
            .as_ref()
            .map(|g| g.iter().copied().map(array).collect::<Vec<_>>()),
    )?;
    out.set_item("electron_count", report.electron_count)?;
    out.set_item("expected_electrons", report.expected_electrons)?;
    out.set_item("formal_charge", report.formal_charge)?;
    out.set_item("compensating_charge", report.compensating_charge)?;
    out.set_item("compensation", compensation)?;
    out.set_item("countercharge_center", countercharge_center)?;
    out.set_item("countercharge_width", countercharge_width)?;
    out.set_item("charge_residual", report.charge_residual)?;
    out.set_item("basis_cutoff", basis_cutoff)?;
    out.set_item("alpha", report.params.alpha)?;
    out.set_item("real_cutoff", report.params.rcut)?;
    out.set_item("reciprocal_cutoff", report.params.gcut)?;
    out.set_item("quadrature_order", report.quadrature_order)?;
    for (name, value) in [
        (
            "quadrature_operator_defect",
            report.quadrature_operator_defect,
        ),
        (
            "quadrature_derivative_defect",
            report.quadrature_derivative_defect,
        ),
        ("quadrature_source_defect", report.quadrature_source_defect),
        ("hermiticity_defect", report.hermiticity_defect),
        ("translation_defect", report.translation_defect),
        (
            "real_space_density_imaginary_max",
            report.real_space_density_imaginary_max,
        ),
        (
            "hartree_potential_imaginary_defect",
            report.hartree_potential_imaginary_defect,
        ),
    ] {
        out.set_item(name, value)?;
    }
    out.set_item("converged", report.converged)?;
    out.set_item("model", report.model)?;
    out.set_item("units", report.units)?;
    out.set_item("warnings", report.warnings)?;
    Ok(out)
}

/// Normalized Cartesian Gaussian nuclear or electron-repulsion Green integral.
#[pyfunction]
#[pyo3(signature = (positions, exponents, angular, cell, pbc=[true,true,true],
    source_charge=None, source_width=None, alpha=None, real_cutoff=None,
    reciprocal_cutoff=None, quadrature_order=96, tolerance=1e-11))]
#[allow(clippy::too_many_arguments)]
fn periodic_gaussian_integral(
    py: Python<'_>,
    positions: Vec<[f64; 3]>,
    exponents: Vec<f64>,
    angular: Vec<[usize; 3]>,
    cell: [[f64; 3]; 3],
    pbc: [bool; 3],
    source_charge: Option<f64>,
    source_width: Option<f64>,
    alpha: Option<f64>,
    real_cutoff: Option<f64>,
    reciprocal_cutoff: Option<f64>,
    quadrature_order: usize,
    tolerance: f64,
) -> PyResult<Bound<'_, PyDict>> {
    use crate::periodic::gaussian_coulomb::{
        CartesianGaussian, GaussianIntegralOptions, GaussianNucleus,
    };
    let nuclear = positions.len() == 3;
    let count = if nuclear { 2 } else { 4 };
    if !matches!(positions.len(), 3 | 4) || exponents.len() != count || angular.len() != count {
        return Err(PyValueError::new_err(
            "provide three centres with two exponents/angular triples for nuclear \
            attraction, or four centres with four for electron repulsion",
        ));
    }
    if !nuclear && (source_charge.is_some() || source_width.is_some()) {
        return Err(PyValueError::new_err(
            "source controls apply only to nuclear attraction",
        ));
    }
    let vector = |v: [f64; 3]| Vec3::new(v[0], v[1], v[2]);
    let geometry = crate::periodic::PeriodicGeometry::new(
        crate::periodic::Lattice::new(vector(cell[0]), vector(cell[1]), vector(cell[2])),
        crate::periodic::features::PeriodicBoundary::new(pbc),
    )
    .map_err(err)?;
    let gaussians: Vec<_> = (0..count)
        .map(|i| CartesianGaussian {
            center: vector(positions[i]),
            exponent: exponents[i],
            angular: angular[i],
        })
        .collect();
    let source = nuclear.then(|| GaussianNucleus {
        center: vector(positions[2]),
        charge: source_charge.unwrap_or(1.0),
        width: source_width.unwrap_or(0.0),
    });
    let report = crate::periodic::gaussian_coulomb::periodic_gaussian_integral(
        &geometry,
        &gaussians,
        source.as_ref(),
        &GaussianIntegralOptions {
            alpha,
            real_cutoff,
            reciprocal_cutoff,
            quadrature_order,
            tolerance,
        },
    )
    .map_err(err)?;
    let out = PyDict::new(py);
    out.set_item("value", report.value)?;
    let components = PyDict::new(py);
    components.set_item("real_space", report.components.real_space)?;
    components.set_item("reciprocal_nonzero", report.components.reciprocal_nonzero)?;
    components.set_item("reciprocal_zero", report.components.reciprocal_zero)?;
    components.set_item("point_self", report.components.point_self)?;
    out.set_item("components", components)?;
    out.set_item(
        "center_gradients",
        report
            .center_gradients
            .iter()
            .map(|v| [v.x, v.y, v.z])
            .collect::<Vec<_>>(),
    )?;
    out.set_item("shape", [positions.len(), 3])?;
    out.set_item("pbc", pbc)?;
    out.set_item("dimension", geometry.dimension())?;
    out.set_item("measure", geometry.measure())?;
    out.set_item("alpha", report.params.alpha)?;
    out.set_item("real_cutoff", report.params.rcut)?;
    out.set_item("reciprocal_cutoff", report.params.gcut)?;
    out.set_item("quadrature_order", report.quadrature_order)?;
    out.set_item("quadrature_value_defect", report.quadrature_value_defect)?;
    out.set_item(
        "quadrature_gradient_defect",
        report.quadrature_gradient_defect,
    )?;
    out.set_item("translation_defect", report.translation_defect)?;
    out.set_item("converged", report.converged)?;
    out.set_item("kind", report.kind)?;
    out.set_item("units", report.units)?;
    out.set_item("model", report.model)?;
    out.set_item("warnings", report.warnings)?;
    Ok(out)
}

/// Dimensional Coulomb foundation using prescribed point/Gaussian charges.
#[pyfunction]
#[pyo3(signature = (positions, charges, cell, pbc=[true,true,true], widths=None,
    compensation="neutral", countercharge_center=None, countercharge_width=None,
    alpha=None, real_cutoff=None, reciprocal_cutoff=None, quadrature_order=96,
    tolerance=1e-11, k_mesh=[1,1,1]))]
#[allow(clippy::too_many_arguments)]
fn periodic_electrostatics<'py>(
    py: Python<'py>,
    positions: Vec<[f64; 3]>,
    charges: Vec<f64>,
    cell: [[f64; 3]; 3],
    pbc: [bool; 3],
    widths: Option<Vec<f64>>,
    compensation: &str,
    countercharge_center: Option<[f64; 3]>,
    countercharge_width: Option<f64>,
    alpha: Option<f64>,
    real_cutoff: Option<f64>,
    reciprocal_cutoff: Option<f64>,
    quadrature_order: usize,
    tolerance: f64,
    k_mesh: [usize; 3],
) -> PyResult<Bound<'py, PyDict>> {
    use crate::periodic::electrostatics::{
        ChargeCompensation, ElectrostaticOptions, ElectrostaticSource,
    };
    let vector = |v: [f64; 3]| Vec3::new(v[0], v[1], v[2]);
    let geometry = crate::periodic::PeriodicGeometry::new(
        crate::periodic::Lattice::new(vector(cell[0]), vector(cell[1]), vector(cell[2])),
        crate::periodic::features::PeriodicBoundary::new(pbc),
    )
    .map_err(err)?;
    let points = geometry.k_points(k_mesh).map_err(err)?;
    let widths = widths.unwrap_or_else(|| vec![0.0; charges.len()]);
    if positions.len() != charges.len() || widths.len() != charges.len() {
        return Err(PyValueError::new_err(
            "electrostatic positions, charges and widths must have the same nonzero length",
        ));
    }
    let compensation = match compensation {
        "neutral" if countercharge_center.is_none() && countercharge_width.is_none() => ChargeCompensation::RequireNeutral,
        "uniform" if countercharge_center.is_none() && countercharge_width.is_none() => ChargeCompensation::UniformBackground,
        "gaussian" => ChargeCompensation::GaussianCountercharge {
            center:vector(countercharge_center.ok_or_else(|| PyValueError::new_err("Gaussian compensation requires countercharge_center"))?),
            width:countercharge_width.ok_or_else(|| PyValueError::new_err("Gaussian compensation requires countercharge_width"))?,
        },
        _ => return Err(PyValueError::new_err("compensation must be neutral, uniform or gaussian; center/width belong only to gaussian")),
    };
    let sources: Vec<_> = positions
        .into_iter()
        .zip(charges)
        .zip(widths)
        .map(|((r, charge), width)| ElectrostaticSource {
            position: vector(r),
            charge,
            width,
        })
        .collect();
    let report = crate::periodic::electrostatics::periodic_electrostatics(
        &geometry,
        &sources,
        &ElectrostaticOptions {
            compensation,
            alpha,
            real_cutoff,
            reciprocal_cutoff,
            quadrature_order,
            tolerance,
        },
    )
    .map_err(err)?;
    let out = PyDict::new(py);
    let array = |v: Vec3| [v.x, v.y, v.z];
    out.set_item("energy", report.energy)?;
    let components = PyDict::new(py);
    components.set_item("real_space", report.components.real_space)?;
    components.set_item("reciprocal_nonzero", report.components.reciprocal_nonzero)?;
    components.set_item("reciprocal_zero", report.components.reciprocal_zero)?;
    components.set_item("point_self", report.components.point_self)?;
    out.set_item("components", components)?;
    out.set_item(
        "forces",
        report.forces.iter().copied().map(array).collect::<Vec<_>>(),
    )?;
    out.set_item("site_potentials", report.site_potentials)?;
    out.set_item("compensation_force", report.compensation_force.map(array))?;
    out.set_item("compensation_potential", report.compensation_potential)?;
    out.set_item("supplied_charge", report.supplied_charge)?;
    out.set_item("compensating_charge", report.compensating_charge)?;
    out.set_item("pbc", pbc)?;
    out.set_item("dimension", geometry.dimension())?;
    out.set_item("measure", report.measure)?;
    out.set_item(
        "reciprocal_vectors",
        geometry
            .reciprocal()
            .into_iter()
            .map(array)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "k_points",
        points.into_iter().map(array).collect::<Vec<_>>(),
    )?;
    out.set_item("alpha", report.params.alpha)?;
    out.set_item("real_cutoff", report.params.rcut)?;
    out.set_item("reciprocal_cutoff", report.params.gcut)?;
    out.set_item("quadrature_order", report.quadrature_order)?;
    out.set_item("quadrature_energy_defect", report.quadrature_energy_defect)?;
    out.set_item("quadrature_force_defect", report.quadrature_force_defect)?;
    out.set_item("force_sum_defect", report.force_sum_defect)?;
    out.set_item("converged", report.converged)?;
    out.set_item("units", report.units)?;
    out.set_item("model", report.model)?;
    out.set_item("potential_convention", report.potential_convention)?;
    out.set_item("warnings", report.warnings)?;
    out.set_item("shape", [sources.len(), 3])?;
    Ok(out)
}

/// Discover the periodic feature boundary without trial calculations.
#[pyfunction]
fn periodic_capabilities(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let capabilities = crate::periodic::features::PeriodicCapabilities::current();
    let out = PyDict::new(py);
    out.set_item("finite_k", capabilities.finite_k)?;
    out.set_item("exact_exchange", capabilities.exact_exchange)?;
    out.set_item("meta_gga", capabilities.meta_gga)?;
    out.set_item("analytic_forces", capabilities.analytic_forces)?;
    out.set_item(
        "fixed_cell_optimization",
        capabilities.fixed_cell_optimization,
    )?;
    out.set_item(
        "fermi_dirac_occupations",
        capabilities.fermi_dirac_occupations,
    )?;
    out.set_item("periodic_ecp_energy", capabilities.periodic_ecp_energy)?;
    out.set_item("periodic_ecp_forces", capabilities.periodic_ecp_forces)?;
    out.set_item("partial_boundaries", capabilities.partial_boundaries)?;
    out.set_item(
        "dimensional_electrostatics",
        capabilities.dimensional_electrostatics,
    )?;
    out.set_item("partial_geometry", capabilities.partial_geometry)?;
    out.set_item(
        "dimensional_basis_operators",
        capabilities.dimensional_basis_operators,
    )?;
    out.set_item("dimensional_xc_grid", capabilities.dimensional_xc_grid)?;
    out.set_item(
        "dimensional_xc_operators",
        capabilities.dimensional_xc_operators,
    )?;
    out.set_item(
        "dimensional_coulomb_operators",
        capabilities.dimensional_coulomb_operators,
    )?;
    out.set_item(
        "dimensional_gaussian_integrals",
        capabilities.dimensional_gaussian_integrals,
    )?;
    out.set_item(
        "charged_lower_dimensional_electrostatics",
        capabilities.charged_lower_dimensional_electrostatics,
    )?;
    out.set_item("vv10", capabilities.vv10)?;
    out.set_item("chg", capabilities.chg)?;
    Ok(out)
}

/// Monkhorst-Pack k-point mesh for a lattice (Cartesian, 1/Bohr).
#[pyfunction]
#[pyo3(signature = (cell, mesh, pbc=[true,true,true]))]
fn monkhorst_pack(
    cell: [[f64; 3]; 3],
    mesh: [usize; 3],
    pbc: [bool; 3],
) -> PyResult<Vec<Vec<f64>>> {
    use crate::periodic::Lattice;
    if cell.iter().flatten().any(|x| !x.is_finite()) || mesh.contains(&0) {
        return Err(PyValueError::new_err(
            "cell vectors must be finite and every mesh dimension must be at least 1",
        ));
    }
    let lat = Lattice::new(
        Vec3::new(cell[0][0], cell[0][1], cell[0][2]),
        Vec3::new(cell[1][0], cell[1][1], cell[1][2]),
        Vec3::new(cell[2][0], cell[2][1], cell[2][2]),
    );
    let geometry = crate::periodic::PeriodicGeometry::new(
        lat,
        crate::periodic::features::PeriodicBoundary::new(pbc),
    )
    .map_err(err)?;
    Ok(geometry
        .k_points(mesh)
        .map_err(err)?
        .iter()
        .map(|k| vec![k.x, k.y, k.z])
        .collect())
}

/// Convert a common-driver report into plain Python objects.
fn value_to_py<'py>(py: Python<'py>, value: &crate::driver::Value) -> PyResult<Bound<'py, PyAny>> {
    use crate::driver::Value;
    Ok(match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => pyo3::types::PyBool::new(py, *b).to_owned().into_any(),
        Value::Int(i) => i.into_pyobject(py)?.into_any(),
        Value::Float(f) => pyo3::types::PyFloat::new(py, *f).into_any(),
        Value::Str(s) => pyo3::types::PyString::new(py, s).into_any(),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(value_to_py(py, item)?)?;
            }
            list.into_any()
        }
        Value::Object(entries) => {
            let dict = PyDict::new(py);
            for (key, item) in entries {
                dict.set_item(key, value_to_py(py, item)?)?;
            }
            dict.into_any()
        }
    })
}

/// Convert Python keyword values (scalars, strings, complex numbers, lists,
/// tuples, dictionaries and NumPy arrays) into the driver value tree.
fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<crate::driver::Value> {
    use crate::driver::Value;
    use pyo3::types::{PyBool, PyFloat, PyInt, PyString, PyTuple};
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = obj.downcast::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    if obj.is_instance_of::<PyInt>() {
        return Ok(Value::Int(obj.extract::<i64>()?));
    }
    if obj.is_instance_of::<PyFloat>() {
        return Ok(Value::Float(obj.extract::<f64>()?));
    }
    if let Ok(s) = obj.downcast::<PyString>() {
        return Ok(Value::Str(s.to_cow()?.into_owned()));
    }
    if obj.is_instance_of::<pyo3::types::PyComplex>() {
        let re: f64 = obj.getattr("real")?.extract()?;
        let im: f64 = obj.getattr("imag")?.extract()?;
        return Ok(Value::Array(vec![Value::Float(re), Value::Float(im)]));
    }
    if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut entries = Vec::new();
        for (key, item) in dict.iter() {
            let key: String = key
                .extract()
                .map_err(|_| PyValueError::new_err("control dictionary keys must be strings"))?;
            entries.push((key, py_to_value(&item)?));
        }
        return Ok(Value::Object(entries));
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        let mut items = Vec::new();
        for item in obj.try_iter()? {
            items.push(py_to_value(&item?)?);
        }
        return Ok(Value::Array(items));
    }
    if obj.hasattr("tolist")? {
        return py_to_value(&obj.call_method0("tolist")?);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "unsupported control value of type {}",
        obj.get_type().name()?
    )))
}

/// Run any calculation registered with the common driver. Positions and
/// lattice rows are Bohr; keyword arguments are the calculation controls.
/// Returns the driver report as dictionaries and lists in atomic units
/// unless a key names another unit.
#[pyfunction]
#[pyo3(signature = (feature, numbers=None, positions=None, cell=None, pbc=None, **controls))]
fn run_feature<'py>(
    py: Python<'py>,
    feature: String,
    numbers: Option<Vec<u8>>,
    positions: Option<Vec<[f64; 3]>>,
    cell: Option<[[f64; 3]; 3]>,
    pbc: Option<[bool; 3]>,
    controls: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut entries = Vec::new();
    if let Some(dict) = controls {
        for (key, item) in dict.iter() {
            entries.push((key.extract::<String>()?, py_to_value(&item)?));
        }
    }
    let controls =
        crate::driver::Controls::from_value(&crate::driver::Value::Object(entries)).map_err(err)?;
    let system = crate::driver::System {
        numbers: numbers.unwrap_or_default(),
        positions: positions.unwrap_or_default(),
        pbc: pbc.unwrap_or(if cell.is_some() {
            [true; 3]
        } else {
            [false; 3]
        }),
        cell,
    };
    let report = py
        .allow_threads(move || crate::driver::run(&feature, &system, &controls))
        .map_err(err)?;
    value_to_py(py, &report)
}

/// Execute an existing Python adapter in the same scoped resource envelope
/// as the common Rust driver. The caller releases the GIL while the private
/// worker acquires it; numerical bindings can release it normally inside.
#[pyfunction]
#[pyo3(signature = (function, args, kwargs, cpu_threads=None, memory_mb=None, log_file=None, log_mode=None, log_level=None))]
fn _with_resources(
    py: Python<'_>,
    function: Py<PyAny>,
    args: Py<PyTuple>,
    kwargs: Py<PyDict>,
    cpu_threads: Option<usize>,
    memory_mb: Option<usize>,
    log_file: Option<String>,
    log_mode: Option<String>,
    log_level: Option<String>,
) -> PyResult<Py<PyAny>> {
    let limits = crate::resources::ResourceLimits::new(cpu_threads, memory_mb).map_err(err)?;
    let logging =
        crate::logger::LogOptions::from_strings(log_file, log_mode, log_level).map_err(err)?;
    let (result, usage, configured) = py
        .allow_threads(move || {
            logging
                .install(|| {
                    let outcome = limits.install(move || {
                        if let Some(usage) = crate::resources::current_usage() {
                            crate::logger::log(&format!(
                                "resources cpu_threads={} memory_mb={:?}",
                                usage.cpu_threads, usage.memory_mb
                            ));
                        }
                        let outcome =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                Python::with_gil(|py| {
                                    if let Ok(name) = function.bind(py).getattr("__qualname__") {
                                        if let Ok(name) = name.extract::<String>() {
                                            crate::logger::log(&format!("Python callable={name}"));
                                        }
                                    }
                                    function.call(py, args, Some(kwargs.bind(py)))
                                })
                            }));
                        let failure = crate::resources::take_failure();
                        let result = match (failure, outcome) {
                            (Some(error), _) => Err(err(error)),
                            (None, Ok(result)) => result,
                            (None, Err(payload)) => std::panic::resume_unwind(payload),
                        };
                        match &result {
                            Err(error) => crate::logger::record_error(&error.to_string()),
                            Ok(value) if crate::logger::enabled(crate::logger::LogLevel::Info) => {
                                Python::with_gil(|py| {
                                    if let Ok(report) = value.bind(py).downcast::<PyDict>() {
                                        for key in [
                                            "energy",
                                            "free_energy",
                                            "scf",
                                            "converged",
                                            "units",
                                            "unit_labels",
                                        ] {
                                            if let Ok(Some(value)) = report.get_item(key) {
                                                if key == "scf" {
                                                    if let Ok(scf) = value.downcast::<PyDict>() {
                                                        for field in [
                                                            "energy",
                                                            "iterations",
                                                            "converged",
                                                            "scf_error",
                                                            "orbital_residual",
                                                        ] {
                                                            if let Ok(Some(value)) =
                                                                scf.get_item(field)
                                                            {
                                                                if let Ok(text) = value.str() {
                                                                    crate::logger::log(&format!(
                                                                        "result scf.{field}={text}"
                                                                    ));
                                                                }
                                                            }
                                                        }
                                                    }
                                                    continue;
                                                }
                                                if let Ok(text) = value.str() {
                                                    crate::logger::log(&format!(
                                                        "result {key}={text}"
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                })
                            }
                            Ok(_) => (),
                        }
                        (
                            result,
                            crate::resources::current_usage(),
                            crate::resources::scope_is_configured(),
                        )
                    });
                    if let Err(error) = &outcome {
                        crate::logger::record_error(&error.to_string());
                    }
                    outcome
                })
                .and_then(|result| result)
        })
        .map_err(err)?;
    let result = result?;
    if let (Some(usage), Ok(report)) = (usage, result.bind(py).downcast::<PyDict>()) {
        if !configured && report.contains("resources")? {
            return Ok(result);
        }
        let metadata = PyDict::new(py);
        metadata.set_item("cpu_threads", usage.cpu_threads)?;
        metadata.set_item("memory_mb", usage.memory_mb)?;
        metadata.set_item("managed_peak_bytes", usage.peak_reserved_bytes)?;
        metadata.set_item("managed_live_bytes", usage.reserved_bytes)?;
        metadata.set_item(
            "scope",
            "registered Rust matrices, ERI storage and workspace allowances; not process RSS",
        )?;
        report.set_item("resources", metadata)?;
    }
    Ok(result)
}

#[pyfunction]
fn _logging_active() -> bool {
    crate::logger::is_active()
}

#[pyfunction]
fn _resource_scope_configured() -> bool {
    crate::resources::scope_is_configured()
}

#[pyfunction]
fn _resource_usage(py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
    crate::resources::current_usage()
        .map(|usage| {
            let out = PyDict::new(py);
            out.set_item("cpu_threads", usage.cpu_threads)?;
            out.set_item("memory_mb", usage.memory_mb)?;
            Ok(out.unbind())
        })
        .transpose()
}

/// Run the `dft-rs` command line with `argv` (without the program name) and
/// return its exit status. Output goes to the process standard streams.
#[pyfunction]
fn cli_main(py: Python<'_>, argv: Vec<String>) -> i32 {
    py.allow_threads(move || crate::cli::run_arguments(argv))
}

#[pymodule]
fn dft_rs_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_with_resources, m)?)?;
    m.add_function(wrap_pyfunction!(_resource_usage, m)?)?;
    m.add_function(wrap_pyfunction!(_logging_active, m)?)?;
    m.add_function(wrap_pyfunction!(_resource_scope_configured, m)?)?;
    m.add_function(wrap_pyfunction!(run_feature, m)?)?;
    m.add_function(wrap_pyfunction!(cli_main, m)?)?;
    m.add_function(wrap_pyfunction!(calculate, m)?)?;
    m.add_function(wrap_pyfunction!(hessian, m)?)?;
    m.add_function(wrap_pyfunction!(vibrations, m)?)?;
    m.add_function(wrap_pyfunction!(raman, m)?)?;
    m.add_function(wrap_pyfunction!(optimize, m)?)?;
    m.add_function(wrap_pyfunction!(saddle, m)?)?;
    m.add_function(wrap_pyfunction!(polarizability, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_polarizability, m)?)?;
    m.add_function(wrap_pyfunction!(electric_field_taylor, m)?)?;
    m.add_function(wrap_pyfunction!(nuclear_taylor, m)?)?;
    m.add_function(wrap_pyfunction!(anharmonic, m)?)?;
    m.add_function(wrap_pyfunction!(nmr, m)?)?;
    m.add_function(wrap_pyfunction!(tddft, m)?)?;
    m.add_function(wrap_pyfunction!(sapt, m)?)?;
    m.add_function(wrap_pyfunction!(basis_sets, m)?)?;
    m.add_function(wrap_pyfunction!(functionals, m)?)?;
    m.add_function(wrap_pyfunction!(periodic, m)?)?;
    m.add_function(wrap_pyfunction!(periodic_scf, m)?)?;
    m.add_function(wrap_pyfunction!(periodic_capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(periodic_electrostatics, m)?)?;
    m.add_function(wrap_pyfunction!(periodic_basis, m)?)?;
    m.add_function(wrap_pyfunction!(periodic_gaussian_integral, m)?)?;
    m.add_function(wrap_pyfunction!(periodic_coulomb_operators, m)?)?;
    m.add_function(wrap_pyfunction!(periodic_xc_operators, m)?)?;
    m.add_function(wrap_pyfunction!(dimensional_scf, m)?)?;
    m.add_function(wrap_pyfunction!(periodic_field_operators, m)?)?;
    m.add_function(wrap_pyfunction!(monkhorst_pack, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
