// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Pure, fixed-template generation of candidate one-center methyl operators.
//! Generation is not a chemical-accuracy or production-force acceptance gate.

use crate::basis::{data, MolBasis};
use crate::driver::continuous_species::gaussian::gaussian_potential;
use crate::driver::continuous_species::reduction_linear::{
    constrained_projection, operator_projection_dataset, ConstrainedProjection, ElectronLedger,
    FrozenFragmentSpec, OperatorProjectionSample, ProjectionConstraints,
};
use crate::driver::Value;
use crate::grid::{GridOptions, MolecularGrid};
use crate::integrals::{kinetic_matrix, nuclear_matrix};
use crate::ks::xc_contribution;
use crate::linalg::{solve_linear_matrix, symmetric_eigen, Matrix};
use crate::math::{Mat3, Vec3};
use crate::pseudo_sub::reduction_basis::{
    build_reduction_basis, cross_basis_overlap, OneCenterBasisTemplate,
};
use crate::pseudo_sub::reduction_geometry::{FragmentSpec, RigidExpansion};
use crate::pseudo_sub::reduction_kernels::{
    charge_operator, core_interactions, projector_overlaps, GaussianCharge, ProjectorTemplate,
};
use crate::pseudo_sub::reduction_reference::{
    extract_methyl_frozen_space, ExtractionOptions, FrozenReferenceModel, MethylBoundary,
};
use crate::pseudo_sub::reduction_support::SupportProjection;
use crate::scf::{run_scf, EriMode, Guess, ScfOptions};
use crate::system::{Atom, Molecule};
use crate::xc::{resolve, FunctionalSpec};
use crate::{DftError, Result};

pub const MODEL_VERSION: &str = "static-v1";
pub const SOURCE_VERSION: &str = "0.3.0";
pub const TEMPLATE_ID: &str = "one-center-h1s-local3-projector8-charge3-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionRule {
    FullAoMetric,
    PhysicalSupportPauliNormalized,
    /// Fit the fixed residual on the actual common production AO subset.
    /// The scalar reference, electron ledger and operator widths are unchanged.
    ReducedAoMetric,
}

impl ProjectionRule {
    pub fn id(self) -> &'static str {
        match self {
            Self::FullAoMetric => "full-ao-metric-v1",
            Self::PhysicalSupportPauliNormalized => "physical-support-pauli-normalized-v1",
            Self::ReducedAoMetric => "reduced-ao-metric-v1",
        }
    }
    pub fn from_id(id: &str) -> Result<Self> {
        match id {
            "full-ao-metric-v1" => Ok(Self::FullAoMetric),
            "physical-support-pauli-normalized-v1" => Ok(Self::PhysicalSupportPauliNormalized),
            "reduced-ao-metric-v1" => Ok(Self::ReducedAoMetric),
            _ => invalid("unsupported fixed operator projection rule"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenerationOptions {
    pub basis_template: OneCenterBasisTemplate,
    pub projection_rule: ProjectionRule,
    pub functional: String,
    pub grid_level: usize,
    pub local_exponents: [f64; 3],
    pub charge_exponents: [f64; 3],
    pub s_exponents: [f64; 2],
    pub p_exponents: [f64; 2],
    pub pauli_level: f64,
    pub regularization: f64,
    /// Empty by default; any supplied probes are construction data, not holdout.
    pub electrostatic_points: Vec<Vec3>,
    pub electrostatic_weight: f64,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            basis_template: OneCenterBasisTemplate::Hydrogen1s,
            projection_rule: ProjectionRule::FullAoMetric,
            functional: "pbe".into(),
            grid_level: 1,
            local_exponents: [0.25, 1.0, 4.0],
            charge_exponents: [0.2, 0.8, 3.2],
            s_exponents: [0.35, 1.4],
            p_exponents: [0.35, 1.4],
            pauli_level: 2.0,
            regularization: 1e-8,
            electrostatic_points: Vec::new(),
            electrostatic_weight: 1.0,
        }
    }
}

impl GenerationOptions {
    fn validate(&self) -> Result<()> {
        let template = Self::default();
        if self.projection_rule == ProjectionRule::ReducedAoMetric
            && self.basis_template == OneCenterBasisTemplate::Hydrogen1s
        {
            return invalid("reduced-AO metric projection requires a fixed carbon AO subset of the all-electron reference; an external-span H1s embedding is not defined by this rule");
        }
        if !matches!(self.functional.as_str(), "lda" | "pbe")
            || self.local_exponents != template.local_exponents
            || self.charge_exponents != template.charge_exponents
            || self.s_exponents != template.s_exponents
            || self.p_exponents != template.p_exponents
            || self.grid_level > 9
            || self
                .local_exponents
                .iter()
                .chain(&self.charge_exponents)
                .chain(&self.s_exponents)
                .chain(&self.p_exponents)
                .any(|v| !v.is_finite() || *v <= 0.0)
            || !self.pauli_level.is_finite()
            || self.pauli_level <= 0.0
            || !self.regularization.is_finite()
            || self.regularization < 0.0
            || !self.electrostatic_weight.is_finite()
            || self.electrostatic_weight <= 0.0
            || self.electrostatic_points.iter().any(|p| !finite(*p))
        {
            return invalid("generation requires LDA/PBE, positive fixed widths/Pauli level and finite nonnegative ridge");
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ProjectionReport {
    pub data_rank: usize,
    pub free_parameters: usize,
    pub regularized_rank: usize,
    pub data_condition_number: Option<f64>,
    pub regularized_condition_number: Option<f64>,
    pub weighted_residual_squared: f64,
    pub regularization_penalty: f64,
    pub constraint_max_residual: f64,
    pub projected_stationarity_residual: f64,
}

impl From<&ConstrainedProjection> for ProjectionReport {
    fn from(p: &ConstrainedProjection) -> Self {
        Self {
            data_rank: p.data_rank,
            free_parameters: p.free_parameters,
            regularized_rank: p.regularized_rank,
            data_condition_number: p.data_condition_number,
            regularized_condition_number: p.regularized_condition_number,
            weighted_residual_squared: p.weighted_residual_squared,
            regularization_penalty: p.regularization_penalty,
            constraint_max_residual: p.constraint_max_residual,
            projected_stationarity_residual: p.projected_stationarity_residual,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenerationDiagnostics {
    pub full_ao_count: usize,
    pub reduced_ao_count: usize,
    pub full_reference_energy: f64,
    pub projected_reference_energy_without_constant: f64,
    pub reference_energy_reconstruction_error: f64,
    pub active_trace: f64,
    pub frozen_trace: f64,
    pub orbital_reconstruction_error: f64,
    pub active_frozen_overlap: f64,
    pub target_reference_pauli_expectation: f64,
    pub sigma_charge_defect: f64,
    pub sigma_dipole_defect: f64,
    pub sigma_projection: ProjectionReport,
    pub electronic_projection: ProjectionReport,
}

/// Sole canonical generated record. There is no implicit named-parser activation.
#[derive(Clone, Debug)]
pub struct ReducedFragment {
    pub model_version: String,
    pub source_version: String,
    pub template_id: String,
    pub source: Molecule,
    pub spec: FragmentSpec,
    pub geometry_map: RigidExpansion,
    pub electron_ledger: ElectronLedger,
    pub options: GenerationOptions,
    /// Reference ions contain +1 at the center, independently of its H basis label.
    pub baseline_ionic_charge: f64,
    /// Neutral Gaussian correction, intrinsic dipoles stored in the body frame.
    pub gaussian_charges: [f64; 3],
    pub gaussian_body_dipoles: [[f64; 3]; 3],
    pub local_amplitudes: [f64; 3],
    /// Upper triangle, row order (0,0),(0,1),...,(7,7); not constrained PSD.
    pub nonlocal_coefficients: [f64; 36],
    pub internal_energy_reference: f64,
    pub diagnostics: GenerationDiagnostics,
    /// Deterministic FNV-1a of exact construction inputs, not a security hash.
    pub construction_fingerprint: String,
}

fn invalid<T>(s: &str) -> Result<T> {
    Err(DftError::InvalidInput(s.into()))
}
fn finite(v: Vec3) -> bool {
    (0..3).all(|i| v.get(i).is_finite())
}
fn add(a: &mut Matrix, b: &Matrix, scale: f64) {
    for (x, y) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *x += scale * y;
    }
}
fn scaled(a: &Matrix, scale: f64) -> Matrix {
    let mut b = a.clone();
    for x in b.as_mut_slice() {
        *x *= scale;
    }
    b
}
fn body(u: Mat3, v: Vec3) -> [f64; 3] {
    std::array::from_fn(|i| u.col[i].dot(v))
}
fn global(u: Mat3, v: [f64; 3]) -> Vec3 {
    u.col[0] * v[0] + u.col[1] * v[1] + u.col[2] * v[2]
}

fn metric(s: &Matrix) -> Result<Matrix> {
    let (values, vectors) = symmetric_eigen(s)?;
    if values.iter().any(|x| *x <= 1e-10) {
        return invalid("generation refuses a rank-deficient full AO metric");
    }
    let mut x = Matrix::zeros(s.rows, s.cols);
    for k in 0..s.rows {
        for i in 0..s.rows {
            for j in 0..s.cols {
                x[(i, j)] += vectors[(i, k)] * vectors[(j, k)] / values[k].sqrt();
            }
        }
    }
    Ok(x)
}

/// The same physical local/projector kernels build the full reference and
/// production-space dictionaries. No converged full Fock is added here.
fn electronic_dictionary(
    basis: &MolBasis,
    atom_count: usize,
    center: Vec3,
    orientation: Mat3,
    options: &GenerationOptions,
) -> Result<Vec<Matrix>> {
    let mut dictionary: Vec<_> = options
        .local_exponents
        .iter()
        .map(|&a| gaussian_potential(basis, atom_count, center, a).map(|value| value.operator))
        .collect::<Result<_>>()?;
    let overlaps = projector_overlaps(
        basis,
        atom_count,
        center,
        &ProjectorTemplate {
            s_exponents: options.s_exponents,
            p_exponents: options.p_exponents,
            orientation,
        },
    )?;
    dictionary.extend(overlaps.coefficient_operators());
    Ok(dictionary)
}

fn project(
    target: Matrix,
    dictionary: Vec<Matrix>,
    x: &Matrix,
    name: &str,
    constraints: Matrix,
    values: Vec<f64>,
    ridge: f64,
    esp: Option<(&[f64], &[Vec<f64>], f64)>,
) -> Result<ConstrainedProjection> {
    let n = dictionary.len();
    let mut dataset = operator_projection_dataset(
        &[OperatorProjectionSample {
            case_id: name.into(),
            group_id: "construction-reference".into(),
            target,
            dictionary,
            metric_weight: x.clone(),
            weight: 1.0,
        }],
        (0..n).map(|i| format!("{name}[{i}]")).collect(),
        "closed-rks-sto3g-fixed-grid".into(),
        format!("methyl-reference-{name}"),
    )?;
    if let Some((targets, features, weight)) = esp {
        use crate::pseudo_sub::fit::{ObservableKind, ReferenceDatum};
        for (i, (&target, feature)) in targets.iter().zip(features).enumerate() {
            dataset.data.push(ReferenceDatum {
                case_id: format!("esp[{i}]"),
                group_id: "construction-reference".into(),
                observable: ObservableKind::Other("electrostatic-potential".into()),
                target,
                scale: 1.0,
                weight,
                linear_features: feature.clone(),
                diagnostics: None,
            });
        }
    }
    constrained_projection(
        &dataset,
        &ProjectionConstraints {
            matrix: constraints,
            values,
            prior: vec![0.0; n],
            regularization_operator: Matrix::identity(n),
            regularization_strength: ridge,
            constraint_absolute_tolerance: 1e-9,
            relative_rank_tolerance: 1e-12,
        },
    )
}

impl ReducedFragment {
    pub fn candidate_status(&self) -> &'static str {
        "candidate-not-chemically-validated"
    }
    pub fn shape_status(&self) -> &'static str {
        "not-generated"
    }
    pub fn response_status(&self) -> &'static str {
        "active-only-no-internal-polarization"
    }
    pub fn force_mapping(&self) -> &'static str {
        "rigid-expansion-J-transpose; body-frame-chain-required"
    }
    pub fn gaussian_sites(&self, center: Vec3, orientation: Mat3) -> Result<Vec<GaussianCharge>> {
        self.validate()?;
        Ok((0..3)
            .map(|i| GaussianCharge {
                center,
                exponent: self.options.charge_exponents[i],
                charge: self.gaussian_charges[i],
                dipole: global(orientation, self.gaussian_body_dipoles[i]),
            })
            .collect())
    }
    pub fn validate(&self) -> Result<()> {
        self.options.validate()?;
        if self.model_version != MODEL_VERSION
            || self.source_version != SOURCE_VERSION
            || self.template_id != self.options.basis_template.id()
            || self.baseline_ionic_charge != 1.0
            || self.electron_ledger.frozen_electron_count != 8
            || self.electron_ledger.fragments.len() != 1
            || self.electron_ledger.fragments[0].effective_ionic_charge != 1
            || self.source.multiplicity != 1
            || self.electron_ledger.full_total_charge != i64::from(self.source.charge)
            || self
                .gaussian_charges
                .iter()
                .chain(self.gaussian_body_dipoles.iter().flatten())
                .chain(&self.local_amplitudes)
                .chain(&self.nonlocal_coefficients)
                .chain(std::iter::once(&self.internal_energy_reference))
                .any(|x| !x.is_finite())
            || self.gaussian_charges.iter().sum::<f64>().abs()
                > 64.0
                    * f64::EPSILON
                    * self
                        .gaussian_charges
                        .iter()
                        .map(|x| x.abs())
                        .sum::<f64>()
                        .max(1.0)
        {
            return invalid("generated record version/template, electron ledger, neutrality or coefficient mismatch");
        }
        let ledger = ElectronLedger::new(
            &self.source.atoms.iter().map(|a| a.z).collect::<Vec<_>>(),
            self.electron_ledger.full_electron_count,
            &[FrozenFragmentSpec {
                explicit_atoms: self.spec.explicit_atoms.clone(),
                frozen_electrons: 8,
            }],
        )?;
        if ledger != self.electron_ledger {
            return invalid("generated record electron ledger is inconsistent with source");
        }
        RigidExpansion::methyl(&self.source, self.spec.clone())?;
        if self.geometry_map.fragment_spec() != Some(&self.spec)
            || format!("{:?}", self.geometry_map.reference_molecule())
                != format!("{:?}", self.source)
        {
            return invalid("generated record geometry map does not match source/spec");
        }
        let full_ao = MolBasis::build(&self.source, &data::load("sto-3g")?)?.nao_sph;
        let mut atoms: Vec<_> = self
            .spec
            .retained_atoms
            .iter()
            .map(|&i| self.source.atoms[i].clone())
            .collect();
        atoms.push(Atom {
            z: 1,
            position: self.source.atoms[self.spec.port.fragment_atom].position,
        });
        let reduced_ao = build_reduction_basis(
            &Molecule::new(atoms),
            self.spec.retained_atoms.len(),
            self.options.basis_template,
        )?
        .nao_sph;
        let d = &self.diagnostics;
        if d.full_ao_count != full_ao
            || d.reduced_ao_count != reduced_ao
            || self.electron_ledger.active_electron_count == 0
            || self.electron_ledger.active_electron_count % 2 != 0
            || self.electron_ledger.active_electron_count > 2 * reduced_ao
            || [
                d.full_reference_energy,
                d.projected_reference_energy_without_constant,
                d.reference_energy_reconstruction_error,
                d.active_trace,
                d.frozen_trace,
                d.orbital_reconstruction_error,
                d.active_frozen_overlap,
                d.target_reference_pauli_expectation,
                d.sigma_charge_defect,
                d.sigma_dipole_defect,
            ]
            .iter()
            .any(|x| !x.is_finite())
            || (d.active_trace - self.electron_ledger.active_electron_count as f64).abs() > 1e-8
            || (d.frozen_trace - 8.0).abs() > 1e-8
            || (d.projected_reference_energy_without_constant + self.internal_energy_reference
                - d.full_reference_energy)
                .abs()
                > 1e-9
        {
            return invalid("generated reference diagnostic/count/energy identity mismatch");
        }
        if self.construction_fingerprint != fingerprint(&self.source, &self.spec, &self.options) {
            return invalid("generated record construction fingerprint mismatch");
        }
        Ok(())
    }
}

/// Ordinary full reference is evaluated once. Every coefficient below is a
/// common operator projection at its fixed active density, never a curve fit.
pub fn generate_methyl(
    source: &Molecule,
    spec: FragmentSpec,
    options: GenerationOptions,
) -> Result<ReducedFragment> {
    options.validate()?;
    let map = RigidExpansion::methyl(source, spec.clone())?;
    let hydrogens: Vec<_> = spec
        .explicit_atoms
        .iter()
        .copied()
        .filter(|i| *i != spec.port.fragment_atom)
        .collect();
    let hydrogens: [usize; 3] = hydrogens.try_into().map_err(|_| {
        DftError::InvalidInput("Me must contain exactly three internal hydrogens".into())
    })?;
    let basis = MolBasis::build(source, &data::load("sto-3g")?)?;
    let grid = MolecularGrid::build(
        source,
        &GridOptions {
            level: options.grid_level,
            weight_cutoff: 0.0,
            ..GridOptions::default()
        },
    )?;
    let functional = resolve(&FunctionalSpec::Combined(options.functional.clone()))?;
    let scf_options = ScfOptions {
        guess: Guess::Core,
        max_iter: 300,
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-11,
        eri_mode: EriMode::InCore { limit_mb: 64 },
        ..ScfOptions::default()
    };
    let full = run_scf(source, &basis, &scf_options, &functional, Some(&grid))?;
    let selection = extract_methyl_frozen_space(
        source,
        &basis,
        &full,
        MethylBoundary {
            carbon: spec.port.fragment_atom,
            hydrogens,
            port_atom: spec.port.host_atom,
        },
        &ExtractionOptions::default(),
    )?;
    let ledger = ElectronLedger::new(
        &source.atoms.iter().map(|a| a.z).collect::<Vec<_>>(),
        selection.full_electrons,
        &[FrozenFragmentSpec {
            explicit_atoms: spec.explicit_atoms.clone(),
            frozen_electrons: 8,
        }],
    )?;
    let model =
        FrozenReferenceModel::new(source, &basis, &grid, &functional, &scf_options, &selection)?;
    let target = model.static_residual_target(options.pauli_level)?;
    let pa = &selection.active_reference_density;
    let x = metric(model.evaluator().overlap())?;
    let kin = map
        .body_frame_kinematics(&map.reference_coordinates())?
        .ok_or_else(|| {
            DftError::InvalidInput("methyl generation requires a nonidentity body frame".into())
        })?;
    let center = kin.center;
    let u = kin.orientation;
    // Complete sigma = ordinary point ion +1 plus a neutral correction.
    let mut zpoint = vec![0.0; source.len()];
    zpoint[spec.port.fragment_atom] = 1.0;
    let point = nuclear_matrix(&basis, source, Some(&zpoint));
    let sites: Vec<_> = options
        .charge_exponents
        .iter()
        .map(|&exponent| GaussianCharge {
            center,
            exponent,
            charge: 0.0,
            dipole: Vec3::zero(),
        })
        .collect();
    let charge = charge_operator(&basis, source.len(), &sites)?;
    let mut sigma_dictionary = Vec::new();
    for i in 0..3 {
        sigma_dictionary.push(charge.charge_operators[i].clone());
        for axis in 0..3 {
            let mut m = Matrix::zeros(basis.nao_sph, basis.nao_sph);
            for cart in 0..3 {
                add(
                    &mut m,
                    &charge.dipole_operators[i][cart],
                    u.col[axis].get(cart),
                );
            }
            sigma_dictionary.push(m);
        }
    }
    let mut electrostatic = target.removed_nuclear.clone();
    add(&mut electrostatic, &target.frozen_hartree, 1.0);
    add(&mut electrostatic, &point, -1.0);
    let exact_dipole = body(u, model.frozen_core_dipole(center)?);
    let mut constraints = Matrix::zeros(4, 12);
    for i in 0..3 {
        constraints[(0, 4 * i)] = 1.0;
        for axis in 0..3 {
            constraints[(axis + 1, 4 * i + axis + 1)] = 1.0;
        }
    }
    let probe_targets = model.frozen_core_potential(&options.electrostatic_points)?;
    let mut esp_targets = Vec::new();
    let mut esp_features = Vec::new();
    if !probe_targets.is_empty() {
        for (k, &p) in options.electrostatic_points.iter().enumerate() {
            let r = (p - center).norm();
            if r < 1e-12 {
                return invalid("construction ESP probe coincides with point ion");
            }
            esp_targets.push(probe_targets[k] - 1.0 / r);
        }
        let mut columns = Vec::new();
        for i in 0..3 {
            for axis in 0..4 {
                let unit = GaussianCharge {
                    center,
                    exponent: options.charge_exponents[i],
                    charge: if axis == 0 { 1.0 } else { 0.0 },
                    dipole: if axis == 0 {
                        Vec3::zero()
                    } else {
                        u.col[axis - 1]
                    },
                };
                columns.push(
                    crate::pseudo_sub::reduction_kernels::electrostatic_potential(
                        &[unit],
                        &options.electrostatic_points,
                    )?,
                );
            }
        }
        for k in 0..esp_targets.len() {
            esp_features.push(columns.iter().map(|c| c[k]).collect());
        }
    }
    let sigma = project(
        electrostatic,
        sigma_dictionary.clone(),
        &x,
        "sigma",
        constraints,
        vec![0.0, exact_dipole[0], exact_dipole[1], exact_dipole[2]],
        options.regularization,
        if esp_targets.is_empty() {
            None
        } else {
            Some((&esp_targets, &esp_features, options.electrostatic_weight))
        },
    )?;
    let mut sigma_matrix = point.clone();
    for (m, &c) in sigma_dictionary.iter().zip(&sigma.coefficients) {
        add(&mut sigma_matrix, m, c);
    }
    let dictionary = electronic_dictionary(&basis, source.len(), center, u, &options)?;
    let mut residual = target.total.clone();
    add(&mut residual, &sigma_matrix, -1.0);
    let (fit_target, fit_dictionary, fit_weight) = match options.projection_rule {
        ProjectionRule::FullAoMetric => (residual.clone(), dictionary.clone(), x.clone()),
        ProjectionRule::PhysicalSupportPauliNormalized => {
            let support = SupportProjection::new(
                model.complement(),
                &selection.frozen_orbitals,
                &selection.full_reference_fock,
                model.evaluator().overlap(),
                options.pauli_level,
            )?;
            (
                support.transform_operator(&residual)?,
                support.transform_dictionary(&dictionary)?,
                support.metric_weight.clone(),
            )
        }
        ProjectionRule::ReducedAoMetric => {
            let mut atoms: Vec<_> = spec
                .retained_atoms
                .iter()
                .map(|&i| source.atoms[i].clone())
                .collect();
            atoms.push(Atom {
                z: 1,
                position: center,
            });
            let reduced = Molecule::new(atoms);
            let reduced_basis =
                build_reduction_basis(&reduced, spec.retained_atoms.len(), options.basis_template)?;
            let cross = cross_basis_overlap(&basis, &reduced_basis)?;
            let transfer = solve_linear_matrix(model.evaluator().overlap(), &cross)?;
            let reduced_overlap = crate::integrals::overlap_matrix(&reduced_basis);
            // Carbon core/valence subsets are exactly contained in the source
            // AO span. Refuse a hidden projection loss or change of basis rank.
            let captured =
                transfer.transpose_matmul_seq(&model.evaluator().overlap().matmul_seq(&transfer));
            if captured.max_abs_diff(&reduced_overlap) > 1e-10 {
                return invalid(
                    "reduced-AO projection does not preserve the declared common AO metric",
                );
            }
            (
                transfer.transpose_matmul_seq(&residual.matmul_seq(&transfer)),
                electronic_dictionary(&reduced_basis, reduced.len(), center, u, &options)?,
                metric(&reduced_overlap)?,
            )
        }
    };
    let electronic = project(
        fit_target,
        fit_dictionary,
        &fit_weight,
        "electronic",
        Matrix::zeros(0, 39),
        Vec::new(),
        options.regularization,
        None,
    )?;
    let mut h = kinetic_matrix(&basis);
    let mut zret = vec![0.0; source.len()];
    for &i in &spec.retained_atoms {
        zret[i] = f64::from(source.atoms[i].z);
    }
    add(&mut h, &nuclear_matrix(&basis, source, Some(&zret)), 1.0);
    add(&mut h, &sigma_matrix, 1.0);
    for (m, &c) in dictionary.iter().zip(&electronic.coefficients) {
        add(&mut h, m, c);
    }
    let mut fitted_sites = sites;
    for i in 0..3 {
        fitted_sites[i].charge = sigma.coefficients[4 * i];
        fitted_sites[i].dipole = global(
            u,
            std::array::from_fn(|a| sigma.coefficients[4 * i + a + 1]),
        );
    }
    let retained: Vec<_> = spec
        .retained_atoms
        .iter()
        .map(|&i| (source.atoms[i].position, f64::from(source.atoms[i].z)))
        .collect();
    let mut core = core_interactions(&fitted_sites, &retained)?.energy;
    for (i, &(ri, zi)) in retained.iter().enumerate() {
        core += zi / (ri - center).norm();
        for &(rj, zj) in &retained[..i] {
            core += zi * zj / (ri - rj).norm();
        }
    }
    let half = scaled(pa, 0.5);
    let engine = crate::scf::JkEngine::from_options(source, &basis, &scf_options)?;
    let (ja, _) = engine.build(&basis, pa, [&half, &half], false)?;
    let xc = xc_contribution(&functional, &basis, &grid, [&half, &half])?;
    let without = pa.frobenius_dot(&h) + 0.5 * pa.frobenius_dot(&ja) + xc.exc + core;
    let e0 = selection.full_reference_energy - without;
    let mut reduced_atoms: Vec<_> = spec
        .retained_atoms
        .iter()
        .map(|&i| source.atoms[i].clone())
        .collect();
    reduced_atoms.push(Atom {
        z: 1,
        position: center,
    });
    let reduced_ao_count = build_reduction_basis(
        &Molecule::new(reduced_atoms),
        spec.retained_atoms.len(),
        options.basis_template,
    )?
    .nao_sph;
    let record = ReducedFragment {
        model_version: MODEL_VERSION.into(),
        source_version: SOURCE_VERSION.into(),
        template_id: options.basis_template.id().into(),
        source: source.clone(),
        spec: spec.clone(),
        geometry_map: map,
        electron_ledger: ledger,
        options: options.clone(),
        baseline_ionic_charge: 1.0,
        gaussian_charges: std::array::from_fn(|i| sigma.coefficients[4 * i]),
        gaussian_body_dipoles: std::array::from_fn(|i| {
            std::array::from_fn(|a| sigma.coefficients[4 * i + a + 1])
        }),
        local_amplitudes: std::array::from_fn(|i| electronic.coefficients[i]),
        nonlocal_coefficients: std::array::from_fn(|i| electronic.coefficients[i + 3]),
        internal_energy_reference: e0,
        diagnostics: GenerationDiagnostics {
            full_ao_count: basis.nao_sph,
            reduced_ao_count,
            full_reference_energy: selection.full_reference_energy,
            projected_reference_energy_without_constant: without,
            reference_energy_reconstruction_error: (without + e0 - selection.full_reference_energy)
                .abs(),
            active_trace: selection.diagnostics.active_electron_count,
            frozen_trace: selection.diagnostics.frozen_electron_count,
            orbital_reconstruction_error: selection.diagnostics.occupied_reconstruction_error,
            active_frozen_overlap: selection.diagnostics.active_frozen_overlap,
            target_reference_pauli_expectation: target.frozen_pauli_energy,
            sigma_charge_defect: sigma.coefficients.iter().step_by(4).sum::<f64>().abs(),
            sigma_dipole_defect: (0..3)
                .map(|a| {
                    (sigma
                        .coefficients
                        .iter()
                        .skip(a + 1)
                        .step_by(4)
                        .sum::<f64>()
                        - exact_dipole[a])
                        .abs()
                })
                .fold(0.0, f64::max),
            sigma_projection: ProjectionReport::from(&sigma),
            electronic_projection: ProjectionReport::from(&electronic),
        },
        construction_fingerprint: fingerprint(source, &spec, &options),
    };
    record.validate()?;
    Ok(record)
}

fn fingerprint(source: &Molecule, spec: &FragmentSpec, options: &GenerationOptions) -> String {
    // Preserve the original generation identity for already saved H1s models.
    // The explicit top-level template ID records the discrete AO alternative.
    let legacy_options = format!(
        "GenerationOptions {{ functional: {:?}, grid_level: {:?}, local_exponents: {:?}, charge_exponents: {:?}, s_exponents: {:?}, p_exponents: {:?}, pauli_level: {:?}, regularization: {:?}, electrostatic_points: {:?}, electrostatic_weight: {:?} }}",
        options.functional, options.grid_level, options.local_exponents,
        options.charge_exponents, options.s_exponents, options.p_exponents,
        options.pauli_level, options.regularization, options.electrostatic_points,
        options.electrostatic_weight,
    );
    let id = options.basis_template.id();
    let mut text =
        format!("{source:?}|{spec:?}|{legacy_options}|{SOURCE_VERSION}|{MODEL_VERSION}|{id}");
    if options.projection_rule != ProjectionRule::FullAoMetric {
        text.push('|');
        text.push_str(options.projection_rule.id());
    }
    let mut hash = 0xcbf29ce484222325u64;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

// The existing common-driver value tree supplies JSON; no serialization
// dependency or implicit regeneration on load is introduced.
fn numbers(values: impl IntoIterator<Item = f64>) -> Value {
    Value::Array(values.into_iter().map(Value::Float).collect())
}
fn indices(values: impl IntoIterator<Item = usize>) -> Value {
    Value::Array(values.into_iter().map(|v| Value::Int(v as i64)).collect())
}
fn object(entries: Vec<(&str, Value)>) -> Value {
    Value::Object(entries.into_iter().map(|(k, v)| (k.into(), v)).collect())
}
fn required<'a>(value: &'a Value, key: &str) -> Result<&'a Value> {
    value
        .get(key)
        .ok_or_else(|| DftError::InvalidInput(format!("missing reduced model field {key}")))
}
fn array(value: &Value) -> Result<&[Value]> {
    match value {
        Value::Array(a) => Ok(a),
        _ => invalid("model array required"),
    }
}
fn string(value: &Value) -> Result<String> {
    value
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| DftError::InvalidInput("model string required".into()))
}
fn number(value: &Value) -> Result<f64> {
    match value.as_f64() {
        Some(x) if x.is_finite() => Ok(x),
        _ => invalid("finite model number required"),
    }
}
fn index(value: &Value) -> Result<usize> {
    value
        .as_i64()
        .and_then(|x| usize::try_from(x).ok())
        .ok_or_else(|| DftError::InvalidInput("nonnegative model integer required".into()))
}
fn floats<const N: usize>(value: &Value) -> Result<[f64; N]> {
    array(value)?
        .iter()
        .map(number)
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| {
            DftError::InvalidInput(format!("model array must contain exactly {N} entries"))
        })
}
fn index_vec(value: &Value) -> Result<Vec<usize>> {
    array(value)?.iter().map(index).collect()
}
fn point(value: &Value) -> Result<Vec3> {
    let p = floats::<3>(value)?;
    Ok(Vec3::new(p[0], p[1], p[2]))
}
fn point_value(p: Vec3) -> Value {
    numbers([p.x, p.y, p.z])
}
fn optional_number(value: Option<f64>) -> Value {
    value.map(Value::Float).unwrap_or(Value::Null)
}
fn load_optional_number(value: &Value) -> Result<Option<f64>> {
    match value {
        Value::Null => Ok(None),
        _ => number(value).map(Some),
    }
}

impl ProjectionReport {
    fn value(&self) -> Value {
        object(vec![
            (
                "ranks",
                indices([self.data_rank, self.free_parameters, self.regularized_rank]),
            ),
            (
                "data_condition",
                optional_number(self.data_condition_number),
            ),
            (
                "regularized_condition",
                optional_number(self.regularized_condition_number),
            ),
            (
                "errors",
                numbers([
                    self.weighted_residual_squared,
                    self.regularization_penalty,
                    self.constraint_max_residual,
                    self.projected_stationarity_residual,
                ]),
            ),
        ])
    }
    fn load(value: &Value) -> Result<Self> {
        let ranks = index_vec(required(value, "ranks")?)?;
        let e = floats::<4>(required(value, "errors")?)?;
        if ranks.len() != 3
            || ranks[0] > ranks[1]
            || ranks[2] > ranks[1]
            || e.iter().any(|x| *x < 0.0)
        {
            return invalid("invalid projection diagnostic dimensions or signs");
        }
        Ok(Self {
            data_rank: ranks[0],
            free_parameters: ranks[1],
            regularized_rank: ranks[2],
            data_condition_number: load_optional_number(required(value, "data_condition")?)?,
            regularized_condition_number: load_optional_number(required(
                value,
                "regularized_condition",
            )?)?,
            weighted_residual_squared: e[0],
            regularization_penalty: e[1],
            constraint_max_residual: e[2],
            projected_stationarity_residual: e[3],
        })
    }
}

impl ReducedFragment {
    pub fn to_value(&self) -> Result<Value> {
        self.validate()?;
        let opts = &self.options;
        let d = &self.diagnostics;
        Ok(object(vec![
            ("model_version", Value::Str(self.model_version.clone())),
            ("source_version", Value::Str(self.source_version.clone())),
            ("template_id", Value::Str(self.template_id.clone())),
            ("status", Value::Str(self.candidate_status().into())),
            ("shape_status", Value::Str(self.shape_status().into())),
            ("response_status", Value::Str(self.response_status().into())),
            ("force_mapping", Value::Str(self.force_mapping().into())),
            (
                "fingerprint",
                Value::Str(self.construction_fingerprint.clone()),
            ),
            (
                "source_atoms",
                Value::Array(
                    self.source
                        .atoms
                        .iter()
                        .map(|a| {
                            object(vec![
                                ("z", Value::Int(i64::from(a.z))),
                                ("position", point_value(a.position)),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("source_charge", Value::Int(i64::from(self.source.charge))),
            (
                "source_multiplicity",
                Value::Int(self.source.multiplicity as i64),
            ),
            (
                "explicit_atoms",
                indices(self.spec.explicit_atoms.iter().copied()),
            ),
            (
                "retained_atoms",
                indices(self.spec.retained_atoms.iter().copied()),
            ),
            (
                "port",
                indices([
                    self.spec.port.fragment_atom,
                    self.spec.port.host_atom,
                    self.spec.port.reference_atom,
                ]),
            ),
            (
                "single_bonds",
                Value::Array(
                    self.spec
                        .single_bonds
                        .iter()
                        .map(|b| indices(b.iter().copied()))
                        .collect(),
                ),
            ),
            (
                "electronic_centers",
                Value::Int(self.spec.electronic_centers as i64),
            ),
            (
                "options",
                object(vec![
                    ("functional", Value::Str(opts.functional.clone())),
                    (
                        "projection_rule",
                        Value::Str(opts.projection_rule.id().into()),
                    ),
                    ("grid_level", Value::Int(opts.grid_level as i64)),
                    ("local_exponents", numbers(opts.local_exponents)),
                    ("charge_exponents", numbers(opts.charge_exponents)),
                    ("s_exponents", numbers(opts.s_exponents)),
                    ("p_exponents", numbers(opts.p_exponents)),
                    ("pauli_level", Value::Float(opts.pauli_level)),
                    ("regularization", Value::Float(opts.regularization)),
                    (
                        "electrostatic_points",
                        Value::Array(
                            opts.electrostatic_points
                                .iter()
                                .copied()
                                .map(point_value)
                                .collect(),
                        ),
                    ),
                    (
                        "electrostatic_weight",
                        Value::Float(opts.electrostatic_weight),
                    ),
                ]),
            ),
            (
                "ledger",
                indices([
                    self.electron_ledger.full_electron_count,
                    self.electron_ledger.frozen_electron_count,
                    self.electron_ledger.active_electron_count,
                ]),
            ),
            (
                "baseline_ionic_charge",
                Value::Float(self.baseline_ionic_charge),
            ),
            ("gaussian_charges", numbers(self.gaussian_charges)),
            (
                "gaussian_body_dipoles",
                Value::Array(
                    self.gaussian_body_dipoles
                        .iter()
                        .map(|p| numbers(*p))
                        .collect(),
                ),
            ),
            ("local_amplitudes", numbers(self.local_amplitudes)),
            ("nonlocal_coefficients", numbers(self.nonlocal_coefficients)),
            (
                "internal_energy_reference",
                Value::Float(self.internal_energy_reference),
            ),
            (
                "diagnostics",
                object(vec![
                    ("ao_counts", indices([d.full_ao_count, d.reduced_ao_count])),
                    (
                        "values",
                        numbers([
                            d.full_reference_energy,
                            d.projected_reference_energy_without_constant,
                            d.reference_energy_reconstruction_error,
                            d.active_trace,
                            d.frozen_trace,
                            d.orbital_reconstruction_error,
                            d.active_frozen_overlap,
                            d.target_reference_pauli_expectation,
                            d.sigma_charge_defect,
                            d.sigma_dipole_defect,
                        ]),
                    ),
                    ("sigma_projection", d.sigma_projection.value()),
                    ("electronic_projection", d.electronic_projection.value()),
                ]),
            ),
        ]))
    }
    pub fn to_json(&self) -> Result<String> {
        Ok(self.to_value()?.to_json_pretty())
    }
    pub fn from_json(text: &str) -> Result<Self> {
        Self::from_value(&Value::parse_json(text)?)
    }
    pub fn from_value(v: &Value) -> Result<Self> {
        use crate::pseudo_sub::reduction_geometry::AttachmentPort;
        let model_version = string(required(v, "model_version")?)?;
        let source_version = string(required(v, "source_version")?)?;
        let template_id = string(required(v, "template_id")?)?;
        if model_version != MODEL_VERSION
            || source_version != SOURCE_VERSION
            || OneCenterBasisTemplate::from_id(&template_id).is_err()
            || string(required(v, "status")?)? != "candidate-not-chemically-validated"
            || string(required(v, "shape_status")?)? != "not-generated"
            || string(required(v, "response_status")?)? != "active-only-no-internal-polarization"
            || string(required(v, "force_mapping")?)?
                != "rigid-expansion-J-transpose; body-frame-chain-required"
        {
            return invalid("unsupported reduced model version, template or capability claim");
        }
        let atoms = array(required(v, "source_atoms")?)?
            .iter()
            .map(|a| {
                Ok(Atom {
                    z: u8::try_from(index(required(a, "z")?)?)
                        .map_err(|_| DftError::InvalidInput("atomic label out of range".into()))?,
                    position: point(required(a, "position")?)?,
                })
            })
            .collect::<Result<_>>()?;
        let mut source = Molecule::new(atoms);
        source.charge = i32::try_from(
            required(v, "source_charge")?
                .as_i64()
                .ok_or_else(|| DftError::InvalidInput("source charge integer required".into()))?,
        )
        .map_err(|_| DftError::InvalidInput("source charge overflow".into()))?;
        source.multiplicity = index(required(v, "source_multiplicity")?)?;
        let port = index_vec(required(v, "port")?)?;
        if port.len() != 3 {
            return invalid("three port anchors required");
        }
        let single_bonds = array(required(v, "single_bonds")?)?
            .iter()
            .map(|a| {
                index_vec(a)?
                    .try_into()
                    .map_err(|_| DftError::InvalidInput("bond requires two indices".into()))
            })
            .collect::<Result<_>>()?;
        let spec = FragmentSpec {
            explicit_atoms: index_vec(required(v, "explicit_atoms")?)?,
            retained_atoms: index_vec(required(v, "retained_atoms")?)?,
            port: AttachmentPort {
                fragment_atom: port[0],
                host_atom: port[1],
                reference_atom: port[2],
            },
            single_bonds,
            electronic_centers: index(required(v, "electronic_centers")?)?,
        };
        let o = required(v, "options")?;
        let options = GenerationOptions {
            basis_template: OneCenterBasisTemplate::from_id(&template_id)?,
            projection_rule: match o.get("projection_rule") {
                Some(value) => ProjectionRule::from_id(&string(value)?)?,
                None => ProjectionRule::FullAoMetric,
            },
            functional: string(required(o, "functional")?)?,
            grid_level: index(required(o, "grid_level")?)?,
            local_exponents: floats(required(o, "local_exponents")?)?,
            charge_exponents: floats(required(o, "charge_exponents")?)?,
            s_exponents: floats(required(o, "s_exponents")?)?,
            p_exponents: floats(required(o, "p_exponents")?)?,
            pauli_level: number(required(o, "pauli_level")?)?,
            regularization: number(required(o, "regularization")?)?,
            electrostatic_points: array(required(o, "electrostatic_points")?)?
                .iter()
                .map(point)
                .collect::<Result<_>>()?,
            electrostatic_weight: number(required(o, "electrostatic_weight")?)?,
        };
        let counts = index_vec(required(v, "ledger")?)?;
        if counts.len() != 3 {
            return invalid("three electron counts required");
        }
        let electron_ledger = ElectronLedger::new(
            &source.atoms.iter().map(|a| a.z).collect::<Vec<_>>(),
            counts[0],
            &[FrozenFragmentSpec {
                explicit_atoms: spec.explicit_atoms.clone(),
                frozen_electrons: counts[1],
            }],
        )?;
        if counts[2] != electron_ledger.active_electron_count {
            return invalid("saved active electron count mismatch");
        }
        let dipoles = array(required(v, "gaussian_body_dipoles")?)?
            .iter()
            .map(floats::<3>)
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| DftError::InvalidInput("three body dipoles required".into()))?;
        let d = required(v, "diagnostics")?;
        let ao = index_vec(required(d, "ao_counts")?)?;
        let e = floats::<10>(required(d, "values")?)?;
        if ao.len() != 2 {
            return invalid("two AO counts required");
        }
        let geometry_map = RigidExpansion::methyl(&source, spec.clone())?;
        let result = Self {
            model_version,
            source_version,
            template_id,
            source,
            spec,
            geometry_map,
            electron_ledger,
            options,
            baseline_ionic_charge: number(required(v, "baseline_ionic_charge")?)?,
            gaussian_charges: floats(required(v, "gaussian_charges")?)?,
            gaussian_body_dipoles: dipoles,
            local_amplitudes: floats(required(v, "local_amplitudes")?)?,
            nonlocal_coefficients: floats(required(v, "nonlocal_coefficients")?)?,
            internal_energy_reference: number(required(v, "internal_energy_reference")?)?,
            construction_fingerprint: string(required(v, "fingerprint")?)?,
            diagnostics: GenerationDiagnostics {
                full_ao_count: ao[0],
                reduced_ao_count: ao[1],
                full_reference_energy: e[0],
                projected_reference_energy_without_constant: e[1],
                reference_energy_reconstruction_error: e[2],
                active_trace: e[3],
                frozen_trace: e[4],
                orbital_reconstruction_error: e[5],
                active_frozen_overlap: e[6],
                target_reference_pauli_expectation: e[7],
                sigma_charge_defect: e[8],
                sigma_dipole_defect: e[9],
                sigma_projection: ProjectionReport::load(required(d, "sigma_projection")?)?,
                electronic_projection: ProjectionReport::load(required(
                    d,
                    "electronic_projection",
                )?)?,
            },
        };
        result.validate()?;
        Ok(result)
    }
}
