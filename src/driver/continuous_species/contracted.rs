// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Unregistered experimental contracted Me8 force provider.
//! Geometry/grid sites, electronic AO centers and projector centers are distinct.
//! The original labelled grid is retained to preserve the measured static model.

use super::few_center::FewCenterEvaluation;
use super::local::{LocalCenter, LocalGaussianSpec};
use super::reduction_linear::{ElectronLedger, FrozenFragmentSpec};
use super::static_core::{static_core_correction_with_auxiliary_roles, StaticCoreParameters};
use super::{ChargeChart, ChargeParameter, ContinuousSpeciesModel, Fingerprint, SpeciesFunctional};
use crate::basis::{aos, data, MolBasis};
use crate::dual::DualN;
use crate::grid::{GridOptions, MolecularGrid};
use crate::integrals::analytic_basis::{self, AnalyticShell};
use crate::integrals::overlap_matrix;
use crate::linalg::Matrix;
use crate::math::{Mat3, Vec3};
use crate::pseudo_sub::reduction::GenerationOptions;
use crate::pseudo_sub::reduction_geometry::{BodyFrameKinematics, RigidExpansion};
use crate::pseudo_sub::reduction_kernels::symmetric_projector_coefficients;
use crate::pseudo_sub::reduction_roles::{layout, CenterTemplate};
use crate::scf::{EriMode, Guess, ScfOptions};
use crate::system::Molecule;
use crate::{DftError, Result};

pub const OPERATOR_ID: &str = "contracted-me8-body-fitted-gaussian3-local3-full-grid-v1";

fn invalid<T>(s: &str) -> Result<T> {
    Err(DftError::InvalidInput(s.into()))
}
fn unsupported<T>(s: &str) -> Result<T> {
    Err(DftError::Unsupported { feature: s.into(), missing: "this unregistered provider implements only fixed fitted-Gaussian electrostatics, static contracted Me8 operators and first nuclear derivatives; no exact frozen-density electrostatics or internal response".into() })
}
fn finite(v: Vec3) -> bool {
    v.to_array().iter().all(|x| x.is_finite())
}
fn add(a: &mut Matrix, b: &Matrix, scale: f64) {
    assert_eq!((a.rows, a.cols), (b.rows, b.cols));
    for (x, y) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *x += scale * y;
    }
}
fn functional(s: &str) -> Result<SpeciesFunctional> {
    match s {
        "lda" => Ok(SpeciesFunctional::Lda),
        "pbe" => Ok(SpeciesFunctional::Pbe),
        _ => unsupported("contracted static functional"),
    }
}

/// D is in the saved body gauge. Widths and all 51 coefficients stay fixed.
/// Conversion from an earlier global-gauge projection is a separate one-time
/// operation, never a per-geometry fit or a positive-semidefinite projection.
#[derive(Clone, Debug)]
pub struct ContractedCoefficients {
    pub template: CenterTemplate,
    pub operator_id: String,
    pub dictionary_id: String,
    pub functional_id: String,
    pub reference_id: String,
    pub gaussian_charges: [f64; 3],
    pub gaussian_body_dipoles: [Vec3; 3],
    pub local_amplitudes: [f64; 3],
    pub body_nonlocal_upper: [f64; 36],
    pub internal_energy_reference: f64,
}
impl ContractedCoefficients {
    pub fn validate(&self) -> Result<()> {
        if self.operator_id != OPERATOR_ID {
            return unsupported("contracted operator recipe identifier");
        }
        functional(&self.functional_id)?;
        if self.dictionary_id.trim().is_empty()
            || self.reference_id.trim().is_empty()
            || self
                .gaussian_charges
                .iter()
                .chain(&self.local_amplitudes)
                .chain(&self.body_nonlocal_upper)
                .chain(std::iter::once(&self.internal_energy_reference))
                .any(|x| !x.is_finite())
            || self.gaussian_body_dipoles.iter().any(|&v| !finite(v))
        {
            return invalid("contracted coefficients need finite values and explicit provenance");
        }
        let scale = self
            .gaussian_charges
            .iter()
            .map(|x| x.abs())
            .sum::<f64>()
            .max(1.0);
        if self.gaussian_charges.iter().sum::<f64>().abs() > 64.0 * f64::EPSILON * scale {
            return invalid(
                "contracted Gaussian correction must be neutral without renormalization",
            );
        }
        Ok(())
    }
}

/// C1s,C2s,Cp0,Cp1,Cp2,H1s,H2s,H3s: only columns2..4 rotate.
pub fn body_rotation(u: Mat3) -> Result<Matrix> {
    if u.col.iter().any(|&v| !finite(v)) {
        return invalid("contracted frame must be finite");
    }
    for i in 0..3 {
        for j in 0..3 {
            if (u.col[i].dot(u.col[j]) - if i == j { 1.0 } else { 0.0 }).abs() > 1e-10 {
                return invalid("contracted frame must be orthonormal");
            }
        }
    }
    if (u.col[0].cross(u.col[1]).dot(u.col[2]) - 1.0).abs() > 1e-10 {
        return invalid("contracted frame must be proper");
    }
    let mut result = Matrix::identity(8);
    let t = aos::cart_to_sph(1);
    for i in 0..3 {
        for j in 0..3 {
            result[(2 + i, 2 + j)] = (0..3)
                .flat_map(|a| (0..3).map(move |b| (a, b)))
                .map(|(a, b)| t[(a, i)] * u.col[b].get(a) * t[(b, j)])
                .sum();
        }
    }
    Ok(result)
}

/// Ambient derivative with respect to U_(world,local), not a second frame map.
fn rotation_component(world: usize, local: usize) -> Matrix {
    let t = aos::cart_to_sph(1);
    let mut result = Matrix::zeros(8, 8);
    for i in 0..3 {
        for j in 0..3 {
            result[(2 + i, 2 + j)] = t[(world, i)] * t[(local, j)];
        }
    }
    result
}

pub fn global_to_body_upper(
    global_upper: &[f64; 36],
    reference_orientation: Mat3,
) -> Result<[f64; 36]> {
    let global = symmetric_projector_coefficients(global_upper)?;
    let r = body_rotation(reference_orientation)?;
    let body = r.transpose_matmul_seq(&global.matmul_seq(&r));
    let mut upper = [0.0; 36];
    let mut k = 0;
    for i in 0..8 {
        for j in i..8 {
            upper[k] = body[(i, j)];
            k += 1;
        }
    }
    Ok(upper)
}

/// Four right-side projector sites. They carry no automatic ionic charge/AOs.
pub fn right_basis(full: &Molecule, map: &RigidExpansion) -> Result<(MolBasis, [usize; 4])> {
    let spec = map.fragment_spec().ok_or_else(|| {
        DftError::InvalidInput("contracted projectors need a declared methyl map".into())
    })?;
    if map.internal_offsets().len() != 3 || full.len() != map.full_atom_count() {
        return invalid("contracted Me8 needs exactly three saved ordered H offsets");
    }
    let ids = [
        spec.port.fragment_atom,
        map.internal_offsets()[0].0,
        map.internal_offsets()[1].0,
        map.internal_offsets()[2].0,
    ];
    if full.atoms[ids[0]].z != 6 || ids[1..].iter().any(|&i| full.atoms[i].z != 1) {
        return invalid("contracted right sites require the original C,H,H,H labels");
    }
    let molecule = Molecule::new(ids.iter().map(|&i| full.atoms[i].clone()).collect());
    let basis = MolBasis::build(&molecule, &data::load("sto-3g")?)?;
    if basis.nao_sph != 8
        || basis.shells.len() != 6
        || basis
            .shells
            .iter()
            .zip([0, 0, 1, 0, 0, 0])
            .any(|(s, l)| s.l != l)
        || basis
            .shells
            .iter()
            .zip([0, 1, 2, 5, 6, 7])
            .any(|(s, off)| s.sph_off != off)
    {
        return invalid("bundled contracted Me8 shell order changed");
    }
    Ok((basis, ids))
}

pub struct ContractedOverlaps {
    pub global: Matrix,
    pub body: Matrix,
    /// Partial derivatives in independent full-skeleton coordinates at fixed U.
    /// Both left-AO and right-projector motion are included exactly once.
    pub atom_derivatives: Vec<[Matrix; 3]>,
    /// Ambient U derivatives, index3*world+local; frame rows are3*local+world.
    pub orientation_derivatives: [Matrix; 9],
}

pub fn overlaps(
    left: &MolBasis,
    atoms: usize,
    right: &MolBasis,
    right_sites: [usize; 4],
    orientation: Mat3,
) -> Result<ContractedOverlaps> {
    if right.nao_sph != 8
        || right.shells.len() != 6
        || right.shells.iter().any(|s| s.atom >= 4)
        || right_sites.iter().any(|&a| a >= atoms)
        || right
            .shells
            .iter()
            .zip([0, 0, 1, 0, 0, 0])
            .any(|(s, l)| s.l != l)
        || right
            .shells
            .iter()
            .zip([0, 1, 2, 5, 6, 7])
            .any(|(s, off)| s.sph_off != off)
        || (0..4).any(|i| (0..i).any(|j| right_sites[i] == right_sites[j]))
        || left.shells.iter().any(|s| s.atom >= atoms)
        || !left.ecps.is_empty()
        || !right.ecps.is_empty()
        || left.n_core_electrons != 0
        || right.n_core_electrons != 0
    {
        return invalid("contracted overlap shell/site contract mismatch");
    }
    let r = body_rotation(orientation)?;
    let matrix = || Matrix::zeros(left.nao_sph, 8);
    let mut global = matrix();
    let mut derivatives: Vec<[Matrix; 3]> = (0..atoms)
        .map(|_| std::array::from_fn(|_| matrix()))
        .collect();
    type D = DualN<f64, 6>;
    for a in &left.shells {
        for b in &right.shells {
            let ao = AnalyticShell::from_placed(
                a,
                std::array::from_fn(|axis| D::variable(a.center.get(axis), axis)),
            )?;
            let beta = AnalyticShell::from_placed(
                b,
                std::array::from_fn(|axis| D::variable(b.center.get(axis), 3 + axis)),
            )?;
            let block = analytic_basis::overlap(&ao, &beta)?;
            for i in 0..block.shape[0] {
                for j in 0..block.shape[1] {
                    let value = block.values[i * block.shape[1] + j];
                    let row = a.sph_off + i;
                    let col = b.sph_off + j;
                    global[(row, col)] = value.re;
                    for axis in 0..3 {
                        derivatives[a.atom][axis][(row, col)] += value.im[axis];
                        derivatives[right_sites[b.atom]][axis][(row, col)] += value.im[3 + axis];
                    }
                }
            }
        }
    }
    let body = global.matmul_seq(&r);
    let atom_derivatives = derivatives
        .into_iter()
        .map(|d| d.map(|v| v.matmul_seq(&r)))
        .collect();
    let orientation_derivatives =
        std::array::from_fn(|entry| global.matmul_seq(&rotation_component(entry / 3, entry % 3)));
    Ok(ContractedOverlaps {
        global,
        body,
        atom_derivatives,
        orientation_derivatives,
    })
}

pub struct ContractedOperator {
    pub operator: Matrix,
    pub atom_derivatives: Vec<[Matrix; 3]>,
}
fn differentiated_operator(b: &Matrix, d: &Matrix, db: &Matrix) -> Matrix {
    let mut value = db.matmul_seq(d).matmul_transpose_seq(b);
    add(&mut value, &b.matmul_seq(d).matmul_transpose_seq(db), 1.0);
    value
}
impl ContractedOverlaps {
    pub fn operator(
        &self,
        body_upper: &[f64; 36],
        frame: &BodyFrameKinematics,
    ) -> Result<ContractedOperator> {
        let atoms = self.atom_derivatives.len();
        if frame.orientation_jacobian.rows != 9
            || frame.orientation_jacobian.cols != 3 * atoms
            || frame
                .orientation_jacobian
                .as_slice()
                .iter()
                .any(|x| !x.is_finite())
        {
            return invalid("contracted frame Jacobian needs full-skeleton columns");
        }
        let d = symmetric_projector_coefficients(body_upper)?;
        let operator = self.body.matmul_seq(&d).matmul_transpose_seq(&self.body);
        let mut atom_derivatives = Vec::with_capacity(atoms);
        for atom in 0..atoms {
            let mut value: [Matrix; 3] =
                std::array::from_fn(|_| Matrix::zeros(self.body.rows, self.body.rows));
            for axis in 0..3 {
                let mut db = self.atom_derivatives[atom][axis].clone();
                for world in 0..3 {
                    for local in 0..3 {
                        add(
                            &mut db,
                            &self.orientation_derivatives[3 * world + local],
                            frame.orientation_jacobian[(3 * local + world, 3 * atom + axis)],
                        );
                    }
                }
                value[axis] = differentiated_operator(&self.body, &d, &db);
            }
            atom_derivatives.push(value);
        }
        Ok(ContractedOperator {
            operator,
            atom_derivatives,
        })
    }
}

fn full_frame(map: &RigidExpansion, coordinates: &[Vec3]) -> Result<BodyFrameKinematics> {
    let frame = map
        .body_frame_kinematics(coordinates)?
        .ok_or_else(|| DftError::InvalidInput("missing contracted body frame".into()))?;
    let mut orientation_jacobian = Matrix::zeros(9, 3 * map.full_atom_count());
    let mut center_jacobian = Matrix::zeros(3, 3 * map.full_atom_count());
    let main = map.fragment_spec().unwrap().port.fragment_atom;
    let ids: Vec<_> = map
        .retained_atoms()
        .iter()
        .copied()
        .chain(std::iter::once(main))
        .collect();
    for (primary, &full) in ids.iter().enumerate() {
        for axis in 0..3 {
            for row in 0..9 {
                orientation_jacobian[(row, 3 * full + axis)] =
                    frame.orientation_jacobian[(row, 3 * primary + axis)];
            }
            for row in 0..3 {
                center_jacobian[(row, 3 * full + axis)] =
                    frame.center_jacobian[(row, 3 * primary + axis)];
            }
        }
    }
    Ok(BodyFrameKinematics {
        center: frame.center,
        orientation: frame.orientation,
        orientation_jacobian,
        center_jacobian,
    })
}

/// Identifies the actual contracted radial data, ordered body sites and grid.
pub fn dictionary_id(
    source: &Molecule,
    map: &RigidExpansion,
    template: CenterTemplate,
    dictionary: &GenerationOptions,
) -> Result<String> {
    let standard = GenerationOptions::default();
    functional(&dictionary.functional)?;
    if dictionary.local_exponents != standard.local_exponents
        || dictionary.charge_exponents != standard.charge_exponents
        || dictionary.s_exponents != standard.s_exponents
        || dictionary.p_exponents != standard.p_exponents
        || dictionary.grid_level != 1
    {
        return unsupported("contracted dictionary requires the fixed common widths and original full-labelled level1 grid");
    }
    if source.len() != map.full_atom_count()
        || source
            .atoms
            .iter()
            .zip(&map.reference_molecule().atoms)
            .any(|(a, b)| a.z != b.z || a.position != b.position)
    {
        return invalid("contracted dictionary source must equal the saved map reference");
    }
    let (right, ids) = right_basis(source, map)?;
    let mut id = Fingerprint::new();
    id.string(OPERATOR_ID);
    id.string(template.id());
    id.string(&dictionary.functional);
    id.string("C1s-C2s-Cp012-H1s-H2s-H3s;R=TtUT-only2to4;D-upper-row-v1");
    for shell in right.shells {
        id.number(shell.atom);
        id.number(shell.l as usize);
        id.number(shell.sph_off);
        for x in shell.exps.into_iter().chain(shell.coefs) {
            id.float(x);
        }
    }
    for &a in &ids {
        id.number(a);
    }
    for &(a, offset) in map.internal_offsets() {
        id.number(a);
        id.vector(offset);
    }
    for atom in &source.atoms {
        id.number(atom.z as usize);
    }
    for x in dictionary
        .local_exponents
        .into_iter()
        .chain(dictionary.charge_exponents)
    {
        id.float(x);
    }
    let t = aos::cart_to_sph(1);
    for &x in t.as_slice() {
        id.float(x);
    }
    id.number(dictionary.grid_level);
    Ok(format!("fnv1a64:{:016x}", id.0))
}

/// Fixed-parameter static model. No ordinary full-fragment Fock is constructed
/// per iteration: only the reduced AO basis and the retained labelled grid.
#[allow(clippy::too_many_arguments)]
pub fn evaluate(
    source: &Molecule,
    map: &RigidExpansion,
    ledger: &ElectronLedger,
    dictionary: &GenerationOptions,
    coefficients: &ContractedCoefficients,
    coordinates: &[Vec3],
    field: Vec3,
    origin: Vec3,
    spin_electrons: Option<[usize; 2]>,
    request_internal_response: bool,
) -> Result<FewCenterEvaluation> {
    coefficients.validate()?;
    if request_internal_response {
        return unsupported("contracted internal frozen response");
    }
    if !finite(field) || !finite(origin) {
        return invalid("contracted field and origin must be finite");
    }
    if coefficients.functional_id != dictionary.functional
        || coefficients.dictionary_id
            != dictionary_id(source, map, coefficients.template, dictionary)?
    {
        return invalid("contracted functional/dictionary provenance mismatch");
    }
    let spec = map
        .fragment_spec()
        .ok_or_else(|| DftError::InvalidInput("contracted evaluation needs a methyl map".into()))?;
    let expected = ElectronLedger::new(
        &source.atoms.iter().map(|a| a.z).collect::<Vec<_>>(),
        source.n_electrons()?,
        &[FrozenFragmentSpec {
            explicit_atoms: spec.explicit_atoms.clone(),
            frozen_electrons: 8,
        }],
    )?;
    if &expected != ledger
        || ledger.fragments.len() != 1
        || ledger.fragments[0].effective_ionic_charge != 1
    {
        return invalid("contracted frozen8 ledger mismatch");
    }
    let active = ledger.active_electron_count;
    let spins = spin_electrons.unwrap_or([active / 2; 2]);
    if spins[0].checked_add(spins[1]) != Some(active)
        || spins[0] < spins[1]
        || source.multiplicity != spins[0] - spins[1] + 1
        || (spin_electrons.is_none() && active % 2 != 0)
    {
        return invalid("contracted fixed-spin electron sector mismatch");
    }
    let full_spins = [spins[0] + 4, spins[1] + 4];
    let expanded = map.evaluate(coordinates)?;
    let molecule = expanded.molecule;
    let atoms = molecule.len();
    let roles = layout(map, source, coordinates, coefficients.template)?;
    let mut basis = roles.basis()?;
    for shell in &mut basis.shells {
        shell.atom = roles.role_to_full_geometry[shell.atom];
    }
    let nao = basis.nao_sph;
    if spins.iter().any(|&v| v > nao) {
        return invalid("contracted occupation exceeds reduced AO space");
    }
    let own = spec.port.fragment_atom;
    let mut charges = vec![0.0; atoms];
    for &a in map.retained_atoms() {
        charges[a] = f64::from(source.atoms[a].z);
    }
    charges[own] = 1.0;
    let auxiliary: Vec<_> = charges.iter().map(|&z| z == 0.0).collect();
    let positions: Vec<_> = molecule.atoms.iter().map(|a| a.position).collect();
    let frame = full_frame(map, coordinates)?;
    let parameters = StaticCoreParameters {
        charge_exponents: dictionary.charge_exponents,
        charge_coefficients: coefficients.gaussian_charges,
        body_dipoles: coefficients.gaussian_body_dipoles,
        projector_s_exponents: dictionary.s_exponents,
        projector_p_exponents: dictionary.p_exponents,
        nonlocal_coefficients: [0.0; 36],
    };
    // Deliberate electrostatic-only reuse. The primitive nonlocal contribution
    // is identically zero; the complete contracted operator is added below.
    // This temporary extra primitive-overlap work is not claimed as a speedup.
    let mut correction = static_core_correction_with_auxiliary_roles(
        &basis,
        &positions,
        &charges,
        own,
        &frame,
        &parameters,
        origin,
        &auxiliary,
    )?;
    let (right, right_sites) = right_basis(&molecule, map)?;
    let overlaps = overlaps(&basis, atoms, &right, right_sites, frame.orientation)?;
    let nonlocal = overlaps.operator(&coefficients.body_nonlocal_upper, &frame)?;
    add(&mut correction.operator, &nonlocal.operator, 1.0);
    for a in 0..atoms {
        for axis in 0..3 {
            add(
                &mut correction.atom_derivative_matrices[a][axis],
                &nonlocal.atom_derivatives[a][axis],
                1.0,
            );
        }
    }
    correction.validate(nao, atoms)?;
    let mut permanent = correction.permanent_dipole;
    for (&q, &r) in charges.iter().zip(&positions) {
        permanent += (r - origin) * q;
    }
    let grid = MolecularGrid::build(
        &molecule,
        &GridOptions {
            level: 1,
            weight_cutoff: 0.0,
            ..GridOptions::default()
        },
    )?;
    let grid_points = grid.len();
    let overlap = overlap_matrix(&basis);
    let options = ScfOptions {
        guess: Guess::Core,
        max_iter: 300,
        e_tol: 1e-13,
        p_tol: 1e-12,
        conv_tol_grad: 1e-11,
        eri_mode: EriMode::InCore { limit_mb: 64 },
        ..ScfOptions::default()
    };
    let chart = ChargeChart {
        reference_charges: charges,
        conserve_total_charge: true,
        parameters: vec![ChargeParameter {
            name: "fixed-contracted-model".into(),
            unit: "dimensionless".into(),
            scale: 1.0,
            bounds: [-1.0, 1.0],
            charge_direction: vec![0.0; atoms],
        }],
    };
    let local: Vec<_> = (0..3)
        .map(|i| LocalGaussianSpec {
            center: LocalCenter::Atom(own),
            exponent: dictionary.local_exponents[i],
            reference_amplitude: coefficients.local_amplitudes[i],
            amplitude_direction: vec![0.0],
            reference_steric_strength: 0.0,
            steric_direction: vec![0.0],
            environment_exponents: vec![1.0; atoms],
            environment_hardness: vec![0.0; atoms],
            excluded_atoms: vec![true; atoms],
        })
        .collect();
    let model = match spin_electrons {
        None => ContinuousSpeciesModel::new_with_auxiliary_roles(
            molecule,
            basis,
            grid,
            active,
            chart,
            functional(&dictionary.functional)?,
            options,
            origin,
            auxiliary,
        ),
        Some(counts) => ContinuousSpeciesModel::new_with_spin_and_auxiliary_roles(
            molecule,
            basis,
            grid,
            counts,
            chart,
            functional(&dictionary.functional)?,
            options,
            origin,
            auxiliary,
        ),
    }?;
    let mut model = model
        .with_local_potentials(local)?
        .with_static_core_correction(correction)?
        .with_fragment_internal_energy(coefficients.internal_energy_reference)?;
    let mut identity = Fingerprint(model.fingerprints.chart);
    identity.string(OPERATOR_ID);
    identity.string(&coefficients.dictionary_id);
    identity.string(&coefficients.reference_id);
    identity.number(spins[0]);
    identity.number(spins[1]);
    model.fingerprints.chart = identity.0;
    let state = model.evaluate(&[0.0], field, None)?;
    let trace = ledger.check_active_trace(&state.total_density(), &overlap, 1e-9)?;
    let spin_trace = state.spin_density().frobenius_dot(&overlap);
    if (spin_trace - (spins[0] - spins[1]) as f64).abs() > 1e-9
        || (state.total_charge() - ledger.reduced_total_charge as f64).abs() > 1e-10
    {
        return invalid("contracted converged electron/charge ledger mismatch");
    }
    let full_gradient = model.nuclear_gradient(&state)?;
    // The one geometry pullback includes projector-site and quadrature forces.
    // There is no second role pullback and no separately added frame torque.
    let gradient = map.pullback_forces(coordinates, &full_gradient)?;
    let forces = gradient.iter().map(|&g| g * (-1.0)).collect();
    Ok(FewCenterEvaluation {
        energy: state.energy(),
        electronic_energy: state.electronic_energy(),
        classical_energy: state.classical_energy(),
        energy_without_internal_reference: state.energy_without_internal_reference(),
        classical_energy_without_internal_reference: state
            .classical_energy_without_internal_reference(),
        fragment_internal_energy: state.fragment_internal_energy(),
        gradient,
        forces,
        role_gradient: full_gradient,
        model_dipole: state.dipole,
        permanent_ionic_dipole: permanent,
        active_electronic_dipole: state.dipole - permanent,
        physical_total_charge: state.total_charge(),
        active_density_trace: trace,
        active_electrons: active,
        frozen_electrons: 8,
        full_electrons: ledger.full_electron_count,
        spin_electrons: spins,
        frozen_spin_electrons: [4, 4],
        full_spin_electrons: full_spins,
        active_spin_trace: spin_trace,
        unrestricted: state.scf.unrestricted,
        spin_status: "fixed-active-spin-sector-with-frozen-singlet-core",
        electronic_centers: coefficients.template.center_count(),
        role_sites: atoms,
        retained_nuclei: map.retained_atoms().len(),
        grid_sites: atoms,
        ao_count: nao,
        grid_points,
        projector_functions: 8,
        electrostatic_sites: 3,
        local_potential_sites: 3,
        response_variables: 0,
        overlap_condition_number: model.overlap_condition_number,
        scf_iterations: state.diagnostics.iterations,
        scf_orbital_residual: state.diagnostics.orbital_residual,
        frontier_gap: state.frontier_gap(),
        template_id: coefficients.template.id(),
        reference_id: coefficients.reference_id.clone(),
        fingerprints: model.fingerprints.clone(),
        active_model_polarizability: None,
        response_residuals: None,
        candidate_status: "candidate-not-chemically-validated",
        shape_status: "not-generated",
        response_status: "unsupported-frozen-internal-response-no-alpha-provider",
    })
}
