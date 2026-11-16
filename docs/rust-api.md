<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 ss0832 -->

# Rust API

Start with the [API guide](api-guide.md) to choose a calculation, compare
interfaces, run an example and interpret units and convergence. This manual
contains the detailed contracts and reference-specific limits.

[Common driver](#common-driver) · [Calculation table](#table-1-rust) · [Typed boundaries](#typed-physical-boundaries-and-checked-scf-states)

New molecular and periodic response calculations, shared report types, solver
controls, and analytic foundation APIs are listed in the
[aligned response API tables](rust-api.md#table-1-rust).

The Rust API uses Bohr for Cartesian inputs and Hartree for energies.
Individual observables retain their documented units, such as MHz or ppm.
XYZ readers accept Angstrom coordinates and convert them to Bohr.

```toml
[dependencies]
dft-rs = { version = "0.4.0", path = "../dft-rs" }
```

Adjust this checkout dependency to the directory containing `Cargo.toml`.
The development version does not establish availability on a crate registry.

The crate is `#![forbid(unsafe_code)]` and depends only on `libm`, `rayon`
and `faer`. The `python` feature adds `pyo3` and builds the extension module;
a Rust consumer never needs it.

Contents: [molecular SCF](#molecular-calculation) ·
[basis/ECP](#basis-sets-and-ecps) · [options](#scf-options-and-nonlocal-correlation)
· [derivatives](#derivatives) · [geometry walks](#geometry-optimization-and-transition-states)
· [external field](#external-electric-field)
· [extra one-electron operator](#extra-one-electron-operator)
· [solvation](#continuum-solvation)
· [properties](#properties-and-files) · [SAPT and response](#sapt-and-response)
· [excited states](#excited-states) · [periodic](#periodic-boundary-conditions)
· [errors and threading](#errors-and-threading)

## Common driver

`dft_rs::driver::run(feature, &System, &Controls) -> Result<Value>` runs
registered common-driver calculations; `driver::registry::features()` lists
both common-driver and dedicated entry points. The functions in
`driver::method::{MolecularMethod, prepare_molecular, run_molecular_reference}`
build the reference from named controls. See [the common driver](rust-api.md#common-calculation-driver).

Registry membership describes an entry point, not every supported reference,
functional or boundary condition. Dedicated entries require their named API;
passing their ID to `run` raises `DftError::Unsupported`. A supported nuclear
energy Hessian does not imply nuclear derivatives of hyperfine, g or ZFS
tensors. Read each property's model and acceptance limits separately.

```rust
use dft_rs::driver::{run, Controls, System, Value};

let system = System::molecule(vec![8, 1, 1], vec![[0.0, 0.0, 0.117], [0.0, 1.433, -0.931], [0.0, -1.433, -0.931]]);
let controls = Controls::from_value(&Value::object().with("xc", "pbe").with("basis", "def2-svp"))?;
let report = run("dipole", &system, &controls)?;
let debye = report.get("dipole_norm_debye").and_then(Value::as_f64);
```

### Weighted molecular XC

`Controls` accepts `xc_terms`, an array of objects with a dimensionless
`weight` and either `xc` or the pair `x` and `c`. The declared coefficients
are used directly on one common SCF reference. Molecular `energy`,
`gradient`, `hessian`, `dipole`, `molecular-multipole` and `orbitals`
accept the array for RKS, UKS, ROKS and GKS. The explicit common-orbital
entry `roks-hessian` accepts it too. A typed Rust caller can also set
`MolecularMethod::xc_terms` with
`xc::WeightedFunctionalTerm` values.

For molecular `energy` and static `quadrupole-polarizability`, a
custom double hybrid can also use `xc_terms` for its common orbital
parent. Supply `pt2_same_spin` and `pt2_opposite_spin` separately.
GKS uses a common coefficient for both spin weights and can apply a
separate semilocal or hybrid target to the resulting parent density.
All weights are dimensionless and enter the model as supplied.

A named global hybrid such as `xc="pbe0"` retains its defined full-range
exact-exchange fraction. An `xc_terms` array declares an explicit
composition. Validation of the named global hybrid and the declared
composition is tracked separately.

The report gives energy in Hartree, analytic Cartesian gradients in
Hartree/Bohr. Hessian rows and columns use `3*atom + axis` and
Hartree/Bohr². The RKS, UKS and ROKS Hessian reports also expose
`dipole_derivatives` with dipole-axis rows, Cartesian-coordinate columns
and units of elementary charge. The ROKS report includes its constrained
static polarizability in Bohr³. `scf_orbital_residual` is in Hartree.
RKS and UKS return
the maximum metric commutator element in the orbital frame. ROKS returns
the maximum active common-orbital effective Fock element, or the
closed-shell metric commutator when the open shell is empty. GKS uses
the spinor solver's physical orbital residual. See the [coverage ledger](response.md#r067-ground-state-properties-exports-relaxation-and-streaming)
for measured residuals and operator restrictions.

### Static molecular electric moments

`dipole` and `molecular-multipole` use the stationary density of the
selected reference. ROKS contracts the total density from its common
spatial orbitals. GKS contracts the real spin trace of its complex
density for the real symmetric electric AO operators. Both include
the effective ionic charges of the chosen core model.

The origin defaults to the center of mass and may be selected by name or
supplied in Bohr. Dipoles are in e·Bohr, with a Debye view. The rank-2
quadrupole in `ranks` uses e·Bohr² in Cartesian, Buckingham traceless and
Racah-normalized spherical conventions. `scf_orbital_residual` is in Hartree.
Collinear double-hybrid electric moments include the relaxed parent,
distinct-target and spin-resolved PT2 contributions. Common-orbital
and spinor correlated moments require their corresponding PT2 response.
See the [reference and acceptance scope](response.md#r060-molecular-multipoles-and-derivatives).

### Electric quadrupole polarizability

`quadrupole-polarizability` uses five Frobenius-orthonormal symmetric
traceless modes `T_b` for an electric field gradient
`G = sum_b lambda_b T_b` at the reported origin. The electronic source
is `+G:rr/2`. If `Theta` is the Buckingham quadrupole, the tensor is
`C_ab = d(Theta:T_a)/d lambda_b`, in e²·Bohr⁴/Hartree.
Equivalently, the static energy term is `-sum_ab C_ab lambda_a lambda_b/6`.
The report includes the mode matrices, raw solver residuals and
the complex real and imaginary responses at the requested frequency.
Static calculations also return `quadrupole_polarizability`. Retarded
frequencies are signed, with nonnegative damping. Imaginary-axis
frequencies are nonnegative and use zero damping.
At a real nonzero frequency, damping evaluates the retarded response
at `z = frequency + i * frequency_damping`. With a real-frequency
drive represented by `exp(-i frequency t)`, the imaginary part gives
the quadrature component of the induced moment.
For the diagonal response of an isolated molecule at zero damping
and away from resonances, that component vanishes up to numerical error.

RKS and UKS use the paired collinear solver. ROKS uses its constrained
common-orbital metric and GKS its full-spin complex kernel. Explicit
weighted XC uses those same operators. Static RKS and UKS double hybrids
differentiate the parent, distinct target when present, and noncanonical
PT2 energy through second order. All six contribution tensors use the
same modes and units as the total. The three separately reported PT2
channels are unweighted. The PT2 correction applies the recipe's
same-spin and opposite-spin coefficients. The raw residuals include the
parent solve, projector, stationarity, orbital frame, particle number
and PT2 amplitudes by Taylor order. Static GKS accepts a common PT2
coefficient and an optional distinct semilocal or hybrid target.
It returns the parent and generalized PT2 tensors, the scaled PT2
correction and any target correction in the same Buckingham modes.
The target energy is evaluated on the parent density Taylor path,
including its second-order relaxation. The spinor solver retains
spin-flip substitutions and reports directional projector,
stationarity, particle-number, frame and pair residuals. Unequal spin
coefficients, GKS nonlocal correlation, and finite-frequency GKS or
ROKS field gradients require their corresponding correlated kernels.
Finite-frequency collinear double hybrids use the
[perturbative quasi-energy response](#frequency-dependent-double-hybrid-electric-response).
The origin-dependent convention follows
the [Buckingham expansion](https://doi.org/10.1016/j.cplett.2020.137555).

### Electric multipole polarizability

The default origin is the molecular center of mass. It follows a rigid
translation of all nuclei, so the moments and the complete static and
dynamic response tensors are invariant to a change of laboratory
coordinate offset. An explicitly changed expansion center is a different
multipole convention: higher ranks mix with lower ranks and are not
individually origin independent. This physical dependence is retained.


`multipole-polarizability` uses the Racah-normalized real solid harmonics
`R_lm` of `molecular-multipole`, in the order `0,+1,-1,+2,-2,...`.
Set `multipole_rank` to 1 through 5. Rank 3 is the default. The molecular
moment is `M_lm = sum_A Z_A R_lm(r_A-o) - Tr(P R_lm)` at origin `o`.
The external interaction is `-sum_m lambda_lm M_lm`. Its electronic
Hamiltonian source is `+R_lm` and the clamped ionic term is linear in
the field. The response is the derivative of the Racah moment itself.
The units are e²·Bohr^(2l)/Hartree. The separate Buckingham quadrupole
command retains its own projected field-gradient normalization.

The static report gives `multipole_polarizability`. Static and dynamic
reports give `multipole_polarizability_real` and
`multipole_polarizability_imag`, origin, signed frequency in Hartree,
mode order, SCF orbital residual, reciprocity defect and raw response
residuals. Retarded frequency and imaginary-axis controls use the same
conventions as the quadrupole command. RKS, UKS, ROKS and GKS use their
existing paired, constrained or spinor kernel. Static RKS/UKS double
hybrids include parent, target and separate PT2 spin channels. Static
GKS double hybrids include the generalized pair tensor with one common
PT2 weight and the available target correction. Collinear double
hybrids also support the
[perturbative quasi-energy response](#frequency-dependent-double-hybrid-electric-response).

### Frequency-dependent double-hybrid electric response

`dynamic-polarizability`, `quadrupole-polarizability` and
`multipole-polarizability` use a perturbative quasi-energy action for
real collinear RKS and UKS parents with integer occupations. For one
input at `+z` and its dual observation at `-z`, the parent solver
supplies the first harmonics and their zero-frequency mixed harmonic.
The distinct target correction contains the target-minus-parent
potential contracted with the mixed density and the corresponding
kernel between the two first-order densities.

The spin-resolved PT2 pair generator uses the orbital Fock matrix
in the moving occupied/virtual frame and subtracts its time-derivative
connection. With the occupied-minus-virtual denominator `A`, the
right and left amplitude equations at `+z` use `A+z` and `A-z`,
respectively. The two field vertices are continued analytically in
`z` and contracted through the transpose dual. The pair term includes
the frequency-dependent vertices and perturbative residues. Parent
poles remain perturbative and an undamped driven double pole returns
a named error. At `z=0` the same equations give the relaxed static
parent, target and PT2 energy Hessian.

The total and the parent, target, scaled PT2, unweighted same-spin
alpha, opposite-spin and same-spin beta tensors have `_real` and
`_imag` keys. Units are e²·Bohr^(2 rank)/Hartree with the feature's
own origin and moment normalization. The signed frequencies and
minimum driven pair denominator are in Hartree.
`frequency_pair_diagnostics` returns each component's parent,
projector, frame and left/right pair residuals. Parent and frame arrays
list their first and mixed harmonics separately in
`frequency_parent_lanes` and `frequency_frame_lanes` order. Unit labels
give each absolute residual's source power. The left and right
first-harmonic amplitude residuals use e·Bohr^rank and their relative
residuals are dimensionless. The reference orbital residual is in
Hartree. The static route continues to report static term tensors.

The frequency action currently covers real collinear scalar
electric perturbations. Learned or nonlocal parent and target
providers, correlated GKS and ROKS, magnetic operators, thermal
populations and nonlinear correlated frequency response require
their corresponding time-dependent paths.

### Weighted periodic XC

`driver::run("periodic-energy", ...)` and `driver::run("stress", ...)`
accept the same `xc_terms` array. The resolved provider enters the existing
periodic reference, analytic Cartesian gradient and affine-strain paths.
Select `reference="rks"` for a singlet, `reference="uks"` for separate spin
orbitals, or the existing `reference="roks"` and `reference="gks"` routes.
ROKS uses the full-rank common-orbital Gamma derivative. The generalized
branch uses its field-free all-electron complex spinor derivative.

The raw `orbital_residual` is in Hartree and comes from the final physical
SCF check of the selected reference. `gradient` uses Hartree/Bohr;
`strain_derivative` uses Hartree per cell and follows
`r'_a = r_a + epsilon_ab r_b` for both positions and cell vectors.
`stress` uses Hartree/Bohr³ with the reference-specific tensor convention
documented in the table below. Weighted empirical dispersion, defining
nonlocal terms, multiple distinct range kernels, PT2 and learned
compositions require additional provider contracts. See
[stress acceptance](response.md#r026-stress-elastic-high-elastic).

## Continuous chemical species: model contracts

Two different model families use the continuous-species infrastructure.
An **all-electron continuous-charge model** varies real nuclear charges at
fixed integer electron number and declared spin. Every represented electron
is solved self-consistently. Integer atom labels still select the orbital
basis and grid recipe; a fractional physical charge does not select a new
ordinary element or a basis independent of those labels.

A **reduced substituent model** removes an explicit fragment's electronic
degrees of freedom and replaces its effect with saved local/nonlocal
operators, electrostatics, a frame and electron bookkeeping. The methyl
example freezes eight electrons: carbon 1s contributes two, and the internal
C-H valence space contributes six. This is not the usual approximation that
freezes only core orbitals. One reduced Me in ethane uses the ledger
18 = 8 frozen + 10 active; the separate Me-times-two experiment uses
18 = 16 frozen + 2 active. Electron/spin/attachment inconsistencies remain
errors rather than finite penalty energies.

The current fixed operator has these **51 primary coefficients**:

| Part | Independent coefficients |
| --- | ---: |
| Local Gaussian amplitudes | 3 |
| Real symmetric matrix between eight nonlocal projectors | 36 |
| Gaussian electrostatic charges | 3 |
| Three body-frame dipole vectors | 9 |

Gaussian widths and projector radial exponents are fixed. The charge
correction obeys its neutrality constraint. A separate scalar **E0** is
included once per fragment in the reported density-independent classical
energy. In the common charge-model path it is added after the SCF, so it
does not enter convergence arithmetic. The reported total includes it: do not add an informational
`fragment_internal_energy` field again. E0 cancels in a reaction difference
only when the same fragment records occur with matching stoichiometry on
both sides; absolute energies and changes of fragment count retain it.

The present four-candidate experiment varies **three simplex coordinates**:
`w = (1-q1-q2-q3, q1, q2, q3)`, with nonnegative weights. The same weights
interpolate the saved coefficient vectors. It does not optimize all 51
coefficients independently. The framework can accept additional compatible,
provenance-bearing candidate records through its development interfaces;
the current public API does not thereby expose an arbitrary candidate-list
optimizer. A useful design result still needs named candidates and an
ordinary all-electron evaluation outside the fitted data.

Atom/fragment mappings, attachment ports and body-frame orientation are
separate from these chemical coordinates. Coordinates of retained atoms and
electronic role sites must follow the declared mapping. A saved frozen
fragment does not gain freely deformable internal bonds merely because its
attachment can move. Low-symmetry fragments require a valid oriented frame
and an appropriate model, not a silently spherical replacement.

Generation and application depend on the functional, orbital/dictionary
basis and fixed recipe. Saved records preserve those identities; the checked
few-center public energy/force subset requires PBE and its declared templates.
Its parser accepting another functional name is not acceptance of that
combination. Other continuous-charge implementations have their own scope.
An unvalidated transfer is reported as unestablished, while a known unsupported
model combination is refused. No functional- or basis-independent chemistry
is established by the present fits.

Parameter derivatives `dE/dq` and `d2E/dq2` at fixed geometry include the
appropriate density response. They do not establish nuclear forces,
nuclear Hessians or mixed geometry/parameter derivatives, which also need
moving-basis, frame, projector and frozen-reference terms. The public
`few-evaluate` contract currently exposes its bounded energy/force scope;
private parameter-Hessian experiments do not promote extra public tensors.

The separate retained-center/Euler derivative backend is still a development
capability. Its closed-shell PBE two-center fixture includes moving AOs,
overlap/Pulay terms, quadrature, static local/nonlocal operators, classical
charge/dipole interactions and orbital response. All 12 energy-gradient
components pass independent four-step checks. The full analytic 12-by-12
Hessian is computed, but only 142 of its 144 ordered gradient-difference
checks currently pass; two center/angle mixed checks remain unresolved.
Separate mixed-energy checks pass for those two entries using explicitly
different length/angle steps (center step `h` bohr, angle step `0.01 h`
radian); their fitted orders are about 1.914 and R2 exceeds 0.99970.
Thus each matrix entry has an independent four-step witness, while the
two failed gradient-difference gates remain failed and recorded.
Static operator tests, a reciprocal component, tensor symmetry or a passing
quadratic contraction do not replace these remaining checks. This is not
full nuclear/pose-Hessian acceptance.

The explicit primitive interface now uses the existing `continuous-species`
driver with `fragment_mode="rigid-evaluate"`, `fragment_model_json` and integer
`derivative_order=0/1/2` (default 1). The strict `rigid-static-core-v1` record
declares native units, a separate full/frozen/active electron ledger, fixed
Gaussian/projector coefficients, proper body axes and Euler angles. Retained
point-ion positions come from `System`. Generation provenance is reported
separately from the application functional and core AO templates. This bounded
public connection requires molecular restricted PBE; unsupported references,
periodicity, fields and schemas are refused explicitly. Private generated
v2/v3 records, including occupied-region dictionaries, are not silently loaded
or made compatible by changing a version string.

One actual Rust 1.85 build of the compatible shared source supplies the Rust
library and Python extension; library/test metadata, strict Clippy and the
complete distributable-source hygiene check pass. On that library the moving
dipole regression passes all 42 component fits, and three managed-storage
checks pass. Three linked public Rust contracts and 17 unique native/ASE/CLI
contracts pass. The completed ASE cases check rejection and nonzero unit
conversion; a successful direct ASE model evaluation was not completed.
The local `dft-rs-python` wheel installs offline into an isolated
environment with the expected version and core identity. A nonzero ten-AO
fixture evaluated through the new explicit record reproduces the retained
energy within 1.43e-14 Hartree and all 12 gradient entries within 1.27e-14 in
their native units. This is an interface equivalence check, not new chemical
validation.

Continuous-species work is stopped at this checkpoint. The concurrent
one/two-thread full-Hessian comparison first exceeded its 1800-second bound;
its recovery and the native/ASE full-Hessian output check were then cancelled
on request. They are uncompleted, not passed. The remaining alias/common-suite
and new source-distribution stages behind that check were not run. The compiled
order-two interface is experimental and lacks this final public-runtime
acceptance. No Me-times-two geometry minimization was rerun. Future work, not
an active calculation plan, includes resolving the two direct force-difference
gates, completing public-Hessian validation and assessing model/basis/functional
transfer independently. These derivative results do not repair the recorded
ethane geometry error or certify the private ten-candidate model's accuracy.

The chart contains retained Cartesian centers in bohr followed by Euler
angles in radians, with `Rz(gamma) Ry(beta) Rx(alpha)` acting on the saved
body frame. Euler energy covectors are not Cartesian forces or world-frame
torques. Center/center, center/angle and angle/angle Hessian blocks have
units Hartree/bohr^2, Hartree/(bohr radian) and Hartree/radian^2 respectively.
Mapping a nonlinear anchor/frame chart requires both its Jacobian and its
second derivative: `J^T H J + sum_i gradient_i * map_hessian_i`.
Frozen internal C-H coordinates are not created by this mapping. No new
Me-times-two geometry optimization was run for the derivative checks.

At this documentation checkpoint:

| Experiment | Actual outcome and limit |
| --- | --- |
| All-electron fractional-charge diatomic | PBE-RKS, charges 6.5/7.5, 14-electron singlet and an N-labelled ten-AO STO-3G recipe. Its fixed-q bond relaxes from 2.8 to 2.270707931 bohr with a checked analytic gradient. This proves neither reduced-fragment geometry accuracy nor a nuclear Hessian. |
| Rigid Me-times-two versus explicit ethane | Both constrained projected-force optimizations converge, at 4.522447803 versus 2.965933880 bohr. The 1.556513923-bohr bond error fails chemical acceptance. |
| Unconstrained explicit ethane | All 24 Cartesian coordinates are free, including C-H lengths, angles and torsion. PBE-RKS/STO-3G converges to a 2.9300998573-bohr C-C bond with maximum atomic force 8.7938e-6 Hartree/bohr below the 1e-5 gate; all 24 initial analytic-gradient components pass four-step checks. This validates the full reference calculation, not the reduced model. |
| Me-times-two with all retained poses relaxed | The bond and both fragment orientations converge after numerical-derivative step checks, at 4.5219923116 bohr (2.3929352792 Angstrom). The full reference is 1.5505420701 Angstrom, so the +0.8423932091-Angstrom error still fails chemical geometry acceptance. Internal C-H stretch/bend coordinates are absent from this reduced model; relaxing its poses does not restore them. These are explicitly numerical derivatives, not analytic nuclear-Hessian acceptance. |
| Original four-candidate mixture | The original CoreSp experiment retains its fluoro SCF failures, including the standard 300-iteration attempt. Later ValenceSp parameter experiments and larger candidate collections use separately identified charts and acceptance records; they do not erase this failure or validate nuclear/pose derivatives. |

These observations remain experimental. The current scientific outcomes and
functional/basis limits are maintained in [response.md](response.md).
The ethane geometry failure is retained there as **future work**: the reduced
bond is about 54.3% too long. The old rigid scan is not a fully relaxed energy
profile, and successful convergence does not make either comparison validated
substituent chemistry. Current work returns to the declared continuous-parameter
optimization; later fitting must preserve this failed benchmark rather than
erase it.

## Persistent calculation logs

The common driver accepts `log_file`, `log_mode` and `log_level` alongside
`cpu_threads` and `memory_mb`. A file alone selects file logging. Modes are
`console` (stderr), `file`, `both` and `none`; levels are `error`, `warn`,
`info` (default) and `debug`. Providing only `log_level` explicitly selects
stderr logging. File modes require a nonempty file path; console/none
refuse an unused file path. Omission creates no file and
inherits an enclosing run. Unconfigured library progress is quiet; warnings
and errors can still use the existing stderr fallback. This is not a promise
that every unconfigured call produces no output.

```rust
let controls = Controls::from_value(&Value::object()
    .with("xc", "hf").with("cpu_threads", 2)
    .with("log_file", "calculation.log").with("log_level", "info"))?;
let report = run("energy", &system, &controls)?;
```

For lower-level Rust calls, `logger::LogOptions::to_file("calculation.log")?`
provides `run(|| calculation())` and `run_with_limits(resource_limits, ||
calculation())`. The latter applies the existing resource contract and
logging together. `install` supports adapters retaining their own error
type; they call `logger::record_error` before returning an error and arrange
`ResourceLimits::install` around Rayon work. Ordinary library callers use
`run` or `run_with_limits`. The old
`logger::init` is a calling-thread compatibility API; call `logger::flush()`
to check its I/O status. Scoped APIs check I/O
at completion and propagate the context to private Rayon workers.
Nested calculations inherit that context and pool. Changing the log destination
inside an active resource pool is refused; configure logging and resources at
the outer boundary with `run_with_limits` to preserve the CPU ceiling.

Records carry a process/run identity, elapsed time and level. Info records
include method/resource settings, existing SCF iteration energies and
residuals, convergence/errors and common-driver result/unit summaries.
Debug adds method detail; lower levels intentionally filter progress.
Files append, flush each record, and retain partial records on errors or
unwinding. A write/flush failure makes the requested logging operation fail;
it cannot silently return a successful calculation. This is flushing to the
OS, not a power-loss durability guarantee. Abrupt process termination may
omit the final record. Join asynchronous work before leaving a scope.

Independent concurrent calculations use independent sinks. A canonical
file path already owned by another active run in the same process is
rejected. Sequential runs may append to the same file. Use distinct files
for independent processes too; no cross-process or hard-link file lock is
claimed. No default embeds a developer path or initializes a global sink.

## Molecular calculation

Response grid ownership, independent VV10/NEO/PBC controls and grid-free
exceptions are specified in the [quadrature contract](conventions.md#quadrature-grids).
Low-level response APIs reuse the supplied `MolecularGrid`; the common
driver preserves the requested grid even for HF properties that still need
quadrature. `MolecularSetup::integration_grid()` borrows the existing XC grid
or constructs the property grid from its retained `GridOptions`.

```rust
use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{GridOptions, MolecularGrid};
use dft_rs::scf::{run_scf, ScfOptions};
use dft_rs::system::Molecule;
use dft_rs::xc::{resolve, FunctionalSpec};

fn main() -> dft_rs::Result<()> {
    let molecule = Molecule::from_xyz_file("water.xyz")?;
    let basis_set = data::load("def2-svp")?;
    let basis = MolBasis::build(&molecule, &basis_set)?;
    let functional = resolve(&FunctionalSpec::Combined("b3lyp".into()))?;
    let grid = MolecularGrid::build(&molecule, &GridOptions::default())?;
    let options = ScfOptions::default();
    let result = run_scf(&molecule, &basis, &options, &functional, Some(&grid))?;
    println!("{:.12}", result.energy);
    Ok(())
}
```

Constructed `Atom` positions are already in Bohr. Atomic number zero is not
a ghost specification: it does not identify which element's basis should be
placed and `MolBasis::build` rejects it. SAPT constructs its dimer-centred
monomer spaces internally without overloading `Atom::z`.

`GridOptions::default()` is the standard grid: Treutler–Ahlrichs M4 radial
points × Lebedev–Laikov angular rules with five-region pruning and the
Becke partition with Treutler's size adjustment, at `level = 3` (radial
points and angular degree by period). `GridOptions::from_controls(level,
preset, scheme, weight_cutoff)` reads the driver controls `grid_level`, `grid_preset`
(`fine`, `ultrafine`, `superfine`; type `GridPreset`),
`grid_angular_scheme` (`lebedev`, `product`; type `AngularScheme`) and
`grid_weight_cutoff` (Bohr^3, default 1e-14; zero keeps every point so the
energy is a smooth function of the nuclear positions);
`MolecularGrid::atomic_point_counts` gives the per-atom point counts before
the partition. Acceptance against frozen default grids of common practice:
`tests/p238_standard_grid.rs`.

## Basis sets and ECPs

```rust
use dft_rs::basis::{data, mixed, MolBasis};

let bundled = data::load("def2-tzvp")?;
let bundled_or_file = data::load_or_read("custom.g94")?;
let mut mixed_set = mixed::load_spec("def2-tzvp; H=def2-svp")?;
mixed::apply_ecp_spec(&mut mixed_set, "Ni=stuttgart-rsc-1997")?;
let basis = MolBasis::build(&molecule, &mixed_set)?;

let names = data::bundled_names();
let effective_charges = basis.effective_charges(&molecule);
```

Gaussian exponents must be finite and positive. ECP headers must match their
element and cannot replace more than Z electrons. These constraints are
checked both by the Gaussian94 parser and, for programmatically constructed
objects, when the molecular basis is built.

### Pseudo-substituent request plumbing

```rust
use dft_rs::pseudo_sub::parse_named_sites;

let requests = parse_named_sites("4=Me; 7=Ph")?; // zero-based atom indices
let site = requests[0].build_for_molecule(&molecule)?;
```

The parser also accepts newline-separated assignments, `#` comments and
`@file` includes, and rejects duplicate atom indices. Recognized names are
Me, Et, tBu, Ph, CF3 and OMe. The final line currently returns a typed
`DftError::Unsupported` after checking that the index exists and the G nucleus
is represented by `Atom { z: 1 }`: local-frame/Jacobian and real Wigner-D mechanisms
are implemented, but no provenance-bearing named parameter set has passed its
required physical validation gates. That named-request route has no accepted
physical parameter consumer. Separate saved-fragment APIs can apply explicit
constructed operators within their bounded model contracts; they do not
activate named requests. CF3 and OMe additionally have no Phase-4 model or acceptance
specification. Parsing is exposed now so configuration handling is shared and
deterministic; it is not a claim that named pseudo-substituent chemistry is
available.

The fitting mechanics are a separate candidate-only API:

```rust
use dft_rs::pseudo_sub::fit::run_synthetic_recovery_gate;

let fit = run_synthetic_recovery_gate()?;
assert!(fit.optimizer.converged);
let text = fit.record.to_text()?; // status=candidate-unaccepted
fit.record.validate_level("synthetic/phase4-fitter-gate")?;
```

`fit_reference_set` additionally exposes the fixed-seed group split, weighted
ridge initialization, Tikhonov prior, diagnostic policy, packed-symmetric B
spectral projection and projected L-BFGS. Candidate records never register a
chemical name. The physical observable evaluator and acceptance data remain a
separate prerequisite.

For local-frame mechanism work, connectivity is explicit rather than inferred
from an empirical distance cutoff:

```rust
use dft_rs::pseudo_sub::frame::{
    build_local_frame_with_connectivity, FrameConnectivity,
};
use dft_rs::pseudo_sub::SiteSymmetry;

let frame = build_local_frame_with_connectivity(
    &molecule,
    g_atom,
    FrameConnectivity {
        x_atom,
        reference_candidates: &bonded_neighbours_of_x,
    },
    SiteSymmetry::Cs,
)?;
```

The caller attests that the candidates are X's other bonded neighbours; the
function validates their indices and ranks only that set by highest Z then
lowest atom index. Consequently an unrelated high-Z atom elsewhere in the
molecule cannot change the frame. The older `build_local_frame` discovers X
by unique nearest distance but refuses if multiple possible references remain.

## SCF options and nonlocal correlation

```rust
use dft_rs::nlc::Vv10Params;
use dft_rs::scf::{EriMode, ScfOptions};

let functional_name = "wb97m-v";
let options = ScfOptions {
    e_tol: 1e-10,
    conv_tol_grad: 1e-5,
    p_tol: 1e-8,
    max_iter: 200,
    incore_mb: 4096,
    eri_mode: EriMode::default(),
    density_fit: None, // Some("def2-universal-jfit".into()) enables RI-J
    vv10: Vv10Params::for_functional(functional_name),
    vv10_grid_level: 3,
    ..ScfOptions::default()
};
```

`resolve` returns the semilocal/exact-exchange functional. Rust callers must
put a defining VV10 term in `ScfOptions`; `Vv10Params::for_functional`
centralizes the name mapping. The CLI and Python API do this automatically.
Empirical D2/D3/CHG corrections are separate from SCF and live in
`dft_rs::dispersion`.

`ScfResult` exposes energies, spin-resolved MO coefficients and densities,
Fock/overlap/core matrices, occupations, spin diagnostics, convergence data,
`e_nlc`, `e_pcm`, and the optional grid electron-count diagnostic.

## Derivatives

```rust
use dft_rs::gradient::analytic_gradient;
use dft_rs::hessian::{analytic_hessian_full, numerical_hessian};
use dft_rs::dispersion::Dispersion;

let gradient = analytic_gradient(
    &molecule, &basis, &options, &functional, Some(&grid)
)?;

let analytic = analytic_hessian_full(
    &molecule,
    &basis,
    &options,
    &functional,
    &gradient.scf,
    Some(&grid),
)?;
let hessian = &analytic.hessian;
let apt = &analytic.dipole_derivatives;

let numerical = numerical_hessian(
    &molecule,
    &basis_set,
    &options,
    &functional,
    "b3lyp",
    Some(&GridOptions::default()),
    &Dispersion::None,
    5e-3,
)?;
```

ECP first- and second-derivative terms are implemented analytically. A
`DerivOrderUnsupported` error means the requested functional derivative is
not available; convergence, response, resource, and algebra errors should
not be converted into a numerical fallback silently.

RI-J nuclear derivatives require derivatives of the three-centre integrals
and Coulomb metric. Until that analytic implementation is present,
`assemble_gradient` and `analytic_hessian_full` explicitly refuse RI-J.
`numerical_gradient`, `numerical_hessian`, and the optimizer differentiate
the RI-J total energy by geometry-rebuilt finite differences instead of
silently substituting exact-J derivatives.

## Geometry optimization and transition states

```rust
use dft_rs::dispersion::Dispersion;
use dft_rs::optimizer::{optimize, OptOptions};
use dft_rs::saddle::{optimize_saddle, SaddleMethod, SaddleOptions};

let minimum = optimize(
    &molecule,
    &basis_set,
    &options,
    &functional,
    "b3lyp",                       // name used for dispersion parameter lookup
    Some(&GridOptions::default()),
    &Dispersion::D3Bj,
    &OptOptions {
        gtol: 3e-4,
        trajectory: Some("path.xyz".into()),
        ..OptOptions::default()
    },
)?;

let saddle = optimize_saddle(
    &molecule, &basis_set, &options, &functional, "b3lyp",
    Some(&GridOptions::default()), &Dispersion::None,
    &SaddleOptions {
        method: SaddleMethod::RsIrfo,
        ..SaddleOptions::default()
    },
)?;
```

Both walkers take a `BasisSet` rather than a `MolBasis` because the basis has
to be rebuilt at every geometry. Both return the full path in
`trajectory: Vec<OptFrame>` and, if `OptOptions::trajectory` is set, stream it
to disk **as the walk proceeds** — appended and flushed per accepted step, so
an interrupted run still leaves its path behind. `OptFrame::to_xyz_string`
puts the step number, energy and `|g|max` on the comment line.

`prfo_step` and `rsirfo_step` are public for externally supplied Hessians.
The latter constructs `P = I - 2tt^T`, minimizes the coupled image function,
and solves for the `S = alpha I` metric that places the RFO solution within
the requested trust radius; it does not rescale an overlong vector.

## External electric field

```rust
use dft_rs::math::Vec3;

let field = ScfOptions {
    external_field: Some(Vec3::new(0.0, 0.0, 5e-3)),
    ..options.clone()
};
let perturbed = run_scf(&molecule, &basis, &field, &functional, Some(&grid))?;
```

The field enters `H_core` as `+E.r` together with the nuclear term
`-Z_A R_A.E`, so the energy, the density and everything derived from them are
the field-perturbed quantities. `dE/dE_ax = -mu_ax` holds to the SCF
convergence threshold, which is how the term is gated.

`assemble_gradient` differentiates the field term too: the electronic
`sum_c E_c Tr[P dD^c/dR_A]` from `deriv_mat::dipole_derivative_matrices`,
plus the nuclear `-Z_A E`. There is no operator-center term, because the
dipole operator is anchored at a fixed origin rather than on a nucleus.
Note that the gradients then sum to `-Q_tot E`, not to zero: a uniform field
breaks translational invariance for a charged system.

`analytic_hessian_full` includes the direct contraction of the second
derivative of all three dipole-integral matrices. The ordinary CPKS response
therefore completes the field-dependent Hessian without a separate response
implementation. `numerical_hessian` remains available as an independent
finite-difference check.

## Extra one-electron operator

```rust
use dft_rs::linalg::Matrix;

let mut extra = Matrix::zeros(basis.nao_sph, basis.nao_sph);
// ... fill `extra` from AO indices, e.g. an embedding potential ...
let options = ScfOptions {
    extra_hcore: Some(extra),
    ..ScfOptions::default()
};
```

`ScfOptions::extra_hcore` adds a caller-supplied one-electron operator to
`H_core` (embedding potentials, the SAPT induction finite-difference gate,
...). Its contract: **the matrix must be independent of the nuclear
coordinates**. It contributes to the energy through `Tr[P·extra]` like any
other core term and to nothing else — `assemble_gradient`,
`analytic_hessian_full`, and `tda_gradient`/the CPKS response kernel never
read this field, because a geometry-independent additive constant has zero
nuclear derivative and, being density-independent, drops out of the
response kernel too.

Do not pass a geometry-dependent operator here: the plain `Matrix` type
cannot carry its own `dR`, so the gradient/Hessian/excited-state code would
silently miss that derivative and return a wrong analytic result with no
error. A geometry-dependent one-electron term needs its own derivative
contribution wired into `gradient.rs`/`hessian/analytic.rs` directly —
`src/pcm.rs`'s C-PCM/IEF-PCM reaction field is the worked example: it is
rebuilt from the density every SCF iteration and ships its own analytic
gradient/Hessian machinery (`cpcm_gradient`, `switch_gradient`,
`PcmSolver::gradient`), which is why it is a dedicated `ScfOptions::pcm`
field rather than being plumbed through `extra_hcore`.

## Continuum solvation

```rust
use dft_rs::pcm::{solvent_by_name, PcmModel, PcmOptions};

let water = solvent_by_name("water").expect("bundled");
let options = ScfOptions {
    pcm: Some(PcmOptions {
        model: PcmModel::Cpcm,       // Cosmo | IefPcm | Ssvpe
        eps: water.eps,
        descriptors: Some(water),
        ..PcmOptions::default()
    }),
    ..ScfOptions::default()
};
```

The reaction field is rebuilt at every SCF iteration, so the solvated density
is self-consistent, and it participates in the CPKS/TDDFT response kernel — a
polarizability or a Hessian sees the solvent relax with the density.
`ScfResult::e_pcm` reports the electrostatic solvation energy.

`solvent_names()` lists the 179 bundled solvents and `solvent_by_name`
resolves names and short aliases. The SMD non-electrostatic term is
`smd_cds_energy`; it refuses element pairs whose surface-tension coefficients
are not determined rather than returning the atomic sum alone, which is wrong
there by hundreds of cal/mol. The cavity itself (`build_cavity`, `s_matrix`,
`d_matrix`, `patch_areas`) and the analytic derivative pieces
(`switch_gradient`, `nuclear_potential_gradient`, `electronic_potential_gradient`,
`cpcm_gradient`) are public for callers building their own solvation model.

## Properties and files

```rust
use dft_rs::properties::{dipole_moment, lowdin_charges, mulliken_charges};
use dft_rs::properties::thermochemistry::{
    thermochemistry, ThermochemistryOptions,
};
use dft_rs::properties::vibrational::vibrational_analysis;

let dipole = dipole_moment(&molecule, &basis, &gradient.scf);
let mulliken = mulliken_charges(&molecule, &basis, &gradient.scf);
let lowdin = lowdin_charges(&molecule, &basis, &gradient.scf)?;
let modes = vibrational_analysis(&molecule, hessian, Some(apt))?;
let thermo = thermochemistry(
    &molecule,
    &modes.frequencies_cm,
    gradient.scf.energy,
    ThermochemistryOptions {
        temperature_kelvin: 298.15,
        pressure_pascal: 101_325.0,
        symmetry_number: 2,
        electronic_degeneracy: molecule.multiplicity,
    },
)?;

dft_rs::properties::molden::write_molden(
    std::path::Path::new("water.molden"),
    &molecule,
    &basis_set,
    &basis,
    &gradient.scf,
)?;
dft_rs::properties::checkpoint::dump_wavefunction(
    std::path::Path::new("water.wfn.txt"),
    &molecule,
    &basis,
    &gradient.scf,
)?;
```

Mulliken and Lowdin charges use ECP effective ionic charges, so their sum is
the molecular charge for all-electron and ECP calculations alike.

`thermochemistry` implements ideal-gas rigid-rotor/harmonic-oscillator
corrections. Energies and totals are Hartree per molecule; entropies are
Hartree/K per molecule. Temperature is kelvin, pressure is pascal, and the
caller supplies the external rotational symmetry number. Atomic, linear and
nonlinear rotors are distinguished from their principal moments. Every
frequency must be finite and positive: transition-state thermochemistry is
refused rather than silently dropping an unstable coordinate.

`dft_rs::properties::natural_bond::natural_bond_analysis` provides the
metric-correct NAO/NPA foundation and Wiberg bond indices. Its result labels
the method as a foundation: it does not yet claim NHO construction,
Lewis-structure search, localized NBOs, or donor-acceptor analysis.

## SAPT and response

```rust
use dft_rs::sapt::sapt_interaction;
use dft_rs::response::polarizability::polarizability;
use dft_rs::response::tddft::{TdaSolver, TdaSpin};

let sapt = sapt_interaction(
    &monomer_a, &monomer_b, &basis_set, &functional, &options, None
)?;
let alpha = polarizability(
    &molecule, &basis, &functional, Some(&grid), &gradient.scf, &options
)?;
```

SAPT reports electrostatics, exchange, both induction directions,
exchange-induction, uncoupled/coupled dispersion, exchange-dispersion, and
the counterpoise-corrected supermolecular comparison.

`polarizability` is analytic CPKS; `raman_activities` differentiates it along
supplied normal modes. `ResponseSolver` underneath is matrix-free (PCG) and
exposes `solve_ov` for a raw occupied-virtual right-hand side and
`solve_ov_block` for many right-hand sides at once — the block form runs one
PCG over all of them, so the iteration sequence is identical to solving them
individually and only the Fock-build scheduling changes.

## Excited states

```rust
use dft_rs::response::tddft::{TdaSolver, TdaSpin};
use dft_rs::response::tddft_grad::{tda_gradient, tda_gradient_with_grid};

let solver = TdaSolver::new(
    &molecule,
    &basis,
    &functional,
    Some(&grid),
    &gradient.scf,
    &options,
    TdaSpin::Singlet,   // Triplet | SpinResolved
)?;
let tda = solver.solve(5)?;          // Tamm-Dancoff
let rpa = solver.solve_rpa(5)?;      // full TDDFT/TDHF, Casida reduced form

let force = tda_gradient(
    &molecule, &basis, &functional, &gradient.scf, &tda[0], &options
)?;

// Pure LDA/SVWN5 uses the same grid as SCF and TDA.
let lda_force = tda_gradient_with_grid(
    &molecule, &basis, &functional, Some(&grid),
    &gradient.scf, &tda[0], &options
)?;
```

`TdaState` carries the excitation energy, the amplitudes `x` per spin, the
oscillator strength, and rotatory strengths in both the length and the
velocity gauge. `solve_dense` is available for small problems and
cross-checks the Davidson path.

`TdaSolver` follows `ScfOptions::eri_mode` for both ordinary and
range-separated exchange. Direct mode evaluates the non-symmetric
transition-density J/K contractions without storing an ERI tensor; explicit
in-core mode retains the existing guarded tensor path. There is no separate
fixed TDDFT in-core limit or implicit TDDFT-only mode switch.

`tda_gradient` is the grid-free HF/CIS-compatible wrapper;
`tda_gradient_with_grid` adds the gas-phase closed-shell singlet pure-LDA
path. Both use a Handy-Schaefer Z-vector assembly: one CPHF solve serves
all `3N` coordinates, and it returns the gradient of the TOTAL excited-state
energy `E_SCF + omega`, nuclear repulsion included. Its scope is a
closed-shell singlet over an HF/CIS or SVWN5/LDA reference. GGA, meta-GGA,
hybrid/RSH, nonlocal-correlation, PCM, triplet, full-RPA, and open-shell
gradients are refused.

## Periodic boundary conditions

`periodic::PeriodicGeometry` additionally provides masked 1D/2D/3D lattice
geometry without using open cell vectors as periodic images. The shared
`periodic::electrostatics::periodic_electrostatics` driver computes prescribed
point/Gaussian Coulomb energies and analytic forces, including explicit
Gaussian compensation for charged wires/slabs. See
[dimensional boundary conventions](periodic.md#1-boundary-conditions-and-electrostatics) for the
potential gauge, source normalization, charge compensation and the electronic
SCF/DFPT connections still being implemented.

`dft_rs::periodic::Cell` provides periodic one-electron matrices, their
nuclear derivatives, Ewald nuclear terms, and `hartree` / `hartree_image_blocks`
for a caller-supplied real-space density. `electron_count` and the Hartree
entry points validate dimensions, finite values, unique translations, and
the Hermiticity condition `D(-T) = D(T)^T` before doing expensive work.
`gamma_density_images` performs the one-point inverse Bloch transform;
using a lone `D(0)` for a Gamma density is physically incorrect when basis
functions overlap between cells.

The geometry, lattice, basis, and short-range cutoff are exposed through
read-only `molecule()`, `lattice()`, `basis()`, and `overlap_cutoff()`
accessors. They cannot be mutated independently of the cached image sums.
Use `rebuilt_with_molecule` (or construct a new `Cell`) after changing the
system so all AO, ECP, and Ewald caches are rebuilt together.

A complete finite-k run, from geometry to forces:

```rust
use dft_rs::periodic::forces::analytic_forces;
use dft_rs::periodic::occupations::OccupationScheme;
use dft_rs::periodic::scf::{run_periodic_scf, PeriodicScfOptions};
use dft_rs::periodic::{Cell, Lattice};
use dft_rs::scf::Guess;

let lattice = Lattice::new(
    Vec3::new(5.5, 0.0, 0.0),
    Vec3::new(0.0, 12.0, 0.0),
    Vec3::new(0.0, 0.0, 12.0),
);
// The cutoff bounds the SHORT-range one-electron lattice sums only; the
// conditionally convergent Coulomb sums use Ewald summation instead.
let cell = Cell::new(molecule, lattice, &basis_set, 11.0)?;
let functional = resolve(&FunctionalSpec::Combined("pbe0".into()))?;
let options = PeriodicScfOptions {
    k_mesh: [4, 1, 1],
    exchange_cutoff: Some(11.0),
    guess: Guess::Sad,
    occupation: OccupationScheme::Insulator,   // or FermiDirac { temperature_kelvin }
    ..PeriodicScfOptions::default()
};

let scf = run_periodic_scf(&cell, &functional, &options)?;
let forces = analytic_forces(&cell, &functional, &scf, &options)?;
```

Atomic positions can be optimized while retaining the lattice exactly:

```rust
use dft_rs::periodic::optimizer::{optimize_fixed_cell, PeriodicOptOptions};

let optimized = optimize_fixed_cell(
    cell.molecule(), cell.lattice(), &basis_set, cell.overlap_cutoff(),
    &functional, &options, &PeriodicOptOptions::default(),
)?;
assert!(optimized.converged);
let final_forces = analytic_forces(
    &optimized.cell, &functional, &optimized.scf, &options,
)?;
```

Every trial rebuilds the immutable periodic caches and minimizes total energy,
or Mermin free energy under smearing. `PeriodicOptResult` owns the exact final
`Cell` paired with its SCF result. Lattice vectors are deliberately immutable;
optimizing them requires the not-yet-complete electronic stress tensor.

`dft_rs::periodic::scf::run_periodic_scf` assembles finite-k HF and
LDA/GGA/meta-GGA Kohn-Sham energies, localized exact exchange, and analytic
all-electron forces. `PeriodicScfOptions::occupation` selects integer or
mesh-global Fermi-Dirac occupations; the result reports internal energy,
Mermin free energy, entropy term, chemical potential, occupations, and the
energy-weighted density needed by the fractional-occupation Pulay force.
`level_shift` applies `lambda (I-P)` in the orthonormal complex Bloch basis
while the physical commutator residual exceeds `level_shift_until`; final
canonicalization always uses the unshifted Fock. Density damping and DIIS
subspace size are likewise public controls and are wired through the CLI and
Python periodic entry point. Core is the compatibility default; GWH and SAD
are selectable, with SAD using a periodic HF Fock seed followed by the normal
finite-k occupation solve. For integer occupations, `use_adiis` and
`adiis_until_error` select a complex finite-k convex interpolation over one
shared coefficient set for every spin and k point before CDIIS takes over.
Fermi-Dirac calculations use CDIIS because this zero-temperature ADIIS model
does not include entropy curvature. Periodic SOSCF remains staged.
The result also retains the validated cell/functional/cutoff/grid plan;
analytic forces reject a result paired with a different cell or an
energy-defining option set instead of differentiating a different energy.

`PeriodicBoundary` records the three periodic axes and validates a k mesh.
`PeriodicCapabilities::current()` exposes complete and staged features.
`PeriodicEcpPlan` exposes translated-center cutoff/convergence information;
periodic ECP energies and analytic forces are included. True dimensional
[Coulomb source kernels](periodic.md#1-boundary-conditions-and-electrostatics), Gaussian overlap/kinetic,
atomic derivatives and [XC quadrature](periodic.md#2-electronic-basis-and-integration-grid) are available
through `periodic::periodic_basis` and `periodic::electrostatics`.
Partial-PBC electronic nuclear/Hartree and SCF/DFPT connections remain pending.
Periodic VV10/CHG-required
functionals are likewise rejected until their nonlocal kernels are complete.

## Magnetic operators

`dft_rs::properties::nmr` supplies the one-electron GIAO operator layer:
`electric_field_operator` (the `(r-C)_j/|r-C|^3` integrals), the paramagnetic
nuclear magnetic-dipole operator, orbital angular momentum about any origin,
and the momentum operator. Each is gated on antisymmetry and an exact
origin-shift identity.

`dft_rs::properties::magnetic::{shielding_tensors, magnetic_response}`
assemble closed-shell exact-exchange shieldings. Their `*_with_grid` variants
add the molecular grid required by GIAO LDA/GGA and by meta-GGA in either
gauge (the current-current response is a grid contraction). CommonOrigin
LDA/GGA calculations need no grid.
The current-corrected meta-GGA path is experimental: its internal FD and
gauge-origin gates are present, but the pinned external NMR implementation
does not supply a meta-GGA full-tensor oracle.
Range-separated exchange uses a separate exchange-only Erf derivative and
response engine. SC-VV10, ECP and open-shell cases return typed
errors rather than incomplete tensors.

## Logging

```rust
dft_rs::logger::init(dft_rs::logger::LogMode::Both, Some(std::path::Path::new("run.log")))?;
dft_rs::logger::log("starting");
```

`LogMode` is `Console | File | Both | None`. This initializer configures the
calling thread for compatibility. New code uses the run-local scopes in
[persistent calculation logs](#persistent-calculation-logs), which isolate
independent concurrent calculations and propagate context to scoped workers.
Call `logger::flush()` to check compatibility-mode I/O failures. Unconfigured
warnings/errors may use stderr; no default log file is created by the library.

## Errors and threading

Fallible functions return `dft_rs::Result<T>`. Match `DftError` variants when
program logic depends on the failure category rather than parsing messages.
`Unsupported { feature, missing }` identifies an unavailable calculation,
reference/feature combination, common-driver adapter or known control in an
inapplicable context. `DerivOrderUnsupported` identifies an unavailable
functional derivative order. Unknown control names, malformed selectors,
missing required inputs and invalid shapes remain `InvalidInput`.
The common driver preserves unhandled reference selectors and refuses them
before ordinary molecular setup instead of substituting the default reference.
An explicitly selected automatic Hessian route can use its documented,
labelled numerical fallback; an unsupported model selection does not authorize
a different reference. Convergence, resource and linear-algebra failures must
propagate rather than be classified as missing implementation.

Rayon parallelizes integral, grid, and derivative batches. Call
`dft_rs::ensure_parallelism()` early in an embedding application to select
the shared Rayon-backed faer execution path. `RAYON_NUM_THREADS` sizes the
pool.

Per-calculation resource settings use `resources::ResourceLimits` rather than
changing environment variables or global pools:

```rust,ignore
let limits = dft_rs::resources::ResourceLimits::new(Some(4), Some(512))?;
let report = limits.run(move || dft_rs::driver::run("energy", &system, &controls))?;
```

The common driver also accepts `cpu_threads` and `memory_mb` controls. Both
are optional positive integers; omission retains the existing behavior.
Integral floats and booleans are refused for these global controls. They
are consumed before feature reference/mode translation; unknown feature
controls still fail, and the original request retains its serialized settings.
`cpu_threads` is a ceiling on workers for this calculation, including nested
Rayon/faer tasks in its private pool. Independent calculations may use
different pools concurrently; their limits are not a process-wide total.
Nested calculations inherit the same pool and budget and cannot reconfigure
them. The existing sequential faer environment override remains respected.

`memory_mb` uses MiB (1024^2 bytes) and bounds a shared ledger of managed Rust
matrix payloads (including copies), dense molecular ERI storage, ERI assembly
payload/metadata, conservative dense eigen/LU workspace allowances, and retained
semilocal/meta-GGA XC and finite-q exchange caches.
Registered storage stays charged until its last owner drops. Checked size
arithmetic rejects overflow; exhausted managed allocations return
`DftError::ResourceLimit`. The infallible Matrix compatibility constructors
unwind only a private allocation refusal, recovered by the resource envelope;
unrelated panics remain panics. Wrap low-level Rust calculations in the
envelope when requesting managed allocation failures as `Result` values.
Managed memory recovery requires an unwind-enabled build; `panic=abort`
builds explicitly refuse `memory_mb`.
Automatic ERI selection retains direct recomputation when its conservative
storage estimate does not fit. Explicit in-core requests fail instead.
Existing `incore_mb`, `pt2_memory_mb` and `td_memory_mb` driver limits are
capped by the remaining budget and are never enlarged.

This is **not a whole-process RSS limit**. Caller inputs, Python/report
objects, grids and basis construction, raw Vec/Taylor/PT2/CI storage,
allocator metadata, thread stacks and other unregistered caches are not
fully accounted. faer allocations use a workspace allowance, not allocator
instrumentation. Existing component-specific budgets remain relevant.
`resources::remaining_bytes` and `reserve` provide the shared lease protocol
for cache owners. Optional caches leave headroom for required workspaces;
a failed optional-cache reservation selects the existing exact recomputation
path. Scoped workers use an 8 MiB stack, so third-order XC dispatch stays in
the same pool and budget instead of starting a separate native helper. Reports from scoped public calls include
`resources.cpu_threads`, `memory_mb`, `managed_peak_bytes`, `managed_live_bytes`
and a scope description. These are accounting diagnostics, not RSS readings.
Managed byte counters are enabled with `memory_mb`; CPU-only scopes leave
memory accounting disabled and report zero byte counters.


### Canonical moving-metric response and Mermin thermochemistry

`response::ensemble::EnsembleSolver::solve_metric(&[F_skeleton_alpha,
F_skeleton_beta], &delta_overlap, &ResponseSolverConfig)` returns a
`MetricEnsembleResponse`. Its `response` is the fixed-per-spin-number
`EnsembleResponse`; `delta_energy_weighted_density` is the total derivative
of `sum_s C_s f_s epsilon_s C_s^T`. Sources and metric are matching finite
AO matrices. The solver uses the existing full Hartree/exchange/XC/nonlocal/
continuum kernel, spectral divided differences with their degeneracy limit,
and GMRES. The moving frame contributes `-{delta_S,f}/2` to P and
`-{delta_S,epsilon}/2` to the Hamiltonian source. Both original self-consistency
and `Tr(S*dP)+Tr(dS*P)` residuals are retained.

`hessian::analytic::analytic_hessian_full` uses this solver at positive
electronic temperature and the existing occupied/virtual solver at zero
temperature. `AnalyticHessianResult` additionally exposes
`energy_weighted_density_derivatives`, `ensemble_response_residuals` and
`ensemble_number_defects`. `density_derivatives` and weighted-density
derivatives are in Cartesian coordinate order `3*atom+axis`. The shared
analytic integral/grid/Pulay skeleton supplies the Mermin Hessian and APT.

`properties::thermochemistry::thermochemistry_from_mermin(mol,
frequencies_cm, mermin_energy, electronic_temperature_kelvin,
electronic_entropy, options)` combines A_e (Hartree) and S_e/k_B with the
existing `ThermochemistryOptions` and ideal-gas RRHO model. The returned
`Thermochemistry` contains physical internal energy, enthalpy and total
entropy, and `gibbs_free_energy=H-T*S`. Positive electronic and nuclear
temperatures must agree; the zero-electronic-temperature case is the usual
ground-state RRHO approximation. Additional electronic degeneracy is supplied
through the options. The common driver reuses this function for
`gibbs-free-energy` and `mermin-free-energy` with `gibbs=true`.

### Static phonon pairing, mass and coupled gap response

`properties::pairing::Lm2005` shares one scalar-generic implementation of
`convolution(x,y,omega,kT)`, `phonon_factor(...)`,
`evaluate(&PairingInput<S>)`, `mass_factor(...)` and
`mass(&PairingInput<S>, &band_weights)`. The mass factor is Hartree^-2,
K is Hartree and Z is dimensionless. Vertices carry mode/q weights; K
excludes final-band weights and Z includes them once. The compatibility
type `Lm2005OffDiagonal` aliases `Lm2005`. The default analytic-limit
quadrature has order 32; `new(order)` accepts 8..128.

`response::superconducting::Lm2005BandFunctional` implements the existing
`GapFunctional` with generated phonon K/Z at every trial chemical potential.
Parameters are energies[n], weights[n], kT[1], phonon energies[m], squared
vertices[m*n*n] and screened Coulomb[n*n]. The supplied-Z
`PhononBandFunctional` and supplied-K/Z `SuppliedBandFunctional` remain
available. `solve_gap_taylor` uses the same canonical complex gap/number/
phase constraint and AD Jacobian for all providers and returns
`GapTaylor::mass_renormalization_coefficients[order][band]` along with the
gap, density, occupation, mu and kernel coefficients. All coefficients
are factorial-normalized in the caller's dimensionless parameter.

The mass uses the static LM2005 particle-hole-symmetric phonon approximation.
No normal-density backreaction, causal kernel or thermodynamic branch
selection is implied. Full controls, units, model assumptions and the
three aligned public entry tables are in [rust-api.md#common-calculation-driver](rust-api.md#common-calculation-driver).

### Connected lattice, electromagnetic and four-component components

`periodic::phonon_bands::ForceConstantModel::long_wave_relaxation(direction,
order, config)` computes factorial-normalized atomic-phase internal
displacements at q=kappa*direction for orders zero through four. Direction
is dimensionless and kappa has inverse-Bohr units. The constrained solve
keeps the mass center fixed and retains the original projected residuals,
acoustic reactions and raw uniform-force defect. Its `LongWaveRelaxation`
contains real/imaginary displacement matrices [order][3N][3], in Bohr^order
per imposed displacement. `born_polarization(&born_charges, volume)` contracts
them to [order][polarization][displacement] in e*Bohr^(order-3).
The Born lattice contribution to type-I flexoelectricity is minus the
second polarization coefficient. Electronic and dynamic-multipole terms
remain necessary for the complete coefficient.

`response::magnetic_mixed::mixed_electromagnetic_energy` extends the existing
`mixed_magnetic_energy` through matching electric and magnetic direction
lists. Each dimensionless parameter may perturb both fields; the shared
analytic orbital/metric/XC response returns the contracted energy derivative
in Hartree through fourth order. The original magnetic function delegates
to this implementation with zero electric directions. Finite-field
`scf::magnetic::{magnetic_field_scf, magnetic_field_gradient}` also consume
`ScfOptions::external_field`. `MagneticFieldResult::electric_field` records
the scalar potential, including nuclear coupling. A charged system uses
the original laboratory energy origin and sum of atomic gradients -Q*E.

`integrals::dirac::restricted_magnetic_balance` takes the same analytic
shells, finite Gaussian nuclei, speed of light and outer derivative order
as `restricted_kinetic_balance`, plus magnetic field and gauge origin.
It forms the full complex restricted-magnetic-balance Hamiltonian and
overlap. `BalancedDiracIntegrals::overlap_imaginary` and
`balanced_kinetic_real/imaginary` retain complex spinor metric and kinetic
blocks. `matrix_coefficients` passes both parts to the existing no-pair
projector response. This common-origin integral component does not yet
provide an interacting Dirac-Coulomb SCF or London-basis model.

`periodic::electron_phonon::mode_vertex<S>` contracts normalized displacement
modes and Pulay-complete complex Fock/overlap derivatives with both endpoint
energies and sqrt(2*M*Omega). Every input retains its scalar derivatives.
`ElectronPhonon::vertices[spin][k][mode]` stores the actual complex band
matrices without a Fermi-surface filter. `pairing_input` checks matching
spin-degenerate frames and assembles the existing `PairingInput` plus final
band weights for the common LM2005 K/Z provider. Explicit screened Coulomb
is required. Nk=1 represents the commensurate supercell; larger meshes
provide only the supercell-Gamma contribution with missing q nodes left
unrenormalized. Full input/report units, shared-driver controls and public
surface mappings are documented in [rust-api.md#common-calculation-driver](rust-api.md#common-calculation-driver).

### Full thermal matrix and Nambu density response

`response::thermal_matrix::fermi_matrix_taylor(H, S, kT, order,
overlap_cutoff, config)` evaluates a full complex Hermitian matrix Fermi
function with a moving positive overlap through order four. Every input
is a factorial-normalized coefficient series in one dimensionless
parameter; omitted higher coefficients mean exact zero. H already includes
the chemical potential. It returns the contravariant density, inverse
metric and raw metric, Hermiticity, complex-embedding, inverse and
commutator diagnostics. Densities and overlap are dimensionless; H and kT
are Hartree. The full reference overlap must be retained.

The implementation transports the moving metric, evaluates a scaled matrix
exponential and uses bounded logistic doubling. If F=f(X), then
f(2X)=F^2/[F^2+(I-F)^2]. The denominator spectrum is bounded below by one
half for physical occupations. Coefficient products keep their matrix
order. No eigenvector derivative, occupied/virtual gap division or
denominator floor is used, including at exact Fermi degeneracy.

`response::nambu::nambu_density_taylor(h, Delta, S, mu, kT, order,
overlap_cutoff, config)` overlays particle/hole space on the supplied full
normal electronic space. Spin is already included in its indices and
Delta must be antisymmetric under ordinary transpose. Nambu ordering is
all electrons followed by their conjugate holes. The operator is
H_N=[[h-mu*S,Delta],[Delta^H,-(h-mu*S)*]], S_N=diag(S,S*).
`NambuTaylor` retains the full density R, normal density R_ee and anomalous
density chi=-R_eh=<c_i c_j>, electron number Tr(S*R_ee), and original
particle/hole, anomalous-antisymmetry and imaginary-number residuals.
Particle/hole covariance is R+tau_x*R.conjugate()*tau_x=S_N.inverse().

This is a grand-canonical kinematic density component on explicit supplied
normal and pairing operators, with general complex interband pairing.
It does not infer a self-consistent normal/pairing functional or canonical
gap constraint. A Dirac provider must supply its intended balanced no-pair
electronic space; doubling an unprojected negative-energy spectrum is not
a no-pair construction. The normal and anomalous density formulation is
described in the [primary finite-temperature superconducting density
formalism](https://arxiv.org/abs/1903.01516). Independent acceptance and
public-driver integration are tracked separately from the existing
band-decoupled self-consistent gap model.

<!-- scalar-london-api:begin -->
### Scalar finite-field London integral kernels

`integrals::london::ScalarFieldPath<S>` carries electric and magnetic fields,
gauge origin, London/common-origin choice, `phase_order` (total retained field
degree, zero through 24), and `derivative_order` (the total scalar nilpotent
order, including all nested directions). These two orders are independent:
the latter is required by the analytic Boys expansion. Coordinates and fields
use Bohr and atomic field units, exponents use Bohr^-2, and charges use e.

`primitive_field_pair(bra,ket,nuclei,field,exact_zero)` returns five complex
scalar pairs: overlap, core Hamiltonian, then x/y/z position integrals.
The primitives are unnormalized and keep their supplied angular powers,
exponents and centers. Core includes kinetic energy, point nuclear attraction,
and electric, orbital-paramagnetic and diamagnetic couplings. Nuclei are
`(center,charge)` scalar tuples. `primitive_field_repulsion(primitives,kernel,
field,exact_zero)` returns one complex Coulomb or range-separated quartet.
All four endpoints, the nuclear positions/charges, exponents and fields retain
their complete scalar jets. No same-q Hermitian projection is imposed.

`exact_zero` is a scalar-algebra callback that may report true only when
every coefficient is zero. For `dual::Taylor<N>`, check all `c` entries;
checking only `value()` would erase moving-center derivatives at zero phase.
Returning false conservatively is algebraically safe but increases work.
Finite-field phase truncation must be converged independently of derivative
order. The existing finite-field one-electron gradient contractions and
contracted shell quartets reuse these same polynomial/primitive kernels.

Independent acceptance compares orders one through four with closed complex
Gaussian moments and an entire complex Boys series, using four separate
geometric step sizes and recorded error fits. This kernel component is not
a completed magnetic-field nuclear Hessian or cubic/quartic force constant:
the complex self-consistent density, metric and stationary-response connection
must also be accepted. The existing bfield-gradient driver remains the
connected physical entry while those higher nuclear orders are completed.
<!-- scalar-london-api:end -->

<!-- finite-magnetic-nuclear:begin -->
### Nuclear response of a converged finite London-field reference

`response::nuclear::magnetic_nuclear_taylor(molecule,basis,functional,grid,
scf_options,magnetic_options,reference,direction,order,config)` returns the
Taylor coefficients of energy and dipole through nuclear order 1-4.
`direction[atom][axis]` is Cartesian; coefficients multiply t^n, so multiply
by n! for a directional derivative. Fields stay fixed in laboratory axes.
Moving metric, nuclear attraction, Coulomb/RSH exchange, atom-owned Becke
weights and semilocal LDA/GGA/meta-GGA follow the same Scalar/complex-response
kernels. Restricted references share one spatial gauge. The finite-field
SCF must retain London degree at least order+2 for nonzero B, and its phase
tail must be converged independently. The common driver estimates a larger
degree before building the reference, reusing this entry for hessian,
cubic-fc-along and quartic-fc-along with optional bfield and field controls.
Native Python uses hessian_report/cubic_fc_along/quartic_fc_along; ASE uses
get_hessian_report/get_cubic_fc_along/get_quartic_fc_along. The direct legacy
native hessian function does not acquire these controls.

The result reports fresh response residuals, projector/stationarity defects
and a reference mismatch check. No finite difference enters production.
This bounded dense integral-jet path uses integer occupations and currently
requires separate ECP, solvent, nonlocal/learned, density-fitting, extra-core
and correlated PT2 providers. It does not establish periodic magnetic or
arbitrary-q DFPT nuclear response. Independent physical/public acceptance is
recorded in response.md and the coverage CSV, not inferred from this interface.
<!-- finite-magnetic-nuclear:end -->

<!-- magnetic-nuclear-operator-api:begin -->
### Moving magnetic-translation one-electron operators

`periodic::magnetic_cell::MagneticCell::nuclear_operator_taylor(basis,
cutoff,wave,operator,direction,order)` differentiates the same continuum
magnetic pair contraction used by `fourier`, `kinetic` and `momentum`.
`MagneticPairOperator` is `Fourier`, `Kinetic`, or `Momentum(axis)` with
axis 0-2. Order is zero through four. Directions are Cartesian Bohr per
path parameter on each atom; every image of an atom moves with it.
The result is a vector of complex spherical-AO Bloch matrices multiplying
t^n; multiply by n! for the nth derivative. Fourier coefficients are
dimensionless, kinetic coefficients Hartree, and momentum coefficients
atomic momentum, with the corresponding path-parameter powers.

The field, flux-compatible cell, magnetic Bloch wave, reciprocal Fourier
wave and ordered magnetic-translation cocycle stay fixed. Translation
images are selected at the reference geometry and their cutoff must be
converged independently. Every ordered AO matrix element is evaluated;
no same-wave Hermitian projection is made. The scalar Fourier primitive
keeps the moving London wave and all centre phases. `dual::Taylor<W>` now
provides `sin_cos()` for curved paths through arbitrary retained order,
using the coupled exact recurrences s'=x'c and c'=-x's.

This is an integral component. It does not by itself include the nuclear
Coulomb energy, relaxed density, correlated amplitudes, finite-electric
Berry enthalpy or finite-q DFPT. Full periodic nuclear derivatives require
those additional shared providers and their independent public acceptance.
<!-- magnetic-nuclear-operator-api:end -->

<!-- magnetic-relaxed-nuclear-api:begin -->
### Relaxed magnetic-torus nuclear energy and direct variational gradient

`MagneticDftBackend::nuclear_energy_taylor(reference,direction,order,config)`
returns MagneticNuclearTaylor: energy coefficients of t^n through order 1-4,
solved_response_order (0/1/1/2 via the static 2n+1 rule),
response_residuals, reference_defect, reference_energy_defect,
projector_defects and stationarity_defects. The fixed cell/flux spinor
occupied subspace relaxes through the common complex response recursion.
Only actually solved orders appear in response/stationarity diagnostics;
the higher trial projector coefficients retain full metric/idempotency.
No dipole or high-order relaxed density is inferred from the energy-only path.
The finite_gamma wrapper MagneticBlochBackend exposes the same method;
crystal_BZ intersector response requires its independent geometry provider.

`MagneticDftBackend::nuclear_energy_slope(reference,direction)` independently
contracts fixed-density integral derivatives and the Pulay term
-Tr[P F P S']. It solves no density-response equation and shares the
physical geometry/energy provider with the Taylor recursion. It supports
the independent displaced-SCF gradient verification of higher orders.
Directions have one Cartesian triplet per source atom; for dimensionless
directions the path parameter is Bohr and energy[n] has Hartree/Bohr^n.
The same finite-Gamma point-source/full-or-range-exchange/generated-SOC
restrictions and pending combinations as docs/rust-api.md#common-calculation-driver apply. No result here
is a complete arbitrary-field, arbitrary-q or all-functional claim.
<!-- magnetic-relaxed-nuclear-api:end -->

The existing `response::nuclear::ContractedForceConstants` now also returns
`response_residuals`: the fresh relative residuals of the actually solved
equations on every distinct cached path. Molecular and periodic contracted
cubic/quartic common-driver reports expose this same dimensionless vector.

<!-- joint-magnetic-nuclear-rust-api:begin -->
### Electric enthalpy on a continuum magnetic reference

MagneticScfOptions.electric_field is a Cartesian [f64;3] bias in
Hartree/(e Bohr), default zero. The 3D finite-Gamma MagneticDftBackend
adds the single-point Berry electric enthalpy to its variational Fock
and energy_components[8]. MagneticBlochBackend exposes this same finite
model; magnetic crystal integration still requires coupled Berry strings.
MagneticDftBackend::electric_dipole and MagneticBlochBackend::electric_dipole
return an optional dipole [f64;3] in e Bohr on the local determinant phase continued near the electron-number-weighted nuclear charge centroid. The shared real/complex functional uses the actual SCF overlap in Q-S, including every moving-metric coefficient; independent phase-cut and short-image translation checks are recorded in response.md.
The existing nuclear_energy_taylor and nuclear_energy_slope include the
electric term and its moving metric/operators with both fields fixed.

SpinorEnergyBackend::validate_occupation(OccupationScheme) is a provider
check called by run_spinor_scf; its default accepts ordinary SCF ensembles.
The Berry-biased provider requires an integer insulating occupied space.
The common driver applies the same requirement on every public route.
The compensation, branch, units and remaining providers are documented
with the three corresponding entries in docs/rust-api.md#common-calculation-driver.
<!-- joint-magnetic-nuclear-rust-api:end -->

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 ss0832 -->

## Shared calculation reference

Every public calculation is listed in the three aligned tables in the Rust, CLI and Python manuals
(Rust, CLI, native Python and ASE); the IDs are identical in all three and
match `driver::registry`, which a unit test enforces. A row describes an
available calculation, not completion of its whole theory family; the
remaining objective is tracked in the [response coverage ledger](response.md#response-coverage-ledger)
and [property_targets.csv](property_targets.csv) (interpretation in
[scope.md](scope.md#13-physical-target-registry)). Theory and validation are in
[response.md](response.md) and [periodic.md](periodic.md); the earlier ground
state, derivative and optimization APIs are in [rust-api.md](rust-api.md),
[cli.md](cli.md) and [python-api.md](python-api.md).

## Contents

- [Common calculation driver](#common-calculation-driver)
- [Table 1: Rust](#table-1-rust)
- [Table 2: CLI](cli.md#table-2-cli)
- [Table 3: native Python and ASE](python-api.md#table-3-native-python-and-ase)
- [Result conventions and physical scope](#result-conventions-and-physical-scope)
- [Analytic foundation APIs](#analytic-foundation-apis)
- [Orbital stability and following conventions](#orbital-stability-and-following-conventions)

## Common calculation driver

`dft_rs::driver` runs registered calculations the same way on every surface:

| Surface | Call |
|---|---|
| Rust | `driver::run(feature, &System, &Controls) -> Result<Value>` |
| CLI | `dft-rs <feature> structure.xyz [--control value ...] [--format text\|json] [--save report.json]` |
| Python | `dft_rs.run_feature(feature, numbers, positions, *, cell=None, pbc=None, **controls)` and named wrappers |
| ASE | `DftRs.get_feature_report(feature, atoms, **controls)` and named `get_*` methods |

The [feature registry](rust-api.md#common-calculation-driver) (`driver::registry`) lists every public
calculation with its Rust, CLI, Python and ASE names and its physical target
IDs. `feature-registry` returns the same rows as data. A unit test keeps the
three aligned tables in the surface manuals identical to the registry, and the Python
suite checks that each named function and ASE method exists.

### Requests

* `System`: atomic numbers and positions in Bohr; for a crystal, lattice rows
  in Bohr and periodic axes. The CLI reads XYZ files and `--cell` in Angstrom;
  ASE converts `Atoms` from Angstrom.
* `Controls`: named values. CLI `--grid-level 4`, Python `grid_level=4` and the
  JSON key `"grid_level"` are the same control. Values may be numbers,
  booleans, strings, lists (`--field 0,0,0.01`) or lists of rows
  (`--cell "5,0,0;0,5,0;0,0,5"`). **Every control that the calculation does
  not read is refused by name**, so a misspelling or an inapplicable option is
  an error rather than a silently ignored input.

#### Reference controls of molecular calculations

`xc` (or `x` with `c`), `omega`, `basis`, `ecp`, `charge`, `multiplicity`
(alias `mult`), `unrestricted`, `guess_mix`, `grid_level`, `dispersion`
(alias `disp`), `density_fit`, `max_iter`, `conv_tol`, `conv_tol_grad`,
`conv_tol_dp`, `incore_mb`, `vv10_grid_level`, `field` (uniform field, a.u.),
`solvent`, `eps`, `guess`, `level_shift`, `damping`, `diis_size`, `no_adiis`,
`no_soscf`, `stabilize`, `electronic_temperature` (Kelvin) and
`electron_offset` (fractional alpha/beta electrons; needs a positive
temperature). Grammar and defaults are those of the earlier CLI and Python
entry points.

`xc_parameters` is an object of functional parameter names and values
(`ResolvedFunctional::with_parameters`): every part of the selected
functional that has a parameter of that name takes the value, and a name no
part has is an error. `xc::catalogue` and `FunctionalInfo` list the
functionals; the parameter names of a part are those of
`XcFunctional::param_names`, for example `tau_r`, `alpha_r`,
`switch_lower` and `switch_upper` of rSCAN, whose published switching
interval of the polynomial interpolation is `[0, 2.5]` (`[1, 1]` gives the
infinitely differentiable SCAN interpolation used to validate its
response). Range separation keeps its own `omega` control. Molecular
driver calculations accept it; the periodic driver does not yet.

The combined registry also accepts `blyp35`: 0.35 full-range HF exchange,
0.65 B88 exchange and full LYP correlation, without an implicit dispersion
or nonlocal term. Resolve it with `FunctionalSpec::Combined("blyp35".into())`
or use the common driver's `xc` control. Energy is in Hartree and nuclear
gradients in Hartree/Bohr. Numerical acceptance is recorded in
[FUNCTIONALS.md](FUNCTIONALS.md); adding the recipe does not certify all
response/reference combinations.

`wc04` and `wp04` use the same registry/control route and the explicit
VWN-RPA convention. Their five independent fitted coefficients and bounded
energy/gradient acceptance are specified in FUNCTIONALS.md; they are not
renormalized to an exchange-complement rule.

| Provider | Common features | Model and units | Raw acceptance measurements |
|---|---|---|---|
| BLYP35 | `energy`, `gradient` | All-electron spherical STO-3G RKS/UKS, integer populations, zero temperature, no added corrections; Hartree and Hartree/Bohr | Water/methyl SCF error <=1.720e-12 Hartree; full five-molecule catalogue maximum error 9.778e-12 Hartree; 12 C1 gradient fits p=1.999584..2.001368, minimum R2=0.99999988; native/ASE/CLI maximum gradient difference 4.441e-15 Hartree/Bohr, energy difference 4.263e-14 Hartree |
| WC04 / WP04 | `energy`, `gradient` | All-electron spherical STO-3G RKS/UKS, integer populations, zero temperature, no added corrections; explicit VWN-RPA convention; Hartree and Hartree/Bohr | Water/methyl maximum SCF errors 2.871e-12 / 2.501e-12 Hartree; all 12 C1 gradient entries per model fitted; maximum public gradient differences 1.999e-15 / 6.218e-15 Hartree/Bohr and energy differences 8.527e-14 / 7.106e-14 Hartree; WP04 five-molecule maximum error 6.435e-11 Hartree; WC04 ferrocene remains unaccepted after an interrupted nonstationary run |
| PKZB | `energy`, `gradient` | All-electron spherical STO-3G RKS/UKS; integer populations, zero temperature; original seven-variable meta-GGA, half-weighted tau, FHC input convention; Hartree and Hartree/Bohr | Maximum water/methyl SCF error 3.567e-12 Hartree; 53 local partial and 12 C1 RKS gradient fits pass; public gradient/energy differences 1.582e-15 Hartree/Bohr and 2.842e-14 Hartree; five-molecule and higher response acceptance remain pending; standard H2/LiH/water/phosphine pass, maximum error 6.196e-12 Hartree; ferrocene reference reached 180-second budget; Rust comparison unrun; 104 smooth local third partial fits pass (p300), no molecular higher-order inference |
| TPSSloc | `energy`, `gradient` | All-electron spherical STO-3G RKS/UKS, integer populations, zero temperature; localized TPSS correlation with TPSS exchange, half-weighted tau; Hartree and Hartree/Bohr | 53 local partial and 12 C1 gradient fits pass; native/ASE/CLI maximum energy/gradient differences 5.685e-14 Hartree and 2.929e-15 Hartree/Bohr; all six p301 tests and both public cases pass after explicitly approved four-entry physical inward reference correction; original raw values, failed logs and thresholds retained; five-molecule and higher response acceptance pending |
| DSD-PBEB95-D3(BJ) | `energy`, `gradient` | Original 2013 typed global meta-GGA double hybrid; .66 HF/.34 PBE-X/.55 standard B95-C/.09 SS/.46 OS; automatic pairwise D3(BJ)(.61,0,0,6.2); all-electron spherical STO-3G RKS/UKS, integer populations; Hartree and Hartree/Bohr | p304: 46 local first/second/third partial fits, 72 structural zeros and 55 molecular term comparisons pass; all 24 independent C1 RKS/UKS gradient fits pass; both RKS/UKS public-route exact-node runs terminate successfully; earlier aggregate timeout trials retained in FUNCTIONALS.md; molecular Hessian/ECP/ROKS/GKS/excited combinations pending |
| Original omegaB2PLYP / omegaB2GP-PLYP (bounded validation) | `energy`, `gradient` | Typed range-separated GGA double hybrids; SR/full-LR exact exchange .53/1 and .65/1; omega .30/.27 Bohr^-1; ITYH SR-B88, LYP and full-Coulomb PT2; no implicit dispersion; Hartree and Hartree/Bohr | p305: 66 local derivative fits, 44 structural zeros, spin/range limits, 110 molecular term comparisons and all 48 C1 RKS/UKS gradients pass; both public energy cases and all four native/common/ASE/CLI gradient cases pass; all point kernels pass after the explicitly approved eight-entry physical-limit correction; original raw values/logs retained; standard five-molecule acceptance pending; no general excited-state acceptance |
| `hsesol`, `wb97x-2-lp`, `wb97x-2-tqz`, `xdh-pbe0` | `energy` | All-electron RKS H₂/STO-3G at 1.4 Bohr. Grid level 9, zero grid weight cutoff. Double hybrids use full-Coulomb PT2. Total energies and raw orbital residuals in Hartree. | Independent four-model totals and double-hybrid spin terms pass. Maximum total error 2.834e-9 Hartree and raw orbital residual 1.301e-16 Hartree. All three PT2 amplitude residuals are zero. [Definitions and energy values](FUNCTIONALS.md#screened-and-xdh-energy-checks) |
| TASK / mTASK | `energy`, `gradient` | All-electron spherical STO-3G RKS/UKS, integer populations, zero temperature; published exchange plus original PW92 correlation, half-weighted tau, FHC input convention; Hartree and Hartree/Bohr | 107 local derivative/parameter fits and 24 C1 RKS gradient fits pass; maximum SCF error 2.416e-13 Hartree; four native/ASE/CLI cases pass, maximum energy/gradient route differences 5.684e-14 Hartree and 2.665e-15 Hartree/Bohr; full local fifth tests pass 84 exchange plus 6 shared correlation entries at one smooth point, retaining 21 failed exploratory measurements; dense cap remains 3; five-molecule, independent UKS nuclear, ECP/ROKS/GKS and excited-state Hessian combinations remain separate |
| EDF2 (bounded validation) | `energy`, `gradient` | Global GGA hybrid, published independent weights and0.1695 HF; all-electron spherical STO-3G RKS/UKS, atomic units | New library/CLI build and portable audit pass;48 local fits, RKS/UKS terms and2 native/ASE/CLI cases pass; all12 C1 RKS gradients and78 independent common-driver Hessian fits pass; local component fifth74fits/442zeros pass; all point kernels pass after the explicitly approved six-entry physical-limit correction; original raw values/logs retained; all five standard molecular energies pass including Fe40 iterations/error1.751e-11 Hartree at unchanged gates; public Hessian/ECP/ROKS/GKS/excited combinations remain separate |
| N12-SX (bounded validation) | `energy`, `gradient`, `hessian` | Screened hybrid GGA, whole fitted local polynomial; HF0.25 short range/zero long range, omega0.11; spherical STO-3G, atomic units | Actual2a5b3ad9 Python/lib/CLI build and portable audit pass; independent full local kernels,86 first-third fits, fourth45fits/95zeros, fifth68fits/184zeros and RKS/UKS term cases pass; all12 C1 RKS gradients and78 independent Hessian fits pass; all4 RKS/UKS native/common/ASE/CLI gradient/Hessian route cases pass; four standard molecular energies pass, Fe reference active; ECP/ROKS/GKS/excited and general mixed-provider combinations remain separate |
| dlDF / B98 / tau-HCTH / hybrid tau-HCTH (bounded validation) | `energy`, `gradient`, `hessian` | dlDF global HF0.6144129 with fitted meta-GGA X/PW92-MOD C and no added dispersion; B98 fit2c HF0.219847 with original-PW92 B97 form; tau-HCTH pure/hybrid HF0/0.15 with shared kinetic exchange and independently fitted correlation | Actual84bd1e14 Python/library/CLI warning-strict build and portable audit pass; independent component points,432 first-third local fits and eight RKS/UKS term cases pass; complete interior fifth380 nonzero fits/1846 zeros and weighted composition checks pass; all270 independent C1 RKS gradient/Hessian fits for dlDF/tau-HCTH pair and all8 RKS/UKS native/common/ASE/CLI gradient cases pass; standard16/20 accepted with4 Fe cases in finite queues; no separate B98 nuclear-Hessian or universal UKS Hessian/excited/ECP/ROKS/GKS claim |
| RSX-QIDH / RSX-0DH (bounded validation) | `energy`, `gradient` | Range-separated global-LR double hybrids; QIDH SR-HF3^(-1/3), omega0.27, PT2=1/3; 0DH SR-HF0.5, omega0.33, PT2=0.125; full LR HF and full-Coulomb same/opposite-spin PT2; no dispersion | Actual8e512c3f library/CLI audit passes;110 local first-third fits,110 independent molecular terms and two public energy cases pass; all48 C1 RKS/UKS gradient fits pass, with one retained failed coarse interval resolved at unchanged gates; all four public-gradient cases pass; ferrocene/Hessian/excited/ECP/ROKS/GKS combinations remain pending |
| PBE1KCIS / mPW1KCIS (bounded validation) | `energy`, `gradient` | Global exact exchange fractions 0.22/0.15 with PBE/mPW91 and full KCIS correlation; molecular STO-3G RKS/UKS, energies in Hartree and gradients in Hartree/Bohr | Independent local fits and four molecular energy decompositions pass in the linked-density harness; raw occupied–virtual residual at most 2.603e-11 Hartree. [Model, term energies and acceptance scope](FUNCTIONALS.md#kcis-correlation-and-global-hybrids) |

Integer determinant selection additionally accepts `occupation_mode`,
`occupied_alpha` and `occupied_beta`; see the
[occupied-projector contract](#occupied-projector-delta-scf-controls).

#### Response solver controls

Every response calculation on the driver reads the same controls:

| Control | Default | Meaning |
|---|---|---|
| `response_tolerance` | `1e-10` | Iteration stops at this relative residual (CPKS routes: absolute residual per amplitude). |
| `response_max_iterations` | `500` | Iteration budget. |
| `response_restart` | `40` | GMRES restart length. |
| `response_accept_residual` | `response_tolerance` | Acceptance bound of the freshly recomputed residual of the original operator. A solve that exhausts its budget within this bound is accepted; its residual is reported. |
| `response_divergence_factor` | `1e8` | Stops a solve whose residual grows by this factor over the smallest residual reached. |
| `response_min_denominator` | `0` (off) | Refuses a response whose zeroth-order denominator (orbital gap or preconditioner diagonal) is below this value, instead of regularizing a near-singular or resonant response. |
| `response_preconditioner_floor` | `1e-3` | Floor of the preconditioner only; the operator and its physical denominators are never changed. |

### Reports

`run` returns a `Value` tree (`driver::value`). Every report carries
`feature`, `version`, `model`, `units` (key to unit name), `converged` and
`warnings`; molecular reports add the reference (`functional`, `basis`,
`scf_iterations`, `unrestricted`, `n_alpha`, `n_beta`, `s_squared`, `energy`,
`energy_components`, optional `stability_following`). Values are atomic units
unless the key names another unit (`*_debye`, `*_cm1`, `*_km_mol`,
`*_angstrom3`). Complex numbers are `[real, imaginary]` pairs. Raw residuals
are reported as measured, before any symmetrization. The JSON writer emits
the shortest round-trip representation of each number, and `--save` writes
the same JSON as `--format json`.
The molecular static-reference API also exposes
`driver::method::CollinearPropertyReference::{Auto,Rks,Uks,Roks}` with
`from_controls` and `name`, and
`driver::method::run_collinear_property_reference(setup, kind)` returning
the same `MolecularReference` and additive energy terms. For full complex
states, `response::spinor::MolecularSpinorBackend::new` implements
`periodic::spinor_scf::SpinorEnergyBackend`; select
`MolecularSpinorFunctionalModel::{SpinCovariant,SpinDiagonal}` explicitly,
then call `run_spinor_scf` to obtain `SpinorScfReport`. `functional_model()`
identifies the energy extension. `response::nuclear::spinor_nuclear_taylor`
accepts that solver's full `SpinorScfReport`, a resolved functional, SCF
options, fixed additional AO operator and nuclear direction; it returns
factorial-normalized analytic `FieldPathTaylor` energy coefficients and raw
projector/stationarity/response diagnostics. The present nuclear provider
covers Coulomb, full/range exchange and globally projected semilocal XC;
nonlocal, solvent and ECP path providers remain pending. Every matrix is in spherical AO order and
atomic units. `properties::lowdin_density_charges(molecule, basis,
real_total_density, overlap)` shares Loewdin analysis between real and
complex-state providers. [Model and validation details](response.md#molecular-complex-spinor-reference-provider).

The Rust providers are `response::nuclear::{spinor_nuclear_taylor,
spinor_nuclear_energy_taylor, SpinorNuclearEnergyTaylor,
correlated_molecular_taylor, MolecularCorrelatedTaylor}`. The spinor
constructors take the supplied molecular grid as well as the converged
spinor reference. The energy-only provider uses the existing 2n+1 theorem;
the complete provider retains all density orders. `driver::nuclear_tensor::
assemble(atoms, order, channels, evaluator)` returns `NuclearTensor`
(energies, flattened gradients, Hessian matrices and path count), sharing
one complete Cartesian assembly across spinor and correlated providers.
Its evaluator returns factorial-normalized energy paths per channel.

`SpinorNuclearContext::new(molecule,basis,grid,functional,reference,
options,operator,max_order,config)` binds the immutable physical reference;
`energy_taylor(direction,order)` shares its descriptor-space kernels across
Cartesian paths. Storage scales with grid points and active descriptors,
not AO count to the fourth power. The `incore_mb` budget bounds this cache;
larger cases retain on-demand matrix-free evaluation. Current and previous
angular rules remain available for actual path-coefficient comparisons.

`XcFunctional::taylor_directional_potentials` and
`ResolvedFunctional::semilocal_taylor_directional_potentials` return
`XcTaylorDirectionalOutput { value, directional }` with energy and all
seven potential paths. Directions may vary along the path; nested duals
differentiate them independently.
External providers can use `XcTaylorInput::evaluate_directional_potentials`
with `xc::taylor::DirectionalTaylorScalar` to instantiate their own generic
energy body without duplicating the mixed derivative machinery.
`periodic::spinor_xc::{
spinor_semilocal_point_taylor, spinor_semilocal_energy_taylor,
SpinorXcPointTaylor}` project the 32 Pauli descriptor paths at explicit
even angular order 4..64. Taylor order and angular convergence are distinct;
the nuclear driver refines the integrated coefficients and returns the
raw angular history. No finite displacement is used by these providers.


## Complete nuclear tensors and reference constraints

The common `gradient`/`hessian` calculations accept explicit `reference`
constraints and HF aliases as described in [response.md](response.md#full-nuclear-tensors-for-generic-double-hybrids).
GKS uses complete complex spinor density blocks; the semilocal nuclear path
contains moving Becke weights, all Pauli density/gradient/tau/current terms
and independently refined angular quadrature. Additional supplied AO spin
operators are held fixed along a nuclear path; no missing SOC integral
derivative is implied. Parent nonlocal, ECP and solvent kernels retain
their documented scope. ROKS/GKS PT2 is not inferred from parent support.

For `double_hybrid_model="custom"`, `xc` selects the stationary parent
and optional `double_hybrid_target_xc` selects a distinct semilocal or
hybrid target evaluated on those orbitals. Both use the same grid settings;
a target grid is built even for an HF parent. The full parent orbital
response is retained for this nonstationary target. Named recipes fix
their target and refuse an override. Additional target nonlocal terms
remain a separately recorded provider limitation. The common driver uses
this control on Rust, native Python and ASE; the CLI spelling is
`--double-hybrid-target-xc`. For example, `xc="hf"`,
`double_hybrid_model="custom"`, `double_hybrid_target_xc="pbe"`,
`pt2_same_spin=0.37`, `pt2_opposite_spin=0.61` defines PBE + weighted PT2
on HF orbitals. All 312 total/target derivative component fits pass
(p=1.817002..2.181338, minimum R2=0.99919170). The common driver agrees
bitwise on all total/target tensor entries, and the final 36-case
native/ASE/CLI/package suite passes with warnings treated as errors.
This defines a supplied model, not a fitted named functional.

Double-hybrid `gradient` and `hessian` use the existing energy API with
`xc` or explicit custom PT2 coefficients. Every gradient component and
Hessian element is returned. `parent_`, `target_`, `pt2_`,
`pt2_same_spin_alpha_`, `pt2_opposite_spin_`, `pt2_same_spin_beta_` and
`dispersion_` prefix complete `gradient`, `hessian` and `nuclear_energy`
outputs. Unscaled spin channels reconstruct weighted `pt2_`; total equals
target + PT2 + dispersion. Units are Hartree, Hartree/Bohr and
Hartree/Bohr^2, with axes [atom][xyz] and [3*atom+axis][3*atom+axis].
Reports include orbital/amplitude response, projector, stationarity,
metric, number and reference residuals. Top-level `converged` requires
these nuclear diagnostics as well as SCF convergence. The new GKS and
correlated Hessian providers return complete Cartesian energy derivatives;
they do not yet generate the additional dipole derivatives, APT or IR
intensities supplied by the existing collinear vibrational driver. Correlated nuclear full-component
fits for four custom parents and the 24-case public suite pass, including
B2PLYP, DSD-BLYP, DSD-PBEP86 and PWPB95 nuclear tensors on native/ASE
and B2PLYP on canonical CLI JSON/plain text and legacy native entries.
Complex PBE/PBE0 nuclear and all four ROKS-class tensor fits pass;
complex TPSS remains unvalidated after failing the order-64 angular gate. `nuclear_xc_cache_bounds`
contains conservative bounds separately from the actual coefficient
changes in `nuclear_xc_angular_history`; physical residual gates are unchanged. Exact scope and failed/resolved
intervals are recorded in response.md. Native `hessian(..., reference=...,
**controls)` routes to this analytic common driver without numerical fallback.
ROKS accepts common response controls on energy/gradient requests as well
as Hessian/polarizability requests. They are validated in every case; only
the latter two require orbital-response equations. ROKS Hessian convergence
also includes its accompanying electric-response calculation.

## Table 1: Rust

[Common requests and report fields](#common-calculation-driver) / [Choose a calculation by property](api-guide.md#find-a-calculation) / [Match the same ID to the CLI and Python](api-guide.md#choose-an-interface)

Molecular arguments `mol, mb, functional, grid, scf, opts` are respectively
`&Molecule`, `&MolBasis`, `&ResolvedFunctional`, `Option<&MolecularGrid>`,
`&ScfResult`, and `&ScfOptions`. Reuse the actual converged reference and its
functional, geometry, basis, integration grid, and calculation options.

The shared collinear reference report carries `scf_orbital_residual`
in Hartree, using the common-orbital residual for ROKS. The standard
molecular GKS report carries `orbital_residual` in Hartree.

| ID | Physical calculation | Entry point and request | Successful return |
|---|---|---|---|
| `continuous-species` | Fixed-geometry all-electron LDA/PBE with real nuclear charges, independent integer electron count and fixed AO/grid templates. | `driver::run("continuous-species", ...)`; `driver::continuous_species::{ContinuousSpeciesModel,ChargeChart,ChargeParameter}` | Analytic parameter gradient/Hessian, dipole/Jacobian, static alpha and optional moving-template nuclear gradient. Closed-shell M1, fixed-spin UKS and local operators pass their stated numerical/public gates; generated frozen-Me candidates additionally use `pseudo_sub::reduction::generate_methyl` and `continuous_species::reduced::evaluate_candidate`, with chemical accuracy unestablished. See response.md.  The common charge, fixed few-center and generated-reduced model reports also return `energy_without_internal_reference` and `classical_energy_without_internal_reference` in Hartree, plus `scf_energy_convention`; state-common constants are added only to reported totals.  Experimental `scalar-evaluate`/`scalar-optimize` add explicit full frozen joint-endpoint actions and site weight optimization; see the model contract below. Saved weight routes pass clean installed checks. `scalar-gradient`/`scalar-geometry-optimize`/`scalar-hessian` require explicit endpoint/state owners; paired RKS and explicitly spin-resolved UKS records remain distinct. Hessian reports include all ordered entries and raw residuals; the stated PBE RKS/UKS fixture and installed-selector gates pass, while real-model minima remain separate. |
| `multicomponent` | Experimental simultaneous electronic/quantum-particle SCF, expectation-position constraints, analytic coupled gradient/Hessian and total static polarizability; `species_stability = report \| follow` gives the lowest eigenvalues of the coupled orbital Hessian (Davidson, one half of the energy Hessian, Hartree) and follows a negative mode with re-convergence until the reference is a local minimum (`interspecies_correlation` epc17, epc17-2 or none). Laboratory center-of-mass kinetic energy is retained; intrinsic vibronic theory remains pending. | `driver::run("multicomponent", ...)`; `multicomponent::{solve,solve_from,gaussian_particle_basis}`; `multicomponent::stability::{lowest_curvatures,follow}`; `constraints::solve`; `response::CoupledResponseSolver`; `derivatives::{directional,cartesian,hessian}` | Disjoint physical energies, quantum states, optional position constraints, analytic geometry tensors and raw coupled residuals. Structured particle/channel and spin-averaged annihilation contracts, units and bounded acceptance are below.  Optional `quantum_mass_response` uses `multicomponent::mass::{gradient,response}` for fixed-basis mass tensors and raw coupled diagnostics. Select `species_electronic_model="dirac-coulomb"` and per-atom `nuclear_rms_radii` in Bohr for the fixed-sector Dirac–Coulomb HF energy with distinguishable quantum nuclei. The report gives the projector reference and raw orbital, density, population and Coulomb reciprocity residuals. `JointNoPairSystem::solve_in_field` and `electric_response` use `JointElectricOperators` at the laboratory origin. For the Dirac joint reference, `field` gives total dipole (e Bohr) and `species_polarizability` returns static or complex alpha (Bohr^3), signed frequency and damping (Hartree), and raw orbital and solve residuals. For a nonrelativistic molecular positron, `field` perturbs the SCF and `species_polarizability` returns the static dipole tensor (Bohr^3) and raw response residuals. [Positron benchmark and conditions](response.md#r076-multicomponent-nuclear-positron-photon-dft). With `species_electronic_model=dirac-coulomb`, `dirac_interaction=coulomb\|gaunt\|breit` selects the electron-electron operator for the coupled SCF and electric response. The current direct and exchange terms and Hermiticity defect are in Hartree. [Model and validation](response.md#r081-relativistic-quantum-nuclear-response).  In the fixed-sector relativistic quantum-nuclear model, `species_field_gradient` supplies five traceless symmetric gradient modes (Hartree/e/Bohr^2), and `species_quadrupole_polarizability` requests their coupled response. Named or explicit `origin` is supported, default zero. `electric_quadrupole` reports Buckingham moments (e Bohr^2), static/complex 5x5 response (e^2 Bohr^4/Hartree), and raw conjugate dipole-gradient 3x5/5x3 blocks (e^2 Bohr^3/Hartree); multiplying the latter moment rows by three gives Buckingham response. Eight source residuals, iterations, reference residual, mode order and signed frequency are exposed. Gaussian-product, full derivative and public acceptance are pending. Nonrelativistic electric additions use `CoupledResponseSolver::dynamic_polarizability` and `electric_quadrupole`. See the [model and verification status](response.md#r076-multicomponent-nuclear-positron-photon-dft). The nonrelativistic `multicomponent::solve_in_field_gradient` and `constraints::solve_in_field_gradient` use `QuadrupoleField` for the fixed-geometry reference. [Scope and biases](response.md#r076-multicomponent-nuclear-positron-photon-dft). |
| `casscf` | Complete-active-space SCF: string-driven determinant CI in the active space (dense below 400 determinants, Davidson above), spin-selected lowest states, state-averaged orbital optimization by preconditioned limited-memory BFGS with energy backtracking; nonrelativistic all-electron Hamiltonian with an in-core integral tensor and a closed-shell core. | `driver::run("casscf", ...)`; `mcscf::casscf::CasProblem::{new,evaluate,optimize,rotate}`; `mcscf::ci::CiSpace::{sigma,lowest,rdm12,spin_squared}` | State energies and average (Hartree), natural occupations, `<S^2>`, CI residuals (Hartree), largest orbital gradient (Hartree), macro iterations and energy history. Orbital gradient and full-space CI acceptance in `p230_casscf_mcpdft`; `derivative_order = 1/2` adds analytic gradients/Hessians (`mcscf::path::pdft_derivatives`; transported-frame orbital and CI response through the requested order on the moving integrals). |
| `mcpdft` | Multiconfiguration pair-density functional theory: one-electron energy, classical Coulomb energy and an on-top functional of the density and on-top pair density of the active-space reference; translated (`t`) and fully translated (`ft`) local and gradient-corrected parents, hybrid fraction, state-specific, linearized (L-PDFT), compressed (CMS-PDFT) and extended (XMS-PDFT) multistate forms. `derivative_order = 1/2` adds analytic gradients and Hessians of the state-specific (hybrid) energies with the complete orbital and CI response along nuclear Taylor paths (`mcscf::path::{pdft_path,pdft_derivatives}`); multistate derivatives, meta-GGA translation and periodic cells pending. | `driver::run("mcpdft", ...)`; `mcscf::pdft::{OnTopFunctional::parse,translate,on_top_energy,state_energy,multistate}` | Final energies (Hartree, ascending), per-state one-electron, Coulomb, on-top, PDFT, active-space and hybrid energies, grid electron and on-top pair counts, intermediate rotation, effective Hamiltonian and eigenvectors, linearized expansion energies, Coulomb objective. Energy acceptance: p230 (8 tests), p232 (9 frozen total-energy cases and 20 term comparisons; maximum term defect 4.418e-7 Hartree). Native gradients/Hessians use Hartree/Bohr and Hartree/Bohr^2; ASE uses eV/Angstrom and eV/Angstrom^2. Axes are [state][atom][axis] and [state][3 atom + axis][3 atom + axis]. derivative_stationarity is the largest remaining orbital/CI residual in Hartree; derivative_solve_residual is dimensionless. The transported orbital/CI Taylor path uses the common matrix-free GMRES and orthogonal retraction. Local Hessian grid convergence and open-active GGA derivative combinations require their recorded acceptance in response.md. Four CASSCF/MC-PDFT public-route cases pass, including gradients/Hessians and atomic/ASE conversions (path stationarity 3.641e-12 Hartree, solve residual 3.380e-13). MC-PDFT derivative state axes follow state_total in reference-state order, while pdft_energies are sorted; CASSCF derivatives follow cas_energies. |
| `superconducting-state` | Canonical static complex singlet gap, chemical potential and phase response on supplied normal bands | `driver::run("superconducting-state", ...)`; `response::superconducting::{GapFunctional,Lm2005BandFunctional,solve_gap_taylor}` | Gap, anomalous density, occupation, mu and diagonal Z Taylor coefficients through order four; `pairing_model="lm2005"` generates phonon K/Z together; raw physical and GMRES residuals; measured acceptance in response.md `pairing_model="nambu"` evaluates full complex interband normal/anomalous density on supplied H/Delta/S, at positive temperature and supplied chemical potential. This model is grand canonical and kinematic; the existing supplied/lm2005 band providers retain canonical gap/number/phase self-consistency. The bracketed canonical onset uses `response::superconducting::critical_temperature` with bracket energies in Hartree. The common control `scdft_temperature_bracket=[cold,hot]` uses Kelvin. It returns the crossing, final bracket, normal-state chemical potential and original gap/eigenvector/number residuals. The fixed-band variational phase selector uses `response::superconducting::select_band_phase` with reciprocal supplied K and uniform Z. It reports the selected phase, condensation free energy and variational stability margin at fixed electron number and temperature. The full retarded path is also available as `response::nambu::nambu_causal_response`. It uses fixed overlap and occupations, signed frequencies, optional damping and a supplied symmetric curvature in Hartree. Dimensionless normal and pairing probe matrices give complex susceptibility in inverse Hartree with axes frequency, observed, driven. [Model and verification](response.md#r050-scdft-gap-anomalous-density). |
| `no-pair-projector` | Electronic-sector projection of supplied balanced H/S or generated finite-Gaussian-nucleus molecular integrals. Use `rkb`, common-origin `rmb` or `london-rmb` for the one-electron basis. | `driver::run("no-pair-projector", ...)`, `properties::no_pair::no_pair_projector_taylor`, `integrals::dirac::london_magnetic_balance` | Factorial-normalized complex H/S and no-pair coefficients, sector gap (Hartree), raw Hermiticity, metric and commutator defects. London NH3/STO-3G maximum metric and relative commutator defects are 2.220e-16 and 3.924e-17. [Models and units](#restricted-magnetic-balance-and-a-complex-four-component-metric). |
| `dirac-coulomb-scf` | Molecular four-component finite-Gaussian-nucleus Dirac–Coulomb HF in the fixed positive nuclear-core sector. The complete restricted-kinetic-balance scalar Coulomb and exchange terms are retained. | `driver::run("dirac-coulomb-scf", ...)` and `properties::no_pair::stationary::DiracCoulombBackend` | Rest-subtracted energy and components in Hartree, Gaussian radii in Bohr, orbital and Coulomb Hermiticity residuals in Hartree, and raw metric, density and population defects. Use `dirac_interaction` to select Coulomb, Gaunt or Breit. Current direct and signed exchange energies and raw Hermiticity defects are in Hartree. `DiracCoulombBackend::new_london` uses a London-RMB basis and the positive sector at the selected uniform field. The same London-RMB basis provides optional Gaunt and Breit direct and exchange corrections at finite field. |
| `two-component-relativistic` | Finite-Gaussian-nucleus X2C, DKH2 or finite-basis matrix ZORA with scalar or spin-orbit operators and a fixed field-free picture. The lifted balanced density supplies Coulomb, full and long-range exchange, semilocal XC and required self-consistent VV10. The resolved functional can also require geometry-only CHG. | `driver::run("two-component-relativistic", ...)`, `properties::two_component::MolecularTwoComponent` and the `x2c_taylor`, `dkh2_taylor` and `matrix_zora_taylor` operator paths | Energy and raw orbital/decoupling residuals in Hartree, radii in Bohr, dipole in e Bohr and complex polarizability in e² Bohr²/Hartree. An optional supplied dimensionless four-component operator is transformed in the same orthonormal chart. The molecular route currently uses even-electron all-electron insulating references. [Molecular model and verification](response.md#r075-breit-and-two-component-relativistic-dft). |
| `dirac-dks` | Fixed-sector molecular four-component Kohn–Sham with finite Gaussian nuclei, full balanced density and variational Coulomb XC. `dirac_interaction` selects Coulomb or its explicit Gaunt/Breit determinant current correction. Current direct and exchange terms and their raw defect are in Hartree. | `driver::run("dirac-dks", ...)` with all-electron even-electron GKS and explicit nuclear RMS radii in Bohr. | Energy, terms, spectral gap and orbital residual in Hartree, dipole in e Bohr, optional complex electric polarizability in e² Bohr²/Hartree and raw response residuals. [Model and verification](response.md#molecular-diraccoulomb-kohnsham). With `dirac_integrals="london-rmb"`, nuclear-core positive sector, Coulomb and range exchange, DFT density and dipole depend on the selected uniform magnetic field. The London-RMB option supports the current corrections with `dirac_interaction=gaunt` or `breit`. |
| `scdft-pairing-kernel` | Static LM2005 singlet phonon K, supplied screened Coulomb and optional generated phonon mass Z on normal bands | `driver::run("scdft-pairing-kernel", ...)`; `properties::pairing::{PairingInput,Lm2005}`; `Lm2005OffDiagonal` is a compatibility alias | Separate K components (Hartree); `band_weights` enables dimensionless Z; mixed parameter path coefficients through order four; separate K/Z quadrature differences and K asymmetry |
| `dynamic-polarizability` | Molecular reference-aware complex dipole susceptibility | `driver::run("dynamic-polarizability", ...); response::dynamic_polarizability; response::nuclear::{open_shell_dynamic_polarizability, spinor_electric_response}` | Full complex 3x3 tensor and original-operator residuals. RKS/RHF, UKS/UHF, constrained ROKS/ROHF and spin-covariant GKS/GHF. [Collinear double-hybrid quasi-energy response](#frequency-dependent-double-hybrid-electric-response). `relativistic_method="dirac"` selects the molecular fixed-sector four-component DKS route. [Model and coverage](response.md#molecular-diraccoulomb-kohnsham). |
| `molecular-medium-optics` | Electric molecular dielectric and slab scattering | `driver::run("molecular-medium-optics", ...)` and `response::medium_optics` | Normal complex indices, inverse-Bohr absorption coefficients, oblique s/p Jones matrices, power fractions and raw local-field, boundary and flux-balance residuals. [Model and scope](response.md#r080-molecular-medium-optics). With `medium_magnetic_field`, the linear orbital magnetic susceptibility correction supplies field-odd Kerr rotation and ellipticity in radians. |
| `field-taylor` | Static directional dipole response through any requested order | `response::electric_field_taylor(mol, mb, functional, grid, scf, opts, direction, order, &config)` | `ResponseReport<ElectricFieldTaylor>`: `dipole_coefficients[order+1][3]`, per-spin `density_coefficients`, order-resolved stationarity, projector, and particle-number defects; HF/PBE0 water alpha/beta/gamma verified at zero and nonzero reference field by twelve independent four-step fits; shared driver accepts prescribed nonthermal real collinear populations; see the nonlinear electric response contract below |
| `field-mixed` | Static mixed dipole derivative | `response::mixed_electric_field_response(mol, mb, functional, grid, scf, opts, directions, &config)` | `ResponseReport<MixedElectricFieldResponse>`: `dipole_derivative[3]`, `accumulation_roundoff[3]`, `directional_evaluations`; shared driver accepts prescribed nonthermal real collinear populations; see the nonlinear electric response contract below |
| `beta` | Complete static first hyperpolarizability | `response::static_beta(mol, mb, functional, grid, scf, opts, &config)`; `response::static_beta_variational(...)` for 2n+1 assembly | `ResponseReport<StaticElectricTensor>`: flat raw `tensor[27]`, shape `[3,3,3]`, `beta_vector/norm`, `frobenius_norm`, field/energy permutation defects and counts, roundoff and physical coefficient defects; variational assembly uses three first-order response solves; shared driver accepts prescribed nonthermal real collinear populations; see the nonlinear electric response contract below Correlated integer RKS/UKS energy paths, including static reference_field bias, are connected; parent/target/SS/OS tensors and raw residuals are returned. Measured subsets and remaining combinations are in response.md. [Constrained static tensors] Named C1 HF ROHF/GHF static subset: all434 independent fits, translation and public10 pass; other class/frequency combinations remain separate. See [measurements](#constrained-static-electric-tensors). Static `reference=gks` with equal `pt2_same_spin` and `pt2_opposite_spin` uses stationary complex spinors. The third-order energy paths return parent, target, target correction, and unscaled and weighted generalized PT2 tensors in e^3*Bohr^3/Hartree^2, with raw response and pair residuals. |
| `gamma` | Complete static second hyperpolarizability | `response::static_gamma(mol, mb, functional, grid, scf, opts, &config)` | `ResponseReport<StaticElectricTensor>`: flat raw `tensor[81]`, shape `[3,3,3,3]`, `gamma_isotropic`, `frobenius_norm`, field/energy permutation defects and counts, roundoff and physical coefficient defects; shared driver accepts prescribed nonthermal real collinear populations; see the nonlinear electric response contract below Correlated integer RKS/UKS energy paths, including static reference_field bias, are connected; parent/target/SS/OS tensors and raw residuals are returned. Measured subsets and remaining combinations are in response.md. [Constrained static tensors] Named C1 HF ROHF/GHF static subset: all434 independent fits, translation and public10 pass; other class/frequency combinations remain separate. See [measurements](#constrained-static-electric-tensors). Static `reference=gks` with equal `pt2_same_spin` and `pt2_opposite_spin` uses stationary complex spinors. The fourth-order energy paths return parent, target, target correction, and unscaled and weighted generalized PT2 tensors in e^4*Bohr^4/Hartree^3, with raw response and pair residuals. |
| `dynamic-beta` | Complete complex quadratic dipole response | `response::dynamic_beta(mol, mb, functional, grid, scf, opts, [z1,z2], &config)` | `ResponseReport<DynamicElectricTensor>`: raw complex `tensor[27]`, output frequency, full equation/projector/number defects, equal-frequency intrinsic interchange measurements, subset solve/kernel counts |
| `dynamic-gamma` | Complete complex cubic dipole response | `response::dynamic_gamma(mol, mb, functional, grid, scf, opts, [z1,z2,z3], &config)` | The same report, raw complex `tensor[81]`, shape `[3,3,3,3]`; no static Kleinman symmetrization |
| `electric-process` | Physical nonlinear optical frequency preset | `response::dynamic_electric_process(mol, mb, functional, grid, scf, opts, process, &[optical_magnitudes], damping, &config)` with `ElectricProcess` | The common complex tensor report with `process`, ordered incoming/outgoing frequencies and all physical residuals |
| `stability` | Molecular orbital stability | `response::stability_from_state(mol, mb, functional, grid, scf, opts, &StabilityOptions)`; `molecular_stability(..., roots, &ResponseSolverConfig)` is a convenience entry | `ResponseReport<MolecularStability>`: channel verdicts, lowest eigenvalues/generators, energy curvatures, dominant orbital excitations, measured residuals and convergence |
| `stab-following` | Internal orbital-instability descent and re-converged reference | `response::stabilize(mol, mb, functional, grid, scf, opts, &StabilizeOptions)` | `StabilizeReport`: initial/final analyses, accepted descent history, `final_state: ScfResult`, `stabilized`, `converged`, warnings |
| `orbital-energy-gradients` | Relaxed molecular orbital eigenvalue gradients and projected degenerate blocks | `response::orbital_energy_gradients(mol, mb, functional, grid, scf, opts, &OrbitalGradientOptions, &config)` | `ResponseReport<OrbitalEnergyGradients>`: selected coordinates, energies/occupations, optional individual gradients, full grouped derivative matrices, energy spreads, commutator/translation/Hermiticity and moving-metric diagnostics |
| `frontier-gap-gradient` | Per-spin neutral-reference LUMO-HOMO orbital gap gradients | `response::frontier_gap_gradient(mol, mb, functional, grid, scf, opts, &OrbitalGradientOptions, &config)` | `ResponseReport<FrontierGapGradient>`: optional spin gaps with HOMO/LUMO IDs and gradients, plus the common orbital derivative report |
| `nuclear-taylor` | Analytic nuclear energy derivatives, including cubic and quartic force constants | `response::nuclear::nuclear_taylor(mol, mb, functional, grid, scf, opts, &direction, order, &config)` | `ResponseReport<NuclearTaylor>`: energy/component coefficients, density/overlap and transported orthonormal Fock coefficients, projector/stationarity/metric/number defects, reference energy/Fock errors |
| `nuclear-mixed` | Mixed nuclear energy derivatives; three/four directions give cubic/quartic force constants | `response::nuclear::mixed_nuclear_response(mol, mb, functional, grid, scf, opts, &directions, &config)` | `ResponseReport<MixedNuclearResponse>`: scalar `energy_derivative`, `accumulation_roundoff`, `directional_evaluations` |
| `molecular-taylor` | Simultaneous nuclear and electric energy/dipole Taylor response | `response::nuclear::molecular_taylor(mol, mb, functional, grid, scf, opts, &MolecularDirection { nuclear, electric }, order, &config)` | `ResponseReport<MolecularTaylor>`: energy/components, dipole and density jets, moving metric and physical residuals |
| `force-constants` | Analytic cubic/quartic tensors on patterns, coordinate blocks or all Cartesian coordinates | `driver::run("force-constants", system, controls)`; `response::force_constants::{analytic_force_constants(..., &patterns, order, &config), cartesian_force_constant_block(..., &coordinates, order, &config), cartesian_force_constants(..., order, &config)}`; molecular arguments precede the shown request | `Value` on the common driver; `ResponseReport<AnalyticForceConstants>`: flat tensor, patterns, component count, independently measured reverse-order defects, Cartesian translation residual when measurable, comparison counts, roundoff and physical residuals ; integer correlated RKS/UKS complete Cartesian/block/pattern tensors include full orbital/PT2 amplitude response and separate target/SS/OS/dispersion channels; coordinate units Hartree/Bohr^order, dimensional-pattern unit Hartree; energy permutation reuse is explicit, with raw residuals and independent acceptance tracked in response.md  ROKS/ROHF and complex GKS/GHF complete exports are connected; full acceptance remains in progress, with raw failures documented in response.md. Geometry-only D1 is accepted through the shared driver using explicit atomic C₆, radii, and scale. [Model and checks](response.md#molecular-d1-energy-and-nuclear-tensors). Positive electronic temperature selects the stationary molecular Mermin derivative for supported ROKS/ROHF and GKS/GHF providers. [Finite-temperature guide](api-guide.md#set-an-electronic-temperature). |
| `anharmonic` | Selectable strict VPT2 or retained-polyad anharmonic bands, IR corrections and optional static Raman | `response::anharmonic::molecular::molecular_anharmonic(mol, mb, functional, grid, scf, opts, mode_ids, include_raman, &anharmonic_config, &response_config)`; `response::anharmonic::vibrational_spectrum(&VibrationalPolynomial, &anharmonic_config)` accepts supplied analytic derivative tensors | `ResponseReport<MolecularAnharmonic>`: normal-mode patterns, cubic/quartic tensors, third nuclear dipole/polarizability derivatives, resolved bands, transformed transition moments, polyad Hamiltonian and diagnostics |
| `molecular-mixed` | Mixed nuclear/electric energy and dipole derivative | `response::nuclear::mixed_molecular_response(mol, mb, functional, grid, scf, opts, &directions, &config)` | `ResponseReport<MixedMolecularResponse>`: `energy_derivative`, `dipole_derivative[3]`, separate accumulation roundoff estimates |
| `static-spectroscopy` | Directional analytic APT, Raman and hyper-Raman | `driver::run("static-spectroscopy", ...)`; `response::nuclear::static_spectroscopic_derivative(mol, mb, functional, grid, scf, opts, &nuclear_direction, electric_order, &config)` | `ResponseReport<StaticSpectroscopicDerivative>`: flat Cartesian tensor of rank 1/2/3, raw `permutation_defect`, `permutation_comparisons`, roundoff and evaluation count; common-driver integer RKS/UKS complete post-SCF target and PT2 amplitude/orbital response; nuclear paths in Bohr, atomic electric units, all ordered tensor entries; complete C1 HF+PT2 RKS/UKS static APT/Raman/hyper-Raman subset validated by all 570 independent fits and all 12 primary native/ASE/CLI checks; raw normalized amplitude, reference, metric, particle-number and stationarity diagnostics and failed-constraint warnings are returned. Use the common driver for target/PT2 energy composition; the low-level Rust entry differentiates its supplied reference functional. [Constrained static tensors] Named C1 HF ROHF/GHF static subset: all434 independent fits, translation and public10 pass; other class/frequency combinations remain separate. See [measurements](#constrained-static-electric-tensors). |
| `periodic-response` | Self-consistent finite-q density response | `dfpt::density::PeriodicDensitySolver::new(cell, functional, reference, scf_options, &options)?`, followed by `solver.solve(&external)` | `ResponseReport<PeriodicDensityResponse>`: complex `density[spin][k][bra AO][ket AO]`, `fourier_density`, reciprocal waves, occupation consistency defect; semilocal and full/range exchange share the same independent shifted-space solver, with physical derivative scope measured in p292 |
| `field-response` | Screened periodic orbital electric/magnetic density and current | `dfpt::field::periodic_field_response(cell,functional,reference,scf_options,&PeriodicFieldResponseOptions)` | `ResponseReport<PeriodicFieldResponse>`: independent k+q density, number/charge observables, separate paramagnetic/diamagnetic current, fields and raw continuity diagnostic |
| `dielectric` | Selected inverse dielectric matrix from screened susceptibility | The same `PeriodicDensitySolver`, followed by `solver.dielectric(&modes)` | `ResponseReport<DielectricResponse>`: complex `screened_chi[row][column]`, `inverse_epsilon[row][column]`, selected reciprocal waves |
| `thermal-response` | Uniform static self-consistent Mermin thermodynamic response | The same `PeriodicDensitySolver`, followed by `solver.solve_thermal(source)` | `ResponseReport<PeriodicThermalResponse>`: density/Fourier variations, `delta_chemical_potentials`, `delta_numbers`, entropy/internal/free-energy variations, occupation and number defects |
| `periodic-electrostatics` | 1D/2D/3D prescribed point/Gaussian Coulomb energy and analytic forces, including compensated charged cells | `periodic::electrostatics::periodic_electrostatics(&PeriodicGeometry, &[ElectrostaticSource], &ElectrostaticOptions)`; `DimensionalCoulomb::pair(displacement, width, exclude_home)` exposes the same Green operator | `ElectrostaticReport`: energy/components, forces, site potentials, countercharge reaction, boundary/measure, Ewald parameters, independent Fourier quadrature defects and potential gauge |
| `periodic-basis` | Dimensional Gaussian Bloch overlap/kinetic, atomic derivatives and optional XC quadrature | `periodic::periodic_basis(mol, mb, &geometry, &PeriodicBasisOptions)` | `PeriodicBasisReport`: k points, complex S/T, `[k,atom,axis]` derivative matrices, raw Hermiticity/translation defects, minimum overlap eigenvalue, optional grid/moving weights and grid overlap defect |
| `periodic-gaussian-integral` | Dimensional normalized Cartesian nuclear attraction or electron repulsion, with analytic centre gradients | `periodic::gaussian_coulomb::periodic_gaussian_integral(&geometry, &[CartesianGaussian], Option<&GaussianNucleus>, &GaussianIntegralOptions)`; `DimensionalCoulomb::gaussian_derivatives(r,width,order)` | `GaussianIntegralReport`: value, split components, centre gradients, actual controls, quadrature/translation defects, physical model, units and convergence |
| `periodic-coulomb` | Contracted spherical dimensional nuclear/Hartree operators, compensated Coulomb components and fixed-density centre derivatives | `periodic::coulomb_boundary::periodic_coulomb_operators(mol,mb,&geometry,Option<&[BlochMatrix]>,&PeriodicCoulombOptions)`; reusable `DimensionalCoulombBackend::new(...).evaluate(density)` | `PeriodicCoulombReport`: complex split matrices/derivatives, source/electronic/Hartree energies, fixed AO density gradients, actual electron count and all raw defects; complete SCF/DFPT remains in progress |
| `periodic-xc` | Dimensional semilocal energy/components and analytic spin/Bloch potentials for supplied complex densities | `periodic::xc_boundary::periodic_xc_operators(mol,mb,&geometry,&functional,&density,&PeriodicXcOptions)`; reusable `DimensionalXcBackend::new(...).evaluate(functional,density)` | `PeriodicXcReport`: weighted semilocal parts, complex potentials `[spin,k,AO,AO]`, separate metric/grid/target electron counts and raw defects; exact exchange/nonlocal metadata and component model |
| `magnetic-response` | Analytic closed/open collinear orbital magnetic spin density response | `properties::magnetic::magnetic_density_response(mol,mb,functional,grid,scf,opts,gauge,origin,&config)` | `ResponseReport<SpinMagneticResponse>`, density/source matrices [spin,B,AO,AO], metric derivatives, actual occupations, fresh relative residuals and raw constraints |
| `nmr` | Orbital NMR shieldings for closed/open collinear references | `properties::magnetic::molecular_nmr_response(...,&config)`; compatible `shielding_tensors[_with_grid]` wrappers | Total/diamagnetic/paramagnetic [nucleus,nuclear moment,B] in ppm, shared reference and response diagnostics; spin thermal/hyperfine shifts remain separate |
| `periodic-field` | Analytic 1D/2D/3D scalar/current/diamagnetic Fourier Hamiltonian derivatives | `periodic::field::periodic_field_operators(mol,mb,&geometry,&PeriodicFieldOptions)` | `PeriodicFieldReport`: density/current/first/second/contact vertices, independent ket/k+q/k+2q spaces, complex mode E/B, charge/model/units; screening and homogeneous-field SCF pending |
| `dimensional-scf` | Self-consistent zero-field 1D/2D/3D collinear HF/semilocal/global/range-separated hybrid DFT with ECP and compensated signed charge | `periodic::scf_boundary::run_dimensional_scf(mol,mb,&geometry,&functional,&DimensionalScfOptions)`; cached `DimensionalScfBackend::new(...).evaluate(&spin_density)` and `reference_fock_at` including shifted hybrid/ECP operators | `DimensionalScfReport`: reference, all energy components, weighted semilocal parts, fresh physical Fock/number/occupation and Coulomb/grid diagnostics |
| `dimensional-field` | Screened 1D/2D/3D orbital scalar/vector-potential response of a converged dimensional reference | `dfpt::field_dimensional::dimensional_field_response(mol,mb,&geometry,&functional,&scf_report,&scf_options,&DimensionalFieldResponseOptions { transfer_fractional, frequency, fixed_number, scalar_potential, vector_potential, solver })` | `ResponseReport<DimensionalFieldResponse>`: full and reduced transfer, wave, source/target k meshes, complex spin/k AO density, E/B fields, number/charge density, separate paramagnetic/diamagnetic currents, continuity, quadrature, occupation and stationary-reference defects |
| `energy` | Molecular total energy with every component (Hartree) | `driver::run("energy", &System::molecule(numbers, bohr_positions), &controls)`; `driver::method::{prepare_molecular, run_molecular_reference}` build and converge the same reference | `Value` report: `energy`, `energy_components` (electronic, nuclear repulsion, XC, nonlocal correlation, continuum electrostatic, dispersion, SMD CDS), `converged`, `scf_iterations`, `unrestricted`, `n_alpha/n_beta`, `s_squared`, optional `stability_following`. With `xc = skala` and `skala_model = <model file>` the exchange-correlation energy and Kohn-Sham matrices come from a Skala model file (`skala::SkalaModel::load`, `skala::skala_fock`); only `energy`, `charges`, `dipole`, `orbitals`, `molden`, `orbitals-save` and `molecular-multipole` accept it. Double hybrids use this same entry: `xc="b2plyp"`, `xc="dsd-pbep86"`, `xc="dsd-blyp"`, `xc="pwpb95"` or `xc="wb97m2"`; the DSD names select the documented 2013 pairwise D3(BJ) recipe. SS/OS PT2 and dispersion are reported separately. Arbitrary parents accept `pt2_same_spin` and `pt2_opposite_spin`. Reports give parent, term energies and raw amplitude/Lagrangian residuals; [model and measured acceptance](response.md#double-hybrid-components-and-pt2-lagrangian).  omegaB97M(2) uses a distinct target on omegaB97M-V orbitals, scaled VV10 with b=10 and separate orbital/target energies. PWPB95 includes reoptimized PW/B95 and SOS PT2; all terms and raw residuals use the shared report. Custom parents retain their complete XC energy; `nonsemilocal_exchange_correlation_energy` (Hartree) separates a model-provider contribution from the semilocal component sum. VV10 remains in its separate nonlocal-correlation field. Integer state selection: `occupation_mode=mom/imom`, complete `occupied_alpha/occupied_beta`, optional `occupation_tracking`; [score, units and pending combinations](rust-api.md#occupied-projector-delta-scf-controls). Fixed nonthermal populations also use these common entries: `occupation_mode=fixed`, `occupations_alpha/occupations_beta`; the raw density-map residual is dimensionless and the unequal-population spectral gap is Hartree. [Constraint, units and remaining combinations](rust-api.md#fixed-orbital-population-controls).  RKS/UKS selectors and HF aliases use the existing driver; ROKS/ROHF and GKS/GHF connections have the [reference-specific scope and diagnostics](response.md#molecular-complex-spinor-reference-provider). [Weighted XC controls and diagnostics](#weighted-molecular-xc). GKS common-weight generalized PT2 energy reports the full-spin correlation, optional target correction and amplitude residual in Hartree.  [ROKS doubles-only PT2 energy and residuals](#restricted-open-shell-double-hybrid-energy). [Molecular D1, TS, MBD, VV09 and fixed-density vdW-DF controls, units and acceptance](response.md#r089-molecular-post-scf-dispersion-and-nonlocal-kernels). [Molecular total-spin values and run logs](response.md#molecular-total-spin-streaming). |
| `mermin-free-energy` | Stationary molecular electronic Mermin free energy A = E - T_e S_e at fixed electron count | `driver::run("mermin-free-energy", ...)`; `electronic_temperature` (K), optional `gradient`; `gibbs=true` adds the shared analytic Hessian and Gibbs report | Mermin free energy and internal energy in Hartree, electronic entropy in k_B, occupations, chemical potentials and raw SCF residuals. ROKS/ROHF fixes both spin counts with shared orbitals. GKS/GHF fixes total electron count. Optional analytic gradient uses the supported stationary provider. Equilibrium Gibbs RRHO uses the RKS/UKS route. [Finite-temperature guide](api-guide.md#set-an-electronic-temperature). |
| `gibbs-free-energy` | Ideal-gas RRHO Gibbs energy from the same reference Hessian | `driver::run("gibbs-free-energy", ...)`; `temperature` (K), `pressure` (Pa), `symmetry_number`, `electronic_degeneracy`; common constrained-reference Hessian controls | `gibbs_free_energy` (Hartree per molecule), frequencies and full `thermochemistry`; positive internal modes, equilibrium temperature, converged reference and free-molecule Eckart model required; source/acceptance limits in response.md |
| `free-energy` | Compatibility alias for `mermin-free-energy`: Mermin free energy `A = E - T S` of a molecular Fermi-Dirac reference | `driver::run("free-energy", ...)` with `electronic_temperature` (Kelvin) and optional `gradient`; `ScfOptions::electronic_temperature` (k_B T in Hartree) and `scf::fermi_dirac_occupations` give canonical per-spin occupations; `assemble_gradient` differentiates `A` with the occupation-weighted energy density | `free_energy`, `internal_energy`, `entropy` (k_B), `entropy_term`, `kt`, per-spin `chemical_potentials` and `occupations`, `ensemble`, optional analytic `gradient` of `A` |
| `gradient` | Analytic molecular nuclear gradient of the total energy | `driver::run("gradient", ...)`; analytic `gradient::assemble_gradient` plus the dispersion gradient; RI-J and SMD CDS are refused rather than differentiated numerically | `gradient[atom][3]` in Hartree/Bohr, raw `translation_residual` (sum minus `-Q E` in a field), `provenance="analytic"`, plus the energy report Integer state selection: `occupation_mode=mom/imom`, complete `occupied_alpha/occupied_beta`, optional `occupation_tracking`; [score, units and pending combinations](rust-api.md#occupied-projector-delta-scf-controls). Fixed nonthermal populations also use these common entries: `occupation_mode=fixed`, `occupations_alpha/occupations_beta`; the raw density-map residual is dimensionless and the unequal-population spectral gap is Hartree. [Constraint, units and remaining combinations](rust-api.md#fixed-orbital-population-controls).  RKS/UKS selectors and HF aliases use the existing driver; ROKS/ROHF uses the existing constrained analytic route; [GKS/GHF nuclear paths include Coulomb/exchange and globally projected semilocal XC; full Cartesian double-hybrid tensors include PT2 amplitude/orbital response, with bounded acceptance scope](response.md#molecular-complex-spinor-reference-provider). [Weighted XC controls and diagnostics](#weighted-molecular-xc). The ordinary molecular report also returns `forces` as the negative Cartesian gradient in Hartree/Bohr. Positive electronic temperature selects the stationary molecular Mermin derivative for supported ROKS/ROHF and GKS/GHF providers. [Finite-temperature guide](api-guide.md#set-an-electronic-temperature). [Generalized double-hybrid gradients](api-guide.md#read-a-generalized-double-hybrid-gradient). |
| `streamed-gradient` | Analytic nuclear gradient in which no integral tensor is stored: the iteration is kept on integral-direct Coulomb and exchange builds, and the gradient contracts the derivative integrals shell quartet by shell quartet, so the memory stays quadratic in the basis size. Same energy and gradient as `gradient`. Restricted HF/PBE STO-3G all-component and public-route acceptance validated; other models remain pending | `driver::run("streamed-gradient", ...)`; `gradient::assemble_gradient` | `gradient` in Hartree/Bohr, `avoided_tensor_megabytes` [Measured restricted HF/PBE STO-3G all-component and public-route scope](validation-matrix.md#t139-integral-direct-molecular-gradient). |
| `charges` | Mulliken and Loewdin charges and Mulliken spin populations | `driver::run("charges", ...)` | `mulliken`, `lowdin`, `mulliken_spin` (alpha minus beta), raw `charge_sum_residual`, plus the energy report  Supports `reference=auto/rks/uks/roks` (HF aliases accepted); ROKS spin semicanonical orbital export and constraints are described in [reference-specific scope](response.md#reference-constraints-for-molecular-populations-and-orbitals).  RKS/UKS selectors and HF aliases use the existing driver; ROKS/ROHF and GKS/GHF connections have the [reference-specific scope and diagnostics](response.md#molecular-complex-spinor-reference-provider). |
| `natural-bond` | Natural bond orbital analysis of the converged SCF reference. Natural atomic orbitals: component-averaged atomic natural orbitals per atom and angular momentum, natural minimal basis of the core and valence subshells of each element (effective-core subshells removed), symmetric orthogonalization of the core, occupancy-weighted orthogonalization of the valence set, Schmidt and symmetric orthogonalization of the Rydberg set, restored natural character. Natural population analysis, natural electron configurations and Wiberg indices `2 sum (D^a^2 + D^b^2)` in the NAO basis. Natural Lewis structure: core orbitals, lone pairs, two-centre and (`nbo_three_center`) three-centre bonds above the occupancy threshold, scanned from 0.95 to 0.75 of the maximum occupancy for the smallest non-Lewis occupancy unless `nbo_lewis_threshold` fixes it; orthogonal hybrids, antibonds and Rydberg orbitals complete an orthonormal NBO basis. NLMOs by Jacobi rotations of the Lewis/non-Lewis density block; second-order donor-acceptor energies `q_i F_ij^2/(F_jj - F_ii)` with the Fock matrix in the NBO basis (`nbo_perturbation_threshold`, Hartree, default 0.5 kcal/mol). Open shells (`nbo_spin = auto`; `total`, `separate`) get one Lewis structure per spin with at most one electron per orbital and natural spin populations. Model: SCF one-particle densities and Fock matrices, full-rank AO metric. Acceptance `p233_natural_bond` (Hartree-Fock restricted and unrestricted references: external NAO oracle within 1.8e-10, Lewis structures, NLMO and second-order checks, spin and rotation limits) and `python/tests/test_natural_bond.py`; other functional classes are not separately accepted | `driver::run("natural-bond", ...)`; `properties::natural_bond::{natural_bond_analysis, natural_atomic_orbitals, natural_bond_orbitals}` | `natural_charges` (e), `natural_populations`, `natural_spin_populations`, core/valence/Rydberg populations, `natural_electron_configurations`, `natural_subshells`, `nao_labels`, `nao_occupancies`, `nao_spin_densities`, `nao_coefficients`, `wiberg_bond_indices`, `channels` per analysed density (`nbos` with type, atoms, occupancy, energy in Hartree and hybrids with polarization coefficients and percent s/p/d/f character; `nbo_coefficients`; `perturbation` stabilizations, Fock elements and gaps in Hartree; `nlmos` with parent and atomic percentages; threshold trials, orthonormality and NLMO residuals) The SCF reference selector supports RKS, UKS, ROKS and globally collinear GKS with real spatial blocks. `nbo_spin_density` and `nbo_natural_orbitals` expose `natural_bond::spectral::NaturalDensityOrbitals`. [Raw defects and units](response.md#reference-selection-and-natural-density-orbitals). `natural_bond::corresponding::corresponding_orbitals` provides the occupied pairing. |
| `igmh` | Isolated all-electron molecular IGMH, RKS/UKS/ROKS/GKS spin-summed SCF density with spherical neutral HF proatoms in the molecular basis. The default uses the published visualization vector. The `hirshfeld` mode uses the true stockholder density gradient. [Definition and numerical checks](response.md#r090-molecular-igmh) | `driver::run("igmh", ...)` or `properties::igmh::analyze` on a supplied AO density | Points in Bohr, density and signed density in electrons/Bohr³, density Hessian in electrons/Bohr⁵, three delta-g fields and optional atomic vectors in electrons/Bohr⁴. Raw SCF, density and gradient closure residuals. |
| `periodic-natural-bond` | Natural bond orbital analysis of a periodic reference: the k-mesh calculation is the calculation of its Born-von Karman supercell, whose real overlap, spin densities and Fock matrices `X[(L mu),(L' nu)] = 1/N_k sum_k X(k) exp(-i k.(R_L' - R_L))` are analysed like a molecule (3D finite-k SCF; open axes through the dimensional SCF). Natural charges, populations and configurations of the home-cell atoms, Wiberg bonds with the cell offset of the partner, the natural Lewis structure per cell (one orbital per translation class, image atoms labelled `C2[1,0,0]`), hybrids and second-order donor-acceptor energies; the NBO controls of `natural-bond`. Model: unshifted k mesh; with fewer than three k points along a periodic axis an atom and its image in the next cell can share supercell orbitals (warning). Validated by `p234_natural_bond_extensions` (unfolded natural charges against the explicit Gamma-point supercell 2.2e-15, translation defect 8.9e-16) | `driver::run("periodic-natural-bond", ...)`; `properties::natural_bond::periodic::periodic_natural_bond(&molecule, &basis, lattice, mesh, &scf, spin, &options)` | `natural_charges` (e), populations, `natural_electron_configurations`, `wiberg_bonds` (atom, partner, `cell_offset`, index), `channels` per spin density (`electrons_per_cell`, Lewis/non-Lewis occupancy per cell, `nbos`, `perturbation` in Hartree), `translation_defect`, `unfolding_imaginary`, `nbo_warnings` |
| `natural-shielding` | Natural chemical shielding analysis: the orbital NMR shielding of the coupled-perturbed magnetic response (GIAO or common origin) as a sum over the NLMOs of the natural Lewis structure. The occupied rotation `T = C_o^T S L` carries the first-order canonical orbitals `C_v U - C_o S^oo/2` to the NLMOs; NLMO `k` contributes `Tr[l_k l_k^T h_mixed]` (diamagnetic) and `Tr[(u_k l_k^T - l_k u_k^T) h_pso]` (paramagnetic); open shells per spin. Contributions sum to the shielding (`decomposition_residual`). Validated by `p234_natural_bond_extensions` (methane: total equal to `nmr-shielding`, equivalent C-H bonds spread 4.9e-13, decomposition residual 3.4e-13) | `driver::run("natural-shielding", ...)`; `properties::magnetic::orbital_shielding(..., [&l_alpha, &l_beta]) -> OrbitalShielding` | `shieldings` per nucleus: isotropic and tensor (ppm), `contributions` per NLMO (label, spin, isotropic, diamagnetic and paramagnetic isotropic, tensor); `decomposition_residual` (ppm), `rotation_residual` |
| `natural-coupling` | Natural J-coupling analysis of a restricted closed-shell reference: the reduced indirect spin-spin couplings (Fermi contact, spin dipole, paramagnetic and diamagnetic spin-orbit) of the `nuclei` pairs as sums over the NLMOs; the triplet and imaginary singlet response amplitudes of the canonical orbitals are rotated to the NLMOs and the diamagnetic term integrates the NLMO densities. Validated by `p234_natural_bond_extensions` (methane C-H: total equal to `spin-spin-couplings`, decomposition residual 6.6e-23) | `driver::run("natural-coupling", ...)`; `response::spin_spin::spin_spin_couplings_by_orbital(..., &nlmos) -> (SpinSpinCouplings, OrbitalCouplings)` | `couplings` per pair: reduced isotropic K and its four terms (atomic units), `coupling_hz` with `nuclear_g`, `contributions` per NLMO; `decomposition_residual`, `rotation_residual`, `response_residual` |
| `natural-energy-decomposition` | Natural energy decomposition of the interaction of molecular fragments (`fragments`: fragment index per atom): the Fock matrix of the complex in its NBO basis with every inter-fragment element deleted, each fragment occupying the eigenvectors of its block up to its Lewis count, gives the localized state; charge transfer `E[P] - E[P_loc]`, electrostatic energy of the deformed fragment densities and nuclei, exchange-correlation remainder `E[P_loc] - sum E_X[P_X] - ES`, and deformation `E_X[P_X] - E_X[isolated]` against SCF fragments in the basis of the complex; the terms sum to the interaction energy. Model: closed- or open-shell SCF densities, functionals without nonlocal correlation, no continuum or effective core potentials, fragments without shared Lewis orbitals. Validated by `p234_natural_bond_extensions` (water dimer: identity 0, local electrostatic exponents 3.12 and 3.06, interaction against the SAPT counterpoise energy 1.2e-12) | `driver::run("natural-energy-decomposition", ...)` | `interaction_energy`, `charge_transfer`, `electrostatic`, `exchange_correlation`, `deformation` (Hartree), `fragments` (atoms, charge, multiplicity, electrons, isolated/deformed energies), `decomposition_residual`, `energy_functional_residual`, `localized_idempotency_residual` |
| `sapt-analysis` | `sapt_variant` selects `response` (default, HF/KS, Heitler-London exchange and projected exchange-dispersion) or `sapt0`, `sapt0-s2`, `ssapt0` (field-free all-electron HF, CPHF induction and uncoupled orbital-pair dispersion; full, single-pair and scaled single-pair exchange, respectively). Perturbation terms use monomer-centred orbitals. `sapt_regularization_eta` (Bohr⁻²) enables the optional R-SRS induction diagnostic with `sapt0` or `sapt-ks-fdds`. `sapt-30` uses canonical uncoupled HF product determinants with exact antisymmetrization through third intermolecular order. `sapt2+3` assembles [full-exchange correlated RHF SAPT2+3](response.md#sapt23-with-full-intermolecular-exchange). `sapt-ks-fdds` uses coupled induction, adiabatic FDDS dispersion and uncoupled KS exchange-dispersion. Other modes accept the NLMO atom-pair electrostatic/induction partition and dimer-minus-monomer-basis induction diagnostic. Optional full response and the potential-only KS asymptotic correction are available in `response`. `SaptDiagnostics` adds [closed-shell correlated terms and relaxed cubic HF induction](response.md#correlated-sapt-terms-and-relaxed-cubic-induction) to `sapt0`. See [models and numerical residuals](response.md#sapt0-with-the-full-antisymmetrizer). | `driver::run("sapt-analysis", ...)`; `sapt::sapt_interaction_variant`; `sapt::sapt_interaction_regularized_variant`; `sapt::sapt_interaction_diagnostics`; `sapt::sapt_interaction_with` for the default | `electrostatic`, `exchange`, `induction` [A, B], `exchange_induction`, `exchange_induction_directions` [A, B], `dispersion`, `exchange_dispersion`, `dispersion_uncoupled`, `interaction_energy`, `interaction_energy_rpa`, `supermolecular_interaction`, `delta`, `third_order` (`sapt-30` or `sapt2+3`), `sapt2_plus3`, `regularized_induction` (eta in Bohr⁻²), `correlation_terms` and `relaxed_induction30` (Hartree), `delta_hf`, `exchange_scale` (dimensionless), `rpa`, `charge_transfer`, `partition`, `fragments` (Hartree) |
| `dipole` | Molecular electric dipole of nuclei plus electrons | `driver::run("dipole", ...)` with optional `origin` | `dipole`, `dipole_norm` (e*Bohr), `dipole_debye`, `dipole_norm_debye`, `origin` (Bohr), `origin_convention`; charged molecules carry an origin warning  Double hybrids and custom PT2 parents use the same API; `parent_dipole`, `pt2_dipole_correction`, `target_dipole_correction` are e Bohr; `response_residuals` are relative, `amplitude_response_residuals` e Bohr and `response_number_residuals` electrons per atomic field. Includes noncanonical amplitude/orbital response and distinct-target exchange/VV10; integer zero-temperature reference. [Measured fits and model limits](response.md#correlated-electric-dipoles-and-periodic-dispersion-force-constants). Fixed nonthermal populations also use these common entries: `occupation_mode=fixed`, `occupations_alpha/occupations_beta`; the raw density-map residual is dimensionless and the unequal-population spectral gap is Hartree. [Constraint, units and remaining combinations](rust-api.md#fixed-orbital-population-controls). [Reference, origin and weighted XC controls](rust-api.md#static-molecular-electric-moments). `relativistic_method="dirac"` selects the molecular fixed-sector four-component DKS route. [Model and coverage](response.md#molecular-diraccoulomb-kohnsham). |
| `orbitals` | Canonical orbitals of the converged reference | `driver::run("orbitals", ...)` with `coefficients`, `density` | Per spin channel: `energies`, `occupations`, `homo_index`, `homo`, `lumo`, `gap`; optional AO `coefficients`/`density` with `overlap` and the spherical AO order Fixed nonthermal populations also use these common entries: `occupation_mode=fixed`, `occupations_alpha/occupations_beta`; the raw density-map residual is dimensionless and the unequal-population spectral gap is Hartree. [Constraint, units and remaining combinations](rust-api.md#fixed-orbital-population-controls).  Supports `reference=auto/rks/uks/roks` (HF aliases accepted); ROKS spin semicanonical orbital export and constraints are described in [reference-specific scope](response.md#reference-constraints-for-molecular-populations-and-orbitals).  RKS/UKS selectors and HF aliases use the existing driver; ROKS/ROHF and GKS/GHF connections have the [reference-specific scope and diagnostics](response.md#molecular-complex-spinor-reference-provider). [Weighted XC controls and diagnostics](rust-api.md#weighted-molecular-xc). |
| `molden` | Molden export of the converged reference (fractional occupations written as they are) | `driver::run("molden", ...)` with `path`; `properties::molden::molden_text` | `path`, `bytes`, `n_mo`, raw `occupation_density_deviation`, `ao_order`, plus the energy report |
| `orbitals-save` | JSON orbital export that refuses occupations not reproducing the SCF density | `driver::run("orbitals-save", ...)` with `path` | The `orbitals` report with coefficients, densities, overlap, `kt`, `chemical_potentials`, `occupation_density_deviation` (refused above 1e-8) and `path`; the file holds the same JSON |
| `molecular-multipole` | Molecular multipoles to rank 8 in three conventions | `driver::run("molecular-multipole", ...)` with `max_rank` (default 4, at most 8) and `origin` | Per rank: `monomials`, raw `cartesian` moments, Buckingham `traceless` components, Racah-normalized real `spherical` Q_lm; `origin`, `conventions` [Reference, origin and weighted XC controls](rust-api.md#static-molecular-electric-moments). |
| `quadrupole-polarizability` | `driver::run("quadrupole-polarizability", ...)` | Analytic electric quadrupole response. RKS/UKS double hybrids return parent, target and spin-resolved PT2 tensors at static and supported nonzero frequencies. Static GKS with a common PT2 weight gives the full-spin pair and target corrections. | Five symmetric traceless modes, e²·Bohr⁴/Hartree, origin (Bohr), raw SCF and response residuals. [Conventions](#electric-quadrupole-polarizability). `relativistic_method="dirac"` also returns both mixed dipole/field-gradient blocks and residuals for all eight sources. [Model and coverage](response.md#molecular-diraccoulomb-kohnsham). |
| `multipole-polarizability` | `driver::run("multipole-polarizability", ...)` | Analytic Racah response to a conjugate electric multipole field. `multipole_rank` is 1 to 5, default 3. The collinear double-hybrid path reports parent, target and PT2 terms. | `2 rank + 1` modes, e²·Bohr^(2 rank)/Hartree, origin (Bohr), signed frequency (Hartree), raw SCF and solve residuals. [Conventions](#electric-multipole-polarizability). `relativistic_method="dirac"` selects the molecular fixed-sector four-component DKS route. [Model and coverage](response.md#molecular-diraccoulomb-kohnsham). |
| `conceptual-dft` | Conceptual DFT descriptors: chemical potential, hardness, softness, electrophilicity, IP/EA, condensed Fukui functions, dual descriptor, local softness | `driver::run("conceptual-dft", ...)` with `route` (`grand_canonical` when `electronic_temperature > 0`, otherwise `integer`; or `koopmans`), `population` (`mulliken`/`lowdin`), `cation_multiplicity`, `anion_multiplicity`. The analytic grand-canonical route uses `response::ensemble::EnsembleSolver` (self-consistent Fermi-Dirac response with occupation response and number constraints) | `route`, `global` (`chemical_potential`, `electronegativity`, `hardness`, `softness`, `electrophilicity`), route-specific `ionization_energy`, `electron_affinity`, `fukui_plus/minus/zero`, `dual_descriptor`, `local_softness*`, or `fukui`, per-spin `chemical_potentials_per_spin`, `hardness_matrix_per_spin` and its raw symmetry defect, response residual and number defect, `conventions` |
| `spin-conceptual-dft` | Spin-channel conceptual DFT with `M = N_alpha - N_beta` | `driver::run("spin-conceptual-dft", ...)`; `grand_canonical` (unrestricted ensemble response) or `integer` (fixed-N spin flips) | Grand canonical: `spin_chemical_potential`, `spin_hardness`, `charge_spin_coupling` and its independently evaluated transpose, `spin_fukui_ss/ns/sn` plus the charge-channel keys. Integer: `spin_chemical_potential_plus/minus`, `spin_hardness`, `spin_fukui_plus/minus`; charge hardness and coupling are `null` with the reason in `absent` |
| `hessian` | Analytic Cartesian Hessian of the total energy with its verdict | `driver::run("hessian", ...)`; `hessian::analytic::analytic_hessian_full` (now also returning `raw_asymmetry` and the relaxed `density_derivatives`) plus the analytic dispersion Hessian. Unsupported analytic terms (RI-J, SMD CDS) are refused, not differenced | `hessian` (Hartree/Bohr^2, rows/columns `3*atom+axis`), analytic `dipole_derivatives` (3 x 3N), raw pre-symmetrization `raw_asymmetry`, acoustic `translation_residual` (absent in a field), `provenance`, plus the energy report Optional `bfield` and `field` select the shared complex London stationary nuclear path (integer occupations); `bfield_series_order` retains at least nuclear order+2. Reports include fresh `response_residuals`, `projector_defects`, `stationarity_defects`, `reference_defect`, `series_order`, `series_tail` and `accumulation_roundoff`. Intrinsic permutation reuse is declared; independent ordered asymmetry and dipole derivatives are not supplied on this magnetic path. Complex PT2 and extra Hamiltonian providers retain separate restrictions. See the externally biased nuclear scope below. Prescribed nonthermal populations use the full spectral moving-metric response, with `ensemble_response_residuals[3N]` (dimensionless), `ensemble_number_defects[3N][2]` (electrons/Bohr), the Hessian and APT. [Population-law contract](rust-api.md#fixed-population-and-thermal-static-response).  RKS/UKS selectors and HF aliases use the existing driver; ROKS/ROHF uses the existing constrained analytic route; [GKS/GHF nuclear paths include Coulomb/exchange and globally projected semilocal XC; full Cartesian double-hybrid tensors include PT2 amplitude/orbital response, with bounded acceptance scope](response.md#molecular-complex-spinor-reference-provider). [Weighted XC controls and diagnostics](rust-api.md#weighted-molecular-xc).  [Occupation-order diagnostics](rust-api.md#scf-occupation-order-diagnostics): per-spin occupied_virtual_gaps (Hartree) and negative-gap warning. Positive electronic temperature selects the stationary molecular Mermin derivative for supported ROKS/ROHF and GKS/GHF providers. [Finite-temperature guide](api-guide.md#set-an-electronic-temperature). |
| `frequencies` | Harmonic normal modes of the selected analytic reference Hessian | `driver::run("frequencies", ...)`; `vibrational_projection="auto"` (or `eckart`, `cartesian`), `electronic_degeneracy`, `temperature` (K), `pressure` (Pa), `symmetry_number`; low-level `properties::vibrational::vibrational_analysis_with_projection` | `frequencies_cm1`, `normal_modes_cartesian` (1/sqrt(amu)), optional `ir_intensities_km_mol` (null without APT), projection count, raw Hessian/reference diagnostics; inherit provider restrictions. Bounded tests and pending integration are in response.md |
| `ir-raman` | Analytic IR intensities and Raman activities. With `raman_frequency` (photon energy in Hartree) the activities are also given with the dynamic polarizability, from the first Taylor coefficient of `alpha(omega)` along every normal coordinate (`response::nuclear::optical_tensor_path`; restricted HF/STO-3G water and C1 ammonia mode/tensor and public-route acceptance validated; other models remain pending) | `driver::run("ir-raman", ...)`; each mode's polarizability derivative is the analytic mixed nuclear/electric response `response::nuclear::static_spectroscopic_derivative(..., mode, 2, ...)`; response controls `response_tolerance`, `response_max_iterations`, `response_restart`, `response_preconditioner_floor` | The frequency report plus `raman_activities` (Angstrom^4/amu), `depolarization_ratios`, `polarizability_derivatives` (Angstrom^2/sqrt(amu)), raw `raman_permutation_defect`, `response_converged` [Measured restricted HF STO-3G dynamic Raman component and public-route scope](validation-matrix.md#t017-dynamic-raman-mode-derivatives). |
| `polarizability` | Static electronic dipole polarizability | `driver::run("polarizability", ...)`; `response::polarizability::polarizability` | `polarizability` (Bohr^3, `alpha_ab = d mu_a / d F_b`), `isotropic`, `anisotropy`, raw `raw_asymmetry`, Angstrom^3 siblings; named double hybrids and custom SS/OS parents use the same driver, with second-order parent orbitals, noncanonical PT2 amplitudes and distinct-target exchange/VV10 kernels (physical acceptance in progress). `parent_polarizability`, `pt2_polarizability_correction`, `target_polarizability_correction` and unscaled aa/ab/bb PT2 tensors use e^2 Bohr^2/Hartree. Six independent static components are expanded; `raw_asymmetry` is null. Pair residuals have shape [6,3] for Taylor orders 0,1,2 with units [Hartree,e Bohr,e^2 Bohr^2/Hartree]; fresh orbital residuals and projector/number/frame defects are separate. Dynamic and higher correlated properties remain pending Prescribed populations and positive electronic temperature use `response::ensemble::EnsembleSolver::polarizability` through this same entry. Reports add the population model, fresh `response_residuals[3]`, iteration counts and `number_defects[3][2]` (e Bohr/Hartree); all tensor columns are solved independently. [Population-law contract](rust-api.md#fixed-population-and-thermal-static-response).  RKS/UKS selectors and HF aliases use the existing driver; ROKS/ROHF uses the existing constrained analytic route; [GKS/GHF nuclear paths include Coulomb/exchange and globally projected semilocal XC; full Cartesian double-hybrid tensors include PT2 amplitude/orbital response, with bounded acceptance scope](response.md#molecular-complex-spinor-reference-provider). Static GKS with a common full-spin PT2 weight returns the total, parent, unweighted pair, weighted pair and optional distinct semilocal or hybrid target tensors in e^2 Bohr^2/Hartree. The six independent components use the stationary spinor energy Hessian, with raw orbital, projector, particle-number and pair diagnostics. [Conventions and measured scope](response.md#r060-molecular-multipoles-and-derivatives). The integer collinear CPKS route reports `response_absolute_residuals` in e Bohr, one maximum fresh occupied-virtual equation defect per Cartesian field. `relativistic_method="dirac"` selects the molecular fixed-sector four-component DKS route. [Model and coverage](response.md#molecular-diraccoulomb-kohnsham). |
| `multipole-gradient` | Analytic nuclear derivatives of multipole moments | `driver::run("multipole-gradient", ...)` with `max_rank` (default 2, at most 8) and `origin`; nuclear term, relaxed CPKS density derivative, moving-basis moment integral derivative and moving-origin term | Per rank: `monomials`, `cartesian` moments and `gradient[monomial][3*atom+axis]`; rank 1 is the atomic polar tensor |
| `response-save` | First-order static response derivatives for reuse; with `frequencies` (and `frequency_damping`) also the amplitudes `X + Y` and `X - Y` of the frequency-dependent field response and its polarizability (`response::dynamic::dynamic_amplitudes`; The p126 strict-unit, zero-frequency static-limit and finite-frequency normal-dispersion checks pass; additional nuclear and frequency combinations remain separate.) | `driver::run("response-save", ...)` with optional `path` (JSON file) and `nuclear` (default true); field responses use `ResponseSolver::{project_ov, solve_ov, delta_density}`, nuclear ones the Hessian CPKS | `field[axis]` with `orbital_rotations[spin][virtual][occupied]` (U^F) and `density_derivative[spin]` (dP/dF, `H' = +F r`), `polarizability`, `nuclear_density_derivatives[3*atom+axis][spin]` (dP/dR), `conventions`, optional `path` |
| `third-derivative-2n1` | All Cartesian third derivatives of the molecular energy by the `2n + 1` rule: for every combined direction one first-order response is solved, the trial density path is continued through third order with the blocks that the projector condition alone fixes, the third-order energy coefficient is evaluated on the moving basis and grid, and the mixed derivatives follow from the polarization identity; no second-order response equation is solved. Same model scope as the nuclear Taylor driver. The p104 baseline validates HF/PBE/PBE0 cubic tensors against complete mixed responses; its independent four-step SCF-energy derivative fits also pass (p=1.995439646..2.002158542, minimum R2=0.999998593470, maximum response residual 4.314e-16, maximum Richardson defect 1.965e-7 Hartree/Bohr^3). See [measurements](response.md#reduced-order-baseline-repeat-two-measured-route-comparison-tests) | `driver::run("third-derivative-2n1", ...)`; `response::nuclear::cubic_force_constants_reduced` | `cubic_force_constants` in Hartree/Bohr^3, `translation_residual` |
| `cubic-fc-along` | Cubic force constants of a molecule contracted with a displacement pattern on the first index, `Phi(u)_jk = sum_i u_i d3E/(dx_i dx_j dx_k)`: the derivative of the Hessian along `probe`, from first-order responses of the combined directions (2n+1 rule) and the polarization identity; no full tensor is formed, and a normal mode as the probe gives its coupling to every pair of coordinates. The p104 baseline validates the PBE zero-field contraction against the complete tensor; this comparison alone does not establish an independent finite-difference derivative fit. See [measurements](response.md#reduced-order-baseline-repeat-two-measured-route-comparison-tests) | `driver::run("cubic-fc-along", ...)`; `response::nuclear::cubic_force_constants_along` | `contracted_force_constants` in Hartree/Bohr^3 per unit pattern Optional `bfield` and `field` select the shared complex London stationary nuclear path (integer occupations); `bfield_series_order` retains at least nuclear order+2. Reports include fresh `response_residuals`, `projector_defects`, `stationarity_defects`, `reference_defect`, `series_order`, `series_tail` and `accumulation_roundoff`. Intrinsic permutation reuse is declared; independent ordered asymmetry and dipole derivatives are not supplied on this magnetic path. Complex PT2 and extra Hamiltonian providers retain separate restrictions. See the externally biased nuclear scope below. Every distinct path also returns its actually solved relative `response_residuals`; these are retained before any tensor permutation reuse. Correlated integer RKS/UKS energy paths, including static reference_field bias, are connected; parent/target/SS/OS tensors and raw residuals are returned. Measured subsets and remaining combinations are in response.md. |
| `quartic-fc-along` | Quartic force constants of a molecule contracted with two displacement patterns, `Phi(u, v)_kl = sum_ij u_i v_j d4E/(dx_i dx_j dx_k dx_l)`, from responses through second order of the combined directions. The p104 baseline validates the PBE zero-field contraction against complete mixed responses; independent derivative and further-combination acceptance remain separate. See [measurements](response.md#reduced-order-baseline-repeat-two-measured-route-comparison-tests) | `driver::run("quartic-fc-along", ...)`; `response::nuclear::quartic_force_constants_along` | `contracted_force_constants` in Hartree/Bohr^4 per unit of both patterns Optional `bfield` and `field` select the shared complex London stationary nuclear path (integer occupations); `bfield_series_order` retains at least nuclear order+2. Reports include fresh `response_residuals`, `projector_defects`, `stationarity_defects`, `reference_defect`, `series_order`, `series_tail` and `accumulation_roundoff`. Intrinsic permutation reuse is declared; independent ordered asymmetry and dipole derivatives are not supplied on this magnetic path. Complex PT2 and extra Hamiltonian providers retain separate restrictions. See the externally biased nuclear scope below. Every distinct path also returns its actually solved relative `response_residuals`; these are retained before any tensor permutation reuse. Correlated integer RKS/UKS energy paths, including static reference_field bias, are connected; parent/target/SS/OS tensors and raw residuals are returned. Measured subsets and remaining combinations are in response.md. |
| `excitations` | Vertical excitation energies, oscillator strengths and (Tamm-Dancoff) rotatory strengths from linear-response TDA or RPA on a closed- or open-shell reference | `driver::run("excitations", ...)` with `nstates` (default 5), `td_method` (`tda` default, or `rpa`), `td_spin` (`auto` default: singlets for a restricted and the spin-resolved manifold for an unrestricted reference; `singlet`, `triplet`, `spin-resolved`), `td_tolerance` (1e-7), `td_max_iterations` (60), `amplitudes` (default false); `response::tddft::TdaSolver::{solve, solve_rpa}`; a Fermi-Dirac or fractionally charged reference is refused | Energy keys plus `td_method`, `td_spin`, `states[]` with `energy` (Hartree), `energy_ev`, `energy_cm1`, `wavelength_nm`, `oscillator_strength` (length gauge), `transition_dipole`, `dominant_transition`; for `tda` also `transition_momentum`, `transition_angular_momentum`, `rotatory_strength` (length gauge, about the coordinate origin), `rotatory_strength_velocity` (origin independent) and both in 1e-40 erg cm^3 (`*_cgs`); optional `amplitudes[spin][virtual][occupied]`; `oscillator_strength_sum` Full-spin `reference=ghf/gks`, `td_method=tda`, `td_spin=spinor/auto` uses the complete complex occupied-empty space, without exact total-spin labels. The isolated HF subset passes all 15 poles, 12 gradient and 144 Hessian entries and native/ASE/CLI routes with a fixed additional AO operator; full-spin RPA, moving SOC, semilocal nuclear mixed kernels and correlated pole derivatives remain pending. Full-spin Hessians currently report derivative tensors without harmonic analysis. See [bounded evidence and raw residuals](excited-state-extension.md#full-spin-hf-acceptance). `reference=roks/rohf`, `td_spin=reference-spin/auto` selects the common-orbital spin-conserving flag TDA with Berry weights (1,2,1). HF doublet/quartet all 312 gradient/Hessian fits and six native/ASE/CLI cases pass; spin-changing flag sectors and correlated pole derivatives remain unconnected. See [flag evidence](excited-state-extension.md#common-orbital-hf-acceptance). `td_nto=true` uses `properties::natural_transition_orbitals::decompose` for TDA. [Normalization and residuals](response.md#natural-transition-orbitals). `td_difference_density` reports signed unrelaxed difference weights for RKS/UKS/GKS. |
| `ecd-spectrum` | Gaussian-broadened vertical absorption and electronic circular-dichroism spectra from TDA states: `epsilon(nu) = (1/K_abs) sum f_n g_n(nu)` and `Delta epsilon(nu) = (nu/K_ecd) sum R_n g_n(nu)` with unit-area Gaussians, `K_abs = 1000 ln10/(pi N_A r_e) = 4.319e-9` and `K_ecd = 3 h c 1000 ln10/(32 pi^3 N_A) = 22.965e-40 erg cm^3` | `driver::run("ecd-spectrum", ...)` with `broadening_ev` (Gaussian standard deviation, required), `points` (1000), `energy_window_ev` (default: five standard deviations beyond the outer states, not below zero) and the `excitations` controls (`nstates` default 10); `td_method = rpa` and `td_spin = triplet` are refused | `states[]` as for `excitations`, `energies_ev`, `wavenumbers_cm1`, `molar_absorptivity`, `ecd_length` and `ecd_velocity` (L mol^-1 cm^-1), `oscillator_strength_sum`, `absorption_integral_oscillator_strength`, `rotatory_strength_sum_cgs` and `ecd_integral_rotatory_strength_cgs` (`[length, velocity]`, 1e-40 erg cm^3)  Correlated PT2 pole and electric/magnetic transition-residue corrections use the same entry and spin controls; bounded full-component and public acceptance passes (see the detailed model scope). |
| `td-state-gradient` | Analytic nuclear gradient of `E_SCF + omega`: singlet HF/pure-LDA closed-shell Z-vector route, or generic transported nuclear-path route | `driver::run("td-state-gradient", ...)` with `root` (1-based), `nstates`, `td_tolerance`, `td_max_iterations`; `response::tddft_grad::tda_gradient_with_grid` includes ECPs and retains its typed model restrictions. `td_gradient_route = path` uses `response::nuclear::excitation_path` for TDA/RPA, open shells, semilocal/hybrid all-electron gas-phase references. Native HF/LDA/hybrid TDA and GGA RPA directional paths pass four-step fits; full path-gradient public acceptance and other reference classes remain pending | Energy keys plus `root`, `excitation_energy`, `excited_state_energy`, `oscillator_strength`, `gradient` (Hartree/Bohr, including requested dispersion), `translation_residual`, `provenance`; measured path fits and raw residuals in response.md Full-spin `reference=ghf/gks`, `td_method=tda`, `td_spin=spinor/auto` uses the complete complex occupied-empty space, without exact total-spin labels. The isolated HF subset passes all 15 poles, 12 gradient and 144 Hessian entries and native/ASE/CLI routes with a fixed additional AO operator; full-spin RPA, moving SOC, semilocal nuclear mixed kernels and correlated pole derivatives remain pending. Full-spin Hessians currently report derivative tensors without harmonic analysis. See [bounded evidence and raw residuals](excited-state-extension.md#full-spin-hf-acceptance). `reference=roks/rohf`, `td_spin=reference-spin/auto` selects the common-orbital spin-conserving flag TDA with Berry weights (1,2,1). HF doublet/quartet all 312 gradient/Hessian fits and six native/ASE/CLI cases pass; spin-changing flag sectors and correlated pole derivatives remain unconnected. See [flag evidence](excited-state-extension.md#common-orbital-hf-acceptance). Collinear positive-SS/OS correlated TDA molecular gradients/Hessians now pass 1,248 complete component fits for custom HF-parent restricted/unrestricted C1 models and four native/ASE/CLI cases. Twelve small unrestricted Hessian entries use independently energy-validated fresh first gradients; all other entries use fresh static energies. No named semilocal/DH or ECP/ROKS/GKS correlated acceptance is inferred. See [correlated model and residuals](excited-state-extension.md#molecular-correlated-tda-nuclear-acceptance). |
| `hyperfine` | Point-nucleus nonrelativistic contact and spin-dipolar tensors of a converged collinear spin density, normalized by `two_s = N_alpha - N_beta`: `A = A_iso I + A_dip`; no electronic degenerate-state average or relativistic effective-spin projection. The dipolar AO operator is the analytic nucleus Hessian of the positive Coulomb potential plus `(4 pi/3) delta_ab` times the contact operator, with no trace projection | `driver::run("hyperfine", ...)`; `properties::hyperfine::{spin_dipole_operators, spin_hyperfine_tensor, HyperfineTensor::scaled}`; open-shell reference and optional `nuclear_g` (one signed value per atom). The same kernel supplies the compatibility contact scalar and full tensor. ECP nuclei have null all-electron outputs and a separate pseudo-valence spin density | `isotropic_coupling_per_g_mhz`, `dipolar_coupling_per_g_mhz`, `total_coupling_per_g_mhz`, `principal_couplings_per_g_mhz`, `principal_axes_per_g`; with `nuclear_g`: isotropic MHz/gauss, `dipolar_coupling_mhz`, `total_coupling_mhz`, `principal_couplings_mhz`, `principal_axes`. Raw trace, asymmetry, eigensystem and orthogonality residuals with per-atom `tensor_converged`; see tensor conventions and measurements below Explicit ROHF/ROKS retain common spatial orbitals. GHF/GKS require `epr_projection=single-kramers-pair` for the approximate odd-electron doublet model, including orbital-current A, spin expectation, spin-squared and metric diagnostics. These extensions are all-electron and common-origin; they do not spin-purify the determinant. Source-connected HF nuclear-A increment: `nuclear_order=0` (default) keeps static output; 1 adds gradients with null Hessian fields; 2 adds the full Hessian in `hyperfine_nuclear`. Optional zero-based `derivative_nuclei` selects target nuclei only; derivatives still span all 3N input Cartesian coordinates. Contact/dipolar/total `_per_g` fields are per signed nuclear-g convention; optional signed `nuclear_g` adds scaled fields. Native/Rust/CLI derivative units are MHz/Bohr and MHz/Bohr^2; ASE converts inverse length factors to its selected length units. Scope is all-electron integer UKS/UHF HF; PBE full6840 validation is running, and ROKS/GKS/ECP/PT2 nuclear-A are refused rather than replaced by a parent. Raw path coefficients retain derivative/k!, indexed 0..order; unavailable zero-order response diagnostics remain null. Original/inner linear residuals and reconstructed/source ratios are separate quantities. Exact adapter04 order mapping is measured; final consolidated artifact/public acceptance remains pending. |
| `nmr-shielding` | Nuclear magnetic shielding tensors `sigma_ij = d2E/(dm_i dB_j)` from the coupled-perturbed magnetic response; `gauge` = giao (default) or common-origin with `gauge_origin` (Bohr). Implemented with bounded molecular acceptance described in response.md | `driver::run("nmr-shielding", ...)`; `properties::magnetic::molecular_nmr_response` | Total, diamagnetic and paramagnetic tensors [nucleus][i][j] and isotropic values in ppm, response residuals Explicit VV10/learned target composition uses the same driver; new nonlocal coupled/public acceptance is in progress in p260_nonlocal_complex_response and response.md. Weighted parent/target nonlocal tensors are returned separately (ppm for shielding, atomic units for coupling), already included in parent/target totals. J amplitude residuals are Hartree; frame/projector residuals are dimensionless. With `relativistic_method="dirac"`, finite nuclear charge and magnetization radii in Bohr and optional `nuclei` select static four-component London-RMB shielding. Returns `shielding_tensors` in ppm with explicit-sector/orbital-relaxation parts and raw projector, reference and CPKS residuals. See [four-component scope](response.md#static-four-component-uniform-magnetic-response). The selected `dirac_interaction` can include magnetic Gaunt or frequency-independent Breit current terms. See [four-component current scope](response.md#uniform-magnetic-gaunt-and-breit-interactions). |
| `nmr-spectrum` | Isotropic NMR spectrum: one Lorentzian of unit area per nucleus at `delta = sigma_reference - sigma_isotropic`; paramagnetic spin-Zeeman shifts are excluded; optional nuclear multiplets use `nmr_couplings`. With `nmr_couplings` and `nuclear_g` every line is split into its first-order multiplet by the indirect spin-spin couplings to the other spin-active nuclei that are not equivalent to it (`nuclear_spins` = twice the spin per atom, `nmr_spectrometer_mhz` = proton frequency). Implemented with bounded molecular acceptance described in response.md | `driver::run("nmr-spectrum", ...)` | Shieldings as for `nmr-shielding`, `chemical_shifts` (ppm), and per plotted element the axis (ppm) and intensity (1/ppm) Correlated spin-free closed-shell FC/SD/PSO/DSO uses the full target/PT2 energy Hessian and relaxed DSO expectation through the same entry; independent acceptance passes all 925 component/channel fits in p259_correlated_spin_spin. The initial 17-case public/ordinary suite passes; the final rebuilt broader suite is tracked in response.md. Open-shell transverse pair response and nonlocal providers remain required. Explicit VV10/learned target composition uses the same driver; new nonlocal coupled/public acceptance is in progress in p260_nonlocal_complex_response and response.md. Weighted parent/target nonlocal tensors are returned separately (ppm for shielding, atomic units for coupling), already included in parent/target totals. J amplitude residuals are Hartree; frame/projector residuals are dimensionless.  Exact mode: `nmr_spin_model=exact`, `nmr_couplings=true`, `nuclear_g`, optional `nuclear_spins` (twice I), `nmr_spin_max_dimension` (1024). Shared isotropic spin Hamiltonian; ppm line positions, integrated line weights, raw Hz and dimensionless eigen residuals, area defects; bounded acceptance in response.md. With `relativistic_method="dirac"`, supply finite nuclear charge and magnetic RMS radii in Bohr for the London-RMB electronic response. The spectrum uses all nuclei's shielding tensors in ppm and optional total indirect J in Hz. Raw response, eigensystem and area residuals follow the common spin Hamiltonian. See [four-component spectrum](response.md#four-component-isotropic-nuclear-spin-spectrum). |
| `spin-spin-couplings` | Indirect nuclear spin-spin coupling of a closed-shell molecule: reduced tensors `K_MN = d2E/(dm_M dm_N)` as the sum of the diamagnetic spin-orbit expectation value (molecular grid), the paramagnetic spin-orbit term (imaginary singlet response with the magnetic Hessian) and the Fermi-contact and spin-dipole terms (triplet response with the spin-resolved orbital Hessian; the dipolar operator is the second derivative of the nuclear attraction integral with respect to the nucleus plus its contact part); with `nuclear_g` the isotropic constants `J = g_M g_N mu_n^2 K_iso / h`. All-electron references. Implemented with bounded molecular acceptance described in response.md UKS/GKS full complex Pauli response with optional fixed-spin-direction constraints is connected; all source/spin components and FC/SD-PSO mixed terms are retained. New molecular acceptance is tracked separately. | `driver::run("spin-spin-couplings", ...)`; `response::spin_spin::spin_spin_couplings` | Tensors in atomic units, `coupling_constants_hz` in Hz `nuclei` must contain distinct atom indices. `coupling_pairs[p]=[M,N]` preserves their input order; tensor axes are `[pair][M-axis][N-axis]`, so reversing a pair requires a transpose and does not make its Cartesian tensor symmetric. The common report includes `axis_convention`; tensors retain atomic units and optional isotropic constants are Hz. Labelled-pair reciprocity and bounded public-route acceptance are measured in response.md; full functional/reference coverage remains separate. Correlated spin-free closed-shell FC/SD/PSO/DSO uses the full target/PT2 energy Hessian and relaxed DSO expectation through the same entry; independent acceptance passes all 925 component/channel fits in p259_correlated_spin_spin. The initial 17-case public/ordinary suite passes; the final rebuilt broader suite is tracked in response.md. Open-shell transverse pair response and nonlocal providers remain required. Explicit VV10/learned target composition uses the same driver; new nonlocal coupled/public acceptance is in progress in p260_nonlocal_complex_response and response.md. Weighted parent/target nonlocal tensors are returned separately (ppm for shielding, atomic units for coupling), already included in parent/target totals. J amplitude residuals are Hartree; frame/projector residuals are dimensionless. With `relativistic_method="dirac"`, `MolecularTwoComponent::dirac_core_source_response` yields the indirect coupling and separates the moving positive-sector and CPKS contributions. Supply `nuclear_magnetic_rms_radii` in Bohr. [Four-component tests](response.md#four-component-indirect-nuclear-spin-coupling). |
| `fermi-contact-couplings` | Fermi-contact part of the indirect nuclear spin-spin coupling: triplet response of the spin density to the contact operator at one nucleus, read at the other. Implemented with bounded molecular acceptance described in response.md UKS/GKS full complex Pauli response with optional fixed-spin-direction constraints is connected; all source/spin components and FC/SD-PSO mixed terms are retained. New molecular acceptance is tracked separately. | `driver::run("fermi-contact-couplings", ...)`; `response::spin_spin::spin_spin_couplings` | Tensors in atomic units, `fermi_contact_hz` in Hz `nuclei` must contain distinct atom indices. `coupling_pairs[p]=[M,N]` preserves their input order; tensor axes are `[pair][M-axis][N-axis]`, so reversing a pair requires a transpose and does not make its Cartesian tensor symmetric. The common report includes `axis_convention`; tensors retain atomic units and optional isotropic constants are Hz. Labelled-pair reciprocity and bounded public-route acceptance are measured in response.md; full functional/reference coverage remains separate. Correlated spin-free closed-shell FC/SD/PSO/DSO uses the full target/PT2 energy Hessian and relaxed DSO expectation through the same entry; independent acceptance passes all 925 component/channel fits in p259_correlated_spin_spin. The initial 17-case public/ordinary suite passes; the final rebuilt broader suite is tracked in response.md. Open-shell transverse pair response and nonlocal providers remain required. Weighted parent/target nonlocal tensors are returned separately (ppm for shielding, atomic units for coupling), already included in parent/target totals. J amplitude residuals are Hartree; frame/projector residuals are dimensionless. |
| `exafs` | Single-scattering EXAFS `chi(k) = S0^2 sum_j f_j(k)/(k R_j^2) exp(-2 k^2 sigma_j^2) exp(-2 R_j/lambda) sin(2 k R_j + phi_j(k))` of one absorber from tabulated amplitudes and phases; no electronic-structure calculation. With angle-resolved tables (`exafs_scattering_angles`) and twice the phase shift of the absorber (`exafs_central_phase`) the double-scattering paths `0 -> j -> l -> 0` with `R_tot/2 <= exafs_rmax` are added in the plane-wave approximation, polarization-averaged for a K edge; scattering of third and higher order is not included. The p121 closed-chain, right-angle and single-scattering-limit checks pass; no energy derivative is involved. | `driver::run("exafs", ...)` | Paths (atom, element, distance in Bohr, sigma^2 in Bohr^2), `k` (1/Bohr), `chi`, `chi_weighted`, Fourier-transform magnitude on `r` (Bohr) |
| `xanes` | Core-level absorption: Tamm-Dancoff states in the core-valence-separated space of the K shell of `xanes_element` (or of `core_orbitals`), with frozen orbitals; all-electron basis required. `core_hole = delta-scf` converges the state with a hole in the first core orbital (unrestricted, occupations by maximum overlap with the initial occupied space) and reports the relaxed ionization energy, its difference to the frozen-orbital estimate `-eps_core` and the excitation energies shifted uniformly by it. The p122 one-electron core-hole limit and HF water O 1s relaxation checks pass; this does not validate state-specific excited-state relaxation. | `driver::run("xanes", ...)`; `response::tddft::TdaSolver::solve_core` | Excitation energies (Hartree), oscillator strengths, core orbital indices and energies, optional Lorentzian spectrum Relaxed core holes reuse the common physical-metric projector-IMOM SCF builder and preserve both spin labels. |
| `nexafs` | Polarization-resolved finite-basis electric dipole absorption with explicit occupied spin orbital lists or K-edge absorber indices. CVS-TDA evaluates unshifted response poles. The relaxed-core-ion model solves one ion per selected spin/core hole and uses its virtual energies and a projected, nonorthogonal determinant transition dipole. | `driver::run("nexafs", ...)` | Lines and Cartesian oscillator-strength tensors, Gaussian or Lorentzian spectrum and core-hole tracking residuals. Energies and full widths are Hartree, dipoles e Bohr and oscillator density inverse Hartree. Connected collinear all-electron CVS-TDA and per-hole relaxed-ion models. Complete continuum, spin-orbit-split edges, state-specific neutral relaxation, common-orbital and spinor response remain open. [Model limits and water results](response.md#r085-near-edge-xray-absorption-fine-structure). |
| `constrained-dft` | Charge and spin constraints on variational integer determinants. Each constraint supplies explicit AO operators or a moving Mulliken fragment projector with a target electron count. | `driver::run("constrained-dft", ...)` | Energy, gradient and forces in atomic units, optional L-BFGS geometry relaxation and coupled static electric response with fixed populations. The report retains the dual Lagrangian, multiplier susceptibility and raw residuals. [Constraint schema and measured results](response.md#r083-constrained-density-functional-theory). |
| `dft-plus-u` | Self-consistent molecular Dudarev and Liechtenstein-FLL Hubbard corrections and optional fixed-projector intersite terms on integer RKS, UKS, ROKS and complex GKS references. Supply `hubbard_pairs` to the common driver or `HubbardModel::with_intersite`. | `driver::run("dft-plus-u", ...)`; `scf::hubbard::run_collinear`; `scf::hubbard::run_roks`; `HubbardModel::evaluate_spinor` and `response_spinor` for fixed projectors | Total and site energies in Hartree, spin occupation matrices in electrons and raw orbital residual in Hartree. GKS uses spin-major occupation matrices. [Site schema and coverage](response.md#r087-molecular-dftu). |
| `pz-sic` | Variational Perdew-Zunger self-interaction correction on fixed-geometry real unrestricted integer orbitals. | `driver::run("pz-sic", ...)`; `scf::sic::run_pz` | Energy, self subtractions by orbital and spin, and raw orbital-gradient residual in Hartree. [Model and coverage](response.md#r088-molecular-pz-sic). |
| `flo-sic-fixed` | Fixed-descriptor Fermi-Lowdin self-interaction correction on a frozen real integer UKS determinant. | `driver::run("flo-sic-fixed", ...)`; `scf::sic::flo_sic::run_fixed_descriptors` | `fod_alpha` and `fod_beta` are Cartesian rows in Bohr. Energies and the parent orbital residual are Hartree. Descriptor densities are electrons/Bohr³. AO density and metric defects are dimensionless. [Model and coverage](response.md#fixed-descriptor-fermi-lowdin-sic). The common driver accepts `fod_gradient`, `fod_optimize` and `flo_self_consistent` for the real calculation. `fod_alpha_imaginary`/`fod_beta_imaginary` select complex descriptors with fixed parent orbitals. [Units, models and verification](response.md#fixed-descriptor-fermi-lowdin-sic). |
| `car-parrinello-md` | Microcanonical propagation of nuclei and occupied orbitals in the symmetric Löwdin frame, with a Euclidean fictitious kinetic metric and RATTLE. The full spinor implementation uses Hermitian constraints. | `driver::run("car-parrinello-md", ...)` | Trajectory, forces, both kinetic energies, extended energy drift and constraint defects. Time and velocities are atomic units. Masses are electron masses, positions Bohr and energies Hartree. PT2 off-shell forces, thermostats and alternative kinetic metrics remain open. [Trajectory model and water tests](response.md#r084-car-parrinello-molecular-dynamics). |
| `magnetizability` | Magnetizability `xi_ab = -d2E/(dB_a dB_b) = -(1/4) Tr[P (r_O^2 delta_ab - r_a r_b)] - (1/2) Tr[dP/dB_a L_b]` with a common gauge origin `O` (default: centre of nuclear charge); origin dependent in a finite basis. Implemented, not yet validated by acceptance tests | `driver::run("magnetizability", ...)`; `properties::magnetic::magnetic_density_response` | Total, diamagnetic and paramagnetic tensors in atomic units (e^2 Bohr^2/m_e), isotropic value also in 1e-6 cm^3/mol  Optional `response_order=1..4` and `magnetic_direction` (atomic magnetic field) add `magnetic_energy_coefficients` and actual factorial-restored `magnetic_energy_derivatives`, in Hartree along B(t)=t*direction, plus raw path/projector/stationarity residuals. Orbital zero-field collinear model; eight restricted HF/PBE/TPSS/PBE0 second/fourth magnetic energy fits pass; remaining tensor/reference/class acceptance is pending.  Optional `magnetic_directions` contains one through four Cartesian field directions. Returns the actual mixed energy derivative for B=sum_j t_j*d_j (dimensionless t_j), Hartree, plus raw response/projector/stationarity defects and polarization roundoff. Repeated and non-orthogonal directions are supported through the same London/common-origin field driver. Matching `electric_directions` optionally adds electric parts to every mixed parameter: E=sum t_j e_j, B=sum t_j b_j. `magnetic_mixed_derivative` then reports the joint energy derivative, Hartree per dimensionless parameters; the same analytic field/metric response is reused.  Static RKS/UKS double-hybrid orbital tensors use the shared complex orbital/PT2 field path for common-origin and London gauges. Parent, target, weighted PT2 and unscaled SS-alpha/OS/SS-beta tensors are reported separately with raw orbital, pair, metric and reference diagnostics. The full total energy is the same post-SCF model as the common energy driver. High-order correlated contractions remain explicitly unsupported. New tensor/public acceptance is pending. Explicit `reference=gks` or `reference=ghf` with `gauge=common-origin` selects the static all-electron spinor orbital tensor. The spinor report includes the fixed-density diamagnetic and relaxed paramagnetic terms and raw response, symmetry and imaginary defects. The finite-basis tensor depends on the origin. The rebuilt HSEsol ammonia-cation scope passes all eighteen independent four-step fits and public route parity. Maximum raw relative response residual 5.125e-11 and zero imaginary susceptibility defect. [Spinor model and acceptance](response.md#static-spinor-orbital-magnetizability-at-a-common-gauge-origin) With `relativistic_method="dirac"`, finite charge radii in Bohr and the London-RMB basis give static `magnetizability` in `e^2*bohr^2/m_e` and `magnetic_dipole` in atomic units. Reports give explicit-sector/orbital-relaxation parts and raw projector, reference and CPKS residuals. See [four-component scope](response.md#static-four-component-uniform-magnetic-response). The selected `dirac_interaction` can include magnetic Gaunt or frequency-independent Breit current terms. See [four-component current scope](response.md#uniform-magnetic-gaunt-and-breit-interactions). |
| `magnetic-quadrupole` | Electronic symmetric-traceless tensor Q = -dE/dG for a source-free field B(r) = B0 + G(r-O). The common Poincaré gauge uses A = (B0 cross r)/2 + ((G r) cross r)/3 with r measured from O. Includes orbital and g=2 Pauli spin terms and the current-invariant meta-GGA convention. | `driver::run("magnetic-quadrupole", ...)` | Orbital, spin and total magnetic quadrupoles, magnetic dipole and raw SCF residual. Magnetic quantities use atomic units. Origin defaults to the centre of mass in Bohr. Connected zero-bias RKS/UKS/ROKS and finite-bias GKS all-electron moments. At changed expansion origin the spin tensor mixes with the spin dipole, and finite common-gauge basis orbital values retain gauge dependence. ECP and PT2 magnetic-gradient operators are pending. [Field convention and measured scope](response.md#r082-molecular-magnetic-quadrupole). |
| `franck-condon` | Franck-Condon progression of a singlet TDA state in the vertical-gradient model: Huang-Rhys factors `S_k = f_k^2/(2 w_k^3)` from the excitation-energy gradient along the ground-state modes, zero temperature, no Duschinsky rotation. With `temperature` > 0 the initial vibrational states carry Boltzmann populations (at most `fc_max_hot` quanta); `fc_model` = vertical-hessian builds the excited-state surface from its analytic gradient and Hessian at the ground-state geometry (TDA or RPA, any functional of the path engine), with its own frequencies and the Duschinsky rotation of the normal modes, and the Franck-Condon integrals follow from the recursion for rotated, displaced oscillators of different frequencies. The p103 H2 vertical-model finite-temperature normalization and zero-temperature vertical-Hessian sum rules pass; broader vibronic spectra remain pending. | `driver::run("franck-condon", ...)` | Huang-Rhys factors, reorganization and 0-0 energies (Hartree), line energies, Franck-Condon factors and quanta, optional line shape |
| `excited-optimization` | Geometry relaxation on the surface of a singlet TDA state by L-BFGS on the analytic excited-state gradient, the state followed by the overlap of squared amplitudes (Hartree-Fock, pure LDA and learned-functional references). Restricted HF/STO-3G isolated-state C1 acceptance validated; other models and crossings remain pending | `driver::run("excited-optimization", ...)` | Relaxed positions (Bohr), excited-state and excitation energies (Hartree), gradient, convergence flag and trajectory [Exact model, full-component acceptance and residuals](validation-matrix.md#t120-isolated-singlet-tda-optimization). Explicit `td_gradient_route=path` connects analytic nuclear-path gradients; new-route acceptance is pending the coordinated rebuild. [Spin/model limits and implementation stages](excited-state-extension.md). |
| `optical-activity` | Optical rotation: electric dipole - magnetic dipole polarizability `G'(omega)` of the paired response in the modified velocity gauge by default (`gauge` = velocity or length; the length gauge depends on `gauge_origin` in a finite basis), `beta = -Tr G'/(3 omega)` and the specific rotation `28800 pi^2 N_A a0^4 nu^2 beta / M`. Implemented, not yet validated by acceptance tests | `driver::run("optical-activity", ...)`; `response::dynamic::dynamic_optical_rotation` | `rotation_tensors` and `beta` (atomic units) per frequency, `specific_rotation` in deg/(dm g/cm^3); the model field distinguishes common-origin length gauge from modified velocity gauge with static subtraction. Nuclear optical-tensor derivative acceptance is tracked separately; periodic and correlated mixed dynamic response remain pending |
| `broken-symmetry-exchange` | Heisenberg coupling `J` of `H = -2 J S_A . S_B` from the high-spin determinant (the multiplicity of the controls) and the broken-symmetry determinant: `J = -(E_HS - E_BS)/(<S^2>_HS - <S^2>_BS)`, with the weak- and strong-overlap limits. With `sites` (the atoms of all magnetic sites) and `site_sizes` (atoms per site) more than two sites are treated: the high-spin determinant and the broken-symmetry determinants with one or two sites reversed are mapped on `E = E_0 - 2 sum J_ij S_i S_j s_i s_j` and all `J_ij` follow from one linear system (`unpaired` per site is required). Implemented, not yet validated by acceptance tests | `driver::run("broken-symmetry-exchange", ...)` | Energies (Hartree), `<S^2>` of both determinants, `exchange_coupling` in Hartree and cm^-1, Mulliken spin populations of the sites |
| `vcd` | Vibrational circular dichroism: atomic axial tensors from the overlap of the nuclear and magnetic coupled-perturbed responses with distributed gauge origins at the atoms by default (`gauge` = distributed-origin or common-origin), rotational strengths `R_k = P_k . M_k` of the harmonic modes; the common-origin variant depends on the origin in a finite basis. Ordinary RKS/UKS subsets are validated; normalized pair-state AAT and correlated Hessian/APT pass 2,025 and 2,250 complete component fits on HF/LDA/GGA/meta-GGA/hybrid parents with RKS/UKS | `driver::run("vcd", ...)` | Atomic axial tensors (e Bohr/hbar), frequencies (cm^-1), dipole and rotational strengths in atomic units and 1e-44 esu^2 cm^2 |
| `spin-susceptibility` | Static spin susceptibility `chi = g_e^2 mu_B^2 / eta` of an unrestricted Fermi-Dirac reference, with the spin stiffness `eta = d(mu_alpha - mu_beta)/d(delta)` of the ensemble response (full Hartree, exchange and exchange-correlation kernel); orbital parts are those of `magnetizability`. Generic molecular Mermin response; measured acceptance in response.md | `driver::run("spin-susceptibility", ...)`; `response::ensemble::EnsembleSolver` | `spin_susceptibility` in atomic units (e^2 Bohr^2/m_e) and 1e-6 cm^3/mol, `spin_stiffness` (Hartree)  `spin_g_factor` defaults to the free-electron value. Reports include `bare_spin_susceptibility`, `bare_spin_stiffness`, `spin_enhancement`, `spin_density_response[spin][AO][AO]`, raw response/number and reference commutator residuals. Magnetization convention M=-g(N_alpha-N_beta)/4; density response is per atomic magnetic field. Fixed total number, spin-only longitudinal response; GGA/meta-GGA/hybrid validation and remaining combinations are tracked in CSV. |
| `g-tensor` | Electronic g tensor `g = g_e + dg_RMC + dg_GC + dg_SO` of an unrestricted open-shell reference: relativistic mass correction, gauge correction and the orbital Zeeman / one-electron spin-orbit cross term from the magnetic response, with nuclear or effective `soc_charges`; no two-electron spin-orbit terms. `gauge = london` (default) uses the London magnetic response and the phase derivatives of the spin-orbit operator, so the tensor does not depend on the origin; `gauge = common-origin` keeps one origin for all basis functions. The p120 nondegenerate water-cation origin/translation and one-centre-limit checks pass; the independent mixed-energy derivative fit remains pending. | `driver::run("g-tensor", ...)`; `properties::magnetic::magnetic_density_response`; `properties::g_nuclear::PauliGNuclearContext` | `g_tensor` [field axis][spin axis], isotropic value, shifts and the three parts in ppm Explicit ROHF/ROKS use the constrained common-orbital response. GHF/GKS require `epr_projection=single-kramers-pair` and return pair-axis `g_tensor`, invariant `g_gram`, spin and angular/operator diagnostics. These extensions support the all-electron common-origin one-electron model; London response and higher multiplet projection remain unavailable. Separate `nuclear_order=1/2` uses integer all-electron HF/UHF and the one-electron Pauli London model: `g_gradient[3N,3,3]` and `g_hessian[3N,3N,3,3]` are actual derivatives in `1/bohr` and `1/bohr^2`; the Hessian is null for order one. ASE converts these to inverse Angstrom powers. Mass/gauge/spin-orbit parts and real/mixed diagnostics remain separate. Nuclear DFT, RO/spinor, ECP/ensemble, nonzero electric field and stability-following requests remain explicit errors; explicit zero field is allowed. Static order zero keeps the preceding capabilities. No two-electron SOC, picture-change or spectral-accuracy claim follows. |
| `spin-spin-zfs` | Direct electron spin-spin D only, H_D=S.D.S; SOC and spin purification excluded. All-electron integer UHF/HF, S>=1. | `driver::run("spin-spin-zfs", ...)`; `properties::zero_field_splitting::spin_spin_nuclear_derivatives` | `d_tensor` [3,3] Hartree, optional `d_gradient` [3N,3,3] Hartree/Bohr and `d_hessian` [3N,3N,3,3] Hartree/Bohr^2. Strict `nuclear_order=0/1/2` (default0); unrequested derivatives null. Atom-major xyz, raw SCF/path/coefficient diagnostics. The backend has 810 bounded HF component fits. One actual compatible native artifact and installed local wheel pass all 25 public cases; all 387 D/gradient/Hessian values agree with the retained canonical Rust report. This is API parity, not additional independent fits. No total D or DFT nuclear-D claim. |
| `roks` | Restricted open-shell Kohn-Sham / Hartree-Fock: one set of spatial orbitals with `N_beta` closed and `N_alpha - N_beta` open shells, Roothaan effective Fock matrix with Pulay extrapolation, started from the unrestricted solution; analytic gradient from the spin densities and the energy-weighted density `sum_s P_s F_s P_s`. No continuum solvation, nonlocal correlation or fractional occupations. Implemented, not yet validated by acceptance tests | `driver::run("roks", ...)`; `scf::roks::run_roks` | Energy and components (Hartree), `s_squared = S(S+1)`, gradient (Hartree/Bohr) |
| `roks-hessian` | Analytic Cartesian Hessian of the restricted open-shell Kohn-Sham / Hartree-Fock energy. In the frame transported with the overlap the state is a pair of orthogonal projectors on the closed and on the open orbitals; idempotency and orthogonality fix every block of a Taylor coefficient except the closed-open, closed-virtual and open-virtual rotations, which solve `sum_s [F_s, D_s] = 0` on the orbital Hessian of the reference. `H_ii = 2 c(e_i)` and `H_ij = c(e_i + e_j) - c(e_i) - c(e_j)` from the second-order energy coefficients `c` of nuclear paths, each with one first-order solve; mixed nuclear-field paths give the dipole derivatives and the infrared intensities; integrals, Becke weights and AO values are differentiated analytically, no displaced calculation. HF, LDA, GGA, meta-GGA, global and range-separated hybrids; integer occupations, gas phase, all-electron, exact Coulomb integrals in memory. Implemented, not yet validated by acceptance tests | `driver::run("roks-hessian", ...)`; `response::nuclear::open_shell_hessian` | `hessian` in Hartree/Bohr^2, `gradient` in Hartree/Bohr, `frequencies_cm1` in cm^-1; raw `flag_defect`, `stationarity_defect`, `reference_energy_error`, `response_residual` [Weighted XC controls and diagnostics](rust-api.md#weighted-molecular-xc). |
| `roks-polarizability` | Static limit of the constrained paired common-orbital susceptibility; closed-open, closed-virtual and open-virtual rotations; generic semilocal/hybrid/RSH, scalar nonlocal/continuum and learned density Hessians | `driver::run("roks-polarizability", ...); response::nuclear::open_shell_dynamic_polarizability` | Full independently contracted `polarizability[3][3]`, dipole, isotropic/anisotropy, reference stationarity, metric and fresh response residuals; e^2 Bohr^2/Hartree; acceptance scope in R001 |
| `dc` | Divide-and-conquer Kohn-Sham: every fragment of atoms (`dc_fragments`, one index per atom; default one atom per fragment) with the buffer atoms within `dc_buffer` forms a subsystem whose Kohn-Sham and overlap blocks are diagonalized; one Fermi function of smearing `dc_smearing` with a common chemical potential occupies all local orbitals and the density is assembled with the partition weights 1, 1/2, 0; no global diagonalization. The energy is the Kohn-Sham functional of that density minus `kT S`. The gradient is the derivative of that free energy: besides the fixed-density terms it carries the response of the assembled density, entropy and chemical potential to the subsystem blocks through one adjoint (relaxation) equation with the Kohn-Sham kernel. HF, LDA, GGA, meta-GGA and hybrids without solvent, nonlocal correlation or density fitting. The molecular HF/PBE/PBE0 incomplete-buffer partition-gradient four-step fits and complete-buffer Mermin limits pass in the p102 baseline. `energy` is the electronic Mermin free energy, `internal_energy` the internal energy (Hartree); Cartesian `gradient` is its partition-relaxed derivative (Hartree/Bohr). Fixed partition membership is assumed. Public, periodic and other-class acceptance remain separate. See [raw fits](response.md#phase-b-molecular-divide-and-conquer-gradient-acceptance). | `driver::run("dc", ...)`; `scf::divide_conquer::run_divide_conquer` | Energies in Hartree, `gradient` in Hartree/Bohr, `dc_buffer` in Bohr, `dc_smearing` in Hartree |
| `mrsf` | Mixed-reference spin-flip TDA on the restricted open-shell triplet: spin-flip matrix `A+` of the `M_S = +1` reference and the pairing block `U` with the `M_S = -1` reference, singlet channel `A+ - U`, triplet channel `A+ + U`; exact exchange and the non-collinear kernel with its density, gradient and kinetic terms (`spin_flip_kernel` = noncollinear or collinear; local limit where the spin density vanishes), open-to-open coupling as published (`mrsf_coupling` = mrsf) or union-space (`ci`). Implemented, not yet validated by acceptance tests | `driver::run("mrsf", ...)`; `response::mrsf::mrsf_tda` | Singlets `S0..` and triplets `T0..` with `omega` from the reference triplet, total energy (Hartree), `<S^2>`, oscillator strengths from `S0` |
| `mrsf-state` | Analytic nuclear gradient of the total energy `E_triplet + omega` of one mixed-reference spin-flip state (`mrsf_state = S0, S1, ..., T0, ...`). The restricted open-shell triplet is followed along each Cartesian direction in the frame transported with the overlap, with its closed-open, closed-virtual and open-virtual rotations solved at first order; the moving projectors give orthonormal closed, open and virtual orbitals, the spin-flip matrices are assembled from the series of the Kohn-Sham blocks and of the projected kernel responses (exact exchange and the non-collinear kernel as series on the moving grid), and the eigenvalue of the channel matrix follows from non-degenerate perturbation theory. No displaced calculation. A root within 1e-6 Hartree of another root of its channel is refused; gas phase, all-electron, integer occupations. Implemented, not yet validated by acceptance tests | `driver::run("mrsf-state", ...)`; `response::nuclear::{mixed_reference_path,mixed_reference_surface}` | `gradient` in Hartree/Bohr, `omega` and `state_energy` in Hartree; `omega_reference_error` is the distance of the path series to the direct solution; raw `root_gap`, `flag_defect`, `stationarity_defect`, `response_residual` |
| `mrsf-hessian` | Analytic nuclear Hessian of the total energy of one mixed-reference spin-flip state from the second-order coefficients of the same path series (`H_ii = 2 c(e_i)`, `H_ij = c(e_i + e_j) - c(e_i) - c(e_j)`), with the rotations of the reference solved through second order, its gradient and the harmonic frequencies. Implemented, not yet validated by acceptance tests | `driver::run("mrsf-hessian", ...)`; `response::nuclear::mixed_reference_surface` | `hessian` in Hartree/Bohr^2, `gradient` in Hartree/Bohr, `frequencies_cm1` in cm^-1 |
| `spin-orbit` | One-shot spin-orbit coupling: `H_SO = (alpha^2/2) sum_A Z_A (l_A . s)/r_A^3` (nuclear or effective `soc_charges`, no two-electron terms) added to the converged collinear Kohn-Sham operator with its spin axis along `spin_axes` and diagonalized in the Kohn-Sham spin orbitals; the band energies carry no self-consistency in the spin-orbit term, and `spin_orbit_self_consistent` adds the energy of the self-consistent two-component state of every spin axis (`response::spinor::spin_orbit_spinor_reference`). Implemented, not yet validated by acceptance tests | `driver::run("spin-orbit", ...)` | Spinor energies, band energy and spin-orbit energy (Hartree) per spin axis Set `reference=gks` (`--reference gks` in the CLI) to include the generated one-electron operator in the spin-covariant SCF. `self_consistent_energies` and `self_consistent_orbital_residuals` are paired per axis in Hartree. |
| `spinor-polarizability` | Dynamic polarizability `alpha_ab(omega)` of the self-consistent two-component state of a molecule: the spinor SCF with `H_SO = soc_scale (alpha^2/2) sum_A Z_A (l_A . s)/r_A^3` in its Fock matrix (spin frame along `spin_axis`, started from the collinear reference), then the coupled linear response of its `2N` spinors with complex amplitudes `X, Y` at each of `frequencies` (optional `frequency_damping`). Coulomb and exact exchange act on the whole spinor density; every density-functional term (semilocal, nonlocal correlation, learned functional, reaction field) is the collinear functional of the two spin-diagonal blocks, in the reference and in the kernel alike, so the response operator is the second derivative of the SCF energy. Without damping the operator is Hermitian and, on a magnetized state, indefinite; a frequency at a pole is reported through the residual, not regularized. No two-electron spin-orbit terms; effective core potentials are refused. Implemented, not yet validated by acceptance tests | `driver::run("spinor-polarizability", ...)`; `response::spinor::{spin_orbit_spinor_reference,spinor_dynamic_polarizability}` | `polarizability_real`, `polarizability_imag` as `[frequency][a][b]` and `polarizability_axial[frequency][c] = eps_cab Im alpha_ab / 2` in e^2 Bohr^2/Hartree; `spinor_energy`, `spin_orbit_energy` in Hartree; `spin_expectation` in units of hbar |
| `magnetic-anisotropy` | Magnetic anisotropy energy by the force theorem: band energies (sum of the occupied spinor energies of `spin-orbit`) of the spin axes `spin_axes` relative to the easiest axis. Implemented, not yet validated by acceptance tests | `driver::run("magnetic-anisotropy", ...)` | `band_energies`, `anisotropy_energies` (Hartree) per spin axis |
| `magnon` | Transverse spin excitations of an unrestricted reference: spin-flip TDA `A = delta_ij F'_ab - delta_ab F_ij - c_x (ab\|ij)` in the spin-lowering and the spin-raising block, with exact exchange and the non-collinear kernel with its density, gradient and kinetic terms (`spin_flip_kernel` = noncollinear or collinear); the lowest spin-lowering energy of a high-spin reference is the Goldstone mode. Implemented, not yet validated by acceptance tests | `driver::run("magnon", ...)`; `response::spin_flip::spin_flip_tda` | `spin_lowering_energies`, `spin_raising_energies`, `goldstone_energy` (Hartree) |
| `mcd` | Default `closed-shell-tda`: canonical A/B terms from complete singlet TDA. Explicit `mcd_model=thermal-manifold`: canonical ensemble of supplied real symmetric sublevel Hamiltonian, magnetic-moment and normalized helicity operators; population C and interlevel mixing separate. Artifact-specific Rust/Python/ASE/both-CLI numerical tests and independent supplied-model comparisons pass. | `driver::run("mcd", ...)` | Default A/B fields retained; thermal model returns populations, finite-field normalized signal, population-C/mixing/total response tensors and explicit units  New source models: `thermal-complex-manifold` retains Hermitian real/imaginary quadratures; `molecular-casci-soc` generates spin-complete CASCI/SOC states in DFT/HF orbitals and fixed/isotropic thermal cross sections. CASCI is not TDDFT; scoped shared Rust/Python/ASE/both-CLI artifact02 acceptance and19 actual molecular/complex replays pass. Moving SOC, quadratic magnetic completion and measured-spectrum agreement remain open. Source follow-up adds explicit `mcd_magnetic_basis=london-linear-frozen-soc`, first-order London metric/Coulomb/optical completion, and per-state spin expectation/S-squared diagnostics; scoped linked acceptance now passes13 Rust cases and46 Python/ASE cases per CLI, with source1044-fit matrices reproduced exactly by the public core. Moving SOC and general relativistic/ROKS/GKS RPA remain separate. |
| `td-hessian` | Analytic Hessian of `E_SCF + omega` for one isolated TDA/RPA state, from nuclear-path series and polarization identities, with harmonic analysis. All-electron integer occupations, gas phase, semilocal/hybrid functionals, without density fitting or nonlocal correlation. Native HF/LDA/hybrid TDA and GGA RPA directional first/second derivatives pass four-step fits; full Cartesian tensor/public acceptance remains pending | `driver::run("td-hessian", ...)` with `root`, `nstates`, `td_method`, `td_spin`; `response::nuclear::excited_state_hessian` | `hessian`, `excitation_hessian` (Hartree/Bohr^2), `gradient` (Hartree/Bohr), `frequencies_cm1`, `reference_residual`, `response_residual`; directional raw measurements in response.md Full-spin `reference=ghf/gks`, `td_method=tda`, `td_spin=spinor/auto` uses the complete complex occupied-empty space, without exact total-spin labels. The isolated HF subset passes all 15 poles, 12 gradient and 144 Hessian entries and native/ASE/CLI routes with a fixed additional AO operator; full-spin RPA, moving SOC, semilocal nuclear mixed kernels and correlated pole derivatives remain pending. Full-spin Hessians currently report derivative tensors without harmonic analysis. See [bounded evidence and raw residuals](excited-state-extension.md#full-spin-hf-acceptance). `reference=roks/rohf`, `td_spin=reference-spin/auto` selects the common-orbital spin-conserving flag TDA with Berry weights (1,2,1). HF doublet/quartet all 312 gradient/Hessian fits and six native/ASE/CLI cases pass; spin-changing flag sectors and correlated pole derivatives remain unconnected. See [flag evidence](excited-state-extension.md#common-orbital-hf-acceptance). Collinear positive-SS/OS correlated TDA molecular gradients/Hessians now pass 1,248 complete component fits for custom HF-parent restricted/unrestricted C1 models and four native/ASE/CLI cases. Twelve small unrestricted Hessian entries use independently energy-validated fresh first gradients; all other entries use fresh static energies. No named semilocal/DH or ECP/ROKS/GKS correlated acceptance is inferred. See [correlated model and residuals](excited-state-extension.md#molecular-correlated-tda-nuclear-acceptance). |
| `td-ecd-giao` | Rotatory strengths of TDA or RPA states in the length gauge with London orbitals, `R_n = d_n . [g . (X - Y)_n + w_n W . (X + Y)_n]`: `g_ai = Fhat_ai - e_i Shat_ai` is the static London property gradient of the magnetic response equations (explicit London derivative of the Kohn-Sham matrix with its two-electron and semilocal parts, exchange of the occupied-occupied overlap response, overlap derivative), and `W_mn = (1/2) [(R_n - O) x <m\|r - O\|n>]` is the ket-centre matrix of the basis that moves with an oscillating field. A paired state carries `X - Y = (A + B)(X + Y) / w` on the magnetic moment. The common-origin and velocity values of the same states are reported next to it. All-electron, gas phase, integer occupations; the supported functionals are those of the London magnetic response. The p117 origin, translation, mirror and common-origin TDA-limit checks pass; the independent four-step London optical-rotation residue fit remains pending. | `driver::run("td-ecd-giao", ...)`; `response::london_ecd::london_rotatory_strengths` | `states[].rotatory_strength_london` in atomic units (`alpha E_h a0^3`) and `_cgs` in 1e-40 erg cm^3; excitation energies in Hartree and eV |
| `roa` | Raman optical activity: derivatives along every normal mode of the electric dipole polarizability, of the optical rotation tensor `G'` and of the dipole-quadrupole polarizability `A` at the photon energies `frequencies`, from the paired response expanded along the mode in the transported frame of the nuclear Taylor driver; invariants `alpha^2`, `beta(alpha)^2`, `alpha G'`, `beta(G')^2`, `beta(A)^2` and the circular intensity differences of forward, backward and right-angle scattering. Velocity gauge by default (`gauge` = velocity or length): the tensors are the differences of the momentum responses at the frequency and at zero frequency, observed with the velocity forms of the dipole and of the quadrupole, so they change with `gauge_origin` exactly as in a complete basis and the invariants do not depend on it; the length gauge depends on the origin in a finite basis. Implemented, not yet validated by acceptance tests | `driver::run("roa", ...)`; `response::nuclear::optical_tensor_path` | Tensor derivatives in atomic units per Bohr sqrt(amu), invariants in their squares, `cid_*` dimensionless, `frequencies_cm1` |
| `hypermagnetizability` | Static hypermagnetizability `eta_ab,cd = d2 alpha_ab / (dB_c dB_d)` as the mixed third derivative of the induced dipole in combined electric and magnetic fields, from order-by-order density-projector equations with a complex Hermitian density (real part: orbital Hessian of the electric response; imaginary part: magnetic Hessian with exact exchange only) and the polarization identity of field directions; also the static polarizability and the magnetizability of the same response. London orbitals by default (`gauge` = london or common-origin): the overlap, one-electron and two-electron integrals are power series in the field, the reference orbitals are transported with the field-dependent metric, and the functional sees the gauge-invariant kinetic energy density, so the tensors do not depend on the gauge origin; the field-free two-electron integrals act through the integral-direct engines and their field coefficients are contracted quartet by quartet, so no integral tensor is stored; the common-origin route depends on `gauge_origin` in a finite basis. With `frequency` > 0 the tensor is the second magnetic-field derivative of the frequency-dependent polarizability of the paired response along magnetic paths (the first derivative, the Faraday tensor, is reported too); `frequency` = 0 gives the static tensor. Molecular restricted-HF static subset validated | `driver::run("hypermagnetizability", ...)`; `response::magnetic_taylor::hypermagnetizability` | `hypermagnetizability`, `polarizability`, `magnetizability`, `hypermagnetizability_anisotropy` in atomic units Restricted HF H2 static tensor: all 81 ordered components agree with independently contracted fourth energy derivatives; Cotton-Mouton orientation and four-frequency static-limit fit accepted. Periodic/correlated and other-class acceptance remains pending. |
| `cotton-mouton` | Cotton-Mouton constant per molecule `C = (2 pi / 27) [D_eta + 2 (alpha_ab xi_ab - 3 alpha xi) / (15 k T)]` with `D_eta = (eta_ab,ab - eta_aa,bb / 3) / 5` from the static hypermagnetizability, polarizability and magnetizability of `hypermagnetizability` at `temperature`. Molecular restricted-HF static subset validated | `driver::run("cotton-mouton", ...)`; `response::magnetic_taylor::hypermagnetizability` | `cotton_mouton_constant`, `orientational_term`, `hypermagnetizability_anisotropy` in atomic units; `temperature` in K Restricted HF H2 static tensor: all 81 ordered components agree with independently contracted fourth energy derivatives; Cotton-Mouton orientation and four-frequency static-limit fit accepted. Periodic/correlated and other-class acceptance remains pending. |
| `magneto-optical` | Faraday rotation tensor `Im d alpha_ab(w) / dB_c` and its invariant `eps_abc Im d alpha_ab / dB_c / 6`: the first-order coefficient in a magnetic field of the frequency-dependent polarizability of the paired response, solved order by order in the transported frame of the field path (complex projector reference, real and imaginary occupied-virtual parts from the paired equations of the reference). For an open-shell reference (`magneto_optical_source` = auto or spin-orbit) the tensor is instead the first-order change of the polarizability under the one-electron spin-orbit coupling `(alpha^2/2) sum_A Z_A l_A.s/r_A^3` (`soc_charges`) with the spins along each axis `k`, `Im d alpha_ab/d lambda` as `magneto_optical_tensor[a][b][k]`: the magnetization-induced Faraday and Kerr response; `magneto_optical_source` = field selects the external-field tensor. London orbitals by default (`gauge` = london or common-origin), without stored integral tensors. Implemented, not yet validated by acceptance tests | `driver::run("magneto-optical", ...)`; `response::field_path::faraday_tensor` | `faraday_tensor`, `faraday_invariant`, `polarizability` in atomic units; `frequency` in Hartree Optional `frequency_damping` (Hartree) uses the causal complex equation. Reports retain both parts of the molecular polarizability and the field or spin-orbit derivative, plus the raw relative and reference residuals. |
| `bfield` | Self-consistent field of a molecule in a finite uniform magnetic field with London orbitals. The integrals are entire functions of the field: overlap and one-electron matrix are the sums of their field series, the field-free two-electron integrals act through the integral-direct engines and their field coefficients are contracted quartet by quartet, with the first two omitted coefficients checked (`series_order`, `series_tail`; the molecule is moved to its geometric centre, which changes only constant phases). The semilocal functional is evaluated from the complex orbitals on the grid with the gauge-invariant kinetic energy density; no current dependence. Orbitals solve the complex Hermitian eigenproblem; the spin Zeeman term `g_e B.S/2` is added with the spins along the field (`spin_zeeman`). HF, LDA, GGA, meta-GGA, hybrids; no ECP, solvent, nonlocal correlation or density fitting. Energy/gradient four-step fits and physical limits measured; remaining combinations tracked in CSV | `driver::run("bfield", ...)`; `scf::magnetic::magnetic_field_scf` | `bfield` in atomic units (2.35052e5 T), `field_energy` and orbital energies in Hartree  `bfield_initial_guess` defaults to auto; failed zero-field convergence may use a direct London-metric core seed. Requested stability following is never bypassed. `energy` is null without a converged zero-field reference; `field_energy` and `converged` describe the actual field state. `series_tail` measures omitted coefficients. Optional `field` (Hartree/e/Bohr) adds a simultaneous uniform electric field, including nuclear coupling and restoration of the charged-system laboratory energy origin. `electric_field` records the applied field. |
| `bfield-gradient` | Analytic nuclear gradient of the energy of the `bfield` state. The derivative of a London orbital with respect to its centre, `exp(-i A.r)[-d chi/dr - (i/2)(B x e).r chi]`, is a combination of primitives on the same centre, so the derivative overlap, one-electron and two-electron integrals come from the same field series; the bra centres follow by complex conjugation, the nuclei of the attraction by exact dual numbers. The quadrature points and Becke weights follow the atoms and the phases the centres, with the functional through its first derivatives; the energy-weighted density `D F D` multiplies the overlap derivative. Energy/gradient four-step fits and physical limits measured; remaining combinations tracked in CSV | `driver::run("bfield-gradient", ...)`; `scf::magnetic::magnetic_field_gradient` | `gradient` in Hartree/Bohr  `bfield_initial_guess` defaults to auto; failed zero-field convergence may use a direct London-metric core seed. Requested stability following is never bypassed. `energy` is null without a converged zero-field reference; `field_energy` and `converged` describe the actual field state. `series_tail` measures omitted coefficients. Optional `field` adds the same electric scalar potential to the London SCF and analytic gradient. For charge Q, `gradient_translation_target=-Q*electric_field` (Hartree/Bohr); `translation_residual` measures the difference from that physical target. |
| `magnetoelectric` | Linear magnetoelectric response of an open-shell molecule: the electric dipole a magnetic field induces to first order in the one-electron spin-orbit coupling `(alpha^2/2) sum_A Z_A l_A.s/r_A^3` (`soc_charges`), `d mu_a/dB_b` for the spins along each axis `k`. It is minus the mixed third derivative of the energy in an electric field component, a magnetic field component and the spin-orbit strength; every combined direction is one field path through third order with complex Hermitian densities, and the polarization identity gives the mixed derivative. London orbitals by default (`gauge` = london or common-origin). Needs an unrestricted open-shell reference. Implemented, not yet validated by acceptance tests | `driver::run("magnetoelectric", ...)`; `response::field_path::spin_orbit_magnetoelectric` | `magnetoelectric_tensor[a][b][k]` in atomic units per unit spin-orbit strength |
| `periodic-energy` | Finite-k periodic Kohn-Sham/Hartree-Fock energy per cell (Mermin free energy at finite temperature), optional analytic forces; exact exchange uses the minimum images of the Born-von Karman supercell of the k mesh (one block per pair of centres, see periodic.md section 31) | `driver::run("periodic-energy", &System { cell: Some(rows_bohr), pbc: [true; 3], .. }, &controls)`; `driver::periodic::{PeriodicMethod, prepare_periodic}`; controls `xc`, `basis`, `ecp`, `charge`, `multiplicity`, `unrestricted`, `k_mesh`, `overlap_cutoff`, `exchange_cutoff`, `grid_level`, `grid_shell`, `hartree_compact_exponent` (speed only: split exponent of the range-separated Hartree build, default chosen from the cell volume), SCF controls, `electronic_temperature` (K), `forces`, `stress`, `dispersion` (alias `disp`: `none`, `d3`, `d3bj`, `chg`; lattice sum `periodic::dispersion::periodic_d3`; `wb97xd` carries `chg` by definition) with `disp_cutoff` (95 Bohr) and `cn_cutoff` (40 Bohr), `three_body` (default false; Axilrod-Teller-Muto term of a D3 model) with `abc_cutoff` (40 Bohr) | `energy`, `free_energy`, `energy_components`, `k_points` (1/Bohr) and `k_points_reduced`, `volume`, optional `forces`/`gradient` (Hartree/Bohr) with `translation_residual`, optional stress fields as for `stress`; with `dispersion`: `dispersion_energy` (included in `energy`, `free_energy` and the forces), `dispersion_model`, `disp_cutoff`, `cn_cutoff` 3D `dispersion="d2"` uses published functional and H-Kr atom parameters, a fixed pair cutoff in Bohr, and no ATM term. The dispersion energy is Hartree per cell, its forces Hartree/Bohr, and its stress Hartree/Bohr^3. [D2 model and verification](response.md#periodic-d2-empirical-dispersion). Correlated lattice path: typed named/custom PT2 parent, full periodic Ewald integrals, every required orbital/pair order through four, integer 3D Gamma supercell model; physical acceptance tracked in response.md. Native Python, ASE and CLI use this same driver.  VV10: `vv10`, `vv10_b`, `vv10_c`, `vv10_cutoff` (Bohr); named -V parents attach their defining term. Energy components separate semilocal/nonlocal correlation. SCF and moving Gamma kernel connected; physical/public acceptance and finite-q/frequency combinations remain pending. Explicit reference=roks/rohf source now connects an integer Gamma common-spatial-orbital reference, with closed beta subspace inside alpha, fresh physical Focks and raw orbital/metric/number diagnostics. HF/PBE source-local open-shell references and all six orbital-rotation four-step fits pass; closed-shell energies match existing periodic RKS within6.7e-16 Hartree. Core initialization/centered CDIIS, three-dimensional Gamma and zero temperature only; ROHF requires an HF provider. Finite k, thermal, shifted/damped initialization, additional nonlocal/PT2 terms and response above first order remain refused. Full-rank Gamma source now also connects analytic nuclear gradients/forces and affine strain/stress. The actual bacf85/2b60bd successor passes canonical Rust HF/PBE references, all six independent four-step orbital-rotation fits (p1.99919..2.12830, minimum R2 .999114), 11 unique public tests and two alias-CLI SCFs. Native/ASE/CLI raw common C, spin densities, physical Focks, units and residuals are retained; independent reconstruction gives common residual at most2.64e-15 Hartree. Those bacf85/2b60bd checks certify reference energy/interfaces. The later full-rank HF/PBE first-derivative subset passes all36 Cartesian/affine-strain four-step fits on the unchanged1e-13 energy,1e-11 density and1e-10 orbital gates. PBE needs a converged localized XC AO radius: xc_ao_cutoff=16 Bohr agrees with20 on this fixture; the original6-Bohr cutoff-crossing failures remain retained. Native gradient/forces use Hartree/Bohr and stress Hartree/Bohr^3; ASE converts lengths and derived units. The b895d970/80b95343 shared core passes15 unique public cases across preserved attempts and all8 native/ASE/both-CLI raw reports (288 force/gradient/stress/strain entries); maximum route difference3.97e-15 and difference from accepted analytic references2.53e-15. One false watchdog stop and one corrected metadata-stub failure remain retained. No new installed-wheel acceptance is claimed; finite-k, geometry relaxation, higher response, general GKS derivatives and material/basis/grid convergence are not inferred. Use damping=0, no_adiis=true for this core/CDIIS route. ASE construction additionally sets use_adiis=False; disabled ADIIS omits only its automatic cutoff, while explicit cutoff controls remain visible to backend validation. Conventional RKS/UKS paths retain their own existing scope. Field-free generalized increment: `reference=gks` or `ghf` on `periodic-energy` uses the complex spinor reference with optional `forces=true` and `stress=true`, without a collinear surrogate. The source-connected 3D integer all-electron/full-AO-rank stress is the complete same-gauge Coulomb/kinetic/complex-exchange/moving-XC/Pulay affine derivative divided by volume. HF/GHF and HF/SVWN5 GKS have measured all-nine-component acceptance; PBE/meta-GGA/hybrid stress and general higher response remain separate gates. `strain_derivative` and its component tensors use Hartree, `stress`/`stress_voigt` use Hartree/Bohr^3 before ASE conversion, and `pressure=-trace(stress)/3`. The ordered raw tensor and `stress_antisymmetric_residual` are retained; the Voigt view selects xx,yy,zz,yz,xz,xy without silently projecting away spatial-grid torque. Additional spin/SOC/field, finite-temperature and discarded-rank derivative sectors remain refused. See R067 in response.md for exact artifact scope, finite-grid limits and retained failures. [Weighted XC controls and diagnostics](rust-api.md#weighted-periodic-xc). Field-free all-electron 1D/2D RKS and UKS use the dimensional finite-k reference for optional forces and stress. Field-free finite-k ROKS uses fixed integer ranks of common complex spatial orbitals on each mesh point, with separate real/imaginary matrices and raw metric, conjugation and orbital residuals. The dimensional RKS/UKS VV10 energy route supplies the reconverged oracle for all 36 force and strain fits. [Verification](response.md#first-geometry-derivatives-of-variational-periodic-vv10). Field-free fixed-integer ROKS VV10 reports `nonlocal_correlation_energy` in Hartree per cell, included in `xc_energy`. [ROKS scope](response.md#first-geometry-derivatives-of-variational-periodic-vv10). The field-free 1D/2D RKS and UKS and 1D/2D/3D ROKS and GKS routes include geometry-only dispersion in the total energy and requested forces. [Periodic dispersion scope and acceptance](response.md#geometry-only-dispersion-with-periodic-common-orbital-references) [Explicit periodic D1 inputs and derivative scope](response.md#periodic-d1-and-analytic-dispersion-derivatives). [Spinor VV10 energy and first geometry derivatives](response.md#first-geometry-derivatives-of-variational-periodic-vv10). |
| `stress` | Analytic stress `sigma = (1/V) dE/d(epsilon)` of the finite-k energy (Mermin free energy at finite temperature), atoms and lattice strained together at fixed fractional coordinates and fractional k points; field-free scalar semilocal ECP and all-electron cells | `driver::run("stress", ...)`; `periodic::stress::analytic_stress(&cell, &functional, &scf, &options) -> PeriodicStress`; periodic reference controls including `dispersion`, `disp_cutoff`, `cn_cutoff` | `stress` and `stress_gpa` (3x3), `stress_voigt` (xx, yy, zz, yz, xz, xy; Hartree/Bohr^3), `pressure` = -tr/3 and `pressure_gpa`, raw `strain_derivative` (Hartree per cell) with `strain_derivative_terms` (kinetic, overlap (Pulay), nuclear attraction, Hartree, exact exchange, exchange-correlation, nuclear repulsion, and `dispersion` with a D3 model), raw `rotation_residual` = max abs(dE/de_ab - dE/de_ba), `forces`, `energy` The dedicated GKS/GHF route shares the field-free 3D Gamma spinor derivative, units and diagnostics of `periodic-energy` with `stress=true`. See [analytic stress](periodic.md#11-analytic-stress-and-cell-relaxation). [Weighted XC controls and diagnostics](rust-api.md#weighted-periodic-xc). Field-free RKS, UKS, fixed-integer ROKS and GKS also report 1D and 2D ordered affine strain with length or area normalization, active-subspace pressure and raw Coulomb/SCF residuals. The field-free scalar ECP force and affine stress include separate translated-centre terms in Hartree/Bohr and Hartree. Collinear Fourier-field force and stress retain fixed reduced wave indices and Cartesian amplitudes, including current and A squared terms. Raw field-energy reconstruction defects use Hartree. The affine term uses Hartree per cell and its stress is divided by the active periodic measure. Field-free 3D RKS/UKS double hybrids use the relaxed Born-von Karman orbital and pair response at fixed reduced mesh, including the uniform Ewald background and monopole strain for charged cells. Ordered target and PT2 derivatives and raw solver residuals are reported. [Validation scope](response.md#r026-stress-elastic-high-elastic). Scalar ECP uses retained projector images on the first-order correlated lattice path. [ECP scope and numerical evidence](response.md#scalar-ecp-on-the-first-order-correlated-lattice-path). Neutral 3D collinear double hybrids accept a fixed Cartesian Berry bias in Hartree/e/Bohr. [Field convention and validation](response.md#correlated-stress-at-fixed-cartesian-electric-bias). Field-free RKS/UKS variational VV10 uses `vv10_cutoff` in Bohr. Conventional 3D finite-k force and stress have a verified NaH subset. The 1D RKS omegaB97X-V wire and 2D UKS omegaB97M-V slab use length or area normalization. Both pass all 18 force and ordered strain fits and the CLI, native Python and ASE comparisons. [Derivative scope and residuals](response.md#first-geometry-derivatives-of-variational-periodic-vv10). Field-free fixed-integer ROKS VV10 reports `nonlocal_correlation_energy` in Hartree per cell, included in `xc_energy`. [ROKS scope](response.md#first-geometry-derivatives-of-variational-periodic-vv10). The field-free 1D/2D RKS and UKS and 1D/2D/3D ROKS and GKS routes include geometry-only dispersion in the force and strain derivatives. [Periodic dispersion scope and acceptance](response.md#geometry-only-dispersion-with-periodic-common-orbital-references) [Explicit periodic D1 inputs and derivative scope](response.md#periodic-d1-and-analytic-dispersion-derivatives). [Spinor VV10 energy and first geometry derivatives](response.md#first-geometry-derivatives-of-variational-periodic-vv10). |
| `opt-fixed-cell` | Fixed-cell relaxation of the atoms (L-BFGS with Armijo backtracking) on analytic finite-k forces | `driver::run("opt-fixed-cell", ...)`; `periodic::optimizer::optimize_fixed_cell`; controls `max_steps`, `gtol` (Hartree/Bohr), `step_max` (Bohr), `history`; periodic reference controls include `dispersion`, `disp_cutoff`, `cn_cutoff` (the D3 energy, forces and stress enter the objective) | Relaxed `positions` (Bohr), `objective`, final `forces`, `optimization_converged`, `iterations`, `trajectory` (`objective`, `grad_max`) and the reference report of the final SCF The Gamma 3D ROKS objective includes geometry-only dispersion and its gradient. [Periodic dispersion scope and acceptance](response.md#geometry-only-dispersion-with-periodic-common-orbital-references) [Explicit periodic D1 inputs and derivative scope](response.md#periodic-d1-and-analytic-dispersion-derivatives). |
| `cell-opt` | Variable-cell relaxation of atoms and lattice minimizing the enthalpy `E + pV` with analytic forces and stress; symmetric deformation of the reference cell (no rigid rotation) | `driver::run("cell-opt", ...)`; `periodic::optimizer::{optimize_cell, CellOptOptions}`; controls `max_steps`, `gtol` (Hartree/Bohr), `stress_tol` (Hartree/Bohr^3, max abs(sigma + pI)), `pressure_gpa`, `step_max` (Bohr), `history`; periodic reference controls include `dispersion`, `disp_cutoff`, `cn_cutoff` (the D3 energy, forces and stress enter the objective) | Relaxed `cell` rows and `positions` (Bohr), `enthalpy`, `external_pressure`, final stress fields and `forces`, `optimization_converged`, `iterations`, `trajectory` (`enthalpy`, `grad_max`, `stress_max`, `volume`) |
| `phonons` | Gamma-point harmonic phonons: every lattice image of an atom moves with it; analytic force constants from the skeleton second derivatives (kinetic, overlap, Ewald nuclear attraction, screened Hartree, XC with periodic Becke weights, Ewald nuclei) and the finite-k coupled-perturbed equations with the exact screened Hartree, semilocal XC and compatible full/long-range exchange kernels; LDA/GGA references and, for insulators, exact exchange (Hartree-Fock, global and range-separated hybrids: second derivatives of the minimum-image exchange sum, exchange kernel, and a separate pairing matrix for the truncated sum); exact exchange with Fermi-Dirac occupations is refused | `driver::run("phonons", ...)`; `periodic::phonon::gamma_force_constants(&cell, &functional, &scf, &options, &solver)`; periodic reference and `response_*` controls | `force_constants` (Hartree/Bohr^2, symmetrized), `raw_force_constants` (before symmetrization: column y is the derivative of the analytic gradient with respect to displacement y), `frequencies_cm1` (negative = imaginary), `normal_modes_cartesian`, raw `raw_asymmetry` and `acoustic_residual` before any correction, `response_solves` (iterations, relative residual), `q` = 0 Correlated lattice path: typed named/custom PT2 parent, full periodic Ewald integrals, every required orbital/pair order through four, integer 3D Gamma supercell model; physical acceptance tracked in response.md. Native Python, ASE and CLI use this same driver. Optional `field` selects analytic nuclear derivatives of the stationary finite-bias Berry electric enthalpy; fixed Cartesian field, insulating Gamma supercell of the k mesh. Reference residual/energy and available response residuals are reported; independent acceptance is tracked in response.md. For magnetic/joint-field nuclear derivatives use `periodic-magnetic-scf`; its accepted finite-Gamma subset and remaining combinations are documented separately. Scalar ECP second-order retained-image coefficients are connected to the correlated path; independent operator, Hessian and installed acceptance are separate. [ECP second-order scope](response.md#scalar-ecp-second-order-lattice-path). Conventional scalar ECP CPKS Hessians are connected through the common driver; raw force constants, acoustic and symmetry residuals use Hartree/Bohr^2. [Separate acceptance](response.md#scalar-ecp-in-the-conventional-periodic-cpks-hessian). The collinear Gamma XC Hessian includes the meta-GGA kinetic-energy-density channel with finite-k electronic occupations. Independent TPSS force and XC operator fits and installed ECP parity are pending. RKS omegaB97X-V scalar-ECP NaH passes all 21 analytic Hessian directions at electronic Gamma. [VV10 evidence](response.md#variational-vv10-gamma-hessian). The three-dimensional Berry-bias route includes the lattice D2, D3/ATM or CHG correction. [Finite-field scope and acceptance](response.md#finite-electric-bias-with-empirical-dispersion). The conventional collinear route reports the accepted SCF `orbital_residual` in Hartree alongside the coupled response residuals. [Weighted-recipe example](response.md#r026-stress-elastic-high-elastic). |
| `phonon-bands` | Phonon dispersion from the analytic Gamma-point force constants of an `N1 x N2 x N3` supercell sampled with the k mesh divided by the supercell (the same Brillouin-zone sampling): every primitive pair is assigned to its shortest supercell images with equal weights on ties, exact at the commensurate q and interpolated elsewhere; no acoustic sum rule is imposed; neutral closed-shell cells | `driver::run("phonon-bands", ...)` with `supercell` (required), `path` (reduced q vertices), `points` (20), `labels`, optional `dos_mesh` with `dos_sigma` (5 cm^-1) and `dos_points` (400); `periodic::phonon_bands::{build_supercell, ForceConstantModel}`; periodic reference and `response_*` controls; `k_mesh` must be divisible by `supercell` | `q_path` (1/Bohr), `q_path_reduced`, `path_distance`, `frequencies_cm1[q][mode]` (negative = imaginary), optional `dos_frequencies_cm1` and `dos` (states per cm^-1 per primitive cell, 3N modes), `supercell`, `supercell_k_mesh`, `supercell_force_constants` (Hartree/Bohr^2, supercell atoms ordered primitive atom fastest, then cells along a1, a2, a3), raw `acoustic_residual`, `force_constant_asymmetry`, `dynamical_matrix_hermiticity`, `response_solves`  Optional `wavevector_direction` (Cartesian 1/Bohr) with `response_order` 0..4 returns mass-weighted and unweighted Fourier coefficients along q(t)=q+t*d. `phonon_phase=cell` uses R and `atomic` uses R+tau_final-tau_initial. Raw Hermiticity is measured without projection; supercell interpolation and missing long-range/electronic flexoelectric terms remain explicit. `lattice_flexoelectric=true` with `wavevector_direction` and order 2..4 adds fixed-mass-center internal relaxation at Gamma. Born charges are computed on the primitive reference or supplied in `born_charge_rows` (3N by 3). `born_lattice_flexoelectric` is the Born-charge lattice part, e/Bohr, contracted twice with the normalized direction; electronic/dynamic-multipole terms remain separate. Collinear periodic double hybrids use the full correlated Gamma Hessian of the force-constant supercell. The fixed-cell 3D all-electron charged route uses uniform Ewald compensation and reports primitive, force-constant and Born-von Karman charges in e. The correlated report retains orbital and PT2 residuals and force constants in Hartree/Bohr^2. Scalar ECP second-order retained-image coefficients are connected to the correlated path; independent operator, Hessian and installed acceptance are separate. [ECP second-order scope](response.md#scalar-ecp-second-order-lattice-path). Conventional scalar ECP CPKS Hessians are connected through the common driver; raw force constants, acoustic and symmetry residuals use Hartree/Bohr^2. [Separate acceptance](response.md#scalar-ecp-in-the-conventional-periodic-cpks-hessian). |
| `phonon-thermo` | Harmonic lattice thermodynamics per primitive cell (zero-point energy, vibrational internal energy, Helmholtz free energy, entropy, constant-volume heat capacity) from the same supercell force constants on a Gamma-centred uniform q mesh; modes below `mode_cutoff_cm1` are excluded and counted | `driver::run("phonon-thermo", ...)` with `supercell`, `q_mesh` (required), `temperatures` (K; default 0, 100, 200, 298.15, 400, 600), `mode_cutoff_cm1` (1); `periodic::phonon_bands::harmonic_thermo` | `zero_point_energy`, `internal_energy`, `free_energy`, `entropy`, `heat_capacity` (Hartree, Hartree/K per primitive cell) with kJ/mol and J/(mol K) siblings, `mesh_frequencies_cm1`, `excluded_modes`, `imaginary_modes`, diagnostics as for `phonon-bands` Collinear periodic double hybrids use the full correlated Gamma Hessian of the force-constant supercell. The fixed-cell 3D all-electron charged route uses uniform Ewald compensation and reports primitive, force-constant and Born-von Karman charges in e. The correlated report retains orbital and PT2 residuals and force constants in Hartree/Bohr^2. Scalar ECP second-order retained-image coefficients are connected to the correlated path; independent operator, Hessian and installed acceptance are separate. [ECP second-order scope](response.md#scalar-ecp-second-order-lattice-path). Conventional scalar ECP CPKS Hessians are connected through the common driver; raw force constants, acoustic and symmetry residuals use Hartree/Bohr^2. [Separate acceptance](response.md#scalar-ecp-in-the-conventional-periodic-cpks-hessian). |
| `electron-phonon` | Electron-phonon coupling from the Gamma displacement responses of a supercell (every primitive wave vector commensurate with it; insulating or Fermi-Dirac references): vertices `g = [C^+ (dF - (e_m + e_n)/2 dS) C]/sqrt(2 w)` with the self-consistent first-order Fock, mode couplings `lambda_nu`, `lambda`, `omega_log`, `omega_2`, the isotropic Eliashberg function, McMillan and Allen-Dynes (with f1 f2) critical temperatures for a given `mu_star`, the critical temperature of the isotropic imaginary-axis Eliashberg equations (linearized gap equation, semi-empirical `mu_star`), and the nonlinear gap `Delta(T)` of the same equations with its Pade (Vidberg-Serene) continuation to the real axis; the eigenvalue derivatives `d e_n/d x` are reported for checks | `driver::run("electron-phonon", ...)` with optional `supercell` (default 1,1,1; `k_mesh` divisible), `electron_smearing` (Hartree; default kT of `electronic_temperature`), `phonon_smearing_cm1` (10), `mode_cutoff_cm1` (5), `mu_star` (0.1), `a2f_points` (400), `eliashberg_cutoff_cm1` (Matsubara energy cutoff; default 10 times the highest included mode), `eliashberg_t_min_k` (0.01, lowest temperature searched), `gap_temperatures_k` (list of K; nonlinear gaps, none by default), `pade_points` (64, Matsubara points of the continuation); `periodic::electron_phonon::{electron_phonon, eliashberg_function, allen_dynes, eliashberg_tc}`, `periodic::eliashberg::{eliashberg_gap, pade_continuation, real_axis_gap}` | `lambda`, `mode_lambda`, `omega_log_k`, `omega_2_k`, `fermi_dos`, `frequencies_cm1`, `a2f_frequencies_cm1`, `a2f`, `tc_allen_dynes_k`, `tc_mcmillan_k` (null when the exponent denominator is not positive), `tc_eliashberg_k` (isotropic Migdal-Eliashberg on the imaginary axis; null when no solution exists above `eliashberg_t_min_k` or no mode is included), `eliashberg_cutoff_cm1`, `eliashberg_gaps` (per `gap_temperatures_k` entry: `temperature_k`, `converged`, `iterations`, `residual`, `delta_iw0_hartree`, `z_iw0`, `delta_real_hartree`, null in the normal state; empty list when no temperature is given), `pade_points`, `eigenvalue_derivatives`, `acoustic_residual` `scdft_pairing=true` adds `scdft_inputs`, static K (Hartree) and dimensionless Z. Supply `screened_pair_coulomb` or choose `scdft_screening=rpa` (thermal head) or `rpa-neutral` (fixed-number Gamma body), with `response_reciprocal_cutoff` in inverse Bohr. Generated Coulomb screening covers the complete 3D k-point difference mesh of a spin-degenerate finite-temperature reference. The phonon contribution retains the commensurate Gamma-supercell scope. [Models and diagnostics](periodic.md#static-microscopic-scdft-coulomb-screening). |
| `quasi-harmonic` | Quasi-harmonic lattice thermodynamics as a fitted volume-grid model: on isotropically scaled cells, the static energy per primitive cell and the harmonic thermodynamics of `phonon-thermo`; at every temperature a third-order Birch-Murnaghan fit of `E(V) + F_vib(V, T)` gives `V(T)` and `B_T`, least-squares quadratics of the vibrational entropy and heat capacity give `alpha = (dS/dV)_T / B_T`, `C_p = C_v + V T alpha^2 B_T` and `gamma = alpha B_T V / C_v`; the grid must bracket every `V(T)` | `driver::run("quasi-harmonic", ...)` with `supercell`, `q_mesh` (required), `scales` (default 0.98..1.02 in steps of 0.01, at least four), `temperatures`, `mode_cutoff_cm1`, `pressure_gpa` (default 0: the state is the point of the fitted isotherm with `-dF/dV = p`, which the volume grid must bracket); `periodic::quasi_harmonic::{quasi_harmonic, quasi_harmonic_at_pressure, eigenvalue_volume_derivatives}` | Per temperature `volume` (Bohr^3), `linear_scale`, `free_energy`, `bulk_modulus_gpa`, `thermal_expansion` (1/K, volumetric), `entropy_j_mol_k`, `heat_capacity_v_j_mol_k`, `heat_capacity_p_j_mol_k`, `gruneisen`, raw `fit_residual_rms`; `static_v0`, `static_b0_gpa`, `static_b0_prime`, `volumes`, `static_energies`, `imaginary_modes_per_volume`; `pressure_gpa`, `gibbs_energy` (`F + p V`), `vibrational_free_energies` `[volume][temperature]`; mode Grueneisen parameters of the fitted volume-grid model (least-squares quadratics of the dynamical matrices in ln V, degenerate first-order perturbation theory): `mode_gruneisen` `[q][mode]` (null below the cutoff), `mode_gruneisen_frequencies_cm1`, `mode_gruneisen_q_reduced`, `mode_gruneisen_volume`, `gruneisen_mode_average` per temperature |
| `elastic` | Second-order elastic constants: clamped-ion strain Hessian from the skeleton strain second derivatives (kinetic, overlap, Ewald nuclear attraction with its background, screened Hartree, XC by exact forward-mode jets with periodic Becke weights, Ewald nuclei) and the strain coupled-perturbed responses; internal-strain coupling from the strain and displacement responses; relaxed ions by `W - Lambda Phi^+ Lambda^T` with the translations projected out; Lagrangian constants `sym[W - delta_ac tau_bd]/V` and stress-strain (Birch) coefficients; collinear semilocal and compatible exact-exchange references with a supported response provider | `driver::run("elastic", ...)`; `periodic::elastic::elastic_constants(&cell, &functional, &scf, &options, &solver) -> ElasticConstants`; periodic reference and `response_*` controls | `clamped_ion`, `relaxed_ion` (Voigt xx, yy, zz, yz, xz, xy; Hartree/Bohr^3) and `_gpa` siblings, `clamped_ion_birch_gpa`, `relaxed_ion_birch_gpa`, `relaxed_ion_eigenvalues_gpa` (Born stability), raw `strain_hessian` `[3a+b][3c+d]` (Hartree per cell, `x -> (1+eps) x`), `internal_strain` `[3a+b][3 atom+c]` (Hartree/Bohr), `internal_relaxation` (Bohr per unit strain), `force_constants`, `stress`, `strain_derivative`, raw `raw_asymmetry`, `internal_strain_asymmetry`, `rotation_residual`, `force_constant_asymmetry`, `acoustic_residual`, `smallest_optical_eigenvalue`, `strain_solves`, `displacement_solves` Uniformly compensated charge includes the analytic Ewald monopole strain Hessian. The 3D collinear scalar ECP path includes affine and internal-strain terms. Charged and scalar ECP numerical acceptance is tracked in the [response ledger](response.md#r026-stress-elastic-high-elastic). Includes the geometry-only CHG and D3 strain, mixed and nuclear curvatures before ionic relaxation. The conventional collinear route reports the accepted SCF `orbital_residual` in Hartree alongside the coupled response residuals. [Weighted-recipe example](response.md#r026-stress-elastic-high-elastic). [Elastic report conventions and strain checks](api-guide.md#choose-elastic-constants-and-strain-diagnostics). |
| `elastic-fd` | Validation route (finite differences are validation only): Lagrangian elastic constants from central differences of the analytic strain derivative over the six symmetric strains `E` (shears with 1/2 per index order), as `sym[FD - S E - E tau]/V` for the composed strain; ions clamped at fixed fractional coordinates or relaxed in every strained cell | `driver::run("elastic-fd", ...)` with `steps` (default 0.01, 0.005), `relax_ions` (false), and for relaxation `gtol` (1e-6 Hartree/Bohr), `max_steps`, `step_max`, `history` | `lagrangian_gpa` per step, `richardson_gpa` (last two steps), `strain_derivative`, `reference_max_force`, `max_residual_force` (relaxed), `all_converged`, `provenance` |
| `elastic-moduli` | Polycrystalline moduli of the analytic elastic tensor: Voigt, Reuss and Hill bulk and shear moduli, Hill Young modulus and Poisson ratio, universal anisotropy index, compliance, Born stability eigenvalues, mass density, isotropic P and S wave speeds of the Hill aggregate and the Debye temperature from the mean sound velocity | `driver::run("elastic-moduli", ...)` with `ions` (`relaxed` default or `clamped`) and `coefficients` (`birch` default or `lagrangian`); `periodic::elastic::elastic_moduli(&c)` for any Voigt tensor | `elastic_tensor_gpa`, `compliance_per_gpa`, `tensor_eigenvalues_gpa`, `born_stable`, `voigt_/reuss_/hill_bulk_gpa`, `voigt_/reuss_/hill_shear_gpa`, `young_gpa`, `poisson_ratio`, `universal_anisotropy`, `density_kg_m3`, `p_wave_velocity_m_s`, `s_wave_velocity_m_s`, `debye_temperature_k`, `stress_gpa`, `selection` |
| `christoffel` | Plane elastic waves of the analytic elastic tensor: `det(C_ijkl n_j n_l - rho v^2 delta_ik) = 0` along given directions | `driver::run("christoffel", ...)` with `directions` (Cartesian rows; default the six axes and face diagonals and [111]), `ions`, `coefficients`; `periodic::elastic::christoffel(&c_gpa, density, n)` | `velocities_m_s[direction]` (ascending; negative = unstable), `polarizations[direction]` (columns), `directions`, `density_kg_m3`, `elastic_tensor_gpa`, `selection` |
| `material-card` | Continuum material card of the analytic elastic tensor (selected by `ions`, `coefficients`): stiffness and compliance (Voigt, engineering shear strains), orthotropic engineering constants `E_i = 1/S_ii`, `nu_ij = -S_ji/S_ii`, `G_23, G_13, G_12`, optionally in rotated material axes (Bond transformation), Hill moduli, density, wave speeds, the reference stress, conventions and residuals; written as JSON or CSV | `driver::run("material-card", ...)` with `rotation` (orthonormal rows = material axes), `card_path`, `card_format` (`json` or `csv`, else from the extension), `ions`, `coefficients`; `geomech::{compliance, engineering_constants, rotate_stiffness}` | `stiffness_gpa`, `compliance_per_gpa`, `young_gpa`, `poisson_ratios`, `shear_moduli_gpa`, `hill_bulk_gpa`, `hill_shear_gpa`, `density_kg_m3`, `vp_m_s`, `vs_m_s`, `reference_stress_gpa`, `tensor_eigenvalues_gpa`, raw residuals, `csv`, `conventions`, optional `rotation`, `card_path` |
| `equation-of-state` | Third-order Birch-Murnaghan equation of state: SCF energies of isotropically scaled cells (fixed fractional coordinates or ions relaxed in every cell) fitted by linear least squares in `(V/V_r)^(-2/3)` (exactly the third-order form); the analytic pressure `-tr(sigma)/3` of every cell is compared with `-dE/dV` of the fit | `driver::run("equation-of-state", ...)` with `scales` (linear factors; default 0.97..1.03 in steps of 0.01, at least four), `relax_ions` (false) and for relaxation `gtol` (1e-5), `max_steps`, `step_max`, `history`; `periodic::eos::fit_birch_murnaghan(&volumes, &energies)` | `v0` (Bohr^3) and `v0_angstrom3`, `e0`, `b0_gpa`, `b0_prime`, `equilibrium_scale`, `volumes`, `energies`, `analytic_pressures_gpa`, `fitted_pressures_gpa`, `fitted_bulk_moduli_gpa`, raw `pressure_fit_gap_gpa` and `residual_rms`, `all_converged` |
| `dielectric-tensor` | Electronic (clamped-ion) dielectric tensor of an insulating reference, including exchange image blocks and screened exchange; static HF water dilute-limit acceptance complements existing semilocal tests by periodic DFPT at fixed macroscopic field: ddk amplitudes from `d_k H` and `d_k S` of the Fock and overlap image blocks, the position operator `R_a = C_v^+ (D_a + i d_a S) C_o + i U^ka` with dipole image blocks, the self-consistent field response with the screened Hartree, semilocal XC and compatible full/long-range exchange kernels, `chi_ab = -(1/(V Nk)) sum 2 Re Tr[U^Eb^+ R_a]`, `eps_inf = 1 + 4 pi chi` | `driver::run("dielectric-tensor", ...)`; `periodic::dielectric::dielectric_tensors(&cell, &functional, &scf, &options, &solver, false)`; periodic reference and `response_*` controls | `epsilon_infinity`, `susceptibility`, raw `raw_asymmetry`, `fock_image_residual` (Fock image blocks against the SCF Fock), `field_solves`; HF water continuum DFPT dilute-limit fit: p=2.970392240 in 1/L, R^2=0.999997818686, maximum raw field residual 3.722e-14; no HF Born-charge claim Correlated lattice path: typed named/custom PT2 parent, full periodic Ewald integrals, every required orbital/pair order through four, integer 3D Gamma supercell model; physical acceptance tracked in response.md. Native Python, ASE and CLI use this same driver.; conventional 3D insulating RKS/UKS includes scalar ECP. [Acceptance](response.md#r023-born-charge-berry-polarization) Includes the CHG or D3 reference energy for these geometry-only terms. |
| `born-charges` | Born effective charges `Z*_(k,ab) = V dP_a/du_(k,b)` from the Berry-connection derivative, self-consistent displacement response, moving-basis overlap/dipole derivatives and ddk amplitudes; zone sums converge with the k mesh | `driver::run("born-charges", ...)`; `periodic::dielectric::dielectric_tensors_with_force_constants(..., true, false)`; compatibility `dielectric_tensors(..., true)` also builds the polar-mode force constants | `born_charges[atom][a][b]` (e), raw `born_sum_rule`, `imaginary_residual`, dielectric fields and `displacement_solves`; Born-only requests reuse identical first-order sources without the unrelated Hessian construction Correlated lattice path: typed named/custom PT2 parent, full periodic Ewald integrals, every required orbital/pair order through four, integer 3D Gamma supercell model; physical acceptance tracked in response.md. Native Python, ASE and CLI use this same driver.; conventional 3D insulating RKS/UKS includes scalar ECP. [Acceptance](response.md#r023-born-charge-berry-polarization) Includes the CHG or D3 reference energy for these geometry-only terms. |
| `piezoelectric-tensor` | Proper piezoelectric tensor `e_d,(ab) = dP_d/d(eps_ab)` of an insulating reference with the polarization in the coordinates of the unstrained cell: clamped-ion from the strain response of the Berry connection, relaxed-ion with Born charges and the internal relaxation (control `relaxed_ion`, default true; `false` gives the clamped-ion tensor of references without a stable ionic relaxation and omits the relaxed-ion entries). Implemented, not yet validated by acceptance tests | `driver::run("piezoelectric-tensor", ...)`; `periodic::dielectric::piezoelectric_tensor(&cell, &functional, &scf, &options, &solver, relax_ions)` | Clamped- and relaxed-ion tensors [polarization axis][Voigt strain] in e/Bohr^2 and C/m^2, unsymmetrized [axis][3a+b] tensors, Born charges; conventional 3D insulating RKS/UKS includes scalar ECP. [Acceptance](response.md#r023-born-charge-berry-polarization) The relaxed-ion route includes CHG or D3 in the internal relaxation. |
| `berry-polarization` | Berry-phase polarization of an insulating reference: string phases `phi_i = -Im ln prod det[C_o(k_j)^+ M(k_j, k_(j+1)) C_o(k_(j+1))]` of the occupied manifold along each reciprocal axis with plane-wave cross-k AO overlaps, averaged over the perpendicular mesh, plus the ionic term; `p_i = -g sum_spin phi_i/(2 pi) + sum Z (r . b_i)/(2 pi)`, `P = (1/V) sum p_i a_i`; defined modulo the polarization quanta (only changes along an insulating path are physical); Fermi-Dirac references are refused | `driver::run("berry-polarization", ...)` with optional `berry_mesh` (default: the SCF `k_mesh`); `periodic::berry_phase::berry_polarization` | `berry_phases` (radians, `[spin][axis]`), `electronic_reduced`, `ionic_reduced`, `total_reduced` (modulo one), `polarization` (e/Bohr^2) and `polarization_c_m2`, `polarization_quanta`, raw `string_spread`, `min_link_log_abs_determinant`, `min_gap`, `fock_image_residual` |
| `crystal-ir` | Polar Gamma-point lattice dynamics from one reference: transverse modes of the analytic force constants, mode effective charges `Z~_m,a = sum Z*_(k,ab) e_m(k,b)/sqrt(m_k)`, infrared intensities `974.8801 \|Z~_m\|^2`, and LO frequencies along given directions from the non-analytic term `(4 pi/V)(q.Z*_k)(q.Z*_k')/(q.eps_inf.q)/sqrt(m_k m_k')` | `driver::run("crystal-ir", ...)` with `directions` (Cartesian rows; default x, y, z) and `mode_cutoff_cm1` (1); `periodic::dielectric::polar_modes(&tensors, &masses, volume, &directions, cutoff)` | `frequencies_cm1`, `mode_effective_charges` (e/sqrt(amu)), `ir_intensities_km_mol`, `lo_directions`, `lo_frequencies_cm1`, `epsilon_infinity`, `epsilon_static`, `born_charges`, raw `born_sum_rule` and `acoustic_residual`, `excluded_modes`; conventional 3D insulating RKS/UKS includes scalar ECP. [Acceptance](response.md#r023-born-charge-berry-polarization) The Gamma Hessian includes CHG or D3. The report gives raw Hessian and Fock-image defects and the field/displacement solve residuals. |
| `static-dielectric` | Static (relaxed-ion) dielectric tensor `eps_0 = eps_inf + (4 pi/V) sum_m Z~_m Z~_m^T / w_m^2` over the modes above `mode_cutoff_cm1` | `driver::run("static-dielectric", ...)`; same route as `crystal-ir` | `epsilon_static` with the `crystal-ir` fields; conventional 3D insulating RKS/UKS includes scalar ECP. [Acceptance](response.md#r023-born-charge-berry-polarization) Uses the same dispersion-corrected Gamma Hessian and response diagnostics as crystal IR. |
| `electronic-structure` | SCF bands, populations and optional complex Bloch orbitals of the 3D finite-k reference | `driver::run("electronic-structure", ...)`, with `coefficients=true` | Per channel `band_energies[k][band]` (Hartree), per-spin `occupations`, `chemical_potential`, `band_edges`, `mulliken`, `mulliken_spin`, `electron_count_residual`; the optional `orbitals` object exports all complex C/S/F/P matrices and raw metric/eigenpair/energy/density defects, with `spin_weight`. [Scope and conventions](response.md#complex-bloch-orbital-coefficient-export). |
| `effective-mass` | Band velocity `d e_n/dk` and inverse effective-mass tensor `d2 e_n/dk_a dk_b` of one band at one k point from analytic first and second k derivatives of the converged Kohn-Sham operator (non-orthogonal perturbation theory over every band of the basis, including the overlap terms); a degenerate cluster (within `degeneracy_tolerance`) is resolved by degenerate perturbation theory along a direction (velocities: eigenvalues of the block of `d_u H - e d_u S`; second derivatives: the second-order block matrix in each eigenspace). Semilocal, full/range-separated hybrid, Hartree-Fock and ECP insulating or Fermi-Dirac references; identical SCF coefficients and image cutoffs | `driver::run("effective-mass", ...)` with `k_point` (reduced), `band` (zero-based), optional `direction` (Cartesian), `spin` (0), `degeneracy_tolerance` (1e-6 Hartree); `periodic::band_transport::BandModel::{new, bands, derivatives}` | `energy`, `velocity` (Hartree Bohr) and `velocity_m_s`, `inverse_mass_tensor` (1/m_e), `inverse_mass_eigenvalues`, `effective_masses` (m_e), `principal_axes`, `cluster`, `degenerate`, with `direction`: `branch_velocities`, `branch_second_derivatives`, `branch_effective_masses`; `reference_level`, raw `fock_image_residual` |
| `boltzmann-transport` | Rigid-band thermoelectric transport with constant or positive energy-dependent scalar relaxation time; degenerate-cluster velocity products retained | `driver::run("boltzmann-transport", ...)`; `periodic::band_transport::{transport_integrals, transport_coefficients}`; `periodic::scattering::{RelaxationTimeTable, transport_integrals, transport_coefficients, frequency_transport, thermoelectric_frequency_transport}`; `scattering_table` rows [absolute band energy in Hartree, lifetime in fs] | Legacy per-tau fields preserved without a table; table mode returns `conductivity` (S/m), `seebeck` (V/K), `electronic_thermal_conductivity` (W/(m K)), unweighted particle count and optional complex frequency conductivity, Seebeck, Peltier and open-circuit heat conductivity; [model and validation](periodic.md#34-energy-dependent-relaxation-time-transport) |
| `hall-transport` | Ordinary weak-field DC and retarded intraband Hall transport of isolated rigid bands, with constant or energy-dependent scalar lifetime; analytic velocities and curvatures | `driver::run("hall-transport", ...)`; `periodic::band_transport::{hall_integrals, hall_coefficient, frequency_transport}`; `periodic::scattering::{hall_integrals, hall_coefficient, frequency_transport, thermoelectric_frequency_transport}`; controls as for `boltzmann-transport`, `hall_onsager_tolerance=1e-10` | Legacy `hall_conductivity_derivative_over_tau_squared` preserved without a table; table mode returns `hall_conductivity_derivative` (S/(m T)), `hall_coefficient` (m^3/C), raw/relative Onsager defects and warnings without projection; dynamic last axis [real,imaginary]; [model and validation](periodic.md#34-energy-dependent-relaxation-time-transport) |
| `coherent-transport` | Covariant independent-particle finite-frequency intra/interband conductivity and intrinsic Hall, including exactly degenerate multiplets; physical AO position connection; semilocal, full/range-separated hybrid and ECP 3D references; frozen operator, no induced Hartree/XC vertex | `driver::run("coherent-transport", ...)`; `periodic::coherent_transport::{CoherentModel, conductivity, occupation_divided_difference}`; `transport_k_mesh`, `transport_frequencies_hartree`, `transport_temperature_k` (300), `transport_mu_shift` (0 Hartree), `scattering_hartree` (0.001 Hartree) | `results.{conductivity,intraband_conductivity,interband_conductivity,antisymmetric_conductivity}` (S/m, [3,3,2], last axis real/imaginary); exp(-i omega t); raw current Hermitian, pair-equation and Fock-image residuals |
| `quantum-hall` | Non-Abelian constant-rank occupied-slice Chern invariant and integer quantum Hall conductance; exact physical AO cross-k metric with reciprocal-boundary sewing; degeneracies inside the occupied multiplet allowed; zero-temperature positive-gap slice of a 3D reference | `driver::run("quantum-hall", ...)`; `periodic::coherent_transport::{CrossMetric, quantum_hall_slice, quantum_hall_from_links}`; `topology_mesh` ([8,8]), `topology_plane_axes` ([0,1]), `topology_fixed_k` (0) | Raw `chern_number`, `slice_hall_conductance` (S, C e^2/h including spin weight), integer residual, sampled minimum gap, maximum plaquette phase, minimum log determinant; no integer rounding; flux/SOC ground states and bulk dimensional response remain tracked work |
| `wannier-flow` | Hybrid Wannier charge centres of the occupied multiplet of an insulating reference: for every transverse wave vector along the flow axis, minus the eigenphases over `2 pi` of the product of the occupied-subspace links `<u_k|u_(k+b)>` (exact cross-k AO overlaps) around the closed loop along the loop axis (the centres sum to minus the electronic reduced Berry-phase polarization over the spin weight); the sum of the centres is continued along the flow and its winding over the closed flow, without integer rounding, is the Chern number of the slice. The phases are those of the unitary part of the loop product. Trivial-chain acceptance in `p229_piezoelectric_and_wannier_flow` (winding 0, centres against the Berry-phase polarization to 1.1e-16); nonzero windings are not yet validated | `driver::run("wannier-flow", ...)`; `periodic::coherent_transport::wannier_flow` | Centres in units of the lattice vector of the loop axis, `winding` dimensionless, gaps in Hartree |
| `periodic-cubic-fc-along` | Cubic force constants of a periodic cell contracted with a Gamma-point displacement pattern on the first index, `Phi(u)_jk = sum_i u_i d3E/(dx_i dx_j dx_k)` per primitive cell, every atom moving with all its lattice images: the derivative of the Gamma-point force constants along `probe`. The k mesh is sampled at the Gamma point of its supercell, where the Bloch sums are real; the integrals of that cell (lattice sums of overlap and kinetic integrals, Ewald sums of the Coulomb interactions with the compact/diffuse partition of the cell, minimum-image exact exchange, quadrature weights and Bloch sums of the orbitals on the grid) are expanded as exact power series along the displacement at the partition of the reference geometry, the reference is converged on the functional of those series (`series_reference_energy_difference` is its distance from the self-consistent field), and first-order responses of the combined directions give the third derivative by the 2n+1 rule and the polarization identity. Insulating all-electron neutral closed-shell cells; LDA, GGA, meta-GGA and hybrid functionals; no learned functional or nonlocal correlation. The series store the Coulomb sums of the supercell as a dense tensor. Implemented, not yet validated by acceptance tests | `driver::run("periodic-cubic-fc-along", ...)`; `response::nuclear::lattice_force_constants_along` | `contracted_force_constants` in Hartree/Bohr^3 per unit pattern and per primitive cell; `probe` dimensionless; energies in Hartree  Accepts `dispersion=d2/d3bj/d3/chg` with a pair cutoff in Bohr. CN and optional ATM settings belong to D3. [D2 model and verification](response.md#periodic-d2-empirical-dispersion); adds analytic dispersion third/fourth derivatives through the existing generic lattice energy. `dispersion_energy_per_primitive_cell` is Hartree; dispersion force-constant arrays and raw `dispersion_acoustic_residual`/tensor `dispersion_permutation_residual` use Hartree/Bohr^order per primitive cell. See [four-step fits](response.md#correlated-electric-dipoles-and-periodic-dispersion-force-constants); remaining electronic combinations retain their stated scope. Optional `field` selects analytic nuclear derivatives of the stationary finite-bias Berry electric enthalpy; fixed Cartesian field, insulating Gamma supercell of the k mesh. Reference residual/energy and available response residuals are reported; independent acceptance is tracked in response.md. For magnetic/joint-field nuclear derivatives use `periodic-magnetic-scf`; its accepted finite-Gamma subset and remaining combinations are documented separately. Every distinct path also returns its actually solved relative `response_residuals`; these are retained before any tensor permutation reuse. The driver includes the implicit CHG term for omegaB97X-D. Combined all-direction validation is pending. |
| `periodic-quartic-fc-along` | Quartic force constants of a periodic cell contracted with two Gamma-point displacement patterns, `Phi(u, v)_kl = sum_ij u_i v_j d4E/(dx_i dx_j dx_k dx_l)` per primitive cell, by the same series with responses through second order (`second_probe` defaults to `probe`). Implemented, not yet validated by acceptance tests | `driver::run("periodic-quartic-fc-along", ...)`; `response::nuclear::lattice_force_constants_along` | `contracted_force_constants` in Hartree/Bohr^4 per unit of both patterns and per primitive cell  Accepts `dispersion=d2/d3bj/d3/chg` with a pair cutoff in Bohr. CN and optional ATM settings belong to D3. [D2 model and verification](response.md#periodic-d2-empirical-dispersion); adds analytic dispersion third/fourth derivatives through the existing generic lattice energy. `dispersion_energy_per_primitive_cell` is Hartree; dispersion force-constant arrays and raw `dispersion_acoustic_residual`/tensor `dispersion_permutation_residual` use Hartree/Bohr^order per primitive cell. See [four-step fits](response.md#correlated-electric-dipoles-and-periodic-dispersion-force-constants); remaining electronic combinations retain their stated scope. Optional `field` selects analytic nuclear derivatives of the stationary finite-bias Berry electric enthalpy; fixed Cartesian field, insulating Gamma supercell of the k mesh. Reference residual/energy and available response residuals are reported; independent acceptance is tracked in response.md. For magnetic/joint-field nuclear derivatives use `periodic-magnetic-scf`; its accepted finite-Gamma subset and remaining combinations are documented separately. Every distinct path also returns its actually solved relative `response_residuals`; these are retained before any tensor permutation reuse. The driver includes the implicit CHG term for omegaB97X-D. Combined all-direction validation is pending. |
| `periodic-cubic-fc` | Complete cubic force constants `d3E/(dx_i dx_j dx_k)` of a periodic cell per primitive cell for Gamma-point displacements (every atom moves with all its lattice images), by the integral series of `periodic-cubic-fc-along`: first-order responses of every combined direction give its third energy coefficient (2n+1 rule) and the polarization identity the mixed derivatives. One path per combined direction; every path recomputes the series. Implemented, not yet validated by acceptance tests | `driver::run("periodic-cubic-fc", ...)`; `response::nuclear::lattice_force_constant_tensor` | `force_constant_tensor` in Hartree/Bohr^3 per primitive cell, indices `3*atom + axis`  Accepts `dispersion=d2/d3bj/d3/chg` with a pair cutoff in Bohr. CN and optional ATM settings belong to D3. [D2 model and verification](response.md#periodic-d2-empirical-dispersion); adds analytic dispersion third/fourth derivatives through the existing generic lattice energy. `dispersion_energy_per_primitive_cell` is Hartree; dispersion force-constant arrays and raw `dispersion_acoustic_residual`/tensor `dispersion_permutation_residual` use Hartree/Bohr^order per primitive cell. See [four-step fits](response.md#correlated-electric-dipoles-and-periodic-dispersion-force-constants); remaining electronic combinations retain their stated scope. Optional `field` selects analytic nuclear derivatives of the stationary finite-bias Berry electric enthalpy; fixed Cartesian field, insulating Gamma supercell of the k mesh. Reference residual/energy and available response residuals are reported; independent acceptance is tracked in response.md. For magnetic/joint-field nuclear derivatives use `periodic-magnetic-scf`; its accepted finite-Gamma subset and remaining combinations are documented separately. Uniformly compensated charged fixed-lattice paths are connected. [Scope and acceptance](response.md#charged-gamma-force-constants-and-higher-elastic-constants). The driver includes the implicit CHG term for omegaB97X-D. Combined all-direction validation is pending. |
| `periodic-quartic-fc` | Complete quartic force constants of a periodic cell for Gamma-point displacements, with responses through second order. Implemented, not yet validated by acceptance tests | `driver::run("periodic-quartic-fc", ...)`; `response::nuclear::lattice_force_constant_tensor` | `force_constant_tensor` in Hartree/Bohr^4 per primitive cell  Accepts `dispersion=d2/d3bj/d3/chg` with a pair cutoff in Bohr. CN and optional ATM settings belong to D3. [D2 model and verification](response.md#periodic-d2-empirical-dispersion); adds analytic dispersion third/fourth derivatives through the existing generic lattice energy. `dispersion_energy_per_primitive_cell` is Hartree; dispersion force-constant arrays and raw `dispersion_acoustic_residual`/tensor `dispersion_permutation_residual` use Hartree/Bohr^order per primitive cell. See [four-step fits](response.md#correlated-electric-dipoles-and-periodic-dispersion-force-constants); remaining electronic combinations retain their stated scope. Optional `field` selects analytic nuclear derivatives of the stationary finite-bias Berry electric enthalpy; fixed Cartesian field, insulating Gamma supercell of the k mesh. Reference residual/energy and available response residuals are reported; independent acceptance is tracked in response.md. For magnetic/joint-field nuclear derivatives use `periodic-magnetic-scf`; its accepted finite-Gamma subset and remaining combinations are documented separately. Uniformly compensated charged fixed-lattice paths are connected. [Scope and acceptance](response.md#charged-gamma-force-constants-and-higher-elastic-constants). The driver includes the implicit CHG term for omegaB97X-D. Combined all-direction validation is pending. |
| `toec` | Third-order Lagrangian elastic constants from `F(t) = (I + 2 t eta)^(1/2)` with clamped ions or first-order internal relaxation. The optional Cartesian field selects the stationary Berry electric enthalpy of a neutral insulator. | `driver::run("toec", ...)`; `response::nuclear::lattice_elastic_constants_at_field` | Hartree/Bohr^3 and GPa in Voigt order xx, yy, zz, yz, xz, xy. `internal_relaxation` in Bohr per unit strain. Raw relative orbital residuals in `response_residuals`. [Scope and acceptance](response.md#charged-gamma-force-constants-and-higher-elastic-constants) |
| `foec` | Fourth-order Lagrangian elastic constants with responses through second order. The relaxed path includes `t u' eta + t^2 u''(eta, eta)/2`; the field convention is the same as for `toec`. | `driver::run("foec", ...)`; `response::nuclear::lattice_elastic_constants_at_field` | Same elastic units and residuals as `toec`, plus `second_internal_relaxation` in Bohr, indexed `[6 I + J][3 atom + axis]`. [Scope and acceptance](response.md#charged-gamma-force-constants-and-higher-elastic-constants) |
| `mode-gruneisen` | Analytic Gamma phonon derivatives for an insulating Gamma supercell. Order one includes first-order ion relaxation and optional shear. Order two gives isotropic log-volume curvature with clamped or relaxed ions. The geometry-only dispersion energy enters the same derivative and relaxation paths. | `driver::run("mode-gruneisen", ...)`; `response::nuclear::{lattice_mode_gruneisen_with_dispersion, lattice_volume_gruneisen_with_dispersion}` | Dimensionless mode parameters and raw relative response residuals. Driver frequencies in cm^-1, force constants in Hartree/Bohr^2 and dispersion energy in Hartree per primitive cell. |
| `periodic-beta` | Clamped-ion electric susceptibility `chi_ab = dP_a/dE_b` of a periodic cell and its field derivative `d2P_a/(dE_b dE_c)` (the second-order susceptibility is half of it). The field enters the energy as `-Omega E.P` with the polarization of the single-point Berry phase of the Gamma-point supercell of the k mesh, `sum_a (E.a_a)/(2 pi) [sum_s Im ln det(1 + D_s dQ_a) - sum_A Z_A b_a.R_A]`, `dQ_a = <m\|exp(i b_a.r) - 1\|n>`: the product of the Berry-phase strings of the mesh. It is a function of the density matrix and joins the integral series of `periodic-cubic-fc-along`, so field components are further coordinates of the same paths and third derivatives follow from first-order responses (2n+1 rule). The discretization of the Berry phase on the k mesh differs from the `dielectric-tensor` route and agrees with it in the limit of a dense mesh. Insulating all-electron neutral closed-shell cells. Nonzero HF water beta passes an independent four-step fit; Raman and remaining combinations retain their separate pending acceptance | `driver::run("periodic-beta", ...)`; `response::nuclear::lattice_field_response` | `susceptibility` and `dielectric_tensor` dimensionless; `second_susceptibility` in atomic units of polarization per squared field; nonzero HF water beta verified with all 27 unprojected index orders and a four-step fit; `field` / CLI `--field x,y,z` selects a self-consistent finite-bias reference through the common Berry enthalpy; Rust `lattice_field_response_at_field` / `lattice_electric_tensor_at_field` preserve zero-field compatibility wrappers. `reference_field` is Hartree/e/Bohr on native routes and converts on ASE; raw path residuals remain explicit. Biased HF/PBE/TPSS/PBE0 alpha/beta/gamma acceptance passes all twelve four-step fits; relaxed-ion biased states need ionic stationarity Correlated lattice path: typed named/custom PT2 parent, full periodic Ewald integrals, every required orbital/pair order through four, integer 3D Gamma supercell model; physical acceptance tracked in response.md. Native Python, ASE and CLI use this same driver. |
| `periodic-gamma` | Static clamped-ion third-order electric susceptibility from the fourth derivative of the existing single-point Berry-phase electric enthalpy; second-order orbital response and the 2n+1 rule. Compatible insulating 3D Gamma-supercell models; four-step fits and static symmetry expansion validated for HF, SVWN5, PBE, TPSS, PBE0 and WB97X | `driver::run("periodic-gamma", ...)`; `response::nuclear::{lattice_electric_tensor,LatticeElectricTensor,ElectricTensorSymmetry}` with polarization derivative order 3 | Flat `tensor`/`third_susceptibility`, shape [3,3,3,3], actual d3P/dE3; Gaussian convention e^2 Bohr^2/Hartree^2; `chi3=tensor/6`. Raw response and integral-series reference residuals; `ElectricTensorSymmetry::IntrinsicPermutation` evaluates 15 independent static components, `Unprojected` evaluates all 81. The raw permutation defect is measured only for `Unprojected`, otherwise null. Relaxed-ion, dynamic, lower-dimensional and correlated combinations remain pending; HF water also passes the unprojected 81-component four-step gamma acceptance; `field` / CLI `--field x,y,z` selects a self-consistent finite-bias reference through the common Berry enthalpy; Rust `lattice_field_response_at_field` / `lattice_electric_tensor_at_field` preserve zero-field compatibility wrappers. `reference_field` is Hartree/e/Bohr on native routes and converts on ASE; raw path residuals remain explicit. Biased HF/PBE/TPSS/PBE0 alpha/beta/gamma acceptance passes all twelve four-step fits; relaxed-ion biased states need ionic stationarity Correlated lattice path: typed named/custom PT2 parent, full periodic Ewald integrals, every required orbital/pair order through four, integer 3D Gamma supercell model; physical acceptance tracked in response.md. Native Python, ASE and CLI use this same driver. |
| `periodic-raman` | Born effective charges `Omega dP_a/du_k` and the displacement derivative of the susceptibility `d chi_ab/du_k` (Raman tensor of the Gamma-point displacement `k`) from the same paths. With `relaxed_ion` the static susceptibility with the lattice contribution and `du/dE` are reported. Implemented, not yet validated by acceptance tests | `driver::run("periodic-raman", ...)`; `response::nuclear::lattice_field_response` | `born_charges` in e; `raman_tensor` in Bohr^-1 |
| `electrostriction` | Strain derivative of the susceptibility at clamped ions, `-(1/Omega) d3F/(dE_a dE_b d eta_I)` with `F = E - Omega E.P`. Field components refer to the reference lattice (fixed reduced field `E.a_i`), so the derivative is the proper one; `susceptibility_strain_derivative` adds the factor from the strain dependence of the volume. With `relaxed_ion` the atoms follow the field and the strain to first order (`u' = -Phi^+ d2F/(du dx)`) and the relaxed tensors are reported next to the clamped ones. Implemented, not yet validated by acceptance tests | `driver::run("electrostriction", ...)`; `response::nuclear::lattice_field_response` | `strain_susceptibility` dimensionless, `[a][b][Voigt]` |
| `piezo-higher` | Proper clamped-ion piezoelectric tensor `-(1/Omega) d2F/(dE_a d eta_I)` and its strain derivative `-(1/Omega) d3F/(dE_a d eta_I d eta_J)` at fixed reduced field; with `relaxed_ion` also the relaxed-ion tensor and its strain derivative. Implemented, not yet validated by acceptance tests | `driver::run("piezo-higher", ...)`; `response::nuclear::lattice_field_response` | `piezoelectric_tensor`, `second_piezoelectric` in e/Bohr^2 |
| `periodic-efield` | Self-consistent field of a periodic cell in a finite uniform electric field: the stationary point of `E - Omega E.P` with the single-point Berry phase, on the functional of the integral series of the Gamma-point supercell of the k mesh, started from the zero-field reference. The field must stay below the Zener limit `gap/(e L)` of the supercell length `L`. Forces are the derivative of that functional at fixed density with the energy-weighted density. Response properties of the finite-field state are not available. Implemented, not yet validated by acceptance tests | `driver::run("periodic-efield", ...)`; `response::nuclear::lattice_finite_field` | `field` in Hartree/(e Bohr); `field_energy` in Hartree per primitive cell; `dipole` in e Bohr modulo the lattice vectors; forces in Hartree/Bohr Correlated lattice path: typed named/custom PT2 parent, full periodic Ewald integrals, every required orbital/pair order through four, integer 3D Gamma supercell model; physical acceptance tracked in response.md. Native Python, ASE and CLI use this same driver. Correlated `response_order=0..4`, `field_direction` in Hartree/e/Bohr per dimensionless path parameter; factorial-normalized energy coefficients in Hartree per primitive cell, unscaled SS/OS coefficients and raw orbital/pair/projector/metric/number/stationarity diagnostics. The three-dimensional Berry-bias route includes the lattice D2, D3/ATM or CHG correction. [Finite-field scope and acceptance](response.md#finite-electric-bias-with-empirical-dispersion). |
| `dielectric-q` | Self-consistent density response function `chi(q, q)` and inverse dielectric function `1 + 4 pi chi/q^2` of a periodic cell at wave vectors commensurate with the k mesh. The potentials `lambda cos(q.r)` and `lambda sin(q.r)` are one-electron operators of the Gamma-point supercell (the pair transforms of its Coulomb sums); their amplitudes are coordinates of the same energy paths as displacements and strain, and the second derivatives of the energy, `(Omega/2)[chi(q, q) +- Re chi(q, -q)]`, add up to `Omega chi(q, q)`. The response includes the Hartree and exchange-correlation kernels and the local fields; `q_points` are reduced coordinates whose products with the k mesh are integers. Implemented, not yet validated by acceptance tests | `driver::run("dielectric-q", ...)`; `response::nuclear::lattice_density_response` | `chi` in electrons/(Hartree Bohr^3); `inverse_dielectric`, `dielectric` dimensionless; `q` in Bohr^-1 |
| `orbital-susceptibility-q` | Insulating 3D Gamma-supercell transverse perturbations with the uniform-potential sum rule. Collinear and all-electron spinor providers use their semilocal XC and exact-exchange density kernels, including the current derivative for meta-GGA. The spinor provider accepts a fixed local spin bias. [Model scope](api-guide.md#orbital-susceptibility-at-finite-q) | `driver::run("orbital-susceptibility-q", ...)`, `response::nuclear::lattice_orbital_susceptibility`, or `periodic::spinor_scf::SpinorDftBackend::orbital_susceptibility_q` | `chi=-4 E2/(volume q^2)` in atomic units of volume susceptibility. Each transverse direction has three raw solve residuals. The spinor route gives the full phase-averaged `chi_transverse` tensor, per-primitive-cell energy matrices and unscaled imaginary and transpose defects. |
| `periodic-dc` | Divide-and-conquer Kohn-Sham of a periodic cell at Gamma (`k_mesh` 1,1,1; use a supercell): the fragments of `dc_fragments` with the atoms within `dc_buffer` by the minimum image are subsystems, whose blocks of the Kohn-Sham and overlap matrices of the cell are diagonalized and occupied with a common Fermi level (`dc_smearing`); the free energy is the Kohn-Sham functional of the assembled density minus `kT` times the partition-weighted entropy. The Kohn-Sham matrices come from the integral series of the cell (`periodic-cubic-fc-along`) at the reference geometry, so no self-consistent field of the whole cell is solved. `forces` gives `-dA/du` from first-order series along each displacement with the relaxation matrix of the partition response, `dA/dx = E^(x)[P] + sum_s Tr[Z_s F_s^(x)] - Tr[W S^(x)]`. Implemented, not yet validated by acceptance tests | `driver::run("periodic-dc", ...)`; `response::nuclear::lattice_divide_conquer` | Free energy in Hartree per cell; forces in Hartree/Bohr; `dc_buffer` in Bohr; `dc_smearing` in Hartree |
| `dc-stress` | Stress of the divide-and-conquer free energy of a periodic cell: the same adjoint with the first-order series along each deformation `1 + eps` of the lattice (atoms at fixed fractional coordinates), symmetrized and divided by the volume. Implemented, not yet validated by acceptance tests | `driver::run("dc-stress", ...)`; `response::nuclear::lattice_divide_conquer` | `stress` in Hartree/Bohr^3 and GPa; `strain_derivative` in Hartree |
| `periodic-nmr-shielding` | Nuclear magnetic shielding in a periodic cell: the mixed second derivative of the energy in the amplitude of the transverse vector potential `a e sin(q.(r - R_A))` (field `a q (q^ x e)` at the nucleus; near the nucleus a Landau gauge centred there) and in the moment of nucleus `A`, whose periodic images form a lattice of dipoles with the vector potential `alpha^2 m x grad_R Phi` of the Ewald potential `Phi` (screened kernel in real space by exact dual numbers, damped kernel in reciprocal space, no `G = 0` component). `sigma = (1/q) d2E/(da dm)` has a diamagnetic expectation value and the paramagnetic term `2 sum y B` of the imaginary response to the wave; of the kernel only exact exchange acts. The value tends to the uniform-field shielding for `q -> 0` and exists only at wave vectors commensurate with the k mesh; wave vectors along two axes give the full tensor by least squares. The shape-dependent bulk magnetization term is not included. LDA, GGA and hybrids; no meta-GGA. Implemented, not yet validated by acceptance tests | `driver::run("periodic-nmr-shielding", ...)`; `response::nuclear::lattice_nmr_shielding` | Shieldings in ppm; `q` in Bohr^-1 |
| `berry-dipole` | Berry curvature dipole `D_ab = int d3k/(2 pi)^3 sum_n f_n d Omega_(n,b)/dk_a` of a periodic cell in its Fermi-surface form `int (-df/de) v_a Omega_b`, on `transport_k_mesh` at `transport_temperature_k`. The curvature comes from the physical velocity matrices of the converged bands (with their AO connection terms): every exactly degenerate level is one multiplet with the matrix `Omega_b = i eps_bcd sum_m v_c\|m><m\|v_d/(e - e_m)^2` over the other levels and contributes `-f'(e) Tr[v_a Omega_b]`, independent of the basis inside the multiplet. The chemical potential is the reference level plus `transport_mu_shift`, or the level that holds `berry_dipole_doping` additional electrons per cell. It vanishes by symmetry with inversion. Implemented, not yet validated by acceptance tests | `driver::run("berry-dipole", ...)`; `periodic::coherent_transport::berry_curvature_dipole` | `berry_curvature_dipole[a][b]` dimensionless; chemical potential in Hartree |
| `lab-frame-spin-spiral` | Energy `E(q)` of a flat spin spiral of reduced wave vector `spiral_q` in a 1D/2D/3D cell, by the variational two-component Bloch SCF in the smallest supercell that holds the spiral (`k_mesh` must be divisible by it; no generalized Bloch condition). The state is seeded by the Zeeman term `sigma.B/2` of `B = h [e1 cos(q.r) + e2 sin(q.r)]` (`h = spiral_seed_field`, `e1 x e2 = spiral_axis`) and then converged again without the seed from the seeded state; `nuclear_soc_scale` (default 0) adds the one-electron Pauli nuclear spin-orbit term to the final SCF. Reports the projection of the spin density on the seeded texture and on the opposite chirality, so a collapsed moment is visible. Total-number occupations, core guess, CDIIS. Implemented, not yet validated by acceptance tests | `driver::run("lab-frame-spin-spiral", ...)`; `periodic::spinor_scf::{BiasedSpinorBackend,SpinorDftBackend,run_spinor_scf}` | `energy`, `seeded_energy`, `seed_energy`, `nuclear_soc_energy` in Hartree per cell of the supplied structure; `spiral_wave_vector` in 1/Bohr; `spiral_spin_projection`, `magnetization` as Pauli expectation values (electrons); `spiral_seed_field` in atomic units of magnetic field |
| `dmi` | Chirality-odd spin-orbit energy of the flat spin spiral of `lab-frame-spin-spiral`: `dmi_energy = [E_SO(direct) - E_SO(mirrored)]/2` with the nuclear spin-orbit expectation value `E_SO` on the states converged without spin-orbit coupling (first order in `nuclear_soc_scale`); the mirrored chirality is the rotation of every spin by pi about `e1`, and the energy change under that rotation is reported as `chirality_rotation_energy_defect`. `dmi_self_consistent` adds the same half difference of SCF energies converged with the spin-orbit term from both chiralities. `dmi_per_wave_number` divides by `\|q\|` and equals a micromagnetic spiralization per cell only in the long-wavelength limit. It vanishes by symmetry with inversion. Implemented, not yet validated by acceptance tests | `driver::run("dmi", ...)`; `periodic::spinor_scf::{BiasedSpinorBackend,SpinorDftBackend,run_spinor_scf}` | `dmi_energy`, `soc_even_energy`, `nuclear_soc_energies`, `dmi_energy_self_consistent` in Hartree per cell of the supplied structure; `dmi_per_wave_number` in Hartree*Bohr |
| `bands` | Band structure along a reduced k path | `driver::run("bands", ...)` with `path` (reduced vertices), `points` per segment (20), optional `labels`; the converged Fock is rebuilt at each path point by `periodic::scf::reference_fock_at` | `k_path` (1/Bohr), `k_path_reduced`, `path_distance`, per channel `band_energies[k][band]`, `chemical_potential` |
| `periodic-spin-response` | Finite-q/frequency longitudinal spin-Zeeman response of a 3D collinear reference | `driver::run("periodic-spin-response", ...)`; `dfpt::spin::periodic_longitudinal_spin_response`; Cartesian `response_q`, frequency controls, complex `spin_probe_field=[re,im]` (default `[1,0]`), `spin_g_factor=2` and shared response solver controls | `density[spin][k]` in the independent `(k+q,k)` space; induced spin/charge electron numbers and magnetic moment per cell, susceptibility per atomic magnetic field, raw `response_residuals` and iterations. Source is +/-g B_z N(q)/4; q in 1/Bohr and frequency in Hartree. Uses the eligible shared adiabatic density/spin screening backend. Spin-only longitudinal probe; uniform static equilibrium uses `pauli-spin-susceptibility`. Orbital/transverse/SOC response is separate. Existing tests: `periodic_longitudinal_spin`; no new acceptance measurement is claimed by this documentation repair. |
| `periodic-field-scf` | Complex self-consistent 1D/2D/3D collinear orbital reference in a real static periodic scalar/vector field; LDA/GGA and full/range-separated exchange; zero mean magnetic flux | `driver::run("periodic-field-scf", ...)`; `periodic::scf_boundary::run_dimensional_field_scf`; `magnetic_scf::StaticPeriodicField`; field wave integer indices, real/imaginary vector and scalar Fourier coefficients, amplitude; dimensional cutoff/charge controls | Hartree energy/free energy and analytic amplitude derivative; full complex spin/k density and Fock; raw orbital and spinful TR defects, stationarity, occupation, particle-number, component-energy and field Hermitian defects; no imposed orbital conjugation; spin Zeeman and nuclear SOC use the spinor route; current-dependent XC and the remaining magnetic response combinations are tracked  Accepted uniform wire/in-plane-slab extension: `uniform_magnetic_field=[Bx,By,Bz]`, `magnetic_origin=[Ox,Oy,Oz]`, `field_amplitude`; wires accept every direction and ordinary slab Bloch gauges accept in-plane fields. Full continuum orbital coupling, common origin AO phase; spinor route adds `sigma.B/2`, nuclear `grad(V) cross (p+A)` and its analytic amplitude derivative. Reduced atomic field, origin in Bohr; energy/derivative in Hartree per cell. Normal slab flux, ECP/current-dependent terms and remaining response combinations are tracked separately. Rust: `UniformOpenField::{gauge_matrix,origin_phase_wave,vector_potential,operators}`, `run_dimensional_uniform_scf`, `SpinorDftBackend::with_uniform_open_field`. Collinear Fourier-field force and stress retain fixed reduced wave indices and Cartesian amplitudes, including current and A squared terms. Raw field-energy reconstruction defects use Hartree. The affine term uses Hartree per cell and its stress is divided by the active periodic measure. Explicit `reference=gks` or `reference=ghf` uses the shared complex spinor SCF. Nuclear spin-orbit coupling is opt-in on this entry. [Controls and validation](response.md#explicit-generalized-reference-on-static-periodic-field-scf). |
| `periodic-spinor-scf` | General complex two-component 1D/2D/3D SCF; full spin coherence, total-number occupations, rotationally invariant LDA and full/range exchange | `driver::run("periodic-spinor-scf", ...)`; `periodic::spinor_scf::{SpinorEnergyBackend,SpinorDftBackend,run_spinor_scf}`; general Hermitian spin/SOC matrices, analytic point/finite-nucleus Pauli SOC, prescribed-potential SOC with kinetic-momentum contact | Hartree total/Mermin energy and amplitude derivative; [k][up AOs,down AOs] complex density/Fock; Pauli spin polarization, raw spinful TR defect, canonical stationarity, occupation consistency, particle number, metric and energy residuals; separate nuclear SOC expectation; core/two-electron SOC, nuclear contact for Fourier vector fields and spin-gradient/current XC remain tracked; uniform flux has the separate magnetic-torus row  Accepted uniform wire/in-plane-slab extension: `uniform_magnetic_field=[Bx,By,Bz]`, `magnetic_origin=[Ox,Oy,Oz]`, `field_amplitude`; wires accept every direction and ordinary slab Bloch gauges accept in-plane fields. Full continuum orbital coupling, common origin AO phase; spinor route adds `sigma.B/2`, nuclear `grad(V) cross (p+A)` and its analytic amplitude derivative. Reduced atomic field, origin in Bohr; energy/derivative in Hartree per cell. Normal slab flux, ECP/current-dependent terms and remaining response combinations are tracked separately. Rust: `UniformOpenField::{gauge_matrix,origin_phase_wave,vector_potential,operators}`, `run_dimensional_uniform_scf`, `SpinorDftBackend::with_uniform_open_field`. Joint03 bounded runtime-verified increment: explicit reference=gks/ghf selects this existing complex Bloch solver; GHF requires an HF energy provider, and ordinary/common-orbital sectors are refused. The same converged SCF mesh supplies band_energies, k_points and band_edges, with normalized band_eigen_residuals and band_metric_defects. No spin factor of two is applied. A gap is reported only for a uniform exact integer occupied prefix at zero electronic temperature; thermal or incompatible occupations leave band_edges null. This adds neither arbitrary-path interpolation nor generalized force/DFPT/Berry/ROKS support. The earlier three-dimensional two-k-point HF spin-rotation baseline passed with complex density and5.55e-17 Hartree energy difference; that earlier baseline did not execute the new report fields. Joint03 now independently checks both GKS/GHF two-k HF mesh reports, their complex density, orbital/occupation/number gates, units and occupied band-edge reconstruction. This bounded HF report proof does not establish generalized PBE or property-response accuracy. |
| `periodic-magnetic-scf` | Uniform-flux continuum London spinor SCF with covariant semilocal XC on slabs and bulk | `driver::run("periodic-magnetic-scf", ...)` with `reference="gks"` or `"ghf"` for an HF provider. Low-level `MagneticBlochBackend` / `MagneticDftBackend` with `run_spinor_scf` | Energy in Hartree per magnetic cell, `reference_type`, flux quanta, SCF residuals, and independent spatial and XC angular diagnostics. The optional finite-Gamma nuclear report and 3D Berry electric branch have their own controls. [Shared settings and scope](api-guide.md#run-a-uniform-magnetic-flux-scf) |
| `dos` | Gaussian-broadened density of states and, with `projected`, its Mulliken decomposition by atom and angular momentum (band weights `sum_(mu in group) Re[conj(C_mu) (S C)_mu]`, which sum to one per band) | `driver::run("dos", ...)` with required `sigma` (Hartree), `points` (1000), optional `energy_window`, optional denser `dos_mesh`, `projected` (default false) | `energies`, per channel `dos` (states/Hartree/cell, both spins for a restricted reference) and `integrated`, `sampling_k_points`, `chemical_potential`; with `projected` also `projection_groups[]` (`atom`, `l`, `label`) and per channel `projected_shells[group][point]`, `projected_atoms[atom][point]`, `projection_sum_residual` 1D/2D/3D RKS, UKS, ROKS and GKS on the stationary k mesh. ROKS uses spin-resolved semicanonical common-orbital blocks. Reports include states per cell, exact window counts and integration and projection residuals. Separate frozen sampling uses the bulk collinear route. Field-free 1D/2D RKS and UKS and 1D/2D/3D ROKS and GKS reference energies include geometry-only dispersion. [Periodic dispersion scope and acceptance](response.md#geometry-only-dispersion-with-periodic-common-orbital-references) [Explicit periodic D1 inputs and derivative scope](response.md#periodic-d1-and-analytic-dispersion-derivatives). [Spinor VV10 energy and first geometry derivatives](response.md#first-geometry-derivatives-of-variational-periodic-vv10). |
| `pauli-spin-susceptibility` | Bare equal/unequal-channel rigid-band thermodynamics; `exchange_enhanced=true` solves the generic static thermal LDA/GGA/meta-GGA spin and charge response for equal or unequal collinear channels, at fixed total number and fixed reference spin bias | `driver::run("pauli-spin-susceptibility", ...)`; `periodic::spin_susceptibility::longitudinal_spin_susceptibility`; `temperatures_k`, `fermi_k_mesh`, `spin_g_factor=2`, common response controls | Susceptibility per cell in atomic units, dimensionless SI volume response, per-spin DOS in states/Hartree/cell; common `chemical_potential_response` in Hartree per atomic magnetic field, `charge_compressibility` in electrons/Hartree/cell, baseline `spin_chemical_potentials` and `reference_spin_bias` in Hartree. Raw unconstrained/constrained number, spatial density sum, Hermiticity, fresh equation and solver residuals. Default reciprocal cutoff 16 inverse Bohr for unrestricted/unequal channels, 6 for restricted channels. [Ensemble, limits and measured acceptance](periodic.md#35-exchange-enhanced-static-spin-susceptibility) Bare mode also accepts unequal collinear bands and `spin_g_factor`: frozen bands are reoccupied with a common chemical potential and zero reservoir spin bias. `spin_electrons`, `fixed_number_spin_dos`, `rigid_band_spin_magnetization`, `chemical_potential_response`, `delta_spin_numbers` and raw number defects describe that ensemble. It differs from the enhanced mode's fixed baseline bias; no self-consistent response is claimed. |
| `cli` | The command line as a library entry point; canonical executable also installed by pip | `dft_rs::cli::run_arguments(args: Vec<String>) -> i32` (arguments without the program name); both Rust executables and the Python entry point share it | Exit status 0 or 1; output on process standard streams; [portable packaging acceptance](cli.md#portable-python-packaging) |
| `piezoelectric-strain` | Algebraic proper d=eS on supplied fixed-electric-field tensors, with explicitly matching clamped/relaxed-ion convention; no SCF, mechanical stability or method-specific property acceptance inferred | `driver::run("piezoelectric-strain", ...)`; `geomech::piezoelectric::strain_coefficients` | Inputs `piezoelectric_stress` 3x6 e/Bohr^2, `piezoelectric_compliance` 6x6 Bohr^3/Hartree, required `ion_response=clamped/relaxed`; engineering Voigt xx,yy,zz,yz,xz,xy. Outputs atomic d, C/N and pm/V. Joint03 bounded Rust/native/ASE/canonical and alias CLI checks pass all18 entries, SI conversions and input gates; the independent four-step fits have p=1.99814..1.99953 and minimum R2=0.999999768. Supplied tensors retain their own physical-validation responsibility. |
| `aggregate-moduli` | Isotropic multiphase mineral aggregate (no electronic structure): Voigt, Reuss and Hill averages, Hashin-Shtrikman bounds in Berryman's form and the self-consistent estimate for spherical grains | `driver::run("aggregate-moduli", &System::default(), &controls)` with `phases` (rows of volume fraction, K, G in GPa; required); `geomech::aggregate_moduli(&phases)` | `voigt_gpa`, `reuss_gpa`, `hill_gpa`, `hashin_shtrikman_lower_gpa`, `hashin_shtrikman_upper_gpa` (each [K, G]), `self_consistent_gpa` (null when the iteration does not converge) |
| `rock-physics` | Porous rock from supplied mineral and fluid data: dry frame (critical porosity, Hertz-Mindlin contact theory, Hashin-Shtrikman upper bound with empty pores, dry spheroidal pores of one aspect ratio by the differential effective medium, the Kuster-Toksoz estimate or Berryman's self-consistent estimate, or given), Gassmann fluid substitution, Biot coefficient and modulus, Skempton coefficient, drained and undrained Poisson ratios, bulk density, saturated wave speeds, Biot characteristic frequency, and the anisotropic Brown-Korringa tensor; physical inputs are required (no defaults) | `driver::run("rock-physics", ...)` with `mineral_bulk_gpa`, `mineral_shear_gpa`, `mineral_density_kg_m3`, `porosity`, `fluid_bulk_gpa`, `fluid_density_kg_m3`, `frame` and its parameters (`critical_porosity`; `coordination_number`, `effective_pressure_gpa`; `dry_bulk_gpa`, `dry_shear_gpa`; `pore_aspect_ratio` for `dem`, `kuster-toksoz` and `self-consistent`), optional `fluid_viscosity_pa_s` with `permeability_m2`, optional `mineral_tensor_gpa` with `dry_tensor_gpa`, optional `mineral_conductivity_w_mk` with `fluid_conductivity_w_mk`; `geomech::{fluid_substitution, brown_korringa}`, `geomech::inclusions::{dem, kuster_toksoz, self_consistent, conductivity_bounds}` | `dry_bulk_gpa`, `dry_shear_gpa`, `saturated_bulk_gpa`, `saturated_shear_gpa`, `biot_coefficient`, `biot_modulus_gpa`, `skempton_b`, `poisson_drained`, `poisson_undrained`, `bulk_density_kg_m3`, `vp_m_s`, `vs_m_s`, `biot_frequency_hz`, optional `brown_korringa_saturated_tensor_gpa`, optional `thermal_conductivity_w_mk` (arithmetic, harmonic and geometric means and the Hashin-Shtrikman bounds) |
| `feature-registry` | Registry of public calculations as data | `driver::registry::{features, find, registry_value}`; `driver::run("feature-registry", &System::default(), &Controls::new())` | `features[]`: `id`, `title`, `space`, `route`, Rust/CLI/Python/ASE entry names and physical target IDs |

Every function returns `Result<...>`. `Frequency` is `Static`,
`Retarded { omega, damping }`, or `Imaginary { frequency }`.
`ResponseSolverConfig` defaults to `tolerance=1e-10`, `max_iterations=500`,
`gmres_restart=40`, and `preconditioner_floor=1e-3`. The last setting affects
preconditioning only. `PeriodicResponseOptions` additionally defaults to
`q=Vec3::zero()`, static frequency, `fixed_number=true`, and
`reciprocal_cutoff=6.0`. `solver.reciprocal_waves()` lists the full screening
set; `solver.plane_wave_source(mode)?` builds a unit-amplitude source for `solve`.
General spin-dependent complex AO perturbations can also be supplied directly.
The Coulomb AO image range is `cell.hartree_real_space_cutoff()`, as in the
reference Ewald build. It is independent of the reciprocal cutoff and can
exceed the metric overlap range. The XC AO range follows the SCF options.

`ThermalSource { delta_temperature_hartree, ensembles }` supplies a uniform
static thermodynamic direction. Temperature here is `k_B T` in Hartree.
Each entry of `ensembles` is `Ensemble::FixedNumber { delta_number }` or
`Ensemble::FixedChemicalPotential { delta_mu }`, consistently with the solver's
`fixed_number` choice. A positive-temperature reference and q=frequency=0
are required. A saturated spin channel without a finite chemical potential
can carry only a zero-number canonical source.

```rust
use dft_rs::response::{dynamic_polarizability, Frequency, ResponseSolverConfig};

// mol, mb, functional, grid, scf, and opts come from the SCF calculation.
let report = dynamic_polarizability(
    &mol, &mb, &functional, grid.as_ref(), &scf, &opts,
    Frequency::Retarded { omega: 0.05, damping: 0.01 },
    &ResponseSolverConfig::default(),
)?;
let alpha_zz = report.value.alpha[2][2];
```
## Result conventions and physical scope

All eight calculations return `shape`, `functional`, `units`, `model`,
`axis_convention`, `converged`, `residuals`, `iterations`, `warnings`, and
`reference`. Reference diagnostics contain `eigen_residual`,
`orbital_residual`, `orthonormality_defect`, `projector_defect`, and
`particle_number_error`. Python additionally returns the SCF `energy`; periodic
reports include `free_energy`, q, frequency, temperature, ensemble, and cutoff.
CLI prints the same diagnostics; Rust stores them in `ResponseReport<T>`.
Nonconverged iterative solves return an error. Nuclear reports additionally
set `converged=false` when the rebuilt moving-basis diagnostics exceed the
larger of the requested tolerance and the reference residual (floor `1e-14`).

`alpha[a][b]` differentiates dipole component a with respect to field b, in
bohr^3. If `mu(t)=sum_n c[n] t^n` and
`E(t)=E_reference+t direction`, `c[n]` includes division by n!: directional
beta is `2*c[2]`, gamma is `6*c[3]`. Mixed derivatives have no factorial
normalization. Every field direction uses atomic electric field units.

Periodic mode indices are zero-based in the full set
`0 < |G+q| <= reciprocal_cutoff`, sorted by magnitude and Cartesian components.
Density is a rectangular `(k+q,k)` object. Python stores each Bloch matrix as
`{"real": matrix, "imag": matrix}`. AO response uses hartree^-1; Fourier
density and screened susceptibility use bohr^-3/hartree. The inverse dielectric
is dimensionless and obeys `epsilon^-1[row][column] = delta[row][column] +
4*pi/|G_row+q|^2 * screened_chi[row][column]`. Selecting output modes retains
all configured modes in the internal screening calculation.

Electric molecular calculations currently support collinear integer-occupation
references, with semilocal, hybrid/range-separated exchange, requested
self-consistent VV10, and linear PCM contributions. They use an adiabatic
response kernel. Nuclear calculations currently include all-electron exact
HF/semilocal/global hybrid/RSH models with analytic moving basis and grid.
They expose nuclear repulsion, explicit nuclear field energy, one-electron, Hartree, global exchange,
additional long-range exchange and semilocal XC coefficient arrays in
`energy_components`. `density_coefficients[n][spin][AO][AO]` and
`overlap_coefficients[n][AO][AO]` retain factorial normalization. With a
Bohr coordinate unit direction, `6*energy_coefficients[3]` has units
Hartree/Bohr^3 and `24*energy_coefficients[4]` Hartree/Bohr^4. Mixed responses
return the derivative itself. ECP, RI-J, nonlocal/empirical corrections,
solvent and additional one-electron operators require further geometry kernels;
this driver refuses them. Reference keywords for native nuclear calls are
those of `nuclear_taylor`, with `dispersion="none"`; explicit displacements
use Bohr on all surfaces, including ASE. The full functional-class goal
remains tracked in the ledger. Periodic density and dielectric calculations currently
support 3D collinear semilocal references, arbitrary Cartesian q, integer or
finite-temperature occupations, static/retarded/imaginary frequency. A q at
which no configured nonzero reciprocal mode remains needs a larger cutoff.
Finite-q exchange, periodic nonlocal correlation, 1D/2D electrostatics,
additional electronic spaces, and nuclear/strain/magnetic sources remain
within the implementation objective.

Molecular mixed paths combine `R(t)=R_reference+t*nuclear_direction` and
`E(t)=E_reference+t*electric_direction`. All nuclear paths are in Bohr,
including ASE; electric paths and reference fields use atomic units. Taylor
energy and dipole coefficients are divided by `n!`; mixed derivatives have
no factorial normalization. `energy_components["nuclear_field"]` includes
the explicit `-sum_A Z_A R_A.E` term. Dipole response also differentiates the
moving AO position operators and nuclear dipole, not only the density.

Spectroscopy returns `d mu/dR` (APT), `d alpha/dR` (Raman), or `d beta/dR`
(hyper-Raman) along the supplied nuclear path. `tensor` is row-major, has
`shape=[3]*electric_order`, and the output dipole index comes first. All
27 hyper-Raman entries have their own ordered field evaluations; the raw
permutation defect is measured before symmetrization. These are static
directional tensors; normal-mode intensities, full Cartesian tensors and
dynamic spectroscopic response remain separately tracked targets.

Analytic force constants use `order=3` or `4`. A request with `patterns`
returns a tensor with one axis per derivative slot and dimension equal to
the number of supplied atom-by-Cartesian Bohr displacement patterns. A
`coordinates` request uses unique IDs `3*atom+axis` in the caller's order;
the returned Cartesian tensor uses Hartree/Bohr^order. Omitting both selects
all `3N` coordinates. Pattern tensors are actual contractions, without
factorial normalization or assumed mass weighting; callers may supply
normal-mode displacement vectors with their chosen mass convention.

Each nondecreasing index tuple is computed analytically. Its distinct reverse
ordering is independently recomputed, and `permutation_comparisons` counts
these comparisons before filling the full tensor's permutations. It is not
a claim that every ordered tuple was independently solved. The complete
Cartesian tensor measures all raw translation sums over every atom, axis
and derivative slot. A smaller coordinate block or arbitrary pattern tensor
has `translation_residual=None`, `translation_comparisons=0`. No acoustic
projection modifies the values. Dense output storage and integral jets must
fit `incore_mb`; large requests should select patterns or coordinate blocks.
Full tensor output has `(3N)^order` entries and `C(3N+order-1,order)` independent
components; the current polarization contractions also cost multiple analytic
directional response evaluations per component.

Thermal results are variations along the supplied path: density variation is
dimensionless, Fourier variation is bohr^-3, entropy variation is in k_B, and
energy/chemical-potential variations use Hartree. Sources can combine temperature
and spin-resolved numbers or reservoirs. The linear equation includes induced
Hartree/XC fields and a mesh-global chemical-potential correction. Reported
`occupation_equation_defect` reevaluates the Fermi occupation derivative using
the final induced field; `number_constraint_defects` measures the canonical
constraints. `delta_free_energy` differentiates `A=E-k_B T S/k_B`, including
the explicit temperature derivative. Its numerical acceptance is tracked
separately from implementation and public-route availability.
The report model declares temperature-independent XC: the current semilocal
functionals have no explicit temperature argument. Fermi occupations, entropy,
and density-induced XC relaxation carry the temperature response.

## Analytic foundation APIs

These Rust building blocks are independently usable. Their presence does not
establish completion of an observable driver or of public-route parity.

- `response::krylov::{LinearOperator, solve, ResponseSolverConfig}` provides
  CG, MINRES, and restarted GMRES with original-operator residual evaluation.
- `response::constraints::{ConstraintSpace, solve_constrained}` solves in a
  constrained space and reports removed-mode multipliers and defects.
- `response::series::{evaluate_series, evaluate_series_many, SeriesScalar}`
  evaluates factorial-normalized runtime-order analytic Taylor coefficients.
  Input vectors have identical length `order+1`; the arena is scoped to the call.
  `evaluate_series_many_result` additionally accepts a fallible analytic
  kernel and restores the scoped arena before returning its error.
- `response::taylor::Taylor<S, N>` supports fixed-width nested Taylor AD.
- `XcFunctional::{taylor_energy, taylor_potentials}` and
  `ResolvedFunctional::semilocal_taylor_*` expose analytic density/parameter
  series through the existing Scalar energy expressions.
- `dfpt::thermodynamic::occupation_response` provides the analytic weighted
  Fermi occupation and chemical-potential response; a standalone primitive is
  separate from a self-consistent temperature-source observable.
- `integrals::analytic::{Primitive, GaussianKernel, overlap, kinetic,
  attraction, repulsion, boys_jet, boys_values}` differentiates unnormalized
  Cartesian Gaussian primitives with respect to centres, exponents, nuclear
  charges, and range separation. `attraction`, `repulsion`, and `boys_jet`
  take a composition order sufficient for the Taylor width and any nested
  dual directions. The nuclear driver connects these jets to contracted
  integrals, metric transport, moving quadrature and density response.
- `integrals::analytic_basis::{AnalyticShell, AnalyticNucleus, IntegralBlock,
  overlap, kinetic, dipole, attraction, repulsion}` contracts and transforms these
  primitive jets into the production spherical AO convention. `from_placed`
  lifts normalized production shells; `normalized` also differentiates
  primitive and contraction normalization for variable basis parameters.
  `IntegralBlock` stores row-major `values` with a rank-2 or rank-4 `shape`.
  `dipole(a, b, axis, origin)` differentiates the position operator in a
  moving spherical basis, with an explicit origin and Cartesian axis.
  Moving quadrature and nuclear response equations are connected through
  `response::nuclear` for the physical models described above.

## Orbital stability and following conventions

Default restricted channels are `internal-real` (singlet A+B),
`external-spin` (triplet A+B), and `external-complex` (singlet A-B).
An unrestricted reference defaults to `unrestricted-internal` (coupled
alpha/beta A+B). Optional `external-complex-spin` and
`unrestricted-complex` test additional imaginary collinear rotations.
`external-generalized` identifies the still-pending noncollinear spin-flip
backend; it is not silently included in the computed space or reported stable.
Periodic, fractional-occupation and generalized-reference stability remain
tracked targets requiring their own physical operators.

`StabilityOptions` defaults to `n_roots=3`, `tolerance=1e-7` Hartree,
`max_iterations=200`, `max_subspace=60`, `block_buffer=3`,
`resolution=None`, automatic channels, `verify_lowest=true`, and
`verify_steps=24`. The verdict resolution is `10*tolerance` unless explicitly
specified; it must not be below the eigensolver tolerance. A lowest value
above resolution is `stable`, below its negative is `unstable`, and inside
the interval is `indefinite`. An empty orbital space has no numerical lowest
value. Davidson accepts fresh **absolute** residual norms, reports their
relative counterparts as well, and returns unconverged numbers with warnings
and `provenance="unconverged"`. Lowest-root verification runs an independent
Rayleigh descent; an overturned root is explicitly unconverged.

The normalized mode satisfies `E''(0)=4*lambda` for a restricted spin-adapted
generator and `E''(0)=2*lambda` for an unrestricted combined-spin generator.
`eigenvector` uses occupied-major/virtual-minor ordering, alpha then beta
when unrestricted. `modes[].amplitudes` retain the native TDDFT
virtual-major/occupied-minor ordering. Dominant entries have magnitude at
least 0.1, are sorted by magnitude, and are limited to six.

Following is opt-in. Its defaults are three descents, angle 0.3 radians and
minimum accepted energy drop 1e-8 Hartree. Both signed exact orthogonal
orbital rotations are re-converged; only a converged lower-energy state is
accepted and re-analysed. Internal following preserves the chosen reference
class. External spin/complex instabilities remain reported and require an
explicit choice of another reference class. `converged` and `stabilized`
answer different questions. The final Rust state can be supplied directly
to another response driver; the dedicated Python/ASE getter returns that
state's energy, densities and orbitals but does not replace other calls'
independently constructed SCF reference.

For a full molecular single point, `native.calculate(..., stabilize=True)`
computes its energy, analytic gradient, charges, dipole, optional population
analysis and wavefunction on the final state. `DftRs(stabilize=True)` applies
the same driver to ordinary ASE properties, with the usual unit conversion.
Explicit `unrestricted=True` permits a broken-symmetry singlet reference;
following itself never changes that choice. These calls use the standard
`StabilizeOptions` defaults; the dedicated stability getter provides custom
analysis/descent controls. A lazy getter that cannot yet accept the chosen
stabilized/forced-unrestricted state raises a clear error instead of silently
constructing another reference. RI-J stabilized forces await an analytic
fitted-integral gradient and do not invoke the existing finite-difference
fallback. Periodic stabilized single points remain pending.

```console
dft-rs stability h2.xyz --unrestricted --stab-follow
dft-rs dynamic-gamma h2.xyz --unrestricted --stab-follow --electric-process optical-kerr --process-frequencies 0.03 --response-damping 0.006
```

```python
from dft_rs import stability, stabilize
report = stability([1, 1], [[0, 0, 0], [0, 0, 4.5]])
followed = stabilize([1, 1], [[0, 0, 0], [0, 0, 4.5]], unrestricted=True)
print(followed["initial_energy"], followed["energy"], followed["stabilized"])
```


The dimensional self-consistent driver uses the common finite-k SCF occupation,
metric, ADIIS/CDIIS and canonicalization iteration. Its dimensional compensated
Green operator replaces the legacy 3D Coulomb convention. Charged 1D/2D cells
require a physical Gaussian countercharge; a uniform background is available
only in 3D. No additional isolated-ion monopole energy is added. The complete
physical conventions, controls, acceptance fits and remaining field connections
are in [Dimensional self-consistent DFT](periodic.md#6-self-consistent-dimensional-dft).

### Hyperfine tensor conventions

`hyperfine` returns the specified collinear state's tensor, not an average
over degenerate electronic states. The contact term is
`(2 pi/3) alpha^2 g_e rho_s(R_A)/(m_p two_s)` per nuclear g in Hartree.
`total_coupling_per_g_mhz[nucleus][a][b]` is contact times the identity plus
the spin-dipolar tensor. `principal_couplings_per_g_mhz` is ascending;
`principal_axes_per_g[nucleus][Cartesian][principal]` contains its column
eigenvectors. With signed `nuclear_g`, the physical tensor is diagonalized
again: negative factors reverse the principal-value order, and the separate
`principal_axes` matches `principal_couplings_mhz`. Axis signs and the basis
inside a degenerate principal subspace are arbitrary.

The raw fields `dipolar_trace_per_g_mhz`, `tensor_asymmetry_per_g_mhz` and
`tensor_eigen_residual_per_g_mhz` are MHz; the last measures the maximum
entry of `A V - V diag(values)` using the unmodified tensor.
`tensor_axes_orthogonality_residual` is the maximum entry of `V^T V - I`.
`tensor_relative_residual` divides the largest trace, symmetry or eigen
defect by the actual total-tensor maximum norm (the exact zero tensor has
zero residual). The per-atom `tensor_converged` requires this residual and
orthogonality to be at most `tensor_residual_tolerance = 1e-10`.
No symmetry or trace projection is applied. The SCF diagnostics remain
those of the common reference report; this expectation value introduces
no extra orbital-response solve. All-electron tensor fields are null at
ECP sites; their pseudo-valence spin density is labelled separately.

Measured HF/PBE0/TPSS methyl tensors on native Python, generic Python, ASE
and both CLI names have maximum per-g trace magnitude 1.564e-13 MHz,
eigen residual 1.137e-13 MHz and axes residual 4.441e-16.
The independent closed-Gaussian operator check is within 5.552e-16 Bohr^-3;
its four-step derivative fits and the resolved electronic-state rotation
comparison are recorded in [response.md](response.md).


<!-- cli-scf-progress:begin -->
### SCF progress on the CLI

The common `dft-rs` entry prints iteration energy, energy change and raw
convergence residuals on stderr by default. Molecular, periodic, common-orbital
open-shell, finite magnetic and spinor SCF loops use the common logger.
JSON/text results remain on stdout. `--log-mode none` explicitly silences
progress; `console` is the common-driver default. `file`/`both` require
`--log-file PATH`. The pip-installed CLI uses this same entry point. Direct
library calls remain quiet unless logging is initialized.
<!-- cli-scf-progress:end -->

<!-- pairing-kernel-scope:begin -->
### Superconducting pairing and phonon mass components

`scdft-pairing-kernel` evaluates LM2005 Eq. (74) on explicitly supplied
normal-band energies, phonon energies, squared weighted vertices, chemical
potential and a screened Coulomb matrix. Each squared-vertex row is a flattened
`[initial band][final band]` matrix for one mode, including its mode/q weight.
Band quadrature weights are excluded from the returned kernel. All entries,
including both triangles, are computed independently. The raw asymmetry is
reported without projection. Phonon and Coulomb matrices are separate additive
terms. `K(epsilon+t*d)=sum_n K_n*t^n` defines the optional path coefficients;
`t` is dimensionless and a derivative is `n! K_n`.

Removable phonon resonances and Fermi-surface limits use analytic
divided-difference/derivative integrals. No denominator floor or finite
difference enters production. Orders 16 and 32 of the analytic-limit quadrature
provide raw absolute and relative error estimates for every requested order;
`converged` requires relative agreement at most `1e-10`. The normal-state
electronic functional is a provider of the input spectrum and vertices, so this
component does not branch on its registered name. It does not establish a
self-consistent superconducting reference by itself. The separate
`superconducting-state` driver connects K and generated or supplied Z to a
canonical complex gap, number and phase solve. The generated static phonon
mass is described below. Normal-density backreaction, the DFPT vertex adapter
and causal dynamical kernels remain pending. See [response.md](response.md).
<!-- pairing-kernel-scope:end -->

<!-- no-pair-component:begin -->
### Four-component electronic-sector projection component

`no-pair-projector` accepts Hermitian four-component H and a positive full-rank
overlap S from a balanced provider. The provider retains responsibility for
kinetic/magnetic balance, finite nuclei, interactions and electronic-space
ordering. `spectrum_separator` must lie inside the electronic/negative-energy
gap; for a free operator shifted by minus the rest energy it is `-c^2`.
The driver does not infer a balance condition from numeric matrices. It computes
the separated electronic projector as a subspace, including its coupling to
negative-energy states in every response order. Intra-sector degeneracies do
not require an arbitrary eigenvector derivative. No gap denominator is floored.

For dimensionless t, `H(t)=H+t*dH`, `S(t)=S+t*dS`; the driver returns
factorial-normalized coefficients through order four. The Rust function also
accepts higher coefficients of those paths. P is contravariant, with `P S P=P`;
the reported covariant projected operator is `S P H P S`. Common moving-overlap
transport and faer eigensolves/products are reused. Reports retain raw
idempotency, transported metric, commutator (Hartree), trace and complex-embedding
defects. The relative commutator divides by the measured coefficient convolution
of operator/projector maximum norms times matrix size. `converged` requires
these relative commutator and other dimensionless defects to be at most `1e-10`.
This component does not establish a self-consistent no-pair Dirac-Coulomb DFT
reference or complete relativistic observables. Those integrations remain pending.

Optional `dirac_operator_real` and `dirac_operator_imaginary` are **dimensionless
covariant one-electron operator** Taylor coefficients indexed
`[order][AO row][AO column]`, including zeroth order. Supply one through
`response_order+1` coefficients; absent real/imaginary parts and omitted higher
orders are exactly zero. Every combined coefficient must be Hermitian, finite
and match the four-component dimension. These public inputs and outputs remain
dimensionless under native and ASE unit conversion. No physical operator or
physical-unit conversion is inferred from numeric matrices.

The optional outputs `projected_operator_coefficients_real/imaginary` contain
factorial-normalized `Q=S P O P S` coefficients, with a precise
`projected_operator_model` label. The low-level Rust function
`properties::no_pair::no_pair_operator_taylor(H,S,O,order,separator,cutoff)`
retains the supplied operator's units and returns `NoPairOperatorTaylor`
(`sector` diagnostics and `operator` coefficients). This is bare one-electron
no-pair projection including moving metric and electronic/negative-sector
response, not decoupled X2C picture change, interacting DKS, two-electron
picture change or a conserved-current construction.

The isolated implementation/oracle acceptance passes 128 independent complex
operator/metric component fits through fourth order: p=2.000009042105408 to
2.0013609184029173, minimum R2=0.9999998761022915, at steps
[0.08,0.04,0.02,0.01] in dimensionless path parameter. All ordered coefficients,
the identity `O=S -> SPS`, Hamiltonian equivalence and input contracts pass.
The rebuilt `tests/no_pair_operator.rs` and native/ASE atomic/physical/both-CLI integration now pass on the scoped follow-up artifact. Static getter binding and explicit resource forwarding also pass after the reproduced Python wrapper defect was corrected. This remains bare operator projection, with the limitations above.



#### Generated finite-nucleus, kinetically balanced operator


The lower-level Rust contract is
`restricted_kinetic_balance<S: Scalar + Send + Sync>(shells: &[AnalyticShell<S>], nuclei: &[GaussianNucleus<S>], speed_of_light: S, outer_ad_order: usize) -> Result<BalancedDiracIntegrals<S>>`.
`GaussianNucleus` stores `center: [S;3]`, `charge: S` and `rms_radius: S`.
`AnalyticShell::normalized` accepts raw radial coefficients and propagates
normalization derivatives; `from_placed` lifts an already normalized basis.
The returned row-major `IntegralBlock<S,2>` fields are `spatial_overlap`,
`kinetic`, `potential`, `small_potential_real/imaginary`,
`hamiltonian_real/imaginary` and `overlap`, plus scalar `nuclear_repulsion`.
Spatial matrices have dimension N, small-potential matrices 2N and full
Hamiltonian/overlap matrices 4N. `outer_ad_order=0..32` must include all outer
nested dual levels; the two endpoint levels are added internally.
`matrix_coefficients<const W:usize>(&real, Option<&imaginary>, order)` converts
`Taylor<W>` matrix entries to `Vec<BlochMatrix>`; it requires `order<W`,
matching shapes and finite output coefficients. It supplies the unchanged
common no-pair projector. Invalid allocation sizes are checked before use.

Set `dirac_integrals="rkb"` and supply molecular atomic numbers, positions,
`basis` (default `sto-3g`) and positive `nuclear_rms_radii` (one per nucleus,
Bohr in native Rust/Python). Gaussian nuclear charge density is
`Z*(omega^2/pi)^(3/2)*exp(-omega^2*r^2)`, `omega=sqrt(3/2)/rms_radius`.
Its potential is `-Z*erf(omega*r)/r`. No isotope radius is guessed. ECP basis
sets are refused because this is the all-electron finite-nucleus operator.
`speed_of_light` defaults to 137.035999084 atomic units. At zero magnetic
field the small basis is `(sigma.p)/(2c)` times the large spatial basis;
AO ordering is `[large up,large down,small up,small down]`.

The rest-energy-shifted Hamiltonian has blocks `[V,T;T,W/(4c^2)-T]`,
`W=(sigma.p)*V*(sigma.p)`, and metric `diag(S,T/(2c^2))`. All spin-orbit
entries of W are retained. Two independent center dual levels act only on
the basis endpoints of each attraction integral, keeping the nuclear
potential fixed during those inner derivatives. Outer AD retains all
center, charge, radius, contraction and exponent derivatives. Every shell
pair is evaluated independently in parallel; no Hermitian triangle is
copied. This is the finite-basis restricted-kinetic-balance model described
in [the primary derivation](https://www.mdpi.com/2073-8994/12/7/1121).

Optional `nuclear_displacements` has shape `[atom][3]`,
`nuclear_radius_direction` has one entry per atom, and
`speed_of_light_direction` is a scalar. Nuclear positions, basis centers,
radii and c follow these linear paths in dimensionless t. The defaults
are zero directions. `response_order=0..4` returns all nonlinear integral,
metric and projector coefficients. The default separator is `-c(0)^2`.
Supplied-matrix controls cannot be mixed with this generated mode.

Additional generated-mode results are
`dirac_hamiltonian_coefficients_real/imaginary`,
`dirac_overlap_coefficients_real`, `kinetic_coefficients`,
`large_potential_coefficients`, `small_potential_coefficients_real/imaginary`
and `nuclear_repulsion_coefficients`. Energies are Hartree, overlap is
dimensionless; all coefficients have factorial normalization. Distinct
Gaussian nuclei interact with their Gaussian Coulomb energy, excluding
individual nuclear self-energies. Reports include unsymmetrized
`hamiltonian_hermiticity_defects` (Hartree) and
`overlap_hermiticity_defects` (dimensionless).

```python
report = dft_rs.no_pair_projector(
    [1], [[0.0, 0.0, 0.0]], dirac_integrals="rkb", basis="sto-3g",
    nuclear_rms_radii=[1.6e-5], response_order=4,
    nuclear_radius_direction=[1.0e-6],
)
```

The finite radius above is an explicit example input, not an isotope table.
The ASE static getter accepts an optional `Atoms`; geometry is always
Angstrom and control/report units follow `units="atomic"` or the default
ASE convention. Generated integrals are connected to the same no-pair
response used for supplied matrices. Restricted magnetic balance, London
basis response and interacting Dirac-Coulomb SCF/XC are separate remaining
work; this one-electron operator does not establish their completion.

<!-- no-pair-component:end -->

<!-- mixed-magnetic-scope:begin -->
### Mixed higher magnetic energy response

`magnetizability(..., magnetic_directions=[d1,...,dn])`, `n=1..4`, uses the
existing analytic London/common-origin field driver and the common exact
polarization identity to obtain `d^n E/(dt1...dtn)` at zero field,
`B=sum_j t_j*d_j`. Directions are atomic magnetic fields and parameters are
dimensionless; the returned mixed derivative and roundoff estimate are Hartree.
This exposes individual full-tensor contractions, including different,
repeated and non-orthogonal magnetic legs, without an additional SCF model.
It computes each directional coefficient and reports measured maximum response,
projector, stationarity and reference defects. Integer-occupation collinear
orbital references retain the field-path restrictions; spin Zeeman,
current-dependent correlation, correlated pair response and periodic magnetic
counterparts remain pending. Existing `response_order`/`magnetic_direction`
directional-series requests can be combined with the mixed request.
<!-- mixed-magnetic-scope:end -->


<!-- canonical-superconducting-state:begin -->
### Canonical complex superconducting state and response

`superconducting-state` solves the static band-decoupled singlet equations
`(1+Z_i)*Delta_i + sum_j K_ij*w_j*chi_j = 0`,
`chi_i=Delta_i*tanh(E_i/(2*kT))/(2*E_i)`,
`E_i^2=(epsilon_i-mu)^2+abs(Delta_i)^2`, and
`N=sum_i w_i*(1-(epsilon_i-mu)*tanh(E_i/(2*kT))/E_i)`.
Each band denotes a two-spin pairing channel; quadrature weights are not
absorbed into K. The complex gap and chemical potential are solved together.
This is the static band-decoupling approximation of
[the primary derivation, Eqs. (3)-(5)](https://arxiv.org/abs/cond-mat/0408686).
The seeded solver uses fixed normal orbitals and phonons. Its scope is static
band-diagonal pairing on the supplied normal state.

Required controls are `band_energies`, `band_weights`, `electronic_temperature`
(Kelvin), `electron_number`, `chemical_potential` (initial guess), and
`initial_gap_real`. `initial_gap_imaginary` defaults to zero. Energies, gaps
and kernels are Hartree in native Rust/Python. Weights and Z are dimensionless;
weights/kT are positive and `1+Z>0`. The particle number lies strictly between
zero and twice the sum of band weights. A supplied real K must already be
reciprocal; no triangle or spin/phase value is copied by symmetry.

`pairing_model="supplied"` (default) requires `pairing_kernel`, shape n by n.
`pairing_model="lm2005-off-diagonal"` instead requires `phonon_energies`,
`pair_couplings_squared` (one flattened n*n row per mode, including mode/q
weights, Hartree squared) and `screened_pair_coulomb` (n by n, Hartree).
Both modes require `mass_renormalization` (one diagonal Z per band). Their
kernel is not frozen at the input mu. `pairing_model="lm2005"` takes the same
phonon/vertex/Coulomb inputs and generates both K and phonon Z at every
responding mu and temperature, including the same band-weight paths. Explicit
mass inputs and directions are rejected in this mode. The mass approximation
and remaining normal-state/causal scope are stated below and in each report.

The seed selects a nonlinear branch and a global phase. A linear phase
constraint relative to the complex seed is included with its Lagrange
multiplier. A zero seed selects the normal stationary solution, with an
explicit warning that its thermodynamic stability has not been established.
At a zero seed, the first channel supplies the phase-coordinate convention.
No artificial nonzero gap is inserted. The same constrained Jacobian is used
for all Taylor orders. Its products are exact forward AD; the common GMRES
solver checks fresh residuals of the original bordered equations. The raw
unbordered physical gap equations, number, phase, multiplier and Goldstone
mode are independently reevaluated. A projected residual alone cannot pass.
Iteration mu, gap and residuals are logged by the common CLI logger.

`response_order=0..4` returns factorial-normalized coefficients along
linear input directions in dimensionless t. Optional directions are
`band_energy_direction`, `band_weight_direction`,
`mass_renormalization_direction`, `temperature_direction` (Kelvin),
`electron_number_direction`, `pairing_kernel_direction` (n by n),
`phonon_energy_direction`, `pair_couplings_squared_direction` (m by n*n),
and `screened_pair_coulomb_direction` (n by n), as applicable to the selected
physical model. Every direction has the units of its input. Common
`response_tolerance`, `response_max_iterations`, `response_restart` and other
response solver controls apply; defaults remain 1e-10, 500 and 40.

Reports contain `gap_real/imaginary`, `gap_coefficients_real/imaginary`,
`chemical_potential` and `chemical_potential_coefficients`,
`anomalous_density_coefficients_real/imaginary`, `occupation_coefficients`,
`quasiparticle_energies`, `quasiparticle_energy_squared_coefficients`, and
`pairing_kernel_coefficients`, `mass_renormalization` and
`mass_renormalization_coefficients` (dimensionless). Coefficient arrays are [order][band] and
kernel arrays [order][initial band][final band]. The anomalous density and
spin-summed occupation are dimensionless before quadrature weights; gap,
mu and K are Hartree, squared energies Hartree squared. The E-squared path
remains analytic at normal-state Fermi-surface zeros, where individual
quasiparticle eigenvalue derivatives need not exist. The thermal factor is
evaluated analytically in E-squared there, with no denominator floor.

`gap_residuals`, `phase_residuals`, `phase_multipliers` and
`goldstone_defects` are Hartree at every order; `number_residuals`,
`reference_scaled_residual` and `linear_residuals` are dimensionless.
`linear_iterations` and `reference_iterations` count actual solves.
`energy_scale` (Hartree) records the fixed residual-scaling convention;
`phase_reference` is a dimensionless complex vector, [band][real,imaginary].
Raw physical residuals are retained alongside scaled solver diagnostics.
`converged` establishes stationarity of this specified model, not stability
or completion of the full SCDFT objective. Native, CLI and ASE run one driver.
The ASE static getter requires no Atoms, converts eV/eV-squared controls and
reports by the same unit labels, and accepts `units="atomic"`.

The Rust provider contract is `GapFunctional::evaluate<S:Scalar+Send+Sync>`
over parameters, the responding mu and complex gap. It returns
`GapFunctionalData<S>` containing energies, weights, kT, diagonal Z and K;
all parameter/mu/gap dependence must retain S and global U(1) covariance.
`solve_gap_taylor<const W>` accepts this provider, parameter `Taylor<W>`
paths, the electron-number path, a `GapState` branch seed, order and the common
`ResponseSolverConfig`. `SuppliedBandFunctional` packs parameters as
energies[n], weights[n], Z[n], kT[1], K[n*n]. `PhononBandFunctional` packs
energies[n], weights[n], Z[n], kT[1], phonon energies[m], squared vertices[m*n*n],
screened Coulomb[n*n], with explicit band/mode dimensions and quadrature.
`Lm2005BandFunctional` uses energies[n], weights[n], kT[1], phonon energies[m],
squared vertices[m*n*n], screened Coulomb[n*n], omitting the supplied Z slots.
Custom providers can retain their own parameter layout and derivatives;
no response equation branches on an electronic functional's registered name.
`anomalous_factor(E_squared,kT)` exposes the shared analytic thermal factor.
#### Canonical SCDFT critical temperature

Supply `scdft_temperature_bracket = [cold, hot]` in Kelvin to select the
normal-state linear pairing onset through `superconducting-state`.
The Rust low-level entry is `response::superconducting::critical_temperature`
with bracket energies `k_B T` in Hartree. The common driver uses the same
`pairing_model`, band weights and K/Z inputs as the canonical gap solve.
Omit `electronic_temperature`, `chemical_potential` and the initial gap.
The normal occupations determine the chemical potential at each trial.
The critical-temperature calculation takes fixed inputs and supports
`response_order=0`.

With `Delta=0`, the thermal factor is
`q_i = tanh((epsilon_i-mu)/(2*k_B*T))/(2*(epsilon_i-mu))` and its limit
at resonance is `1/(4*k_B*T)`. Positive `w_i q_i` and `1+Z_i` make
the real reciprocal pairing map similar to the symmetric matrix
`-D K D`, with `D_i=sqrt(w_i q_i/(1+Z_i))`. Search the oriented
bracket where its leading eigenvalue decreases through one. The band gaps'
linearized equation follows the static singlet model in
[Lüders et al., Eqs. (33) and (35)](https://arxiv.org/abs/cond-mat/0408685).
The returned `linearized_gap_mode` has unit Euclidean norm in the original
gap coordinates and arbitrary physical amplitude. When several crossings
exist, the input bracket determines the selected one.

For a half-filled band with weight 1, `Z=0.2` and `K=-0.012 Hartree`,
the closed-form result is `k_B T_c=0.0025 Hartree`.

```console
dft-rs superconducting-state --pairing-model supplied --band-energies '[-0.003]' --band-weights '[1.0]' --mass-renormalization '[0.2]' --pairing-kernel '[[-0.012]]' --electron-number 1 --scdft-temperature-bracket '[400,1200]' --format json
```

The report includes `critical_temperature_k` and its final bracket in
Kelvin, the normal-state chemical potential in Hartree and spin-summed
occupations before band weighting. The leading eigenvalue and raw
linearized-gap, eigenvector and particle-number residuals are dimensionless.
`kernel_asymmetry` reports the largest absolute reciprocity defect of K in
Hartree. The driver refuses intervals without the required cold and hot
endpoint signs. The normal spectrum is fixed throughout the bracket.
The pairing onset and the normal-state electronic thermal response
address different perturbations. A thermodynamic phase comparison requires
the free energies of the competing branches.

For a fixed-band reciprocal `pairing_model="supplied"` with uniform Z,
set `scdft_phase_selection=true` at one positive `electronic_temperature`.
The solver determines the canonical normal state, uses the leading unstable
mode to seed pairing and tests the selected phase against a variational
free-energy bound. Supply the same band energies, weights, electron number,
kernel and Z convention as for the seeded solver. Omit the initial gap,
chemical-potential guess, temperature bracket and response directions.
`response::superconducting::select_band_phase` takes the supplied model
parameters with thermal energy in Hartree. The common driver uses Kelvin.

```sh
dft-rs superconducting-state --scdft-phase-selection true --band-energies '[0.031]' --band-weights '[0.73]' --mass-renormalization '[0.2]' --pairing-kernel '[[-0.0387567445370204]]' --electron-number 0.73 --electronic-temperature 1263.100099216 --format json
```

The result includes `selected_phase`, `condensation_free_energy` as
`F(selected) - F(normal)`, `pairing_energy`, the dimensionless
`pairing_stability_margin`, both entropies divided by Boltzmann's constant,
and the raw gap, phase and particle-number residuals. Energies and chemical
potentials use Hartree in Rust, CLI and native Python and eV in ASE's
default units. Positive and negative margins give the sign of the
minimum variational bound. The paired state's global phase gives a zero
mode. The solver reports an error if the normal state is unstable and no
paired solution satisfies both the bound and the free-energy check.
[Model and acceptance](response.md#r050-scdft-gap-anomalous-density)

The `electron-phonon` DFPT adapter's `scdft_inputs` supplies the band
energies, weights and phonon/Coulomb inputs to this model. Use the adapter's
`scdft_electron_number` for `electron_number` and replace the single
temperature and chemical potential with `scdft_temperature_bracket`.
The adapter's k/q sampling restrictions still apply.

<!-- canonical-superconducting-state:end -->

### Generated diagonal phonon mass and its response

`scdft_pairing_kernel(..., band_weights=[...])` and the corresponding CLI/ASE
entry generate dimensionless `mass_renormalization` Z in addition to K.
Weights must be finite, positive and have one entry per normal band. A mode
vertex includes its mode/q weight; Z additionally includes one final-band
quadrature weight. The returned K does not include band weights. Supplied
screened Coulomb contributes only to K, with no implicit Coulomb mass.

The static particle-hole-symmetric phonon mass follows the overall minus-J
convention in [PRB 88, 014514, Eq. (10)](https://arxiv.org/abs/1305.0390).
The confluent Fermi derivative product is consistent with Eq. (44) of that
derivation. This is the LM2005 symmetric mass approximation; no
particle-hole-asymmetric kernel improvement is inferred.

The implementation rewrites the mass using `f(x)=1/(1+exp(x/kT))`,
`D(b,y)=f[b,b,y]=(f'(b)-(f(b)-f(y))/(b-y))/(b-y)` and
`M(x,y)=(n_B(Omega)+f(x))*D(x-Omega,y)
+(n_B(Omega)+f(-x))*D(x+Omega,y)`:
`Z_i=sum_j,nu w_j*abs(g_ij,nu)^2*(M(xi_i,xi_j)+M(xi_i,-xi_j))
/tanh(xi_i/(2*kT))`.
For coincident arguments, `D(b,y)=integral_0^1 t*f''(y+t*(b-y)) dt`.
For `xi_i=0`, the odd numerator is divided analytically using its integrated
x derivative and `x/tanh(x/(2*kT))`. These identities preserve every
temperature, energy, mu, vertex and weight derivative, including paths that
cross the Fermi surface. No denominator is shifted or clipped.

The component accepts optional `band_energy_direction`,
`chemical_potential_direction`, `temperature_direction` (Kelvin),
`phonon_energy_direction`, `pair_couplings_squared_direction`,
`screened_pair_coulomb_direction` and `band_weight_direction`. Directions
have the units and shapes of their values and define a dimensionless linear
parameter t. `mass_renormalization_coefficients[order][band]` are
factorial-normalized; multiply order n by n! for the derivative.
`mass_model` states the convention. `mass_quadrature_absolute_errors` and
`mass_quadrature_relative_errors` compare orders 16 and 32 separately from
the Hartree-valued K errors. Both mass diagnostics are dimensionless and
every requested relative error must be at most 1e-10 for `converged=true`.

`properties::pairing::Lm2005::{mass_factor,mass}` supplies the scalar-generic
Rust implementation, shared by both public calculations. The factor is
Hartree^-2; `mass(&PairingInput<S>, &band_weights)` returns `Vec<S>` Z.
`Lm2005OffDiagonal` remains an alias for compatibility. Its existing
`convolution`, `phonon_factor`, `evaluate` and quadrature constructors keep
their signatures. `Lm2005BandFunctional` connects generated K/Z to the same
gap/number/global-phase AD Jacobian, so chemical-potential response is
included in both. `GapTaylor::mass_renormalization_coefficients` retains Z
for supplied and generated models alike.

These calculations operate on supplied normal bands and phonons, with
explicit optional parameter paths. Normal-density/phonon self-consistency,
the DFPT-to-pairing vertex adapter, general interband pairing and causal
external-frequency response remain separate pending work. No electronic
functional or reference-class coverage is inferred solely from this
provider-independent band component. Acceptance measurements are recorded
in [response.md](response.md).

### Noncanonical orbital optimization curvature

`scf::soscf::OrbitalHessian::fock_commutator(kappa, &[F_alpha,F_beta])`
returns the fixed-potential orbital-Hessian action in the existing packed
virtual/occupied rotation order, with the same R/U normalization as
`gradient` and `project`. It uses `F_vv*kappa-kappa*F_oo` at the current
orbitals. Stored `eps` values and the positive floor in `diagonal()` are
preconditioners only; they do not replace this operator after an orbital
rotation. The density-dependent action is added by the caller from the
shared cached Coulomb/exchange kernel.

The ordinary SCF uses this energy-tested optimizer for fixed-tensor
quadratic Coulomb/exchange models. Once a variational step is accepted it
retains that mode while escaping a saddle; a temporarily larger gradient
does not trigger an uphill extrapolation. Negative-curvature CG termination
can supply a preconditioned downhill gradient trial, which still requires
the actual line-search checks. Other density terms keep the ordinary
SCF/DIIS path until their complete optimization kernels are connected.
This optional optimizer does not restrict the full generic response kernel.
Occupied and virtual blocks are recanonicalized at an integer stationary
reference without changing its density. The energy/gradient/density
acceptance thresholds remain unchanged.

`OrbitalHessian::finite_density_change(kappa)` evaluates the finite density
change on the unitary orbital path as nested commutators of the occupation
matrix. At the optimizer's trust-region steps it retains quadratic diagonal
population changes even when subtracting two floating-point projectors would
erase them. The SCF uses this analytic path difference for quadratic energy
work and refreshes the preconditioner's orbital energies. Independent closed
sine/cosine identities test rotations down to 1e-11 radians. This is an
algebraic energy-difference evaluation, not a finite-difference derivative,
energy floor, or relaxed convergence condition.


### Explicit thermodynamic potential names

Use `mermin_free_energy` / `dft-rs mermin-free-energy` for the molecular
electronic Mermin potential `A=E-T_e*S_e`. `electronic_temperature` is Kelvin,
`entropy` is the electronic entropy divided by k_B, and `kt` is k_B*T_e in
Hartree. The RKS/UKS gradient differentiates A at fixed electronic
temperature and electron counts per spin. For `reference="gks"` or
`"ghf"`, the scalar report uses one Fermi level and preserves the
total electron count. Spinor occupation-response derivatives and
Gibbs thermochemistry have separate acceptance gates. Nuclear translation,
rotation, vibration and pV belong to the separate RRHO model.

Use `gibbs_free_energy` / `dft-rs gibbs-free-energy` for the molecular ideal-gas
RRHO Gibbs potential `G=H-T*S`. `temperature` is Kelvin, `pressure` is Pa and
`symmetry_number` is the rotational symmetry number. The existing analytic
Hessian and thermochemistry calculation is reused. A positive electronic
temperature uses the full occupation/chemical-potential/metric response of
the Mermin surface. For equilibrium Gibbs, nuclear and positive electronic
temperatures must agree. The default nuclear temperature is the positive
electronic temperature, otherwise 298.15 K. The report includes zero-point and
thermal vibrations, rigid rotations, translations, electronic-degeneracy
entropy and `pV=k_B*T` per molecule. No geometry relaxation is performed;
the supplied structure must represent the desired minimum. An imaginary
vibrational mode makes the Gibbs request fail, while the frequencies route
continues to report the saddle and withholds thermochemistry.

The primary result keys are `mermin_free_energy` and `gibbs_free_energy`.
`free_energy_kind` states the potential and approximation. Existing
`free_energy(...)`, `get_free_energy_report(...)`, `free-energy` and the
molecular `free_energy` result key remain explicit Mermin compatibility
aliases. ASE's standard `results["free_energy"]` remains the electronic
Mermin potential required for force-consistent electronic energies; it is
not a Gibbs thermochemistry calculation. Native reports use Hartree and
Bohr. ASE report getters use eV and Angstrom unless `units="atomic"`.

Periodic electronic SCF reports expose `mermin_free_energy` in addition to
the historical `free_energy`. `reference_mermin_free_energy` and
`delta_mermin_free_energy` disambiguate electronic reference and thermal
response reports. The common-driver `phonon-thermo` and `quasi-harmonic`
reports instead expose `helmholtz_free_energy` for their lattice Helmholtz
potential, preserving its original per-cell units. `quasi-harmonic` also
exposes `gibbs_free_energy=F+pV`, retaining `gibbs_energy` as an alias.

### Explicit nonlinear preset arguments and ASE units

Both `dynamic_beta` and `dynamic_gamma` expose `process`, `frequencies`,
`damping` and `input_frequencies` as named keyword arguments. The dedicated
ASE getters accept the same alternatives. All seven gamma presets in the
nonlinear table use the common cubic-response equations; no duplicate
property-specific solver or tensor symmetrization is introduced. An explicit
leg list cannot be combined with a preset, and beta/gamma reject a preset
of the wrong order. Negative optical legs keep positive incoming damping;
static DC legs have zero frequency and damping.

Native Python and CLI frequencies/damping are Hartree. ASE uses eV unless
`units="atomic"` is selected. Its aliases delegate to one conversion
boundary, for both explicit legs and presets. The gamma tensor converts as
`e^4*bohr^4/hartree^3`; the ordered output is a molecular cubic dipole
susceptibility. The absorptive TPA response and Kerr/EFISH tensors still
require the physical polarization, density, refractive-index and optical
intensity factors appropriate to any requested macroscopic coefficient.

```python
mermin = dft_rs.mermin_free_energy(numbers, positions, electronic_temperature=6000)
gibbs = dft_rs.gibbs_free_energy(numbers, positions, temperature=298.15, pressure=101325)
thg = dft_rs.dynamic_gamma(numbers, positions, process="thg", frequencies=[0.05], damping=0.005)
fwm = calc.get_dynamic_gamma(atoms, process="four-wave-mixing",
                             frequencies=[0.05, 0.07, 0.09], damping=0.005,
                             units="atomic")
```


### Gibbs thermochemistry from the Mermin entry

The connected electronic providers use ground-state XC with Fermi-Dirac
occupation entropy; an additional explicitly temperature-dependent XC
free-energy model is not implied. Double-hybrid finite-temperature
correlation and its Gibbs/Hessian combinations remain pending in the CSV.

`mermin_free_energy(..., gibbs=True)` and `dft-rs mermin-free-energy --gibbs`
return both named potentials using one converged electronic reference. The
default Mermin call remains an electronic calculation without a Hessian.
The Gibbs option adds the analytic Cartesian Hessian of A_e at fixed T_e,
Eckart projection, harmonic modes and the shared ideal-gas RRHO calculation.
The Hessian uses orbital, occupation, chemical-potential and energy-weighted
density response, including both Pulay metric terms; no finite differences
or frozen-occupation substitution are used. The dedicated Gibbs entry and
the frequencies entry use these same implementations.

For positive T_e, equilibrium requires `temperature=electronic_temperature`.
Omitting `temperature` selects T_e; for T_e=0 it selects 298.15 K and uses
the ground-electronic-state RRHO approximation. Unequal positive electronic
and nuclear temperatures are rejected with the compatibility condition.
`pressure` is positive Pa and `symmetry_number` a positive integer. The
spin multiplicity counts additional degenerate electronic states outside
the fixed-spin-channel Fermi ensemble, as on the existing frequencies route.

The electronic report retains `internal_energy=E_e`, `entropy=S_e/k_B` and
`mermin_free_energy=A_e=E_e-T_e*S_e`. In `thermochemistry`, `internal_energy`
and `enthalpy` include nuclear RRHO contributions, and `entropy` includes
electronic, vibrational, rotational and translational entropy in Hartree/K.
At equilibrium `gibbs_free_energy=H-T*S_total=A_e+G_RRHO`; the electronic
entropy is counted once. The nested report separates every energy and
entropy term. `gradient=True` still requests the gradient of A_e, not the
geometry derivative of RRHO Gibbs. No geometry optimization or anharmonic
thermochemistry is implied. At an imaginary mode the Gibbs request fails.

Raw `ensemble_response_residuals` are the original GMRES relative equation
residuals for every Cartesian source. `ensemble_number_defects` are measured
`Tr(S*dP)+Tr(dS*P)` per spin, electrons/Bohr. These arrays are empty for the
zero-temperature occupied/virtual solver. `raw_asymmetry` is recorded before
Hessian symmetrization. A vanishing Fermi-surface weight permits fixed-number
response with an undetermined in-gap chemical-potential gauge; number-changing
response still requires a resolved Fermi-surface weight. Functional classes
reuse their existing complete kernel; no registered-name branch is used.

```sh
dft-rs mermin-free-energy molecule.xyz --xc pbe --gibbs --temperature 298.15 --pressure 101325
dft-rs mermin-free-energy molecule.xyz --xc pbe --electronic-temperature 6000 --gibbs
```

```python
report = dft_rs.mermin_free_energy(numbers, positions, xc="pbe",
                                  electronic_temperature=6000, gibbs=True)
electronic_A = report.mermin_free_energy
equilibrium_G = report.gibbs_free_energy
ase_report = calc.get_mermin_free_energy(atoms, gibbs=True, pressure=101325)
```

<!-- connected-response-api:begin -->

### Restricted magnetic balance and a complex four-component metric

The existing `no-pair-projector` entry accepts `dirac_integrals="rmb"` with
the same geometry, basis, explicit Gaussian nuclear RMS radii and speed of
light as RKB. Optional `magnetic_field` and `magnetic_direction` are atomic
magnetic units; `gauge_origin` is Bohr. B(t)=B0+t*dB, with dimensionless t.
Positions, radii and c may simultaneously follow their existing directions.
No isotope radius is guessed. Native Rust/Python use Hartree/Bohr; the ASE
getter accepts Angstrom/eV, or `units="atomic"` except for Atoms geometry.

The small basis is chi_S=(sigma.pi)*chi_L/(2c), pi=p+A,
A=B cross (r-O)/2. Restricted magnetic balance is the finite-field
nonrelativistic balance condition described in this
[primary derivation](https://par.nsf.gov/servlets/purl/10176394).
The shared integral builder forms T_B=(sigma.pi)^2/2 and
W_B=(sigma.pi)*V*(sigma.pi). Its rest-energy-shifted blocks are
H=[V,T_B;T_B,W_B/(4c^2)-T_B], S=diag(S_large,T_B/(2c^2)). Both real and
imaginary metric parts are retained. Orbital, spin Zeeman, diamagnetic,
spin-orbit and small-basis derivative terms follow the same scalar path.

Polynomial nuclear-attraction moments share the existing Gaussian angular
recurrence. Inner endpoint duals differentiate the basis functions while
holding the vector potential fixed; outer AD retains field, origin, basis
and finite-nucleus dependencies. Shell pairs and spin entries are computed
independently. The existing metric/no-pair subspace transport differentiates
the full complex H/S through order four. Degenerate spin states remain a
subspace. No denominator, negative-energy gap or eigenvalue is shifted.

Additional results are `balance`, `dirac_overlap_coefficients_imaginary`
(dimensionless), `balanced_kinetic_coefficients_real/imaginary` (Hartree).
All existing projector, projected Hamiltonian, metric, Hermiticity and
commutator diagnostics retain their conventions. This common-origin basis
has no London phases; gauge-origin changes alone can change finite-basis
values. Joint translation of basis, nuclei and origin is covariant. This
component supplies magnetic balance, not interacting Dirac-Coulomb SCF or
relativistic exchange-correlation/observable completion.

With `dirac_integrals="london-rmb"`, each large and small Gaussian includes
its London orbital phase. The complex H/S coefficients take the same
`no_pair_projector_taylor` route. The global `gauge_origin` cancels between
the endpoint phases and the vector potential. Use `magnetic_phase_order`
through eight to converge a finite base field. At zero field the default
phase order is `response_order` and the requested magnetic coefficients
are exact in that order. The model reports the phase order and the
imaginary large nuclear-attraction matrix in Hartree. The C1 ammonia
STO-3G check gives a largest metric defect of `2.220e-16` and largest
relative commutator defect of `3.924e-17`. [Validation](response.md#r049-no-pair-dirac-coulomb)

### Joint electric and magnetic fields

`magnetizability` accepts matching `electric_directions` and
`magnetic_directions`, each n by 3 for n=1..4. A direction may contain both
fields; E=sum_j t_j e_j, B=sum_j t_j b_j. The existing
`magnetic_mixed_derivative` is the actual joint energy derivative with
respect to the dimensionless parameters. Directional coefficients and the
analytic polarization identity use one six-dimensional source. Repeated
directions share the existing binomial contractions. The raw response,
projector, stationarity, reference and accumulation-roundoff fields retain
their meanings. Zero-field collinear integer occupations and the declared
orbital/current-functional restrictions of the field path still apply.

`bfield` and `bfield-gradient` additionally accept the existing molecular
`field` electric control. The London one-electron scalar potential, nuclear
electric energy and analytic nuclear derivative are all included. The
internal recentering correction Q*E.dot(shift) restores the original
laboratory energy of a charged system. `electric_field` records E;
`gradient_translation_target=-Q*E`, not zero, for charge Q. The translation
residual measures sum_atom gradient minus this target. Electric inputs are
Hartree/e/Bohr natively, eV/e/Angstrom in the ASE report getters. Energies and
gradients follow the same usual unit conversion; no finite fields are used
to produce analytic response or forces.

### Long-wave internal relaxation and the Born lattice contribution

`phonon-bands` with `lattice_flexoelectric=true` adds the constrained
internal-displacement solution of its actual real-space IFC model. A neutral
cell, `wavevector_direction` and order 2..4 are required. Its normalized
Cartesian direction n defines q=kappa*n, with kappa in inverse Bohr, separate
from the existing dimensionless q-path coefficient convention. The atomic
phase includes R+tau_final-tau_initial. In the mass-weighted space, the three
excluded translations are sqrt(M_atom)*e_axis. Internal coefficients obey
sum_atom M_atom*u_atom,n=0 at positive order. The common constrained GMRES
solver reevaluates the original projected equations and acoustic reactions;
it does not modify IFCs, restore a sum rule or floor an optical denominator.

`long_wave_relaxation` is a list of objects, each with integer `order`,
`displacement_real/imaginary` [3N][3] in Bohr^order, `polarization_real/imaginary`
[polarization][acoustic displacement] in e*Bohr^(order-3),
`relative_residuals`, `center_of_mass_defects` (Bohr^order), and total Cartesian
`acoustic_reaction_real/imaginary` in Hartree*Bohr^(order-2). Displacements
are per imposed macroscopic displacement. `long_wave_uniform_force_defect`
is the raw IFC force under uniform displacement, Hartree/Bohr^2.
`long_wave_converged` checks that raw defect against the stated response
tolerance; original equation residuals have their separate solver gates.

Born charges come from the primitive cell's existing insulating DFPT provider,
without recomputing unrelated force constants. Alternatively,
`born_charge_rows` supplies 3N rows of three displacement-axis charges,
ordered atom then polarization. `long_wave_born_charges` retains the values.
The Born contribution is P_n=sum_atom Z*_atom*u_atom,n/volume.
`born_lattice_flexoelectric=-P_2` is its type-I coefficient contracted with
n twice: P_a=mu_(a,beta,gamma,delta)*partial_gamma partial_delta u_beta,
with units e/Bohr. It is a contribution, not the complete flexoelectric
tensor. The long-wave decomposition is discussed in the
[primary lattice-response derivation](https://arxiv.org/abs/2112.10721).
Clamped-ion electronic polarization, dynamic multipole/metric/current
terms, surface conventions and convergence of the finite-supercell IFC
range remain necessary for the full objective.

### DFPT mode vertices connected to static superconducting kernels

The existing `electron-phonon` driver now retains the complex mode vertices
g_mn^nu=sum_x e_nu(x)/sqrt(2*M_x*Omega_nu)
*[F^x_mn-(epsilon_m+epsilon_n)*S^x_mn/2], including the self-consistent
Fock response and both endpoint Pulay energies. No Fermi-surface Gaussian
filter removes a vertex; those weights apply only to the separate lambda
and a2F contractions. Nonpositive/below-cutoff modes are excluded explicitly
without frequency shifts.

`scdft_pairing=true` requires positive `electronic_temperature` and explicit
`screened_pair_coulomb` on flattened k/bands. `scdft_inputs` contains
`band_energies`, `band_weights`, `chemical_potential`, `electronic_temperature`,
`phonon_energies`, `pair_couplings_squared` and `screened_pair_coulomb`, with
the usual pairing units. It can be passed directly to
`scdft_pairing_kernel(**report.scdft_inputs)`. `scdft_pairing_kernel`,
`scdft_phonon_pairing_kernel` (Hartree), `scdft_mass_renormalization`
(dimensionless), `scdft_selected_modes`, `scdft_spin_defect` (Hartree) and
`scdft_electron_number` are returned. Each final band receives weight 1/Nk
once. The squared physical Gamma-mode vertex receives no extra mesh weight.

Nk=1 covers the commensurate supercell model. At Nk>1, only transitions
within a supercell k block are present: this is the Gamma-node contribution
with its mesh weight, not a renormalized estimate of unavailable q points.
Spin-degenerate matching frames are checked rather than imposed; general
interband singlet/triplet pairing and independent spin-frame transport are
not inferred. The same LM2005 K/Z provider handles this input. To solve the
canonical gap, pass these inputs to `superconducting_state` with
`pairing_model="lm2005"`, explicit `electron_number` and a chosen complex gap
seed. A zero seed tests the normal stationary branch, not superconductivity.
Normal-state/vertex parameter response, screened-Coulomb generation and
arbitrary-q adapters remain necessary parts of the full SCDFT objective.
<!-- connected-response-api:end -->

<!-- nambu-density-api:begin -->
### Full Nambu density on a supplied electronic space

The existing `superconducting-state` entry accepts `pairing_model="nambu"`
through Rust's common driver, `dft-rs superconducting-state`, native
`superconducting_state(**inputs)` and ASE
`DftRs.get_superconducting_state(**inputs)`. No atoms are required for
supplied operators. This selector computes a grand-canonical matrix density;
it does not replace the canonical self-consistent band gap providers.

Required square matrices are `nambu_normal_real` and `nambu_pairing_real`,
in Hartree. Optional matching `_imaginary` matrices default to zero.
`nambu_overlap_real/imaginary` defaults to the identity metric. Each prefix
also accepts `_direction_real/imaginary`, defining a linear path in the
same dimensionless parameter t. `chemical_potential` (Hartree), optional
`chemical_potential_direction`, positive `electronic_temperature` (Kelvin),
optional `temperature_direction`, `response_order` (0..4),
`metric_eigenvalue_cutoff` and standard response convergence controls apply.
ASE uses eV for all Hamiltonian/pairing/mu inputs and their directions by
default; `units="atomic"` uses Hartree. Overlap and its directions are
dimensionless, and temperature remains Kelvin.

The full electronic indices already include spin. Pairing obeys
Delta=-Delta^T, and normal h/S are Hermitian without projection.
Nambu order is all electrons then their conjugate holes:
H_N=[[h-mu*S,Delta],[Delta^H,-(h-mu*S)*]], S_N=diag(S,S*).
The same moving-metric matrix Fermi function retains every interband
normal and anomalous component. It uses analytic coefficient products,
scaled matrix exponentiation and bounded logistic doubling; eigenvector
derivatives and physical-gap denominator floors are unnecessary at
degeneracy. This follows the normal/anomalous density formulation of the
[primary finite-temperature superconducting formalism](https://arxiv.org/abs/1903.01516).

`nambu_density_coefficients_real/imaginary` has shape [order][2n][2n].
`normal_density_coefficients_real/imaginary` and
`anomalous_density_coefficients_real/imaginary` have shape [order][n][n].
They are dimensionless contravariant coefficients, with rho=R_ee and
chi=-R_eh=<c_i c_j>. `electron_number_coefficients` is Tr(S*rho), in
electrons. `nambu_hamiltonian_coefficients_real/imaginary` is Hartree;
`nambu_overlap_coefficients_real/imaginary` is dimensionless. Every series
is factorial normalized. Raw dimensionless diagnostics include
`particle_hole_defects`, `pairing_antisymmetry_defects`,
`number_imaginary_defects`, `metric_defects`,
`density_hermiticity_defects`, `relative_commutator_defects`,
`complex_embedding_defects`, `maximum_inverse_residual` and
`exponential_tail_estimate`. Counts `exponential_terms` and
`doubling_steps` describe the matrix-function work.

Particle/hole covariance is R+tau_x*R.conjugate()*tau_x=S_N.inverse().
The full positive overlap is retained. A Dirac provider must explicitly
supply its intended balanced no-pair electronic space; this density
evaluation does not create it. Self-consistent normal/pairing potentials,
general canonical gap/phase constraints, screened-interaction generation,
external-frequency response and backreaction remain separate required
extensions. This model's acceptance is recorded separately from the
existing band-decoupled gap acceptance in response.md.
<!-- nambu-density-api:end -->

<!-- external-field-nuclear-scope:begin -->
### Nuclear derivatives about an externally biased reference

Field derivative order and nuclear-coordinate derivative order are distinct.
Magnetic energy derivatives and Cotton-Mouton response do not by themselves
establish nuclear force constants in a magnetic field.

| Reference | Nuclear gradient | Nuclear Hessian | Cubic/quartic nuclear derivatives |
|---|---|---|---|
| Molecular uniform electric field | Connected | Connected | Connected on supported integer-occupation nuclear Taylor paths |
| Molecular finite orbital magnetic field | Connected through bfield-gradient; HF directional orders 1-4 accepted | Common analytic driver connected | Common analytic contracted tensors connected; HF B-only fits accepted |
| Molecular simultaneous electric and magnetic fields | Restricted H2 and charged H2+ Zeeman fits accepted; PBE0 water analytic-path fits passed, public tensors separate | Neutral restricted and charged unrestricted HF tensors accepted on all public surfaces | The same two HF references accepted on all public surfaces; other classes remain per-entry |
| Periodic finite Berry electric field | Connected on the insulating Gamma/Born-von Karman enthalpy model | HF H2 independent directional fits and public tensor comparisons passed | HF H2 directional fits and public contracted tensors passed; other classes/combinations require acceptance |
| Periodic 3D finite-Gamma point-source magnetic field | Connected on the full spinor exchange/SOC model | HF tensor public comparisons passed | HF/custom range-exchange/SOC order-1..4 fits passed; HF contracted tensor public comparisons passed |
| Periodic 3D finite-Gamma point-source simultaneous electric/magnetic field | Seven crystal metrics and model/limit cases passed order-1..4 fits | Both CLI names/native Python/ASE tensor checks passed | All five report forms passed on the same finite-Gamma subset |
| Periodic magnetic BZ or 1D/2D sources | Nuclear connections remain required | Required | Required |
| Arbitrary-q DFPT on finite electric/magnetic references | Per-q field-biased connection required | Required | Required |

Nuclear paths include the field's moving-basis, metric/Pulay and source
terms on each supported model. Positive-temperature Mermin Hessians have
separate occupation-response acceptance. Third/fourth nuclear paths keep
their stated integer-occupation restrictions. ECP, continuum, nonlocal,
learned, spinor and correlation support follows each provider's actual
restrictions. The complete double-hybrid amplitude/orbital nuclear response
and remaining field/reference/functional combinations stay within scope.
<!-- external-field-nuclear-scope:end -->

<!-- biased-nuclear-api:begin -->
### Nuclear response of a finite-bias periodic electric enthalpy

`phonons`, `periodic-cubic-fc`, `periodic-quartic-fc`,
`periodic-cubic-fc-along` and `periodic-quartic-fc-along` accept the existing
`field` control (Hartree/e/Bohr in Rust/native/CLI; eV/Angstrom in the
default ASE convention). All routes use the shared nuclear path engine.
The field is fixed in Cartesian coordinates while the atomic positions,
Gaussian basis, overlap metric and Berry coupling are differentiated.
`series_reference_energy` and `field_energy`, when present, describe the
stationary biased electric enthalpy per primitive cell. The legacy
`energy` field still describes the zero-field starting supercell.

Rust supplies `lattice_path_energy_at_field`,
`lattice_force_constants_along_at_field` and
`lattice_force_constant_tensor_at_field`. The tensor entry accepts nuclear
orders two through four; the existing zero-field tensor retains its
three/four-order API. The wrappers share one implementation and do not
duplicate a physical kernel. Integer-occupation insulating three-dimensional
Gamma/Born-von Karman boundary assumptions and existing functional-provider
restrictions apply. No functional registered-name branch is introduced.

The finite-field harmonic driver reports freshly recomputed response
residuals, reference residual, force constants (Hartree/Bohr^2 per primitive
cell), modes and frequencies (cm^-1). Intrinsic tensor permutation is used
for production. Independent ordered asymmetry has not been evaluated on
this path and is not fabricated as a zero residual. Cubic and quartic
force constants have Hartree/Bohr^3 and Hartree/Bohr^4 per primitive cell.
The full tensor driver also reports every raw response residual.

Independent acceptance uses four displaced-cell enthalpy/force steps
against production analytic coefficients through nuclear order four.
Native/ASE atomic and physical units and both CLI variants are checked
separately. Measured acceptance is recorded only after those tests pass.
This connection does not establish finite magnetic/joint-field nuclear
response, arbitrary-q biased nuclear DFPT or correlated PT2 nuclear
response; those remain required by the full objective.
<!-- biased-nuclear-api:end -->

<!-- scalar-london-api:begin -->
### Scalar finite-field London integral kernels

`integrals::london::ScalarFieldPath<S>` carries electric and magnetic fields,
gauge origin, London/common-origin choice, `phase_order` (total retained field
degree, zero through 24), and `derivative_order` (the total scalar nilpotent
order, including all nested directions). These two orders are independent:
the latter is required by the analytic Boys expansion. Coordinates and fields
use Bohr and atomic field units, exponents use Bohr^-2, and charges use e.

`primitive_field_pair(bra,ket,nuclei,field,exact_zero)` returns five complex
scalar pairs: overlap, core Hamiltonian, then x/y/z position integrals.
The primitives are unnormalized and keep their supplied angular powers,
exponents and centers. Core includes kinetic energy, point nuclear attraction,
and electric, orbital-paramagnetic and diamagnetic couplings. Nuclei are
`(center,charge)` scalar tuples. `primitive_field_repulsion(primitives,kernel,
field,exact_zero)` returns one complex Coulomb or range-separated quartet.
All four endpoints, the nuclear positions/charges, exponents and fields retain
their complete scalar jets. No same-q Hermitian projection is imposed.

`exact_zero` is a scalar-algebra callback that may report true only when
every coefficient is zero. For `dual::Taylor<N>`, check all `c` entries;
checking only `value()` would erase moving-center derivatives at zero phase.
Returning false conservatively is algebraically safe but increases work.
Finite-field phase truncation must be converged independently of derivative
order. The existing finite-field one-electron gradient contractions and
contracted shell quartets reuse these same polynomial/primitive kernels.

Independent acceptance compares orders one through four with closed complex
Gaussian moments and an entire complex Boys series, using four separate
geometric step sizes and recorded error fits. This kernel component is not
a blanket acceptance of magnetic-field nuclear force constants. The
complex self-consistent density, metric and stationary-response connection
is separately accepted on the restricted and charged open-shell subsets
below; unsupported provider combinations remain explicitly tracked.
<!-- scalar-london-api:end -->

<!-- finite-magnetic-nuclear-measurements:begin -->
### Measured finite-magnetic nuclear subset on the common driver

Hessian and contracted cubic/quartic outputs use Hartree/Bohr^2, ^3 and ^4
for dimensionless probe entries. Optional bfield is atomic magnetic flux
density and field is Hartree/(e Bohr). Native hessian_report and ASE
get_hessian_report accept these controls; the legacy direct native hessian
entry retains its own signature. Plain CLI hessian with --bfield selects
the analytic common driver even without --format.

Six restricted H2 cases passed 24 independent nuclear order-1..4 fits:
p=1.991552236..2.029039426, minimum R2=0.999970126791. HF H2 simultaneous
E/B tensors passed native Python, ASE atomic/physical and both CLI names.
Maximum fresh response residual is 6.789960111282125e-16, reference defect
2.220446049250313e-16 and stationarity defect 9.906788930869048e-15.
London degree was 6/7/8 for Hessian/cubic/quartic; projector defects were
measured as zero on this two-orbital example. Raw fits and the failed
coarse quartic interval are retained in response.md. Charged unrestricted
H2+ with spin Zeeman now also passes four independent order-1..4 fits and
the same three public tensors. Its constrained collinear spin populations
are reported with charge, multiplicity, unrestricted, n_alpha and n_beta.
The alpha/beta Zeeman offsets are +/-g_e|B|/4. Raw translation contractions
are below 6.884e-15 without repair. PBE0 water analytic-path fits also pass; water public tensors,
ROHF/ROKS, molecular generalized, correlation and the remaining periodic
magnetic/DFPT combinations are not inferred from these subsets.
<!-- finite-magnetic-nuclear-measurements:end -->

### Sample London basis functions and kinetic momentum

On a uniform-flux `MagneticCell`, call
`orbital_values_with_momentum(&basis, &points, cutoff)` for the Bloch
AO matrix and the three Cartesian `p + A` matrices in x, y, z order.
Each complex matrix has one row per point and one column per spherical
AO. Give points and the image cutoff in Bohr. The matrices have units
of Bohr^-3/2 for normalized AO values and Bohr^-5/2 for momentum
acting on those AOs. The image phase includes the cell's magnetic
Bloch wave and the London phase uses its gauge origin.
`orbital_values` and `orbital_momentum_values` return an individual
matrix through the same evaluator. The shared driver uses the combined
evaluation for magnetic SCF and current samples.
[Operator and SCF validation](response.md#covariant-semilocal-scf-on-uniform-flux-cells)

<!-- periodic-magnetic-nuclear-api:begin -->
### Nuclear response on a finite uniform-flux periodic reference

`driver::run("periodic-magnetic-scf",...)`, CLI `dft-rs periodic-magnetic-scf`,
native `periodic_magnetic_scf` and ASE `get_periodic_magnetic_scf` use the
same nuclear response driver. Optional nuclear_order=1..4 adds the nested
nuclear_response object. With nuclear_direction (3N dimensionless entries),
energy_path[n] contains coefficient E_n and derivative n! E_n. Its units are
Hartree/Bohr^n; n=0 is the total model energy. The parameter along this path
has length units. Without nuclear_direction, order 1 returns gradient
(Hartree/Bohr), order 2 returns hessian (Hartree/Bohr^2), order 3 returns a
cubic matrix contracted with probe (Hartree/Bohr^3), and order 4 returns a
quartic matrix contracted with probe and second_probe (Hartree/Bohr^4).
second_probe defaults to probe. Tensor axes are 3*atom+axis (x,y,z).
Input directions/probes are dimensionless on native and ASE surfaces.

The fixed-cell finite_gamma provider differentiates point-source Ewald
energy, moving overlap, covariant kinetic energy, electron-nuclear Fourier
vertices, full/range exchange, Zeeman metric and generated nuclear SOC.
Integer occupations constrain the full spinor occupied subspace; relaxed
density/metric/projector coefficients use the shared complex response
solver. Every source and AO image moves with its atom while magnetic field,
quantized flux, Bloch sector and selected lattice sums remain fixed. The
model's source/image/reciprocal cutoffs must be converged independently.
Its finite-sector nuclear derivatives do not assert infinite-crystal
acoustic sum rules at fixed magnetic Bloch phase.

Energy derivatives use the variational 2n+1 rule. The reported
solved_response_order is 0/1/1/2 for energy orders 1/2/3/4.
Stationarity/residual arrays cover only actually solved orders; an empty
array is not a measured zero. The complete-density path remains the
independent oracle. Unit labels are root-qualified dotted paths;
energy_path units use each row's order. Diagnostic coefficient norms use
the dimensionless unit-length path t/a0, with energy measured in Hartree.

Fresh response_residuals are relative; projector_defects are dimensionless;
stationarity_defects, reference_defect and reference_energy_defect are
Hartree. Tensor reports declare intrinsic permutation reuse and include
accumulation_roundoff; independent ordered asymmetry is not fabricated.
The 3D point-source HF/custom-range-exchange/SOC subset passed independent
order-1..4 fits. HF scalar, gradient, Hessian and contracted cubic/quartic
reports passed native Python/ASE atomic+physical/canonical+alias CLI checks.
A charged one-electron state with three-face flux and generated SOC also
passed independent fits and scalar-path public checks. Measured failures,
raw residuals and resolved intervals are recorded in response.md. Semilocal/nonlocal,
finite-width/countercharge, double-hybrid, magnetic crystal_BZ, 1D/2D,
arbitrary-q DFPT nuclear combinations remain separate required providers.
Fixed Cartesian Berry electric bias on the 3D finite-Gamma model is now
connected through the common complex functional and is under independent
physical/public acceptance. Requested dispersion/nonlocal terms that this
reference cannot supply are explicitly refused instead of being omitted.
<!-- periodic-magnetic-nuclear-api:end -->

<!-- joint-magnetic-nuclear-electric-api:begin -->
### Simultaneous electric and magnetic bias on a finite periodic cell

The existing periodic-magnetic-scf Rust/common-driver, canonical CLI,
native periodic_magnetic_scf and ASE get_periodic_magnetic_scf accept the
existing field control, in addition to magnetic_field. Native/Rust/CLI
electric components use Hartree/(e Bohr); default ASE uses eV/Angstrom.
The same 3D finite_gamma reference feeds nuclear_order=1..4 and all its
scalar, gradient, Hessian and contracted-tensor reports. Both fields and
the magnetic Bloch phase stay fixed while all atomic/source/AO centres move.

The electric term uses M=I+P(Q-S), Q=exp(i b_a.r) in the magnetic AO
space, with occupation one for the full spinor manifold. Its energy is
sum_a (field dot a_a)/(2pi) times [arg det M_a - sum_A Z_A b_a.R_A].
The Hermitian Fock and density kernel are analytic derivatives of this
same energy, including complex off-diagonal density. Moving operators,
overlap/Pulay terms and the occupied projector keep the full nuclear jets.
The real collinear periodic field implementation delegates to this same
functional. No registered functional name selects the response algorithm.

reference_field reports the Cartesian bias. electric_dipole is the dipole
per magnetic cell (e Bohr), on the local Berry branch, and
energy_components.orbital_field is its electric enthalpy -field dot dipole.
electric_field_model states the static finite-cell approximation, integer
spinor subspace, local phase branch and compensation convention in the report.
Uniform charge compensation is centred at half the sum of the three
lattice vectors; its fixed dipole is included. An explicit Gaussian
countercharge uses its actual centroid. Charged-cell polarization therefore
uses this stated compensation/origin convention. Static finite-width
references do not yet imply their nuclear-response provider.

The total electronic phase is continued near the electron-number-weighted
effective nuclear charge centroid; higher Taylor coefficients continue
that local branch without rewrapping. Branch-cut translation acceptance
is tracked in response.md. The link determinant
must be nonzero and occupations insulating integers. This is a finite-cell
metastable electric-enthalpy model, not a steady metallic DC transport
state or an assertion of a global polarization branch in a Chern crystal.
Independent image-cutoff field-energy/Fock defects join the operator
convergence test. Magnetic-BZ Berry strings, open-boundary electric
coupling and the remaining functional/source/finite-q nuclear providers
remain required. Sixteen physical joint-field/zero-field-limit fits and charged scalar
public-route comparisons have passed. Neutral full tensor checks are
currently running; no complete-program claim follows from this entry.
<!-- joint-magnetic-nuclear-electric-api:end -->

For a cubic magnetic cell of side 5 Bohr (2.645886054515 Angstrom), one
flux quantum through its xy face corresponds to Bz=2*pi/25 in atomic
units. With a two-atom cell.xyz in Angstrom, a scalar nuclear path through
order four can be requested with the existing CLI:

```text
dft-rs periodic-magnetic-scf cell.xyz --cell '[[2.645886054515,0,0],[0,2.645886054515,0],[0,0,2.645886054515]]' --xc hf --basis sto-3g --guess core --magnetic-model finite_gamma --magnetic-field 0,0,0.25132741228718347 --field 0.0007,0.0005,-0.001 --nuclear-order 4 --nuclear-direction '[0.07,-0.05,-0.3,-0.11,0.09,0.3]' --format json
```

The corresponding native call uses Bohr geometry; ASE can use the same
atomic-unit controls via DftRs(units="atomic"):

```python
controls = dict(xc="hf", basis="sto-3g", guess="core",
                magnetic_model="finite_gamma",
                magnetic_field=[0, 0, 0.25132741228718347],
                field=[0.0007, 0.0005, -0.001], nuclear_order=4,
                nuclear_direction=[0.07, -0.05, -0.3, -0.11, 0.09, 0.3])
report = dft_rs.periodic_magnetic_scf(numbers, positions_bohr, cell_bohr,
                                     **controls)
adapted = dft_rs.DftRs(units="atomic").get_periodic_magnetic_scf(atoms,
                                                               **controls)
```

The caller supplies atoms, numbers and geometry. Cutoff, basis, SCF and
response convergence must be established for that particular system.
To obtain a full gradient or Hessian, omit nuclear_direction and set
nuclear_order to 1 or 2. For a cubic/quartic contracted matrix, use order
3 or 4 with probe and, for two distinct quartic contractions, second_probe.


### Occupied-projector delta-SCF controls

`occupation_mode="mom"|"imom"`, together with complete zero-based
`occupied_alpha` and `occupied_beta` column lists, selects a determinant
from a converged initial reference. The default is `"aufbau"` with no lists.
MOM updates the target occupied space each cycle; IMOM keeps the initial
target. The score is `w_j = sum_i |C_target,i^T S C_candidate,j|^2` in the
physical AO metric. It is invariant under orbital signs and rotations
within the target occupied space. This is explicitly the squared-projector
variant; the amplitude-sum definition is a different selection rule.
All occupied columns are returned first, with energy order within the
occupied and virtual blocks. Thus the complete MO energy array need not
be ascending. Empty spin spaces are valid. Tied boundary weights are
reported without introducing a score floor.

State selection reuses the ordinary energy/Fock builder, XC provider,
full/range exchange, overlap orthogonalization and Pulay gradients.
It does not dispatch on a functional's registered name. Integer total
electron number and absolute spin projection must match the system;
different alpha/beta occupied subspaces require `unrestricted=True`.
Fermi smearing, fractional electron offsets and ground-state stability
descent are incompatible with this integer excited-determinant selector.
The ground-state SOSCF descent is skipped for selected saddle states.
Physical-Fock selected-density residuals are checked in addition to the
commutator and density/energy changes, so a wrong occupied eigenspace
cannot pass solely because its commutator vanishes.

The optional `occupation_tracking` object reports the mode, initial SCF
energy, state-minus-initial SCF energy (Hartree), complete input indices,
raw projector weights, occupied/omitted boundary gaps and the maximum
selected-density-map residual in the orthonormal AO frame (dimensionless).
An absent boundary gap is null for an empty/full spin space, not a
physical zero. The common root `units` labels preserve ASE energy conversion and
leave weights/gaps unchanged. Core-hole relaxation uses this same IMOM
driver and preserves alpha/beta labels when the removed electron makes
the remaining beta population larger than alpha.

The accepted energy subset is real collinear restricted/unrestricted
HF, SVWN5, PBE, TPSS, PBE0 and CAM-B3LYP on H2. Ten restricted/unrestricted HF/LDA/GGA/meta-GGA/hybrid
finite-electric-field force fits pass with four steps (restricted grid 4,
unrestricted grid 6); the coarse-grid failures remain documented. Water
also tests the ordinary occupied-subspace limit and a singly promoted
unrestricted PBE0 energy on native/ASE/CLI surfaces. These are state-specific
determinant energies/forces, not spin-projected singlet, multiplet-ROKS
or linear-response excitation energies. Full excited-state Hessians,
third/fourth derivatives, correlated double-hybrid amplitude/target
response, geometry transport by cross-overlap, complex/spinor selection
and periodic k/q occupation tracking remain required acceptance and
implementation work. Parent-functional convergence alone does not certify
these combinations. No additional dependency is used.

Rust callers use `scf::occupations::{DeterminantOccupation,OverlapReference}`
with `ScfOptions.occupation` or `driver::method::MolecularMethod.occupation`.
`run_scf_with_occupations(molecule,basis,options,functional,grid,reference,
occupation)` accepts a converged same-metric reference explicitly. The
common driver accepts the same control names. The supplied reference must
share the current AO metric; a geometry change needs cross-overlap transport.

```rust
use dft_rs::driver::{run, Controls, System, Value};
let system = System::molecule(vec![1,1], vec![[0.0,0.0,-0.7],[0.0,0.0,0.7]]);
let controls = Controls::from_value(&Value::object()
    .with("xc", "pbe0").with("basis", "sto-3g")
    .with("occupation_mode", "imom")
    .with("occupied_alpha", vec![1_i64]).with("occupied_beta", vec![1_i64]))?;
let report = run("energy", &system, &controls)?;
```


### Scalar dimensional source-energy foundation

`periodic::electrostatics::DimensionalCoulomb::energy_components_scalar<S: Scalar>(sources: &[ElectrostaticSource], positions: &[[S;3]], cos_sin: impl Fn(S)->(S,S)+Copy) -> Result<ElectrostaticComponents<S>>`
evaluates the same 1D/2D/3D real, nonzero-reciprocal and zero-mode Green
kernel as the ordinary f64 `pair` and `evaluate` methods. `S` may be a
Taylor or nested dual scalar. The callback returns (cos,sin), in that order.
Charges, Gaussian widths, lattice, cutoffs and the selected reference image
partition remain fixed; only the supplied positions vary. Source coordinates
and widths are Bohr, charges are electron-charge units, energy is Hartree;
an nth displacement derivative is Hartree/Bohr^n. `ElectrostaticComponents<S>`
defaults to f64 and retains its real/nonzero/zero/self/mean components and
`total()`. Coincident finite Gaussian sources use analytic radial limits,
including nilpotent zero-mode coordinates; point self terms remain excluded.

The source kernel passes 24 independent four-order fits and closed zero-mode
oracles. It is a low-level differentiable foundation; it does not by itself
supply a strain/cell derivative or an electronic self-consistent response.
The finite-Gamma magnetic nuclear backend now connects 2D slabs and 3D
point/finite-Gaussian nuclei plus a fixed explicit Gaussian countercharge to
this kernel and to the corresponding Fourier form factors. Only the 3D
point-source relaxed magnetic/joint-field subset is independently accepted.
New slab/finite-width/countercharge relaxed-SCF and public nuclear acceptance
remain required. Nonzero slab electric bias, wire nuclear response,
semilocal/nonlocal moving magnetic kernels, magnetic-BZ and arbitrary-q
providers are not inferred from this source-only acceptance.


### Fixed orbital population controls

The ordinary `energy`, `gradient`, `dipole`, `charges`, `orbitals`, Molden
and JSON orbital reports accept `occupation_mode="fixed"` with
`occupations_alpha` and `occupations_beta`. Each population is in [0,1]
per spin orbital. Arrays follow ascending Fock energy, and omitted trailing
columns have zero population. Their sums must equal the system's integer
alpha/beta counts (absolute tolerance 1e-10); no population is renormalized
or clipped. Both arrays must fit the retained overlap rank. Restricted
references require equal arrays after zero padding; different arrays use
`unrestricted=true`. Empty arrays are valid for an empty spin channel.

```rust
use dft_rs::driver::method::{MolecularMethod, prepare_molecular, run_molecular_reference};
use dft_rs::scf::occupations::FixedOccupations;
let method = MolecularMethod {
    xc: Some("pbe0".into()),
    fixed_occupations: Some(FixedOccupations {
        populations: [vec![0.8, 0.2], vec![0.8, 0.2]],
    }),
    ..Default::default()
};
let setup = prepare_molecular(&[1,1], &[[0.,0.,-0.7],[0.,0.,0.7]], &method)?;
let state = run_molecular_reference(&setup)?;
assert!(state.scf.fixed_occupation_residual.unwrap() < setup.options.p_tol);
# Ok::<(), dft_rs::DftError>(())
```

Low-level clients set `ScfOptions.fixed_occupations` to the same typed
constraint and call `scf::run_scf`. The energy/Fock/XC/exchange/continuum
builders and analytic weighted Pulay gradient are the common ones; no
functional-name dispatcher supplies a substitute energy. The fixed-density
constraint is `P_sigma=C_sigma diag(f_sigma) C_sigma^T`, with `df/dR=0`.
SCF extrapolation and convergence use the complete population map rebuilt
from the physical Fock, rather than accepting a commuting density with
different populations. Ground-state SOSCF is not applied to this constraint.
The fresh final map is reevaluated after the final physical Fock build.
Faer handles the shared matrix/eigensolver operations, and the existing
Rayon integral/XC scheduling remains in use.

`energy` is the prescribed-density internal energy in Hartree. There is no
thermal entropy term or chemical-potential fit. This is an orbital-stationary
ordinary density functional at supplied populations; it does not implement
an exact ensemble-weight-dependent XC prescription or remove ghost
interactions. It therefore cannot be identified with a canonical multiplet,
an optimized excitation ensemble, a spin-projected state or FOD analysis.
See [Janak's variational identity](https://doi.org/10.1103/PhysRevB.18.7165)
for the occupation derivative of a stationary density-functional energy.

Reports include `occupation_model`, per-spin `occupations` (dimensionless),
`fixed_occupation_residual` (maximum orthonormal-AO density-map element,
dimensionless), and `fixed_occupation_gap` (smallest absolute energy gap
between differently populated orbitals, Hartree; null when none exist).
An exact degeneracy with unequal populations carries an explicit branch
warning. Derivative acceptance requires a resolved smooth branch; equal
populations within a degenerate subspace are invariant under its rotations.
Reported MO energies are unmodified, and no denominator or spectral floor
is applied. `ScfResult::has_integer_occupations` tests the actual reported
leading-column populations. Fractional states have null HOMO/LUMO/gap
labels in the frontier report; their complete orbital spectrum is retained.

For fractional occupations `s_squared` reports the independent-orbital
occupation ensemble spin moment,
`Sz^2+(N_alpha+N_beta)/2-Tr(P_alpha S P_beta S)
+sum_sigma,i f_sigma,i(1-f_sigma,i)/4`.
This includes the number-fluctuation term and is not an exact canonical
multiplet spin. The integer limit reduces to the determinant expression.
Its independent 16-determinant H2 oracle is part of the acceptance suite.

Fixed populations cannot be combined with Mermin thermal smearing,
fractional electron offsets, MOM/IMOM, determinant seeds or ground-state
stability descent. The Mermin/free-energy API requires thermal equilibrium;
use `energy` for this constraint. Static electric, Cartesian Hessian and APT
responses use the spectral solver described below. Other integer-projector
response routes refuse fractional references rather than dropping population
factors. Higher response, correlated PT2, ROKS, complex/spinor and periodic
fixed-population combinations, FOD and
weight-dependent ensembles remain tracked as pending in R073/T163. No
whole-family or arbitrary-functional-class validation is inferred from the
accepted collinear parent subset. Measured oracles and four-step fits are
recorded in [response.md](response.md#fixed-population-acceptance).

### Fixed-population and thermal static response

The existing `driver::run("polarizability", ...)` and `driver::run("hessian", ...)`
accept the same population controls as `energy`. The common driver chooses
the spectral population law from the reference, independently of the functional
name. `hessian::analytic::analytic_hessian_full` also exposes the density and
energy-weighted-density derivatives used in the force contraction and APT.
The moving basis, grid, Hartree, full/range-separated exchange and connected
XC/nonlocal/continuum terms retain their ordinary shared implementations.
Existing unsupported skeleton terms retain their explicit restrictions.
Reference and response integral engines share the configured `incore_mb`
memory policy: cache a full n^4-double tensor when it fits, otherwise use
direct recomputation. Explicit in-core limits and density fitting retain
their selected behavior. Cached and direct engines use the same physical
kernel; this choice does not change occupations or response denominators.

```rust
use dft_rs::response::{ensemble::EnsembleSolver, ResponseSolverConfig};
let solver = EnsembleSolver::new(
    &setup.molecule, &setup.basis, &setup.functional,
    setup.grid.as_ref(), &state.scf, &setup.options,
)?;
let config = ResponseSolverConfig::default();
let electric = solver.polarizability(&config)?;
assert!(electric.responses.iter().all(|r| r.relative_residual <= config.acceptance()));
# Ok::<(), dft_rs::DftError>(())
```

`EnsembleSolver::new` accepts a converged prescribed-population reference at
zero electronic temperature or a positive-temperature Fermi-Dirac reference.
`population_model()` identifies the derivative constraint. `polarizability`
returns `EnsemblePolarizability { tensor, responses }`: an unsymmetrized 3 by 3
tensor `alpha_ab=d mu_a/d F_b` in e^2 Bohr^2/Hartree, and three independent
`EnsembleResponse` solutions in x/y/z field order. Each solution contains the
two AO density derivatives, fresh original-operator relative residual,
iteration count and per-spin number defects. No tensor component is copied
from a symmetry-related solve. A pre-existing electric bias is retained.

`solve(Some(&[V_alpha,V_beta]), [dN_alpha,dN_beta], &config)` accepts finite
matching AO potentials and returns the same response type. `solve_metric`
accepts the fixed-density Fock derivatives and overlap derivative and returns
`MetricEnsembleResponse`, including `delta_energy_weighted_density`.
Thermal number directions include the chemical-potential constraint.
For prescribed populations only zero number directions are admissible;
the reported zero chemical-potential change is an unused gauge convention.

For fixed populations `W_ij=(f_i-f_j)/(epsilon_i-epsilon_j)` for unequal
populations and zero for equal populations, including the diagonal. All
orbital pairs participate, including partially occupied nominal virtual
orbitals. An exactly degenerate subspace must have equal populations.
`response_min_denominator` rejects smaller differently populated gaps without
changing the operator; no denominator floor supplies a physical regularization.
At finite temperature the diagonal and continuous degenerate limit are the
Fermi derivative, with each spin count conserved. These thermal and fixed
constraints are distinct even when their reference populations happen to match.

Common polarizability reports add `response_population_model`,
`response_residuals[3]` (dimensionless), `response_iterations[3]` and
`number_defects[3][2]` (electron-count derivative per electric field,
e Bohr/Hartree). Hessian reports retain `ensemble_response_residuals[3N]`
and `ensemble_number_defects[3N][2]` (electrons/Bohr), plus the
Hartree/Bohr^2 Hessian and `dipole_derivatives[3][3N]` (e).
Hessians are symmetrized only after recording `raw_asymmetry`; the electric
tensor and APT are computed directly. Harmonic frequencies/IR on a fixed
population surface are constrained-state quantities; equilibrium RRHO
thermochemistry is withheld and `gibbs-free-energy` refuses that nonthermal
reference. Nonlinear, magnetic, periodic and correlated population responses
remain separate unaccepted combinations.

Measured four-step fits, original-operator residuals, physical AO number
defects, public-route comparisons and the five independent molecular tensors
are recorded in the [spectral-response acceptance](response.md#fixed-population-spectral-response-acceptance).

### Prescribed-population nonlinear electric response

`response::electric_field_taylor`, `mixed_electric_field_response`,
`static_beta`, `static_gamma`, `static_electric_tensor` and
`static_beta_variational` accept a converged real collinear reference with
`ScfOptions::fixed_occupations`. The same operations are available from
`driver::run("field-taylor" | "field-mixed" | "beta" | "gamma", ...)`.
Every per-spin orbital population remains fixed, including differently
populated pairs within the conventional occupied and virtual subspaces.
These are constrained nonthermal derivatives at zero frequency. They do not
include the temperature/chemical-potential response of a thermal population.

The common spectral kernel solves for skew orbital generators in
`D(t)=exp(K(t)) diag(f) exp(-K(t))`, with `K(0)=0`. Equal-population rotations
are a gauge; unequal-population exact degeneracy is rejected by the reference
kernel. The algorithm never divides by an occupation difference or replaces
the physical orbital gap by a preconditioner floor. Each order reevaluates
the complete nonlinear Fock coefficient and its full stationarity residual.
The fixed-grid semilocal and learned energy/potential Taylor providers,
Hartree, full/range-separated exchange, connected VV10 and linear continuum
terms are reused by both the integer and constrained recursions.

`ElectricFieldTaylor::fixed_population: Option<FixedPopulationDiagnostics>`
is `Some` on this path. Its reference density/population defects and
order-resolved population, orbital orthogonality and relative commutator
defects measure the active constraints. All are atomic-unit residuals;
the relative commutator uses the actual matrix-product norm bound. The
original `projector_defects` and `ReferenceDiagnostics::projector_defect`
retain their measured non-idempotency and are not constrained to zero.
`StaticElectricTensor::projector_constraint` identifies whether idempotency
is active; `fixed_population_defect` is the maximum of the active spectrum,
metric and relative stationarity diagnostics. Mixed derivatives aggregate
the convergence flags and raw measurements from all directional solves.

For `E(t)=E0+t*d`, `t` is dimensionless, `d` is Hartree/(e Bohr), and
`dipole_coefficients[n]` is `d^n mu/dt^n / n!`, in e Bohr. The mixed API
returns actual directional derivatives in e Bohr. Full beta and gamma
contain actual Cartesian derivatives in e^3 Bohr^3/Hartree^2 and
e^4 Bohr^4/Hartree^3. All ordered slots remain independently evaluated.
The optional three-solve variational 2n+1 beta uses the same spectrum
constraint and complete nonlinear source as the recurrence.

The nonlinear common driver refuses correlated PT2 requests until the
corresponding amplitude/orbital and distinct-target Lagrangian sources are
connected. Higher thermal, moving-nuclear fixed-population, magnetic,
spinor and periodic population responses require their own acceptance;
this static molecular interface does not establish those combinations.
Measured acceptance intervals and unresolved validation cases are recorded
in [response.md](response.md#prescribed-population-nonlinear-electric-acceptance).

For the moving-nuclear core, `response::nuclear::NuclearTaylor` (also the
`MolecularTaylor` value type) now carries optional `fixed_population`
diagnostics. `nuclear_taylor` and `molecular_taylor` use the same isospectral
orbitals in the analytically transported AO overlap metric. The raw
`projector_defects` of a genuinely fractional density remain nonzero and are
reported; the active constraint is its preserved population spectrum. The
separately implemented excited/reduced projector equations retain their
integer-reference requirements. HF R/U finite-electric-field orders one
through four have independent core acceptance in response.md; remaining
functional and public high-order nuclear combinations are still pending.

## Experimental multicomponent Rust API

`multicomponent::solve(molecule, electronic_basis, electronic_options,
functional, grid, particles, correlation, options)` returns
`Result<MulticomponentResult>`. All lengths are Bohr, energies Hartree,
charges positive elementary-charge units and masses electron masses.
The electronic `ResolvedFunctional` is independent of the interspecies
provider; there is no electronic functional-name dispatch in the solver.
The current experimental electronic reference is real, collinear, zero
temperature and Aufbau. The common driver rejects a double-hybrid request
until its jointly stationary correlated density provider is connected.

| Type or entry | Contract |
|---|---|
| `QuantumParticle` | `label`, finite `charge`, positive finite `mass`, independent normalized Gaussian `basis`, optional zero-based `replaced_atom`, and `correlated_proton`. Replacing an atom removes exactly its matching all-electron classical point charge. `None` adds a particle. Every particle is distinguishable and has one normalized Hartree orbital. |
| `gaussian_particle_basis(center, exponents, lmax)` | Builds an uncontracted normalized particle basis with every requested exponent for angular momenta `0..=lmax` (`lmax <= 3`). Centers are finite Bohr coordinates; positive exponents are Bohr^-2. The core solver also accepts a caller-supplied multicenter `MolBasis`. |
| `MulticomponentOptions` | `tolerance=1e-10`, `max_iterations=500`, `mixing=0.5`. The tolerance separately bounds measured orbital commutators (Hartree in the orthonormal frame), density-map defects and particle-number errors. `mixing` is the initial normalized orbital step in the descent search, with `0 < mixing <= 1`; it never changes a physical operator or denominator. |
| `InterspeciesCorrelation` | A `Sync` local provider supplies `description()`, `evaluate(total_electron_density, total_proton_density)` and `is_zero()` (default false). Only an identically zero provider can omit a grid. Densities are Bohr^-3. All quantum densities marked `correlated_proton` enter the total proton argument. |
| `CorrelationPoint` | `energy` is energy per volume (Hartree/Bohr^3); `electron_potential` and `proton_potential` are the two functional derivatives (Hartree). Both enter the respective orbital equations on the same integration grid. |
| `Epc17` | Public parameters `a`, `b`, `c` default to `2.35, 2.4, 3.2`. Eq. (13) of [Yang et al.](https://doi.org/10.1063/1.4996038); parameters were fitted to proton densities. Inputs must be nonnegative and finite; a denominator pole for any physical density is rejected. The exact one-sided zero-density potential limits are retained. |
| `NoCorrelation` | Explicitly zero interspecies dynamical correlation; all kinetic and Coulomb terms remain. This mean-field model can overlocalize proton densities. |
| `QuantumState` | `coefficient`, `density`, `overlap`, `fock`, `orbital_energies`, `kinetic_energy`, `classical_energy`, `particle_number`, `position`, `position_variance`, `orbital_residual`, `density_residual`. Matrices use row-major AO order. `position_variance` is the diagonal central second moment in Bohr^2; it is not a classical optimized atom coordinate. |
| `MulticomponentResult` | `energy` and the disjoint electronic-one-body, electronic-Coulomb, electronic-exchange/XC, electronic-nonlocal, electron-quantum-Coulomb, distinct-quantum-pair-Coulomb, interspecies-correlation and classical energy terms; `quantum` states carry the remaining kinetic and classical terms. `electronic_kinetic` is an additional diagnostic already included in `electronic_one_body`. Also returns `electronic_density`, fresh `electronic_orbital_residual`/`electronic_density_residual`, `iterations`, `converged`, `energy_history` and `residual_history`. |

The nonlinear orbital step is checked with the actual coupled functional.
Analytic endpoint work and fresh commutator reduction resolve a local descent
step if energy subtraction is rounded; no residual is replaced by an energy
floor. Electronic Coulomb/exchange integral caches are shared between frozen
inner solves. The scoped density-dependent potential is kept out of
`ScfOptions::extra_hcore`, and the coupled result does not expose an ordinary
`ScfResult` that could be differentiated with missing species terms.

A supplied ground-state interspecies provider needs only its energy and
both potentials. If its analytic higher kernel is unavailable, the optional
nuclear Newton accelerator is disabled and the same variational SCF equations
and residual gates remain active. Numerical/input errors are propagated.
Geometry/electric response still requires the full analytic kernel and never
replaces a missing physical response term by zero or finite differences.

Unselected nuclei stay clamped. Selecting every nucleus gives a laboratory
quantum-particle energy retaining center-of-mass motion. Internal-coordinate
mass polarization, rotational/vibronic eigenstates, identical-nucleus
statistics, periodic quantum particles
and correlated positron/photon providers remain required work. This is an
experimental physical subset of R076/T166, not completion of that family.

### Coupled quantum-nuclear response and analytic geometry API


`quantum_mass_response=true` (`--quantum-mass-response true` on the CLI)
adds `mass_response.gradient`, the flattened row-major `hessian` with
`hessian_shape=[Q,Q]`, independently assembled `raw_asymmetry`, and per-mass
coupled/reference/position residuals and particle-number defects. Masses stay
in electron masses (`m_e`) on every surface: gradients use Hartree/m_e and
Hessians Hartree/m_e^2 natively, or eV/m_e and eV/m_e^2 through ASE. The
position-response residual uses Bohr/m_e or Angstrom/m_e. This differentiates
the laboratory-frame model at fixed Gaussian bases, charges, correlation
parameters and position targets. It removes no COM kinetic energy and adds no
identical-particle statistics. The default is false and returns `mass_response=null`.
The isolated HF C1 H3+ two-isotope candidate passes all two gradient and four
ordered Hessian four-step fits: p=2.002752467556..2.004439020151, minimum
R2=0.999998680142; response residual at most 8.462e-15 and number defect at
most 3.331e-16. The scoped rebuilt artifact passes the permanent mass suite, native/ASE/both-CLI tests and default-false regressions; its source/core hashes are in `molecular-response-followup-20261101-01/final-acceptance.json`.


The experimental multicomponent geometry implementation differentiates both
species at one stationary energy. Finite differences are acceptance oracles
only. Unconstrained quantum Gaussian centers are finite-basis parameters;
they are not physical clamped-nucleus positions. For a molecular surface
parameterized by quantum nuclear expectation positions, use the constrained
reference below, following the position convention of
[Xu and Yang (2020)](https://doi.org/10.1063/5.0014001).
Neither path removes laboratory center-of-mass kinetic energy.

| Public Rust entry or type | Contract, units and conventions |
|---|---|
| `multicomponent::constraints::solve(molecule, basis, electronic_options, functional, grid, particles, correlation, options, targets, config)` | Returns `MulticomponentResult` after simultaneous electronic/nuclear stationarity and `mean(r_I)=targets[I]` (Bohr). Targets can differ from Gaussian centers. Each frozen nuclear Fock uses its analytic position susceptibility, measured range/null-space checks and a multiplier residual line search; the outer SCF reconverges all species. Orbital steps are retracted onto the position manifold. No unconstrained collective translation inverse, penalty energy or finite-difference Jacobian is used. |
| `constraints::PositionConstraints` | `targets: Vec<[f64;3]>` (Bohr), `multipliers: Vec<[f64;3]>` (Hartree/Bohr), `residual` and outer-SCF `residual_history` (maximum Cartesian expectation-position errors, Bohr), accumulated nuclear-inner-Newton `iterations`. Sign: `L=E+sum lambda_I dot (mean(r_I)-target_I)`. `MulticomponentResult::position_constraints` is `Some` only for a constrained reference; physical `energy`, its history and disjoint energy terms exclude multiplier work. Nuclear `fock` and orbital energies include `lambda dot (r-target)`. |
| `multicomponent::response::CoupledResponseSolver::new(molecule, electronic_basis, options, functional, grid, particles, correlation, state)` | Requires the matching jointly converged real collinear reference. Reuses the common electronic Fock kernel and both interspecies density endpoints. Restricted electronic amplitudes use sqrt(2) normalization, preserving the Hermitian cross-species Hessian without changing a gap. |
| `CoupledResponseSolver::{dimension,apply_real}` | Orbital dimension and matrix-free coupled Hessian action. Physical gaps are retained. `apply_real` is the orbital block at fixed multipliers; the constrained `solve` additionally includes position multipliers. |
| `CoupledResponseSolver::solve(electronic_sources, quantum_sources, config)` | Static AO-source response for two electronic spin matrices and one matrix per quantum particle. Sources must be finite symmetric matrices in matching physical AO spaces. For a constrained reference, targets stay fixed and an augmented Hermitian orbital/position system is solved. Common MINRES defaults: relative residual 1e-10, 500 iterations; residuals are reevaluated with the actual operator. |
| `response::CoupledDensityResponse` | Physical `electronic_density: [Matrix;2]`, `quantum_density: Vec<Matrix>`, normalized orbital `amplitudes`, `relative_residual`, `reference_residual`, `iterations`, `converged`. Constrained results additionally contain `position_multipliers` and `position_residual` for the linearized constraints. Density responses are per unit of the supplied perturbation. |
| `CoupledResponseSolver::polarizability(config)` | Total static dipole response, including quantum-charge motion or its constrained response. Returns `CoupledPolarizability { tensor: [[f64;3];3], raw_asymmetry, responses }`; tensor and raw asymmetry are Bohr^3. All three source directions are evaluated independently. |
| `multicomponent::mass::{gradient,response}` | `gradient(particles,state)` returns the stationary quantum-mass gradient. `response(molecule,basis,options,functional,grid,particles,correlation,state,config)` constructs the matching coupled solver and returns `MassResponse`: all ordered relaxed mass-Hessian entries, independent column responses, raw asymmetry, particle-number and kinetic-energy reconstruction defects. Masses use electron masses (`m_e`); gradient/Hessian units are Hartree/m_e and Hartree/m_e^2. Gaussian bases, charges, correlation parameters and position targets stay fixed; the laboratory COM kinetic energy remains. Invalid masses, unmatched dimensions, changed kinetic/overlap/position data and unconverged references are rejected. |
| `InterspeciesCorrelation::kernel_action(electron, proton, [delta_e,delta_p])` | Both analytic potential variations, in Hartree. Densities/density variations are Bohr^-3. A supplied provider must implement its derivative; an absent kernel is not replaced by a numerical derivative. `Epc17` implements the mixed kernel, exact Maxwell symmetry and the physical nodal tangent limit. Undefined nonzero own-density tangents at a zero density are rejected. |
| `multicomponent::derivatives::directional(molecule, basis, options, functional, grid, particles, correlation, state, direction, config)` | Supply finite dimensionless Cartesian vectors in nuclear order, optionally followed by vectors for the added-particle Gaussian sites in particle order. A nuclear-only direction holds the added sites fixed. The path parameter is Bohr. Differentiates moving Gaussian integrals, overlap transport, moving grid coordinates/weights, electronic XC, both interspecies XC endpoints and all distinct-species Coulomb terms. Each attached quantum center follows its owning atom. A constrained expectation-position target follows its basis site and the derivative includes the multiplier response. |
| `derivatives::CoupledGeometryDerivative` | `gradient` (Hartree/Bohr), `hessian`, `explicit_hessian`, `orbital_relaxation` (Hartree/Bohr^2), `reference_energy_error` (Hartree), `response_residual`, `reference_residual`, `position_response_residual`, `iterations`, `converged`. `hessian=explicit_hessian+orbital_relaxation` uses the stationary coupled Lagrangian. The reference energy error is the signed analytic order-zero reconstruction minus the independently solved reference energy. |
| `derivatives::cartesian(..., state, order, config)` | `order=1` performs 3N axial analytic evaluations and returns the Cartesian gradient; `order=2` also assembles the full relaxed Hessian from analytic directional polarization. Invalid orders are rejected. All atomic coordinates use the molecule's atom order. |
| `derivatives::hessian(..., state, config)` | Convenience entry for `cartesian(..., 2, ...)`. Returns `CoupledGeometryHessian { gradient: Vec<[f64;3]>, hessian: Option<Matrix>, directional_evaluations, response_residuals, position_response_residuals, reference_energy_errors, converged }`. Hessian order is `(3*atom+axis,3*atom+axis)`, in Hartree/Bohr^2. An order-1 Cartesian request returns `hessian=None`; an uncomputed tensor is never represented by physical zeros. |

The variational geometry formula uses analytic metric-transported Taylor
coefficients, the complete coupled orbital Hessian, and, when constrained,
the position KKT block. The multiplier contribution to the relaxed Hessian
is retained, including the explicit derivative of the target. Off-diagonal
Cartesian elements use polarization of independently solved directions;
they are mirrored after scalar assembly, without claiming an independent
AB/BA reciprocity measurement.

The first connected geometry subset uses all-electron Coulomb integrals,
zero-temperature Aufbau R/U electronic orbitals, and independent normalized
nuclear Hartree orbitals with optional local EPC17. The ordinary electronic
functional supplies its own analytic kernel/Taylor provider. ECP, RI,
nonlocal-correlation moving-species terms, correlated PT2 densities,
and periodic quantum particles still require their corresponding derivative
providers. The all-nucleus
laboratory model and constrained expectation-position model do not establish
intrinsic translational/rotational eigenstates, identical-nucleus statistics
or general nonadiabatic dynamics. Acceptance and failed intervals are retained
in [response.md](response.md).

### ROKS convergence fallback (acceptance pending)

`roks`, `roks-hessian` and `roks-polarizability` retain their existing
common-driver contracts and units.
The ROKS SCF implementation now includes a common-spatial-orbital Newton
fallback near convergence or after DIIS stagnation. It reuses the complete
spin-density response kernel, preserves closed/open/virtual occupation
classes, and accepts steps by the actual electronic-energy Armijo condition.
`no_soscf=true` disables it; Rust uses `ScfOptions::use_soscf=false`.
This is an optimizer change, not a change of the energy functional or of
SCF convergence tolerances. No new physical API or unit convention is added.
Production compilation and all-feature warning checks pass. The common-orbital
units and p115/p118 molecular PBE water-cation acceptance pass; the measured
four-step exponents, residuals and retained OH failures are recorded in
`response.md`. These gates cover the measured cases, with the remaining
functional, periodic, degenerate and dynamic combinations still open.


The magnetic CPKS kernel fixes its direct-integral mask at the reference density.
Every nonzero field source is solved against the original operator, including
nearly axial sources as small as 3.900e-17 (fresh relative residual 7.947e-16);
components are not discarded by comparing their magnitude to another axis.
The existing magnetic/NMR/VCD entries and units are unchanged.


### Molecular reference-aware dipole response

The common `dynamic-polarizability` driver selects `reference=rks/rhf`,
`uks/uhf`, `roks/rohf`, or `gks/ghf`. All nine Cartesian contractions are
returned independently. `frequency` and `frequency_damping` are Hartree;
`omega` retains its separate range-separation meaning in common controls,
and `damping` its SCF mixing meaning. Legacy CLI `response_omega`,
`response_damping` and `imaginary_frequency` are aliases of the electric
controls, with duplicate aliases rejected. Native Python preserves its
`omega`, `damping`, `imaginary`, `units="bohr^3"` and `unit_labels` contract,
translating electric controls before calling the same driver. ASE accepts
additional reference controls through `get_dynamic_polarizability` or
`get_feature_report("dynamic-polarizability", ...)`.

ROKS dynamics is the constrained time-dependent common-orbital model, with
closed-open, closed-virtual and open-virtual rotations. The paired time-even
and time-odd Hessians carry occupation-difference Berry weights 1, 2, 1;
an unrestricted reference is not substituted. The static limit also serves
`polarizability` with an explicit ROKS reference and `roks-polarizability`.
The generic learned Fock and density Hessian, scalar nonlocal correlation
and continuum density kernels are reused when selected; thermal and tracked
common-orbital populations and constrained stability following remain separate.

GKS/GHF uses a full complex self-consistent spin/AO density, not its diagonal
spin blocks. `response::spinor::spinor_polarizability_with_kernel` is the
shared paired projector solver; it accepts an explicit reference-bound
Hessian action. `response::nuclear::spinor_electric_response` reuses the
moving-basis provider at fixed geometry and returns `SpinorElectricReport`:
`values`, `reference_fock_defect`, actual `angular_history`, and separate
conservative `cache_bound_history`. The descriptor cache respects `incore_mb`;
uncached application remains available. Non-Hermitian transition densities
are extended as H+iA from two Hermitian directions; density/current and both
exchange endpoints remain present. A mismatch between the reference Fock and
the response provider fails instead of returning a substituted susceptibility.
The additional AO source is held fixed. GKS nonlocal/continuum/learned and
correlated spinor providers are still missing, and angular convergence must
pass independently for every requested functional and path.

The correlated zero-frequency route reuses the full parent density and
noncanonical PT2 orbital/amplitude response, with separate unscaled SS-alpha,
OS and SS-beta tensors. Causal finite-frequency PT2 action kernels and the
ROKS/GKS correlated connection remain pending; the static energy Hessian is
not presented as a frequency-dependent double-hybrid susceptibility.

Polarizability unit labels use `e^2*bohr^2/hartree`, including anisotropy
and reciprocity defects; this is essential for conversion to ASE
`e^2*angstrom^2/eV`. The historical native summary remains `bohr^3`.
All frequency aliases carry Hartree labels. The CLI text renderer retains
legacy `alpha a b real imag`, `dipole_coefficient n ...`, and space-separated
`dipole_derivative ...` rows as views of the common report; JSON is unchanged.

Acceptance and measured bounds are recorded in response.md under R001.

### Collinear ECD, VCD and NMR functional providers

The registered `ecd-spectrum`, `vcd` and `nmr-spectrum` calculations accept
`reference="rks"` or `reference="uks"` (CLI `--reference rks|uks`). Global
and range-separated exact exchange use the parent's declared coefficients;
local, gradient, kinetic-density and learned providers share the existing
response machinery. For the learned provider pass `xc="skala"` and an
explicit `skala_model` file. The magnetic adapter evaluates the complete
nonlocal reverse pass once and passes its weighted density/gradient/kinetic
partials to the same London-phase and current-current contractions used by
semilocal providers. Weights are not divided out, including at zero-weight
points. The kinetic input is gauge covariant, `tau-j_p^2/(2 rho)`; this
extension leaves a real zero-field SCF unchanged. It is adiabatic and
collinear. Learned-model accuracy is a separate question from derivative
and route consistency.

`ecd-spectrum` reports TDA line strengths and Gaussian-broadened absorption
and ECD (L mol^-1 cm^-1); `broadening_ev` is a standard deviation, not a
full width. Length-gauge strengths depend on the coordinate origin in a
finite basis; the velocity-gauge strengths and their convention remain
separately reported. This is a vertical spectrum, without vibronic band
shapes or a correlated excitation kernel.

`vcd` returns every electronic/nuclear atomic axial-tensor element
(`[3*atom+axis][magnetic axis]`, e Bohr), polar derivatives, harmonic
frequencies and rotational strengths. Distributed origins are the default;
`gauge="common-origin"` with `gauge_origin` (Bohr in native/Rust) provides
the explicit finite-basis origin convention. No tensor element is filled
by symmetry in acceptance. `nmr-spectrum` uses orbital shielding tensors
(ppm), referenced chemical shifts and unit-area Lorentzians; its full width
is `nmr_broadening` (ppm). Optional first-order J multiplets still have the
closed-shell spin-coupling restriction. Open-shell Curie/paramagnetic shifts
and a strong-coupling spin Hamiltonian are separate calculations.

Magnetic reports retain raw `response_residuals`, reference diagnostics and
the learned kinetic-input assumption in `warnings`. The independent learned London/basis-current/current-Hessian acceptance
passes all 144 independent imaginary AO elements on a fixed C1 quadrature
(p=1.987237..2.037561, minimum R2=0.99997674). All 432 VCD axial elements
across RKS/UKS and six functional providers pass `p252_spectroscopy_functionals`
(p=1.924539..2.136165, minimum R2=0.99956060). The 43-case public/units
suite passes. Measured intervals and retained failures are in response.md.
Double-hybrid ECD transitions, VCD mixed response and NMR magnetic PT2
amplitude/orbital derivatives remain pending. A parent-only calculation is
not a correlated spectrum.

### Complex PT2 path primitives

`properties::pt2_response::complex_taylor_response(problem, ovov_coefficients,
fock_coefficients, coefficients)` shares the real amplitude recurrence.
The parent is a real semicanonical `Pt2Problem<f64>`; each higher coefficient
holds complex excitation-ket sources `(a i|b j)` and Hermitian complex Fock
matrices. Occupied indices are contragredient, virtual indices covariant,
and energies contract `conj(g) t`. `Pt2TaylorResponse<Complex>` contains
factorial-normalized SS-alpha/OS/SS-beta/correlation energies, all amplitude
coefficients and freshly recomputed real residual norms. Nothing discards
imaginary components or changes the excitation denominators.

`properties::pt2_response::complex_transform_path(nao, integrals, orbitals,
energies, occupied)` transforms London AO coefficient series into those
sources. Coefficients are `[order][spin][AO][MO]`, AO integrals are in
chemist order and the output is their conjugated occupied-virtual source.
The transformation reuses the real packed SIMD contractions, with exact
complex phase sectors; no response-density screening threshold is added.
These are reusable analytic path primitives. Completion of a spectral
observable requires its transition/mixed-response driver and acceptance,
which remain under implementation; no extra CLI calculation is introduced.

`properties::pt2_response::complex_projector_frames(projectors, occupied)`
returns direct rotations and their maximum raw metric/projector defect.
The zeroth projector is the diagonal semicanonical occupied projector;
all higher entries are complex coefficient matrices. The algorithm reuses
the London noncommuting inverse-square-root metric transport. It preserves
whole degenerate occupied/virtual spaces and exposes no within-space
energy-gap division. Its independent finite-matrix acceptance uses a
Cayley orbital path and Newton-Schulz polar transport.

The complex integral/amplitude primitives pass 270 independent first/second
coefficient fits against direct AO summation and dense complex pair systems,
p=1.823113..2.181805, minimum R2=0.99925411; two unsuccessful intervals
are retained and resolve on the printed alternate intervals. Amplitude
coefficient residuals are at most 2.406e-17 and the independent orbital-phase
energy defect is 4.337e-19. The original real correlated-alpha regression
retains all 252 passing component/channel fits. These measurements validate
path primitives; correlated spectral entry-point acceptance remains required.

### Complex correlated field providers

`response::field_path::field_wavefunction_path` returns a
`FieldWavefunctionPath`: the stationary parent energy/dipole series,
optional distinct-target energy, occupied counts, complete AO orbital
coefficients, matching transported MO Fock coefficients and the raw orbital
frame defect. Inputs include a `FieldPath` and optional factorial-normalized
`[order][spin]` Hermitian core coefficients. All perturbations share the
same zero-order reference; no parent-density 2n+1 truncation is used for a
nonstationary target or correlation energy.

`response::correlated_field::correlated_field_taylor` combines that provider
with every London integral endpoint and noncanonical complex PT2 amplitudes.
`CorrelatedFieldTaylor` preserves total, target, unscaled SS-alpha/OS/SS-beta
and weighted correlation energy coefficients (Hartree), all complex
amplitudes, actual amplitude residuals and the imaginary-energy defect.
It accepts the existing PT2 memory budget and response solver configuration.
These new combined providers are undergoing independent finite-field
acceptance; their presence does not validate a correlated spectral driver.

`scf::magnetic::magnetic_field_scf_with_core` uses the existing finite-field
SCF with explicit Hermitian spin operators. Matrices are in the London AO
basis of the geometrically centred molecule and already contain the finite
perturbation strength. Energies are Hartree; fields use atomic units.
Dimensions, finiteness and Hermiticity are checked; unequal spin operators
require an unrestricted reference. The ordinary finite-field API is unchanged.

### Correlated orbital NMR on the existing API

The existing `nmr-shielding` and `nmr-spectrum` calculations accept the same
named/custom double-hybrid controls as `energy`: `xc`,
`double_hybrid_model`, `double_hybrid_target_xc`, `pt2_same_spin`,
`pt2_opposite_spin` and `pt2_memory_mb`. The common RKS/RHF and UKS/UHF
selectors are retained. The spectrum uses the existing reference shieldings
and Lorentzian line model; no separate correlated-spectrum API is added.

`shielding_tensors` are the mixed derivative of the complete target-plus-PT2
energy. `parent_shielding_tensors`, `target_shielding_tensors`,
`pt2_shielding_tensors`, `pt2_same_spin_alpha_shielding_tensors`,
`pt2_opposite_spin_shielding_tensors` and
`pt2_same_spin_beta_shielding_tensors` retain all `[nucleus][moment][B]`
entries in ppm. The three spin channels are unscaled; the PT2 tensor is
`c_ss*(SS_alpha+SS_beta)+c_os*OS`. The diamagnetic tensor is the complete
first derivative with respect to the explicit B/m contact operator; its
post-SCF density includes parent-orbital relaxation. The paramagnetic tensor
is the remainder. Energies/decompositions reuse the common energy report.
Amplitude, orbital-frame, projector, stationarity, reference and imaginary
energy residuals remain explicit; convergence tests inspect the raw values.

The integer zero-temperature orbital model excludes spin-Zeeman/Curie shifts.
Correlated J-coupling multiplets, nonlocal/learned field-path providers and
causal PT2 transition/VCD response remain separate work. The complex post-SCF field provider now uses the explicit
`CurrentInvariant` kinetic-density model, `tau - |j|^2/(2 rho)`, for both
the analytic energy/Fock series and its cached magnetic response kernel.
The independent finite-SCF oracle `magnetic_field_scf_with_core` uses the
same physical model; `magnetic_field_scf` and the existing finite-B nuclear
paths retain the `Mechanical` model. `FieldWavefunctionPath` and
`MagneticFieldResult` identify their model with `kinetic_density_model`.
The meta-GGA connection has independent TPSS full-component magnetic and
common-origin NMR acceptance, recorded below; other combinations are not
inferred from those measurements.

`properties::correlated_magnetic::correlated_molecular_nmr_response` returns
`CorrelatedNmr`, whose `response` is a `ResponseReport<Vec<Shielding>>` and
whose separate parent/target/PT2/spin tensors and diagnostic arrays preserve
the complete decomposition described above. It takes the generic parent and
optional target providers, reference/grid/SCF options, PT2 coefficients,
gauge/origin, memory budget and shared response solver configuration.

The direct complex projector-frame oracle additionally passes all 44
independent first/second coefficient fits over [0.08,0.06,0.04,0.03],
p=1.998542..2.002907, minimum R2=0.99999986, with no failed interval.


Correlated orbital NMR acceptance (`p255_correlated_nmr`, 2026-10-23)
passes all 810 common-origin Cartesian component/channel fits for restricted
C1 H4 and unrestricted C1 H5 with HF/PBE0 parents: total, parent,
SS-alpha, OS and SS-beta. The 808 second-order stencils have
p=1.807113..2.131252 and minimum R2=0.99911588; two cancellation cases
use the prescribed fourth-order mixed stencil, p=3.941553/4.022333,
minimum R2=0.99998602. All 218 failed interval attempts remain printed.
Ordinary steps are [0.04,0.03,0.02,0.015], refined through
[0.003,0.0025,0.002,0.0015]; the fourth-order intervals are
[0.04,0.03,0.02,0.015] and [0.02,0.015,0.01,0.0075].
The original restricted and unrestricted failures are retained separately;
SCF density/orbital convergence is tightened to 1e-14. No error floor,
component omission or relaxed gate is used. Maximum amplitude residual is
3.553e-15, imaginary-energy defect 1.666e-16, transported-frame defect zero
at printed precision, and the full parent-shielding limit differs by at
most 2.942e-12 ppm. The run passes in 399.15 s. This validates the named
common-origin molecular PT2 subset; full London/translation acceptance,
nonlocal/learned correlated fields, correlated transitions and spin-spin
multiplets remain separate. TPSS common-origin acceptance and the new
VCD connection are recorded below.

The complete magnetic energy Hessian acceptance (`p254_correlated_magnetic`)
passes 180 Cartesian/channel fits for restricted C1 H4 and unrestricted
C1 H5 with HF/PBE0 parents. Every total/parent/SS-alpha/OS/SS-beta entry is
computed independently: p=1.810889..2.168594, minimum R2=0.99935513.
All 243 failed interval attempts remain in the output. Resolved intervals
are [0.08,0.06,0.04,0.03], [0.02,0.015,0.01,0.0075] and
[0.005,0.00375,0.0025,0.001875] atomic units. The finite-field phase
expansion was extended from order 12 to 16 after the omitted tail of
4.706e-11 failed the unchanged 1e-12 gate. Maximum fresh amplitude/field
residuals are 1.421085471521e-14/2.026057371854e-15; the largest imaginary
energy defect is 3.885780586188e-16.

Public correlated-NMR acceptance passes all eight new cases: six native,
common-driver and ASE comparisons and two canonical CLI comparisons. The
first CLI fixture used a different Bohr conversion constant; correcting
the XYZ fixture to `dft_rs.units.BOHR_IN_ANGSTROM` resolves both failures
without changing the 1e-8 ppm gate. The companion spectroscopy, unit,
documentation and ledger checks passed in that run. These are HF/PBE0
custom PT2 and B2PLYP common-origin orbital-spectrum checks; they do not
certify PT2 ECD/VCD, coupled-spin NMR spectra or the new meta-GGA model.

The current-invariant complex XC pullback passes all 38 independent fits
of its energy and both spins' 18 independent Hermitian Fock coordinates,
through derivative order two: p=1.966887..2.001269, minimum R2=0.99998051,
steps [0.08,0.06,0.04,0.03], no failed interval. The oracle differentiates
the energy in density coordinates using forward AD and exact trigonometric
AO phases; it does not reuse the production potential pullback. TPSS full
magnetic energy Hessians (`p254_correlated_magnetic`) pass all 90 RKS/UKS
Cartesian/channel fits: p=1.811654..2.099307, minimum R2=0.99932393;
141 failed interval attempts are retained. The complete TPSS mixed NMR acceptance
(`p255_correlated_nmr`) passes all 405 RKS/UKS Cartesian/channel fits:
p=1.801623..2.147692, minimum R2=0.99939482, with all 123 failed interval
attempts retained. Every accepted TPSS stencil is second order; the
four-step intervals are printed per component. Together with the 810
HF/PBE0 fits this gives 1,215 accepted common-origin NMR entries. The
TPSS unrestricted full parent-tensor limit differs from the independent
magnetic provider by at most 2.948752353404e-13 ppm. This does not certify
London correlated shieldings, pair spin couplings, finite-temperature
response, nonlocal targets or other unmeasured combinations.

The existing `vcd` entry is now connected to
`properties::correlated_axial::correlated_molecular_axial_response` for
custom/named double hybrids. It uses the complete parent nuclear and
magnetic orbital frames and noncanonical PT2 amplitude derivatives, plus
the nuclear ket-basis half-overlap. Its harmonic force field and APT come
from the same target-plus-PT2 energy through `correlated_molecular_taylor`;
the existing nuclear tensor assembler and scalar-generic dispersion paths
are reused. The measured common-origin double-hybrid VCD subset is recorded in response.md. `p256_correlated_axial` compares every AAT
entry with independent displaced real/complex SCF determinants and direct
Slater cross-overlap sums; measured acceptance is recorded in response.md. The independent full Hessian/APT acceptance is `p257_correlated_vibrational`; public VCD cases are in `python/tests/test_correlated_vcd.py`.

`properties::pt2_axial::{PairAxialState, PairStateTangent, PairAxialOverlap}`
provide the reusable normalized first-order pair-wavefunction model. The complete double-substitution tensors avoid a fixed-word orbital limit. Nuclear and magnetic
connections are complete anti-Hermitian MO connections. The zero-field
nuclear tangent is real and the magnetic tangent purely imaginary. The
model uses `M=1+sum_c w_c <T_c|T_c>` and
`Im<d_R Psi|d_B Psi> = (I_parent+sum_c w_c I_c)/M
- M_R Im<Psi|d_B Psi>/(2 M^2)`. SS/OS weights multiply channel bilinears
and their norm; negative custom weights are an explicit signed continuation
and a nonpositive resulting norm is refused. Linear imaginary cancellation is explicitly evaluated; the cross-channel
terms are reduced by the one-body commutator selection rule described below. This is the
normalized MP1 AAT model of the
[analytic pair-wavefunction formulation](https://arxiv.org/abs/2501.06632),
with generic orbital-generating DFT references and explicit spin scaling.

`vcd` retains the controls `gauge`, `gauge_origin`, the reference controls
and generic `double_hybrid_model`, `pt2_same_spin`, `pt2_opposite_spin`,
`pt2_memory_mb`. Common origins and atom-distributed common origins are
explicit; they do not include London basis derivatives. Tensor axes remain
`[3*nucleus + nuclear_axis][magnetic_axis]`, imaginary overlap in e*bohr/hbar.
New diagnostics are `parent_atomic_axial_tensors_electronic`,
`pt2_axial_channel_bilinears` (unscaled intermediate-normalized SS-alpha,
OS, SS-beta bilinears, not additive normalized corrections),
`pt2_wavefunction_norm`, `pt2_wavefunction_norm_derivatives`,
`pt2_magnetic_connections`, `axial_connection_defects`,
`axial_cancellation_defects`, `amplitude_response_residuals`,
`orbital_frame_defects` and `vibrational_response_residuals`.
No finite differences enter these production paths. London pair AATs,
nonlocal/learned correlated field providers and genuine correlated ECD
transition moments remain separate, unfinished work.


`MolecularCorrelatedTaylor` also exports its complete `orbitals`, `occupied` and unscaled `amplitudes` Taylor data. These reuse the already computed moving-metric/nuclear path for wavefunction observables; no second orbital or amplitude solve is introduced. `vcd` returns the complete `hessian` (Hartree/Bohr^2) and `dipole_derivatives` (e, dipole-axis by nuclear-coordinate) on both ordinary and correlated routes.

Pair AAT baseline acceptance (2026-10-24), before the exact commutator
reduction, includes every common-origin Cartesian entry and every unscaled
channel for C1 H4/RKS and C1 H5/UKS:

| Parent | AAT component/channel fits | Accepted p range | Minimum R2 | Failed interval attempts retained |
|---|---:|---|---|---:|
| HF | 405 | 1.920500..2.019506 | 0.99986947 | 0 |
| PBE0 | 405 | 1.938054..2.005580 | 0.99996573 | 1 |
| TPSS | 405 | 1.904521..2.086913 | 0.99981642 | 0 |

The independent oracle converges real displaced SCF determinants and
complex finite common-origin-field determinants, transforms each PT2
amplitude by direct four-AO sums, and evaluates determinants of cross-AO
orbital overlaps. Nuclear steps are [0.02,0.015,0.01,0.0075] Bohr and
field steps are 0.03 times those numbers in atomic magnetic units; the
additional printed intervals resolve any failed fit. No element is copied
from tensor symmetry. The separate HF complete force-field/APT acceptance
passes all 450 entries (369 Hessian elements and 81 APT elements),
p=1.964973..2.135546, minimum R2=0.99976884, without failed intervals.
The expanded all-five-parent VCD acceptance passes all 2,025 AAT/channel and 2,250 Hessian/APT fits.

The production pair contraction now uses the exact fermionic identity
`Im<G_R T|G_B T> = -Im<T|G([K_R,K_B])T>/2` for anti-Hermitian orbital
connections. `G(K)T = Tr_occ(K)T - pair_action(K,T)` reuses the same
occupied-minus-virtual tensor operation as the complex PT2 amplitude
equations. Nuclear/magnetic amplitude cross terms and both contributions
to the magnetic Berry connection, `G(K_B)T + T_B`, are retained. Thus no
triple-substitution state expansion is allocated: each contraction scales
as O(N^5) with O(N^4) coefficient storage. `cancellation_defect` measures
the explicit single-space linear cancellation and the commutator's
anti-Hermitian defect; the cross-channel reduction is an exact one-body
spin-sector selection rule tested by the full Slater-overlap oracle.

`PairAxialState::with_memory_limit(dimensions, occupied, amplitudes,
memory_mb)` checks the estimated coefficient/matrix workspace before
allocation. `new` uses the existing default memory budget. The molecular
driver forwards `pt2_memory_mb`; no fixed orbital-count mask is introduced.
The exact contraction passes the full-crate/all-parent regression,
including SVWN5 and PBE parents. A preflight caught an omitted amplitude
Berry term during this optimization; it was restored without changing any
fit threshold, and a normalized two-state closed-form/four-step regression
plus a field-dependent global-phase limit were added. The failed run is
retained outside the source package.

AAT labels now explicitly include hbar: `e*bohr/hbar` (hbar=1 in atomic
units), and `pt2_magnetic_connections` is `e*bohr^2/hbar`. The latter's
length-square factor is essential to ASE conversion and to reconstructing
the normalized AAT from its reported channel/norm diagnostics in either
unit system. The numerical AAT convention is unchanged. Public acceptance
includes this reconstruction and the physical conversion of every tensor.


### Correlated TDA transition API and kernel acceptance

`response::correlated_tda::correlated_tda_states(parent, basis, scf, eri,
problem, pt2, coefficients, nstates, memory_mb)` returns
`CorrelatedTdaResult`. The parent is the existing generic `TdaSolver`;
`problem` and `pt2` are the same stationary spin-resolved full-Coulomb PT2
objects used for energy. Reference occupations/energies must agree.
The combined parent-resolvent and tensor workspace is checked before
allocation. Semicanonical occupied-virtual gaps remain physical; no
within-space or response denominator is floored. Real collinear RKS and
UKS references are supported by this model; exactly degenerate parent
blocks use symmetric metric transport, including off-diagonal overlap
terms, rather than independent scalar normalization. The entire block is
returned if a requested root count cuts through it. Individual residue
vectors within a degenerate block use that normalization gauge; subspace
observables are the invariant quantities of this truncation.

The result contains `states` (including all three electric, momentum and
angular-momentum components), `parent_energies`, `energy_corrections`,
`singles_correction_norms`, signed `doubles_norms`, `pole_residuals`,
`normalization_defects`, `smallest_double_denominators`, `parent_residual`
and `model`. Pole residuals are freshly evaluated second-order coefficient
equations, not residuals of a fully resummed frequency-dependent response.
Normalization defects are also through second order. Strengths use the
reported corrected energies and moments in the existing line-strength
formulas, without re-expanding their products. Large correction norms
remain visible; numerical convergence does not establish perturbative or
chemical accuracy. The model retains the adiabatic parent kernel and a
normal-ordered residual interaction, and is separate from the stationary
correlated static-response and London-origin models.

The same `ecd-spectrum` / `ecd_spectrum` / `get_ecd_spectrum` entry accepts
existing `double_hybrid_model`, `pt2_same_spin`, `pt2_opposite_spin` and
`pt2_memory_mb` controls. `nstates`, `td_spin`, `td_tolerance`, broadening
and energy-window conventions are unchanged. The report adds the
`pt2_transition_*` diagnostics and correlated ground-energy decomposition.
Native transition dipoles are e*bohr, momentum coefficients 1/bohr and
angular-momentum coefficients dimensionless (the real coefficient of
the anti-Hermitian r-cross-gradient operator). ASE converts the first
two to e*Angstrom and 1/Angstrom. Existing rotatory atomic/cgs and spectral
line conventions are unchanged. Coupled double poles are named errors;
no causal resummation or distinct-target excitation kernel is inferred.

The independent complete-determinant oracle in
`tools/oracle/pt2_propagator.py` freezes all 452 nonstructural A/B/metric,
singles-doubles coupling and electric/time-odd property-gradient
derivatives. Production contractions agree with the direct commutators
within 2e-14. Every derivative fits four signed interaction strengths
[0.16,0.12,0.08,0.06], p=1.991859..2.012248,
minimum R2=0.99999757; no failed interval is removed or floored.
The weighted coupling is adjoint under the SS/OS pair metric, and signed
coefficient linearity passes. The molecular `p258_correlated_ecd` run
passes every HF/SVWN5/PBE/TPSS/PBE0/synthetic-learned RKS H4 and UKS H5
case, with zero-correlation transition-vector defects at most 3.609e-16
and coefficient-linearity defects at most 4.441e-16. Both electric and
magnetic moments change when PT2 is enabled. B2PLYP water/water-cation
checks also pass. The full dense pole fits and public native/ASE/CLI acceptance pass; these measurements do not upgrade unmeasured combinations.

Additional common-origin PT2 NMR acceptance covers all LDA/GGA RKS H4 and
UKS H5 entries, including every unscaled spin channel and mixed magnetic
contact term. These 810 independent mixed-energy fits bring the accepted
HF/SVWN5/PBE/TPSS/PBE0 total to 2,025. The leading interval is
[0.04,0.03,0.02,0.015] in the dimensionless mixed path parameter; all
printed finer/coarser intervals and failed attempts remain available.

| Additional parent | Complete fits | Accepted p range | Minimum R2 | Failed intervals retained |
|---|---:|---|---:|---:|
| SVWN5 | 405 | 1.800247..2.134919 | 0.99937729 | 243 |
| PBE | 405 | 1.823402..2.163056 | 0.99950894 | 240 |

The shared relaxed-density refactor also passes the complete
`p251_collinear_correlated_alpha` regression, preserving its independent
four-step component fits and zero-coefficient parent limits.

All 80 independent pole-energy/electric/momentum/magnetic transition
components pass the dense generalized-eigenproblem interaction-strength
fits: p=1.855798..2.126792, minimum R2=0.99948066. One PBE0/RKS momentum
entry fails the leading interval (p=2.221595) and resolves on
[0.05,0.0375,0.025,0.01875]; that failed interval is retained. Nine further
component fits validate the full degenerate normalization metric against
an independent dense inverse square root. The phase convention is fixed
by the largest parent electric transition component, then momentum or
angular momentum if the electric vector is exactly zero. All amplitudes
and moments change sign together; intensities do not change. Fully dark
states retain an unobservable eigenvector phase. The parent anchor keeps
the perturbative vectors continuous as correlation coefficients vary.
An initial native/ASE/CLI run exposed only this arbitrary-sign mismatch;
the common TDA phase routine now fixes it without changing fit tolerances.
The raw failed run is retained, and the phase-corrected public rerun passes all 28 tests.

Dense parent resolvents and tensor workspace have one combined budget.
Reported pole/normalization residuals are the second-order coefficient
equations. London, learned/nonlocal correlated magnetic providers, PT2 J
multiplets and causal resummation remain separate work.

Final bounded transition acceptance passes 20 public/metadata tests on the
latest one-electron-limit implementation. The broader phase-corrected
public run passes all 28 cases, including eight correlated orbital-NMR
checks. Native/common driver, ASE atomic/physical units, canonical CLI,
ordinary excited-state regressions, ledger/API documentation and warning-
as-error lint gates pass. A Pauli-forbidden repeated-hole or repeated-
particle same-spin substitution is excluded exactly from the propagator,
including denominator diagnostics; it cannot create a false double pole
in a one-electron state. An empty active doubles space reports null
denominator entries through the driver. The independent one-electron
limit, all 452 kernel/property-gradient fits, all 80 pole/moment fits,
nine degenerate-metric fits and the parent-anchored phase regression pass.
The release-feature artifact audit reports four files with no private
build roots. No release of the complete response program is inferred.

LDA/GGA London magnetic energy Hessians also pass all 180 added
Cartesian/channel fits, bringing the five-parent total to 450:

| Additional parent | Complete fits | Accepted p range | Minimum R2 | Failed intervals retained |
|---|---:|---|---:|---:|
| SVWN5 | 90 | 1.802151..2.169804 | 0.99942876 | 169 |
| PBE | 90 | 1.800689..2.189245 | 0.99917439 | 156 |

The first LDA/GGA runs fail before reaching their weak-field asymptotic
range (16/12 restricted components unresolved). Their small parent gaps
are printed; PBE/RKS has a 0.02291652636862479 Hartree occupied-virtual
gap. The new interval [0.00125,0.0009375,0.000625,0.00046875] atomic
magnetic units resolves the remaining components without changing any
operator, denominator or fit gate. All four earlier intervals and both
failed runs remain recorded. This pure-B London acceptance does not yet
validate London B/nuclear-moment contacts or London pair AAT curvature.

The next magnetic connections remain explicit work: London mixed NMR and
pair AATs, nonlocal/learned complex field coefficients, and the correlated
FC/SD/PSO/DSO nuclear-pair response. They are not complete from this
bounded ECD/VCD/orbital-NMR milestone.

### Correlated nuclear-moment response and learned field provider

All three existing coupling/NMR entries accept the ordinary double-hybrid
controls (`xc=b2plyp` or an explicit parent and PT2 coefficients). Reports
add `coupling_parent_tensors`, `coupling_target_tensors`,
`coupling_correlation_tensors`, `coupling_same_spin_alpha`,
`coupling_opposite_spin`, `coupling_same_spin_beta` (atomic units),
`coupling_response_converged`, `coupling_response_residuals` and
`coupling_path_residuals` (dimensionless),
`coupling_amplitude_residuals` (Hartree in alpha^2-scaled nuclear-source coordinates),
and `coupling_stationarity_defects` (Hartree). The model is a spin-free
closed-shell energy Hessian; longitudinal response is not substituted for
the transverse response required by an open shell. Full source, amplitude,
orbital and DSO relaxation is retained. No new calculation identifier is
needed. Measured scopes and raw residuals are in response.md.

The learned complex field provider evaluates the entire global network
with the tau-|j|^2/(2 rho) chain rule; orders 0..7 are connected. Its
38 independent energy/full-Fock coefficient fits and 810 full mixed-NMR
SCF/PT2 fits pass on RKS/UKS and common-origin/London gauges. Nonlocal two-endpoint magnetic
providers and London pair VCD are separate unfinished combinations.

`properties::correlated_spin_spin::correlated_spin_spin_couplings(...)`
returns `CorrelatedSpinSpin`. Its `response.value` is `SpinSpinCouplings`;
`parent`, `target`, `correlation`, `same_spin_alpha`, `opposite_spin` and
`same_spin_beta`, `parent_nonlocal` and `target_nonlocal` have the identical
four-term tensor structure. The two nonlocal terms are weighted parts of
the corresponding parent/target totals. Raw
`path_residuals` are dimensionless; `amplitude_residuals` and
`stationarity_defects` are Hartree. The amplitude residuals use alpha^2-scaled
nuclear-source coordinates.
The arguments specify the parent, optional distinct target, molecular grid,
stationary reference, options, generic PT2 coefficients, ordered distinct
nuclei, contact-only switch, memory budget and solver configuration.

`response::spin_spin::nuclear_magnetic_sites(mol,basis,nuclei,contact_only)`
returns `Vec<NuclearMagneticSite>`. Fields: `atom` (index), `contact`
(unscaled AO delta-function matrix), `triplet[3*k+a]` (FC+SD including
(g_e/2) alpha^2, electronic spin k and nuclear moment a), and `orbital[a]`
(real antisymmetric PSO matrix with -i alpha^2 removed). The factory is
shared by ordinary and correlated calculations and checks the all-electron
basis and unique atom indices. Nuclear sources are expressed in atomic
magnetic-moment units in a fixed real AO basis.

Measured magnetic checkpoint (2026-10-24): 2025 full London PT2 NMR fits, 810 global learned mixed-NMR fits and 925 full FC/SD/PSO/DSO PT2 J fits pass. Raw residuals, failed/resolved intervals, model conventions and the bounded open combinations are in response.md.

### Explicit auxiliary field energy models

`response::field_path::FieldEnergyModel::new(&functional)` selects an
explicit semilocal/exact-exchange target. `.with_vv10(coefficient,params)`
returns a validated weighted two-endpoint model;
`.with_learned(Arc<SkalaModel>)` attaches the entire supplied graph.
`.functional()` returns its functional provider. Target auxiliaries are
independent of the parent. `None` means the complete parent energy model.

The low-level advanced variants are
`field_path::field_wavefunction_path_for_model`,
`correlated_field::correlated_field_taylor_for_model`,
`properties::correlated_magnetic::correlated_molecular_nmr_response_for_model`
and `properties::correlated_spin_spin::correlated_spin_spin_couplings_for_model`.
They take `Option<FieldEnergyModel>` in the existing target position and
share the compatibility wrappers' equations, units and diagnostics.
`FieldWavefunctionPath` additionally returns `parent_nonlocal_energy` and
`target_nonlocal_energy`: factorial-normalized, weighted Hartree
coefficients on the same path. Other orbital/Fock and target-energy fields
keep their existing conventions.

The existing NMR/J calculation identifiers and Python/ASE methods accept
the registered double-hybrid recipes with explicitly composed nonlocal
targets. Custom registered VV10 targets use the declared parameters through
`double_hybrid_target_xc`; a parameter coefficient is explicit in the
Rust model. These nonlocal molecular routes are code-connected and under
independent coupled/public acceptance; do not infer validation from the
earlier semilocal or learned magnetic cases. The first unified-body
nonlocal kernel passes 38 full-component fits and a U(1) defect
6.606856988584e-20. Measured intervals/residuals are in response.md.

`ScfOptions::nonlocal_grid(&molecule)` builds the shared dedicated VV10
quadrature. `vv10_grid_weight_cutoff` is a finite nonnegative weight
threshold (default 1e-14); the molecular driver copies
`grid_weight_cutoff` to it. Zero keeps all points in derivative acceptance.
`vv10_grid_level` retains its meaning. The finite magnetic SCF result adds
`e_nlc` (Hartree), `nonlocal_parameters` and `learned_functional`; the
existing field-energy report exposes `field_nonlocal_correlation`
(Hartree). The older force routine refuses these auxiliary providers until
their moving nuclear/grid/phase derivatives are connected.


The nonlocal magnetic-energy Hessian passes 408 independent HF/PBE0
RKS/UKS component/channel fits, with independently varied parent and target
VV10 terms. Mixed nuclear-moment acceptance remains separately tracked.
The existing NMR reports add `parent_nonlocal_shielding_tensors` and
`target_nonlocal_shielding_tensors` (ppm); J reports add
`coupling_parent_nonlocal_tensors` and `coupling_target_nonlocal_tensors`
(atomic units). Each term is weighted and already included in its parent
or target total. Raw J `coupling_amplitude_residuals` are Hartree in the
alpha^2-scaled source coordinates; `coupling_path_residuals` contain only
dimensionless orbital-frame/projector defects. Both enter convergence.
Detailed measured intervals and model limitations remain in response.md.


### Complete correlated force-constant tensors

The current force-tensor public/ordinary/unit/metadata/API/ledger/Ruff/
packaging suite passes all 39 cases with Python warnings as errors and
strict units. The independently fitted physical tensors, original
failures and resolved intervals retain their separate evidence.

Raw complete-tensor maxima in this subset: orbital response
2.810776e-15, absolute amplitude 2.775558e-15 Hartree, relative amplitude
9.786505e-13, stationarity 4.973799e-13 Hartree and unprojected acoustic
defect 1.850076e-12 Hartree/Bohr^order. Full per-reference values and
the independent stencil fits are recorded in response.md.

The bounded HF-parent/PBE-target complete-tensor acceptance contains
44152 independent Cartesian/pattern component/channel fits across
RKS/UKS. Failed original intervals and resolved four-step fits are
retained in [response.md](response.md#complete-correlated-nuclear-tensor-connection-2026-10-26).
Correlated property `energy_components` now comes from the same complete
target/PT2/dispersion decomposition as `energy`, not the parent SCF
component map. This acceptance does not certify other named models or
references without their independent tests.

The common driver bounds both numeric buffers and typed report tensor
values through `incore_mb`; the PT2 coefficient workspace additionally
uses `pt2_memory_mb`. Oversized exports return the required/allowed MiB
before allocating the output tensors. Coordinate blocks or pattern
tensors provide bounded exports for larger molecular systems.

The existing `force-constants` calculation now uses `driver::run` on all
surfaces. `response_order=3` or `4` selects actual energy derivatives,
without factorial normalization. Omit `coordinates`/`patterns` for all
`(3N)^order` row-major Cartesian elements. `coordinates=[...]` selects
unique IDs `3*atom+axis` in the supplied axis order; `force_coordinates`
is its CLI compatibility alias. `patterns[pattern][atom][xyz]` selects a
complete tensor on arbitrary Bohr displacement patterns;
`nuclear_directions` is its CLI compatibility alias. Select either a
coordinate list or patterns. Legacy ordinary plain-text CLI output and
low-level Rust response functions remain available.

Integer RKS/UKS double hybrids and custom SS/OS models use the same full
moving-basis/grid, metric-transport, orbital and noncanonical amplitude
coefficients as the contracted exports. Parent and target stay generic.
Additional flat tensors and roundoff arrays use prefixes `parent_`,
`target_`, `pt2_`, `pt2_same_spin_alpha_`, `pt2_opposite_spin_`,
`pt2_same_spin_beta_`, `dispersion_`. SS/OS are unscaled;
`tensor = target_tensor + pt2_tensor + dispersion_tensor`. Parent/target
exclude the separately reported empirical dispersion. Coordinate units
are Hartree/Bohr^order; supplied dimensional patterns define dimensionless
path parameters and the corresponding tensor unit is Hartree. ASE
converts these labels to eV/Angstrom^order or eV, including input patterns
and `reference_field` consistently. Native compatibility wrappers retain
a summary `units` string and authoritative `unit_labels`.

The shared multi-channel tensor assembler evaluates independent scalar
energy components and reuses intrinsic permutations. Correlated exports
therefore report `permutation_defect=None`, `permutation_comparisons=0`
and `assembly=energy-polarization`. `directional_evaluations` counts
unique cached analytic paths; `polarization_requests` includes requests
served by that cache. Every channel has its own unprojected acoustic
`translation_residual` when all Cartesian coordinates are present;
partial blocks/patterns retain None and zero comparisons. All legs and
Cartesian translation axes are measured, without an acoustic projection.
Absolute/relative amplitude, orbital, metric, projector, stationarity,
number and reference residuals remain part of the common report.

Finite electric bias is supported. Complete correlated finite-magnetic,
ROKS/ROHF, GKS/GHF, moving parent/target nonlocal nuclear and periodic
combinations remain separate providers. Ordinary empirical/solvent surface
terms are refused on the ordinary complete-tensor route while their
coefficient provider is absent; they are not omitted from an energy total.
The correlated route includes its explicit analytic empirical dispersion.
Independent full-component acceptance and measured subsets are recorded
in response.md; an implemented export does not imply that every tensor,
functional/reference or physical combination has passed acceptance.

### Correlated high-order molecular exports

<!-- Complete correlated nuclear acceptance: prepared -->

The complete four-parent PT2 cubic/quartic acceptance in `p312_correlated_complete_nuclear` is prepared but has not yet been compiled or numerically run. It requests C1 H4/STO-3G restricted HF/PBE/TPSS/PBE0, zero external field, grid level 0 with `grid_weight_cutoff=0`, and custom unregularized SS/OS coefficients 0.37/0.61. Every actual common-driver Cartesian tensor is checked along 364 cubic and 1,365 quartic unisolvent directions, independently for total, parent, target, weighted PT2 and unscaled SS-alpha/OS/SS-beta: 48,412 fits in total. The oracle solves fresh displaced SCFs and directional Hessian response equations. Cubic and quartic tensors use hartree/bohr^3 and hartree/bohr^4 respectively; the contracted directional Hessian uses hartree/bohr^2. Each curve uses four distinct steps, with the unchanged absolute exponent gate 0.2 and R-squared above 0.999; all failed intervals remain recorded. Every displaced orbital and PT2 response, projector, stationarity, metric, number, frame and reference diagnostic must be finite and within 1e-12. Returned original-equation and inner-correction residuals are separately checked; the unfloored reconstructed-source ratio stays visible. Test preparation and a closed-form fit-auditor check are not numerical molecular acceptance. Existing measured subsets and matrix cells remain unchanged until the complete requested runs finish.

The current nuclear diagnostic/public subset passes all nine cases: complete RKS cubic and UKS quartic custom HF-parent/PBE-target matrices on native/common/ASE/canonical CLI, plus API/ledger/Ruff checks. This checks the new relative and unchanged absolute residual exports. The broader named-model nuclear rerun is separate and remains in progress.

`amplitude_response_residuals` retains the fresh maximum absolute PT2
equation residual in Hartree. `amplitude_relative_residuals` additionally
reports, per order, the maximum over spin channels of
`||A0 t_n - rhs_n||_2 / ||rhs_n||_2`, where
`rhs_n = g_n - sum_(k=1..n) A_k t_(n-k)`. The full physical equation
is recomputed after the solve. No norm floor or modified denominator is
used: a zero source requires zero residual. High-order convergence applies
the requested dimensionless tolerance to this relative residual while
retaining the unscaled absolute measurement. Large Taylor coefficients
therefore do not fail solely because of the units or source magnitude.

The existing static `beta`/`gamma` entries accept named double hybrids
or a generic parent, optional distinct target and SS/OS coefficients on
integer RKS/UKS references, including `reference_field` electric bias.
They return all 27/81 Cartesian elements from full parent-orbital and
noncanonical amplitude energy coefficients through orders three/four.
The same polarization and tensor-summary helpers are reused. No new
calculation identifier is introduced. `assembly=energy-polarization` names
the actual calculation; `tensor_assembly=projector` remains an accepted
compatibility request. Parent-only variational 2n+1 assembly is not used
for a nonstationary post-SCF target.

Additional tensors are `parent_tensor`, `target_tensor`, `pt2_tensor`,
`pt2_same_spin_alpha_tensor`, `pt2_opposite_spin_tensor`,
`pt2_same_spin_beta_tensor`, `parent_nonlocal_tensor` and
`target_nonlocal_tensor`, in e^3*bohr^3/hartree^2 for beta and
e^4*bohr^4/hartree^3 for gamma. SS/OS tensors are unscaled;
total=target+weighted PT2. Nonlocal tensors are weighted parts already in
their corresponding totals. Target auxiliaries remain explicitly composed.
`energy_permutation_defect`/`intrinsic_permutation_defect` are null and
their comparison counts zero: scalar-energy permutation reuse is not an
independent observed-dipole-leg Maxwell measurement. Native legacy electric
wrappers retain a summary `units` string and authoritative `unit_labels`;
the common driver/CLI returns the labels in `units`.

The existing `cubic-fc-along`/`quartic-fc-along` entries also accept these
correlated collinear references. Every remaining Cartesian matrix element
is returned, with `probe` and optional `second_probe` as leading contraction
patterns. Parent, target, PT2, unscaled SS-alpha/OS/SS-beta and dispersion
matrices use the corresponding prefixes on `contracted_force_constants`.
Units are hartree/bohr^3 or hartree/bohr^4 per dimensionless patterns;
total=target+weighted PT2+dispersion. Parent/target matrices exclude the
separately returned empirical dispersion. The static electric bias is
`reference_field`; the finite-London `field`/`bfield` correlated nuclear
combinations remain separate. This contracted export does not complete the
uncontracted correlated Cartesian rank-three/four API.

All routes retain response, amplitude, projector, stationarity, metric,
particle-number, frame and reference diagnostics. Amplitude/stationarity
and reference defects use Hartree, number defects electrons, frame/metric
defects dimensionless; coefficients use the reported normalized atomic-unit
path coordinates. `pt2_memory_mb` also bounds the high-order integral and
amplitude workspaces. ROKS/ROHF and spinor GKS/GHF correlated constraints,
moving molecular nonlocal nuclear providers (both parent and target,
including omegaB97M(2)), ECP/solvent/RI paths and
finite-frequency correlated kernels retain their own open coverage.


### Common-orbital/spinor high-order scope and SCF diagnostics

On the existing `force-constants` call, `reference=roks/rohf/gks/ghf`
selects the common-orbital or full complex-spinor provider through order
four. Tensor/block/pattern selection and units follow the existing call.
`parent_tensor` and `dispersion_tensor` sum to `tensor`. Reference,
response, projector and nuclear-stationarity defects retain their raw
values, alongside the complete-Cartesian acoustic residual; a failed
gate sets `converged=false`. GKS accepts the prescribed Hermitian
`gks_operator_real`/`gks_operator_imaginary` source; a fixed AO source
does not imply a moving nuclear SOC model. Constrained/spinor PT2,
nonlocal/learned spin kernels and arbitrary-q periodic high-order tensors
remain separate. Independent all-component/public acceptance is in
progress; retained SCF and quartic failures and measured subsets are in
[response.md](response.md#common-orbitalspinor-tensors-and-scf-globalization-2026-10-26).

ROKS Newton optimization uses compensated quadratic finite-density work
when the provider is quadratic. After complex SCF stagnation, energy
globalization remains active for CDIIS proposals and backtracked convex
canonical updates. Rust providers may declare exact density-quadratic
energies with `SpinorEnergyBackend::quadratic_density_energy` (default
false); nonlinear providers retain actual energy evaluations. This
declaration changes numerical work evaluation only. Final physical
Fock, number, occupation, density, residual and energy gates are unchanged.
Globalization returns to local CDIIS after a factor-100 physical residual
reduction, retaining the same final convergence gates. The retained 12-case
SCF suite passes at 1e-14 reference tolerances; the four-route gradient/API/
ledger/Ruff suite passes nine tests. Raw numbers and failures are recorded
in response.md. CLI iteration energies, residuals and accepted fallback
steps go to stderr.

`periodic::spinor_scf::SpinorDftBackend` provides
`SpinorEnergyBackend::orbital_kernel` at a fixed periodic reference
density. The kernel applies the Hartree, full- and long-range exchange,
and noncollinear LDA Fock differential to one Hermitian density change
per k point. The same provider is available to the periodic spinor
trust-region augmented-Hessian solver for integer occupations. The
output is one Hermitian Fock change in Hartree per k point. The finite-q
spinor response also constructs the projected GGA and meta-GGA Hessian
at the converged reference. Its managed memory includes the cached
local descriptor matrices.
[Fixed-mesh acceptance](response.md#fixed-mesh-periodic-spinor-density-kernel)

For correlated molecular nuclear outputs, `parent_nonlocal_tensor` and
`target_nonlocal_tensor` (or the corresponding gradient, Hessian and
`contracted_force_constants` prefixes) give weighted nonlocal parts
already included in parent/target totals. They use the same units as
their total derivative. Rust's
`response::nuclear::correlated_molecular_taylor_for_model` accepts the
same typed `FieldEnergyModel` as electric response. Full moving-pair
and public nuclear acceptance remains pending; no complete nonlocal
functional/reference coverage is inferred from the new fields.


### Common-orbital energy and explicit correlated target providers

`response::nuclear::open_shell_nuclear_energy_taylor(mol, basis,
functional, grid, scf, orbitals, options, direction, order, config)` returns
`Result<OpenShellTaylor>`. `direction: &[[f64;3]]` has one Bohr displacement
per atom and defines `R(t)=R(0)+t*direction`; `order` is 1..=4. Supply the
converged common spatial orbitals returned by `run_roks_with_orbitals`,
not independently rotated alpha/beta canonical frames. The optional grid
is required for semilocal terms and must match the SCF energy. The
reference electric bias is in `ScfOptions`; this entry changes nuclear
coordinates only. It solves rotation equations through `floor(order/2)`
and retains all geometric/metric orders. `energy[n]` is the
factorial-normalized Hartree coefficient, so an actual contracted nth
derivative is `n! * energy[n]`. `solved` records the highest solved
rotation order; `flag_defects` checks both projectors and their mutual
orthogonality, `stationarity_defects` checks solved commutator orders,
`residuals` gives relative response residuals, and
`reference_stationarity`/`reference_energy_error` retain the zero-order
Hartree defects. Inspect `converged` and raw defects. Additional
nonlocal/learned/ECP/solvent/RI/operator kernels and constrained PT2 have
their own open coverage. The empty open class has the restricted limit.
Complete tensor assembly and its still incomplete acceptance are described
above and in response.md.

`response::nuclear::correlated_molecular_taylor_for_model(molecule,
basis, parent, target, grid, reference, options, coefficients, direction,
order, config)` returns `Result<MolecularCorrelatedTaylor>` for integer,
zero-temperature molecular RKS/UKS references, order 0..=4.
`target: Option<FieldEnergyModel<'_>>` explicitly composes the distinct
target functional, weighted VV10 parameters and optional shared learned
graph. The `None` target means the responding parent energy. The parent
reference, quadratures, basis and `ScfOptions` must describe the same
calculation. `coefficients: Pt2Coefficients` supplies same/opposite-spin
weights; `MolecularDirection` contains nuclear and electric path directions
in atomic units. Geometry, metric, full orbital and noncanonical pair
amplitudes respond through every required order. The result retains
orbital frames, parent/target/PT2/total coefficient channels and their
raw response/amplitude/particle/frame diagnostics. New `parent_nonlocal`
and `target_nonlocal` coefficient vectors are weighted Hartree parts
already contained in their parent/target channels. They are diagnostics,
not extra summands. The original `correlated_molecular_taylor` signature
remains a wrapper for a functional-only target and its defining VV10
parameters; use the explicit model entry for independently supplied
target auxiliaries. Moving nonlocal nuclear all-component/public acceptance
remains pending, so this connection does not certify all combinations.

`response::field_path::FieldEnergyModel::nonlocal(&self)` returns
`Option<(f64, Vv10Params)>`: the dimensionless energy weight and declared
VV10 parameters. `learned(&self)` returns
`Option<&Arc<SkalaModel>>`, borrowing the explicitly supplied graph without
copying it. These read-only accessors let nuclear and electric providers
share one typed composition; they do not infer a model from its name.

The latest complex-SCF globalization uses occupied-subspace conjugate
gradients with Cayley density increments and the same energy/Fock backend;
the expanded retained set passes 18 cases at unchanged 1e-14 reference
gates. ROKS projector coefficients retain summation remainders throughout
the physical density/stationarity path. Full quartic/public acceptance is
still in progress: the preceding optimized public suite passed 12/14 and
retained two ROKS quartic convergence failures. See the dated raw results
in response.md; no broader functional/reference coverage is inferred.

The two previously failing ROKS quartic public selections now pass all
four routes with their original gates: eight reports have worst relative
response 7.860938497533e-12, raw projector
6.058451752097e-28, stationarity
9.730564927445e-12 Hartree and reference
1.783486552887e-13 Hartree. The internal rotation solve is
stricter when needed to satisfy both requested relative and absolute
residual bounds. Physical operators and public thresholds are unchanged;
complete-component/functional acceptance is tracked separately in response.md.


### Structured quantum-particle and positron correlation contract

The existing multicomponent entry accepts either legacy quantum_atoms,
quantum_masses and positron_positions, or a particles list, never both.
Each object has kind=nucleus, positron, muon, negative-muon or particle;
atom is a zero-based owner and position has three Cartesian coordinates.
A nucleus replaces its matching classical point charge. Other kinds add a
distinguishable Hartree orbital. A custom particle requires charge and mass.
Optional exponents, lmax, correlation and label override the shared defaults.
Masses are in electron masses; atomic units use Bohr and inverse Bohr squared.
ASE mode converts nested positions/exponents from Angstrom/inverse Angstrom
squared once, using the same dotted control-unit registry as scalar controls.

positron_correlation=drummond-2011 (alias qmc-2011) selects the local
dilute-positron relaxation energy; none selects the Coulomb-only model.
Each particle can instead specify its correlation explicitly. Distinct
correlation channels add their electron potentials and kernels; particle
densities within a channel are summed. The electronic functional is generic.
The model is a zero-positron-density impurity approximation, not a general
finite-positron-density or intrinsic vibronic theory. Laboratory COM kinetic
energy, fixed unselected nuclei and distinguishable-particle assumptions remain.

annihilation_enhancement accepts ipm, drummond-2011, barbiellini-ap,
barbiellini-sl and barbiellini-qmc. Optional nonnegative dimensionless
annihilation_gradient_alpha applies the gradient damping; zero is exactly
the undamped model. annihilation.positrons contains contact_density and
enhanced_contact_density (Bohr^-3 natively), rate (inverse atomic time),
rate_per_ns (ns^-1) and lifetime_ps (ps), per normalized positron. The
spin-averaged convention is lambda=pi*alpha_fs^3*int(n_+ n_- gamma).
ASE converts contact densities to Angstrom^-3; the atomic-time, ns and ps
rates/lifetimes retain their labelled units.

Independent p281 component evidence includes 16 four-step local kernel-row
fits, seven high-precision point/term oracles through density 1e-300 and five
closed Gaussian annihilation integrals (maximum enhanced-contact defect
3.119726699197e-14 Bohr^-3). These are local/component checks, not molecular
all-component derivative acceptance. All four full-crate p281 cases pass;
the two none/QMC public cases pass native Python, ASE atomic/physical units
and canonical CLI (4.16 s). The retained TPSS electronic SCF failure and
broader regression status are in response.md.


Rust correlation providers use InterspeciesCorrelation::{evaluate,kernel_action}
with total number densities in Bohr^-3, energy density in Hartree/Bohr^3,
potentials in Hartree and a symmetric two-density kernel action. The
CorrelationModel::{new,none,single,channels,len,is_empty,is_zero,descriptions}
container replaces a single fixed proton provider. QuantumParticle has
correlation_channel: Option<usize>; solve, solve_from, constraint, response,
stability and derivative entry points all consume the same model.
correlation::{by_name,interspecies_correlation,positron_correlation} construct
providers; PositronRelaxation::drummond_2011 supplies the published parameters
and relaxation(r_s) returns Delta and its first two r_s derivatives in Hartree.
annihilation::{Enhancement,Annihilation,annihilation} expose contact models,
the AO/grid integral and rate_per_ns(). Enhancement::{by_name,factor} select
and evaluate the contact enhancement. CorrelationPoint stores energy,
electron_potential and partner_potential. All new connection and limiting
evidence is bounded as described above.


### Shared variational SCF globalization

Molecular integer-occupation RKS/UKS uses the same complete energy/Fock
evaluation for iterations, occupied-subspace line searches and final
reports, including full/range exchange, semilocal XC, VV10, learned terms
and the continuum reaction field when present. After stagnation, the
existing optional orbital optimizer can reuse the complex-spinor
limited-memory BFGS/Cayley implementation with independent real spin
projectors. The restricted block has weight two; unrestricted blocks have
weight one. Local DIIS resumes after a hundredfold reduction of the true
orbital residual. A failed line search only restarts DIIS; it never accepts
SCF convergence. Fresh physical Fock and particle-population map checks keep
the requested energy, orbital and density tolerances unchanged.

Rust controls this with ScfOptions::use_soscf; the common CLI/native/ASE
control no_soscf=true disables the optional globalization. This does not
change units or introduce a separate physical calculation. Thermal and
prescribed/selected occupations, and potential-only asymptotic corrections,
retain their existing algorithms. The retained TPSS quantum-potential SCF
and coupled derivative regressions are measured separately in response.md;
connecting an energy term does not by itself validate every response tensor.


The TPSS-family total-density iso-orbital cap uses the physical inward
derivative when upper-bound roundoff would select inconsistent branches
in otherwise equivalent AO contractions. Its capped energy value is
unchanged; inputs genuinely beyond the floating-point enclosure keep the
constant extension. This is a shared functional-body convention, including
its Taylor and dense kernels, without a response-driver name switch.
The independent point and all-component molecular checks are p283;
the resolved retained p223 coupled-Hessian regression and remaining
acceptance are recorded in response.md. API units and requested residual gates are unchanged.


Measured endpoint acceptance: all 266 p283 fits pass (14 point, 18 gradient,
162 Hessian, 18 alpha and 54 beta entries), p=1.955170..2.013108 and minimum
R2=0.99989964. The four failed larger-step mixed-Hessian intervals are
retained. The coupled TPSS directional Hessian on multicomponent now passes
p=1.997884081962, R2=0.999999698518 on h=[0.04,0.02,0.01,0.005] Bohr;
response residual 2.033352129667e-14, reference energy defect
-2.242650509743e-14 Hartree and Richardson defect 5.532513308593e-9
Hartree/Bohr^2. All six p223 cases pass. Ordinary molecular full-tensor
acceptance does not imply complete coupled multicomponent Cartesian,
periodic or correlated acceptance. The shared-SCF public aggregate passes
50 cases; endpoint-specific full tensor route comparisons are recorded in
response.md when measured.


Static polarizability has two deliberately distinct report entries.
polarizability is the response d mu/d F, labelled e^2*bohr^2/hartree;
ASE physical mode converts it to e^2*Angstrom^2/eV. The explicit
polarizability_angstrom3 entry is the volume convention and remains in
Angstrom^3 on every surface. Their native atomic-unit numbers correspond,
but their physical unit conversions must not be interchanged. Beta uses
e^3*bohr^3/hartree^2. The endpoint public test's first attempt incorrectly
used the volume factor for the response entry; the corrected expectation
checks both entries without changing the calculation or a tolerance.


The common molecular meta-GGA response kernel uses a bounded immutable-reference
XC derivative cache. Rust sets its budget through ScfOptions::incore_mb;
the canonical CLI uses --incore-mb and native Python/ASE use incore_mb.
One eighth of this existing memory allowance is available to the local
derivative arrays. Insufficient budget uses the identical on-demand kernel;
there is no new property entry or unit convention. All 2560 fixed-density AO
entry fits across TPSS/revTPSS/M06-L/M06-2X pass, with bitwise cache parity,
p=1.877946..2.154855 and minimum R2=0.99900211. This is local-kernel evidence;
complete molecular, periodic, correlated and public-budget acceptance have
their separately stated scope in response.md. Requested residual gates are
unchanged.

Measured cache-enabled common-driver acceptance passes all 266 p283
component fits and all six p223 cases. The RKS/UKS public budget
comparison passes every alpha entry bitwise on native Python, ASE
atomic mode and canonical CLI at incore_mb=1 and 2048. The complete
endpoint gradient/Hessian/alpha/beta public comparisons pass; maximum
beta route defect is 8.881784197001e-16 atomic units. These are bounded
acceptance claims, not full release-wide regression certification.


The nmr-spectrum route also supports exact isotropic spin mixing. Use
nmr_couplings=true and nmr_spin_model="exact"; nuclear_spins contains twice I,
nuclear_g the signed g factors, and nmr_spin_max_dimension (default 1024)
the complete Hilbert-space resource limit. The first-order mode is the
default. Physical shieldings of all active partners set the Hamiltonian;
nmr_reference_shieldings change only the plotted ppm reference. Exact mode
retains all scalar flip-flop terms and coherent isotope-group transverse
detection. It returns raw eigen residuals in Hz, a dimensionless backward
residual, integrated-area sum-rule defects and an explicit convergence flag.
The common response_tolerance gate is unchanged. Line positions are ppm;
line weights are integrated dimensionless areas. Normalized high-temperature
transverse strengths exclude absolute spin populations, quadrupolar and
anisotropic terms, pulse sequences and relaxation. This does not enable
unimplemented electronic shielding/J reference or functional combinations.
The independent AB/dense/limiting acceptance and current public scope are
in response.md; no full-program completion is claimed.

All existing basis controls now select 78 embedded sets. The additional
thirty version-pinned BSE sets are listed in third_party/bse/NOTICE.md and
catalogue.csv. Rust uses basis::data::load/bundled_names; the canonical CLI
uses --basis pcseg-1, native calls use basis="pcseg-1", and ASE uses
DftRs(basis="pcseg-1"). Units and atomic/physical conversions are unchanged.
All published supported elements are retained; no missing-element fallback
is invented. A DK-optimized basis is not a relativistic-Hamiltonian switch.
Data/term validation is separate from chemical or basis-convergence accuracy.


### Full-spin electronic nuclear J response

The existing `spin-spin-couplings` and `fermi-contact-couplings` calculations
connect the unchanged UKS/UHF reference or the self-consistent complex
GKS/GHF reference to the same full spin/AO response kernel. The current
provider covers all-electron HF, semilocal and full/range-separated hybrid
Hamiltonians with integer populations. FC, SD, PSO and explicit DSO are
retained; `spin_orbit_mixed_tensors` includes both FC/SD-PSO source orders
and is included in `reduced_couplings`. All Cartesian/spin source components
are solved independently. Reference/functional names do not select kernels.

`spin_response_constraint="fixed-spin-direction"` imposes
`Tr[S_perp1 dP]=Tr[S_perp2 dP]=0` using two orthogonal axes transverse
to the reference total spin and a Lagrange-multiplier saddle system.
The total-spin magnitude can relax. A zero/unresolved reference moment
has no spin direction and is refused with instructions to use free
response for a spin-paired state or a separate local-constraint provider
for a compensated magnet. Constraining all three spin components at a
collinear determinant would be singular: its longitudinal Jacobian is
zero despite nonzero second-order expectation changes. That draft
constraint is not used as a finite-field ensemble.
`"free"` (default) requests the unmodified unconstrained response and can be singular
at a Goldstone mode. No energy gap or Hessian eigenvalue is replaced.
The fixed-spin calculation excludes free-spin thermal Curie enhancement.
Common-orbital ROKS, open-shell PT2, ECP, PCM, learned and nonlocal spinor
response providers remain separate; no support is inferred for them.

Reports retain `[pair][M-axis][N-axis]` order and the requested nucleus
ordering. Couplings are atomic units and optional isotropic constants are Hz.
`response_residual` is the fresh relative saddle-equation residual;
`spin_response_constraint_residuals` contains the absolute derivatives of
the original spin constraints, in atomic units. Rank-revelation discards
only numerically dependent constraint projections and reports their norms
as `spin_response_discarded_projection_norms` (dimensionless), together
with `spin_response_constraint_rank`. `response_reference_fock_defect` is
Hartree; angular/cache histories have `[dimensionless, Hartree, Hartree]`
columns. The bounded all-component and public acceptance status is recorded
in response.md; code connection alone is not validation.

Rust low-level entry points are
`response::spinor::spinor_operator_response_with_kernel` and
`constrained_spinor_operator_response_with_kernel`: a generic one-body
operator list and a matrix-free interaction kernel return full complex
susceptibilities, all density-response spin blocks, fresh residuals and
constraint diagnostics. The constrained form also returns multipliers in
the original operator basis. `response::nuclear::spinor_one_body_response`
and `spinor_constrained_one_body_response` attach the shared molecular
spin-covariant functional provider. `response::spin_spin::spinor_nuclear_magnetic_sources`,
`spin_direction_constraints`, `total_spin_constraints` and `spinor_spin_spin_couplings` build physical
operators and return `SpinorSpinSpinCouplings` with the complete Ramsey
decomposition. `response::spinor::embedded_collinear_spinor_reference`
embeds a stationary integer UKS reference without reoptimizing it; it
checks the full-spin Fock, metric, eigen and number residuals. It is not
a substitute for the separate ROKS common-orbital Hessian.


### Molecular ECP nuclear Taylor connection

The existing nuclear/force-constant calculations now obtain the moving
local and nonlocal ECP core coefficients from
`integrals::ecp::ecp_nuclear_taylor`. Basis and ECP centres both move;
effective charges agree with reference SCF. Coefficients are factorial
normalized, orders 0..4, in atomic units. Complete tensors use the existing
`force-constants` controls, canonical CLI and native/ASE methods, with the
existing Hartree/Bohr^order unit conventions. No separate ECP API or
functional-name response branch is introduced. p289 passes all 228 AO/order fits through fourth order; p290 passes
all 936 gradient/Hessian fits across AE RHF/RKS and ECP RHF/RKS/UHF/UKS.
Complete cubic fits pass all six AE/ECP RHF/RKS/UHF/UKS models (2184 independent directions); AE RHF and ECP RHF each pass all 1365 quartic directions. Other complete quartic combinations remain in acceptance. Scalar-ECP ROHF/GHF paths additionally pass all 312 gradient/Hessian fits and all eleven public route checks; see response.md for their separate higher-order counts and retained failures. ROKS/GKS and multicomponent ECP providers remain separate.

Bounded complex-spin acceptance: 193 one-body/constraint fits and 81 C1
GHF electronic Ramsey fits pass (minimum R2 0.99990020/0.99999941).
The fixed-spin-direction UKS/GHF public routes pass with fresh response
residuals 7.814105880151068e-13/9.802464131734452e-12 and reference Fock
defects 1.1102230246251565e-15/9.992007221626409e-16 Hartree.
All component/ensemble restrictions and failed/resolved intervals are
recorded in response.md; ROKS and open-shell PT2 J remain incomplete.

### Finite-q full and range-separated exchange connection

The existing `periodic-response`, `dielectric` and `thermal-response`
calculations use the shared density solver with the reference's same-spin
full/range exchange fractions, exchange cutoff and k mesh. q is Cartesian
inverse Bohr; frequencies are Hartree. Scalar AO density response is per
Hartree, Fourier density is bohr^-3/Hartree, and the dielectric convention
is epsilon^-1=I+v chi with screened chi. Bra/ket spaces at (k+q,k) remain
independent. Distance-selected Gaussian quartets and finite-mesh
minimum-image weights define the exchange model; cutoff/mesh convergence
must be checked. CLI density, field and thermal response share complete
`density spin k bra_AO ket_AO real imaginary` rows in their report density
units, together with the Fourier table. `--grid-weight-cutoff` has the
same nonnegative, dimensionless meaning as native/ASE `grid_weight_cutoff`.

Rust exposes `dfpt::exchange::FiniteQExchange::{new,apply}`: one complex
AO density per k point maps to exchange matrices in Hartree before the
-cx/-clr contraction. The complete Coulomb/erf/erfc kernel passes 12288
four-step entry fits (p=2.002891, minimum R2=0.99999944); its Gamma
reference-Fock defect is 1.665334536938e-16 Hartree. The response equations
use generic functional providers and fractions, without name branches.

Post-repair physical scalar-density acceptance passes all 360 independent
Gamma charge/spin-source AO output fits for restricted HF/PBE0/CAM-B3LYP
on the spin-stable C1 triclinic model. Each of six specified sources fits
all 20 independent AO density outputs. At
h=[0.04,0.02,0.01,0.005] Hartree source coordinates, all fits pass without a
failed interval: p=1.995728514..2.006077233, minimum R2=0.999997493811;
maximum fresh response residual is 6.956075636512e-15. Eighteen
single/two-k-point static/retarded/imaginary dense cases pass with maximum
full-complex density RMS difference 2.991769568875e-12 and raw response
residual 5.640863983520e-11. These are bounded model tests, not a complete
physical q/cutoff convergence certificate.

All 26 native Python/ASE/canonical CLI periodic checks pass. The twenty-four
HF/PBE/PBE0/CAM-B3LYP parent/mesh/frequency cases compare every complex AO entry; maximum
route difference is 6.081411436930e-12 at the unchanged 1e-11 gate, with
maximum response residual 8.142231598842e-13. The default SCF stderr trace
is checked. Returned-reference consistency separately passes 32
RKS/RHF/unequal-population UKS/UHF parent/mesh/temperature cases: every
rebuilt F[P] entry agrees exactly, and independent entropy/Mermin defects
are at most 1.776356839400e-15 Hartree. All 1152 CLI AO entries
are checked exactly once, with missing/duplicate/non-finite rejection.
Eleven existing Rust density/thermal, dimensional/charged, energy/folding/
translation and bounded force-constant regression tests pass again.

The UKS/UHF reference-consistency test does not certify general
spin-polarized response tensors. ROKS/GKS, learned/nonlocal/PT2, dimensional
and other q/frequency/temperature combinations remain separate work. The
retained cutoff-six, spin-unstable and stale-final-field failures, resolved
intervals and raw residual measurements are in response.md.

## Spin-complete pair and common-orbital providers

The following lower-level APIs extend the shared response machinery. Their
Ramsey/public-driver integration and broader physical acceptance remain in
progress in [response.md](response.md#spin-complete-pair-and-common-orbital-work-2026-10-30).
They do not change the advertised scope of an existing coupling calculation.

| Rust entry | Inputs and coefficient convention | Returned values and scope |
|---|---|---|
| `properties::pt2_response::spinor_transform_path` | Spatial `nao`, packed spatial Coulomb Taylor tensors, full complex AO-by-spinor orbital Taylor matrices, semicanonical energies and occupied-spinor count | Spin-summed `(a i|b j)` coefficients `[order][i,a,j,b]`; all spin traces, without a spin-AO fourth-rank expansion |
| `properties::pt2_response::generalized_taylor_response` | Semicanonical energies, occupied count, complex transformed sources and full MO Fock coefficients, including order zero | `GeneralizedPt2Taylor`: unscaled energy/amplitude coefficients and recomputed absolute/relative pair-equation residuals; full spinor substitutions, including complex base integrals |
| `response::nuclear::spinor_wavefunction_path` | Molecular/basis/grid/functional/reference/options context, fixed reference spin/AO operator, added Hermitian core coefficients and solver configuration; added order zero is exactly zero | `SpinorWavefunctionPath`: parent energy/dipole, full spinor orbitals/Fock/density, occupied count, frame/particle-number defects, angular history and physical response residuals |
| `response::nuclear::spinor_correlated_one_body_taylor` | The same context and core path, plus one finite common PT2 coefficient | `SpinorCorrelatedPath`: parent path, spin-complete unscaled pairs, parent plus scaled correlation, imaginary-energy defect; equal SS/OS scaling only, without a distinct target or orientational constraint |
| `response::nuclear::open_shell_operator_response` | Molecular context, converged integer common-orbital ROKS/ROHF state and common orbitals, Hermitian AO sources `[source][alpha,beta]`, `Frequency`, solver configuration | `OpenShellOperatorResponse`: `susceptibility[a][b]=d<H_a>/d lambda_b`, residuals/iterations, reference stationarity and metric defect; constrained common-orbital real/even and imaginary/odd response |

Energies and Fock matrices use Hartree, AO coordinates use Bohr. The path
parameter is dimensionless. Taylor coefficients are factorial normalized,
so the order-n derivative is `n! * coefficient[n]`. Operator-response units
are the product of the supplied operator units divided by Hartree; source
scaling is explicit. Electric polarizability is minus the susceptibility of
the electric-energy source. Explicit observable/contact derivatives must be
added by the physical property provider. Unequal SS/OS scaling needs its own
covariant spin-pair definition; a collinear label is not silently retained
through a transverse rotation. The common-orbital source API does not supply
a transverse spin-adapted ROKS Ramsey tensor. The generalized one-body PT2
provider additionally passes all 108 parent/PT2/total Hessian entries of six
physical two-site Fermi-contact sources in p298; the conditioned coordinate
is lambda=alpha_fs^2*m and the physical second derivative has alpha_fs^4
scaling. This bounded source acceptance does not establish a public
open-shell correlated J calculation.

High-angular nuclear scope: the existing molecular basis controls accept g,
h and i (l<=6), without a separate calculation API. The current C1 HF
acceptance has 1872 complete AE/ECP RHF/UHF gradient/Hessian fits and
776 independent scalar-ECP AO matrix comparisons. Complete cubic
fits pass all 364 directions for AE RHF/UHF g/h/i and scalar-ECP g/h RHF
(2912 directions). AE RHF g/h/i quartics each pass all 1365 directions
(4095 total); h/i include five/three resolved fourth-order stencil fits.
The g AE UHF quartic also passes all 1365 directions (1360 second-order,
five fourth-order, minimum R2=0.99926600); all 34 failed intervals are
retained. Remaining ECP/h/i-UHF quartics require separate complete runs. Independent HF terms and all twelve energy public-route
cases pass. Other higher-order/reference combinations remain in acceptance. Units and full measurements,
including the resolved integral-workspace failure and later optimization
limits, are in [response.md](response.md#c1-ghi-nuclear-component-acceptance-2026-10-30).

The ECP scalar kernel reuses exact sphere moments and repeated primitive
monomial contractions. All 5323 old/new high-angular Cartesian entries,
including raised intermediates for order four, are bitwise equal; all
228 AO/order fits, 624 nuclear regression entries and 776 independent
ECP matrix entries pass. Isolated kernel timings and bounded cache storage
are in response.md; they do not establish a whole-calculation speedup.
On the combined build containing the exact optimized ECP source, all thirteen
native/ASE atomic+physical/canonical-CLI cases pass; maximum independent
energy defect is 1.438849039914203e-13 Hartree.

The nine selected continuation models pass all 12285 independent C1 nuclear
quartic directions: 12245 second-order and 40 fourth-order error fits,
minimum R2=0.999010185261, with all 228 failed earlier intervals retained.
The initial six-model h-boundary set contributes 8190 directions. The three
additional compact C1 s/i HF models (scalar-ECP RHF, scalar-ECP UHF and
all-electron UHF) each pass all 1365 directions on the identical fixed
library, adding 4095 directions. In the i subset, 4077 second-order fits give
p=1.811921818572..2.195963003999 and 18 fourth-order fits give
p=3.888446720385..4.152374561445; minimum R2=0.999109974377 and maximum
reported linear residual9.979730888808e-13. The fixed source/optimization
artifact is identified by the Immutable20 table in response.md. This is
bounded model evidence; the later common response-frame precision extension
requires its own affected regressions. The two ECP PBE RKS/UKS quartics remain
deferred and unvalidated within the full objective. No arbitrary cumulative
basis stack, registered-functional/reference matrix or full-program
completion is inferred.

The default molecular SOSCF iteration can hand off unresolved, repeatedly
stalled projector steps to a fresh physical-Fock/CDIIS sequence. Energy,
density and gradient convergence gates are unchanged; no new control is
required. The retained i-shell UHF displacement now converges in twelve
iterations with physical commutator 5.329070518201e-15 Hartree at the
original 1e-14 energy/density/gradient controls. Its formerly failed cubic
direction passes the original four-step fit (p=1.997920, R2=0.99999993).
The combined native/ASE atomic+physical/canonical-CLI suite passes all
thirteen cases, including strict recovery; independent catalogue energies
agree within 1.438849039914e-13 Hartree. The repaired i-UHF cubic also
passes all 364 directions (363 second-order fits and one fourth-order fit,
minimum R2=0.99907621); all five failed intervals remain recorded.
Other reference/functional and scalar-ECP higher-order combinations retain
their separate acceptance scope.

## Explicit anharmonic treatment selection

`response::anharmonic::AnharmonicConfig.method` selects `AnharmonicMethod::RetainedPolyad` (the compatible default) or `AnharmonicMethod::StrictVpt2`. `AnharmonicMethod::parse` accepts exactly `retained-polyad` and `strict-vpt2`. The returned `AnharmonicSpectrum.method` records the choice.

Strict VPT2 truncates cubic interactions at second order and quartic/Watson terms at first order, and includes normalized transition moments through second order. Reported intensities square those corrected moments; they are not an intensity expansion truncated at order two. Coupled near-resonant reported states cause an explicit error, without denominator floors or an automatic fallback. The retained-polyad method diagonalizes the contact-transformed retained Hamiltonian and can include higher-order contributions through that diagonalization. It remains an explicitly named approximation with its existing resonance controls and validity warnings.

Python and CLI reports return `anharmonic_method`, `energy_order` (the nominal construction order), `energy_treatment`, `transition_moment_order`, `intensity_treatment`, `resonance_treatment`, `eigensystem_kind` and `convergence_scope`. Strict effective-Hamiltonian matrices contain diagonal perturbative levels and identity vectors; zero eigensolver diagnostics do not claim an exact eigensolution of a coupled Hamiltonian. Numerical convergence alone does not establish perturbative validity. Strict `exterior_mixing_norm` denotes the largest first-order wavefunction-correction norm among reported harmonic states, as stated in its warnings; retained-polyad keeps the earlier exterior-transformation diagnostic. Frequencies are cm^-1, IR intensities km/mol and static Raman activities Angstrom^4/amu. A selected-mode calculation omits unselected-mode couplings in either method.

## Collisionless thermal electric response

`dynamic-polarizability` accepts the explicit common control `thermal_response="collisionless"` (CLI `--thermal-response collisionless`). Python `dynamic_polarizability` and the ASE getter forward this control. The default `integer` preserves integer-projector validation. The new opt-in requires a converged positive-temperature real collinear canonical Fermi reference. Its population-weighted Liouvillian retains all ordered unequal-population pairs and both spin channels. Reference populations remain fixed: diagonal occupation relaxation and chemical-potential response are absent, so its zero-frequency limit differs from static isothermal Mermin response.

Rust `response::thermal_dynamic::{ThermalSolver, polarizability}` exposes the parent local/GGA plus exact-exchange kernel. The common driver rejects correlated composite, spinor and constrained common-orbital extensions; kinetic/meta, nonlocal and continuum kernels are also excluded by the core capability checks. A resolved parent functional alone does not encode external correlation metadata. No causal PT2 contribution or public real-time integrator is implied.

Reports include all nine complex tensor entries, residuals, iterations, the actual reference occupations, electronic temperature and spin chemical potentials in Hartree, and per-axis/per-spin response particle-number defects in electrons*e*bohr/Hartree. `population_defect` measures consistency with the reference Fermi populations. Thermal `reference_projector_defect` describes physical non-idempotency rather than a convergence error. Full compiled-artifact acceptance is recorded separately from metadata compilation and independent test-oracle checks.

For MCD, `faraday_a_terms` provides the canonical coefficient of the line-shape derivative in observation frequency, while `a_terms` preserves the legacy signed first-line-moment representation: `faraday_a_terms=-a_terms`. `b_terms` includes the ground-state permanent-dipole subtraction. Complete degenerate levels must be summed; orientation and circular-helicity prefactors are not already applied. Returned `mcd_convention` and unit labels state these distinctions explicitly.

<!-- Retained response coefficients: begin -->
The existing common-driver `static-spectroscopy` APT/Raman/hyper-Raman entries use retained physical coefficient assembly for the connected integer real-collinear parent/target/PT2 path. No new property identifier or user control is needed. Taylor coefficient n is derivative/n!; energy, nuclear/electric units and Cartesian axis order are unchanged. Rust `NuclearTaylor::density_corrections` and `MolecularCorrelatedTaylor::orbital_corrections` expose the low matrix coefficients in the same units as their leading coefficients; keep both parts for subsequent cancellation-sensitive contractions. Raw number, projector, metric, stationarity, orbital-frame and amplitude residuals refer to the retained physical state. Python, ASE and canonical CLI use the same driver. Complete C1 HF+PT2 RKS/UKS static spectroscopy passes all 90/90 cases and 570/570 independent fits: APT90, Raman180 and hyper-Raman300. All 30 hyper-Raman coordinate/reference cases pass, including the previous 27 failures. Every ordered Cartesian slot is assembled; no tensor value is copied by symmetry. Public-route acceptance: lint exit 0, docs exit 0, public exit 0, parent-regression exit 0. Separate PBE/TPSS/PBE0/B2PLYP composition integrations are not inferred from these primary public checks. Overall exit 0. Additional preceding-artifact composition integration: composition-pbe reached its unchanged 2400-second wall-time budget. This is incomplete acceptance, not a measured numerical mismatch; the remaining named compositions were not executed. The original timeout is retained, with no functional-class promotion. The earlier parent-route regression retained a stale uncentered evaluation count (57); centered polarization requires 6*8+3*6=66 actual calls for its nonzero nuclear/three-axis fixture. The independent combinatorial expectation is corrected without changing tensor, residual, permutation or error-fit gates; the initial failed assertion is retained. The final correction-loss tracking is included in this linked acceptance artifact. The identical frozen earlier fixed-population regression passes all five unchanged tests (exit 0, 615.36 s, four Rayon workers). Its original single-worker timeout is retained. This confirms that rerun; it is earlier-artifact evidence, not inferred new-artifact or nonlinear population acceptance.
<!-- Retained response coefficients: end -->


### Periodic particle-number diagnostics

The source common periodic reference report adds `particle_number_diagnostics`:
occupied and overlap-metric electron numbers, the XC-grid integral when available,
signed number errors, relative grid error, and a 0.01 gross-discrepancy advisory.
All are dimensionless particle counts/ratios and retain their values in ASE units.
Unavailable grid quantities and their advisory result are null. The advisory
does not change SCF convergence, rescale density, adapt cutoffs, or certify
physical convergence when absent. Independently converge the one-electron,
interaction, AO/quadrature and k supports before interpreting responses.

The additive helper passes observed LiF/NaCl and unavailable-grid checks.
Combined native/ASE/canonical CLI HF/PBE cases now pass count and unit
consistency, including null unavailable HF grid diagnostics. These checks
validate the report contract and do not certify material convergence.

<!-- Constrained static electric tensors: begin -->
## Constrained static electric tensors

The existing `static-spectroscopy`, `beta` and `gamma` common-driver entries
accept `reference="roks"`/`"rohf"` and `reference="gks"`/`"ghf"`.
`static-spectroscopy` with `electric_order=2` or `3` returns all nine Raman
or twenty-seven hyper-Raman entries for the supplied nuclear direction.
`beta` returns all 27 first-hyperpolarizability entries; `gamma` returns all
81 static second-hyperpolarizability entries. Each ordered slot is assembled
by analytic scalar-energy polarization. Identical homogeneous path
coefficients, including exact reversal parity, are reused; tensor slots are
not filled by symmetry. `tensor_assembly="projector"` and `"energy"` select
this stationary contraction. The separate collinear three-solve
`"variational"` shortcut is not inferred for constrained references.

Rust `response::nuclear::open_shell_molecular_energy_taylor` and
`SpinorNuclearContext::molecular_energy_taylor` consume
`MolecularDirection { nuclear, electric }`. Their energy coefficients are
factorial normalized through order four; public tensors are actual
derivatives without factorial factors. The common-orbital closed/open/virtual
flag and full complex spin/AO density respectively obey the physical
stationarity and metric constraints. GKS retains both density parts through
the same physical core, Hartree, full/range exchange and spin-covariant
density/gradient/tau/current XC providers. Its cached kernel is used only at
its exact expansion center. No functional-name response branch is added.
`SpinorNuclearEnergyTaylor::reference_energy_defect` is the retained
zeroth-order energy minus the input SCF energy, in Hartree; it accompanies
the existing Fock, projector, solved-order and angular diagnostics.

All surfaces use `driver::run`: Rust, canonical `dft-rs static-spectroscopy`
or `beta`/`gamma`, native `static_spectroscopic_derivative`/`beta`/`gamma`,
and ASE `get_static_spectroscopic_derivative`/`get_beta`/`get_gamma`.
Native paths use Bohr, fields Hartree/(e Bohr), and tensor units
e^r Bohr^r/Hartree^(r-1): r=2/3 for contracted Raman/hyper-Raman, r=3/4
for beta/gamma. ASE converts paths, fields, fixed Hermitian spinor sources
and reports through the shared unit registry. The calculator forwards its
SCF `guess` and preserves configured response tolerance/iteration controls;
explicit call controls take precedence. An absent `guess` delegates the
reference-family default (molecular SAD,
periodic core); explicit guesses are forwarded. The native legacy summary falls
back to the authoritative tensor unit label when no duplicate summary exists.
Raw response, projector, stationarity, reference and accumulation diagnostics
remain available; no compatibility wrapper constructs an alternate reference.

Local, full-exchange and range-exchange providers are connected generically.
The measured subset below is biased molecular HF/STO-3G in a C1 geometry,
not all functional classes. Nonlocal, learned, continuum, fractional,
periodic and dynamic combinations retain their separate capability checks
and validation status. Constrained PT2 is rejected until matching amplitude
and orbital providers are connected; no parent-only double-hybrid tensor is
returned. RKS/UKS correlated support remains a separate measured subset.

Acceptance: all434 independent biased C1 HF ROHF/GHF static Raman/hyper-Raman/
beta/gamma fits, original translation gates and public10 cases pass. Other
functional/reference, dynamic, correlated and periodic combinations retain
their separate restrictions. Affected four-class nuclear regression timed
out before measurements and remains unverified; response-cost work continues.
See [response.md](response.md#constrained-static-electric-tensors) for exact
intervals, residuals, model, retained failures and artifact identities.
<!-- Constrained static electric tensors: end -->

<!-- Active descriptor Taylor kernels: begin -->
## Active descriptor Taylor kernels (2026-11-01)

The shared Taylor potential implementation uses two active descriptors for
LDA bodies, five for GGA bodies and seven for meta-GGA bodies. Inactive
potential arrays remain exactly zero in the unchanged seven-descriptor
output contract. The public external-body evaluator retains seven inputs;
`xc::taylor::DirectionalTaylorScalar` remains the original nongeneric
seven-descriptor public alias, preserving constructor type inference;
active descriptor types remain private to the family drivers. Parameter
Taylor paths and the independently varying directional path are retained.
No response dispatch branches on a registered functional name, and no new
CLI, native Python or ASE control is needed. Existing common-driver routes
consume the same providers.
Energy density uses Hartree/Bohr^3; density, gradient-contraction and kinetic
descriptors retain their existing atomic-unit definitions. Taylor entries
are coefficients of t^n (the nth derivative divided by n!); the
physical path fixes the units of t and the potential partials.

The permanent `src/xc/taylor_tests.rs` suite compares every energy, potential
and mixed-potential coefficient against the seven-lane evaluation through
order five, including moving parameters. A warning-strict standalone build
of the exact changed source passes five tests. Independent closed
exponential first-through-fourth derivatives cover all 468 unisolvent
directions across descriptor dimensions 2, 5 and 7, with 1390 fits across
energy, potential and mixed-potential representations. At
h=[0.08,0.04,0.02,0.01], p=2.000005591487..2.000070978831 and minimum
R2=0.999999999662. Every error is measured without a floor; no interval
fails. Separate fourth-order Richardson and closed derivative checks pass.
An isolated earlier prototype also preserves all 1008 compared coefficients
against the frozen library. These are arithmetic model contracts, not
chemical functional/reference acceptance or a whole-program speed claim.

Physical combination status and retained timeouts are recorded in
[response.md](response.md#active-descriptor-taylor-kernels-2026-11-01).
<!-- Active descriptor Taylor kernels: end -->

<!-- Reference-independent harmonic API: begin -->
### Reference Hessian harmonic postprocessing

The two harmonic entries reuse the selected analytic Hessian, including
its convergence diagnostics and functional/reference restrictions. A
nonzero laboratory field/additional AO operator selects unprojected modes
under `vibrational_projection=auto`; Gibbs RRHO is withheld for that model.
Absent APT data yield null IR intensities. Fixed-spinor electronic degeneracy
defaults to one and can be supplied explicitly. The named HF ROHF/complex-GHF frequency and atomic Gibbs public subsets
pass. Actual59 also passes the fixed B2PLYP/H2 full-correlation contract;
other correlated classes and complete component coverage remain pending;
no parent-only
correlation approximation is silently substituted. Exact model, units,
bounded measured fits and pending checks are in
[response.md](response.md#reference-independent-harmonic-analysis).

Retained nuclear paths expose Rust
`NuclearTaylor::coefficient_equations: Option<Vec<CoefficientEquationDiagnostics>>`;
correlated common-driver gradient/Hessian reports and their harmonic
postprocessor expose `coefficient_equations`. The actual59 entries
retain `source_norm` and `reconstructed_norm` in Hartree and the unfloored
`reconstructed_relative_residual` (dimensionless). `linear_relative_residual`
is the fresh relative residual of the selected returned coefficient in its
original A*u=-b equation; `inner_linear_relative_residual` separately records
the largest relative residual of individual correction solves. Both are
dimensionless. An accurate discarded inner solution cannot certify the
returned coefficient. Only linearly admissible coefficients enter the
best-physical-iterate selection; tiny nonzero sources still require a solve.
`None`/null means the backend does not supply these extra diagnostics.
Absolute stationarity/metric/projector/number/reference gates are unchanged.
Actual59 strict metadata, library/CLI/portable audit, 14 closed tests,
five harmonic Rust tests and five native/ASE/CLI tests pass. The fixed
B2PLYP/H2 contract checks the full Hessian and both linear residuals.
The large unfloored reconstruction ratio remains visible and warned about.
Original failed H2 reports remain recorded in response.md. This bounded
result does not certify other correlated models or the full4368 rerun;
that stronger rerun checks raw convergence on every displaced gradient too.
The existing unchanged p39 third/fourth-order nuclear Taylor regression
passes all five tests on actual59 in 1811.908s. This affected regression
is measured separately from gradient/Hessian, earlier624 ROKS and
spring-Hessian fits. It does not replace the pending all4368 correlated
rerun or establish complete high-order tensor coverage for other models.
<!-- Reference-independent harmonic API: end -->


<!-- Represented-pair residual certification checkpoint -->
Current original57 canonical acceptance is 4368/4368 (HF 1092, PBE 1092, TPSS 1092, PBE0 1092); 49 failed intervals are retained. Its correlated leaf terminated with two tests passed, including the zero-PT2 limit (maximum gradient defect 2.442490654175344e-15 and ordered-Hessian defect 2.012279232133096e-15). Independent canonical fits span p=1.802732926104004..2.195880426067760, minimum R2=0.999026043601219. The enclosing manager exit1 remains preserved for its earlier H2/public failures; it does not replace the actual correlated-leaf exit0. This original artifact checked raw diagnostics at base Hessians; it does not certify every displaced gradient of the repaired artifact. This remains partial for the broader program; the original57 correlated chain is terminal. The stronger repaired-artifact status is stated separately below. Actual59 retains its measured 14 closed, five harmonic, five public and five unchanged p39 test passes, including the recorded H2 diagnostics. A separate algebraic counterexample shows that rounding retained high-plus-low amplitudes before the original Hessian action can hide a low-part relative residual. Those bounded passes do not certify that generic case. Source60 preserves both operator images and adds the counterexample and tiny-source dispatch regressions. The exact repaired core in recovered joint02 (manifest ec007b0e645f6f4ffc6ad692a3cbe82749d00ebb1d2de38f23a81ebf93c22fa0) passes all six warning-strict319-target module/Clippy configurations. The first joint metadata failure from two reference-test module imports is retained; the two canonical-import repairs leave every production byte unchanged. The one audited official Rust1.85 wheel/library and CLI have completed successfully. On exact joint02 rlib3a881819430757f0a773720ab4bef08fa81901b4c8656cace321a1f3ef0a61d2, all16 exact-source closed tests (including lost-low and tiny-source cases), five harmonic Rust tests and five native/ASE/canonicalCLI tests pass. The full-PT2 B2PLYP/H2 frequency is 4750.103488154000 inverse centimetres; the42 actual split-part returned-equation residuals have maximum 4.375957819920e-16, and physical stationarity has maximum 7.219401424865e-17 Hartree. The unfloored reconstructed-source ratio maximum 1.195234215610e+00 and its warning remain recorded separately. This is bounded source60 numerical evidence; its unchanged p39 regression now passes all five tests on the same repaired artifact (1036.29 seconds of test time). Independent four-step directional fits give D3 p=2.000325081038, R2=0.999999992984 and D4 p=1.998278797624, R2=0.999999801093; Richardson ratios are 15.7937096387 and 15.8908477116. This certifies those two directional higher-order checks, not complete rank-three/four tensors. The stronger every-sample4368 rerun has now passed all4368 canonical components on exact repaired joint02 rlib3a881819 (HF/PBE/TPSS/PBE0 1092 each), with 49 failed intervals retained. Independent p=1.802732929711707..2.195880426061605, minimum R2=0.999026043824687. Every displaced gradient and base Hessian passed convergence and all nine finite raw bounds. The zero-PT2 limit passed with gradient/Hessian defects 1.554312234475e-15/1.776356839400e-15. The actual correlated stage took 34944.750 wall seconds with observed peak group RSS 36061184 bytes. The original supervisor disappeared while its exact numerical worker survived; a separate observer captured that exact worker's native OS exit status as 0 without restarting or signalling it. The complete Rust test report and independent canonical refit certify the numerical result; the supervisor loss remains recorded. This closes the stated C1 restricted four-parent/seven-channel fixture, not all methods, geometries, complete higher-order tensors, crystal properties or chemical accuracy. The historical actual59 tests remain separately attributed and do not certify the former rounded-low case. The idle source59 all4368 successor was superseded before launch. Its replacement gates every displaced gradient and Hessian sample and independently closes the exact canonical component IDs at terminal. Operators, scientific thresholds, failed intervals and reference values are unchanged.

<!-- Typed physical boundaries: begin -->
## Typed physical boundaries and checked SCF states

Rust additionally exposes `quantities::Coordinates<Bohr>` and
`Coordinates<Angstrom>`, separate energy-per-molecule and energy-per-cell
types, a distinct molar-energy conversion, `EnergyGradient`, `Force`,
unweighted `CartesianHessian` and three-dimensional `Stress`.
Gradient-to-force conversion changes the sign explicitly. Unit conversion
uses the existing CODATA constants and checks finite values, including
overflow after scaling. The wrappers preserve the underlying storage;
coordinate conversion consumes and scales the existing allocation.

`driver::typed::MolecularProblem` owns immutable molecular conditions.
Its `iterate` method exposes a final trial energy even if SCF did not
converge. Only the checked transition creates `ScfSolution<Converged>`,
whose methods call the existing common gradient and Hessian implementations.
The certificate records the requested tolerance and fresh residuals for
that exact problem; it does not establish orbital stability, a ground
state, basis/grid accuracy or response convergence. Coordinates changed
with `with_coordinates` create a new problem and new index owners.
Old solutions remain bound to their original conditions. Exported
`InitialGuess` densities have no convergence or derivative capability;
the existing seeded SCF still checks their new-metric projector and number
constraints. Cross-geometry density reprojection is not implicit.

`indices::AtomId`, `AoId`, `ShellId`, `AuxAoId`, `MoId`,
`AlphaOrbitalId`, `BetaOrbitalId` and `SpinorId` distinguish index kinds.
`IndexSpace` additionally checks ownership/generation and bounds at runtime.
The typed molecular facade binds its atom, spherical-AO and shell spaces
to the immutable problem. Alpha/beta orbital IDs are not a decomposition
of a general spinor. Other low-level orbital/auxiliary containers have not
all been migrated.

These are additional Rust contracts on the common collinear SCF path;
they do not add a new calculation ID or silently substitute a reference.
The existing research APIs and dynamic Python/ASE/CLI interfaces retain
their input and output conventions and runtime checks. XYZ parsing now
uses an explicit typed Angstrom-to-Bohr conversion. A caller can still
mislabel raw values at construction; types cannot infer their true unit.
Compile-fail documentation examples cover unit, meaning, normalization,
index-kind and convergence misuse. Runtime acceptance is tracked
separately from existing all-component physical derivative acceptance.
<!-- Typed physical boundaries: end -->


### Static correlated hypermagnetizability scope

The shared static collinear driver also connects the full analytic E/E/B/B
tensor, electric polarizability and magnetizability for supported PT2 models.
The nine energy channels retain total, parent, target, weighted PT2,
SS-alpha, OS, SS-beta and parent/target nonlocal contributions. Tensor entries
follow eta_abcd = -d4E/(dE_a dE_b dB_c dB_d); existing output units and gauge
conventions apply. Raw parent-field, amplitude, projector, stationarity and
orbital-frame defects accompany the calculation. Complete independent
mixed-tensor acceptance is in progress in p317; a connected driver is not a
validated method combination. Correlated nonzero-frequency hypermagnetizability
remains unsupported.



### Trust-region augmented-Hessian SCF

Molecular common-driver calculations accept `scf_optimizer="auto"`
(default), `"trah"`, or `"legacy"`. The CLI spelling is
`--scf-optimizer trah`; native `run_feature` and ASE
`get_feature_report` use the same string control. Rust additionally exposes
`ScfOptions::orbital_optimizer` / `OrbitalOptimizer`.

The scaled augmented-Hessian eigenproblem uses matrix-free Davidson products
of the complete physical density kernel. Actual versus predicted electronic
energy work determines step acceptance and the trust radius. RKS/UKS use
independent real orbital blocks, ROKS retains its shared closed/open/virtual
constraints, and molecular GKS uses independent real/imaginary spinor
rotations. Integral and RI tensors are shared between SCF and its response
kernel. Functional names never select a curvature formula.

Automatic mode activates TRAH on difficult nonlinear collinear iterations
and stalled supported spinor backends; the existing quadratic collinear
Newton endgame remains. Forced mode starts once a determinant frame is
available. Rejected or unresolved steps return to local iterations without
counting as convergence. Legacy mode retains the previous DIIS/Newton/
quasi-Newton sequence. `no_soscf=true` disables the collinear orbital
globalizer and the new spinor TRAH route; the existing spinor fallback
retains its historical behavior. It cannot be combined with forced TRAH. Thermal or occupation-selected states
and backends lacking the full analytic kernel refuse forced TRAH.

For a conventional double hybrid, TRAH minimizes the **parent SCF energy**.
The same parent orbitals subsequently enter the existing target/SS/OS PT2
evaluation; this option does not turn it into an orbital-optimized PT2 model
and does not cure a divergent perturbative correction. All existing energy,
density and orbital-residual convergence thresholds remain independent.
Stderr records radius/rotation norm (dimensionless), actual/predicted work
(Hartree), agreement ratio, AH shift, raw inner residual and Hessian-product
count. Numerical acceptance and remaining combinations are recorded in
[the response ledger](response.md); the newly connected solver remains
under acceptance testing until those measurements pass.


The `nmr-shielding` and `nmr-spectrum` common-driver routes now also
dispatch constrained common-orbital and full complex spinor shielding.
Measured common-origin ROHF/GHF acceptance covers all 36 C1 components
each; the four-reference closed-shell exact-spin spectrum limit is checked.
UKS spectra share the standalone full-Pauli Ramsey provider. GKS London,
open-shell ROKS transverse Ramsey terms and correlated ROKS/GKS NMR remain
incomplete. Common-orbital London transport has separate acceptance in
progress. `coupling_response_residual` (raw atomic units) participates in
spectrum convergence; full FC/SD/PSO/DSO tensors retain their model labels.
See [measurements and the ethyl-acetate example](response.md#constrained-reference-nmr-and-correlated-mixed-field-measurements)
before selecting a reference/gauge combination.


### Restricted-open-shell double-hybrid energy

The shared `energy` calculation accepts `reference="roks"` with a named
double-hybrid recipe or a custom parent and explicit `pt2_same_spin` /
`pt2_opposite_spin`. For example, Rust/native Python/ASE controls
`xc="b2plyp", reference="roks", charge=1, multiplicity=2` and CLI
`--xc b2plyp --reference roks --charge 1 --multiplicity 2` select the same
constrained common-orbital parent.

The existing parent solver diagonalizes the separate alpha/beta occupied
and virtual Fock blocks without changing either density. The shared PT2
contraction then evaluates doubles only. It does not add an open-shell
MP2 singles correction. This convention follows the restricted-open-shell
double-hybrid construction (DOI:
[10.1021/jp9042864](https://doi.org/10.1021/jp9042864)) and the explicit
doubles-only convention in section 3.8 of
[10.1021/acs.jctc.1c00093](https://doi.org/10.1021/acs.jctc.1c00093).
All explicitly represented electrons are correlated; the correction is
full-Coulomb, unregularized and uses the supplied SS/OS coefficients.

`energy`, `reference_energy`, the three unscaled `pt2_same_spin_alpha`,
`pt2_same_spin_beta`, `pt2_opposite_spin` terms and weighted
`pt2_correlation` are in Hartree. The raw
`reference_common_orbital_residual`,
`reference_semicanonical_residual`, `pt2_amplitude_residual` and
`pt2_lagrangian_residual` are also in Hartree.
The independent-spin `reference_orbital_residual` need not vanish at a
valid common-orbital stationary point; its nonzero value is retained.
`reference_eigen_residual` is a relative Frobenius residual and is dimensionless; ASE preserves it without an energy conversion.
`pt2_excitation_model` identifies the doubles-only convention.

Source `a2aa595c08f960c4` passes both `p320_roks_double_hybrid` tests:
all three spin terms agree with an independent direct-AO determinant sum,
and closed-shell, zero-correlation and one-electron limits pass.
The asymmetric NH3+ B2PLYP/STO-3G example (unpruned level-1 grid) gives
E=-55.340070390800975 and weighted PT2=-0.014684869880666217 Hartree;
common-orbital/semicanonical residuals are
4.558793230067e-11/3.552713678801e-15 Hartree.
Public route checks in `test_trah_scf.py` are still running. ROKS correlated gradients and other responses still
require the constrained PT2 Lagrangian and are not validated by an energy
comparison; unsupported routes return a named error. See the coverage
ledger for the measured scope.


## Experimental full frozen scalar action

The existing `continuous-species` feature also accepts
`fragment_mode="scalar-evaluate"` and `"scalar-optimize"` with an opaque
native-unit `fragment_model_json` record of version `frozen-scalar-action-v1`.
Rust exposes `driver::continuous_species::scalar::ScalarActionRecord`;
native Python uses `evaluate_reduced_fragment(..., fragment_mode=..., model=...)`,
and ASE uses `get_reduced_fragment` with the same selectors. The canonical
CLI uses `--fragment-mode scalar-evaluate --fragment-model-json <JSON>`.
These weight modes identify the saved first state's raw geometry and refuse
changed geometry. The separately declared fixed-endpoint affine geometry modes
below optimize coordinates; weight optimization itself does not move nuclei.

The record stores explicit site/candidate identities, every joint endpoint
tuple, per-state integer active electron counts and full/frozen electron
ledgers, normalized ordered s/p AO functions, fixed frozen orbitals, reactant
construction identities/energies/residuals, and an explicit functional object.
The functional object uses the existing `xc`, `x`/`c`, `xc_terms`, `omega` and
`xc_parameters` grammar. The evaluator capability contract permits the
implemented all-electron closed-shell semilocal action; hybrids, nonlocal,
learned, ECP, periodic and noninteger sectors are refused. Present chemical
and scalar derivative measurements are PBE examples, not validation of every
family-supported provider. Settings/provider changes require a new matching
construction record; metadata alone cannot establish physical provenance.

`fragment_weights` concatenates one simplex per saved site. The joint action
weights are products of site weights, and every joint endpoint scalar keeps
its electronic cross interactions. Missing combinations are errors; they
are not completed by adding separate site scores. The endpoint count grows
as the product of the site candidate counts. Full endpoint integrals and
frozen-density Hartree/XC actions are retained, so this is **not** a compressed
fast potential. Each state has its own fixed common AO chart; differing
reaction-state electron/AO counts are not silently equated.

`fragment_weight_strategy` selects `joint` (default), `one-pass` (one block
cycle in saved site order), or `cyclic` (repeat block cycles). The same existing
projected L-BFGS optimizer is reused. `fragment_weight_tolerance` defaults to
1e-4 Hartree; `fragment_weight_max_iter` defaults to 160 (per block, and the
cycle cap for `cyclic`). Energy, density and orbital SCF thresholds remain
separate `conv_tol`, `conv_tol_dp`, `conv_tol_grad` controls, defaulting to
1e-14 Hartree, 1e-13 and 1e-12 Hartree respectively. No tolerance is weakened
to obtain a weight convergence flag. Failed warm SCFs are logged and may
retry from an independent core guess at unchanged gates.

Reports contain state energies before common reference addition, physical
electronic/nuclear components, per-term energies, all independent site
weight derivatives, raw SCF/particle/idempotency diagnostics, and actual
optimizer convergence, residual, iteration/evaluation count and block
history. At each site the independent coordinates are `q_i=w_i` for i>0 and
`w_0=1-sum(q)`; gradients are in saved site/candidate order. Orbital/reference
commutator residuals and weight derivatives have Hartree units; metric and
particle defects are dimensionless. ASE converts Hartree entries to eV and
leaves the saved JSON in native units. CLI logs every active SCF iteration
energy and residual on stderr by default. State-common reference constants
are removed inside compensated scalar accumulation before rounding, and
reaction coefficients must cancel them. State-dependent strain/polarization
is not removed as a reference constant.

`continuous_species_scalar_api` checks every one of the four independent
two-site simplex derivatives on a C1 algebra fixture at four steps, plus
serialized/direct parity and bounded joint/block optimizer behavior. The
initial eight-iteration optimizer smoke runs did not converge; their flags
remain false. Actual substituent accuracy and optimized reaction enthalpies
are separate acceptance tasks. The common driver and installed CLI/native/
ASE regression tests must pass on the same frozen source before public
route acceptance is recorded.


### Restricted and unrestricted analytic nuclear Hessians

The ordinary `hessian` and `frequencies` calculations reuse one analytic
CPKS implementation for RKS/RHF and UKS/UHF. UKS retains separate alpha and
beta occupied/virtual spaces, including unequal spin populations. The
Cartesian Hessian is in Hartree/Bohr^2 before any mass weighting; ordering
is `3*atom + axis`. No finite-difference fallback is used by these common
routes. Analytic moving-basis, moving-grid and metric contributions follow
the selected provider's capability contract.

For integer populations, `orbital_response_residuals` contains one fresh
maximum absolute defect of the original CPKS equation per Cartesian
perturbation, in Hartree/Bohr. Each must pass the actual solver tolerance
before the Hessian is returned. `scf_orbital_residual` is in Hartree;
`raw_asymmetry` is measured before symmetrizing the Hessian. Ensemble
populations retain their separate dimensionless `ensemble_response_residuals`.
ASE converts the new dimensional residuals to eV/Angstrom and eV unless
atomic report units are requested. Molecular minimum checks and RRHO
thermochemistry use the analytic Hessian. An SCF-converged or geometry-
converged result alone does not establish a vibrational minimum.

Double hybrids use the same public Hessian entry with the existing parent,
target and PT2 response provider. Its orbital residuals are dimensionless;
PT2 amplitude residuals are in Hartree, and unscaled SS-alpha/OS/SS-beta
nuclear tensors have the same Cartesian units as the total. The new UKS H5
HF-parent check passes all 1680 component fits. The raw orbital relative
residual is 3.3271284980532005e-15 (dimensionless), and the amplitude
residual is 8.326672684688674e-17 Hartree. Other parent classes and installed
public-route extensions have separate acceptance states; see the
[correlated nuclear checkpoint](response.md#open-shell-correlated-nuclear-component-acceptance)
and [report guide](api-guide.md#choose-a-hessian-and-minimum-check).

This ordinary molecular path does not substitute for the second derivatives
of a saved continuous-species model: its moving projection, frozen density
and active response require their own analytic Hessian and acceptance.
The C1 RKS/open-shell UKS acceptance in `p321_collinear_hessian` passes
all 1,560 independent gradient/Hessian component fits across ten bounded
HF/LDA/GGA/meta-GGA/global-hybrid fixtures. The largest fresh CPKS defect
is 4.278521981149197e-14 Hartree/Bohr; all four installed native/ASE/CLI
Hessian/frequency route checks pass. See `response.md` for the measured
intervals, retained failures and exact scope. This does not certify every
provider/reference or the saved continuous-species model Hessian.


### Fixed-endpoint affine scalar geometry

`fragment_mode="scalar-gradient"`, `"scalar-geometry-optimize"` or
`"scalar-hessian"` selects explicit `fragment_state` and `fragment_endpoint`
identities. The endpoint must declare
`geometry_transport="fixed-internal-affine-v1"` and `geometry_atom_owners`:
one zero-based raw-center owner per full atom, covering every raw center.
Positions belong to the selected state's raw atom labels. Full atoms follow
their owners by translation at fixed laboratory internal offsets. A frozen
reactant coefficient span is normalized in the current AO metric; displaced
target SCF orbitals never regenerate it.

The existing `frozen-scalar-action-v1` with `reference="rks"` keeps paired
frozen spatial orbitals and an even `active_electrons` count. It is not
reinterpreted as unrestricted data. The separate
`frozen-scalar-collinear-action-v1` requires `reference="uks"`, explicit
`active_spin_electrons=[Nalpha,Nbeta]` in each state, and independent
`frozen_orbitals_alpha` and `frozen_orbitals_beta` matrices in every endpoint.
An empty spin bath is an AO-row array of empty rows. The raw charge and
multiplicity must match the active spin counts; full counts must equal
active counts plus the respective frozen ranks. Both spin maps must retain
the common raw rank. Missing spin blocks, conflicting selectors and an old
RKS record merely relabeled UKS are errors. UKS weight interpolation is not
connected; these UKS selectors operate on one saved endpoint only.

Rust exposes `ScalarActionRecord::calculate_geometry` and
`calculate_hessian`. Native Python uses `evaluate_reduced_fragment`, ASE uses
`get_reduced_fragment`, and the canonical CLI uses the same `fragment_mode`,
`fragment_state`, `fragment_endpoint` and `fragment_model_json` controls.
All call the existing analytic scalar/geometry machinery and shared geometry
optimizer. There is no second SCF or optimization implementation in the API.

Geometry controls `max_steps`, `gtol` and `step_max` default to 200,
1e-5 Hartree/Bohr and 0.2 Bohr. SCF gates remain separate. Hessians use the
common `response_*` controls. Reports return physical and reference-free
energies, every Cartesian gradient and force, raw/full positions and actual
SCF and geometry convergence diagnostics. UKS particle, idempotency, chart
and frozen-overlap errors are explicit `[alpha,beta]` arrays; old RKS fields
remain scalars. `scalar-hessian` additionally returns the complete `hessian`
and `raw_hessian` in `(3*atom+axis,3*atom+axis)` order, `raw_asymmetry`, the
reference orbital residual and every original-equation directional response
residual. Directions list Cartesian unit vectors, then all pairs `i<j`.
Hessians/asymmetry use Hartree/Bohr^2, absolute response/gradient-consistency
residuals Hartree/Bohr, and relative residuals are dimensionless. ASE converts
these once to eV and Angstrom units; model JSON always retains native units.

The model fixes laboratory offsets and a frozen electronic space. It does
not supply a freely rotating fragment frame, internal fragment relaxation,
mixed-weight geometry or full molecular thermochemistry. A stationary affine
geometry is not automatically a minimum. A constrained spectrum requires
the pulled-back physical mass metric `A^T M_full A`; only three translations
are guaranteed null directions. No six-mode projection or molecular RRHO
heat is inferred from this API.

Exact cached source ed98122e8a1b2459 passes every ordered PBE Hessian
component for both restricted and unequal-spin fixtures: 81 entries each,
with minimum R2=0.9999391422/0.9999701827 and no failed intervals. The
explicit-density LDA/GGA kernel additionally passes 800 components with
27 coarse intervals retained and resolved. Separate clean wheel and sdist
installs each pass all 29 scalar/schema tests, including RKS/UKS Hessian
native-Python/ASE/canonical-CLI parity, plus four ordinary Hessian/frequency
tests. Two optional legacy generated-template checks are skipped; no
scalar-Hessian check is skipped. Units, original-equation residuals and
default CLI SCF stderr are checked. See response.md for full artifact
identities and raw numerical scope. These test-fixture and installed-route
passes do not certify a real-model optimized minimum or reaction heat.

The saved scalar capability contract currently permits generic LDA/GGA only. Exact exchange, range separation, meta-GGA, nonlocal correlation and Skala remain refused by the evaluator; the broader ordinary Hessian regression does not extend this model capability. Measured model derivative/chemical acceptance remains PBE-only.

## SCF occupation-order diagnostics

Common collinear reference reports, including `energy`, `hessian` and
`frequencies`, return `occupied_virtual_gaps` in Hartree, ordered alpha then
beta. Each entry is the lowest virtual minus highest occupied energy of
the retained stationary orbital blocks. An empty occupied/virtual block,
fractional populations or a common-orbital constrained reference gives
`null` for the unavailable quantity. A negative gap produces an explicit
warning: orbital stationarity does not certify Aufbau ordering, electronic
stability or the global ground state. Use the independent `stability` route
to characterize orbital curvature. The accepted trust-region determinant
keeps the original energy, actual density-step and fresh orbital-residual
convergence tests. Installed acceptance of this new diagnostic is pending.
