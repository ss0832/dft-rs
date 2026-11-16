<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 ss0832 -->

# Implementation conventions

Rules that every change to dft-rs follows: code, tests, documentation and
release artifacts. They describe how a feature is built, wired, validated
and recorded. The physics of individual methods is in
[response.md](response.md), [periodic.md](periodic.md),
[FUNCTIONALS.md](FUNCTIONALS.md) and [THEORY.md](THEORY.md).

## Contents

- [Scope and claims](#scope-and-claims)
- [Language, dependencies and portability](#language-dependencies-and-portability)
- [Code hygiene](#code-hygiene)
- [Units and reports](#units-and-reports)
- [Wiring a calculation](#wiring-a-calculation)
- [Exchange-correlation functionals](#exchange-correlation-functionals)
- [Quadrature grids](#quadrature-grids)
- [Common calculation settings](#common-calculation-settings)
- [Derivative validation](#derivative-validation)
- [Oracles](#oracles)
- [Acceptance tests](#acceptance-tests)
- [Verification by implementation contract](#verification-by-implementation-contract)
- [Ledgers](#ledgers)
- [Documentation](#documentation)
- [Changelog and releases](#changelog-and-releases)

## Scope and claims

- The objective is the complete molecular, periodic and high-order response
  program. A partial milestone, an interface, a refusal or a passing subset
  of tests is never described as completion of a feature or of the program.
- A release covers the calculations that are implemented and validated at
  that point; every combination that is not is recorded as such in the
  [ledgers](#ledgers), not hidden.
- An unsupported input is refused with a named error (`DftError` is
  `#[non_exhaustive]`), never answered with a plausible number.

## Language, dependencies and portability

- Code, comments, identifiers, error messages, test output, documentation
  and changelog entries are written in English.
- Computational dependencies are only those declared in `Cargo.toml`
  (`libm`, `rayon`, `faer`, optional `pyo3`). The Python package uses only
  its declared dependencies.
- Names of external reference implementations, in any spelling, do not
  appear in source, documentation, tests, file names or distributions.
  Programs used only to produce frozen oracle values are named where the
  oracle is described.
- Auxiliary automation and validation scripts are Python `.py` files; no
  shell-specific script formats are added.
- Nothing hardcodes a workspace, home directory, user name or session path.
  Resource paths are resolved relative to the package (for tests,
  `env!("CARGO_MANIFEST_DIR")`) or accepted as explicit inputs.
- The author metadata (`ss0832` and its e-mail address) is preserved.
- Privacy cleanup preserves third-party authorship, copyright, notices,
  license texts, source URLs and attribution. Machine-local build roots
  may be made portable without removing upstream package identities,
  dependency edges or public provenance URLs. Redistribution includes
  the complete bundled notices and license texts, not only links to them.

## Code hygiene

- Compiler warnings and Python test warnings are errors
  (`cargo clippy --all-features --lib --bins --tests -- -D warnings`,
  `python -W error -m pytest`). `cargo fmt` is applied to every change.
- Python correctness lint is mandatory: install the development-only tool with
  `python -m pip install '.[lint]'` (also included in `.[test]`), then run
  `python tools/lint.py`. It checks all of `python/` and `tools/`, including
  oracle generators, with the explicit Ruff `E9`/`F` rules (syntax and
  Pyflakes correctness). Findings fail; unused imports are removed rather
  than suppressed. `python/tests/test_lint.py` runs this gate and checks its
  rejection of invalid syntax, undefined names and unused imports. The
  runner works from any directory, accepts explicit paths and `--fix`, and
  disables caches. Ruff is not a runtime or computational dependency.
- Unused code is removed. Diagnostics are not suppressed with
  `allow(dead_code)`; data needed only by tests stays behind `cfg(test)`.
  Shared test helpers live in `tests/support/` and are included with
  `#[path = "support/<file>.rs"] mod <name>;` only by tests that use every
  item of the file.
- Each file keeps its own line-ending style.
- CLI calculations report SCF iteration energies and convergence residuals
  on stderr by default.
- Response and derivative code never branches on a functional's registered
  name; it branches on the functional family and hybrid parameters only.

## Units and reports

- Rust and the native Python layer use atomic units (Bohr, Hartree,
  atomic units of field). The ASE calculator uses Angstrom and eV and
  converts through the unit labels of `CONTROL_UNITS` and `REPORT_UNITS`
  in `src/driver/units.rs`. The adapter uses the package CODATA 2018
  energy and length factors, matching native calculations and XYZ input;
  it does not inherit the installed ASE release's default CODATA edition.
- Every control and every report field carries a unit label; a new key
  without one fails the strict unit check (`DFT_RS_STRICT_UNITS=1`).
- Reports state the model and its assumptions and return raw residuals
  (response, stationarity, reference) next to the values, so that a result
  can be judged without rerunning it.

## Wiring a calculation

Every completed calculation reaches Rust, the `dft-rs` CLI, native Python
and ASE through the common driver:

1. a driver function (`src/driver/<area>.rs`) reads its controls and ends
   with `controls.finish(id)`, which refuses unknown controls;
2. a dispatch arm in `src/driver/molecular.rs::run` or
   `src/driver/periodic.rs::run`;
3. an entry `feature!(id, title, space, driver, rust, cli, python, ase,
   [targets])` in `src/driver/registry.rs`, immediately preceded by a
   `// settings-contract: PROFILE` applicability declaration;
4. unit labels for its controls in `CONTROL_UNITS` and for its report in
   `REPORT_UNITS` or the report's own `units` block;
5. a wrapper in `python/dft_rs/native.py` exported from
   `python/dft_rs/__init__.py`, and an ASE getter in
   `python/dft_rs/ase_calculator.py`;
6. one row in each of the three aligned tables of
   [rust-api.md](rust-api.md), [cli.md](cli.md) and
   [python-api.md](python-api.md), in registry order, with units,
   conventions, model assumptions and raw residual measurements;
7. ledger entries (see [Ledgers](#ledgers));
8. an acceptance test `tests/pNNN_<topic>.rs` and, for public routes, a
   Python test in `python/tests/`.

The registry test and `python/tests/test_api_documentation.py` enforce that
the three tables list exactly the registered calculations.
The `settings-contract` comments are mandatory source-audit declarations.
They are distinct from the public `FeatureSpec` fields serialized by
`feature-registry`; runtime parity checks compare those public fields and
validate the source declarations separately.

## Exchange-correlation functionals

- A functional is one generic energy-density body `fn body<S: Scalar>(...)`
  monomorphized by the LDA, GGA or meta-GGA driver; dense derivatives
  through third order and Taylor series of arbitrary order come from the
  same body by automatic differentiation. Construction macros and shared
  primitives (spin scaling, PW92 and VWN parameter sets, enhancement
  factors, B97 power series) are used instead of repeated code.
- Parameters are declared with names. A parameter with a physical
  dimension declares its uniform-scaling power, so the coordinate-scaling
  test holds without an exclusion list.
- Values at an empty gradient that are a convention rather than the true
  limit (`sigma_is_absent`) are declared in the boundary test, and the
  single-spin branch of every functional must be the limit of its
  two-spin expression.
- Meta-GGA functionals whose reference definition enforces the
  Fermi-hole-curvature bound (`sigma <= 8 rho tau`) set the driver flag
  that applies it to the input values.
- New functional classes need independent energy and term-by-term oracles
  (for double hybrids separate same-spin and opposite-spin PT2
  contributions) and physical limiting checks. Every component is compared
  point by point with frozen reference values, and every combined entry
  with frozen SCF energies (see [Oracles](#oracles)).

## Quadrature grids

- The default molecular and periodic grid is Treutler-Ahlrichs M4 radial
  quadrature, Lebedev-Laikov angular rules, five-region pruning and the
  Becke partition with Treutler's size adjustment. Published numerical
  tables are stored as static data and verified by tests (every Lebedev
  rule against closed-form monomial integrals); they have no generator at
  build or run time.
- Quadrature points whose partitioned weight is below `grid_weight_cutoff`
  are dropped. Derivative acceptance tests set it to `0`, which keeps the
  energy a smooth function of the nuclear positions; with a nonzero cutoff,
  points entering and leaving the grid add noise to displaced quantities.

Grid resolution and physical-model support are separate contracts. A finer
grid does not enable an unsupported reference or derivative. The controls
below apply to implemented paths; pure Gaussian analytic integrals and
post-processing of supplied tensors have no XC quadrature to refine.

| Integration | User controls and defaults | What is kept separate |
| --- | --- | --- |
| Molecular semilocal XC, its orbital/electric response and moving nuclear derivatives | `grid_level`: integer 0..9, default 3; `grid_preset`: `level` (default), `fine`, `ultrafine`, `superfine`; `grid_angular_scheme`: `lebedev` (default) or `product`; `grid_weight_cutoff`: finite, nonnegative Bohr^3, default 1e-14 | The response uses the reference grid. Moving-grid derivatives retain its points, atom ownership and unpartitioned weights and differentiate the partition and moving AOs; they do not rebuild a default-resolution grid. |
| Molecular VV10 | Independent `vv10_grid_level`, integer 0..9, default 3; common-driver `grid_weight_cutoff` also sets its pruning | VV10 uses its own level-based Lebedev grid, not the semilocal preset or angular scheme. Rust additionally exposes `ScfOptions::vv10_grid_weight_cutoff`. |
| Noncollinear spin projection | Molecular `spinor_xc_tolerance`, finite and positive, default 1e-10 | This is spin-sphere integration, independent of the real-space molecular grid. The reference checks successive angular rules up to order 128 and reports its history. Nuclear response starts from the reference order and checks its own rules using `response_tolerance` (default 1e-10). Low-level Rust point evaluators additionally accept an explicit even angular order 4..128. |
| HF indirect nuclear spin-spin coupling | The requested molecular grid controls select the diamagnetic expectation quadrature even though HF SCF has no XC grid | This is distinct from direct electronic spin-spin ZFS. HF direct ZFS and its analytic derivatives are grid-free; the contact-only Ramsey term does not depend on this quadrature. |
| Multicomponent electron/quantum-particle XC and EPC | The same molecular grid controls; an EPC grid is built with them even for an HF electronic parent | Electron and quantum-particle densities are evaluated on this common grid. There is no separate public proton/EPC grid level. `quantum_exponents` and `quantum_lmax` control the quantum Gaussian basis, not integration resolution; Gaussian kinetic/Coulomb integrals are analytic. Annihilation integration uses the same requested grid. |
| MC-PDFT on-top functional | The common molecular grid controls, including with an HF starting reference | Active-space size, orbital basis and on-top functional are independent controls. |
| Conventional periodic XC and its response | The same level/preset/scheme/weight controls; `grid_shell` defaults to 2 and controls the periodic partition neighbourhood | `k_mesh` (default 1,1,1), response `q`/`q_points`/`q_mesh`, reciprocal/source cutoffs and `xc_ao_cutoff` are independent convergence parameters. |
| Dimensional real-space XC/field response | Molecular-style grid controls; `grid_cutoff` defaults to 12 Bohr | The cutoff bounds open radial tails and the image neighbourhood, rather than selecting radial/angular point counts. Electrostatic `quadrature_order` (default 96), `operator_tolerance` (1e-11) and `grid_tolerance` (1e-5) are separate. |
| Magnetic-torus XC | `magnetic_grid_order`, 16..128; driver default `16 + 4*grid_level` (28 at level 3); refinement by 8 up to 128 with raw history | Open-direction `magnetic_open_order` defaults to 64 and `grid_cutoff` to 12 Bohr. Atomic-grid presets, a product angular selector and nondefault weight pruning are refused here: this grid is tensor-product quadrature. |

The common driver exposes the four molecular grid controls through Rust
`MolecularMethod`/`PeriodicMethod`, Python `run_feature` and named report
functions, ASE calculator parameters/per-call controls, and CLI hyphenated
options. Dedicated legacy bindings expose the options in their actual
signature: every integrating electronic response has a level control, but
named presets and weight pruning are not universal legacy keywords. Unsupported
keywords fail explicitly. The dimensional Python/ASE route forwards its
preset, scheme and cutoff to the same `GridOptions::from_controls` parser.
Both CLI names expose `--vv10-grid-level` on legacy molecular response paths.
Dimensioned controls are atomic units in Rust/native Python/CLI; ASE converts
explicit `grid_weight_cutoff` between Angstrom^3 and Bohr^3, including both
calculator parameters and per-call overrides. Omission retains the native
default, while the value zero is unchanged by that conversion.
The common `periodic-field-scf` and `periodic-spinor-scf` routes also accept
positive finite `operator_tolerance` (default 1e-11) and `grid_tolerance`
(electron-count error, default 1e-5), matching the dimensional Python binding.
The noncollinear periodic spin-sphere refinement uses `operator_tolerance`;
it is independent of real-space level and the electron-count gate.

In Rust, low-level response functions accepting `Option<&MolecularGrid>` use
the supplied grid. For an HF Ramsey calculation, pass an explicit grid (or
use `MolecularSetup::integration_grid`/the common driver) to select its
diamagnetic quadrature; `None` retains the low-level default. Rust
`GridOptions` also offers pruning and per-element/per-atom radial/angular
overrides. These additional typed fields are not public Python keywords.
Learned nonlocal XC forces the effective molecular partition-weight cutoff
to zero; this preserves its defining unpartitioned-grid sums.

Frozen continuous-species recipes are not covered by a universal grid
override: their saved generation/application recipe and tested grid remain
part of their experimental provenance. Generic grid settings do not establish
new generation or transfer validation; use the exact model's current ledger.

## Common calculation settings

`Controls` is the single named request representation. Molecular and periodic
adapters resolve it through the existing `MolecularMethod`/`PeriodicMethod`
and SCF options; response solvers retain their own applicable tolerance and
iteration controls. A response tolerance is not an SCF tolerance. Nested
calculations must derive their options from the enclosing resolved request,
preserving explicit overrides, rather than constructing fresh defaults.
GKS projects the resolved SCF settings, including damping, into its spinor
solver. Nested integer-SOC calculations retain their historical solver defaults
when no override is supplied; explicit `max_iter`, `conv_tol`, `conv_tol_dp`,
`conv_tol_grad`, `damping` and `diis_size` also reach that internal SCF.
Collinear ADIIS and level shifting govern the collinear seed and are not options
of the distinct spinor iteration algorithm.
The run envelope owns `cpu_threads`, `memory_mb` and logging. Memory accounting
covers registered allocations and reserved workspaces, not whole-process RSS.

For ASE common reports, per-call keywords override calculator parameters;
method defaults apply after those two sources. Geometry and dimensional
controls cross the ASE/atomic unit boundary once. Integer-valued controls
must not be rounded or accept Boolean values. Ordinary molecular ASE
energy/force requests use the common route when additional reference controls
are configured. Legacy getters that cannot forward a configured control
refuse it by name; they must not silently substitute a default. The legacy
dynamic-polarizability adapter keeps optical `damping` in Hartree distinct
from optional dimensionless `scf_damping`.

Every registry row declares one applicability family below. The existing
`tools/api_reference.py --check` and permanent documentation tests reject
missing, unknown or space/route-incompatible declarations. The declaration
records what must be reviewed when a route changes; it is not evidence that
every functional, reference or response order has passed numerical tests.

| Registry family | Applicable settings and limits |
| --- | --- |
| `molecular-common` | Common-driver molecular reference, applicable XC/property grids and route-specific response controls. Reference/provider restrictions remain explicit. |
| `molecular-dedicated` | Individually bound molecular constructor and its declared signature; an option absent from that binding is not automatically available. |
| `periodic-common` | Common periodic reference, real-space quadrature, lattice sums, k mesh and property-specific q/response controls. These resolutions are distinct. |
| `periodic-dedicated` | Individually bound SCF/response options with their declared defaults and applicability. |
| `periodic-operator` | Prescribed basis, charge, density or field operator; no implicit electronic SCF. Numerical quadrature/cutoffs apply only where the kernel uses them. |
| `supplied-data` | Physical data processing; electronic basis, functional and SCF grid are inapplicable. Units and applicable run-envelope controls still matter. |
| `conditional` | The selected supplied-data/generated-operator/reference mode determines applicability; inspect that mode before forwarding reference controls. |
| `multicomponent` | Electronic SCF plus separate species/coupled iteration and particle-basis controls. XC/EPC quadrature is distinct from analytic Gaussian integrals. |
| `multiconfiguration` | Active-space/state controls plus the selected orbital reference and applicable on-top quadrature. A determinant response is not substituted for multistate response. |
| `frozen-model` | Saved recipe/provenance determines the model; generic reference settings do not establish transferability. This experimental model remains frozen with its documented failures. |
| `cli` | Callable command dispatcher; the selected command owns its settings. |

An analytic grid-free HF kernel is marked not applicable for XC quadrature,
not unsupported. HF Ramsey diamagnetic terms are an exception: their property
quadrature still uses the requested molecular grid. Moving nuclear grids
retain the supplied quadrature points and weights; VV10 has an independent
nonlocal grid. See [Quadrature grids](#quadrature-grids) for exact options.

Regression evidence includes direct inspection of resolved Rust SCF options,
HF property-grid parity through the public routes, integer/unit validation,
and an intentionally iteration-limited SCF whose saved histories test actual
mixing and iteration forwarding. These checks complement scientific derivative
tests. Registry coverage alone does not close the ongoing audit of every
nested property calculation or establish scientific release acceptance.

## Derivative validation

Finite differences are validation only; production derivatives are
analytic, automatic-differentiation or response derivatives.

- Every new or extended derivative has an independent multi-step error fit
  in its acceptance test: at least four distinct steps, the production
  value against central differences, and a least-squares fit
  `log|error| = log C + p log h`. The exponent, the fit quality (R2), the
  prefactor and the step interval are reported and recorded.
- Every independent component of a derivative tensor is fitted (gradient,
  Hessian, polarizability and hyperpolarizability, dipole and Raman
  derivatives, magnetizability, susceptibilities, strain derivatives,
  elastic constants), not one component or one direction. Components that
  share finite-difference evaluations are grouped
  (`tests/support/components.rs`): one displaced pair of gradient runs
  gives a gradient component and a whole Hessian column; a field along one
  axis gives a column of the polarizability; mixed fields give the
  off-diagonal second derivatives. A symmetric tensor of higher rank may be
  validated through its contractions along as many linearly independent
  generic directions as it has independent components.
- Test geometries have no symmetry element, so that no component vanishes
  by symmetry and every fit sees a measurable error.
- No error floor is applied and no failed step is discarded. A component
  whose fit fails on the first interval is recomputed on a finer interval
  (against higher-order terms) and then a coarser one (against noise); when
  an anomalously small h^2 coefficient cancels against the h^4 term, a
  fourth-order stencil fitted against order 4 is the last stage
  (`tests/support/stencils.rs`). Every fit is printed, failed ones marked,
  and the failure and the resolved interval are documented.
- Closed forms, dense solutions, conservation identities (translation and
  acoustic sum rules) and Richardson checks remain additional requirements.

## Oracles

- Independent reference values are produced by scripts in `tools/oracle/`
  and frozen in `tests/oracle_data/`. Oracle programs are used strictly as
  numerical oracles: they are never a dependency of the package, its build
  or its data generation, and no production table is generated from them.
- Each oracle file records its provenance (program version, settings,
  geometry units). Regenerating an oracle is an explicit, reviewed step.
- Point-by-point functional oracles cover physical points and the boundary
  regions (small density, full spin polarization, the iso-orbital limits).

## Acceptance tests

Incremental verification is selected by the explicit provider, assembly,
solver and property boundaries in [verification by implementation contract](#verification-by-implementation-contract).
Keep per-functional derivative/domain/recipe checks and independent
representative integrations for every distinct reference, descriptor and
physical backend. An unchanged shared contract can avoid a new all-names by
all-properties sweep; it cannot supply missing higher-order or unconnected
provider evidence. Existing tests and the full release-suite requirements
remain unchanged.

The acceptance matrix includes both `wb97x-d` (range-separated exchange
and CHG dispersion) and `wb97x-v` (range-separated exchange and
self-consistent VV10 correlation) for every affected property/reference
route where their defining operators apply. Record the full functional's
result or the named unsupported capability in the response ledgers. A
successful semilocal parent or an energy-only result supplies evidence
for that narrower calculation.

- **Every component of a derivative is validated.** A derivative
  acceptance test fits each independent component of the derivative
  (see [Derivative validation](#derivative-validation)); a test that fits a
  single component, a single Hessian column or the contraction along one
  direction is incomplete and does not validate the derivative. When a
  feature itself returns a directional quantity (for example force constants
  contracted with a displacement pattern), every entry of that quantity is
  fitted.
- An acceptance test names in its header what it validates and against
  which independent route, and records the measured exponents, fit quality,
  step intervals and any failure with its resolution.
- A failing assertion must be investigated. Preserve its original data and
  threshold. A revised numerical acceptance criterion needs an explicit
  dimensional error budget, conditioning/cancellation and inverse-step
  amplification analysis, independent evidence and disclosed earlier
  exploration. Compensated evaluation is preferred where useful. A justified
  noise-floor criterion does not turn the original failed rate fit into a
  pass, or excuse a physical, convergence or quadrature error.
- Long or memory-heavy acceptance runs are made faster by optimizing the
  production code, never by shrinking the test.
- The measured numbers of every run are recorded in
  [response.md](response.md) in the part of their topic; chronological run
  records are kept in its last part.

## Verification by implementation contract
Functional expressions and response machinery have explicit interfaces, but
they are not one interchangeable code path. Verification is organized around
those interfaces and distinct physical backends. Existing tests and numerical
thresholds are retained. This strategy changes how additional runs are chosen;
it does not turn registration or automatic differentiation into acceptance.

### Functional-provider contract

`xc/traits.rs::GridVars` supplies
`(rho_a,rho_b,sigma_aa,sigma_ab,sigma_bb,tau_a,tau_b)`, with
`sigma_st=grad(rho_s).grad(rho_t)` and
`tau_s=1/2 sum_i f_i |grad(phi_i)|^2`. Effective local dimensions are 2 for
LDA, 5 for GGA and 7 for meta-GGA. `XcOut` stores energy per volume, first
potentials and packed second/third partials. In atomic units rho, sigma and
tau have dimensions Bohr^-3, Bohr^-8 and Bohr^-5, respectively; energy density
is Hartree/Bohr^3. Empty channels and the mixed-spin sigma factors are part of
the contract, not conventions that may be guessed independently by consumers.

`XcFunctional` exposes weighted `eval_add`, family/order metadata and parameter
controls. Built-in LDA/GGA/meta drivers share scalar-generic expressions and
dense orders 0..3. `xc/taylor.rs::XcTaylorInput` carries seven Taylor-normalized
paths and optional parameter paths; trait methods expose energy, potential and
mixed directional-potential series. Their defaults return unsupported, so an
external provider with dense derivatives is not automatically Taylor-capable.
`ResolvedFunctional::semilocal_taylor_*` combines weighted provider parts;
composite parameter paths have separate restrictions from static parameter
replacement. `HybridParams` and required nonlocal/PT2 terms remain separate.

Keep per-body independent energy, spin/domain/parameter/limit and all required
derivative-component checks. Keep per-name family/maximum-rung, coefficients,
exchange range/omega, required-nonlocal and parent/target/PT2 SS/OS/dispersion
recipe checks. Weighted recipes can reuse already validated bodies plus
composition/linearity contracts and selected molecular integrations. They do
not need every existing property repeated merely because the name is new.

### Orders requested by actual interfaces

| Consumer/operation | Required local energy derivatives | Other obligations |
|---|---|---|
| Ground energy / SCF potential | 0 / 1 | Density, sigma, tau and integration/Fock conventions |
| Ordinary static or adiabatic dynamic response kernel | 2 | `ResponseSolver::kernel_only`; LDA/GGA contraction and the separate tau-aware `response/mgga_kernel.rs` both request order 2 |
| Ordinary analytic nuclear Hessian XC terms | 2 | `hessian/xc_terms.rs` and `mgga_terms.rs`, plus basis/grid/metric/Pulay terms |
| Energy-only Taylor coefficient of order n | Through n | Normalized series, mixed polarization and parameter-path conventions |
| Potential/Fock Taylor path through n | Through n+1 | `response/field_path.rs` consumes `semilocal_taylor_potentials` |
| Independent mixed directional potential path through n | Through n+2 | Full mixed source, not one sampled directional energy coefficient |
| Cubic energy/beta/Raman and quartic energy/gamma contributions | Generally through 3 and 4 | Audit the requested energy/potential path; dense order-3 support does not certify fourth-order or arbitrary-order Taylor response |

Stationary reduced energy routes may exploit the 2n+1 rule. Nonvariational
distinct targets and PT2 require their actual orbital/amplitude response order;
the stationary shortcut is not transferable merely because both use jets.
One registry-direction cubic check is not full third/fourth tensor acceptance.

### Shared assembly and solver contracts

Check once per distinct path: packing/permutation/factorials; inactive zeros;
rho/sigma/tau-to-AO contraction; kernel action against finite Fock differences;
self-adjointness where physically applicable; dense versus iterative solves;
physical residual reconstruction and constraints; and observable contractions
against independent complete-component finite-field or nuclear references.
A synthetic polynomial provider can populate every descriptor slot, while
representative physical providers check that the real interface is connected.

| Distinct path | Representative integrations still required |
|---|---|
| LDA / GGA / meta-GGA | At least one real model per descriptor path and relevant spin sector; tau crosses and kinetic contractions are a separate path |
| Full and range-separated exchange | HF/global hybrid and screened/full-range limits; separate ERI kernels, omega dependence and recipe coefficients |
| ROKS | Coupled closed/open/virtual flag-projector Hessian and Berry weights; UKS evidence does not replace it |
| GKS | Complex full-spin source phases, spin-covariant descriptors, transverse potentials, angular convergence and current terms; a collinear limit is complementary |
| ECP | Core/effective-charge and angular-projector geometry coefficients through each required order, basis angular channels, and selected ECP+functional+reference integrations |
| Double hybrid/PT2 | Orbital transforms, physical pair equations, separate SS/OS and parent/target terms, complex coefficients and constraints; equal-coefficient generalized PT2 does not certify unequal-SCS or distinct targets |
| Nonlocal/learned/PCM and extra sources | Explicit provider connection, source/geometry derivatives and refusal of unavailable combinations; local jets omit these backends |

Special domains remain separate: density cutoffs, empty spins/gradients,
Fermi-hole-curvature projection/seed policy, iso-orbital endpoints, piecewise
SCAN-family joins, screening limits and deliberate smooth-reference variants.
High-order smoothness must be established at the actual points and paths.
For example, current spinor nuclear preparation refuses several
nonlocal/learned/PCM/DF/non-Coulomb combinations; catalogue availability is not
evidence that those branches are connected.

### Applied scheduling

The existing `tools/verify_release.py NAME` first runs a dedicated module-loading
gate. Cargo supplies the package, targets and feature graph. The gate enumerates
all distinct named root-feature closures, including the `default` flag even
when its dependency list is empty. Each configuration runs Clippy on all Cargo
targets with `--forbid=clippy::duplicate_mod` and warnings denied. The gate also
sets `--cap-lints=forbid` so inherited warning caps cannot conceal a duplicate.
Inherited caps are removed only from the check's environment: adding another
cap flag alone does not reliably re-enable Clippy analysis. Other compiler
flags and the caller's environment are preserved; malformed caps fail planning.
The manifest
also explicitly forbids that named lint. Local `allow`/`expect` attributes cannot
lower it; no same-target duplication exceptions are registered. Legitimate
`use`/re-exports, separate compilation targets and mutually exclusive `cfg`
branches are not duplicate module loads.
A planning error or compiler failure stops before Rust tests, builds or reused
Python artifacts can produce a release verdict, preserving the original exit
status and separate per-configuration logs. Feature inventories above eight
names or 32 effective configurations fail for explicit review rather than
silently sampling. The current package has six effective feature configurations.
Configurations run serially; Cargo defaults to one build job unless the caller
explicitly supplies `CARGO_BUILD_JOBS`.
`--python-only` remains a diagnostic mode that reuses an existing build, without
claiming Rust source validation.

Before the full Python suite, the launcher runs the existing pure typed-record
export test against the selected Python-feature Cargo target. It sets
`DFT_RS_TEST_FEW_CENTER_MODELS` for both CLI passes and retains the generated
bundle and producer log in a unique validation directory. A failed exporter,
missing output or invalid JSON stops validation before pytest. This step runs
no SCF and makes no chemical-accuracy claim. `--python-only` never compiles;
when selecting the fixed few-center public tests, supply that variable from
a matching previous export. Other explicitly selected Python tests need no
model bundle.

`python tools/verify_release.py modules --module-only` runs just this gate;
`--list-module-checks` prints its Cargo-derived plan without compiling.
`--module-target TRIPLE` selects an installed target for module-only checks.
The [compiler-aware Clippy implementation](https://github.com/rust-lang/rust-clippy/blob/master/clippy_lints/src/duplicate_mod.rs)
tracks compiler-loaded out-of-line modules and canonicalizes their source paths.
It handles ordinary resolution, `#[path]`, nested modules, lexical/symlink
aliases, `cfg_attr`, macro-generated module declarations and test-only modules.
The dependency-free Cargo fixtures in `python/tests/test_module_loading.py`
verify those boundaries and both rejection and legitimate-sharing cases.
This is not a regex model of Rust syntax or a scan for identical file contents.
Coverage is for the selected target and actual Cargo/build-script environment;
other target triples and custom cfg environments require separate invocations.
Textual `include!` reuse inside inline modules and doctest snippet ownership
are not claimed by this out-of-line module lint. Identity is by canonical path,
not by equal contents or distinct hard-link names. Existing Rust compilation,
doctest and numerical gates retain their separate roles. Virtual workspaces
require package-level runs instead of an implicit partial scan.

`tools/lint.py` also audits product text and the existing Cargo distribution inventory for
Japanese prose, private local paths, unapproved contact addresses, machine names
and conversation identifiers. It reports only file, line and rule, never the
leaked value. Python matrix operators are not contact addresses; Greek symbols
and scientific notation remain valid. Narrow exceptions preserve declared
package-author metadata, required license/third-party notices, reserved example
contacts, and original bibliographic entries with DOI/year provenance. These
exceptions do not permit private paths or conversation identifiers. Original
approval records stay in excluded private evidence; public documents describe
their scientific scope without copying conversation identifiers.
The existing portable-build and PEP 517 hooks share this policy. Actual wheel
members and source archives are checked too, including private work/cache
exclusions. Failed archive audits preserve the bytes and fail the build.
Source and wheel text classification is shared, including extensionless
`LICENSE`/`NOTICE` files: required attribution stays intact, while private paths
or conversation identifiers remain forbidden inside legal notices too.
CycloneDX dependency-author contacts are retained only at standard author
locations for matching package name/version/purl entries, using the authors in
resolved Cargo metadata. Arbitrary properties do not receive that exception;
dependency identities, author attribution and graph links are not rewritten.

The existing `tools/api_reference.py --check` also resolves explicit Python
exports to documented declarations and rejects missing export lists, duplicate
exports, undefined names and conflicting imports. Its mutation regressions in
`python/tests/test_api_documentation.py` deliberately break each contract;
the audit requires no compiled extension. The common-driver registry test
checks callable native/ASE entries and identical top-level Python aliases.
`test_objects.py` rejects fractional or out-of-range atomic numbers and
fractional supercell repeats before they can silently change a structure;
NumPy integer inputs and supercell atom order remain covered. These checks
run in the existing Python suite alongside physical route and unit tests.
The source registry also supplies lightweight forwarding tests for every
common-driver wrapper, including required keyword arguments and compatibility
adapters. Public CLI/Python/ASE name collisions and missing registry rows in
the aligned manuals fail the source-only audit. Direct calls and structure
objects share atomic-number and periodicity validation. Frequency aliases
are checked against resolved metadata and all complex alpha entries on a
representative real reference; conflicting aliases fail explicitly. These
interface checks do not certify additional physical model combinations.
The retained Python response regression exposed an obsolete expectation of
three directional evaluations for three repeated electric directions. The
current centered polarization evaluates four nonzero paths (-3d/2, -d/2,
d/2, 3d/2). Its metadata expectation now checks four on both native and ASE
routes; every existing dipole-series, mixed-derivative and residual tolerance
is retained. The initial failure remains in the API audit logs.

The current plan retains every per-functional provider/recipe test and chooses
new physical acceptance by uncovered code path. T139 checks HF and PBE
integral-direct gradient assembly; T017 checks HF moving-geometry/dynamic Raman
assembly, not all XC nonlinear contributions. Its LDA/GGA/tau nonlinear
representatives remain separate unless their existing tests establish them.
T120 uses one isolated restricted-HF singlet-TDA surface with complete
gradient/state/curvature acceptance rather than a sweep of all names.

The existing provider/Taylor and solver framework suites are reusable contract
targets (`p27_xc_taylor`, `p24_response_framework`); the seven-variable spinor
descriptor suite (`p244_spinor_xc_taylor`) is a distinct boundary. Their exact
passed subset and source/artifact identities must accompany reuse. In
particular, p27's single rho direction and p225's registry direction do not
replace complete higher-order mixed tensors or molecular acceptance.

No end-to-end test is removed. Additional all-names x all-properties runs are
not scheduled when a documented, unchanged interface and representative path
already cover the risk. New descriptor, reference, source, range, boundary,
constraint or geometry mechanisms add an edge to the coverage graph and need
their own integration evidence. The current evidence boundaries are recorded
in [validation-matrix.md](validation-matrix.md).

## Ledgers

### Status vocabulary

These definitions are authoritative for the CSV, Markdown method matrices
and implementation catalogues. Exact machine columns use the lowercase
tokens below; prose capitalization and the stated legacy aliases have the
same meanings. Long `implementation_status` descriptions are scoped notes,
not additional enum values. A test filename alone does not establish that
it was executed on the current artifact.

| Label | Meaning and assignment/evidence requirement |
| --- | --- |
| `implemented` | The algorithm and required connection exist. This alone makes no verification claim; the declared public scope must compile and be reachable. |
| `conditionally_supported` | A complete route exists only for the stated reference, provider, environment, order and input restrictions. Name those restrictions and retain guards outside them. |
| `unimplemented`, `not_implemented` | The complete requested method/route is absent. Adjacent operators, a parser or a lower-order implementation cannot establish it. The latter is a prose/legacy alias; CSV availability uses `unimplemented`. |
| `internal_only_prototype`, `backend_only` | An internal/experimental Rust backend exists without the complete declared common-driver/native/ASE/CLI connection or acceptance. A callable Rust prototype does not establish the other surfaces. |
| `implemented_subset` | Legacy method-matrix label: code is connected for a bounded subset. Read its validation state separately. |
| `implemented_not_yet_validated`, `implemented_unvalidated` | Code exists, but the independent acceptance required for the exact method/reference/order is absent or incomplete. Compilation is insufficient. |
| `verified_within_stated_scope`, `validated_subset` | Named gates actually passed for the specified inputs, method, order, units and artifact. Derivatives require independent component-wise multi-step evidence. Public claims also need actual public-route tests; no inference to other references, orders or environments. |
| `partial` | Some bounded gates passed; other stated gates are unexecuted or unresolved. Identify missing coverage. This is not full acceptance. |
| `partial_with_failures` | Some checks passed and specific retained failures coexist in the row's scope. Identify failed models/components and counts in scope/evidence; do not hide them in a pass total. |
| `untested` | No executed acceptance for the exact target/scope is recorded. Source and test preparation alone do not change it. |
| `failed`, `FAIL` | An executed criterion was not met. Keep its raw result, criterion and scope. Complementary evidence can resolve a distinct criterion without erasing the original failure. |
| `PASS`, `passed` | One executed gate succeeded, with assertions and process outcome when available. This labels that gate, not a whole feature or release. |
| `not_audited` | Records have not established implementation/applicability or verification for this class/reference. It means unknown, not automatic support or definite absence. |
| `not_applicable` | The dimension has no mathematical role in this route, such as XC quadrature for a grid-free HF operator. Explain why; this differs from missing implementation or silently dropping an applicable setting. |
| `pending`, `planned`, `not_started`, `deferred` | Work or acceptance remains outstanding. Record whether implementation, connection or validation is missing and why it is deferred. Adjacent components do not complete the target. |
| `source_only` | A proposal/edit is recorded without required compilation/actual-artifact tests. Do not advertise it as compiled or accepted. |
| `experimental` | A modifier indicating bounded assumptions, validation or applicability. It does not replace the availability/validation columns or imply chemical transferability. |
| `running`, `in_progress` | An identified live process/stage is actually active. Historical records can keep original wording; current summaries must change after termination. |
| `stopped`, `interrupted` | Execution ended before its full scope, including a release freeze. Completed results remain bounded evidence; omitted work is uncompleted, not a numerical pass or automatic scientific failure. |
| `unsupported` | A valid request lacks a supported reference/model/environment/order route. Rust returns `DftError::Unsupported`, Python `NotImplementedError`; another reference is not substituted. Malformed/unknown input is a separate input error. |

This release retains historical scientific evidence without an exhaustive
new physical sweep. CSV partial/untested states are conservative; per-method
`validated_subset` cells retain only their stated historical scope. Numerical
fallbacks must identify their method and cannot count as analytic support.
Required license and scientific-author attribution, the project author
handle and existing author contacts are preserved. Private local paths,
hostnames and conversation identifiers are not distribution content.

- `docs/property_targets.csv` lists the physical targets (one row per
  target, including the functional/reference matrix and evidence columns) with
  their implementation and validation status, API entry points and tests.
  The appended `availability` column describes implementation/public
  applicability, independently of `validation_state`. Existing identifiers,
  columns and detailed historical evidence remain compatible. Availability
  is `implemented`, `conditionally_supported`, `unimplemented`,
  `internal_only_prototype`, `not_audited` or `not_applicable`. Validation is
  `verified_within_stated_scope`, `partial`, `partial_with_failures`,
  `untested`, `failed`, `not_audited` or `not_applicable`. Partial historical
  evidence does not certify the current whole-library scientific matrix.
  A row spanning several models uses its scope/evidence cells to identify
  exactly which subset passed, failed or remains internal. In particular,
  an internal fixed-grid prototype or numerical fallback is not public
  analytic support. The earlier `implementation_status` prose is retained
  as a detailed compatibility field, not a universal validation verdict.
- The [response coverage ledger](response.md#response-coverage-ledger)
  lists every observable and perturbation with its required scope, status,
  entry points, tests and the functional-class by reference-type matrix.
  `tools/ledger_check.py` parses its **Status** and **Tests** lines.
- A status names what is implemented and what is validated; a row whose
  tests are listed does not also say that acceptance tests are pending.
  `tools/ledger_check.py` (run by `python/tests/test_ledger.py`) checks the
  ledgers against the registry and the test files.
- The [functional catalogue](FUNCTIONALS.md#functional-catalogue) records
  the status of every requested functional.

## Documentation

- The three API manuals keep aligned tables in registry order. A `|` inside
  a code span in a Markdown table is written `\|`.
- Documentation describes the package itself; it contains no local paths,
  machine details or session history.

## Changelog and releases

- `CHANGELOG.md` follows Keep a Changelog; entries go under
  `## [Unreleased]` and version headings carry no dates.
- A release is built by `tools/make_release.py --zip` from the `include`
  whitelist of `Cargo.toml`, and its content is audited: local histories,
  logs, environments, reference inputs and build directories are excluded,
  and no machine-specific path may remain. `tools/verify_release.py` runs
  the full Rust regression and the Python suites in a frozen copy. Invoke it
  with a Python interpreter containing `.[test]` dependencies. The verifier
  stages the compiled extension and both CLI names itself, runs the existing
  pytest suite with each CLI and preserves nonzero statuses in per-command
  logs under `validation/`; no external validation runner is required.
  `--python-only --target-dir TARGET` reuses a release build. Optional
  `--test-path PATH` is allowed only with `--python-only` for diagnostics;
  that subset is not full release acceptance.
- Include the public `docs/` tree and coverage ledger in both Python
  artifacts. The wheel installs them in `dft_rs/docs/` and includes every
  documentation file in its wheel RECORD. The sdist keeps them in
  `docs/` at the source root, with the original file contents.
- Verify the actual pip artifacts with `python tools/verify_release.py NAME
  --installed-only --wheel WHEEL --sdist SDIST --rust-cli PATH`. Each install uses a separate
  environment managed by pip from the verifier's Python interpreter.
  Before installation, compare the sdist's source inventory and the
  wheel's public documentation by SHA-256 against the frozen sources.
  The check runs the installed `dft-rs` console script from
  outside the source tree, verifies its version and registry, and compares
  molecular and periodic reports to native Python and the standalone Rust CLI
  with no optional dependencies. It then installs the declared `ase` extra in each
  environment and compares the same energy and stress reports with ASE.
  Before and after dependency installation, pip's recorded source archive
  must match the SHA-256 of the supplied artifact. The verifier saves the
  input archive and standalone CLI hashes in the validation logs.
  Build the required standalone CLI from the same frozen source. Both omegaB97X-D and
  omegaB97X-V are mandatory molecular acceptance functionals. Preserve their
  CHG dispersion and variational VV10 paths in the scientific regression
  and in the installed constrained DFT and Car-Parrinello MD checks.
  Add `--wheelhouse DIR` to resolve the declared sdist build and ASE
  dependencies without an index.
- Full regression acceptance requires the full Rust suite with
  `DFT_RS_STRICT_UNITS=1`, the Python suites, the registry and documentation
  tests and the ledger check to pass on the same source. The v0.3.0 local
  source-ZIP milestone instead records its bounded API/package acceptance
  explicitly. Its six-configuration all-target checks compile tests; they
  do not execute every scientific test. Unexecuted tests and known physical
  failures remain disclosed limitations, not inferred passes. See
  [the delivery scope](../CHANGELOG.md#remaining-scientific-work).
