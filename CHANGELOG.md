<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 ss0832 -->

# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This project is 0.x and was never published, so it does not yet promise
API stability. `DftError` is `#[non_exhaustive]` so that naming a new
refusal is not a breaking change.

## [Unreleased]

## [0.4.0]

- Publish the current numerical verification record with passing cases,
  retained failures, deferred work and incomplete runs listed in the
  implementation status and response ledgers.

- Add molecular IGMH fields from the converged SCF charge density. Sample
  points or affine grids and split the interaction field into inter- and
  intrafragment contributions across the common interfaces. Export matching
  interaction and signed-density CUBE files, SVG slices and a rotatable
  browser view with an IGMH/NCI scatter plot in Python. The self-contained
  HTML packs the full mesh and sampled points, shows the full NCI gradient
  range, and lets users set unsigned or signed density filters and
  signed colour limits.

- Differentiate the same real-space nuclear attraction image sum in the
  periodic SCF, force and Gamma Hessian paths. Reuse batched shell-pair
  integrals and check all ordered operator entries on two Ewald sums.

- Document how to find the same calculation across Rust, CLI, native Python
  and ASE. Add route and unit examples, fill periodic verification pointers
  and update the scoped finite-frequency double-hybrid status.
- Consolidate verification contracts in the implementation conventions
  and target-specific measurements in the validation matrix. Keep the
  shared calculation descriptions in one API reference, with links from
  the corresponding CLI and Python/ASE sections.

- Add common-driver routes for molecular density constraints, Car-Parrinello
  trajectories in a moving Löwdin frame, polarized finite-basis near-edge
  spectra and the orbital/Pauli magnetic quadrupole. The model and numerical
  acceptance scopes are recorded in response ledger entries R082–R085.
- Require omegaB97X-D and omegaB97X-V in the installed constrained DFT
  and Car-Parrinello checks. Add the open-shell UKS/ROKS trajectory
  regression and cover both functionals in the moving-frame derivative tests.
- Add multi-atom electron population constraints with analytic forces,
  constrained geometry relaxation and static electric response. Share the
  molecular VV10 charge kernel with the full spinor response.
- Extend the moving Löwdin Car-Parrinello integrator to complex GKS
  orbitals with Hermitian position and tangent constraints.
- Include variational periodic VV10 in collinear analytic forces and
  stress. Reuse the pair derivatives for field-free wires, slabs and
  bulk cells.
- Use the SCF overlap for dimensional XC electron-count diagnostics
  when the grid AO image radius differs from the overlap cutoff.
- Connect periodic VV10 energy, forces and stress to the fixed-integer
  common-orbital ROKS paths, with a shared bulk XC grid builder.
- Add catalogue-driven molecular gradient and static polarizability
  tests that retain per-provider failures and all Cartesian fits.

- Connect the translated scalar ECP fixed-density Hessian to the existing
  periodic CPKS force constants and phonon spectrum drivers.

- Reuse the molecular scalar ECP recurrence for second-order periodic
  nuclear and affine paths, including the mixed chain-rule term.
  Extend correlated phonon acceptance to the charged scalar ECP cell.

- Connect retained scalar ECP image derivatives to the first-order periodic
  lattice path used by collinear double-hybrid energies, forces and stress.
  Document the separate XC AO range and strain-step convergence checks.
- Report the actual collinear reference type in periodic response reports.
  Connect neutral three-dimensional collinear double-hybrid stress to
  the local Berry electric enthalpy at fixed Cartesian bias.
- Share the reduced-field projection between fixed and moving periodic
  correlated paths.

## [0.3.1]

The v0.3.1 local source-distribution checkpoint collects the work since
v0.3.0 listed below. It is a stopping point, not a claim of exhaustive
scientific validation: the full Rust and Python regressions were not rerun
on this exact source. The release checks on the staged source are the
all-configuration Clippy module gate, clean wheel and source-distribution
installs with public calculations through Python, ASE and both console
scripts, and the archive audits of `tools/make_release.py`. The last
combined candidate before this checkpoint passed Clippy, the release build,
six exact-source integration suites and clean wheel/source installs. Its
post-run audit first stopped because the runner expected 81 components in
the collinear Hessian log, which holds 9 gradient and 81 Hessian
components; with the count corrected the independent refits pass 90 of 90
(p = 1.979 to 2.006, minimum R2 0.99997) and 1560 of 1560 for the
collinear Hessian matrix (44 failed first intervals retained and resolved).
Full trust-region response-matrix acceptance, the
remaining reaction families of the continuous-substituent model and the
gaps listed under 0.3.0 remain open.

- Add shared trust-region augmented-Hessian molecular SCF adapters for
  independent spin, constrained common orbitals and complex spinors.
  Preserve all physical convergence gates, share immutable integral caches,
  and resolve small energy work through analytic Fock density-path quadrature.
  Full response-matrix acceptance remains in progress.
- Route unrestricted NMR spectra through the existing complete Pauli Ramsey
  provider and retain coupling residuals in spectrum convergence. Add
  constrained common-orbital and complex-reference shielding checks.

- Add a task-oriented API guide with a shared H2 example for Rust, CLI,
  native Python and ASE, explicit unit and convergence interpretation,
  functional composition limits, and a guide to similar calculation names.
  Link it from the README and all three aligned API manuals.

- Add explicit unit and force/gradient types, owner-checked indices and an
  immutable molecular SCF facade with checked convergence capabilities.
  Reuse the existing SCF, gradient and Hessian kernels; unfinished iterates
  expose trial energies without acquiring stationary-state operations.
- Keep common fragment reference energies outside SCF arithmetic and expose
  reference-free total/classical energies through existing continuous,
  few-center and generated-reduced model reports. Fixed replacement charts
  preserve integer electron ledgers and a common operator space; generated
  substituent chemistry remains separately unvalidated.
- Resolve the nodal tracked-orbital force validation interval while retaining
  original failed intervals, independent fits and unchanged scientific gates.
  Match the saved-fragment periodic-refusal test to the established ASE
  exception contract.
- Validate the bounded correlated magnetizability tensor and causal-PT2
  response paths from dedicated frozen-source builds; retain full-program
  coverage gaps and raw failed/resolved derivative intervals.

- Add weighted XC specifications to the molecular RKS, UKS, ROKS and
  GKS energy and analytic gradient driver, with reference-specific SCF
  residuals and explicit guards for unconnected operators.
- Reduce transient copies of molecular SCF density and Fock matrices.
- Route the dedicated periodic GKS/GHF stress command through the existing
  analytic spinor strain derivative.
- Check pip-installed console scripts from both distribution formats. Fix
  Windows source ZIP publication and metadata path handling.
- Record repeated wall time, peak process memory, managed memory and SCF
  diagnostics for paired CLI benchmarks.

## [0.3.0]

The third-party provenance review preserves all bundled upstream notices,
data headers and license texts. The Python license-file selection now also
includes complete third-party `LICENSE` files, correcting the omission of
the BSE BSD notice from wheels. Provenance and license coverage have
permanent packaging regressions; no numerical or Python API payload changes
are part of this metadata correction.

The bounded source-distribution checks use official Rust 1.85: all six
Cargo/Clippy configurations pass for 347 targets each, followed by a complete
development-profile library/CLI build and execution of the README Rust
consumer. The actual PEP 517 wheel installs offline into a fresh environment
as `dft-rs-python==0.3.0`, imports as `dft_rs` and executes public calculations.
The public suite completes 563 cases: 562 pass in the retained initial run;
the sole failure confused a source-only settings-audit comment with a public
registry field and passes after a test-only correction. All 170 source/package
guards pass after repairing missing test-environment executable settings, and
five Rust common-setting propagation/validation tests pass. Original failed
records are retained. One mixed continuous-substituent numerical test stays
deselected under the model freeze; its ordinary resource checks run separately.
An extra legacy large-grid scientific oracle remains incomplete. These checks
do not certify the entire scientific suite, every platform/Python version, or
the default optimized fat-LTO release profile.

The v0.3.0 local source-distribution milestone prioritizes public API,
common-setting and packaging consistency. It does not claim exhaustive
scientific validation. Work is consolidated to one source owner; further
scientific expansion and batches are paused. Original failures are retained.
The repaired PBE nuclear-A run stopped after 79/90 directions and 6004/6840
component fits (5974 pass, 30 fail, 836 uncompleted), rather than passing its
full matrix. General SOC/RO/spinor nuclear EPR, broader periodic physical
convergence, full experimental NEO nuclear-Hessian acceptance, and reduced
continuous-model chemical accuracy remain incomplete.

The separate nuclear-g adapter exposes only integer all-electron HF/UHF
Pauli/London first and second Cartesian derivatives, with explicit inverse
length units and retained real/mixed response diagnostics. Static order zero
keeps its existing routes. Nonzero electric fields, stability following and
RO/spinor/DFT nuclear requests remain explicit capability errors. A nonzero
electric-field reference-origin inconsistency was reproduced and remains
future work; guards were not relaxed. Exact-artifact public tests are required
for this source increment independently of older static acceptance.

Common-setting propagation now retains requested HF property quadrature,
independent VV10 resolution, dimensional-grid controls and explicit periodic
operator/grid tolerances. ASE no longer rounds integer settings or discards
configured SCF mixing, temperature, initial guesses and additional driver
controls. Dynamic optical linewidth and SCF damping have distinct arguments.
GKS and nested integer-SOC SCFs derive applicable explicit settings from the
resolved common reference, while unspecified legacy SOC defaults are retained.
One actual compatible library and offline-installed `dft-rs-python` wheel pass
109 settings/public-route cases and five Rust grid/configuration regressions;
24 source/API guard tests cover all170 explicit registry applicability
declarations and reject an omitted declaration. This is targeted propagation
evidence, not numerical acceptance of every property/reference combination.
Known but inapplicable controls raise `NotImplementedError`; unknown names
remain invalid input (`ValueError`), and failed SCF convergence remains an error.

Rust installation examples retain a local checkout path and an explicit matching
version without implying registry publication. The release version updater and
guards support both string and inline Cargo dependencies, preserve paths and
feature lists, update current-version labels and refuse ambiguous versions.

PBE correlation now evaluates the algebraically equivalent tail enhancement
without subtracting nearly equal energies. A centered `exp_m1` evaluation
preserves the complete AD/Taylor dependence, and the shared PW92 logarithm
uses `ln_1p`. Functional parameters, density cutoffs and convergence criteria
are unchanged. Independent 80-digit references and all 120 dense/runtime
gradient and ordered-Hessian four-step fits pass; ordinary-density, uniform,
fully polarized and zero-density provider checks are retained separately.
These pointwise conditioning corrections do not establish the cause or
resolution of the original PBE nuclear-A baseline: all 6840 entries were
attempted, with 6795 passes and 45 retained failures. Exact new shared-artifact
and affected molecular acceptance remain pending.

The direct `spin-spin-zfs` common driver now connects Rust, native Python,
ASE and both CLI names for integer all-electron HF/UHF with S>=1. Strict
`nuclear_order=0/1/2` returns D, its nuclear gradient and its nuclear Hessian
with unrequested blocks null. SOC, spin purification, total D and unaccepted
ROKS/GKS/DFT nuclear-D combinations remain explicit exclusions. One actual
compatible shared library and clean offline-installed `dft-rs-python` wheel
pass all 25 public cases, including units, raw convergence diagnostics,
partial logs, typed unsupported errors and managed-budget rejection. All
387 D/gradient/Hessian values agree with the retained Rust-handler output;
this is cross-artifact API parity, separate from the 810 independent backend
fits and seven adapter tests. The three missing canonical test files are
restored; the p312 first-derivative helper repair preserves all samples and
thresholds and passes warning-strict metadata. Final ZIP and complete
EPR/PBC/NEO scientific acceptance remain pending.

### Remaining scientific work

This local source-ZIP milestone prioritizes API, common-setting, installation
and package checks. The unfinished scientific items below are disclosed future
work, not prerequisites for delivering this partial milestone. No further
scientific batch is required or implied by this checklist. Source availability,
an enum selector, an unsupported error or an API smoke does not close a physical
calculation. Full-program scientific acceptance remains separate.

- [x] Retain the resolved ferrocene11/11 component comparisons for both recipes;
  preserve old unstable references and independently justified corrections.
- [x] Retain EPC17 underflow/alias corrections and their bounded kernel, NH3
  field-response and public-interface evidence without claiming all NEO scope.
- [x] Complete the bounded Gamma HF/PBE periodic common-orbital reference-energy
  path, independent orbital-component fits and native/ASE/both-CLI checks.
- [ ] Close methyl GKS with the original orbital, occupation, operator/XC and
  fresh physical-reference gates; the H4 subset does not close methyl.
- [ ] Finish all6840 PBE nuclear-A fits and exact-artifact common-driver
  integration, preserving original failed intervals and corrected diagnostic orders.
- [ ] Implement and validate the requested remaining g/D nuclear property
  derivatives and SOC prerequisites; energy/operator Hessians are distinct.
- [ ] Finish periodic ROKS and general GKS forces/stress and their actual
  response/property routes in the requested method-by-property matrix. Reuse
  existing phonon/elastic/Berry/Born/piezo/IR/band postprocessing only with
  physically valid reference-specific tensors; retain all unsupported domains.
- [ ] Establish independent real-crystal validation and numerical-support
  convergence; retained LiF/NaCl pressure discrepancies are not removed by fitting.
- [ ] Close remaining NEO coupled-SCF/response cases and full-component
  geometry acceptance, distinguishing older TPSS failures from newer accepted subsets.
- [ ] After EPR prerequisites, finish the preserved48,412 C1 HF/PBE/TPSS/PBE0
  responding-PT2 cubic/quartic fits with the original gates and actual native exits.
- [ ] Reconcile one final compatible source/native identity, required exact-artifact
  regressions, API3/ledger/lint, units, resource/logging contracts and documentation.
- [ ] After the API/package checks, create the requested reproducible v0.3.0 source ZIP: include
  implementation, needed licensed bundled data, tests, examples, licenses and
  existing documentation; exclude .work, caches, environments, private/reference
  data, machine/personal paths and unlicensed external content. Private delivery
  of the archive does not authorize a tag, push, public upload, publication or
  package-index registration.

Continuous-species work is frozen at the user's stop instruction. Its interrupted
optimization, failed model accuracy and incomplete tests remain documented with
future directions; they are neither restarted nor relabelled as accepted chemistry.

The existing source-release helpers now share Cargo's include inventory and
the mandatory source/privacy policy. Staging uses a fresh destination without
deleting previous results; ZIP creation verifies package identity, exact
membership, per-file SHA256, CRCs and required licenses before exposing the
archive, and refuses to replace an existing output. Twenty-one synthetic
archive/staging regressions and 48 existing build-backend regressions pass.
That earlier draft inventory contained 1,581 source members and was not final
source or scientific acceptance. The delivery record identifies the actual
frozen inventory and the bounded checks executed for the delivered archive.


Periodic common-orbital ROKS now has an actual integer Gamma reference-energy
path on the existing periodic J/K/XC operators. The non-symmetric H3/HF and
H3/PBE cases converge in11/10 iterations and pass all six independent
four-step orbital-rotation fits; closed-shell energies agree with existing
periodic RKS within6.7e-16 Hartree. The bacf85/2b60bd successor passes canonical
Rust and native/ASE/canonical/alias CLI reference-energy checks. Core/CDIIS
scope, raw constrained stationarity and explicit unsupported response domains
are documented. A later full-rank Gamma increment now connects analytic
ROKS nuclear gradients/forces and affine stress; all36 HF/PBE ordered
four-step energy fits pass with unchanged gates after explicit XC AO-range
convergence. Original PBE6-Bohr image-cutoff failures are retained.
The b895d970/80b95343 shared-core repeat passes15 unique cases and all8
raw native/ASE/both-CLI reports; original interrupted attempts remain retained.
Phonons, relaxation, finite-k
ROKS and general GKS response remain unclosed. The inactive ADIIS default-forwarding correction is Python-only;
its explicit-control regression passes and prior failed public attempts remain.

NEO EPC17 positive-density product-underflow and alias-channel corrections now
pass 64 pointwise fits, all nine bounded C1 NH3 field-response fits and targeted
Rust/native/ASE/both-CLI checks on one corrected artifact. Historical failed
intervals remain retained. Current source also connects scoped dedicated static
EPR routes and the HF nuclear-A report adapter, whose exact order-index repair
is verified; final consolidated EPR acceptance, methyl GKS and the
full PBE nuclear-A batch remain unfinished. Supplied-tensor proper d=eS now
passes all18 independent fits and joint03 Rust/native/ASE/both-CLI checks;
27 bounded public cases and four Rust tests pass. GKS/GHF HF two-k mesh
band reports also pass, while periodic ROKS and general GKS property response
remain unimplemented. The same joint core passes the two-model NEO public
response and mixed-alias channel smokes. No release/publication is implied.

Continuous-species work is paused at the requested stopping point. The explicit
experimental `rigid-static-core-v1` interface is connected through the existing
Rust/Python/ASE/CLI entry, with separate center and Euler unit blocks. One shared
Rust 1.85 library/extension builds and passes strict Clippy/source hygiene; the
installed local wheel passes 17 unique bounded native/ASE/CLI contracts and
three linked Rust contracts. The same core passes 42 moving-dipole component
fits and three budget/overflow/reservation checks. A nonzero explicit-record
energy/12-gradient comparison agrees within 1.43e-14/1.27e-14 native units.
The original two ordered force-difference failures remain, with separate
passing mixed-energy witnesses. Final public-Hessian tests were timed out or
cancelled, not accepted; follow-on common-suite/alias/sdist checks were not run.
No additional geometry minimization was performed. Model accuracy and transfer,
including the documented ethane geometry failure, remain future work. See the
[scoped API record](docs/rust-api.md#continuous-chemical-species-model-contracts).

The Python distribution is now named `dft-rs-python`, with `dft_rs` imports
and the `dft-rs` console command. The Rust crate keeps its `dft-rs` name.
Wheel/sdist metadata and artifact-name checks enforce the Python identity.
This is local development packaging; no index publication is implied.

The installed shared-core wheel additionally passes all 14 bounded
HF-collinear nuclear-A public cases, including signed nuclear g factors,
ordered targets, ASE atomic/converted units and both matching CLI binaries.
The current 162-case capability/error module passes on the same installed
core: 20 new malformed ASE/CLI selectors, 48 dedicated static EPR selector
cases and 94 remaining regressions. Dedicated static ROKS/ROHF/GKS/GHF
selection no longer has a stale blanket-rejection test. This is public API
and installed-package evidence, not acceptance of PBE or g/D nuclear
derivatives, methyl GKS, or the newer periodic ROKS source.

<!-- v0.3.0 API consolidation: begin -->
### Bounded v0.3.0 integration milestone

The current cycle consolidates the ferrocene investigation, molecular
continuous-species experiments and EPR work, with their API and packaging
regressions. This development checkpoint does not certify the complete
physical program or publish a release.

At the earlier bounded API checkpoint, the hyperfine/g-tensor preflight
rejected then-unconnected ROKS/ROHF/GKS/GHF selectors before SCF. The dedicated
static providers described above supersede that blanket refusal; their
nuclear derivative and model restrictions remain separate. The common
capability contract uses typed `Unsupported` errors for unavailable IDs,
adapters, reference combinations and known controls in the wrong context.
Python/ASE preserve `NotImplementedError`; both CLI names fail nonzero.
Unknown controls, malformed selectors and missing required inputs remain
input errors. Routing no longer consumes selectors before their handler;
reused Rust Controls cannot hide a previously read unsupported request.

One actual Rust 1.85 root build supplies the Rust library and Python core.
Fourteen exact-source pure units and 42 linked Rust integration tests pass.
The installed local `dft-rs-python` wheel passes 142 unique native/ASE/CLI
capability cases, 120 logging/resource/saved-model/ASE/thermal cases and
the multicomponent invalid-input contract. Both optional legacy carbon
templates were actually enabled and tested. The compatibility CLI passes
28 capability and 14 logging/saved-model cases; the routine release Python
entry point passes the 142-case module. Local wheel/sdist identity,
isolated offline name-based installation, version/core-hash and portability
checks pass. Earlier runner, fixture and link-timeout failures remain in
the private evidence; no physical threshold was weakened and the healthy
root library was reused for recovery.

This acceptance covers the explicit temporary EPR guard, not newly developed
reference extensions or EPR nuclear derivatives. Those need their own
scientific gates and shared public integration. Conditional continuous
charge-basis recipes and optimizer penalties are not advertised as accepted
controls here. Saved fragment routes keep their bounded PBE/template and
electron-ledger contracts; chemical accuracy remains unestablished.

Readiness of the final v0.3.0 milestone still depends on the separately
reported ferrocene, continuous-model and EPR outcomes. The package name is
`dft-rs-python`, the import remains `dft_rs`, and no registry publication,
tag or release-wide certification is implied.
<!-- v0.3.0 API consolidation: end -->

<!-- Complete correlated nuclear acceptance: prepared -->
### Prepared complete correlated nuclear acceptance

The complete four-parent PT2 cubic/quartic acceptance in `p312_correlated_complete_nuclear` is prepared but has not yet been compiled or numerically run. It requests C1 H4/STO-3G restricted HF/PBE/TPSS/PBE0, zero external field, grid level 0 with `grid_weight_cutoff=0`, and custom unregularized SS/OS coefficients 0.37/0.61. Every actual common-driver Cartesian tensor is checked along 364 cubic and 1,365 quartic unisolvent directions, independently for total, parent, target, weighted PT2 and unscaled SS-alpha/OS/SS-beta: 48,412 fits in total. The oracle solves fresh displaced SCFs and directional Hessian response equations. Cubic and quartic tensors use hartree/bohr^3 and hartree/bohr^4 respectively; the contracted directional Hessian uses hartree/bohr^2. Each curve uses four distinct steps, with the unchanged absolute exponent gate 0.2 and R-squared above 0.999; all failed intervals remain recorded. Every displaced orbital and PT2 response, projector, stationarity, metric, number, frame and reference diagnostic must be finite and within 1e-12. Returned original-equation and inner-correction residuals are separately checked; the unfloored reconstructed-source ratio stays visible. Test preparation and a closed-form fit-auditor check are not numerical molecular acceptance. Existing measured subsets and matrix cells remain unchanged until the complete requested runs finish.

<!-- Returned-coefficient residual repair: begin -->
### Retained coefficient residuals (bounded measured acceptance)

Validate the actual returned coefficient against its original linear response
equation before selecting the physical iterate. Keep inner-correction and
unfloored physical reconstruction diagnostics separate. Actual59 strict
metadata/library/CLI/portable checks, 14 closed contracts, five harmonic Rust
tests and five native/ASE/CLI tests pass, including fixed B2PLYP/H2. Original
failed reports and a private harness visibility failure remain retained.
The unchanged p39 third/fourth-order regression passes all five tests
on actual59 in 1811.908s. The original57
four-parent rerun has HF1092 and PBE1092 measured components; TPSS/PBE0 are
pending. New59 all4368 acceptance will additionally gate every displaced
gradient's convergence/raw diagnostics. No scientific threshold is relaxed.
<!-- Returned-coefficient residual repair: end -->


<!-- Active descriptor Taylor kernels: begin -->
## Active descriptor Taylor kernels (2026-11-01)

Specialize shared LDA/GGA Taylor potential arithmetic to two/five active
descriptors while preserving seven-slot outputs, the original nongeneric
external scalar alias and parameter/directional paths. Five warning-strict
arithmetic tests pass1390 complete first-to-fourth error fits over468
independent directions; all coefficients through order five match the
full-lane path.
Immutable56 strict library/CLI/path audits pass; physical reruns remain separate; all 624 unchanged four-class nuclear fits pass.
No derivative, residual or error-fit gate is relaxed.
<!-- Active descriptor Taylor kernels: end -->













<!-- Reference-independent harmonic analysis: begin -->
Connect ROHF/ROKS and complex GHF/GKS analytic Hessians to the existing
frequency and Gibbs calculation IDs through a shared postprocessor. Add
explicit rigid-motion projection and electronic-degeneracy controls, retain
all Cartesian modes under fixed laboratory sources and report unavailable
IR data as null. Route double-hybrid text-CLI frequency requests through
the correlated common driver. Preserve the initial B2PLYP tiny-source
relative-residual failure; tighten retained coefficient stopping to require
both existing absolute and relative targets. Four full-library harmonic/model contracts, 144 C1 component fits and
ROHF/complex-GHF native/ASE/CLI frequency and atomic Gibbs subsets pass.
Preserve the remaining correlated tiny-source reconstruction failure;
separate fresh linear residuals from unfloored reconstructed source/norm/ratio
diagnostics. The follow-up source repair and correlated acceptance remain
pending. This is not
full-program or functional-class completion.
<!-- Reference-independent harmonic analysis: end -->

<!-- Constrained static electric tensors: begin -->
Connect generic parent ROHF/ROKS and full complex GHF/GKS stationary energy
to existing static-spectroscopy, beta and gamma on all four public surfaces.
Retain flag, metric, AO/Fock and scalar-energy low parts through mixed
polarization; use exact binary path scaling for retained-capability providers.
All434 independent C1 HF Raman/hyper-Raman/beta/gamma fits, original
translation gates and public10 cases pass. Preserve every failed interval
and earlier failing checkpoint. Fix ASE initial-guess/solver-control forwarding
while retaining family defaults, and legacy authoritative-unit summaries.
The separate collinear HF+PT2 rerun passes570 fits; full PBE native/ASE
composition passes3832.47s at unchanged scientific gates, old timeouts retained.
The four-class C1 ROKS nuclear rerun and independent refit pass all 624
components, with original timed-out attempts retained and no weaker gates. No broader
functional, correlated constrained, dynamic, periodic or full-program completion
is inferred. Exact scopes and raw measurements are in response.md.
<!-- Constrained static electric tensors: end -->

<!-- Retained response coefficients: begin -->
Retain physical matrix/orbital coefficient remainders in high-order molecular response and PT2 contractions; replace subtractive semilocal tangents with common forward-dual kernels. Complete C1 HF+PT2 RKS/UKS static spectroscopy passes all 90/90 cases and 570/570 independent fits: APT90, Raman180 and hyper-Raman300. All 30 hyper-Raman coordinate/reference cases pass, including the previous 27 failures. Every ordered Cartesian slot is assembled; no tensor value is copied by symmetry. Preserve the failed coarse-grid Richardson evidence and resolve the unchanged four-step/ratio gates at quadrature levels 3 and 4; confirm the unchanged five-test earlier fixed-population retry. The final correction-loss tracking is included in this linked acceptance artifact.
<!-- Retained response coefficients: end -->









- Correct the declared Rust minimum to 1.85 for the locked dependency graph.
  Audit selected Cargo dependency requirements during portable and Python
  wheel builds and in the existing source verification suite. Actual Rust
  1.85.0 wheel/library/CLI builds and isolated installation now pass with the
  bounded opt1/codegen256 integration profile; optimized-release acceptance
  remains separate.

- Target v0.3.0 development for continuous chemical-space exploration and
  the extension of real-substituent effective-center models. This version
  change does not certify pending model or release-wide acceptance gates.
  Fixed-population closed-shell/fixed-spin and explicit local-potential,
  steric and attached-frame mechanisms have bounded numerical acceptance.
  The initial frozen methyl compression is a candidate: its nine-point
  relative-energy MAE is 0.456471 Hartree and fails chemical acceptance.
  Later candidate refinements retain separate acceptance records.
- Add per-calculation CPU ceilings and managed memory budgets through the
  existing Rust, CLI, native Python and ASE routes. Scope managed payloads
  and workspace allowances explicitly; this is not a whole-process RSS cap.
  A complete frozen library/CLI build and linked resource/cache/public
  checks pass. Correct direct-core resource argument/type handling and
  public function signatures; retain the original failures. Earlier resource06
  acceptance passes 300 Python and 24 Rust cases, three real calculation
  comparisons and the existing release entrypoint with both CLI names.
  Keep saved-model ASE getters in the same resource envelope so per-call
  overrides take precedence over constructor defaults without reconfiguration.
  The later fixed-source Rust 1.85 integration passes 318 installed-wheel
  cases, both additional AO-template cases, a saved-model CLI-alias case,
  26 linked Rust tests and all 27 core13 derivative fits. Confirm actual
  per-call resource overrides and the three numerical comparisons without
  promoting compressed-model chemical accuracy. Clarify that ASE Atoms
  positions remain in Angstrom while origin controls follow the unit mode.

- Connect static APT/Raman/hyper-Raman to the common molecular driver,
  including generic integer RKS/UKS post-SCF target and PT2 amplitude/orbital
  response. Preserve existing native/ASE names and CLI component rows.
  Validate the complete C1 HF+PT2 RKS/UKS APT/Raman subset with270
  independent component fits and nine native/ASE/CLI checks; retain all
  failed intervals and displaced-SCF branch switches. The later retained-coefficient record above completes this bounded
  hyper-Raman subset; remaining combinations and raw failures are recorded
  in response.md.
- Use full occupied/virtual Fock blocks in the common real orbital response
  frame and retain low intermediate parts in moving-metric congruences.
  All225 algebraic component fits pass their first four-step second-order
  intervals, including the shared compensated Coulomb/exchange series
  and48 dense orbital-retraction entries.
  The initial C1 molecular spectroscopy failures are resolved within
  the bounded retained-coefficient acceptance above. Reuse scalar directional coefficients across ordered static
  spectroscopy slots without storing AO paths or copying tensor entries.
- Refine the actual integer-reference orbital Fock gradient with the shared
  response kernel and SCF retraction. Warning-strict builds, portable
  artifacts and all-feature Clippy pass. The complete C1 diagnosis passes
  all 270 APT/Raman fits again plus 30 bounded UKS hyper-Raman fits; the initial 27
  hyper-Raman failures are resolved by the retained-coefficient record above.
  Independent 80-digit density/orbital traces identify remaining coefficient
  precision errors; these are retained rather than hidden in diagnostics.
- Accumulate scalar nuclear projector, commutator, particle-number and
  source convolutions with the existing compensated product sums; avoid
  rounding cancelling matrix/trace fragments separately. Physical operators,
  denominators and acceptance thresholds are unchanged; bounded numerical acceptance
  of this precision extension is recorded above.

- Resolve six EDF2 and eight omega-B2 zero-gradient reference discrepancies
  after independent high-precision/isolated-C diagnosis and explicit user
  approval. Preserve raw fixtures, failures and all scientific gates; focused
  point/local/term regressions and point regeneration pass. No production
  physics changes; standard five-molecule and broader combinations remain
  separate.

<!-- BEGIN C1 QUARTIC ACCEPTANCE -->

- Validate all 1365 independent C1 nuclear quartic directions for each of 9
  targeted models (12285 fits): g scalar-ECP RHF, g scalar-ECP UHF, s/p/d/f
  scalar-ECP UHF, h scalar-ECP RHF, h scalar-ECP UHF, h all-electron UHF, i
  scalar-ECP RHF, i scalar-ECP UHF, i all-electron UHF. Independent raw-error
  refits confirm the unchanged power and fit-quality gates; all failed
  intervals remain recorded. All 9 selected models are complete; 2 other
  models are deferred and unvalidated at the user-selected boundary.

<!-- END C1 QUARTIC ACCEPTANCE -->

- Add original omegaB2PLYP and omegaB2GP-PLYP typed range-separated recipes,
  independent spin-PT2/exchange terms and complete local derivative tests.
  Keep partial acceptance explicit: eight raw zero-gradient kernel
  comparisons fail unchanged; TPSSloc approval is not extended.


- Add the original DSD-PBEB95-D3(BJ) typed recipe with published independent
  exchange, correlation, spin-resolved PT2 and automatic D3(BJ) weights;
  verify local partials and independent molecular energy terms (p304).
- Apply the explicitly approved four TPSSloc physical inward kernel
  references after independent high-precision investigation. Preserve raw
  references and failures; unchanged p301 six tests and two public routes
  pass without modifying production equations or acceptance thresholds.


### Added

- Add EDF2 with published independent weights and0.1695 exact exchange.
  After explicit resumption approval, the strict shared build,48 local fits,
  RKS/UKS terms,90 C1 gradient/Hessian fits and public energy/gradient routes
  pass. Component fifth74fits/442zeros pass; six original zero-gradient
  kernel comparisons remain failed and unchanged. Broader combinations
  and five standard molecules remain separate.
- Correct the VWN5 spin-stiffness coefficient to its exact defining
  -1/(6*pi*pi), fixing a digit typo without changing references or gates.
  New default-binary EDF2 terms/local derivatives and repaired-VWN5 fifth
  tests pass; broader dependent-recipe combinations remain separate.

- Add TASK/mTASK exchange and explicit original-PW92 combined recipes
  through the generic meta-GGA driver. Complete local partials through
  third order, all eleven parameter partials per model, independent
  RKS/UKS energy terms, all 24 C1 RKS gradients and native/ASE/CLI routes
  pass. Smooth-local fifth tensors pass with failed exploratory intervals
  retained; five-molecule and general excited-state acceptance remain separate.

- Add localized TPSS correlation and its explicit TPSS-exchange recipe.
  The warning-strict build, 53 local partial fits, 12 C1 gradient fits,
  RKS/UKS terms and native/ASE/CLI checks pass. Four raw zero-gradient
  reference-kernel comparisons remain failed; pointwise acceptance is
  partial, with the original references and thresholds preserved.

- Add original PKZB exchange/correlation through the shared meta-GGA
  driver, with complete local first/second component oracles and fits,
  RKS/UKS energy terms, 12 C1 RKS gradient fits and native/ASE/CLI checks.
  The later local-Taylor repair artifact also passes local/energy
  regressions. Five-molecule and general higher-response gates remain
  separate; no universal excited-state acceptance is inferred.

- Reuse exact ECP sphere moments and primitive monomial contractions,
  preserving all 5323 old/new high-angular Cartesian entries bitwise,
  including fourth-derivative intermediates. Bound the lazy moment cache
  to 1736448 bytes at the existing limit; isolated i/i kernel timing changes
  from 3.451603 to 0.038584 s without changing quadrature or screening.
  All 228 AO/order fits, 624 nuclear regression entries, 776 independent
  matrices, twelve HF term cases and full warning-strict Clippy pass.
  The combined optimized build also passes all thirteen high-angular
  native/ASE/canonical-CLI cases.

- Extend C1 complete nuclear acceptance to g/h/i spherical shells: all
  1872 all-electron/scalar-ECP RHF/UHF gradient/Hessian entries pass.
  Eight complete cubic models pass 2912 independent directions: AE RHF/UHF
  g/h/i and scalar-ECP RHF g/h. AE RHF g/h/i quartics each pass 1365
  directions (4095 total); h/i retain all 26/19 failed intervals and five/three
  resolved fourth-order stencils. Independent scalar-ECP quadrature covers 776 entries, and twelve
  independent HF term/native Python/ASE/CLI energy cases pass. The angular
  limit remains i. Bounded local Taylor storage matches 13825 old coefficients
  and passes 2144 one-electron order-one-to-four fits. The isolated i/i probe
  peaks at 5524 KiB. Preserve the original i-UHF displaced-SCF failure;
  generic physical-Fock/CDIIS stagnation recovery now converges the strict
  regression in twelve iterations with residual 5.329e-15. The repaired
  failed-direction four-step cubic fit and warning-strict build/Clippy pass.
  All thirteen native Python/ASE atomic+physical/canonical-CLI energy and
  strict-recovery cases pass on the combined Python build. The repaired
  i-UHF cubic passes all 364 directions with five failed intervals retained;
  g AE UHF quartic also passes all 1365 directions (1360 second-order and
  five fourth-order fits, minimum R2=0.99926600), retaining all 34 failed
  intervals. Remaining ECP/h/i-UHF tensors stay under acceptance.

- Add BLYP35, WC04 and WP04 named global-hybrid compositions through the
  existing generic providers. WC04/WP04 explicitly retain the named
  VWN-RPA convention and independent fitted coefficients. Independent
  point/component/kernel and RKS/UKS energy oracles, complete C1 gradients
  and native Python/ASE/canonical CLI checks pass. BLYP35 also passes the
  complete five-molecule catalogue, as does WP04; WC04 ferrocene
  stationarity remains separately tracked. Preserve the original
  registry-Taylor failed intervals and thresholds while adding a resolved
  coarser fourth-order interval for the existing revM11 cancellation.

- Add a spin-complete complex PT2 source/amplitude provider and full-spin
  one-body orbital paths sharing the existing nuclear response engine.
  Generalize common-orbital response to complex spin-diagonal operators.
  Independent acceptance passes 41 pair/physical transverse fits and 18
  common-orbital static fits plus 108 full physical two-site FC source fits;
  mixed time-even/time-odd vertices also match
  complete one-electron spectral sums at three frequencies. Full transverse
  ROKS and open-shell correlated Ramsey assembly remain required.

- Rebuild the final periodic SCF field and energy at the returned density,
  validate its physical commutator/occupation/free-energy residuals, and
  use accelerated iterations after failed final checks. The old stored-pair
  regression fails independently. All 32 restricted/unrestricted
  reference/mesh/temperature cases, eleven existing Rust regression tests
  and twenty-six native Python/ASE/CLI response cases pass. Every CLI AO
  index is checked exactly once; missing/duplicate/non-finite output fails. The repaired core
  also passes all 360 complete Gamma derivative fits and eighteen
  finite-q complex dense cases without changing thresholds.

- Covariant complex finite-q full/range exchange on the shared periodic
  density solver, with reference-consistent pair weights and no same-q
  Hermitian projection. All 12288 kernel entry fits pass; coupled HF/hybrid
  physical acceptance passes 360 independent Gamma-field component fits
  and eighteen finite-q mesh/frequency dense cases. Full complex AO
  native Python/ASE/canonical CLI acceptance passes all twenty-six cases.

- Connect analytic local/nonlocal ECP nuclear Taylor coefficients through
  order four to the shared molecular response path, including moving basis
  and operator centres and effective nuclear charges. Complete C1 s/p/d/f
  kernel acceptance passes 228 fits and all six gradient/Hessian references
  pass 936 fits. All six complete cubic cases pass 2184 independent directions;
  AE RHF, AE RKS/PBE and ECP RHF each pass all 1365 quartic directions. Other quartic
  combinations remain in acceptance. Scalar-ECP ROHF/GHF providers pass all
  312 gradient/Hessian fits and eleven native Python/ASE/CLI route checks.
  Tighten coefficient solves against independently reconstructed absolute
  stationarity residuals without modifying physical operators or gates.
- Record 193 complex one-body/constraint fits, 81 full C1 GHF electronic
  Ramsey fits, the complete restricted limit and a fresh 156-component C1
  GHF nuclear regression. Fixed-spin-direction ensembles retain magnitude
  relaxation; the free ensemble is the default. ROKS and open-shell PT2 J
  remain required.

- Full complex one-body susceptibility and equality-constrained response
  with unchanged physical Hessians, shared UKS/GKS electronic Ramsey sources,
  complete FC/SD-PSO mixed terms, raw constraint/reference residuals and
  existing-driver API connections. New molecular/constraint acceptance is
  tracked separately; ROKS and open-shell PT2 remain required.
- A sufficient C1 geometry certificate for derivative fixtures and explicit
  documentation distinguishing complete independent component fits from
  directional, planar-molecule and public-route evidence.

- Thirty version-pinned Basis Set Exchange orbital bases, bringing the
  embedded catalogue to 78. All supported elements and every nonzero
  primitive coefficient match independent JSON exports. Retain upstream
  notices, exact source URLs and payload/catalogue hashes.
- A development-philosophy-only bibliographic comparison; no code or
  implementation from the cited work is used as a reference.
- Exact isotropic nuclear-spin Hamiltonian spectra on the existing NMR
  route, with arbitrary spin quantum numbers, full scalar flip-flop terms,
  coherent isotope-group detection, bounded untruncated Hilbert spaces and
  raw eigen/sum-rule diagnostics. Electronic response restrictions remain
  those of the shielding/J providers; public acceptance is tracked separately.


## [0.2.2]

Source checkpoint of the current implementation. Complete release-wide Rust/
Python regression and the full periodic elastic acceptance remain uncertified;
bounded measurements and remaining combinations are recorded in response.md.

### Added

- Add EDF2 with published independent weights and0.1695 exact exchange.
  After explicit resumption approval, the strict shared build,48 local fits,
  RKS/UKS terms,90 C1 gradient/Hessian fits and public energy/gradient routes
  pass. Component fifth74fits/442zeros pass; six original zero-gradient
  kernel comparisons remain failed and unchanged. Broader combinations
  and five standard molecules remain separate.
- Correct the VWN5 spin-stiffness coefficient to its exact defining
  -1/(6*pi*pi), fixing a digit typo without changing references or gates.
  New default-binary EDF2 terms/local derivatives and repaired-VWN5 fifth
  tests pass; broader dependent-recipe combinations remain separate.

- Reuse bounded immutable-reference meta-GGA XC derivatives in the common
  molecular response solver, with identical on-demand fallback. All 2560
  fixed-density kernel entry fits pass across four existing models; all
  outputs agree bitwise. The cache-enabled six-case coupled regression, all
  266 molecular component fits and RKS/UKS all-public-route alpha cache-budget
  comparisons pass. Broader combinations and periodic acceptance remain separate.

- Stabilize the physical inward derivative at the TPSS-family iso-orbital
  endpoint while retaining its capped value and the genuinely super-bound
  constant extension. All 266 point/complete RKS/UKS nuclear/electric fits
  pass, with four retained failed intervals. The original TPSS coupled
  directional Hessian now passes p=1.997884081962 and R2=0.999999698518;
  broader coupled tensors remain separate.
- Expand elastic acceptance from selected column maxima to all 81 strain
  and 54 internal-strain entries per parent, with independent staged fits
  and unchanged strict exponent, R2 and Richardson gates. The retained
  nine-entry shear diagnostic is resolved; the complete fresh suite remains
  a separate incomplete gate.

- Share the complete molecular density-energy/Fock evaluation between SCF,
  variational line searches and final reports. Reuse the complex occupied-
  subspace optimizer for stalled real RKS/UKS integer projectors without
  changing physical convergence gates. Retained TPSS quantum-potential
  acceptance and remaining combinations are recorded separately.
- Resolve the p5 rotational-regression quadrature failures with an explicitly
  converged unpruned Lebedev grid, retaining the original 1e-8 gates.
  All eight gradient regressions pass; the default grid is not assigned
  the refined fixture's precision.


- Connect structured quantum particles and summed correlation channels to the
  existing multicomponent driver, including dilute-positron local correlation
  and spin-averaged two-photon annihilation reports. Preserve the laboratory
  model assumptions and remaining method restrictions.
- Retain representable positron kernel actions and enhanced contact integrals
  in dilute density tails without a density floor; add independent point/term,
  four-step kernel and closed Gaussian oracles. All four p281 tests and both
  none/QMC all-public-route cases pass; molecular positron tensor acceptance
  remains separate.
- Fix native structured-particle defaults and recursively convert structured
  ASE controls from the common dotted unit registry.
- Record all 5187 complete ROHF and all 5187 complex GHF cubic/quartic
  channel fits, including independent displaced cubic-response checks for
  the two cancellation-limited quartic entries. Transported orbital L-BFGS resolves all
  32 retained displaced cases at the unchanged 500-iteration/1e-14 gates;
  the named complete GKS high-order fixture passes; other families remain open.
- Retain the reproduced unrestricted TPSS multicomponent SCF failure as a
  regression, now resolved by shared variational globalization and the
  iso-orbital derivative fix. Complete coupled Cartesian response remains
  unvalidated; its directional subset passes the original gates.


- Complex orbital-manifold globalization retains stable Cayley density
  increments, tangent transport and generic variational energy/Fock providers.
  ROKS high-order projector recurrences retain summation remainders in the
  physical path; the complete C1 ROHF cubic/quartic fixture passes.


- Common-orbital ROKS/ROHF and full complex GKS/GHF complete cubic/quartic
  tensor connections on the existing common force-constants API, and
  moving molecular parent/distinct-target VV10 plus PT2 nuclear paths.
  Independent all-component/public acceptance remains in progress with
  retained failures and raw diagnostics in response.md.
- SCF numerical robustness: stable common-orbital quadratic-density work,
  true restricted empty-open-class limit, and persistent energy-descent
  globalization of complex CDIIS after stagnation. Physical convergence
  gates are unchanged; retained displaced failures are regression tests.
  Twelve retained cases and nine public/metadata/lint cases pass; complete
  high-order tensor acceptance is recorded separately in response.md.


- Named double hybrids on the existing `xc` selection of every energy
  interface: B2GP-PLYP, B2K-PLYP, B2T-PLYP, mPW2-PLYP, mPW2K-PLYP,
  PBE0-DH, PBE-QIDH, PBE0-2, TPSS-QIDH, SCAN0-DH, SCAN-QIDH, r2SCAN0-DH,
  r2SCAN-QIDH, revDSD-PBEP86-D3(BJ) and the xDH recipe XYG3 (B3LYP
  orbitals, distinct target). Term-by-term oracles
  (`tools/oracle/gen_double_hybrids.py`): parent, target, one-electron,
  Hartree, exact exchange, every semilocal part, unscaled same-spin and
  opposite-spin PT2, D3(BJ) and total for H2, LiH, water, phosphine,
  ferrocene and the LiH cation doublet (`tests/p263_named_double_hybrids.rs`).
- Global and range-separated hybrids SOGGA11-X, M05, M05-2X, M06-HF, APF
  (0.411 B3PW91 + 0.589 PBE0, without the APFD dispersion model),
  RCAM-B3LYP and LC-BOP on the existing SOGGA11, M06, CAM and LC forms, and
  the meta-GGA VSXC as the VS98 part of the M06 form; the M05-type and VSXC
  coefficient tables were checked against libxc by linear least squares
  (residual below 1.1e-11).
- The MN12/MN15 meta-nonseparable family (`xc::mn12`): MN12-L, MN12-SX
  (screened exact exchange 0.25, omega = 0.11), MN15-L and MN15 (cx = 0.44),
  with the MN12 exchange form and the M08 correlation form; the index
  order of the 40 exchange coefficients and the total-density kinetic
  variable of the correlation were settled against libxc to 5e-14
  (`tools/oracle/probe_m08_form.py`).
- Functional parameter overrides on the molecular driver: the control
  `xc_parameters` (an object of parameter names and values; CLI
  `--xc-parameters '{...}'`, native Python and ASE keyword
  `xc_parameters={...}`) sets every part of the selected functional that has
  the named parameter (`ResolvedFunctional::with_parameters`); a name no
  part has is an error, and range separation keeps the `omega` control.
- rSCAN takes its switching interval as the parameters `switch_lower` and
  `switch_upper` (published `[0, 2.5]`, unchanged energies); `[1, 1]` gives
  the infinitely differentiable SCAN interpolation with the same branch
  constants. `tests/p280_rscan_validation.rs` validates rSCAN in two parts:
  the first three Taylor coefficients of the energy density at points on
  both sides of the polynomial junction, inside the polynomial range and on
  both SCAN branches, along 7, 28 and 84 generic directions per point (all
  833 fits pass, with the fourth-order stencil where the h^2 and h^4 terms
  cancel). The earlier all-component smooth-interpolation claim was incorrect:
  38 components failed. Test-only branch continuation now independently
  passes all 126 molecular gradient, Hessian, polarizability and first
  hyperpolarizability fits, p=1.918816..2.051137 and minimum R2=0.99964605.
  These fits retain each reference quadrature point's interpolation branch;
  they do not certify unfrozen finite-step branch crossings. The
  junction joins value and first three derivatives only, so finite
  differences of molecular rSCAN energies do not follow a power law beyond
  that order (fourth-order stencil of water alpha_zy: p = 2.95, against
  3.74 with the smooth interpolation; the `tau - tau_W` floor has no
  effect); rSCAN therefore left the p225 response set.
- Functional catalogue: `xc::catalogue::catalogue()` resolves every
  canonical exchange, correlation, combined and double-hybrid name into a
  record with its family, exact-exchange split, range separation, required
  nonlocal term and PT2 coefficients; `xc::double_hybrid::DOUBLE_HYBRID_NAMES`
  lists the named double hybrids and a registry test keeps it equal to the
  recognized names. `dft-rs --list-functionals` lists the double hybrids
  and native Python gains `functionals()`.

- Complete correlated cubic/quartic Cartesian, coordinate-block and pattern
  force tensors on the existing force-constants common API, with full
  orbital/PT2 amplitude response, separate physical channels and explicit
  permutation-reuse/acoustic diagnostics. Ordinary tensor assembly shares
  the same multi-channel polarization core. Independent acceptance and
  remaining reference/functional combinations are recorded in response.md.

- Shared CODATA 2018 unit factors for ASE, native and XYZ input, with frozen
  boundary checks. High-order PT2 paths expose fresh relative amplitude
  residuals alongside the unchanged absolute Hartree residuals; the common
  high-order convergence gate uses the requested relative tolerance.

- Correlated static beta/gamma full tensors and cubic/quartic contracted nuclear
  matrices on the existing common APIs, reusing full orbital and noncanonical
  PT2 response. Finite electric bias, separate SS/OS/target/dispersion channels,
  explicit permutation-reuse metadata and bounded independent acceptance are
  recorded in response.md. Complete Cartesian nuclear exports have their
  own bounded acceptance; remaining physical combinations stay open.

- Exact sparse coefficient support for nonlocal response: constant and
  t^s-only paths evaluate without zero-padded higher-order arithmetic.
  No numerical cutoff or molecular symmetry assumption is introduced;
  the independent acceptance still evaluates every tensor component.

- Separate weighted parent/target nonlocal NMR and J tensors on the existing
  APIs. PT2 J amplitude-equation diagnostics carry Hartree units separately
  from dimensionless frame/projector defects; all remain convergence gates.

- Explicit independent parent/target auxiliary field models and shared VV10
  endpoint rows for real SCF, owned/runtime Taylor and complex magnetic
  paths. The full-complex kernel passes 38 fits and molecular magnetic
  energy Hessians pass 408 full-component fits. Mixed nuclear-moment and
  public nonlocal acceptance are tracked separately. Dedicated-grid weight policy
  is shared, including zero pruning in derivative tests.

- Shared global learned complex-field energy/Fock pullback and finite magnetic
  SCF connection, with 38 independent full-component coefficient fits
  and 810 coupled NMR fits across both gauges and RKS/UKS. Full London
  PT2 NMR adds 2025 component/channel fits across five parent families.
- Full target/PT2 closed-shell FC/SD/PSO/DSO pair-response connection on the
  existing coupling and NMR-multiplet APIs; 925 independent component/channel fits pass; the initial 17-case public
  suite passes, with the final broader rerun recorded in response.md.

- Generic second-order PT2 TDA pole and electric/magnetic transition-residue
  connection on the existing ECD spectrum API, with explicit model, memory
  checks and raw pole/normalization diagnostics; 452 independent kernel/property-gradient and 80 dense pole/moment fits pass;
  20 final public/metadata tests and a broader 28-case regression pass,
  including a common parent-anchored real transition-state phase convention
  and exact Pauli exclusion in the one-electron propagator limit.

- Normalized first-order pair-wavefunction VCD AAT connection with full
  parent orbital, PT2 amplitude and nuclear half-overlap response; the
  exact one-body commutator reuses the PT2 tensor action and avoids explicit
  triple-substitution storage, with a checked workspace budget. The
  existing VCD driver also uses the full correlated Hessian and APT.
  HF/LDA/GGA/meta-GGA/hybrid pair AATs pass 2,025 component/channel fits
  and full Hessian/APT passes 2,250; 15 public and 49 broader regression
  cases pass. The one small-h^2-coefficient cancellation is resolved by an
  independently fitted fourth-order stencil, retaining every failed interval. No unmeasured combination
  is marked validated.

- Complete complex PT2 field paths and explicit mixed one-electron contact
  coefficients, connected to the existing orbital NMR APIs with all spin
  channels and diagnostics. HF/PBE0 common-origin NMR passes 810 component/channel
  fits and eight public cases; TPSS magnetic Hessians pass 90 further fits.
  Current-invariant TPSS mixed NMR adds 405 passing component/channel fits;
  correlated transition/VCD and remaining providers are not marked complete.

- Shared real/complex noncanonical PT2 amplitude recurrence, with conjugated
  excitation sources and contragredient occupied indices; complex London
  AO-to-MO transformations reuse the packed real SIMD implementation.
  Independent all-component dense/finite-path acceptance is recorded in
  response.md and rust-api.md. Spectral PT2 integration remains in progress.

- Shared weighted density-functional partials for learned London-phase
  and gauge-covariant current response, reused by orbital NMR and VCD
  for RKS/UKS. Independent complete-matrix derivative acceptance and
  functional/reference spectroscopy tests record their actual scope;
  correlated spectral response remains open.

- Common molecular dynamic-polarizability driver with RKS/RHF, UKS/UHF,
  constrained common-orbital ROKS/ROHF and full complex spin-covariant
  GKS/GHF reference selection. Static, retarded and imaginary-axis models
  use generic functional providers and return every Cartesian component.
- Shared time-even/time-odd common-orbital response with its physical
  occupation-difference Berry metric, and fixed-geometry reuse of the
  spinor descriptor Hessian with reference-Fock and angular diagnostics.
- All-component four-step acceptance for constrained, complex spinor and
  restricted/unrestricted correlated susceptibility, including separate
  same-spin/opposite-spin terms. Measurements and remaining combinations
  are tracked in response.md; this is not completion of R001.

### Fixed

- Match the original PW92 reference convention's rounded spin-stiffness
  normalization without changing the precise PW92-MOD local base used by
  PBE/TPSS families. Independent energy/potential/kernel checks and all
  first/second/third density partial fits pass; all 4554 prior
  TPSS/revTPSS/PBEloc values remain bitwise identical.

- Correlated property energy-component maps now use the same target, PT2
  and dispersion decomposition as the energy API, instead of retaining
  parent SCF components. RKS/UKS and default-dispersion regressions pass.
- Complete force-tensor native/ASE wrappers preserve their common legacy
  unit summary alongside authoritative unit labels. Failed initial tests
  and the independent corrected reruns remain recorded in response.md.
- The periodic nuclear attraction is truncated per shell pair, by the
  distance of the pair as for the overlap and kinetic images, in the
  energy, the gradient, the Hessian, the strain derivatives and the strain
  Fock blocks. It was truncated per lattice translation (`|T| <= cutoff`),
  so supercells lost the translation symmetry of their primitive cell: in a
  two-cell hydrogen chain at a 14 Bohr cutoff, moving an atom or its image
  in the other cell changed the energy by 4.5e-7 Hartree (now 0), the
  supercell force constants of equivalent atoms differed by 5.9e-5, the
  dynamical matrix had a Hermiticity defect of 3.2e-8 and the Gamma
  acoustic modes came out at -0.2 cm^-1; the 14 Bohr energy moved from
  4.4e-5 to 5.6e-7 Hartree of the 20 Bohr one.
- The periodic exact exchange selects its shell quartets by the
  separations of the shell centres (both charge distributions and both
  density pairs within the exchange cutoff) instead of by the cell
  translations (`|T|, |C|, |D| <= cutoff`), in the exchange matrix, the
  low-dimensional variational map, the gradient and virial, the force
  constants, the first-order exchange matrices and the strain second
  derivatives, which now share one quartet enumeration
  (`periodic::exchange_quartets`). A doubled hydrogen chain at a 14 Bohr
  cutoff differed from its primitive cell by 5.4e-7 Hartree per cell with
  PBE0; it now differs by 2.1e-12 (PBE0) and 3.4e-12 (Hartree-Fock), and
  moving an atom or its image in the other primitive cell gives energies
  equal to 8e-15. The selection is symmetric under the exchange of the two
  density factors, so the truncated exchange energy is variational: the
  exchange-matrix derivative used in place of the pairing matrix now
  reproduces the density derivative of the exchange gradient to 3.7e-15.

### Changed

- Recorded 44,152 independent complete correlated cubic/quartic
  Cartesian/pattern component/channel fits for the bounded finite-bias
  HF-parent/PBE-target RKS/UKS model, and 1,750 complete static beta/gamma
  fits across five representative parents and both references. Original
  failed intervals and separately resolved fits are retained. The
  current complete-force public/ordinary/unit/packaging suite passes
  all 39 cases. Periodic, nonlocal and constrained/spinor fourth-order
  provider gaps remain explicit in response.md.

- p3 runs every SCAN-based functional on the oracle's own level-9 grid:
  SCAN0 on level 8 missed its level-9 oracle by -2.27e-6 (water) and
  -4.51e-6 (CH3), a grid difference rather than a functional defect.

- The finite magnetic field SCF (`bfield`, `bfield-gradient` and the paths
  built on it) stores the sum of the London field coefficients of the
  repulsion integrals once per SCF when the tensor fits the `incore_mb`
  budget (`integrals::london::summed_repulsion_series`,
  `contract_summed_repulsion`) instead of regenerating the integral series
  in every iteration; larger bases keep the quartet-direct path. For water
  in STO-3G at B = 0.05 a.u. (series order 18) the SCF fell from 65 s per
  iteration to 76 s in total; the stored contraction matches the direct
  one to 1.4e-17. Together with the shifted-Gaussian series of the pair
  phases the SCF of the distorted p108 water (series order 18) now takes
  7.0 s in total; before, one series generation did not finish in 1800 s.
  The term-by-term fallback of `primitive_repulsion_coefficients` (series
  orders above 32 and scalar-generic jets) builds the Hermite expansion
  coefficients and the Coulomb tensor once per primitive quartet
  (`integrals::analytic::HermiteQuartet`), matching the term-by-term
  evaluation to 3e-16 relative.
- p225 fits beta from central differences of the analytic polarizability
  at the field-polarized reference (all 27 components) instead of second
  differences of the dipole, whose noise grows as 1/h^2 and failed 24
  components; the fits add fourth-order stencil stages for the large
  higher field derivatives of SCAN.

- Spin-sphere reference and response quadratures can refine consistently
  through order 128; actual operator defects still enforce the requested
  tolerance, including failures at the former order-64 ceiling.

- Native Python and ASE dynamic-polarizability calls use the common driver;
  historical Python frequency and unit keys and CLI frequency aliases remain
  available. Static ROKS polarizability uses the same constrained kernel.
- Restored legacy field-Taylor/mixed-response text rows in the shared report
  renderer and retained paired complex-alpha rows. Correct electric tensor
  unit labels preserve the ASE conversion e^2 Bohr^2/Hartree while retaining
  the historical native bohr^3 summary and frequency metadata.
- Removed the stale ROKS learned-functional refusal; the existing learned
  energy, Fock and density Hessian are now connected to the common orbitals.

## [0.2.1]

This is a partial milestone of the accepted full response program. The
functional/reference matrices in docs/response.md and property_targets.csv
remain authoritative for unfinished and unvalidated combinations. The full
legacy Rust/Python suites have not been rerun for this milestone; acceptance
below is scoped to the stated features and regressions.

Measured acceptance for this milestone:

| Scope | Independent entries / cases | Result |
|---|---:|---|
| ROHF and complex GHF nuclear tensors | 156 each | All entries pass retained multi-step fits |
| SVWN5/PBE/TPSS/PBE0 ROKS nuclear tensors | 624 | p=1.811786..2.193221; minimum R2=0.99913778 |
| Restricted correlated nuclear channels on HF/PBE/TPSS/PBE0 parents | 4368 | p=1.802733..2.195880; minimum R2=0.99902605 |
| Distinct PBE-on-HF total/target nuclear tensors | 312 | p=1.817002..2.181338; minimum R2=0.99919170 |
| Complex PBE/PBE0 nuclear tensors | 312 | PBE/PBE0 pass; the earlier TPSS order-64 failures are retained, and consistent order-128 refinement now passes 156 complete TPSS nuclear entries |
| Spinor descriptor kernels (four functional classes) | 1552 | p=1.937461..2.085495; minimum R2=0.99989851 |
| Common-driver regression | 5 | Strict-unit exports and ordinary calculations pass |
| Versioned related Rust regression | 15 | Common driver, Bloch exports, collinear/GKS references and all spinor descriptor kernels pass |
| Final versioned public/ASE/package regression | 36 | All pass in 322.20 s, warnings treated as errors; both CLIs and the Python core report 0.2.1 |

The final gate includes native/ASE/CLI custom-target tensors, ordinary
ROKS/GKS lazy Hessians, named double-hybrid tensors, periodic Bloch exports,
ASE cache compatibility and stabilized-reference refusal, dependency-graph
normalization, Ruff and API/ledger consistency. Default-feature checking
and all-feature/lib/bin/test Clippy pass without compiler warnings. Four
native artifacts pass the private-build-root audit. The source release
retains failed angular acceptance and unfinished combinations explicitly;
these checks are not a claim that the full response program is complete.

- Expose optional `double_hybrid_target_xc` on the existing custom-model
  energy/property APIs. Its nonstationary target retains full parent orbital
  response; named recipes refuse an override. Additional target nonlocal
  terms retain their documented provider limitation.
- Preserve explicit references and custom correlated targets in ASE's lazy
  `get_hessian`, sharing its routing decision with ordinary energy/forces.
- Separate conservative XC cache bounds from actual operator residuals.
  Every actual energy/potential path still passes the original fresh angular
  convergence gate; descriptor-bound convergence alone is not certification.

- Accept and validate shared response controls on ROKS energy/gradient
  routes; propagate Hessian and auxiliary electric-response convergence to
  the top-level report instead of reporting SCF convergence alone.
- Verify the correlated zero-coefficient limit against all parent nuclear
  tensor entries, and add complete distinct-target/ROKS-class acceptance.

- Connect generic molecular GGA/meta-GGA/hybrid complex nuclear paths with
  moving Becke grids, kinetic/current descriptors, mixed Taylor potentials
  and raw angular refinement. Reuse stationary 2n+1 energy evaluation.
- Connect complete molecular double-hybrid nuclear gradients/Hessians to
  the common API, including all SS/OS channels and full parent orbital/PT2
  amplitude response. Preserve remaining nonlocal and reference combinations.
- Share Cartesian tensor assembly across spinor/correlated providers; fit
  every independent component and all correlated channels in acceptance.
- Fix missing thermal-orbital-save diagnostic unit labels found in strict
  common-driver regression.

- Add a reusable molecular complex spinor energy/Fock provider with the
  shared spin-covariant semilocal and complete exchange providers. Connect
  GKS/GHF to the common energy, populations and orbital-export APIs, with
  all-component variational fits and independent public-path checks.
- Add canonical ROKS/ROHF energy, gradient, Hessian and polarizability
  reference routing through the existing analytic implementations; preserve
  unresolved generalized nuclear/response combinations in the scope ledger.
- Fix missing unit labels in molecular orbital coefficient exports and
  prevent a learned-functional ROKS request from silently omitting its XC.


- Add a development-only Ruff correctness gate for Python sources, tests
  and tools, with a portable runner and acceptance tests. Remove unused
  imports and use the explicit solvent wrapper in the PCM oracle generator.
- Connect existing molecular `charges` and `orbitals` to explicit RKS,
  UKS and ROKS constraints through a shared typed reference provider,
  retaining common functional and additive energy providers. ROKS orbital
  export uses the existing spin semicanonical occupied/virtual blocks.
- Add optional complex Bloch coefficient export to `electronic-structure`
  through the common Rust, CLI, native Python and ASE driver. Reports include
  the overlap, Fock and per-spin density matrices and raw metric, eigenpair,
  energy and density reconstruction residuals.

## [0.2.0]

This release packages the features implemented through 0.2.0-alpha.2 and the
corrections below. It is a partial milestone of the full accepted response
program, not its completion; the ledger in [response.md](docs/response.md)
and the API tables list what is connected and how far each item is
validated. No global completion percentage is inferred.

Validation state of this release:

- The standard Lebedev grid is the default, and the grid equivalence (p238),
  functional molecules (p237), DFT oracle (p3), functional coverage (p14) and
  XC boundary (p22) suites pass on it. The pointwise functional oracle of
  p225 passes in every component.
- Derivative tests converted to every-component fits that pass on this
  source: p118 (27 dipole derivatives, 9 polarizability components, 9
  solvated gradient components), p124 (27 Raman components) and p125 (9
  gradient components).
- Converted to every-component fits, with the defects found by the
  conversion fixed, but not yet rerun on this source after the fix: p115
  (restricted open-shell Hessian), p116 (spin-flip derivatives) and the p225
  response fits (gradient, Hessian, polarizability, strain and quantum
  nuclear derivatives for the new functionals).
- Converted to every-component fits but not yet run: p105, p106, p107, p108,
  p110 and p119.
- All other derivative tests still fit selected components or directions
  and are labelled single-component validation until they are converted.
- The full Rust and Python suites have not been rerun on the new default
  grid; the results above are per suite.

- Fix the unit labels of the `mrsf` report: the state lists carried one
  combined label that the strict unit check (`DFT_RS_STRICT_UNITS=1`)
  rejects, so `mrsf` failed in strict mode; the fields now have their own
  labels (`singlets.omega`, `singlets.energy`, ...).
- Speed up the periodic Gamma-point XC Hessian (phonons, cubic force
  constants, Grueneisen and elastic paths): the periodic Becke weight
  derivatives skip the derivatives of cells below 1e-30 of the largest one
  (they stay in the normalization and cannot change a digit of the result),
  and the pair accumulation no longer builds the sparse 3x9x9 distance
  Hessians. For a two-atom chain the weight derivatives fell from 840 s to
  31 s of thread time.

- Replace the PowerShell basis download script by `tools/fetch_basis.py`
  (standard library only, same sets and PROVENANCE header), and add
  `docs/conventions.md` with the implementation conventions of the package.

- Add the control `grid_weight_cutoff` (Bohr^3, default 1e-14) on the
  molecular and periodic drivers, native Python (`calculate`,
  `periodic_scf`) and ASE: quadrature points whose partitioned weight is
  below it are dropped, and `0` keeps every point so the energy is a smooth
  function of the nuclear positions. With the default cutoff, points
  entering and leaving the grid as atoms move put noise near 1e-13 into
  displaced gradients; derivative acceptance tests now use `0`.
- Derivative acceptance fits every component (gradient, Hessian,
  polarizability and hyperpolarizability, dipole and Raman derivatives,
  magnetizability, susceptibilities, strain derivatives) instead of one
  component or one direction; shared helper `tests/support/components.rs`.

- Fix P86 correlation (BP86, B3P86, DSD-PBEP86) wherever only one spin
  channel carries density: the single-spin branch used the spin factor
  d(1) = 2^(-1/2) instead of the limit 2^(1/3) of the two-spin expression,
  so its gradient term was 2^(5/6) times too large. A new p22 test requires
  the single-spin branch of every functional to be the limit of the
  two-spin expression; the old continuity test approached only from below
  the density cutoff and could not see it.
- Merge `docs/functional_catalogue.csv` into the "Functional catalogue"
  section of `docs/FUNCTIONALS.md` and `docs/response_coverage.csv` into the
  "Response coverage ledger" section of `docs/response.md` (readable entries
  with a functional-class by reference-type table); `tools/ledger_check.py`
  reads the new section.

- Fix the TPSS family (TPSS, revTPSS, modTPSS) and B95 correlation (B95 and
  its hybrids BB95, B1B95, BB1K, mPW1B95, PW6B95, PWB6K, PWPB95) where
  `tau < tau_W`: the meta-GGA driver now applies the Fermi-hole-curvature
  bound to their inputs as it already did for the SCAN family. The TPSS
  correlation also caps the total-density indicator `tau_W / tau` at one,
  and rSCAN floors its indicator numerator `tau - tau_W` at zero, as the
  reference implementations do. Values at physical densities are
  unchanged; at the oracle point with `tau < tau_W` revTPSS/modTPSS
  exchange deviated by up to 100 % and B95 by 5e-4 before.
- rSCAN takes `tau_r` and `alpha_r` as parameters with declared
  uniform-scaling powers (5 and 0), so the scaling test holds without an
  exclusion.
- Add global hybrids B3LYP*, X3LYP, B3P86, mPW3PW91, B1LYP, B1PW91, BHandH,
  BHandHLYP, PBE0-1/3, mPW1PW91, mPW1LYP, mPW1PBE, mPW1K, B97-1, B97-2,
  B97-3, B97-K, TPSS0, revTPSSh, SCAN0, r2SCAN0, r2SCANh and r2SCAN50
  (`HybridParams::global`).

- Change the molecular and periodic XC quadrature to the standard
  atom-centred grid: Lebedev-Laikov angular rules (published orbit
  generators for degrees 3-131, each checked against closed-form monomial
  integrals) replace the Gauss-Legendre x uniform-phi product rule as the
  default, the five-region near/far-nucleus pruning replaces the former
  degree-11 core shell for every element, the Becke partition uses
  Treutler's size adjustment (square root of the Bragg radius ratio), the
  Treutler-Ahlrichs xi table extends beyond Ar, and `grid_level` 0-9 now
  sets radial points and angular degree by period (level 3: 50/75/80/90
  radial points, degree 29/29/35/35 for H-He/Li-Ne/Na-Ar/K-Kr). This
  changes every grid-based energy at fixed `grid_level`; the former rule
  stays available as `grid_angular_scheme=product`. New driver controls
  `grid_preset=fine|ultrafine|superfine` (per-atom (75, 302), (99, 590),
  (150/225, 974)) and `grid_angular_scheme=lebedev|product` on Rust
  (`GridOptions::from_controls`, `GridPreset`, `AngularScheme`), the CLI,
  native Python (`run_feature`, `calculate`, `periodic_scf`) and ASE
  calculator parameters. Acceptance: `tests/p238_standard_grid.rs`.

- Add natural bond orbital analysis on the shared Rust/CLI/native
  Python/ASE driver (`natural-bond`): natural atomic orbitals with the
  natural minimal basis and occupancy-weighted orthogonalization, natural
  population analysis and electron configurations, Wiberg indices in the
  NAO basis, the natural Lewis structure with orthogonal hybrids, antibonds
  and Rydberg orbitals, natural localized molecular orbitals and
  second-order donor-acceptor energies; open shells get one Lewis
  structure per spin and natural spin populations. The `--nbo` option of
  `charges` and the `nbo` block of the Python SCF result now use the same
  NAOs (previously a symmetric orthogonalization with atom-block
  diagonalization) and report spin-resolved Wiberg indices with a zero
  diagonal.

- Extend the natural bond orbital analysis: relaxed MP2/double-hybrid and
  CASSCF (also the MC-PDFT state) densities through `nbo_density`, natural
  resonance theory (`nbo_resonance`), periodic cells through the
  Born-von Karman unfolding (`periodic-natural-bond`), and the natural
  chemical shielding (`natural-shielding`), natural J-coupling
  (`natural-coupling`) and natural energy decomposition
  (`natural-energy-decomposition`) analyses. Two-centre bonds are now
  formed from the lone-pair-depleted density without depleting each other,
  so equivalent bonds stay equivalent.

- Add `sapt-analysis`: SAPT of two fragments of one system with the
  gradient-regulated asymptotic correction of Kohn-Sham monomers
  (`ScfOptions::asymptotic_correction`, `ks::grac_correction`),
  full-response (RPA) dispersion by sum over states and by the
  Casimir-Polder integral, charge transfer from the dimer-centred basis and
  an atom-pair partition of electrostatics and induction over natural
  localized molecular orbitals (`sapt::sapt_interaction_with`,
  `sapt::SaptConfig`). `sapt_interaction` is unchanged.
  `TdaSolver::orbital_hessians` and `TdaSolver::rpa_from_hessians` expose
  the dense `A + B`, `A - B` and the RPA states built from them.

- `piezoelectric-tensor` takes `relaxed_ion` (default true); `false`
  reports the clamped-ion tensor of references without a stable ionic
  relaxation. `wannier-flow` reports the hybrid Wannier centres with the
  sign of the Berry phase (they were the negated centres), and the second
  susceptibilities of the periodic field response carry the parseable unit
  label `e*bohr/hartree`.

- Add the functionals `pbesol`, `rpbe`, `pw91` (PW91 exchange and
  correlation), `scan` (original SCAN) and `lrc-wpbeh` through the generic
  drivers, with the split components `pbesol`, `rpbe`, `pw91` and `scan`;
  frozen libxc energies and pointwise first derivatives in
  `tests/oracle_data`. `interspecies_correlation` accepts `epc17-2`
  (c = 6.6) and `epc17-1` as an alias of `epc17`.

- `multicomponent` takes `species_stability` (`none`, `report`, `follow`)
  and `species_stability_roots`: the lowest eigenvalues of the coupled
  electronic/quantum-particle orbital Hessian by a Davidson iteration
  (`multicomponent::stability`), and with `follow` steps along a negative
  mode with re-convergence (`multicomponent::solve_from`) until the
  reference is a local minimum. Lattice paths reuse the integral series of
  paths that differ only in their field component (`piezo-higher`,
  `electrostriction`, `periodic-raman`).

- `periodic-dc` acceptance: the incomplete-buffer case of
  `p110_periodic_divide_conquer` used a buffer that covered the whole cell;
  it now uses 1.8 Bohr.

- Parseable unit labels for `berry-dipole` (`fermi_weight`,
  `1/hartree/bohr^3`), `orbital-susceptibility-q` (per-entry labels of the
  wave-vector points; `q` now converts to inverse Angstrom on the ASE
  route) and `periodic-nmr-shielding` (per-entry labels of the nuclei and
  their points); strict unit checking refused these reports.

- SAPT refuses a monomer whose internal orbital-Hessian curvature is not
  positive (a partially filled degenerate shell or an SCF saddle point)
  before evaluating any second-order term. The previous guard on the size
  of the first-order rotation depended on the arbitrary orientation of the
  degenerate solution: for triplet oxygen beside water the hole now lands
  in the p orbital that the water field cannot rotate, and the guard no
  longer fired.

- Add complete-active-space SCF (state-specific and state-averaged,
  string-driven determinant CI with spin adaptation of degenerate states)
  and multiconfiguration pair-density functional theory with translated and
  fully translated local and gradient-corrected on-top functionals, a hybrid
  fraction and the linearized, compressed and extended multistate forms, on
  the shared Rust/CLI/native Python/ASE driver (`casscf`, `mcpdft`). Measured
  internal and frozen energy/term oracles are recorded in the response
  manual. Add analytic state-specific gradients and Hessians through the
  shared response solver and orbital retraction; multistate derivatives,
  meta-GGA translation and periodic cells remain pending.

- Keep magnetic response integral screening fixed at the reference density.
  Solve nonzero magnetic sources with a fresh residual even when their
  magnitude is near a symmetry zero; do not replace them by zero based on
  the size of another field component.

- Keep explicit XANES core-hole indices within each spin channel's occupied
  space and label frequency-dependent saved-response entries for strict unit
  validation. Include the coverage-ledger checker in source distributions so
  its packaged Python acceptance test can run on another computer.

- Add an experimental simultaneous electronic/quantum-nuclear variational
  model with exact Gaussian interspecies Coulomb terms and independent local
  EPC17 correlation. Connect expectation-position constraints, matrix-free
  coupled response and analytic moving-basis gradients/Hessians to the shared
  Rust/CLI/native Python/ASE driver. Keep laboratory center-of-mass and
  intrinsic vibronic limitations explicit; acceptance is recorded in the
  existing response manual and coverage ledgers.

- Share the complete fixed-tensor Coulomb/exchange kernel between electronic
  Newton optimization and response. Evaluate finite orbital energy work in
  the orbital frame and use Armijo descent, including steps with a temporarily
  increasing gradient. Refine the retained overlap metric, compensate small
  residual contractions, and parallelize independent stored-tensor J/K rows
  with the existing rayon pool; retain physical operators and tolerances.
  Initialize quantum-nuclear calculations in their distributed charge
  potential, avoiding an unnecessary converged point-nucleus BO prerequisite
  and a duplicate electronic integral-cache build.

- Connect prescribed-population nonlinear molecular electric response, mixed derivatives and full static beta/gamma to the common driver; reuse the complete semilocal and learned Fock Taylor sources and retain physical spectrum/metric diagnostics.

- Connect prescribed populations to the shared matrix-free static electric
  response and moving-metric Hessian/APT, including nominal virtual/virtual
  population rotations. Preserve the canonical thermal occupation derivative,
  exact degenerate-subspace conditions and raw response/number diagnostics.
  Nonthermal harmonic surfaces do not acquire equilibrium Gibbs thermochemistry.
  Share the configured integral-cache policy between SCF, response kernels and
  Hessians instead of regenerating affordable tensors on every trial vector.

- Add prescribed spin-orbital populations to the common collinear SCF and
  analytic force driver, with physical population-map convergence, independent
  occupation-ensemble spin moments and explicit thermal/ensemble-XC distinctions.
  Validate five molecular PBE energy/component/dipole oracles and record the
  ground-state experimental comparison. Explicit fixed/MOM/IMOM controls also
  use the common driver for default CLI text output.

- Consolidate the complete calculation reference into the existing Rust,
  CLI and native Python/ASE manuals; preserve all 157 aligned entries and
  physical contracts, and check each table against the same registry.
- Add functional-independent real-collinear occupied-projector MOM/IMOM
  delta-SCF to the ordinary SCF path and reuse it for core-hole relaxation.
  Energy oracles and finite-electric-field gradient fits cover the accepted
  parent-functional subset; further methods and response combinations remain
  tracked rather than inferred.
- Share dimensional source Green kernels between real energies/forces and
  analytic Taylor derivatives through fourth order, including finite Gaussian
  sources and fixed countercharges; relaxed magnetic nuclear acceptance remains
  separate.
- Record the expanded DFT variant scope and completed long public finite-bias
  nuclear repeat. The frozen alpha.2 archive is unchanged.

## [0.2.0-alpha.2]

This alpha extends the frozen alpha.1 source archive. It is a partial milestone
of the full accepted response program. Physical scope and validation intervals
are recorded in [response.md](docs/response.md), with functional/reference
combinations in the two coverage CSV files. No global completion percentage
is inferred from API, test-file or matrix-cell counts.

### Added

- Add EDF2 with published independent weights and0.1695 exact exchange.
  After explicit resumption approval, the strict shared build,48 local fits,
  RKS/UKS terms,90 C1 gradient/Hessian fits and public energy/gradient routes
  pass. Component fifth74fits/442zeros pass; six original zero-gradient
  kernel comparisons remain failed and unchanged. Broader combinations
  and five standard molecules remain separate.
- Correct the VWN5 spin-stiffness coefficient to its exact defining
  -1/(6*pi*pi), fixing a digit typo without changing references or gates.
  New default-binary EDF2 terms/local derivatives and repaired-VWN5 fifth
  tests pass; broader dependent-recipe combinations remain separate.

- Five double-hybrid energy recipes: B2PLYP, DSD-PBEP86-D3(BJ),
  DSD-BLYP-D3(BJ), wB97M(2), and PWPB95-D3(BJ). Independent component
  oracles include separate same/opposite-spin PT2 contributions. Static
  molecular correlated dipole and polarizability include amplitude and orbital
  response; generic periodic orbital/pair Taylor connections remain subject to
  the acceptance restrictions in the ledger.
- Periodic static gamma, finite-bias alpha/beta/gamma and analytic nuclear
  paths through fourth order. Molecular London finite-B and joint E/B nuclear
  checks include neutral/charged HF H2 and hybrid PBE0 water. The 3D finite
  magnetic Gamma, point-source, integer full-spinor model supports E, B and
  joint E/B nuclear derivatives through fourth order, with neutral full
  gradient/Hessian and contracted cubic/quartic public-route comparisons.
  Independent joint-field fits cover all seven crystal metrics, custom
  full/range exchange with nuclear Pauli SOC, and a charged coherent spinor.
- Periodic two-endpoint nonlocal-correlation energy and moving-density kernels;
  high-order periodic dispersion; constrained long-wave phonon moments and the
  Born-charge lattice contribution to flexoelectricity; electrostriction paths.
- Molecular contact/dipolar hyperfine tensors, enhanced spin response,
  Cotton-Mouton/higher magnetic and mixed magnetic paths, optical-activity
  paths, and an energy-dependent scalar scattering-time transport model.
  Each has its own stated reference, dimensional and correlation limits.
- Finite-nucleus balanced one-electron Dirac components, metric no-pair
  projector response, static LM2005 phonon pairing and number/phase response,
  and full interband Nambu density response on supplied operator spaces.
  These components do not establish a self-consistent Dirac-Coulomb reference
  or the complete superconducting density-functional response program.
- Explicit `mermin-free-energy` and `gibbs-free-energy` APIs, with an optional
  RRHO Gibbs calculation on the same Mermin reference. The old `free-energy`
  entry remains a documented compatibility alias.
- A pip console entry named `dft-rs`, portable native-build remapping and
  dependency-metadata normalization, warning-as-error builds/tests, default
  SCF iteration diagnostics, and generated native Python/ASE API contracts.
- Functional/reference coverage matrices: 158 target rows and 68 response
  families, each with 28 class/reference columns. The three aligned API
  tables contain 157 matching IDs (151 in alpha.1). Forty-three new numbered
  Rust acceptance files are present; their presence alone is not test success.

### Fixed

- Periodic Berry electric enthalpy now uses the actual SCF overlap in
  `Q-S`, including every moving-metric Taylor coefficient. Independent
  pair-charge image cutoffs no longer change the occupied projector metric.
- The shared real/complex Berry functional continues the electronic phase
  near the electron-number-weighted nuclear charge centroid. Neutral rigid
  translations across determinant phase cuts preserve energy and dipole;
  charged magnetic translations obey the explicit electric-work convention.
- Finite-field initialization, London-series tails, metallic occupation/spin
  constraints and reported units/residuals have dedicated regression coverage.

### Validation for this milestone

- Warning-as-error checks pass for all Cargo targets, both default and Python
  features; selected Rust term-oracle/component tests pass 11 tests and
  the common Berry subset passes five tests.
- Python/ASE/CLI public regressions pass 35 tests for each executable name.
  The unchanged post-overlap-repair electric and nuclear independent
  multi-step fits pass, with raw measurements in response.md.
- The pip wheel installs outside the source tree; `dft-rs --version`, native
  and CLI energy oracles, and retained author metadata pass.
- The full regression catalogue was not rerun to completion. The long
  finite-bias nuclear public tensor repeat remains running separately.
  All unaccepted feature/reference combinations retain their ledger status.

### Remaining scope

The CSV `pending`, `not_audited`, `implemented_unvalidated` and `backend_only`
entries remain open. In particular, complete 1D/2D and magnetic-BZ E/B nuclear
response, arbitrary-q biased nuclear DFPT, all semilocal/current/nonlocal and
double-hybrid correlated response combinations, complete flexoelectricity,
self-consistent no-pair Dirac-Coulomb, superconducting normal-density
backreaction/dynamics, periodic correlated optical activity and the other
unaccepted registry combinations are not declared complete.

## [Earlier development notes]


Work toward 0.2.0 is delivered as `0.2.0-alpha.N` milestones. An alpha is a
verified subset of the response, periodic DFPT, superconductivity and
continuum-parameter objective; it is not completion of that objective. The
[physical target registry](docs/scope.md#13-physical-target-registry) and the
[response coverage ledger](docs/response.md#response-coverage-ledger) describe the scoped evidence.

### Added

- Add EDF2 with published independent weights and0.1695 exact exchange.
  After explicit resumption approval, the strict shared build,48 local fits,
  RKS/UKS terms,90 C1 gradient/Hessian fits and public energy/gradient routes
  pass. Component fifth74fits/442zeros pass; six original zero-gradient
  kernel comparisons remain failed and unchanged. Broader combinations
  and five standard molecules remain separate.
- Correct the VWN5 spin-stiffness coefficient to its exact defining
  -1/(6*pi*pi), fixing a digit typo without changing references or gates.
  New default-binary EDF2 terms/local derivatives and repaired-VWN5 fifth
  tests pass; broader dependent-recipe combinations remain separate.

- `berry-polarization`: Berry-phase polarization of insulating references
  with a selectable string mesh.

- Periodic wB97X-D: the Chai-Head-Gordon pair dispersion as a lattice sum
  (`dispersion = chg`, attached automatically), the Axilrod-Teller-Muto
  three-body term of D3 (`three_body`, `abc_cutoff`), and dispersion force
  constants in `phonons`, `phonon-bands`, `phonon-thermo` and
  `quasi-harmonic`.

- Exact-exchange force constants: `phonons`, `phonon-bands`, `phonon-thermo`
  and `quasi-harmonic` accept Hartree-Fock, global and range-separated
  hybrids (including `wb97xd`) for insulators; `raw_force_constants` is
  reported next to the symmetrized matrix.

- `quasi-harmonic` at a pressure (`pressure_gpa`, Gibbs energy) and mode
  Grueneisen parameters of the fitted volume-grid model.

- Molecular three-body dispersion (`dispersion = d3abc`, `d3bjabc`) with
  analytic gradient and Hessian.

- Skala learned functional (`xc = skala`, `skala_model`): pure-Rust
  evaluation and reverse pass of a model file with file-defined
  hyperparameters, self-consistent energies and one-electron properties of
  molecules; `tools/convert_skala_model.py`.

- Periodic interface: extended-XYZ lattices on the CLI, `Crystal` and
  `PeriodicCalculator` in Python, and an ASE adapter that routes energy,
  forces and stress through the common driver (dispersion models, `kpts`,
  extra controls, periodic `get_hessian`, `get_relaxed_atoms`, `BandPath`).

- Effective core potentials in the analytic excited-state gradient.

- Faster periodic Hartree build: the reciprocal sums run over a half space and the split exponent follows the cell volume (`hartree_compact_exponent` overrides it); the matrix is unchanged within the screening tolerance.

- `hyperfine` on the common driver: isotropic Fermi-contact couplings (MHz, gauss)
  from the spin density at the nuclei of an open-shell reference, with `nuclear_g`.

- Mulliken-projected density of states on `dos` (`projected`, by atom and
  angular momentum) and `pauli-spin-susceptibility`: Fermi-level density of
  states, bare Pauli spin susceptibility, electronic heat capacity and
  Sommerfeld coefficient of the rigid bands.

- Molecular excited states on the common driver: `excitations` (TDA/RPA
  energies, oscillator strengths and Tamm-Dancoff rotatory strengths),
  `ecd-spectrum` (Gaussian-broadened absorption and circular-dichroism
  curves with their sum-rule integrals) and `td-state-gradient` (analytic
  singlet TDA gradient for Hartree-Fock and pure LDA references).

- Spheroidal-pore dry frames on `rock-physics` (`dem`, `kuster-toksoz`,
  `self-consistent` with `pore_aspect_ratio`; Berryman's inclusion factors
  checked against a numerical Eshelby tensor) and thermal-conductivity means
  and Hashin-Shtrikman bounds (`thermal_conductivity_w_mk`).

- Add physical normal-flux magnetic slab finite/crystal references, dimension-specific full/range exchange singularity integration, continuous-normal Coulomb quadrature and raw refinement histories on all public routes. Independent term oracles, four k meshes and strict four-step derivative fits retain the original thresholds.

- Continuum uniform fields for arbitrary wire directions and in-plane slab
  fields through both dimensional field drivers, with analytic orbital
  contact, spin Zeeman, nuclear kinetic-momentum SOC contact and amplitude
  derivatives. Independent four-step fits, seven-metric geometry checks and
  matched Rust/CLI/native Python/ASE comparisons cover the accepted subset.
  Normal slab flux, open-axis Coulomb and slab crystal integration are also accepted; the complete remaining combinations stay in the coverage ledger.

- Magnetic Bloch Brillouin-zone integration with separate transfer spaces,
  intersector full/range exchange, analytic Gaussian singularity integration
  and XC from averaged spin densities. The finite-torus model remains explicit.

- Fixed-flux additional-spinor-operator energy derivatives, decomposed local
  Pauli particle/electric currents and physical B-to-minus-B London-basis
  time-reversal diagnostics on the uniform magnetic reference driver.
  Independent source/interaction oracles and four-step fits cover all routes.
- A portable Cargo build helper that remaps private build roots and audits
  native artifacts; relocated runtime tests use embedded resources without
  the source tree. Existing author and email metadata are retained.

- Uniform-flux continuum London spinor SCF on a compatible 3D magnetic
  Gamma torus (`periodic-magnetic-scf`), with periodic Coulomb/full/range
  exchange, uniform Zeeman and Pauli nuclear kinetic-momentum contact.
  The finite-cell model and its remaining infinite-crystal/open-axis
  combinations are tracked independently.
- General complex two-component dimensional SCF (`periodic-spinor-scf`),
  total-number Fermi-Dirac occupations, full spin-off-diagonal exchange,
  rotationally invariant LDA, prescribed-potential Pauli SOC with its
  kinetic-momentum contact, periodic spin Zeeman fields and analytic
  one-electron point/finite-nucleus SOC in the dimensional Coulomb gauge.
  Rust, CLI, native Python and ASE use one variational energy/Fock driver.
  Gradient/current XC, nonlocal-core/two-electron SOC, scalar relativistic
  terms and the remaining uniform-field/response combinations stay tracked;
  the separate magnetic-cell driver provides the accepted 3D uniform models.
- Complex collinear SCF in real static periodic scalar/vector fields
  (`periodic-field-scf`), retaining the full complex localized exchange
  map and analytic field-amplitude derivative, with raw orbital/spinful
  time-reversal diagnostics and explicit zero-net-flux model conventions.

- Ordinary weak-field Hall conductivity derivatives and Hall coefficients
  (`hall-transport`) from analytic isolated-band velocities and curvatures,
  with electron/hole sign conventions, SI units and raw Onsager diagnostics
  on the common Rust, CLI, native Python and ASE driver, including causal
  finite-frequency intraband tensors and full/range-separated hybrid/ECP bands.
- Covariant independent-particle intra/interband conductivity and intrinsic
  Hall (`coherent-transport`), exact degenerate occupation limits, physical AO
  connections, and non-Abelian occupied-slice Chern/quantum Hall response
  (`quantum-hall`). Magnetic-flux/SOC references and complete dimensional
  transport combinations remain tracked work.
- Masked local/nonlocal ECP ground-state operators and shifted hybrid reference
  fields in dimensional SCF, with a separate ECP energy contribution.

### Fixed

- Periodic exact exchange on a finite k mesh used every Born-von Karman
  replica of a density image; it now uses the minimum image of every pair of centres, which removes the
  dependence on an exchange cutoff larger than the supercell (energies,
  forces, stress and force constants of hybrids with coarse meshes change).

- Gamma-point phonons of cells whose first-order overlaps vanish by symmetry
  failed with "k-point density is not Hermitian"; the response kernel now
  projects the response density onto its Hermitian part.

- The low-level periodic SCF again refuses a functional with a built-in pair
  dispersion unless the caller states that it adds the lattice sum.

- Avoid unrelated second-order force-constant construction for Born-only
  requests while preserving the shared first-order sources and compatibility
  polar-mode calculation.

- Use the same AO-pair image selection for periodic dipoles, dipole nuclear
  derivatives, overlap metrics and physical cross-k links. The previous
  origin-centered dipole sphere broke the current Hermitian identity.

### Added (continued)
- Analytic band velocities and effective-mass tensors (`effective-mass`, with
  degenerate perturbation theory for band clusters) and constant-relaxation-time
  Boltzmann transport (`boltzmann-transport`: sigma/tau, Seebeck, electronic
  thermal conductivity, carrier concentrations) from k derivatives of the
  converged Kohn-Sham operator.
- Periodic DFT-D3 dispersion (zero and Becke-Johnson damping): lattice sums with
  coordination numbers through all images, analytic forces and strain
  derivative, on `periodic-energy`, `stress`, `opt-fixed-cell` and `cell-opt`
  (`dispersion`, `disp_cutoff`, `cn_cutoff`); second-derivative features
  refuse it until the dispersion Hessian exists.
- Common response framework: `ResponseReport<T>`, matrix-free CG/MINRES/GMRES
  solves with constraint spaces, and arbitrary-order Taylor jets for semilocal,
  VV10 and Gaussian-integral kernels.
- Molecular response: dynamic polarizability, complete static and dynamic
  beta/gamma tensors with nonlinear optical presets, mixed field responses,
  nuclear Taylor and mixed derivatives, analytic cubic/quartic force constants,
  directional APT/Raman/hyper-Raman, Watson VPT2 anharmonic bands with
  mechanical and electrical intensities, orbital energy and frontier gap
  gradients, orbital stability analysis with stability following, and
  collinear orbital magnetic response/NMR.
- Periodic response: finite-q screened density, inverse dielectric matrix,
  Mermin thermal response, screened electric/magnetic field response, and
  1D/2D/3D dimensional electrostatics, basis, Coulomb, XC, SCF and field
  response.
- Common calculation driver (`driver::run`, `dft-rs <feature>`,
  `dft_rs.run_feature`, `DftRs.get_feature_report`) with a feature registry,
  named controls that refuse unread inputs, JSON/text reports and `--save`.
  Driver calculations: energy, Mermin free energy, analytic gradient, charges,
  dipole, orbitals, multipoles to rank 8 and their analytic gradients,
  analytic Hessian, frequencies with RRHO thermochemistry, analytic IR/Raman,
  polarizability, response-derivative export, Molden and orbital export,
  conceptual DFT (grand-canonical, integer and Koopmans routes; charge and
  spin channels) and the feature registry.
- Three-dimensional finite-k periodic calculations on the driver: energy
  with analytic forces, electronic structure, band structures, DOS, the
  analytic stress tensor (`stress`, ASE `get_stress`), fixed-cell relaxation
  (`opt-fixed-cell`) and variable-cell relaxation at a hydrostatic pressure
  (`cell-opt`).
- Gamma-point phonons (`phonons`) from analytic force constants: skeleton
  second derivatives of every periodic energy term (overlap, kinetic, Ewald
  nuclear attraction, screened Hartree, XC with analytic periodic Becke
  weight second derivatives, Ewald nuclei), first-order Fock matrices and
  the finite-k coupled-perturbed equations, for insulating LDA/GGA
  references.
- Second-order elastic constants (`elastic`) with clamped and relaxed ions:
  skeleton strain second derivatives of every periodic energy term
  (closed-form real-space blocks and exact forward-mode jets for the
  reciprocal, background and XC parts), first-order strain Fock and overlap
  matrices, strain and internal-strain coupled-perturbed responses,
  Lagrangian constants and stress-strain coefficients; the validation route
  `elastic-fd` differentiates the analytic stress.
- Continuum moduli (`elastic-moduli`: Voigt, Reuss and Hill moduli, wave
  speeds, Debye temperature) and Christoffel wave velocities
  (`christoffel`) from the analytic elastic tensor.
- Phonon dispersion and DOS (`phonon-bands`) and harmonic lattice
  thermodynamics (`phonon-thermo`) from analytic supercell force constants.
- Third-order Birch-Murnaghan equation of state with analytic pressures
  (`equation-of-state`) and the quasi-harmonic model (`quasi-harmonic`).
- Electronic dielectric tensor (`dielectric-tensor`) and Born effective
  charges (`born-charges`) by periodic DFPT with the covariant k derivative,
  and the polar Gamma-point analysis (`crystal-ir`, `static-dielectric`:
  infrared intensities, LO-TO splitting, relaxed-ion dielectric tensor).
- Fermi-Dirac (Mermin) metallic DFPT force constants: the orbital-pair
  response with Lindhard factors, a fixed-particle-number Fermi-level shift
  and a GMRES solve, extending the Gamma-point `phonons` response to
  metallic LDA/GGA references.
- Electron-phonon coupling (`electron-phonon`) from the self-consistent
  displacement responses of a supercell: vertices, `lambda`, `a2F`,
  McMillan and Allen-Dynes critical temperatures and the isotropic
  imaginary-axis Eliashberg `tc_eliashberg_k` (controls `eliashberg_cutoff_cm1`,
  `eliashberg_t_min_k`), the nonlinear isotropic gap `Delta(T)` and its Pade
  (Vidberg-Serene) continuation to the real axis (`gap_temperatures_k`,
  `pade_points`).
- Continuum rock physics on supplied data (`aggregate-moduli`,
  `rock-physics`: Hashin-Shtrikman bounds, dry frames, Gassmann,
  Brown-Korringa, Biot and Skempton parameters) and continuum material cards
  of the analytic elastic tensor (`material-card`, JSON or CSV).
- Range-separated screened periodic Hartree build for the finite-k SCF,
  forces and stress (compact/diffuse split, Schwarz-ordered neighbour
  search, density-contracted plane-wave transforms).
- Molecular Fermi-Dirac references (`electronic_temperature`, fractional
  `electron_offset`) and their ensemble linear response with occupation
  response.
- Response solver controls: acceptance residual, divergence stop and minimum
  physical denominator.
- `cli::run_arguments` and Python `cli_main` expose the command line as a
  library entry point.
- `dft-rs` is the canonical CLI name; `dft_rs_cli` remains an alias.
- Every new calculation is available from Rust, the CLI, native Python and
  ASE; see [docs/rust-api.md#common-calculation-driver](docs/rust-api.md#common-calculation-driver).

### Added, not yet validated by acceptance tests

The calculations below compile, pass the lint checks and are connected to
Rust, the CLI, native Python and ASE through the common driver. Their test
files (`tests/p101_*.rs` to `tests/p116_*.rs`) are written and have not
been run; no fit or residual has been measured for them. They are not part
of a verified milestone.

- Periodic response in the supercell of the k mesh from Taylor series of
  the lattice sums along displacement, strain and field paths:
  `periodic-cubic-fc-along`, `periodic-quartic-fc-along`,
  `periodic-cubic-fc`, `periodic-quartic-fc`, `toec`, `foec`,
  `mode-gruneisen`, `periodic-beta`, `periodic-raman`, `electrostriction`,
  `piezo-higher`, and the self-consistent field of a cell in a finite
  uniform electric field (`periodic-efield`).
- Response of periodic cells at commensurate wave vectors: `dielectric-q`,
  `orbital-susceptibility-q` and `periodic-nmr-shielding` (without the
  bulk magnetization term).
- Divide-and-conquer on periodic cells: `periodic-dc`, `dc-stress`.
- Molecules in a finite uniform magnetic field with London orbitals:
  `bfield`, `bfield-gradient`.
- Spin-orbit response of molecules: `magneto-optical` with
  `magneto_optical_source = spin-orbit`, `magnetoelectric`,
  `spinor-polarizability` (self-consistent two-component state and its
  dynamic polarizability), `spin_orbit_self_consistent` on `spin-orbit` and
  `magnetic-anisotropy`.
- Periodic two-component references: `lab-frame-spin-spiral`, `dmi`;
  `berry-dipole` of metallic or doped cells.
- Restricted open-shell and spin-flip surfaces: `roks-hessian`,
  `mrsf-state`, `mrsf-hessian`.

### Changed

- The documentation was consolidated from 23 to 10 Markdown files: the
  dimensional/periodic notes into `docs/periodic.md`, the driver notes into
  `docs/rust-api.md#common-calculation-driver` and `docs/response.md`, the target registry notes into
  `docs/scope.md`, and the functional validation and D4 notes into
  `docs/FUNCTIONALS.md`. Chronological acceptance checkpoints are collected in
  the acceptance record of `docs/response.md`.
- Periodic Python bindings accept the same basis specifications and
  Gaussian94 file paths as the CLI.

## [0.1.1]

Predominantly a correctness release. A new sweep over the boundary of every
functional's domain found 182 failing functional/point/order combinations on
the 0.1.0 tree; all are fixed, and the sweep ships as a gate.

### Fixed

- **The periodic XC quadrature was not a smooth function of the geometry.**
  The local Becke denominator kept image centres inside a hard sphere around
  every grid point, so the partition of unity, the SCF energy and every
  derived quantity jumped by 1e-10 to 5e-9 Hartree whenever a centre crossed
  it; central differences of the energy of the p67 fixture stopped converging
  below h = 0.01. Every centre now carries a C-infinity switching function of
  its distance to the point (full weight inside `r_cut - (V/N)^(1/3)`, zero
  from `r_cut`), also as a competitor in the other cell functions. Value,
  gradient, strain virial, Hessian and the strain/displacement jets use the
  same weights. Periodic and dimensional XC energies shift slightly (wherever
  a centre sits in the switching zone).

- **PBE-family functionals returned a NaN Kohn–Sham potential whenever a spin
  channel was empty.** A hydrogen atom, H₂⁺, He⁺, or any N_β = 0 state with
  `pbe`, `pbe0`, `revpbe`, `hse06`, `hse03`, `lc-wpbe`, `lc-wpbe-hjs` or
  `lc-pbe` returned NaN instead of an energy. PBE correlation had no ζ = ±1
  branch, and because the exponent there is 2/3 the singularity struck at the
  *first* derivative — the potential itself. Atomization energies go through
  E(H) as a matter of course. Every open-shell oracle in the suite was CH₃
  doublet or O₂ triplet, both with N_β > 0, so nothing covered it.

- **`--x slater --c tpss` and every other split spec mixing a meta-GGA with a
  non-meta component panicked on any density**, not only at a boundary. Split
  specs could never resolve to `Family::MetaGga`, so `vtau` was never
  allocated and the meta-GGA driver indexed an empty vector. `tpss` and
  `r2scan` appear in both the exchange and correlation name tables, so
  `--x slater --c tpss`, `--x pbe --c r2scan` and `--x tpss --c vwn5` were all
  affected.

- **LDA correlation produced NaN second derivatives at full spin
  polarization**, poisoning everything built on `v2rho2`: CPKS, the analytic
  Hessian, polarizabilities and TDDFT. Slater exchange and LYP returned `-inf`
  as well.

- **B88, TPSS, P86, HJS-ωPBE and r²SCAN returned NaN for every derivative at
  zero density gradient** — including `vrho`, not merely `vsigma`, because a
  NaN in a forward-mode dual contaminates all tangent slots. Reached at the
  centre of any closed-shell atom and at any point of exact symmetry.

- **HSE06 and HSE03 returned NaN second derivatives at zero gradient**, via an
  overflow in the wPBEh kernel when ω = 0.

- **D3(BJ) returned NaN when the coordination-number weights underflowed**,
  reachable when a geometry step brings two atoms close together. C₆ no longer
  collapses to zero there either: the interpolation now asymptotes to the
  nearest reference point, which is the physical answer.

- **Atomic masses beyond krypton were fabricated as `2.5 × Z` amu**,
  mis-scaling every frequency, ZPE and IR intensity for exactly the heavy
  elements the bundled LANL and Stuttgart ECP sets exist to treat — Ru by
  −4.2 % in frequency, Pd by −3.8 %, I by −2.1 %. It was nearly exact at Pt and
  Au, where A/Z ≈ 2.5, so a spot check on gold would have confirmed the bug
  rather than exposed it. The table now covers Z = 1…118 from IUPAC 2021 with
  the standard-weight vs longest-lived-isotope rule recorded per element, and
  there is no fallback at all.

- **Ghost centres crashed vibrational analysis** by dividing by zero; they are
  now refused by name.

- **`frequencies --solvent ief:water` (and `ssvpe:`) aborted** instead of
  falling back to the numerical Hessian that was available throughout.

- **`periodic` and `sapt` silently ignored `--disp`, `--nlc`, `--solvent`,
  `--field`, `--nbo`, `--dump-wfn` and `--molden`**, printing a number with
  none of them applied. They are refused by name now, from one
  command × option table that also drives the docs and a contract test.
  `--unrestricted` was the worst of these: `PeriodicScfOptions` has a
  `reference` field that the CLI never set, so asking for a broken-symmetry
  periodic run silently got restricted behaviour.

- **SMD's `G_CDS` was not a smooth function of geometry.** The SASA neighbour
  shell used a hard cutoff, so the value jumped as an atom crossed it —
  measured second difference 4.22e-5 Ha, now 1.7e-18. Degenerate neighbour
  shells (methane's carbon, octahedral centres) no longer commit to an
  arbitrary quadrature axis. Separately, `water_pair_free`'s second conjunct
  was a tautology that collapsed the gate to a hard-coded triple.

### Added

- Add EDF2 with published independent weights and0.1695 exact exchange.
  After explicit resumption approval, the strict shared build,48 local fits,
  RKS/UKS terms,90 C1 gradient/Hessian fits and public energy/gradient routes
  pass. Component fifth74fits/442zeros pass; six original zero-gradient
  kernel comparisons remain failed and unchanged. Broader combinations
  and five standard molecules remain separate.
- Correct the VWN5 spin-stiffness coefficient to its exact defining
  -1/(6*pi*pi), fixing a digit typo without changing references or gates.
  New default-binary EDF2 terms/local derivatives and repaired-VWN5 fifth
  tests pass; broader dependent-recipe combinations remain separate.

- **RS-I-RFO transition-state optimization.** A Householder image function
  (`P = I - 2tt^T`) flips the followed Hessian mode, retains coupling to its
  complement, and solves the restricted generalized augmented-Hessian
  equation with `S = alpha I`. The trust radius is enforced by solving for
  `alpha`, not by scaling an overlong step. Rust, CLI (`--saddle-method
  rsirfo`) and Python (`saddle(method="rsirfo")`) expose it alongside P-RFO.
- **Ideal-gas RRHO thermochemistry** from Eckart-projected frequencies:
  zero-point, thermal internal-energy and enthalpy corrections, separated
  entropy terms, and Gibbs free energy at caller-selected temperature,
  pressure, rotational symmetry number and electronic degeneracy.
- **Published D2/D3 damping coverage** for additional common functionals;
  unavailable parameter combinations are typed refusals rather than guessed
  defaults.

- **Pseudo-substituent fitter mechanics and public request plumbing.** The
  standalone `fit_substituent --synthetic-check` binary uses the library's
  deterministic group-wise 80/20 split, normalized weighted least squares,
  Tikhonov prior, Frobenius-correct packed-B eigenvalue projection and
  projected L-BFGS; it recovers a known synthetic `c/B` parameter set far
  below the required `1e-6` gate and writes an explicitly
  `candidate-unaccepted` record containing level, seed, split and provenance
  fingerprint. CLI `--pseudo-sub` and Python `pseudo_sub=` share the strict
  zero-based Me/Et/tBu/Ph/CF3/OMe parser and stop before SCF with typed
  missing-parameter errors. No physical named record or nonzero operator is
  enabled; the staged reference evaluator and all v1/v2 acceptance data remain
  prerequisites.

- **Direct TDDFT J/K and periodic level shifting.** TDA/RPA now use the same
  direct or guarded in-core J/K engine selected by `ScfOptions`, including a
  separate direct Erf exchange engine for RSHs; HF/PBE/LC-PBE operator and
  spectrum parity are gated against in-core results. Periodic SCF adds an
  auto-released complex-Bloch virtual level shift, with `--level-shift`,
  `--damping` and `--diis-size` wired through the CLI and Python. Periodic
  Core remains the compatibility default, while explicit GWH and SAD guesses
  are now available; SAD builds one periodic HF Fock from on-the-fly atomic
  densities before the ordinary finite-k occupation solve. Complex finite-k
  ADIIS now interpolates one shared convex coefficient set over every spin and
  k point, including imaginary Bloch blocks, before handing off to CDIIS;
  `--no-adiis` and `--adiis-until-error` are wired through CLI and Python.
  Finite-temperature runs deliberately use CDIIS because the zero-temperature
  ADIIS model has no entropy curvature. Periodic SOSCF remains unimplemented.
- **Fixed-cell periodic geometry optimization.** A public analytic-force
  L-BFGS path now rebuilds the complete periodic cell/SCF at every trial,
  preserves the exact final Cell/SCF identity, and minimizes total energy or
  Mermin free energy under smearing. Rust, `periodic --relax`, and Python
  `periodic_scf(relax=True)` expose the same convergence and history controls.
  Lattice vectors remain fixed: electronic stress and cell relaxation are not
  claimed.
- **Gas-phase closed-shell singlet LDA-TDA analytic gradients.** The
  grid-aware API adds the LDA `f_xc` and `f_xxc:T:T` response contractions and
  the full moving atom-centred-grid excited-state Lagrangian. Distorted-water
  SVWN5 agrees with rebuilt-grid finite differences in all nine components
  (worst `4.45e-8 Ha/Bohr`), and independent gates cover the third-kernel
  contraction, zero-amplitude ground-gradient limit and translation sum rule.
  GGA/meta-GGA, hybrid/RSH, VV10, PCM, triplet, RPA and open-shell cases remain
  explicit typed refusals rather than incomplete forces.
- **GIAO NMR for range-separated exact exchange, LDA, and GGA, with an
  experimental current-corrected meta-GGA path.** The RSH path uses
  an exchange-only Erf field derivative (`J'_erf` is structurally excluded)
  plus the matching long-range CPHF coupling, with finite-omega origin and
  large-omega HF-limit gates. LDA adds a batched London-phase `dVxc/dB`
  contraction, gated against an independent complex-phase finite difference,
  antisymmetry, the exact origin-shift identity, and an end-to-end SVWN5
  three-origin shielding. GGA adds the separately observable phase-gradient
  term and is gated with unequal spin densities plus end-to-end PBE, B3LYP,
  and CAM-B3LYP three-origin shieldings. Meta-GGA adds the conventional-tau
  London terms and the Dobson current-current CPHF kernel; it is internally
  FD/origin gated but is not release-complete because the pinned external NMR
  oracle does not support meta-GGA. SC-VV10 remains an explicit refusal.
  Full total/diamagnetic/paramagnetic tensors for HF, SVWN5, PBE,
  BLYP, B3LYP, and CAM-B3LYP are also frozen against PySCF 2.14.0 plus a
  source-hash-pinned pyscf-properties commit.
- **XDM dispersion** (exchange-hole dipole moment), implemented from Becke &
  Johnson's papers as the answer to D4's absence rather than a substitute for
  it. Its C₆/C₈/C₁₀ are computed from the Becke–Roussel exchange hole of the
  actual converged density, so it needs no locked reference database — which
  is precisely what makes D4 unavailable here. The only bundled data is a
  small free-atom polarizability table from an independent atomic-physics
  compilation; the companion free-atom *volume* is computed from this
  project's own `sad_density` rather than tabulated, which makes the
  free-atom limit exact by construction. Measured against the classic
  noble-gas C₆ at HF/aug-cc-pVTZ: Ne–Ne −2 %, Ar–Ar −13 %, He–He +18 %.
  Those are reported as measured, not tuned. Per-functional BJ damping
  parameters are **not** bundled — they are empirical fits this project
  cannot verify independently, so `a1`/`a2` are supplied by the caller and a
  request without them is refused by name. No analytic gradient, and no
  three-body term. The CLI exposes the energy through `--disp xdm` for
  `energy`/`charges`, with required `--xdm-a1` and `--xdm-a2`; derivative
  commands refuse it before starting an SCF.
- **Analytic Hessian for empirical dispersion** (D2, D3(zero), D3(BJ), CHG),
  seeded one dual level deeper than the gradient. `--disp` no longer forces
  the numerical path.
- **Analytic Hessians in a uniform external field.** Direct nested-bra and
  mixed bra-ket second derivatives of all three dipole-integral matrices add
  the field skeleton to the existing CPKS response. The integral contraction
  and the complete field-dependent HF Hessian are separately gated against
  finite differences. Frequencies, P-RFO, Raman and Python Hessian/vibration
  entry points now share one analytic-first dispatcher; only a typed
  unsupported derivative falls back numerically.
- **Analytic IEF-PCM and SS(V)PE Hessians.** The variational reaction-field
  Fock derivative now differentiates `q_bar = (K^-1 R + R^T K^-T)v/2` by the
  adjoint equations, retaining distinct `K_a` and `K_a^T` applications for
  IEF-PCM and their symmetric average for SS(V)PE. Both complete Hessians are
  gated against rebuilt-SCF finite differences of the analytic gradient.
- **Third-order XC derivative primitives** for every LDA/GGA/meta-GGA driver,
  exposing all 4/35/84 symmetric blocks. `tests/p22_xc_boundaries.rs` sweeps
  every registry functional over the boundary of its domain at orders 0--3,
  plus a finite-difference-of-second-derivatives gate and a test that pins exactly
  which functionals report a stated *convention* rather than a true one-sided
  limit at σ = 0.
- **`tests/p21_cli_contract.rs`**: the command × option acceptance table.
- **The Python surface widened**: `polarizability`, `raman`, `optimize`,
  `saddle`, `nmr`, `tddft` and `sapt` as
  entry points, and `field=` / `solvent=` / `eps=` on `calculate`, resolving
  through the same 179-entry solvent table and the same `smd:`/`cosmo:`/
  `ief:`/`ssvpe:` grammar as the CLI. `sapt` takes two monomer geometries
  rather than one array plus `--natoms-a`. The CIS excited-state gradient
  stays Rust/CLI-only.
- **Pseudo-substituent wiring** (`src/pseudo_sub/`) with no physics attached:
  a per-atom basis override, the site table, and a `SiteOperator` with no way
  to be nonzero. Gated by methane computed as "C + 3H + 1G" against ordinary
  methane under both Hartree–Fock and a grid-based functional. Named
  substituent specs now share a strict zero-based parser (`4=Me; 7=Ph`,
  comments and `@file` includes) and recognize Me/Et/tBu/Ph/CF3/OMe. All six
  still refuse by name: no v1/v2 fitted record has passed the required gates,
  and CF3/OMe lack even a model/validation specification. The independently
  gated local-frame Jacobian and real Wigner-D mechanisms are present, but no
  nonzero `V_G` is assembled. See `docs/scope.md` §10b.
- Physical invariants that had no gate: the virial ratio's approach to 2,
  Koopmans for frozen-orbital HF, aufbau ordering, dipole origin-independence
  and its `Q·d` shift for an ion, `CᵀSC = I` on the Rust side, UKS ≡ RKS on
  seven functionals, size consistency separated from dispersion's deliberate
  R⁻⁶ tail, Td methane's t₂ degeneracy and vanishing dipole, overlap positive
  definiteness, polarizability positive definiteness, the rotational/torque
  sum rule and rotational covariance of analytic gradients, Hessian symmetry,
  the APT translational sum rule including a cation, exchange under uniform
  coordinate scaling, ERI permutation symmetry for d/f shells and the
  erf/erfc kernels, grid convergence and orientation stability, and the
  structural relations of an excitation spectrum.

### Changed

- **The wPBEh s-rescaling is now smooth**, which is a deliberate departure
  from libxc and the only one in this project. libxc's short-circuits are
  discontinuous by −5.141e-4 at s = 1 and +1.616e-3 at s = 15; the bare
  Henderson softplus that removes them is *also* wrong, mapping s = 0 to
  −1.892e-4 and violating the uniform-gas limit. The offset-corrected form
  satisfies both. Cost, measured: HSE06/HSE03/LC-ωPBE totals move by
  1.0–1.8e-4 Ha (≈ 0.1 kcal/mol) away from libxc. The hole model itself is
  still pinned pointwise at 1e-9 against the same frozen libxc points,
  evaluated through libxc's own rescaling kept as a test reference. See
  `docs/FUNCTIONALS.md`.
- `sr_factor`'s large-a expansion gained its u⁶/420 term, cutting the
  discontinuity at its switch point from 2.15e-6 to about 1e-9.
- `DftError` gained `Unsupported { feature, missing }` and is now
  `#[non_exhaustive]`. "Not implemented" is no longer indistinguishable from
  "bad input", so fallbacks branch on the variant instead of parsing messages.
- **Grid atomic radii unified on one source.** The noble-gas entries were
  Clementi *calculated* radii inside an otherwise Bragg–Slater table, which
  flips the sign of the Becke size-adjustment for a rare gas beside a light
  atom. Now Slater 1964 throughout, matching what PySCF uses so the oracle
  comparison is like for like. This is a consistency fix, **not** a
  convergence improvement: measured, level-3 error improves 14× for NeH⁺ and
  gets worse for most He-containing geometries. The converged answer is
  unaffected either way.
- Documentation reconciled with the code in eleven places. Five had told
  readers that a shipped, gated capability does not exist — GIAO NMR
  shieldings, periodic ECP analytic forces, the ECP analytic Hessian,
  open-shell SAPT and non-aqueous SMD. One `THEORY.md` paragraph, duplicated
  verbatim, asserted the ECP analytic Hessian is refused and cited a test as
  proof; that test exists nowhere in the tree.

### Known limitations recorded during this release

- **The rotational sum rule is quadrature-limited for meta-GGA functionals.**
  Its translational partner is an algebraic identity of the Becke partition of
  unity and holds at ~1e-15 at any grid density; the rotational one also needs
  the *angular* quadrature to be isotropic, and this grid is a product
  (Gauss–Legendre × Chebyshev) rule rather than Lebedev. At grid level 9:
  svwn5 ~4e-10, b3lyp ~3.5e-9, cam-b3lyp ~2.5e-9, but m06l 2.8e-6, m06-2x
  3.2e-6, ωB97X-D 1.1e-5. It is gated as a convergence trend, and
  `src/grid/mod.rs` already anticipates the Lebedev backend that would fix it.
- The analytic Hessian is force-symmetrised before being returned, so
  `H = Hᵀ` holds by construction rather than as an emergent property of the
  CPKS solve. The test on it is a regression gate on that guarantee.
- **D4 remains unimplemented, and the question was re-examined rather than
  inherited.** PySCF does not avoid the licence — it takes a runtime
  dependency on the LGPL-3.0 `dftd4` and ships no data, which is compliant but
  unavailable here (no FFI, `forbid(unsafe_code)`, four-crate allowlist). The
  reference polarizabilities live in a 9,427-line generated file with no
  licence header of its own, inheriting the repository's LGPL. See
  `docs/FUNCTIONALS.md` for the full record, including one Apache-2.0 near-miss and why
  it was not used. XDM (above) is what was done instead — not a workaround
  for the licence, but a model that does not need the database at all.




<!-- Represented-pair residual certification checkpoint -->
Current original57 canonical acceptance is 4368/4368 (HF 1092, PBE 1092, TPSS 1092, PBE0 1092); 49 failed intervals are retained. Its correlated leaf terminated with two tests passed, including the zero-PT2 limit (maximum gradient defect 2.442490654175344e-15 and ordered-Hessian defect 2.012279232133096e-15). Independent canonical fits span p=1.802732926104004..2.195880426067760, minimum R2=0.999026043601219. The enclosing manager exit1 remains preserved for its earlier H2/public failures; it does not replace the actual correlated-leaf exit0. This original artifact checked raw diagnostics at base Hessians; it does not certify every displaced gradient of the repaired artifact. This remains partial for the broader program; the original57 correlated chain is terminal. The stronger repaired-artifact status is stated separately below. Actual59 retains its measured 14 closed, five harmonic, five public and five unchanged p39 test passes, including the recorded H2 diagnostics. A separate algebraic counterexample shows that rounding retained high-plus-low amplitudes before the original Hessian action can hide a low-part relative residual. Those bounded passes do not certify that generic case. Source60 preserves both operator images and adds the counterexample and tiny-source dispatch regressions. The exact repaired core in recovered joint02 (manifest ec007b0e645f6f4ffc6ad692a3cbe82749d00ebb1d2de38f23a81ebf93c22fa0) passes all six warning-strict319-target module/Clippy configurations. The first joint metadata failure from two reference-test module imports is retained; the two canonical-import repairs leave every production byte unchanged. The one audited official Rust1.85 wheel/library and CLI have completed successfully. On exact joint02 rlib3a881819430757f0a773720ab4bef08fa81901b4c8656cace321a1f3ef0a61d2, all16 exact-source closed tests (including lost-low and tiny-source cases), five harmonic Rust tests and five native/ASE/canonicalCLI tests pass. The full-PT2 B2PLYP/H2 frequency is 4750.103488154000 inverse centimetres; the42 actual split-part returned-equation residuals have maximum 4.375957819920e-16, and physical stationarity has maximum 7.219401424865e-17 Hartree. The unfloored reconstructed-source ratio maximum 1.195234215610e+00 and its warning remain recorded separately. This is bounded source60 numerical evidence; its unchanged p39 regression now passes all five tests on the same repaired artifact (1036.29 seconds of test time). Independent four-step directional fits give D3 p=2.000325081038, R2=0.999999992984 and D4 p=1.998278797624, R2=0.999999801093; Richardson ratios are 15.7937096387 and 15.8908477116. This certifies those two directional higher-order checks, not complete rank-three/four tensors. The stronger every-sample4368 rerun has now passed all4368 canonical components on exact repaired joint02 rlib3a881819 (HF/PBE/TPSS/PBE0 1092 each), with 49 failed intervals retained. Independent p=1.802732929711707..2.195880426061605, minimum R2=0.999026043824687. Every displaced gradient and base Hessian passed convergence and all nine finite raw bounds. The zero-PT2 limit passed with gradient/Hessian defects 1.554312234475e-15/1.776356839400e-15. The actual correlated stage took 34944.750 wall seconds with observed peak group RSS 36061184 bytes. The original supervisor disappeared while its exact numerical worker survived; a separate observer captured that exact worker's native OS exit status as 0 without restarting or signalling it. The complete Rust test report and independent canonical refit certify the numerical result; the supervisor loss remains recorded. This closes the stated C1 restricted four-parent/seven-channel fixture, not all methods, geometries, complete higher-order tensors, crystal properties or chemical accuracy. The historical actual59 tests remain separately attributed and do not certify the former rounded-low case. The idle source59 all4368 successor was superseded before launch. Its replacement gates every displaced gradient and Hessian sample and independently closes the exact canonical component IDs at terminal. Operators, scientific thresholds, failed intervals and reference values are unchanged.
