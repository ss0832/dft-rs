<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 ss0832 -->

# Excited-state spin and analytic nuclear derivatives

This work extends the existing excited-state paths. It is not universal
excited-state acceptance. The frozen T120 singlet optimization evidence in
[the target-specific acceptance record](validation-matrix.md#t120-isolated-singlet-tda-optimization)
predates the new driver routing below. New tests must run against a rebuilt
library before their results can be promoted in the coverage ledger.

## Existing architecture and first integration

`response/tddft.rs::TdaSolver` assembles a spatial spin-adapted singlet or
triplet problem on a restricted closed-shell reference. On an unrestricted
reference it assembles spin-conserving alpha/beta singles. It supports TDA
and paired RPA. The scalar excitation problem and reference multiplicity
are different controls: `multiplicity` chooses reference occupations;
`td_spin` chooses the excitation channel.

`response/nuclear/frame.rs::excitation_path` transports reference projectors,
the AO metric and excitation amplitudes in a moving nuclear frame. Its
nondegenerate eigenvalue recurrence differentiates the actual response
operator. `excited_state_hessian` uses order-two coefficients for every
Cartesian direction and pair sum, including all mixed Hessian entries.
There is no displaced calculation in this production Hessian.

The original optimizer called the restricted singlet HF/pure-LDA Z-vector
gradient directly. The first new integration shares fixed-state gradient
evaluation with `td-state-gradient` and accepts the explicit
`td_gradient_route="path"` option. The default remains `"z-vector"`.
The path option differentiates the reference and excitation energies
analytically at every optimizer point; a failed path convergence report is
a typed `ResponseNotConverged` error. Reports identify the actual
`td_method`, `td_spin`, `td_gradient_route` and final path residual. Existing
restrictions of the underlying provider still apply. This connection is
prepared for rebuilt-artifact verification; it is not evidence that all
functionals or spin sectors have been accepted.

## Physical spin contract

| Reference / excitation space | Meaning and required extension |
|---|---|
| Restricted closed shell, spin-adapted singles | Singlets and triplets are accessible. A quintet or higher-spin state cannot be obtained by relabeling a single particle-hole excitation from a singlet. Higher excitation rank or a suitable different reference/model is required. |
| Collinear unrestricted spin-conserving singles | Fixed electron number and spin projection; arbitrary valid reference occupations are possible. Spin contamination prevents assigning a generally exact excited-state `S` or `2S+1`. Require explicit projection/channel metadata and independent spin diagnostics. |
| Restricted open shell | Closed/open/virtual orbital constraints and spin-adapted configuration couplings require their own equations. Reusing the unrestricted operator does not certify them. |
| Mixed-reference spin-flip | Existing `driver/mrsf.rs` and spin-flip derivative paths are separate from ordinary TDA. Their singlet/triplet models are not general arbitrary-spin configuration spaces. Extend and test their actual channel matrices. |
| Generalized/noncollinear/SOC | Full complex spinor excitations, transverse kernels and phase/metric transport are required. Exact multiplicity is not generally a good quantum number, especially with SOC; report the spinor model and measurable spin character. |
| Degenerate or crossing roots | A single-root Hessian need not exist as a smooth scalar derivative. Require a transported subspace/effective-state model, gap diagnostics and explicitly defined adiabatic/diabatic quantities. Do not silently swap roots or regularize the physics with a denominator floor. |

The current optimizer follows squared configuration-amplitude overlaps. The
T120 oracle independently tracks AO transition densities, but that oracle
does not replace the production tracker. Robust transported tracking and
degenerate subspace derivatives remain implementation tasks.

## Functional and backend contracts

| Family / backend | Required analytic content and acceptance boundary |
|---|---|
| HF and local/semilocal LDA/GGA/meta-GGA | The moving reference, grid/weights/AOs, descriptor metric and projected response kernel must all contribute. A second nuclear derivative of the kernel generally consumes energy-provider derivatives through fourth order. Dense `eval(2)` or a first/second pointwise test alone is insufficient. Meta-GGA includes tau and all cross channels. |
| Global and range-separated hybrid | Keep the semilocal contract plus moving full/SR/LR exchange integrals and coefficient/omega conventions. Use actual range-kernel paths; do not infer them from a global hybrid coefficient test. |
| Nonlocal, learned, solvent and fitted terms | Explicit geometry-dependent response-kernel providers are required. Respect existing typed refusals for unconnected terms; provider availability in a catalogue is insufficient. |
| Double hybrid | Existing `response/correlated_tda.rs` defines a second-order propagator correction: parent adiabatic operator, singles Hessian/metric, doubles coupling and transition corrections. Its nuclear derivatives must differentiate this same pole model, reference orbitals, denominators, amplitudes, metric and target coefficients. A ground-state PT2 Hessian is not an excited-state pole Hessian. |
| ECP and other references | Moving core/projector jets and reference constraints compose with each above only after their interfaces are connected and independently tested. Separate family tests do not certify the cross product. |

No new double-hybrid excited-state approximation is selected here.
`docs/FUNCTIONALS.md` distinguishes TD(A)-DH/CIS(D) corrections from
correlation-kernel response. The implemented propagator model is the concrete
starting point. A request to replace it with a distinct CIS(D), spin-scaled
or state-specific model requires an explicit scientific definition before
implementation; it cannot be inferred from a named ground-state recipe.

## Staged implementation and validation

1. Connect existing analytic path gradients to optimization without changing
   the default model. Rebuild once in coordination with the other owners.
   `tests/excited_spin_paths.rs` prepares all 12 gradient and all 144 Hessian
   entry fits for a C1 two-electron four-center HF triplet, plus a genuine
   optimization step and independent fresh endpoint. Both Rust tests pass on
   rebuilt artifact09 (`9b5bee22f4b8373e511cf8186aa12e79125a5ee494659c679da1473e00f55704`).
   The subsequent Python/native/ASE/CLI triplet optimizer test passes on the rebuilt cf6 artifact described below.
2. Exercise each distinct semilocal descriptor, spin and exchange path with
   full-component independent four-step fits and residual/gap checks. Retain
   per-provider mixed derivatives through the required order and per-recipe
   registration/coefficients. Follow [verification by implementation contract](conventions.md#verification-by-implementation-contract)
   rather than duplicating every property for every name.
3. Extend transported state/subspace tracking and the required open-shell,
   spin-flip and complex spinor excitation operators. Test spin/rotation
   limits and reference constraints separately; do not attach unjustified
   exact multiplicities.
4. Differentiate the explicitly identified correlated pole model and its
   doubles/metric equations; certify SS/OS and zero-correlation limits, full
   derivative components and public reports independently.
5. Add representative native Python, ASE and canonical CLI acceptance for
   each completed model. T120 covers the isolated HF singlet; the triplet optimizer step and full-spin HF public routes have the separately bounded acceptance below.

Full-suite and all-family completion remain pending. Builds are serialized
with the functional and geometry owners after the high-angular-momentum
memory incident; no timing benchmark or security workaround is involved.

## Existing-path baseline evidence

Before the coordinated rebuild, the derivative filter above passes against
the earlier immutable library `a1c6cb7f342cff4a3f37937563934fc362d6da9065af468501eefd773d473e92`.
All 156 entries pass the first `[0.04,0.02,0.01,0.005]` interval, with
`p=1.981311..2.006563` and minimum `R2=0.99997561`. No failed interval is
discarded. The serial Rayon-one/faer-one correctness run records peak child
RSS `13188` KiB. Raw component errors, commands, test hash and artifact
identity are retained in the local acceptance record
`excited-spin-paths-20261030/baseline-triplet/full-fit-results.json`.

This establishes the existing HF triplet gradient/Hessian path and the new
test's oracle/interval. It does not test the newly edited optimizer, its
convergence guard, the changing geometry implementation, or new Python/ASE/CLI
routes. Those remain pending the shared current-artifact build and rerun.

The follow-up artifact09 rerun passes both the complete derivative filter
and the newly edited triplet optimizer's real step/fresh endpoint. Its
49.96-second serial correctness run is not a performance benchmark. The
recorded child RSS maximum includes the test compiler and must not be
compared with the runtime-only baseline RSS. Python/CLI acceptance remains
separate.

## Full-spin implementation in progress

`response/spinor_tda.rs` now provides the full complex occupied/empty TDA
space and an isolated Hermitian-pole recurrence through second order.
Negative poles are preserved as instability diagnostics. Degenerate-root
derivatives are typed refusals, with no denominator floor. The exact new
module passes a separate source harness: all three triplet components and
the singlet of independently assembled HF spin couplings, complex orbital
phase covariance, kernel complex-linearity and memory-budget diagnostics,
and all two-parameter gradient/Hessian entries against a closed-form
complex two-by-two eigenvalue. These contracts do not replace molecular
acceptance or public-library integration.

`nuclear/spinor/excited.rs` connects the static pole problem to the existing
spin-covariant molecular kernel, retaining both Hermitian components of a
non-Hermitian transition density. `excited_path.rs` builds moving complex
orbital/Fock and complete Coulomb/full-range exchange matrix jets, followed
by analytic pole derivatives and Cartesian Hessian polarization. Its initial
nuclear implementation explicitly refuses semilocal mixed-kernel derivatives
until that provider is connected. The full-spin common driver identifies
`td_spin=spinor`, not an unjustified exact multiplicity; paired full-spin RPA
remains a separate pending implementation.

These source changes have compiled and passed the bounded molecular/public acceptance below. Later integral/provider changes are not silently included in that immutable artifact.

## Full-spin HF acceptance

The Python/lib/bin snapshot `spinor-excited-integration-20261030` has library
SHA256 `cf6e5879deac8c36f283a19d07fb7df7ef8bd9486b884ef4451ad921d03bc165`.
On a genuine C1 four-center, three-electron HF reference with fixed transverse
real and imaginary AO sources, `spinor_excited_nuclear` compares all 15 poles
with an independent spin-orbital Hamiltonian assembled directly from AO ERIs.
All 12 nuclear gradient and all 144 Hessian entries pass independent four-step
fits. There are 146 second-order and 10 fourth-order accepted fits; minimum
R2 is 0.99907082. All 29 failed exploratory intervals remain in the evidence;
three previously unresolved Hessian entries resolve on a finer interval
without changing gates or introducing an error floor.

Native Python, ASE and CLI all pass each of `excitations`,
`td-state-gradient` and `td-hessian`. Maximum CLI differences are
6.22e-15 Ha for poles, 1.47e-14 Ha/Bohr for gradients and
1.34e-13 Ha/Bohr^2 for Hessians. The maximum raw response diagnostic is
6.70e-14 and fresh-Fock defect is 9.99e-16. These diagnostics are not a new
relative residual definition; the later atomic library separates relative
solve residuals from dimensional stationarity and projector/frame defects.
The full-spin Hessian report currently provides derivative tensors without
the collinear harmonic-analysis postprocessing. Additional fixed AO sources
have zero explicit nuclear derivative, so this is not moving nuclear SOC.

The local evidence record is
`excited-spin-paths-20261030/spinor-final-results.json`, with exact commands,
artifact/test hashes, all raw fit intervals and public-route differences.
Tests use Rayon one and faer one. No timing result is claimed.

The later atomic library
`f87dbddfe21946953f30c887fa3ba3ede60edf6ccd200420865d720f3cb16399`
also passes four full-spin operator/pole contracts, the two bidirectional
XC contracts (350 derivative fits and 96 normalized coefficients), and five
representative provider/composition fifth-order tests. The latter cover
1154 nonzero full symmetric local entries and 958 structural zeros at a
smooth interior point; 11 exploratory failed intervals are retained.
Weighted-recipe checks include SVWN5, PBE, TPSS, PKZB, PBE0 and CAM-B3LYP.
This does not certify branch endpoints, all registered providers, or a
connected semilocal excited nuclear Hessian. Global spin projection adds a
spin-scaling tangent, so its second nuclear mixed-kernel path generally
requires collinear provider derivatives through fifth order.

The common-orbital flag TDA operator separately passes full independently
assembled spin-adapted determinant Hamiltonian comparisons for singlet,
doublet and quartet HF references (matrix errors below 2.39e-15 Ha).
The determinants' S-squared and Berry norms are checked explicitly. This is
the spin-conserving common-orbital flag space, not transverse spin response.
Its common-driver routes and analytic nuclear pole derivatives are still
being connected; these low-level tests do not promote public ROKS support.

Remaining implementation includes moving semilocal transverse mixed kernels,
common-orbital nuclear pole paths/public routes, transported state/subspace
tracking, full-spin RPA and derivatives of the existing second-order
correlated propagator. Generalized unequal SS/OS scaling needs an explicitly
defined spin model; the invariant equal-coefficient model can be extended
without choosing a fixed spin axis implicitly. None of these gaps is closed
by registering a functional or by ground-state PT2 acceptance.

## Moving semilocal and common-orbital integration

The new `spinor_xc/directional.rs` differentiates every global-projection
potential coefficient in an independent density direction, including the
spin-scaling tangent. Tests use a complete Walsh direction basis in all
active descriptor channels: four for LDA, sixteen for GGA and thirty-two for
meta-GGA. All 4,044 independent potential/energy coefficient fits through
nuclear order two pass; 1,104 inactive output coefficients are exact zeros.
Minimum R2 is 0.99906525; all 26 failed exploratory intervals remain recorded.
The same complete test passes on the actual new compiled library, not only
an extracted-source harness. This is a smooth interior fixed-quadrature
contract, not branch-boundary or molecular pole acceptance.

The moving-grid implementation reuses the original Pauli AO potential
contraction. An independent density finite difference checks all 120
Hermitian potential-matrix coefficients for a generic moving-AO/weight
point across LDA/GGA/meta-GGA, with minimum R2 0.99993836 and no failed fit
interval. This composition test is combined with the complete descriptor
direction contract, rather than claiming one tangent is a full kernel test.
The molecular excitation path now connects this mixed kernel, independently
refines its spin-angular rule, and retains both Hermitian components of a
complex transition density. Molecular full-component acceptance is pending.

The common-orbital path now transports closed/open/virtual flags, retains
noncanonical Fock and interaction jets, and solves reference rotations
through the full pole-derivative order. The real paired flag TDA operator
uses the existing Berry weights. Both the Cartesian analytic Hessian and
common-driver excitation/gradient/Hessian routes are implemented. Prepared
doublet/quartet tests independently assemble the determinant Hamiltonian
at every displaced geometry; their full 156-entry acceptance must pass
before the corresponding ledger cells are promoted.

Snapshot `excited-mixed-integration-20261031-03`, library SHA256
`5b14dc0d1b7697d4f02b511d90fa30fce03355ea4d18a18fc8099f34c49a6f4e`,
compiles with warnings denied. Its geometry files deliberately come from
the peer's accepted budget12 snapshot; the peer's live geometry work is
excluded and provenance is recorded. This artifact passes the 4,044-entry
contract. The first flag driver run failed at missing report-unit object
initialization before numerical acceptance. Report-only snapshot04,
SHA256 `0d8b276de53406547c604b36a4559e34c6b735f4ff05ea648011f632027ecf86`,
also compiles with warnings denied; its strict-unit test then rejected the
noncanonical `states[]` label spelling. That metadata correction is queued
in the next shared Python/lib/CLI snapshot. Both failed runs remain visible;
neither is a numerical acceptance pass or a relaxed unit check.

A neutral four-center/coarse-grid LDA preflight also failed to converge its
collinear seed in 400 iterations. A more gapped charged fixture and refined
fixed grid are separate preflights; they do not replace that failure or
certify all physical references. Healthy runs remain resource bounded and
are not interrupted to obtain an easier status.

Local evidence: `spinor-xc-directional-20261030-07/final-results.json`,
`excited-spin-paths-20261030/mixed-point-library-contract/final-results.json`,
and `spinor-grid-mixed-20261030-02/final-results.json`. Existing isolated
full-spin HF public/nuclear evidence remains on its separately named cf6
artifact. New semilocal or flag code does not silently extend that evidence.

## Common-orbital HF acceptance

The shared Python/lib/CLI artifact `task-pw92-integration-20261030` has library
SHA256 `236be05047a1c10a428d9567f045da345183cefdffd163170a05885dfbac848e`.
The C1 four-H+, three-electron doublet and quartet each pass all 12 gradient
and 144 Hessian entries against independent determinant-Hamiltonian scalar
energies and displaced analytic gradients. Every fit passes the first
`[0.04,0.02,0.01,0.005]` second-order interval. Doublet p is
1.963899..2.009326, minimum R2 0.99990620; quartet p is
1.991308..2.005607, minimum R2 0.99999484. No failed numerical fit interval
occurs in these two final runs; earlier report-unit failures remain distinct.

All six combinations of two reference multiplicities and three calculations
pass native Python, ASE and canonical CLI comparisons under strict units.
Maximum differences are 2.67e-15 Ha for poles, 1.49e-15 Ha/Bohr for gradients
and 2.09e-15 Ha/Bohr^2 for Hessians. Maximum reference stationarity is
3.58e-16 Ha and maximum fresh relative response residual is 1.40e-14.
The model is the reference-spin common-orbital flag space, not a transverse
or spin-changing approximation. Its Hessian output currently contains
derivative tensors without harmonic-analysis postprocessing. Evidence is
`excited-spin-paths-20261030/flag-final-results.json`, with every raw fit,
independent determinant check, command, test hash and artifact identity.

The same artifact passes the existing full-spin HF public three-case
regression and all 4,044 projected descriptor coefficient fits. Its five
representative fifth-order provider/composition tests pass; original PW92
adds six independent fifth partials (p1.968396..2.007210, minimum
R2 0.99992838). TASK/mTASK add 84 distinct exchange fifth partials and 840
structural zeros, with 21 failed exploratory intervals retained and resolved
at unchanged gates (including fourth-order stencils). The recipes' repeated
six PW92 partials are the same provider, not twelve new independent entries.
TASK/mTASK plus PW92 minimum R2 is 0.99909330. These are smooth-interior local
contracts; they do not establish excited-state molecular acceptance for
those named meta-GGA recipes or resolve the separate TPSSloc oracle issue.

The new correlated-reference driver guard explicitly refuses an unconnected
ROKS/GKS double-hybrid excitation model, rather than substituting a parent
TDA pole. That post-shared-build guard has a separate queued six-case test;
it is not included in the shared artifact acceptance above. Nuclear
derivatives of the existing second-order propagator remain implementation
work; generalized unequal same-spin/opposite-spin scaling additionally needs
an explicitly defined spin model. Existing ground-state PT2 derivatives do
not certify excited correlated poles.

## Post-guard current integration

The next isolated library14 has SHA256
`add409f917bdacdce82c56cf3c4b11ce594839969acdd78f894148edb807132b`.
It starts from the response owner's validated root13 geometry/absolute-residual
repair and adds the explicit correlated-reference model guard and shared-test
import repair. Compilation denies warnings. All six flag-target tests pass:
the three independent spin/Hamiltonian limits, both full 156-component nuclear
cases, and all six ROKS/GKS correlated-request refusals. All 312 numerical fits
again pass without a failed interval. Detailed evidence is
`excited-spin-paths-20261030/flag-complete-post-guard-14/final-results.json`.
The strict all-feature library/binary/test Clippy audit passes root13 plus the
shared-test import repair (13a); it predates the small model-guard branch.

A paired comparison also retains the complete earlier cf6 independent
acceptance: all fifteen HF poles and oscillator strengths are unchanged;
maximum gradient difference is 1.64e-15 Ha/Bohr, and total/excitation Hessian
differences are below 4.14e-14 Ha/Bohr^2 on shared236. This checks every tensor
entry across the interaction/integral refactor, rather than relabeling the old
finite-difference record as a new run. See
`excited-spin-paths-20261030/spinor-hf-paired-artifacts.json`.

## Normalized eigenpair foundation and initially canceled integration

`response/spinor_tda/pole_vector.rs` adds genuine normalized eigenvector
coefficients through order two, needed to differentiate the existing
second-order propagator and its transition residues. The fixed-base overlap
gauge, fresh eigen-equation and normalization coefficients are checked.
Degenerate roots and nonfinite responses are explicit failures, with no
denominator floor. The old pole-energy arithmetic order is preserved through
shared private preparation/energy helpers.

Before promotion, four isolated exact-source prototype tests pass. All 24
real/imaginary eigenvector gradient/Hessian entries pass the first four-step
interval (p1.998239..2.004382, minimum R2 0.99999873). Independently constructed
unitary paths check every root in dimensions 2,3,5,7,15: 96 known energy
coefficients and 936 complex vector coefficients. Complex basis-phase
covariance, degenerate and one-level limits, and overflow diagnostics pass.
Thirty legacy energy coefficients compare bitwise with accepted library14.
Two earlier overflow-diagnostic expectation failures remain recorded; the
final implementation checks nonfinite derived coefficients explicitly.
The permanent test retains independent closed-form/unitary oracles rather
than comparing a function with itself after integration.

At the initial pause, whole-crate/Python/CLI integration of this foundation
was unrun. The escalated EDF2/VWN5/eigenpair launch request was aborted by the user;
no stage, snapshot, compiler or new
artifact was observed. No retry or alternative route was used before the
user explicitly approved resumption after reconnecting. The functional owner's independent EDF2
preflight also retains six finite zero-gradient B88/recipe kernel reference
discrepancies; these are distinct from the earlier TPSSloc failures, and no
oracle or gate has been changed to mask either set.

Evidence: `pole-vector-prototype-20261031-06/final-results.json`,
`excited-extension-20261031-checkpoint.json`, and the functional owner's
`edf2-20261030-checkpoint.json`. The two original LDA molecular preflights
have no terminal molecular derivative acceptance. Read-only process inspection
after reconnecting found neither original process; their logs have no exit
record. The weak-source reference and static spectrum completed before
interruption, but its moving second-order path did not produce a result.
These runs are preserved as interrupted, not relabeled as corrected-VWN5 verification.

Remaining correlated nuclear work must differentiate the same existing
normal-ordered propagator: parent eigenvectors, full noncanonical pair
resolvents, singles kernel, metric, doubles coupling and explicit parent/target
ground energy. This foundation alone is not a correlated excited Hessian.
Generalized unequal SS/OS scaling still requires an explicitly defined spin
model, independently of the existing invariant equal-coefficient case.

## Approved resumption and actual eigenpair acceptance

The user-approved shared build now completes with warnings denied and the
portable source audit passing. The EDF2/VWN5/eigenpair library SHA256 is
`3d8d7d44ae3965fa0f2d6bfb27e5c51584ac3ed9661014ffc1784c4d40caf5a0`.
Its immutable manifest includes the exact three eigenpair source/test files,
the bounded EDF2 registration and original-VWN5 constant correction. Later
SCF recovery and correlated contraction source are excluded from this build.

All three permanent eigenpair tests pass against this actual library:
24 real/imaginary vector gradient and Hessian fits, all roots of the five
known unitary-path models, phase covariance and failure diagnostics. All 24
fits pass the first interval: p1.998239..2.004382, minimum R2 0.99999873.
The six doublet/quartet public excitation, gradient and Hessian cases also
pass through native Python, ASE and canonical CLI under strict units and
warnings-as-errors. Their independent full nuclear fit evidence remains
the earlier HF acceptance; public route agreement alone adds no new
functional-family or molecular derivative certification.

EDF2's six parameter-distinct semilocal components pass all 74 nonzero
symmetric fifth partials at the existing smooth interior point:
p1.971098..2.025567, minimum R2 0.99994040. All 442 defining-equation zeros
are checked exactly. Repeated B88/LYP bodies have distinct part indices and
parameter labels. The first attempt tried to fit finite-difference roundoff
at LYP's exact affine-sigma zeros and failed; that log remains retained.
The corrected test asserts those zeros from the defining affine equation,
and preserves every nonzero fit gate. The weighted recipe versus independent
dense-third composition contract also passes. No original boundary oracle
entry is changed, and these smooth-interior tests do not resolve the
separate EDF2 or TPSSloc zero-gradient reference disagreements.

Evidence: `edf2-integration-20261030/build.json`,
`excited-spin-paths-20261030/pole-vector-actual-edf2/final-results.json`,
`excited-spin-paths-20261030/flag-public-edf2/results.json`,
`excited-spin-paths-20261030/edf2-shared-fifth-02/final-results.json`, and
`excited-spin-paths-20261030/edf2-weighted-third/results.json`.

## Noncanonical correlated-pole contraction

`response/correlated_tda/pole_path.rs` implements analytic coefficients
through second order for the existing propagator contraction
`u†(K - omega M)u + (C†u)†(omega I - D)^-1(C†u)`. It transports the normalized
parent eigenpair and solves the full noncanonical double-space resolvent
coefficient by coefficient. Every coefficient is checked against the fresh
equation; resonances and nonfinite results fail explicitly. No orbital-gap
diagonal approximation, denominator floor or finite difference is used in
production. Inputs are already normalized physical single/double tensors;
this interface does not define a new spin scaling model.

The exact child-source harness passes three tests against the accepted
shared library. A separate closed-form two-by-two eigenvector and adjugate
inverse oracle verifies all 30 real/imaginary double-vector and correction
gradient/Hessian components with four steps. All pass the first interval:
p1.999026..2.000980, minimum R2 0.99999992. Independent complex rotations of
both spaces, resource limits, missing jets and resonance diagnostics pass.
The initial test compile's missing fixed-array annotation is retained.
Evidence: `correlated-pole-path-20261031-03/final-results.json`.

The initial exact-source result alone is not a molecular correlated
excited-state Hessian. At that stage physical molecular jets were still missing; the later molecular
adapter and bounded complete nuclear acceptance are recorded below.
The current contraction assumes a positive-metric normalized double space;
an indefinite signed spin metric needs an explicitly defined adapter.

## Combined actual-library acceptance

The next combined Python/library/CLI artifact contains the completed
EDF2/eigenpair source, the independently accepted SCF Newton-stagnation
recovery and the new correlated-pole contraction. Its library SHA256 is
`bee31520b8a38baa39f49321e3481146ef20370b68fb6bd95f2ed0b007a789dc`.
All sources are frozen in the original combined16 manifest. The first
tool-owned build lost its process and produced no complete artifact or exit
record. Read-only inspection established no remaining compiler before
recovery into a separate stage16b from the unchanged snapshot. Both original
logs and partial objects remain; no numerical failure or success is inferred
from the interruption. The recovered library and canonical CLI compilation
deny warnings and pass the private-root artifact audit.

Actual whole-crate acceptance now passes all three correlated contraction
tests, including the 30 complete gradient/Hessian fits, complex basis
rotations, zero-correlation limit and typed failure diagnostics. All three
eigenpair tests and all six common-orbital HF tests also pass on this same
artifact. The latter repeats the complete 312 doublet/quartet nuclear fits:
p1.963899..2.009326, minimum R2 0.99990620. All 366 fits across these three
targets pass their first interval without a failed numerical fit.

The fixed Python/CLI bundle passes all thirteen high-angular public cases,
including the former i-UHF Newton-stall point through native, ASE atomic and
physical units, and CLI at unchanged 1e-14 SCF gates. The existing full-spin
HF excitation/gradient/Hessian public three-case regression also passes.
These public comparisons check route consistency; the independent physical
reference and full nuclear fit records remain explicitly scoped above.

Evidence: `excited-pole-path-integration-20261031-16b/build.json`,
`excited-pole-path-acceptance-20261031-16b/results.json`, and the three
`excited-spin-paths-20261030/*_contract-actual16b/final-results.json` files.
This historical16b result predates the molecular adapter accepted below.
Complete semilocal spinor molecular tensors remain under acceptance. The full strict Clippy record
on snapshot15 predates this new child; no full-suite or later live-tree
Clippy pass is inferred from warnings-strict compilation.

## Molecular correlated TDA nuclear acceptance

The existing normal-ordered full-Coulomb second-order TDA propagator now has
an analytic molecular adapter through second nuclear order. It transports the
complete Fock and four-index MO integrals, ground amplitudes, singles kernel,
metric, coupling and noncanonical doubles operator before the normalized
parent-eigenpair/resolvent contraction. Ground target energy derivatives use
the complete solved orbital order. Production uses no finite differences,
diagonal-gap replacement or denominator floor. Positive collinear SS/OS
coefficients are supported; this does not define unequal spin scaling for
spin-mixed states.

`response::nuclear::{correlated_excitation_path,correlated_excited_state_hessian}`
and the common `td-state-gradient`/`td-hessian` driver are integrated on
library17b SHA256
`4c3b9019aec02d91203c71f00ae8a14ddb302e446d10dd01a6d084d00bee1fa2`.
The static adapter independently matches all 232 frozen determinant-oracle
kernel/metric/coupling entries (maximum difference 7.81e-18); all three
noncanonical pole contracts pass, including the empty-double-space limit.

Acceptance uses noncoplanar C1 four-H systems with all-electron HF adiabatic
parents: two-electron restricted singlet with SS=OS=0.2 and three-electron
unrestricted spin-conserving space with SS=0.17, OS=0.31. For each model all
four channels (correlated ground energy, parent pole, pole correction, total
excited energy) pass all 12 gradients and 144 Hessian entries: 624 fits each.
Every displaced static reference rebuilds the physical propagator. Complex
cross-geometry AO-metric transition overlaps track the isolated parent state;
minimum overlaps are 0.963903 and 0.966489, minimum parent gaps 0.049660 and
0.236153 Hartree. These are physically defined collinear subsets, not general
exact-spin multiplicity claims.

Restricted acceptance has 622 second-order fits p1.874073..2.045759 and two
fourth-order fits p3.974290, minimum R2 0.99944978. Unrestricted acceptance has
624 second-order fits p1.890022..2.133289, minimum R2 0.99901459. Twelve tiny
unrestricted Hessian entries use differences of fresh first-order analytic
gradients after all 48 first derivatives had independently passed static-energy
fits; no second-order path is called by that supplemental oracle. The other
612 unrestricted entries and all 624 restricted entries use fresh static
energy differences. Original noisy intervals and the coarse-interval AO
state-overlap failure remain recorded. Neither numerical nor tracking gates
were relaxed. The first aggregate harness exited unsuccessfully on the
unrestricted case after the restricted case passed; the final separate
unrestricted harness exits successfully.

All four public gradient/Hessian cases (restricted/unrestricted, native/ASE/
canonical CLI) pass with strict units and warnings as errors. Largest observed
tensor route difference is 3.36e-14 atomic units. Gradients/forces have opposite
signs, Hessians are Hartree/Bohr^2, and individual ground/pole/correction tensors
are reported separately. Evidence:
`molecular-correlated-excited-20261031-final-results.json`, with exact commands,
artifact/test hashes, every fitted component, failed intervals and raw logs.
Tests: `tests/collinear_transition_path_contract.rs`,
`tests/correlated_pole_path_contract.rs`,
`tests/correlated_excited_nuclear.rs`,
`python/tests/test_correlated_excited_paths.py`.

This closes molecular correlated HF-parent nuclear integration. Semilocal,
range-separated, ECP and named double-hybrid molecular correlated combinations
still require representative acceptance. Correlated ROKS/GKS, indefinite or
negative double-space metrics, spin-changing sectors and full-spin RPA remain
unconnected. Complete semilocal spinor molecular tensors are under independent
acceptance on combined19, not inferred from local jet tests.


## Family-wide molecular spinor acceptance in progress

The requested scope includes LDA, GGA, meta-GGA, global/range-separated
hybrids, correlated double hybrids and the learned provider. Permanent
`tests/semilocal_spinor_excited_acceptance.rs` cases prepare complete 312-entry
pole/absolute-energy gradient/Hessian fits for SVWN5, PBE, TPSS, PBE0 and
CAM-B3LYP. The fixed spatial grid is deliberately small and unpruned, with
zero weight cutoff; this tests derivatives, not grid accuracy. All references,
angular convergence gates, source terms and cross-geometry state tracking are
kept. The immutable19 LDA run reached its 5,400-second resource budget before
the analytic tensor completed; it has no accepted nuclear components. The
timeout and original log are retained in
`spinor-active-acceptance-20261031-19/results.json`. Remaining family
executions are pending. Preparing or compiling a case is not its acceptance.

Combined19 contains the completed17b collinear correlated adapter, active
4/16/32 directional descriptor seeds and the approved ECP reuse source. Its
warnings-strict Python/library/CLI build and portable audit pass. Actual19
passes all 4,044 descriptor coefficient fits and all thirteen high-angular
public cases. Full Clippy across all features/library/binaries/tests passes
on19a, which differs only in explicit Result handling in one overflow test;
the initial19 test lint failure is retained. Evidence:
`spinor-active-integration-20261031-19/build.json`,
`spinor-active-audit-fixed-20261031-19a/clippy-results.json`.
The later static-point seed specialization is independently paired against19:
594 coefficients compare exact bits and all 4,044 coefficient fits pass.
It is included in immutable21, whose warnings-strict library, Python and CLI
build and portable audit pass. The actual21 descriptor and isolated-eigenpair
contracts pass; a PBE 48-point reference and all twelve poles also pass a
preflight. This does not certify nuclear derivatives.

The complete standard B95 fifth tensor passes 66 nonzero fits and 396 exact
raw-tau defining-equation zeros at one smooth interior descriptor point.
Companion B88 has 12 fits and 114 defining zeros. All 78 fits pass the first
interval, p1.971387..2.025567, minimum R2 0.99994181. The actual weighted
DSD-PBEB95 semilocal composition also matches independent dense-third
contractions. These are local/provider contracts and do not certify a
spinor correlated molecular Hessian. Evidence:
`excited-spin-paths-20261030/b95-fifth-actual19/final-results.json`.

The user explicitly selected global spin projection for the learned spinor
extension. `skala::spinor::spinor_skala_grid` evaluates the complete nonlocal
network at one shared spin direction across every grid point, then averages
`E + D_spin E`. Its weighted reverse potentials include nonlocal grid-cross
terms and the density dependence of the spin-scaling seed. Existing raw tau
is retained; electromagnetic gauge/current extensions are not silently added.
Synthetic network tests now pass the original collinear limit,
global spin-rotation covariance and every active density derivative and
transverse kernel: 40 gradients plus all 1,600 kernel entries. Of these,
1,561 use ordinary native finite-endpoint subtraction; the remaining 79 tiny
cross-grid entries use stable divided-difference arithmetic for the same two
finite evaluations. Every endpoint is independently checked against ordinary
native evaluation; the supplementary oracle does not use infinitesimal AD.
All original 2,109 failed interval fits are retained. Accepted exponents are
p1.858090..2.191051, minimum R2 0.99902156, at unchanged gates. Evidence:
`skala-spinor-projection-20261031-05/final-results.json`. The first unused
harness import/metadata compile failures are also retained. This is an exact
source foundation result, also reproduced on the compiled21 library. SCF, moving AO/grid,
molecular excitation and public-route connections remain work. A deterministic random network validates implementation, not physical
accuracy of a fitted learned model. No new model weights or changed reference
fixtures are distributed.

On November 1, the static projected-kernel chain rule passes all 1,348
independent point-component fits: LDA 4+16, GGA 16+256 and meta-GGA 32+1,024
gradient/kernel entries. Complete column comparisons against the original
Taylor implementation at angular rules 4 and 8 differ by at most
6.67e-16 in potentials and 4.45e-16 in kernels. One fine-step TPSS fit was
dominated by cancellation; its failed intervals remain recorded, and an
additional four-step fourth-order interval resolves it at unchanged gates.
Cutoff, sigma and kinetic clipping boundaries, unavailable dense derivatives
and nonfinite intermediate jets retain the existing Taylor fallback.
Evidence: `native-spinor-kernel-contract-20261101-01` and `-02`.

The moving transition-kernel cache separately passes all 120 independent
matrix-coefficient fits, including moving AO values, weights and the complete
second-order density path. Cached and direct results differ by at most
1.74e-18; repeated application is bitwise identical. Both supported angular
rules, explicit unsupported-rule errors and the memory-budget fallback are
tested. Evidence: `spinor-grid-mixed-cache-20261101-01/results.json`.
These are exact-source tests against the immutable21 provider library;
whole-crate integration of both cache changes and new molecular 312-entry
acceptance remain pending. The immutable22 HF regression passes all fifteen
poles and 156 nuclear derivative fits; it does not establish semilocal,
correlated full-spin or learned molecular acceptance.
