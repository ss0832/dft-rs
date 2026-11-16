<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 ss0832 -->

# Validation matrix: models, references and independent evidence

All ECP, GGA, meta-GGA, hybrid, double-hybrid, ROKS and GKS families are
verification targets. A pass in one row does not fill its Cartesian product
with other rows. This summary distinguishes directly measured, owner-reported
fixed-artifact and historical documented evidence. Detailed component errors,
thresholds and source/artifact identities remain in the named acceptance
records. See the [implementation status](development-status.md) for the
current integrated source checkpoint.

- [Functional and reference evidence](#current-bounded-family-evidence)
- [Next molecular tests with d orbitals](#next-molecular-tests-with-d-orbitals)
- [ECP and nuclear derivatives](#ecpnuclear-intersection-evidence)
- [Excitation integration](#excitation-integration-evidence)
- [Later source integration](#later-source-integration-checkpoint)
- [Target-specific acceptance](#target-specific-acceptance)
  - [Integral-direct molecular gradient (T139)](#t139-integral-direct-molecular-gradient)
  - [Dynamic Raman mode derivatives (T017)](#t017-dynamic-raman-mode-derivatives)
  - [Response-save consistency](#existing-response-save-consistency-check)
  - [Isolated singlet TDA optimization (T120)](#t120-isolated-singlet-tda-optimization)

## Current bounded family evidence

| Family / reference | Energy and derivative evidence | Response / limits | Remaining combinations |
|---|---|---|---|
| GGA: restricted PBE | T139 HF/PBE STO-3G water/C1 ammonia: every gradient entry, 42 total fits, stored/direct and public-route comparisons pass on artifact `a1c6cb7f...` | ECP/PBE and ordinary response have separate rows below | Other bases/references and nonlinear properties are not inferred |
| Meta-GGA | PKZB: 53 first/second local fits, 104 complete local third partials, 12 C1 RKS gradient fits and four bounded standard molecular energies accepted. TASK/mTASK: owner-reported 107 local/parameter fits, 24 C1 RKS gradient fits and four public cases pass shared `236be050...` | Complete smooth-interior fifth contracts cover TPSS/PKZB and TASK/mTASK exchange; moving spinor descriptor paths pass 4,044 coefficients | Point/high-order contracts leave molecular excited Hessians and ECP/reference intersections open. TPSSloc retains the four original zero-gradient failures and records the later corrected checks |
| New global GGA hybrids: BLYP35/WC04/WP04 | Artifact `a1c6cb7f...`: 69 nonzero local density-potential/kernel fits plus 36 complete neutral-RKS C1 gradients; fixed-density RKS/UKS term references; 14/15 RKS five-molecule cases pass | Six native/common/ASE/CLI cases pass, including UKS route consistency; registry/recipe checks pass | UKS routes are not independent UKS force fits. Named-model Hessian/APT, alpha/beta/Raman/magnetic/high-order/periodic, ECP, ROKS and GKS remain unrun here |
| Double-hybrid/generalized PT2 | Separate named energy/SS/OS oracle suites remain in their ledger. New full-spin equal-SS/OS HF-parent model uses dense physical pair and complex-AO transform references | Owner-reported p296 41 fits, p297 18 static fits and nine spectral comparisons at three frequencies, p298 108 full two-site FC parent/PT2/total field-Hessian fits pass | No inference to ECP, unequal-SCS, distinct targets, ROKS PT2, constrained high orders or complete SD/PSO/DSO assembly |
| ROKS / ROHF | HF reference-spin flag TDA doublet/quartet: all 24 gradients and 288 Hessian entries pass independent C1 determinant-Hamiltonian acceptance on shared `236be050...` | Six native/ASE/CLI spectrum/gradient/Hessian cases pass; existing p297 and ECP/HF nuclear evidence remain separate | Spin-changing flag sectors, correlated pole derivatives, semilocal molecular excitation acceptance and ECP+functional combinations remain distinct gaps |
| GKS / GHF | Isolated full-spin HF TDA: all 15 poles and 156 gradient/Hessian fits accepted on cf6; complete paired tensors agree with shared `236be050...` within 4.14e-14 Ha/Bohr^2 | Full-spin public three-case regression passes on shared artifact; compact ECP/HF nuclear and equal-coefficient PT2 retain their own records | Semilocal moving-kernel molecular tests remain unfinished; no moving SOC, unequal-SCS correlated pole model, full-spin RPA or ECP+all-functionals inference |
| Periodic range-separated hybrids | RKS omegaB97X-V NaH with scalar ECP passes all 15 force and ordered strain fits at finite k. The 1D RKS omegaB97X-V wire and 2D UKS omegaB97M-V slab pass 18/18 each, with minimum R² of 0.99942343 and 0.99967865. | CLI, native Python and ASE agree for both lower-dimensional fixtures. The omegaB97X-D NaH polar case passes the Gamma Hessian, Born-sum and directional LO-TO checks and has an imaginary optical mode. | Grid refinement, stable relaxed-ion SOEC and broader reference/provider coverage remain open. [VV10](response.md#first-geometry-derivatives-of-variational-periodic-vv10) [CHG](response.md#dispersion-curvature-acceptance) |

The [catalogue sweep](response.md#molecular-response-across-the-registered-functional-catalogue)
enumerates named recipes for all four molecular references and targets
Cartesian gradients and static polarizability. The test harness has
passed Clippy. Numerical acceptance is pending.

The hybrid worker reports BLYP35 and WP04 all five RKS molecular energies pass.
WC04 passes four; its ferrocene SCF is nonstationary and the final attempted
run has no recorded successful exit. Its disappearance is not convergence
or evidence of a separate lower root; the execution cause remains unconfirmed.
All fifteen independent
reference energies were generated, but a lower unconverged energy is not a
validated root. Bounded alternative reference guesses also failed their
unchanged criteria. No reference, gate or scientific definition is substituted.

PKZB is compiled and its bounded provider/RKS/UKS/public and four standard
molecular checks pass; its ferrocene independent reference hit its wall budget,
so the fifth comparison is unrun. TPSSloc has accepted local/SCF/C1/public
subsets. The four original zero-gradient FXC failures are preserved. A later
[reference correction](FUNCTIONALS.md#tpssloc-approved-reference-application-2026-10-31)
passes the affected comparisons, and all five standard molecular energies
subsequently pass. TASK/mTASK are compiled with the explicit
original PW92 convention; their owner reports all 24 C1 gradient fits passing.
No reference or gate is changed to turn a failed comparison into a pass.
None of these provider subsets certifies ECP+meta-GGA or every response model.

T120 has a separate isolated restricted-HF/STO-3G singlet-TDA optimization
acceptance: 24 full Cartesian gradient fits, dense/AO-metric state references,
internal curvature and native/ASE/CLI endpoints. This validates neither other
functional optimizers nor crossings; see [the exact scope](#t120-isolated-singlet-tda-optimization).

The subsequent requested spin/Hessian extension has a separate
[architecture and pending-work record](excited-state-extension.md).
The HF flag and full-spin subsets below have actual rebuilt-artifact evidence;
new semilocal and correlated combinations retain explicit unfinished status.

## Next molecular tests with d orbitals

The next molecular validation priority is a real molecule with d-type
Gaussian basis functions in the full relaxed calculation. Start with C1
distortions of neutral ammonia and its doublet cation in the bundled
def2-SVP basis, which has a nitrogen d polarization shell. Confirm that
this shell contributes to the molecular calculation. Use neutral RKS
and open-shell UKS, ROKS and GKS, with omegaB97X-D and omegaB97X-V as
the required functional cases.

Check analytic gradients, Cartesian Hessians, and third- and
fourth-order nuclear force constants with zero grid weight cutoff.
Use independent four-step fits for every component or a complete
full-rank set of directions. Record absolute errors, raw convergence
residuals, failed intervals and representative common-driver, CLI,
native Python and ASE agreement. A separate transition-metal fixture
with occupied valence d character extends the scope from basis
polarization to d-electron chemistry. These cases are planned.
The previous compact s/p/d/f numerical models retain their own
[acceptance record](response.md#priority-provider-connections-and-c1-spdf-acceptance-2026-10-29).

## ECP/nuclear intersection evidence

The following owner-reported `p290_spdf_nuclear_components` cases use a C1
20-AO s/p/d/f model with 12 nuclear coordinates and the older immutable
library prefix `fead0f84`. Independent fresh displaced analytic gradients
and Hessians, at least four steps and complete unisolvent sets are used;
all failed intervals remain recorded. These derivative checks do not by
themselves establish a new absolute-energy oracle.

| Core / reference / functional | Gradient + Hessian | Complete cubic | Complete quartic |
|---|---|---|---|
| All electron RHF/HF | Passed | 364 passed | 1365 passed |
| All electron RKS/PBE | Passed | 364 passed | Earlier 1364/1365 checkpoint. All 1365 pass in the later response record |
| Scalar nonlocal ECP RHF/HF | Passed | 364 passed | 1365 passed |
| Scalar nonlocal ECP RKS/PBE | Passed | 364 passed | 187/1365 checkpoint; partial |
| Scalar nonlocal ECP UHF/HF | Passed | 364 passed | Unrun at this checkpoint. All 1365 pass on a later fixed artifact |
| Scalar nonlocal ECP UKS/PBE | Passed | 364 passed | Queued/unrun |

The [later quartic record](response.md#complete-c1-spdf-cubic-and-quartic-results-2026-10-30)
gives the completed all-electron PBE and scalar-ECP UHF results with their
source and fit details. The two incomplete ECP DFT quartic suites retain
their separate status.

All six gradient/Hessian cases comprise 936 complete component fits.
Separately, p293 compact five-AO C1 ECP/HF ROHF and GHF each pass 156
gradient/Hessian fits, 364 cubics and 1365 quartics on fixed snapshot03;
eleven native/ASE/CLI cases pass there. This is not full 20-AO ROKS/GKS
or ECP+other-functional acceptance. New ECP+meta-GGA/hybrid/double-hybrid
acceptance is unrun in the reported tranche. Higher g/h/i AO acceptance is
in progress and remains separate; parser/transform support and an angular
cap are not tests. The response owner separately reports 156 gradient/Hessian
fits for each all-electron HF g/h/i case and 156 for the ECP g case. These
were the earlier lower-order subsets. Subsequently the response owner reports
all twelve g/h/i, AE/ECP, RHF/UHF gradient/Hessian cases (1,872 fits), complete
g/h AE RHF cubics (364 each) and complete g AE RHF quartic (1,365) passing
artifact11. The original g-cubic memory failure is retained and resolved by
the local-Taylor integral workspace, not by reducing numerical coverage.
An earlier overflowing/asymmetric i/ECP reference remains rejected; the later
independently quadrature-built oracle passes 776 ECP-matrix comparisons and
twelve HF term cases. These are distinct reference records. Root snapshot13
additionally matches 13,825 one-electron coefficients exactly and passes
2,144 independent kernel fits; its remaining physical checkpoints and strict
audit have their own terminal statuses and are not inferred from compilation.

The final direct-map ECP memoization artifact08 passes 228 kernel fits and
312 p293 gradient/Hessian fits. Its source identity differs from the earlier
complete quartic artifact, so the old quartic result is not silently relabelled
as a complete rerun of artifact08. Full Clippy/unit acceptance on snapshot05
and changed-target Clippy on snapshot07 are also distinct from numerical
coverage of every later implementation.

## Ownership and reuse rules

The functional-catalogue worker owns named-provider/recipe and molecular
energy checks. The response/ECP worker owns ECP recurrences, high-order nuclear
and ROKS/GKS/PT2 response acceptance. The independent validation worker owns
the Hermite/matrix contracts and T017/T139/T120 subsets. Existing healthy
jobs are reused through fixed artifacts and are not duplicated.

Tests are selected using [implementation contracts](conventions.md#verification-by-implementation-contract).
Per-functional local derivatives and registration/recipe checks remain, while
unchanged response/property mechanisms use explicit representative integration
coverage. “Documented”, “passed”, “partial”, “running” and “unrun” are distinct
states; neither a family label nor the list of available names supplies missing
cross-product evidence.

## Excitation integration evidence

The common-orbital doublet/quartet 312 component fits all pass their first
four-step interval, with minimum R2 0.99990620 and 0.99999484 respectively.
All eighteen native/ASE/CLI comparisons of six cases pass. The corresponding
full-spin HF three-case public regression also passes, and every paired
gradient/total-Hessian/excitation-Hessian entry agrees with the earlier
independently accepted cf6 calculation (maximum 4.14e-14 in atomic units).
See [exact models, units and residuals](excited-state-extension.md#common-orbital-hf-acceptance).
All 4,044 moving projected descriptor coefficients pass on that same shared
artifact, while the 120 moving AO/grid composition coefficients have their
separate exact-source evidence. Neither is molecular semilocal pole acceptance.

The correlated ROKS/GKS driver guard passes all six explicit rejection cases on library14; the same artifact repeats all 312 flag nuclear fits and three spin/Hamiltonian limits successfully. The later collinear molecular nuclear acceptance is recorded below; correlated
ROKS/GKS derivatives remain implementation work. A bare semilocal parent pole
must not be reported as the requested correlated double-hybrid model.

## Later source integration checkpoint

Following explicit user approval after reconnecting, the shared EDF2/VWN5
Python/library/CLI build passes with warnings denied and portable audit.
Library `3d8d7d44...` contains the normalized eigenpair foundation; all three
permanent tests pass on it, including every one of the 24 vector derivative
fits. The six HF doublet/quartet public cases also pass that new bundle.
All 74 nonzero fifth partials of the six parameter-distinct EDF2 semilocal
parts and 442 exact defining-equation zeros pass at one smooth interior point;
the original boundary-reference failures remain separate and unchanged.

The new noncanonical correlated-pole contraction now has actual whole-crate
acceptance for all 30 correction/double-vector derivative components on
combined16b (`bee31520...`), alongside all 24 eigenvector fits and all 312
common-orbital HF doublet/quartet nuclear fits. That Python/CLI bundle also
passes thirteen high-angular public cases, including strict Newton-stall
recovery, and three full-spin HF excited public cases. A molecular adapter supplying complete
physical propagator tensor jets is still missing; this is not a molecular
correlated excited Hessian. Original LDA molecular preflight processes were
absent after reconnecting without terminal results, so their status is
interrupted/unverified. Existing accepted cells remain tied to exact artifacts.
See [the resumed integration and remaining work](excited-state-extension.md#approved-resumption-and-actual-eigenpair-acceptance).


Combined17b now passes all 1,248 molecular correlated derivative components
across the restricted/unrestricted HF-parent positive-SS/OS TDA subsets and
four native/ASE/CLI cases. Twelve unrestricted Hessian components use fresh
first-order gradients after independent energy-oracle acceptance of every
first derivative; all other components use fresh static energies. Raw failed
intervals and the earlier aggregate failure are retained. This is not named
semilocal/DH, ECP or ROKS/GKS cross-product acceptance. See
[exact model, methods and residuals](excited-state-extension.md#molecular-correlated-tda-nuclear-acceptance).


The expanded molecular spinor family request is tracked in the existing
[family-wide acceptance section](excited-state-extension.md#family-wide-molecular-spinor-acceptance-in-progress).
The 312-component LDA run hit its resource limit before the analytic tensor
completed. GGA, meta-GGA and global/range-hybrid cases are prepared and
acceptance is pending. The learned global-projection foundation has separately
accepted all 1,640 density/kernel fits and collinear/rotation invariants;
79 tiny entries use independently checked stable finite-endpoint arithmetic,
with every original failed subtraction retained. It has no molecular acceptance. Spinor correlated DH remains separate from the proven
collinear HF-parent1248 milestone. Combined19 strict all-feature Clippy passes
on the test-only corrected19a; no later live-source pass is inferred.

## Target-specific acceptance
This record reconciles specific implemented-but-unvalidated catalogue entries
with measured acceptance. A passing subset does not certify every functional,
reference, perturbation or periodic combination of its aggregate ledger row.
Production finite differences are not introduced.

### Fixed source and execution scope

The library is the immutable Python-feature build used for the Hermite
integration checks in [performance.md](performance.md). Source-manifest
SHA-256 is `201f42f0c24e79c6062be8ea5c637eaca1a93bf166f6c11f92ad857be36de721`;
the linked library SHA-256 is
`a1c6cb7f342cff4a3f37937563934fc362d6da9065af468501eefd773d473e92`.
Rust tests are compiled with warnings denied and use two Rayon threads,
sequential faer and `DFT_RS_STRICT_UNITS=1`. Python tests use `-W error` and
the same freshly built extension/canonical CLI. No additional heavy build
or timing comparison is part of this acceptance.

The production Hermite, matrix and functional kernels match that snapshot.
Later common-orbital/generalized PT2 and nuclear-response edits belong to a
separate acceptance effort; these checks do not certify those later changes.
Source/test hashes, every error sample, compiler command and complete output
are retained with the fixed validation artifacts outside the package.

### T139: integral-direct molecular gradient

The existing `p226_excited_optimization_and_streamed_gradient` streamed-gradient
test formerly fitted only one Cartesian component. It now retains the water
case and adds a genuinely C1 four-atom ammonia geometry. HF and PBE/STO-3G
are checked at zero temperature with restricted integer occupations, no
extra corrections, `grid_weight_cutoff=0`, energy tolerance `1e-13`, density
tolerance `1e-12` and orbital-gradient tolerance `1e-11`.

All 42 Cartesian gradient entries (9 water and 12 ammonia, two functionals)
pass independent central energy-difference fits at
`[0.02,0.01,0.005,0.0025]` Bohr. Exponents are `1.997161..2.006627`, minimum
R2 is `0.99999558`. Every measured error is positive and printed; no interval
or sample is dropped or floored. All entries pass the original stricter
`abs(p-2)<0.05`, `R2>0.9999` bounds on the first interval; additional staged
intervals and a fourth-order stencil are available without relaxing those
original bounds.

Streamed/stored gradient differences are at most `1.124e-13` Hartree/Bohr;
energy differences at most `7.105e-14` Hartree; translational residuals at
most `3.582e-15` Hartree/Bohr. All reports state `integral_storage=none` for
the streamed path. Native Python, the generic feature driver, ASE atomic
reports and the canonical CLI agree for both ammonia cases; maximum CLI
gradient discrepancy is `3.108624468950e-15` Hartree/Bohr and energy
discrepancy `4.263256414561e-14` Hartree. Explicit stored/native comparison
retains the same physical thresholds as the Rust test.

Only the streamed-gradient test is executed from p226; its excited-state
optimization test and T120 acceptance remain separate. Other functional
classes, references, periodic gradients and additional model terms are not
validated by these HF/PBE restricted checks.

### T017: dynamic Raman mode derivatives

The existing distorted-water HF/STO-3G acceptance passes all 27 mode/tensor
entries (three vibrational modes, full 3x3 polarizability derivative per
mode) at photon energy `0.1` Hartree, with independent displaced-molecule
spinor polarizabilities at zero spin-orbit coupling as the reference.
Every component passes the first interval `[0.08,0.04,0.02,0.01]` Bohr,
with `p=1.995572..2.007256`, minimum R2 `0.99999103`. The small-photon
`0.001`-Hartree/static-activity relative difference is `6.071e-6`, within the
unchanged `1e-3` bound.

A genuinely C1 four-atom ammonia case additionally passes all 54 entries
(six vibrational modes, all nine tensor entries). The complete current p124
suite passes three tests and all 81 water/ammonia component fits, with
`p=1.978644..2.063736`, minimum R2 `0.99979176`. Every component passes the
first interval at the unchanged shared bounds `abs(p-2)<0.2`, `R2>0.999`;
no finer/coarser fallback was required. Both geometries retain every grid
point, and no nonzero component is replaced by a contraction or omitted.
Native Python/ASE/canonical CLI checks on water pass with positive dynamic
convergence flags. A normal-mode sign is aligned through its displacement
vector before comparing its tensor derivative; every tensor entry and mode
remains in the comparison. Maximum CLI tensor discrepancy is `1.8638e-11`
Angstrom^2/sqrt(amu), and the largest dynamic response residual is
`9.490918890968e-11`. Activities are Angstrom^4/amu; photon energies are
Hartree. Broader functional/reference and periodic dynamic Raman acceptance
remains separate.

### Existing response-save consistency check

`p126_dynamic_response_save` passes both strict-unit report tests: at zero
frequency, the dynamic/static rotation defect is `6.799e-15`, other complex
amplitude parts are exactly zero, and polarizability defect is `3.109e-15`.
The polarizability trace is `7.17687659` statically and `7.30261485` at
`0.1` Hartree, below the first pole. This confirms an already documented
subset; it does not validate the pending periodic response-save model.

### T120: isolated singlet TDA optimization

This acceptance is restricted HF/STO-3G, all-electron, neutral closed-shell
formaldehyde, zero temperature, with a genuinely C1 initial geometry in a
generic laboratory orientation. The target is the isolated first singlet
Tamm-Dancoff state. No solvent, electric field, density fitting, nonlocal
correction or PT2 contribution is included. The existing p226 example passes
its original trajectory, gradient and two-component checks, but that example
alone is not the complete-component evidence below.

`excited_optimization_validation` independently solves the small TDA matrix
densely. At the initial and optimized points its four excitation energies
also agree with forced Davidson solutions to at most `3.247402347029e-15`
Hartree. The scalar reference energy is the independently converged HF energy
plus its dense first TDA eigenvalue. Across displaced geometries, states are
matched through the AO transition density `D=C_v X C_o^T`, using
`Tr[D_ref^T S_ref,point D_point S_ref,point^T]` and both self norms. This
reference is invariant to a state sign or a compensated occupied/virtual
orbital representation change; it does not copy the optimizer's squared-MO-
amplitude heuristic. The best match must remain the first energy-ordered
root, exceed `0.9`, and exceed every alternative by more than `0.5`.

All 24 Cartesian gradient entries at the C1 start and optimized endpoint pass
independent four-step dense-energy fits at `[0.02,0.01,0.005,0.0025]` Bohr:
`p=1.99762408..2.00150071`, minimum R2 `0.9999996846`. The original stricter
`abs(p-2)<0.05` and `R2>0.9999` gates are retained; all components pass the
first interval. Grid-weight cutoff is zero, SCF energy/density/orbital-gradient
tolerances are `1e-13`/`1e-12`/`1e-11`, and TDA tolerance is `1e-11`.
No errors are zeroed, floored or discarded. Minimum measured AO state overlap
is `0.9994140105759`, and minimum first/second-root gap is
`0.1590756627286` Hartree over the checked displaced systems.

Optimization takes 18 iterations, decreases energy from
`-112.205909320625` to `-112.214955267664` Hartree, and reaches maximum
gradient `6.617138775443e-6` Hartree/Bohr, below the unchanged `1e-5` target.
The trajectory is non-increasing, and the separate fresh excited-state
gradient report reproduces the endpoint energy/gradient within the original
`1e-9` Hartree / `1e-8` Hartree/Bohr bounds. Native Python, ASE atomic reports
and canonical CLI each converge in 18 steps to that same energy. Their fresh
endpoint gradient-report defects are zero at printed precision; interatomic
distances agree within `1e-5` Bohr.

Minimum character is checked in all internal directions, not only two
Cartesian diagonal curvatures. All Cartesian gradient differences are formed
at four steps; translation/rotation vectors define an independently
orthonormalized six-dimensional internal subspace. Every projected curvature
matrix has six positive eigenvalues. Its smallest eigenvalue changes from
`0.0522608673093` at `h=0.02` to `0.0522906140310` Hartree/Bohr^2 at
`h=0.0025`, with a positive margin over the observed final refinement change.
The unsymmetrized finite-difference Hessian discrepancy decreases from
`1.6312e-4` to `2.5605e-6` Hartree/Bohr^2. These finite differences are
validation-only curvature evidence; they do not validate the separate
production analytic excited-state Hessian feature.

The optimizer's general tracking remains a squared-configuration-amplitude
heuristic. Per-step AO-transported identity/gap diagnostics and crossings,
near-degenerate or higher states, other initial basins, LDA/learned/other
functional or reference paths, extra terms and periodic models are not
certified by this isolated HF case. In this frozen acceptance artifact the
optimizer invokes the native TDA-gradient backend, distinct from the generic
`td-state-gradient` path route. The subsequent explicit path connection is
tracked in [excited-state-extension.md](excited-state-extension.md); the
frozen evidence does not validate that newer source automatically.
