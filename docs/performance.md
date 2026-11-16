<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 ss0832 -->

# Numerical kernel performance and validation

Use `python tools/bench.py --baseline-exe OLD --exe NEW --save comparison.json`
for five runs per executable and case. The two executables alternate under
the same explicit CPU and memory settings. The report retains each trial,
including failed calculations, and gives the median and median absolute
deviation of wall time, peak process RSS and peak managed memory. Managed
memory covers the resource ledger. Peak RSS covers the executable process.
The periodic high-order cases compare every FC3/FC4 or clamped- and
relaxed-ion elastic tensor returned by the chosen driver. The comparison
also includes the first- and second-order internal relaxations and any
separately reported dispersion derivatives. The charged Li+ reference
cases are available as `--only high-order/periodic`.

For one atom per primitive cell, the TOEC and FOEC drivers skip the
internal-relaxation response paths. Repeated motion of that atom is a
uniform translation, and the translation-free internal space has dimension
zero. The reports retain zero first- and second-order relaxation tensors,
as appropriate for the requested order. Both ion conventions share the
same elastic tensor. A matched before/after timing is pending.

The timing harness also completed five alternating PBE0/6-31G water
pairs at one CPU thread and 1024 MiB using the same executable for both
labels. All ten trials returned -76.3010134896547 Hartree, eight SCF
iterations and a last logged residual of 5.363e-9 Hartree. The two
timing medians were 0.293152 and 0.294812 s, with median absolute
deviations of 0.010553 and 0.008936 s. Both RSS medians were 26980352
bytes and both managed peaks were 595992 bytes. This measures the
run-to-run variation of the harness on this input.

The molecular SCF now lends the current density and Fock slices to the
DIIS histories and allocates the diagonalization fallback only when
extrapolation provides no Fock. In five alternating HF/STO-3G water
trials per build at one worker and 1024 MiB, the median managed peak
changed from 158208 to 157032 bytes. Median RSS changed from 9338880 to
8736768 bytes. Both builds took seven iterations with a last logged
residual of 6.77e-12 Hartree in every trial. Median wall times were
0.039812 and 0.040308 seconds with median absolute deviations of
0.004389 and 0.000526 seconds. These concurrent-validation measurements
establish a bounded memory comparison. The wall-time difference is
within the observed run variation.

For PBE0/6-31G water at grid level one, the same five-pair protocol
measured median times of 0.196782 and 0.187271 seconds with median
absolute deviations of 0.002576 and 0.009409 seconds. Median managed
peaks were 597344 bytes in both builds. Median RSS was 11296768 and
11309056 bytes. Both builds took eight SCF iterations and ended at the
same logged residual of 5.363e-9 Hartree. The timing intervals overlap
at the scale of their observed variation.

## Closed-shell molecular GKS

For the dense static spinor XC descriptor kernel, an exactly z-collinear
base makes the projected charge and spin variables independent of
azimuth. Each of the two descriptor derivatives contributes at most one
transverse trigonometric factor. Four azimuth samples therefore integrate
the kernel's degree-two Fourier polynomial. The polar rule and convergence
settings are preserved. A nonzero transverse descriptor uses the full rule.

At polar order 12, the reduced and full descriptor Hessians, potential
vectors, and energy agree within 8.105e-15 in absolute atomic units for
SVWN5, PBE, M06-L, M06, M06-2X, and M06-SX at a smooth collinear point.
The kernel comparison also tests nonzero transverse descriptors as small
as 1e-30. For M06 at polar order 20, five alternating local timings
give median wall times of 5.397304 s for 40 azimuths and 0.526395 s
for four azimuths. The ranges are 3.973557 to 6.627288 s and 0.343792 to
0.664006 s. This comparison covers the dense single-point kernel.
The CLI pair case `--only response/azimuth/gks/m06` measures the molecular
calculation, including all nine response components, raw residuals and
both process RSS and peak managed memory.

Exact repeated spin blocks use a single spatial diagonalization. The
spinor XC evaluation also reuses the density descriptors across angular
refinements. Distorted water/STO-3G calculations at main grid level 2
used one thread and a 1024 MiB managed-memory budget. Five alternating
pairs per functional used the same CPU affinity and build settings.
All retained observables passed with absolute and relative comparison
tolerances of 1e-8 in reported units.

| Functional | Baseline/candidate wall time (s) | Baseline/candidate time MAD (s) | Baseline/candidate peak RSS (bytes) | Timing ratio |
|---|---:|---:|---:|---:|
| omegaB97X-D | 166.81750 / 0.96531 | 1.98502 / 0.03303 | 48144384 / 67334144 | 172.81 |
| omegaB97X-V | 165.06852 / 1.34352 | 5.52578 / 0.05867 | 48373760 / 64946176 | 122.86 |
| omegaB97M-V | 428.19566 / 1.69145 | 34.95737 / 0.04202 | 49057792 / 64978944 | 253.15 |

The median managed peak is 14775936 bytes in every case. Both builds
take two SCF iterations. In table order, the candidate energies are
-75.29899432715958, -75.29140227665927 and -75.30271770483498 Hartree,
with orbital residuals 2.482968e-11, 2.670497e-12 and 5.444023e-12 Hartree.
Peak RSS rises by 18.30, 15.80 and 15.18 MiB, respectively. The benchmark compares
the immediately preceding source with the repeated-block and
angular-reuse candidate. Other functional and open-shell cases have
separate acceptance.

## Scoped resource-control acceptance

The earlier frozen v0.3.0 integration `resources-v030-integration-20261031-06`
passes all 300 selected Python tests (44.42s, warnings as errors), 24 linked
Rust resource/cache/linear-algebra tests, all-feature/default-feature library
and whole-unit Clippy checks, API/ledger checks and Python lint. The original
eight direct-core Python failures and three Rust failures now pass against
the new library. The actual repository-owned release launcher also passes
the enhanced 21 resource contracts with each CLI name in `--python-only`
diagnostic mode; its full-release gates remain separate. Both CLI names
report the resolved CPU/memory settings and the same molecular energy.

The final opt1/codegen256 library build takes 963.91s on one assigned CPU,
with sampled compiler peak RSS 3717468 KiB. Library, Python package,
registry and both CLI versions agree on 0.3.0; package metadata retain the
existing author. Artifact identities and all original failures are retained
separately. The response body remains the exact frozen bundle33, excluding
later response/model changes; this API acceptance does not promote their
separate scientific gates.

A subsequent Python composition check finds that the newer saved-model ASE
getter reapplies constructor resource defaults inside the common envelope,
overriding explicit per-call limits. Its original nested-reconfiguration error
is reproduced with the real resource06 bridge and mocked model dispatch.
Removing only that redundant forwarding makes the new override regression
pass; stored calculator settings remain unchanged. The captured Python-layer
suite passes 300 cases and detects one stale generated source-link reference;
regenerating the existing API block and rerunning that failed check passes.
These Python-only results are separate from the immutable resource06 package
and from later continuous-model scientific acceptance.

A later fixed-source integration combines those resource controls with the
accepted saved-fragment/core13, SCF-recovery and retained-response changes.
Actual Rust 1.85.0 builds its PEP 517 wheel, library, three CLI executables and
eight integration-test harnesses with warnings denied (opt1, codegen256,
no LTO). The wheel installs in an isolated CPython 3.14.4 environment; package,
core, registry and console versions are 0.3.0, author metadata are unchanged,
and every RECORD hash and known private build root is audited. All six wheel
Python files exactly match the fixed source. This is a local integration
profile, not optimized-release or full scientific-program acceptance.

The installed-wheel suite passes 318 cases. Its two opt-in SP12/core13 cases
then pass with the existing flags enabled, and a separate saved-model case
passes through the second CLI name. All 26 selected linked Rust tests pass,
including the five existing nuclear-Taylor gates across semilocal, meta-GGA,
hybrid, range-separated and unrestricted paths. The core13 derivative gate
passes all 27 independent gradient/dipole/active-alpha component fits:
p=1.993043..2.002308, minimum R2=0.99999705, retaining all four steps and
original thresholds. This does not establish compressed-model chemical
accuracy. Constructor resource defaults are overridden by actual one/three
worker, 32 MiB saved-model calls with zero energy/force discrepancy.

The PBE-energy, damped-HF response and periodic-LDA comparisons also pass at
one/three workers and 32 MiB: maximum discrepancy is 1.776357e-15, largest
managed peak is 482680 bytes and process maximum RSS is 47216 KiB. These are
single small cases with initialization, not a speedup claim. The managed
ledger is not a process RSS cap; caller/Python objects, grid/basis inputs,
raw Taylor/PT2/CI vectors, unregistered allocations and stacks remain outside
its scope. Original disconnect/collection/cache-bookkeeping failures are
retained separately. The changed pytest node-list cache is preserved before
its original bytes are restored; all 1497 original file hashes and all five
built artifact hashes agree. Detailed local evidence is recorded in
`api-resources-m3-20261031-03/acceptance-separated.json` under the existing
validation workspace. Later few-center/response work is outside this artifact.

The per-calculation `cpu_threads`/`memory_mb` implementation reuses Rayon,
faer and existing component memory controls. An isolated warning-strict
harness compiles the actual resource, matrix and complex-arithmetic modules
against the existing dependency artifacts, without rebuilding the complete
application. The initial 22 tests pass, including shared atomic budget contention,
live matrix-copy accounting, tiny/overflowing allocations, nested inheritance,
independent concurrent one/three-worker settings and unchanged linear algebra
oracles. The numerical comparison retains an absolute 1e-11 gate. Both CPU
settings measure 533504 peak managed bytes for the same workload; the full
isolated test process takes 0.31s and records maximum RSS 5008 KiB. These small
contract measurements are not molecular performance or RSS enforcement claims.
Initial harness-only unused-function diagnostics remain recorded alongside
the resolved warning-strict build.

The first extension adds recovery before an FFI boundary and checks that CPU-only
scopes preserve unbudgeted behavior: all 24 tests pass in 0.45s. An isolated
extension compiled from the exact new PyO3 bridge also
passes six contract groups covering concurrent limits, inherited settings,
allocation overflow and typed budget refusals. Its initial leaked
PanicException failure is retained. The fallback bridge restores typed errors;
legacy bindings without an inner fallible guard can still emit PyO3's panic
diagnostic before that restoration. The common driver and ordinary SCF entry
have an inner allocation guard to avoid that crossing.

The initial combined Python/ASE adapter, source/discovery, version, ledger and
release-launcher checks pass all 40 selected cases (30.77s, warnings as errors)
using the older library and mocked resource callbacks. That older artifact
does not implement the new native envelope; it is not promoted by those tests.

Subsequent optional-cache reservation coverage raises the isolated total to
25 passing tests (0.16s). All-feature library and whole-unit Clippy checks,
a default-feature library Clippy check, and frontend checks of the four
permanent integration crates pass. These do not constitute a linked-library
runtime gate.

The first complete v0.3.0 resource artifact now builds the Python-feature
library, both CLI names, the substituent fitter and the whole unit-test binary.
All-feature/default-feature library and whole-unit Clippy checks pass, as does
the source/artifact portable audit. The opt1/codegen256 library build takes
882.87s with sampled peak compiler RSS 3742756 KiB on one assigned CPU;
these are build measurements, not molecular runtime or memory-limit claims.
The retained response bundle is exactly the published degree-four bundle33;
its separate spectroscopy acceptance is not inferred from these API checks.

The initial integration-harness command omitted direct Rayon linkage and
failed before test execution. Its original log remains retained. Correcting
only the harness linkage against the unchanged library passes nine resource,
one third-order XC, three linear-algebra and one finite-q cache tests, plus
seven private resource/cache/dispatch tests. The one/three-worker nested
Rayon/faer workload retains the 1e-11 comparison gate and measures 533504
managed peak bytes for each setting. Shared semilocal/meta-GGA and periodic
exchange cache lifetime, budget contention and exact recomputation fallback
now have linked-application evidence.

Three real calculations on the final artifact also pass the unchanged
absolute 1e-11 comparison against unbudgeted execution with one and three
workers and a 32 MiB budget:

| Calculation | One/three-worker managed peaks (bytes) | Maximum absolute difference |
| --- | --- | --- |
| Molecular PBE energy, water/STO-3G | 179760 / 374584 | 0 |
| Damped molecular HF response, H2/STO-3G, all real/imaginary alpha entries | 52160 / 52160 | 0 |
| Periodic LDA energy, He/STO-3G, 2x2x2 k mesh | 482680 / 482680 | 1.78e-15 Hartree |

The Python measurement process peaks at 50184 KiB RSS despite the 32 MiB
managed budget, demonstrating why the two quantities are distinct. These
small runs share the assigned validation CPUs; their single-sample times
do not establish parallel speedup or large-system memory bounds.

All 55 resource/public/version/source tests and 237 existing API tests pass
against that artifact with warnings treated as errors. The latter include
finite-temperature native/ASE/CLI parity and nonconvergence propagation,
without deselecting a scientific integration. Additional direct-core tests
then reproduce eight Python failures and three Rust failures: global limits
leak into reconstructed reference controls, and integral floats are accepted
only by the direct Rust/core resource parser. Their strict, feature-local
control fix is covered by the final compiled-artifact results above.

A broader forwarding guard also exposes 102 native wrappers outside
`native.__all__` without the common outer resource signature. Their existing
inner `run_feature` still applies runtime settings. Applying the existing
adapter to every local public native function resolves all 142 registry
forwarding/signature cases on the unchanged library. Both original failures
and the isolated adapter result remain separate. Final combined API
acceptance follows the existing PBC and shared continuous-model/SCF2
compiler order; no healthy peer compiler was interrupted.

## Declared compiler requirement audit

The original declared Rust minimum 1.82 was below the actual locked graph:
faer requires 1.84 and its selected generativity dependency requires 1.85.
The new source guard reproduces that incompatibility before the declaration
is corrected to 1.85. All 11 build-backend/source tests pass after correction.
The portable Cargo helper and wheel/editable backend audit selected
non-development dependency edges for the requested platform, normalize
patch versions and fail before compilation on incompatible declarations.
The existing full pytest suite also checks the actual source graph.
That initial graph checkpoint used Rust 1.97.1 and did not establish execution
with Rust 1.85. The later combined integration above closes that compiler and
isolated-wheel execution gap with actual Rust 1.85.0.

## Hermite-Coulomb tensor storage sweep

`RTensor::build` overwrites every valid `(t,u,v)` entry with
`t+u+v <= reach` at each auxiliary level. Its inputs are entries of the
preceding level with total order at most `reach-1`, all of which have already
been written. Clearing the whole destination cube before this sweep is
therefore redundant. Removing that clear saves `l*(l+1)^3` zero stores per
build without changing any floating-point operation, tensor layout or valid
entry. Allocations remain initialized; entries outside the final simplex
remain zero. There is no new physical model, derivative route or tolerance.

Validation of the storage change covers 163,800 baseline/current entries,
all bitwise identical. Orders `0..=24`, four exponent/center combinations,
and both Boys-function and synthetic signed base arrays are included.
The production module's original closed-form/recursive tests, Boys-function
quadrature tests and vector tests pass alongside the baseline tests: 12
tests in the isolated source harness. The additional memoized-recursion
test checks every valid component at orders `0,1,2,4,6,8,12,16`, including
co-centered and nonsymmetric centers, with the existing `1e-12` relative/
absolute bound. These are kernel checks, not full derivative acceptance or
whole-crate release certification.

A preliminary paired optimized run used single-thread batches, ten samples
per implementation, alternating execution order, and identical
`-Copt-level=3 -Ccodegen-units=1 -Clto=fat -Dwarnings` flags. Inputs were
`p=1.7`, `pc=(0.31,-0.83,0.57)` and Boys values at `p*|pc|^2`.
The ratio is baseline median divided by candidate median:

| Maximum order | Builds per sample | Baseline/candidate time |
|---|---:|---:|
| 0 | 1,000,000 | 0.905 |
| 2 | 1,000,000 | 1.149 |
| 4 | 503,178 | 1.177 |
| 6 | 177,271 | 1.157 |
| 8 | 87,066 | 1.282 |
| 12 | 22,608 | 1.894 |
| 16 | 6,048 | 2.180 |
| 24 | 562 | 1.360 |

Other builds and numerical acceptance jobs were running concurrently.
These measurements establish a promising kernel improvement, with visible
timing noise, rather than a stable hardware-independent gain. Order zero
performs no clearing in either implementation and showed a slower sample
median; a uniform speedup is not established. No end-to-end speedup is claimed.

## Reproducing the checks

Preserve the unmodified `src/integrals/hermite.rs` outside the distributable
source before a change. Use the same compiler and unchanged math/Boys
modules for both versions. The portable helper snapshots its input bytes,
compiles the actual source modules without a full library build, runs their
unit tests, checks every valid entry bitwise, then records alternating
warm batches and raw timing samples:

```text
python -W error tools/bench_hermite.py --baseline BASELINE/hermite.rs --output results.json --contention-note "otherwise idle"
python -W error tools/bench_hermite.py --baseline BASELINE/hermite.rs --output verification.json --verify-only
```

The JSON records source SHA-256 values, compiler/version flags, processor,
logical CPU count, thread/environment settings, contention note, regression
output, numerical equality count, checksums and individual timings. The
kernel itself is sequential; Rayon and faer are not linked into this
harness. Keep raw outputs and machine metadata outside packaged docs.
An initial exploratory harness used separate optimized executables; the
reusable helper runs alternating batches in one executable. Its verification
mode has passed; the same-process timing mode still needs a quiet run.

Representative existing integration baselines also pass unchanged:
`p1_integrals` (9 tests) and `p2_hf_oracle` (3 tests), including integral
closed forms and permutations, screened/in-core consistency, independent
HF energy oracles, DIIS and initial-guess independence. These were linked
against a completed library from before the storage change, so they are
baseline evidence, not integration certification of the modified library.
The follow-up rebuild includes the modified production kernel, the Python
feature and all binaries. It uses a fixed source snapshot, an independent
build directory, one Cargo job, package optimization level 1, 256 codegen
units and no link-time optimization. Artifact/source SHA-256 identities and
exact commands are recorded with the validation outputs. Portable artifact
audit passes. Against that new library, the unchanged `p1_integrals` (9),
`p1_oracle` (3), `p2_hf_oracle` (3) and `p38_hartree_consistency` (1) tests
pass with warnings denied, `RAYON_NUM_THREADS=2`,
`DFT_RS_FAER_THREADS=1` and `DFT_RS_STRICT_UNITS=1`.

A separately linked new-library harness also passes all 163,800
baseline/current bitwise comparisons and the diagnosed high-precision
component check below, alongside the three original baseline Hermite tests
(5 tests total). Native Python, ASE and canonical CLI BLYP35 route comparisons
pass for closed/open shells (2 tests): maximum gradient discrepancy
`4.440892098501e-15` Hartree/Bohr and energy discrepancy
`4.263256414561e-14` Hartree. The metadata/lint/API-documentation tests pass
all 7 cases after the authored functional documentation is placed outside
the generated declaration reference.

This closes affected integration coverage for the storage change; a full
Rust suite, full Python numerical suite and end-to-end performance comparison
remain unrun. Subsequent unrelated source edits require their own validation;
the recorded checks refer to their fixed source and artifact identities.

## High-order recurrence conditioning

An exploratory reverse-axis recursive comparison exposed an existing
rounding-sensitive example: order `12`, `p=3.4`,
`pc=(1.13,-0.37,0.91)`, component `R(2,0,9)`. The production x-first value
is `40.14032548098248`; a z-first memoized recurrence gives
`40.140325481152104`. The difference `1.6962e-10` exceeds the unchanged
`1e-12*max(abs(reference),1)` bound (`4.0140e-11`). The bitwise comparison
includes this case and confirms that the storage optimization preserves
the baseline value exactly. This exploratory reverse-axis test failed;
it is not described as accepted accuracy evidence.

The retained recursive storage-sweep test uses the conventional x/y/z
priority and preserves the original bound, geometries and all components.
It validates iteration/storage against memoized recursion without making
a stronger cross-axis conditioning claim. No threshold has been relaxed.

For this one failed exploratory component, an independent finite polynomial
in the base Boys values was evaluated with 50, 80 and 120 decimal digits.
Its integer coefficients are
`t!/(2^i*i!*(t-2i)!)` independently in each Cartesian axis. The base Boys
integrals use a converged positive-term series and high-precision exponential;
input floats are converted exactly, rather than through their printed
decimal spellings. High-precision x-first and z-first recurrences agree with
the closed polynomial. The exact-input reference is
`40.1403254809965462957177582760041252324064251784...`.
The unchanged production value differs by `-1.40633e-11`, within the original
`4.01403e-11` bound for this component, while the reverse-axis floating-point
oracle differs by `1.55557e-10` and fails it.

If the already rounded double-precision base-tensor entries are held fixed,
their high-precision polynomial instead gives `40.140325481050182397...`;
the production and reverse-axis sweeps differ from it by `-6.76994e-11`
and `1.01921e-10`, respectively. This separates base rounding from recurrence
rounding and shows why exchanging floating-point recursion axes is not an
exact reference here. The before/after production values have the same
bits, `404411f62f73cec0`. This is a one-component diagnosis of the retained
failure, not a claim that general high-order conditioning is resolved.

## Rayon and faer correctness contracts

The matrix wrapper already sends dense products to faer's blocked kernels
using borrowed row-major views. Transposed products and occupied-column Gram
matrices avoid full transpose/density temporaries. `ensure_parallelism`
selects the shared Rayon pool once; `DFT_RS_FAER_THREADS=0` or `1` selects
sequential faer execution. Work already parallelized over independent tasks
can use `matmul_seq` and the sequential transpose methods to keep dense
products from adding another parallel layer. The scalar Hermite recurrence
has dependencies between auxiliary levels and small typical working sets;
adding parallel scheduling inside each tensor has not been justified.

`tests/parallel_linalg_contract.rs` adds scalar-oracle coverage for rectangular
and zero-dimensional row-major products, both transpose directions, occupied
Gram contractions, and independent outer Rayon tasks using sequential faer
products. Binary-rational inputs require exact numerical equality; additional
nonbinary inputs retain a `1e-12*max(abs(reference),1)` bound and print raw
componentwise maxima. The test is intended to run in separate processes with
sequential and shared-pool faer settings, without changing process-global
parallelism during a run. No production matrix or scheduling code is changed.
All 3 tests pass in both separate processes (`DFT_RS_FAER_THREADS=1` and
`2`, `RAYON_NUM_THREADS=2`); the outer-task test additionally uses local Rayon
pools with 1 and 2 workers. Binary-rational results match exactly, and the
largest observed nonbinary scalar-oracle discrepancy is
`5.329070518200751e-15` in either faer setting. This validates the existing
execution contracts without establishing any new performance claim.


## Fixed-geometry magnetic response cache comparison

The fixed-nucleus full-spin one-body provider reuses the existing SCF
Coulomb/exchange engines instead of repeatedly contracting nuclear Taylor
coefficients of a zero displacement. Both complex/transverse blocks are
retained, including the imaginary Coulomb action for non-Hermitian
transition densities. Actual moving-nucleus response retains its Taylor
provider. Compensated accumulation and physical residual thresholds are
unchanged. The spinor and collinear magnetic providers now use the existing
memory-aware SCF cache-selection policy; screening stays fixed at the
reference density when the direct path is selected.

The pair-kernel matrix comparison checks 16,384 real entries per model for
full exchange, an explicit global/long-range mixture, and no exchange.
All 49,152 comparisons pass, with maximum defect 1.110223024625e-16.
Frozen source 6f4d0cd9a68986c1af05cefdae5259cfcda547033b8a8f98849fda08d999ce15
also passes all 81 independent C1 GHF Ramsey source-pair fits:
p=1.9993486529937496..2.0029433608406406,
minimum R2=0.9999994079247017, no failed intervals. The later collinear
cache source e4f3244f6b4f59379001917ece6e9bd6d2bb8fc8ddc5479685594288193eb04a
passes all 144 p318 shielding fits and seven p228 spectroscopy regressions.
All 72 earlier failed shielding intervals remain; the resolved exponents
are 1.988517012216169..2.110009592518886 with minimum R2=0.9991654994861668.

Five alternating water GKS/HF/STO-3G exact-spin NMR trials compare that
candidate with source a2aa595c08f960c48a23287576f66d6296a4d7a69295a2a69c95b69ed6e60e97.
Both use opt-level 1, no LTO, 128 code-generation units, one worker,
1024 MiB, common origin, unpruned level-1 quadrature and identical
convergence controls. These are matched validation-build measurements;
they do not establish optimized-release or larger-molecule performance.

| Measurement | Baseline median / MAD | Candidate median / MAD |
| --- | --- | --- |
| Wall time (s) | 0.674164690 / 0.028925689 | 0.328110539 / 0.009561478 |
| Peak process RSS (bytes) | 24088576 / 139264 | 23789568 / 45056 |
| Peak managed memory (bytes) | 749880 / 0 | 768888 / 0 |

The observed time ratio is 2.05469. Managed memory increases by 19,008
bytes to retain the reusable cache; this is principally a time improvement.
All shielding/decomposition components, coupling pairs/constants, chemical
shifts and every sampled spectral point pass the shared numerical gate
(atol=rtol=1e-8 in the report units). SCF uses two spinor iterations in
both executables. The benchmark is available as
`python tools/bench.py --only nmr/gks/hf/sto3g --baseline-exe OLD --exe NEW`.
The later two-sided Krylov preconditioner is not included in this comparison.

The candidate water energy is -74.96302316286685 Hartree and proton
J=-20.513871127968244 Hz. The raw orbital residual is
2.544928439295956e-13 Hartree and maximum coupling-source relative residual
is 9.714718088395279e-11; the baseline values are
2.505625911005890e-13 Hartree and 9.623315735778899e-11.


<!-- molecular-hessian-cache-checkpoint -->
### Molecular partition Hessian cache checkpoint

The molecular Hessian now caches six-coordinate Becke pair derivatives once
per grid point and reuses them across all nuclear blocks. It retains zero-safe
products and the dense nested-dual oracle. Meta-GGA uses the same weight
provider as LDA/GGA and the connected nonlocal paths. Ordered XC batch
accumulation keeps at most two batches per worker alive, preserving the
original floating-point reduction order.

The exact-module acceptance fits every one of 624 C1 gradient/Hessian
entries independently at four steps: p=1.985117940663..2.081375712643,
minimum R2=0.999542208654, with no failed intervals. Dense-dual and
single-center/zero-factor limiting checks also pass. A 13-atom, one-thread
kernel comparison alternates old/new execution five times each on 192
identical moving-grid samples. Median times are 1.925979595 and
0.561265349 seconds (3.43149-fold); every one of 299520 output entries
agrees within 4.440892098501e-16. Median peak RSS is 12108 versus
12252 KiB; this local cache trades 144 KiB measured RSS for speed.
Managed Vec workspace accounting is unavailable for this isolated kernel.

These kernel measurements do not establish whole-Hessian speed or memory
savings. Immutable candidate 2400c006d91df8668309079268193b9b8433d6db94937f9e38b9ac78bab69468
passes all ten nuclear cases (1560 component fits, with the same 44
retained/resolved intervals as the baseline) and all five optimizer contracts.
Across all 1440 Hessian entries its maximum difference from the fixed
baseline is 5.551115123126e-17 Hartree/Bohr^2; all energies and all atomic
polar tensor entries agree bitwise. The paired physical timing run was
aborted after its UKS water-dimer cation repeatedly left a nearly converged
TRAH determinant: orbital residual 6.55e-11 Hartree passed its 1e-10 gate
while density RMS change remained 1.6e-7 above its 1e-10 gate. Returning
to extrapolated diagonalization raised the energy by about 0.0146 Hartree
and restarted the cycle. Captured failure logs are retained outside the
package. No complete whole-Hessian timing comparison is accepted.
A new frozen-core regression reproduces nonconvergence after 120 iterations
(143.72 seconds, final residual 1.942826754198e-7). The isolated handoff fix
retains the accepted trust-region path until the original physical density
map and energy gates pass; its post-fix validation is pending.

Its Clippy test build detected duplicate loading of the common component-fit
helpers; the test now imports the existing in-crate acceptance module. That
change does not alter the production kernel. Subsequent source
b3f88b65fe7d552215f795d27139f2469df627e7b260f12d5b4dd02b2ee61a5f,
including the shared off-shell skeleton, passes all-feature library/binary/
test Clippy with warnings denied (6m05s). Its duplicate release build was
explicitly superseded by the ensuing unified ordinary/model checkpoint.
The latter numeric gate remains pending; no release or general performance
claim follows from these checkpoints.

The benchmark runner now atomically saves each completed timing trial.
An interrupted case is explicitly marked `measuring`, the overall record
has `complete=false`, and no numerical acceptance is assigned until all
requested trials have been compared. Console summaries are flushed after
each case. The interruption regression preserves the first trial without
accepting a partial case; all eleven benchmark tests pass with warnings
treated as errors. This runner change does not alter the ongoing immutable
cached-kernel timing measurement.


<!-- scalar-density-cache-five-pairs -->
## Explicit-density scalar response cache: bounded diagnostic

Source ed98122e8a1b2459aaf092e5bb5ab995cc491cdf4c2f1c84bbf315d9f8b0ce34,
library b8e447e64d638a08ddd751efe5fe3d169e4a98d1734d0b36ff72287d82648a26.
Five alternating fresh-process cached/streamed pairs per functional use the
same C1 H4+ four-AO fixture, grid level 1, zero grid-weight cutoff, one CPU
thread, 128 MiB managed ceiling, explicit in-core J builder, spin densities
and density perturbation. Only the optional XC derivative-cache budget is
changed; no SCF is fabricated or timed. Each process records construction,
first call and 20 repeated calls, whole-process Linux peak RSS and managed
allocation peaks. Every one of the 32 AO outputs is bitwise equal across
both modes and all five repetitions.

| Median per-call measurement | SVWN5 streamed | SVWN5 cached | PBE streamed | PBE cached |
|---|---:|---:|---:|---:|
| First call / seconds | 0.081981252 | 0.083936547 | 0.340087686 | 0.364416822 |
| Repeated call / seconds | 0.077965568900 | 0.004916500850 | 0.327479612100 | 0.008997780950 |
| Peak RSS after first call / bytes | 8470528 | 8810496 | 8716288 | 10207232 |
| Peak RSS after repeated calls / bytes | 8474624 | 8810496 | 8724480 | 10231808 |
| Managed live allocation after first call / bytes | 10056 | 360000 | 10056 | 1546560 |
| Managed peak after first call / bytes | 80704 | 394432 | 85576 | 1622080 |
| Managed peak after repeated calls / bytes | 80704 | 394688 | 85832 | 1622336 |

This cache trades bounded memory for repeated contraction speed. It does not
improve the initial provider evaluation or reduce memory, and these small
static-kernel timings do not establish a whole-molecule Hessian speedup.
The cache falls back to the shared streamed provider when its budget does
not fit. The separate ordinary full-calculation benchmarks above retain
their own sources, measurements and acceptance scopes.

The separate exact-source unit test compares the new contraction with the
previous nuclear-Taylor density kernel in five alternating pairs. Median
Taylor/cached times are 1.143161552/0.005037829 seconds (SVWN5) and
2.548254102/0.009190892 seconds (PBE); the first cached calls take
0.082037787/0.300053353 seconds. Maximum errors over all 32 AO outputs are
2.220446049250e-16/3.330669073875e-16 respectively. These measurements
isolate the old repeated Taylor work. They do not assign the streamed
cache diagnostic's RSS measurements to the Taylor implementation or imply
a whole-molecule speedup of the same factor.

<!-- scalar-pair-response-five-pairs:begin -->
## Frozen-model Hessian pair-response reuse

This five-pair benchmark compares immutable source20
ed98122e8a1b2459aaf092e5bb5ab995cc491cdf4c2f1c84bbf315d9f8b0ce34
with source22 fece69515e3f08db71fd4cd5283d51469e0c9ab5f34ccdb05a9373734f5973fb.
Each fresh process uses the unchanged nonsymmetric acceptance fixture and
times the complete production Hessian call; only the independent validation
finite-difference loops are omitted. The fixtures are PBE/STO-3G, grid level 1,
grid_weight_cutoff=0, one thread, and the same 64 MB in-core ERI setting.
The run memory_mb limits are 512 for the restricted four-AO fixture and 384
for the unrestricted five-AO fixture, unchanged between versions. Both
libraries use the same release profile (optimization 1, no LTO, 128 codegen
units); probe compilation uses optimization 1. Before/after order alternates
for five trials per reference. Process time additionally includes reference
generation and startup, so it is distinct from the Hessian-call time.

| Fixture | Hessian median before / after (s) | Process median before / after (s) | Peak RSS median before / after (bytes) | Managed peak median before / after (bytes) | Median paired speed ratio |
|---|---:|---:|---:|---:|---:|
| RKS | 8.870359 / 7.286100 | 9.278568 / 7.685555 | 28266496 / 28307456 | 2542872 / 2542872 | 1.227540 |
| UKS | 20.830260 / 15.069099 | 21.870362 / 16.067523 | 37367808 / 37343232 | 4957800 / 4957800 | 1.393895 |

Every pair improves and the before/after Hessian timing ranges do not
overlap. The raw five samples, their median absolute deviations and paired
ratios are preserved in the independent audit. All 81 raw Hessian entries
are bitwise equal in all ten paired comparisons. All 45 directional
curvatures agree within 2.220446049250313e-16/3.3306690738754696e-16
Hartree/Bohr^2 for restricted/unrestricted; maximum fresh response residuals
are 3.0531133177191805e-16/2.636779683484747e-16 Hartree/Bohr.
The full independent four-step component gates also pass on the exact
new library. Managed peaks are unchanged; the small RSS shifts do not
establish a memory improvement. This measures small complete analytic
Hessian calls, not real-molecule throughput, geometric or chemical accuracy.
The two real neutral-model Hessian runs remain a separate ongoing comparison.
<!-- scalar-pair-response-five-pairs:end -->

<!-- scalar-actual-neutral-pair-reuse:begin -->
## Completed actual neutral-model Hessian comparison

The same optimized neutral methyl-endpoint model completes on both
source20 and source22-r2, with identical input and probe hashes. All 900
ordered/symmetric Hessian entries and all 27 constrained eigenvalues and
frequencies are bitwise equal. The unchanged settings are one thread,
memory_mb=1024, saved PBE union-AO model/grid level 1, cutoff zero, SCF
energy/density/orbital tolerances 1e-14/1e-13/1e-12 and absolute response
tolerance 1e-12. Whole completed-run times are 5766.805 and 1581.383 s,
a single-run ratio of 3.64668458. These times include the initial model
evaluation and constrained-spectrum analysis; they are not five repetitions
or a whole-process memory benchmark.

Source22 phase timings are 511.423 s shared preparation, 335.584 s for
30 Cartesian responses and 704.357 s for 435 checked pair curvatures.
The original run and all matrices remain available. The larger observed
ratio does not replace the separate five-pair fixture statistics or imply
the same speedup for other systems. The completed constrained minimum and
mass/projection limits are recorded in response.md; no full molecular
thermochemistry follows from the timing result.
<!-- scalar-actual-neutral-pair-reuse:end -->

## Molecular VV10 response in ROKS SCF

The benchmark displaces N along x by 0.02 Bohr from the C1 ammonia
cation fixture. The calculation uses omegaB97X-V/ROKS, STO-3G,
level-0 main and VV10 grids, zero grid-weight cutoff, one thread and
a 1024 MiB managed limit. Energy, density and orbital SCF tolerances
are 1e-13 Hartree, 1e-12 and 1e-12 Hartree. Both binaries use the
release profile with optimization level 1 and link-time optimization
disabled. The benchmark runs five alternating candidate/baseline pairs.

| Build | Wall time median ± MAD (s) | Observed wall time range (s) | Median peak RSS (bytes) | Managed peak (bytes) |
| --- | ---:| ---:| ---:| ---:|
| Nested dual pair evaluation | 47.00059 ± 0.20632 | 45.68353–48.89934 | 26,869,760 | 629,672 |
| Shared analytic density fields | 6.41412 ± 0.27912 | 6.13500–6.87474 | 26,869,760 | 629,672 |

Every trial returned -55.37651340070984 Hartree and converged in
seven iterations with the same raw orbital residual of
1.782128792581e-13 Hartree. The energy gate was 1e-10 Hartree
absolute. The median speedup is 7.33× and both memory medians are
equal. The baseline executable SHA-256 is
`e982c287e024316b2ff0bd95e408720dce791e02ef624e72e0d06a82934b9fc8`
and the candidate SHA-256 is
`23293e26bf161c74eaeb0efda70f8c358dfe8d2b1dc84d3642430bc365fdb2ad`.
Run the case with
`python tools/bench.py --only scf/roks/vv10/wb97x-v --baseline-exe OLD --exe NEW`.
[Independent derivative and API checks](response.md#molecular-vv10-density-kernel).

## Spinor XC kernel at zero spin

A single collinear projection supplies the charge and isotropic spin
blocks of the zero-spin Hessian. The dense comparison uses the analytic
spherical projection at angular order 8. At Rust release optimization
level 1, five alternating trials of 20 evaluations per lane use the same
input point and one thread. Times below are medians in milliseconds per
point, with the observed minimum and maximum in parentheses.

| Functional | Spherical projection | Single projection |
| --- | ---: | ---: |
| PBE | 29.503 (27.525–33.641) | 0.151 (0.101–0.155) |
| TPSS | 194.386 (191.684–199.771) | 0.393 (0.347–0.395) |
| M06-L | 186.736 (109.797–191.906) | 0.395 (0.240–0.427) |

These measurements cover one point kernel. The full grid cache reserves
managed storage of up to 2,048 bytes per GGA point or 8,192 bytes per
meta-GGA point for its descriptor Hessian. The grid build, orbital
solves and process RSS have separate performance gates.
[Point and numerical scope](response.md#projected-gga-and-meta-gga-current-kernel)

## Screened Hartree contractions for charged Li

The unstrained triclinic Li+ `elastic` run uses SVWN5/RKS, STO-3G,
electronic Gamma, 13.8 Bohr overlap cutoff, 24 Bohr XC AO radius,
level-1 quadrature and zero grid weight cutoff. With one thread and a
1024 MiB managed limit, each profiled Hartree matrix build takes
29.20 to 33.74 seconds. XC takes 0.006 to 0.011 seconds per iteration.
The seven SCF cycles and final fresh field evaluation finish in 254.63
seconds at energy -7.037238120346 Hartree per cell and orbital residual
1.222e-14 Hartree.

The Hartree matrix and stress now share the wave norms, coefficient
bounds and suffix bounds across shell pairs with the same reciprocal
potential. Stress also shares the three wave-vector derivative
coefficient sets per compact or diffuse channel. Every term retains its
previous screening predicate and accumulation order.

An isolated comparison compiled the old and new contraction functions
with optimization level 1. Both used the Li basis, 225 translated shell
pairs, 29,370 reciprocal vectors and identical prescribed complex
potential coefficients. Five alternating one-thread trials gave the
following wall times.

| Contraction | Median (s) | MAD (s) | Range (s) |
| --- | ---: | ---: | ---: |
| Coefficient setup per shell pair | 2.350 | 0.138 | 1.995–2.675 |
| Shared coefficient setup | 1.807 | 0.338 | 1.469–2.158 |

All 1,250 real and imaginary matrix elements agree bitwise in each
trial. The full CLI time, RSS and residual comparison is a separate
checkpoint. The benchmark case is
`python tools/bench.py --only stress/periodic/svwn5/charged-li --baseline-exe OLD --exe NEW`.

The same profile spent 277.81 seconds in the Hartree stress before
entering the elastic calculation. In the fixed-density Hartree Hessian,
moving the common owner of a real-space quartet translates its four
centres together. The mixed strain–displacement contribution of that
quartet is identically zero. In a one-site cell, the displacement also
multiplies both reciprocal density transforms by a common phase. The
phase cancels in their products and the zero-wave-vector term is
invariant. The corresponding contractions are skipped. The full ordered
81-entry strain Hessian is still evaluated.

The charged-cell elastic timing case uses
`python tools/bench.py --only elastic/periodic/svwn5/charged-li --baseline-exe OLD --exe NEW`.
It compares the ordered Hessian and the relaxed and clamped tensors,
including internal-strain, relaxation and force-constant entries. It
records the strain and displacement solve diagnostics with the SCF and
memory measurements.
