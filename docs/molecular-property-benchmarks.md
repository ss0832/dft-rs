# Molecular property benchmarks

The initial physical benchmark is complete: eight molecular species, sixteen
static dipole/polarizability conditions, two optimized water comparisons and
three explicit-geometry primary water tensor comparisons. The additional
five species and their user-requested longer benzene calculation all finished
successfully. The measured benzene delay is predominantly repeated ERI
gradient evaluation during a slowly converging optimizer tail; it is not a
SCF divergence or response failure. Results support plausibility of the
implemented finite models while demonstrating large quantitative basis/model
errors for polarizabilities. This is not completion of the wider property
implementation/acceptance program.

## Scope and reproducibility

The initial baseline compares gas-phase closed-shell H2O, H2 and HF electric
dipoles and static electronic dipole polarizabilities. Later sections add the
explicitly named DFT, vibrational, Raman, NMR and hyperfine subsets. None of
these cases certifies untested reference, functional or derivative combinations. No production implementation, reference fixture or acceptance
threshold was changed. This is the single new comparison document authorized
by the user; subsequent cases belong in this file.

The frozen public native Python package and canonical CLI are in the local
validation bundle `.work/validation/edf2-integration-20261030` (paths relative
to the repository root). Its `build.json` records successful compilation and
the exact snapshot. The compiled Python extension SHA256 is
`8d59be3c2175aca4f2a34fdccd885a4a04e5e247a17bc23c0c9f1c62f2d3039a`;
CLI SHA256 is
`3b000c112b89fc4b830d4cc72f9dd263b24011ced1dd26cc2df5637f61735207`.
The initial baseline uses only HF routes; subsequent named DFT cases below have their own scope. The bundle's unrelated EDF2
boundary failures are not treated as accepted by this investigation.

The existing HF electric-response evidence is in
[response.md](response.md#water-finite-field-and-continuum-dfpt-acceptance).
The public contracts are in [python-api.md](python-api.md). The initial
experiment uses `calculate` and `polarizability`; an independent field check
and the zero-frequency `dynamic_polarizability` route investigate agreement
at the same structure. Agreement at one structure is not an all-component,
general derivative acceptance claim.

Local, nondistributed reproduction scripts are
`.work/validation/molecular_property_benchmark_20261031.py`,
`molecular_property_crosscheck_20261031.py`, and
`molecular_property_optimized_20261031.py` in the same directory. Raw reports,
complete geometries, controls and independent results are in
`.work/validation/molecular-property-benchmark-20261031/` as `results.json`,
`crosscheck.json` and `optimized.json`. Run each script with the existing
`.work/validation/python-env/bin/python` under WSL. Each uses one numerical
thread, existing compiled artifacts and no build. The first script verifies
the compiled extension hash before importing it.

## Conditions and conventions

All calculations use neutral singlet restricted HF, all electrons, no ECP,
no fitted Coulomb operator, no solvent, no applied bias, zero frequency and
clamped nuclei. SCF energy tolerance is 1e-12 Hartree, density-change
tolerance 1e-11 and maximum iterations 200. The initial structures are fixed
near-equilibrium test inputs, not claimed optimized structures: water
r(OH)=0.9572 Angstrom and HOH=104.52 degrees; H2 r(HH)=0.7414 Angstrom;
HF r(HF)=0.9168 Angstrom. Exact coordinates are in the raw report.

Water lies in yz, with z the angle bisector pointing toward H. H2 and HF lie
along z; HF has F at the origin and H at positive z. Dipoles include the
nuclear contribution and point from negative to positive charge. Alpha is
`d(mu_i)/d(F_j)` with energy coupling `-mu.F` and electronic Hamiltonian
`+F.r`. Native units are Hartree/Bohr; 1 dipole a.u. is 2.541746473 Debye;
1 polarizability a.u. is `(0.529177210903)^3` Angstrom cubed. The reported
isotropic value is trace(alpha)/3, not trace(alpha).

These are electronic values at fixed geometry. They contain no vibrational
averaging, zero-point correction, thermal population, rotational orientation
polarization, liquid environment or local-field correction. Experimental
gas-phase vibrational-state moments are useful model checks but are not exact
targets for these uncorrected values. No temperature-dependent correction or
experimental uncertainty has been invented.

## Initial fixed-geometry results

Alpha components and isotropic values below use Angstrom cubed; dipole
magnitudes use Debye. All six SCF reports converged. Zero H2 dipoles below
4e-15 Debye are consistent with inversion symmetry; relative error against
zero is undefined.

| Molecule | Basis | Dipole magnitude | alpha xx | alpha yy | alpha zz | Isotropic alpha |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H2O | STO-3G | 1.725797104 | 0.006209344 | 0.770891927 | 0.314403335 | 0.363834868 |
| H2O | aug-cc-pVDZ | 1.998495622 | 1.085069211 | 1.338484460 | 1.192599868 | 1.205384513 |
| H2 | STO-3G | 1.13e-15 | 0 | 0 | 0.455748667 | 0.151916222 |
| H2 | aug-cc-pVDZ | 3.39e-15 | 0.652765220 | 0.652765220 | 0.968426397 | 0.757985612 |
| HF | STO-3G | 1.288681356 | 0.000329991 | 0.000329991 | 0.465093890 | 0.155251291 |
| HF | aug-cc-pVDZ | 1.930735085 | 0.561469298 | 0.561469298 | 0.827665711 | 0.650201436 |

The vanishing/minuscule transverse response in the minimum basis is a basis
restriction, not evidence by itself of a sign or unit defect. All diagonal
responses are nonnegative; the larger basis restores transverse response.
Raw off-diagonal terms are retained rather than projected away.

## Published theoretical comparison

Fiedler, Gao and Truhlar, *J. Chem. Theory Comput.* **7**, 852-856 (2011),
[DOI 10.1021/ct1006373](https://doi.org/10.1021/ct1006373),
[accessible primary manuscript, Table 4](https://pmc.ncbi.nlm.nih.gov/articles/PMC3518311/#T4),
reports water tensors in the same yz orientation. The
[primary supporting information](https://doi.org/10.1021/ct1006373.s001),
page 1, specifies own-method optimized HF geometries; Table S1, PDF page 2,
gives isotropic values. This overrides an assumption that these are measured
geometry calculations. The downloaded PDF and its publisher metadata are
preserved locally; the PDF MD5 matches publisher metadata
`ff26e861073ebfb7044f044a951cdebf`.

Water was therefore optimized independently through the public `optimize`
route, gtol=1e-7 Hartree/Bohr and maximum 60 steps, retaining full trajectories.

| Basis | Optimized OH / Angstrom | HOH / degrees | Computed isotropic alpha | Published Table S1 | Signed difference | Relative difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| STO-3G | 0.989409339 | 100.0268875 | 0.400791733 | 0.40 | +0.000791733 | +0.198% |
| aug-cc-pVDZ | 0.943627220 | 105.9317046 | 1.181234910 | 1.26 | -0.078765090 | -6.251% |

The optimized STO-3G diagonal tensor is (0.005936394, 0.816209599,
0.380229205), versus Table 4 (0.01, 0.82, 0.38), consistent with its printed
precision. The aug-cc-pVDZ diagonal tensor is (1.075718377, 1.303889855,
1.164096498), versus (1.11, 1.42, 1.27). The second discrepancy remains an
explicit unresolved literature-condition/result discrepancy, not an accepted
agreement and not yet an identified implementation bug. Individual optimized
coordinates and response settings are absent from the downloaded SI.

### Explicit-geometry primary benchmark

A stronger comparison is available in G. Avila, *J. Chem. Phys.* **122**,
144310 (2005), [DOI 10.1063/1.1867437](https://doi.org/10.1063/1.1867437),
[primary author-deposited PDF](https://digital.csic.es/bitstream/10261/23787/1/JCP122-144310.pdf).
Table I, page 144310-3, specifies both geometries and HF/aug-cc-pVDZ static
tensors. Section II specifies spherical angular functions. Its x is our z
(bisector), its y is our x (out of plane), and its z is our y. The following
comparison uses that explicit permutation, atomic units and the identical
bond lengths/angles. Three separate public `polarizability` calls are
preserved in `primary-matched.json`; reproduce them with the additional
`molecular_property_primary_20261031.py` script in the validation directory.

| OH / Angstrom | HOH / degrees | Computed diagonal in paper's xyz order / a.u. | Published diagonal / a.u. | Maximum absolute difference / a.u. |
| ---: | ---: | --- | --- | ---: |
| 0.95843 | 104.44 | (8.064932772, 7.328154739, 9.055079889) | (8.0649, 7.3282, 9.0551) | 4.52615e-5 |
| 0.79843 | 74.44 | (6.550609602, 6.588158837, 6.359136258) | (6.5506, 6.5882, 6.3591) | 4.11625e-5 |
| 1.11843 | 134.44 | (9.122131206, 8.124246298, 14.428183530) | (9.1221, 8.1242, 14.4282) | 4.62975e-5 |

All nine entries agree with the printed four decimal places (rounding half
width 5e-5 a.u.); maximum relative discrepancy is below 0.0007 percent.
The canonical CLI independently reproduced the first native Python tensor
within 9e-15 a.u. and the energy identically; it converged in nine SCF
iterations and reported raw tensor asymmetry 1.33e-15 a.u.
The two displaced structures test the physical response away from equilibrium;
they are not asserted stable molecular geometries or thermal ensembles.
This independent primary agreement supports the implemented static response.
The other paper's larger aug-cc-pVDZ result remains visible above.

A targeted diagnostic now tests its geometry convention: evaluating
aug-cc-pVDZ at the independently optimized HF/STO-3G water geometry yields
(1.107286421, 1.419280375, 1.269978980) Angstrom cubed and isotropic
1.265515259 Angstrom cubed. Every tensor entry rounds to that paper's
(1.11, 1.42, 1.27). The source is explicitly HF, the quantity is electric
dipole polarizability, and an axis permutation cannot change the isotropic
value. Both references label the same augmented correlation-consistent DZ
basis; the explicit-geometry primary comparison and independent ground-state
agreement support the implemented basis. Neither a frequency nor a
vibrational average was specified for that paper's HF tensor table. These
observations make a geometry/convention discrepancy a concrete explanation
candidate, but the SI explicitly says own-method optimized geometries and
provides no coordinates. The inferred alternative geometry is not asserted
to be the actual author input. The SI isotropic 1.26 is also not the nearest
0.01 rounding of 1.265515259. The result is therefore classified as an
unresolved source-condition inconsistency, not an identified code defect.
The diagnostic input and output are in `cross-geometry-diagnostic.json`.

## Experimental comparison and limitations

Shostak, Ebenstein and Muenter, *J. Chem. Phys.* **94**, 5875-5882 (1991),
[DOI 10.1063/1.460471](https://doi.org/10.1063/1.460471), reports molecular-beam
electric-resonance measurements of gas-phase water. Its abstract expression
for the rotationless moment gives 1.85498 Debye for v=(0,0,0).
[NIST's dipole table](https://cccbdb.nist.gov/diplistx.asp) independently
identifies that vibrational-ground-state value. The publisher full article
was not retrievable here; the original abstract was retrievable via its DOI
metadata record, so uncertainties and detailed beam conditions are not
asserted. The fixed-geometry STO-3G result differs by -0.129182896 Debye
(-6.964%); aug-cc-pVDZ differs by +0.143515622 Debye (+7.737%). These
percent-level model deviations are not numerical-acceptance failures.

For HF, NIST compiles a gas-phase vibrational-ground-state magnitude of
1.827 Debye. The fixed-geometry values differ by -0.538318644 Debye
(-29.464%) and +0.103735085 Debye (+5.678%), respectively. This is explicitly
a compilation-level plausibility comparison: the original experimental full
text and uncertainty have not been verified. It is not promoted to a primary
experimental acceptance target.

## Independent checks and investigation status

An independently evaluated restricted-HF calculation at each of the six exact
input geometries reproduces the ground-state energies within 7.19e-8 Hartree
and dipole vector entries within 1.10e-7 Debye. For aug-cc-pVDZ the maximum
energy difference is 2.85e-13 Hartree. Raw independent versions, inputs and
values remain outside the distributable source in
`independent-ground-state.json` and the local crosscheck script. The small
minimum-basis differences are retained, without claiming bitwise agreement
between separately supplied basis data and numerical solvers.

For fixed-geometry STO-3G water the static and zero-frequency dynamic
polarizability routes agree to floating-point accuracy. Re-converged SCFs at
positive and negative fields on all three axes, with steps 0.002, 0.001,
0.0005 and 0.00025 a.u., independently confirm the response; raw full matrices
are in `crosscheck.json`. These symmetric structures are physical plausibility
checks, not a replacement for the existing C1 all-component acceptance tests.
The maximum finite-field matrix errors decrease as 8.52035e-6, 2.13029e-6,
5.32772e-7 and 1.33406e-7 a.u. The diagonal yy/zz log-error slopes are
2.0150/1.9990; the very small xx response gives 1.3747 and fails a strict
second-order-fit interpretation because of cancellation. All four points
are retained. Independent finite-field SCFs also have noise up to 1.05e-5
a.u. in this interval. Neither check is labeled a new derivative acceptance.
The zero-frequency dynamic/static maximum defect is 1.78e-15 a.u.

The six initial molecular values, two optimized water calculations and nine
explicit-geometry primary tensor comparisons are complete. The STO-3G water
result and the explicit-geometry aug-cc-pVDZ results support plausibility.
The other published aug-cc-pVDZ discrepancy is inconclusive pending its
unreported geometry/response details; the cross-geometry diagnostic above
closely reproduces its tensor despite conflicting with the SI geometry rule.
No physical anomaly has been established
in the completed static benchmark. Experiment suggests the expected
limitations of minimum-basis and uncorrelated calculations; it is not a reason
to adjust production values or tolerances.

The larger-basis dynamic/field crosscheck completed normally (exit 0;
original process PID 44738 is absent). All six ground-state crosschecks are
recorded. For fixed-geometry aug-cc-pVDZ water the zero-frequency dynamic and
static tensors differ by at most 3.20e-14 a.u.; response residuals are
(4.89e-12, 9.40e-12, 6.96e-12). The four finite-field maximum matrix errors
are (5.00845e-4, 1.25210e-4, 3.13151e-5, 7.84220e-6) a.u. The xx/yy/zz
four-step log-error slopes are 1.99903, 2.00009 and 2.00000. Independent SCF
finite-field values agree within 4.92e-4, 1.24e-4, 4.89e-5 and 4.66e-5
a.u.; the finer independent steps exhibit cancellation/SCF noise, retained
without claiming a clean asymptotic fit. No launched small-molecule process
remains running.

## Additional medium-molecule suite

The user requested approximately five additional species beyond the initial
three. Benzene, pyridine, benzaldehyde, acetamide and dimethyl sulfide cover
an aromatic hydrocarbon, nitrogen heteroaromatic, conjugated carbonyl,
amide and sulfur-containing organic molecule, respectively. They have
12, 11, 14, 9 and 9 atoms. The same primary paper above provides explicit
HF/STO-3G theoretical polarizabilities for every species (SI Table S1),
allowing a reproducible model comparison without an unsupported metal
reference or an expensive large diffuse-basis calculation.

Initial 3D conformers come from the public PubChem records
[benzene CID 241](https://pubchem.ncbi.nlm.nih.gov/compound/241),
[pyridine CID 1049](https://pubchem.ncbi.nlm.nih.gov/compound/1049),
[benzaldehyde CID 240](https://pubchem.ncbi.nlm.nih.gov/compound/240),
[acetamide CID 178](https://pubchem.ncbi.nlm.nih.gov/compound/178), and
[dimethyl sulfide CID 1068](https://pubchem.ncbi.nlm.nih.gov/compound/1068).
Exact downloaded conformers and retrieval URLs are preserved locally in
`medium-inputs.json` and the corresponding SDFs; these are starting guesses,
not experimental geometries. The public optimizer uses HF/STO-3G,
gtol=1e-6 Hartree/Bohr, maximum 100 steps, SCF energy/density tolerances
1e-11/1e-10 and maximum 200 SCF iterations. Static dipole and polarizability
are then calculated with STO-3G and 3-21G at the same optimized geometry.
The second basis is a controlled sensitivity check; it is not claimed to
use a separate 3-21G optimized geometry matching the paper's stated convention.
Calculations run sequentially with one numerical thread and a 600-second
per-molecule budget; timeouts and incomplete optimizations are not accepted.
The reproduction script is `molecular_property_medium_20261031.py`; each
case has its own JSON and log in the existing local result directory, with
one `medium-journal.json` supervisor record. The first unconditioned benzene
public optimization exceeded its 600-second diagnostic budget (600.089 s),
with no completed property result. This is a time-budget event, not a
numerical failure or evidence that the optimization would not converge.
The interrupted attempt is archived in
`benzene-unconditioned-attempt.json` and `.log`. Subsequent starting structures
are improved by an independent HF/STO-3G analytic-gradient BFGS calculation
(maximum 100 iterations, target maximum component 2e-7 Hartree/Bohr) before
calling the same public optimizer with its unchanged 1e-6 criterion. The
preconditioner's convergence status, final gradient and coordinates are
retained. It is only a source of starting coordinates; all reported property
values still come from the frozen public dft-rs interfaces. Results will
update this same document; no additional documentation files are created.
The benzaldehyde preconditioner reported precision loss at maximum gradient
2.780e-7, missing its stricter 2e-7 target; that status is retained. The public
optimizer independently evaluated 2.809e-7 and met the original 1e-6 condition.
No claim is made that the tighter preconditioner target was achieved, and
the production convergence condition was not relaxed.

The user explicitly requested a longer original benzene run and investigation
of its runtime. The original-input public optimization is queued after the
active serial suite with a 3600-second budget and identical convergence
controls. This run uses the production optimizer and records the XYZ
trajectory, phase wall times and the library's `DFT_RS_SCF_TRACE` output.
The original raw journal wording `timeout; not accepted` is retained as
historical output and must not be interpreted as a numerical convergence
verdict.

Source inspection of the frozen `src/optimizer.rs` shows a fresh SCF and
analytic gradient for each evaluation, including each of up to five
energy-backtracking trials per optimizer iteration. It calls `run_scf`
without passing the previous density as an explicit warm start. Gradient
timing labels distinguish one-electron and ERI-derivative contractions.
These are potential cost multipliers, not yet a measured attribution for
the interrupted benzene attempt; the extended run's measured timings below
provide the attribution. No optimizer or gradient code was edited.

### Completed additional-species values

All five additional species have completed both public property
calculations. Dipole magnitudes are Debye and isotropic polarizabilities are
Angstrom cubed. Both basis results use the same publicly verified stationary
HF/STO-3G geometry; the 3-21G column is a basis sensitivity comparison.

| Molecule | STO-3G dipole | STO-3G alpha | Published STO-3G alpha | Signed alpha difference | Relative difference | 3-21G dipole | 3-21G alpha |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Benzene | 6.16519e-7 | 4.753903088 | 4.75 | +0.003903088 | +0.0822% | 2.02010e-6 | 7.200900045 |
| Pyridine | 2.059188115 | 4.397112912 | 4.39 | +0.007112912 | +0.1620% | 2.505093166 | 6.635191201 |
| Benzaldehyde | 1.915543652 | 6.065611367 | 6.06 | +0.005611367 | +0.0926% | 3.382361424 | 9.006756746 |
| Acetamide | 2.377278807 | 2.364212913 | 2.36 | +0.004212913 | +0.1785% | 3.603547030 | 3.868694819 |
| Dimethyl sulfide | 0.874409163 | 2.634372056 | 2.63 | +0.004372056 | +0.1662% | 1.927502523 | 5.108981028 |

Public optimizer maximum Cartesian gradients are 9.936e-7, 1.862e-7,
2.809e-7, 5.445e-8 and 1.552e-7 Hartree/Bohr, respectively. All are below
the unchanged 1e-6 criterion. Benzene, acetamide and dimethyl sulfide agree with the printed source
rounding interval. Pyridine and benzaldehyde exceed a strict half-last-digit
interval by 0.002113 and 0.000611 Angstrom cubed; those differences are
retained and not called exact agreement. The source's unreported coordinates
and rounding convention prevent attributing these small residuals to code.
All computed polarizability eigenvalues are positive. Smallest eigenvalues
across the two bases are 0.8676, 0.9420, 1.1792, 1.2752 and 1.6803 Angstrom cubed.

The larger basis raises the mean response by 51.5%, 50.9%, 48.5%, 63.6% and 93.9%,
respectively. Relative to the broader-model scales identified above, the
minimum basis is low by approximately 52-64%; 3-21G remains low by 28-32%.
For the primary higher-level theoretical targets, benzaldehyde is low by
52.61% / 29.63% and dimethyl sulfide by 64.35% / 30.87% with STO-3G / 3-21G.
This supports a physically sensible response to basis enlargement, while
showing that neither inexpensive basis provides quantitative polarizability
predictions. It is not grounds for adjusting an implementation to experiment.
Dipoles change differently, so increased basis size is not represented as
monotonic improvement of every observable. No conformer search or vibrational
correction was performed; stationary geometry is not a full minimum/stability
certification. Benzene's residual dipole below 2.1e-6 Debye is compatible
with a nearly symmetric structure at the stated gradient tolerance; relative
error against its exact zero permanent dipole is undefined. Full Cartesian
tensors and their spectra are retained without enforcing symmetry.

### Benzene runtime evidence and transport recovery

The first extended run produced one complete evaluation before its process
disappeared during the tool transport outage, without a terminal journal.
Both original supervisor and matching numerical processes were verified
absent after reconnection. Its exact log, JSON and trajectory were archived
with prefix `transport-interrupted-1-`; the interruption reason is unconfirmed
and is not a numerical failure. The original WSL VM and other agents' healthy
jobs remained alive. At recovery it had 10570 MiB available memory and no
swap use, which does not support a current memory-pressure explanation.
The longer run was restarted with the same initial geometry, scientific
controls and 3600-second budget as an independent hidden process, after
confirming no duplicate. This preserves the user's request and isolates it
from the foreground tool-session lifetime.

Its measured first evaluation at the original structure used nine SCF
iterations (converged residual 2.49e-10), 2.779 seconds for SCF, 0.056 seconds
for the one-electron gradient and 11.744 seconds for the ERI derivative.
The ERI gradient accounts for 80.6% of those recorded stages. It had already
written initial gradient maximum 7.142e-3 Hartree/Bohr, so this is a real
optimization problem requiring further steps, not merely waiting to print a
converged result. No polarizability response construction/solve had started.
The model has 12 atoms, 36 spherical STO-3G AOs, 21 occupied and 15 virtual
spatial orbitals; the logged in-core ERI tensor is 13 MiB. Runs use one
numerical thread and the identified artifact's opt-level 1 build, so these
are controlled validation timings rather than portable production-performance
claims. The interrupted original 600-second process used approximately
72 MiB RSS and essentially one CPU; no evidence of Python overhead or SCF
iteration explosion as the dominant cause was observed.

Together with the source path, the measured evidence identifies repeated
ERI-gradient evaluation as the main per-step cost candidate. Backtracking
may multiply that cost; the final trajectory and evaluation counts are
needed to distinguish ordinary step count from excessive retries. The
restarted run retains CPU time, wall time, peak RSS, SCF/gradient stage logs
and subsequent property phase times. No performance fix has been applied.

An observed checkpoint of the detached run at 182.92 seconds had used
180.10 CPU seconds (98.46% of one core), with 73436 KiB peak RSS and no
swap pressure. It had completed 11 gradient evaluations and recorded the
initial structure plus 10 accepted steps. Thus there was no evidence of
repeated rejected gradient evaluations at that checkpoint. ERI gradients
consumed 135.936 seconds, each taking 11.701-13.062 seconds; SCFs took
7-10 iterations and approximately 2.4-3.4 seconds per evaluation. Gradient
maximum fell from 7.142e-3 to 3.013e-5 by step 5, 5.721e-6 by step 6,
and 3.391e-6 by step 10. The later reduction is slower and still above the
unchanged 1e-6 target. This directly explains why additional optimizer
iterations cost substantial time even though SCF convergence is healthy.
It is not evidence of a polarizability-response or Python bottleneck.
Read-only samples are preserved in `benzene-profile-history.jsonl` and
`benzene-live-profile.json`, with AO/occupation dimensions, SCF iteration
counts, stage timings, process CPU/wall time and memory readings.

The extended original-input run completed successfully (exit 0) after
783.705 seconds including its two property calculations. The public
optimizer took 740.823 seconds, accepted 51 steps and reached maximum
gradient 9.936018e-7 Hartree/Bohr. There were exactly 52 gradient evaluations
for the initial point plus 51 accepted steps, so repeated rejected-gradient
backtracking was not the cause in this run. The SCFs used 7-10 iterations
per point and 135.531 seconds in total. ERI derivatives used 599.811 seconds,
one-electron derivatives 2.552 seconds, and other optimizer overhead about
2.929 seconds. Thus ERI derivatives consumed 81.0% of optimizer wall time,
SCF 18.3% and other measured work less than 1%. The slow tail is explicit in
the trajectory: after about 600 seconds the maximum gradient was still
1.280e-6, above the unchanged target. The original 600-second attempt has no
trajectory, so its exact interruption point is not reconstructed; the same
input's extended run demonstrates that 600 seconds is too short under these
controlled conditions.

Post-optimization STO-3G energy/alpha calls took 2.745/5.547 seconds and
3-21G energy/alpha calls took 9.292/24.015 seconds. Those alpha durations
include their own SCF work; they are not pure matrix-solve times. They cannot
explain the optimizer delay, which occurred before either response call.
The whole child used 776.948 CPU seconds in 782.476 wall seconds (99.3% of
one core), with peak RSS 283756 KiB (277.1 MiB, including the larger-basis
property calculations). The optimizer checkpoints stayed near 72 MiB RSS.
This rules out a dominant scheduling wait or memory exhaustion in the
measured run, while keeping the unrelated transport interruption distinct.
Exact final accounting is in `benzene-final-profile.json` and the successful
terminal journal; all earlier interruptions remain preserved.

### Why the final optimizer steps improve slowly

Two stored structures, steps 20 and 21, were re-evaluated with the same frozen
public `calculate(..., gradient=True)` route after all property calculations
finished, sequentially on one thread. Both energies exactly reproduced the
stored values in the returned floating-point representation.

With `s = R21 - R20` and `y = g21 - g20`, the measured `s.y` is
9.11019e-13 Hartree. The unchanged optimizer only appends a curvature pair
when `s.y > 1e-10`; it therefore skips this positive-curvature pair and
continues using its existing limited-memory approximation. Maximum
displacement is 2.80031e-6 Bohr, and the largest gradient component belongs
to H index 10 along z. Approximately 47.8% of gradient squared norm is normal
to the carbon ring plane, so both small out-of-plane and in-plane corrections
remain; this is not merely a single unresolved bond stretch.

This verifies a concrete mechanism that prevents updating the approximation
in the observed slow tail. It does not by itself prove that changing the
threshold improves total runtime or preserves robustness elsewhere. A
separate optimizer performance investigation should assess a scale-aware
curvature condition or molecular preconditioner, together with the existing
convergence/derivative gates. The expensive ERI-gradient kernel is the other
clear optimization target. Neither production change was made for this
benchmark, and no reference value or tolerance was adjusted to obtain agreement.

The final diagnostic also completed successfully. All launched numerical
jobs belonging to this investigation are terminal; result journals and the
earlier budget/transport interruptions remain available. Only this one new
documentation file was created; raw inputs and scripts remain outside the
distributable source under the existing validation workspace.

## Harmonic frequencies and infrared intensities: completed tranche

The next bounded physical comparison covers formaldehyde (four atoms, six
modes) and dimethyl sulfide (nine atoms, 21 modes, one of the five previously
added species). All 27 harmonic wavenumbers and all 27 IR intensities were
computed through the public common-driver `run_feature("frequencies", ...)`
interface. The frozen artifact and its hashes are unchanged. Its reports
identify the analytic Hessian/APT route and six projected translation/rotation
modes; both molecules have zero imaginary modes. This uses the existing
analytic HF Hessian route: the AE RHF C1 gradient/Hessian coverage in p290
has 156 independent component fits, while p62 records the molecular
Hessian/APT/IR driver connections. This real-molecule comparison does not
replace C1 derivative acceptance or extend it to unrelated methods.

### Water source follow-up

The primary manuscript and SI geometry contradiction described above remains
unresolved. A further search of the publisher record and the authors'
[2010-2014 publication list](https://truhlar.chem.umn.edu/publications/journal-articles/2010-2014)
did not identify a correction to Part 1. The listed correction
10.1021/ct3005134 belongs to Part 2 (10.1021/ct100638g), and is not evidence
for changing Part 1's numbers. No additional author coordinates or response
settings were obtained. The alternative-geometry diagnostic is therefore
still a supported explanation candidate, not a proven correction; the
original comparison and discrepant values are retained. No water scalar
benchmark was needlessly repeated for this tranche.

### Conditions, references and limitations

Both models are isolated neutral closed-shell HF/STO-3G molecules without
solvent, ECP, fitted Coulomb integrals, dispersion, external fields or
electronic temperature. The SCF controls are energy tolerance 1e-12 Hartree,
orbital-gradient tolerance 1e-10, density-change tolerance 1e-11, maximum
300 iterations and an in-core allowance of 512 MiB. One numerical thread
was used, with no build. Formaldehyde started from the three-decimal Cartesian
geometry in its NIST calculation record and was publicly reoptimized to
gtol=1e-7; it converged in nine steps to maximum gradient 4.8123e-8
Hartree/Bohr. Dimethyl sulfide reuses the previously accepted public
HF/STO-3G geometry (maximum gradient 1.5523e-7). Complete coordinates are
in the per-case JSON files. Thus structures are stationary at the specified
level, rather than experimental or higher-level structures.

The primary computational data records are NIST CCCBDB Release 22:
[formaldehyde HF/STO-3G](https://cccbdb.nist.gov/energy3x.asp?basis=20&casno=50000&method=1)
and [dimethyl sulfide HF/STO-3G](https://cccbdb.nist.gov/energy3x.asp?basis=20&casno=75183&method=1),
under "Vibrational Frequencies calculated at HF/STO-3G". Those tables print
integer unscaled wavenumbers and two-decimal IR intensities. Reference mode
numbers and symmetries below are their labels. The records were retrievable
through indexed primary-page content; direct HTTP downloads returned 403,
and the underlying job-file page returned 500. No exact input/output file,
mass table or convergence settings were retrieved. The published Cartesian
coordinates are rounded to three decimals, not full-precision reproduction
inputs. These source limitations prevent treating every last printed digit
as a strict numerical acceptance target.

Wavenumbers are harmonic cm^-1; integrated double-harmonic IR strengths are
km/mol. Normal-mode phases cannot change these scalar strengths. Reference
matching uses ascending frequency and retains the source mode number; close
pairs should also be interpreted as small spectral clusters. No intensity
matching was used to choose the assignment. No empirical frequency scale
factor was fitted or applied, including the separate 0.8165 column offered
by the database. Temperature 298.15 K, pressure 101325 Pa and symmetry number
2 only specify the returned RRHO thermochemistry; they do not add thermal,
anharmonic, solvent or rotational-band corrections to the harmonic spectrum.

The public driver uses standard atomic weights H=1.008, C=12.011,
O=15.999 and S=32.06 u. This is not an isotope-specific spectrum, and this
driver contract exposes no isotope-mass override. The independent mass
sensitivity check below is kept separate from these public results.

### All-mode theoretical comparisons

Formaldehyde values are listed in reference mode order. DMS denotes dimethyl
sulfide in the subsequent table. Raw floating-point values, all mode vectors,
units and residual diagnostics are retained locally.

| Mode | Symmetry | Computed harmonic / cm^-1 | NIST harmonic / cm^-1 | Computed IR / km mol^-1 | NIST IR / km mol^-1 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | A1 | 3498.384435 | 3499 | 1.823034 | 1.82 |
| 2 | A1 | 2099.231178 | 2100 | 8.457884 | 8.48 |
| 3 | A1 | 1767.133711 | 1767 | 4.159818 | 4.15 |
| 4 | B1 | 1278.607979 | 1279 | 6.155883 | 6.15 |
| 5 | B2 | 3645.242762 | 3646 | 19.880822 | 19.89 |
| 6 | B2 | 1397.388370 | 1398 | 29.571135 | 29.59 |

| DMS mode | Symmetry | Computed harmonic / cm^-1 | NIST harmonic / cm^-1 | Computed IR / km mol^-1 | NIST IR / km mol^-1 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 11 | A2 | 181.812475 | 183 | 7.07e-10 | 0.00 |
| 15 | B1 | 190.927209 | 192 | 0.587642 | 0.59 |
| 7 | A1 | 339.424696 | 340 | 0.266613 | 0.27 |
| 6 | A1 | 917.280712 | 918 | 0.551213 | 0.55 |
| 21 | B2 | 978.884169 | 980 | 0.115557 | 0.12 |
| 20 | B2 | 1115.445947 | 1116 | 0.016092 | 0.02 |
| 10 | A2 | 1153.803440 | 1154 | 4.72e-11 | 0.00 |
| 14 | B1 | 1182.630855 | 1183 | 0.136868 | 0.14 |
| 5 | A1 | 1253.356831 | 1254 | 13.255536 | 13.26 |
| 19 | B2 | 1693.960656 | 1694 | 16.683659 | 16.71 |
| 4 | A1 | 1721.123609 | 1722 | 0.544888 | 0.55 |
| 9 | A2 | 1826.000002 | 1826 | 6.25e-9 | 0.00 |
| 13 | B1 | 1827.555226 | 1828 | 6.338142 | 6.34 |
| 18 | B2 | 1839.687960 | 1840 | 6.933914 | 6.93 |
| 3 | A1 | 1842.617552 | 1843 | 0.255761 | 0.26 |
| 2 | A1 | 3548.305686 | 3548 | 1.659081 | 1.66 |
| 17 | B2 | 3549.287475 | 3549 | 1.575252 | 1.58 |
| 12 | B1 | 3725.135825 | 3725 | 8.088994 | 8.10 |
| 8 | A2 | 3726.374692 | 3727 | 6.02e-8 | 0.00 |
| 16 | B2 | 3744.324526 | 3744 | 0.003131 | 0.00 |
| 1 | A1 | 3745.532703 | 3746 | 0.903244 | 0.90 |

Maximum absolute wavenumber differences are 0.76882 cm^-1 for formaldehyde
and 1.18753 cm^-1 for DMS; mean absolute differences are 0.54650 and 0.50642
cm^-1. The maximum relative differences are 0.04375% and 0.64892%; the latter
belongs to the lowest, approximately 183 cm^-1 mode. Maximum absolute IR
differences are 0.02212 and 0.02634 km/mol. Percent errors are unsuitable for
zero or weak rounded reference intensities. All four A2 DMS modes have IR
intensity below 6.1e-8 km/mol without forcing exact zeros, consistent with
the symmetry selection rule. These are physically plausible finite-model
spectra, but not exact agreement with every printed source interval.

As a derived units check, harmonic zero-point energies are 6842.9942 and
20051.7361 cm^-1, versus the database's unscaled 6844.2 and 20055.1 cm^-1.
This is half the frequency sum, not an independent additional derivative
acceptance. Mass/source precision differences must also be considered here.

### Independent numerical and mass checks

Independent analytic HF Hessians at the exact same two structures reproduce
all public harmonic frequencies, using the identical standard atomic weights,
within 5.49e-5 and 9.99e-5 cm^-1, respectively. Raw Hessians and all frequencies
are saved, rather than checking only selected stretches. The first ordinary
reference-SCF attempt exhausted 50 cycles under its stricter energy/gradient
targets of 1e-13/1e-11 and supplied no accepted Hessian. That failed attempt
is retained. Increasing its iteration budget to 200, without changing either
target, converged for both molecules; no second-order fallback was needed.

An independent reanalysis then substitutes explicit main-isotope masses
1.00782503223, 12, 15.99491461957 and 31.9720711744 u from the
[NIST isotopic-mass tables](https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?all=all&ascii=ascii&ele=&isotype=all).
This is a mass-convention sensitivity experiment, not a modified public
calculation or claimed isotope API. Formaldehyde's largest difference from
the printed theoretical frequencies falls to 0.38290 cm^-1, with all six
entries inside the integer-rounding interval. DMS still differs by 1.16970
cm^-1 at its lowest mode; seven entries remain outside a strict half-last-
digit interval. Thus mass convention can explain the formaldehyde scale
of discrepancy, but does not fully reconcile DMS. The absent original mass,
geometry and convergence records prevent assigning the residual DMS source
difference to a specific cause. The independent same-mass agreement does
not support calling it a production numerical defect.

### Experimental model assessment for formaldehyde

Nakanaga, Kondo and Saeki, *J. Chem. Phys.* **76**, 3860-3865 (1982),
[DOI 10.1063/1.443527](https://doi.org/10.1063/1.443527), reports gas-phase
FTIR measurements with polymerization controlled and overlapping bands
separated by simulations accounting for Coriolis interactions. The primary
publisher abstract was retrieved. Numerical fundamentals, integrated
intensities and reported uncertainties below are the
[NIST transcription of that paper](https://cccbdb.nist.gov/exp2x.asp?casno=50000).
The original PDF tables, exact pressure/temperature and uncertainty coverage
were not retrieved; these are not asserted to be one-standard-deviation
uncertainties. No solid/solution spectrum is mixed into this gas comparison.

| Mode | Assignment | Computed harmonic / cm^-1 | Experimental fundamental / cm^-1 | Computed double-harmonic IR / km mol^-1 | Experimental band IR / km mol^-1 | Reported uncertainty / km mol^-1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | symmetric CH2 stretch | 3498.384 | 2782 | 1.8230 | 75.5 | 7.1 |
| 2 | CO stretch | 2099.231 | 1746 | 8.4579 | 74.0 | 5.3 |
| 3 | CH2 scissors | 1767.134 | 1500 | 4.1598 | 11.2 | 1.0 |
| 4 | CH2 wag | 1278.608 | 1167 | 6.1559 | 6.5 | 0.6 |
| 5 | antisymmetric CH2 stretch | 3645.243 | 2843 | 19.8808 | 87.6 | 8.0 |
| 6 | CH2 rock | 1397.388 | 1249 | 29.5711 | 9.9 | 1.0 |

The harmonic wavenumbers exceed the measured fundamentals by 9.56-28.22%.
Those are different physical definitions: electronic-structure/basis error,
mechanical/electrical anharmonicity and vibration-rotation effects are not
separated by this comparison. IR deviations are strongly mode-dependent:
the symmetric stretch is low by 97.59%, CO stretch by 88.57%, whereas the
rock is high by 198.70%. The wag alone is comparatively close. Agreement
with the same-level theoretical table alongside these large experimental
differences shows that this small-basis HF/double-harmonic model is not a
quantitative prediction of the gas-phase IR spectrum. It does not justify
tuning fixtures or correcting production intensities to experiment. The
missing primary measurement conditions also preclude a precision experimental
acceptance claim.

### Reproduction and final status of this tranche

The serial public driver script is
`.work/validation/molecular_vibrational_benchmark_20261031.py`; its two case
budgets were 3600 seconds, while both completed successfully in 6.855 and
29.515 seconds including setup. Actual analytic frequency/IR calls took
1.401 and 27.431 seconds. DMS timing includes 18.124 seconds of second ERI
derivatives, 4.107 seconds of first-derivative Fock construction and 2.926
seconds of CPKS solving. The independent script is
`crosscheck_vibrational_mass_20261031.py`, and
`molecular_vibrational_cli_20261031.py` reproduces formaldehyde using the
canonical CLI. Its frequencies and IR intensities agree exactly with the
native common-driver report in the returned floating-point values; default
SCF progress, exact arguments, JSON and exit code are retained.

All raw results are inside the existing benchmark directory's
`vibrational-tranche/` subdirectory: `journal.json`, per-case reports/logs,
`summary.json`, `independent-mass-crosscheck.json`, the initial failed SCF
record, and `experimental-formaldehyde-reference.json`. All launched jobs
for this tranche are terminal. No new documentation MD/CSV file, build,
production edit, reference-fixture edit, threshold change or publishing
operation was performed. The completed scope is two real-molecule harmonic
frequency/IR comparisons; Raman, anharmonic fundamentals and excitation
energies remain separate future tranches rather than implied acceptance.

The primary matching targets in SI Table S1 are 4.75, 4.39, 6.06, 2.36
and 2.63 Angstrom cubed for the five species in the order above. Their
rounding half width is 0.005 Angstrom cubed, not a newly relaxed numerical
acceptance threshold. Isotropic comparisons avoid unreported orientation
choices. Matching the same basis tests implementation plausibility, while
the additional 3-21G result tests sensitivity to basis incompleteness.

The paper's Table 1 also supplies useful broader-model scales: benzaldehyde
12.80 Angstrom cubed is its own MP2/aug-cc-pVDZ calculation (footnote c),
and dimethyl sulfide 7.39 Angstrom cubed is its own MP2/aug-cc-pVTZ result
(footnote a, nominally at MP2/aug-cc-pVDZ geometry). The SI instead states
HF/STO-3G geometries for MP2 references, so that geometry inconsistency is
disclosed for these scales too. They are primary published theoretical
values, not experimental measurements or exact same-method acceptance
targets. Benzene 10.32, pyridine 9.18 and acetamide 5.67 Angstrom cubed in
the same table are compiled literature values; they are kept distinct from
the authors' own calculations and are not represented here as independently
retrieved primary experimental measurements. Missing temperature, frequency,
phase-specific details and uncertainties preclude a precision experimental
test against those compiled numbers.

## GGA and hybrid permanent dipoles: completed DFT tranche

Four additional closed-shell gas-phase electronic dipoles use PBE/STO-3G and PBE0/STO-3G through the frozen public `calculate` API. PBE0 is the 25% exact-exchange PBE hybrid, corresponding to the NIST name PBE1PBE. These are ground-state density expectations, not a claim that all hybrid vibrational, magnetic or high-order routes are accepted. The ground-state common-driver subset is tracked by T008/R067; existing restricted GGA/global-hybrid component and SCF evidence is in [response.md](response.md). No new derivative route was needed.

Each calculation fixes the method-specific NIST Cartesian structure as actually printed (0.001 Angstrom precision); it does not substitute the earlier HF-optimized structures. The source optimizes its structures but does not supply full-precision coordinates, SCF/grid settings or uncertainties on these pages. Thus this matches the published rounded coordinates, with a remaining geometry mismatch to the unpublished full-precision reference calculation. The neutral singlet total dipole includes nuclear and electronic terms, uses the same Cartesian axes, and points along negative z. No solvent, temperature averaging or vibrational correction is included.

| Molecule | Method | Calculated magnitude (D) | NIST (D) | Difference (D) | Relative difference |
| --- | --- | ---: | ---: | ---: | ---: |
| formaldehyde | PBE | 1.262657826 | 1.262 | +0.000657826 | +0.05213% |
| formaldehyde | PBE0 | 1.379688673 | 1.379 | +0.000688673 | +0.04994% |
| dimethyl-sulfide | PBE | 0.809722932 | 0.810 | -0.000277068 | -0.03421% |
| dimethyl-sulfide | PBE0 | 0.886907697 | 0.887 | -0.000092303 | -0.01041% |

Primary calculation records: NIST CCCBDB Release 22, each page’s “Geometric Data” and “Electric dipole moments” sections: [formaldehyde PBE](https://cccbdb.nist.gov/energy3x.asp?basis=20&casno=50000&method=61), [formaldehyde PBE1PBE](https://cccbdb.nist.gov/energy3x.asp?basis=20&casno=50000&method=54), [dimethyl sulfide PBE](https://cccbdb.nist.gov/energy3x.asp?basis=20&casno=75183&method=61), [dimethyl sulfide PBE1PBE](https://cccbdb.nist.gov/energy3x.asp?basis=20&casno=75183&method=54). PBE here means exchange plus correlation, not exchange alone.

The native grid sequence is 3, 4, 5 with all partition weights retained; SCF tolerances are energy 1e-12 Hartree, orbital residual 1e-10 and density 1e-11, at most 300 iterations. The largest grid-4 to grid-5 dipole change is 1.36446e-6 D. Independent same-geometry calculations at grid levels 5 and 6 differ from the final native values by at most 2.54003e-7 D. The two grid families are not assumed point-identical. All four SCFs and independent references converge; the entire serial tranche took 220.165 s, 219.281 CPU s and 316.688 MiB peak RSS.

**Interpretation:** all four comparisons are physically plausible. The formaldehyde differences (0.000658 and 0.000689 D) exceed a strict half-last-digit 0.0005-D rounding interval. This is retained, not rounded away or labelled exact reproduction. Independent agreement and the small grid tail isolate the remaining discrepancy to unavailable source conditions, including coordinate rounding; they do not by themselves prove which missing condition accounts for it. The dimethyl-sulfide values lie inside that interval. These observations do not establish experimental accuracy of a minimal basis, nor do they validate a different property merely because it shares an SCF reference.

## Property inventory and staged physical-comparison coverage

The complete 170-row [property_targets.csv](property_targets.csv) inventory and the associated Markdown verification sections were inspected. It contains 128 observable targets; the remaining 42 are components (5), foundations (3), interfaces/exports (6), models/controls (15), references (1), required methods (9), and validation/algorithms (3). A local snapshot preserves every row and its original verification fields without rewriting the catalogue. “Implemented”, independent numerical acceptance, public API parity and real-molecule literature comparison are different forms of evidence; none substitutes for the others.

| Inventory group | Physical comparison in this report | Boundary / next stage |
| --- | --- | --- |
| T004/T006/T008, ground-state energies, gradients, dipoles | Eight-species HF baseline and four PBE/PBE0 dipoles; energies/forces support the stated structures | No general functional/reference certification from these samples |
| T013–T017, Hessian, harmonic frequencies, IR and Raman | 27 HF harmonic modes/IR lines; dynamic Raman follow-up below | Excited, anharmonic and periodic spectra require distinct comparisons |
| T019/T020, static/dynamic alpha | Static suite and zero-frequency consistency; unresolved water source-condition discrepancy retained | Finite-frequency tensor/dispersion coverage remains limited |
| T039, hyperfine; T044, NMR shielding | Priority real-molecule follow-up below | T003 NMR spectrum and T043 indirect nuclear J coupling are separate observables |
| T021–T033, excited/optical/chiroptical spectra; T034–T037/T112/T119, nonlinear/anharmonic response; T069/T075/T080/T081, magnetic tensors and ROA | No real-molecule literature result claimed by the current tranche | Select source-matched accepted subsets after the three user-named priorities; scope depends on the reference/basis/functional |
| Periodic, elastic, phonon, transport, material and specialized/multicomponent targets | Outside this gas-phase molecular comparison | Need their own phase, boundary, temperature and model references; molecular results cannot certify them |

Dynamic Raman, orbital NMR shielding and electron–nuclear hyperfine tensors are the current priorities, rather than the entire requested property programme. API completeness is being audited separately; this report records scientific values and reference compatibility. No unbenchmarked catalogue cell is marked accepted by this inventory reconciliation.

## Dynamic Raman, NMR shielding and hyperfine tensors: completed priority tranche

These three additional observables were actually evaluated through public APIs. Raman reuses the specifically accepted immutable `hermite-integration-20261030` Python extension (SHA256 `a1c6cb7f342cff4a3f37937563934fc362d6da9065af468501eefd773d473e92`), whose T017 evidence includes all 81 C1-ammonia mode/tensor fits and the distorted-water static limit in [the target-specific acceptance record](validation-matrix.md#target-specific-acceptance). This certifies the stated restricted HF/STO-3G subset, not all DFT Raman combinations. Magnetic calculations use the earlier identified frozen `edf2-integration-20261030` extension. Existing NMR mixed-field and hyperfine operator/component acceptance is recorded in [response.md](response.md#orbital-magnetic-response-and-nmr) and its “Complete collinear hyperfine tensors” section; the following physical comparisons add scientific evidence without claiming a complete functional/basis/reference matrix.

### Water dynamic Raman at 514.5 nm

`run_feature('ir-raman', ...)` uses photon energy 0.088558508322 Hartree (514.5 nm), zero damping, the previously converged HF/STO-3G water structure (0.989409339 Angstrom, 100.0268875 degrees), and the analytic harmonic normal modes. Atomic masses are the same average masses used in the harmonic tranche. The dynamic response converges with reported residual 9.6002e-11. Activities are Angstrom^4/amu; they are not measured scattering intensities. The invariant is 45 times the squared isotropic derivative plus 7 times the squared anisotropy; the reported perpendicular/parallel depolarization ratio uses 3 gamma-squared divided by (45 isotropic-squared plus 4 gamma-squared). Mode signs do not affect either invariant.

| Water mode | Harmonic frequency (cm^-1) | Static activity | Dynamic activity, 514.5 nm | Dynamic/static | Dynamic depolarization | Published CCSD/W6 dynamic activity (depolarization) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bend | 2169.851313 | 9.265222 | 9.406079 | 1.015203 | 0.724360 | 0.8 (0.71) |
| symmetric stretch | 4139.635233 | 47.811523 | 50.395055 | 1.054036 | 0.179391 | 119 (0.04) |
| antisymmetric stretch | 4390.672987 | 21.551193 | 22.615631 | 1.049391 | 0.750000 | 28 (0.75) |

The primary theoretical values are Avila, JCP 122, 144310 (2005), [DOI10.1063/1.1867437](https://doi.org/10.1063/1.1867437), Table VIII p144310-7, final CCSD/W6 column at 514.5 nm. Equations7–10 define these activities from polarizability derivatives with respect to normal coordinates at equilibrium, using the fitted correlated polarizability surface near rOH=0.95843 Angstrom and angle104.44 degrees. Although the article also calculates rovibrational spectra and averages elsewhere, those corrections must not be assumed to enter this derivative-defined TableVIII comparison. The retained original article text supplies this table and its defining equations; the publisher/author PDF could not be reobtained for a fresh screenshot (access/bot challenge). Original experimental Murphy tables were not retrieved, so their transcription in Avila is not promoted here to an independently checked primary experimental measurement.

**Interpretation:** dispersion increases all three activities by 1.52%, 5.40% and 4.94%, with depolarization ratios inside the physical 0–0.75 interval. Against CCSD/W6 the activity differences are +1075.76%, -57.65% and -19.23%; the symmetric-stretch depolarization is also much higher. Thus HF/STO-3G is quantitatively inadequate for these water Raman intensities, especially bending. This is a real model warning, not a passing accuracy claim. The comparison changes basis, electronic correlation, equilibrium structure and normal-coordinate transformation; it does not isolate a code defect. Both sides use equilibrium polarizability derivatives for the activities. Existing independent full-tensor derivative fits remain the implementation evidence. No frequency-fourth-power, thermal population, linewidth or polarization collection factor was applied: these are activities, not normalized experimental spectra. An exact same-model dynamic-activity source would be needed to sharpen the diagnosis.

### Absolute orbital NMR shielding and reference conventions

`nmr` computes London/GIAO orbital shielding tensors in ppm at fixed neutral-singlet geometries. The tensor axes are nucleus, nuclear magnetic-moment axis, external magnetic-field axis. Isotropic shielding is its trace/3; an individual shielding tensor is not forced symmetric. The origin is initially zero Bohr. These are absolute shieldings sigma, not chemical shifts delta. A chemical shift requires the same-isotope, same-model reference, with delta approximately sigma(reference)-sigma(sample); no TMS or arbitrary reference has been silently inserted. At fixed nuclear positions the nonrelativistic electronic orbital shielding does not contain isotope-mass, spin-Zeeman, hyperfine, Curie, vibrational or thermal effects. Experimental observation of oxygen shielding would use 17O, whereas 16O has spin zero.

The exact-coordinate primary calculation is Jensen et al., PCCP18,21145–21161(2016), [DOI10.1039/C6CP01294A](https://doi.org/10.1039/C6CP01294A), Table1 coordinates and Table3 GTO/pcS-0 results. Water Bohr coordinates are O(0,0,-0.125), H(+/-1.4375,0,1.025); methane is C at zero and H at the four tetrahedral sign combinations of1.184860 Bohr given in the retained input. The functional is explicitly selected as B3LYP5/VWN5. The paper’s functional-library alias definition identifies its B3LYP as VWN5; native B3LYP is a separate VWN3 convention. Archived paper input decks were not available, so this convention identification is documented rather than hidden. Standard contracted pcS-0 data were downloaded from the primary basis distributor; exact bytes and SHA256 are retained. The publisher’s direct page returned403, while its indexed primary table supplied the numerical targets.

| Molecule/nucleus | Calculated B3LYP5/pcS-0, grid7 (ppm) | Published (ppm) | Difference (ppm) | Grid5 to7 change (ppm) |
| --- | ---: | ---: | ---: | ---: |
| water, O | 332.115618006 | 332.1143 | +0.001318006 | -3.003e-07 |
| water, H (equivalent) | 31.929952578 | 31.9300 | -0.000047422 | -2.668e-08 |
| methane, C | 192.784525953 | 192.7857 | -0.001174047 | -1.737e-05 |
| methane, H (equivalent) | 32.484245986 | 32.4843 | -0.000054014 | -4.240e-08 |

All magnetic equations converge below1e-10 relative residual. The largest final source difference is0.00131801ppm (water O), about4parts-per-million of the shielding magnitude, and the methane C difference is-0.00117405ppm. Both exceed the paper’s printed rounding interval and the measured final grid tail; they remain small unresolved source-condition discrepancies, not exact reproduction. Proton differences are about-0.0000474ppm (water) and-0.0000540ppm (methane). A native VWN3 diagnostic gives water O332.022509745/H31.928489396ppm, visibly different from VWN5. Thus functional conventions matter; no implementation was tuned to match a table. Full source input/basis contractions, quadrature and convergence settings would be needed to resolve the remaining millippm offsets. There is no present evidence requiring a production correction.

As a separate experimental plausibility check, H2 at rHH=0.7414 Angstrom gives HF/STO-3G sigma(1H)=27.748575476ppm and HF/aug-cc-pVDZ=26.472215313ppm. Garbacz et al., JPCA116,11896–11904(2012), [DOI10.1021/jp309820v](https://doi.org/10.1021/jp309820v), primary abstract, measures isolated zero-density H2 at300K as26.293(5)ppm. The respective differences are+1.455575ppm (+5.54%) and+0.179215ppm (+0.682%). The electronic equilibrium calculation omits correlation and rovibrational/300K corrections, so this is not a test within the experimental0.005ppm uncertainty. Simultaneous translation of the aug-cc-pVDZ molecule and change of GIAO origin changes the full tensor by at most 4.157e-13ppm.

The additional HF/cc-pVDZ water run at0.9572 Angstrom/104.52degrees gives O347.957378311/H31.380325077ppm. Vaara et al., JCP109,8388(1998), [DOI10.1063/1.477501](https://doi.org/10.1063/1.477501), TableI p8390, reports O347.5/H31.35ppm with London orbitals, but calls its geometry “experimental” without numerical coordinates. Differences+0.457378/+0.030325ppm therefore cannot be called exact same-geometry errors. Its later optimized RAS geometry is a different table and was not substituted. The exact-coordinate comparison above is the stronger implementation check.

### Methyl radical hyperfine contact and spin-dipolar tensors

The public `hyperfine` calculation is an isolated planar neutral CH3 doublet (S=1/2), rCH=1.078 Angstrom, unrestricted PBE0 and explicitly contracted IGLO-III. Supplied dimensionless nuclear g factors are1.4048236 for13C and5.5856946893 for each1H; 12C would have zero nuclear spin and no EPR carbon hyperfine splitting. The Hamiltonian convention is S.A.I. Couplings are signed MHz, with gauss converted using the free-electron g factor; no absolute-value operation hides negative proton couplings. The model includes Fermi contact and spin dipole, without spin-orbit, rotational, vibrational or matrix/environment averaging.

| Nucleus | Isotropic MHz | Isotropic G | Ascending principal values, MHz |
| --- | ---: | ---: | --- |
| 13C | 79.18020033 | 28.25346565 | 0.35318368, 0.35318884, 236.83422845 |
| 1H (three equivalent sites) | -64.98257613 | -23.18740007 | -105.36210292, -64.08939297, -25.49623249 |

A primary experimental anisotropic comparison is Kubota et al., JCP71,4771–4776(1979), [DOI10.1063/1.438315](https://doi.org/10.1063/1.438315), publisher abstract: proton principal values-93.7,-60.3,-31.4MHz and isotropic-61.8MHz. The crystal was CH3COONa.3D2O irradiated at77K and analyzed below24K with methyl tunneling. Our gas-phase electronic proton isotropic value is-64.98258MHz (difference-3.18258MHz,5.15% in magnitude); principal differences are-11.66210,-3.78939,+5.90377MHz. The sign, order and scale are plausible, while anisotropy is quantitatively imperfect. Solid matrix, tunneling, temperature, geometry and electronic model differ, and no experimental uncertainties are supplied by the accessible abstract. These differences cannot isolate a tensor implementation bug.

The independent same-basis PBE0 calculation reproduces nuclear spin densities to 9.372e-08 electrons/Bohr^3 and total energy to 3.309e-09Hartree. Grid3 to5 changes contact couplings by at most 3.234e-04MHz. The raw dipolar trace is at most 2.274e-13MHz, total-trace/contact defect 7.105e-14MHz and full eigen-reconstruction defect 5.684e-14MHz. These checks retain all tensor entries; isotropic values were not enforced by replacing the trace.

### Reproducibility, anomaly classification and remaining scope

All owned jobs are terminal. The initial magnetic/Raman supervisor records Raman exit0(33.24s), NMR exit0(40.65s) and the retained missing-basis hyperfine setup exit1. The separate IGLO-III hyperfine run and exact-coordinate NMR/grid refinement subsequently exit0. Logs, reports, source URLs, basis bytes/hashes and the compact final residual summary are under `molecular-property-benchmark-20261031/magnetic-raman-tranche/`. Scripts are `.work/validation/molecular_magnetic_raman_20261031.py`, `molecular_nmr_matched_20261031.py` and the basis retrieval helpers. Run the former with `raman`, `nmr` or `hyperfine` for the individual calculation; run the NMR script once, then with `--refine` for grid7. The secondary EPR archive is evidence only, not an accepted input. No dependencies were installed.

The earlier Fiedler water discrepancy is still unresolved. Its resolution requires the authors’ full-precision coordinates and row mapping, exact basis/contraction/angular definitions, static versus finite-frequency settings, response/SCF/field controls and unrounded tensors or author clarification. Repeating the already matched Avila tensors or the existing geometry diagnostics does not supply those missing source data.

## Independent NMR isolation and authoritative-source follow-up

The remaining Table3 NMR offsets were investigated with a genuinely independent orbital-shielding implementation, not another route to the same compiled kernel. Its official upstream shielding modules were read and retained locally with source hashes, and used with the already present numerical environment; no package was installed and no distributable dependency changed. Both codes use exactly the retained pcS-0 contractions, exact published Bohr coordinates, B3LYP/VWN5 and London/GIAO orbitals. Independent SCFs retain all grid points, use energy1e-13/orbital1e-11 tolerances, and run grids6 and7. Independent coupled response uses tolerance1e-12 and at most200 iterations. Full tensors are compared after transposing external-field/nuclear-moment indices to the native nuclear-moment/external-field order.

| Molecule | Independent grid7 heavy nucleus (ppm) | Independent grid7 H (ppm) | Native minus independent heavy nucleus (ppm) | Maximum full-tensor difference (ppm) |
| --- | ---: | ---: | ---: | ---: |
| water | 332.115614649 | 31.929952561 | +3.357e-06 | 9.362e-06 |
| methane | 192.784525796 | 32.484245960 | +1.564e-07 | 1.564e-07 |

**Diagnosis:** the independent calculation reproduces the native isotope-independent electronic shieldings to9.362e-6ppm for every water tensor entry and1.564e-7ppm for every methane entry. The approximately0.0013ppm published-table offset is present in both implementations. This strongly excludes a defect specific to the native response implementation as its cause at that scale. It does not identify which unpublished source condition produced the offset. The Jensen paper specifies the GTO basis and GIAO convention but does not provide its GTO quadrature, SCF/response settings, numerical basis literals or archived inputs; it does not report a Table3 GTO convergence defect. A missing setting, basis revision, constants or source rounding remains a hypothesis, not a demonstrated source error. Its alternative multiwavelet values approach a different basis-set limit and must not replace the pcS-0 targets. Thus the implementation-side investigation closes without a production change, while exact reconstruction of the historical table remains limited by source provenance.

The four independent calculations all completed in 27.661 seconds on one core.

The Fiedler water primary-source search likewise yielded no new full-precision coordinates or row mapping. The previously stated data requirements remain; the matched Avila tensor calculations were not repeated.

## Inventory-driven static beta, orbital magnetizability and CIS excitation tranche

This completed tranche adds T034 (static first hyperpolarizability), T081 (orbital magnetizability) and T021 (vertical molecular excitation energies), following the170-row inventory. It does not promote the other catalogue cells to validated status. Existing full HF electric-tensor acceptance is in the “Complete static beta and gamma tensors” section of [response.md](response.md); ordinary restricted singlet TDA and magnetic-response subsets have their separate acceptance records there. Physical comparisons below are restricted to the actual named references, basis, observable and frequency. No new property route, production code or reference fixture was introduced.

### Static first hyperpolarizability of water

The fixed literal published geometry is rOH=0.957Angstrom, angle104.5degrees, oxygen at zero and hydrogens symmetric in the yz plane. Calculations use neutral singlet HF or PBE/STO-3G, zero electronic temperature, no solvent, and clamped nuclei. The public analytic variational2n+1 route evaluates all27 raw Cartesian beta entries from three first-order responses. Convention: E(F)=E0-mu.F-alpha.F.F/2-beta.F.F.F/6+..., and beta_bar is the norm of the vector sum_j(beta_ijj+beta_jij+beta_jji)/3; no EFISH3/5 or factorial rescaling is inserted.

The source is Liu, Peng and Gu, [author preprint DOI10.20944/preprints202510.1887.v1](https://www.preprints.org/manuscript/202510.1887), Section4.1, Eq98 and Table1. A peer-reviewed version is linked as Chemistry7,181(2025), DOI10.3390/chemistry7060181, but its direct body returned429; the numbers here are explicitly attributed to the accessed author version rather than assuming unchanged publication tables. Source SCF/response thresholds are1e-11/1e-9 and DFT grid99x590. Native thresholds are1e-13 energy,1e-11 orbital/response and1e-12 density. Native named Fine75x302 and UltraFine99x590 use their own radial mapping/pruning; a matching grid name is not point identity.

| Method/grid | Calculated beta_bar (au) | Primary author value (au) | Difference (au) | Relative difference |
| --- | ---: | ---: | ---: | ---: |
| HF/ultrafine | 9.715641404 | 9.724434 | -0.008792596 | -0.090418% |
| PBE/ultrafine | 9.484697182 | 9.490396 | -0.005698818 | -0.060048% |
| PBE/fine | 9.484698548 | 9.490396 | -0.005697452 | -0.060034% |

These small source differences are retained rather than called exact reproduction. PBE Fine-to-UltraFine changes the norm by only1.36629e-6au, far less than its source offset. Independent field-differentiated SCF dipoles at the identical literal geometry test every tensor entry at four field steps0.004,0.002,0.001,0.0005au. These are independent point oracles on the symmetric source geometry, not replacement complete-C1 acceptance fits. All raw components, including zeros and cancellation noise, remain in the JSON; no step is discarded or floored.

| Independent oracle | Raw maximum tensor errors at the four steps (au) | Finest-two Richardson maximum tensor error (au) |
| --- | --- | ---: |
| HF | 3.62785e-04, 9.05531e-05, 2.25150e-05, 5.50526e-06 | 2.99421e-06 |
| PBE | 3.50328e-04, 8.96828e-05, 1.99417e-04, 7.77432e-04 | 1.04767e-03 |

The PBE fine0.001/0.0005au errors increase; those points remain explicitly unresolved because dipole/SCF cancellation noise is amplified by division by h-squared. Two new coarse points0.016/0.008au and the retained0.004/0.002 samples give maximum-error exponent p=1.994406, R-squared=0.99997678, and Richardson maximum tensor error 5.62360e-05au. No noisy fine point is discarded. This scalar maximum-error diagnostic on a symmetric physical geometry is not a complete C1 component-acceptance certificate.

A separate precision diagnostic samples the corners of rOH=0.957+/-0.0005Angstrom and angle104.5+/-0.05degrees, assuming ordinary rounding of the printed geometry, and also the pre-existing0.9572Angstrom/104.52degree structure. These are explicitly labelled additional geometries, not fitted replacements for the source or new experimental uncertainties.

| Geometry rOH / angle | HF beta_bar (au) | HF alpha isotropic (au) | HF common-origin xi isotropic (au) |
| --- | ---: | ---: | ---: |
| 0.9570 / 104.50 | 9.715641404 | 2.453847251 | -2.649803887 |
| 0.9572 / 104.52 | 9.720354385 | 2.455279393 | -2.649917933 |
| 0.9565 / 104.45 | 9.703866572 | 2.450270767 | -2.649518079 |
| 0.9565 / 104.55 | 9.696052397 | 2.450068876 | -2.649332059 |
| 0.9575 / 104.45 | 9.735225848 | 2.457628147 | -2.650275417 |
| 0.9575 / 104.55 | 9.727427120 | 2.457429267 | -2.650088705 |

**Interpretation:** independent agreement at the literal geometry, the small PBE grid tail and the measured geometry sensitivity distinguish implementation behavior from source-condition compatibility. The sampled range is a sensitivity diagnostic, not a rigorous uncertainty interval or proof of the authors’ actual coordinates. Full-precision Cartesian inputs and their grid definition would be required for an exact historical reconstruction. No reference or geometry was adjusted to force agreement.

### Static orbital magnetizability and gauge distinction

The same fixed geometry and SCF models were evaluated with the common-origin gauge at oxygen(0,0,0)Bohr and separately with London orbitals. The sign convention is xi_ab=-d2E/(dB_a dB_b), isotropic trace/3, units e^2 a0^2/m_e. Spin Zeeman and thermal/Curie terms are excluded. The primary author preprint Table2 gives static HF-2.650016 and PBE-2.611421au. Its archived magnetic input and exact origin/basis-phase settings were not retrieved; the near common-origin agreement identifies a compatible comparison, not proof of identical gauge inputs. The London results are retained as different finite-basis magnetic models rather than compared as an implementation error against a common-origin candidate.

| Method/gauge | Diamagnetic (au) | Paramagnetic (au) | Total isotropic (au) | Source total (au) | Difference (au) |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF/london | not reported | not reported | -2.459838295 | -2.650016 | +0.190177705 |
| HF/common-origin | -3.003557752 | 0.353753865 | -2.649803887 | -2.650016 | +0.000212113 |
| PBE/common-origin | -3.009844762 | 0.398560602 | -2.611284160 | -2.611421 | +0.000136840 |
| PBE/london | not reported | not reported | -2.402271335 | -2.611421 | +0.209149665 |

All successful reports require their fresh response convergence and retain raw asymmetry and residuals. Negative isotropic values indicate diamagnetism and have plausible magnitude. Finite common-origin bases retain origin dependence, while London orbitals address origin dependence through field-dependent bases; the two approaches need not agree in a minimal basis. The HF geometry-sensitivity rows above also quantify the source-coordinate precision issue for the common-origin result. An initial harness call used the invalid keyword `common`; the typed error was preserved, and only the documented `common-origin` spelling was retried. This is not a production physics defect.

### Formaldehyde vertical singlet CIS excitation

The public `excitations` route uses ordinary restricted HF singlet TDA(CIS), STO-3G, four roots and tolerance1e-11. No geometry optimization or excited-state relaxation is mixed into the vertical energy. The fixed primary MP2/6-311++G** ground-state geometry is CO1.2122Angstrom, CH1.1044Angstrom, HCO121.94degrees, in the yz plane with z along CO. Foresman, Head-Gordon, Pople and Frisch, JPC96,135–149(1992), [DOI10.1021/j100180a030](https://doi.org/10.1021/j100180a030), Section4B and TableI p141, gives the lowest singlet1A2(n-pi*) CIS/STO-3G energy4.249eV. The accessed public copy supplies the original text/table. Its experimental column is not treated as an independently retrieved primary experiment.

The paper generally freezes occupied core orbitals in its vertical CIS space; the current public route includes all occupied orbitals. An independent dense CIS matrix therefore checks both the full space and the space excluding the two lowest occupied orbitals(O1s/C1s). At this C2v minimal basis the first A2 excitation is unaffected by that exclusion; the measured difference is retained below instead of assuming universal core insensitivity.

| Root | Native full-space CIS (eV) | Independent full-space CIS (eV) | Independent frozen-core CIS (eV) |
| --- | ---: | ---: | ---: |
| 1 | 4.248913083 | 4.248912843 | 4.248912843 |
| 2 | 9.426380190 | 9.426379963 | 9.426842615 |
| 3 | 12.273769920 | 12.273769896 | 12.274387452 |
| 4 | 13.510779996 | 13.510780140 | 13.510780140 |

The first native state differs from the primary4.249eV value by -0.000086917eV; the first full-minus-frozen result is -8.308e-15eV. Across four roots, native versus independent full-space energies differ by at most 2.395e-07eV. State oscillator strengths and dominant transitions are retained in the report; no symmetry-forbidden intensity is imposed as an artificial zero. This is a numerical and source-compatible CIS comparison, not proof of experimental accuracy or general frozen-core equivalence.

### Runtime diagnosis, recovery and final state

The frozen comparison artifact uses compiler opt-level1. PBE UltraFine beta took316.643s, while its SCF took1.741s; Fine beta took229.414s. This is response work after SCF, not SCF divergence. Code inspection of the immutable source finds30 second-order Fock evaluations from nine ordered beta field-pair contractions; exactly zero second-order density still triggers two linear GGA grid contractions per evaluation that cancel. The nonlinear grid/AO/Taylor paths are rebuilt, and seven-channel dual Taylor evaluation repeatedly allocates its per-point series arena. This explains a CPU-bound, low-persistent-memory pattern. These are measured total/SCF times and code-call counts; no sampled timing fraction for individual routines is claimed.

Frozen-source anchors are `src/response/electric_tensor/variational.rs:126-155`, `src/response/nonlinear.rs:350,372,518-564`, `src/response/mod.rs:498-523,615`, `src/xc/taylor.rs:202`, `src/xc/gga.rs:67` and `src/response/series.rs:123-147`. A zero-density shortcut and normalized quadratic-direction reuse are potential owner-reviewed optimizations; they require unchanged full-component/regression validation and are not production edits in this comparison.

The slower PBE London magnetizability uses six full second-order complex field paths(three axes plus three mixed directions), solving both first- and second-order orbital stationarity. It is not an accidental fourth-order request. Each direction rebuilds geometry/integral/point caches, performs grid passes for Fock orders0/1/2 and for energy, with additional Krylov/refinement work. This is at least24 base Taylor-grid passes before those iterations. HF has no semilocal grid contribution, and the common-origin command instead uses the analytic first-order magnetic response. Frozen anchors are `src/driver/magnetizability.rs:132,345` and `src/response/field_path.rs:559-580,614,754,871-928,1798,2054-2097`. This call-path difference explains why a small AO basis can still have a long PBE London runtime; no per-routine timing fraction was measured.

A transport outage left the original PBE beta call without a terminal report; after reconnecting, its process was verified absent before any restart. Original partial JSON/logs remain under `transport-interrupted-*`. The detached resume supervisor preserves completed HF results, resumes only unfinished work and records exit0. The queued independent-oracle and CIS jobs start only after that one-core job ends and both record exit0. Initial invalid-gauge input and interrupted execution are retained separately from converged scientific results.

Reproduce with the retained scripts `.work/validation/molecular_nonlinear_magnetic_20261031.py`, `diagnose_molecular_beta_20261031.py` and `molecular_cis_benchmark_20261031.py`. Full reports, all raw finite-field samples, geometry controls, sources, immutable extension hashes, performance evidence and durable journals are in `nonlinear-magnetic-tranche/` and `excitation-tranche/`. All owned jobs are terminal. No confirmed production defect, gate relaxation, fixture change, additional docs file, package installation or publication occurred. Unmatched historical inputs remain provenance limitations rather than silently accepted conditions.

## UV-Vis normalization, achiral CD and expanded scope

The four formaldehyde vertical singlet CIS states above were evaluated through public `ecd-spectrum` with4001energy points and Gaussian standard deviation0.2eV(FWHM0.470964eV). Molar absorptivity is L mol^-1 cm^-1. This is an electronically broadened stick spectrum without vibronic structure, solvent or temperature averaging, not a fit to experimental band maxima. The lowest n-pi-star transition is electric-dipole forbidden in the planar model; its tiny computed oscillator strength does not predict the experimentally vibronically activated band intensity.

| Root | Excitation (eV) | Wavelength (nm) | Native oscillator strength | Independent oscillator strength |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4.248913083 | 291.802153 | 2.0402739509e-32 | 1.0204623782e-29 |
| 2 | 9.426380190 | 131.528960 | 1.1329699267e-02 | 1.1329699428e-02 |
| 3 | 12.273769920 | 101.015580 | 3.1017594232e-01 | 3.1017594215e-01 |
| 4 | 13.510779996 | 91.766869 | 1.4176092591e-29 | 3.1627762639e-28 |

The selected oscillator sum is0.321505641586648, and integration of the broadened spectrum gives0.321505641586648, with the reported4.319e-9 conversion from the molar-absorptivity integral over cm^-1. This is a normalization check of the four selected roots, not a complete Thomas-Reiche-Kuhn sum-rule test of all16electrons or experimental cross-section agreement. Native and independent oscillator strengths are compared explicitly above.

Planar achiral formaldehyde supplies a CD limiting case: maximum absolute CD is9.0625e-13(length gauge) and1.8286e-13(velocity gauge) in L mol^-1 cm^-1, with raw rotatory sums and moments preserved rather than replaced by zeros. An active chiral molecule and a source-matched measured CD spectrum remain separate comparisons. The state report returns transition angular momentum alongside electric moments; that is not represented as validation of a permanent magnetic dipole moment.

The user expanded the programme to CD, MCD, Cotton-Mouton, beta/gamma, hyper-Raman, magnetic moments, magnetic/higher susceptibilities, nuclear spin-spin couplings, refractive index, absorption/transmission/reflection, Pockels/Kerr and VPT2. The existing170-row inventory remains the coverage source. Current completion applies to the explicitly computed slices above, not to all these targets. A nonzero molecular beta norm alone does not establish a bulk Pockels effect: orientational averaging and material symmetry matter. Bulk optical quantities require an explicit number density, frequency, phase, thickness, interfaces and local-field model; an isolated-molecule result will not be labelled a universal refractive index or transmission.

The completed VPT2 tranche below includes a full-mode water case and an explicitly reduced resonant formaldehyde case. Reduced-mode results are not represented as full-mode VPT2 predictions.


## Molecular anharmonic bands: strict VPT2 versus retained-state mixing

Three public `optimize`/`anharmonic` calculations completed on the immutable `edf2-integration-20261030` Python extension (SHA256 `8d59be3c2175aca4f2a34fdccd885a4a04e5e247a17bc23c0c9f1c62f2d3039a`). H2/STO-3G uses its complete single vibrational mode; water/6-31G uses all three; formaldehyde/STO-3G explicitly retains only mode IDs2(scissor) and4(symmetric CH stretch). The last case reuses the earlier public stationary geometry and omits couplings to the other four modes. It is a reduced Hamiltonian resonance diagnostic, not a full formaldehyde fundamental prediction.

All use restricted HF, isolated gas-phase stationary structures, zero temperature, no empirical scaling, fixed average atomic masses H1.008/C12.011/O15.999u, and one numerical core. Existing bundled STO-3G/6-31G definitions are unchanged; their recorded provenance is BSE with original-paper citations, not a newly transcribed paper/SI basis. No software-exported basis was used or added. Public optimization uses gtol1e-8; SCF energy/gradient/density tolerances are1e-12/1e-10/1e-11. Response tolerance is1e-10, maximum500iterations. Maximum forces are2.60e-10/1.78e-9/4.81e-8Ha/Bohr for H2/water/formaldehyde. Full coordinates and every input are retained in raw JSON.

The projected, electron-mass-weighted normal-coordinate arrays contain actual cubic/quartic derivatives; the Hamiltonian applies factors1/3! and1/4!. Rotational projection, inverse inertia, Coriolis couplings and the rotor-specific Watson constant are retained. H2 is linear with zero Watson/Coriolis terms; water is nonlinear with Watson constant-6.3991108794e-5Ha, Eckart defect5.10e-16 and metric defect2.22e-16. Resonance controls are1e-8Ha with coupling factor1, maximum1024basis states and256retained states. No small denominator was replaced by a floor.

### Water: primary same-method/basis comparison

Buczek, Kupka and Broda, J Mol Model17,2029-2040(2011), [DOI10.1007/s00894-010-0913-3](https://doi.org/10.1007/s00894-010-0913-3), Table3 p2037, reports RHF/6-31G harmonic and PT2 differences from its experimental column; adding those signed differences gives the source frequencies below. The authors separately optimize each model. Full-precision source coordinates, isotope masses and resonance thresholds are unavailable; the table is integer precision. This is a primary theoretical comparison, not independent retrieval of the experiments cited by that paper.

| Water mode | Native harmonic | Source harmonic | Native retained-state band | Independent strict VPT2 | Source PT2 | Strict minus source | Native IR (km/mol) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bend |1736.8524|1737|1671.3055|1669.7288|1670|-0.2712|122.6381|
| Symmetric OH stretch |3988.1511|3989|3726.6161|3816.6422|3817|-0.3578|1.11217|
| Antisymmetric OH stretch |4145.0600|4145|3877.6087|3956.2996|3956|+0.2996|40.4083|

All frequency columns are cm^-1. Independent strict VPT2 uses ladder-operator matrix elements from the returned tensors: first-order quartic plus rotational kinetic contributions and second-order cubic sums over physical harmonic denominators. It contains no retained-space higher-order diagonalization. Its three frequencies agree with the primary PT2 table within0.36cm^-1. This strongly supports the actual force/rotation tensor conventions for this case; unavailable historical inputs preclude an exact digit-for-digit reconstruction. The published source supplies no matched anharmonic intensities, so the IR values remain computed predictions rather than source-validated intensities.

The public routine diagonalizes a retained effective Hamiltonian, partially resumming internal mixing beyond strict second order. It therefore gives stretch bands90.3839 and78.3913cm^-1 below the primary PT2 values despite the strict agreement. The returned exterior mixing norm0.69231 explicitly warns that truncation needs independent convergence checks. Numerical response convergence is not physical validity of this resummation. These native stretch values remain scientifically questionable as conventional VPT2 predictions; they are not accepted by relabelling the primary comparison. A strict-order output option or independently converged resonance treatment is an owner-reviewed next step, not a change made to force agreement here.

### Hydrogen: independent diagnosis of anomalous level spacing

| H2/STO-3G quantity | Native retained model | Independent strict VPT2 | Converged full quartic Taylor model |
| --- | ---: | ---: | ---: |
| Fundamental (cm^-1) |5165.66461|5279.79868|5376.48955|
| Overtone from vacuum (cm^-1) |10472.91844|10358.62786|10770.53155|
| Overtone minus twice fundamental (cm^-1) |+141.58922|-200.96950|+17.55245|

The harmonic is5480.76817cm^-1; returned Phi3=6.5810309907e-5 and Phi4=7.0067861083e-6 in the recorded mass-weighted atomic units. An independent three-state retained Hamiltonian agrees with the native matrix to3.47e-18Ha. Independent strict matrix elements and the closed form agree to6.94e-18Ha: E_n=w(n+1/2)+Phi4(2n^2+2n+1)/(32w^2)-Phi3^2(30n^2+30n+11)/(288w^4). Scaling the cubic by lambda and quartic/second-order terms by lambda^2 gives a retained-minus-strict remainder with four-step exponent4.00150 and R-squared0.999999979. This identifies a higher-order resummation difference rather than a factorial/sign error in the tested Hamiltonian.

Direct diagonalization of the same quartic polynomial with12/24/48/96/192oscillator functions converges to the last column (24-to192 fundamental change below1e-9cm^-1). This is convergence of the local quartic polynomial, not the full molecular potential. Its positive spacing curvature shows that the unusual native sign cannot simply be attributed to a production algebra bug or solely to basis incompleteness. Native exterior mixing0.66957 and its warning, the114.13407cm^-1 strict/native fundamental separation, and the211cm^-1 native/quartic separation all prevent claiming a reliable physical anharmonic prediction at full interaction strength.

Dickenson et al., PRL110,193601(2013), [DOI10.1103/PhysRevLett.110.193601](https://doi.org/10.1103/PhysRevLett.110.193601), [author PDF](https://www.fuw.edu.pl/~krp/papers/vib_splitting_prl.pdf), TableII, gives para-1H2 X-state J=0 fundamental4161.16632(18)cm^-1 from collisionless molecular-beam laser combination differences. Its BO theory is4163.40350cm^-1; nonadiabatic/relativistic/QED corrections matter at precision scale. Our average-mass HF/minimal-basis native and strict results exceed experiment by1004.49829(24.14%) and1118.63236(26.88%)cm^-1. These are inadequate quantitative molecular models, not precision experimental validation; the mass/electronic/local-potential and resummation differences are explicit.

H2 native fundamental/overtone IR intensities are1.9082e-29/1.5959e-29km/mol, consistent with the electric-dipole-forbidden limit without imposing artificial zeros. STATIC Placzek Raman activities are21.92586/0.153283Angstrom^4/amu. They are not dynamic Raman cross sections and have no matched primary intensity benchmark in this tranche.

### Formaldehyde: explicitly reduced resonant model

| Dominant selected occupation (scissor,stretch) | Band (cm^-1) | Native IR (km/mol) | Dominant weight |
| --- | ---: | ---: | ---: |
| (1,0) |1761.45244|4.38169|0.99755|
| (0,1) |3437.49113|0.851462|0.88177|
| (2,0) |3536.91336|0.487131|0.90750|
| (1,1) |5167.74482|0.0824675|0.84857|

The harmonic stretch3498.38443 lies near twice the scissor1767.13371cm^-1. The returned effective stretch/overtone matrix coupling is31.49573cm^-1 and diagonal detuning-1.35062cm^-1. The eigenstates have different weights because this is a nine-state retained problem, not an isolated two-state avoided crossing: the stretch-labelled state contains8.944% overtone weight, and the overtone-labelled state contains8.634% stretch weight. Three additional resonant states are retained; exterior mixing0.31839 triggers the same validity warning. The independent strict reduced-model fundamentals1761.44590/3423.18577cm^-1 are diagnostics, not the correct nonperturbative treatment of the near resonance. Unselected-mode coupling and matched full-mode primary intensity/fundamental references remain unresolved. The source RHF/6-31G formaldehyde table is deliberately not treated as equality evidence for this STO-3G two-mode calculation.

### Evidence, recovery and terminal resources

Raw inputs, full arrays, bands, residuals and warnings are under `.work/validation/molecular-property-benchmark-20261031/vpt2-tranche/`. Reproduce with `.work/validation/molecular_vpt2_benchmark_20261031.py` using the frozen Python artifact, then the independent scripts `diagnose_vpt2_hydrogen_20261031.py` and `diagnose_vpt2_strict_levels_20261031.py`. Two initial calls failed on an unsupported harness keyword before optimization; the original logs/results are archived as `*-initial-api-mismatch-*`, and only the harness was corrected. No production/reference edit or gate relaxation occurred.

Water took276.400s worker wall/257.443s CPU, formaldehyde121.355/116.496s, and H2 1.166/0.717s; worker peak RSS was below93MiB. Both independent tensor diagnostics took below0.1s internally. Every owned molecular supervisor, worker and diagnostic is terminal exit0; the one-core slot is released. No heavy build, additional docs file or new basis definition was created. The principal finding is a source-compatible strict VPT2 water force field alongside consequential retained-space resummation/validity limitations; neither native convergence nor a low matrix residual is presented as universal molecular accuracy.

### Public selection of strict VPT2 and retained-polyad treatments

The observed method distinction is now implemented as an explicit selector. Rust uses `AnharmonicConfig.method` with `AnharmonicMethod::StrictVpt2` or `RetainedPolyad`. Native Python and ASE accept `method="strict-vpt2"` or `"retained-polyad"`; the CLI accepts `--anharmonic-method`. The default remains retained-polyad, preserving existing scientific outputs. Unknown method names fail explicitly. Strict nonresonant VPT2 rejects coupled near resonances, including indirect second-order coupling, rather than flooring denominators or silently switching methods.

The strict treatment includes normalized wavefunctions and linear/quadratic/cubic observable terms through second order. It reports the square of the corrected transition moment, not an intensity series truncated at second order. Method, energy/intensity/resonance treatment and eigensystem metadata distinguish diagonal perturbative levels from a retained-state eigensolution. Numerical convergence is separate from perturbative validity. Force-constant generation, reference data and existing derivative gates are unchanged.

The isolated core suite passes ten tests, including four unchanged retained-method regressions, coupled and uncoupled degeneracies, direct/indirect resonance rejection, all three dipole and nine polarizability components, phase covariance and the identical native water force/rotation fixture. Independent four-step comparisons give an energy exponent4.00107 and dipole-moment exponents4.01078/3.99836/3.99951, all R-squared above0.9999989. These fits check the expected higher-order remainder of the new perturbative treatment; they do not replace the existing force-constant derivative fits. The full isolated library/Python frontend passes with warnings denied. Runtime Python/ASE/both-CLI acceptance and the repeated public water calculation are tracked separately in the local strict-VPT2 integration record; they are not inferred from the isolated core pass.

The compiled isolated bundle now passes the actual Python, ASE and both CLI-name tests with warnings treated as errors: 6 tests passed, none skipped. Tests check default versus explicit retained outputs, identical force tensors across method choices, a closed-form one-mode strict frequency, method-specific metadata, strict resonance rejection and unknown-method errors. The portable-artifact audit also passes. The bundle is the previously accepted base plus only the anharmonic selector/core delta; it is not a claim that unrelated work in the live checkout has been release-certified.

The public strict API was then run at the previously stored HF/6-31G water minimum, without another geometry optimization. It gives the following actual fundamentals. The force tensors, normal modes and harmonic frequencies are compared directly with the retained-run inputs in the raw record. The paper values are those discussed above, not newly fitted targets.

| Treatment | Bend (cm^-1) | Symmetric stretch (cm^-1) | Antisymmetric stretch (cm^-1) |
| --- | ---: | ---: | ---: |
| Compatible retained-polyad | 1671.305513 | 3726.616133 | 3877.608691 |
| Standard strict-vpt2 | 1669.728785 | 3816.642162 | 3956.299630 |
| Published standard VPT2, rounded | 1670 | 3817 | 3956 |

The strict public result agrees with the independent identical-force calculation within2e-5cm^-1 and with the printed paper values within0.36cm^-1. This agreement concerns these frequencies and stated conditions; it does not establish experimental accuracy or validate all intensities, resonances or functional/basis combinations. Existing derivative thresholds and source references remain unchanged. Reproduction commands, source hashes, artifact hashes, raw reports, warnings and test logs are in the local `strict-vpt2-integration-20261031` validation record.

The public water acceptance took764.71seconds wall time under concurrent work. At the recorded CPU-affinity move, its process had accumulated199.52CPU seconds; both owned threads were moved from contended CPU14 to the released CPU15, without changing scientific inputs or other jobs. The result file was written122.00seconds later. The before/after affinity record is retained in `water-resource-adjustment.json`; this is evidence of scheduling contention, not a measured kernel timing fraction. All compared force tensors and modes remain exactly equal to the earlier run.



## Static second hyperpolarizability of hydrogen fluoride

The public static `gamma` route was evaluated for neutral singlet HF with HF/aug-cc-pVDZ,32spherical AOs, F at the origin and H on+z at a declared rHF=0.9168Angstrom. This is a chosen fixed geometry; the exact experimental separation used in the source paper was not retrieved. No solvent, damping, thermal or vibrational correction is included. The unscaled tensor convention is E(F)=E(0)-mu.F-alpha:FF/2-beta:FFF/6-gamma:FFFF/24; gamma uses e^4 a0^4/Eh^3 atomic units. All81tensor components are retained in the raw record.

Avramopoulos et al., JCTC6,3365(2010), [DOI10.1021/ct100471k](https://doi.org/10.1021/ct100471k), Eq1 and Table8 p3370, give the following static longitudinal values. Table8 footnote b only states that experimental geometry was used; its reference54 is Cade and Huo, JCP47,614(1967), [DOI10.1063/1.1711938](https://doi.org/10.1063/1.1711938). The accessible original-paper endpoint did not expose that numerical distance. Axis reversal does not change gamma_zzzz; signed odd-rank moments require the source atom orientation, which also remains unknown.

| Quantity/model | Computed (a.u.) | Published (a.u.) | Computed minus published | Relative difference |
| --- | ---: | ---: | ---: | ---: |
| gamma_zzzz, HF/aug-cc-pVDZ | 219.6799648 | 219 | +0.6799648 | +0.3105% |
| gamma_zzzz, comparison to CCSD(T) | 219.6799648 | 284 | -64.3200352 | -22.6479% |

The second row deliberately compares different electronic models and is not an implementation-error measurement. The HF difference is small and physically plausible, but exceeds a half-unit rounding interval if the printed219is interpreted as nearest-integer rounding. Missing exact source coordinates and calculation details prevent an exact-reproduction claim. The native isotropic contraction is160.808880996au and mu_z=+0.759609625au; the latter's sign describes only our stated axis orientation. No parameter or fixture was adjusted to match the paper.

Maximum relative response residual is6.61e-11 across504reported solves (maximum14iterations), field-permutation defect7.11e-15, energy-permutation defect6.07e-10, projector defect2.22e-16, stationarity residual7.49e-10 and number defect4.76e-14. No warnings were returned. Wall/CPU time is303.55/248.74seconds, peak RSS75.76MiB on one core. Code inspection identifies168directional third-order response calculations for the27ordered field triples, with no cache shared across triples; this explains the workload but is not a measured timing fraction or an additional derivative certificate.

Raw controls, geometry, immutable artifact identity, full tensor, diagnostics and execution log are in `molecular-property-benchmark-20261031/gamma-tranche/`. The reusable driver is the local `run_gamma.py` in that directory. The calculation is terminal. This single static HF comparison does not certify frequency-dependent gamma, bulk Kerr coefficients or other functional/reference combinations.


## Nine-crystal periodic physical benchmark: initial campaign

This additional campaign is in progress. Prepared structures, started jobs,
SCF convergence, numerical convergence and physical validation are separate
statuses. The nine cases are seven inorganic crystals, one urea molecular
crystal and one ferrocene organometallic-complex crystal. No isolated molecule
or finite cluster substitutes for a crystal. The seven inorganic structures
use two-atom primitive fcc cells with their ideal A4/B3/B1 basis positions.

| Crystal | Phase | Static reference cubic a (Angstrom) | Experimental bulk target (GPa) |
|---|---|---:|---:|
| Diamond | A4 | 3.543 | 443 |
| Silicon | A4 | 5.416 | 99.2 |
| 3C-SiC | B3 | 4.342 | 225 |
| LiF | B1 | 3.964 | 69.8 |
| NaF | B1 | 4.579 | 51.4 |
| NaCl | B1 | 5.565 | 26.6 |
| MgO | B1 | 4.184 | 165 |

The targets and phase definitions are from Tables II and V of
[Csonka et al., Phys. Rev. B 79, 155107 (2009)](https://doi.org/10.1103/PhysRevB.79.155107),
also available in the [author manuscript](https://arxiv.org/abs/0903.4037).
The lattice targets remove thermal and zero-point expansion; the bulk targets
include thermal corrections but do not generally remove zero-point effects.
These are reference targets, not lattice constants or moduli computed here.

Urea uses the full 16-atom P -4 21 m cell at 12 K, a=b=5.565 and c=4.684
Angstrom, from [the neutron study](https://doi.org/10.1107/S0108768184002135)
via [COD 9017318](https://www.crystallography.net/cod/9017318.html).
Ferrocene uses the full 42-atom P 21/a mean-coordinate rigid-body model,
a=10.443(5), b=7.572(4), c=5.824(4) Angstrom and beta=120.95(8) degrees,
from [Brock and Fu (1997)](https://doi.org/10.1107/S0108768197005132)
via [COD 2101932](https://www.crystallography.net/cod/2101932.html).
This is a static full-occupancy approximation to the orientationally
disordered monoclinic phase. Its CIF omits the measurement temperature; it
must not be labelled the ordered 98 K polymorph or a thermal disorder
ensemble. Attempts to retrieve the ordered structure did not provide an
accessible structure file. The ferrocene CIF's exact Hall-group operations
are written explicitly in a private derived file to resolve ASE's unsupported
long space-group spelling; the original CIF remains unchanged. Expanded
cells have exactly C2H8N4O2 and C20H20Fe2 compositions, respectively.

The initial diagnostic uses the frozen EDF2 integration artifact already
identified above, restricted neutral singlet all-electron PBE/STO-3G, no ECP,
no dispersion, no external fields, zero electronic temperature, Gamma mesh,
AO overlap reach 12 Bohr, grid level 2, two Becke shells and zero grid-weight
cutoff. SCF energy/density-change gates are 1e-10 Hartree/1e-8, with 150
maximum iterations. A Gamma-only minimal-basis single point is not a
converged material prediction. Urea and ferrocene lattice cohesion will also
require a dispersion model and broader-basis/cutoff checks before a
quantitative lattice comparison is meaningful. No Kohn-Sham gap is compared
with an optical or quasiparticle gap, and absolute total energies from
different methods are not treated as comparable observables.

Local nondistributed inputs, source CIFs, full geometry/controls, frozen
extension hashes, stdout/stderr, elapsed times and per-case status are in
`.work/validation/crystal-benchmark-20261031/`. The serial supervisor is
`.work/validation/crystal_benchmark_20261031.py`; invoke it with the existing
validation Python environment. `--tag baseline --timeout 1800` runs the seven
inorganic cases and `--tag baseline --molecular --timeout 1800` runs the two
molecular crystals. The first short-limit diamond attempt timed out after
240.04 s and is retained. The restarted diamond SCF has not yet produced a
completed report at the time of this initial record. One job runs at a time;
the initial diamond job uses one numerical thread and subsequent workers use
at most two Rayon threads/one linear-algebra thread. A separate owned-worker
RSS guard records and terminates any benchmark worker exceeding 1.5 GiB;
it never signals peer jobs. A timeout or memory termination is not convergence.

The periodic DFPT audit is saved privately as `dfpt-gap-map.json` beside the
raw results. Static/retarded/imaginary-frequency finite-q density response,
uniform Mermin temperature/chemical-potential/number response and orbital
scalar/vector Fourier-field response already have distinct public routes.
They do not establish periodic real-time propagation, which remains pending
in T164/R074. Meta-GGA covariant kinetic-density current/contact terms and
ECP nonlocal-core field vertices remain separate missing connections.
Full/range-separated exchange screening is already connected for matching
HF/hybrid references; its existence is distinct from complete hybrid current
physical acceptance. The original field report's LDA/GGA-only model label
was stale and is corrected without introducing an unjustified hybrid guard.

The changed field module, adapted only to public Cell accessors for external
linkage, compiles with warnings as errors against accepted23 dependencies.
All five p59 tests pass, including the new matching-PBE0/CAM-B3LYP uniform
scalar-gauge test. Original independent nonlinear field fits and thresholds
are unchanged: four-step exponents 1.999949222058..1.999961514747 and minimum
R-squared 0.999999999827. Original dense-screening residuals are
1.462869748124e-16..2.461094815816e-16, while their raw continuity defects
remain 0.02244206806315..0.02244898773735. The latter are retained finite-basis
defects, not projected away or certified by the small solver residuals.
This is a narrow module acceptance, not a full live-tree library/package
build. Source/library hashes and exact scope are in
`patched-field-acceptance.json`; p33/p37 thermal/dynamic rechecks are recorded
separately. No completed nine-crystal physical acceptance is claimed.

## Molecular DFPT coverage boundaries

A source/API/ledger audit covers13molecular response categories and23target rows, not the complete170-item programme. Here DFPT means a self-consistent perturbation response around a declared electronic reference; a Fourier-domain equation is not by itself a real-time propagation API. Periodic crystal coverage is audited separately.

| Molecular response category | Actual boundary established by the audit |
| --- | --- |
| Static integer-occupation electric/nuclear response | Implemented with substantial property/provider-specific acceptance; no universal higher-order acceptance follows from the shared solver. |
| Static finite-temperature response | Implemented Mermin response includes population weights, diagonal Fermi derivatives, moving overlap/energy-weighted density and per-spin chemical-potential response under prescribed particle-number changes. Existing p165 records three passing tests; p135 records one static collinear spin-transfer test. |
| Finite-frequency alpha/beta/gamma | Integer-reference routes support the declared real/imaginary frequency and damping conventions with bounded acceptance. A separate collisionless finite-temperature alpha increment is implemented with explicit opt-in; its initial full runtime gate fails and the corrected integration remains pending. Thermal nonlinear beta/gamma and bath relaxation are not established. |
| Finite electric bias | Existing static/electric-response acceptance is explicitly reference-, derivative- and functional-specific. Bias support does not imply finite-temperature nonlinear dynamics. |
| Orbital magnetic/joint electric-magnetic response | Bounded integer-reference London-field and nuclear acceptance exists. Positive-temperature orbital magnetic references are explicitly unsupported in the audited paths. Static thermal spin response excludes orbital magnetism. |
| Magnetic optical response | Existing frequency-dependent magnetic Taylor paths have bounded HF acceptance, but accept real nonnegative omega rather than arbitrary complex damped frequency. They do not establish a general response about an arbitrary nonzero magnetic reference. |
| Correlated finite-frequency response | Accepted static target-plus-correlation and specific corrected-pole calculations do not supply the missing general causal correlated dynamic action kernel. |
| Real-time propagation | No production pulse/trajectory/time-integrator API is established for T164; a test-only time-dependent oracle is not a public implementation. |

The static thermal solver differentiates the free energy A=E-TS and enforces its stated number constraints. The conceptual route called grand-canonical returns documented Legendre/hardness/Fukui derivatives via prescribed particle-number directions; its name alone is neither a defect nor evidence of fixed-chemical-potential bath susceptibility. Collisionless frequency response, isothermal static occupation relaxation and a response coupled to a particle/thermal bath have different limits and must be specified before implementing a thermal dynamic extension.

The audit reproduced an input-boundary defect with a genuinely converged HF/STO-3G H2 reference at r=3Bohr,T=40000K and per-spin occupations0.852646964/0.147353036. The ordinary polarizability report rejects this reference, but the low-level TDA constructor/direct-amplitude path accepted it and returned an integer-projector response. That returned number is retained only as invalid-result evidence, not as a finite-temperature observable. A common constructor guard and a genuine finite-temperature/T0 regression were added to `src/response/tddft.rs` and the existing `tests/p25_dynamic_response.rs`. The isolated strict-VPT2 artifact predates this guard and cannot certify that edit. Initial isolated and metadata gates, exact commands, hashes and the original failure remain in the guard ownership and metadata records.

The immutable `fractional-dynamic-guard-integration-20261031` bundle closes that guard's runtime gate: all four linked p25 tests, six anharmonic native/ASE/CLI tests, ten reference-aware dynamic public tests and the explicit thermal-refusal/T0 public controls pass. Its library SHA256 is `d334d9a45017690cf924fe49423d1d6046f525d846784c5488ec73f60dbf057c`; Python extension is `272ecc7922e000db5eec9b3adbe7540089a05abc5b1d5aa0174c170b84177749`. The actual low-memory opt-level-0 build took238.02seconds, peaked at2978056KiB RSS and passed warnings-strict and portable audits; p25 took555.47seconds. Build settings changed resource use, not scientific tolerances or equations. This closes the invalid-input guard, not the separate new thermal dynamic model.

The new positive-temperature electric increment uses all ordered unequal-population MO pairs for each spin and an explicit `thermal_response="collisionless"` choice. It holds reference Fermi populations fixed and reports their temperature, chemical potentials, consistency and number defects; its zero-frequency limit differs from isothermal Mermin response. Local/GGA and exact-exchange parent kernels are the bounded target. Correlated composite, spinor, constrained common-orbital, kinetic/meta, nonlocal and continuum extensions are excluded. The default integer route keeps its guard.

The first actual new-library p309 gate has3 passing tests and1 nonconverged test. Independent dense comparison isolates the initial HF C1-ammonia failure to restarted, unpreconditioned GMRES: restart40 reaches500iterations at residual7.64e-9, while a full112-vector space gives residual2.91e-15 and density error2.33e-16. An exact free-Liouvillian right preconditioner was then added, retaining the requested restart40/max500 and fresh original-equation residual acceptance. Independent operator checks pass20 HF/PBE frequency/reference cases, four exact-zero diagonal cases and216 four-step electric-response fits (slopes1.980515–2.002256, minimum R-squared0.99997358), at most24iterations and residual9.78e-13. These are isolated mathematics checks, not completed actual-library acceptance of the correction. A separate public spinor-dispatch bypass was also corrected after retaining its failed refusal tests. The next actual integration waits for the reserved PBC and original-owner builds; neither failure was hidden by a looser tolerance.

The audit evidence is `molecular-property-benchmark-20261031/molecular-dfpt-scope-audit.json`, including source identities, raw historical acceptance logs, exact source locations, current surface mappings and six prioritized increments. The initial boundary failure and later fix records remain separate. Missing, unsupported and nonconverged cases must remain distinct from physical zeros in any future screening dataset; no latent-model or training implementation was started here.

### Measured molecular DFT response cost and bounded reuse

A clean terminal timing run uses C1 ammonia, PBE/STO-3G,8AOs,43326 grid points (level3, weight cutoff0), one CPU and the same opt-level-0 actual library throughout. SCF converges in11iterations,15.75CPU/17.20wall seconds. A symmetric A+B response action costs15.05CPU seconds; separately evaluating exactly its256-point batches of reference XC second derivatives costs14.33CPU seconds, a95.2% cost ratio. These separately timed stages are not an exclusive profiler partition. AO values/gradients take about0.15CPU seconds, reference-density/gradient assembly about0.01seconds; constructors cost0.07/0.11CPU seconds. Earlier contended samples and an expired supervisor wait are retained separately; only the short retry has a captured clean terminal state.

This measured duplication motivates a lazy per-reference local/GGA derivative cache, now written with unchanged contractions and streamed fallback. It retains only active derivative arrays, bounds stored data plus slot headers by min(64MiB, one eighth of the configured incore_mb memory budget), checks allocation arithmetic and array shapes, and does not cache AO values. Its immutable solver borrows prevent reuse across a changed reference, geometry, grid or functional. For this43326-point GGA grid, derivative data alone require6238944bytes. The first action still pays provider evaluation; expected gains concern subsequent actions/RHS. Full AO/spin equality, memory fallback and actual before/after timing remain required before claiming an accepted speedup. These opt-level-0 costs must not be extrapolated into a release-build speedup; optimized-build measurements are a separate performance gate. Raw baseline evidence is `response-timing-tranche/retry-brief`; the shared full-compiler handoff is recorded outside docs.

The bounded corrected snapshot passes warnings-strict full-library, actual cfg(test), p25 and latest p309 metadata compilation. This includes typechecking the complete all-AO cache regression; it does not execute that regression. All16 original HF/SVWN5/PBE/PBE0 closed/open-shell, geometry/grid fixtures independently converge on the preceding actual library, without omitted cases (maximum orbital residual1.50e-13). The metadata-only run's peak RSS is1771656KiB. Its long502.45second unit-typecheck wall time was traced to an explicit CPU14 affinity shared with PBC compilation; moving only the owned metadata threads to15 preserved progress and peer settings. All owned jobs are terminal. Actual corrected-artifact tests and cache performance remain queued after the parent-reserved PBC and original-owner compiler windows; no new full compiler was started ahead of them. Commands, source hashes, typecheck outcomes and affinity records are in `mcd-thermal-cache-metadata-20261031`.


### Periodic checkpoint: first completed SCF and DFPT rechecks

At 2026-10-31 04:29 UTC the first actual diamond PBE/STO-3G Gamma-cell
calculation has completed: E=-74.06683247966433 Hartree per two-atom cell,
six SCF iterations, 953.40 s inside the calculation (955.72 s supervisor
wall time). The independent final Fock rebuild reports orbital residual
6.963e-10 Hartree, density change 1.698e-11 and energy change 5.684e-14
Hartree. This is initial SCF convergence only; material-property acceptance
and k/basis/cutoff convergence remain open. Silicon is running, with the
remaining seven crystals queued. `status-summary.json` records these states.

To avoid another SCF for analytic stress, subsequent cases use the local
Rust reference/stress harness `crystal_measurements_worker_20261031.rs`,
linked to the frozen accepted23 library. Its SCF, screened-Hartree and
periodic-driver source files match the original EDF2 integration snapshot
byte for byte. The harness first saves the full completed SCF report, then
attempts analytic stress on the two-atom inorganic cells; a later stress
timeout cannot erase or certify the earlier SCF evidence. A periodic helium
interface check gives exactly zero energy and stress differences from the
public native route; `measurement-worker-ready.json` preserves both reports,
library/source/executable hashes and build command. This is a local
measurement harness, not a new package API or independent physical oracle.
Many-atom urea/ferrocene stress is explicitly deferred in this first bounded
run; no successful pressure is inferred from a saved SCF.

The original p33 finite-q frequency-domain density suite passes both tests
(20.73 s), and p37 uniform finite-temperature Mermin response passes its
independent displaced-temperature SCF/occupation/entropy/number checks
(111.69 s). The latter is the existing 40000 K H2 periodic fixture with
two temperature-difference scales and Richardson extrapolation, not a new
complete four-step thermal derivative acceptance or a 300 K material
prediction. Together with p59's five tests, eight existing/new narrow
periodic checks pass. The aligned API manuals were regenerated after the
corrected Python contract, and `tools/api_reference.py --check`, Python
warnings-as-errors compilation and changed-source rustfmt checks pass.

No numerical field kernel was changed. An initial diagnostic interpretation
that hybrid success implied a missing exchange-current model was withdrawn
after tracing the self-consistent exchange kernel and its gauge
transformation. The retained reproduction log's `UNSUPPORTED_ACCEPTED`
labels describe that withdrawn interpretation; they must not be treated as
a confirmed physical defect. No new hybrid refusal remains in the source.
The actual production change corrects stale result metadata; the new test
checks the uniform scalar gauge with matching PBE0/CAM-B3LYP parents.

## MCD permanent-dipole defect and explicit Faraday conventions

A same-basis independent fixed-orbital CIS comparison for distorted water (HF/STO-3G, complete10-state singlet manifold) identifies a genuine MCD B-term defect. Zero-field excitation energies and transition strengths agree within6e-13, while every B term disagrees; the largest defect is0.114045222852atomic units. For one state the old native result is -0.290020473310 versus -0.404065696162 from the independent field derivative. These are fixed-CIS-space response quantities, not experimental band strengths or relaxed finite-field SCF derivatives.

Every discrepancy equals the omitted ground-state permanent-dipole counterterm within4.70e-12. All30 state/axis derivatives and all10 cyclic sums pass independent four-step second-order fits, with refined slopes1.999915–1.999991. Adding a scalar identity to the electric-dipole operator changes the independent derivative by at most2.22e-16, exposing why a permanent-dipole addition must cancel. The production correction removes that extra addition from an already ground-subtracted diagonal state-to-state moment. Initial failed values and independent-helper source identities remain in `molecular-property-benchmark-20261031/mcd-tranche`.

Faraday A conventions were separately checked against [Faber et al., JCP153,114105(2020)](https://doi.org/10.1063/5.0013398), Eqs3–5, and a helium P manifold with magnetic slopes -1/2,0,+1/2. The canonical first-level A sum is +1.0172751278atomic units. Compatible legacy `a_terms` retains its opposite signed first-line moment; new `faraday_a_terms=-a_terms` is the canonical coefficient multiplying a derivative in observation frequency. The B correction is a bug fix, whereas this A distinction is explicit representation metadata. Orientation averages, circular-polarization factors and broadening conventions must still be supplied to construct an experimental spectrum.

The complete old source fails the new regression as expected. The actual rebuilt library in `mcd-thermal-integration-20261031` passes all three focused Faraday tests, including40 four-step fits (slopes1.998184–2.000704, minimum R-squared0.9999997785) and the previous symmetry/degenerate-rotation checks. Native/common/ASE/both-CLI comparisons pass all10 reports; independent differences are at most4.50e-12 for water B and3.05e-13 for helium A. The MCD correction is accepted on this artifact, while its separate new thermal-response gate fails and must not be represented as whole-bundle acceptance. No threshold, reference fixture or basis coefficient was changed to obtain agreement. Exact commands/hashes and the retained initial failures are in `mcd-tranche/correction-final-results.json`.

## Helium magneto-optics: basis dependence and an independent field check

The public Cotton-Mouton route was evaluated for a neutral singlet He atom, HF, clamped nuclei, electronic temperature0, incident wavelength1064nm and undamped frequency45.5633525316/1064Hartree. For a single center at the origin, the London phase is unity and agrees with the common-origin convention. All81 mixed electric/magnetic tensor components were retained. The gas comparison uses273.15K and pressure1atm, separately from the electronic reference temperature.

Cadene et al.'s primary measurement, [arXiv:1307.6394](https://arxiv.org/abs/1307.6394), Eq47, gives kCM=+(2.35+/-0.13)e-16 T^-2 atm^-1 at these optical/gas conditions. Eq48 converts the atomic anisotropy Deta into gas birefringence: kCM=6.18381e-14 Deta/T. Here Deta=(sum_ab eta_abab-sum_ab eta_aabb/3)/5. The native per-molecule Cotton constant (2pi/27)Deta is a conventional intermediate, rather than the gas birefringence coefficient itself.

| Basis | Spherical AOs | Deta (atomic units) | kCM (T^-2 atm^-1) | Difference from experiment |
| --- | ---: | ---: | ---: | ---: |
| aug-cc-pVDZ | 9 | -0.0488374231 | -1.10562455e-17 | -104.70% |
| aug-cc-pVTZ | 23 | +0.3843917312 | +8.70219818e-17 | -62.97% |
| aug-cc-pVQZ | 46 | +0.5932093008 | +1.34295940e-16 | -42.85% |

The DZ wrong sign is physically inadequate at this model/basis; it is not sufficient evidence of an implementation defect. An independent complex-HF/dense-TDHF field calculation using exactly the TZ basis and minimal coupling h=h0+B Lz/2+B^2(x^2+y^2)/8 reproduces Deta=0.3843917355 after extrapolation, differing by4.25e-9atomic units. All four magnetic steps0.02,0.01,0.005,0.0025 were retained; the fitted second-order error has slope2.00041 and R-squared0.999999898. Zero-field polarizabilities agree within5.66e-13. An initial independent-helper exchange-index error and its correction remain in raw evidence; no native kernel was changed to obtain agreement.

The same calculations give Faraday Verdet magnitudes3.16136e-5,3.49566e-5 and3.68282e-5 atm^-1 rad T^-1 m^-1. The primary Eq31 measurement is +(4.17+/-0.13)e-5. Only magnitudes are compared until Fourier/helicity sign conventions are matched. The QZ magnitude is11.68% low. Neither tensor sequence is basis converged; the remaining Cotton-Mouton discrepancy cannot be assigned entirely to correlation.

Basis counts and HF energies independently match the primary tables in [Ema et al., Computation10,65(2022)](https://doi.org/10.3390/computation10050065), within their printed half-rounding. This checks basis identity, not higher-response convergence. Additional author basis coefficients were not retrieved: the article advertises a MOLPRO supplement, but the accessible article did not provide its contents and subsequent requests were rate limited. No coefficients were reconstructed. The QZ calculation completed in1014.48seconds wall/782.84seconds CPU, peak119308KiB; maximum response residual9.88e-11. All jobs are terminal. Raw results, all components, steps, conventions and initial failures are in `molecular-property-benchmark-20261031/magneto-optical-tranche`.

## Chiral VCD: matched Cartesian electronic axial tensors

The primary calculation is [Shumberger and Crawford, JCTC20,7254(2024)](https://doi.org/10.1021/acs.jctc.4c00747), with its [author supporting information](https://doi.org/10.1021/acs.jctc.4c00747.s001). The published SI coordinates and atom ordering were used exactly, HF/STO-3G, common gauge origin0, without GIAOs. For each four-atom example all36 electronic atomic axial tensor components were compared to the first three columns of the respective SI table, in atomic units.

| Case | SI geometry / tensor tables | Components within printed half-rounding | Maximum absolute difference |
| --- | --- | ---: | ---: |
| Chiral hydrogen dimer arrangement | S1 / S4 | 36/36 | 4.96319e-7 |
| (P)-hydrogen peroxide | S3 / S12 | 36/36 | 4.86384e-7 |

Mirroring peroxide gives exactly the expected axial/polar tensor transformation, unchanged harmonic frequencies and reversed native modal rotational strengths. This is a matched electronic-AAT comparison for T002. It does not establish matched published modal VCD intensities: the paper uses a common MP2/cc-pVDZ Hessian for its spectra, whereas the native HF/STO-3G Hessian at that geometry has one imaginary mode (-746.186cm^-1). The source's common normal-mode vectors/force constants are needed to close that comparison. The artificial hydrogen arrangement is likewise not evidence of a stable gas-phase compound. Raw source provenance, exact coordinates, all tensor values and mirror checks are in `molecular-property-benchmark-20261031/vcd-tranche`; these jobs completed in less than20seconds combined.

## Hydrogen gas: conditional refractivity and slab optics

HF/aug-cc-pVDZ H2 at R=1.40117Bohr gives undamped dynamic polarizability diag(4.4956330532,4.4956330532,6.7173295663) and isotropic5.2361985576atomic units at546.2252nm. All nine components were computed; the maximum response residual is2.18e-13. This is a clamped electronic reference at temperature0.

For an explicitly specified ideal gas at273.15K and101325Pa, orientational averaging and the Lorentz-Lorenz equation give n=1.0001309908566. The primary [Peck and Huang, JOSA67,1550(1977)](https://doi.org/10.1364/JOSA.67.001550), TableI, gives refractivity139.304ppm at546.2252nm under standard conditions. Native refractivity is8.31314ppm lower (-5.9676%). The paper does not assign absolute accuracy to every printed digit. Its TableIV inferred polarizability5.5722atomic units also depends on its density/reduction convention; inversion using our ideal-gas density gives5.56850 instead. Correlation, basis, rovibrational averaging and gas-density conventions remain model differences, rather than an isolated implementation defect.

As explicit postprocessing, a coherent vacuum/gas/vacuum slab of thickness1m at normal incidence gives R=1.243472943e-9, T=0.9999999987565267 and the raw conservation remainder1-R-T=3.33e-16. A single interface gives R=4.289089276e-9. Undamped below-resonance electronic absorption is0 m^-1 in this model; scattering, collisions, vibrational absorption and cell windows are excluded. Exact vacuum, zero-thickness and quarter-wave limits were checked. These conditional calculations are neither a native bulk-material API nor measured transmission/reflectance validation. Raw formulas, controls and results are in `molecular-property-benchmark-20261031/gas-optics-tranche`; the run completed in19.34seconds.

## H2 dc-Kerr response and the inversion-symmetric electro-optic limit

The full public frequency-ordered gamma tensor was computed for HF/aug-cc-pVDZ H2 at R=1.40117Bohr, wavelength632.8nm, zero damping and clamped nuclei. For native U_abcd=gamma_abcd(-omega;omega,0,0), the dc-Kerr contraction is sum_st(3U_stst-U_sstt)/10. It is not the ordinary isotropic gamma average. This follows the conventions in [Shelton and Rice, ChemRev94,3(1994)](https://doi.org/10.1021/cr00025a001), Eq5/TableI and Eq7; no additional permutation factor of3 multiplies an actual mixed derivative.

The result is gammaK=310.9517047312atomic units, with zero imaginary part. The primary [Shelton and Rugar, ChemPhysLett201,364(1993)](https://doi.org/10.1016/0009-2614(93)85084-2), p366 Table1 and p367, gives718.5+/-7.2atomic units after subtracting a quantum-rotor contribution from the zero-density experiment at25.2C. That value contains electronic and vibrational contributions; the native value is56.72% smaller and is not a matched-model implementation oracle. The source's alternative rotation treatment also changes the inferred value. Exact fixed-nuclei bond length and the air/vacuum conversion of the printed nominal wavelength remain unmatched. Its measured retardation definition, Eq1, and the tensor/field factors were checked directly against the lawful author PDF.

The separate electro-optic beta(-omega;omega,0) calculation gives maximum absolute component1.16e-12atomic units, consistent with molecular inversion symmetry. This is a molecular Pockels limiting-case check, not a noncentrosymmetric bulk electro-optic coefficient. All raw81 gamma and27 beta entries, residuals and controls are in `molecular-property-benchmark-20261031/kerr-tranche`. The run completed in132.91seconds wall/65.23seconds CPU, peak190308KiB; it shared one CPU with a diagnostic timer during part of execution, so its wall time is not an uncontended performance benchmark.

An independent same-basis check reoptimizes the HF electronic density under each static electric field and evaluates the dense optical TDHF polarizability there. Its second field derivative reconstructs all81 ordered tensor entries. All four steps0.008,0.004,0.002,0.001atomic units are retained; the Kerr contraction approaches native with second-order slope2.001263 and R-squared0.999999893. The h^2/h^4 extrapolated limit310.9517047248 differs from native by -6.43e-9atomic units. This35.57second check isolates the large experimental mismatch from a native contraction/response-factor defect at the chosen electronic model. Basis/correlation convergence and vibrational corrections are still required to identify its physical origin quantitatively; independent agreement does not establish experimental accuracy.

## HD indirect nuclear spin-spin coupling and independent checks

The public full Ramsey `spin-spin-couplings` route was evaluated for the neutral electronic singlet at R=1.40117Bohr (0.7414672326Angstrom), HF/aug-cc-pVDZ with18spherical AOs. The two nuclei use explicit dimensionless g factors5.5856946893(1H) and0.8574382338(2H). This is a clamped-nuclei electronic calculation without nuclear-motion, temperature, mass-dependent electronic or intermolecular corrections. Nuclear magnetic vector potentials are centered at the actual nuclei; this route does not accept an external uniform-field gauge-origin setting. Electron anomalous-g factors and conversion from energy to Hz are retained consistently.

The primary electronic reference is Helgaker et al., Molecular Physics110,2611(2012), [DOI10.1080/00268976.2012.729097](https://doi.org/10.1080/00268976.2012.729097), author manuscript Table2 p2615 at the same separation. It is a different, correlated FCI/basis-limit model, not a same-HF-basis implementation oracle.

| HD contribution | Native HF/aug-cc-pVDZ (Hz) | Published electronic FCI (Hz) | Difference (Hz) |
| --- | ---: | ---: | ---: |
| Fermi contact | 54.46960409 | 40.2515 | +14.21810409 |
| Spin dipolar | 0.54053483 | 0.4479 | +0.09263483 |
| Paramagnetic spin-orbit | 0.61801216 | 0.8344 | -0.21638784 |
| Diamagnetic spin-orbit | -0.20097460 | -0.3141 | +0.11312540 |
| Total | 55.42717648 | 41.2197 | +14.20747648 |

The isotropic total is34.47% larger than the correlated electronic reference. Garbacz, Chemical Physics443,1-7(2014), [DOI10.1016/j.chemphys.2014.08.002](https://doi.org/10.1016/j.chemphys.2014.08.002), gives a300K gas HD value43.136(7)Hz under its stated dilute-mixture measurements/corrections; the native point is28.49% higher. The2012 reference separately reports a zero-point correction+1.8943Hz and a further300K contribution+0.1993Hz, yielding its stated43.3132Hz. Those corrections were not added to our HF point, and were not counted twice. Different correlation, finite basis and nuclear-motion treatments preclude identifying the experimental difference as an implementation error.

The full native pair tensor is diagonal(66.13404426,66.13404426,34.01344092)Hz, with off-diagonal magnitudes below7.4e-15Hz. Term closure is1.42e-14Hz and maximum response residual6.33e-12. Replacing only the isotope g factor gives H2 J=361.07473764Hz and H2/HD=6.51439890258, equal to gH/gD; the complete reduced tensor is unchanged. This tests magnetic isotope conversion, not physical isotopologue rovibrational averaging.

The dominant discrepancy was investigated independently at the exact same bundled basis. Opposite-spin nuclear-contact perturbations were applied in four central step pairs0.008/0.004/0.002/0.001. The independently extrapolated FC value54.4696040817Hz differs from native by1.16e-8Hz, with slope1.9999374 and R-squared0.99999999986. The physical Hamiltonian coefficient and h-versus-hbar conversion were derived from Helgaker, Watson and Handy, JCP113,9402(2000), [DOI10.1063/1.1321296](https://doi.org/10.1063/1.1321296), Eqs3/4/9; no native conversion constant was fitted. An independent DSO density integral on successively refined grids4/6/8 gives-0.200974808486/-0.200974846675/-0.200974849099Hz, differing from native by2.48e-7Hz at the last grid. These are symmetric-point checks, not replacements for prior complete-C1 derivative gates. PSO and SD were not independently recomputed in this tranche.

The independent FC agreement establishes that the large dominant-term difference belongs to the chosen HF/basis model. This model is quantitatively poor for HD J, despite small equation residuals. Its HF DSO fallback constructs an internal default quadrature when no XC grid is supplied; identical results under public grid_level3/5 do not establish convergence. The separate independent DSO refinement supplies a measured check here. This control limitation is retained rather than disguised as a successful native grid study.

Raw tensors, controls, sources, immutable artifact/basis hashes, every finite-field sample and independent quadrature values are in `molecular-property-benchmark-20261031/spin-spin-tranche/`. Native and independent calculations are terminal; no reference, gate or implementation parameter was adjusted to match the published values.


### Periodic checkpoint: silicon stress, added structure families and spin DFPT

The extended scope contains twelve electronic crystal cases. At the
2026-10-31 05:21 UTC checkpoint, two initial SCFs have completed, 3C-SiC is
running, and nine cases remain queued. No case has a completed k/basis/grid
convergence acceptance. Silicon completed SCF plus analytic stress:
E=-571.3036973237838 Hartree per two-atom cell, seven iterations,
final reference orbital residual 3.099e-10 Hartree and density change
3.234e-10. The SCF took 803.18 s; SCF plus stress took 1440.60 s
(1442.93 s supervisor wall time). At the supplied static reference
a=5.416 Angstrom, its pressure is +51.0442544673641 GPa, rotation residual
4.1434819401181e-15 Hartree and sampled Gamma KS gap 2.70074868440926 eV.
The large pressure is an unresolved numerical/model discrepancy, not a
pass; k-mesh, basis and quadrature scans are still required. The gap is
a mesh-sampled KS quantity, not an experimental optical/fundamental gap.

The first diamond 240 s timeout and two interrupted silicon setup/SCF
attempts remain separate from successful results. The original silicon
worker exited shortly after its obsolete wrapper/supervisor was replaced;
its exact exit status was not available. Absence and lack of output were
verified before starting exactly one retry, which completed successfully.
There have been no 1800 s hard-timeout or RSS-limit failures in the resumed
campaign. Its new supervisor extends runtime when needed instead of
automatically abandoning a progressing reference. The active SiC process
at 05:21:27 UTC had elapsed 1607 s, current RSS 52524 KiB and peak RSS
203976 KiB. The measured hotspot is screened Hartree: approximately
136..164 s per build, 7330 AO pairs (3898 compact), 285 output translations
and roughly 2.4 million real-space quartets, with two Rayon threads.
This is computational cost, not a demonstrated memory or SCF-divergence
failure. Exact states and resource readings are in `status-summary.json`
and `live-resource-checkpoint.json`.

The added perovskite is pure KMgF3, P m -3 m, a=3.9859 Angstrom, five atoms,
from [Burns et al. (1996), COD 9006363](https://www.crystallography.net/cod/9006363.html).
The added spinel is the published fully occupied normal MgAl2O4 model,
Mg tetrahedral/Al octahedral, F d -3 m, a=8.0898 Angstrom and oxygen
u=0.26330, at 0.001 kbar, from
[Finger, Hazen and Hofmeister (1986), COD 9005807](https://www.crystallography.net/cod/9005807.html).
Its 14-atom fcc primitive cell reconstructs the 56-atom conventional cell
with fractional-position defect below 1e-10. Neither CIF states measurement
temperature; no specific temperature is inferred. The mixed-occupancy
1991 spinel samples were inspected and are not silently converted to a
normal ordered cell. Both selected cells use neutral restricted all-electron
models with existing bases, no ECP and no imposed site charges.

The kaolin case is the [Bish (1993) 1.5 K neutron kaolinite structure](https://doi.org/10.1346/CCMN.1993.0410613),
[COD 9009234](https://www.crystallography.net/cod/9009234.html), including all
structural H atoms. The centred triclinic C 1 conventional cell has
a=5.1535, b=8.9419, c=7.3906 Angstrom, alpha=91.926, beta=105.046,
gamma=89.797 degrees. The 17-atom Al2Si2O9H4 primitive cell reconstructs
all 34 conventional-cell atoms with maximum fractional-position defect
2.9893669801409083e-16. Its structural OH is retained; no intercalated or
pore water, substituted sites or stacking-fault ensemble is added. The
initial electronic baseline is PBE without dispersion and therefore cannot
by itself certify interlayer cohesion.

Actual periodic pair-only PBE-D3(BJ) component calculations have completed
for urea, ferrocene and kaolinite, with pair cutoffs 60, 80, 95, 120 and
160 Bohr and coordination cutoff 40 Bohr. At the default 95 Bohr pair
radius the energies are -0.0352353380666255, -0.105442158306389 and
-0.0780999218435587 Hartree per respective primitive cell. Two-cell
energy-per-primitive defects are at most 1.27675647831893e-15 Hartree;
translation energies/forces and strain replication also pass their retained
component gates. These are dispersion components, not completed electronic
SCFs, total lattice energies or full material acceptance.

For kaolinite, a rigid-sheet separation scan changes only the normal
component of the repeat vector, preserving intra-sheet Cartesian geometry
and the lateral stacking registry. At 40/95 Bohr CN/pair cutoffs, the
dispersion-only work of separation is 0.306659880785 J/m^2
and the corresponding two-surface energy contribution is
0.153329940392 J/m^2. Increasing the vacuum by
another 20 Angstrom changes this finite-cutoff component energy by zero.
Pair-cutoff work values are
95.0 Bohr: 0.306659880785, 120.0 Bohr: 0.306796937242, 160.0 Bohr: 0.306876085295 J/m^2.
Separate CN-cutoff scans are preserved in the same raw report. This omits
electronic hydrogen bonding/electrostatics, relaxation, thermal/ZPE and
particle/pore physics; it is not a full cleavage energy or soil strength.

Existing soil-related paths are mineral elastic/material cards, orientation
and aggregate moduli, dry-frame/contact/spheroidal-pore models, and
Gassmann/Biot/Brown-Korringa coupling. Crystal elastic data alone do not
determine porosity, saturation, permeability, drainage, friction, plasticity
or particulate strength. No such physical inputs have been fabricated.
The [published 2011 sheet-silicate tensor study](https://doi.org/10.2138/am.2011.3558)
also distinguishes crystal tensors from shale texture/pores; matched
primary tensor/continuum comparisons remain work, not an inferred soil pass.

The new `periodic-spin-response` connection implements longitudinal
finite-q/frequency spin-Zeeman sources using the existing collinear
screening kernel. All 15 independent closed-Gaussian-source/dense complex
cases across SVWN5/PBE/TPSS/PBE0/CAM-B3LYP and static/retarded/imaginary
frequencies pass (maximum all-entry defect 1.829236089858e-13). All
120 original shared-gate four-step real spin-density component fits
pass for PBE/TPSS/PBE0 cosine and sine probes, over 0.04..0.005 atomic
magnetic-field amplitudes. Their measured p range is
1.920199..2.069206, minimum
R-squared 0.99958007; the unchanged shared gates are
abs(p-2)<0.2 and R-squared>0.999. The earlier verbal assertion that coarse
TPSS fits had failed those gates and been resolved on a finer interval was
incorrect. They passed the shared gates. A separate finer TPSS interval
0.01..0.00125 with additional abs(p-2)<0.02 and R-squared>0.99999 gates is
running; all original measurements remain retained.

The 40000 K PBE test gives equilibrium static susceptibility
0.6309073177191 atomic units, spatially uniform collisionless response
6.165984353252e-33, finite-q retarded moment magnitude
1.963078163771e-3, and occupation consistency defect
1.727980278124e-15. A thermal static reciprocal-lattice probe without its
required common-number multiplier is explicitly refused. These results
establish the stated limit distinction, not real-time propagation or a
complete orbital/SOC/transverse magnetic model.

The source/API connection has not yet had its serialized full-library,
CLI/native/ASE artifact build. Module tests use accepted23 dependencies;
the old artifacts do not contain the new public feature. Source ownership
and the build request are in `.work/validation/pbc-ownership-20261031.json`
and `pbc-build-request-20261031.json`. There is no overlap with the proposed
non-PBC continuous-species M1 `src/scf/mod.rs` hook or its driver module.
No paused spinor worker was restarted. All running physical jobs remain
active; this checkpoint is not closure of the twelve-case validation.


### Periodic longitudinal spin: strict TPSS component fits and build handoff

The additional strict TPSS fine-step acceptance passed all 40 independent cosine/sine spin-density components on the C1 cell. For steps 0.01, 0.005, 0.0025 and 0.00125, measured exponents span 1.994431..2.005896, with minimum R2=0.99999772; all meet |p-2|<0.02 and R2>0.99999. Runtime was 554.06 s. No original shared threshold, SCF gate or error sample was loosened or removed. The original 120 coarser PBE/TPSS/PBE0 component fits remain separately preserved.

The public native/ASE/CLI acceptance file is prepared and syntax checked, and the three API tables pass their source-alignment check. A frozen ownership bundle applies only explicit shared-file hunks and new files, with a successful staged apply dry run. Execution of the new public route still requires the serialized complete build after the existing peer reservation; old frozen modules do not contain this API. This is a scoped longitudinal frequency-domain implementation, not completion of all electromagnetic or real-time DFPT.


### Kaolinite continuum calculation with published elastic inputs

The existing `aggregate-moduli` native driver was actually evaluated with published mineral/interparticle data, independently of the queued crystal electronic calculation. Sayers and den Boer, [JGR Solid Earth 123 (2018), Tables 2 and 4](https://doi.org/10.1029/2018JB015600), supply the best-fitting transversely isotropic ideal-kaolinite tensor and the interparticle-medium parameters. This is a fitted TI representation of a published DFT tensor, not the full triclinic tensor of the Bish experimental structure used in the periodic job.

Independent dense compliance inversion gives random-orientation Voigt/Reuss/Hill (K,G) values of (68.1222,42.8033), (52.9902,26.5409) and (60.5562,34.6721) GPa. The directional Young moduli are (160.7451,160.7451,83.4789) GPa in the TI axes, retaining the softer layer-normal direction. The tabulated C66=58.8 GPa is rounded; this calculation uses C66=(C11-C12)/2=58.85 GPa, preserving exact TI symmetry.

For an explicitly isotropized spherical-grain mixture with mineral volume fraction 0.935 and interparticle (K,G)=(2.208,0.174) GPa, the existing driver returns mixture Hill (39.5230,17.4632) GPa and self-consistent (52.3074,30.3231) GPa. Positive elastic spectrum, the pure-phase limit, independent mixture Voigt/Reuss/Hill identities and bound ordering all pass. This spherical isotropic calculation does not reproduce the paper anisotropic oblate-platelet Willis model or establish soil strength, porosity, drainage or plasticity prediction. No crystal DFT elastic result is inferred from these input values. The full inputs, outputs, library hash and scope are preserved in `kaolinite-published-continuum-card.json` under the local campaign artifacts.


### Periodic execution and measured Hartree scheduling optimization

At the 2026-10-31 05:57 UTC checkpoint, diamond, silicon and 3C-SiC have completed initial periodic SCFs; silicon and SiC also returned analytic stress. LiF is running, with eight later cases queued. SiC converged in eleven iterations to -322.70240569138826 Hartree, with final orbital residual 6.868e-11 Hartree, density change 6.908e-11 and free-energy change 1.705e-13 Hartree. SCF plus stress took 2909.66 s (2911.78 s supervisor wall time). Its pressure at a=4.342 Angstrom is +139.410540320904 GPa, rotation defect 3.42108112706761e-15 Hartree, quadrature electron count 20.0010525541307 and Gamma-sampled KS gap 4.96336584527274 eV. The large pressure is unresolved; no physical acceptance or k/basis/grid convergence is inferred. SiC stress timings include 865.21 s Hartree and 84.11 s exchange-correlation.

The next LiF reference has 31355 significant part pairs, 12436 compact pairs and 2123 output translations. Its first Hartree build evaluated 9485134 real-space quartets in 332.66 s, including 52.14 s preparation. The second build took 269.67 s. These are measured costs, not a memory or timeout failure. The owned 1.5 GiB RSS guard now remains active until an explicit completion marker instead of expiring after 24 hours; no peer or active SCF was signalled during guard renewal.

A narrow optimization parallelizes matrix construction over significant bra pairs, then assembles each output in the original pair order. It changes neither integrals, screening, physical cutoffs nor SCF gates. Mixed compact/diffuse s/p C1 matrices agree with the frozen original for all three split parameters. On the actual SiC basis/lattice, both workload orders use the same synthetic translation-even AO density, Hartree tolerance 1e-13, uncapped pair distances, 285 output translations, compiler profile and two threads. Every matrix element agrees exactly. Original/candidate times are 181.745735/133.065892 s and, in reversed order, 171.462215/128.594829 s: 1.366x and 1.333x kernel speedups. This is not a converged SCF density or an end-to-end SCF speedup measurement. Smaller kernels show mixed timing changes; the gain is workload dependent.

Additional temporary memory holds one shell-pair-sized matrix per significant bra, with linear growth; there is no persistent or four-index cache. The source hunk is integrated and frozen for the pending serialized full-library/native/CLI build. Existing full-library matrix/gradient regressions, public spin-route tests and end-to-end crystal timing still require that artifact. The in-flight LiF job retains its original frozen library. The twelve-case physical and convergence scope remains open.


### Expanded periodic physical-property scope beyond the named examples

The user examples are nonexhaustive. This inventory follows the existing registry/MD/CSV and related mechanical, dielectric, thermal and electromechanical quantities; it is not a claim about every conceivable solid-state observable.

The following are execution and implementation obligations, not completed crystal results. Existing entry points alone do not establish their functional/reference, strain, frequency or convergence coverage.

| Family | Existing routes / inspected source | Required numerical scope and open acceptance |
|---|---|---|
| Stationary structures, stress and EOS | `stress; opt-fixed-cell; cell-opt; equation-of-state` | Diamond/Si/SiC and ionic cells; cell and internal equilibrium before responses. Initial SCFs are not structural convergence. Establish basis/k/grid/reach convergence and pressure/force stationarity. |
| Second-order elasticity | `elastic; elastic-fd` | Clamped/relaxed SOEC, internal strain, prestress-corrected Lagrangian/Birch coefficients. Analytic route exists. Actual equilibrium crystal tensors, all-component validation and numerical convergence remain pending. |
| Compliance and mechanical observables | `elastic-moduli; christoffel; material-card` | Compliance, engineering moduli, Voigt/Reuss/Hill, direction-dependent waves and Debye estimates. Use the same validated tensor; a scalar/isotropic card does not reproduce an anisotropic clay or soil. |
| Third-/fourth-order elasticity | `toec; foec` | TOEC/FOEC with defined Lagrangian strain, clamped/relaxed ions and prestress conventions. Existing analytic Gamma-supercell routes; expensive dense Taylor workspace and finite-k convergence must be assessed before launch. |
| Proper piezoelectricity | `piezoelectric-tensor; piezo-higher` | Nonzero zincblende 3C-SiC; centrosymmetric diamond/NaCl as zero controls. Existing routes are not physical acceptance. Check orientation/sign, proper versus improper tensor, ionic relaxation and polarization convention. |
| Electrostriction and coupled strain/field response | `electrostriction; piezo-higher` | SiC and centrosymmetric ionic/perovskite controls with finite nonzero electrostriction. Audit the reported strain derivative of susceptibility separately from constant-stress strain/E^2 and strain/P^2 coefficients; do not rename one as another. |
| Polarization, Born charges and dielectric tensors | `berry-polarization; born-charges; dielectric-tensor; static-dielectric` | Ionic crystals, KMgF3, spinel and molecular/kaolinite cells where stable. Electronic versus relaxed-ion response; polarization branch, fixed E/D convention, charge sum rule and k convergence. |
| IR, mode charges and LO-TO splitting | `crystal-ir; phonons; phonon-bands` | LiF/NaF/NaCl/MgO and polar SiC, with direction-dependent nonanalytic limit. Requires matched Born charges/epsilon, masses and acoustic/charge sum-rule diagnostics; Gamma alone is not dispersion. |
| Harmonic lattice dynamics | `phonons; phonon-bands; phonon-thermo` | Force constants, q dispersion/DOS and acoustic/optical branches of selected crystals. Force-constant/DFPT comparison, raw ASR, supercell/q convergence and instability interpretation remain required. |
| Mode volume/strain response | `mode-gruneisen; quasi-harmonic` | First and second mode Gruneisen coefficients, with branch/degenerate-subspace tracking. Analytic first Gamma-mode route exists. A dedicated second-order mode route was not identified; implement it with an explicit log-volume convention and fourth derivatives. Thermal averages are separate. |
| Vibrational and quasi-harmonic thermodynamics | `phonon-thermo; quasi-harmonic` | ZPE, vibrational F/U/S/Cv/Cp, thermal expansion, finite-pressure equilibrium and thermodynamic Gruneisen. QHA routes exist, but stable volume-dependent q-sampled spectra and mode tracking are not supplied by initial Gamma SCFs. |
| Nonlinear dielectric and Raman response | `periodic-beta; periodic-gamma; periodic-raman` | Static higher electric susceptibility and displacement/strain-electric couplings. Retain static/bias/relaxation and intrinsic tensor permutation conventions. Static routes do not establish frequency-dependent electro-optics. |
| Anharmonic and electron-phonon targets | `periodic-cubic-fc; periodic-quartic-fc; periodic-cubic-fc-along; periodic-quartic-fc-along; electron-phonon` | Cubic/quartic IFCs and documented electron-phonon couplings. Directional contractions are not full tensors; coupling vertices are not automatically phonon linewidths or lattice thermal conductivity. |
| Static, thermal and dynamic DFPT | `periodic-response; field-response; dielectric; periodic-spin-response; thermal-response` | Occupation/chemical-potential/entropy consistency, finite-q electric/orbital/spin probes and frequency response. Scoped source/public tests pass. Real-time propagation and selected covariant meta-GGA/ECP current connections remain separate missing work. |
| Other documented periodic electronic/magnetic spaces | `electronic-structure; effective-mass; boltzmann-transport; hall-transport; quantum-hall; wannier-flow` | Retain the complete registry/CSV inventory, with appropriate carrier, magnetic and dimensional references. Insulating bulk examples do not certify metallic transport, topological or lower-dimensional spaces. User-paused spinor work remains paused. |
| Microscopic-to-continuum connection | `material-card; aggregate-moduli; rock-physics` | Kaolinite crystal elasticity versus explicitly supplied aggregate/pore/fluid parameters. Published TI continuum-card calculations have run separately. Perfect dry crystal responses do not determine soil strength, porosity, drainage or plasticity. |

Three minimal-basis Gamma SCFs and two stresses returned; LiF retry is running. Zero of twelve has a completed stationary-structure/basis/k/grid physical-property acceptance. Si and SiC large pressures remain unresolved.

Periodic D3 energy/forces/stress/optimization exists and separate D3 derivatives are available, but PeriodicMethod::from_controls currently rejects dispersion for the general response features. Full dispersion Hessian/mixed/high-order connections must be audited/connected before using relaxed molecular/mineral responses.

Proposed explicit outputs: gamma1=-d ln(omega)/d ln(V), curvature=d2 ln(omega)/d ln(V)^2=-d gamma1/d ln(V), and optionally q=d ln(gamma1)/d ln(V) only when defined. Do not conflate these or an undefined acoustic/unstable mode with a numeric zero.

Existing high-order paths represent k sampling by a Gamma supercell. A 2x2x2 SiC/STO-3G supercell has 112 AOs; five dense four-index coefficient payloads alone require 112^4*5*8=6294077440 bytes, before response/grid/workspace. This is a lower-bound estimate, not RSS or a generic impossibility claim.

Additional explicitly retained categories:

| Family | Existing routes / inspected source | Required numerical scope and open acceptance |
|---|---|---|
| Other documented magnetic/spin/nuclear responses | `pauli-spin-susceptibility; orbital-susceptibility-q; periodic-magnetic-scf; lab-frame-spin-spiral; dmi; periodic-nmr-shielding` | Retain the separate magnetic, current and shielding target rows and their actual reference restrictions. A closed-shell nonmagnetic zero control does not validate finite magnetic or noncollinear responses. User-paused spinor work remains paused. |
| Other documented boundary/operator spaces | `periodic-electrostatics; periodic-basis; periodic-gaussian-integral; periodic-coulomb; periodic-xc; periodic-field; dimensional-scf; dimensional-field` | Inventory the documented boundary/electrostatic/Gaussian/XC operators separately from bulk material properties. Three-dimensional bulk results do not certify wires/slabs/open directions; operator acceptance is distinct from experimental crystal-property validation. |

The full matched registry entries, CSV rows and inspected MD headings are preserved in `expanded-periodic-property-scope.json`. No MD/CSV file was added. Numerical property results will be staged per validated reference/property rather than inferred across all twelve cases.


### Accepted public periodic artifact and durable campaign continuation

Frozen artifact02 completed library and canonical CLI compilation and portable artifact audit. Six native/ASE/CLI longitudinal-spin tests pass, together with the p38 Hartree matrix/energy oracle, five p59 field cases, all twelve independent C1 Hartree gradient fits and the new spin dense/source, static-limit and positive-temperature checks. The first unit-conversion fixture failure is retained: its ASE reference controls had not been converted to physical units. The corrected fixture passes unchanged numerical gates; no product-kernel tolerance was loosened.

The new measurement worker is compiled separately against the accepted artifact, with zero energy/stress defect against both the new public route and the retained old helium oracle. It is adopted only for future queued cases. In-flight LiF retains its original mapped library and recorded controls. Detached supervisors, waiters and the owned-worker RSS guard survived the subsequent connection callback and were verified alive before continuing. No duplicate build or crystal calculation was launched. The detailed artifact hashes, PIDs, expanded scope and still-open physical acceptance are in `parent-checkpoint-current.json`.

## Five further molecular species and measured response optimization

The five additions are methanol, acetonitrile, acetone, thiophene and aniline. They extend the existing eight-molecule set with alcohol, nitrile, carbonyl, sulfur heteroaromatic and aromatic-amine environments. Each completed a converged public HF/STO-3G optimization followed by HF/STO-3G and HF/3-21G dipole/static-polarizability calculations at that same geometry, eight singlet CIS roots and GIAO shielding tensors. All five numerical comparison cases are complete; their experimental/model interpretation and TMS-referenced shifts are given below. These deliberately small basis comparisons isolate implementation from model accuracy; neither basis is established as a quantitative gas/solution spectroscopic model.

Initial comparison failures are retained. The independent methanol static-response solver stopped with fresh relative residuals up to 6.99e-6 despite a nominal tighter tolerance. An independent matrix-free solve with fresh original-equation residuals below 6.28e-14 reduces the maximum native/reference polarizability difference from 1.06412e-6 to 2.12189e-8 atomic units. Similarly, the independent magnetic response initially has residuals 4.91e-7..8.22e-7; a fixed occupied-metric block and converged virtual-space solve gives residuals 5.20e-14..9.72e-14 and reduces the maximum methanol shielding-tensor difference from 1.01297e-4 to 3.27674e-7 ppm. No native tolerance or comparison threshold was relaxed. Dense independent CIS diagonalization also resolves the auxiliary iterative solver's false convergence flags: all eight energy differences are at most 6.55e-13 Hartree and oscillator-strength differences at most 8.03e-12. These are auxiliary-solver accuracy findings, not evidence of native property defects.

The primary thiophene beam-Stark dipole is 0.5296(13) D, rather than the older compiled 0.550 D: Spoerel and Consalvo, [Chemical Physics 239 (1998), 199-206](https://doi.org/10.1016/S0301-0104(98)00275-4), abstract and electric-dipole section. The primary aniline microwave abstract instead reports inversion-state-resolved a-axis components, 1.129+/-0.005 D for 0+ and 1.018+/-0.005 D for 0-: Lister et al., [Journal of Molecular Structure 23 (1974), 253-264](https://doi.org/10.1016/0022-2860(74)85039-8). These components must not be compared directly with a calculated dipole norm or substituted for the separately compiled total 1.530 D.

Actual profiling on the retained 14-AO methanol reference identifies the weighted magnetic ERI contraction as a substantial cost: 4.02 CPU seconds, against less than 0.01 for overlap, 0.05 for core, 0.08 for all paramagnetic spin-orbit operators and 0.66 for all mixed operators. Separate calls and contention make these measurements unsuitable for summing into an exclusive whole-job fraction. A bounded optimization reuses ordinary moment-integral shell quartets while preserving each magnetic phase/sign contribution separately. It does not assume eightfold symmetry of the magnetic derivative itself or change screening thresholds. All 24 synthetic C1 s/p/d cases, symmetric and nonsymmetric densities, two origins and full/range-separated kernels agree with the original contraction to at most 1.344e-17. The literal production module's eight tests and the final full-crate artifact's same eight tests pass. The actual public-route and performance acceptance below is complete. Raw failures, source identities and component results remain in the existing local validation campaign. No new documentation file was added.

### Final molecular-response artifact and measured speedups

Frozen V2 passed its full library/CLI build and portable audit, two complete AO/spin cache tests, all eight magnetic-ERI tests, four p25 tests, four p309 tests, three Faraday tests, native/ASE/both-CLI public gates and six anharmonic/VPT2 public tests. The first magnetic test filter selected only the six older tests; the broader filter was then actually run and passed all eight. No compilation-only result is substituted for these executions. The exact rlib starts b1434959a36f, Python library69cef02cd3b7 and both CLI artifacts a13cc3aee5bc; complete hashes and commands are in the existing local `mcd-thermal-cache-integration-20261031-v2/final-acceptance.json`.

The thermal-response test has 180 second-order electric-kick component fits, p=1.980515..2.002256 and minimum R2=0.99997358. Separately, its independent nonlinear time-domain oracle has 36 fourth-order timestep fits, p=3.924533..4.082262 and minimum R2=0.99976474. All existing gates pass. These oracle trajectories validate the native frequency-domain response; they do not expose a public real-time propagation API. The original nonconverged thermal run and the corrected preconditioner evidence remain preserved. This acceptance covers the explicitly tested reference/function/property slices, not every property in the inventory or subsequent peer source changes.

The semilocal derivative cache was timed with one thread on the same optimized artifact, C1 NH3/PBE/STO-3G, eight AOs, 43326 grid points, level3 and zero grid-weight cutoff. Both cases retain explicit ERI InCore128MiB; only the response-cache budget control changes from0 to64MiB, corresponding to an8MiB cache allowance. These are internal Rust timing controls, not a new public cache option. The first A+B action costs0.82 versus0.83 CPU seconds, while median subsequent action time falls from0.76 to0.05 seconds, a15.2x gain. Full input/output vectors and reference coefficients are identical as IEEE754 binary64 payloads, including signed-zero checks.

For all three RHS of the damped dynamic polarizability at omega=0.21 and damping=0.13 Hartree, response CPU time falls from263.78 to17.13 seconds, a15.40x gain. Response wall time is264.184 versus17.149 seconds; full SCF-plus-response wall time is265.478 versus18.420 seconds. All nine complex tensor entries,29iterations per RHS and every residual are exactly unchanged; maximum residual is5.61e-11. Process peak RSS is11340 versus17704KiB. This is a workload-specific measurement, not a universal DFT speedup or a hard process-memory cap. The cold action still pays derivative construction. The raw cache-off/on outputs, exact payload comparison and resource controls are in `cache-performance-ready/run-b1434959a36f-v2-final` within the local campaign.

On the retained methanol HF/STO-3G geometry, ordinary public NMR CPU time changes from18.588 to15.554 seconds and wall time from18.597 to15.632 seconds. Maximum old/new shielding-tensor difference is1.99e-12ppm, and the strict independent difference is3.28e-7ppm. Native, common, ASE and both CLI routes agree. Observed process peak RSS changes from94004 to98032KiB; these whole loaded-bundle measurements do not isolate the workspace allocation from every other binary/runtime difference. No8x full-NMR gain is inferred from integral symmetry.

An additional public Rust resource-path check explicitly stores the ordinary ERI tensor for both SCF and magnetic response with InCore256MiB. All54methanol tensor entries differ from the public-driver result by at most4.03e-13ppm; energy difference is zero and maximum response residual7.40e-13. The worker uses1.17CPU seconds and22548KiB peak RSS. Its entry point and assigned CPU differ from the Python-driver timing, so this is not presented as a controlled universal speed ratio. It demonstrates an existing exact integral-reuse path suitable for still-unrun molecular reference calculations. It does not claim that the default Python/CLI path has acquired this resource-policy change.


### Completed five-molecule values and physical interpretation

All five additions are complete: ten static conditions, forty ordered singlet CIS roots and405GIAO shielding-tensor entries agree with independent calculations at the identical geometry and exact shared basis definition. Maximum differences are9.89e-12Hartree in energy,8.90e-8D in a dipole component,1.10e-7atomic units in polarizability,1.35e-12Hartree in a CIS energy,1.65e-11 in oscillator strength and8.39e-7ppm in shielding. All original gates and native convergence checks pass. The smallest eigenvalue of a symmetrized static polarizability is5.3148atomic units; maximum raw reciprocity defect is1.21e-7atomic units. NMR tensors themselves are not assumed symmetric.

Every molecule is neutral and closed shell. Both static basis calculations use its converged public HF/STO-3G geometry, without solvent, nuclear-motion or thermal corrections. NMR and CIS use HF/STO-3G. The final aniline NMR and the TMS reference use the independently checked public Rust explicit-InCore resource path; the other molecular NMR calculations retain their recorded native/common route. This resource distinction does not change the electronic model. Starting conformers, final Bohr coordinates, all controls, basis/artifact hashes, complete tensors and forty-state data are retained in the existing local `added-five-tranche` records.

| Molecule (atoms) | Dipole STO-3G / 3-21G, D | Gas experimental dipole, D | Isotropic static alpha STO-3G / 3-21G, Angstrom^3 |
|---|---:|---:|---:|
| Methanol (6) | 1.509446 / 2.212515 | 1.672, indexed compilation | 1.183573 / 2.056888 |
| Acetonitrile (6) | 3.096297 / 3.944410 | 3.92197(13), primary abstract | 1.725117 / 3.008818 |
| Acetone (10) | 1.918415 / 3.150708 | 2.9345(22), primary Table III | 2.566183 / 4.520061 |
| Thiophene (9) | 0.104984 / 1.101263 | 0.5296(13), primary measurement | 3.847619 / 6.476888 |
| Aniline (14) | 1.496981 / 1.789752 | 1.530, compiled total | 5.570056 / 8.273185 |

Methanol uses the [NIST neutral-molecule entry](https://cccbdb.nist.gov/exp2x.asp?casno=67561&charge=0), which cites Sastry, Lees and Van der Linde, [JMS88,228(1981)](https://doi.org/10.1016/0022-2852(81)90367-2); the original numerical table was not retrieved. Acetonitrile's newer value is the authors' abstract for Gadhi et al., [JChimPhys92,1984(1995)](https://doi.org/10.1051/jcp/1995921984), mirrored in its bibliographic record; its full original uncertainty treatment was not retrieved. For acetone, Dorosh and Kisiel, [ActaPhysPolA112,S-95(2007)](https://doi.org/10.12693/APhysPolA.112.S-95), TableIII pS-102, includes field-calibration uncertainty; the smaller statistical error in TableII is not substituted for it. Thiophene and aniline provenance and the distinction between a component and a total are stated above. These rotational/inversion-state measurements are not clamped-nuclei equilibrium dipoles.

Relative to these qualified dipole references, the3-21G deviations are+32.3%,+0.57%,+7.37%,+107.9% and+17.0%, respectively. The small acetonitrile deviation does not validate the other molecules. In particular, thiophene changes from a strongly underestimated to a strongly overestimated dipole when only this small basis is enlarged. The NIST-indexed Gussoni et al. analysis gives comparison alpha volumes3.210,4.280,6.270,9.010 and11.581Angstrom^3, [JMolStruct447,163(1998)](https://doi.org/10.1016/S0022-2860(97)00292-5). Its original measurement frequencies/temperatures and uncertainties were not recovered, so these are compiled plausibility guides, not matched static-response acceptance targets. Neither basis is established as quantitatively converged.

Aniline illustrates why a total alone can be misleading. Its STO-3G total is only2.16% below the compiled1.530D, but calculated absolute principal-axis components are(0.309059,0.000000704,1.464731)D; at3-21G on the same geometry they are(0.342920,0.000001173,1.756593)D. Axes are ordered by increasing inertia using the ordinary isotopologue masses. The primary measured a-component1.129D cannot be replaced by the near-agreeing total. The optimized geometry has C-N1.444495Angstrom, HNH107.597degrees and NH-bisector/ring-plane angle50.809degrees, versus the primary substitution-structure C-N1.402+/-0.002Angstrom, HNH113.1+/-2degrees and reported amino out-of-plane angle37.483+/-2degrees. The geometric angle is explicitly constructed from the two normalized NH bonds and the least-squares carbon-ring plane; a clamped stationary structure and an inversion-averaged substitution structure are distinct. These results identify substantial geometry/model limitations, rather than evidence of a native implementation error hidden by the total.

The lowest calculated singlet CIS sticks are:

| Molecule | Energy, eV | Oscillator strength |
|---|---:|---:|
| Methanol | 12.237209 | 0.00538176 |
| Acetonitrile | 8.235386 | 4.30e-15 |
| Acetone | 4.477565 | 7.39e-15 |
| Thiophene | 8.329373 | 0.127463 |
| Aniline | 7.536415 | 0.0260069 |

Thiophene's next root is8.575066eV with f=0.170805. Holland et al., [PCCP16,21629(2014)](https://doi.org/10.1039/C4CP02420F), Table1, give low bright EOM-CCSD states5.78/6.13eV with f=0.081/0.084 at the experimental geometry and a406-orbital augmented/Rydberg basis. Their Table3 gas vibronic origin5.157eV and broad band near5.5eV are separate observables. The minimal-basis sticks are strongly displaced from this low-energy manifold, but an exact state-by-state assignment is not inferred from energy ordering alone. The first acetone stick lies near the broad room-temperature n-to-pi-star envelope described around275nm by Koch et al., [JQSRT109,2037(2008)](https://doi.org/10.1016/j.jqsrt.2008.02.010); its tiny clamped oscillator strength and the measured vibronically enabled finite-temperature absorption are consistent with different observables. This proximity is not a quantitative vertical-energy or intensity validation. Exact numerical band tables for the other retrieved primary sources were unavailable; no missing values or assignments were invented. The computed forty roots remain an implementation comparison, not a complete absorption spectrum, sum-rule closure or bulk absorption coefficient.

### Same-model TMS referencing and experimental chemical shifts

A separately public-optimized17-atom TMS reference was actually evaluated with the same HF/STO-3G model and GIAO convention. All153shielding-tensor entries agree with the strict independent calculation within4.60e-7ppm. Mean carbon/proton shieldings are249.43496795/33.75277281ppm, with equivalent-site spreads5.20e-5/1.28e-5ppm. Site groups below follow the archived SDF connectivity and atom order, not fitted shift assignments. Methyl values are arithmetic means of clamped sites; they are not a simulated thermal exchange average.

The conversion is delta_ppm=(sigma_TMS_ppm-sigma_site_ppm)/(1-sigma_TMS_ppm*1e-6). Experimental values are Gottlieb, Kotlyar and Nudelman, [JOC62,7512(1997)](https://doi.org/10.1021/jo971176v), Table1 p7513 for1H and Table2 p7514 for13C, first solvent columnCDCl3. The [original article copy](https://www.epfl.ch/schools/sb/research/isic/wp-content/uploads/2018/10/NMR_Chemical-Shifts_Common_Laboratory.pdf) specifies internal TMS,24+/-1C,300.1/75.5MHz. No uncertainty was supplied for the selected solute entries.

| Molecule / site | Isotope | Calculated delta, ppm | Experimental delta, ppm | Difference, ppm |
|---|---|---:|---:|---:|
| Methanol CH3 | 13C | 50.269 | 50.41 | -0.141 |
| Methanol CH3 | 1H | 2.214 | 3.49 | -1.276 |
| Methanol OH | 1H | -0.546 | 1.09 | -1.636 |
| Acetonitrile CH3 | 13C | 21.354 | 1.89 | +19.464 |
| Acetonitrile CN | 13C | 110.275 | 116.43 | -6.155 |
| Acetonitrile CH3 | 1H | 2.264 | 2.10 | +0.164 |
| Acetone CO | 13C | 157.969 | 207.07 | -49.101 |
| Acetone CH3 | 13C | 31.663 | 30.92 | +0.743 |
| Acetone CH3 | 1H | 2.438 | 2.17 | +0.268 |

The19.46ppm nitrile-methyl and49.10ppm carbonyl discrepancies are much larger than native/independent numerical differences. This coarse gas/clamped model is unsuitable for quantitative chemical-shift screening across these environments. Methanol carbon's much closer agreement does not establish general accuracy. In addition to basis/correlation and geometry convergence, the solution comparison lacks solvent, nuclear-motion, concentration/exchange and thermal corrections; OH is particularly condition dependent. The current evidence isolates the discrepancy from the native same-model tensor implementation, but does not partition the experimental disagreement among those model effects. No implementation or reference data was tuned to these measurements.

The first thiophene NMR result was not checkpointed before a harness schema error and was explicitly recomputed; its failure log remains. Other checkpointed native values were reused during auxiliary-solver recovery. The reference engine's large virtual scratch reservation conflicted with the initial2GiB address-space allowance; a labelled4GiB validation-only allowance resolved the reservation without changing formulas or accuracy gates. Its256MB engine memory hint is recorded but does not bound that fixed C scratch reservation. These are not hard process-RSS promises. Final per-case acceptance, exact source identities and the chemical-shift conversion are in `added-five-tranche/recovery-summary.json`, `physical-comparisons.json`, `tms-reference/results.json` and `nmr-shift-comparisons.json` within the existing local campaign.

### Aniline resource-boundary timing and remaining response hotspot

The77-AO3-21G aniline tensor requires281224328bytes,268.196MiB, just beyond the initial256MiB integral-storage budget. A controlled same-artifact, same-geometry, one-thread comparison changes only that existing budget to384MiB. CPU time falls from98.244 to25.397seconds and wall time from98.402 to25.498seconds, a3.87x CPU gain. All reported energy components differ by at most5.80e-12Hartree and dipole components by2.02e-10atomic units; all convergence gates pass. Peak process RSS rises from83028 to463096KiB. An integral-cache allowance is therefore not a process memory cap.

SCF iterations change from20 to6. Source inspection explains an additional effect beyond integral reuse: the accepted HF implementation permits its cached quadratic Newton step for a fixed in-core tensor, while the screened density-dependent direct path does not meet that condition. The ratio combines storage reuse and this convergence-path difference; it is not an exclusive J/K-kernel speed measurement or a general GGA Newton claim. This is one fresh sample per condition in the recorded order, with no statistical universal speed claim.

The original aniline public CIS call completed after1770.648seconds wall/1525.480seconds CPU, including its converged SCF. The425-dimensional default dense A construction evaluates every column, and its dedicated HF J/K engine uses the raw direct mode rather than the SCF automatic cache choice. That source-supported path remains a further performance target; no exclusive time fraction was measured and its healthy run was preserved to completion. The accepted explicit-InCore public Rust NMR call for the same molecule took55.217seconds wall/50.745seconds child CPU and passed every independent tensor gate. These are different properties and are not compared as a speed ratio. The production improvements actually closed here are the derivative cache and magnetic quartet reuse; later changes to default response resource routing require their own coordinated integration and complete-operator/eigenpair checks.

## LiF terminal stress and analytic log-volume response checkpoint

LiF completed the original PBE/STO-3G Gamma reference and analytic stress without changing its controls: E=-105.78207484114978 Hartree, twelve SCF iterations, sampled KS gap2.405586053138203 eV, pressure-65.56133999010856 GPa and rotation defect5.4000070821255404e-15 Hartree. Total SCF-plus-stress wall time5827.197460403 seconds; stress Hartree1563.547 seconds and XC69.693 seconds. This large tensile pressure joins the unresolved Si/SiC pressure discrepancies. It is not a converged physical response reference, and the sampled KS gap is not a measured fundamental or optical gap. The existing serial supervisor proceeded to NaF using accepted artifact02; no healthy calculation was restarted.

New analytic log-volume response source includes physical displacement/mass and Lagrangian/log-volume chain rules, first/second internal relaxation, degenerate-subspace spectral response, explicit internal-stationarity/stability checks for relaxed ions, raw residuals and undefined acoustic/unstable logarithms. All58 independent spectral/coordinate component fits pass, together with four closed/symmetry/input tests. A short warnings-denied metadata check of the curated whole source passed in131.639 seconds. This does not replace a full artifact, public-route execution, actual independent crystal-path fits or material-property acceptance: those remain pending in the serialized build/validation queue. Existing aligned API tables were regenerated from the common registry.

A further exact Ewald-partition study on the LiF geometry used the same synthetic density, uncapped1e-13 screening and every matrix entry. Default split1.92468087 took161.256 seconds; split7 took202.877 seconds and split10 took191.733 seconds, with maximum entry changes5.0248e-12 and5.1505e-12. These slower choices are not adopted. The failed initial calculation-sized linker allowance and successful compiler-specific retry are retained in private attempt records. No physical cutoff or acceptance gate was relaxed.


### Abnormal-pressure diagnosis: operator convergence before material interpretation

The requested LiF/NaCl investigation first checked the B1 structures, Bohr/Angstrom conversion, fractional coordinates, primitive/conventional volume ratio4, unlike-neighbour multiplicity6, charges, multiplicities and12/28 electron counts. These are consistent. The production convention is sigma=V^-1 dE/d strain and P=-Tr(sigma)/3. A small private-harness GPa-factor discrepancy was found and corrected to the existing engine CODATA constant; it changes LiF by only0.00002404GPa and NaCl by-0.000004288GPa, so it cannot explain the anomaly. Original raw results are retained. A misleading physical-direction sentence in the stress documentation was corrected; the numerical stress convention was already correct.

Fully reconverged NaCl energy differences at the unchanged original PBE/STO-3G/Gamma/grid2/shell2 settings give isotropic pressures11.71480829GPa at h=0.008 and11.69908991GPa at h=0.004, approaching analytic11.69384757GPa. The corresponding derivative errors are0.00062144372 and0.00015542497Hartree, nearly a factor4 decrease. Finer steps and independent xx strain remain in progress; two intervals are not represented as the complete four-step acceptance.

LiF has a much stronger numerical warning: its nominal12-electron SCF integrates to11.31870183 electrons on the XC grid. The Li p Gamma overlap diagonal is0.44891307685 with the original12Bohr analytic image cutoff,0.10309575037 with24Bohr, and0.10345324902 from the original real-space grid. The analytic S12/S18 maximum change is0.38025176 and S18/S24 still0.01682314. Raising only quadrature level2 to5 at fixed12Bohr leaves the grid/analytic-S12 discrepancy near0.3454. At level4/shell3, increasing the AO evaluation reach to18Bohr reduces the grid/analytic-S24 maximum discrepancy to0.00042675. This directly establishes severe overlap/AO-tail incompleteness in the original LiF model, rather than attributing it to basis size alone. An actual30Bohr same-basis/same-grid/same-k/fixed-structure calculation is running; it has not yet established a corrected pressure or physical acceptance.

The actual nuclear Ewald terms obey isotropic energy homogeneity to1.49e-11Hartree for LiF and5.03e-11 for NaCl, with consistent four-step isotropic/xx energy derivatives and Ewald-parameter invariance. A LiF Hartree bilinear reciprocity check has defect1.49e-14 at the actual SCF density-image radius32.42693Bohr, falling to1.11e-16 at extended radii. This tested counterexample search does not confirm a Hartree variational bug. An independent all-electron Gaussian PBE check with the exact same basis parameters and Gamma structures gives preliminary pressures+49.79547GPa for LiF and+11.81631GPa for NaCl; its Gaussian auxiliary fitting and atomic quadrature differ, and separate convergence runs remain active. No correction has been fitted to these reference results.

The diamond solid-basis finite-k pilot saved a converged SCF at-75.9708564194567Hartree, sampled KS gap4.87362959eV and grid count12.00404038. Its later XC-stress phase exceeded the private1.5GiB RSS guard and was terminated at08:27:48UTC, retaining the original failure and saved SCF. Earlier wording that treated the stale last log as an ongoing process is superseded by this terminal record. Pressure is unavailable. This is a workspace failure, not physical acceptance; future recovery must allocate a justified bounded workspace or batch the exact AO work. All unaffected original crystal jobs continue.


### LiF terminal stress and analytic log-volume response checkpoint

LiF completed the original PBE/STO-3G Gamma reference and analytic stress without changing its controls: E=-105.78207484114978 Hartree, twelve SCF iterations, sampled KS gap2.405586053138203 eV, pressure-65.56133999010856 GPa and rotation defect5.4000070821255404e-15 Hartree. Total SCF-plus-stress wall time5827.197460403 seconds; stress Hartree1563.547 seconds and XC69.693 seconds. This large tensile pressure joins the unresolved Si/SiC pressure discrepancies. It is not a converged physical response reference, and the sampled KS gap is not a measured fundamental or optical gap. The existing serial supervisor proceeded to NaF using accepted artifact02; no healthy calculation was restarted.

New analytic log-volume response source includes physical displacement/mass and Lagrangian/log-volume chain rules, first/second internal relaxation, degenerate-subspace spectral response, explicit internal-stationarity/stability checks for relaxed ions, raw residuals and undefined acoustic/unstable logarithms. All58 independent spectral/coordinate component fits pass, together with four closed/symmetry/input tests. A short warnings-denied metadata check of the curated whole source passed in131.639 seconds. This does not replace a full artifact, public-route execution, actual independent crystal-path fits or material-property acceptance: those remain pending in the serialized build/validation queue. Existing aligned API tables were regenerated from the common registry.

A further exact Ewald-partition study on the LiF geometry used the same synthetic density, uncapped1e-13 screening and every matrix entry. Default split1.92468087 took161.256 seconds; split7 took202.877 seconds and split10 took191.733 seconds, with maximum entry changes5.0248e-12 and5.1505e-12. These slower choices are not adopted. The failed initial calculation-sized linker allowance and successful compiler-specific retry are retained in private attempt records. No physical cutoff or acceptance gate was relaxed.


### Pressure diagnosis follow-up: four-step NaCl and exact workspace recovery

NaCl isotropic fully reconverged energy derivatives now pass all four steps h=0.008,0.004,0.002,0.001: p=1.9988980391 and R2=0.9999999553. The finest pressure is11.6941758757GPa, compared with the analytic11.6938475732GPa. Independent xx strain checks remain active. This validates the derivative of the specified coarse model; it does not establish material convergence or equilibrium.

All18 independent all-electron Gaussian reference SCFs completed. At quadrature level4 the two auxiliary choices give LiF+49.63784241 and+49.62386320GPa, and NaCl+11.85116960 and+11.83768875GPa. The basis parameters, structures and Gamma sampling match; integral fitting and quadrature differ, and each pressure uses only one strain interval. These are diagnostic comparisons, not fitted corrections or a converged physical reference.

The native same-basis/grid/k/structure LiF30Bohr attempt terminated before SCF under the private1.5GiB RSS guard after986.348s, including953.317s in the core-Hamiltonian stage. The observed guard sample was1675112KiB. No corrected native LiF pressure is available. Full-grid point/image lists and strain moments were identified as avoidable workspaces. Exact64-point AO batches and streamed strain moments now preserve every tested value bitwise, including complex non-Gamma values, gradients, Hessians and all strain moments. Independent four-step fits for648 spatial derivative components pass unchanged gates; the fourth-order gradient cross-check has maximum error9.05435e-10. A portable source-local regression was added. The full owned artifact and actual LiF30 memory/SCF/stress acceptance remain pending, with no physical cutoffs or convergence thresholds relaxed.


## Molecular follow-up: independent closures and remaining physical scope

This checkpoint supplements the five added molecules and the accepted cache/runtime artifact. It does not certify every entry of the property inventory. Numerical agreement within an identical electronic model, comparison with a published different model, and comparison with experiment remain distinct.

### Newly executed independent comparisons

HD at R=1.40117Bohr, HF/aug-cc-pVDZ, now has all nine elements independently compared for each Ramsey contribution, using gH=5.5856946893 and gD=0.8574382338. Independent isotropics are FC54.4696040935241, SD0.54053482511245, PSO0.61801216264574 and DSO-0.20097484583112Hz, total55.4271762354512Hz. Maximum native/independent component differences are respectively2.27e-10,4.74e-11,4.82e-12 and3.35e-7Hz; the total maximum is3.35e-7Hz. The SD tensor includes both traceless mixed FC/SD terms; their isotropic cancellation must not be mistaken for absence. Independent fresh response residuals are at most1.82e-13. The DSO quadrature tail changes by2.75e-9Hz. This closes the earlier missing independent PSO/SD comparison at this point; it does not establish generic C1 derivative certification or experimental HD accuracy. Evidence: `spin-spin-tranche/independent-full-ramsey-results.json` in the existing local campaign.

The public methyl-radical PBE0/IGLO-III result at rCH=1.078Angstrom, doublet, now has an independent full contact-plus-spin-dipolar comparison for all nine tensor components at each nucleus. Independent contact terms are79.1803056428MHz for13C and-64.9825794296/-64.9826052135/-64.9826052135MHz for the protons. Maximum full-tensor discrepancy is1.062e-4MHz, below the predeclared1e-3MHz gate; the spin-dipolar maximum is4.46e-6MHz and its trace is below2.85e-14MHz. Independent/native grids6/5 differ, with energy difference3.31e-9Hartree and independent stationarity7.19e-11. An initial auxiliary spin-normalization factor error is retained separately and was corrected from the operator definition without changing production output or gates. These are one-body spin-density expectations, not a newly validated CPHF route. Evidence: `magnetic-raman-tranche/independent-hyperfine-tensor-results.json`. This paragraph concerns only the public IGLO-III calculation.

Nonzero chiral ECD is now tested for the primary-SI (P)-H2O2 geometry and its mirror, HF/STO-3G, all27 singlet CIS roots. The [primary SI](https://doi.org/10.1021/acs.jctc.4c00747.s001) supplies geometry, not a matched published ECD-strength reference. Maximum independent differences are2.67e-13Hartree for energies,3.80e-12 for oscillator strengths,1.97e-13au for length-gauge and2.05e-13au for velocity-gauge rotatory strengths. Mirror energies and oscillator strengths are identical and all rotatory strengths reverse exactly. Root1 is7.904567549eV, f=0.003459882, Rlength=+11.46569476 and Rvelocity=-19.13387335 in10^-40erg cm^3. Both gauges independently match; their substantial disagreement is an unresolved basis/model gauge error, not an implementation mismatch. Full12001-point Gaussian curves with sigma=0.2eV match independently to2.72e-9 relative peak CD, including the reduced integral/stick-sum conversion. Initial independent origin/sign convention errors are archived. No ECD production defect was confirmed. Evidence: `ecd-chiral-tranche/results.json`, `analysis.json` and `results-initial-convention.json`.

Pure orbital B^4 response is also separated from the earlier mixed E^2B^2 Cotton-Mouton result. Centered He HF/aug-cc-pVDZ gives d4E/dB4=-0.729183355665864au, independently-0.729183355665771au, difference9.24e-14au. The convention chi4=-d4E/dB4 gives+0.729183355665864au. An oblique normalized magnetic direction gives the same scalar. Four auxiliary coupling steps u=B^2=0.08,0.04,0.02,0.01 give p=2.006325 and R2=0.999997323 without discarded points or error floors. For this occupied m=0 reference Lz annihilates the occupied space and d4E/dB4=12d2E/du2; negative u is only an analytic finite-basis validation probe. This is an isotropic atomic scalar and rotational check, not full molecular/C1 fourth-rank acceptance, a spin response, or a matched experimental hypermagnetizability comparison. Evidence: `magneto-optical-tranche/pure-b4/results.json`.

### Exact scope of the three reported timing improvements

All timings below use one computation thread, including linear-algebra thread controls. Integral/cache allowances are not process RSS caps. These are measured workloads, not universal speed claims.

| Workload and controlled change | Before / after CPU seconds | Before / after wall seconds | Peak RSS KiB | Numerical check |
|---|---:|---:|---:|---|
| C1 NH3 PBE/STO-3G,8AO,43326 grid points, grid3/cutoff0; derivative cache off/on; omega0.21 and eta0.13Hartree,3 RHS | 263.78 /17.13 (15.3987x) | response264.184151343 /17.149470092; whole265.477768062 /18.419983169 | 11340 /17704 | All9 complex responses and operator vectors bitwise identical;29 iterations each, residuals2.65e-11/5.60e-11/3.49e-11 |
| Methanol HF/STO-3G,14AO, static GIAO origin0; magnetic quartet reuse, old/new immutable artifacts | 18.588261335 /15.553745221 (16.325% lower) | 18.597329724 /15.632372230 | 94004 /98032 | All54 tensor entries differ at most1.99e-12ppm; independent max3.29e-7ppm |
| Aniline HF/3-21G,77AO, retained geometry; integral allowance256/384MiB | 98.244307 /25.396651 (3.8684x) | 98.401949 /25.497597 | 83028 /463096 | Energy components within5.80e-12Hartree, dipole2.02e-10au; iterations20/6 |

The cache comparison uses the same accepted V2 binary on CPU15. Explicit integral storage128MiB is unchanged; only the internal derivative-cache control changes, from no cache to6238944bytes of data plus headers within8MiB allowance. Cold operator construction takes0.82/0.83CPU seconds. Three warm constructions take[0.81,0.76,0.73]/[0.05,0.04,0.06]seconds, median15.2x. The dynamic response ratio is one trial. SCF thresholds are energy1e-12, density1e-11, gradient1e-10Hartree-based atomic units; response tolerance1e-10, max500, restart40.

The NMR comparison uses fresh processes per artifact, CPU15, energy1e-14/density1e-13/gradient1e-12, integral allowance256MiB, response1e-12/max500/restart40. It is not a separately controlled cold/warm experiment. All common/legacy/ASE/two-CLI routes also pass parity below1e-8ppm, and the independent gate remains1e-4ppm. Process RSS includes bundle/interpreter differences and is not an isolated workspace measurement.

Aniline uses CPU14, the same old immutable HF artifact on both sides, energy1e-12/density1e-11/gradient1e-10 and max300. Its281224328-byte integral tensor crosses the budget boundary. The in-core fixed quadratic HF model also enables the existing Newton step, explaining20 versus6 iterations; the ratio combines reuse and convergence-path changes. It is one fresh sample per condition, not an exclusive J/K speedup or a GGA claim.

V2 library SHA256: `b1434959a36f954cf20e8a97c3acd787fcd000caa7d06fee25fe05c1cc237780`; Python extension `69cef02cd3b7a098ca96c69e2b2735a3d6c561a9680f2230b2b88a7dd2b5cf15`; both CLIs `a13cc3aee5bc73af8ef8bd2331d5eff87d2893eb241a9366914229c599255391`. The old NMR comparison and both aniline cases use library `3d8d7d44ae3965fa0f2d6bfb27e5c51584ac3ed9661014ffc1784c4d40caf5a0` and extension `8d59be3c2175aca4f2a34fdccd885a4a04e5e247a17bc23c0c9f1c62f2d3039a`. The cache trial is same-binary; NMR is a cross-artifact comparison. Library optimization level is1, not an optimized-release throughput claim. Raw final acceptance is `mcd-thermal-cache-integration-20261031-v2/final-acceptance.json` under the validation root; exact per-trial output remains in the campaign.

### Requested-property coverage: completed scope and outstanding evidence

| Requested property | Actual executed evidence | Still missing or unresolved |
|---|---|---|
| Dynamic Raman | Water514.5nm activities/depolarization; accepted HF/STO-3G mode derivatives | Matched absolute measured spectrum, population/polarization/frequency factors; general functional/basis scope |
| Hyperfine | Full public methyl contact/SD tensors independently matched | Experimental phase/motion and basis convergence; no universal open-shell acceptance |
| CD and UV-Vis | Formaldehyde matched primary CIS root; five added molecules all40 roots; nonzero peroxide ECD/mirror/full broadened conversion | Published same-model peroxide strengths, vibronic/solvent/thermal bands and basis convergence |
| VCD | Primary-SI Cartesian AATs plus NEW matched six-mode optimized-peroxide HF/STO-3G frequency/IR/VCD comparison after explicit isotope postprocessing | Public isotope selector absent; measured spectra/ensembles and the distinct MP2 normal-mode reference remain unverified |
| MCD | CIS A/B; supplied real/complex thermal ensembles; generated spin-complete CASCI with nuclear SOC in DFT/HF orbitals; OH/SH absolute spectra and all202944 independent complex operator entries; scoped Rust/Python/ASE/both-CLI acceptance | General TDDFT/SOC, magnetic gauge completion, two-electron SOC and matched measured-spectrum agreement remain open |
| Cotton-Mouton | He1064nm aug-DZ/TZ/QZ and gas-condition comparison | QZ42.85% below experiment; not basis/correlation converged; not universal tensor certification |
| First/second hyperpolarizability | Water beta and HF gamma tensors; source-model comparisons | Matched converged reference conditions and wider models; experimental effective responses include extra physics |
| Hyper-Raman | Scalar owner's numerical derivative work | Latest inspected numerical32 rank3 has13 passes/17 failures (400/570 fits overall);33 only type/lint evidence at this checkpoint. Physical acceptance blocked by unresolved numerical gate, not by an assumed successful API |
| Magnetic dipole | Transition operators plus NEW polarized H2+ nonzero permanent spin moment from public field energies and independent one-electron operators | Dedicated vector output, nonzero orbital/SOC moment, bound-g and thermal ensemble remain unverified |
| Susceptibility/hypersusceptibility | Orbital water second response; NEW complete C1 peroxide pure B4 tensor (15 independent components), He scalar and separate mixed E2B2 | Basis/gauge convergence, Curie/thermal susceptibility and matched physical pure B4 reference remain open |
| Spin-spin coupling | HD full FC/SD/PSO/DSO tensors independently matched | Correlation/basis/nuclear-motion convergence and line-spectrum comparison; native DSO grid-control limitation retained |
| Refractive index/absorption/transmission/reflection | H2 Lorentz-Lorenz gas index; explicitly conditional coherent slab R/T; electronic oscillator-to-spectrum unit checks | Native bulk constitutive validation; nonzero absorption, scattering, windows, measured slab conditions and phase/density model |
| Pockels/Kerr | H2 inversion-zero beta and dc-Kerr all81 independent electronic entries | Nonzero material Pockels tensor; bulk Kerr constitutive mapping, vibrational/rotational/thermal terms; measured H2 discrepancy remains |
| Strict/resummed VPT2 | Explicit compatible method selector; water strict primary values within0.36cm^-1 and identical-force oracle; retained H2/reduced H2CO examples | Full resonant molecular/intensity reference; strict rejects unsupported resonances, reduced polyad is not a full spectrum |
| DFPT/PBC/finite-temperature/dynamic extensions | Separate coordinated periodic and runtime evidence; see response ledger and crystal sections | Molecular finite-temperature collisionless response is not automatic thermal material acceptance; periodic owner retains crystal convergence and linked acceptance |

The scalar spectroscopy owner retains hyper-Raman rank3 correction and its numerical gates. The periodic owner retains material/DFPT convergence; the API owner retains shared CPU/memory controls. No paused worker was restarted, no new functional was added, and no production/reference value was changed by these follow-up comparisons. All newly launched molecular follow-up processes are terminal. The heavy compiler reservation remains released to the previously queued owner. Outstanding source-data, model-convergence and failed numerical gates above remain open rather than being counted as completed physical validation.

The exact LiF30 AO/strain workspace measurement completed on the actual20324-point,10-AO grid: AO values/gradients/Hessians took29.9717s and streamed strain moments41.9015s, with process VmHWM52484KiB. This is a kernel/workspace measurement, not a full-SCF RSS claim or a corrected pressure. The portable source-local regression passes warning-strict whole-unit metadata checking in48.6105s. The full artifact waits behind the resource-controls owner's explicit compiler reservation; dedicated managers will execute the physical-path/public regressions and one guarded same-input LiF30 recovery after that build.

The independent NaCl xx-strain check also completed at all four steps: p=1.9995951910,R2=0.9999999918; the finest equivalent pressure is11.6940739668GPa against analytic11.6938475732GPa. Cubic symmetry makes the other diagonal components equal; this is not a claim that zero shear components admit nonzero-error fits. The existing independent C1 nine-component stress regression is included in the new artifact's queued checks. Urea's initial periodic SCF completed at-444.007741808739Hartree in14 iterations and3997.912s, integrated electron count64.0000795105 and sampled KS gap6.182797432eV. Its initial run did not request stress. Eight distinct initial crystal SCFs are now complete; these coarse fixed-geometry values do not close the material-convergence or expanded-property acceptance.

Extending only the LiF analytic overlap image sum to42Bohr independently resolves the metric discrepancy. Maximum successive full-matrix changes for12,18,24,30,36,42Bohr are undefined,0.380251758837,0.0168231377911,7.95178070e-5,6.92761444e-8,7.55517870e-12. The Li p diagonal approaches0.1030161632774 from the original0.4489130768498. All ten converged overlap eigenvalues agree with the independent same-basis all-electron reference within3.81e-13. Thus the metric disagreement is a real-space image-tail truncation error, not a basis-parser normalization mismatch. Basis coefficients and density were not rescaled.30Bohr remains a diagnostic pressure-recovery point;36Bohr or larger and quadrature/k convergence must still be assessed before material acceptance.


## Molecular permanent moment, complete pure-B4 tensor and matched modal VCD

This bounded continuation closes three earlier molecular gaps using the same accepted V2 extension SHA256 `69cef02cd3b7a098ca96c69e2b2735a3d6c561a9680f2230b2b88a7dd2b5cf15`. It adds no functional, production implementation or reference adjustment. Earlier atomic-only B4 and Cartesian-only VCD limitations remain valid descriptions of their original cases; the new results below extend that scope.

### Polarized H2+ permanent spin magnetic moment

Public finite-field calculations use H2+, charge+1, multiplicity2, HF/cc-pVDZ, R=2Bohr, a normalized bond/field axis (0.310112280972,-0.470170232442,0.826299174462), one thread and the model electron g=2.00231930436256. The alpha-excess branch is quantized along the positive field. Its signed electronic moment is mu=-dE/dB, with magnitude g/2=1.00115965218128Bohr magnetons; the vector is (-0.310471903355,0.470715466377,-0.827257394102)muB. This is the saturated spin-sublevel moment, not the unpolarized zero-field ensemble average or a thermally averaged effective Curie moment. Nuclear spin/rotation, molecular binding corrections to g and relativistic/QED corrections are absent.

Because the field is parallel to the bond, the London phases vanish. The one-electron generalized eigenproblem with h(B)=h0+B^2(r_perp^2)/8 independently reproduces every field energy after adding gB/4, maximum difference3.886e-15Hartree. Four fields0.008,0.004,0.002,0.001au are calculated with spin Zeeman on and off. Their differences give g in [2.00231930436168,2.00231930436479]. All three nonzero components of the forward energy-derived moment have first-order error exponents in [0.999991901,0.999991901], minimum R2=0.999999999982, toward the independent spin expectation. No step was removed or floored. This verifies the programmed free-electron Zeeman coupling and independently determined orbital energy; it does not predict electron g from the molecular wavefunction. The selected alpha-excess state is the upper Zeeman branch, not the thermal ground ensemble. Energy-derived moment postprocessing is explicitly separate from a dedicated public permanent-moment vector API.

[Loch, Stengler and Werth, PRA38,5484(1988)](https://doi.org/10.1103/PhysRevA.38.5484), abstract, reports electron g=2.0022837(18) averaged over H2+ vibrational states4,5,6, versus theory2.0022828(8). The implemented free-electron g exceeds that molecular measurement by35.60436e-6, about17.78ppm. This known omitted bound-g/rovibrational physics prevents a precision experimental-accuracy claim; the parameters were not fitted to this experiment. The physical nonzero spin moment is now exercised, while nonzero permanent orbital moments and general SOC/thermal moments remain unverified here.

### Non-atomic C1 pure orbital B4 tensor

The molecule is H2O2/HF/STO-3G, with the existing primary-SI geometry displaced explicitly by(+0.037,-0.023,+0.019)Bohr on atom0 and(-0.011,+0.029,+0.013)Bohr on atom2 to remove its symmetry. This is a fixed nonstationary molecular geometry, not an equilibrium structure or a measured pure-B4 reference. All positions and atom ordering remain in `molecular-moment-b4-tranche/results.json`. Common gauge origin is(0,0,0); spin Zeeman and current-dependent correlation are excluded. The reported tensor T_abcd=d4E/(dB_a dB_b dB_c dB_d) is the actual fourth energy derivative, with no factorial; pure hypersusceptibility is -T.

Fifteen linearly independent generic directions reconstruct all15 fully symmetric components, design condition number24.5622. Independent finite-field complex RHF solves the same full integral Hamiltonian, with freshly evaluated occupied/virtual stationarity below1e-12 and explicit particle/projector checks for recovered points. The third central difference of its Hellmann-Feynman first derivative estimates the fourth energy derivative. The original0.08,0.04,0.02,0.01au interval FAILED several reconstructed-component fits because higher terms and cancellation obscure the leading error; its maximum extrapolated component difference was7.25e-5au. It remains archived in `results-original-component-fit-failure.json`. The resolved four intervals are 0.005,0.00375,0.0025,0.00125au, with the SAME gates and all15 component fits. Exponents span1.995229924..2.055792074, minimum R2=0.999863320770. Independent extrapolated components differ by at most1.607e-08au, below the predeclared1e-5au gate. No component, failed interval or error is silently removed or floored. This is a complete molecular tensor comparison, distinct from the prior isotropic He scalar and mixed E2B2 Cotton-Mouton response.

| Component | Actual fourth energy derivative, au |
|---|---:|
| zzzz | -0.417608999131 |
| yzzz | 0.116714984991 |
| yyzz | -0.594487498431 |
| yyyz | -0.0828380624543 |
| yyyy | -5.6981113995 |
| xzzz | -0.105304786066 |
| xyzz | 1.10545559309 |
| xyyz | 0.160247443693 |
| xyyy | 0.816223844436 |
| xxzz | -1.34199627498 |
| xxyz | 0.00556678321343 |
| xxyy | -2.34769850027 |
| xxxz | -0.372712049902 |
| xxxy | 4.72928922439 |
| xxxx | 5.05375860053 |

The defining common-origin coupled-HF pure-B4 framework is [Pagola et al., JCP120,9556(2004)](https://doi.org/10.1063/1.1695551). Its accessible primary author record/abstract describes gaugeless basis and coordinate-system convergence; its full numerical tables were not retrieved, so none are invented or claimed to match this displaced peroxide. Basis/gauge convergence and a matched physical pure-B4 reference remain open.

The initial auxiliary script passed an irrelevant response-solver control to finite-field SCF and was rejected before calculation. That error is retained. Some independent complex DIIS/Newton field solves subsequently stalled at the unchanged tight controls; their failures are retained too. Recovery used direct dense complex orbital-stationarity equations with the same1e-12 fresh-gradient gate, preserving completed points and unchanged native outputs. The zero-field approximate orbitals serve only as an orthonormal coordinate chart; their failed convergence flag is recorded and is not used to accept a physical point. An intermediate0.02,0.01,0.005,0.0025 interval improved the component difference to6.60e-7au but still FAILED one reconstructed-component exponent (2.425); this interval also remains archived. The finer interval initially reduced the component difference to5.80e-8au but FAILED that component's R2=0.996682, despite its exponent1.96555. Re-evaluating the SAME f64 integral inputs, Hamiltonian, field steps and components with extended-precision auxiliary orbital exponentials/contractions resolved this cancellation: the full original gates then passed without changing production results, fitting thresholds or inputs. Both the failed double-precision run and the extended results remain available. These are validation-harness, precision and interval recoveries, not a production solver fix or a claimed production speedup. Measured CPU14 contention led to moving only this owned process to idle CPU5, preserving one computation thread; the event and thread affinities are in `affinity-adjustment.json`.

### Six matched peroxide VCD normal modes

[Shumberger and Crawford, JCTC(2024)](https://doi.org/10.1021/acs.jctc.4c00747), reference32, links the primary authors' public source. The archived reference commit `0435144dae38dcc55bcccab74d4a4b7940655c7b`, test013 lines18-40, supplies an optimized HF/STO-3G peroxide geometry plus six frequency/IR/VCD values. This is a different, stable geometry from the earlier SI tensor example: the public V2 `vcd` calculation converges with no imaginary vibrational modes.

The public result uses average atomic masses and projected modes and is retained unchanged. For like-for-like comparison only, independent local postprocessing reweights its SAME newly computed Cartesian Hessian, APT and total electronic+nuclear AAT with1H=1.00782503223u and16O=15.99491461957u and the authors' unprojected six largest modes. No downloaded force constants are used as input; the downloaded author force file is archived only. This is not a new public isotope-selector implementation. With S=M^(-1/2)L, P=APT*S and Mmag=AAT^T*S, R=P dot Mmag; the arbitrary mode sign cancels. No extra1/omega or empirical rescaling is inserted.

| Frequency, cm^-1 | Calculated IR, km/mol | Calculated VCD, 10^-44 esu^2 cm^2 | Author VCD |
|---:|---:|---:|---:|
| 4148.276714 | 30.302597 | 50.537641 | 50.538 |
| 4140.889426 | 12.713776 | -53.527755 | -53.528 |
| 1781.049498 | 2.072667 | 28.606793 | 28.607 |
| 1589.639387 | 46.080354 | -14.649887 | -14.650 |
| 1486.951066 | 0.016843 | 1.098493 | 1.098 |
| 184.632666 | 134.199175 | 101.378036 | 101.378 |

All six modes match the original author gates0.1cm^-1,0.1km/mol and0.5VCD units, with actual maxima0.000125966cm^-1, 0.001674541km/mol and0.000492680VCD units. The author reference explicitly reverses a cited external program's sign; this source convention is recorded, and NO native result was negated to obtain agreement. The average-mass independent comparison has maximum frequency difference0.369978cm^-1, showing why isotope matching matters. Full public average-mass outputs, isotope postprocessing and source signs are separated in `vcd-modal-tranche/results.json` and `comparison-metadata.json`.

This closes a matched six-mode theoretical VCD comparison rather than merely a Cartesian electronic tensor check. It does not establish a measured solution/gas VCD spectrum, conformer/temperature populations, broadening, anharmonic corrections or the earlier paper's distinct common-MP2-Hessian table. VCD elapsed6.55s/CPU4.41s, peak109836KiB, one thread. The final molecular moment/B4 continuation elapsed23.891s/CPU23.278s on one assigned core, peak152752KiB, with reused earlier points and failed-attempt timings separately retained; this is not the total campaign cost. All owned processes are terminal and CPU5/14/15 are released. No compiler was launched or reservation reclaimed.

A separate heteronuclear C1 stress regression now passes all27 components across SVWN5,PBE and TPSS on accepted artifact02, with four distinct strain steps and zero grid-weight cutoff. The old complete-grid reference suite took120.552s; its full output and an earlier test-literal compilation failure are retained. The exact-batched full artifact must still rerun the same gates. NaCl fully reconverged isotropic term slopes at h=0.001 are nuclear108.75261294,kinetic0.61680403,nuclear-attraction-198.44233200,Hartree88.52785105,XC0.19835481Hartree, summing to-0.34670917131Hartree within7.11e-12. These individual slopes include density response; comparison with bare analytic stress terms must include the density chain/Pulay contribution. A fresh unchanged-parameter centre SCF and analytic component ledger are running for that comparison, with no restored or forged derivative identity.


## Thermal MCD C: explicit effective-manifold implementation and physical limits

The existing molecular `mcd` route explicitly rejects unrestricted/smeared references and computes closed-shell singlet-TDA A/B terms only. Its finite-temperature fractional-orbital response therefore cannot be counted as a paramagnetic C term. Missing ingredients for automatic molecular C spectra are a compatible ground magnetic multiplet, SOC/spin-resolved state-to-state optical strengths, coherent magnetic operators, their thermal density, and the required host/rotation/conformer model. This continuation implements only the independently verifiable supplied-real-manifold part; it does not silently claim those missing generators.

`mcd_model="thermal-manifold"` uses H(B)=H0-B.M with all matrices supplied explicitly, temperature in Kelvin and field in atomic units (1au=235051.756758T). Three supplied real symmetric Q operators encode normalized left-minus-right optical helicity. The canonical density is exp(-H/kT)/Z, not Fermi occupations. It returns finite-field populations and normalized helicity, plus separate zero-field population-C and interlevel-mixing tensors. Within a degenerate level the C derivative is beta times the normalized magnetic/optical covariance, including off-diagonal coherences; interlevel divided differences use expm1 and no denominator floor. Multiplying the population tensor by kT gives its reported C numerator. The finite-field signal contains both supplied-manifold mechanisms where present, and is not labelled a pure C term at arbitrary field. Q is field independent; this state-mixing tensor is not the complete excited-state Faraday A/B response and includes no field derivative of the optical operator. Degeneracy tolerance affects only the decomposition, not the total derivative.

The default `closed-shell-tda` path retains its A/B semantics. Unknown model names, nonpositive temperature, mismatched/non-symmetric matrices and inapplicable SCF controls are errors. Only real symmetric supplied models are supported here. The ASE adapter selects this model without injecting calculator-default SCF parameters; per-call invalid SCF parameters still reach strict driver rejection. Inputs and outputs carry units; geometry is explicitly marked unused. A complete curated shared artifact now passes actual Rust/native Python/ASE/both-CLI numerical acceptance; exact scope and artifact evidence are recorded below. Historical source-local tests remain explicitly distinguished.

Eight source-local unit cases pass. Across four fields0.008,0.004,0.002,0.001, all9 pure-population and all9 population-plus-mixing components are fitted, and an additional18 fits separately verify each returned population-C and interlevel-mixing component using frozen-eigenvector versus full field calculations:36 fits total, p=1.9997395374..1.9999453444, minimum R2=0.999999995443. Near-degenerate grouping is separately checked to preserve the total response; unsupported complex input has a named real-operator error. Tests also cover nonzero two-level C, the Curie limit, saturation, continuous degenerate divided differences, basis/energy-offset invariance, population normalization, explicit units and invalid inputs. All original gates are retained. The literal new MCD driver/module linked to the accepted V2 numerical library has worker SHA256 `0b8670abf5d8537aaf5ae5daaaa0d18e048e962ca885de27d8f7a18685837650`. This is explicitly a source-local artifact, not a newly rebuilt common-driver Python/CLI bundle.

A numerical stability defect was independently reproduced: adding an irrelevant common energy origin of-1000Hartree changed a finite-field normalized signal by3.83997e-10, exceeding the predeclared1e-11 invariance gate. The corrected calculation subtracts the common H0 origin BEFORE applying the field and diagonalizing, then computes populations/response from centered energies. Original absolute eigenenergies remain available, supplemented by `mcd_thermal_relative_energies`, `mcd_zero_field_relative_energies` and `mcd_energy_origin`, all explicitly labelled in Hartree. Across tested origins0,-1,-100,-1000Hartree, the corrected signal is bit-identical; an eight-case regression also checks positive origins, all signal components, populations and relative energies. Raw before/after results remain in `thermal-mcd-manifold-20261031/energy-origin-{before,after}.json`. No physical reference or acceptance gate was changed. Fresh eigen residuals refer to the centered Hamiltonian. The rebuilt public routes now also pass this unchanged energy-origin regression; original failure evidence is retained.

### Actual OH effective-model C response

The [primary OH/Ar experiment](https://doi.org/10.1021/jp970076q) used1.7-7K,0-4T and random orientations. Its full site-resolved parameters and calibrated optical tables were not retrieved. A [later primary model paper](https://ir.canterbury.ac.nz/bitstreams/e1d31f06-307c-4e72-bc5b-60ba6d965a86/download), Harrison and Williamson, J. Indian Inst. Sci.85,391-402(2005), pp397-400 Eqs16-31, supplies the SO/crystal-field doublets, Boltzmann partition function, polarized absorption and orientation integral. Its own p401 conclusion says that the model is inappropriate for SH and probably OH because host coupling is missing. Consequently the following are matched calculations of its declared model limits, not an accepted fit to the OH/Ar measurements or a free-rotating gas/DFT spectrum.

For a separated lower doublet, Eq31 gives C0/A0=3*kappa*integral_0^1[(g_parallel*u^2/g_theta)*tanh(g_theta*muB*B/(2kT))]du, g_parallel=ge+2*kappa, g_perp=ge*sqrt(1-kappa^2), g_theta^2=g_parallel^2*u^2+g_perp^2*(1-u^2). We set ge=2 and explicitly use the large-splitting lower-doublet approximation, with kappa=1 and0.54 from the experimental reduction-factor range. These are limiting model inputs, not fitted site parameters or predicted molecular g values. The native supplied2x2 Hamiltonian/magnetic operators are diagonalized for each orientation; its normalized Q is fixed algebraically from the same source helicity definition, with no post hoc sign or scale fit. A separate adaptive integration of Eq31 is the independent oracle.

| kappa | T,K | C0/A0 at0.1T | at1T | at4T |
|---:|---:|---:|---:|---:|
| 1.00 | 1.7 | 0.0789266285213 | 0.706289243746 | 1.378503521522 |
| 1.00 | 4.2 | 0.0319798288547 | 0.313503862034 | 0.990731569372 |
| 1.00 | 7.0 | 0.0191904096825 | 0.190519170736 | 0.690059925986 |
| 0.54 | 1.7 | 0.0328295141487 | 0.302365374962 | 0.628079248759 |
| 0.54 | 4.2 | 0.0132979990084 | 0.131099718928 | 0.436229197869 |
| 0.54 | 7.0 | 0.00797954221906 | 0.0793844408138 | 0.295023898201 |

All zero-field values are exactly zero. Across all24 conditions, native source-local response versus independent Eq31 differs by at most2.1094e-15;64/128-point orientation quadrature changes by at most2.5535e-15. Population normalization defect is2.22e-16, eigen residual2.54e-21Hartree and interlevel mixing is zero for this isolated doublet. Six signed-field derivative fits over0.04,0.02,0.01,0.005T give p=1.99987067..1.99999521, min R2=0.999999998877. Field reversal defect is5.55e-17. The weak-field/high-temperature B/T ratio is1.000000000002318 of the analytic limit; at kappa1,T=.001K,B4T the ratio approaches1.499999956734, versus the source limit1.5. These checks distinguish genuine magnetic-sublevel population C and saturation from closed-shell smearing. The high-temperature check is an algebraic limit of the supplied isolated doublet; it does not validate neglect of higher electronic manifolds at high physical temperature.

No absolute molar intensity, linewidth, vibronic band, host-site distribution or experimental-error bar was invented. Orientation averaging is explicit validation postprocessing, not a new native powder-spectrum API. The original OH site inputs, general complex spinor transitions and automatic DFT/SOC manifold generation remain physical/implementation gaps. Raw input/output JSONL and source locations are in the existing `mcd-tranche/thermal-effective-model-{results.json,raw.jsonl}` and `thermal-primary-research.json`; local compiler/tests are in the separate existing validation root `thermal-mcd-manifold-20261031`. The actual model tranche used one CPU,4.09s wall,2.42s Python CPU plus1.04s worker CPU, peak105448KiB; all its processes are terminal. No full compiler was launched alongside the prior PBC reservation, and no new functional or excluded basis data was introduced.


### NaCl term-by-term pressure acceptance and consistent LiF supports

A fresh original-parameter NaCl centre SCF/stress completed in545.583s and reproduced pressure11.69384757318GPa, energy-615.217111162063Hartree and grid integral28.00340118861. Isotropic analytic strain-derivative terms are kinetic-0.66511464509,Pulay-0.11090141157,nuclear attraction-197.04247805032,Hartree88.47416378532,XC0.24512669892,nuclear repulsion108.75250418496Hartree. Independently contracting each fully reconverged density derivative with its centre operator separates the density chain from individual energy slopes. All six terms pass four-step fits for both isotropic and xx strains: twelve fits, p=1.99944426..2.00115678,minimumR2=0.9999997179. The total density chain is consistent with the overlap/Pulay term. No term-specific stress kernel defect is identified under this fixed coarse model; basis,k,grid and equilibrium accuracy remain separate.

The LiF30Bohr recovery consistently expands short-range S/T/nuclear image support and the default XC AO reach. The existing Hartree reach also includes the one-electron cutoff, so its density-image radius increases from about32.43 to60Bohr. Analytic stress uses the corresponding supports. Ewald splitting and geometry stay fixed. The result must therefore be interpreted as joint image/AO/density support convergence, not a change to overlap alone. An independent Fourier-Poisson regression now checks all complex Li/F p-block entries at Gamma and two generic k points: the original12Bohr errors are0.345897,0.323355,0.304683, and the42Bohr errors are3.84e-13,2.31e-13,1.35e-13.

The owned volume/cache/AO-batching library,canonical CLI and portable audit completed successfully at10:21:42UTC and the compiler was released. The first physical-path volume test correctly hit the existing charged-cell/background-derivative refusal because its positive fixture was HHe+. That fixture choice was incorrect for this API. The original source and failures remain frozen; the guard is preserved, and a neutral genuinely C1 HHHe model with a nonempty virtual space now exercises90 independent volume-jet components on the same artifact. A separate negative test preserves the charged-cell refusal. No second-Gruneisen physical acceptance is inferred from metadata, spectral tests or build success.


### Actual-density normalization diagnosis and completed neutral volume oracle

At the actual fully reconverged LiF h=+0.008 original12Bohr state, the density has Tr(DS12)=12.000000000000009 but Tr(DS42)=11.374017383741984. The actual original XC integral is11.371332656656620. Thus the dominant deficit persists even with the independently converged metric and cannot be repaired by quadrature refinement alone. At fixed density, extending AO reach12to30 at grid2/shell2 gives11.367021376045583;30to36 changes no reported digit. Grid4/shell3/AO36 gives11.372704443271909; higher-grid/partition checks continue. This state is at+0.008 strain and is not substituted for the unstrained reported11.3187018286-electron state. No density was rescaled or restored as a validated SCF object.

For that same actual density, extending the Hartree density-image radius32.68634to49.02951Bohr changes the energy by-3.58114e-12Hartree and every complex matrix entry by at most1.40688e-12; extending to65.37269 produces zero further reported matrix/energy change. This actual-density convergence check is stronger evidence than bilinear reciprocity alone and does not identify Hartree density-image truncation as the pressure anomaly.

The corrected unstrained LiF30Bohr SCF now converges at-105.653002051964Hartree in10 iterations,3709.053s, with grid integral11.9921479621168. Analytic stress remains in progress. The blockwise implementation crossed the previous RSS failure and its process peak was still below50MiB at the end of SCF; final stress peak is recorded separately. Remaining grid error must still be converged. The shared resource candidate independently rejects the exact AO test at16MiB with a typed allocation error and completes at64MiB, with managed Matrix peak34602400bytes; unregistered bounded image/grid/process work remains outside that ledger, so this is not an RSS cap.

The neutral C1 HHHe volume-response test completed both positive and charged-refusal cases in1262.188s: all90 independent first/second dynamical-jet components pass four-step gates. Native/ASE/CLI tests continue on the same compiled artifact. The long legacy finite-k stress regression was approaching an undersized1800s Python timeout; only that controller was replaced, preserving its original Rust process and all gates. The observer retains its full numerical harness summary and explicitly does not invent an unavailable OS exit code. Unaffected Hartree,XC,spin and diffuse-image regressions continue independently.

A private selective-shell-pair nuclear-attraction prototype avoids discarded full concatenated matrices. All3072 Cartesian/spherical SPDF scalar entries across Coulomb/Erf/Erfc and matched periodic image blocks are bitwise equal to the existing implementation. One-CPU ABBA timings on a small three-block C1 fixture are0.42608/0.39268s for the original and0.02373/0.02399s for the prototype. This is a small-kernel measurement, not production adoption or a full-crystal speedup. A narrow unchanged-body scalar pair-helper seam is proposed to the shared integral owner; that shared file and running artifacts remain untouched.


### Linked public thermal-manifold acceptance

The immutable `thermal-mcd-public-integration-20261031-01` bundle was curated from accepted resource06 plus only the owned thermal implementation/routing/tests. Its full library and both CLIs pass the portable audit and warnings-strict checks. Actual linked Rust tests pass all36 component fits (p=1.9997395374..1.9999453444, minimum R2=0.999999995443) and preserve the three closed-shell Faraday regressions. Native named/common Python, ASE atomic/physical-unit routes, explicit resource controls, canonical/alias CLIs, invalid temperatures/populations/matrices/complex inputs and energy-origin invariance pass: 18 cases per CLI configuration, with no skips. One routing-specific mock test is included in that count and does not replace the numerical route cases.

The same loaded extension reruns all24 declared OH lower-doublet model conditions and six field fits. Maximum normalized Eq31 difference is 2.10942374679e-15; this retains the primary-source approximation and experimental limitations above. A separate non-chemical real four-level model containing two doublets verifies all9 population-C,9 mixing and9 total tensor components against an independent matrix-exponential Frechet derivative and projector-trace decomposition; all27 independent four-step fits pass. This is a general multilevel algebra test, not a source-matched four-state OH or experimental benchmark. Full numeric comparisons are retained in `four-level-public-oracle.json`.

Artifact SHA256: library `ef2bf6a01c2e92148be2ff8148a2484c09d4e73766c679878ce26b1e752f5b3c`, extension `ee21600f054f2fc0ba311fe952a3f2a31d23a99fe8dd32f43df27101e3d545ed`, canonical CLI `d3e21434eb32cd558a8d8afc7877c1741a5b2e4568dfc043daed13a25fd28f0e`, alias `f9ad06bde0961f001606387ab38f45c67c2d6e84f524f33b03d6bd77299addad`. Exact input/source hashes, timings, thread settings, observed memory and logs are in `thermal-mcd-public-integration-20261031-01/final-acceptance.json`, `state.json`, `sources.json` and `OH-effective-model-clean/thermal-effective-model-results.json`. The first frontend attempt omitted the declared Cargo MSRV environment; its unchanged original warnings and state remain archived. Restoring the same MSRV1.85 environment used by the accepted baseline passed without changing source, lint or numerical gates. The first completed OH harness left its stdout pipe unclosed at interpreter shutdown; the original warning/output is retained, and an explicit stream-close correction reruns all24 conditions and six fits with warnings treated as errors and no warning. Its final evidence is in `OH-effective-model-clean`; no scientific formula or gate changed. All owned processes are terminal and the compiler reservation is released. Automatic molecular DFT/SOC sublevels/transition generation, complex operators, absolute/vibronic/host spectra and matched experimental open-shell C acceptance remain unimplemented or unestablished; this bounded model must not be counted as those capabilities.


### Terminal corrected LiF pressure and finite-k coverage distinction

The same fixed B1 LiF geometry (a=3.964Angstrom), PBE/STO-3G, Gamma, grid2/shell2 and unchanged strict SCF gates now completes with consistently expanded30Bohr image support: E=-105.653002051964Hartree, P=+51.3770339797293GPa, grid electron number11.9921479621168 of12, and rotation residual3.4660e-15Hartree. The retained12Bohr result was P=-65.5613159505990GPa and grid number11.3187018286. The measured image-support deficiency changes the pressure sign; this is not a pressure-sign/unit/primitive-volume correction. The full corrected calculation took5401.282s, of which SCF took3709.053s, and observed process peak RSS was80508KiB. Earlier1.5GiB-guard failures remain retained. This successful completion demonstrates the exact batching memory recovery, not a matched end-to-end speedup over a completed earlier run.

The correction includes the one-electron and XC supports and dependent density-image reach consistently. No density normalization or physical/SCF tolerance changed. Remaining grid error and finite-k/basis/equilibrium convergence are explicitly unfinished; +51.377GPa is a diagnostic pressure at the specified fixed geometry. A registered refinement worker started after this terminal result, reusing the same immutable Cell for separate grid4/shell3/Gamma, grid6/shell4/Gamma and grid6/shell4/k2 references, with fresh SCF and preserved density/Fock/overlap/core matrices before stress.

The existing long finite-k stress regression completed in4832.29s, including the retained failed coarse shear interval and a passing resolved interval. Its derivative sampling covers3components; it does not establish all9finite-k components. The separateGamma XC batching suite passes all27components across LDA/GGA/meta-GGA. An additional C1 finite-k2x1x1 PBE test now fits all9components and explicitly requires measurable intercell overlap. Public volume parity initially failed only the ill-conditioned signed square roots of acoustic zero eigenvalues; original logs remain. The ongoing rerun preserves all matrix/eigenvalue/nonacoustic/gamma gates and additionally checks acoustic signed squares within128machine-eps times spectral norm using persisted native/ASE/CLI reports.


### Completed twelve-structure overlap support and nine-component finite-k oracle

The actual twelve crystal inputs now have completed Gamma overlap-matrix support checks at12,18,24,30,36,42Bohr. The table gives maximum absolute matrix-entry differences for selected successive radii. These are numerical-support diagnostics, not twelve converged electronic/property calculations. In particular, the original ferrocene12Bohr calculation continues as retained diagnostic evidence; its appreciable overlap-tail error cannot be silently reclassified as a converged crystal reference.

| Crystal | AO count | Delta S(12to18) | Delta S(30to36) | Delta S(36to42) |
|---|---:|---:|---:|---:|
| diamond | 10 | 2.134343e-06 | 0.000000e+00 | 0.000000e+00 |
| ferrocene-monoclinic-average | 156 | 9.493382e-03 | 1.040834e-17 | 1.592831e-26 |
| kaolinite-bish-1p5K | 85 | 3.528937e-05 | 0.000000e+00 | 0.000000e+00 |
| lithium-fluoride | 10 | 3.802518e-01 | 6.927614e-08 | 7.555179e-12 |
| magnesium-aluminate-spinel | 94 | 1.328170e-04 | 4.747538e-29 | 0.000000e+00 |
| magnesium-oxide | 14 | 2.586747e-04 | 0.000000e+00 | 0.000000e+00 |
| potassium-magnesium-fluoride-perovskite | 37 | 1.085848e-02 | 2.824436e-16 | 9.860761e-32 |
| silicon-carbide | 14 | 5.423510e-05 | 0.000000e+00 | 0.000000e+00 |
| silicon | 18 | 1.208031e-04 | 0.000000e+00 | 0.000000e+00 |
| sodium-chloride | 18 | 6.296853e-05 | 0.000000e+00 | 0.000000e+00 |
| sodium-fluoride | 14 | 1.161050e-04 | 0.000000e+00 | 0.000000e+00 |
| urea | 48 | 2.905127e-06 | 0.000000e+00 | 0.000000e+00 |

The added finite-k2x1x1 C1 PBE regression now passes all9independent stress-component four-step fits over h=0.04,0.02,0.01,0.005: p=1.990358to2.011312 and minimum R-squared0.99999374. Runtime is170.882s. It requires an explicit overlap contrast greater than1e-6 between k points, so a numerically isolated molecular cell cannot stand in for the finite-k acceptance. The earlier27Gamma XC component fits and existing three-component finite-k regression are retained separately.

### Accepted force-Hessian streaming and public volume parity

The follow-up resource-stream01 artifact completed its full library/CLI build
and portable audit, followed by all27Gamma LDA/GGA/meta-GGA stress fits,
all9finite-k PBE stress fits, the diffuse-image overlap oracle, and the two
HF/PBE native/ASE/CLI particle-number report cases. These latter cases verify
matching ASE input conversions and unchanged dimensionless metric/grid counts,
including null unavailable HF grid diagnostics. The new report fields are
therefore accepted on this artifact, superseding the earlier source-only note.

The force path retains global AO values/gradients and streams only the
force Hessians in64point blocks. Every lower-order entry matches the former
Hessian build bitwise on SPD Gamma/generic-k controls; the retained648spatial
derivative component fits also pass. For the actual120002point LiF grid6/shell4
with10AOs and8kpoints, the principal dense AO array sizes are1,536,025,600bytes
for full Hessians versus614,410,240bytes for global values/gradients plus819200
bytes for a local Hessian block. These are allocation-size estimates, not an
observed whole-process RSS or a measured whole-SCF speedup. Large-case runtime
memory and physical convergence remain separate checks.

The neutral clamped-ion volume API rerun completed4pytest cases in3806.69s.
Actual native, ASE and CLI reports additionally pass the stricter acoustic
signed-square comparison within128machine-eps times the reference spectral
norm. The native/ASE acoustic squared-frequency difference is3.95567e-20
against a2.58529e-17roundoff bound. All original force-constant, dynamical-jet,
reference-eigenvalue, nonacoustic-frequency and first/second volume-response
gates are retained. The original failed square-root comparison is preserved;
acoustic frequencies are neither floored nor silently set to zero. The90
positive physical-path component fits validate clamped ions. A nontrivial
stationary/stable relaxed-ion positive oracle and actual crystal gamma values
remain pending, distinct from the tested stationarity/charged-model refusals.

The fixed-geometry LiF30Bohr grid4/shell3/Gamma reference now converges at
-105.6550286669295Hartree, with grid number11.9985370634313, compared with
grid2/shell2 energy-105.6530020519641Hartree and number11.9921479621168.
No stress was requested for this intermediate grid-only step; the grid6/k
pressure checks continue. Reusing the same immutable Cell reduces the measured
core-Hamiltonian stage from1039.358s on the first case to0.027s on the next
grid-only case. This is existing geometry-cache reuse by the workflow, not a
claim about whole-SCF speed or cross-geometry invalidation.

Published solid-optimized pob-DZVP-rev2 private validation inputs now cover
H,Li,C,N,O,F,Na,Mg,Al,Si,Cl,K,Fe. All404raw exponents/signed coefficients
match the independently structured input representation bitwise, and every
selected atomic AO normalization check passes. The structured data contains no
ECPs for these elements. The primary definitions and rev2 reference are the
[author's basis page](https://www.chemie.uni-bonn.de/bredow/de/software/basis_sets)
and [2019 paper](https://doi.org/10.1002/jcc.26013). These are private benchmark
inputs, not a new public source registration or redistribution-license claim.
Their additional LiF/NaCl/SiC calculations and matched failed-diamond recovery
have not started; resource admission remains unresolved. Initial crystal SCFs,
input preparation and implementation tests do not complete the twelve-crystal,
eighteen-family physical validation scope.

The original 12Bohr LiF model now has two independently reconverged isotropic
energy-difference pairs: pressure is -64.70144523GPa at strain step0.008 and
-64.09849115GPa at0.004, versus analytic -65.56131595GPa. The discrepancy grows
from0.85987 to1.46282GPa; these two points do not establish derivative convergence
and must not be reported as a successful stress check. The existing series
continues at0.002/0.001 and independent axial strains. The energy decomposition
closes within1.42e-14Hartree for these four SCFs. Grid electron numbers span
11.26308–11.37133 rather than12. The logged Hartree pair/translation sets also
change under strain, so support/topology effects remain under investigation;
those counts alone do not establish the cause of the residual stress mismatch.
Fully relaxed-density energy-term derivatives are not the individual fixed-density
analytic stress terms. Corrected-support grid6/shell4/Gamma SCF has separately
converged at -105.6555582858601Hartree with grid number11.9997643487228;
its analytic stress is still pending, and no stationary-material acceptance is
inferred from this reference.

At14:19UTC the grid6/shell4 analytic stress completed. At unchanged B1
a=3.964Angstrom, PBE/STO-3G, Gamma sampling and30Bohr image support, pressure is
**+49.5989181772GPa**, energy -105.6555582858601Hartree and grid number
11.9997643487228/12. The ten-iteration SCF took2460.627s and the complete
SCF/stress step5457.440s. Nuclear-attraction, Hartree and XC stress stages took
440.492,1228.489 and1326.022s respectively. The existing batch advanced to
2x2x2 sampling without restarting or restoring a synthetic SCF reference.

The same-support change from grid2/shell2 is -1.7781158025GPa. Its individual
pressure contributions change by -0.01799088 kinetic,+0.00920818 Pulay,
-0.08861840 nuclear attraction,+0.10256635 Hartree,-1.78328106 XC and0 Ewald,
all inGPa. Both quadrature density and partition shell change in this comparison;
the XC contribution is the main observed change, not a separated attribution
to radial/angular quadrature alone. Direct -Tr(stress)/3 conversion reproduces
the reported pressure exactly; strain-term closure is1.97e-15Hartree and the
symmetrized stress closure is1.86e-17Hartree/Bohr^3. These arithmetic identities
are not an independent derivative test. Independent same-basis/Gamma energy
differences give about+49.624 to+49.638GPa with two auxiliary fits; those use
only h=0.004 and retain integration/fitting uncertainty, so they do not provide
four-step acceptance. The remaining positive pressure is a nonstationary
minimal-basis result, not an experimental material prediction. Proper-basis,
k-point, equilibrium and corrected-support multistep stress acceptance remain
open; the original-support failing evidence is retained.

The original12Bohr h=0.002 pair also completed: its energy-difference pressure
is -61.91006965GPa, farther from analytic -65.56131595GPa than either coarser
pair. Saved-matrix postprocessing of all six converged h=0.008/0.004/0.002
states gives a maximum Tr(DS)-12 defect8.88e-15, spin-density difference0,
DSD-D maximum1.78e-15 and generalized Fock-commutator maximum4.92e-9.
The minimum sampled orbital gap is0.0779517Hartree. These distinguish the
nearly exact truncated-metric electron count from the deficient real-space
grid count. They do not establish cross-geometry SCF-branch continuity, remove
the stress discrepancy or replace corrected-support four-step acceptance.

The KMgF3 perovskite initial electronic reference completed with exit0:
Pm-3m, five-atom cubic cell a=3.9859Angstrom from
[COD9006363](https://www.crystallography.net/cod/9006363.cif), undoped x=0.
The source does not specify measurement temperature. The calculation is static,
neutral, restricted all-electron PBE/STO-3G with29alpha and29beta electrons,
Gamma sampling, grid3/shell3 and zero grid-weight cutoff. The30Bohr one-electron
support was selected by the actual-basis Gamma-overlap convergence preflight;
this does not certify every operator or k mesh. Fourteen SCF iterations give
E=-1086.3974215517298Hartree and grid count57.9994180024372/58, in10513.927s.
Fresh final checks give residual7.084e-11, RMS density change1.031e-11 and
energy change4.320e-12Hartree. The sampled Gamma KS gap is3.24346968eV,
not an optical or fundamental experimental gap. Stress was not requested for
this initial many-atom case; its pressure remains absent rather than zero.
Nine of the twelve distinct initial crystal SCFs now have converged saved
results. None of these counts closes stationary material/property acceptance.
The existing serial controller retains spinel and kaolinite, waiting for its
unchanged CPU/memory admission conditions before each next calculation.

The serial controller subsequently admitted the MgAl2O4 spinel at15:21UTC.
This is the14-atom primitive repeat from
[COD9005807](https://www.crystallography.net/cod/9005807.cif), Fd-3m origin
choice2. Its input uses the CIF MgA tetrahedral and AlB octahedral site
assignments (the CIF has no partial-occupancy column); cation inversion or
statistical disorder is not modelled. The source pressure is0.001kbar and
measurement temperature is unstated. PBE/STO-3G/Gamma, grid3/shell3 and24Bohr
support are the initial diagnostic settings. No converged spinel result or
mechanical-property acceptance is inferred from starting this calculation.

The running LiF2x2x2 calculation uses the implementation's unshifted mesh:
fractional reciprocal coordinates (i/2,j/2,k/2), i,j,k=0,1, with equal1/8
weights and no half-mesh offset. Thus it includes Gamma and the seven other
0-or-1/2 combinations. Cartesian k vectors are generated from the reciprocal
cell; strain comparisons hold the fractional coordinates fixed. This explicit
definition should also be used for matched independent k-point comparisons.

A retained public-response record was re-audited: frozen spin-public02
(rlib9833c7eae4ed44d09f7e112bb324523c9f762d7f7ab0835808fab83b92ccf772)
has six native/ASE/canonical-CLI longitudinal-spin cases passing in51.71s,
and all eleven retained Rust/public acceptance checks exit0. The first ASE
physical-unit test failure remains archived; its retry converts the reference
controls consistently and changes no numerical tolerance. These are existing
finite-q/frequency longitudinal local-Zeeman and ensemble controls, not a new
run on resource-stream01, a material magnetic-response benchmark, or evidence
of orbital, transverse, SOC or real-time-propagation coverage.

The LiF2x2x2 SCF subsequently passed its fresh final check: ten iterations,
5191.701s, E=-105.9407556578543Hartree, grid count11.9997731435794/12 and
sampled KS gap7.70250967eV. The final residual is4.773e-11, RMS density
change1.875e-11 and energy change4.547e-13Hartree. At the same geometry,
basis,30Bohr support and grid6/shell4, Gamma gave E=-105.6555582858601Hartree
and gap5.86284293eV. The same-model energy change is -0.285197371994Hartree
per primitive cell; these two meshes therefore cannot be called k-converged
merely because both grid electron counts are close to12. This is a sampled
KS comparison, not an optical/fundamental gap or a comparison of absolute
energies across methods. The converged SCF report and operators were saved
before analytic stress, which remains pending. The positive Gamma pressure
and old-support failing derivatives are not waived by this SCF completion.

Complex-matrix postprocessing of that saved k2 state gives12.000000000000009
metric electrons per primitive cell with equal1/8 weights. The maximum
imaginary trace is8.20e-33, idempotency defect2.89e-15 and generalized
Fock-commutator defect8.42e-11; alpha/beta densities agree exactly. No density
renormalization or SCF-object reconstruction was used. The finite grid-count
error and pending stress/convergence checks remain separate.


## Automatically generated complex CASCI/SOC molecular MCD

The new `mcd_model="molecular-casci-soc"` source generates common spatial orbitals from the requested existing DFT/HF reference, constructs every determinant in an explicitly bounded contiguous active space and diagonalizes the physical Coulomb Hamiltonian plus nuclear one-electron SOC. All spin projections are included. The functional generates orbitals only: these are CASCI excitation energies, **not TDDFT**. `mcd_frozen_core` and `mcd_active_orbitals` specify spatial counts; an excessive determinant count is refused rather than silently truncated. The same state space supplies complex magnetic/electric operators and canonical fixed-electron many-electron Boltzmann populations. No empirical molecular g factor, optical scale or measured excitation energy is supplied.

H(B)=H(CASCI)+H(SOC)-B.M, with M=-(L+ge*S)/2 in atomic magnetic dipole units and ge=2.00231930436256. Nuclear SOC uses full nuclear charges at scale1. The original common-origin model lacks London-orbital completion, two-electron SOC and quadratic diamagnetic terms; its default origin is the nuclear-charge centre and it does not establish arbitrary-origin independence. The later explicitly selected London-linear/frozen-SOC increment below supplies and verifies the first-order gauge completion only.

Positive-frequency Lorentz lines have analytic area4*pi^2*alpha*transition_energy*(p_initial-p_final)*|e.d|^2, using the declared Hartree HWHM `spectrum_width`. Cross sections are bohr^2 per molecule. Left polarization is (e1+i*e2)/sqrt(2) in a right-handed(e1,e2,k) frame; MCD is left minus right. The normalized ratio divides by their mean and returns null if it is zero. These are absorption spectra, not a complete causal dispersive or vibronic response. `mcd_orientation="isotropic"` streams polar/azimuthal quadrature with field parallel or antiparallel to light. State/transition tables are the representative input orientation; separate diagnostics retain each orientation's populations/residuals. Complex transition phases are not averaged.

`energy` remains the orbital-generating reference energy. `state_energies_relative` plus `energy_origin` are active-electron levels; adding `frozen_core_electronic_energy` and input nuclear repulsion reconstructs total CASCI levels. Optical energy differences cancel these common constants. The separate `thermal-complex-manifold` route retains supplied Hermitian matrices as real base controls plus matching `_imag` quadratures. Realification duplicates numerator and partition equally; returned energies/populations count physical states only. Exact Hermiticity is required and imaginary data are never silently discarded.

### Source-local numerical evidence

The permanent source-local Rust adapter passes11 cases, including the previous real-thermal tests, all27 new complex population/mixing/total component fits against an independent Pauli formula, phase covariance and invalid input. Full H2/STO-3G CI gives-1.137275943782718Hartree with either HF- or PBE-generated orbitals, within2.05e-14 of an independent identical-coefficient calculation. OH frozen-core CAS(7,4) differs by1.90e-12Hartree in the corresponding scalar comparison. A separate complex four-level matrix-exponential oracle verifies all27 components and a general complex unitary transformation within5.56e-16.

Three independent molecular integral calculations verify **all1344** Coulomb/SOC/magnetic/electric matrix elements,448 per case. Maximum errors are1.44e-13Hartree,2.78e-16au and1.10e-15e*Bohr. Independently diagonalized polarization-projector spectra reproduce native isotropic MCD within8.93e-11bohr^2. The first same-name STO-3G oracle failed its unchanged overlap gate because of rounded basis coefficients; identical original coefficients resolve it. A sulfur oracle then exposed a loader that sorted interleaved SP shells; preserving actual input shell order resolves overlap and all operator/spectrum comparisons. Both original harness failures remain. No native basis or acceptance threshold was changed.

### Generated molecular values and primary experimental comparison

PBE generates the following common orbitals; the physical model is nuclear-SOC CAS(7,4), not PBE-TDDFT. Ground bond lengths are OH0.96966Angstrom and SH1.3409Angstrom. OH uses1.7K/4T; SH1.69K/1.01T. All spectra are unshifted with an explicitly diagnostic100cm^-1 Lorentz HWHM, which is neither fitted nor equated to measured second-moment bandwidth. The environment is an isolated fixed molecule, with isotropic orientations.

| Case | Pi-to-first-Sigma vertical gap, cm^-1 | Pi SOC splitting, cm^-1 | Source-local wall,s |
|---|---:|---:|---:|
| OH-PBE-STO3G-order6 | 48785.735426 | 166.183811 | 0.681 |
| SH-PBE-STO3G | 47789.424351 | 483.342183 | 1.035 |
| SH-PBE-631Gss | 36976.577232 | 575.761879 | 19.180 |

Assignments are checked from projected Lz^2. Zero-field Kramers-pair defects are at most2.14e-14Hartree. OH zero-field MCD is5.56e-17bohr^2 and field-reversal defect1.09e-14. Zero-SOC Pi splitting is3.12e-9cm^-1; orbital-degenerate Pi states can retain orbital C without SOC, so nonzero MCD is not automatically a failed limit.

Orientation orders6/10 differ by0.00253848bohr^2 and are not converged. The original successive18/24 criterion also fails its predeclared1e-8 gate(difference2.71e-8); this record is retained. A stronger independent absolute sphere integral at orders48/64 agrees within4.78e-15 and establishes native order24 true error8.76e-11bohr^2, satisfying the same1e-8 absolute target. No point was discarded or target relaxed. A complex unitary transformation in that independent generated-matrix spectral oracle changes spectra by8.70e-13bohr^2; this is not a new public supplied-transition-spectrum API.

The [primary SH matrix experiment](https://ir.canterbury.ac.nz/bitstreams/e1d31f06-307c-4e72-bc5b-60ba6d965a86/download), Harrison and Williamson,J.IndianInst.Sci.85,391-402(2005),TableI, gives SH/Ar barycentre30400+/-20cm^-1, spectra at1.01T and field dependence at1.69K. The STO-3G vertical assignment is17389cm^-1 higher;6-31G** reduces this unmatched difference to6577cm^-1. These are **not matched vertical/vibronic or isolated/Ar conditions** and do not establish experimental agreement. Predicted nuclear-only SH SOC also exceeds the376.9cm^-1 gas value discussed by that paper. Active-space/basis/orbital convergence and missing two-electron SOC, host and vibronic physics must be separated before diagnosing a code defect; independent complete-operator calculations agree with the declared implementation.

The [primary OH/Ar paper](https://doi.org/10.1021/jp970076q) measures1.7-7K,0-4T with multiple host sites and orbital reduction; those host parameters are not fabricated here. The [OH](https://webbook.nist.gov/cgi/cbook.cgi?ID=C3352576&Mask=1000) and [SH](https://webbook.nist.gov/cgi/cbook.cgi?ID=C13940211&Mask=1000) compilations distinguish electronic minima,0-0 transitions and equilibrium geometries. They are not substituted for fixed-ground-geometry vertical references. Calibrated numeric experimental traces/site strengths were not retrieved. No per-spectrum scale or energy shift is used to manufacture agreement.

The curated shared Rust/Python/ASE/both-CLI artifact **passes scoped numerical acceptance** in `mcd-automatic-soc-public-integration-20261031-02/final-acceptance.json`:13 build/runtime checks,10 Rust cases with63 complete derivative fits,35 Python/ASE cases per CLI and portable audit. Fresh actual public-API replay on2026-11-01 passes19 cases with zero difference from saved physical outputs. Independent calculations against those fresh outputs verify1344 one-hole and201600 general CAS(7,5) complex operator entries; maximum Hamiltonian error2.78e-13Hartree and MCD spectrum error8.93e-11bohr^2. Native extension SHA-256 is `251c3f92ae09adb1db1e55f5964152be074c8ff12861c3004fd1f6965c488b3f`. One CPU thread on CPU10 and a512MiB managed calculation budget were used for replay; that budget is not an RSS cap. The first artifact failed strict nested unit labels; dotted per-field labels and an ASE conversion regression correct the contract without changing model values. The original failure is retained. This curated base excludes later API-M3/few-centre/root53 work and is not whole-live-tree acceptance. Full inputs, operator/spectrum oracles, failed attempts and orientation evidence are in `mcd-complex-casci-20261031-01`. General TDDFT/SOC thermal manifolds, magnetic gauge completion, two-electron SOC, active-space convergence and matched experimental spectral agreement remain separate obligations. No new functional, runtime dependency or excluded private basis input is introduced.

## Periodic recovery and pressure diagnosis, 2026-11-01

All sixteen original-support LiF strain SCFs completed with their original gates. The four central-difference steps are 0.008, 0.004, 0.002 and 0.001, with fractional atomic coordinates and fractional k points held fixed. Their errors against the production analytic derivative **fail** to converge: the isotropic fit gives p=-0.7825099842, R2=0.9169392298; the xx fit gives p=-0.3573185974, R2=0.6650917516. No step was removed and no error floor was used. The finest equivalent pressures are -61.6957465202 GPa and -60.0714065240 GPa, respectively, compared with the original analytic -65.5613159506 GPa. This is the deficient 12-Bohr support, PBE/STO-3G, Gamma, grid2/shell2 diagnostic, not corrected-support or stationary material acceptance. All sixteen saved densities satisfy metric electron number within 1.60e-14, idempotency within 2.00e-15, and the generalized Fock commutator within 4.93e-9. These checks do not supply the missing centre-state matrices or prove cross-geometry branch continuity.

The corrected-support LiF k2 SCF remains saved and converged, but its subsequent analytic stress did **not** complete. The unchanged 1.5-GiB process RSS guard terminated it at 1,583,080 KiB after kinetic, overlap, nuclear-attraction (806.927 s) and Hartree (2528.911 s) stress stages. No k2 pressure is available. The earlier Gamma grid6 pressure +49.5989181772 GPa remains abnormal and physically unaccepted. A new run must use the separately accepted exact force-Hessian streaming artifact, reconverge its electronic reference, and retain the guard; a saved matrix is not a validated restorable derivative identity.

The original ferrocene crystal worker likewise ended at the RSS guard, 1,949,492 KiB, without a converged SCF. Its overlap/core-Hamiltonian construction alone took 33,332.759 s. This is an actual crystal failure, not replaced by any molecular ferrocene result. Persistent SCF AO storage and core-integral cost require separate treatment; force-Hessian streaming alone cannot establish recovery of this first-SCF memory failure. The spinel worker remains alive; kaolinite is queued. Nine of twelve initial crystal SCFs have converged, but no fully stationary basis/grid/k-converged material response reference is accepted.

After an explicitly allocated short CPU12 preflight, the resource-scoped small worker compiled and all four overlap-support checks passed. The private published pob-DZVP-rev2 inputs give 20 AOs for LiF, 33 for NaCl and 32 for SiC; their Gamma overlap matrices have successive 24-to-30-to-36-to-42-Bohr changes below 1e-12. The diamond check preserves its previous 28-AO k2/grid3 input. These are overlap-support checks only, not density/Hartree/XC/stress, finite-k, basis, equilibrium or property acceptance. No additional SCF was launched by the preflight. Its short CPU allocation is released; longer numerical work still needs a confirmed nonoverlapping allocation. Full evidence, original guard events, all raw difference steps, matrix diagnostics, hashes and prepared inputs remain under the existing private crystal-benchmark validation directory; no new MD/CSV file was added.

Subsequently, explicit continuing allocation authorized the fresh proper-basis LiF SCF/stress and, separately, ten sequential canonical-CLI functional controls. LiF PBE/pob-DZVP-rev2 at the same B1 geometry, Gamma, grid4/shell3 and 24-Bohr support converged in thirteen iterations: E=-107.07412916476535 Hartree, metric electron number 12.000000000000014, quadrature number 11.998399177063995, sampled KS gap 8.059517844411769 eV. Its fresh residual is 1.266e-9, density change 1.096e-10 and energy change 1.705e-13 Hartree; SCF time is 981.347 s on one thread. Analytic stress completed with pressure **+87.1033130691 GPa**, so the abnormality is not resolved by this basis change. Total time was 1921.367 s, peak RSS 133,828 KiB; nuclear, Hartree and XC stress stages took 392.978, 155.651 and 387.993 s. Pressure reconstructed from the stress and its component sum agrees to roundoff, and strain-term closure is 1.11e-15 Hartree; these are arithmetic checks, not independent energy-derivative acceptance. Same-basis finite-k and independent-reference comparisons and supported-condition four-step strain fits remain necessary. This different-basis energy is not directly compared as a physical observable with the earlier STO-3G total energy. The new memory-controlled process is an actual new SCF, not a reconstructed derivative plan from old matrices.

The ten exact-name controls below completed SCF, analytic forces and stress through the accepted canonical common driver. The synthetic triclinic H2 primitive cell has inversion symmetry because its two sites are identical. Its original private input label incorrectly said C1; the input and results are retained, with this correction, and are **not** a symmetry-free all-component derivative gate or an experimental material. These are neutral two-electron RKS/STO-3G/Gamma controls with 24-Bohr support, grid3/shell3 and zero grid-weight cutoff. Maximum force-sum residual is 5.06e-15 Hartree/Bohr, metric number error 4.45e-16, quadrature number error 5.61e-9, and sampled peak RSS 28,464 KiB. Raw energies identify each model execution; they are not an across-functional accuracy ranking.

| Requested functional | Energy, Hartree per cell | Pressure, GPa | Wall time, s |
|---|---:|---:|---:|
| SVWN5 | -1.121295644160 | -0.02258291620 | 172.121 |
| PBE | -1.152166295682 | -0.03780829882 | 207.906 |
| TPSS | -1.166876331084 | -0.08070617270 | 214.171 |
| PBE0 | -1.154432373740 | -0.08934109614 | 178.818 |
| HSE06 | -1.154718090022 | -0.08979765284 | 196.241 |
| CAM-B3LYP | -1.156260898462 | -0.09228353844 | 199.806 |
| dlDF | -1.214016895615 | -0.10232223189 | 172.500 |
| B98 | -1.162218058977 | -0.08385948764 | 189.027 |
| tau-HCTH | -1.178413549248 | -0.05926143700 | 218.319 |
| Hybrid tau-HCTH | -1.169548770451 | -0.07690994984 | 193.698 |

No result is inferred for an unexecuted name, spin/reference combination, response route, defining nonlocal composition or PT2 contribution. Asymmetric multi-orbital controls, material comparisons and independent derivative/kernel validation remain separate. Existing double-hybrid correlated-periodic routes must retain their full model and remain lower priority. The current common-driver default AO image cutoff is 20 Bohr; the failed original 12-Bohr benchmark was explicit input. Neither radius is certified universally by this work. Basis/cell-aware policy changes must separately verify overlap, kinetic, density, nuclear, Hartree, XC and exchange support, grid and k sampling, preserve explicit user values and report effective settings.

Nine independent same-orbital-basis Gamma LiF SCFs also saved converged results. Central differences at strain +/-0.004 give 87.0566469607 GPa with auxiliary set A/grid3, 86.8927905541 with set A/grid4, and 86.9001115762 with set B/grid4. Different auxiliary fitting and atomic quadratures remain explicit numerical approximations; these are single-interval pressure checks, not four-step acceptance. The large positive pressure is reproduced independently, so a native-only sign/volume error is an insufficient explanation. A monitor asserted on an empty child cmdline at exit after the ninth result was saved; its original log is retained and that last OS exit is unknown. All nine saved results were verified without reexecution. Only the never-started same-basis native k2 SCF/stress was launched in recovery. Basis/grid/k and fixed-geometry model errors and native corrected-support strain fits remain open.

The proper-basis native LiF unshifted2x2x2 calculation completed both SCF and analytic stress with OS exit0. At the same fixed3.964Angstrom B1 cell, PBE/pob-DZVP-rev2,24Bohr support and grid4/shell3, E=-107.31468131934636Hartree, metric electron number12.000000000000002, quadrature number11.99851735743141, and pressure **+11.5967599613GPa**. The Gamma-to-k2 pressure change is **-75.5065531078GPa**, demonstrating strong sampling sensitivity, not k convergence or equilibrium. Twelve SCF iterations took1469.584s; nuclear-attraction, Hartree and XC stress stages took363.131,151.727 and1564.432s, respectively. Total native time was3557.063s on one thread, sampled peak RSS614852KiB, rotation defect1.385e-15. The sampled KS gap9.568167eV is not an optical/fundamental experimental gap. Independent same-basis finite-k pressure checks, corrected-support native four-step energy/stress fits, denser k/grid tests and an appropriate equation of state remain required. This fixed reference geometry was not optimized with the present PBE/basis/sampling, so zero pressure there is not imposed by definition.

The independent same-orbital-basis unshifted 2x2x2 LiF calculation completed three fresh PBE SCFs at isotropic strains 0 and +/-0.004, with fixed fractional coordinates and fractional k points. Its distinct Gaussian auxiliary fitting and atomic quadrature give pressure +11.3305663855 GPa, versus the native analytic +11.5967599613 GPa. The difference -0.2661935758 GPa remains material and unaccepted. This single interval does not replace matched native four-step energy derivatives, quadrature/image-support convergence or an equilibrium equation of state. All three independent SCFs converged with OS exit 0; peak RSS stayed below 501000 KiB.

An additional independent integral check compared every complex overlap matrix entry at all eight k points: 20x20x8 complex entries, or 6400 real entries. With the explicit spherical AO ordering mapped consistently, maximum entry errors at integral precisions 1e-8, 1e-10 and 1e-12 were respectively 2.78666e-14, 2.77556e-14 and 2.81997e-14, passing the unchanged 1e-12 gate. The weighted density trace was 12.000000000000009 electrons. This validates the present overlap image support for these sampled points; it does not certify electron-nuclear, Hartree or XC support, generic complex k/q response, grid integration or stress. The failed container-indexing attempt is retained separately; the successful check changed only array container normalization.

At the saved native LiF k2 density, independent one-electron core matrices converge toward the native matrices: maximum entry differences are 7.81e-9, 1.08e-10 and 2.00e-12 Hartree for integral precisions 1e-8, 1e-10 and 1e-12; the final density contraction differs by 1.26e-12 Hartree. This does not test their strain derivatives. Raw independent fixed-density Hartree/XC evaluations then show a larger energy discrepancy: the native Hartree/XC energies are 27.5818118903/-12.3516361301 Hartree, whereas independent grid level4 gives 27.5817966584/-12.3522894955 Hartree. The differences are -1.52319e-5/-6.53365e-4 Hartree. Independent grid levels3,4,5 integrate 12.0009632109, 12.0001133142 and 12.0000239413 electrons, versus native 11.9985173574. Level5 still differs from the native XC energy by -6.23962e-4 Hartree. These distinct quadratures and auxiliary Coulomb fitting indicate XC dominates the observed energy difference, but do not establish a source defect or explain the pressure derivative quantitatively. Native fixed-density grid/support tests and matched strain energies remain open. The155.81-second diagnostic, peak RSS505672 KiB, remains a strict failure because the reference library implicitly closed temporary files during shutdown and emitted ResourceWarnings; the earlier complex-cast warning is also retained. Energy imaginary parts were explicitly measured at9.91e-17 Hartree, but neither warning-bearing attempt is accepted. Explicit temporary-file cleanup is prepared for a later serialized rerun.

The quantum-nuclear scope is specifically NEO-DFT: electrons and selected quantum nuclei with explicit masses, charges, normalization and statistics. Existing molecular multicomponent SCF, EPC17 correlation and coupled analytic response are reuse candidates; the R076 coverage ledger still lists periodic NEO as pending. A periodic implementation must replace the selected classical nuclei consistently, define localization/cell and zero-mode conventions, and validate coupled electronic/nuclear response. Molecular laboratory-frame results, classical BO dynamics and diagonal BO corrections do not establish periodic NEO acceptance.

Full periodic TDDFT/RPA excited-state gradients and analytic nuclear Hessians are an additional open scope. RPA here retains the excitation/de-excitation coupling and the indefinite X/Y norm, not an RPA correlation-energy approximation; the distinction from TDA Hessians follows the [full-TDDFT analytic-Hessian formalism](https://doi.org/10.1063/1.3659312). The source audit found paired analytic excitation/Hessian recursion already present for molecular references, while existing finite-q periodic frequency response does not itself supply excited-state eigenvectors or nuclear Hessians. A narrow transported-frame operator seam is prepared for review only; it has not been compiled or adopted. Initial real-Gamma/supercell reuse is a stage, not acceptance of arbitrary finite-k/q complex derivatives.

The same scope includes physically admissible multiplicities, RKS/UKS/ROKS/GKS distinctions, spin-conserving and spin-changing blocks, state/subspace tracking and spin diagnostics. Current molecular spinor and common-orbital flag excited drivers explicitly refuse full RPA; ordinary periodic collinear/longitudinal response does not establish their missing spin-flip or generalized-reference combinations. Periodic orbital-field/nuclear-SOC infrastructure is not general Dirac/X2C/DKH excited-state acceptance. Molecular simultaneous electronic/quantum-particle SCF is a non-BO model, but its molecular driver rejects periodic inputs; it does not establish periodic quantum-nuclear couplings or Hessians. These periodic relativistic/non-BO and GKS/ROKS combinations require their actual boundary, species, magnetic-order and derivative models, independent tests and common surfaces. The separately paused spinor task has not been restarted, and no new functional was added.


## London first-order molecular MCD and full-RPA follow-up, 2026-11-01

The source now offers explicit `mcd_magnetic_basis="london-linear-frozen-soc"` alongside the unchanged `common-origin` default. London orbital phases generate first-order overlap, one-electron and Coulomb coefficients. The connection `-S_prime/2` acts on every Hamiltonian, four-index repulsion and optical-operator index in the same zero-field orthonormal active frame. Spin Zeeman is separate; no orbital-Zeeman term is counted twice. Optical operators retain their own field derivative. Zero-field nuclear SOC is frozen in that frame: moving SOC, two-electron SOC and quadratic magnetic terms are **not implemented by this increment**. This is an explicitly linear-field model, not general relativistic finite-field self-consistency. Selecting an origin by convention is not substituted for the London completion.

Three OH/HF/STO-3G gauge origins give identical magnetic/optical derivative matrices and MCD differences no larger than9.993e-16bohr^2. On an asymmetric three-hydrogen doublet, independently constructed full-Fock creation matrices and finite-field metric diagonalization verify all1044 independent nonzero imaginary magnetic/optical derivative components along three linearly independent generic fields. The initial wider-step interval did not satisfy the original strict gates. The retained refined interval[0.02,0.01,0.005,0.0025]au passes all1044 components: p=1.98119027633..2.05621008777, minimumR2=0.999809291664. Exactly linear spin terms and structural zeros are checked directly, with their raw errors retained; no logarithmic error floor is used. Initial harness attempts incorrectly treated the real spin-Zeeman derivative, then its exactly linear imaginary part, as a nonlinear orbital coefficient. Both failures remain archived. Correcting that classification did not alter production formulas or remove a failed step. The original common-origin finite-field defect remains retained.

The London derivative proof is linked to the rebuilt Rust/Python/ASE/both-CLI artifact. Source hash, input/output matrices, all fits and failures are in `mcd-london-linear-20261101-01`. Warning-strict source Clippy/build passes. All1044 source-proven derivative matrices are exactly reproduced by the actual public library; full details and all failed attempts remain in `molecular-response-followup-20261101-01/final-acceptance.json`.

The molecular full-RPA Hessian route already differentiates the paired A/B response through analytic nuclear Taylor transport. It returns the excited-state total Hessian and a distinct excitation-only Hessian. A fresh actual public-API HF/STO-3G distorted-water calculation gives excitation0.45179603232343934Hartree, exact stored Hessian symmetry and translation defect1.132e-11Hartree/Bohr^2. The first90 complete gradient/Hessian component fits fail3 components on[0.04,0.02,0.01,0.005]Bohr; the original failed interval is retained. The initial exploratory harness used broader fit gates and is not final acceptance. Re-evaluation uses the original strict1.9<p<2.1 andR2>0.999 gates. Staged complete four-step gradient intervals accept89 components; an independent mixed scalar-energy stencil resolves component74 on[0.02,0.01,0.005,0.0025]Bohr. All90 then pass: p=1.95264267920..2.09874508328, minimumR2=0.999464586668. Every coarse, fine and failed interval remains retained in molecular-rpa-strict-intervals-20261101/final-strict-validation.json. The original fixed-root isolated-state test does not establish crossings, degenerate-subspace transport, spin flip or every functional/reference.

A separate actual native counterexample proves that the old RPA solver returned a zero-energy state for `A-B=1,A+B=-1`. Source now rejects nonpositive/nonfinite squared frequencies instead of clipping them; a permanent regression checks unstable, zero and positive-frequency metric cases. RPA gradients now select the analytic path by default while explicit unsupported Z-vector requests remain errors. The Hessian API source adds the ground-state contribution and explicit total-versus-excitation quantity metadata. The rebuilt follow-up artifact passes the actual stability and public-API regressions; earlier artifacts are not represented as containing the correction.

Arbitrary multiplicity requires separate spin-conserving and spin-changing spaces. Existing collinear spin-resolved response is not universal spin-flip support. ROKS/GKS paired RPA, fully moving SOC and matched experimental MCD agreement remain open. Parallel isolated multicomponent mass-response and no-pair operator-property work is recorded separately; neither a nuclear-SOC matrix nor BO nuclear dynamics is labeled complete relativistic or non-BO DFT.


### Exact current response coverage of this follow-up

An independent dense full-RPA calculation reproduces the water excitation within1.19e-13Hartree. Its analytic gradient differs by1.874e-8Hartree/Bohr, exceeding the unchanged1e-8 comparison gate. Tightening response convergence and replacing its iterative eigenvectors with independently normalized dense eigenvectors do not resolve that difference. Independent energy-only sixth-order Richardson gradients instead agree with the dft-rs analytic gradient within9.98e-11Hartree/Bohr; the other analytic gradient differs from those scalar energies by1.870e-8. This isolates an unresolved analytic-reference discrepancy rather than proving a dft-rs defect. The failed external gradient/Hessian gates remain failed and are not promoted by the energy check. Three external-gradient Hessian fits also remain unresolved. Detailed raw records are in `molecular-rpa-hessian-independent-20261101-hf-04`; earlier failed response/iterative attempts remain preserved.

| Route | Executed evidence | Remaining scope |
|---|---|---|
| Restricted HF full A/B RPA | Water total excited-state energy: all9 gradients and81 Hessian four-step fits; excitation-only energy: all9 diagonal Hessian entries independently pass | The81 excitation-only Hessian entries have not all received a separate direct check. Crossings, degenerate subspaces and external analytic-gradient discrepancy remain; new stability/API replay passes |
| Restricted PBE full A/B RPA | H2 total excited-state energy: all6 gradients and36 Hessian entries pass scalar-energy fits; excitation-only energy: all36 Hessian entries independently pass staged four-step scalar-pole fits | Accepted artifact251c3f92; the separate original analytic-gradient replay is terminal and passes all42 strict component fits |
| ROKS/GKS full RPA and arbitrary spin sectors | Actual current drivers audited; both explicitly refuse paired RPA | Paired flag/spinor response, spin-changing blocks and their analytic nuclear derivatives require implementation and independent acceptance |
| SOC mixed-spin diagnostics | New source returns per-state spin vector and S-squared; H2 singlet/triplet and H3 doublet/quartet traces exactly6/27 | These are spin-complete CASCI diagnostics, not universal pure-spin TDDFT or GKS/ROKS RPA |
| Relativistic operator property | Bare covariant no-pair operator projection128 component fits pass; original projector preserved | New common-driver public replay passes; no interacting relativistic DFT, X2C decoupling or conserved-current claim |
| Non-BO mass response | Simultaneously variational electron/quantum-particle model: every2 mass gradients and4 relaxed Hessian entries plus4 Gaussian mass fits pass | New driver/public replay passes; distinguishable lab-frame Hartree particles, no intrinsic COM removal or identical-nuclear statistics |

The shared follow-up snapshot is prepared from accepted MCD02 plus only owned narrow increments. Its first preparation failed one documentation newline hunk; the exact rejection is retained and a normalized-text resume succeeds. No scientific test or source threshold was changed. `molecular-response-followup-20261101-01/resume-preparation-02.json` records21 source/test hashes and required linked tests. The subsequently completed serialized full build and public replay pass as recorded below; the earlier compiler wait is retained in the preparation record. No new documentation file or functional was added, and paused spinor work was not restarted.


### Linked follow-up acceptance and measured gauge invariance

The scoped rebuilt library passes all13 new Rust cases (201 complete component fits),46 Python/ASE tests per CLI name, retained closed-shell MCD and five multicomponent plus one projector and one stable-spectrum regressions. Portable artifact audit passes. Native library SHA-256 is `6d963b12bf20422c3b7ab8475cde7ba87847d8ccd801c894dae6a51b26d807cb`; the exact snapshot and every modified source/package hash are recorded in `molecular-response-followup-20261101-01/final-acceptance.json`. The actual public water RPA Hessian/gradient and H3 London operator matrices reproduce their earlier complete-component proof values exactly. First test collection failed a helper path; the subsequent actual static getter failure exposed a resource wrapper that had lost static binding. Both failures remain archived. Only the Python binding was corrected, with an explicit CPU/memory forwarding regression; no numerical gate was loosened.

Matched actual public OH/HF/STO-3G CAS(7,4),rOH0.96966Angstrom,T4.2K,B4T, fixed light/field alongz, Lorentz HWHM100cm^-1, photon grid0.20..0.25Hartree(101points): three gauge origins[0,0,0],[1.3,-0.8,0.4],[-2.1,0.7,1.8]Bohr give common-origin MCD variation1.82453519848e-7bohr^2 versus London-linear/frozen-SOC1.33226762955e-15bohr^2. London magnetic matrix differences are exactly zero. Field-reversal defect is6.218e-11bohr^2; peak absolute MCD is1.41236365981938bohr^2. These are gauge/implementation checks, not evidence of experimental agreement or a complete finite-field SOC Hamiltonian.

PBE/STO-3G H2 at the explicitly saved non-axis-aligned geometry has full-RPA excitation0.9341533192390493Hartree. The original analytic Hessian took1286.323seconds, including an initially contended CPU allocation. Its42 scalar-energy component fits pass on[0.08,0.04,0.02,0.01]Bohr with original strict1.9<p<2.1 andR2>0.999 gates: p=1.97901288811..2.02888059361, minimumR2=0.999947680110.284 uncached scalar points complete in31.115seconds; these finite differences validate the analytic result and are not a production Hessian implementation. The separate originally launched analytic-gradient replay is now terminal:49/49 calls completed and42/42 strict fits pass on[0.02,0.01,0.005,0.0025]Bohr, p=1.99950867109..2.00008757402 and minimumR2=0.999999983734. Both independent checks and the contention history are retained separately; minimal-basis implementation acceptance is not experimental excitation-energy agreement.

The excitation-only Hessian is separately checked against differences of the scalar RPA pole, excluding the ground-state energy. For H2/PBE,34of36 components pass on[0.08,0.04,0.02,0.01]Bohr; the symmetric pair(1,4)/(4,1) initially fails that coarse interval. Four additional scalar calculations in2.442seconds complete the refined interval[0.04,0.02,0.01,0.005]Bohr, givingp=1.97260275287 andR2=0.999947282411 for both entries. The resulting staged36-component acceptance hasp=1.97260052876..2.00300358911 with the original strict gates. Original failed fits, all four new reports, startup warnings and the clean canonical-interpreter audit remain retained in `molecular-rpa-excitation-only-20261101`. This is not one common interval and is not substituted for a direct water81-component excitation-only check. Water's9 diagonal excitation-only entries independently pass on[0.02,0.01,0.005,0.0025]Bohr withp=1.99490588583..2.02264157740 and minimumR2=0.999965254361.

Measured latency diagnosis found that the initially assigned CPU10 overlapped an existing crystal[2,10] mask. Moving only this owned calculation toCPU13 preserved its state and restored8.0CPU-seconds in8.009wall-seconds; the foreign process was untouched. This is scheduling contention correction, not a claimed code speedup. An eight-second userspace sampling window found broad Taylor arithmetic/path symbols; nearest-symbol attribution is not a complete exclusive call profile. An isolated logarithm fusion removes temporary allocations and matches5040 XC coefficients bitwise, but six alternating microbenchmarks do not show consistent speedup. It remains isolated and is not promoted into production.

A later root56 public test occupied the sameCPU13 and reduced the original replay to approximately50% of one core. Once the separately owned NEO diagnostic releasedCPU14, an explicit recorded handoff moved only the original replay and its existing/future workers toCPU14, preserving process identity and scientific controls. Eight wall seconds then delivered eight CPU seconds; the other owner's test was untouched. Both overlap intervals and affinity transitions remain in the original replay's durable monitoring records and are excluded from code-speedup claims.

The non-BO scope in this program is specifically NEO-DFT: coupled electrons and selected quantum nuclei, not classical Born-Oppenheimer dynamics. Existing molecular Hartree quantum-nuclear orbitals and EPC17 electron-nuclear correlation are reused. The accepted mass response holds the Gaussian basis and constraints fixed; it does not establish periodic NEO, identical-nuclear exchange statistics, or every coupled excited-state derivative.

An additional numerical NEO check uses deliberately distorted C1 NH3, ten electrons and one quantum proton of mass1836.15267343 electron masses. The electronic model is HF/STO-3G with existing EPC17-1 electron-proton correlation, [DOI10.1063/1.4996038](https://doi.org/10.1063/1.4996038). The proton uses fixed-center Gaussian exponents[4,8]Bohr^-2 and angular momenta throughp, with an unconstrained normalized Hartree orbital. This is a finite-basis laboratory-frame implementation test, not a matched experimental or literature polarizability benchmark. At electric bias[0.0013,-0.0009,0.0017]au, its coupled dipole polarizability is approximately[[5.019112845564,1.203945801177,0.667548782874],[1.203945801177,6.592684357711,-0.341598084786],[0.667548782874,-0.341598084786,1.922919263972]]Bohr^3. The quantum-particle response is included.

The tighter wider-field run completes26 fresh joint SCFs in146.116seconds. Electron/proton number errors stay below8.89e-15/4.45e-16; response-number errors stay below2.45e-15/6.96e-16. Translating the neutral system changes energy by6.40e-14Hartree, dipole by4.11e-15au and polarizability by8.00e-14Bohr^3. Nevertheless, the unchanged four-step convergence gates accept only6of9 tensor components on[0.032,0.016,0.008,0.004]au. The xx,xy,yx fitted orders are2.228964,1.576729,2.108196 and remain failed. The initial constrained-position rejection and all failed unconstrained intervals are retained; no all-component acceptance is claimed. Raw records and exact controls are in `neo-alpha-20261101-unconstrained-04/independent-summary.json`.

Tighter shared-point SCFs change dipoles by at most3.42e-14au and do not resolve the failed fits. Phase-aligned quantum-orbital sign crossings are measured at finite field: the negative x step0.0005au crosses grid points with maximum proton density6.28e-7, whereas at0.000125au the remaining crossing has density8.47e-51. This establishes moving discrete-grid node sectors, not their causal contribution to every error. The smallest interval[0.000125,0.0000625,0.00003125,0.000015625]au completes12 new SCFs plus14 hash-identified prior fresh states in35.53seconds; none of its9 complete fits satisfies both strict gates. Small absolute errors are not promoted to convergence acceptance. Source inspection finds occupied-orbital-square base densities, physical2*psi*delta_psi tangents and an exact-zero EPC17 branch, without a finite density floor in that inspected path. Quadrature resolution and independent variational/kernel checks remain open; no production defect has yet been isolated. Exact signed errors, node counts, source hashes and source-audit limits remain in stages05/11/12 and `neo-alpha-20261101/density-path-source-audit.json`.

The primary full-TDDFT Hessian formalism is described by [Liu and Liang(2011)](https://doi.org/10.1063/1.3659312). Its accessible abstract is not a matched numeric molecule/basis/geometry reference and is not substituted for the actual component tests above.

A further primary MCD reference, [Hattori et al.(2024)](https://doi.org/10.1021/acs.jpclett.4c02793), supplies selected numerical Faraday B terms for the open-shell radicals PyBTM and F2PyBTM in official supplementary TablesS3/S4. Its calculation uses TDA-M06-2X/6-31G(d,p), PCM dichloromethane and50 doublet excited states; it is neither a thermal C-term reference nor the present CASCI model. The retrieved seven-page supplement contains plotted experimental MCD but no numerical wavelength trace or Cartesian geometries. Exact temperature, gauge and absolute line-shape conventions are also insufficient for a matched absolute comparison. No experimental agreement or radical TDDFT B-term acceptance is claimed from these tables.

### ROKS/GKS magnetic reference dispatch correction

An actual public OH/HF/STO-3G doublet reproducer on library6d963b12 finds a physical reference-selection bug: `energy(reference="roks")` gives-74.3616254361038Hartree, whereas `energy(reference="auto")` gives-74.36273359054296Hartree. Calling `magnetizability(reference="roks")` returns exactly the latter UKS reference energy. The1.10815443916mHartree discrepancy is an ignored reference request, not basis or experimental model error. All three references converge. Exact controls, reports and source hashes remain in `molecular-property-benchmark-20261031/mcd-tranche/magnetic-reference-fallback-reproduction.json`.

The source dispatcher now explicitly refuses unconnected ROKS/ROHF/GKS/GHF magnetizability, hypermagnetizability, Cotton-Mouton, magnetoelectric, magneto-optical and finite-field paths instead of returning automatic-collinear results. Thirty-six native/ASE/CLI contract cases are added; their linked-artifact execution is **pending**, so the old library is not represented as fixed. This guard corrects the silent replacement; it does not implement the missing reference-specific response. Existing restricted H2 B2/B4 fits, ROKS complex linear one-body fits and general spinor finite-matrix fits are reusable components, not full self-consistent moving-London B1-through-B4 acceptance. The requested ROKS/GKS paths still require their constrained/common-orbital or full-spin response, complex magnetic Hamiltonian and metric/interaction derivatives, explicit spin branch and gauge conventions, actual open-shell molecular finite-field tests and public parity. The separately paused worker remains paused.

A source-local unrestricted spinor HF/STO-3G test on nonplanar H4+ now verifies one finite-bias magnetic energy contraction through fourth order. With B0=[0.013,-0.007,0.011]au and B(t)=B0+t*[0.6,-0.2,0.1]au, the fourth derivative is-26964.07747626844Hartree per atomic-field unit to the fourth power; the direction is not normalized. The independent spin-scalar-conditioned finite-field replay passes all four derivative orders on[0.032,0.016,0.008,0.004], with exponents2.000011..2.002406, minimumR2=0.999999799564 and maximum Richardson absolute error9.633e-9. A wrong spin branch at one initial point was diagnosed by its negative stability eigenvalue and rerun from a corrected initial spin orientation without changing the Hamiltonian or thresholds. Original failures remain archived. This is a common-origin, finite-bias, selected-direction energy check with density response through order3; it does not establish a full B4 tensor, fourth-order density response, London gauge independence, semilocal DFT or public GKS magnetic support. Evidence: `mcd-tranche/common-origin-source-local-attempt11/direct-spin-scalar-conditioning-reproduced.json`.

### Electron-proton correlation alias correction

An actual two-quantum-nucleus test identifies a separate channel-grouping bug: replacing one `epc17` selector by its equivalent `epc17-1` alias changes the total energy by+0.0012937735354237034Hartree while both electronic and nuclear densities remain identical. The entire change is in the EPC energy: textual aliases incorrectly created separate correlation channels. The source now canonicalizes equivalent selectors before channel grouping and preserves the distinct `epc17-2` parameterization. Three public-surface regression cases are prepared; execution on a newly linked artifact remains pending. The existing EPC17 alpha derivative failures above are separate and are not resolved by this correction. Raw evidence is retained in `neo-correlation-audit-20261101/alias-comparison.json`.

The names `epc17` and `epc17-1` both denote the local energy density f=-x/(a-b*sqrt(x)+c*x), x=rho_e*rho_p, with(a,b,c)=(2.35,2.4,3.2). The separately implemented `epc17-2` uses(2.35,2.4,6.6). A retrieved independent NEO benchmark reports both parameter sets in Section2.1: [Khan and Tonner-Zech(2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11915487/), DOI10.1002/jcc.70082. This corroborates the existing coefficients. A subsequent primary-source check corrects the earlier report's attribution of the second parameterization solely to grid-based zero-point energies: [Brorsen, Schneider and Hammes-Schiffer(2018)](https://doi.org/10.1063/1.5037945), Section IV after Fig.3, explicitly identifies fitting the experimental HCN proton affinity. The earlier assertion that this was not a direct proton-affinity fit is withdrawn. The source comment describing proton-affinity fitting should not be changed on that basis. No formula or coefficient is changed by this provenance or alias correction; neither establishes EPC18/EPC19 implementation or closes the failed alpha derivative fits.

A separate regularity check follows directly from the EPC17 formula. For fixed positive electron density and a real nuclear orbital, the local energy contains a term proportional to `-|psi|^3`; it is C2 but generally not C3 at an exact orbital node. Central second-energy differences of a discrete quadrature can consequently have an O(|h|) nodal error. This is a pointwise/discrete-grid statement: spatial integration can smooth a moving simple node, so no universal C2 limit of the continuum NEO energy is asserted. This does not relax the existing strict fits or establish the cause of the failed polarizability entries. The saved field tests contain weighted node crossings as large as8.16735e-8 at step0.004, falling to4.08269e-10 over the finer recorded interval; they must be assessed through the prepared fixed-orbital/grid diagnostic rather than dismissed from one negligible tail crossing. Density-product underflow is a separate confirmed source-level defect whose native Rust regression and adoption remain pending.

### Periodic LiF strain continuation

The inherited single native LiF isotropic strain+0.004 calculation is terminal with OS exit0 and convergence=true. At volume106.3502917550081Bohr^3, PBE/pob-DZVP-rev2, unshifted2x2x2 mesh,24Bohr image cutoff, grid level4/shell3, zero grid-weight cutoff and0K, its energy is-107.31515282441718Hartree per cell after11 reported SCF iterations. The independent final physical rebuild logs Fock residual4.753e-9Hartree and density change4.863e-10. Wall time is1910.794s, peak RSS588408KiB, with one requested thread and1280MiB managed memory; managed allocation is not a process RSS cap. Overlap-metric electron count is12.000000000000012 and quadrature count11.998531505914693. This one converged energy is not a central pressure difference, a four-step derivative fit, or basis/grid/k-converged material acceptance. Seven other prepared strain inputs remain unlaunched while the serialized integration queue proceeds. The CLI artifact has SHA-256 prefix `20ceb3b52de02486`.

### Periodic full-RPA implementation controls

Candidate03 first failed warning-strict metadata on an unused `plane_wave_contraction` import. Candidate04 removes only that import; the original source and failure remain retained. All four metadata commands, the actual library build and two integration executables now terminate successfully. The exact candidate Rust library SHA256 is `f4c91ba50ce570fce59d961885ad144c122a39d01be0675cd1be32d309120e4f`; the shared-library SHA256 is `b19d8db1a0e55e67cdc3c98943f5ece44771211b17812280d57be2056af7ff09`. Evidence is under `pbc-stream-rpa-nuclear-integration-20261101-04/full01`.

The complex matrix test checks the independent closed form omega=sqrt(a*a-b*b-c*c) with a nonzero imaginary coupling, indefinite excitation/de-excitation metric, phase covariance and refusal of unstable, incomplete or degenerate derivative cases. Three test functions pass. All12 pole-gradient/Hessian components with respect to three synthetic matrix parameters also pass an independent replay of the original raw four-step errors on[0.08,0.04,0.02,0.01]: strict1.9<p<2.1 andR2>0.999, p=1.99959897381..2.00042271626, minimumR2=0.999999986994. The existing helper's wider exponent half-width0.2 is not substituted for this strict audit. These matrix parameters are not nuclear displacements of a crystal.

The second executable passes its complete collinear spectrum-to-coupled-density comparison for HF,PBE,PBE0,TPSS, meshes[1,1,1]/[2,1,1], q=0/[0.043,0.017,-0.021]Bohr^-1, and static/retarded frequency(omega,damping)=(0.13,0.04)Hartree. Its eight converged reference SCFs use an asymmetric three-atom H/H/He artificial cell and compact one-function-per-atom basis. Across32 combinations,1728 real/imaginary AO source columns yield5760 output-matrix comparisons with complex RMS difference below1e-9; operator residual, metric and eigenpair limits remain unchanged. This is consistency between two representations of the same kernel, not an independent physical kernel, basis/cutoff convergence or experimental spectrum oracle.

On one admitted logical CPU, library compilation took1529.514s with sampled compiler RSS peak3724936KiB; the density consistency test reports27.32s of test runtime. These are not controlled speedup measurements. New AO/cache unit parity and derivative fits, new full-RPA public-surface coverage and stationary material comparisons remain pending; general periodic analytic nuclear excited-state Hessians, arbitrary-spin response, relativistic and NEO combinations are not established by these controls.

A subsequent same-library public continuation passes five focused tests: HF/PBE particle-number diagnostics through native, physical-unit ASE and each of the two CLI executables, followed by SVWN5 stress/decomposition/unit parity through native/ASE/canonical CLI. Frozen scientific assertions and model parameters are unchanged; the private test overlay adds only one requested CPU thread and1280MiB managed memory. Both actual CLI links terminate successfully and have SHA256 `fd1ccdb587fbffdcf7f178c75a34d2922510363b2a0f04b63cd04ae8c4664024`. The matching native-library hash remainsb19d8db1. Portable artifact audit passes. Peak sampled process-family RSS is120528KiB for linking and88528/90480KiB for the number/stress checks. Evidence: `pbc-stream-rpa-nuclear-integration-20261101-04/public01/state.json` and individual logs. These controls do not close independent derivative, material-convergence or the new full-RPA Python/ASE/CLI API gaps.

### Matched LiF performance diagnostic

The same LiF+0.004 input was subsequently replayed with the candidate04 CLI after unit compilation yielded before spawn on the unchanged6GiB available-memory gate. The old CLI20ceb3b5 and new CLIfd1ccdb5 receive identical geometry bytes and all calculation options: RKS/PBE,pob-DZVP-rev2,2x2x2 k mesh,24Bohr cutoff,grid4/shell3,zero grid-weight cutoff,0K,core guess,energy tolerance1e-10Hartree,density tolerance1e-8,one compute thread on CPU12 and1280MiB managed memory. The basis hash4d0c1276 matches its retrieval record predating the original calculation. Both processes start with fresh scientific caches. OS caches and concurrent background load are not controlled.

| Quantity | Original accepted CLI | Candidate04 CLI |
|---|---:|---:|
| Energy, Hartree/cell | -107.31515282441718 | -107.31515282441718 |
| Reported SCF iterations | 11 | 11 |
| Process wall time, s | 1910.793876 | 862.905473 |
| Overlap/core Hamiltonian time, s | 999.156 | 235.407 |
| XC grid construction time, s | 194.883 | 151.130 |
| Sampled peak RSS, KiB | 588408 | 590132 |
| Reported managed peak, bytes | 571277088 | 571277088 |

All19 selected scientific scalars, including every reported energy component, particle-number diagnostics,volume and iteration count, are bitwise identical; maximum absolute difference is0. The new process uses858.796849 CPU seconds; baseline CPU time was not recorded. The historical wall-time ratio is2.21437, with no measured memory reduction. This one matched-input result supports further evaluation of the exact optimizations; it is not an ABBA/statistical or universal speedup guarantee and does not waive the unexecuted unit/derivative gates. It also supplies no new equilibrium-material or experimental agreement. Raw commands, geometry/basis hashes,phase timings and complete scalar differences remain in `crystal-benchmark-20261031/lif-source04-matched-replay-01/{state.json,comparison.json,profile-summary.json}`.

### Periodic partial-XC refinement and serial execution

The practical completion targets remain separate from local algebra acceptance.
The current independently executed scope relevant to the requested analytic
Hessians is summarized below; a blank combination must not inherit a nearby
result. The detailed reference-class ledger remains in `response.md`.

| Domain and response | Executed evidence | Remaining scope |
|---|---|---|
| Molecular restricted HF full RPA | Distorted-water 90 gradient/total-energy Hessian entries | Other reference classes, degeneracies and moving SOC remain separate |
| Molecular restricted PBE full RPA | H2: all 42 gradient/total-energy Hessian entries; exponent 1.99950867–2.00008757, minimum R-squared 0.9999999837 | This small-system result does not certify all GGA/hybrid or open-shell combinations |
| Molecular common-orbital open-shell HF | Previously accepted doublet/quartet reference-spin TDA derivatives | Full RPA and genuine spin-changing sectors are distinct gates |
| Molecular generalized full RPA | Transport foundation tests executed | Paired-spin molecular nuclear hooks remain source/metadata preparation; no executed molecular Hessian claim |
| Periodic full RPA | Complex metric and fixed-k density-response tests executed | Complete nuclear excited-state Hessians, transported frames and public molecular/material cross-checks remain open |
| Relativistic response | Previously accepted finite-nucleus and frozen-source component subsets | Interacting relativistic SCF, picture change and full excited Hessians are not established by those subsets |
| Molecular NEO HF/EPC17 | Distorted-ammonia coupled polarizability: six of nine strict component fits | Three components remain failed; coupled-solver and alias-fix public artifact tests are pending |
| Real crystals | Ten of twelve initial SCFs completed | Zero stationary basis/grid/k-converged material references; kaolinite continues and ferrocene memory failure remains unresolved |

Excitation-energy derivatives and excited-state total-energy derivatives are
different outputs. Neither fixed-density response nor finite-difference
validation is described here as an analytic nuclear implementation.

The source04 strict unit follow-up executed all 72 independent fixed-density
nuclear-coordinate and cell-strain entries for SVWN5, PBE, TPSS and THCTHhyb.
All entries satisfy the unchanged second-order criteria: selected exponents
1.900423–2.094141 and minimum R-squared 0.99914313. All 27 failed coarser
interval fits remain in the raw evidence. The previously unresolved SVWN5
cell zz entry resolves on steps 0.01, 0.005, 0.0025 and 0.00125, with exponent
1.969157 and R-squared 0.99993258. This closes that synthetic partial-XC
gate; it does not validate full self-consistent stress or material properties.
Production formulas and acceptance thresholds were unchanged.

The focused unit binary is identified by SHA-256
`ee216a74eb4e15800c1dda1ab0fb4aaac1a2196d9abc8ff5274b7b1ef40ea5f3`.
Compilation took 789.649 seconds and the focused runtime test 205.173 seconds
on one reserved CPU, with sampled family RSS maxima 4,317,032 and 14,972 KiB,
respectively. These are execution measurements, not a controlled speedup.
Component fits and all raw signed differences are retained under
`.work/validation/pbc-stream-rpa-nuclear-integration-20261101-04/unit03/`.

### Completed spinel initial reference and measured bottlenecks

The owned periodic zero-order nuclear caller has now been adopted into the
working source from the executed source04 postimage. The complete preimage
chain and exact shared scalar-helper body were checked before application;
no scalar-helper, threshold or in-flight artifact was changed. Adoption
evidence is retained in
`.work/validation/pbc-verified-nuclear-caller-adoption-20261101-01/manifest.json`.
This source edit does not put the optimization into the older running
spinel/kaolinite artifact or the separately frozen API/core60 wheel. A new
whole-tree integration and controlled timing/RSS comparisons remain required.
The persistent SCF AO-cache memory policy is a separate, already staged
source04 change requiring reconciliation with the current SCF edits and
the permanent strict test controls.

The inherited normal-spinel MgAl2O4 calculation exited normally after 68446.09s (about19.01hours). The 14-atom primitive cell contains two formula units, with geometry from [COD9005807](https://www.crystallography.net/cod/9005807.cif). Conditions were all-electron PBE/STO-3G, Gamma only, 24Bohr AO image support, grid3/shell3, zero grid-weight cutoff, fixed nuclei and zero electronic temperature. The structure's experimental temperature is unspecified. This is an initial electronic reference, not a basis/k-converged material-property acceptance.

| Quantity | Actual result |
|---|---:|
| Energy, Hartree/primitive cell | -1946.7055287149658 |
| Volume, Bohr^3 | 893.2025710531027 |
| Spherical AOs / electrons | 94 / 140 |
| SCF iterations | 13 |
| Sampled Gamma KS gap, eV | 4.1688693118175655 |
| Grid-integrated electrons | 140.00333756626503 |
| Fresh final Fock residual, Hartree | 1.772e-8 |
| Fresh final density RMS | 2.534e-9 |

[Pilania et al., Communications Materials1,84(2020)](https://doi.org/10.1038/s43246-020-00082-2), Figure3a discussion, reports a PBE direct Gamma gap of5.11eV for relaxed normal MgAl2O4. Its computational methods specify500eV PAW plane waves and a Gamma-centered9x9x9 mesh for14-atom cells. The present gap is about0.941eV (18.4%) lower. Geometry, basis and full Brillouin-zone sampling are not matched. An independent same-geometry/STO-3G calculation and convergence checks are needed to distinguish implementation differences from model errors. Neither experimental agreement nor a code defect is established by this discrepancy.

The measured initial overlap/core-Hamiltonian construction took45407.677s, 66.34% of total wall time. Repeated Hartree fields took18312.170s and XC fields4359.396s; initial XC-grid construction took262.073s. Screened-Hartree preparation is nested in the Hartree time and is not added again. These historical timings include background load and establish no controlled spinel speedup. The process used two Rayon threads, one faer thread and its inherited two-CPU affinity, with a sampled1.5GiB RSS guard. No exact final peak-RSS value was retained.

Exact inputs, result/library hashes, all phase samples and physical-final-check values are retained in `crystal-benchmark-20261031/spinel-terminal-profile-20261101-v2.json`; the literature discrepancy is separated in `spinel-gap-comparison-20261101.json`. The corrected profile reads the 94 orbital energies from the spin/k/band array; the earlier summary incorrectly counted one k point as the AO dimension and is retained as failed summary evidence. Raw energies and convergence records were unchanged. Stress was not requested. The existing controller advanced to its queued kaolinite calculation after normal spinel termination.

### Completed matched LiF center and pressure-only cost audit

The source04 public stress calculation at the existing fixed LiF center has
completed. Conditions are PBE/pob-DZVP-rev2, unshifted 2x2x2 k sampling,
24 Bohr image support, grid4/shell3, zero grid-weight cutoff and zero
electronic temperature. The primitive volume is 105.08423022388008 Bohr^3.
Energy is -107.31468131934636 Hartree and pressure is
11.596759961334078 GPa. All nine stress entries, all nine strain derivatives,
energy, volume and pressure are exactly equal to the saved prior native
report. This is a same-model replay, not an independent implementation or
equilibrium-material validation. The completed four-step signed-strain
closure is reported separately below.

The final physical SCF check completed at 660.708 seconds after 12 iterations,
with Fock residual 6.281e-10 Hartree and density RMS 6.119e-11. The full public
report took approximately 11670.75 seconds with sampled peak RSS 629348 KiB,
one CPU thread and a 1280 MiB managed-memory setting. Managed-memory accounting
is not a hard process-RSS limit. The force translation residual is
6.858064136710679e-16 Hartree/Bohr.

| Timed phase | Seconds |
|---|---:|
| After final SCF check, before stress construction | 8682.377 |
| Stress kinetic | 0.460 |
| Stress overlap | 0.308 |
| Stress nuclear attraction | 686.674 |
| Stress Hartree | 268.488 |
| Stress exchange-correlation | 1368.427 |

The first interval includes force construction and other dispatch work;
existing markers do not isolate its nuclear derivative cost. The public
stress command requests forces by default. Its existing explicit
`forces=false` control, including CLI `--forces=false`, permits a subsequent
pressure-only request to omit that output. Defaults and explicitly requested
forces remain unchanged. Matched stress-component parity, wall time and peak
RSS tests are prepared but have not yet been executed; no speedup is claimed
for this control. Independent force and force-Hessian gates remain required.

A geometry-only audit identifies an AO-image pair at radius
23.98249958069001 Bohr for the 24 Bohr support predicate. Its selection changes
at positive isotropic strain 0.0007297162354202857; all four declared positive
steps cross that boundary. The corresponding matrix and energy contributions
have not been measured, so this observation establishes neither a derivative
defect nor the cause of the pressure discrepancy. The frozen interval and
thresholds remain unchanged, with all outcomes retained. Exact geometry and
predicate provenance are in
`crystal-benchmark-20261031/lif-source04-pressure-closure-01/support-boundary-geometry.json`.

The initial supervisor failed while atomically replacing its state file
with `PermissionError`; the competing file handle has not been identified.
Its original calculation worker was preserved and the completed report was
recovered without a restart. The unavailable original OS exit status is not
reported as zero. This operational failure remains separate from numerical
results. A bounded replacement-retry helper passed four failure-injection
tests for future controllers; the original hash-pinned controllers remain
unchanged. After a later transport interruption, both the previous supervisor
and its unfinished negative-strain worker were confirmed absent, with an empty
report and retained partial log. A new foreground continuation uses the tested
checkpoint helper, preserves both completed reports without recalculation, and
stores the retried unfinished point separately. Its recovery manifest is
`crystal-benchmark-20261031/lif-source04-pressure-closure-01/transport-recovery-02/manifest.json`.
No scientific controls or acceptance criteria changed. Complete center comparisons and phase samples are
retained in
`crystal-benchmark-20261031/lif-source04-pressure-closure-01/center-completed-profile.json`.

### Completed four-step LiF isotropic pressure closure

All nine declared reports are saved and converged: the center, eight signed
strain points, and one of those strain reports reused with its exact
same-artifact output hash. No completed center or positive 0.008 point was
recalculated during transport recovery. The existing interval, controls and
criteria were retained without an error floor or omitted step.

The central energy difference uses fixed fractional atom and k coordinates:
`P(h) = -[E(+h)-E(-h)]/(2*h*3*V0)`. The analytic pressure is
11.596759961334078 GPa, with V0=105.08423022388008 Bohr^3.

| Isotropic strain h | Difference pressure, GPa | Difference minus analytic, GPa |
|---|---:|---:|
| 0.008 | 11.625692298541729 | 0.0289323372076514 |
| 0.004 | 11.603990601114402 | 0.0072306397803241 |
| 0.002 | 11.598567400639139 | 0.0018074393050621 |
| 0.001 | 11.597211735298513 | 0.0004517739644365 |

The four-step fit gives exponent **2.0002989894** and R-squared
**0.999999999011**, passing the unchanged criteria 1.9<p<2.1 and
R-squared>0.999. This accepts this declared isotropic trace consistency
check at the stated model and numerical controls. It does not establish
all stress components, other reference classes, relaxed or converged material
properties, or agreement with an independent implementation or experiment.
The geometric support boundary noted above did not prevent this particular
four-step fit from passing; its isolated energetic contribution is still
unmeasured.

The separate independent h=0.004 result remains 11.330566385498162 GPa.
At the same finite step, the native result differs by
**0.273424215616240 GPa**. The earlier analytic-native versus independent
finite-step difference was 0.266193575835915 GPa; the native finite-step
correction is only 0.007230639780324 GPa. Thus the internal derivative check
does not resolve the cross-implementation discrepancy. Distinct quadrature
and auxiliary Coulomb approximations remain to be isolated. No reference
value, functional coefficient or tolerance was adjusted to fit this result.
The count of stationary, basis/grid/k-converged material references remains
zero.

The historical supervisor and transport failures remain explicit. The
continuation exits with its retained operational failure status even though
the numerical four-step gate passes; neither a missing OS exit status nor
a failed attempt is rewritten as success. Exact report hashes, energies,
artifact identities and independent recomputation of the fit are retained in
`crystal-benchmark-20261031/lif-source04-pressure-closure-01/final-numerical-evidence.json`.
The matched finite-step comparison is retained in
`crystal-benchmark-20261031/lif-source04-pressure-closure-01/matched-reference-comparison.json`.

Postprocessing the saved h=0.004 reports verifies matching strained cells,
atom positions, basis hashes and fractional k points, and closes each energy
ledger. The nuclear finite-difference pressure agrees within 8.3e-11 GPa.
The native-minus-independent difference splits into +0.0522453488 GPa in
the XC energy difference and +0.2211788669 GPa in the remaining electronic
energy difference. That remaining term combines one-electron and Hartree
energies, which are not separately exposed by these native reports.
These are differences of terms at separately self-consistent densities,
not individually variational stress components. Density relaxation transfers
contributions between terms, so the split does not identify an XC or Hartree
implementation defect. No new SCF or field evaluation was needed. The
reproducible arithmetic and input hashes are retained in
`crystal-benchmark-20261031/lif-source04-pressure-closure-01/saved-energy-pressure-decomposition.json`.

The saved-density independent diagnostic was subsequently replayed with
strict resource cleanup. That replay exits successfully without the earlier
resource warnings. All non-timing JSON fields, including 29 numerical fields
at grid levels 3, 4 and 5, are bit-identical to the retained earlier run.
At level 5 the independent grid integrates 12.0000239413 electrons, and the
maximum induced-field difference remains 2.710872e-4 Hartree. This closes
the cleanup/warning issue only; it neither reconverges a density nor resolves
the pressure discrepancy. The original warning failure remains preserved,
and no new SCF was performed for the comparison audit.

### Completed inherited kaolinite initial electronic reference

The inherited fixed-structure calculation completed normally after 23 SCF
iterations. The local Rust harness used the frozen volume02 library
`cf85cf623ff0be158dccdf5dab91ab3c273c04236b3bf0ed74479067a5f972ab`:
all-electron nonmagnetic PBE/STO-3G, 17 atoms, 130 electrons, 85 spherical
AOs, Gamma sampling, 24 Bohr AO support, grid level 3/shell 3, and zero grid
weight cutoff. The structure is the primitive reconstruction of
[the 1.5 K diffraction structure](https://www.crystallography.net/cod/9009234.cif),
associated with DOI 10.1346/CCMN.1993.0410613. This temperature describes
the measured geometry; electronic occupations were at zero temperature.
No relaxation, quantum-nuclear treatment, dispersion correction or stress
calculation was performed.

The energy is **-1720.7461119581876 Hartree per primitive cell**, volume
1109.1158671182425 Bohr^3. The final physical rebuild gave commutator residual
4.274e-8, occupation-map density RMS 7.729e-9, energy change 6.821e-13
Hartree and zero entropy change. These satisfy the unchanged respective
thresholds 1e-5, 1e-8 and 1e-10 Hartree. The earlier failed physical check
with density RMS 1.094e-8 remains in the log; it was followed by further
iterations rather than a relaxed gate. Grid integration gives
130.00053687420282 electrons, an independent numerical-support diagnostic
rather than a claim of grid convergence.

The sampled **Gamma KS gap is 3.33930767002239 eV**. Both spin blocks have
65 occupied orbitals, with HOMO 0.3239400040566671 and LUMO
0.4466572974659557 Hartree. The saved-band audit confirms sorted spectra,
integer occupations and correct gap extraction. Absolute periodic orbital
energies depend on the potential convention and are not ionization energies.

This low gap remains a physical-comparison concern. The initially retrieved
publisher abstract gave a 6.2-8.2 eV range without method-specific values.
A subsequent [public author-uploaded text of Nisar et al., Phys. Rev. B 84,
075120 (2011)](https://www.researchgate.net/publication/224860832_Optical_gap_and_native_point_defects_in_kaolinite_studied_by_the_GGA-PBE_HSE_functional_and_GW_approaches)
resolves that ambiguity: Section III.C.1, p. 075120-3 reports **PBE 4.9 eV,
HSE06 6.2 eV and G0W0 8.2 eV**. The higher interval must not be used as a
PBE acceptance range. The native gap is 1.56069233 eV below the paper's
PBE value, still a method-context comparison rather than an implementation
error measurement.

The reference uses PAW/plane waves at 500 eV and a 4x2x3 mesh, with a
34-atom cell; its relaxed PBE lattice differs from the fixed native
17-atom primitive reconstruction. Exact relaxed coordinates and a k-resolved
direct/indirect assignment remain unavailable. These conditions do not match
the all-electron minimal-basis Gamma calculation. Basis, quadrature, support,
geometry and implementation effects require independent isolation. At fixed
potential, adding k points cannot increase the minimum sampled indirect gap;
Gamma sampling alone is not an explanation for a low gap. No code defect or
model-error explanation is established by the present comparison. The
updated source audit is
`crystal-benchmark-20261031/kaolinite-primary-method-audit-20261101.json`.

Elapsed time was 42725.937 s (about 11.87 h) at Rayon 2/faer 1, with sampled
peak RSS 1471084 KiB under the existing 1572864 KiB guard. Disjoint logged
costs were initial overlap/core construction 11480.352 s, repeated Hartree
24667.056 s and XC 6359.644 s, approximately 26.9%, 57.7% and 14.9% of
elapsed time. Screened preparation is nested inside Hartree and is not added
again. These historical timings do not measure a source04 speedup. A separate
source review identified redundant same-density field rebuilding after a failed
final check; reuse still requires shared-core implementation and matched
correctness, timing and memory tests before adoption.

Raw identity, controls, physical gates and phase totals are retained in
`crystal-benchmark-20261031/kaolinite-terminal-profile-20261101.json`;
band, occupation and bounded periodic-neighbor diagnostics are retained in
`crystal-benchmark-20261031/kaolinite-saved-gap-audit-20261101.json`.
This closes the inherited initial SCF, not stationary material-property or
new public-interface parity acceptance. The count of stationary,
basis/grid/k-converged material references remains zero.

The bounded follow-up protocol is saved as
`crystal-benchmark-20261031/kaolinite-minimal-followup-protocol-20261101.json`.
The old report retained eigenvalues and occupations but no AO density or
orbital coefficients, so an exact warm restart cannot be reconstructed from
it. The first requested calculation is one independent PBE/STO-3G Gamma
reference on the saved geometry, exporting complete density, orbitals and
metric data. No repeat of the 11.87-hour native cold calculation is requested.
AO-order and overlap checks must precede any density transfer. Subsequent
fixed-density grid sensitivity can reuse the one-electron/Hartree terms and
change only XC quadrature; this is not a reconverged gap test. Full basis,
support and k convergence remains conditional on those diagnostics.
These calculations are prepared, not executed or admitted ahead of the
existing Fe/common-basis serial work.

### Reference-preserving static EPR extension

The dedicated common-orbital providers retain the ROKS/ROHF closed, open and
virtual spaces. The g response uses their paired constrained orbital Hessian;
the unrestricted Hessian is not substituted. The current compatible shared
artifact passes all 23 canonical Rust tests, 141 independent component fits,
and eight actual four-reference g/hyperfine routes through native Python,
ASE and the installed canonical CLI. These are common-origin static routes;
the generalized routes explicitly use the single-Kramers-pair approximation.
The current molecular numbers and remaining limits are recorded below.
Initially, seven source-local tests
actually passed: two common-origin g tests, one HF/PBE methyl hyperfine test,
and four complex-spin-field/cofactor-projection tests. All nine independent
ROHF orbital-Zeeman/SOC mixed-energy components pass four-step fits on
`h=[0.08,0.04,0.02,0.01]`: exponents 1.998196--2.000907, minimum R-squared
0.99999944. The energy oracle uses complex one-electron eigenvalues separately
from the response kernel.

The additional interacting C1 neutral H4 triplet test in
`tests/p_epr_common_orbital_interacting.rs` has three alpha and one beta
electron, including a closed orbital, two open orbitals and a virtual orbital.
All nine one-electron g-SOC components pass independent four-step fits on
`h=[0.08,0.04,0.02,0.01]`: exponents 1.999732--2.000085 and minimum R-squared
0.9999999952. The validation-only complex common-orbital HF optimizer contracts
raw AO electron-repulsion integrals and follows its own constrained orbital
stationarity; it does not reuse the production JK builder or response Hessian.
This is a common-origin one-electron SOC response check, not validation of
two-electron SOC, London response or nuclear derivatives.

| Actually computed quantity | HF common orbitals | PBE common orbitals |
|---|---:|---:|
| Distorted methyl carbon contact, MHz per nuclear g | 307.65290684 | 306.63987904 |
| Distorted methyl g isotropic, common origin | 2.0026514976 | 2.0027121111 |
| Native constrained orbital stationarity, hartree | 4.80e-14 | 2.95e-14 |

These are implementation diagnostics with STO-3G, grid level 2 and zero
grid-weight cutoff, at the C1 Bohr geometry in
`tests/p_epr_common_hyperfine.rs`. They are not basis/grid-converged chemical
predictions or comparisons to experimental methyl couplings. All four nuclei's
complete contact and spin-dipolar tensors were evaluated; tensor symmetry and
dipolar trace were checked.

All six actual C1 methyl native routes, ROKS/PBE, ROHF/HF and GHF/HF times
g and hyperfine, subsequently execute successfully with the full static
artifact. The first common-orbital g attempt retained a stationarity failure
of 1.064e-10; tightening the SCF to energy/density 1e-13 and gradient 1e-12,
and requesting response tolerance 1e-12, resolved it with stricter settings.
The origin is the carbon position. GHF retains the original SCF
tolerances and reaches an orbital residual of 5.63e-13. Its pair-axis tensor
must be compared through `g g^T`, rather than individual gauge-dependent
columns. These native calculations do not establish chemical basis/grid
convergence or the separate GKS molecular acceptance.

The generalized providers retain all complex spin-density blocks, orbital
currents and self-consistent one-electron SOC. A raw noncollinear nuclear field
does not define an effective-spin tensor by itself. The optional
`single-kramers-pair` model projects operators into the determinant and its
time reverse using cofactors, including singular occupied overlaps. Its
one-electron rotated-pair invariants and three-electron transitions against an
independent 20-state Fock-space calculation pass. SOC GHF methyl converged at
energy -39.01323495048831 hartree, orbital residual 6.62e-13 and spin-squared
expectation 0.7578928623; the PBE spinor calculation remains in progress.
Stationarity alone does not establish angular-XC convergence: the new EPR
entry checks the operator flag and reports the actual refinement diagnostics.

The projected g matrix uses field axes for rows and pair-Pauli axes for columns;
`g g^T` is invariant to pair-basis rotations. A single contaminated determinant
can bias transverse couplings substantially, as discussed in the
[primary generalized-HF analysis](https://doi.org/10.1080/00268976.2023.2192820).
No spin purification, isolated-doublet proof, general higher-multiplet tensor,
two-electron SOC, or London response is inferred. The initial common-orbital
g provider also rejects missing meta-GGA current-response terms explicitly.

The dedicated static driver passes warning-free metadata, compilation and two
runtime tests covering all four references times both static properties on H,
all nine signed nuclear-g scalings, and explicit unsupported-model guards.
One full isolated library and wheel have now been built from the accepted
baseline plus the owned static changes. All eight routes execute through its
actual new native extension, CLI entry function and ASE getters; all nine
entries, energies and atomic/converted units agree. The CLI subprocess retains
default SCF energies and residuals on stderr. Four canonical Rust tests linked
to the same library also pass, including the interacting H4 response test.
All eight routes also pass with `DFT_RS_STRICT_UNITS=1` using the previously
installed `dft-rs` launcher selected through PATH and the actual candidate
extension selected through PYTHONPATH. These are atomic routing and scoped
molecular checks. Final composition with the newer shared API and moving-nucleus
providers, installation of that distribution, and chemically converged system
predictions remain separate gates.
Static reference tests do not establish the moving-nucleus A derivatives.

The four-reference static branch, strict control units and separate nuclear-A
preflight are now connected in the shared source. A current combined candidate,
including the LDA recovery and complete raw SCF diagnostics below, passes the
whole-library metadata check with warnings treated as errors. The initial check
retained a missing Cargo-version environment failure and an unused import from
the older integration baseline; supplying the package metadata and removing only
that import resolves them without changing response equations. This typecheck
does not establish execution of the final combined library or distribution.
The older full static runtime evidence above remains a separate source version.

An isolated point-kernel optimization uses 4/16/32 active automatic-differentiation
directions for LDA/GGA/meta-GGA. Energies, all 32 returned coefficients and
separate functional contributions are bit-identical to the original implementation
for SVWN5, PBE, TPSS, PBE0 and CAM-B3LYP at angular orders 4, 8 and 16.
The independent scalar oracle in `tests/support/spinor_xc_scalar_energy.rs`
constructs `f + D_spin f` from explicit projected densities, gradients,
current-corrected kinetic densities and first functional partials. It does not
call the production spinor potential or response kernel. All 84 active
descriptor gradients pass four-step fits on `[0.08,0.04,0.02,0.01]`, with
exponents 1.996119--2.010506 and minimum R-squared 0.99999169; no failed intervals,
floors or dropped steps. This verifies derivatives of the fixed order-16
quadrature, not physical angular convergence. Initial matched point benchmarks
give approximately 1.37 times PBE and 2.71 times LDA throughput; these are kernel
measurements, not full-SCF speedups. The tested narrow point-function change is
now applied to the shared source; kernel, directional and Taylor providers are
preserved. The isolated full static artifact includes this optimization, while
the final common API artifact still requires composition and acceptance.

The formatted dedicated driver also passes the strengthened every-entry atomic
g/hyperfine Gram-matrix, same-functional collinear-energy and spin-squared limit
checks. The earlier eight generated-report conversion checks used an accepted
extension solely for imports. The later full-artifact checks described above
actually call the new EPR native implementation and supersede that earlier
FFI limitation without changing its retained evidence.

The exact grid branch now considers only the spin descriptors consumed by the
functional family: local density for LDA, density and gradients for GGA, and
all kinetic/current descriptors for meta-GGA. This retains spatially varying
full complex Pauli densities; local LDA spin diagonalization does not constrain
the molecule to a common spin axis. The independent local spin-eigenvalue
oracle fits all 16 Hermitian two-AO density components at four steps, with
exponents 1.999962--2.000138 and minimum R-squared 0.9999999987. The independent
zero-gradient GGA limit with transverse inactive currents, complete active
noncollinear PBE/TPSS Fock parity, gauge checks, inactive-NaN rejection and
local spin-density positivity controls pass. These four tests are in
`tests/p_epr_spin_xc_family_grid.rs`. The final common native artifact has not
yet been rebuilt with this additional classifier change.

The additional SOC methyl/SVWN5 GKS calculation is not accepted. Both a rotated
spin seed and a z-axis seed reached small orbital residuals but failed the
unchanged density/occupation consistency gate before the bounded run ended.
For the z seed the early orbital residual was approximately 2.74e-11 hartree
while occupation consistency remained approximately 3.21e-10 against 1e-11.
Quasi-Newton steps based on total-energy changes near 1e-14 hartree then
increased the residual. This is a measured convergence obstacle, not a
validated stationary EPR tensor or an established diagnosis of its cause.
The original methyl/PBE spinor baseline is preserved separately. No methyl
GKS tensor is inferred from the atomic public checks or the local grid fits.

The separate three-electron C1 H4+ case now converges with the production
global-spin recovery function in a source-local integrated XC/SCF test.
It uses SVWN5/STO-3G, grid level 2 with all 21,488 weights retained,
one-electron SOC and an unconstrained complex reference. The initial z spin
axis is a seed only. At total energy -1.6769848451687972 hartree, the orbital
residual is 8.6263e-13, occupation consistency 7.295e-12 and density change
3.950e-12. XC angular energy/operator defects are zero on the exact LDA
limit. Fourteen global rotations and 120 total SCF iterations take about
47 seconds for the accepted run; maximum scalar-energy/SOC-work mismatch
is 7.33e-16 hartree. Complete g and all four nuclear A tensors are evaluated;
the determinant has spin-squared expectation 0.7512079762 in the declared
single-pair model. These are diagnostic model values, not chemically
converged predictions.

The rotation is the minimum of a four-dimensional quaternion Rayleigh
quotient. It preserves every complex density block and the physical
Hamiltonian, rather than fixing a spin direction. All three independent
spin torques pass dense-complex four-step energy checks on
`[0.08,0.04,0.02,0.01]` (exponents 1.999855, minimum R-squared 0.9999999986).
The canonical tests in `tests/p_epr_global_spin_rotation.rs` also pass
invalid-input, metric invariance and total-iteration-budget checks. The
production helper restricts recovery to insulating molecular semilocal LDA;
HF and other functional families retain their ordinary solver. The dedicated
driver passes five source-local regressions after this change. The canonical
public H4+ regression in `tests/p_epr_generalized_lda.rs` is prepared for the
final common artifact; its public execution has not yet been claimed.

The subsequent actual shared-library acceptance executes that public H4+
regression and both native Python EPR routes. Both native calculations converge
in 120 total iterations at -1.6769848451687976 hartree, with orbital residual
8.62620177495e-13, density change 3.96493948784e-12, occupation consistency
7.29749594086e-12, zero particle-number error and metric defect 4.44e-16.
All raw gates and the XC operator convergence gate pass. The two native routes
take 93.97 seconds together on one thread. These remain minimal-basis model
diagnostics, with the explicit single-Kramers-pair approximation and common
gauge origin, rather than converged molecular predictions.

The same actual-library static suite passes 16 of 17 tests. Its remaining
failure is a rotated hydrogen GKS/PBE PSD-boundary rejection, reproduced in
the physical native backend before its first XC evaluation. At 711 of 5,372
grid points a normalized rank-one density has a computed spin magnitude above
its charge density by less than one machine epsilon (maximum relative deficit
2.00545701e-16). This is an arithmetic boundary error, not evidence of a
physically indefinite reference. A narrow correction admits only a relative
64-epsilon arithmetic allowance and repairs negative roundoff projected
eigenvalues while preserving total charge. It retains every positive minority
density, the existing functional density cutoff, all SCF gates and the generic
AD projection and high-order response children unchanged.

The independent static correction suite passes all six tests; the unmodified
baseline fails the new rank-one test. Rotated hydrogen at three density scales
matches independent scalar collinear energies to relative error 1.56e-16 and
every complex Fock coefficient to at most 1.40e-14. Positive minority and
genuinely indefinite/negative-charge controls pass. All 84 active-descriptor
and 16 complex AO-density four-step derivative fits also pass with their
existing thresholds. The repaired actual library subsequently passes all
20 canonical static EPR tests. Its matching audited wheel passes all eight
atomic g/A routes through native Python, ASE and the installed canonical CLI,
with strict units and signed-isotope scaling. Both native H4+ g/A calculations
also pass all unchanged raw gates at 120 iterations. This closes the static
PSD repair, without establishing the separate chemical or nuclear-derivative
capabilities below.

The methyl/SVWN5 case still fails after 600 active seconds with the same
recovery: occupation consistency remains 2.32e-10 against 1e-11. Its earlier
plain-SCF and recovery failures are retained. H4+ success does not close this
different molecular case, and the common native artifact has not yet been
rebuilt with the new recovery helper.
The retained 41-cycle log ends at orbital residual 1.9765e-11, rotation angle
2.7407e-4 rad and occupation consistency 2.3184e-10. Over the last 20 cycles,
the occupation-error ratio is approximately 0.99189 per cycle (log-fit
R-squared 0.99999942). The largest energy/rotation-work discrepancy is
1.30e-14 hartree. These measurements identify a slow coupled electronic and
spin-orientation iteration despite an orbital residual below its threshold;
they establish neither a kernel defect nor an accepted stationary tensor.
The decay fit is diagnostic and is not used to extrapolate acceptance.

A subsequent same-start C1 methyl/SVWN5 diagnostic compares DIIS histories
of four and twelve at 96 physical iterations each. Both remain unconverged:
occupation defects are 2.934944e-10 and 2.934994e-10, respectively, against
1e-11. Their orbital residuals are approximately 2.5019e-11 and their printed
energies agree at -38.86163683563977 hartree. Reconstructing each canonical
density from its actual bands and occupations reproduces the reported defect.
Projection of that defect onto the three global SU(2) density tangents
accounts for 99.99945--99.99949 percent of its squared Euclidean AO Frobenius
norm. This norm is basis dependent. The result identifies a global-spin
orientation slow mode that changing the DIIS history did not resolve in this
interval; with SOC, this orientation is not a removable gauge freedom.
No unconverged g or A tensor is accepted from either endpoint. The diagnostic
takes 377 active seconds, interleaved with the preserved baseline calculation;
its per-case wall times are not used as matched performance measurements.

A safeguarded orientation-secant candidate resolves the methyl/SVWN5 slow
mode on the actual accepted full-spin backend. It reaches all unchanged
stationarity gates in 37 total physical iterations and four rotation cycles:
energy -38.86163683564086 hartree, orbital residual 6.65728e-14,
occupation consistency 6.86796e-13, density change 1.08067e-12 and zero
particle-number and final energy-change errors. Two extrapolated spin seeds
are accepted only after fresh physical SCF segments reduce the measured
stationarity defect without an energy increase beyond arithmetic resolution.
Every attempted segment counts against the original iteration budget.
The five-test private suite also passes the existing three torque fits,
invalid-input/budget controls and an independent dense-complex quaternion/work
oracle. It takes 124.82 active seconds with sampled group RSS 53,192 KiB.
This is an isolated solver result, not yet a changed native solver or evidence
that methyl/PBE has converged. The subsequent actual projected methyl tensors
are finite, with spin-squared expectation 0.7520675647. H4+ also meets the
original gates with the candidate in 120 iterations. However, agreement of
SCF energies to 6.44e-15 hartree does not establish convergence of this
single-pair model's tensor: the largest ordered `g g^T` change relative to
the accepted default reference is 1.24475e-4. Tightening only the spinor
density, occupation and particle-number gates from 1e-11 to 1e-13 produces
another converged H4+ state in 272 iterations, but changes the laboratory
`g g^T` entries by as much as 1.07086e-3. The three principal g magnitudes
change by up to approximately 2.41e-7. Pair-Pauli gauge rotations cannot
remove a change in `g g^T`. The physical orientation and approximate
time-reversed-pair subspace remain under investigation. These results concern
the initial private variant; the corrected implementation and its separate
native-acceptance boundary are described below. No relaxed tensor-convergence
threshold or converged nuclear g derivative is inferred from these results.

For methyl, the matched 1e-13 spinor density/occupation/number run converges
in 53 iterations and six cycles at -38.86163683564083 hartree, occupation
consistency 3.52478e-14 and density change 5.92026e-14. It preserves the
ordinary molecular seed and every other physical/numerical setting.
Relative to the 37-iteration result, the largest ordered `g g^T` change is
1.41509e-6. The largest change in a nuclear A singular value is
5.71788e-4 MHz per nuclear g; all four nuclear `A A^T` and `g A^T` matrices
were also compared. Raw pair-axis columns change substantially and are not
used as a gauge-invariant convergence test. The final frozen-density SOC
torque components are approximately `[-2.83e-15,5.06e-15,2.65e-16]` hartree.
This private run passes all five solver/control tests in 210.99 active
seconds, with sampled group peak RSS 53,384 KiB. A further 1e-14 interval
also converges, in 69 iterations and eight cycles: energy
-38.86163683564084 hartree, orbital residual 3.09182e-15, occupation
consistency 3.16414e-15, density change 4.82947e-15 and number error
1.77636e-15. All five private solver/control tests pass in 260.57 active
seconds, with sampled peak RSS 53,460 KiB. Relative to the 1e-13 run, the
largest laboratory `g g^T` change falls to 3.27835e-8 and the largest g
principal-value change to 4.54388e-11. The largest A singular-value change
is 1.31882e-5 MHz per nuclear g. The three g magnitudes are
`[1.9987402747499614,1.9991138239055388,2.0022212736801297]`.
These are matched numerical convergence results for the explicit minimal-
basis single-pair approximation. They do not resolve the distinct H4+
orientation issue or establish chemical accuracy.

Matched tests with the unchanged native solver also find H4+ tensor
sensitivity: tightening only the spinor density/occupation/number gates
from 1e-11 to 1e-13 changes the largest laboratory `g g^T` entry by
2.85029e-4 and the largest principal g magnitude by 3.30896e-8, at 120 and
320 iterations respectively. The effect therefore predates the secant
extension; this comparison does not establish a unique physical cause.
The initial private extension additionally applied a global spin rotation
after the orbital and occupation stationarity gates had already passed,
while the iteration-to-iteration density gate was still settling. The
corrected extension continues physical SCF from the same full complex
density at that point. Its tight H4+ test converges in 130 iterations,
with orbital residual 1.54562e-14, occupation consistency 9.78662e-14,
density change 9.81992e-14 and energy change 8.88178e-16 hartree. Relative
to its default-gate result, the largest `g g^T` change is 1.24638e-5 and
the largest principal g change is 1.34247e-9. The remaining tensor/state
sensitivity is retained as an open validation issue.

The final implementation passes five source-local solver/control tests for
each of methyl and tight H4+, including the original three independent
torque fits, and strict matching-toolchain Clippy. The final methyl test
uses 53 iterations and reproduces the tighter result above. This corrected
LDA-only recovery has been adopted in source without changing convergence
thresholds or the total physical-iteration budget. HF, GGA and meta-GGA
retain their existing solver routes. A new canonical-driver methyl g/A
regression is present in source. The earlier accepted native and an
intermediate shared build predate this recovery, and the intermediate build
also predates the PSD and static-point corrections. A subsequent compatible
shared library now includes all nine reviewed static-source/test files;
native compilation and linking of its canonical test executable succeed.
Actual public-route runtime acceptance of that static version now passes:
23/23 canonical Rust tests, including both g and A for C1 H4+ and methyl,
and all eight native-Python/ASE/installed-CLI reference/property routes.
The 141 independent component fits retain their original four-step
intervals and thresholds: measured exponents 1.996119--2.010506 and minimum
R-squared 0.99999169. The three torque fits additionally retain their
stricter exponent and fit-quality gates. Signed nuclear-g scaling, atomic
and ASE units, actual native identity, default CLI SCF stderr, and explicit
unsupported nuclear-reference errors pass. No additional root build was
used for these tests.
All twelve current public-test crates pass
strict metadata checks against the existing public APIs, separately as
Cargo compiles integration tests. This establishes test-source compatibility;
it is separate from the subsequently executed same-artifact tests above.

The actual installed Python API also computes both methyl GKS tensors with
the same SVWN5/STO-3G, C1 Bohr geometry, grid level 2, zero grid-weight
cutoff, neutral charge and doublet seed as
`tests/p_epr_generalized_lda.rs`. The origin is the carbon position; the
spin-axis input is a seed, not a final collinear constraint. Both calls
converge in 37 physical iterations and four recovery cycles at
-38.86163683564086 hartree. Orbital residual is 6.65728e-14, occupation
consistency 6.86796e-13, density change 1.08067e-12, and final energy-change
and particle-number errors are zero. The spin-squared expectation is
0.7520675646772892. The g principal magnitudes are
`[1.9987402742802132,1.9991138233697856,2.0022212746184707]`.

This 37-iteration result is specifically **SVWN5/STO-3G**. The original
PBE/STO-3G molecular calculation is a separate retained run and remains
unaccepted until its actual terminal result and physical operator/SCF gates
are available. Its functional has not been substituted by SVWN5.
The exact accepted static artifact is the combined library with rlib SHA256
`e0b74ea186f076a3d0227c420792a2b24d4b750cb1f0386363f5ea2decd1a0cb`,
native-core SHA256
`5925210a4677896ebf655da9d7fcd398cbb9676b892eb5f50f6923883bc98207`,
and installed wheel SHA256
`5242c37260679116965bc1281c486bb115f0c49745bf0a5cf735c2d48bc452c5`.

| Physical SCF acceptance gate | Required | Actual methyl SVWN5 g and A |
|---|---:|---:|
| Orbital residual | < 1e-10 | 6.657277327525115e-14 |
| Density change | < 1e-11 | 1.0806679986638534e-12 |
| Occupation consistency | < 1e-11 | 6.867961060974537e-13 |
| Particle-number error | < 1e-11 | 0 |
| Energy change, hartree | < 1e-12 | 0 |
| Physical operator converged | true | true |
| Total physical iterations | <= 500 | 37 |

The following rounded signed matrices use laboratory field/nuclear axes
for rows and the computed pair-Pauli axes for columns. g is dimensionless;
all A matrices are MHz per signed nuclear g, before an isotope factor.
The raw tensors need not be symmetric in this pair frame. Full tensors
and every convergence diagnostic were retained from the actual native calls.

```text
g = [[ 1.549550336785,  1.262062095113,  0.043128463131],
     [-1.259693680804,  1.549967754503, -0.080317658071],
     [-0.083998613432,  0.035041011546,  2.000144739180]]
A_C = [[ 199.130699211004, 162.195975605399,   9.532741022871],
       [-161.880487701381, 199.159137881192, -16.195819902889],
       [ -11.206097988006,   5.284943983224, 430.309521221436]]
A_H1 = [[ 7.745882767860,  9.230001504093, -5.498231066257],
        [ 3.323446671181, -0.177766525587, -0.772215536258],
        [-2.997837111613, -3.594918741046, -3.908639235078]]
A_H2 = [[-0.298299672875, -4.216429361891,   0.994966527138],
        [-7.393281858912,  4.279881564091,  -4.605817606526],
        [ 3.134807562722, -2.328129377953, -10.677347307357]]
A_H3 = [[1.257107699004, 9.388748801342,  3.577698819316],
        [1.383070275452, 8.294217197807,  3.475667695931],
        [0.631837885927, 4.153297303262, -4.001288046772]]
```

These are implementation and numerical-convergence diagnostics for the
explicit minimal-basis pair approximation, not chemically converged methyl
predictions. Each native report retains its own pair frame; separate g/A
API calls are not silently aligned. The tighter intervals and the H4+
conditioning issue below remain distinct from this public-route acceptance.

Both H4+ routes were also executed through the same installed native Python
core, reusing the existing environment: g and A each converge in 120 physical
iterations and 14 recovery cycles at -1.676984845168804 hartree, with orbital
residual 2.16482e-14 and all original density, occupation, number and energy
gates satisfied. These two native calls complete the explicit three-electron
regression separately from the Rust tests and the nine-electron methyl calls.

The next H4+ interval, at density/occupation/number tolerance 1e-14, does
not pass within the unchanged 500-iteration budget: final orbital residual
3.28167e-15, occupation consistency 1.54876e-14, density change
4.22717e-14, and energy-change and particle-number errors both
8.88178e-16. Four control tests pass and the physical-reference test fails.
No tensor is emitted from that unconverged state. The first run was
interrupted by an atomic progress-file replacement error; the retry reused
the identical executable, reproduced all 54 overlapping recorded physical
proposals exactly, and reached this numerical terminal result. Both attempts
are retained. This failed interval is not replaced by the passing 1e-13
interval, nor interpreted as proof of an unavoidable arithmetic floor.
A private comparison with 32-iteration physical SCF segments retains the
same tighter gates and total iteration budget. It reaches the 600-active-
second runtime bound, with no converged tensor or complete control suite;
this is a runtime-limited diagnostic, not a 500-iteration convergence result.
The longer segments also activate the shared solver's energy-globalization
heuristic. Observed steps as small as 2.98e-8 have energy changes near
2.22e-16 hartree while orbital residuals remain near 1.11e-8. The comparison
therefore does not isolate SCF-history resetting alone. This slower variant
is not adopted. A separate private candidate retains eight-iteration
segments and tries bounded physical SCF polishing near stationarity. Its
trial-selection rule does not change any final convergence gate; a trial
must halve the stationarity score or pass every original gate, in addition
to the operator and energy safeguards. That optional experiment's supervisor
disappeared while its child was stopped. The owned child was reclaimed after
identity checks, with no scientific result inferred. Its source was not
adopted. Validation of the compatible shared library takes priority.

Actual methyl/PBE initial-density profiling uses the same C1 geometry,
STO-3G, grid level 2, zero grid-weight cutoff and physical SOC backend. It
captures the initial complex density before evaluating a spinor Fock, rather
than launching a second physical spinor SCF. All 27,324 spatial grid points
take the noncollinear path. Of these, 2,002 are below the existing built-in
density cutoff; this count is diagnostic and no new screening is applied.
A representative dense point takes 2.002 seconds at angular order 128:
1.330 seconds for native second functional derivatives, 0.032 seconds for
the variational chain contraction and 0.089 seconds for the separate
functional components. The remaining time includes descriptor projection
and temporary data preparation. A low-density tail point still costs about
0.54 seconds at this order while returning exact zero for the existing
built-in provider. These are fixed-point timings, not measured full-SCF
speedups or evidence that the legacy methyl/PBE calculation has converged.

The static point implementation now bounds its temporary angular descriptors
to 256 directions, preserving quadrature nodes, weights, provider parameters
and scalar accumulation order. It skips a Jacobian contraction only when
that Jacobian is exactly zero and the complete native Hessian was already
checked for finite entries. The initial source-local suite passes eight tests:
all original 84 active-descriptor and 16 complex AO-density fits, the PSD
boundary controls, bitwise energy/potential/component comparisons for five
functional compositions and a generic quartic provider. The latter checks
all four active LDA derivatives against independent sphere moments at
32,768 angular directions, using four fixed steps with exponent 2.000000
and R-squared 1.00000000. Every provider batch is at most 256 points.
Separate fresh processes over the same executable and eight identical
PBE/order-128 calls measure sampled peak RSS 75,768 versus 6,568 KiB and
kernel times 17.528 versus 16.364 seconds. This is about 91 percent less
sampled process memory and 7 percent more throughput for this kernel input.
The directional, kernel and Taylor child modules and every scientific
tolerance are unchanged. The new point implementation has source-local
acceptance; it is not included in the earlier accepted PSD native artifact.

The follow-up static GGA projection constructs its exact sparse descriptor
jets directly. Densities are linear and gradient contractions quadratic in
the independent Pauli descriptors; no density/gradient mixed terms exist.
The generic projection remains unchanged for meta-GGA and all higher-order
response paths. The final source-local suite passes ten tests, including
all previous bitwise and derivative gates, the actual methyl/PBE initial
descriptor capture and 20 independent sphere-moment fits (four LDA and all
16 GGA descriptors). The GGA oracle uses the independent fourth sphere
moment of its gradient Gram matrix; every fixed-step fit has exponent
2.000000 and R-squared 1.00000000. No cutoffs, angular rules or tolerances
change. For the actual captured dense PBE point, matched angular-128 time
falls from 2.081 to 1.529 seconds; for the exact-zero provider tail it falls
from 0.594 to 0.0385 seconds. Energy, all 32 returned partials and functional
components remain bit-identical. These point timings do not establish a
full-SCF speedup. This sparse point update is applied to the source and
passes its local suite; the earlier accepted native artifact is unchanged.

Replacing the native dense second-derivative call with the existing public
Taylor directional callback was also measured, rather than assumed faster.
All seven potential and Hessian-vector entries agree within 1e-12 for five
functional compositions. However, 1,024 PBE points take 0.0624 seconds through
the native dense batch and 0.440 seconds through repeated order-zero Taylor
directional calls. The latter route is not adopted for this static kernel.
This is a consistency/cost check of existing callbacks, not validation of a
new XC derivative or a change to the physical model.

A matched replay of the tight H4 reference on the accepted shared library
reproduces 130 physical iterations and 15 cycles, retaining all original
gates. It additionally saves the independent spin expectation and coherent
g/A state. The default-to-tight spin-axis difference is 0.00130055 radians
(about 0.075 degrees). Aligning only these independently computed spin axes
for diagnosis reduces the laboratory `g g^T` difference from 1.24638e-5 to
4.97094e-8; the maximum principal-g drift is 1.34248e-9. The raw output
tensors remain the acceptance quantities. No rotated tensor is promoted.
The frozen-orbital global-spin SOC curvature eigenvalues are
`[1.14928e-11,1.96556e-10,2.07219e-10]` hartree per radian squared.
All six independent analytical curvature components pass four-step
closed-SU(2) energy checks on `[0.08,0.04,0.02,0.01]`, with exponents
1.997931--2.000540 and minimum R-squared 0.99999952. The weak spin pinning is
consistent with the observed tensor sensitivity. This diagnostic fixes the
occupied orbitals; it does not establish relaxed stability, ZFS, an isolated
exact doublet, or nuclear derivatives. The original tighter 1e-14 failure
and unrotated tensor-convergence limitation remain.

A separate private PBE fixed-density optimization evaluates the exact
Hessian-vector contraction with mixed f64 duals, retaining the existing energy
expressions, parameters, density branches and angular rules. All 25 active
kernel entries match the native dense result; all 15 independent entries pass
four-step energy fits on `[0.04,0.02,0.01,0.005]`, with exponents
1.988035--2.132530 and minimum R-squared 0.99906524. With all returned values
consumed, 32,768 evaluations take 1.180324 seconds for the dense path and
0.452830 seconds for the product path. This is a kernel timing, not an SCF
speedup. The projected point at angular order 128 takes 1.514045 versus
0.614682 seconds; all 32 partials differ by at most 2.23e-16 over orders
4, 8, 16, 32 and 128, and the complete complex AO operator differs by
1.05e-17. All 16 independent descriptor fits pass the original four-step
gate, with exponents 1.999682--2.000650 and minimum R-squared 0.99999995.
The shared generic-provider seam passes whole-library metadata and Clippy
under the original package lint policy; its canonical public tests pass
typechecking and strict Clippy. Actual execution of a newly composed native
library remains pending. These results do not establish a new native PBE SCF
or resolve the separately running original methyl/PBE calculation.

The actual initial C1 methyl/PBE complex density was also tested on all 27,324
spatial points, with eight spatial AOs. At angular order 4 the complete XC
energy/operator calculation takes 38.988636 seconds before and 16.536120
seconds after the product rearrangement, with identical energy/components and
maximum complex operator difference 4.45e-16 hartree. At order 8 the operator
difference is 1.12e-16 hartree; its wall times include the explicitly recorded
cooperative pauses and are not interpreted as an isolated speed ratio. The
original angular 4-to-8 change is 2.80648e-5 hartree in XC energy and
1.1831084e-3 hartree in the operator, failing the required 1e-10 hartree gate.
This establishes a separate quadrature-convergence obstacle as well as the
measured dense-kernel cost.

A further private quadrature experiment subtracts a collinear auxiliary
integrand and adds its analytical symmetric endpoint integral. For the
collinear spin variable s, the identity is
`integral[-1,1](f(s)+s*f'(s))/2 = (f(1)+f(-1))/2`.
The reference has a fixed laboratory z axis and includes its complete
descriptor Jacobian. Its auxiliary density is
`(1+128*EPSILON)*rho + 4*RHO_CUTOFF`, so its endpoints avoid native cutoff
branches. The target density, functional and density cutoff retain their
original values. This changes the finite angular quadrature approximation;
it is distinct from the exact Hessian-product rearrangement above.
All 16 independent descriptor fits and complete collinear-limit checks pass.
On the same full initial density, the controlled 4-to-8 energy change is
7.994e-14 hartree and the operator change is 1.77824e-9 hartree. The latter
still fails 1e-10 hartree. Order-16 continuation uses a bit-identical saved
initial density and gives an 8-to-16 energy change of 2.487e-14 hartree and
operator change of 4.68393e-10 hartree, which also fails the original operator
gate. No changed default, physical spinor SCF, nuclear response or high-order
derivative acceptance is inferred from this private experiment.

A second private control blends fixed-axis and local-magnetization-axis
references with a smooth binomial weight. The complete reference, axis and
weight chain rule is retained. All 48 independent descriptor checks pass
on the declared complete base, polarized and zero-magnetization intervals.
The original fine-step zero-spin cancellation failure remains recorded;
all 16 original fine components were collected before using the complete
coarser interval `[0.16,0.08,0.04,0.02]`. Component 10 then has exponent
2.005380 and R-squared 0.99999528. No component-specific interval or error
floor is used. A separate independent contraction of full complex spin
density matrices validates all 16 Hermitian AO-density components on
`[0.04,0.02,0.01,0.005]`, with exponents 1.999820--2.005165 and minimum
R-squared 0.99999822. On the same full methyl initial density, adaptive
order 4-to-8 changes the energy by 6.5814e-13 hartree and the complete
operator by 5.74069e-9 hartree. The operator still fails 1e-10 hartree.
The unchanged-density order-16 continuation completes with exit 0 and
384.143 process CPU seconds. Its energy change is 1.68754e-14 hartree;
the complete operator change is 1.2797413e-10 hartree, which still fails
the unchanged 1e-10 gate. Exit 0 records execution rather than angular
acceptance. These source-local variational checks do not establish a
converged physical PBE spinor SCF or installed public implementation.

The private compositional provider also passes complete-field limiting
oracles with nonzero SOC: both a collinear state and a rotated state match
all 200 real/imaginary Fock entries to 1.066e-14 hartree and all ten energy
components to 1.999e-15 hartree. These fixtures exercise the existing exact
collinear grid branch; they validate Hamiltonian composition rather than
general noncollinear quadrature. A further moving-frame quadrature retains
the complete frame, weight and reference chain rule. Independent scalar
Rodrigues rotations and all spin Gram entries are checked. All 96 independent
descriptor and complex-AO component fits pass, including both rotation
charts, their equator and zero magnetization: exponents are
1.999297--2.005165, with minimum R-squared 0.99999822. All 16 original
zero-spin fine fits remain recorded, including one cancellation failure;
the complete declared coarse interval is used for acceptance. The moving
frame's same-density full-grid order 4-to-8 check completes in 237.854 wall
seconds, including cooperative pauses, and 177.534 active seconds.
The two quadratures use 35.804 and 133.638 process CPU seconds. Their energy
difference is 2.14939e-13 hartree, but the complete Fock difference is
1.74938e-9 hartree and fails the unchanged 1e-10 gate. The order-16
continuation completes in 520.365 process CPU seconds and 781.574 wall
seconds, with a 3.01981e-14 hartree energy change and 2.47729e-10 hartree
complete Fock change. It also fails the original gate. An independent
contraction of all three uniform-spin generators measures an XC torque norm
1.31096e-14 hartree/radian for this result. This checks finite-rule spin
covariance; it does not correct tensors or replace full-operator convergence.

Separating the polar and azimuthal orders identifies unnecessary cost for
this saved density. At fixed polar order 16, reducing the azimuthal count
from 32 to 8 changes the energy by 2.665e-15 hartree and every complex Fock
entry by at most 8.882e-16 hartree. Measured process CPU time falls from
520.365 to 137.247 seconds, about 3.79 times faster for this complete XC
field. Polar 8-to-16 refinement at azimuthal count 8 still changes the Fock
by 2.47729e-10 hartree. The reduced azimuthal count is not a universal
default: the separate complex three-point C1 fixture requires polar 16 and
azimuthal 32. A private controller checks both refinements independently,
with a cache confined to one density evaluation. All 16 independent complex
AO-density fits for that controller pass the unchanged four-step gates;
changed-density invalidation and invalid-tolerance checks also pass. Actual
methyl controller calculation subsequently passes both angular checks at
polar 32 and azimuthal 8. The maximum complete-operator defect is
8.04859e-13 hartree and the maximum energy defect is 8.88178e-16 hartree;
seven rules are evaluated. Its XC contribution is -6.532493218909836
hartree, not a converged molecular total energy. The run takes 738.441
process CPU seconds and 1133.498 wall seconds, including cooperative pauses.
The physical-backend limiting tests also pass all 200 real/imaginary Fock
entries and all ten energy components. Its one-entry density cache returns
bit-identical fields for exact repeats and recomputes after a one-ulp density
change. The actual private PBE spinor-SCF/g/A run has now started with a
bit-identical initial density and the same compiled test executable. The
original 500-iteration maximum and all physical convergence thresholds are
unchanged. No physical-SCF, shared/native, chemical or new-default acceptance
is inferred before that run and its public integration complete.

#### Static EPR continuation: measured limits and open acceptance gates

This checkpoint separates accepted static reference routes from unfinished
PBE convergence and later physical validation. The accepted common-origin
artifact has library hash `e0b74ea1` and native hash `5925210a`. Its 23 Rust
tests, 141 independent four-step fits and eight reference/property public
routes cover the stated ROHF, ROKS, GHF and GKS fixtures. RO references retain
their common-orbital constraints. Generalized results use the explicit
odd-doublet, single-Kramers-pair approximation with full complex one-electron
SOC; they do not establish arbitrary multiplicity, multiple-pair projection,
London-gauge coverage, two-electron SOC or nuclear derivatives.

The 37-iteration methyl result is specifically SVWN5/STO-3G, grid level 2,
all weights retained, nine electrons and a doublet seed. Its energy is
-38.86163683564086 hartree; orbital, density and occupation residuals are
6.65728e-14, 1.08067e-12 and 6.86796e-13. It is not a PBE result and does
not establish chemical accuracy from this minimal basis.

The separate private PBE/STO-3G run retains all 27,324 grid points, eight
spatial AOs, full complex SOC and the original physical gates: 500 maximum
iterations, energy 1e-12 hartree, density/occupation/particle-number 1e-11,
and orbital/angular operator 1e-10. At this checkpoint 4 complete
physical field evaluations are preserved. The latest total energy is
-39.211474001097265 hartree, at polar order 32 and azimuthal
order 8; its independently checked angular energy and
operator defects are 8.881784e-16 and
9.648955e-13 hartree. Consecutive field calls may include
line-search trials: they are not accepted SCF iteration counts or a final
convergence decision. No final physical SCF or same-state g/A result is
accepted at this checkpoint.

Completed field wall times span 1119.304--1333.239 seconds,
including cooperative sharing with the preserved original PBE calculation.
The controlled rule evaluates seven polar/azimuthal combinations for each
of these densities. Exact density repeats can reuse one complete converged
field; a one-ulp density change invalidates that cache. The current private
controller may retain at most 42 rule fields within a single density call;
there is no cache transfer between geometries, functionals or densities.
For this eight-AO fixture measured aggregate RSS is small, but this is not
a validated large-basis memory policy. A bounded read-only observer now
measures one complete field's actual CPU and wall time. Reusing prepared
AO descriptors or removing repeated azimuthal work in an auxiliary collinear
reference are candidates, not accepted performance improvements. The older
PBE process is preserved; CPU use alone does not reveal its SCF iteration
or establish numerical progress because that artifact emits no iteration
log. Neither process is restarted to obtain a more convenient benchmark.

The controlled quadrature has independently checked first derivatives:
96 descriptor/complex-AO component fits include both moving-frame charts,
their equator and zero magnetization. These checks do not supply a second
or higher derivative of the changed finite-rule model. The exact kernel
product and controlled quadrature are still private candidates rather than
executed compatible shared/native integration. Public Rust, native Python,
ASE and canonical CLI parity must be rerun on their eventual exact artifact.

The weakly pinned H4 state remains a separate limitation. Tightened SCF
changes the unrotated laboratory-frame g Gram matrix by 1.24638e-5, although
the principal-g change is 1.34248e-9. Frozen-spin curvature eigenvalues are
approximately 1.15e-11, 1.97e-10 and 2.07e-10 hartree/radian squared.
Diagnostic spin-frame alignment reduces the discrepancy but is not applied
as a correction and does not close laboratory-tensor acceptance. This
frozen-orbital calculation is not a relaxed ZFS or a nuclear derivative.

No finite-difference acceptance threshold has been relaxed in this tranche.
All original failed steps, angular gates and raw results remain. Any later
roundoff-based criterion must state the observable scale, contraction and
conditioning error, stencil amplification by inverse powers of the step,
SCF/response/quadrature uncertainty and truncation error before untouched
cases are evaluated. Machine epsilon alone is not a dimensional derivative
error budget. The nuclear owner may separately improve only the contact and
dipolar contractions and local imports in `spin_hyperfine_tensor`, with
identical-input high-precision evidence; reference selection, operators and
spin normalization are outside that local change.

Remaining gates are PBE physical convergence and same-state g/A, compatible
native/public integration, unrotated tensor conditioning, basis/spatial-grid
convergence and the separately requested high-spin, gauge, relativistic and
nuclear-response combinations. Unsupported errors or source typechecking
do not close those gates. Machine-readable current limits and artifact
identities are retained in `epr-static-current-limits-20261103.json`, with
raw numerical evidence outside the distributable source.

### EPR nuclear derivatives: source-local HF acceptance

For a distorted C1 CH3 doublet with all-electron HF/STO-3G, the source-local
calculation passes all 6,840 canonical component fits: 912 Cartesian
hyperfine-gradient entries and 5,928 Hessian projections. The four nuclear
sites each report contact, all nine ordered dipolar entries and all nine
ordered total entries. The 78 independent directions span the complete
symmetric Hessian over the 12 nuclear coordinates. Exponents are
1.978495--2.093096 and the minimum R-squared is 0.999512679, under the
unchanged strict gates `1.9 < p < 2.1` and `R2 > 0.999`. No floors, missing
component IDs or discarded failed samples are used.

Evidence is
`epr-hyperfine-gradient-oracle-20261102-02/hf/actual-summary.json`.
The test takes 540.329 seconds on one computation thread, with sampled group
peak RSS 87,732 KiB. Sampled group RSS and cumulative child maxRSS are
different measurements. This is a direct Rust source-local provider linked
to the identified accepted core, not an updated installed Python/CLI build.

New source separates moving contact/spin-dipolar operator coefficients,
compensated high/low spin-density contractions, and complete Cartesian
hyperfine-property derivatives. The shared stationary nuclear Taylor solver
supplies moving-basis, metric, SCF and quadrature response. A second metric
transformation is not added. The property surface contains signed contact,
all nine dipolar entries and all nine total entries per selected nucleus.
Its initial contract is nonrelativistic all-electron integer collinear SCF.
The low-level input is a stationary SCF functional. A correlated recipe is
a distinct type, so the public adapter must reject that recipe before
extracting its SCF parent; the parent alone cannot identify a correlated
request. Unsupported reference/particle domains must not be silently
substituted. The HF result does not certify constrained common-orbital,
spinor, relativistic, ECP, periodic or correlated nuclear-response routes.

The proposed common control is strict integer `nuclear_order=0,1,2`, with
default zero preserving the existing static property. Order one computes
only gradients and returns an uncomputed Hessian as absent; order two
assembles gradients and Hessians with explicit Taylor-factorial conversion.
Physical derivative units are MHz/Bohr and MHz/Bohr^2 after applying the
dimensionless isotope factor once. Raw operator and contraction bounds
have order-dependent inverse-length units; they are not dimensionless
physical error estimates.

The first actual source-local compile used
`epr-hyperfine-nuclear-20261102-02/manifest.json` and failed before any SCF.
Its diagnostics identified an invalid PT2-field access on the SCF functional
and an unspecified diagnostic-vector element type. Both have been repaired
in source; the failed attempt and its terminal resource release are retained.
The corrected freeze
`epr-hyperfine-nuclear-20261102-03/manifest.json` was actually compiled with
warnings treated as errors. The remaining failure was test-wrapper
visibility: shared public test helpers were hidden inside a private outer
module and diagnosed as unused. No SCF started. This failed attempt and
its safe terminal resource release are also retained. The prepared fourth
context makes that wrapper public, matching a standalone integration-test
crate, with no warning suppression and no change to any production source,
shared helper, equation, step size or fit threshold. It is
`epr-hyperfine-nuclear-20261102-04/manifest.json`. Its warning-strict compile
passes in 49.569 seconds and its one-SCF operator smoke passes in 1.097
seconds. All static tensor entries agree with the independent native
operator route within the asserted `1e-8 MHz per nuclear g` bound.

The original scalar-second-difference HF test remains a retained **FAIL**:
18 of 6,840 strict fits fail on `[0.04,0.02,0.01,0.005]` Bohr. Tightening
SCF precision alone does not close every failure. Coarser intervals reduce
fine-step cancellation but expose higher-order truncation in other entries;
these experiments and the rejected `1e-14` response-convergence requests
are retained. They are not presented as passes or hidden with an error floor.

The resolved verification uses independently scalar-verified first property
derivatives at freshly displaced SCFs for the Hessian finite differences.
The oracle evaluates only order-one paths; the production Hessian still
uses analytic order-two paths and complete Cartesian polarization. The
Hessian interval is `[0.01,0.005,0.0025,0.00125]` Bohr, extending to a
**finer**, not omitted, smallest step. The 912 first-gradient fits retain
the original scalar oracle and original interval. The kernel equations,
geometry, functional, basis, convergence gates and component coverage are
unchanged. Static operator parity, raw tensor symmetry, translation,
gradient-only/full-order parity and invalid-input checks also pass.

The actual static contacts at this C1 geometry are:

| Site | Contact, MHz per nuclear g |
|---|---:|
| C | 222.6844969756 |
| H1 | -25.9743426635 |
| H2 | -31.0618201391 |
| H3 | -24.7088769233 |

No isotope factor is assumed. The reference energy is
`-39.052305630450746 Hartree`. The geometry is the Bohr geometry in
`tests/p_hyperfine_nuclear.rs`, not an optimized experimental methyl
structure. These minimal-basis values are implementation checks, not
basis-converged predictions or a new experimental agreement claim.

The PBE full-component batch remains running in its analytic-response
stage; no PBE fit acceptance is inferred from HF. Call-level profiling of
one matched PBE path has completed and passed its convergence check:
381.264 seconds wall time on one thread, of which 378.774 seconds is the
shared nuclear response and 1.723 seconds is reference SCF. The first- and
second-order solves report 246 and 183 iterations and residuals
`2.769e-14` and `1.371e-14`. The four moving hyperfine operators together
take about 0.563 seconds, and their trace contractions about 0.0011
seconds. This identifies a shared-response bottleneck, rather than an
SCF or new-operator bottleneck. It is a single-path measurement, not a
general speedup claim or all-component PBE acceptance. The audit is
`epr-hyperfine-pbe-profile-20261102-01/probe/actual-summary.json`.

The owned output implementation now checks coordinate/path/tensor sizes,
reserves the fixed output buffers before assembly, and retains separate
reservations for the actual diagnostic capacities. Reservations have the
lifetime of their owning result, which deliberately has no deep `Clone`.
Three source-local order/overflow/budget-lifetime tests pass, and a warning-
strict source-local compile passes in 57.342 seconds. The permanent
all-component HF test on this output-only change also passes: all 6,840
canonical fits, zero failures, 534.794 seconds, sampled group peak RSS
86,732 KiB, and identical exponent/R-squared extrema to the earlier
accepted source-local test. Its audit is
`epr-hyperfine-output-workspace-20261102-01/hf/actual-summary.json`.
This remains source-local acceptance. Output reservations
are not a total-process RSS cap or a claim that all shared AO/Taylor
temporary allocations are covered. The evidence is
`epr-hyperfine-output-workspace-20261102-01/output-unit-tests/results.json`.

The planned public adapter still requires units, reference/recipe guards,
nonconvergence handling and one audited shared Rust/native-Python/ASE/CLI
artifact. No public A nuclear-Hessian result, vibrational correction,
g/D nuclear derivative, generalized relativistic extension or measured
spectral agreement is established by this HF source-local result. The
earlier EPC17 and independent kaolinite follow-ups remain separate work.

### EPR response timing and bounded optimization experiments

The private immutable-frame orbital-action cache has a warning-strict
compiled-library experiment, not a production-core adoption. Its affected
HF regression passes all 6,840 fits in 501.664 seconds. The next signed
sample record is complete: all 5,928 Hessian sample rows are present, and
an independent reconstruction finds no error/fit disagreement, missing ID
or duplicate. The evidence is
`epr-response-cache-derivatives-20261102-01/hf/independent-signed-review.json`.
The earlier complete HF records and the original 18 scalar-oracle failures
remain unchanged.

For a matched C1 methyl UKS PBE/STO-3G order-two path, all four nuclei and
all 228 signed constant/first/second coefficient entries are compared
against the unchanged baseline library. The maximum difference after
factorial conversion to derivatives is `7.1054e-14`, below the declared
`1e-8` bound. Both one-thread runs use the unpruned level-two grid,
zero grid-weight cutoff, 512 MiB managed memory, and response tolerance
`1e-12`. The cache is cold on entry; reuse is only inside the path. The
observed path-time ratio is 1.153. These are single trials on different
virtual CPU IDs with other independent work present, so this is not a
general speedup or memory-reduction claim. Full PBE four-step acceptance
and public dispatch are still unclosed. Signed values, exact artifact
hashes and timings are in `epr-response-cache-comparison-20261102-01.json`.

A forwarding-only provider timer isolates a major remaining cost without
another library build. All 228 coefficients are bitwise unchanged. Of
457.667 path seconds, provider calls consume 236.558 seconds (51.69%):
ordinary Taylor potentials consume 64.658 seconds and mixed directional
Taylor potentials 170.706 seconds, with ordinary second-order batch
evaluation consuming 1.193 seconds. The rest includes outer geometry/AO
contractions and other shared-response work. Solver iteration count alone
therefore does not identify the full bottleneck. The raw call/order counts
are in `epr-xc-timing-wrapper-20261102-01/provider-timing-audit.json`.

Existing configured parallelism gives a separate, useful resource-scaling
result on the same baseline library: one thread takes 455.818 path seconds
and four threads take 147.030 seconds (observed ratio 3.100). The actual
four-thread pool and four allowed CPU affinities are checked. All 228
signed coefficients and the reference energy are identical; no equation,
interval, grid or convergence gate changes. Total child CPU time increases
from 450.152 to 548.262 seconds, so the shorter wall time uses more CPU
work. Sampled peak group RSS is 334,052 versus 327,404 KiB; these single
samples establish no general memory saving or hard RSS cap. Managed memory
and the external runtime RSS guard remain 512 MiB. This is a one-path
scaling result, not full four-thread PBE acceptance. The evidence is
`epr-pbe-four-thread-benchmark-20261102-01/scaling-comparison.json`.

A separate exact-bit Taylor-constant reuse candidate passes 3,360 scalar
coefficient comparisons and its generation/signed-zero checks, but both
paired microbenchmarks are slower and peak arena capacity is unchanged.
It is rejected for production. Its reduced logical arena use is not
reported as a measured process-memory improvement. The record is
`epr-series-constants-probe-20261102-01/performance-decision.json`.

### Nuclear-A common-report adapter checkpoint

The additive HF nuclear-A report adapter passes its warning-strict
source-local compile and one-SCF C1 methyl test. Selected nuclei retain
their requested order while derivative coordinates include every input
atom. Signed positive, negative and zero nuclear g factors are applied
once; contact, dipolar and total gradient/Hessian fields are checked.
Order-one Hessians remain null. Unsupported reference/model requests and
failed or nonfinite response diagnostics are refused. Existing static
report fields are retained without a second SCF. The maximum raw
translation sums are `4.76064e-13 MHz/g/bohr` for the gradient and
`2.13163e-12 MHz/g/bohr^2` for the Hessian.

An initial nested-unit-label failure is preserved and corrected using
the common root table of dotted paths. On the actual generated HF report,
the unchanged pure ASE conversion functions check 5,928 physical values
and 2,808 raw order-dependent operator diagnostics. Every physical value
matches its independent length-factor conversion exactly, input data are
unchanged, and null Hessians remain null. This tests the report boundary;
it does not execute an installed native or ASE property route. The records
are `epr-nuclear-public-adapter-20261102-03/probe/actual-summary.json`
and `epr-nuclear-public-adapter-20261102-03/ase-conversion-audit.json`.

Independent review subsequently found an order-index defect in the
adapter's response-diagnostic rows: nonconstant response arrays start at
order one, while operator arrays include order zero. The original actual
HF report confirms the shift and is preserved in
`epr-nuclear-public-adapter-20261102-03/diagnostic-order-defect-amendment.json`.
Its physical tensor and operator-unit checks remain separate evidence;
its response-diagnostic order mapping is defective. The additive adapter
uses checked order-minus-one indexing only for the nonconstant arrays,
with null unmeasured zeroth response diagnostics. The permanent exact-value
order-one/order-two regression and the actual HF report rerun pass in
adapter04. All 5,928 physical report values are unchanged; 780 response
diagnostic values move to their correct order, with unchanged operator,
accumulation and coefficient-equation ordering. ASE conversion also passes
for those 780 corrected values. The repair audit is
`epr-nuclear-public-adapter-20261102-04/order-repair-audit.json`.
The nuclear-A equations and running PBE calculation are unchanged.

The adapter and common source hooks are now present in the current
checkout, including control registration and the static-handler call.
This source observation is separate from exact current shared-artifact
Rust/Python/ASE/both-CLI numerical tests, which remain pending coordinated
integration. Public DFT nuclear-A stays
unsupported until its complete numerical acceptance. The existing full
PBE calculation continues independently; no completed PBE component
matrix, g/D nuclear derivative or experimental spectrum acceptance is
inferred from this checkpoint.

The exact review bridge also passes a private whole-crate warning-strict
metadata check in 41.920 seconds, with sampled peak group RSS 1.56 GiB.
It combines the static-owner baseline with the nuclear-A files and narrow
common routing/units hunks, without native code generation or changes to
peer live files. This is typechecking only; rebasing the latest common API
changes and the shared-artifact numerical tests remain required. The
record is `epr-nuclear-common-metadata-20261102-01/state.json`.

A second private composition preserves the separately compiled API library
baseline and the completed static EPR files, excluding newer unaccepted
spin-recovery changes. It passes warning-strict metadata in 38.289 seconds
with peak group RSS 1.56 GiB. Its source manifest is verified; the API
baseline's original CLI timeout remains a failure record and is not
promoted to whole-package acceptance. Latest common-API fixes, including
the early saved-model bypass, still require coordinated rebase before a
shared public numerical artifact. This second typecheck is recorded in
`epr-nuclear-api-composition-20261102-01/state.json`.

The permanent derivative test now prints both sets of raw analytic path
diagnostics before their convergence assertions. Removing those print
statements leaves the original test text unchanged: equations, steps and
fit gates are preserved. Warning-strict compilation and the existing small
output-budget unit pass; no full physical fit suite is repeated for this
logging-only change. The old live PBE executable retains its frozen
logging behavior. The exact supplemental test patch and audit are in
`epr-nuclear-test-diagnostics-20261102-01/logging-only-audit.json`.

### Shared-artifact HF nuclear-A acceptance

The coordinated shared library now passes the complete permanent C1 methyl
HF/STO-3G nuclear-A regression. All 6,840 canonical component IDs and signed
samples are present exactly once, with zero failed fits or signed-error
reconstruction disagreements. The measured exponents span
1.978495238 to 2.093096386; minimum R-squared is 0.999512679. Four step
sizes, the existing strict gates and the independent displaced-gradient
Hessian reference are unchanged. The actual shared-library run takes
548.242 seconds on one computational thread with sampled peak group RSS
84.83 MiB. This timing is a verification measurement, not a general
performance comparison. The Rust library hash begins `2a30243423f2378d`;
its coordinated source manifest begins `38462a09504ca187`.

On the matching native extension, 14 distinct public cases are reconciled:
the native common driver and wrapper, ASE atomic and converted reports,
both CLI names, selected nuclear ordering, positive/negative/zero explicit
nuclear g factors, null first-order Hessians, preservation of static
tensors, malformed input, unsupported nuclear-reference/model requests
and response nonconvergence. The initial run passes 13 cases and fails
one because the test also supplies nuclear-response controls to its
separate static comparison. The API correctly rejects those unused
controls. Removing only those controls from that static call makes the
single failed case pass; the other 13 cases are not repeated. The original
failure remains archived. No production equation, physical criterion or
native artifact is changed for this correction.

The native extension hash begins `c18988e989991d74`. These are actual
language-surface calculations using a private source overlay of the
matching package and both already linked CLI binaries. Clean wheel
installation remains the API owner's separate acceptance. The records
are `epr-nuclear-shared-artifact-20261102-01/independent-hf-audit.json`
and `epr-nuclear-shared-artifact-20261102-01/public-surface-summary.json`.
The permanent public tests are `python/tests/test_hyperfine_nuclear.py`.

This closes the earlier shared-artifact pending status for this collinear,
all-electron integer HF nuclear-A increment. Complete PBE acceptance is
still pending the unchanged original calculation, and public DFT nuclear
A remains unsupported. Neither g/D nuclear derivatives nor arbitrary
reference, relativistic, periodic or experimental-spectrum acceptance is
inferred. No additional full-library compiler is started by this property
verification; the existing coordinated artifact is reused.

Two additional direct Rust common-driver cases pass on the same rlib,
covering the actual molecular report and the typed capability/input
boundary. The 5,928 Cartesian physical report values agree exactly with
the accepted adapter04 report. Maximum translation sums are
`4.77840e-13 MHz/g/bohr` and `2.13163e-12 MHz/g/bohr^2`; the returned
Cartesian Hessian asymmetry is zero. The complete independent fits above
remain the numerical derivative evidence. No full component suite is
repeated for these two interface cases. Their record is
`epr-nuclear-shared-artifact-20261102-01/rust-driver-report-audit.json`.

For this unoptimized C1 methyl fixture, the actual common-driver HF energy
is `-39.052305630450746 hartree`. Derivatives of the carbon contact
coupling per nuclear g with respect to its own Cartesian position are
`[13.7065905811, -91.5291702440, -156.5061459485] MHz/g/bohr`, and its
carbon-x/carbon-x second derivative is `71.5797001446 MHz/g/bohr^2`.
These are isolated-molecule, clamped-nucleus model values in the permanent
test geometry and fixed laboratory axes, without vibrational averaging.
Their numerical acceptance does not establish experimental accuracy of
this minimal basis or an equilibrium-geometry spectrum.

<!-- owner2-epr-followthrough-20261102 -->

### EPR continuation: retained failures and independently checked spin-spin derivatives

The exact shared native artifact `c18988e9` passed all 14 installed-wheel
nuclear-hyperfine public cases, including both matching CLI executables.
The associated HF acceptance retains all 6840 strict derivative fits.
PBE remains unaccepted: the current original-protocol checkpoint has
910/912 passing gradient fits and
5885/5928 passing Hessian fits, with every failed row retained.
All 90 declared directions and their exact 6840 fit/signed IDs are complete,
with no missing or duplicate IDs. The 45 failed views remain failed (2
gradient and 43 Hessian views). The original manager and final child are
terminal. One earlier direction's OS return code was lost during transport;
its complete signed failed data are retained and no exit code is invented.
The four steps are
0.04, 0.02, 0.01 and 0.005 bohr; neither thresholds nor failed steps were removed.
The original monolithic PBE attempt ended on the host-memory safety guard.
Its resource failure is distinct from the recovered calculation's numerical failures.

| EPR quantity | Executed numerical scope | Installed/public status in this checkpoint | Remaining limitation |
| --- | --- | --- | --- |
| A nuclear gradient and Hessian, HF | All 6840 original fits pass | 14 installed native/ASE/CLI cases pass | Integer all-electron collinear reference only |
| A nuclear gradient and Hessian, PBE | 910/912 original gradient and 5885/5928 original Hessian fits pass | Not accepted | 45 original failed views; repaired run stopped at 6004/6840, with 30 failed and 836 uncompleted |
| g nuclear gradient and Hessian, HF Pauli model | 2519/2520 fixed-step fits; all 2520 agree with independent analytic equations; 4368 Cartesian assembly entries checked | Shared-artifact canonical regressions pass; bounded HF/UHF London public adapter and contract tests are provided | One fixed-step failure remains; no DFT/ROKS/GKS derivative inference |
| Actual methyl g | 56 directional fits and 57 physical-origin covariance identities pass | Source-linked molecular validation | Distorted fixed geometry and minimal basis; unmatched experiment |
| Direct spin-spin D nuclear derivatives | All 810 fixed-step fits pass; 387 public values match Rust exactly | 25 installed cases and console pass | HF/UHF integer all-electron S >= 1, no SOC |
| Historical two-electron HF Pauli anisotropy model | 809/810 D and 2424/2430 susceptibility fixed-step fits; independent analytic comparison; H2 Cartesian 243/243 pass | Shared-artifact canonical regressions pass; no public SOC-D driver | Bare one-electron SOC; historical failed curves retained |
| Extended HF SOC nuclear response, paired NH triplet | All 243 Cartesian D fits pass; 25/27 auxiliary scalar fits pass | Experimental source; three canonical shared-library tests pass (the source-leaf count of four includes an internal unit); no public SOC-D route | No two-electron SOC, spin purification, semilocal/ROKS/GKS or public-driver inference |
| Extended HF SOC nuclear response, quartet fixture | 808/810 Cartesian D fits and 90/90 scalar fits pass; all 810 D derivatives agree with independent 70-digit equations | Experimental source; no public driver | Two rate-fit failures retained; not universal multiplicity acceptance |
| Total D with general SOC and two-electron SOC | Not established by the preceding component tests | Not accepted | Full reference-specific spin-manifold and moving-operator response |
| ROKS/GKS or periodic A/g/D nuclear derivatives | Not validated here | No support inferred from static EPR extensions | Separate dedicated response and physical tests required |
| Vibrationally averaged EPR tensors | Not established here | Not accepted | Vibrational EPR contraction and matched state/temperature data |

The new direct spin-spin D implementation differentiates moving Coulomb
pair operators and the self-consistent density. It uses the convention
H_D = S.D.S and includes direct minus exchange, without trace projection.
The C1 four-center triplet-dication Gaussian benchmark passed all 810
independent gradient/Hessian fits against a separately implemented
70-digit scalar reference. The original double-precision scalar-reference
run's 151 failed Hessian fits remain recorded. Only reference arithmetic
changed: analytic values, geometry, four fixed steps
(0.008, 0.004, 0.002, 0.001 bohr) and criteria were identical.
The assembled Cartesian implementation also passed all 810 fits:
p = 1.982730496 to 2.008014490,
minimum R squared = 0.999979328.
Translation residuals were 3.388e-21
and 2.202e-20 for gradients and Hessians.
The independent 810 fits are source-linked finite-basis validation.
The later exact native artifact `5925210a` also passed all 25 installed-wheel
public cases and both matching CLI executables, recorded in
`spin-spin-zfs-installed-20261103-01/results.json`. That public acceptance is
also backed by exact parity of all 387 returned D/gradient/Hessian values
with the accepted Rust-handler report (maximum difference zero). The
installed console entry point passed without an external private Rust binary.
The capability is
restricted to integer all-electron HF/UHF, S >= 1, and direct electron
spin-spin D. SOC, spin purification and other references are excluded.

A real triplet NH calculation at a chosen bond length of 2 bohr, with STO-3G,
gave axial spin-spin D = 2.044215614239 cm^-1 (HF) and
1.960945208434 cm^-1 (PBE, unpruned grid level 2).
The finite-element PBE value 1.861 cm^-1 in
[Ghosh et al., Table III](https://arxiv.org/html/1902.07377v1)
differs by 9.85% and 5.37%, respectively. The geometry and basis are unmatched;
this comparison tests plausibility and does not isolate implementation error.
SOC and vibrational averaging are excluded from these calculated values.

A genuine intermediate-product underflow/overflow defect in the new D
contraction was reproduced, fixed with a bounded multiplication order and
upward accumulation bounds, and checked by exact power-of-two regressions.
All 810 physical fits remain passing. Selecting the necessary Taylor width
preserves normal physical coefficients bit for bit and speeds the isolated
static contraction kernel by about 2.6 times in interleaved measurements;
this is not a whole-SCF speedup claim.

Moving SOC/mixed operators retain three failed gradient fits out of 7344
nonzero independent entries. Their 216 analytically specified AO-selection
zero checks are separate. Two independently nested AD implementations agree
on 171990 coefficients through second order (maximum difference 6.94e-16).
The retained failed curves are explained by opposing h^2 and h^4 terms;
5880 diagnostic coefficients through seventh order independently agree to
2.78e-16. These diagnostics do not change the original failed-fit verdicts.
The coordinated library `b895d970` executed all 2520 Pauli-g directional
gradient/Hessian fits. The original double-precision scalar reference had
eight failures; an independently implemented 70-digit Gaussian reference,
with identical analytic values and four steps, retains one failure out of
2520. For that component, a separate analytic bordered-eigenpair reference
agrees with the implementation to 1.80e-20. Its fourth-, sixth- and
eighth-order coefficients reproduce the failed finite-step curve: opposing
h^2 and h^4 truncation terms explain the non-asymptotic slope. This is a
diagnosis, not a changed acceptance interval; the failed verdict remains.
All 2520 gradient/Hessian components were also compared directly with the
independent 70-digit analytic equations; their maximum absolute difference
was 1.33e-19. This additional equation-level proof does not relabel the
original four-step convergence-rate failure as a pass.
The Rust context now assembles complete Cartesian outputs in coordinate
order 3*atom+axis, including the explicit factor of two for second Taylor
coefficients. All 336 gradient and 4032 Hessian scalar entries of total g
and its retained contributions agree with an independently transformed
70-digit analytic oracle to 2.54e-20 and 1.56e-19, respectively. Exact-source
canonical leaf tests passed order selection, tensor symmetry, translation
and typed-error checks. This is assembly/equation-level acceptance on the
finite-basis fixture; it does not close the retained directional fit failure
or establish installed public nuclear-g acceptance.
The same artifact separately passed the independent magnetic-density,
static-term and London-origin tests. Complete g nuclear acceptance and
installed public SOC-D validation remain open.
For the actual C1 methyl radical, UHF/STO-3G independently passed all
56 directional g-gradient/Hessian component fits at the same four steps.
The determinant has S squared = 0.7658866164, without spin projection.
The original combined test nevertheless failed: it incorrectly required
GC and SO intermediate contributions to be separately origin invariant.
Independent static calculations show equal and opposite partial changes
as large as 1.272e-5, while the physical total g changes by only 4.03e-17.
The first-origin derivative data and failed test are retained. The missing
second-origin nuclear witness completed: all 57 mass-invariance, GC/SO
covariance-cancellation and physical-total-g identities passed. The maximum
residual through second nuclear order was 9.35e-16.
An exact orbital time-reversal reconstruction now avoids three of the six
mixed-field solves. On this same eight-AO UHF/STO-3G methyl fixture, CPU 5,
one thread and a 512 MiB managed workspace, the nuclear path took
222.504 seconds before and 116.835 seconds after; context plus path took
334.118 and 231.172 seconds. This is a single matched molecular timing,
giving warm/cold ratios of 1.904/1.445, not a universal speedup guarantee.
All 84 scalar coefficients agree to 3.42e-18; the maximum physical
second-derivative difference is 6.83e-18. Peak sampled RSS was 12,537,856
versus 12,767,232 bytes, so no memory reduction is claimed. The separate
78-direction regression compared all 2520 derivatives and 59904 density
parity coefficients. Original failed convergence-rate verdicts remain.
Strict matching-toolchain Clippy and warning-strict compilation passed;
all 13104 scalar coefficients in the 78-direction paired regression remain
bit identical after the iterator-only cleanup and error-contract repair.
The optimization is in the reviewed Rust source, opt-in, and not yet in an installed
public nuclear-g route; its compatibility default retains both explicit paths.
The use of distributed gauge origins for electronic g is motivated by
[Glasbrenner et al., JCP 148, 214101](https://doi.org/10.1063/1.5028454);
the partial-term cancellation quoted here is from the actual recorded
calculations, not an experimental or literature-matched accuracy claim.
The computed physical g principal values (singular values, rather than raw
diagonal entries) are 2.0030010010, 2.0029552770 and 2.0022388679 for this
distorted structure. Their spread is 7.6213e-4.
[Popov et al., JCP 126, 134504](https://doi.org/10.1063/1.2715589)
report a g anisotropy of 2.5e-5 for methyl in solid argon, measured over
14--35 K with rotor-state populations and an axial low-temperature spectrum.
The accessible author abstract does not supply a matched isolated-molecule
tensor at this geometry. The phase, structure, basis, missing two-electron
SOC and rotational/vibrational averaging differ. These numbers therefore
do not establish experimental agreement or isolate an implementation error;
no per-spectrum scale or shift has been applied.
An additional source-local SOC-D prototype now differentiates the
constrained HF orbital-response equations analytically through second
nuclear order, including moving Gaussian integrals, the stationary density,
overlap metric, occupied projector transport and transverse-spin constraints.
Its initial domain is an all-electron two-electron HF triplet. It excludes
two-electron SOC, other references and general spin projection; it is not a
registered or installed public SOC-D feature.
All 78 nuclear directions executed. Independent fixed-step tests passed
809/810 D fits and 2424/2430 nonzero susceptibility fits; all seven failures
remain. A separate 70-digit analytic implementation uses orbital-exponential
HF stationarity and reduced spin-channel KKT equations: all 810 D and 2430
nonzero susceptibility derivatives agree to maximum absolute differences
1.83e-23 Hartree/bohr^order and 1.86e-13 in unscaled response units.
Its higher coefficients independently explain the retained failed curves.
There are also 4860 exact spin-selection zero checks.
Common translation and occupied/virtual subspace rotations leave the
prototype coefficients invariant to 1.39e-27 and 3.64e-27 Hartree/bohr^order.
For the actual triplet electronic state of H2 at the specified unrelaxed
geometry, HF/6-31G** passed all 18 directional D-gradient/Hessian tensor fits
at the same four steps. Its static value agrees with the independent full
spinor constrained-response path to 1.04e-25 Hartree. This triplet-state
test is not an equilibrium ground-state H2 or measured EPR-spectrum comparison.
The same constrained HF model also completed all Cartesian coordinates for
this triplet H2 case: all 243 fixed-four-step tensor-gradient/Hessian fits
passed against 288 independent displaced static calculations, with no step
selection. This does not establish a general spin-projected SOC ZFS theory.
The complete SOC plus direct-spin-spin D and broader molecular acceptance
remain unfinished.
A bounded diagnostic repeated the original PBE failed Hessian direction 15
at all four unchanged steps and both signs. All nine saved SCF energies
matched exactly. Among 41400 grid points, a small number crossed the fixed
1e-13 density branches. This establishes branch crossings, but does not
isolate their contribution to the failed hyperfine derivatives; no cutoff
or formula was changed, and no failed fit was promoted.
For the same failed direction, nine order-zero scalar calculations also
compared ordinary references with stationary-frame-polished references at
the original four steps. Static values changed by at most 3.58e-10 MHz per
nuclear g, large enough to affect a second difference at the smallest step.
Nevertheless, all four original failed views (entries 0, 10, 14 and 18)
remained failed in both modes. The ordinary replay agrees with saved
finite-difference values to 1.14e-9 MHz per nuclear g per bohr squared.
This 792.89-second, four-thread diagnostic used no nuclear derivative in
its scalar reference; it neither proves a sole cause nor repairs the
remaining PBE validation. No alternative interval was selected.
A separate corrected-artifact diagnostic tightened only SCF tolerances
from (1e-14, 1e-13, 1e-12) to (1e-16, 1e-15, 1e-14), retaining all nine
references, the same saved analytic vector and four original steps.
All SCFs converged, but 72/76 fits passed instead of 73/76: entries 0, 10,
14 and 18 failed. This does not support a remedy based solely on tighter
SCF thresholds. It does not overwrite the ongoing full-run verdicts.
Another bounded diagnostic captured the same nine H15 references and
checked their stored densities, MO coefficients, overlap and operators
using 100-digit arithmetic. Stored-density contractions and exact
MO-density contractions retained all three carbon-nucleus failed entries
(0, 10, 18). A metric-normalized occupied projector
C_occ*(C_occ^T*S*C_occ)^(-1)*C_occ^T reduced this diagnostic to one failed
entry (10), whose R squared was still only 0.983728. The maximum stored
density versus exact MO-product difference was 7.78e-16, occupied-metric
defect 2.69e-15 and commutator 6.28e-15 Hartree. Normalization changed
static couplings by at most 4.05e-11 MHz per nuclear g, enough to affect
the smallest second difference. These results show a precision effect,
not a gross stale density/MO mismatch. No mode was selected for acceptance:
the calculation still uses f64 operators and converged subspaces, so it is
not an independent high-precision physical DFT solution. The original
failed results and all four steps remain unchanged.
The subsequent local consistency investigation confirmed a separate
low-density numerical-loss defect in the PBE correlation expression.
For c = gamma phi^3, r = expm1(-epsilon_c/c) and q = beta t^2/(gamma r),
combining epsilon_c with its gradient correction gives the exact stable
form -c log1p(r / [1 + (1+r) q (1+q)]). The shared functional owner applied
this rearrangement with a reciprocal large-q form, an anchored expm1
evaluation, and log1p in the PW92 local base. Parameters and density
cutoffs were preserved.
An independent 80-digit value/gradient/Hessian oracle demonstrated the
defect. The actual repaired live source passed all 80 fixed-step local
descriptor checks and all 27 first/second helper-partial checks; its
maximum relative difference from the oracle was 2.81e-15. Runtime
exponential coefficients through order 31 were also checked independently.
The original unweighted tail discrepancies do not establish a molecular
error: with actual AO/grid weights, the measured potential difference was
only 8.22e-18 Hartree. The shared functional owner separately completed
120 fixed-four-step point-provider fits and 912 normal, polarized,
uniform and zero-density provider-value checks. These are local provider
regressions. A coherent rebuilt artifact and the original molecular
protocol must still be executed.
An additional directional diagnostic checked 84 Taylor coefficients through
order six at the two observed tail points along five coordinate directions
and one mixed direction. They agree with independent 100-digit original-form
arithmetic to a maximum relative difference of 7.32e-14. All 24 third/fourth
directional derivative checks passed at the unchanged four steps. These are
not complete high-rank mixed-tensor or molecular acceptance tests.
The repaired source does not relabel the 45 original A failures as passes.
The corrected shared library `1c58c2a1` was then built successfully and the
unchanged original molecular test was relinked. Initial directions gradient
3 and Hessian 15 passed 147/152 original-protocol component fits, versus
146/152 previously. One prior failure passed and five remain; no previously
passing component in this subset regressed. The maximum analytic change was
1.71e-13 in the corresponding derivative units. The shared artifact contains
other reviewed changes, so this comparison does not isolate causal effects
of one source edit. That same-protocol regression was deliberately stopped for the v0.3.0
API/distribution milestone. Its frozen checkpoint contains 79/90 complete
directions and 6004/6840 completed fits: 5974 pass, 30 fail, and 836 fit IDs
remain uncompleted. A partial Hessian direction was interrupted; this is
not a complete scientific pass or an SCF failure. Fifteen of the original
45 failed views are resolved in the completed subset, with no new failed
view there. The original 6795/6840 result and all 45 failures are retained
separately. All owned scientific processes are terminal; no batch is still
running for this release.

#### Displaced first response check of hyperfine H15

The continuation uses the same C1 methyl radical, PBE/UKS, STO-3G, level-2
unpruned grid and zero weight cutoff. The reference energy is
-39.25343704885103 Hartree. The SCF tolerances for energy, density and
orbital convergence are 1e-14, 1e-13 and 1e-12. The response tolerance is
1e-12. On the original steps [0.01, 0.005, 0.0025, 0.00125] Bohr, the
oracle differences fresh order-one hyperfine responses at each pair of
signed displacements. The analytic second-order path supplies the
comparison. The original oracle uses a second difference of static
hyperfine values, which amplifies sample error as
1/h². The displaced first-response oracle amplifies its sample error as
1/h.

The carbon Fermi contact curvature along H15 is 14.388987989100693 MHz
per nuclear g per Bohr². Its four raw errors are 5.846570e-5,
1.466147e-5, 3.667839e-6 and 9.166812e-7 in the same units. All 76
ordered views pass on this direction, with p=1.998083 to 2.020372,
minimum R²=0.99994635 and maximum Richardson error 8.207734e-10.
The analytic first- and second-order response residuals are 2.540858e-14
and 1.872535e-14. The scalar-oracle failures and the broader matrix
counts remain recorded above. Complete acceptance requires the 78
independent Hessian directions and all 12 Cartesian gradient directions.

#### Additional SOC and g checks

The same shared library also passed both canonical g and constrained-SOC
nuclear tests, including Cartesian shapes, Hessian symmetry, translation
and explicit unsupported-domain checks. Their one-thread runtimes were
12.03 and 1.01 seconds. This closes the source-leaf-to-shared-library link
gap; it neither repeats the independent component-fit suites nor changes
their retained failed curves or public experimental capability limits.
The experimental bare-one-electron HF SOC implementation was subsequently
extended to separate occupied alpha/beta spaces. It returns nominal spin,
the actual reference spin-square, and the spin-Hamiltonian normalization.
It retains the 3..12 AO bound and explicit exclusion of semilocal,
ROKS/GKS, ECP, two-electron SOC and environmental response. The original
two-electron directional coefficients remain bit-identical.
For actual NH at a fixed 2-bohr bond along (1,2,4), HF/STO-3G has five
alpha and three beta electrons and reference spin-square 2.0130100283.
All 243 Cartesian D gradient/Hessian fits pass against 289 independently
solved static HF energy-curvature problems at the original four steps.
The independent reference shares zeroth-order Gaussian integrals and SCF
inputs but imports no production SOC or nuclear-response implementation.
Its static tensor differs by at most 6.78e-21 Hartree, and arbitrary complex
occupied/virtual subspace rotations change it by at most 1.22e-21 Hartree.
The auxiliary orientation-independent energy retains two failed rate fits
out of 27; these were not relabeled as passes.
The three-electron quartet fixture separately passes 808/810 Cartesian
tensor fits and all 90 scalar fits against 1153 static references. All 810
tensor derivatives also agree with an independent 70-digit Gaussian/HF
Taylor calculation to 1.48e-26 Hartree/bohr^order. The two retained mixed
Hessian failures arise in a curve with a very small h-squared error term and
an opposing h-fourth term; the independent high-precision reference itself
has fitted exponent 4.557 at the same four steps and fails the original
approximately-second-order criterion. No interval or verdict was changed.
The extended source passes configured Rust 1.85 Clippy and four permanent
source-linked regressions, including spin coherent-state normalization,
paired-spin NH translation and unsupported doublet handling. All 387 NH
assembled tensor values are unchanged after formatting. The exact extension
has now passed all three canonical integration tests on cumulative library
`ae3b6080...` / native `1c1f316e...`; all 387 NH values remain bit identical.
The source-local count of four includes an additional internal unit test,
which is distinct from those three integration tests. The initial saved
integration finalizer incorrectly expected four; saved-log reconciliation
verified the exact three-test source/name set without rerunning or removing
any scientific test. There is still no public SOC-D driver or general,
experimentally certified ZFS support. The spin normalization is
consistent with the spin-coherent relation in Eq. 6 and response channels
in Eqs. 22, 37 and 43 of the
[author preprint](https://doi.org/10.26434/chemrxiv-2023-2kh59-v2).
The same cumulative library also passed the canonical complete-Cartesian
Pauli-g regression. This is actual linked-library execution, while the
remaining original fixed-step g failure and missing public nuclear-g
route remain separate, unchanged limitations.
A separately frozen complementary policy now checks only the retained
g failure `(order=2, direction=62, entry=23)`. Its original rate verdict
is still failed. The independent 100-digit implicit derivative differs
from production by 1.79734e-20 per bohr squared. The predetermined
engineering precision allowance is gamma_m*kappa times the derivative
and natural Pauli scale, with m=2*n_AO^3*(order+1)^2, binary64 epsilon,
ell=(2*0.7)^(-1/2) bohr and independently computed maximum border/metric
condition 11.627754: the resulting allowance is 2.26169e-16. It is not a
certified whole-program bound or a replacement for the original rate gate.
Full independent 100-digit scalar central differences, including all
higher nuclear orders at every original step, predict the signed errors
with residuals 1.79..1.81e-20. The corresponding 70-to-100-digit static
drift is 3.87e-68. This case therefore passes the complementary formula
precision check while retaining its failed near-quadratic-rate result.
The policy acknowledges earlier exploratory comparisons and was fixed
before this new complementary evaluation. No other component, PBE result,
reference family or public route is promoted by it. The added permanent
regression and existing Cartesian g regression both passed on the actual
cumulative shared library (14.87 seconds), with configured MSRV Clippy.
An independent private buffer-reuse microbenchmark found exact parity of
72 Taylor/potential coefficients and passed three scope/panic/stale-handle
regressions. Its median gain was only 1.046..1.058 for repeated local calls,
with a 256-KiB per-thread retention cap. No full molecular speedup or shared
managed-budget integration was established, so this candidate was not
adopted in production.
A separate stack-scratch candidate preserved all 144 tested Taylor/potential
coefficients bit for bit and passed both primitive boundary/error tests.
Across five alternating paired microbenchmark rounds, its median ratios
were 0.962..1.057 depending on order and input. This inconsistent local
gain did not justify adoption; production storage remains unchanged.
The earlier rejected multiple-interval acceptance proposal was not executed
or used for any result above. No release ZIP is claimed ready.

<!-- /owner2-epr-followthrough-20261102 -->
