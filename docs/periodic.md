<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 ss0832 -->

# Periodic and dimensional calculations

Conventions, models, public interfaces and acceptance of the 1D/2D/3D
periodic layer, in the order the layers build on each other: boundary
electrostatics, the masked Gaussian basis and quadrature, Coulomb
integrals and contracted operators, semilocal XC operators, the
self-consistent dimensional reference, analytic electromagnetic vertices
and the screened field response. The finite-k 3D SCF, forces and the
finite-q density/dielectric/thermal responses are described in
[THEORY.md](THEORY.md) and [response.md](response.md); every public entry
point is listed in the three tables of [rust-api.md#common-calculation-driver](rust-api.md#common-calculation-driver). The chronological
acceptance checkpoints of this layer are kept in the
[acceptance record](response.md#acceptance-record).

## Contents

- [1. Boundary conditions and electrostatics](#1-boundary-conditions-and-electrostatics)
- [2. Electronic basis and integration grid](#2-electronic-basis-and-integration-grid)
- [3. Gaussian Coulomb integrals](#3-gaussian-coulomb-integrals)
- [4. Contracted Coulomb operators](#4-contracted-coulomb-operators)
- [5. Semilocal XC operators](#5-semilocal-xc-operators)
- [6. Self-consistent dimensional DFT](#6-self-consistent-dimensional-dft)
- [7. Electromagnetic Fourier vertices](#7-electromagnetic-fourier-vertices)
- [8. Screened electric and magnetic fields](#8-screened-electric-and-magnetic-fields)
- [9. Range-separated screened Hartree](#9-range-separated-screened-hartree)
- [10. Finite-k energy, forces and electronic structure on the driver](#10-finite-k-energy-forces-and-electronic-structure-on-the-driver)
- [11. Analytic stress and cell relaxation](#11-analytic-stress-and-cell-relaxation)
- [12. Gamma-point force constants and phonons](#12-gamma-point-force-constants-and-phonons)
- [13. Elastic constants](#13-elastic-constants)
- [14. Phonon dispersion and lattice thermodynamics](#14-phonon-dispersion-and-lattice-thermodynamics)
- [15. Equation of state and quasi-harmonic thermodynamics](#15-equation-of-state-and-quasi-harmonic-thermodynamics)
- [16. Dielectric tensor and Born effective charges](#16-dielectric-tensor-and-born-effective-charges)
- [17. Continuum rock physics and material cards](#17-continuum-rock-physics-and-material-cards)
- [18. Fermi-Dirac (Mermin) response](#18-fermi-dirac-mermin-response)
- [19. Electron-phonon coupling and superconducting Tc](#19-electron-phonon-coupling-and-superconducting-tc)
- [20. Periodic DFT-D3 dispersion](#20-periodic-dft-d3-dispersion)
- [21. Band velocities, effective masses and Boltzmann transport](#21-band-velocities-effective-masses-and-boltzmann-transport)
- [22. Ordinary weak-field Hall transport](#22-ordinary-weak-field-hall-transport)

## 1. Boundary conditions and electrostatics

*Dimensional periodic geometry and electrostatics.*

`periodic::PeriodicGeometry` represents a periodic subspace inside Cartesian
three-dimensional space. The mask selects lattice vectors, not Cartesian
components: wires/slabs may be tilted and slabs may be skew. Subspace duals
satisfy `a_i . b_j = 2*pi*delta_ij` on periodic axes; open entries are zero.
The measure is length, area or volume in `Bohr^dimension`.

Translations, reciprocal vectors, wrapping, exact closest images and k meshes
use only periodic directions. Wrapping preserves the open projection. Open
lattice vectors may be zero; changing their magnitude does not change the
electrostatic answer. Open k-mesh dimensions must be one. Dependent periodic
vectors and excessive direct-sum enumeration requests return explicit errors.

`periodic::electrostatics::DimensionalCoulomb` combines screened real-space
lattice sums with Fourier series in the periodic subspace and Fourier
integrals in the open subspace. The 2D zero mode retains its transverse
dependence; the 1D zero mode retains its logarithmic dependence. Neither
uses a 3D `4*pi/G^2` substitute. The uniform derivation of dimensional
Ewald sums is described in
[Tornberg's primary paper](https://arxiv.org/abs/1404.3534).

Normalized Gaussian sources have density
`q/(pi*w^2)^(3/2)*exp(-|r-R|^2/w^2)`. Pair widths are
`sqrt(w_i^2+w_j^2)` and bare pair interactions are `erf(r/w_pair)/r`.
Zero width denotes a point with its bare self energy excluded. Gaussian
physical self energies are finite and included. Real-space differences use
complementary error functions to avoid cancellation of saturated erfs.
Forces differentiate the real and Fourier operators analytically.

`periodic_electrostatics` accepts prescribed charges, not an electronic
density inferred from atomic numbers. It returns energy/four components,
analytic forces, charge-averaged site potentials, countercharge force and
potential, charge bookkeeping, potential convention, Ewald controls,
and independently measured quadrature defects. This calculation does not
replace an electronic SCF calculation.

| Compensation | Dimension | Convention |
|---|---|---|
| `neutral` | 1D, 2D, 3D | Supplied charges sum to zero; open directions isolated |
| `uniform` | 3D | Uniform compensating density; zero-average potential/tin-foil boundary |
| `gaussian` | 1D, 2D, 3D | Countercharge `-sum(q)` with explicit Bohr center/width; only periodic directions replicated |

The Gaussian countercharge is a physical external density. Its center is
fixed during source differentiation; its reported reaction restores total
force balance. Changing its width/center changes the model. Translate the
countercharge with the sources when testing rigid translation. Replicate
the actual countercharges when comparing supercells. Replacing many by one
wider Gaussian changes the charge distribution. An infinite uniform density
in open space is not accepted as a wire/slab compensation model.

The 3D zero-mode energy is
`-pi*Q^2/(2*alpha^2*V)+pi*Q*sum(q_i*w_i^2)/V`. The Gaussian width term
restores the removed real-space mean and cancels for neutral cells.
[The independent charged Gaussian Fourier gate](#3-gaussian-coulomb-integrals)
documents the correction and preserves the original failed measurement.
No Makov-Payne or other
isolated-defect correction is added by this driver. Source and countercharge
use the same isolated-slab zero mode or logarithmic wire gauge at one Bohr.
Complete neutralized energies are invariant to a constant potential shift.
A Gaussian site potential is averaged against that site's density; including
the countercharge, `E=0.5*sum(q_i*phi_i)` holds exactly.

Fourier integrals use the existing Gauss-Legendre generator. The driver
evaluates rules of order `n` and `2*n`, returns the latter and reports their
energy/force differences. Defaults are `n=96` and absolute tolerance `1e-11`
in atomic units. Real/reciprocal cutoff convergence remains an independent
requirement; `converged` alone does not certify lattice-sum truncation.
Automatic real cutoffs account for the largest source width and screening
length. Each pair uses its actual distance cutoff and complete skew-lattice
translation bounds. Changing the Ewald split changes the work partition.

All four entry points appear in [three aligned API tables](rust-api.md#common-calculation-driver). CLI
XYZ/cell input uses Angstrom; widths, countercharge center/width and real
cutoff are explicit Bohr inputs. Native Python uses atomic units. The ASE
getter converts Atoms geometry/cell from Angstrom and returns the atomic-unit
report without altering ordinary self-consistent energies/forces.

Dimensional source kernels, geometry, [Gaussian basis operators and the
open-direction XC grid](#2-electronic-basis-and-integration-grid) are implemented. The
[dimensional Gaussian primitive nuclear/Hartree integrals](#3-gaussian-coulomb-integrals)
now share the same Green operator and analytic derivatives. The
[contracted AO/Bloch nuclear/Hartree driver](#4-contracted-coulomb-operators)
adds finite-k complex density contraction, fixed-density centre derivatives,
and source/electronic/Hartree component energies in this same gauge.
The [dimensional semilocal XC driver](#5-semilocal-xc-operators) also
connects the open-direction grid to spin/Bloch functional potentials.
Complete self-consistent SCF and finite-q DFPT connections remain in the original
program. `periodic_capabilities()`
distinguishes `dimensional_electrostatics=True` and `partial_geometry=True`
and `dimensional_coulomb_operators=True` from `partial_boundaries=False`
for the complete electronic SCF driver.
Partial-PBC SCF requests still identify missing electronic connections.
This foundation is not completion of the full PBC/DFPT scope.

### Independent acceptance

`p50_dimensional_boundaries` checks all seven nonempty masks, subspace duals,
skew images against brute force, exact closest images, open-size independence,
the closed alternating-wire energy `-4*ln(2)/L`, existing 3D Ewald energy/forces
with uniform background, point/Gaussian energies against an unsplit direct
real-space oracle, site-potential contraction, translation, rotation and
physical supercell replication. Different Ewald splits are compared.

Independent energy central differences validate analytic forces at Bohr
steps `0.02,0.01,0.005,0.0025`. All steps are retained without error floors.
All six fits pass `abs(p-2)<0.02`, log-space `R_squared>0.99999` and
Richardson agreement within `1e-9 Hartree/Bohr`:

| Dimension | Charge model | p | R_squared |
|---|---|---|---|
| 1D | Neutral | 1.999987587810 | 0.999999999990 |
| 1D | Gaussian countercharge | 1.999971708466 | 0.999999999946 |
| 2D | Neutral | 1.999987668416 | 0.999999999990 |
| 2D | Gaussian countercharge | 1.999972442928 | 0.999999999948 |
| 3D | Neutral | 1.999987668008 | 0.999999999990 |
| 3D | Gaussian countercharge | 1.999972523667 | 0.999999999948 |

The Python/ASE/CLI suite compares charged reports across all three dimensions
and repeats four-step force fits through native Python.

## 2. Electronic basis and integration grid

*Dimensional electronic basis and integration grid.*

`periodic::periodic_basis`, CLI `periodic-basis`, native `periodic_basis`
and ASE `DftRs.get_periodic_basis` use the same driver. It builds Gaussian
Bloch overlap and kinetic matrices, analytic fixed-lattice atomic derivatives,
and optional atom-centred quadrature. The operator is independent of charge,
occupations and functional class. This is an electronic basis foundation;
self-consistent reduced-dimensional nuclear/Hartree and DFPT connections
remain in the original program.

### Geometry and conventions

`PeriodicGeometry` selects any subset of the lattice rows. Only those rows
create images and reciprocal vectors. Open rows may be zero and never define
a vacuum thickness. A tilted wire or slab uses the dual of its actual
periodic subspace. The basis driver also supports the isolated zero-image
limit. Every open mesh axis must have one point.

For Gaussian AOs, the positive-phase convention is

```text
chi_mu^k(r) = sum_T exp(i k.T) chi_mu(r-T)
O_mu,nu(k) = sum_T exp(i k.T) <chi_mu(0)|O|chi_nu(T)>
```

Overlap `S(k)` and kinetic `T(k)` are returned as separate real and imaginary
matrices, with unshifted k points in `[0,1)` fractional order. Hermiticity is
measured on the actual matrices and derivatives; matrices are not repaired
or symmetrized. The minimum overlap eigenvalue is measured with the existing
real embedding. A non-positive value produces a warning, not a denominator
floor or a replacement basis.

The image cutoff is applied to each actual shell-pair centre separation.
Enumeration uses exact subspace-dual bounds and a local integer origin,
so moving an atom by many lattice vectors does not enlarge the search.
At fixed cell, each atomic derivative includes both the home AO and every
translated copy of the same physical atom. Translation sums are measured
from these raw derivatives.

All existing spherical Gaussian angular momenta and contracted basis sets
reuse the molecular overlap/kinetic engines. This driver does not supply an
ECP potential or exchange-correlation energy merely because such a basis
is loaded. The basis and quadrature are usable with arbitrary functional
providers when the corresponding electronic operators are connected.

### Open-direction quadrature

`grid=Some(GridOptions)` in Rust or `grid_level=0..9` in Python enables the
grid. The CLI uses `--basis-grid --grid-level N`. It integrates one periodic
cell and the open Cartesian directions using the existing radial and
angular quadratures, with a local Becke denominator over the actual masked
atom images. Weights always carry the physical three-dimensional measure
`Bohr^3`, including for a 1D wire or 2D slab; the geometry's cell measure is
separately length, area or volume in `Bohr^dimension`.

`grid_cutoff` (`--basis-grid-cutoff`) controls the radial extent and Becke
image neighbourhood in Bohr. It is independent of the AO `basis_cutoff`
(`--overlap-cutoff`), whose default here is 12 Bohr. Both cutoffs and grid
level require independent convergence. They are not Coulomb cutoffs.
The quadrature overlap defect compares the actual integral of
`conj(chi_mu^k) chi_nu^k` against the analytic Bloch overlap and includes
all three truncation/quadrature errors.

`grid_derivatives=True` (`--basis-grid-derivatives`) additionally returns
atomic derivatives of the partition weights, including movement of the
grid point with its owner. The point derivative is the negative sum of
the fixed-point atom derivatives. Thus a simultaneous translation leaves
every returned moving weight unchanged. The radial/angular raw weight
does not change under fixed-cell atom translations.

Finite image and weight selections are held fixed when differentiating.
Derivative validation resolves an interval away from changing image
membership; cutoff convergence is a separate requirement. No finite
differences build the production matrices or derivatives.

### Switched local Becke partition

The periodic Becke weight of a grid point `p` is evaluated over the image
centres `c_i` within the physical radius `r_cut = (shell + 1.5) (V/N)^(1/3)`
of `p` (`V/N` is the volume per atom, so a supercell selects the same
neighbourhood). Cutting that list hard at `r_cut` makes the partition of
unity, and with it every XC quantity, jump whenever an image centre crosses
the sphere as atoms or the cell move: a centre entering changes the
denominator `sum P_i` and also the factors `s(nu_ij)` of the cell functions of
all other centres. Every centre therefore carries a smooth switching function
of its distance `r = |p - c|` to the point,

```text
g(r) = 1 - s(t),  t = (r - r_on)/(r_cut - r_on),  s(t) = e^(-1/t) / (e^(-1/t) + e^(-1/(1-t)))
```

with `g = 1` for `r <= r_on` and `g = 0` for `r >= r_cut`, a C-infinity step
(all derivatives vanish at both ends, which later third-order strain
derivatives need). The switching zone is one length per atom wide,
`r_on = r_cut - (V/N)^(1/3)`, so `r_on >= 1.5 (V/N)^(1/3)`; `g` depends only
on physical distances, so size extensivity is unchanged. The cell function of
centre `b` is

```text
P_b = g_b prod_(c != b) [1 - g_c (1 - s(nu_bc))],   w_b = P_b / sum_i P_i
```

where `s(nu)` is Becke's three-fold iterated polynomial with the size
adjustment of the two radii. A centre that is switched out (`g_c = 0`) leaves
the cell functions of all others untouched, and the weights still sum to one
exactly (the normalization is `sum_i P_i`, so any positive `P_i` gives a
partition of unity). The literal form `g_b P_b / sum g_i P_i` with the
unswitched Becke factors `s(nu_bc)` for every listed competitor is not
smooth: a competitor entering at `r_cut` still multiplies the cell functions
of the others by `s(nu_ic) = 1 - O(eps^8)` with differing `eps`, and the
smoothness fit below fails for it (p = -1.2, R2 = 0.37 for shell 1 under shear).
The radial points beyond `r_cut` carry exactly zero weight and are skipped.
Dimensional grids switch the outer 30 % of their `grid_cutoff` the same way.

Both radii are proportional to `L = (V/N)^(1/3)`, so `g` is a function of
`r/L` alone and the switching zone moves smoothly with the volume. A strain
that changes the volume therefore acts on the weight through the distances and
through `L`: the strain virial carries the extra term
`delta_ab/3 dw/d ln L` (`dg/d ln L = -r g'(r)` at fixed distances), and the
strain jets divide every point-centre distance by `det(1 + eps)^(1/3)` before
switching. With a hard sphere, the volume enters through the sphere itself:
a centre crosses it for many points when the cell dilates, and the jumps
accumulate in proportion to the step. Central differences of the analytic
strain derivative against `eps_xx` (p69 column [0][0], components (1,1) and
(2,2)) then keep a constant offset of 2.1e-5 to 2.7e-5 for h = 0.04 down to
0.0025 instead of falling like `h^2`.

Value, fixed-point gradient (`periodic_weight_gradient`,
`dimensional_weight_gradient`), strain virial (`periodic_weight_virial`),
owner-riding Hessian (`periodic_weight_hessian`) and the nested jets of the
strain and displacement second derivatives (`becke_weight_jet`) all use the
same weights: the closed forms for the first derivatives, and nested
forward-mode jets of `ln q` and `ln g` in the three distances `(|p - c_b|,
|p - c_c|, |c_b - c_c|)` for the second derivatives. No finite differences
enter production.

Why it matters. With the hard sphere the SCF energy of the p67 chain fixture
(SVWN5/STO-3G, k 4x1x1, grid level 4, `grid_shell` 4, overlap cutoff 18) is a
function of the geometry with small jumps of about 1e-10 to 5e-9 Hartree:
central differences under the shear `eps[2][1]` of the energy against the
analytic `dE/d eps = 1.794804325e-3` gave errors of -4.01e-5, -1.24e-5,
-2.78e-6, -7.83e-7, -6.50e-7 and -1.12e-6 for h = 0.08, 0.04, 0.02, 0.01, 0.005
and 0.0025, so below h = 0.01 the error grows like 1/h, and it changed
irregularly with `grid_shell`. Such jumps are invisible to the energy but
corrupt every finite-difference validation, the force constants, relaxations
and the third derivatives built on them.

Smoothness of the quadrature itself (`periodic::grid::tests`, independent of
the SCF): a lattice sum of normalized Gaussians (exponent 0.5) on the atoms of
a skewed two-atom cell (H and O, grid level 2) integrates to one electron
per atom and cell for every geometry, and the quadrature is differentiated
along the shear `eps[2][1]` and along a displacement of one atom. For a smooth
quadrature the central-difference derivatives `D(h)` of five steps
(h = 1e-2 down to 6.25e-4) differ by `|D(h) - D(h/2)| ~ h^2`; the fit of
those differences gives p = 2.0047 (R2 0.999999) for the shear and p = 1.9996
(R2 1.000000) for the displacement at `grid_shell` 1, and p = 2.0097 (R2
0.999994) and 1.9997 (R2 1.000000) at `grid_shell` 2. With the hard sphere
the successive differences do not fall (4.3e-4, 4.1e-4, 1.7e-3, 1.2e-3 for
the shear; p = -0.64, R2 0.64, and p = 0.42, R2 0.40 for the displacement).

Audit of the other thresholds that make a quantity depend discontinuously on
the geometry:

| Threshold | Where | Jump bound | Status |
|---|---|---|---|
| Hard sphere of the local image list | `local_images`, `dimensional_local_images` | 1e-10..5e-9 Ha (measured above) | replaced by the switching zone |
| Competitors entering the cell functions | `local_becke_weight`, jets | same order, first order at points between atoms | replaced by `q = 1 - g_c (1 - s)` |
| Volume dependence of `r_on`, `r_cut` | `Switch`, strain virial and jets | a hard sphere moving with the volume gave a constant offset 2e-5 in strain-derivative differences | smooth (`g` depends on `r/L`), included in virial and jets |
| Radial points beyond `r_cut` | `PeriodicGrid::build_with_images` | exactly zero (`g = 0`) | exact |
| `weight_cutoff` (default 1e-14) | `PeriodicGrid::build_with_images` | at most `1e-14 |rho e_xc|` per point crossing, below 1e-14 Ha for the fixtures; points are far from their owner | kept; the derivative fixtures use 0 |
| `|g| < 1e-26` of 0 or 1 (`|x| > 60`) | `Switch::value` | below 1e-26 relative | kept (needed to keep nested jets finite) |
| `BECKE_CELL_SCREEN` | `becke_weight_jet` | below 1e-24 of the partition sum | kept |
| AO point/shell screening and `overlap_cutoff` image lists | `xc_ao_images`, image blocks | the AO or overlap tail at the cutoff, about `exp(-alpha_min R^2)` (3e-24 for H STO-3G at 18 Bohr); basis and cutoff dependent | kept; converge `overlap_cutoff` |
| Density floors inside functional providers | `src/xc` | not changed here | not audited as part of the grid |

### Report and units

The three [API tables](rust-api.md#common-calculation-driver) show the aligned entries and controls.

| Quantity | Shape | Unit/convention |
|---|---|---|
| `overlap_real/imag` | `[k,AO,AO]` | Dimensionless, canonical spherical AO order |
| `kinetic_real/imag` | `[k,AO,AO]` | Hartree |
| `overlap_derivatives_real/imag` | `[k,atom,axis,AO,AO]` | Per Bohr, fixed Cartesian k and lattice |
| `kinetic_derivatives_real/imag` | The same | Hartree/Bohr |
| `grid.points` | `[point,3]` | Bohr, unwrapped Cartesian positions |
| `grid.weights/raw_weights` | `[point]` | Bohr^3, partitioned/unpartitioned |
| `grid.owners` | `[point]` | Zero-based physical atom ID |
| `grid.moving_weight_derivatives` | `[point,atom,3]` or absent | Bohr^2 |

The grid and its derivative arrays are absent when not requested. Rust
returns the corresponding typed matrices and `PeriodicGrid`. The CLI
prints complete indexed matrix, derivative, grid and moving-weight rows.
Native geometry is in Bohr; CLI XYZ and cell inputs are in Angstrom.
ASE converts the Atoms geometry/cell and returns the atomic-unit report;
explicit cutoffs remain Bohr and ordinary ASE SCF results are unchanged.
Controls for occupations, functionals or electrostatic compensation do
not belong to this calculation and are rejected when explicitly supplied.

### Independent acceptance

`tests/p52_dimensional_basis.rs` independently sums closed contracted
s-Gaussian overlap and kinetic formulas over a brute-force lattice.
It checks the isolated limit, all periodic dimensions, complex phases,
raw translation/Hermiticity defects, arbitrary p/d shells against the
existing 3D path, atom cell relabelling, open-vector independence, all seven
boundary masks and primitive/supercell band folding.

Independent displaced Gaussian operators validate atomic derivatives at
four steps `0.04,0.02,0.01,0.005 Bohr`. Fixed- and moving-point Becke
derivatives use `0.02,0.01,0.005,0.0025 Bohr`. Every fit retains its four
positive errors and must pass `abs(p-2)<0.02`, log-space
`R_squared>0.99999`, step-halving ratio four within 3%, and complementary
Richardson checks. No error floors or relaxed criteria are used.
The accepted measurements are:

| Dimension | Derivative | Step interval (Bohr) | p | R_squared |
|---|---|---|---|---|
| 1D | Bloch overlap | 0.04..0.005 | 1.999977915056 | 0.999999999967 |
| 1D | Bloch kinetic | 0.04..0.005 | 1.999885585125 | 0.999999999122 |
| 2D | Bloch overlap | 0.04..0.005 | 1.999980573284 | 0.999999999975 |
| 2D | Bloch kinetic | 0.04..0.005 | 1.999901254242 | 0.999999999346 |
| 3D | Bloch overlap | 0.04..0.005 | 1.999980227488 | 0.999999999974 |
| 3D | Bloch kinetic | 0.04..0.005 | 1.999901355648 | 0.999999999348 |
| Isolated | Fixed-point Becke | 0.02..0.0025 | 1.999700836115 | 0.999999993983 |
| Isolated | Moving-point Becke | 0.02..0.0025 | 1.999700836152 | 0.999999993983 |
| 1D | Fixed-point Becke | 0.02..0.0025 | 1.999699992383 | 0.999999993949 |
| 1D | Moving-point Becke | 0.02..0.0025 | 1.999699990064 | 0.999999993949 |
| 2D | Fixed-point Becke | 0.02..0.0025 | 1.998873973690 | 0.999999828257 |
| 2D | Moving-point Becke | 0.02..0.0025 | 1.998873971370 | 0.999999828256 |
| 3D | Fixed-point Becke | 0.02..0.0025 | 1.998875555608 | 0.999999828811 |
| 3D | Moving-point Becke | 0.02..0.0025 | 1.998875553361 | 0.999999828810 |

The first grid-overlap gate with level 3 and independent 12 Bohr cutoffs
measured maximum errors `2.42e-8,2.11e-7,4.20e-6,3.70e-6` in the
isolated, 1D, 2D and 3D cases, below the `5e-5` acceptance bound.
The stronger non-real Bloch grid gate measured maximum real/imaginary
overlap errors `2.32e-7`, `5.23e-6` and `6.23e-6` in 1D, 2D and 3D.
Both parts retain the same `5e-5` bound. Boundary-phase and reciprocal-shift
identities also passed; the latest independent Rust basis suite has eight tests.

`python/tests/test_dimensional_basis.py` checks the native, ASE and running
CLI matrices and derivatives in every periodic dimension. It repeats the
four-step operator fits and compares the complete open-direction grid and
moving weights across all public paths. The next
[primitive nuclear/Hartree integral milestone](#3-gaussian-coulomb-integrals)
adds normalized Cartesian Coulomb components. The following
[contracted AO/Bloch Coulomb driver](#4-contracted-coulomb-operators) connects
those components to complete complex nuclear/Hartree operators and
fixed-density component energies. SCF, finite-q and response connections
remain in progress. The [semilocal XC connection](#5-semilocal-xc-operators)
uses these Bloch values and spatial derivatives for arbitrary weighted
functional providers, with independent metric/grid particle diagnostics.

The public native fits use a different k mesh and retain nonzero imaginary
components. Their overlap/kinetic p values are respectively
`1.999980306532/1.999901242202` (1D),
`1.999980044002/1.999901230158` (2D), and
`1.999979863878/1.999901102638` (3D), with minimum log-space
R_squared `0.999999999344` on the same `0.04..0.005 Bohr` interval.
The public basis/boundary/orbital/native/ASE acceptance checkpoint passed
68 Python tests; it overlaps the previously documented suites rather than
constituting 68 additional unique physical cases. Both CLI names and the
Python-feature release library built successfully with the original
dependencies and lockfile. Existing ASE/NumPy deprecation warnings persist.

## 3. Gaussian Coulomb integrals

*Dimensional Gaussian Coulomb integrals.*

`periodic::gaussian_coulomb` evaluates normalized Cartesian primitive nuclear
attraction and electron-repulsion integrals with the same 1D/2D/3D Coulomb
Green function as [prescribed-charge electrostatics](#1-boundary-conditions-and-electrostatics).
It supplies electronic nuclear/Hartree components, including p, d and higher
Cartesian angular powers, before contracted AO/Bloch and density assembly.
It is independent of the chosen exchange-correlation functional. The common
calculation is available on all four surfaces in [three API tables](rust-api.md#common-calculation-driver).
Full lower-dimensional SCF and finite-q DFPT assembly remain in progress.

### Mathematical convention

A primitive is `N_l(alpha) (r-A)^l exp(-alpha*|r-A|^2)`, normalized in
three-dimensional Cartesian space even for wires/slabs. The nuclear source
has charge Z, centre C and Gaussian width w; w=0 denotes a point. Attraction
is `-Z * integral g_a(r) g_b(r) G_D(r-C) dr`; its sign is part of the driver.
An ERI is `integral g_a(r) g_b(r) G_D(r-r') g_c(r') g_d(r') dr dr'`.
`G_D` replicates only the specified periodic subspace and integrates the open
directions without introducing a vacuum supercell.

Gaussian products and Cartesian powers use the existing Hermite expansion.
The construction follows [McMurchie and Davidson's original integral paper](https://doi.org/10.1016/0021-9991(78)90092-X),
with dimensional Green derivatives replacing the free-space Coulomb tensor.
The dimensional Ewald splitting follows
[Tornberg's primary derivation](https://arxiv.org/abs/1404.3534).
These papers define the expansion and boundary model; acceptance comes from
the independent numerical gates below.

For a product exponent p=alpha_a+alpha_b, the smeared pair widths are
`sqrt(1/p+w^2)` for attraction and `sqrt(1/p+1/q)` for an ERI.
Hermite contractions include all normalization, product-centre and volume
factors. Every component uses the same potential gauge. Nuclear and Hartree
primitive integrals may have nonzero integrated charge; an electronic energy
must combine nuclear, electronic and compensating contributions consistently.
The primitive API does not add a compensating charge independently to each
integral.

`DimensionalCoulomb::gaussian_derivatives(r,width,order)` returns actual mixed
Cartesian derivatives through the requested total order, without factorial
normalization. The tensor reports the real, nonzero reciprocal and zero-mode
contributions separately. It includes finite coincident-centre values.
Real-space derivatives use the existing Boys/Hermite recurrences. Reciprocal
derivatives use the exact multivariate recurrence of
`exp(i*G.r-t^2*|P_open*r|^2)`, including off-diagonal entries of the open
projector for tilted wires/slabs. The slab zero mode is differentiated
analytically; the wire zero mode retains its logarithmic gauge at one Bohr.
No same-q Hermitian projection or derivative finite difference is applied.

All Gaussian centre gradients use the exact raising/lowering identity
`d_Ai g_l = 2*alpha*g_(l+ei)-li*g_(l-ei)` with the original normalization.
The nuclear centre uses an independent derivative of the Green tensor.
The raw sum of all centre gradients is reported as `translation_defect`.

#### Charged 3D Gaussian zero mode

For pair width w and volume V, the zero-average Green constant is
`-pi/(V*alpha_Ewald^2)+pi*w^2/V`. The real-space Gaussian correction removes
mean `pi*w^2/V`, which the constant must restore. For sources with total
charge Q, the corresponding energy is
`-pi*Q^2/(2*V*alpha_Ewald^2)+pi*Q*sum(q_i*w_i^2)/V`.
The width term cancels in neutral systems and is essential for a charged
Gaussian density with uniform compensation.

An independent full Fourier oracle exposed the missing width term: the
charged-cell energy was 0.08968313938354877 Hartree against
0.11568568404281729 Hartree. Correcting the Green constant gives
0.11568568404276942 Hartree, an absolute defect of 4.79e-14 Hartree.
The original failing measurements are preserved in validation records;
no thresholds, oracle values or physical denominators were changed.

### Inputs and report

| Input | Convention |
|---|---|
| Geometry and cell | Bohr in Rust/native; Angstrom XYZ/cell in CLI; ASE converts Atoms geometry |
| PBC mask | Selects lattice vectors; all seven nonempty masks are valid |
| Exponents | Positive finite Bohr^-2, one per Gaussian |
| Angular powers | Non-negative Cartesian `[lx,ly,lz]` per Gaussian |
| Nuclear case | Three positions: A, B, C; two exponents/angular triples; charge default 1, width default 0 |
| ERI case | Four positions: A, B, C, D; four exponents/angular triples; source controls rejected |
| Width and real cutoff | Explicit Bohr on every surface |
| Ewald alpha and reciprocal cutoff | Bohr^-1 |
| Quadrature and tolerance | Initial order 96, absolute tolerance 1e-11; reported result uses twice the order |

Rust takes `CartesianGaussian`, `GaussianNucleus` and `GaussianIntegralOptions`.
Python takes `positions, exponents, angular, cell` and keyword controls.
ASE exposes `get_periodic_gaussian_integral(atoms, *, exponents, angular, ...)`;
atomic numbers do not implicitly specify the primitive source charge.
The CLI requires `--gaussian-exponents` and `--gaussian-angular`, with comma
separated exponents and semicolon separated Cartesian triples. A complete
nuclear example is:

```text
dft-rs periodic-gaussian-integral centres.xyz --cell 5,5,20 --pbc 1,1,0 --gaussian-exponents .6,.4 --gaussian-angular "1,0,0;0,1,0" --gaussian-source-charge 2 --gaussian-source-width .3
```

| Report field | Meaning |
|---|---|
| `value` | Normalized primitive integral in Hartree |
| `components` | Real, reciprocal nonzero, reciprocal zero and zero `point_self`; exact sum to value |
| `center_gradients` | Hartree/Bohr, in input centre order; shape `[3,3]` or `[4,3]` |
| `quadrature_value_defect` | Difference between independent n and 2n Fourier rules |
| `quadrature_gradient_defect` | Maximum norm of the corresponding centre-gradient difference |
| `translation_defect` | Raw norm of the total centre gradient |
| `converged` | Finite value/gradients and all three defects within tolerance |
| `kind`, `model`, `units`, `warnings` | Physical integral, gauge, units and cutoff-convergence requirements |
| `dimension`, `measure`, `pbc` | Boundary metadata on Python/CLI/ASE; supplied `PeriodicGeometry` in Rust |
| `alpha`, cutoffs, `quadrature_order` | Actual numerical controls; Rust groups cutoffs in `params` |

Quadrature agreement does not certify lattice truncation. Real/reciprocal
cutoffs and Ewald splitting must be converged independently. Automatic
cutoffs account for Gaussian width and derivative order. The direct tensor
recurrence has a checked finite work limit; overflowing angular powers or
tensor sizes are rejected before allocating or evaluating normalization.
SCF, XC, occupation, k mesh and charge-compensation controls do not belong
to the primitive integral and explicitly supplied no-op controls are rejected.

### Independent acceptance

`tests/p53_dimensional_gaussians.rs` compares 3D p/d attraction and ERIs to
an independent full reciprocal sum using ordinary shifted polynomial Gaussian
moments, with no Hermite expansion or Ewald split in the oracle. Absolute
defects are 4.12e-15 Hartree for attraction and 6.81e-17 Hartree for the ERI,
below the unchanged 1e-11 bound. A separate charged Gaussian/uniform-background
Fourier gate checks the corrected zero mode. Coincident normalized s products
are finite and match the full Fourier oracle without removing their physical
Gaussian self contribution.

All dimensions pass repartitioning of the Ewald split, exchange within
Gaussian pairs, pair exchange, rigid translation of all centres, component
energy sums and raw analytic gradient conservation. Scalar Green values and
gradients agree with the separately implemented source-pair evaluation.

Every derivative fit retains four strictly positive errors at
`0.02,0.01,0.005,0.0025 Bohr`. The acceptance requires `abs(p-2)<0.02`,
log-space `R_squared>0.99999`, step-halving ratio four within 3%, and a
complementary Richardson check within 1e-8. No errors are discarded or floored.
The 18 new Rust fits are:

| Dimension | Derivative | p | R_squared |
|---|---|---|---|
| 1D | Green (1,0,0) | 1.999967213585 | 0.999999999918 |
| 1D | Green (2,0,0) | 1.999995275177 | 0.999999999998 |
| 1D | Green (1,1,1) | 1.999959055820 | 0.999999999887 |
| 1D | Green (2,1,1) | 2.000010579842 | 0.999999999992 |
| 2D | Green (1,0,0) | 1.999969602984 | 0.999999999906 |
| 2D | Green (2,0,0) | 1.999995269627 | 0.999999999998 |
| 2D | Green (1,1,1) | 1.999958933768 | 0.999999999885 |
| 2D | Green (2,1,1) | 2.000010450182 | 0.999999999992 |
| 3D | Green (1,0,0) | 1.999971280364 | 0.999999999894 |
| 3D | Green (2,0,0) | 1.999995057413 | 0.999999999998 |
| 3D | Green (1,1,1) | 1.999958846877 | 0.999999999886 |
| 3D | Green (2,1,1) | 2.000010661816 | 0.999999999992 |
| 1D | Attraction centre | 2.000027107347 | 0.999999999939 |
| 1D | ERI centre | 1.999949922642 | 0.999999999831 |
| 2D | Attraction centre | 2.000120180409 | 0.999999999324 |
| 2D | ERI centre | 1.999950768343 | 0.999999999837 |
| 3D | Attraction centre | 1.999959369140 | 0.999999999906 |
| 3D | ERI centre | 1.999959577782 | 0.999999999909 |

The new independent Rust suite passes seven tests. With the existing
dimensional source and basis suites, the current kernel regression passes
21 tests. The compiled native extension, ASE and canonical CLI pass 33
public tests across the Gaussian, source-boundary and basis suites. The
alias CLI passes seven additional route/contract invocations; these repeat
the corresponding physical cases. Every split component, all centre
gradients, physical units, numerical controls and raw quadrature defects
agree across the routes within 1e-12 in atomic units.

The public primitive suite repeats the six Gaussian-centre fits above and
adds independent nuclear-source-centre fits, on the same four-step interval:

| Dimension | Derivative | p | R_squared |
|---|---|---|---|
| 1D | Nuclear source centre | 1.999920132624 | 0.999999999583 |
| 2D | Nuclear source centre | 1.999918059761 | 0.999999999599 |
| 3D | Nuclear source centre | 1.999917952669 | 0.999999999601 |

Both CLI names and the Python-feature library build in release mode. The
default-feature CLI release build also passes, with seven repeated route
and contract tests on each actual executable. The existing 14-test CLI
contract regression passes unchanged.
The
all-target Python-feature compilation and formatting checks pass. The
distribution-source audit has zero restricted content/names, non-English
source, PowerShell scripts or reference/validation inputs. The 25 calculation
IDs match in all three API tables; all 139 physical targets and 67 theory
families remain registered. Original dependencies and the lockfile are
unchanged. This is an electronic integral foundation checkpoint, with
contracted AO/Bloch density assembly, SCF and finite-q DFPT still pending.
The subsequent [contracted Coulomb milestone](#4-contracted-coulomb-operators)
connects the AO/Bloch density assembly; full SCF and finite-q DFPT remain
required after that later checkpoint.

## 4. Contracted Coulomb operators

*Contracted dimensional Coulomb operators.*

`periodic::coulomb_boundary` connects the [dimensional Gaussian Green
integrals](#3-gaussian-coulomb-integrals) to the actual contracted spherical
AO basis, complex Bloch density and nuclear/countercharge sources. The common
driver is available on all four surfaces in [three aligned API tables](rust-api.md#common-calculation-driver).
It evaluates nuclear attraction, the exact linear Hartree map, energy
decompositions and analytic fixed-density centre derivatives in 1D/2D/3D.
It supplies electronic components for the remaining complete dimensional
SCF and finite-q DFPT connection; it is not a self-consistent reference.

### AO and Bloch conventions

Placed-shell coefficients already contain the shared `(l,0,0)` radial
normalization and contracted-shell normalization. The new contraction uses
unnormalized Cartesian polynomial integrals with these coefficients, followed
by the existing exact real-solid-harmonic transform. It does not multiply
the normalized Cartesian primitive factors a second time. One derivative
tensor is shared by all Cartesian components of each primitive pair/quartet.

Actual shell separations select images at `basis_cutoff`. This independent
Gaussian overlap cutoff is never used as a Coulomb truncation. Electrostatics
uses the common dimensional Ewald Green operator, with separately converged
real and reciprocal cutoffs. The dimensional Fourier treatment follows
[Tornberg's primary derivation](https://arxiv.org/abs/1404.3534).
Open directions are not wrapped or replicated and their cell-vector sizes
do not enter the operator.

The unshifted mesh and uniform weights define
`D_cd(U) = sum_k exp(-i*k.U) P_cd(k)/Nk`.
Every complex density element is retained. The periodic Hartree contraction
is `J_ab(T)=sum_(cd,U) D_cd(U) (a(0)b(T)|c(0)d(U))_periodic`.
The Green integral already includes translations of the complete source
product; no second source-cell sum is added. Spherical source density blocks
are transformed as `C_c * D_cd * C_d^T` before contraction.

The output Bloch convention is `O(k)=sum_T exp(i*k.T) O(T)`.
Hermitian input densities need not obey time reversal. Their inverse Bloch
images can be complex and are never projected onto real matrices. The
conjugate image/product pairs cancel the imaginary physical Hartree potential;
the remaining raw imaginary image defect is measured before the Bloch
transform. No final Hermitian projection repairs a failed input or build.
Signed Hermitian density directions are valid inputs to the linear Hartree
map for static response.

### Charge and energy conventions

Effective nuclear charges are the existing basis charges with ECP cores
subtracted. Nuclear widths default to zero points, excluding their bare self
energy; positive widths denote normalized Gaussian densities with finite
physical self energy. The formal charge is an explicit model input and fixes
`N_expected = sum(Z_effective)-formal_charge`.
The actual supplied density count is `sum_k Tr(P(k)S(k))/Nk`.
A mismatch is returned as `charge_residual` and prevents a converged
physical density report; it does not modify the density or denominator.

| Compensation | Condition | Physical source |
|---|---|---|
| `neutral` | Formal charge zero | No countercharge; isolated open directions |
| `uniform` | 3D | Uniform charge `-formal_charge`, zero-average potential |
| `gaussian` | Any periodic dimension | Fixed co-periodic Gaussian `-formal_charge` with explicit centre/width |

Countercharge amplitude is fixed by the requested formal charge, rather than
silently adjusted to an unconverged density. Its centre follows the atom
centres in the derivative axis as one additional independent centre.
Nuclear attraction includes the potential of both ions and the countercharge.
The source energy includes all ion/countercharge cross and self terms.
Every term uses the same zero-average 3D, isolated-slab or logarithmic wire
gauge at one Bohr.

The joint Coulomb energy is
`E_sources + sum_k Tr(P V)/Nk + sum_k Tr(P J[P])/(2*Nk)`.
Each source energy has four reported components; electronic attraction and
Hartree have the three Green components. Their sums reproduce the joint
energy. The uncoupled source energy is a gauge component of this combined
electronic problem, not a standalone neutral source calculation.
Physical energies require the specified compensation and particle constraint.
No isolated-defect correction is added.

All centre derivatives are analytic. Atom movement includes every home or
translated Gaussian belonging to that atom, both output AO centres, both
density-product centres and each physical nuclear source. The independent
countercharge centre has its own derivative. `energy_gradients` holds the
supplied AO density coefficients, Cartesian k and lattice fixed. Moving atoms
then changes the nonorthogonal metric and can change `Tr(P S)`; a finite
displaced fixed-density report therefore need not satisfy the reference
particle constraint. These derivatives are energy components for SCF forces,
which additionally require stationarity, occupation/charge constraints and
the metric/Pulay contribution.

### Interfaces and report

Rust can retain `DimensionalCoulombBackend::new(mol,mb,geometry,options)`,
which caches nuclear operators and overlap, and call `evaluate(density)`
for successive density states/directions. Cached geometry, basis and k points
are immutable; `geometry()` and `k_points()` return shared views.
The convenience `periodic_coulomb_operators` uses the same backend.

| Surface | Inputs |
|---|---|
| Rust | Existing molecule/basis/geometry; optional `&[BlochMatrix]`; `PeriodicCoulombOptions` |
| Native Python | Atomic numbers, Bohr positions/cell; `density_real/imag[k,AO,AO]`; charge, basis and controls |
| ASE | `get_periodic_coulomb_operators(atoms,*,basis=None,charge=None,**controls)`; Atoms geometry converted to Bohr |
| CLI | `periodic-coulomb`, Angstrom XYZ/cell; optional complete density CSV |

Explicit nuclear/countercharge widths, countercharge centre and real/image
cutoffs use Bohr on every surface. Ewald alpha and reciprocal cutoff use
inverse Bohr. Density is total electron density in the native spherical AO
ordering. It does not specify separate spin occupations or infer SCF results.
No density requests nuclear operators/source components only; Hartree,
electronic energy/count and fixed-density energy gradients are absent.
SCF/XC/temperature controls are rejected when explicitly supplied.

The CLI density file requires the literal header `k,row,col,real,imag`
and exactly one finite row per matrix element. Indices are zero-based;
duplicate, missing, non-finite or out-of-shape entries are rejected.
Rows may be reordered; blank lines and `#` comments are permitted.
`--nuclear-widths` supplies one width per physical atom.
Ewald, compensation and quadrature controls are shared with
`periodic-electrostatics`.

| Report | Shape/convention |
|---|---|
| Nuclear/Hartree matrices | `[k,AO,AO]`, complex, Hartree |
| Split matrices | `[k,component,AO,AO]`, real/nonzero reciprocal/zero reciprocal |
| Centre derivatives | `[k,centre,axis,AO,AO]`, Hartree/Bohr |
| Source/fixed-density energy gradients | `[centre,3]`, Hartree/Bohr |
| Energy components | Source four terms, electronic attraction and Hartree three terms |
| Boundary and charge | PBC, dimension/measure, actual/target electron count, formal and compensating charge |
| Actual controls | Image/Ewald cutoffs, alpha, final Fourier quadrature order |
| Raw diagnostics | Quadrature operator/derivative/source defects; Hermiticity, translation, density imaginary maximum and potential imaginary defect |

The independent Fourier rules are `n` and `2n`; the latter is returned,
with defaults `n=96` and absolute tolerance `1e-11`. Every numerical
defect and particle residual must pass for `converged`. Agreement of the
two rules does not certify image or Ewald cutoff convergence. The physical
model, units and limitations are part of every public report.

### Independent acceptance

`tests/p54_dimensional_coulomb.rs` verifies positive, negative and zero
formal charge in all dimensions, correct Gaussian/uniform compensation,
particle constraints, Ewald repartitioning, complex densities without time
reversal, raw potential reality, Hermiticity/translation, Hartree linearity
and weighted self-adjointness for signed complex directions. All seven
nonempty masks are compared after removing the unused open cell vectors.
Malformed densities, angular/AO offsets and incompatible compensation
are rejected before contraction.

The independent full 3D reciprocal operator uses the existing separately
validated Fourier AO pair matrices, without the dimensional Ewald/tensor
contraction. Complete p/d spherical nuclear and Hartree matrices agree
within `3.43e-14` and `5.78e-15 Hartree`, respectively, against an
unchanged `1e-11` bound. The primitive polynomial-moment Fourier oracle
also remains passing after commonizing the normalization-free contraction.

An additional independent finite-nuclear test evaluates the full unsplit
Fourier energy of the joint physical Gaussian nuclear/countercharge and
complex contracted electronic density. Brute integer-box AO images and the
separate Fourier AO-pair routine avoid the dimensional Green/tensor builder.
Source, electron/source, Hartree and joint energy defects are respectively
0, 1.60e-14, 3.17e-15 and 1.13e-14 Hartree, all below the same 1e-11 bound.
Thus agreement is checked term by term rather than through cancellation of
errors in the total. The complete contracted Rust suite now contains eight
tests; the finite-nuclear component gate passed separately.

Four-step displaced operator/energy fits retain every positive error at
`0.02,0.01,0.005,0.0025 Bohr`. Requirements remain
`abs(p-2)<0.02`, `R_squared>0.99999`, halving ratio four within 3%,
and complementary Richardson agreement within `1e-8`.
These are fixed supplied AO density derivatives, not substituted SCF forces.

| Dimension | Derivative | p | R_squared |
|---|---|---|---|
| 1D | Nuclear matrix | 2.000010102906 | 0.999999999994 |
| 1D | Hartree matrix | 1.999994047179 | 0.999999999999 |
| 1D | Fixed-density Coulomb energy | 1.999996199715 | 1.000000000000 |
| 2D | Nuclear matrix | 2.000008371590 | 0.999999999995 |
| 2D | Hartree matrix | 1.999989327454 | 0.999999999998 |
| 2D | Fixed-density Coulomb energy | 1.999995637440 | 1.000000000000 |
| 3D | Nuclear matrix | 2.000005135539 | 0.999999999998 |
| 3D | Hartree matrix | 2.000001352537 | 1.000000000000 |
| 3D | Fixed-density Coulomb energy | 1.999994595828 | 0.999999999999 |

The first six contracted Rust tests passed together with seven primitive
and 14 existing CLI tests. The all-mask gate passed separately.
The native/ASE/canonical-CLI Coulomb, primitive, boundary and basis checkpoint
passed 40 Python tests; the alias passed four repeated route/contract tests.
These suites overlap earlier checkpoints and are not additional unique cases.
After adding the finite-nuclear public paths, both completed default-feature
CLI builds passed seven route/contract tests per executable, covering point
and finite-width nuclei in all three dimensions. The staged native extension
retains its Python-feature build. The initial run that began before the default
build completed is retained separately; acceptance uses the subsequent run.

The public STO-3G fits exercise a different physical contracted basis and
retain non-real density entries. They use the same four steps and unchanged
acceptance bounds as the Rust fits:

| Dimension | Derivative | p | R_squared |
|---|---|---|---|
| 1D | Nuclear matrix | 2.000024480796 | 0.999999999964 |
| 1D | Hartree matrix | 2.000009740851 | 0.999999999989 |
| 1D | Fixed-density Coulomb energy | 1.999973610358 | 0.999999999949 |
| 2D | Nuclear matrix | 2.000021316849 | 0.999999999967 |
| 2D | Hartree matrix | 2.000003301085 | 0.999999999977 |
| 2D | Fixed-density Coulomb energy | 1.999973538863 | 0.999999999949 |
| 3D | Nuclear matrix | 2.000018546944 | 0.999999999985 |
| 3D | Hartree matrix | 2.000029072492 | 0.999999999976 |
| 3D | Fixed-density Coulomb energy | 1.999973196109 | 0.999999999946 |

Full raw errors and intervals are preserved in the validation checkpoint.
The complete 139-target/67-family response program remains active; complete
dimensional electronic SCF and finite-q DFPT are still required.

## 5. Semilocal XC operators

*Dimensional semilocal DFT operators.*

The common `periodic_xc_operators` driver and immutable `DimensionalXcBackend`
connect contracted spherical Bloch AOs to the existing functional-provider
interface in 1D, 2D and 3D. The physical atom-centred quadrature covers one
cell in periodic directions and all open directions. Unused open vectors
are never repeated, wrapped or used to scale the integration measure.
Arbitrary complex Hermitian spin densities are retained without a
time-reversal assumption. All uniformly weighted k points contribute to
the same density before energy or potential evaluation.

The calculation returns the **semilocal component**: LDA, GGA, meta-GGA,
or the semilocal parts of a hybrid/range-separated/nonlocal functional.
Exact-exchange coefficients and required nonlocal tails are reported as
separate metadata; their energies are not included in this component result.
The explicit component model is shared by Rust, CLI, native Python and ASE.
This calculation supplies SCF operators but does not solve SCF stationarity.
Full dimensional SCF, nonlocal operators and finite-q DFPT remain required
within the original response program.

### Conventions and variational derivative

Input density matrices have shape `[spin=2,k,AO,AO]`, with spin order
alpha/beta and native spherical AO ordering. The local density is
`rho_s(r) = average_k sum_ab P_s,ab(k) Phi_a,k(r) Phi_b,k(r)*`.
GGA uses all Cartesian density-gradient contractions; meta-GGA additionally
uses `tau_s = 0.5 average_k sum_ab P_s,ab grad(Phi_a) . grad(Phi_b)*`.
The same complex values and spatial derivatives construct the potentials,
so a fixed-geometry density direction satisfies
`dE = average_k sum_s Tr(delta_P_s(k) V_s(k))`.
The reported potential is in Hartree, without an extra k-weight factor.

Every weighted functional part is also independently evaluated at energy
order zero on the same density variables. `component_energy_defect` compares
their sum with the composite order-one energy. No calculation branches on
registered functional names; custom weighted providers use this same path.
The grid cache retains the highest AO derivative order requested when it is
built. It can evaluate a lower functional rung, and rejects a higher rung
that needs missing AO data. Shifted Bloch meshes reuse the reference
quadrature through `PeriodicXcGrid::at_dimensional_k_points`; this is a
ground-state field on another mesh, not a finite-q transition response.

Analytic overlap matrices provide the physical electron count independently
of quadrature. The target is `sum(Z_effective) - formal_charge`; the density
is never renormalized by this driver. Grid/metric and requested/actual
particle discrepancies are returned separately. Local XC requires no
Coulomb compensation model, even for charged wires/slabs; constructing the
complete electrostatic SCF still requires the explicit physical compensation
described in [the Coulomb operator contract](#4-contracted-coulomb-operators).

### Public controls and reports

All entries appear under `periodic-xc` in the three aligned [API tables](rust-api.md#common-calculation-driver).
Rust uses `PeriodicXcOptions`; native Python accepts `density_real` and optional
`density_imag`; ASE uses `get_periodic_xc_operators`. The CLI requires complete
alpha/beta density files through `--alpha-density-matrices` and
`--beta-density-matrices`. Both files use the strict
`k,row,col,real,imag` format of `periodic-coulomb`.

| Control/result | Convention |
|---|---|
| `basis_cutoff` / CLI `--overlap-cutoff` | AO image and overlap-pair cutoff; default 12 Bohr |
| `grid_cutoff` / CLI `--basis-grid-cutoff` | Independent radial/Becke physical cutoff; default 12 Bohr |
| `grid_level` / CLI `--grid-level` | Existing quadrature controls; default 3 |
| `operator_tolerance` / CLI `--xc-operator-tol` | Default absolute `1e-11` for raw Hermiticity, component-energy and particle diagnostics |
| `grid_tolerance` / CLI `--xc-grid-tol` | Default absolute `1e-5` electron count against the analytic overlap metric |
| `potential_real`, `potential_imag` | `[spin,k,AO,AO]`, Hartree; no time-reversal projection |
| `component_energies`, `component_labels` | Weighted energies in provider part order; Hartree |
| `electron_count`, `grid_electron_count` | Separate analytic metric and physical quadrature counts |
| `charge_residual` | Actual minus requested electron count; signed |
| `grid_number_defect` | Absolute quadrature minus metric count |
| `hermiticity_defect` | Measured before Hermitian roundoff projection |
| `converged` | All operator, component, particle and grid-count gates pass; does not certify SCF convergence or cutoff convergence |

CLI XYZ/cell input and ASE geometry use Angstrom and are converted to Bohr.
Native geometry and all explicit cutoffs use Bohr. SCF, force, temperature
and Coulomb-compensation flags are refused for this component calculation
before file reads when they do not apply.

### Independent acceptance

`tests/p55_dimensional_xc.rs` checks all seven masks, independence from unused
open cell vectors, arbitrary weighted providers, complete complex potentials,
component-by-component linear assembly and invalid inputs. The Slater-only
energy density `-3/4 (6/pi)^(1/3) sum_s rho_s^(4/3)` and its potential are
assembled independently from the separately validated point-first Bloch AO
routine. Maximum energy and full complex-potential defects are respectively
`1.23e-15` and `2.23e-16 Hartree`, below the unchanged `1e-11` bound.

All 18 density-direction fits across six functional classes and three
dimensions retain the four dimensionless steps `0.04,0.02,0.01,0.005`.
The classes include LDA, GGA, meta-GGA, global hybrid, range separation and
the semilocal part with an explicit VV10 requirement. Requirements are
`abs(p-2)<0.02`, log-space `R_squared>0.99999`, halving ratio four within 3%
and complementary Richardson agreement within `1e-8 Hartree`.
The measured exponents lie in `1.999985913324..2.000026014660`, with minimum
`R_squared=0.999999999937`. Every raw positive error and step is retained.

The shared Bloch AO builder also has four-step spatial gradient/Hessian gates.
Native Python independently repeats LDA/GGA/meta-GGA density-direction fits
using STO-3G and exercises every complex potential and diagnostic through
the running CLI and ASE. Build and final regression results are recorded
only after their actual completion. The current checkpoint passed 39 native,
ASE and canonical-CLI XC/Coulomb/basis tests; the alias passed ten repeated
route/contract cases. The library regression passed 345 tests with three
existing ignored gates. A compatibility-only metadata wrapper preserves
the existing `PeriodicXcContribution` result fields; after this wrapper and
the refined spatial fixture, six shared XC tests, four new integration tests
and 14 existing CLI contract tests passed on the final numerical code.
All-target Python-feature checks and Python-feature library/CLI release
builds passed. These checkpoint counts overlap and are not a unique aggregate.
The full 139-target/67-family objective remains active.

The first public checkpoint preserved a physical grid/metric failure in
three 3D cases: 8 Bohr AO images and a 10 Bohr grid at level 3 gave a
7.17e-4 electron-count defect despite conserving the analytic metric count.
Increasing AO images alone left a level-3 quadrature error above 1e-5.
Independent image, grid-cutoff and quadrature probes retain all ten measured
combinations. At 12 Bohr AO/grid cutoffs and level 4 the defect is 9.69e-6;
at 16 Bohr and level 4 it is 6.80e-6. The acceptance tolerance stays 1e-5.
The resolved public interval uses 12 Bohr and level 4; the initial failed
checkpoint and all nine initial four-step fits remain recorded. A dedicated
regression requires the underresolved grid to remain unconverged, preserving
the metric and the original threshold instead of normalizing the grid count.

The refined spatial Bloch-AO gate exercises actual tilted images in all three
dimensions at a 15 Bohr AO cutoff. Its gradient exponents lie in
1.999255470545..1.999255832269; Hessian exponents lie in
2.003545944328..2.003553217871, with minimum R_squared 0.999999159752.
All six fits retain the same four Bohr steps, halving-ratio and Richardson
gates without discarded steps or error floors.

The resolved public density fits span `1.999960491709..2.000061249571`, with
minimum log-space `R_squared=0.999999999577`. Their nine full raw error arrays
and the nine initial fit arrays are preserved in the validation checkpoint.

## 6. Self-consistent dimensional DFT

*Dimensional self-consistent DFT.*

`periodic::scf_boundary::run_dimensional_scf` connects the dimensional analytic
Gaussian overlap/kinetic, nuclear/Hartree and semilocal XC operators to the
common finite-k SCF iteration. It supports every nonempty periodic-axis mask,
positive/negative/zero formal charge with explicit compensation, collinear
restricted/unrestricted spin, integer insulating or mesh-global Fermi-Dirac
occupations, LDA/GGA/meta-GGA providers and full/long-range exact exchange.
Functional providers supply weighted local terms and exchange coefficients;
the driver does not select a calculation formula by registered name.
Field-free RKS/UKS and fixed-integer ROKS can also attach the variational
VV10 lattice pair term. The ROKS force and stress use its own stationary
common-orbital multiplier.
The [VV10 acceptance record](response.md#first-geometry-derivatives-of-variational-periodic-vv10)
gives the 3D NaH, 1D wire and 2D slab force and stress results,
including all-component fits and the 1D/2D API comparisons.
The SCF and XC charge checks share the overlap metric defined by
`overlap_cutoff`. The independent `xc_ao_cutoff` controls Bloch AO
evaluation at the quadrature points.

All open lattice vectors are unused in the physical image, reciprocal, grid
and exchange sums. An open k-mesh axis must contain one point. The geometry
can therefore have zero or dependent open vectors without introducing vacuum
images. The boundary in `DimensionalScfOptions.scf` must match the immutable
`PeriodicGeometry`; `DimensionalScfOptions::for_boundary` sets this consistently.

### Energy and charge model

The reported per-cell total energy is

```text
E = E_source + Tr_k(P_total T) + Tr_k(P_total V_nuclear)
    + Tr_k(P_total V_ECP) + Tr_k(P_total V_external)
    + 1/2 Tr_k(P_total J[P_total])
    - cx/2 sum_spin Tr_k(P_spin K_full[P_spin])
    - clr/2 sum_spin Tr_k(P_spin K_erf[P_spin]) + E_semilocal + E_VV10
F = E - T S
```

`Tr_k` is the uniformly weighted complex Hermitian trace on the original
integration mesh. ECP, external-field and VV10 terms enter when selected
by a compatible model. The nuclear/countercharge source, electronic attraction and
Hartree operators share one dimensional Green gauge, including the physical
open-direction zero mode. Formal charge fixes the target number and a fixed
countercharge of opposite sign. A Gaussian countercharge repeats only along
periodic axes; its position and width are physical model inputs in Bohr.
Uniform compensation is allowed only in 3D. `neutral` requires zero formal
charge. A separate isolated-ion monopole correction would count electrostatic
energy twice and is not included.

Localized exact exchange uses physical-subspace images and the free-space
full or erf electron repulsion integral, with inverse Bloch spin densities.
The bounded image energy contains two density factors. The analytic exchange
potential differentiates both, using the direct contraction and its adjoint.
The adjoint has images out to twice the energy-image cutoff. This makes the
finite-cutoff energy variational, rather than relying on matrix Hermitianization
to fix a nonsymmetric density map. The image cutoff remains an explicit
approximation to converge independently.
The zero-field driver requires time-reversal paired densities for this real
localized contraction. Generic supplied complex Hermitian densities remain
available for the Coulomb/XC component APIs; a magnetic SCF requires the complex
exchange contraction and a field-dependent reference instead.

### Public interfaces

The three aligned tables in [API reference](rust-api.md#common-calculation-driver) contain `dimensional-scf`.

```rust
use dft_rs::periodic::scf_boundary::{run_dimensional_scf, DimensionalScfOptions};
let mut options = DimensionalScfOptions::for_boundary(geometry.boundary());
options.scf.k_mesh = [3, 1, 1];
let report = run_dimensional_scf(&mol, &basis, &geometry, &functional, &options)?;
```

```text
dft-rs dimensional-scf cell.xyz --cell "3.7,0,0;0,0,0;0,0,0" \
  --pbc 1,0,0 --k-mesh 3,1,1 --xc pbe --grid-level 4
```

```python
from dft_rs import dimensional_scf
report = dimensional_scf(numbers, positions_bohr, cell_bohr,
    pbc=[True, False, False], k_mesh=[3, 1, 1], xc="pbe", grid_level=4)
report = calculator.get_dimensional_scf(atoms, k_mesh=[3, 1, 1], grid_level=4)
```

Native coordinates/cell/cutoffs/widths are Bohr; CLI XYZ/cell vectors are
Angstrom, explicit cutoffs/widths/compensation centers remain Bohr. ASE converts
Atoms geometry to Bohr; this explicit report getter returns Hartree per cell
and does not overwrite the ordinary ASE property cache. k points are Cartesian
inverse Bohr; temperature is kelvin. Densities/Fock/energy-weighted densities
are `[spin,k,AO,AO]`, overlap/core Hamiltonian `[k,AO,AO]`, complex matrices use
separate `_real`/`_imag` arrays. No spin sum or imaginary part is omitted.

Defaults are basis/grid radii 12 Bohr, neutral compensation, 96/192-order
Coulomb quadrature agreement 1e-11, grid number tolerance 1e-5, operator tolerance
1e-11, core guess, 100 iterations, energy tolerance 1e-9, density RMS 1e-8,
orbital-gradient tolerance sqrt(energy tolerance), and insulating occupations.
Controls include independent exchange cutoff, nuclear widths, Coulomb splitting
and real/reciprocal cutoffs, grid level, temperature, spin, guess, ADIIS/CDIIS,
damping/level shift and stopping criteria. Rust's legacy `scf.grid_shell` must
retain its default; `grid_cutoff` specifies the physical dimensional radius.
The CLI rejects unrelated controls before reading the structure.

The ASE getter inherits the calculator's configured k mesh, temperature,
grid level, SCF tolerances, iteration controls and exchange cutoff. Its basis
radius defaults to the calculator's `overlap_cutoff` (20 Bohr), and its
iteration limit defaults to 150; explicit getter keywords override these
settings. Configured ECP, density fitting or stability following is reported
as pending rather than silently replaced by the all-electron exact-Hartree
model. A nondefault legacy `grid_shell` is rejected in favor of `grid_cutoff`.

`scf_converged` reports the iteration stopping condition. `converged` additionally
tests freshly rebuilt F[P], band-metric stationarity, occupation consistency,
particle number, compensated Coulomb quadrature, translation/Hermiticity,
grid number and component sum diagnostics. All defects are raw measurements.
Finite-temperature `projector_defect` measures P squared minus P and can be
nonzero physically; `occupation_consistency` is the applicable thermal test.
The report returns total/source/kinetic/attraction/Hartree/full-exchange/
long-range-exchange/semilocal energies and every weighted semilocal component.

Finite-temperature free energies use the provider's ground-state XC energy
and the variational noninteracting Fermi-Dirac entropy. The local XC provider
has no explicit temperature argument; this checkpoint therefore uses the
zero-temperature XC approximation in the Mermin functional. An explicit
thermal XC provider and its temperature derivatives remain additional work.

### Independent acceptance and remaining connections

`tests/p58_dimensional_scf.rs` tests all seven masks with charges -1, 0, +1,
actual self-consistent particle number, raw physical residuals and component
closure. A normalized primitive Gaussian helium reference compares kinetic
and exchange terms to closed expressions and the combined compensated Coulomb
energy/total energy to the independent spherical neutral atomic limit.

Eighteen four-step variational paths (1D/2D/3D times HF, SVWN5, PBE, PBE0,
CAM-B3LYP, r2SCAN) compare analytic spin Fock contractions against central
energy differences on metric-normalized orbital rotations. All positive errors
at h = 0.004, 0.002, 0.001, 0.0005 radians are retained. Fits report p, R2 and
the interval; the existing p/R2, halving-ratio and Richardson requirements
remain unchanged. Both home-cell and nonzero neighboring exchange-image
intervals are checked. Numerical values and public-route regression results
are recorded in [Response validation](response.md).

The first neighboring-image test exposed a finite-cutoff exchange-map bias:
the error approached 2.3e-5 rather than decreasing quadratically. That failed
fit is retained in the validation record. The resolved implementation adds
the analytic derivative of the second density factor, including its additional
image translations; the same interval and acceptance thresholds then pass.

`python/tests/test_dimensional_scf.py` compares actual native, ASE and canonical/
alias CLI spin matrices, occupations, energies and diagnostics across signed
1D/2D/3D cases; reassembles the self-consistent Coulomb and semilocal terms
through their component drivers, exercises six functional classes and finite
temperature, and checks incompatible controls.

This checkpoint is a zero-field all-electron collinear reference. Finite-q
dimensional Sternheimer screening, current-dependent field kernels, Berry
uniform electric fields, flux-compatible magnetic SCF, spinor/Dirac/Nambu
references, dimensional short-range/nonlocal ECP operators and lattice nonlocal
correlation remain tracked implementation. Legacy 3D SCF/DFPT wrappers retain
their calculation plan; this reference has a separate dimensional provenance
and cannot be passed to their 3D derivative driver. Full dimensional forces,
stress and field response are not inferred from fixed-density component
gradients or the SCF stationarity flag. The full accepted response objective
and every target/family remain active.

The legacy `Cell` exact-exchange reference still uses its original bounded
direct image map. Migrating that reference and its derivative paths to the
paired-density variational contraction remains required; the new dimensional
exchange acceptance does not certify that separate legacy map.

## 7. Electromagnetic Fourier vertices

*Dimensional periodic electromagnetic vertices.*

`periodic::field::periodic_field_operators` builds actual analytic
Hamiltonian derivatives for scalar and vector Fourier potentials in 1D, 2D
and 3D periodic subspaces. All seven periodic masks and positive, zero and
negative formal charges use the same contracted spherical Gaussian path.
This component is independent of the XC provider and occupation model. It
provides the source and contact operators needed for periodic electric and
magnetic response; screening and induced observables remain separate work.

The mode convention is `exp(i q.r-i z t)`. `wave` is a Cartesian
vector in inverse Bohr. `scalar_potential` is electron potential energy
`V=-phi` in Hartree, and `vector_potential` is the reduced atomic
vector potential entering `(p+A)^2/2`. Complex input and output values
use `real,imag`. The physically associated mode fields are

```text
E_q = i q V_q + i z A_q
B_q = i q cross A_q
```

Their invariance under `A -> A+i q Lambda`,
`V -> V-i z Lambda` and the measured `|q.B|` are verified.
A real field combines the conjugate mode `(-q,-conjugate(z))`.
Scalar/vector coefficients for one mode need not be real.

The contracted vertices are:

```text
N(q;k)_mu,nu = <chi_mu,k+q | exp(i q.r) | chi_nu,k>
J_a(q;k) = <chi_mu,k+q | {exp(i q.r),p_a}/2 | chi_nu,k>
dH/dlambda = V N(q) + sum_a A_a J_a(q)
d2H/dlambda2 = (A.A) N(2q)
d2H/dA_a(q)dA_b(q) = delta_ab N(2q)
```

The second derivative has no conjugation in `A.A`. Opposite-mode mixed
contact is obtained from the zero-wave density vertex; arbitrary mixed waves
use their summed wave. This report returns the contact density vertex at
`2q` for the supplied single mode, without constructing a redundant
nine-component diagonal Cartesian tensor.

Analytic Hermite Fourier moments evaluate `N`; original primitive
contractions with ket angular powers raised/lowered evaluate `J`.
The anticommutator includes `q_a N/2`. Radial coefficients and spherical
transforms retain the original normalization. No real-space quadrature or
finite differences enter the production operator. Pair images are selected
by actual shell separation within `basis_cutoff`, default 12 Bohr.
Open directions are never repeated. This finite AO image sum has a user
cutoff; this report does not claim to certify cutoff convergence.

The ket mesh is the dimensional unshifted mesh plus a Cartesian
`k_offset`, default zero. The report constructs and labels independent
`k+q` and `k+2q` targets even for noncommensurate modes. It neither
folds those targets onto the input mesh nor Hermitianizes a fixed nonzero-q
block. The exact comparison is `N(q;k)=N(-q;k+q)^dagger`, with the same
relation for `J`. At zero q, the density vertex equals the analytic Bloch
overlap. Rows are target spherical AOs and columns are ket spherical AOs;
the current vertex has a leading Cartesian axis.

Rust accepts `PeriodicFieldOptions` and returns `PeriodicFieldReport`.
The common Python/native entry is `periodic_field_operators(...)`, the
ASE getter is `calc.get_periodic_field_operators(atoms,...)`, and the CLI
entry is `dft-rs periodic-field cell.xyz --cell "..." --pbc 1,1,0`.
ASE converts positions and cell from Angstrom to Bohr. Explicit wave,
k offset, scalar/vector amplitude and frequency controls retain native units.
CLI controls are `--field-wave`, `--field-k-offset`,
`--field-scalar`, `--field-vector` and `--field-frequency`.
Wave and offset use three comma-separated values; scalar and frequency use
two; vector components use three semicolon-separated complex pairs.
The frequency pair is real Hartree frequency and nonnegative damping. Rust
also accepts the common static/retarded/imaginary `Frequency` type.

The report includes all scalar/current/first-order/second-order/contact matrices,
actual ket and target meshes, electric and magnetic mode fields, formal charge,
dimension, units, axis convention, model and raw transversality defect. It
contains no SCF convergence flag because no electronic state is solved here.

Acceptance compares all vertices against an independent closed s-Gaussian
Fourier/momentum oracle using a brute-force translation box for every mask and
charge sign. The maximum scalar/current/contact difference is
`1.332339907734e-15`, with the unchanged `1e-12` gate. q reversal,
non-Hermitian finite-q blocks, zero-q overlap and electromagnetic gauge
identities are checked independently. Nine spatial-current derivative fits
for contracted s/s, p/d and d/p shells use four centre steps
`0.02,0.01,0.005,0.0025` Bohr. Their exponents span
`1.999942139505..1.999980249706`, with `R2>=0.999999999775`.
Each positive error, the step interval, halving ratio and Richardson check are
retained; gates are `|p-2|<0.02`, `R2>0.99999`, 3% halving-ratio
agreement and `1e-8` Richardson agreement.

Periodic electric and magnetic response remain part of the full objective.
The next connections include the self-consistent dimensional ground state,
field-source screening, gauge-covariant current/meta-GGA kernels, induced
current plus diamagnetic contacts, homogeneous electric Berry polarization,
and uniform magnetic flux-compatible supercells. A static q=0 scalar/constant
vector mode produces zero E and B: it represents potential/twist controls,
not a homogeneous electric or magnetic field. Complete finite-field SCF,
magnetic periodic NMR, susceptibilities and their higher/mixed responses are
not declared complete by this source-operator milestone.



The combined public checkpoint passes 29 tests and the executable alias 20
repeated route/contract cases. Both ordinary default-feature CLI binaries
pass the same 20 cases after completed builds. The shared library regression
passes 346 tests with three existing ignored gates; final all-target compilation,
14+3+2 targeted tests, formatting and the 30-ID/139-target/67-family source audit
pass. These counts are overlapping checkpoints. Original dependencies and
lockfile remain unchanged; the full response objective is still in progress.

## 8. Screened electric and magnetic fields

*Screened periodic electric and magnetic fields.*

`dfpt::field::periodic_field_response` connects analytic scalar/current Fourier
vertices to the common self-consistent finite-q density solver. Its three
aligned entries in [API reference](rust-api.md#common-calculation-driver) expose the same Rust, canonical CLI,
compiled Python and ASE calculation. The result contains induced density,
paramagnetic and diamagnetic number current, charge current, actual electric
and magnetic field amplitudes, reference diagnostics and the Krylov residual.

### Field convention and physical model

For the convention `exp(i q.r-i z.t)`, the electron Hamiltonian is

```text
H(lambda) = (p + lambda A(r))^2 / 2 + lambda V(r) + H_density[P(lambda)]
V = -phi
H'(q) = V N(q) + sum_a A_a J_a(q)
J_a(q) = 1/2 {exp(i q.r), p_a}
E(q,z) = i q V + i z A
B(q,z) = i q cross A
```

Scalar potential is electron potential energy in Hartree. Vector potential
uses reduced atomic momentum units; no unreported factor of the speed of
light is applied. The amplitudes define a dimensionless directional strength
`lambda`: all induced values are derivatives at zero strength, including the
supplied amplitudes. To obtain a tensor, repeat with independent unit sources.

The orbital response solves
`delta P = chi0(z) [H' + (v_Coulomb + f_XC) delta P]` by matrix-free GMRES.
The source and observable use the same one-electron image cutoff as the
reference overlap/kinetic matrices. The induced Hartree operator retains its
separate AO-product reach and complete reciprocal sphere. Both cutoffs are
convergence parameters. The reference density is integrated on its original
k mesh; the bra operator and orbitals are built independently at every k+q,
including noncommensurate q. A single-q density is never Hermitianized.

This current calculation uses 3D all-electron collinear providers and
adiabatic density screening. Functional names do not select its equations.
Meta-GGA requires covariant kinetic-density sources and explicit current/contact
kernels. The shared density solver now connects the complex full/range-separated
exchange response kernel for matching HF/hybrid parents. This connection is
not the independent hybrid field/current acceptance: the dense screening and
nonlinear field fits below retain their original LDA/GGA scope. The
`p59_periodic_field_response` matching-parent hybrid scalar-gauge regression
separately checks that a spatially constant scalar potential creates neither
an electric/magnetic field nor an induced density/current, including at a
finite retarded frequency. A mismatched semilocal reference cannot test this
connection because its reference fingerprint already fails.

Self-consistent exchange transforms with the density matrix under a local
phase; this is distinct from prescribing a fixed nonlocal core operator.
The distinction is relevant to
[the gauge invariance of self-consistent Hartree-Fock/RPA](https://doi.org/10.1103/PhysRevA.19.1876).
Finite-basis gauge completeness and current continuity remain numerical
convergence requirements, not consequences of a small Krylov residual.
Nonlocal-core
currents, explicit current-dependent XC, spin Zeeman, SOC/spinor and Dirac
terms are further physical models. These extensions remain in the full
coverage ledger and are not certified by the present LDA/GGA acceptance.

### Observables and diagnostics

All spin/k/AO entries are preserved. For a probe at -q,

```text
delta n_q = sum_spin,k Tr[N(q,k)^dagger delta P(q,k)] / Nk
delta j_para,a = sum_spin,k Tr[J_a(q,k)^dagger delta P(q,k)] / Nk
delta j_dia,a = A_a N_e
delta j_number = delta j_para + delta j_dia
delta rho_charge = -delta n_q
delta j_charge = -delta j_number
```

These are cell-integrated quantities; divide by the physical 3D cell volume
to obtain number/charge/current density per volume. The explicit current
derivative pairs -q and +q, giving the overlap vertex N(0). The N(2q) contact
in the [bare vertex driver](#7-electromagnetic-fourier-vertices) instead
belongs to two +q Hamiltonian derivatives and is not substituted here.

`continuity_defect = abs(z delta n_q - q.delta j_number)` is a raw finite-basis
measurement. The GMRES residual is re-evaluated with the actual screening
operator; convergence of that equation does not establish gauge completeness
of a finite Gaussian basis. Neither observable nor denominator is adjusted to
force the continuity diagnostic to zero. Reference eigen/stationarity/metric/
occupation/particle-number defects and the model approximation are returned.

The retarded frequency is `z=omega+i*damping` in Hartree. Imaginary frequencies
use `z=i*frequency` with no real damping. Integer or mesh-global finite-
temperature occupations use the reference ensemble; uniform static fixed-number
response constrains each collinear spin population. Explicit thermal XC is
not inferred from the noninteracting Fermi-Dirac reference occupations.

The implemented static canonical occupation constraint recognizes explicit
q=0. A nonzero reciprocal-lattice representation of the folded uniform Bloch
channel still requires its canonical occupation constraint connection and
thermal acceptance. The reciprocal-mode nonlinear fits below use integer
occupations; they do not certify that separate finite-temperature limit.

### Public interfaces

```rust
use dft_rs::dfpt::field::{periodic_field_response, PeriodicFieldResponseOptions};
let mut request = PeriodicFieldResponseOptions::default();
request.response.q = q_cartesian;
request.scalar_potential = scalar_amplitude;
request.vector_potential = vector_amplitudes;
let report = periodic_field_response(&cell, &functional, &reference,
    &scf_options, &request)?;
```

```text
dft-rs field-response cell.xyz --cell 4.7 --xc pbe \
  --response-q .043,.017,-.021 --response-omega .13 --response-damping .04 \
  --field-scalar .37,.22 --field-vector "0,0;0,0;.27,.19" \
  --reciprocal-cutoff .85
```

CLI geometry/cell vectors use Angstrom; Cartesian q and reciprocal cutoff
use inverse Bohr. Complex amplitudes are real,imag pairs, with vector axes
separated by semicolons. Existing periodic SCF controls apply, including
temperature in kelvin. `--field-wave`, `--field-k-offset`, `--field-frequency`
and source-mode controls belong to other requests and are rejected here.

```python
from dft_rs import periodic_field_response
report = periodic_field_response(numbers, positions_bohr, cell_bohr,
    xc="pbe", q=(.043,.017,-.021), omega=.13, damping=.04,
    scalar_potential=.37+.22j, vector_potential=(0j,0j,.27+.19j),
    reciprocal_cutoff=.85)
report = calculator.get_periodic_field_response(atoms,
    q=(.043,.017,-.021), scalar_potential=.37+.22j,
    vector_potential=(0j,0j,.27+.19j), omega=.13, damping=.04,
    reciprocal_cutoff=.85)
```

Native geometry is Bohr. ASE converts Atoms geometry and inherits calculator
SCF controls; q, potentials and frequency remain explicit atomic-unit inputs.
The getter returns a native report without replacing the ordinary ASE cache.
Density is `[spin][k]` of dictionaries containing complete AO `real`/`imag`
matrices. Scalar/vector observable pairs use `_real`/`_imag` fields. k points,
target k+q points and the screening reciprocal modes are explicitly reported.

### Independent acceptance and remaining scope

`tests/p59_periodic_field_response.rs` compares a mixed noncommensurate
electric/magnetic source to a small dense complex solve, independently
contracts its observables, verifies the diamagnetic sign/contact and retains
the unprojected continuity defect. Separate electric and magnetic real fields
are compared to a nonlinear two-band SCF oracle at four strengths. That oracle
analytically diagonalizes its Hermitian 2x2 Fock with Pauli components,
evaluates full semilocal potentials and reciprocal Hartree at every iterate,
and includes the quadratic minimal-coupling Hamiltonian. Its core is recentered
on the converged reference so that its declared Coulomb truncation matches
the response kernel. It does not claim a derivative of an unconverged Ewald
tail or of an unspecified SCF field model. Every positive error is retained;
the existing p/R2, halving-ratio and Richardson thresholds apply unchanged.
Measured intervals/results are recorded in [Response validation](response.md).

`python/tests/test_periodic_field_response.py` compares actual native, ASE
and CLI density/current values for electric, magnetic and mixed sources in
LDA and GGA, checks field signs and shifted mesh indices, and exercises the
finite-temperature constant scalar gauge and invalid controls. Two additional
q=0 cases specify a dynamic uniform electric amplitude E through A=-i E/z
and check number conservation through all public surfaces.

Finite-q dimensional screening for 1D/2D, general functional/current kernels,
uniform static Berry electric fields, uniform magnetic flux cells and finite-
field SCF remain implementation targets. At q=0,z=0 scalar/vector potentials
have zero electric/magnetic field; they must not be presented as a uniform
static field calculation. At q=0 and nonzero frequency a vector potential
represents a dynamic uniform electric field. Finite-q transverse vector
potentials represent periodic zero-net-flux magnetic fields. The full accepted
response objective and all registered targets/families remain active.

## 9. Range-separated screened Hartree

The finite-k 3D SCF and its analytic forces build the Hartree matrix with
`periodic::screened_coulomb` instead of summing every `erfc` quartet of an
explicit image supercell (that exhaustive build stays available as
`Cell::reference_hartree_image_blocks` and is the test oracle).

Every contracted shell is split into a compact part (primitive exponents at
or above `compact_exponent`, default 3 Bohr^-2) and a diffuse part. Any
interaction involving a pair of two diffuse parts is summed exactly in
reciprocal space with the bare `4 pi / G^2` kernel, because such a pair's
Fourier transform is negligible beyond `G = 2 sqrt(2 e_c ln(1/eps))`.
Interactions of two compact pairs use the Ewald split with
`alpha^2 = e_c`: screened `erfc` quartets in real space plus the damped
reciprocal half and its neutralizing background. Each piece is an Ewald sum
of one pair of charge distributions with its own background and is
therefore independent of the splitting parameter; the bare reciprocal
pieces are its `alpha -> infinity` limit. Quartets are skipped when the
Schwarz bound times the density block is below the tolerance (default
`1e-13`) or when the two Gaussian charge distributions are farther apart
than the reach of `erfc(alpha_eff r)/r`, with
`alpha_eff^2 = rho alpha^2 / (rho + alpha^2)`.

Cost controls: compact kets are visited in decreasing order of their Schwarz
bound so that the scan stops at the first one below the tolerance, the
real-space reach of each pair shrinks with its Schwarz product
(`sqrt(ln(S/eps) + 4)/alpha_eff + 1` Bohr), and its lattice cells are
enumerated around the bra-ket displacement instead of being tested one by
one. Ket densities are transformed without per-`G` matrices, and on the
`|G|`-ordered mesh every pair transform stops where its Gaussian damping
times the largest remaining coefficient is below 1e-18.

`screened_hartree_derivatives` returns the Hartree energy at a fixed AO
density, its analytic nuclear gradient (reciprocal pair-transform
derivatives, background, and `erfc` quartet derivatives on all four centres)
and the centre virial used by the strain derivative.

Acceptance:

- Against the exhaustive build on an s+p basis with compact and diffuse
  primitives, `J(k)` agrees within 6.3e-12 for splits 0.4, 1.0 and 2.0;
  the splits agree with each other within 6.9e-12.
- On a converged finite-k H2 density (3x1x1 mesh) the screened build takes
  1.43 s against 105 s for the exhaustive one. Screened builds with other
  splits and a 1e-15 tolerance agree within 3.0e-11; the exhaustive build
  differs by 3.3e-9, which is the truncation of its own image supercell.
- Analytic Hartree gradient at fixed density: four-step fits over
  0.02..0.0025 Bohr give p = 1.999924 and 1.999809 (R^2 = 1.00000000); the
  energy equals `1/2 sum Tr[D^T J]` within 1e-10 and the gradient sums to
  zero under a rigid translation.
- Rock-salt LiH (STO-3G, 2x2x2 mesh, core-Hamiltonian density; 8 threads
  while another build ran): 21706 pairs and 7239 G vectors at the former
  default split. One Hartree build took 98.8 s with the cell-by-cell
  partner scan (39.3 million quartets), 33.1 s after the Schwarz-ordered
  neighbour search (9.5 million quartets). With the split 1, 2, 4, 8, 16 it
  takes 33.1, 26.3, 24.8, 37.3 and 55.2 s; the larger splits trade
  real-space quartets (0.25 million at 16) for reciprocal vectors (458319 G
  at 16). All splits agree within 4.7e-10 in `J(k)` and 2.1e-10 Hartree in
  the Hartree energy; the default is 3. A 3D crystal SCF therefore costs
  about 25 s per iteration for this cell; the reciprocal transforms are now
  the dominant cost and are the next optimization target.

## 10. Finite-k energy, forces and electronic structure on the driver

`periodic-energy`, `electronic-structure`, `bands` and `dos` run the 3D
finite-k SCF through the common driver (Rust `driver::run`, the CLI, Python
and ASE; see [rust-api.md#common-calculation-driver](rust-api.md#common-calculation-driver)). Lattice rows and positions are Bohr (the CLI
reads Angstrom). k points are reported as Cartesian vectors (1/Bohr) and in
reduced coordinates `f_i = a_i . k / (2 pi)`. Band structures rebuild the
converged Fock operator at each path point with
`periodic::scf::reference_fock_at` and solve `F(k) c = e S(k) c`; the DOS is a
Gaussian broadening with an explicit `sigma` (no default), optionally on a
denser mesh. Band edges are reported for integer occupations.

Acceptance (H2 chain, SVWN5/STO-3G, 4x1x1 mesh):

- Path points that coincide with mesh points reproduce the SCF band
  energies within 1e-8 Hartree; the integrated DOS at mid-gap equals the
  electron count within 1e-6; the Mulliken electron count residual is below
  1e-8; the CLI JSON energy equals the Rust driver within 1e-9.
- Analytic force against central differences of the energy, with converged
  cutoffs (overlap 18 Bohr, grid level 4, two grid shells): fitted over
  0.16..0.02 Bohr, p = 2.008795, R^2 = 0.99999887; the Richardson
  combination of the two smallest steps differs from the analytic value by
  3.7e-7.
- Recorded failures: with coarse cutoffs (overlap 12, grid level 2, one
  shell) the 0.02..0.0025 interval gives p = 0.547 (R^2 = 0.154); with
  converged cutoffs, 0.04..0.005 gives R^2 = 0.99994830 and 0.08..0.01 gives
  R^2 = 0.99998760 (halving ratios 4.014, 4.020, 4.098). The total energy
  carries a non-smooth component near 5e-9 Hartree from real-space
  truncations; locating it (AO and grid screening) is open work.

## 11. Analytic stress and cell relaxation

The common `stress` calculation requests the same analytic operator as
`periodic-energy` with `stress=true`. For `reference=gks` or `ghf`, both
routes use the complex-spinor affine derivative. Its field-free
dimensional scope, raw diagnostics and acceptance are recorded
in [the stress coverage entry](response.md#r026-stress-elastic-high-elastic).
Use `forces=false` to request stress without nuclear forces.
For controls, output units and the same four-interface example, see the
[periodic API guide](api-guide.md#periodic-example-stress-and-gamma-phonons).

The full stress target covers each physically defined periodic
reference, functional, ECP and field combination. The bulk tensor uses
energy per volume. The 1D and 2D targets use the periodic length and area
with an active-space affine strain. For quantized magnetic fields the
strain derivative must identify the fixed-flux convention.

`periodic::stress::analytic_stress` returns the strain derivative
`dE/d(epsilon_ab)` of the converged finite-k energy (the Mermin free energy
at finite temperature) and the stress `sigma = (1/V) dE/d(epsilon)` in
Hartree/Bohr^3, with `epsilon` applied to the lattice rows and to every
atom at fixed fractional coordinates and fixed fractional k points (the
Bloch phases `k . T` are strain invariant). The reported tensor is the
symmetric part; `rotation_residual = max |dE/de_ab - dE/de_ba|` measures
the rotational invariance of the truncated sums. A negative trace means a
cell that wants to shrink; the pressure is `-tr(sigma)/3`. The ASE
`get_stress` property uses the Voigt order (xx, yy, zz, yz, xz, xy) in
eV/Angstrom^3.

At a stationary SCF only the explicit dependence at fixed density images
`D(T)` enters, plus the Pulay term `-sum_T W(T) . dS(T)/d(epsilon)`. Gaussian
functions move rigidly with their centres; for every lattice sum the strain
derivative is the centre virial `sum_X (dE/dX_a) X_b` over all basis,
nuclear and grid centres including images, plus the wave-vector dependence
`-sum_G G_a dE/dG_b` and the `1/V` prefactors of reciprocal sums
(`periodic::strain`):

| Term | Strain derivative |
|---|---|
| Kinetic, overlap (Pulay) | `(dO/dA)_a (A - B - T)_b` per image pair |
| Scalar semilocal ECP | `(dU/dA)_a (A - C)_b + (dU/dB)_a (B - C)_b` for each retained bra, ket and core image, with the occupied density at fixed reduced k |
| Nuclear attraction | screened charges: `(d/dA)(A - C) + (d/dB)(B - C)` per pair and nuclear image `C`; reciprocal half: centres, `d rho/dG_b = i <a|r_b e^{iG.r}|b>`, kernel `w [2 G_a G_b (1/4alpha^2 + 1/G^2) - delta_ab]`; overlap background `-delta_ab` |
| Hartree (screened build) | centre virial of the `erfc` quartets and pair transforms, bra wave vectors, `1/(V G^2)` kernels with and without damping, background `-E_bg delta_ab` |
| Exact exchange | centre virial of the translated quartets |
| Exchange-correlation | rigid Bloch AOs at a strained point (`sum_T e^{ik.T} d_a phi(u) u_b`, `u = r - A - T`, and the Hessian analogue), point motion with its owner atom, periodic Becke weights `sum_C (dP/dC)_a (C - R_owner)_b` |
| Nuclear repulsion | Ewald strain derivative; charged cells add `-q^2` times the Ewald strain derivative of a unit charge (monopole correction) |

Cells with effective core potentials are refused (the strain derivative of
the translated semi-local operator is not implemented).

`cell-opt` (`periodic::optimizer::optimize_cell`) minimizes the enthalpy
`E + pV` over the atomic positions and a symmetric deformation `F` of the
reference cell with L-BFGS and Armijo backtracking; the gradients are
`F^T dE/dR` and `(dE/d(epsilon) + pV I) F^-T`, and convergence requires
`max |dE/dR| < gtol` and `max |sigma + pI| < stress_tol`. `opt-fixed-cell`
relaxes the atoms only.

Acceptance of each term at fixed density (`periodic::stress::tests`; cell
rows (4.2, 0, 0), (0.3, 4.6, 0), (0.2, -0.25, 4.9) Bohr, two hydrogens with
an s and a p shell each contracted from a compact and a diffuse primitive;
components [0][0], [0][1], [2][1]; four-step central-difference fits):

| Term | Steps | p | Smallest R^2 |
|---|---|---|---|
| Kinetic | 4e-3..5e-4 | 1.999396, 1.999901, 1.997024 | 0.99999870 |
| Overlap | 4e-3..5e-4 | 1.999994, 2.000146, 1.999898 | 1.00000000 |
| Nuclear attraction | 4e-3..5e-4 | 2.000025, 2.000099, 2.000032 | 1.00000000 |
| Hartree, split 1 | 4e-3..5e-4 | 2.000012, 1.999981, 1.999980 | 1.00000000 |
| Hartree, split 3 | 4e-3..5e-4 | 1.999999, 2.000273, 1.999937 | 0.99999997 |
| Exact exchange | 4e-3..5e-4 | 2.000002, 1.999955, 1.999901 | 1.00000000 |
| XC SVWN5 | 6.4e-2..8e-3 | 1.998424, 1.989492, 1.993366 | 0.99998895 |
| XC PBE | 6.4e-2..8e-3 | 1.997298, 1.987232, 1.992192 | 0.99998430 |
| XC r2SCAN | 6.4e-2..8e-3 | 1.999481, 1.993128, 1.995220 | 0.99999456 |

The XC oracle uses the cell dilated by 2.2 and three Becke shells. Recorded
failures: the periodic Becke denominator counts the images inside a sphere
whose radius scales with `(V/N)^(1/3)`, and images crossing it change the
quadrature discontinuously. In the compact cell (one shell) SVWN5 [0][0]
gave errors 1.76e-4, 8.4e-5, 1.47e-4, 1.97e-4 over 4e-3..5e-4 (p = -0.13,
R^2 = 0.095); dilated by 1.6, 1.9e-5, 7.9e-6, 5.2e-6, 5.8e-6 (p = 0.58,
R^2 = 0.77); dilated by 2.2 with one shell, the shear [0][1] over
6.4e-2..8e-3 had halving ratios 4.07, 4.46, 4.89 (p = 2.157,
R^2 = 0.99925), a strain-independent staircase slope near 5e-7 from the
crossing images. Three shells remove it. A smooth periodic partition would
remove this source of the energy noise floor; it is open work.

Acceptance of the self-consistent stress (`p67_periodic_stress`; skewed
H2 chain cell, SVWN5/STO-3G, 4x1x1 mesh, overlap cutoff 18 Bohr, grid level
4, four Becke shells): central differences of the SCF energy over strain
steps 0.04..0.005 give for dE/de[0][0] p = 1.998296 (R^2 = 0.99999919) with a
Richardson difference of -2.05e-7, and for the shear dE/de[0][1]
p = 1.931182 (R^2 = 0.99999960, constant halving ratio 3.81) with a
Richardson difference of -1.27e-8. With two Becke shells [0][0] gave
p = 1.934 (R^2 = 0.99978); the Richardson differences at 0.01/0.005 were
-4.98e-6 (two shells), -3.29e-6 (three) and -2.05e-7 (four): the images
crossing the Becke neighbourhood bias a central difference by a
strain-independent staircase slope. The analytic stress is unchanged
between orbital-gradient tolerances 1e-6 and 1e-9. The rotation residual
is 5.47e-5 Hartree for this grid: the angular quadrature is fixed in the
laboratory frame, so a rigid rotation changes `Exc`; the [0][1] and [2][1]
fits validate the antisymmetric part component by component.

Relaxations: fcc helium (STO-3G, SVWN5, 2x2x2) at 5 GPa converges in four
steps from a 6.0 Bohr cubic edge (volume 54.0 -> 61.39 Bohr^3) with a
monotone enthalpy and a final stress within 2e-6 Hartree/Bohr^3 of -p; a
uniform 1% compression or expansion of the relaxed cell raises the
enthalpy. The fixed-cell relaxation of the chain lowers the energy
monotonically to forces below 2e-4 Hartree/Bohr.

## 12. Gamma-point force constants and phonons

`periodic::phonon::gamma_force_constants` returns the analytic Hessian of the
finite-k energy for Gamma-point displacements, in which every lattice image
of an atom moves with it. The driver feature `phonons` diagonalizes the
mass-weighted force constants (electron masses) and reports frequencies in
cm^-1 (negative = imaginary) with the Cartesian mode patterns. The raw
asymmetry and the acoustic sum rule residual are reported before the
symmetrization; no acoustic correction is applied.

With the response density `P^y`, the response energy-weighted density
`W^y`, and the first-order skeleton Fock and overlap matrices `F^x`, `S^x`,

```text
H_xy = E_skel,xy + Nk^-1 sum_{k,s} (P^y . F^x - W^y . S^x)(k)
```

The skeleton second derivatives are taken at fixed density images:

| Term | Implementation |
|---|---|
| Kinetic, overlap (Pulay) | mixed bra-ket second derivatives of every image pair (`periodic::hessian::two_centre_hessian`); a pair whose two centres belong to one atom moves rigidly and vanishes |
| Nuclear attraction | screened charges: three-centre blocks of each pair and nuclear image with the sum rules `AC = -AA - AB`, `BC = -BA - BB`, `CC = AA + AB + BA + BB`; reciprocal half: nested and mixed pair-transform blocks, pair-structure-factor and structure-factor-structure-factor terms; overlap background |
| Hartree (screened build) | `erfc` quartet mixed second derivatives on four centres; reciprocal `sum_G K Re[rho_x* rho_y + rho* rho_xy]` by class with per-atom derivative transforms; compact background `-pi N_c^2/(2 alpha^2 V)` |
| Exchange-correlation | the molecular grid kernel on a local basis of lattice images per batch of points (`periodic::xc_hessian`), with analytic periodic Becke weight second derivatives (`periodic::grid::periodic_weight_hessian`) |
| Nuclear repulsion | analytic Ewald Hessian (`periodic::ewald::ewald_hessian`) |

The first-order Fock matrix collects `dH_core/dR`, the screened Hartree
derivative at fixed density (`screened_hartree_fock_derivatives`: bra
centres, coefficients through the ket derivative transforms, background,
and the four centres of the real-space quartets) and the XC Fock
derivatives folded from the local image basis. The coupled-perturbed
equations at every k point are solved in real coordinates by
preconditioned conjugate gradients with the exact screened Hartree and the
semilocal XC kernels.

Scope of this implementation: LDA and GGA references with integer
occupations, or with Fermi-Dirac occupations through the orbital-pair
response of section 18, and exact exchange for insulators (section 31).
Meta-GGA functionals and ECP cells are refused; finite-q phonons are
separate work (phonon bands and DOS,
Born charges and the LO-TO splitting are described in sections 14 and 16).

Acceptance of the skeleton terms at fixed density (non-orthogonal cell
with s and p shells contracted from compact and diffuse primitives;
columns (0,0), (1,2), (0,1); four-step fits of central differences of the
analytic gradient of the same term over 4e-3..5e-4 Bohr unless stated):

| Term | p | Smallest R^2 |
|---|---|---|
| Kinetic | 1.999998, 1.999993, 1.999995 | 1.00000000 |
| Overlap | 1.999997, 2.000000, 1.999994 | 1.00000000 |
| Nuclear attraction | 1.999998, 1.999988, 1.999996 | 1.00000000 |
| Screened Hartree, splits 1 and 3 | 1.999858..2.000004 | 1.00000000 |
| Hartree Fock derivative, splits 1 and 3 | 1.999991..2.000012 | 1.00000000 |
| XC skeleton and Fock derivative, SVWN5 and PBE (cell dilated 2.2, three Becke shells, 1.6e-2..2e-3) | 1.996904..2.000120 | 0.99999944 |
| Ewald Hessian (four columns) | 2.000000..2.000012 | 1.00000000 |
| Periodic Becke weight (owner-riding, six points) | 1.999985..2.000024 | 1.00000000 |

The molecular XC Hessian kernel was generalized to a batch of points with
a weight-derivative provider for this purpose; its molecular LDA, GGA and
meta-GGA finite-difference tests pass unchanged.

Acceptance of the force constants (`p68_gamma_phonons`; skewed H2 chain,
SVWN5/STO-3G, 4x1x1 mesh, overlap cutoff 14 Bohr, grid level 3, three
Becke shells), against central differences of the analytic SCF forces:
column (1,0) over 0.04..0.005 Bohr gave p = 2.006675 (R^2 = 0.99999916)
with a Richardson difference of 9.1e-7; column (0,1) gave halving ratios
4.03, 3.92, 5.50 (p = 2.128805, R^2 = 0.99717469), the smallest step at the
noise of the SCF forces, with a Richardson difference of 7.1e-7; the fit
therefore uses 0.08..0.01 Bohr. The raw asymmetry is 9.0e-15, the acoustic
residual 6.0e-14 and the three acoustic frequencies are below 0.0013
cm^-1; the coupled-perturbed solves converge in two or three conjugate
gradient iterations. Two numerical points of the solve: the response
densities are projected onto the time-reversal-symmetric subspace
(a numerically zero source otherwise leaves it), and the linear kernel is
applied to the unit-normalized density, because the screened builds use
absolute thresholds.

Cost: the real-space quartet second derivatives of the Hartree skeleton
and one screened Hartree build per conjugate-gradient iteration and
displacement dominate; for 3D crystals with many compact pairs (for
example rock-salt LiH, section 9) this is currently impractical, and
block solves sharing one Hartree build and a larger split exponent for the
derivative builds are the planned remedies.

## 13. Elastic constants

`periodic::elastic::elastic_constants` returns the analytic second-order
elastic constants of the finite-k energy with clamped and with relaxed ions.
The strain convention is that of section 11: atoms, lattice and every image
centre map as `x -> (1 + epsilon) x` with the density images `D(T)` labelled
by integer translations and fixed fractional k points; Gamma-point
displacements `u` act before the strain, `E((1 + epsilon)(x + u))`. With the
skeleton second derivatives at fixed density and the coupled-perturbed
responses of section 12 extended to the nine strain components,

```text
W_(ab),(cd)   = E_skel + Nk^-1 sum_{k,s} (P^cd . F^ab - W^cd . S^ab)(k)
Lambda_(ab),u = E_skel + Nk^-1 sum_{k,s} (P^u . F^ab - W^u . S^ab)(k)
W_rel         = W - Lambda Phi^+ Lambda^T
C_abcd        = sym_(ab) sym_(cd) [W_abcd - delta_ac tau_bd] / V
B_abcd        = C_abcd + (delta_ac s_bd + delta_bc s_ad + delta_ad s_bc
                + delta_bd s_ac - 2 delta_cd s_ab) / 2
```

`Phi` is the Gamma force-constant matrix, `Phi^+` its inverse outside the
three uniform translations, `tau = sym(dE/d(epsilon))` and `s = tau / V` the
Cauchy stress. `C` are the Lagrangian (energy) elastic constants, exactly
because `W = V C(eta) + delta_ac tau_bd` for `eta = (F^T F - 1)/2`, and `B`
the stress-strain (Birch) coefficients that govern stability and wave
speeds under stress; the two coincide at zero stress. `Lambda` is
reported from both response routes (strain responses with displacement
matrices and the reverse) with their largest difference, and the ionic
relaxation per unit strain is `-Phi^+ Lambda^T`. The antisymmetric part of
`W - delta_ac tau_bd` measures the rotation non-invariance of the
lab-frame quadrature and is reported, not removed.

| Term | Skeleton second derivative at fixed density | First-order strain matrix |
|---|---|---|
| Kinetic, overlap (Pulay) | `O_ac R_b R_d` per pair, `R = A - B - T` (`two_centre_strain_second`) | `(dO/dA)_a R_b` (`strain_fock::two_centre_strain_blocks`) |
| Nuclear attraction | real space: pair and charge second blocks with relative coordinates; reciprocal half by exact nested jets; background `shift N` with `d(1/det)` | screened charges referred to the charge; reciprocal centre virial, wave-vector and kernel terms; background (`nuclear_attraction_strain_blocks`) |
| Scalar ECP | analytic bra and ket second blocks referred to each translated projector, with the mixed strain/displacement chain term | Bloch sum of the retained image virials at fixed reduced k (`ecp_strain_blocks`) |
| Hartree (screened build) | real-space `erfc` quartets in closed form; reciprocal kernels and compact background by exact jets | bra centre virial and wave-vector blocks at fixed coefficients, coefficient strain derivatives by first-order jets of the ket densities, kernels and volume, background, four-centre virial of the quartets (`screened_hartree_strain_fock`) |
| Exchange-correlation | exact nested jets of points riding their owners, AO image centres, periodic Becke weights and the exact second-order functional expansion (`xc_strain_second`) | first-order jets of the same quantities with the `f_xc` expansion of `v_rho`, `v_sigma`, `v_tau` (`xc_strain_fock`) |
| Nuclear repulsion | analytic Ewald strain second derivatives (`ewald_strain_second`) | none |

The XC jets keep exactly the point/shell pairs inside the AO cutoff of the
SCF quadrature, and constants enter jets by scaling rather than products.

The conventional collinear CPKS route carries semilocal density and
kinetic-energy-density terms and compatible exact exchange. For uniform
charge compensation, the affine Hessian includes the negative unit-charge
Ewald strain Hessian multiplied by the square of the net charge. The
monopole contributes only to the affine skeleton. The 3D collinear
scalar ECP path includes the affine skeleton and the two strain/displacement
response routes. Numerical acceptance varies by functional, reference and
occupation model. See the
[response ledger](response.md#r026-stress-elastic-high-elastic).

The driver feature `elastic-fd` is the validation route (finite
differences are validation only): central differences of the analytic
strain derivative over the six symmetric strains `E`, with clamped or
relaxed ions, give `sym[FD - S E - E tau]/V` because the difference at a
strained cell measures the composed strain.

Acceptance of the skeleton strain second derivatives at fixed density
(non-orthogonal cell with s and p shells contracted from compact and
diffuse primitives; strain columns [0][0], [0][1], [2][1] and displacement
columns (0,0), (1,2); four-step fits of central differences of the
validated strain derivative of the same term over 4e-3..5e-4, at a strained
cell including the composed-strain term `delta_bc dE/d(eps_ad)`):

| Term | p | Smallest R^2 |
|---|---|---|
| Kinetic, overlap | 1.999993..2.000007 | 1.00000000 |
| Nuclear attraction, real space | 1.999988..2.000016 | 1.00000000 |
| Nuclear attraction, reciprocal (jets) | 1.999985..2.000018 | 1.00000000 |
| Nuclear attraction, background | 1.999983..2.000004 | 1.00000000 |
| Screened Hartree, splits 1 and 3 | 1.999976..2.000018 | 1.00000000 |
| Ewald nuclei | 2.000000..2.000012 | 1.00000000 |
| XC, SVWN5 (cell dilated 2.2, three Becke shells, 6.4e-2..8e-3) | 1.996021..2.007498 | 0.99999147 |

The XC jet first derivative agrees with the virial route of section 11 to
4.2e-15. Each XC central difference rebuilds the periodic grid, so the
regression keeps one strain-strain and one strain-displacement column per
functional (SVWN5 and PBE); the SVWN5 values above cover every column.

Acceptance of the first-order strain matrices (image blocks matched by
lattice index against central differences of the image-block builds at
strained cells over 4e-3..5e-4; XC against the SCF potential at fixed Bloch
densities over 6.4e-2..8e-3):

| Matrix | p | Smallest R^2 |
|---|---|---|
| Overlap, kinetic | 1.999984..2.000007 | 1.00000000 |
| Nuclear attraction | 1.999988..2.000004 | 1.00000000 |
| Screened Hartree, splits 1 and 3 | 1.999939..2.000012 | 1.00000000 |
| XC, SVWN5 and PBE | 1.998346..1.998805 | 0.99999988 |

Contracted with the density, the matrices reproduce the energy strain
derivatives of section 11 (the Hartree matrices twice the Hartree term) to
relative 1e-12.

The [stress and elasticity ledger](response.md#r026-stress-elastic-high-elastic)
records the bounded clamped and relaxed comparisons, raw residuals and the
pending charged and scalar-ECP SOEC gates. TOEC and FOEC use the
[Lagrangian strain path](response.md#charged-gamma-force-constants-and-higher-elastic-constants).

The analytic constants were also compared with the validation route
`elastic-fd` on the coarse chain (k mesh 2x1x1, overlap cutoff 12 Bohr, grid
level 2, one Becke shell): C11 = 428.67 GPa analytic against 428.77 (steps
0.01) and 428.70 (Richardson, 0.01/0.005); C22 = -61.57 against -61.45 and
-61.34. The negative vacuum-direction constants of this test system are a
property of the model energy, not of the derivative: with STO-3G the H 1s
functions still overlap across the 7 Bohr interchain gap, the reference
carries sigma_yy = +7.7 GPa, and central differences of the SCF energy along
eps_yy give dE/de = 0.04602 and d2E/de2 = -0.3202 at h = 0.02 against the
analytic 0.04590 and -0.3232.

The negative C22 and C33 of this fixture are therefore physical properties
of the model energy. The scan below repeats the analytic `elastic` route on
the same chain (two H atoms at (0, 0, 0) and (1.5, 0.1, 0.05) Bohr, SVWN5/STO-3G,
k mesh 2x1x1, overlap cutoff 12 Bohr, grid level 2, one Becke shell, clamped
ions) with the cell rows `[3.6, 0, 0]`, `[0.3, d, 0]`, `[0, 0.4, d]` Bohr, so
that `d` sets both transverse lattice lengths and the volume is `3.6 d^2`:

| d (Bohr) | C22 (GPa) | C33 (GPa) | sigma_yy (GPa) | C11 V (GPa Bohr^3) |
|---|---|---|---|---|
| 7 | -61.5677 | -59.1346 | 7.6551 | 75,617 |
| 8 | -21.8588 | -21.2458 | 2.2825 | 74,904 |
| 9 | -7.4084 | -7.3870 | 0.6670 | 74,727 |
| 10 | -2.2639 | -2.2622 | 0.1872 | 74,683 |
| 11 | -0.5887 | -0.6174 | 0.0546 | 74,655 |

`C11 V` is computed from C11 = 428.669, 325.102, 256.264, 207.453 and
171.385 GPa at volumes 176.40, 230.40, 291.60, 360.00 and 435.60 Bohr^3. With
the Gaussian tails of neighbouring chains overlapping at 7 Bohr the reference
is under tensile interchain stress (sigma_yy = +7.7 GPa) and the energy is not
at a minimum with respect to transverse strain. C22 goes monotonically to
zero with successive ratios 2.82, 2.95, 3.27 and 3.85, as fast as a Gaussian
overlap tail decays; C22 approaches C33 as the chains decouple; and the
per-chain axial stiffness `C11 V` converges (75,617 to 74,655). The negative
values at d = 7 Bohr are a property of the fixture, not a sign error, and are
kept as regression targets.

Cost: the nested strain jets of the reciprocal Hartree and nuclear sums are
accumulated over blocks of reciprocal vectors (one array per transform; a
rayon fold per task reached 11.6 GB on the chain), and the second-order
strain jets use a packed symmetric Hessian (`dual::Jet2`, 55 numbers instead
of 100 for nested duals).

### Continuum moduli and elastic waves

`elastic-moduli` and `christoffel` start from the analytic tensor selected
by `ions` (relaxed by default) and `coefficients` (stress-strain `birch` by
default, or `lagrangian`), in GPa. `periodic::elastic::elastic_moduli`
returns the Voigt, Reuss and Hill bulk and shear moduli, the Hill Young
modulus and Poisson ratio, the universal anisotropy index
`5 G_V/G_R + K_V/K_R - 6`, the compliance and the eigenvalues of the
symmetrized tensor (Born stability when all are positive). The density uses
isotope-averaged atomic masses; the isotropic P and S wave speeds are those
of the Hill aggregate and the Debye temperature follows from the mean sound
velocity `v_m = [(2/v_s^3 + 1/v_p^3)/3]^(-1/3)` and the atom number density.
`periodic::elastic::christoffel` solves
`det(C_ijkl n_j n_l - rho v^2 delta_ik) = 0` for any direction and returns
ascending phase velocities with their polarizations; a negative velocity
marks an unstable direction. Closed-form oracles: an isotropic tensor
(`K = lambda + 2 mu/3`, `G = mu`, Voigt = Reuss, `v_p`, `v_s` in every
direction) and a cubic tensor (`G_R = 5(C11 - C12)C44/(4C44 + 3(C11 - C12))`,
`[100]` and `[111]` velocities); the Lagrangian and Birch relations are
checked on a synthetic hydrostatically stressed strain Hessian.

## 14. Phonon dispersion and lattice thermodynamics

`phonon-bands` and `phonon-thermo` obtain phonons at arbitrary wave
vectors from the analytic Gamma-point force constants of an
`N1 x N2 x N3` supercell (section 12). The supercell is sampled with the k
mesh divided by the supercell, so it samples the same Brillouin zone as
the primitive cell; the mesh must be divisible. Supercell atoms are
ordered with the primitive atom fastest, then the cells along `a1`, `a2`,
`a3` (`periodic::phonon_bands::build_supercell`).

`ForceConstantModel::from_supercell_hessian` assigns each primitive pair
`(k0, k'n)` to its shortest supercell images `n.A + L.A_s` (equal weights on
ties within 1e-8 relative) and

```text
D_(ka),(k'b)(q) = sum_(n,L) w Phi_(ka),(k'b)(n) exp(i q . (n.A + L.A_s)) / sqrt(m_k m_k')
```

is exact at every wave vector commensurate with the supercell, where all
image phases are one, and a Fourier interpolation elsewhere. Masses are
the isotope-averaged atomic masses in electron masses; frequencies are in
cm^-1 with imaginary modes negative. No acoustic sum rule is imposed: the
raw residual `max |sum_(k',n) Phi(k0; k'n)|`, the force-constant asymmetry
and the Hermiticity defect of the raw dynamical matrix are reported.

`phonon-thermo` evaluates per primitive cell, on a Gamma-centred uniform q
mesh, the harmonic zero-point energy, `U = sum (w/2 + w n)`,
`F = sum [w/2 + kT ln(1 - exp(-w/kT))]`, `S = (U - F)/T` and the
constant-volume heat capacity, averaged over the mesh. Modes at or below
`mode_cutoff_cm1` (the Gamma acoustic modes and imaginary modes) are left
out and counted (`excluded_modes`, `imaginary_modes`).

Scope follows the selected Gamma Hessian provider. The supercell
repeats the primitive charge and spin imbalance with the same number of
replicas used for its atoms. The supported 3D collinear double-hybrid
path includes relaxed parent orbital and PT2 amplitude response and
supports a uniformly compensated charged supercell. The
[Gamma phonon ledger](response.md#r020-dynamical-matrix-phonons) records
the model limits and distinguishes fixed-operator and relaxed-spectrum
checks.

The interpolated bands use the supercell force-constant model. Its
long-range dipole-dipole correction is a separate directional calculation
through the supported [crystal IR and LO-TO routes](response.md#r024-crystal-ir-loto).
The supercell result and the analytic finite-q density response use
different perturbation models. For the dispersion, thermodynamic and
mode-derivative checks see the
[phonon and quasi-harmonic ledger](response.md#r030-gruneisen-quasiharmonic).

## 15. Equation of state and quasi-harmonic thermodynamics

`equation-of-state` fits the SCF energies of isotropically scaled cells
(`scales` are linear factors; ions at fixed fractional coordinates or
relaxed in every cell) by the third-order Birch-Murnaghan form. That form
is a cubic polynomial in `s = (V/V_r)^(-2/3)`, so the least-squares fit is
linear and exact for Birch-Murnaghan data (`periodic::eos`):

```text
E_V = P1 s',  E_VV = P2 s'^2 + P1 s'',  E_VVV = P3 s'^3 + 3 P2 s' s'' + P1 s'''
s' = -2s/(3V),  s'' = 10s/(9V^2),  s''' = -80s/(27V^3)
p = -E_V,  B = V E_VV,  B' = -1 - V E_VVV / E_VV
```

with `P_k = d^k E/ds^k`. The analytic pressure `-tr(sigma)/3` of every
scaled cell (section 11) is reported beside `-dE/dV` of the fit, and the
fitted bulk modulus at every volume is reported for comparison with
`(B11 + 2 B12)/3` of the analytic stress-strain coefficients of a cubic
cell.

`quasi-harmonic` is a fitted volume-grid model. On every scaled cell it
computes the static energy per primitive cell (the supercell energy over
the number of cells, at the same Brillouin-zone sampling) and the harmonic
thermodynamics of section 14; at every temperature it fits
`E(V) + F_vib(V, T)` by the Birch-Murnaghan form for `V(T)` and the
isothermal `B_T`, and fits the vibrational entropy and heat capacity by
least-squares quadratics in `V`:

```text
alpha = (dS/dV)_T / B_T,   C_p = C_v + V T alpha^2 B_T,   gamma = alpha B_T V / C_v
```

The grid must bracket every `V(T)` (refused otherwise). States at a
pressure and mode Grueneisen parameters of the fitted volume-grid model are
described in section 32; analytic mode Grueneisen parameters need the strain
derivatives of the force constants (third derivatives) and are separate
work.

The equation-of-state tests compare the fitted pressure and bulk modulus
with analytic stress and elasticity. The [quasi-harmonic ledger](response.md#r030-gruneisen-quasiharmonic)
lists volume-grid and mode-derivative checks and their coverage.

## 16. Dielectric tensor and Born effective charges

`dielectric-tensor` and `born-charges` evaluate the electronic (clamped-ion)
dielectric tensor and the Born effective charges of an insulating LDA/GGA
reference by density-functional perturbation theory at fixed macroscopic
field (`periodic::dielectric`). The position operator acts on the occupied
Bloch states as the covariant k derivative. With the Fock image blocks
`H(T)` of the converged density (kinetic, Ewald nuclear attraction,
screened Hartree and the XC potential on the SCF quadrature; their Bloch
sums reproduce the SCF Fock, reported as `fock_image_residual`), the overlap
image blocks, the dipole image blocks `D_a(T) = <m(0)| r_a |n(T)>` and
`d_a M(k) = sum_T i T_a exp(ik.T) M(T)`:

```text
U^ka_ai = -[C_v^+ (d_a H - e_i d_a S) C_o]_ai / (e_a - e_i)
R_a     = C_v^+ (D_a + i d_a S) C_o + i U^ka
(e_a - e_i) U^Eb + [C_v^+ G(P^Eb) C_o] = -R_b        (H' = +E.r)
chi_ab  = -(1/(V Nk)) sum_(k,s) 2 Re Tr[U^Eb^+ R_a],   eps_inf = 1 + 4 pi chi
```

`R_a` is independent of the dipole origin because `C_v^+ S C_o = 0`. The
Born charges are the displacement derivatives of the Berry-phase
polarization `P_a = -(1/(V Nk)) sum Tr A_a` with the Berry connection
`A_a = C_o^+ (S i d_a C_o + (D_a + i d_a S) C_o)`. With the self-consistent
displacement amplitudes `U^x`, `S~^x = C_o^+ S^x C_o`, the moving-basis
dipole derivatives `D^x_a` and the k derivative of `S^x`, integrating the k
derivative of the response by parts over the zone gives

```text
d Tr A_a = 2 Re Tr[U^x^+ R_a] - Re Tr[S~^x C_o^+ (D_a + i d_a S) C_o]
         + Re Tr[i C_o^+ S^x C_v U^ka] + Re Tr[C_o^+ (D^x_a + i d_a S^x) C_o]
Z*_(k,ab) = Z_k delta_ab - Nk^-1 sum_(k,s) d Tr A_a / d u_(k,b)
```

The gauge part of the occupied k derivative cancels between the
overlap-response and skeleton terms; the imaginary parts of the zone sum
cancel (`imaginary_residual`). Like the Berry phase, the zone sum converges
to the continuum value as the k mesh is refined. A rigid translation leaves
the AO coefficients unchanged and moves the dipole operator by the overlap,
so `sum_k Z*_k = 0` holds to the translation invariance of the response
(`born_sum_rule`).

Scope: insulating LDA/GGA references; Fermi-Dirac references, exact
exchange, meta-GGA functionals and ECP cells are refused. The static
(relaxed-ion) dielectric tensor, crystal infrared intensities and the LO-TO
splitting follow from these quantities and the Gamma force constants and
are separate work.

Acceptance (`periodic::dielectric::tests`, skewed H2 chain, SVWN5/STO-3G,
overlap cutoff 12 Bohr, grid level 2):

- The Fock image blocks reproduce the SCF Fock to 3.1e-14 (4x1x1 mesh) and
  3.8e-15 (8x1x1).
- ddk: central differences of the occupied projector `C_o C_o^+` from Fock
  and overlap Bloch matrices rebuilt at `k +- h` (4x1x1 mesh, second k point,
  h = 0.04..0.005 1/Bohr) against `C_v U C_o^+ + h.c. - C_o (C_o^+ d S C_o) C_o^+`:
  p = 2.005566, 1.998656, 1.998743 along x, y, z (smallest R^2 0.99999793).
- Long-wavelength limit (8x1x1 mesh): the macroscopic dielectric function
  `1/eps^-1_00(q)` of the independent finite-q density response with all
  local fields (reciprocal cutoff 8 1/Bohr) against the DFPT
  `eps_xx = 3.428624` along `q = q x`: errors 0.07325, 0.01873, 0.004689,
  0.001152 over q = 0.02..0.0025 1/Bohr, p = 1.997115, R^2 = 0.99995783.
  Over q = 0.08..0.01 the errors were 0.7935, 0.2679, 0.07325, 0.01873
  (halving ratios 2.96, 3.66, 3.91; p = 1.808552, R^2 = 0.99742019): the
  largest q still carries the q^4 term, so the fit uses the finer interval.
- The chain is centrosymmetric with inversion exchanging the two atoms, so
  its Born charges vanish by symmetry; the computed charges are below 2.4e-6
  with a sum-rule residual of 6.7e-7 and an imaginary residual of 1.0e-15.
- Polar modes (closed form): a diatomic model with charges +-Z along x and
  one spring reproduces `eps_0 = eps_inf + 4 pi Z^2/(V k)`,
  `w_LO^2 = w_TO^2 + 4 pi Z^2/(V eps_inf mu)`, the Lyddane-Sachs-Teller
  relation `w_LO^2/w_TO^2 = eps_0/eps_inf` to 1e-12, transverse modes for a
  wave vector normal to the dipoles, and the infrared intensity.

The [Born-charge ledger](response.md#r023-born-charge-berry-polarization)
records the response providers and checks beyond the limits above.
[Crystal IR and LO-TO](response.md#r024-crystal-ir-loto) has its own
directional and dimensional conventions.

## 17. Continuum rock physics and material cards

`geomech` turns computed (or supplied) mineral properties into continuum
parameters. Moduli are in GPa, densities in kg/m^3, porosity is a
fraction; physical inputs have no defaults.

| Calculation | Content |
|---|---|
| `aggregate-moduli` (data) | Isotropic multiphase aggregate: Voigt, Reuss and Hill averages; Hashin-Shtrikman bounds in Berryman's form `K = Lambda(G_max/min)`, `G = Gamma(zeta(K, G)_max/min)`, `zeta = G/6 (9K + 8G)/(K + 2G)`; the self-consistent estimate for spherical grains by damped fixed-point iteration (reported only when it converges) |
| `rock-physics` (data) | Dry frame: critical porosity `K_dry = K_s (1 - phi/phi_c)`, Hertz-Mindlin contact theory, the Hashin-Shtrikman upper bound with empty pores, dry spheroidal pores of one aspect ratio (`dem`, `kuster-toksoz`, `self-consistent`), or given values; Gassmann `K_sat = K_dry + alpha^2 M` with the Biot coefficient `alpha = 1 - K_dry/K_s` and modulus `1/M = phi/K_f + (alpha - phi)/K_s`; Skempton `B = alpha M / K_sat`; drained and undrained Poisson ratios; bulk density and saturated wave speeds; the Biot characteristic frequency `eta phi / (2 pi rho_f kappa)`; the anisotropic Brown-Korringa tensor from mineral and dry stiffness tensors; means and Hashin-Shtrikman bounds of the thermal conductivity |
| `material-card` (periodic) | The analytic elastic tensor (relaxed or clamped ions, stress-strain or Lagrangian coefficients) with compliance, orthotropic engineering constants `E_i = 1/S_ii`, `nu_ij = -S_ji/S_ii`, `G_23, G_13, G_12`, optionally rotated to material axes by the Bond transformation, Hill moduli, density, wave speeds, the reference stress, conventions and residuals, as JSON or CSV |

`geomech::adiabatic_constants` converts isothermal to adiabatic constants,
`C^S = C^T + (T V/C_V) beta beta^T` with `beta = C^T alpha`.

Spheroidal pores (`geomech::inclusions`). An inclusion of moduli
`(K_i, G_i)` and aspect ratio `a` (oblate `a < 1`, prolate `a > 1`) in a
matrix `(K_m, G_m)` takes the strain `T e0`,
`T = [I + S C_m^-1 (C_i - C_m)]^-1`, with Eshelby's tensor `S` of the
spheroid. Averaged over orientations, `P = T_iijj / 3` and
`Q = (T_ijij - T_iijj / 3) / 5` (Berryman's closed form in `theta(a)` and
`f(a)`; the sphere gives `P = (K_m + 4 G_m/3)/(K_i + 4 G_m/3)` and
`Q = (G_m + zeta_m)/(G_i + zeta_m)`). The three frames use them as follows:

| Frame | Equations |
|---|---|
| `kuster-toksoz` | `(K - K_m)(K_m + 4 G_m/3)/(K + 4 G_m/3) = sum x_i (K_i - K_m) P_i` and `(G - G_m)(G_m + zeta_m)/(G + zeta_m) = sum x_i (G_i - G_m) Q_i` (dilute inclusions) |
| `dem` | `dK/dy = (K_i - K) P(K, G)/(1 - y)`, `dG/dy = (G_i - G) Q(K, G)/(1 - y)` from the mineral at `y = 0` to the porosity, integrated by RK4 with step doubling to 1e-10 relative |
| `self-consistent` | `sum x_i (K_i - K) P_i(K, G) = 0`, `sum x_i (G_i - G) Q_i(K, G) = 0` over spherical mineral grains and the pores, by damped fixed-point iteration; refused when it does not converge (porosity above the percolation threshold of the pore shape) |

The driver uses dry pores (`K_i = G_i = 0`) of one `pore_aspect_ratio`; the
library functions take any inclusion moduli and several inclusion families.
With `mineral_conductivity_w_mk` and `fluid_conductivity_w_mk` the report
adds the arithmetic, harmonic and geometric means and the Hashin-Shtrikman
bounds `[sum x_i/(l_i + 2 z)]^-1 - 2 z` (`z` the smallest or the largest
phase conductivity) of the saturated rock.

Acceptance of the inclusion code (`geomech::inclusions::tests` and
`p73_geomechanics`, measured): the closed-form `P` and `Q` agree with an
independent numerical Eshelby tensor (Mura's depolarization integrals by
Gauss-Legendre quadrature and a 9x9 solve for `T`) to 1.7e-13 relative over
32 cases (quartz or calcite matrix; dry, water, quartz and calcite
inclusions; `a` from 0.01 to 20); the sphere, penny-crack (`a` = 1e-5) and
needle (`a` = 1e5) limits are reached within 1e-3; the RK4 global error of
the DEM integration has p = 4.044655 (R^2 = 0.99999167, porosity steps
0.05 to 0.00625); DEM and Kuster-Toksoz differ by `O(phi^2)` (p = 1.958645,
R^2 = 0.99995962, `phi` = 0.04 to 0.005); the self-consistent estimate for
spheres reproduces the aggregate estimate to 1e-10; for dry spherical
pores Kuster-Toksoz coincides with the upper Hashin-Shtrikman bound while
DEM and the self-consistent estimate lie below it (driver, quartz,
`phi` = 0.15: `K_dry` = 28.7319 GPa for the bound and Kuster-Toksoz,
28.2789 GPa for DEM, 27.6499 GPa for the self-consistent estimate); the
conductivity means and bounds are ordered and the upper bound equals the
Maxwell-Garnett formula.

Acceptance (`geomech::tests`, `p73_geomechanics`, `p71_equation_of_state`):
the bound ordering `Reuss <= HS- <= self-consistent <= HS+ <= Voigt` for a
three-mineral aggregate and the collapse of all estimates for one phase;
Gassmann in its classical form, its limits (`phi -> 0` gives the mineral,
`K_f -> 0` the dry frame) and the Biot identities
`K_sat = K_dry + alpha^2 M`, `B = alpha M / K_sat`; Brown-Korringa reduces
to Gassmann for isotropic tensors; isotropic engineering constants and
rotation invariance; the adiabatic bulk modulus `K_S = K_T (1 + alpha_V gamma T)`;
refusal of missing physical inputs; the CLI route without a structure file;
the material card against the rotated analytic tensor and its CSV file.

Not yet included: clay-water interfaces (swelling pressure, hydration,
adsorption, exchange and surface energies, interlayer sliding), the lattice
thermal conductivity and temperature-dependent card entries.

## 18. Fermi-Dirac (Mermin) response

The coupled-perturbed solve of section 12 assumes integer occupations. For a
Fermi-Dirac reference (`electronic_temperature` > 0, chemical potential
`mu` per spin) the occupations `f_n = 1/(1 + exp((e_n - mu)/kT))` change with
the perturbation, so the response is carried by all orbital pairs instead of
the virtual-occupied block (`periodic::phonon`: `MetalChannel`, `Lindhard`,
`MetalState`, `MetalOperator`, `ResponseSolver::solve_metal`). Per spin and
k point the generalized orbitals `C` and energies `e_n` of the converged
Fock and overlap give `f'_n = df/de = -f_n (1 - f_n)/kT`. With the
orbital-basis first-order Fock `V = C^+ (F^x + G[P^x]) C` (skeleton Fock plus
the self-consistent kernel response) and overlap `S~ = C^+ S^x C`,
orthonormality of the orbitals and stationarity of the Mermin free energy at
fixed temperature give the first-order density and the first-order
energy-weighted density in the orbital basis:

```text
P~_nm = D_nm V_nm - G_nm S~_nm - delta_nm f'_n mu^x
W~_nm = G_nm V_nm - X_nm S~_nm - delta_nm e_n f'_n mu^x
```

with the divided differences of the occupations (the Lindhard factors)

```text
D_nm = (f_n - f_m)/(e_n - e_m)
G_nm = (e_n f_n - e_m f_m)/(e_n - e_m)
X_nm = (e_n^2 f_n - e_m^2 f_m)/(e_n - e_m)
```

For `|e_n - e_m| < 1e-9` Hartree the factors are replaced by their
degenerate limits (with the mean of the two orbitals):
`D = f'`, `G = f + e f'` and `X = 2 e f + e^2 f'`. The AO matrices are
`P^x = C P~ C^+` and `W^x = C W~ C^+`. The shift of the Fermi level that keeps
the number of electrons of every spin channel fixed follows from
`sum_(k,n) (P~_nn + f_n S~_nn) = 0`, because `N = Tr[P S]`:

```text
mu^x = sum_(k,n) f'_n (V_nn - e_n S~_nn) / sum_(k,n) f'_n
```

which is linear in `V` and `S~`. Writing `P~ = P~_lin(V) + P~_const`, where
`P~_const` is the part of `P~` produced by `S~` alone (with the shift
`mu^x` evaluated from `S~` only), the self-consistent amplitude solves

```text
V - C^+ G[C P~_lin(V) C^+] C = F~^x + C^+ G[C P~_const C^+] C
```

over all orbital pairs and all k points, where `G` is the screened Hartree
plus semilocal XC kernel of the insulating solve. The operator is a general
(non-symmetric in the real packing) linear operator, so it is solved by
restarted GMRES (`response::krylov`), the unknowns being the real and
imaginary parts of the `n x n` blocks of every k point and spin. After the
solve `P^x` is projected onto the time-reversal-symmetric subspace, and the
response carries the total first-order Fock `F^x + G[P^x]`, `P^x` and `W^x`
in the same fields as the insulating response, so the pairing is unchanged:

```text
H_xy = E_skel,xy + Nk^-1 sum_(k,s) (P^y . F^x - W^y . S^x)(k)
```

The force constants, electron-phonon vertices and everything built on them
(sections 12 to 15 and 19) therefore use one assembly for both
occupations. In the gapped limit (`kT` small against the gap) `f'` vanishes,
`D_nm`, `G_nm` and `X_nm` of occupied-virtual pairs reduce to `+-1`,
`+-e_occupied` and `+-e_occupied^2` over the gap, while for two occupied
orbitals `D = 0`, `G = 1` and `X = e_n + e_m`, so that
`P~_oo = -S~` and `W~_oo = V - (e_n + e_m) S~`, the occupied-occupied terms of
the insulating density and energy-weighted density; the unknown reduces to
the virtual-occupied block and the insulating response is recovered.

Scope: LDA and GGA references with Fermi-Dirac occupations; exact exchange, meta-GGA functionals and ECP cells are refused
as in section 12. The reference chemical potential must be available for
both spin channels. The finite-q thermal (Mermin) density response is
described in [response.md](response.md).

The [Gamma phonon ledger](response.md#r020-dynamical-matrix-phonons)
records the two 3000 K LDA force-constant column fits, the failed coarse
interval and the separate gapped thermal limit. The column tests verify
the recorded fixture components.

## 19. Electron-phonon coupling and superconducting Tc

`electron-phonon` evaluates the electron-phonon vertex, the coupling
constants and the critical temperature of a metallic or insulating
reference from the analytic displacement responses of an `N1 x N2 x N3`
supercell (`periodic::electron_phonon`). The Gamma point of the supercell
contains every primitive wave vector commensurate with it, and the supercell
k mesh folds the primitive one (the mesh must be divisible, as in section
14), so the Gamma vertices of the supercell are the electron-phonon matrix
elements of the primitive cell at those wave vectors. With the
self-consistent first-order Fock `F^x = F^x_skel + G[P^x]` of every Cartesian
displacement (section 12 or 18), the overlap derivative `S^x`, the
mass-weighted mode patterns `e_nu` of frequency `w_nu` and the orbitals `C`:

```text
g^nu_mn(k) = [C^+ (dF^nu - (e_m + e_n)/2 dS^nu) C]_mn / sqrt(2 w_nu)
dF^nu = sum_x e_nu(x) F^x / sqrt(M_x),   dS^nu likewise
```

The average of the two band energies makes the vertex Hermitian for the
non-orthogonal basis, and its diagonal is the eigenvalue derivative
`d e_n/d x = [C^+ (F^x - e_n S^x) C]_nn` (Hellmann-Feynman), reported as
`eigenvalue_derivatives` (Hartree/Bohr, `[spin][k][band][3 atom + axis]`).
Masses are the isotope-averaged atomic masses in electron masses; modes at
or below `mode_cutoff_cm1` (the Gamma acoustic modes and imaginary modes)
are left out and counted in `excluded_modes`.

The electronic delta functions are Gaussians of width `electron_smearing`
(Hartree) around the chemical potential (the mid-gap for integer
occupations). With `N_F = Nk^-1 sum_(k,n) d(e_n - mu)` per spin, the
double-delta coupling of every mode is

```text
lambda_nu = 2 / (N_F w_nu Nk) sum_(k,m,n) |g^nu_mn|^2 d(e_m - mu) d(e_n - mu)
lambda    = sum_nu lambda_nu
omega_log = exp(sum_nu lambda_nu ln w_nu / lambda)
omega_2   = sqrt(sum_nu lambda_nu w_nu^2 / lambda)
a2F(w)    = 1/2 sum_nu w_nu lambda_nu d(w - w_nu)
```

per spin channel, averaged over the two spins, with a Gaussian of width
`phonon_smearing_cm1` for the phonon delta functions in `a2F` (so that
`2 int a2F/w dw = lambda`). `fermi_dos` is reported in states per Hartree per
spin per supercell, `omega_log_k` and `omega_2_k` in K.

The critical temperature is reported in three forms for the Coulomb
pseudopotential `mu_star` (a semi-empirical input, default 0.1):

- McMillan, `T_c = omega_log/1.2 exp(-1.04 (1 + lambda)/(lambda - mu* (1 + 0.62 lambda)))`,
  with `omega_log` as the prefactor frequency (`tc_mcmillan_k`);
- Allen-Dynes, the same expression multiplied by `f1 f2` with
  `f1 = [1 + (lambda/L1)^(3/2)]^(1/3)`, `L1 = 2.46 (1 + 3.8 mu*)`,
  `f2 = 1 + (omega_2/omega_log - 1) lambda^2/(lambda^2 + L2^2)` and
  `L2 = 1.82 (1 + 6.3 mu*) omega_2/omega_log` (`tc_allen_dynes_k`); both are
  null when the exponent denominator is not positive;
- the isotropic Migdal-Eliashberg equations on the imaginary axis
  (`tc_eliashberg_k`, `periodic::electron_phonon::eliashberg_tc`).

The Coulomb pseudopotential of the Eliashberg solve refers to the hard
Matsubara cutoff `eliashberg_cutoff_cm1`: `mu*` acts unscaled on every
frequency below it, and the gap vanishes above it. The renormalization `Z` is
not cut off: its sum over the frequencies beyond the cutoff (where `Delta = 0`)
is added as the normal-state tail, evaluated from the Lorentzian sums with an
Euler-Maclaurin remainder (`Spectrum::tail`). Truncating `Z` at the cutoff
raises `T_c` by 1.3 % at 10 w and 0.26 % at 20 w for `lambda` = 0.6; the
oracle sums `Z` untruncated. At fixed `mu*` the critical temperature therefore rises
with the cutoff. The independent pure-Python oracle
`validation/validate_eliashberg_oracle.py` (all Matsubara indices of both
signs, untruncated `Z`, power iteration; no dft-rs code) gives, for one
Einstein mode `w` and `mu* = 0.1`:

| lambda | T_c/w at 10 w | 20 w | 40 w | Allen-Dynes T_c/w | ratio at 10 w / 20 w / 40 w |
|---|---|---|---|---|---|
| 0.6 | 0.027838 | 0.029053 | 0.030209 | 0.023425 | 1.188 / 1.240 / 1.290 |
| 1.0 | 0.079912 | 0.081489 | 0.083071 | 0.073169 | 1.092 / 1.114 / 1.135 |

The solver reproduces these values to a relative 2e-5 or better. At `mu* = 0` the cutoff dependence is below
0.05 % and the solve agrees with Allen-Dynes within 4 %: the oracle ratios at
20 w are 1.0358, 1.0195 and 1.0089 for `lambda` = 0.6, 1.0 and 1.5. The
McMillan, Allen-Dynes and Eliashberg values for `mu* > 0` are consequently
not interchangeable without rescaling the pseudopotential between cutoffs,

```text
mu*(w_c') = mu*(w_c) / (1 + mu*(w_c) ln(w_c / w_c'))
```

and the Allen-Dynes `mu*` is the one defined at the conventional cutoff of
about ten times the highest phonon frequency (the default of
`eliashberg_cutoff_cm1`).

The Eliashberg solve uses the discrete spectrum, so that
`lambda(n) = sum_nu lambda_nu w_nu^2/(w_nu^2 + (2 pi n T)^2)`. With the
positive Matsubara frequencies `w_n = pi T (2n + 1)` up to the cutoff
`eliashberg_cutoff_cm1` (default ten times the highest included mode), the
linearized gap equation is

```text
rho Delta_n = sum_m pi T [lambda(n - m) + lambda(n + m + 1) - 2 mu*] Delta_m / (w_m Z_n)
Z_n = 1 + (pi T / w_n) sum_m [lambda(n - m) - lambda(n + m + 1)]
```

The kernel is symmetrized with `diag(sqrt(Z_n w_n))` and diagonalized densely;
`T_c` is the temperature at which the largest eigenvalue `rho` equals one,
found by bisection in `ln T` after bracketing from above. The search floor is
`eliashberg_t_min_k` (default 0.01 K); when `rho < 1` at the floor, or no
mode is included, no solution is reported (`null`). The number of Matsubara
frequencies is capped at 4000.

The unit tests of `eliashberg_tc` use a single Einstein mode: the
strong-coupling limit `T_c -> 0.1827 w sqrt(lambda)` for `mu* = 0`; the
values of the independent oracle (`lambda` = 0.6, 1.0, 1.5 at `mu* = 0`, and
`lambda` = 0.6, 1.0 at `mu* = 0.1` for the cutoffs 10 w and 40 w, within 0.5 %)
together with a ratio to Allen-Dynes between 1.0 and 1.06 at `mu* = 0`; and
the absence of a solution at weak coupling (`lambda` = 0.05, `mu* = 0.2`).
The vertex is checked through its diagonal: the eigenvalue
derivatives `d e_n/d x` against four-step fits of central differences of
the band energies of displaced cells (`p75_electron_phonon`, metallic H2
chain with Fermi-Dirac occupations), together with the identity
`sum_nu lambda_nu = lambda`.

Model assumptions: isotropic (k- and band-averaged) coupling, a
semi-empirical `mu*`, a linearized gap equation (no gap function and no
renormalized quasiparticle properties below `T_c`), Gaussian broadening of
both delta functions, and only the wave vectors commensurate with the
supercell; there is no fine-q interpolation, so `lambda` carries the
sampling error of the electronic and phonon meshes and the smearings.
Anisotropic Eliashberg theory and fine-q interpolation remain open.
This describes the original isotropic path. The
[electron-phonon ledger](response.md#r021-electron-phonon-vertex) records
the additional exchange-enabled checks and their model bounds. Separate
[superconducting DFT routes](response.md#r050-scdft-gap-anomalous-density)
cover static pairing and supplied-band gap, number and Nambu-density
subsets.

The [electron-phonon regression](response.md#repaired-metallic-electron-phonon-vertex-regression)
records four passed tests, including nine physical band-energy derivative
fits and the resolved metallic LDA step interval. The
[electron-phonon ledger](response.md#r021-electron-phonon-vertex) records
model and sampling coverage.

### Static microscopic SCDFT Coulomb screening

`electron-phonon` with `scdft_pairing=true` exposes the normal bands,
Pulay-complete Gamma-supercell phonon vertices and Coulomb matrix through
`scdft_inputs`. The default `scdft_screening=supplied` requires the
explicit `screened_pair_coulomb` matrix. Select `scdft_screening=rpa`
or `rpa-neutral` to generate the Coulomb matrix from the 3D normal
reference. These modes require finite electronic temperature, spin-degenerate
time-reversal-symmetric bands and the complete uniform reference k mesh.
They use `response_reciprocal_cutoff` in inverse Bohr in Rust, the CLI and
native Python. ASE mode uses inverse Angstrom for that control.

For each transfer q of the k-point difference mesh, define the
dimensionless transition form `F_G = <a,k+q|exp(i(q+G).r)|b,k>` and
`rho_G = conj(F_G)`. The analytic Gaussian Bloch plane-wave integrals
also serve the finite-q density solver. The independent susceptibility
is the sum of `rho rho^dagger` times
`2 (f(e_a)-f(e_b))/(V Nk (e_a-e_b))`, using the shared Fermi derivative
at degeneracy. Solve `W = (diag(|q+G|^2/(4 pi))-chi0)^-1` on the
complete reciprocal sphere and project `rho^dagger W rho / V` on each
time-reversed singlet pair. The generated real Coulomb matrix is in
Hartree in the same k-major, band-minor order as `scdft_inputs`.
The SCDFT gap solver applies the k weights separately. This is the static
full local-field contraction in
[Floris et al., Eq. (26)](https://www-old.mpi-halle.mpg.de/mpi/publi/br/gross_persoenlich/FSLPLMFGCM07.pdf).

- `rpa` keeps the q=0, G=0 component using the finite-temperature
  static long-wavelength response at fixed chemical potential. The
  homogeneous potential is the canonical overlap operator and its
  contribution to `-chi0[0,0]` is the normal electronic compressibility
  per volume. The microscopic inverse includes the head and local-field
  wings. Zero compressibility in this mode returns a named unsupported
  error.
- `rpa-neutral` constrains the q=0 density response to fixed electron
  number and removes the constant Coulomb mode of the neutral finite cell.
  Its susceptibility body applies the Schur projection
  `chi0[body,body] - chi0[body,0] chi0[0,body] / chi0[0,0]` when the
  number response is nonzero. The other q transfers use the same
  microscopic screening as `rpa`.

The report gives q samples, reciprocal modes and cutoff, normal electron
number and occupation defects, spin and time reversal defects, raw and
scaled inverse residuals, and pair imaginary and reciprocity defects.
It lists `gamma_compressibility` in Bohr^-3/Hartree and, for `rpa`,
`gamma_head = W[0,0]/V` in Hartree. The Fermi-level crossing flag checks
whether a sampled band spans the chemical potential. A positive thermal
compressibility can also come from thermally activated occupations when
the sampled spectrum has a gap. Converge the normal occupations, k mesh,
reciprocal cutoff, basis and real-space cutoffs for an application.
An insulating long-wave q-cell average remains open. Coulomb screening
uses the complete commensurate difference mesh, and the phonon contribution
still uses only the available Gamma-supercell modes. The isotropic
Eliashberg calculation has its separate `mu_star` control.

### Nonlinear gap and real-axis continuation

`periodic::eliashberg` solves the nonlinear isotropic equations for the same
discrete spectrum, Matsubara grid, cutoff and `mu*` treatment as
`eliashberg_tc` (the spectrum `lambda(n)`, the grid `w_n = pi T (2n + 1)` up
to the cutoff and the kernel are shared code; `mu*` acts unscaled on every
frequency below the cutoff). With the even-gap symmetry
`Delta_(-n-1) = Delta_n`:

```text
Z_n = 1 + (pi T / w_n) sum_m [lambda(n - m) - lambda(n + m + 1)] w_m / sqrt(w_m^2 + Delta_m^2)
Z_n Delta_n = pi T sum_m [lambda(n - m) + lambda(n + m + 1) - 2 mu*] Delta_m / sqrt(w_m^2 + Delta_m^2)
```

For `Delta -> 0` the square roots become `w_m`, the weight of the
renormalization sum becomes one and the second equation becomes
`Z_n Delta_n = sum_m pi T K_nm Delta_m / w_m`, which is the linearized gap
equation above with `rho = 1` and the same `Z_n`, term by term. The nonlinear
gap therefore vanishes at the temperature where the largest eigenvalue of the
linearized kernel is one. `eliashberg_gap` iterates `Delta <- F(Delta)` from a
constant guess (default 0.2 times the highest included mode) with linear
mixing (`mixing`, 0.3) and Anderson acceleration (`anderson_depth`, 5; zero
gives plain mixing), until the relative max-norm change is below `tolerance`
(1e-10) or `max_iter` (2000) iterations are used; `converged`, `iterations`
and `residual` are reported. The cost of one iteration is `O(N^2)` for `N` at
most 4000 Matsubara frequencies. Above `T_c` the iteration collapses: when
`max |Delta|` falls below `1e-14` times the highest mode the gap is returned as
exactly zero with `converged = true`, the normal state.

`eliashberg_gap` is the `gap_temperatures_k` output of the driver: for every
temperature `eliashberg_gaps` holds `Delta(i w_0)`, `Z(i w_0)`, the iteration
record and `delta_real_hartree`, the real part of `Delta(omega + i eta)` at
`omega = 0.1 Delta(i w_0)` and `eta = 1e-8` times the highest included mode
(null for `Delta = 0`).

The gap on the real axis follows from the Matsubara values by the
Vidberg-Serene continued fraction (`pade_continuation`, `real_axis_gap`): the
Thiele coefficients `a_p = g_p(z_p)` of
`g_p(z) = (g_(p-1)(z_(p-1)) - g_(p-1)(z)) / ((z - z_(p-1)) g_(p-1)(z))` are
evaluated with the three-term recursion `A_(n+1) = A_n + (z - z_(n-1)) a_n A_(n-1)`
(`B` likewise, `f = A_N/B_N`). The interpolant passes exactly through the
`pade_points` (default 64) lowest Matsubara values, which carry the
double-precision noise of the iteration; the continued fraction is a known
ill-conditioned continuation, so the point count is a control and the
real-axis gap is only as reliable as the resulting plateau against the
number of points. A non-finite coefficient is refused with a request for
fewer points.

Limiting checks (`periodic::eliashberg::tests`): the interpolant reproduces
an exact rational function at off-axis points; for an Einstein mode the gap
is zero above the linearized `T_c`, positive below it, and obeys the
mean-field law `Delta_0^2 ~ (T_c - T)` close to `T_c` (log-log fit);
in weak coupling (`lambda` = 0.3, `mu*` = 0) the ratio `2 Delta_0/T_c`
with `Delta_0` from the continuation is near the BCS value 3.53 and agrees
with `Delta(i w_0)`; in strong coupling (`lambda` = 2.5, `mu*` = 0.1) the
ratio `2 Delta(i w_0)/T_c` exceeds the BCS value.

P75B_ACCEPTANCE


## 20. Periodic DFT-D3 dispersion

`periodic-energy`, `stress`, `opt-fixed-cell` and `cell-opt` accept
`dispersion` (alias `disp`: `none`, `d2`, `d3` for zero damping, `d3bj`, or
`chg`) and add the corresponding lattice sum. For D3 the two-body energy
uses the molecular parameters
(`periodic::dispersion`; S. Grimme, J. Antony, S. Ehrlich, H. Krieg, J. Chem.
Phys. 132, 154104 (2010); Becke-Johnson damping: S. Grimme, S. Ehrlich,
L. Goerigk, J. Comput. Chem. 32, 1456 (2011)). The reference C6 values,
covalent radii, `r2r4` and `r0ab` tables, the coordination-number counting
function and the Gaussian C6 interpolation are those of the molecular code
(`src/dispersion/data/*.csv`, provenance in their headers); nothing is
duplicated. The damping parameters come from the functional, as in the
molecular code.

With the atoms `i` at `r_i`, the translations `T` and `v = r_j + T - r_i`,
`r = |v|` (the prime excludes `i = j`, `T = 0`):

```text
CN_i = sum_j sum_T' cn(r),  cn(r) = 1/(1 + exp(-16 (R_ij/r - 1)))      r < cn_cutoff
E    = -1/2 sum_ij C6_ij(CN_i, CN_j) A_ij
A_ij = sum_T' [s6 g6(r) + s8 c8_ij g8(r)]                             r < disp_cutoff
```

with `g_n = 1/(r^n + f^n)`, `f = a1 sqrt(c8_ij) + a2` (BJ) or
`g_n = 1/(r^n + 6 (rs_n r0_ij)^alpha_n r^(n - alpha_n))` (zero damping),
`c8_ij = 3 r2r4_i r2r4_j`. Self-image pairs (`i = j`, `T != 0`) are included
with the same 1/2. Because `C6_ij` does not depend on the image, the lattice
sums are accumulated once per ordered atom pair and every derivative is
closed form:

```text
dE/dx  = -1/2 sum_ij C6_ij dA_ij/dx + sum_i Ebar_i dCN_i/dx
Ebar_i = -sum_j A_ij dC6_ij/dCN_i          (first argument; C6_ij(a,b) = C6_ji(b,a))
dr/d eps_ab = v_a v_b / r                  (x -> (1 + eps) x for atoms and cell)
```

`dC6/dCN` comes from forward-mode duals of the interpolation. The strain
derivative of a distance-only energy is symmetric, so the rotation residual
of the dispersion term vanishes to rounding. The dispersion strain
derivative is reported as the term `dispersion` of
`strain_derivative_terms`, and the energy as `dispersion_energy` (included in
`energy` and `free_energy`). Defaults: `disp_cutoff` = 95 Bohr and
`cn_cutoff` = 40 Bohr; the derivatives are those of the truncated sums (the
pair term at 95 Bohr is below 1e-11 Hartree per pair on the acceptance
fixture). `d2` uses the same 3D pair enumeration with constant coefficients,
published functional `s6` and logistic damping. Section 29 covers the
CHG pair term, D3 three-body ATM and dispersion force constants.
[D2 verification](response.md#periodic-d2-empirical-dispersion)

Acceptance (`periodic::dispersion::tests`, measured; skewed C/N/O cell
`[[9.1,0,0],[1.4,8.6,0],[0.9,1.3,9.8]]` Bohr, PBE parameters):

| check | D3(BJ) | D3(zero) |
|---|---|---|
| molecule in a 60 Bohr box against the molecular energy | 2.2e-19 | -2.5e-21 |
| force atom 1 x, h = 0.02..0.0025 | p = 2.000335, R^2 = 0.99999999 | p = 2.000236, R^2 = 1.00000000 |
| force atom 2 z | p = 2.000402, R^2 = 0.99999999 | p = 2.000391, R^2 = 0.99999999 |
| dE/d eps[0][0], h = 0.004..0.0005 | p = 1.999998, R^2 = 1.00000000 | p = 1.999793, R^2 = 1.00000000 |
| dE/d eps[0][1] | p = 2.000275, R^2 = 1.00000000 | p = 1.999842, R^2 = 1.00000000 |
| dE/d eps[2][1] | p = 2.000093, R^2 = 1.00000000 | p = 2.000101, R^2 = 1.00000000 |
| force sum | 1.7e-20 | 3.4e-21 |
| E(95 Bohr) - E(80 Bohr) | -3.67e-7 | -3.67e-7 |

The pair cutoff convergence is monotone (E at 40, 60, 80 and 95 Bohr:
-2.65591e-3, -2.66108e-3, -2.66233e-3, -2.66270e-3 Hartree for D3(BJ)); one
atom in a cubic cell gives an isotropic strain derivative (off-diagonal
below 1e-20). On the driver (`tests/p76_periodic_dispersion.rs`, hydrogen
chain, PBE/STO-3G, k 2x1x1) `stress` with `d3bj` and `d3` differs from the
calculation without dispersion by exactly the lattice sum: energy within
1.1e-16, forces within 1.0e-17 and strain derivative within 2.7e-17, the
terms add to the total exactly, `phonons`, `elastic` and `dielectric-tensor`
refuse the control, and the CLI route reproduces the Rust driver.


## 21. Band velocities, effective masses and Boltzmann transport

`effective-mass` and `boltzmann-transport` differentiate the converged
Kohn-Sham operator in k (`periodic::band_transport`). The real-space Fock
and overlap image blocks `F_T`, `S_T` of the reference (the same blocks as
the dielectric DFPT of section 16; `fock_image_residual` reports
`max |bloch(F_T) - F_SCF|` on the SCF mesh, 4.1e-15 for the test chain)
give `H(k) = sum_T exp(ik.T) F_T`, `S(k)` and their derivatives
`d_a M = sum_T i T_a exp(ik.T) M_T`, `d_a d_b M = -sum_T T_a T_b exp(ik.T) M_T`
at any k: off the SCF mesh this is the exact non-self-consistent band
structure of the converged density.

For `H c_n = e_n S c_n` with `c_m^+ S c_n = delta_mn`, differentiating the
eigenvalue equation gives the amplitudes `d_b c_n = sum_m c_m U^b_mn`,
`U^b_mn = V^b_mn/(e_n - e_m)` for `m != n` and `U^b_nn = -S~^b_nn/2` from the
normalization, with `V^a_mn = c_m^+ (d_a H - e_n d_a S) c_n` and
`S~^a_mn = c_m^+ d_a S c_n`. Differentiating `v_n,a = V^a_nn` along `k_b`:

```text
d2 e_n/dk_a dk_b = c_n^+ (d_a d_b H - e_n d_a d_b S) c_n - v_a S~^b_nn - v_b S~^a_nn
                   + sum_(m != n) 2 Re[V^a_mn* V^b_mn] / (e_n - e_m)
```

The sum runs over every band of the basis, so the tensor is exact within the
basis (atomic units: velocities in Hartree Bohr, `1/m*` in `1/m_e`). A
cluster `D` of bands within `degeneracy_tolerance` is treated by degenerate
perturbation theory along a unit direction `u`: the velocities are the
eigenvalues of `W = [c^+ (H_u - e S_u) c]_DD`, and the second derivatives of
the branches are the eigenvalues, in each eigenspace of `W`, of

```text
M = [c^+ (H_uu - e S_uu) c]_DD - (W S~ + S~ W) + 2 sum_(m not in D) V_.m V_m. / (e - e_m)
```

(Rellich: branches split at first order keep the diagonal of `M` in the
eigenbasis of `W`; for one band `M` is the formula above). `effective-mass`
refuses a degenerate band without a direction.

`boltzmann-transport` evaluates, on a dense Monkhorst-Pack mesh
`transport_k_mesh`, the rigid-band, constant-relaxation-time integrals

```text
L^(alpha)_ab = g/(V Nk) sum_(n,k) v_a v_b (e - mu)^alpha (-df/de)
sigma/tau = e^2 L0,   S = -(1/(e T)) L0^-1 L1,   kappa_e/tau = (1/T)(L2 - L1 L0^-1 L1)
```

with `g = 2` for a closed-shell reference (per-spin sums otherwise). A
degenerate cluster contributes the basis-independent `sum_ij Re[W^a_ij W^b_ji]`
in place of `v_a v_b`. States with `|e - mu| > delta_window kT` (default 20)
are counted for the carrier density but left out of `L` (their weight is
below `e^-20`). The chemical potentials are shifts from the reference level
(the SCF chemical potential, or the mid-gap of an insulating reference) or
are solved by bisection for requested carrier concentrations. SI factors:
`sigma/tau` in S/(m s) (`e^2 E_h/(hbar^2 a0)` = 1.90164e23), `S` in V/K
(`k_B/e`) and `kappa_e/tau` in W/(m K s) (`E_h^2 k_B/(hbar^2 a0)`). Along a
direction without carriers `L0` is singular and `S`, `kappa_e` are reported
as null; in a nearly insulating direction (the vacuum directions of the test
chain) the off-diagonal Seebeck elements are ill-conditioned and carry no
physical meaning. Model assumptions: rigid bands, energy-independent `tau`,
no Hall or magnetic-field terms; semilocal references only (exact exchange
and ECP cells are refused, as for the dielectric DFPT).

Acceptance (measured):

- `periodic::band_transport::tests` on a parabolic band `e = k^2/(2 m*)`,
  `m* = 0.7` (867152 states): degenerate metal `sigma/tau` against `n/m*`
  ratio 1.000012; Wiedemann-Franz `kappa_e/(sigma T)` against `pi^2/3`
  ratio 0.997815 (Sommerfeld correction of order `(kT/mu)^2`); non-degenerate
  Seebeck -7.502970 `k_B/e` against `-[(e_c - mu)/kT + 5/2]` = -7.5
  (ratio 1.000396).
- `tests/p77_band_transport.rs`, hydrogen chain (SVWN5/STO-3G, k 4x1x1) at
  `k = (0.31, 0.07, 0.05)`/Bohr along `u = (0.8, 0.5, 0.33)/|.|`, central
  differences of the band energies with h = 0.04, 0.02, 0.01, 0.005:
  band 0 velocity p = 2.001031 (R^2 0.99999993), curvature p = 2.000929
  (0.99999994); band 1 velocity p = 1.999521 (0.99999998), curvature
  p = 1.998190 (0.99999978); the mixed element `M_xy` from a four-point
  mixed difference p = 1.999144 (0.99999995).
- Tetragonal neon (a = 5, c = 6.5 Bohr, k 2x2x2): the p_x/p_y pair at Gamma
  is exactly degenerate (the product angular grid keeps the 90-degree
  rotation about z; a cubic cell splits the three 2p bands by 9.6e-6
  Hartree at grid level 2 because the product grid breaks the x/z
  permutation). Degenerate perturbation theory gives zero branch velocities
  (below 1e-19) and second derivatives -0.12515373 and -0.00941925 `1/m_e`
  along `(0.9, 0.35, 0.2)/|.|`; the sorted band energies give p = 1.999496
  and 1.999339 (R^2 0.99999998, 0.99999997).
- Transport driver on the chain at 300 and 1000 K: requested carrier
  concentrations -1e20 and 2e20 cm^-3 are recovered to 1.4e-15 relative,
  `sigma/tau` is exactly symmetric, and the CLI route reproduces the Rust
  inverse-mass tensor exactly.

## 22. Ordinary weak-field Hall transport

`hall-transport` uses the same reference, transport mesh, temperatures,
chemical-potential shifts or carrier concentrations as `boltzmann-transport`.
This is a DC response. The fixed-density Kohn-Sham operator is differentiated analytically in k,
including the moving overlap metric and all interband terms in the isolated
band curvature. This is the ordinary semiclassical Hall response in the
constant-relaxation-time, rigid-band approximation; it does not include
Berry-curvature or coherent interband transport, a quantized Hall plateau,
or the self-consistent change of the reference in a finite magnetic field.
The latter calculations remain separate targets in the coverage ledger.
The present common-driver reference is the 3D all-electron semilocal SCF;
hybrid, ECP and dimensional Hall drivers remain pending. No response is
selected by the registered functional name.

For electron charge `q_e = -1` in atomic units, expansion of the steady
Boltzmann equation

```text
q_e E.v f' + q_e (v x B).grad_k delta_f = -delta_f/tau
```

gives `delta_f_E = -tau q_e E_b v_b f'`. The mixed electric/magnetic term
contains `(v x B).grad_k v_b`; the derivative of `f'` vanishes because
`(v x B).v = 0`. With `epsilon_xyz = +1`, the conductivity derivative is

```text
H_abc = g/(V Nk) sum_(n,k) (-df/de) v_a sum_(d,e) epsilon_dec v_e M^-1_bd
sigma_ab(B) = tau L0_ab + tau^2 sum_c B_c H_abc + O(B^2)
R_abc = d rho_ab/d B_c = -sum_(u,v) (L0^-1)_au H_uvc (L0^-1)_vb
E_a = rho_ab j_b
```

Here `M^-1` is `d2 e/dk dk`, rather than an inverted mass tensor.
Conductivity axes are (current, electric field, magnetic induction);
resistivity axes are (electric field, current, magnetic induction), each
Cartesian x,y,z. Thus a parabolic electron band has `H_xyz = -n/(m_x m_y)`
and `R_yxz = -1/n`. A parabolic valence-band hole reverses both signs.
This fixes the tensor convention independently of any Hall sign convention
in other software. The semiclassical tensor expressions are also described
by [Madsen and Singh (2006), Eqs. 7-14](https://arxiv.org/pdf/cond-mat/0602203).

`results.hall_conductivity_derivative_over_tau_squared` is a 3x3x3 tensor
in S/(m T s^2), with SI factor `e^3 E_h^2 a0/hbar^4`.
`results.hall_coefficient` is `d rho/dB` in m^3/C (ohm m/T), with factor
`a0^3/e`; tau cancels. It is null when the zero-field conductivity cannot
be inverted. No denominator floor, conductivity floor, pseudoinverse or
antisymmetric projection is used. Bands in the actual union of requested
Fermi windows must be isolated within `degeneracy_tolerance`. A contributing
degenerate cluster gives a typed error identifying its k point, spin and
bands, rather than arbitrarily selecting branches. Degenerate clusters
outside every requested window do not affect this Hall calculation.

`raw_hall_onsager_residual` is the measured `max|H_abc + H_bac|` in atomic
units; `relative_hall_onsager_residual` divides this by `max|H_abc|` (null
for identically zero H). The k integration can violate this identity on a
coarse mesh; the raw value is retained. `converged` describes the reference
SCF, and does not certify k-mesh convergence of transport. The report also
retains the raw Fock-image reconstruction residual. The derivative and
resistivity coefficient, zero-field conductivity, Seebeck and electronic
thermal conductivity share one Rust driver on CLI, native Python and ASE.

`hall_onsager_tolerance` (CLI `--hall-onsager-tolerance`, default `1e-10`)
controls the diagnostic `results.hall_onsager_satisfied`: true or false for
a nonzero Hall tensor, null for identically zero H. A defect above this
tolerance adds a warning to refine the transport mesh and check the Fermi
window. It does not alter any operator, weight, tensor or coefficient, and
passing this necessary symmetry check does not certify all transport
components as mesh converged. The default `delta_window=20` is retained;
the tighter acceptance uses 40 to resolve small tail-quadrature errors.
The zero-field and Hall windows both include their endpoints.

Independent acceptance in `tests/p78_hall_transport.rs` and
`python/tests/test_hall_transport.py` covers parabolic electron/hole bands
with unequal masses, the Drude limit, conductivity inversion in SI, raw
Onsager diagnostics, singular conductivities and invalid controls. Every
new Lorentz derivative is compared with central differences of the DFT
band velocity at four distinct steps; the independent finite-B Drude
resolvent supplies an additional four-step fit and Richardson check.
Measured exponents, fit quality and raw residuals are recorded in
[response.md](response.md); finite differences occur only in acceptance.

## 23. Retarded coherent conductivity and quantum Hall

`hall-transport` and `boltzmann-transport` additionally accept the paired
controls `transport_frequencies_hartree` and `relaxation_time_fs`. Frequencies
are signed angular frequencies, not cycles per second. With exp(-i omega t),
the intraband kinetic solution is `sigma(omega)=tau L0/(1-i omega tau)` and
`d sigma/dB=tau^2 H/(1-i omega tau)^2`. The extra `frequency_results` contain
actual conductivity (S/m, [3,3,2]) and Hall derivative (S/(m T), [3,3,3,2]);
the final axis is [real, imaginary]. Static outputs retain their units per
tau or tau squared. This is an intraband approximation with a physical
constant relaxation time; the coherent calculation below adds interband
terms. The poles lie below the real frequency axis.

Band operators now include the SCF's translated ECP and full/range-separated
exact-exchange terms with identical coefficients, density images and physical
cutoffs. No response calculation branches on a functional's registered name.
This extension concerns the frozen band operator; it does not supply the
induced exchange kernel of self-consistent hybrid DFPT. Converge the AO image
radius for diffuse bases: an insufficient radius can destroy the positive
Bloch metric. A negative metric is rejected, never repaired by an eigenvalue
floor. Acceptance compares shifted-band energies with the independent SCF
field reconstruction, not just the image interpolation itself.

`coherent-transport` computes the independent-particle Kubo-Greenwood tensor
from every band pair. It accepts `transport_k_mesh` and
`transport_frequencies_hartree`; defaults are `transport_temperature_k=300`,
`transport_mu_shift=0` Hartree and `scattering_hartree=0.001`. Its occupation
weight is `(f_n-f_m)/(e_m-e_n)`, with the analytic `-f'` limit only at exact
equality. Stable exponential differences retain arbitrarily small physical
splittings. The retarded denominator is `omega+e_n-e_m+i eta`; eta is a
declared scattering rate, not a numerical floor. Exactly degenerate blocks
enter through all their matrix elements, so unitary rotations inside those
blocks leave the result unchanged. The reported intra/interband decomposition
uses entire equal-energy blocks. See
[the independent-particle expression and flat-band limits](https://arxiv.org/pdf/2212.03192).

The physical AO current is
`v_mn=H'_mn-e_m S'_mn+i(e_m-e_n) D_mn`, including the AO position connection.
It is division free at degeneracy and Hermitian because `D^+=D+i S'`.
Omitting this connection loses intracell positions and even atomic optical
transitions. The raw current Hermitian defect, retarded pair-equation residual
and Fock-image residual are reported before any symmetry projection.
`results.{conductivity,intraband_conductivity,interband_conductivity,
antisymmetric_conductivity}` use S/m, shape [3,3,2]. The antisymmetric part
contains intrinsic Hall within this frozen independent-particle model. An
induced Hartree/XC vertex and a field-induced reference are separate remaining
calculations; their omission is stated in the model.

`quantum-hall` computes a non-Abelian occupied-multiplet Chern invariant on
an ordered two-dimensional Bloch slice. `topology_mesh` defaults to [8,8],
`topology_plane_axes` to [0,1] and `topology_fixed_k` to zero on the remaining
reciprocal axis. The reference has insulating zero-temperature occupations.
The occupied multiplet may be internally degenerate but must be separated
from its unoccupied complement. Physical links are the exact Gaussian
plane-wave cross-k metric, including reciprocal-boundary sewing. Coefficient
overlaps alone would remove the AO connection. The determinant of each
occupied link supplies a unit phase; principal plaquette phases sum to the
raw Chern number. This is a lattice-gauge topological invariant, not a finite
difference derivative. The implementation follows the
[non-Abelian lattice-gauge construction](https://arxiv.org/pdf/cond-mat/0503172).

Outputs include raw `chern_number`, spin-weighted `slice_hall_conductance`
(C e^2/h in S), sampled minimum gap, integer residual, maximum plaquette phase
and minimum log absolute link determinant. Neither the Chern number nor the
links are rounded or floored. Singular links require mesh refinement and a
gap check. Verify the invariant and spectral gap on refined meshes; an
integer on one coarse mesh does not prove the gap remains open between its
points. A 3D slice conductance is not a 3D bulk conductivity. The generic
link kernel supports providers with broken time reversal, and acceptance
uses nonzero Chern bands and degenerate copies. The current self-consistent
DFT connection remains collinear and zero field: flux-compatible magnetic
and SOC reference operators, true 1D/2D electronic combinations, and coherent
ordinary finite-B Hall remain tracked work. Time-reversal-symmetric references
must exhibit the corresponding Hall cancellation.

## 24. Complex orbital references in static periodic fields

`periodic-field-scf` uses the compensated dimensional Coulomb and XC backend
on the actual 1D/2D/3D geometry. It retains complex spin densities, Focks and
energy-weighted densities throughout SCF. The real field is
`A(r)=lambda [A_q exp(i q.r)+A_q* exp(-i q.r)]`, with an analogous electron
scalar potential. Integer `field_wave_indices` specify q in the physical
reciprocal subspace; open-axis indices are zero. A noncommensurate real field
requires a compatible supercell. This is a finite-field calculation with
`(p+A)^2/2`, including the constant and +/-2q components of A squared.
It is a periodic zero-net-flux magnetic field, distinct from the uniform-flux
magnetic-translation problem; see [periodic zero-flux Bloch Hamiltonians](https://arxiv.org/abs/1108.5651).

The localized exchange derivative acts on real and imaginary density image
components. Their Hermitian parities are respectively
`D_re(T)=D_re(-T)^transpose` and `D_im(T)=-D_im(-T)^transpose`.
Using the real parity for both would erase orbital-current exchange.
Hartree, exchange, XC, the full field energy and the Fock use one variational
functional. No orbital-conjugation averaging is applied to this reference.
The analytic amplitude derivative contracts the converged density with
`H1+lambda H2`; this differentiates the zero-temperature energy or the
fixed-number, fixed-temperature Mermin free energy. Displaced calculations
are acceptance oracles only.

All routes report full complex spin/k matrices, raw orbital-conjugation and
physical spinful-time-reversal defects, number and occupation errors, fresh
stationarity and component-energy defects, and the unprojected field
Hermitian residual. Physical spin-1/2 time reversal exchanges spin channels:
spin polarization or antiferromagnetic order may break it even while each
spin separately retains orbital conjugation. Net spin alone is insufficient.

The current model is collinear orbital minimal coupling with ordinary LDA/GGA
XC or complete complex full/range-separated exchange. It omits explicit
spin Zeeman and SOC. Orbital-field meta-GGA needs its covariant current/kinetic
density; nonlocal ECP needs covariant nonlocal-core field operators. Those
requests are not silently evaluated as bare kinetic-only magnetic models.
General spinor/SOC, current-dependent XC, spontaneous orbital-current seeds,
uniform flux cells and connecting their induced response kernels remain part
of the accepted objective. The existing zero-field transport driver keeps its
declared reference; it is not automatically a response of this finite-field
SCF. A finite-field reference with broken orbital conjugation must be tested
against reversed fields and its actual residuals before a Hall claim.

The dimensional zero-field ground state also includes translated local and
nonlocal ECP operators and a separate `ecp_energy` component. Its shifted
frozen-reference Fock includes the same ECP and full/range-separated exchange.
All electronic image sums respect the physical periodic-axis mask. Open
directions have no Bloch wave number. This ground-state extension does not
supply the nonlocal-core vector-potential/current response.

## 25. General complex spinor references

`periodic-spinor-scf` and `periodic::spinor_scf::run_spinor_scf` retain a full
complex Hermitian 2AO density at each k point. The order is spin-major (all
up AOs, then all down AOs). One mesh-global chemical potential fixes the
total electron count; no separate spin populations or orbital-conjugation
constraint are imposed. CDIIS uses a common history across k points and
canonical overlap-metric commutators. Convergence is rechecked against a
fresh physical Fock, canonical occupation density, particle number, metric,
density change and total/Mermin energy change. The new solver defaults to
500 iterations and 1e-10 energy, density and orbital tolerances.

`SpinorEnergyBackend` provides the variational energy and full spinor Fock
without registered-name branches. `SpinorDftBackend` connects dimensional
Coulomb, ECP scalar operators, full/range-separated exchange and arbitrary
local spin-density XC providers. The four Hermitian Pauli orbital matrices
P0, Px, Py and Pz include general complex spin-off-diagonal density; exchange
acts linearly on each matrix. The LDA evaluates local eigenvalues
(n +/- |m|)/2 and differentiates them to construct all three magnetic XC
components. General spin-gradient, kinetic-density and current providers
require their covariant energy and potential before they can use this adapter.

The general Hermitian `spinor_operator_real`/`spinor_operator_imag` controls
are paired flattened [k][2AO][2AO] Hartree arrays. They enter both the energy
and the Fock and scale with `field_amplitude`. The analytic amplitude derivative
includes their expectation value. `spin_field_real`/`spin_field_imag` describe
three Fourier coefficients in atomic magnetic-field units; the conjugate mode
is included. The spin Zeeman field also contains curl(A). `soc_scale` (1)
multiplies the Pauli operator sigma.(grad(V) cross p)/(4c^2) of the prescribed
periodic electron potential. All directions and both conjugate modes remain
complex. Its kinetic-momentum contact sigma.(grad(V) cross A)/(4c^2) includes
the zero and +/-2q components and the second amplitude derivative.

`nuclear_soc_scale` (1) includes the analytic one-electron Pauli nuclear SOC.
The same dimensional point/Gaussian-nucleus Green operator supplies the nuclear
potential and this SOC. Integration by parts expresses its matrix through
mixed bra/ket Gaussian derivatives, with no nuclear cutoff Coulomb model:
<a|grad(V) cross p|b> = i <grad(a)|V cross grad(b)>.
`nuclear_widths` supplies one finite nonnegative Gaussian width per nucleus,
in Bohr; an empty list chooses point nuclei. Nuclear SOC is fixed when the
external field amplitude varies. `nuclear_soc_energy` is its separately
reported Hartree expectation; it is already included in the `spin_soc` energy
component and must not be added again. Nonlocal ECP SOC requires its angular
momentum resolved core operator. Combining nuclear SOC and an orbital vector
potential additionally requires the nuclear grad(V) cross A contact, which
remains tracked; setting the nuclear scale to zero explicitly selects the
prescribed-operator model. Two-electron SOC, scalar relativistic corrections,
Dirac theory and magnetic translations for uniform flux remain separate work.

The common API is `driver::run("periodic-spinor-scf", ...)`,
`dft-rs periodic-spinor-scf`, native `periodic_spinor_scf(...)`, and ASE
`get_periodic_spinor_scf(...)`. Reports include full density/Fock parts,
Pauli spin polarization, chemical potential, occupations, term energies,
analytic amplitude derivative and raw residuals. Native positions/cell are
Bohr; ASE converts Angstrom. The spinful time-reversal diagnostic uses
Theta=i sigma_y K and swaps spin blocks with the off-diagonal minus signs.
It is reported without any projection. These references do not establish
completion of the full spinor response or uniform-flux program.

Nuclear SOC uses the same refined quadrature as the scalar nuclear operator.
`nuclear_soc_quadrature_defect` compares coarse and refined Pauli coefficients;
`nuclear_soc_hermitian_defect` reports the raw unprojected operator measurement,
both in Hartree. `coulomb_operator_defect` and `xc_grid_number_defect` remain
separate raw measurements. `scf_converged` denotes canonical SCF convergence;
the overall `converged` also requires the physical backend and nuclear SOC
quadrature criteria. A small stationarity residual cannot substitute for a
converged integral or density grid.

The physical local-spin-density convention is described in the
[noncollinear spin-density formulation](https://www.openmx-square.org/tech_notes/tech2-1_0/tech2-1_0.html).
Uniform-flux Bloch states require the magnetic-translation algebra described
in [Bloch electrons in a uniform magnetic field](https://harvest.aps.org/v2/journals/articles/10.1103/PhysRev.133.A1038/fulltext);
ordinary field-periodic Bloch functions cannot substitute for that basis.

## 26. Uniform-flux continuum London references

`periodic-magnetic-scf` supplies a separate continuum calculation on a
three-dimensional finite magnetic torus. The supplied cell is the magnetic
cell and must obey B.(a_i cross a_j)/(2 pi) in Z on all faces. The field is
in atomic units for A=B cross (r-O)/2 and kinetic momentum p+A. Invalid
noninteger flux reports all measured face fluxes and the compatible-cell
condition. A rational primitive flux requires a supplied enlarged magnetic
cell; automatically constructing that cell remains a separate task.

The magnetic Gamma sector uses ordered commuting generators rather than
ordinary periodic orbitals. With T=A n, the generator is
U_T^O psi(r)=exp[-i (B cross T).(r+O)/2] psi(r-T), multiplied by the
cocycle exp[-i pi sum_{i<j} m_ij n_i n_j]. This factor is essential for odd
integer flux. The London orbital at R is exp[-i (B cross (R-O)).r/2] g_R(r).
Its image coefficient is the cocycle times exp[i B.(R cross T)/2]. Changing
O multiplies all orbitals by the common gauge phase, so continuum AO overlap,
kinetic and density-product operators are unchanged. Raw boundary and gauge
covariance are verified in all seven crystal systems. The torus quantization
and magnetic translation structure are described in
[quantum mechanics on a magnetic torus](https://arxiv.org/abs/1103.0034) and
[the magnetic translation group](https://arxiv.org/abs/hep-th/0205053).

London Gaussian products supply analytic Fourier matrices R_ab(G), including
their full phase attenuation. Kinetic integrals use the exact local momentum
-i grad(g_R)+B cross (r-R) g_R/2, retaining all linear and quadratic orbital
field terms. The nuclear potential, Hartree and exchange use the same
zero-average periodic Coulomb kernel v_G=4 pi/(V G^2), with G=0 omitted.
Hartree contracts the charge density with R(G); full spinor exchange applies
sum_G v_G R(G) P R(G)^dagger. Long-range exchange uses its defining
exp[-G^2/(4 omega^2)] multiplier and generic functional coefficients.
The finite magnetic-cell interaction defines a finite many-electron model;
its exchange convention must not be identified with an independently
converged infinite-crystal k integral. Magnetic cell-size convergence is
required for that limit.

The Pauli nuclear term is sigma.(grad V cross (p+A))/(4 c^2). Analytic Fourier
gradient and kinetic-momentum vertices include its nuclear vector-potential
contact. Point and finite Gaussian nuclei use the same source potential;
the existing source Green self-energy convention is retained. Uniform
Zeeman is sigma.B/2. A supplied general Hermitian spinor operator is additional
to these terms and is specified in the London AO metric, in Hartree.
Nuclear_soc_energy is a diagnostic expectation already included in spin_soc.
Kinetic energy already includes the orbital field; orbital_field is zero in
this energy decomposition and must not be added a second time.

The local spin-density provider uses the same rotational density eigenvalues
as the dimensional spinor adapter. Its quadrature covers the centered cell
with a tensor Gauss-Legendre rule. `magnetic_grid_order` specifies the initial
fine order per direction; the initial order-8 rule independently measures XC
energy and Fock defects. The provider increases the order by eight, reusing
the preceding fine rule as its next coarse rule, until all three defects
meet `magnetic_operator_tolerance` or order 128 is reached. Failure at that
limit remains unconverged. Fine-grid particle number is compared against the analytic
London overlap, without any weight/density normalization or error floor.
These three raw grid defects participate in overall convergence.
`magnetic_grid_initial_order` records the requested order, and
`magnetic_grid_order` the order actually used (zero when there is no local XC).
`xc_grid_refinement_history` retains every evaluated row, including failed
refinements, as [fine order, energy defect in Hartree/cell, Fock defect in
Hartree, number defect in electrons]. Repeated orders can describe different
SCF densities; the history is not sorted or reduced to successful rows.

Rust uses `MagneticCell`, `MagneticScfOptions`, `MagneticDftBackend` and
`run_spinor_scf`; the common driver exposes `periodic-magnetic-scf`, native
`periodic_magnetic_scf` and ASE `get_periodic_magnetic_scf`. All three aligned
API tables contain the corresponding controls and units. The defaults are
500 SCF iterations and 1e-10 energy/density/orbital tolerances on the new
driver; explicit common reference controls override them. The reciprocal
production cutoff is 32 inverse Bohr and its separate 0.8-cutoff measurement
checks nuclear, Hartree, full/range exchange and SOC terms individually,
so one-electron cancellation cannot conceal a failed operator.
The image diagnostic compares overlap and kinetic matrices at basis_cutoff
and 0.8*basis_cutoff; convergence of all image-dependent quantities still
requires independently increasing that physical cutoff. Source-Ewald
cutoffs and splitting are independent of the electronic reciprocal cutoff.

No separate spin populations or orbital conjugation constraint are imposed.
The common variational solver reports fresh physical-Fock stationarity,
canonical occupations, total number, metric, density/energy changes and
Mermin free energy. Both SCF and backend convergence are required. The
current physical adapter supports local density and full/range exchange.
Infinite-crystal magnetic k integration, 1D/2D isolated uniform-flux
operators, nonlocal-core/two-electron SOC, covariant gradient/current XC,
scalar relativistic/Dirac terms and the full magnetic response hierarchy
remain tracked parts of the accepted objective.

The dimensional Fourier-field and spinor drivers also accept rank-one and
rank-two cells with zero unused rows. Only physical periodic vectors define
their length/area and reciprocal dual; three periodic dependent vectors
remain invalid. The magnetic torus itself requires full rank three.

The independent normalized single-Gaussian kinetic oracle also includes
finite-cell images. For displacement T and wave w=-B cross T/2, its pair
overlap is exp[-a T^2/2-w^2/(8a)] times the magnetic cocycle. Its kinetic
moment is that overlap times
3a/2-a^2 T^2/2+w^2/4+B^2/(16a)-B^2 w^2/(128a^2).
The expectation is the image kinetic sum divided by the image overlap sum.
The isolated value 3a/2+B^2/(16a) applies only when image terms vanish;
it cannot replace this finite-torus oracle at arbitrary cell sizes.

#### Fixed-flux operator differentiation and local Pauli currents

`spinor_amplitude` scales only an additional supplied Hermitian London AO
operator. The cell flux, B and nuclear SOC scale remain fixed. Its stationary
energy derivative is `Tr[P H_generator]`, including full orbital/spin relaxation
in P. At fixed total number and temperature it is also the Mermin free-energy
derivative. The optional generic `exchange_coefficients=[cx,clr,omega]` replaces
the full/range exchange coefficients without changing semilocal part weights;
omega is nonnegative in inverse Bohr, and omega=0 has exactly zero long-range
kernel. No response branch depends on the functional's registered name.

`current_points` are explicit Cartesian Bohr triples on every route, including
ASE. The local Pauli particle current is the sum of kinetic current
`Re sum_ab P_ab phi_b* (p+A) phi_a`, spin magnetization current `curl(m)/2`, and
nuclear SOC current `nuclear_soc_scale m cross grad(V)/(4 c^2)`.
The spin-density spatial derivatives include the London phases analytically.
The source gradient uses the same analytic Ewald Green function as the source
energy, rather than truncating a pointwise nuclear gradient at a hard Fourier
cutoff. Samples on point nuclei are undefined and rejected. Number and spin
densities are per Bohr^3; source gradient is Hartree/Bohr; particle currents
are electrons per Bohr^2 atomic time. Electric current has the opposite sign,
with electron charge -1. An additional supplied nonlocal AO operator needs its
own vector-potential vertex; this reported local Pauli decomposition does not
infer that missing vertex.

Physical time reversal maps B to -B, conjugates the London spatial basis,
and reverses spin. `spin_coefficient_conjugation_defect` describes coefficients
in the fixed London frame alone. A spin-compensated coefficient matrix can
have this defect zero while retaining an orbital current and broken physical
time reversal. `external_field_breaks_time_reversal` and
`time_reversal_convention` make this distinction explicit. Independently
self-consistent B and -B references with a time-reversal-even extra coupling
verify equal energy/density and opposite spin and each current component.

#### Infinite-crystal magnetic Bloch integration

`magnetic_model="crystal_bz"` selects magnetic Brillouin-zone integration;
the backward-compatible default `finite_gamma` retains the finite torus.
The ordered London image coefficient also contains `exp(i k.T)`. The continuum
overlap, kinetic, nuclear and Pauli operators are built independently at each
Cartesian magnetic Bloch k. A transfer vertex is
`R(left,right,q)=<phi(left)|exp(i q.r)|phi(right)>` with
`q+right-left` reciprocal. The two spaces are independent; its adjoint is the
opposite transfer with swapped spaces, not a Hermitian projection within q.
`fourier_transfer` accepts arbitrary finite left/right k satisfying this rule.

Hartree uses the averaged periodic density. Full/range exchange uses
`mean_right sum_G v(q) R(left,right,q) P(right) R(left,right,q)^dagger`,
where `q=G+left-right` and `v(q)=4 pi/(V q^2)`; the range multiplier is
`exp[-q^2/(4 omega^2)]`. All energy traces and total-number constraints are
averaged over the equal-weight k mesh. Local-density XC uses the averaged
charge/spin densities before evaluating the energy or potential; averaging
independent k-point XC energies would be physically incorrect. The adaptive
grid measures every k potential and the common energy/particle number.
Source energy appears once per cell, not once per k point.

For crystal exchange, a Gaussian auxiliary integrates the isolated q=0 limit.
For inverse-length width eta, the kernel correction is
`chi=2 eta/sqrt(pi)-mean_k sum_(G,q!=0) 4 pi exp[-q^2/(4 eta^2)]/(V q^2)`.
The combined transfers form the reciprocal mesh of the cell enlarged by the
mesh counts. The correction adds `chi S(k) P(k) S(k)` to the exchange map;
the energy contains the corresponding half trace. The long-range auxiliary
has width `eta_eff=(eta^-2+omega^-2)^-1/2`, giving its correct omega=0 limit.
This is analytic subtraction/addition at the singular limit; every physical
nonzero q denominator is unchanged. See the [Gaussian auxiliary derivation](https://arxiv.org/abs/0909.1648).

`magnetic_exchange_auxiliary_width` defaults to 0.5 inverse Bohr; zero selects
raw punctured quadrature for independent comparison. The finite torus never
uses this correction. `exchange_singularity_correction` reports [full,LR]
coefficients in Hartree. SCF and reciprocal/grid residuals measure the chosen
quadrature, and do not certify an infinite-mesh limit. Refine k meshes and
compare auxiliary widths independently. A one-point crystal quadrature with
zero auxiliary width agrees with the finite torus; a corrected one-point
crystal quadrature is a different integration approximation.

### Uniform magnetic fields with open axes

`periodic-field-scf` and `periodic-spinor-scf` accept
`uniform_magnetic_field=[Bx,By,Bz]`, `magnetic_origin=[Ox,Oy,Oz]` and
`field_amplitude`. Native coordinates/origin are Bohr; ASE converts the
geometry from Angstrom. The field is reduced atomic `curl A` at unit
dimensionless amplitude. The actual field is the input vector times
`field_amplitude`.

For wires, every field direction has a linear gauge periodic along the
wire. For ordinary slab Bloch sectors, the field must be in the physical
plane. Rust `UniformOpenField` constructs the gauge from the embedded
periodic subspace, including tilted cells and zero unused open-axis vectors.
The wire transverse gauge agrees with the translation-preserving continuum
construction in the [nanowire magnetic-state derivation](https://arxiv.org/abs/1211.5444).
The implementation adds the axial symmetric gauge and evaluates all
Gaussian matrix elements analytically, including `A.A` and the complete
linear anticommutator. It retains complex SCF densities and full complex
exchange. The electrostatic source, Hartree and range-exchange models retain
their physical dimensional boundaries and explicit compensation controls.

The two-component route also includes `sigma.B/2` and the nuclear Pauli
contact `sigma.[grad(V) cross A]/(4 c^2)`, using the same dimensional
point/finite-nucleus Green integral as scalar attraction and nuclear SOC.
The latter is linear in field amplitude and contributes to its variational
energy derivative. Four-step independent source-displacement and relaxed
SCF derivative fits are required by the acceptance suite.

Origin changes are represented by the common AO phase
`exp(i lambda (M origin).r)`. This maps the operator to the origin-zero
gauge frame and makes matrix results invariant within this gauge family.
The orbital-only route omits spin Zeeman/SOC; use the spinor route for
those terms. Nonlocal ECP, gauge-covariant meta-GGA/current providers,
remaining response combinations are still tracked; normal slab flux uses the magnetic-cell route below.
Normal slab flux is not rounded, projected away or replaced by a 3D
periodic Coulomb model. It uses a flux-compatible magnetic cell and open-direction magnetic Coulomb integrals. Current extension acceptance
status and raw measurements are recorded in `response.md`.

### Magnetic slab and its crystal integral

`periodic-magnetic-scf` accepts physical 2D PBC with zero unused lattice rows.
Only actual periodic faces obey `B.(a_i cross a_j)/(2 pi)` integer flux;
arbitrary in-plane field components add no fictitious face constraint.
The finite Gamma model and `magnetic_model="crystal_bz"` are explicit choices.
For crystal slabs only periodic axes have k-mesh counts above one. Vertices
act in distinct left/right spaces at `q=G+k_left-k_right+n q_normal`.
There is no same-q Hermitian projection. Nuclear/Hartree operators average
charge across sectors; local XC uses the averaged full spin density, and
exchange retains spin coherence between distinct k sectors.

Open-normal Coulomb integration uses an even Gauss rule under a tangent
coordinate change. The zero in-plane wave retains
`-2 pi abs(z-z')/area`; analytic subtraction removes its divergent constant.
For long-range exchange the zero mode is referenced to equal normal
coordinates, with exactly zero kernel at omega=0. Finite transverse Gaussian
countercharge is required for charged slabs; a three-dimensional uniform
background is never substituted. Initial `magnetic_open_order=64` compares
separate fine/coarse normal rules and can refine to 512. Actual orders and
all failed nuclear/SOC comparisons are reported. Density-dependent Hartree
and exchange defects remain separately gated at the chosen final rule.

The slab singularity auxiliary integrates the leading normal Coulomb pole:
`chi_full=sqrt(pi) eta-sum_q 2 pi exp(-q^2/(4 eta^2))/(A Nk abs(q))`.
Long-range auxiliary terms include `erfc(abs(q)/(2 omega))`, with analytic
continuum integral `2 eta/sqrt(pi) atan(omega/eta)`.
These dimension-specific coefficients add `chi S_k P_k S_k` without changing
any nonzero physical denominator. The finite Gamma model has no auxiliary;
one-point raw crystal quadrature agrees with it. Four independent mesh
refinements and closed Gaussian normal integrals are recorded in response.md.
All Rust/CLI/native Python/ASE entries and units are aligned in rust-api.md#common-calculation-driver.
Remaining functional classes, core/current operators and response combinations
are still part of the full accepted objective.

## 27. Berry-phase polarization

`berry-polarization` evaluates the polarization of an insulating reference
from the Berry phases of the occupied manifold. For each reciprocal axis
`b_i` the strings `k_j = k_perp + j b_i / N_i` give

```text
phi_i(k_perp) = -Im ln prod_j det[C_o(k_j)^+ M(k_j, k_(j+1)) C_o(k_(j+1))]
p_i = -g sum_spin <phi_i> / (2 pi) + sum_atoms Z (r . b_i) / (2 pi),   P = (1/V) sum_i p_i a_i
```

with the cross-k metric `M` of plane-wave AO integrals. `p_i` is defined
modulo one and only differences along an insulating path are physical. The
string mesh is the SCF k mesh by default (`berry_mesh` overrides it; other
points use the reconstructed Fock operator).

Acceptance (`tests/p96_berry_polarization.rs`, SVWN5/STO-3G):

- Rigid translation of all atoms by a known reduced vector: the total
  reduced polarization changes by less than 2e-15 on every axis.
- Born charges from central differences of the polarization (h = 0.005 Bohr,
  validation only) against the analytic charges of `born-charges` on a polar
  chain (H2 with a helium atom, k meshes N x 1 x 1). Component along the
  chain: differences 6.43e-3, 1.94e-3, 9.84e-4, 3.39e-4 for N = 4, 6, 8, 12
  (`p = 2.657`, `R^2 = 0.9982` in `1/N`). The two components along the
  directions with a single k point keep constant differences of 3.9e-3 and
  9.1e-4: there the Berry phase is the single-point formula and the analytic
  route a one-point zone sum, and with 7 Bohr between the chains the two
  samplings are not equivalent, so they are not compared.
- A molecule in cubic boxes of edge L = 12, 14, 16, 20 Bohr: on string
  meshes 2, 4, 8 the dipole converges with the second order (ratios of
  successive differences 3.967, 3.976, 3.981, 3.988), and the extrapolated
  limit differs from the dipole of the isolated molecule by 2.72e-3, 1.63e-3,
  1.08e-3, 5.43e-4 (`p = 3.150`, `R^2 = 0.99966` in `1/L`), the polarization
  of the periodic images. Recorded failed setting: with one string point per
  axis the differences were 2.98e-2, 2.24e-2, 1.74e-2, 1.14e-2 (`p = 1.88`).

## 28. Projected density of states and Fermi-level thermodynamics

`dos` with `projected` decomposes every band by the Mulliken weights
`w = sum_(mu in group) Re[conj(C_mu) (S C)_mu]` of the groups (atom, angular
momentum); the weights of a band sum to one. `pauli-spin-susceptibility`
evaluates, for the rigid bands at a fixed electron count and temperature,

```text
L_n = w sum (e - mu)^n f (1 - f) / kT
N(E_F) = L_0,   chi = mu_B^2 L_0,   C = k_B (L_2 - L_1^2 / L_0) / kT
```

(the default bare mode, without exchange enhancement). The susceptibility
formula above is the equal-channel limit at g=2. Unequal channels use
`chi=(g/4)^2 4 D_alpha D_beta/(D_alpha+D_beta)` with a common
chemical-potential response; see [section 35](#35-exchange-enhanced-static-spin-susceptibility)
for the ensemble and its two-channel report. Thermal moments and heat
capacity use the sum over both explicit spin channels. The chemical potential is taken as the middle
of the plateau of the electron count, which is the mid-gap value for an
insulator at low temperature.

Acceptance (`tests/p98_fermi_level_and_projected_dos.rs`,
`periodic::fermi_thermodynamics::tests`):

- Hydrogen fluoride chain (gap 0.3171 Hartree): shells sum to their atom and
  atoms to the total with a residual of 5.7e-14 at a peak of 298.8 states
  per Hartree; the occupied integrals of the atom projections are 9.23734543
  and 0.76265373 against the Mulliken populations 9.23734626 and 0.76265374
  of the density matrix; the SCF mesh and the reconstructed mesh agree to
  3.0e-10. The same checks hold for PBE, TPSS, PBE0 and M06 on a hydrogen
  chain (residual at most 7.8e-14, integrated populations 0.99999993).
- Metallic hydrogen chain at 3000 K: `N(E_F)` 9.46727741 states per Hartree
  equals the independent sum over the SCF bands to 1e-9; the chemical
  potential equals the SCF one. Central differences of the magnetization
  under a rigid Zeeman shift (steps 0.1 to 0.0125 kT): `p = 1.998973`,
  `R^2 = 0.99999993`. Central differences of the band energy in kT (steps
  0.025 to 0.003125 kT): `p = 2.003760`, `R^2 = 0.99999906`; the recorded
  failed interval 0.1 to 0.0125 kT gave `p = 2.057`, `R^2 = 0.99980`
  (fourth-order term visible with six k points). With PBE: `p = 1.999218`
  and `p = 2.001074`.
- A ladder of equally spaced levels reproduces `L_0 = g`, `C = S =
  (pi^2/3) kT g` and a Wilson ratio of one to 1e-9; the molar susceptibility
  and the Sommerfeld coefficient agree with `mu_B^2 N_A g` and
  `(pi^2/3) k_B^2 N_A g` from tabulated constants to 1e-6.

## 29. Built-in pair dispersion, three-body term and dispersion force constants

The lattice sum of section 20 also evaluates the Chai-Head-Gordon pair term
of range-separated hybrids with built-in dispersion (`dispersion = chg`,
attached automatically to `wb97xd`): constant `C6_ij = sqrt(C6_i C6_j)` on
the D2 tables and `g(r) = r^6 / (r^12 + a R_r^12)`. With `three_body` a D3
model adds the Axilrod-Teller-Muto term

```text
E_3 = (1/3) sum_i sum_{a<b} s9 sqrt(C6_ia C6_ib C6_ab) [3 cos cos cos + 1] / (r_ia r_ib r_ab)^3
      / (1 + 6 (R0 / (r_ia r_ib r_ab))^(alpha/3))
```

over the partners of every atom inside `abc_cutoff` (default 40 Bohr), with
`R0` the product of the pair radii scaled by `rs9 = 4/3` and `alpha = 16`.
All terms are written once as a generic energy function of the positions and
lattice rows; forces, strain derivative and force constants of the
three-body term, and the force constants of every model, are forward duals
of that function on fixed image lists. `phonons`, `phonon-bands`,
`phonon-thermo` and `quasi-harmonic` add the dispersion force constants (and
`quasi-harmonic` the dispersion energy) of the lattice sum. Phonons of
`wb97xd` itself use the exact-exchange force constants of section 31.

Acceptance (`periodic::dispersion::tests`, `tests/p76_periodic_dispersion.rs`):

- Pair term: an isolated molecule in a 60 Bohr box equals the molecular
  energy to 2e-21; forces `p = 1.999959` and `p = 2.000005`, strain
  derivatives `p = 1.999688` to `1.999986` (four steps each).
- Generic energy against the closed sums: energy, forces and strain
  derivative agree to 2e-18 for every model.
- Force constants against central differences of the analytic forces (steps
  0.02 to 0.0025 Bohr): `p = 2.000124` and `2.000238` (BJ), `2.000132` and
  `2.000248` (zero damping), `1.999797` and `1.999986` (pair term),
  `2.000124` and `2.000237` (BJ with three-body term); the acoustic sum is
  2e-18 and a molecule in a box reproduces the molecular Hessian to 2e-20.
- Three-body term: single triangles reproduce the closed form with explicit
  angles to all printed digits (1.606907901409e-6, -1.140271975125e-6,
  -1.256685320254e-6 Hartree for an acute, an obtuse and a nearly linear
  triangle); forces `p = 2.000332` and `2.000404`, strain derivatives
  `p = 2.000001` to `2.000315`.
- Driver: `wb97xd` on a hydrogen chain gives -1.0958134222 Hartree with the
  dispersion energy -9.078927866e-4 Hartree equal to the direct lattice sum;
  stress and forces with `chg` and with the three-body term differ from the
  reference by the direct lattice sums to 1e-16; the phonon force constants
  with `d3bj` and `chg` equal the plain ones plus the dispersion force
  constants to 5e-17 (highest frequency 5387.91 against 5503.82 cm^-1 with
  `d3bj`).

## 30. Cost of the periodic Hartree build

Long periodic physical checks are marked ignored and can be selected with
`cargo test --release -- --ignored`. Their cost includes rebuilding the
Ewald-split Hartree operator over lattice shells at every k point and SCF
iteration. Even a minimal-basis cell can be expensive when the lattice and
reciprocal shells are large.

The reciprocal sums of the range-separated Hartree build run over one vector
of every pair `(G, -G)` with doubled weight, and the split exponent between
the reciprocal and the real-space part follows the cell volume
(`3 (54 / V)^(2/3)` clamped to `[1, 3]` Bohr^-2; `hartree_compact_exponent`
overrides it). The default local screening threshold is `1e-16` for the
bulk Hartree matrix, energy and response. The accumulated derivative
error depends on strain lever arms and the number of lattice terms.
The fixed-density split check and charged Li convergence measurements
are recorded in [response.md](response.md). Earlier unit tests reproduced
the exhaustive reference to 6e-12 over the tested splits. Measured time
of one build in the earlier screening configuration (four
threads): hydrogen-helium chain (three functions, 313 Bohr^3) 3.2 s before,
0.77 s after; lithium hydride chain (six functions, 397 Bohr^3) 38.8 s
before, 9.6 s after; dense fcc helium (54 Bohr^3) 0.11 s, where the larger
exponent is kept. The Hartree build remains the dominant cost of the
periodic SCF and of every response solve.

## 31. Exact exchange on a finite k mesh and exact-exchange force constants

The inverse Bloch transform of a k mesh is periodic in the Born-von Karman
supercell of the mesh, so the density between two centres and between their
replicas `T + N_i a_i` carries the same block. The real-space exchange sums
use every pair of centres once: the element `(mu, nu)` of the density block
and of the exchange block of translation `T` is scaled by the minimum-image
weight of the separation `R_nu + T - R_mu` of its two atoms (one over the
number of replicas of the same, smallest length; zero when a shorter
replica exists), in the SCF, the forces, the stress, the force constants and
the band model alike. Deciding by the separation of the atoms rather than
by the cell translation makes a supercell and its primitive cell use the
same pairs. The exchange energy `sum w D . K[w D]` no longer depends on an
exchange cutoff that exceeds the supercell, and a coarse mesh no longer
double counts; converge the k mesh until the supercell exceeds the range of
the density matrix. When a displacement or strain changes the minimum
image of an active AO pair, this finite-mesh exchange model can have an
energy jump. The analytic derivatives describe the branch with the
reference image selection. A measured Gamma crossing and its local
acceptance interval are recorded in the [stress ledger](response.md#r026-stress-elastic-high-elastic).

Hartree-Fock energies (Hartree) of the hydrogen chain of section 12
(STO-3G, 7 Bohr between chains):

| k mesh | exchange cutoff (Bohr) | every replica | minimum images |
|---|---|---|---|
| 2x1x1 | 5.0 | -1.0072722787 | -1.0770497395 |
| 2x1x1 | 8.0 | -1.9243779087 | -1.0790985092 |
| 2x1x1 | 12.0 | -3.3310346924 | -1.0790999716 |
| 4x1x1 | 5.0 | -1.0735019884 | -1.0739163757 |
| 4x1x1 | 8.0 | -1.6592860311 | -1.0754103703 |
| 4x2x2 | 8.0 | -1.0764305464 | -1.0755562660 |

`phonons` and the supercell routes built on it accept Hartree-Fock, global
and range-separated hybrids for insulating all-electron references. Three
exchange terms enter (`periodic::exchange_response`): the second derivatives
of the truncated image sum at fixed density, from the mixed second
derivatives of every translated quartet with the same-centre blocks by
translational invariance; the derivative of the exchange matrix of the SCF
as the source of the coupled-perturbed equations, with the exchange kernel
`-c K[P^(1)]` acting on the response density; and, in the pairing with the
response density, the density derivative of the exchange gradient, which is
the average of that matrix and of its adjoint map because the truncated sum
is not symmetric in its two density factors. With these the unsymmetrized
force constants (`raw_force_constants`: column `y` is the derivative of the
analytic gradient with respect to displacement `y`) are exact at any
cutoff. `raw_asymmetry` measures the truncation and vanishes once the cutoff
exceeds the supercell (5.9e-7 at 8 Bohr, 9.0e-12 at 12 Bohr and 1.7e-14 at
16 Bohr on the 2x1x1 mesh); `force_constants` is the symmetric part.
Dielectric, elastic and electron-phonon responses still refuse exact
exchange, and so do Fermi-Dirac references.

A source that vanishes by symmetry (the first-order overlap of a cell whose
atoms are all inversion centres) leaves only rounding noise in the
occupied-occupied response density, which the unit normalization of the
response kernel magnified into a non-Hermitian matrix ("k-point density 0
is not Hermitian", relative error 1.88 on a doubled helium cell). The kernel
now measures the anti-Hermitian part on the absolute scale, refuses more
than 1e-9 relative to `max(1, largest element)`, and projects it out.

Acceptance:

- `periodic::exchange_response::tests` (two-atom s,p cell, fixed density
  images, full-range and erf-attenuated kernels, steps 0.004 to 0.0005
  Bohr): second derivatives against central differences of the exchange
  gradient `p = 1.999916, 1.999953, 1.999976` and `1.999909, 1.999959,
  2.000006` (`R2 = 1.00000000`); exchange-matrix derivative
  `p = 1.999999, 1.999988, 1.999989, 2.000005`; the pairing matrices
  reproduce the density derivative of the gradient to 1.8e-14 and 2.7e-15,
  where the exchange-matrix derivative in their place is off by 2.5e-5 and
  3.5e-5.
- `periodic::exchange::tests::minimum_image_weights_partition_every_replica_class`:
  the weights of every replica class of a 4x2x1 mesh add up to one.
- `tests/p68_gamma_phonons.rs` (hydrogen chain, k 2x1x1, exchange cutoff 8
  Bohr, raw column of atom 1 along x against central differences of the
  analytic forces, steps 0.04 to 0.005 Bohr): Hartree-Fock `p = 2.000305`
  (`R2 = 0.99999999`, Richardson difference 9.2e-10, raw asymmetry 5.9e-7);
  PBE0 `p = 2.000096` (`R2 = 1.00000000`, 3.5e-10, 1.5e-7); wB97X-D with
  its pair dispersion `p = 1.987563` (`R2 = 0.99998910`, 4.5e-8, 3.6e-7;
  halving ratios 3.917, 3.980, 3.995: the largest step carries the
  fourth-order term of the damped pair dispersion). Acoustic residuals
  3.7e-14 to 3.8e-14 Hartree/Bohr^2.
- `tests/p20_periodic_energy.rs` (run with `--ignored`): a doubled helium
  chain with half the k mesh has twice the Hartree-Fock energy of the
  primitive cell within the 5e-8 Hartree of the test, with an exchange
  cutoff beyond half the Born-von Karman period.
- `tests/p79_transport_extensions.rs`: the band model rebuilds the Fock
  matrix of the SCF from the same minimum-image blocks to 1.4e-15
  (Hartree-Fock), 1.1e-14 (PBE0) and 1.7e-15 (LC-wPBE) Hartree.
- Doubled helium cell of the same file (column of atom 1 along x, steps 0.08
  to 0.01 Bohr): errors 7.892e-4, 1.959e-4, 4.890e-5, 1.222e-5,
  `p = 2.004223`, `R2 = 0.99999884`, Richardson difference 6.9e-9, acoustic
  residual 2.8e-13.

## 32. Quasi-harmonic states at a pressure and mode Grueneisen parameters

`quasi-harmonic` takes `pressure_gpa`. At a pressure the state of every
temperature is the point of the fitted isotherm with `-dF/dV = p`, found by
bisection on the volume grid, which must bracket it (the refusal names the
fitted pressures at the ends of the grid). The isotherm needs no minimum of
its own; `B_T = V d2F/dV2` and `alpha = (dS/dV)_T / B_T` are taken at that
volume, `C_p - C_v = V T alpha^2 B_T` and `gamma = alpha B_T V / C_v` hold
at any pressure, and `gibbs_energy = F + p V` is reported.

Mode Grueneisen parameters `gamma = -d ln omega / d ln V` come from the
dynamical matrices of the same volume grid at fixed reduced q: every element
is fitted by a least-squares quadratic in `ln V`, and the eigenvalue
derivatives at the reference volume (scale 1, or the first scale) follow
from first-order perturbation theory with the derivative matrix diagonalized
inside every degenerate eigenspace, `gamma = -(d lambda / d ln V) /
(2 lambda)`. They are properties of the fitted volume-grid model, like the
other quasi-harmonic outputs; an analytic derivative needs third-order
force constants and is not implemented. `mode_gruneisen` is `[q][mode]`
(null below `mode_cutoff_cm1`), `mode_gruneisen_frequencies_cm1` the
frequencies of the fitted matrices, `gruneisen_mode_average` the
heat-capacity-weighted average per temperature at the reference volume, and
`vibrational_free_energies` the `[volume][temperature]` table.

Acceptance:

- `periodic::quasi_harmonic::tests`: on the Einstein-Birch-Murnaghan model
  at 8e-5 Hartree/Bohr^3 the closed-form pressure at the returned volume is
  8.00000008e-5, 8.00000224e-5 and 8.00000119e-5 at 0, 100 and 300 K; a grid
  on the compressed branch without a free-energy minimum gives 2.99999994e-4
  for 3e-4. Thermodynamic Grueneisen parameter against the model value for
  relative volume spacings 0.02 to 0.0025: errors -4.54e-4, -1.08e-4,
  -2.65e-5, -6.61e-6, `p = 2.032168`, `R2 = 0.99993429`. Mode Grueneisen
  parameters of power-law eigenvalues for log-volume spacings 0.02 to
  0.0025: `p = 2.000897, 2.000122, 2.000040` (`R2 >= 0.99999995`).
  Degenerate perturbation theory: eigenvalue derivatives -0.5908326913195889
  and 0.49083269131960244 against the closed form -0.5908326913195984 and
  0.49083269131959834.
- `periodic::eos::tests::isotherm_fit_accepts_a_branch_without_a_minimum`:
  pressure of a compressed branch 1.5081011996238643e-3 against the closed
  form 1.5081011996238146e-3.
- `tests/p70_phonon_dispersion.rs` (compressed fcc helium, edge 6 Bohr,
  scales 0.98, 1, 1.02, 1.04, 2x1x1 supercell and q mesh, 10.3 GPa): volumes
  53.532, 53.853 and 54.516 Bohr^3 and `B_T` 46.99, 45.30 and 42.79 GPa at
  0, 300 and 600 K; the single stable zone-boundary mode (272.77 cm^-1) has
  `gamma = 2.2319` at the reference volume against the thermodynamic 2.2166
  and 2.2549 at the volumes of 300 and 600 K. The two transverse
  zone-boundary modes are imaginary at every volume of this coarse model and
  are excluded (`imaginary_modes_per_volume`).

## 33. Structure input and the object interface

Without `--cell`, the comment line of an extended XYZ file supplies the
lattice of a periodic driver calculation (`Lattice="..."` with nine numbers
in Angstrom, optional `pbc="T T F"`). In Python, `dft_rs.Crystal` holds a
structure in Bohr and `dft_rs.PeriodicCalculator` binds it to reference
controls and exposes every periodic calculation as a method; the ASE adapter
routes energy, forces and stress of fully periodic `Atoms` through one
`periodic-energy` calculation, which makes the dispersion models and every
other driver control available from calculator parameters. The entries are
listed in the Python/ASE table and structure examples in `docs/python-api.md`.

Acceptance: `tests/p68_gamma_phonons.rs` (the extended-XYZ lattice gives the
energy of the `--cell` option, and an open axis in the comment line is
refused by the three-dimensional driver).

## 34. Energy-dependent relaxation-time transport

The `boltzmann-transport` and `hall-transport` common drivers accept
`scattering_table` rows `[absolute_Kohn_Sham_energy_hartree, lifetime_fs]`.
Energies must be strictly increasing and lifetimes positive and finite.
A single row specifies a constant lifetime. Between rows the logarithm of
the lifetime is interpolated linearly; outside the interval the endpoint
lifetime is constant. Shift the table energies together with the band
energies when changing the energy zero. This input supplies a phenomenological
scalar lifetime; it does not calculate microscopic scattering rates.
All moments use the requested `delta_window` about each chemical potential.
Converge this window together with the k mesh when lifetimes vary strongly.

For `w = g/(V Nk) (-df/dE)`, the production integrals are
`K_n,ab = sum w tau(E) v_a v_b (E-mu)^n` and
`H_abc = sum w tau(E)^2 v_a (v cross e_c).grad_k v_b`.
The particle count retains the unweighted occupations. Energy-dependent
transport moments follow the energy-resolved transport distribution; see
[the relaxation-time derivation](https://arxiv.org/abs/1807.07513).
No lifetime derivative enters the Lorentz vertex because the Lorentz flow
is tangent to a constant-energy surface. At signed angular frequency
`omega`, each state's lifetime is replaced by
`tau(E)/(1-i omega tau(E))`, and this propagator is squared for Hall.
Averaging lifetimes before solving this equation changes the model.

With a table, DC fields are `conductivity` (S/m), `seebeck` (V/K),
`electronic_thermal_conductivity` (W/(m K)) and, for Hall,
`hall_conductivity_derivative` (S/(m T)) and `hall_coefficient` (m^3/C).
Without a table the existing per-tau fields and controls are preserved.
Frequencies can accompany a table without `relaxation_time_fs`; supplying
both kinds of lifetime is rejected. The convention is `exp(-i omega t)`;
dynamic tensors have a last axis `[real, imaginary]`. The Hall derivative
has axes `[current, electric field, magnetic induction]`; resistivity has
`[electric field, current, magnetic induction]` and
`d rho/dB = -K0^-1 H K0^-1`. The inverse uses the existing condition test,
without a denominator floor. Singular conductivity returns null Seebeck,
open-circuit heat conduction and Hall coefficient.

The energy column has the common unit label `[hartree, fs]`; ASE's default
unit mode converts only that column to eV and keeps lifetime in fs. Use
`units="atomic"` for a table whose energies are in Hartree at the ASE entry.
The same bracketed label is registered for the control and returned table.

Finite-frequency thermoelectric transport uses all three causal moments:
`S = -K0^-1 K1/kT`, `Pi = -K1 K0^-1`,
`kappa = (K2-K1 K0^-1 K1)/kT`. The frequency entries add complex `seebeck`
(V/K), `peltier` (V, heat current per electric current) and
`electronic_thermal_conductivity` (W/(m K), zero electric current).
The production complex constitutive inverse uses faer's pivoted LU of its
six-dimensional real representation, after homogeneous normalization.
Normalization preserves the operator; the relative singularity condition
remains `|det(K0/scale)| > 1e-13`. No physical denominator is floored.
`raw_thermoelectric_inverse_residual` measures the normalized inverse
equation; `raw_kelvin_residual` is the unprojected
`max|Pi-kT S^T|` in Hartree/e. These residuals and the derived Seebeck,
Peltier and heat tensors are null when conductivity is singular. The imposed thermal drive uses a local
equilibrium bath, without hydrodynamic collision or magnetic heat-current
vertex corrections. Independent heat-current fits give exponents
1.999258363 (real) and 1.999679831 (imaginary), with R^2 above 0.999999963.
The cofactor, static, reversal and tiny-carrier-weight checks differ by at
most 3.667e-16; raw inverse and Kelvin residuals are 1.869e-16 and
1.552e-17. One- and four-thread lifetime evaluations give identical sums.

Rust providers `periodic::scattering::{transport_integrals, hall_integrals,
frequency_transport}` accept a positive lifetime callback, which can capture
temperature and chemical potential. No registered functional name appears
in this quadrature. The same converged band operator supplies semilocal,
hybrid and ECP calculations. Scalar cluster lifetimes retain zero-field
degenerate-subspace velocity products; the ordinary Hall model still needs
isolated bands. Interband coherence, induced reference vertices, a microscopic
collision operator and reduced-dimensional combinations are separate targets.
Rayon evaluates sufficiently large sets of independent scalar weights;
the tensor sums retain state order for deterministic quadrature.

`p128_energy_scattering` passes seven native tests: independent dense moments
and Hall sums (maximum defect 2.221e-16), constant-time, static, frequency
reversal and energy-origin limits (maximum defect 2.483e-16), invalid inputs,
and four-step electric, Hall and heat-current response fits. Raw Onsager defect of the isotropic shell oracle
is exactly zero, without projection. Public-route acceptance passes eleven
tests for each CLI name, including native Python, the generic driver, ASE
and the canonical `dft-rs` executable, with SVWN5, PBE0 and TPSS references.
The `[hartree, fs]` unit label passes the existing common validation;
ASE's default eV conversion changes the energy column only. The coarse
four-by-four-by-four public fixtures have large raw Hall Onsager defects:
20.2450520316602 (SVWN5), 11.623732681700451 (PBE0) and
13.937573629753144 (TPSS), with relative defects 0.9558096721,
1.3567101035 and 0.9728511852. These values are reported with warnings,
without projection; this sampling does not establish converged Hall
reciprocity. Fock-image residuals are at most 5.053e-15 Hartree.
Full measured errors and the
unsuccessful wider interval are recorded in [response.md](response.md).

## 35. Exchange-enhanced static spin susceptibility

`pauli-spin-susceptibility` with `exchange_enhanced=true` solves the full
static semilocal spin and charge screening equation through the existing
`PeriodicDensitySolver`. Generic LDA/GGA/meta-GGA derivatives supply the
kernel without dispatch on a registered functional name. The source is
`dF_alpha/dB=+g S/4`, `dF_beta/dB=-g S/4`, and
`M=-g(N_alpha-N_beta)/4`. Spin populations exchange particles at fixed
total number. A polarized reference retains its baseline spin bias
`b=(mu_alpha-mu_beta)/2`; this is a constrained spin-biased response,
which need not equal the zero-bias equilibrium susceptibility.

Two grand-canonical density solves determine the Zeeman response `x_Z`
and the common-reservoir response `x_mu` to source `-S` in both spins.
The physical constraint is `mu'=-deltaN_Z/deltaN_mu` and
`x=x_Z+mu' x_mu`. A zero compressibility and zero unconstrained number
change give an undetermined multiplier, reported as null; an incompatible
nonzero number change is rejected. No denominator floor is used.
Equal-channel references recover spin cancellation and zero common
multiplier. Unequal channels can redistribute charge locally, so the
raw spatial sum of spin densities is retained even when its cell overlap
trace vanishes. Independently fixing each spin number would remove the
Pauli occupation response and is not the ensemble calculated here.

Each positive `temperatures_k` entry has its own self-consistent thermal
reference. `fermi_k_mesh`, when supplied, becomes both the SCF and
response mesh. The default `spin_g_factor` is 2. The default
`response_reciprocal_cutoff` is 16 inverse Bohr for unrestricted or
non-singlet references, and remains 6 for restricted singlets. An explicit
cutoff overrides either default. Common solver defaults are relative
residual 1e-10, 500 iterations and GMRES restart 40. A small equation
residual certifies the configured operator, not convergence of its
reciprocal Hartree representation; a cutoff scan remains necessary.
Negative susceptibility and charge compressibility are retained and
flagged as instabilities of the stationary reference.

Reports contain screened and bare susceptibility per cell in atomic
units, their ratio, and dimensionless SI volume susceptibility
`4 pi alpha^2 chi_cell/Omega`. Bare response uses the same orbitals and
occupations with the induced kernel removed and its own common-number
multiplier. Per-spin thermally broadened DOS `D_alpha,D_beta` is evaluated
at each spin's own reference chemical potential. Its independent oracle
is `(g/4)^2 4 D_alpha D_beta/(D_alpha+D_beta)`, zero when both DOS values
vanish. A restricted reference reduces this to `(g/4)^2 D_total`.

Rows include `spin_chemical_potentials`, `reference_spin_bias`,
`chemical_potential_response`, `bare_chemical_potential_response`,
`charge_compressibility`, `bare_charge_compressibility`,
`unconstrained_delta_spin_numbers`, constrained `delta_spin_numbers`,
raw bare/screened number defects, spatial spin-density sum, Hermiticity,
reference residuals and solver residuals. `constrained_equation_residual`
is reevaluated with the actual density operator and constrained source:
relative L2, or absolute L2 for an exactly zero source. The compatibility
`chemical_potential` field is the average of the two reference values,
null if either is indeterminate. Native energy and chemical potentials
are Hartree; common multiplier is Hartree per atomic magnetic field;
compressibility and DOS are per Hartree per cell. ASE conversion uses
eV and inverse eV for these quantities and leaves the field atomic.
Raw count and equation residuals are unchanged by that conversion.

Rust entry: `periodic::spin_susceptibility::longitudinal_spin_susceptibility`.
CLI, native Python, generic feature dispatch and ASE use the same driver
in all three [API tables](rust-api.md#common-calculation-driver).

With `exchange_enhanced=false`, the rigid-band branch now accepts both
explicit spin channels. `periodic::fermi_thermodynamics::collinear_fermi_thermodynamics`
reuses the existing common Fermi-level solve with weight 1/N_k per spin
level. It reoccupies the frozen bands at each requested temperature with
one chemical potential and zero spin reservoir bias. The frozen orbital
reference can have its own spin-population constraints; those are not
imposed on the thermodynamic reoccupation. Reports retain its original
spin chemical potentials separately as
`orbital_reference_spin_chemical_potentials`.

Writing D=D_alpha+D_beta, the rigid common-number multiplier is
`mu'=(g/4)(D_alpha-D_beta)/D`, and the spin DOS is
`D_spin=4 D_alpha D_beta/D`. Susceptibility is `(g/4)^2 D_spin`.
For inert bands D=0, susceptibility is zero and the multiplier is null;
no floor is used. `spin_electrons`, `spin_chemical_potentials`,
`fixed_number_spin_dos`, `rigid_band_spin_magnetization`,
`chemical_potential_response`, `delta_spin_numbers`,
`raw_charge_constraint_residual` and `thermal_number_residual` expose the
reoccupation and its raw conservation defects. This is a frozen-band
quantity, rather than a self-consistent derivative of the orbital
reference. The enhanced branch instead fixes the baseline reference
spin bias and reconverges each thermal reference as described above.

Bare heat capacity remains the fixed-total-number derivative of the
physical band energy, using the two-channel moments L_n. The
Sommerfeld estimate uses the total DOS D; the Wilson ratio uses
`(pi^2/3) kT D_spin/C`, reducing to the original formula for equal
channels. `spin_g_factor` defaults to 2 in both modes. This algebra
does not require a functional kernel, so compatible frozen hybrid bands
use the same route. Correlated PT2, orbital and SOC responses are separate
pending calculations. Acceptance of this bare extension is tracked in
`p139_rigid_spin_thermodynamics` and response.md.

Acceptance uses `periodic::scf::fock_at_density` for independent nonlinear
finite-Zeeman-field SCF at fixed total number and fixed baseline bias,
without calling the response kernel. `p129_spin_susceptibility` preserves
the restricted SVWN5/PBE/TPSS four-step fits and bare/g-scaling limits.
`p138_periodic_spin_constraint` additionally validates susceptibility,
the common chemical-potential derivative and every AO density response
entry for unequal channels. On four fields [0.08,0.04,0.02,0.01] atomic
units, measured exponents are 1.987114863..2.000141917 with
R^2 >= 0.999988836549. Maximum Richardson defect is 2.481e-10,
fixed-number defect 1.111e-16 and fresh equation residual 9.163e-11.
The Hartree/common-number limit and g-squared susceptibility/g-linear
multiplier scaling pass. The failed cutoff-8 interval, resolved cutoff-16
intervals, full raw errors and public-route checks are recorded in
[response.md](response.md).

The enhanced class is static longitudinal spin-only response for
3D collinear semilocal references. Its equilibrium common-number constraint
does not establish a dynamic response. A separate finite-q/frequency
spin-only connection is described below; complete hybrid, orbital/SOC,
other dimensional/electronic spaces and correlated spin coverage remains
separate.

## 36. Longitudinal spin-Zeeman response at finite q and frequency

`periodic-spin-response`, native `periodic_spin_response` and ASE
`get_periodic_spin_response` connect to
`dfpt::spin::periodic_longitudinal_spin_response`. The new connection is
under independent component and public-route acceptance; registration is
not completion of the complete magnetic DFPT program.

The probe is longitudinal along the collinear reference's spin axis:

```text
N_mn(q) = <chi_m(k+q)|exp(i q.r)|chi_n(k)>
H'_alpha(q) = +g B_z N(q)/4
H'_beta(q)  = -g B_z N(q)/4
delta M_z(q) = -g [delta N_alpha(q)-delta N_beta(q)]/4
chi_zz(q,z) = delta M_z(q)/B_z
```

The existing shifted-Bloch density solver supplies semilocal spin screening
and full/range-separated exchange. Its actual reference and provider guards
are retained. This local spin source does not introduce an orbital vector
potential, transverse spin, spin relaxation or SOC; an orbital-current
calculation needs its own covariant vertices. In particular, density-kernel
support for meta-GGA does not establish meta-GGA orbital current response.

`response_q` is Cartesian inverse Bohr; `frequency`/`response_omega` and
`frequency_damping`/`response_damping` are Hartree angular frequencies.
`imaginary=True` selects the imaginary axis. `spin_probe_field=[real,imag]`
is the reduced magnetic-field amplitude (default `[1,0]`) and
`spin_g_factor` defaults to 2. With `exp(i q.r-i z.t)`, densities and moments
are complex per-cell Fourier amplitudes; the final scalar axis is
`[real,imag]`. The susceptibility is null for an exactly zero probe,
rather than division by an artificial field floor. AO response densities,
raw residuals, occupation consistency and reference controls remain
separate numerical evidence.

The equilibrium Pauli and collisionless limits differ. Without spin-orbit
coupling or spin relaxation, a spatially uniform Zeeman operator commutes
with each collinear spin Hamiltonian, so its finite-frequency population
response vanishes. The equilibrium static susceptibility can be nonzero
because Fermi populations redistribute between spins at fixed total number.
Spatially uniform static requests therefore use section 35 and are refused
by the new dynamic route. A static finite-temperature probe whose q is a
reciprocal-lattice vector also requires a common-number multiplier for its
intra-cell perturbation and is refused here; proximity to a reciprocal
vector is identified only within floating-point lattice roundoff. No
separate fixed-spin constraint is silently substituted for that ensemble.

`tests/periodic_longitudinal_spin.rs` uses a C1 four-centre Gaussian model,
an independent closed s-Gaussian Fourier source, every complex AO density
entry against a dense solve, collisionless/static-limit checks and
independent nonlinear field SCFs for four-step fits of all real symmetric
spin-density components for cosine and sine probes. The 120 PBE/TPSS/PBE0
component fits passed. A further 40 strict TPSS fits on steps
0.01, 0.005, 0.0025 and 0.00125 give p=1.994431..2.005896 and minimum
R2=0.99999772, meeting |p-2|<0.02 and R2>0.99999. Source/library hashes
and raw errors are retained in the local crystal validation bundle and the
existing physical benchmark report. Frozen public artifact02 subsequently
passed six native/ASE/CLI tests and the Rust matrix, field and spin
acceptance checks. The retained first unit-conversion fixture failure was
resolved by converting its reference controls to ASE units, without changing
the physical calculation or loosening numerical thresholds.
These checks do not certify real-time propagation, all functional/reference
combinations or converged material susceptibilities.

## 37. Log-volume spectral derivatives

`periodic::gruneisen::mode_volume_derivatives(D, D_x, D_xx)` differentiates
every eigenvalue branch of a real symmetric mass-weighted dynamical matrix,
where `x = ln(V/V0)`. The derivative matrices are analytic inputs. It first
diagonalizes `D_x` within a degenerate eigenspace, then the effective second
derivative within any remaining degeneracy. Coupling to other eigenspaces
is retained in the second derivative. Only roundoff-sized asymmetry and
degeneracy are merged, with their raw defects/eigenvalues preserved; small
optical eigenvalues and inter-subspace gaps are not regularized.

The returned conventions are `gamma = -d ln(omega)/dx` and
`curvature = d2 ln(omega)/dx2 = -d gamma/dx`. The latter differs from
`d ln(gamma)/dx`. Zero and imaginary frequencies have undefined logarithmic
derivatives. A crystal caller must also identify uniform translations and
mark their coefficients undefined; this helper does not enforce an acoustic
sum rule or establish a stationary reference.

`tests/periodic_gruneisen_spectral.rs` fits both derivatives of every mode
of generic four-dimensional models, including a threefold degeneracy with
a residual twofold first-derivative degeneracy. Independent mode tracking
uses eigenvector overlaps at all four volume steps. All sixteen fits on
0.04, 0.02, 0.01 and 0.005 passed: p=1.998955..2.001672 and minimum
R2=0.99999981. Closed analytic frequency laws, basis invariance,
instabilities and invalid inputs are checked separately. The new module
was compiled with warnings denied on the accepted linear-algebra library.
The additional material-to-physical isotropic strain chain rule has 42
independent first/second matrix-component fits on steps 0.08, 0.04, 0.02
and 0.01, all passing the same gates. The real-space force constants are
converted with `Phi_phys = Phi(s)/(1+2s)` and
`ln(V/V0) = 3 ln(1+2s)/2`; omitting either conversion changes both derivatives.

An additional analytic lattice path is staged as
`response::nuclear::lattice_volume_gruneisen`, selected by
`mode-gruneisen` with `gruneisen_order=2` through the common driver.
`relaxed_ion=False` holds fractional coordinates fixed. Relaxed ions include
both first and second internal relaxation and require measured internal
stationarity within `gruneisen_force_tolerance` (default 1e-5 Hartree/Bohr)
and positive optical force constants. This new tolerance does not alter SCF
or response residual gates. Reference pressure and raw ionic/response
residuals are returned. The route is isotropic and rejects shear requests.
The ordinary order-one route remains the default. Code is staged; full
artifact/public-route acceptance, independent crystal-path derivative checks
and converged material validation remain pending. These changes are not
included in frozen artifact02.

Both analytic orders accept geometry-only periodic D2, D3(zero), D3(BJ),
their available three-body variants, and the functional-required CHG tail.
The same fixed image list enters the derivatives for the reference pressure,
force constants, and internal relaxation. The `dispersion_energy` field is
per supercell and `dispersion_energy_per_primitive_cell` is per primitive
cell, both in Hartree. `series_reference_energy_difference` compares two
electronic references at the same geometry, where their dispersion energies
cancel. `response_residuals` reports the raw relative Krylov residuals for
both orders. Dispersion-corrected acceptance results are tracked under
[R030](response.md#r030-gruneisen-quasiharmonic).

The mixed-derivative driver builds and retains Taylor coefficients through
the requested energy order. A fourth-order result uses fourth-order
geometry and energy jets with second-order orbital response. The API for
an explicit directional `2n+1` energy series still returns its full
series. Paired run-time and peak-memory measurements of the mixed-path
change are pending.


## Particle-number diagnostics and fixed numerical supports


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

A fixed electron trace in an insufficient overlap metric does not establish the real-space particle number. The converged image metric, AO reach and quadrature must agree independently. Choose converged numerical settings before a strain/field/displacement chart and inspect support-topology changes; automatically growing a cutoff inside a derivative chart can create jumps. The advisory is therefore diagnostic, not an adaptive convergence algorithm. Second-order volume response retains the neutral all-electron insulating path-series scope and its charged-cell/background/ECP refusals.
