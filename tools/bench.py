# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Benchmark repeated CLI calculations with matching resource settings.

Supply both builds to run each candidate trial followed by a baseline trial.
`--save` writes a JSON snapshot and `--compare` reads previous timings.

Live baseline comparisons and repeated candidate trials must agree in every
retained observable within --observable-atol + --observable-rtol * abs(reference).
Both tolerances default to 1e-8 in the reported atomic units. Failures retain
their timing samples but suppress speed reporting and return a nonzero status.
A historical --compare file supplies timing context only; use --baseline-exe
for matched numerical acceptance against the previous executable.

Deliberately modest sizes: the point is to be re-runnable in minutes, not to
showcase the largest system that fits.

    python tools/bench.py                       # run everything
    python tools/bench.py --only hess           # substring filter
    python tools/bench.py --save base.json
    python tools/bench.py --baseline-exe OLD --exe NEW --save comparison.json
    python tools/bench.py --compare base.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EXE = os.path.join(
    ROOT, "target-perf", "release",
    "dft-rs.exe" if os.name == "nt" else "dft-rs",
)


def _benzene() -> list[str]:
    out = ["12", "benzene"]
    for k in range(6):
        a = k * math.pi / 3.0
        out.append(f"C {1.39 * math.cos(a):.6f} {1.39 * math.sin(a):.6f} 0.000000")
    for k in range(6):
        a = k * math.pi / 3.0
        out.append(f"H {2.48 * math.cos(a):.6f} {2.48 * math.sin(a):.6f} 0.000000")
    return out


GEOMETRIES = {
    "hydrogen_magnetic_slab": ["1", "Hydrogen in a square magnetic slab", "H 0 0 0"],
    "helium_hydride": ["2", "Finite Gaussian nuclei for Dirac-Coulomb HF",
                      "He  0.063501265 -0.142877847  0.047625949",
                      "H   0.746139267  0.164044935 -0.243421517"],
    "biased_h2": ["2", "H2 with a fixed spinor operator",
                  "H  0.071 -0.101  0.103",
                  "H  0.834  0.073 -0.042"],
    "open_h3_cell": ["3", "Asymmetric open-shell cell",
                     "H  0.0687930374 -0.1111272143  0.0370424048",
                     "H  1.0107286728  0.1058354422 -0.1005436701",
                     "H  0.2116708844  0.8996012585  0.1746284796"],
    "lithium_ion_cell": ["1", "Li+ site in a triclinic cell",
                         "Li  0.0899601259  0.2169626565 -0.1005436701"],
    "lithium_hydride": ["2", "Tilted lithium hydride",
                        "Li  0.07 -0.12 0.19", "H  0.86 0.47 1.39"],
    "scalar_ecp_cell": ["3", "Asymmetric scalar ECP cell",
                        "H   0.0687930374 -0.1111272143  0.0370424048",
                        "Na  1.0583544211  0.2381297448 -0.1640449353",
                        "H  -0.2434215169  0.6826386014  0.8572676815"],
    "water": ["3", "water",
              "O 0.000000  0.000000  0.117300",
              "H 0.000000  0.757200 -0.469200",
              "H 0.000000 -0.757200 -0.469200"],
    "distorted_water": ["3", "distorted water",
                        "O 0.06879303741739 -0.11112721428963 0.03704240476321",
                        "H 0.83080822111771 0.10054367007157 0.58738670410233",
                        "H -0.25929683334247 0.75143163948226 0.35984050341404"],
    "displaced_ammonia": ["4", "distorted ammonia cation",
                          "N 0.079376581635449986 -0.14287784694381001 0.10054367007156999",
                          "H 0.94193543540733993 0.17992025170702000 -0.042334176872239997",
                          "H -0.25929683334247000 0.87314239798994986 0.21696265647022997",
                          "H 0.16404493537992998 -0.41275822450433997 1.0107284728247299"],
    "ethanol": ["9", "ethanol",
                "C -0.748 -0.015  0.024", "C  0.558  0.775  0.038",
                "O  1.693 -0.069 -0.011", "H -0.716 -0.678  0.898",
                "H -0.815 -0.640 -0.877", "H -1.637  0.621  0.076",
                "H  0.609  1.412 -0.851", "H  0.593  1.416  0.930",
                "H  2.470  0.492  0.023"],
    "benzene": _benzene(),
    "water_dimer": ["6", "water dimer",
                    "O 0.000 0.000 0.000", "H 0.000 0.000 0.957",
                    "H 0.927 0.000 -0.240",
                    "O 0.000 0.000 2.980", "H 0.000 0.757 3.549",
                    "H 0.000 -0.757 3.549"],
    "nico4": ["9", "Ni(CO)4",
              "Ni 0.000  0.000  0.000",
              "C  1.052  1.052  1.052", "O  1.716  1.716  1.716",
              "C -1.052 -1.052  1.052", "O -1.716 -1.716  1.716",
              "C  1.052 -1.052 -1.052", "O  1.716 -1.716 -1.716",
              "C -1.052  1.052 -1.052", "O -1.716  1.716 -1.716"],
}

BASIS_INPUTS = {
    "compact_hydrogen.gbs": "H 0\nS 1 1.0\n0.7 1.0\n****\n",
}
MAGNETIC_SLAB_EDGE_ANGSTROM = 9.0 * 0.529177210903
TRICLINIC_CELL_ANGSTROM = (
    "4.445088572,0.370424048,0.158753163;"
    "0.211670884,4.709677177,0.317506327;"
    "0.105835442,-0.264588605,4.868430340"
)

PERIODIC_CORRELATED_OPTIONS = [
    "--xc", "b2plyp", "--basis", "sto-3g", "--dispersion", "none",
    "--cell", TRICLINIC_CELL_ANGSTROM,
    "--k-mesh", "1,1,1", "--overlap-cutoff", "5.5", "--xc-ao-cutoff", "16",
    "--grid-level", "1", "--grid-shell", "1", "--grid-weight-cutoff", "0",
    "--conv-tol", "1e-12", "--conv-tol-grad", "1e-11", "--conv-tol-dp", "1e-11",
    "--response-tolerance", "1e-11",
]

CHARGED_LI_OPTIONS = [
    "--reference", "rks", "--xc", "svwn5", "--basis", "sto-3g", "--charge", "1",
    "--cell", TRICLINIC_CELL_ANGSTROM, "--k-mesh", "1,1,1", "--overlap-cutoff", "13.8",
    "--xc-ao-cutoff", "24", "--grid-level", "1", "--grid-shell", "1",
    "--grid-weight-cutoff", "0", "--max-iter", "500",
    "--conv-tol", "1e-13", "--conv-tol-dp", "1e-12", "--conv-tol-grad", "1e-11",
]

# label, subcommand, geometry, extra arguments
CASES = [
    ("scf/slab/magnetic-m06l", "periodic-magnetic-scf", "hydrogen_magnetic_slab", [
        "--xc", "m06l", "--basis", "compact_hydrogen.gbs",
        "--cell", f"{MAGNETIC_SLAB_EDGE_ANGSTROM:.17g},0,0;0,{MAGNETIC_SLAB_EDGE_ANGSTROM:.17g},0;0,0,0",
        "--pbc", "1,1,0", "--magnetic-model", "finite_gamma",
        "--magnetic-field", f"0,0,{math.tau / 81:.17g}",
        "--overlap-cutoff", "3", "--magnetic-reciprocal-cutoff", "14",
        "--magnetic-open-order", "64", "--nuclear-widths", ".3",
        "--magnetic-grid-order", "48", "--grid-cutoff", "8",
        "--ewald-alpha", ".55", "--ewald-real-cutoff", "14",
        "--ewald-reciprocal-cutoff", "8", "--conv-tol", "1e-13",
        "--conv-tol-dp", "1e-12", "--conv-tol-grad", "1e-11",
    ]),
    ("response/azimuth/gks/m06/sto3g", "polarizability", "biased_h2", [
        "--reference", "gks", "--xc", "m06", "--basis", "sto-3g",
        "--grid-level", "0", "--grid-weight-cutoff", "0",
        "--conv-tol", "1e-12", "--conv-tol-grad", "1e-10", "--conv-tol-dp", "1e-10",
        "--response-tolerance", "1e-10",
    ]),
    ("gradient/gks/b2plyp/lih", "gradient", "lithium_hydride", [
        "--reference", "gks", "--xc", "b2plyp", "--basis", "sto-3g",
        "--grid-level", "0", "--grid-weight-cutoff", "0",
        "--conv-tol", "1e-13", "--conv-tol-grad", "1e-12", "--conv-tol-dp", "1e-12",
    ]),
    *[(f"gks/closed-shell/{xc}/sto3g", "orbitals", "distorted_water", [
        "--reference", "gks", "--xc", xc, "--basis", "sto-3g",
        "--grid-level", "2", "--vv10-grid-level", "0", "--grid-weight-cutoff", "0",
        "--conv-tol", "1e-11", "--conv-tol-grad", "1e-9", "--conv-tol-dp", "1e-9",
        "--max-iter", "200", "--density", "true",
    ]) for xc in ("wb97x-d", "wb97x-v", "wb97m-v")],
    ("scf/hf/sto3g",       "energy",      "water",       ["--xc", "hf", "--basis", "sto-3g"]),
    ("scf/dirac-coulomb/sto3g", "dirac-coulomb-scf", "helium_hydride", [
        "--basis", "sto-3g", "--charge", "1", "--speed-of-light", "7.4",
        "--nuclear-rms-radii", "[0.12,0.081]", "--conv-tol", "2e-13",
        "--conv-tol-grad", "1e-11", "--conv-tol-dp", "1e-11",
    ]),
    ("scf/b3lyp/631g",     "energy",      "benzene",     ["--xc", "b3lyp", "--basis", "6-31g"]),
    ("scf/pbe0/def2svp",   "energy",      "ethanol",     ["--xc", "pbe0", "--basis", "def2-svp"]),
    ("scf/pbe0/631g",      "energy",      "water",       ["--xc", "pbe0", "--basis", "6-31g", "--grid-level", "1"]),
    ("scf/m062x/631g",     "energy",      "ethanol",     ["--xc", "m06-2x", "--basis", "6-31g"]),
    ("scf/wb97mv/631g",    "energy",      "water",       ["--xc", "wb97m-v", "--basis", "6-31g"]),
    ("scf/roks/vv10/wb97x-v/sto3g", "energy", "displaced_ammonia", [
        "--reference", "roks", "--xc", "wb97x-v", "--basis", "sto-3g",
        "--charge", "1", "--multiplicity", "2", "--grid-level", "0",
        "--vv10-grid-level", "0", "--grid-weight-cutoff", "0",
        "--conv-tol", "1e-13", "--conv-tol-dp", "1e-12", "--conv-tol-grad", "1e-12",
        "--max-iter", "300",
    ]),
    ("scf/ecp/lanl2dz",    "energy",      "nico4",       ["--xc", "b3lyp", "--basis", "lanl2dz"]),
    ("grad/b3lyp/631g",    "gradient",    "benzene",     ["--xc", "b3lyp", "--basis", "6-31g"]),
    ("grad/m062x/631g",    "gradient",    "ethanol",     ["--xc", "m06-2x", "--basis", "6-31g"]),
    ("hess/svwn5/sto3g",   "hessian",     "benzene",     ["--xc", "svwn5", "--basis", "sto-3g", "--grid-level", "2"]),
    ("hess/b3lyp/sto3g",   "hessian",     "ethanol",     ["--xc", "b3lyp", "--basis", "sto-3g"]),
    # The same molecular grid and unrestricted settings exercise the shared
    # partition-weight kernel in LDA/GGA and meta-GGA Hessian assembly.
    ("hess/partition-cache/rks/pbe/sto3g", "hessian", "water_dimer", [
        "--xc", "pbe", "--reference", "rks", "--basis", "sto-3g",
        "--charge", "0", "--multiplicity", "1",
        "--dispersion", "none", "--grid-level", "1", "--grid-weight-cutoff", "0",
        "--conv-tol", "1e-12", "--conv-tol-grad", "1e-10", "--conv-tol-dp", "1e-10",
        "--max-iter", "400",
    ]),
    ("hess/partition-cache/uks/pbe/sto3g", "hessian", "water_dimer", [
        "--xc", "pbe", "--reference", "uks", "--basis", "sto-3g",
        "--charge", "1", "--multiplicity", "2",
        "--dispersion", "none", "--grid-level", "1", "--grid-weight-cutoff", "0",
        "--conv-tol", "1e-12", "--conv-tol-grad", "1e-10", "--conv-tol-dp", "1e-10",
        "--max-iter", "400",
    ]),
    ("hess/partition-cache/rks/tpss/sto3g", "hessian", "water_dimer", [
        "--xc", "tpss", "--reference", "rks", "--basis", "sto-3g",
        "--charge", "0", "--multiplicity", "1",
        "--dispersion", "none", "--grid-level", "1", "--grid-weight-cutoff", "0",
        "--conv-tol", "1e-12", "--conv-tol-grad", "1e-10", "--conv-tol-dp", "1e-10",
        "--max-iter", "400",
    ]),
    ("hess/partition-cache/uks/tpss/sto3g", "hessian", "water_dimer", [
        "--xc", "tpss", "--reference", "uks", "--basis", "sto-3g",
        "--charge", "1", "--multiplicity", "2",
        "--dispersion", "none", "--grid-level", "1", "--grid-weight-cutoff", "0",
        "--conv-tol", "1e-12", "--conv-tol-grad", "1e-10", "--conv-tol-dp", "1e-10",
        "--max-iter", "400",
    ]),
    ("freq/pbe/631g",      "frequencies", "water",       ["--xc", "pbe", "--basis", "6-31g"]),
    ("opt/pbe/sto3g",      "optimize",    "water",       ["--xc", "pbe", "--basis", "sto-3g"]),
    ("alpha/b2plyp/sto3g", "polarizability", "water", ["--xc", "b2plyp", "--basis", "sto-3g", "--grid-level", "1"]),
    ("quadrupole/hf/cc-pvdz/static", "quadrupole-polarizability", "water", [
        "--xc", "hf", "--basis", "cc-pvdz", "--dispersion", "none",
        "--conv-tol", "1e-12", "--conv-tol-grad", "1e-10", "--conv-tol-dp", "1e-10",
        "--grid-weight-cutoff", "0",
    ]),
    ("quadrupole/hf/cc-pvdz/retarded", "quadrupole-polarizability", "water", [
        "--xc", "hf", "--basis", "cc-pvdz", "--dispersion", "none",
        "--conv-tol", "1e-12", "--conv-tol-grad", "1e-10", "--conv-tol-dp", "1e-10",
        "--grid-weight-cutoff", "0", "--frequency", "0.04", "--frequency-damping", "0.004",
    ]),
    ("quadrupole/gks/svwn5-target", "quadrupole-polarizability", "biased_h2", [
        "--xc", "hf", "--reference", "gks", "--basis", "sto-3g",
        "--double-hybrid-target-xc", "svwn5",
        "--pt2-same-spin", "0.27", "--pt2-opposite-spin", "0.27",
        "--grid-level", "0", "--grid-weight-cutoff", "0", "--guess", "core",
        "--conv-tol", "1e-13", "--conv-tol-grad", "1e-12", "--conv-tol-dp", "1e-12",
        "--origin", "[0.23,-0.50,0.42]",
        "--gks-operator-real", "[[-0.052,0,0.019,0],[0,0.044,0,-0.022],[0.019,0,0.052,0],[0,-0.022,0,-0.044]]",
        "--gks-operator-imaginary", "[[0,0,0.017,0],[0,0,0,0.011],[-0.017,0,0,0],[0,-0.011,0,0]]",
    ]),
    ("tddft/b3lyp/631g",   "tddft",       "water",       ["--xc", "b3lyp", "--basis", "6-31g", "--nstates", "5"]),
    *[(f"nmr/{reference}/hf/sto3g", "nmr-spectrum", "water", [
        "--reference", reference, "--xc", "hf", "--basis", "sto-3g", "--dispersion", "none",
        "--gauge", "common-origin", "--grid-level", "1", "--grid-weight-cutoff", "0",
        "--conv-tol", "1e-12", "--conv-tol-grad", "1e-10", "--conv-tol-dp", "1e-10",
        "--response-tolerance", "1e-10", "--nmr-couplings", "true", "--nmr-spin-model", "exact",
        "--nmr-reference-elements", "1", "--nmr-reference-shieldings", "30",
        "--nuclear-spins", "0,1,1", "--nuclear-g", "0,5.5856946893,5.5856946893",
        "--nmr-points", "401",
    ]) for reference in ("roks", "gks")],
    ("solv/cpcm/b3lyp",    "energy",      "ethanol",     ["--xc", "b3lyp", "--basis", "6-31g", "--solvent", "water"]),
    ("solv/ief/b3lyp",     "energy",      "ethanol",     ["--xc", "b3lyp", "--basis", "6-31g", "--solvent", "ief:water"]),
    ("sapt/pbe0/sto3g",    "sapt",        "water_dimer", ["--xc", "pbe0", "--basis", "sto-3g", "--natoms-a", "3"]),
    ("periodic/hf/sto3g",  "periodic-energy", "water", ["--xc", "hf", "--cell", "6", "--overlap-cutoff", "5"]),
    ("stress/periodic/hf/scalar-ecp", "stress", "scalar_ecp_cell", [
        "--reference", "rks", "--xc", "hf", "--charge", "1",
        "--basis", os.path.join(ROOT, "tests", "data", "periodic_scalar_ecp.g94"),
        "--cell", TRICLINIC_CELL_ANGSTROM,
        "--k-mesh", "1,1,1", "--overlap-cutoff", "5.5", "--dispersion", "none",
        "--forces", "true", "--conv-tol", "1e-12", "--conv-tol-grad", "1e-11",
        "--conv-tol-dp", "1e-11",
    ]),
    ("stress/periodic/svwn5/charged-li", "stress", "lithium_ion_cell", [
        *CHARGED_LI_OPTIONS, "--forces", "false",
    ]),
    ("elastic/periodic/svwn5/charged-li", "elastic", "lithium_ion_cell", [
        *CHARGED_LI_OPTIONS, "--response-tolerance", "1e-11",
    ]),
    ("energy/periodic/b2plyp/sto3g", "periodic-energy", "biased_h2", PERIODIC_CORRELATED_OPTIONS),
    ("response-cache/forces/b2plyp/sto3g", "periodic-energy", "helium_hydride", [
        *PERIODIC_CORRELATED_OPTIONS, "--charge", "1", "--forces", "true",
    ]),
    ("response-cache/phonons/b2plyp/sto3g", "phonons", "helium_hydride", [
        *PERIODIC_CORRELATED_OPTIONS, "--charge", "1",
    ]),
    *[(f"high-order/periodic/{feature}/pbe/sto3g", feature, "lithium_ion_cell", [
        "--xc", "pbe", "--basis", "sto-3g", "--charge", "1",
        "--cell", TRICLINIC_CELL_ANGSTROM,
        "--k-mesh", "1,1,1", "--overlap-cutoff", "7.0",
        "--grid-level", "0", "--grid-shell", "1", "--grid-weight-cutoff", "0",
        "--conv-tol", "1e-12", "--conv-tol-grad", "1e-10", "--conv-tol-dp", "1e-10",
        "--response-tolerance", "1e-10",
    ]) for feature in ("toec", "foec")],
    ("periodic/roks/pbe/sto3g", "periodic-energy", "open_h3_cell", [
        "--reference", "roks", "--xc", "pbe", "--basis", "sto-3g",
        "--cell", "4.498006293,0,0;0.1640449354,4.9213480623,0;0.1428778469,-0.1005436701,5.3446898306",
        "--multiplicity", "2", "--overlap-cutoff", "6", "--xc-ao-cutoff", "16",
        "--grid-level", "1", "--grid-shell", "1", "--grid-weight-cutoff", "0",
        "--guess", "core", "--damping", "0", "--no-adiis",
        "--conv-tol", "1e-13", "--conv-tol-grad", "1e-10", "--conv-tol-dp", "1e-11",
        "--max-iter", "300",
    ]),
]


def _wait_peak_rss(proc):
    if os.name == "nt":
        import ctypes as ct
        from ctypes import wintypes as win

        class Counters(ct.Structure):
            _fields_ = [("size", win.DWORD), ("faults", win.DWORD),
                        ("peak", ct.c_size_t), ("remaining", ct.c_size_t * 7)]

        kernel = ct.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [win.DWORD, win.BOOL, win.DWORD]
        kernel.OpenProcess.restype = win.HANDLE
        kernel.CloseHandle.argtypes = [win.HANDLE]
        memory_info = kernel.K32GetProcessMemoryInfo
        memory_info.argtypes = [win.HANDLE, ct.POINTER(Counters), win.DWORD]
        memory_info.restype = win.BOOL
        handle = kernel.OpenProcess(0x0410, False, proc.pid)
        proc.wait()
        if not handle:
            raise ct.WinError(ct.get_last_error())
        try:
            counters = Counters()
            counters.size = ct.sizeof(counters)
            if not memory_info(handle, ct.byref(counters), counters.size):
                raise ct.WinError(ct.get_last_error())
            return counters.peak
        finally:
            kernel.CloseHandle(handle)

    _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    return usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)


def _measure(argv):
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as error:
        start = time.perf_counter()
        with subprocess.Popen(argv, stdout=output, stderr=error) as proc:
            peak_rss = _wait_peak_rss(proc)
            elapsed = time.perf_counter() - start
        output.seek(0)
        error.seek(0)
        stdout, stderr = (file.read().decode("utf-8", errors="replace") for file in (output, error))
    trial = {"seconds": elapsed, "peak_rss_bytes": peak_rss, "exit_code": proc.returncode}
    if proc.returncode != 0:
        trial["error"] = (stderr or stdout).strip()
        return trial
    try:
        report = json.loads(stdout)
        if report.get("converged") is False:
            raise ValueError("calculation did not converge")
        if report.get("response_converged") is False:
            raise ValueError("response did not converge")
        if any(key in report for key in ("quadrupole_polarizability",
                                        "quadrupole_polarizability_real")):
            trial["observables"] = {key: report[key] for key in (
                "energy", "quadrupole_polarizability", "quadrupole_polarizability_real",
                "quadrupole_polarizability_imag", "parent_quadrupole_polarizability",
                "target_quadrupole_polarizability_correction",
                "pt2_quadrupole_polarizability_correction",
            ) if key in report}
        elif report.get("feature") == "hessian":
            # A faster derivative kernel must preserve every force constant,
            # atomic polar tensor element and its fresh equation residual.
            trial["observables"] = {key: report[key] for key in (
                "energy", "hessian", "dipole_derivatives",
                "orbital_response_residuals", "scf_orbital_residual",
            )}
        elif report.get("feature") == "periodic-magnetic-scf":
            # Preserve the variational operator and all spatial checks at the
            # stationary density when comparing magnetic quadrature timings.
            trial["observables"] = {key: report[key] for key in (
                "energy", "energy_components", "density_real", "density_imag",
                "fock_real", "fock_imag", "scf_iterations", "magnetic_grid_order",
                "orbital_residual", "particle_number_error", "coulomb_operator_defect",
                "image_operator_defect", "xc_grid_energy_defect",
                "xc_grid_operator_defect", "xc_grid_number_defect",
            )}
        elif report.get("feature") == "nmr-spectrum":
            # Timing acceptance includes every shielding, coupling and sampled
            # spectral component. Energy agreement alone cannot accept a faster
            # response kernel; degenerate eigenvectors need not match individually.
            trial["observables"] = {key: report[key] for key in (
                "energy", "shielding_tensors", "diamagnetic_tensors", "paramagnetic_tensors",
                "chemical_shifts", "coupling_pairs", "coupling_constants_hz",
                "spectrum_axis", "spectrum_intensity", "spin_hilbert_dimension",
            )}
        elif report.get("feature") == "elastic":
            # Preserve the ordered strain Hessian before the Voigt projection.
            trial["observables"] = {key: report[key] for key in (
                "energy", "stress", "strain_derivative", "strain_hessian",
                "internal_strain", "internal_relaxation", "force_constants",
                "clamped_ion", "relaxed_ion", "clamped_ion_birch_gpa",
                "relaxed_ion_birch_gpa",
            )}
        elif report.get("feature") == "gradient" and "generalized_pt2_gradient" in report:
            trial["observables"] = {key: report[key] for key in (
                "energy", "gradient", "forces", "parent_gradient", "generalized_pt2_gradient",
                "pt2_gradient", "target_gradient", "dispersion_gradient",
            )}
        elif "stress" in report:
            # Preserve the ordered derivative, term decomposition and all
            # requested forces as well as the symmetric physical stress.
            trial["observables"] = {key: report[key] for key in (
                "energy", "strain_derivative", "strain_derivative_terms", "stress",
                "stress_voigt", "pressure", "forces",
            ) if key in report}
        elif "energy" in report:
            trial["observables"] = {"energy": report["energy"]}
            if report.get("feature") == "dirac-coulomb-scf":
                trial["observables"]["energy_components"] = report["energy_components"]
            elif report.get("feature") == "orbitals" and report.get("reference_type") == "gks":
                # Occupied projectors and Fock elements are gauge independent.
                # Comparing individual degenerate orbital vectors is not.
                trial["observables"].update({key: report[key] for key in (
                    "energy_components", "spin_expectation", "mulliken", "lowdin",
                )})
                trial["observables"]["channels"] = [
                    {key: channel[key] for key in (
                        "energies", "occupations", "density_real", "density_imaginary",
                        "fock_real", "fock_imaginary",
                    )} for channel in report["channels"]
                ]
            elif report.get("double_hybrid_model") is not None and "series_reference_residual" in report:
                trial["observables"].update({key: report[key] for key in (
                    "energy_components", "orbital_reference_energy", "pt2_same_spin_alpha",
                    "pt2_opposite_spin", "pt2_same_spin_beta",
                )})
            # A response optimization must retain every requested physical
            # component. Energy equality alone cannot validate path reuse.
            trial["observables"].update({key: report[key] for key in (
                "polarizability", "alpha_real", "alpha_imag", "response_residuals",
                "forces", "force_constants", "raw_force_constants",
                "supercell_force_constants", "frequencies_cm1",
                "force_constant_tensor", "contracted_force_constants",
                "dispersion_force_constants", "dispersion_force_constant_tensor",
                "elastic_constants_clamped", "elastic_constants_relaxed",
                "high_order_clamped", "high_order_relaxed", "internal_relaxation",
                "second_internal_relaxation", "dispersion_energy_per_primitive_cell",
            ) if key in report})
        trial["managed_peak_bytes"] = report["resources"]["managed_peak_bytes"]
        if "path_evaluations" in report:
            trial["path_evaluations"] = report["path_evaluations"]
        trial["scf_iterations"] = report.get("scf_iterations")
        trial["residuals"] = {key: value for key, value in report.items() if "residual" in key}
        if report.get("feature") == "elastic":
            trial["residuals"].update({key: report[key] for key in ("strain_solves", "displacement_solves")})
        if report.get("feature") == "dirac-coulomb-scf":
            for key in ("metric_defect", "density_change", "coulomb_hermiticity_defect", "particle_number_error"):
                trial["residuals"][key] = report[key]
        log_values = re.findall(r"(?:SCF residual\|\s*=|residual=)\s*([0-9.eE+-]+)", stderr)
        if log_values:
            trial["last_logged_scf_residual"] = float(log_values[-1])
    except (ValueError, KeyError, TypeError, AttributeError) as exception:
        trial["error"] = str(exception)
    return trial


def _observable_mismatch(actual, expected, atol, rtol, path="observables"):
    """Compare every retained tensor component before accepting a timing trial."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            return f"{path}: observable keys differ"
        for key in expected:
            mismatch = _observable_mismatch(actual[key], expected[key], atol, rtol, f"{path}.{key}")
            if mismatch:
                return mismatch
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return f"{path}: tensor shapes differ"
        for index, (value, reference) in enumerate(zip(actual, expected)):
            mismatch = _observable_mismatch(value, reference, atol, rtol, f"{path}[{index}]")
            if mismatch:
                return mismatch
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if (not isinstance(actual, (int, float)) or isinstance(actual, bool)
                or not math.isfinite(actual) or not math.isfinite(expected)):
            return f"{path}: non-finite or non-numeric observable"
        if abs(actual - expected) > atol + rtol * abs(expected):
            return f"{path}: {actual!r} differs from {expected!r}"
    elif actual != expected:
        return f"{path}: observable values differ"
    return None


def _summarize(trials):
    if any("error" in trial for trial in trials):
        return {"trials": trials}
    summary = {"trials": trials}
    for key in ("seconds", "peak_rss_bytes", "managed_peak_bytes"):
        samples = [trial[key] for trial in trials]
        median = statistics.median(samples)
        summary[key] = {"median": median, "mad": statistics.median(abs(v - median) for v in samples)}
    return summary


def _executable_identity(path):
    """Identify the measured payload without recording machine-local paths."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"name": path.name, "bytes": size, "sha256": digest.hexdigest()}


def _save_checkpoint(args, results, complete=False):
    """Preserve completed trials without labelling an interrupted run accepted."""
    if not args.save:
        return
    destination = Path(args.save)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False,
                                         dir=destination.parent, suffix=".tmp") as output:
            temporary = Path(output.name)
            json.dump({"cpu_threads": args.cpu_threads, "memory_mb": args.memory_mb,
                       "executables": args.executable_identity,
                       "complete": complete, "cases": results}, output, indent=2)
            output.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, default=Path(DEFAULT_EXE))
    ap.add_argument("--baseline-exe", type=Path)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--cpu-threads", type=int, default=1)
    ap.add_argument("--memory-mb", type=int, default=1024)
    ap.add_argument("--only", default="", help="substring filter on the label")
    ap.add_argument("--save", metavar="FILE", help="write timings as JSON")
    ap.add_argument("--compare", metavar="FILE", help="diff against a saved run")
    ap.add_argument("--observable-atol", type=float, default=1e-8)
    ap.add_argument("--observable-rtol", type=float, default=1e-8)
    args = ap.parse_args()

    if min(args.trials, args.cpu_threads, args.memory_mb) < 1:
        ap.error("--trials, --cpu-threads and --memory-mb must be positive")
    if any(not math.isfinite(value) or value < 0 for value in
           (args.observable_atol, args.observable_rtol)):
        ap.error("observable tolerances must be finite and non-negative")
    if args.baseline_exe and args.compare:
        ap.error("--baseline-exe and --compare are mutually exclusive")
    executables = {"candidate": args.exe.resolve()}
    if args.baseline_exe:
        executables["baseline"] = args.baseline_exe.resolve()
    for executable in executables.values():
        if not executable.is_file():
            ap.error(f"executable not found: {executable}; build it with cargo build --release")
    args.executable_identity = {name: _executable_identity(path)
                                for name, path in executables.items()}

    base = {}
    if args.compare:
        with open(args.compare) as fh:
            base = json.load(fh)

    results = {}
    failed = False
    with tempfile.TemporaryDirectory(prefix="dft-rs-bench-") as directory:
        scratch = Path(directory)
        for name, lines in GEOMETRIES.items():
            (scratch / f"{name}.xyz").write_text("\n".join(lines) + "\n", encoding="ascii")
        for name, contents in BASIS_INPUTS.items():
            (scratch / name).write_text(contents, encoding="ascii")
        for label, feature, molecule, extra in CASES:
            if args.only and args.only not in label:
                continue
            case_args = [str(scratch / value) if value in BASIS_INPUTS else value
                         for value in extra]
            flags = [feature, str(scratch / f"{molecule}.xyz"), *case_args,
                     "--format", "json", "--cpu-threads", str(args.cpu_threads),
                     "--memory-mb", str(args.memory_mb)]
            trials = {name: [] for name in executables}
            for _ in range(args.trials):
                for name in executables:
                    expected = args.executable_identity[name]
                    if _executable_identity(executables[name]) != expected:
                        trial = {"error": "executable changed before trial"}
                    else:
                        trial = _measure([str(executables[name]), *flags])
                        if _executable_identity(executables[name]) != expected:
                            trial["error"] = "executable changed during trial"
                    trials[name].append(trial)
                    results[label] = {key: {"trials": samples} for key, samples in trials.items()}
                    results[label]["status"] = "measuring"
                    _save_checkpoint(args, results)
            result = {name: _summarize(samples) for name, samples in trials.items()}
            results[label] = result
            errors = [trial["error"] for samples in trials.values() for trial in samples if "error" in trial]
            if not errors:
                reference = trials.get("baseline", trials["candidate"])[0].get("observables")
                if not reference:
                    errors.append("no numerical observables available for acceptance")
                else:
                    for name, samples in trials.items():
                        for index, trial in enumerate(samples):
                            mismatch = _observable_mismatch(
                                trial.get("observables"), reference,
                                args.observable_atol, args.observable_rtol)
                            if mismatch:
                                errors.append(f"{name} trial {index + 1}: {mismatch}")
                result["numerical_acceptance"] = {"passed": not errors, "errors": errors.copy(),
                                                  "atol": args.observable_atol,
                                                  "rtol": args.observable_rtol,
                                                  "reference": "baseline" if "baseline" in trials
                                                  else "candidate first trial"}
            result["status"] = "failed" if errors else "passed"
            _save_checkpoint(args, results)
            if errors:
                print(f"{label}: FAILED: {errors[0]}", file=sys.stderr, flush=True)
                failed = True
                continue
            current = result["candidate"]
            elapsed = current["seconds"]["median"]
            previous = result.get("baseline") or base.get("cases", {}).get(label, {}).get("candidate")
            before = previous.get("seconds", {}).get("median") if previous else base.get(label)
            row = (f"{label:<23} {elapsed:9.3f} s  "
                   f"{current['peak_rss_bytes']['median'] / 2**20:8.2f} MiB RSS  "
                   f"{current['managed_peak_bytes']['median'] / 2**20:8.2f} MiB managed")
            if before:
                row += f"  {before / elapsed:7.2f}x"
            print(row, flush=True)

    if not results:
        ap.error("--only matched no benchmark case")

    if args.save:
        _save_checkpoint(args, results, complete=True)
        print(f"\nsaved to {args.save}", flush=True)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
