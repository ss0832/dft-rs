# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""ASE calculator backed by dft-rs.

The native API uses atomic units. This adapter performs the ASE-required
eV/Angstrom conversions and caches ASE properties plus the explicitly lazy
Hessian and vibrational-analysis helpers.

Units
-----
Everything ASE itself defines is converted at this boundary and nowhere
else: ``energy`` and ``free_energy`` in eV, ``forces`` in eV/Angstrom,
``dipole`` in e*Angstrom, the Hessian in eV/Angstrom^2, and positions and
cell vectors converted to Bohr on the way in. All energy and length factors
come from ``dft_rs.units`` (CODATA 2018), matching the native core and XYZ
reader independently of ASE's default CODATA edition. ``charges`` are populations
and dimensionless. Frequencies stay in cm^-1 and IR intensities in km/mol,
which is what ``ase.vibrations`` reports too.

The report getters (``get_phonons``, ``get_elastic``, ``get_energy_report``,
...) take their controls in eV and Angstrom and return the reports of the
common driver in eV and Angstrom. Both conversions follow the unit labels of
the driver: every control and every numeric report entry has one, a label
begins with a unit expression (``hartree/bohr^2``, ``1/bohr``, ``e*bohr``),
and the Hartree and Bohr factors of that expression are converted, the
labels in ``report["units"]`` being rewritten to match. Calculator
parameters with a dimension (``conv_tol`` in eV, ``overlap_cutoff`` in
Angstrom, ...) follow the same rule. What stays as labelled on every
surface: a control or entry whose name carries its unit
(``scattering_hartree``, ``pressure_gpa``, ``frequencies_cm1``), entries in
spectroscopic or SI units, dimensionless numbers, and quantities whose
atomic unit is not a product of powers of energy and length (magnetic
fields, labelled ``atomic units ...``).

``units="atomic"`` as a calculator parameter makes the calculator
parameters and the reports atomic (Hartree, Bohr); as a keyword of one call
it applies to the controls and the report of that call. The ASE properties
above are in eV and Angstrom in either case.

The dedicated getters (``get_nmr``, ``get_dynamic_polarizability``,
``get_periodic_basis``, ...) follow the same rule through the labels in
``result["unit_labels"]``: their keyword arguments with a dimension are
given in eV and Angstrom and their results come back converted, unless
``units="atomic"``. A one-line ``units`` summary of such a result describes
the native atomic-unit result and is not rewritten. The blocks
``results["scf"]``, ``results["wavefunction"]`` and ``results["nbo"]`` of
``calculate`` are converted as well, and ``results["scf"]["units"]`` names
the unit system of that block.

Laziness
--------
An energy-only call does not compute a gradient. Requesting forces later
runs a gradient calculation because the native API does not retain SCF state
between calls; ASE then caches that result for the unchanged geometry.

``get_hessian`` and ``get_vibrations`` are not ASE properties and are never
triggered by ``get_potential_energy``/``get_forces``. Each computes on first
request and caches against the geometry, cell, PBC flags and every
calculator parameter, returning copies so a caller cannot corrupt the cache.
They cache separately, so asking for both runs two Hessians: the native
``vibrations`` entry point builds its own, and there is no binding yet that
accepts a precomputed one.
"""

from __future__ import annotations

import functools
import numbers
import operator
import re

import numpy as np

try:
    from ase.calculators.calculator import (
        Calculator,
        PropertyNotImplementedError,
        all_changes,
    )
    from .units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
except ImportError as exc:  # pragma: no cover
    raise ImportError("the dft-rs ASE calculator requires ase") from exc

from . import native


_ATOMIC = {"hartree": Hartree, "bohr": Bohr}
_WORD = re.compile(r"[A-Za-z0-9_]+")
_POWER = re.compile(r"-?\d+|[A-Za-z_][A-Za-z0-9_]*|\([A-Za-z_][A-Za-z0-9_]*[+-]\d+\)")
# A control or report entry whose name ends with its unit keeps that unit.
_NAMED_UNIT = ("_hartree", "_bohr")
_CONTROL_UNITS = None


def _expression(text):
    """Hartree and Bohr factors of the unit expression at the start of
    ``text`` and the length of that expression, or ``None`` when ``text``
    does not start with one.

    A factor is ``(unit, sign, power)``. A ``/`` divides by the one factor
    that follows it. A power that names an entry of the same object
    (``bohr^rank``, ``bohr^(rank-1)``, ``bohr^(order+1)``) is ``(entry, offset)``
    for the power ``entry + offset``."""
    position, factors, sign = 0, [], 1
    while True:
        word = _WORD.match(text, position)
        if word is None:
            return None
        position = word.end()
        power = 1
        if text.startswith("^", position):
            token = _POWER.match(text, position + 1)
            if token is None:
                return None
            position = token.end()
            spelled = token.group(0)
            if spelled.startswith("("):
                inner = spelled[1:-1]
                operator = "+" if "+" in inner else "-"
                entry, offset = inner.split(operator)
                power = (entry, int(offset) if operator == "+" else -int(offset))
            elif spelled.lstrip("-").isdigit():
                power = int(spelled)
            else:
                power = (spelled, 0)
        name = word.group(0).lower()
        if name in _ATOMIC:
            factors.append((name, sign, power))
        following = text[position:position + 1]
        if following == "*":
            sign = 1
        elif following == "/":
            sign = -1
        else:
            return factors, position
        position += 1


def _renamed(expression):
    renamed = re.sub("hartree", "eV", expression, flags=re.IGNORECASE)
    return re.sub("bohr", "angstrom", renamed, flags=re.IGNORECASE)


def _ase_unit(label):
    """Conversion of a unit label to eV and Angstrom.

    Returns ``(factors, renamed)``: the Hartree and Bohr factors of the
    leading unit expression (for a bracketed list, one list of factors per
    element of the last array axis) and the label with that expression
    rewritten. Returns ``None`` when the label does not begin with an
    expression in Hartree or Bohr."""
    text = label.strip()
    if text.startswith("["):
        close = text.find("]")
        if close < 0:
            return None
        columns, names = [], []
        for item in text[1:close].split(","):
            item = item.strip()
            parsed = _expression(item)
            if parsed is None or item[parsed[1]:parsed[1] + 1] not in ("", " ", ";", "("):
                return None
            columns.append(parsed[0])
            names.append(_renamed(item[:parsed[1]]) + item[parsed[1]:])
        if not any(columns):
            return None
        return columns, "[" + ", ".join(names) + "]" + text[close + 1:]
    parsed = _expression(text)
    if parsed is None or not parsed[0]:
        return None
    if text[parsed[1]:parsed[1] + 1] not in ("", " ", ",", ";", "("):
        return None
    return parsed[0], _renamed(text[:parsed[1]]) + text[parsed[1]:]


def _factor(factors, siblings):
    """Multiplier from atomic units to eV and Angstrom, or ``None`` when a
    power names an entry that is not an integer of the same object."""
    result = 1.0
    for name, sign, power in factors:
        if not isinstance(power, int):
            entry, offset = power
            value = siblings.get(entry) if isinstance(siblings, dict) else None
            if isinstance(value, bool) or not isinstance(value, int):
                return None
            power = value + offset
        result *= _ATOMIC[name] ** (sign * power)
    return result


def _scaled(value, factor, integers=False):
    """Multiply every real number of a nested value by ``factor``. Integers
    in a report are counts and indices and stay as they are; in a control
    (``integers=True``) an integer is a physical value like any other."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, numbers.Integral):
        return value * factor if integers else value
    if isinstance(value, (numbers.Real, complex)):
        return value * factor
    if isinstance(value, dict):
        return type(value)((key, _scaled(item, factor, integers)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return [_scaled(item, factor, integers) for item in value]
    return value


def _scaled_columns(value, columns, integers=False):
    """Scale the last axis of a nested list, one factor per element."""
    if not isinstance(value, (list, tuple)):
        return value
    if len(value) == len(columns) and not any(isinstance(item, (list, tuple, dict)) for item in value):
        return [_scaled(item, factor, integers) for item, factor in zip(value, columns)]
    return [_scaled_columns(item, columns, integers) for item in value]


def _converted(value, label, siblings, *, inverse=False, integers=False):
    """``(value, label)`` in eV and Angstrom (``inverse``: the value from eV
    and Angstrom to atomic units), or ``None`` when the label asks for no
    conversion."""
    parsed = _ase_unit(label)
    if parsed is None:
        return None
    factors, renamed = parsed
    if isinstance(factors[0], list):
        columns = [_factor(column, siblings) for column in factors]
        if any(column is None for column in columns):
            return None
        if inverse:
            columns = [1.0 / column for column in columns]
        return _scaled_columns(value, columns, integers), renamed
    factor = _factor(factors, siblings)
    if factor is None:
        return None
    return _scaled(value, 1.0 / factor if inverse else factor, integers), renamed


def _to_ase_units(report):
    """A driver report in eV and Angstrom.

    Labels are looked up by the dotted path of an entry (array indices
    omitted) and then by its name; the label of an object or an array covers
    everything below it. The labels of the converted entries are rewritten
    and the given report is left as it was."""
    # Driver reports keep their labels in `units`; the dedicated functions,
    # whose `units` is a one-line summary, in `unit_labels`.
    source = "units" if isinstance(report.get("units"), dict) else "unit_labels"
    units = report.get(source)
    if not isinstance(units, dict):
        return report
    labels = dict(units)
    # Labels whose entry kept its atomic value although the label names
    # Hartree or Bohr (a name that carries its unit, a power that could not
    # be read): they are not rewritten.
    kept = set()

    def convert(key, path, value, siblings):
        if isinstance(value, dict) and isinstance(value.get("unit_labels"), dict):
            # A nested result with labels of its own.
            return _to_ase_units(value)
        name = path if path in units else key
        label = units.get(name)
        if isinstance(label, str):
            if key.endswith(_NAMED_UNIT):
                kept.add(name)
                return value
            outcome = _converted(value, label, siblings)
            if outcome is None:
                if _ase_unit(label) is not None:
                    kept.add(name)
                return value
            labels[name] = outcome[1]
            return outcome[0]
        if isinstance(value, dict):
            return type(value)(
                (child, convert(child, f"{path}.{child}", item, value)) for child, item in value.items()
            )
        if isinstance(value, list):
            return [
                convert(key, path, item, siblings) if isinstance(item, (dict, list)) else item
                for item in value
            ]
        return value

    converted = type(report)(report)
    for key, value in report.items():
        if key not in ("units", "unit_labels", "unit_system"):
            converted[key] = convert(key, key, value, report)
    # A report can label entries it does not contain this time (an optional
    # output); those labels are rewritten too, so that none is left behind
    # in Hartree or Bohr.
    for name, label in units.items():
        if name in kept or not isinstance(label, str) or name.rsplit(".", 1)[-1].endswith(_NAMED_UNIT):
            continue
        parsed = _ase_unit(label)
        if parsed is not None:
            labels[name] = parsed[1]
    converted[source] = labels
    converted["unit_system"] = (
        "ASE units (eV, Angstrom) wherever a unit label names them; every other entry as labelled"
    )
    return converted


def _control_units(feature=None):
    """Base controls plus the registered getter's contextual unit contract."""
    global _CONTROL_UNITS
    if _CONTROL_UNITS is None:
        registry = native.feature_registry()
        _CONTROL_UNITS = dict(registry["control_units"])
    if feature is None:
        return _CONTROL_UNITS
    # The same control name can denote a dimensional path or dimensionless
    # contraction weights. Reuse its existing typed getter contract, rather
    # than giving the generic route a second independent unit catalogue.
    getter = next((entry["ase"] for entry in native.feature_registry().get("features", [])
                   if entry["id"] == feature), None)
    return {**_CONTROL_UNITS, **_DEDICATED.get(getter, {})}


def _controls_to_atomic(controls, feature=None):
    """Controls in eV/Angstrom converted by their registry paths, once.

    Dotted labels also describe fields of structured objects and lists of
    objects. A text label on an enclosing object does not hide its typed
    children (for example ``particles.position`` and ``particles.exponents``).
    """
    units = _control_units(feature)

    def convert(path, value, siblings, inherited=None):
        if hasattr(value, "tolist"):
            value = value.tolist()
        label = units.get(path, inherited)
        if isinstance(value, dict):
            return {key: convert(f"{path}.{key}", item, value, label)
                    for key, item in value.items()}
        if isinstance(value, (list, tuple)) and any(isinstance(v, (dict, list, tuple)) for v in value):
            return [convert(path, item, siblings, label) for item in value]
        outcome = None
        if value is not None and isinstance(label, str) and not path.endswith(_NAMED_UNIT):
            outcome = _converted(value, label, siblings, inverse=True, integers=True)
        return value if outcome is None else outcome[0]

    return {name: convert(name, value, controls) for name, value in controls.items()}


class DftRs(Calculator):
    """ASE interface to dft-rs.

    Parameters mirror :func:`dft_rs.native.calculate`. In particular,
    ``ecp`` is independent of ``basis``, ``density_fit`` selects an RI-J
    auxiliary basis, and ``dispersion='auto'`` enables built-in VV10/CHG
    terms for functionals that define them.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from dft_rs import DftRs
    >>> atoms = molecule("H2O")
    >>> atoms.calc = DftRs(basis="def2-svp", xc="b3lyp")
    >>> energy = atoms.get_potential_energy()  # eV
    >>> forces = atoms.get_forces()            # eV/Angstrom
    >>> hessian = atoms.calc.get_hessian(atoms)  # eV/Angstrom^2
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress", "dipole", "charges"]
    default_parameters = {
        "basis": "sto-3g",
        "ecp": None,
        "xc": "hf",
        "dispersion": "auto",
        "density_fit": None,
        "charge": 0,
        "unrestricted": False,
        "stabilize": False,
        "multiplicity": 1,
        "grid_level": 3,
        "grid_preset": None,
        "grid_angular_scheme": None,
        "grid_weight_cutoff": None,
        "vv10_grid_level": 3,
        "wavefunction": False,
        "nbo": False,
        "max_iter": 150,
        "conv_tol": None,
        "conv_tol_grad": None,
        "conv_tol_dp": 1e-8,
        "incore_mb": 2048,
        "cpu_threads": None,
        "memory_mb": None,
        "log_file": None,
        "log_mode": None,
        "log_level": None,
        "k_mesh": (1, 1, 1),
        "kpts": None,
        "units": "ase",
        "overlap_cutoff": None,
        "exchange_cutoff": None,
        "grid_shell": 2,
        "electronic_temperature": 0.0,
        "level_shift": None,
        "damping": 0.25,
        "diis_size": 8,
        "use_adiis": True,
        "adiis_until_error": None,
        "guess": None,
    }

    def __init__(self, **kwargs):
        """Initialize the ASE calculator and empty derivative-result caches.

        Keyword arguments are ASE calculator parameters and documented DFT
        reference controls. Dimensional inputs use eV/Angstrom by default;
        units="atomic" selects Hartree/Bohr for controls and report getters.

        guess=None preserves the reference driver default (SAD for molecular
        common-driver calculations, core for periodic calculations). An
        explicit guess is forwarded to the selected common reference.
        """
        super().__init__(**kwargs)
        self._hessian = None
        self._hessian_key = None
        self._vibrations = None
        self._vibrations_key = None

    def set(self, **kwargs):
        """Set parameters without discarding invalid integers equal to defaults."""
        if "use_adiis" in kwargs and not isinstance(kwargs["use_adiis"], (bool, np.bool_)):
            raise ValueError("use_adiis must be Boolean")
        for name in ("charge", "multiplicity", "grid_level", "vv10_grid_level",
                     "grid_shell", "max_iter", "incore_mb", "diis_size"):
            if name not in kwargs:
                continue
            value = kwargs[name]
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be an integer without rounding")
            try:
                kwargs[name] = operator.index(value)
            except TypeError:
                raise ValueError(f"{name} must be an integer without rounding") from None
        return super().set(**kwargs)

    def _geometry(self, atoms):
        numbers = [int(z) for z in atoms.get_atomic_numbers()]
        positions = (atoms.get_positions() / Bohr).tolist()
        return numbers, positions

    def _key(self, atoms):
        return (
            tuple(int(z) for z in atoms.get_atomic_numbers()),
            atoms.get_positions().tobytes(),
            atoms.cell.array.tobytes(),
            atoms.get_pbc().tobytes(),
            tuple(sorted((key, repr(value)) for key, value in self.parameters.items())),
        )

    @staticmethod
    def _is_periodic(atoms):
        pbc = np.asarray(atoms.get_pbc(), dtype=bool)
        return bool(pbc.any())

    def _k_mesh(self):
        """Monkhorst-Pack mesh: ``kpts`` (the ASE spelling) overrides ``k_mesh``."""
        kpts = self.parameters.get("kpts")
        mesh = self.parameters.k_mesh if kpts is None else kpts
        try:
            mesh = list(mesh)
            if any(isinstance(value, (bool, np.bool_)) for value in mesh):
                raise TypeError
            mesh = [operator.index(value) for value in mesh]
        except TypeError:
            raise ValueError("kpts/k_mesh takes three positive integers without rounding") from None
        if len(mesh) != 3 or min(mesh) < 1:
            raise ValueError("kpts/k_mesh takes three positive integers (a Monkhorst-Pack mesh)")
        return mesh

    # Calculator parameters with a dimension. `None` selects the default of
    # the native layer, given here in atomic units; a value is in the units
    # of the calculator (eV and Angstrom unless `units="atomic"`).
    _DIMENSIONAL_DEFAULTS = {
        "conv_tol": 1e-9,
        "conv_tol_grad": None,
        "overlap_cutoff": 20.0,
        "exchange_cutoff": None,
        "level_shift": 0.0,
        "adiis_until_error": 0.1,
        "grid_weight_cutoff": None,
    }

    def _calculator_units(self):
        mode = str(self.parameters.get("units") or "ase").lower()
        if mode not in ("ase", "atomic"):
            raise ValueError("units is 'ase' (eV, Angstrom) or 'atomic' (Hartree, Bohr)")
        return mode

    def _unit_mode(self, controls):
        """``"ase"`` or ``"atomic"``: the ``units`` keyword of one call or the
        calculator parameter of the same name."""
        mode = controls.pop("units", None)
        if mode is None:
            return self._calculator_units()
        mode = str(mode).lower()
        if mode not in ("ase", "atomic"):
            raise ValueError("units is 'ase' (eV, Angstrom) or 'atomic' (Hartree, Bohr)")
        return mode

    def _atomic(self, name):
        """A dimensional calculator parameter in atomic units."""
        value = self.parameters.get(name)
        if value is None:
            return self._DIMENSIONAL_DEFAULTS[name]
        if self._calculator_units() == "atomic":
            return float(value)
        return float(_controls_to_atomic({name: value})[name])

    def _parameter(self, name):
        """A calculator parameter as the native layer takes it."""
        if name in self._DIMENSIONAL_DEFAULTS:
            return self._atomic(name)
        return self.parameters[name]

    def _integer_parameter(self, name):
        value = self.parameters[name]
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be an integer without rounding")
        try:
            return operator.index(value)
        except TypeError:
            raise ValueError(f"{name} must be an integer without rounding") from None

    def _extra_parameters(self):
        return {key: value for key, value in self.parameters.items()
                if key not in self.default_parameters and value is not None}

    def _extra_controls(self, feature=None):
        """Calculator parameters outside the built-in list are driver controls,
        for example ``DftRs(..., three_body=True, disp_cutoff=30.0)``, in the
        units of the calculator and returned in atomic units."""
        extra = self._extra_parameters()
        return extra if self._calculator_units() == "atomic" else _controls_to_atomic(extra, feature)

    def _merge_report_controls(self, merged, controls, mode, feature):
        weighted_call = "xc_terms" in controls
        weighted_calculator = (
            "xc_terms" in merged and merged.get("xc") == self.default_parameters["xc"]
        )
        if "xc" not in controls and (weighted_call or weighted_calculator):
            merged.pop("xc", None)
        merged.update(controls if mode == "atomic" else _controls_to_atomic(controls, feature))

    @staticmethod
    def _path_vertices(path):
        """Reduced vertices of a k or q path; an ASE ``BandPath`` with one
        continuous segment is converted to its special points."""
        if hasattr(path, "special_points") and hasattr(path, "path"):
            from ase.dft.kpoints import parse_path_string

            segments = parse_path_string(path.path)
            if len(segments) != 1:
                raise ValueError(
                    "a band path with jumps has several segments: pass one continuous segment at a time"
                )
            return [[float(x) for x in path.special_points[name]] for name in segments[0]]
        return [[float(x) for x in vertex] for vertex in path]

    def _native_options(self, *, allow_reference_controls=False, grid_controls=False):
        p = self.parameters
        extended = (*self._changed_scf_parameters(), *self._extra_parameters())
        if extended:
            raise PropertyNotImplementedError(
                "this legacy getter cannot forward " + ", ".join(extended)
                + "; use a supported common report route or an explicitly configured Rust reference")
        if not allow_reference_controls and (p.stabilize or p.unrestricted):
            raise PropertyNotImplementedError("this property getter builds its own reference and does not yet accept stabilized/forced-unrestricted states; use the CLI --stab-follow for response properties")
        grid = {
            "grid_preset": p.grid_preset,
            "grid_angular_scheme": p.grid_angular_scheme,
            "grid_weight_cutoff": self._parameter("grid_weight_cutoff"),
        }
        if not grid_controls and any(value is not None for value in grid.values()):
            raise ValueError(
                "grid_preset, grid_angular_scheme and grid_weight_cutoff reach this getter "
                "through the common driver; use get_feature_report or the named get_*_report "
                "methods")
        return ({**grid} if grid_controls else {}) | {
            "basis": p.basis,
            "ecp": p.ecp,
            "xc": p.xc,
            "dispersion": p.dispersion,
            "density_fit": p.density_fit,
            "charge": self._integer_parameter("charge"),
            "multiplicity": self._integer_parameter("multiplicity"),
            "grid_level": p.grid_level,
            "vv10_grid_level": p.vv10_grid_level,
            "max_iter": self._integer_parameter("max_iter"),
            "conv_tol": self._atomic("conv_tol"),
            "conv_tol_grad": self._atomic("conv_tol_grad"),
            "conv_tol_dp": float(p.conv_tol_dp),
            "incore_mb": self._integer_parameter("incore_mb"),
            "nbo": bool(p.nbo),
        }

    def _uses_common_molecular_reference(self):
        return bool(self._changed_scf_parameters() or self._extra_parameters())

    def _run(self, atoms, *, gradient):
        if self._uses_common_molecular_reference():
            if self.parameters.wavefunction or self.parameters.nbo:
                raise PropertyNotImplementedError(
                    "selected references use the common energy/gradient and orbitals report getters; combined wavefunction/NBO export is not connected"
                )
            result = self.get_feature_report("gradient" if gradient else "energy", atoms, units="atomic")
            if not result["converged"]:
                raise RuntimeError("the selected reference did not converge; adjust its state selection or convergence controls")
            return result
        numbers, positions = self._geometry(atoms)
        result = native.calculate(
            numbers,
            positions,
            gradient=gradient,
            wavefunction=bool(self.parameters.wavefunction),
            unrestricted=bool(self.parameters.unrestricted),
            stabilize=bool(self.parameters.stabilize),
            **self._native_options(allow_reference_controls=True, grid_controls=True),
        )
        if not result["converged"]:
            raise RuntimeError(
                "dft-rs SCF did not converge in "
                f"{result.get('iterations', result.get('scf_iterations', 'unknown'))} iterations; inspect response diagnostics, raise max_iter or change the guess"
            )
        return result

    def _run_periodic(self, atoms, *, forces, response=None):
        p = self.parameters
        if p.stabilize or p.unrestricted:
            raise PropertyNotImplementedError("periodic stabilized/forced-unrestricted reference integration is not yet available")
        if p.density_fit is not None:
            raise ValueError("density_fit is not yet available in periodic dft-rs SCF")
        if str(p.dispersion).lower() not in {"auto", "none"}:
            raise ValueError(
                "empirical/nonlocal dispersion is not yet available in periodic dft-rs SCF"
            )
        if bool(p.nbo):
            raise ValueError("NAO/NPA analysis is not yet defined for Bloch orbitals")
        numbers, positions = self._geometry(atoms)
        result = native.periodic_scf(
            numbers,
            positions,
            (atoms.cell.array / Bohr).tolist(),
            basis=p.basis,
            ecp=p.ecp,
            xc=p.xc,
            charge=self._integer_parameter("charge"),
            multiplicity=self._integer_parameter("multiplicity"),
            pbc=[bool(value) for value in atoms.get_pbc()],
            k_mesh=self._k_mesh(),
            overlap_cutoff=self._atomic("overlap_cutoff"),
            exchange_cutoff=self._atomic("exchange_cutoff"),
            grid_level=self._integer_parameter("grid_level"),
            grid_shell=self._integer_parameter("grid_shell"),
            max_iter=self._integer_parameter("max_iter"),
            conv_tol=self._atomic("conv_tol"),
            conv_tol_grad=self._atomic("conv_tol_grad"),
            conv_tol_dp=float(p.conv_tol_dp),
            forces=forces,
            wavefunction=bool(p.wavefunction),
            electronic_temperature=float(p.electronic_temperature),
            level_shift=self._atomic("level_shift"),
            damping=float(p.damping),
            diis_size=self._integer_parameter("diis_size"),
            use_adiis=bool(p.use_adiis),
            adiis_until_error=self._atomic("adiis_until_error"),
            guess=str(p.guess or "core"),
            response=response,
            grid_preset=p.grid_preset,
            grid_angular_scheme=p.grid_angular_scheme,
            grid_weight_cutoff=self._parameter("grid_weight_cutoff"),
        )
        if not result["converged"]:
            raise RuntimeError(
                "dft-rs periodic SCF did not converge in "
                f"{result['iterations']} iterations; raise max_iter or adjust convergence settings"
            )
        return result

    def get_periodic_basis(self, atoms=None, *, basis=None, **options):
        """Return dimensional Gaussian S/T and optional quadrature in atomic units.

        Reference geometry is converted from Angstrom. Explicit basis/grid
        cutoffs remain Bohr; this getter preserves ordinary SCF results.
        """
        atoms = atoms if atoms is not None else self.atoms
        if atoms is None:
            raise ValueError("periodic basis requires Atoms")
        return native.periodic_basis(atoms.get_atomic_numbers().tolist(),
                                     (np.asarray(atoms.get_positions())/Bohr).tolist(),
                                     (np.asarray(atoms.cell)/Bohr).tolist(),
                                     pbc=[bool(v) for v in atoms.get_pbc()],
                                     basis=basis or self.parameters["basis"], **options)

    def get_dimensional_scf(self, atoms=None, *, basis=None, xc=None, charge=None,
                            multiplicity=None, **options):
        """Return zero-field dimensional SCF; convert only Atoms geometry to Bohr."""
        atoms = atoms if atoms is not None else self.atoms
        if atoms is None:
            raise ValueError("dimensional SCF requires Atoms")
        if self.parameters["ecp"] is not None:
            raise PropertyNotImplementedError(
                "dimensional SCF requires its short-range/nonlocal ECP operators")
        if self.parameters["density_fit"] is not None:
            raise PropertyNotImplementedError(
                "dimensional SCF uses the exact contracted Hartree operator; "
                "dimensional density fitting remains pending")
        if self.parameters["stabilize"]:
            raise PropertyNotImplementedError(
                "dimensional orbital-stability following requires its orbital Hessian")
        if self.parameters["grid_shell"] != self.default_parameters["grid_shell"]:
            raise ValueError(
                "dimensional SCF uses grid_cutoff; the legacy grid_shell must retain its default")
        options.setdefault("guess", self.parameters.get("guess") or "core")
        for name in ("k_mesh", "grid_level", "grid_preset", "grid_angular_scheme",
                     "grid_weight_cutoff", "max_iter", "conv_tol", "conv_tol_grad",
                     "conv_tol_dp", "guess", "exchange_cutoff", "use_adiis",
                     "adiis_until_error", "damping", "level_shift", "diis_size"):
            options.setdefault(name, self._parameter(name))
        options.setdefault("basis_cutoff", self._atomic("overlap_cutoff"))
        options.setdefault("temperature", self.parameters["electronic_temperature"])
        options.setdefault("unrestricted", self.parameters["unrestricted"])
        return native.dimensional_scf(atoms.get_atomic_numbers().tolist(),
            (np.asarray(atoms.get_positions())/Bohr).tolist(),
            (np.asarray(atoms.cell)/Bohr).tolist(), pbc=[bool(v) for v in atoms.get_pbc()],
            basis=basis or self.parameters["basis"], xc=xc or self.parameters["xc"],
            charge=self.parameters["charge"] if charge is None else charge,
            multiplicity=self.parameters["multiplicity"] if multiplicity is None else multiplicity,
            **options)

    def get_dimensional_field_response(self, atoms=None, *, basis=None, xc=None, charge=None,
                            multiplicity=None, **options):
        """Return screened dimensional orbital field response; convert only Atoms geometry to Bohr."""
        atoms = atoms if atoms is not None else self.atoms
        if atoms is None:
            raise ValueError("dimensional SCF requires Atoms")
        if self.parameters["ecp"] is not None:
            raise PropertyNotImplementedError(
                "dimensional SCF requires its short-range/nonlocal ECP operators")
        if self.parameters["density_fit"] is not None:
            raise PropertyNotImplementedError(
                "dimensional SCF uses the exact contracted Hartree operator; "
                "dimensional density fitting remains pending")
        if self.parameters["stabilize"]:
            raise PropertyNotImplementedError(
                "dimensional orbital-stability following requires its orbital Hessian")
        if self.parameters["grid_shell"] != self.default_parameters["grid_shell"]:
            raise ValueError(
                "dimensional SCF uses grid_cutoff; the legacy grid_shell must retain its default")
        options.setdefault("guess", self.parameters.get("guess") or "core")
        for name in ("k_mesh", "grid_level", "grid_preset", "grid_angular_scheme",
                     "grid_weight_cutoff", "max_iter", "conv_tol", "conv_tol_grad",
                     "conv_tol_dp", "guess", "exchange_cutoff", "use_adiis",
                     "adiis_until_error", "damping", "level_shift", "diis_size"):
            options.setdefault(name, self._parameter(name))
        options.setdefault("basis_cutoff", self._atomic("overlap_cutoff"))
        options.setdefault("temperature", self.parameters["electronic_temperature"])
        options.setdefault("unrestricted", self.parameters["unrestricted"])
        return native.dimensional_field_response(atoms.get_atomic_numbers().tolist(),
            (np.asarray(atoms.get_positions())/Bohr).tolist(),
            (np.asarray(atoms.cell)/Bohr).tolist(), pbc=[bool(v) for v in atoms.get_pbc()],
            basis=basis or self.parameters["basis"], xc=xc or self.parameters["xc"],
            charge=self.parameters["charge"] if charge is None else charge,
            multiplicity=self.parameters["multiplicity"] if multiplicity is None else multiplicity,
            **options)

    def get_periodic_field_operators(self, atoms=None, *, basis=None, charge=None, **options):
        """Bare scalar/vector Fourier vertices; convert Atoms geometry to Bohr."""
        atoms = atoms if atoms is not None else self.atoms
        if atoms is None:
            raise ValueError("periodic fields require Atoms")
        return native.periodic_field_operators(
            atoms.get_atomic_numbers().tolist(),(np.asarray(atoms.get_positions())/Bohr).tolist(),
            (np.asarray(atoms.cell)/Bohr).tolist(),pbc=[bool(v) for v in atoms.get_pbc()],
            basis=basis or self.parameters["basis"],
            charge=self.parameters["charge"] if charge is None else charge,**options)

    def get_periodic_xc_operators(self, atoms=None, *, basis=None, xc=None, charge=None, **options):
        """Return semilocal components and spin/Bloch XC potentials in Hartree.

        Atoms geometry is converted to Bohr; explicit cutoffs and supplied
        AO density matrices retain native units and ordering.
        """
        atoms = atoms if atoms is not None else self.atoms
        if atoms is None:
            raise ValueError("periodic XC operators require Atoms")
        return native.periodic_xc_operators(
            atoms.get_atomic_numbers().tolist(), (np.asarray(atoms.get_positions())/Bohr).tolist(),
            (np.asarray(atoms.cell)/Bohr).tolist(), pbc=[bool(v) for v in atoms.get_pbc()],
            basis=basis or self.parameters["basis"], xc=xc or self.parameters["xc"],
            charge=self.parameters["charge"] if charge is None else charge, **options)

    def get_periodic_coulomb_operators(self, atoms=None, *, basis=None, charge=None, **options):
        """Return contracted Coulomb components and fixed-density derivatives.

        Atoms geometry is converted to Bohr. Explicit widths/cutoffs remain
        atomic-unit inputs; the density uses the native spherical AO ordering.
        """
        atoms = atoms if atoms is not None else self.atoms
        if atoms is None:
            raise ValueError("periodic Coulomb operators require Atoms")
        return native.periodic_coulomb_operators(
            atoms.get_atomic_numbers().tolist(), (np.asarray(atoms.get_positions())/Bohr).tolist(),
            (np.asarray(atoms.cell)/Bohr).tolist(), pbc=[bool(v) for v in atoms.get_pbc()],
            basis=basis or self.parameters["basis"],
            charge=self.parameters["charge"] if charge is None else charge, **options,
        )

    def get_periodic_gaussian_integral(self, atoms=None, *, exponents, angular, **options):
        """Primitive Coulomb integral, converting Atoms geometry to Bohr.

        Exponents use Bohr^-2; explicit source widths/cutoffs use Bohr.
        Three positions select nuclear attraction with the source last;
        four select an ERI. Atomic numbers do not specify primitive charges.
        """
        atoms = atoms if atoms is not None else self.atoms
        if atoms is None:
            raise ValueError("periodic Gaussian integral requires Atoms")
        return native.periodic_gaussian_integral(
            (np.asarray(atoms.get_positions())/Bohr).tolist(), exponents, angular,
            (np.asarray(atoms.cell)/Bohr).tolist(),
            pbc=[bool(v) for v in atoms.get_pbc()], **options,
        )

    def get_periodic_electrostatics(self, atoms=None, *, charges, widths=None,
                                    compensation="neutral", countercharge_center=None,
                                    countercharge_width=None, **options):
        """Prescribed-charge Coulomb report for Atoms.pbc, in atomic units.

        Atom positions/cell are converted from Angstrom. Widths and the
        countercharge center/width are explicit Bohr quantities. Charges
        are supplied by the caller; no electronic density is inferred.
        This getter does not alter ordinary self-consistent ASE results.
        """
        atoms = atoms if atoms is not None else self.atoms
        if atoms is None:
            raise ValueError("periodic electrostatics requires Atoms")
        return native.periodic_electrostatics(
            (np.asarray(atoms.get_positions())/Bohr).tolist(), charges,
            (np.asarray(atoms.cell)/Bohr).tolist(),
            pbc=[bool(v) for v in atoms.get_pbc()], widths=widths,
            compensation=compensation, countercharge_center=countercharge_center,
            countercharge_width=countercharge_width, **options,
        )

    def get_periodic_field_response(
        self, atoms=None, *, q=(0,0,0), scalar_potential=0j,
        vector_potential=(0j,0j,0j), omega=0.0, damping=0.0, imaginary=False,
        fixed_number=True, reciprocal_cutoff=6.0, response_tol=1e-10,
        response_max_iter=500, response_restart=40,
    ):
        """Screened per-cell number/charge/current report in reduced atomic units."""
        atoms=atoms if atoms is not None else self.atoms
        if atoms is None or not self._is_periodic(atoms):
            raise ValueError("periodic field response requires periodic atoms")
        vector=[complex(v) for v in vector_potential]
        if len(vector)!=3:
            raise ValueError("vector_potential requires three Cartesian components")
        scalar=complex(scalar_potential)
        return self._run_periodic(atoms, forces=False, response=dict(
            response_q=list(q),response_scalar_potential=[scalar.real,scalar.imag],
            response_vector_potential=[[v.real,v.imag] for v in vector],
            response_omega=omega,response_damping=damping,response_imaginary=imaginary,
            response_fixed_number=fixed_number,response_reciprocal_cutoff=reciprocal_cutoff,
            response_tol=response_tol,response_max_iter=response_max_iter,response_restart=response_restart,
        ))["response"]

    def get_periodic_density_response(
        self, atoms=None, *, q=(0, 0, 0), mode=0, omega=0.0, damping=0.0,
        imaginary=False, fixed_number=True, reciprocal_cutoff=6.0,
        response_tol=1e-10, response_max_iter=500, response_restart=40,
    ):
        """Return the native finite-q density report in atomic units.

        Geometry and cell are converted from ASE Angstrom to Bohr. Cartesian
        q and reciprocal_cutoff are already inverse Bohr; frequency is Hartree.
        """
        atoms = atoms if atoms is not None else self.atoms
        if atoms is None or not self._is_periodic(atoms):
            raise ValueError("periodic density response requires periodic atoms")
        return self._run_periodic(atoms, forces=False, response=dict(
            response_q=list(q), response_density_mode=mode,
            response_omega=omega, response_damping=damping,
            response_imaginary=imaginary, response_fixed_number=fixed_number,
            response_reciprocal_cutoff=reciprocal_cutoff, response_tol=response_tol,
            response_max_iter=response_max_iter, response_restart=response_restart,
        ))["response"]

    def get_periodic_thermal_response(
        self, atoms=None, *, delta_temperature=1.0, delta_numbers=(0.0, 0.0),
        delta_mu=(0.0, 0.0), fixed_number=True, reciprocal_cutoff=6.0,
        response_tol=1e-10, response_max_iter=500, response_restart=40,
    ):
        """Return static thermodynamic path variations in native units.

        Geometry is converted from Angstrom; temperature directions use Kelvin,
        chemical potentials Hartree, and number directions particles per cell.
        The calculator needs positive electronic_temperature in Kelvin.
        """
        atoms = atoms if atoms is not None else self.atoms
        if atoms is None or not self._is_periodic(atoms):
            raise ValueError("periodic thermal response requires periodic atoms")
        return self._run_periodic(atoms, forces=False, response=dict(
            response_q=[0.0, 0.0, 0.0], response_thermal=True,
            response_delta_temperature=delta_temperature,
            response_delta_numbers=list(delta_numbers), response_delta_mu=list(delta_mu),
            response_fixed_number=fixed_number, response_reciprocal_cutoff=reciprocal_cutoff,
            response_tol=response_tol, response_max_iter=response_max_iter,
            response_restart=response_restart,
        ))["response"]

    def get_periodic_dielectric(
        self, atoms=None, *, q=(0, 0, 0), modes=(0,), omega=0.0, damping=0.0,
        imaginary=False, fixed_number=True, reciprocal_cutoff=6.0,
        response_tol=1e-10, response_max_iter=500, response_restart=40,
    ):
        """Return epsilon^-1=I+v*screened_chi in native atomic units."""
        atoms = atoms if atoms is not None else self.atoms
        if atoms is None or not self._is_periodic(atoms):
            raise ValueError("periodic dielectric response requires periodic atoms")
        return self._run_periodic(atoms, forces=False, response=dict(
            response_q=list(q), response_modes=list(modes),
            response_omega=omega, response_damping=damping,
            response_imaginary=imaginary, response_fixed_number=fixed_number,
            response_reciprocal_cutoff=reciprocal_cutoff, response_tol=response_tol,
            response_max_iter=response_max_iter, response_restart=response_restart,
        ))["response"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        """Populate the ASE result dictionary for the requested properties."""
        super().calculate(atoms, properties, system_changes)
        want_forces = "forces" in properties
        want_stress = "stress" in properties
        periodic = self._is_periodic(self.atoms)
        if want_stress and not (periodic and all(self.atoms.get_pbc())):
            raise PropertyNotImplementedError(
                "dft-rs defines the stress only for fully periodic Atoms"
            )
        if periodic:
            unsupported = set(properties) & {"dipole", "charges"}
            if unsupported:
                names = ", ".join(sorted(unsupported))
                raise PropertyNotImplementedError(
                    f"periodic dft-rs does not define ASE properties: {names}"
                )
            if not self.parameters.wavefunction:
                # One calculation of the common driver supplies the energy,
                # the forces and the analytic stress, with the dispersion
                # model of the calculator if there is one.
                result = self._periodic_report(
                    "periodic-energy", self.atoms, _atomic=True, forces=want_forces, stress=want_stress
                )
                if not result["converged"]:
                    raise RuntimeError(
                        "dft-rs periodic SCF did not converge; raise max_iter or adjust convergence settings"
                    )
            elif want_stress:
                result = self._periodic_report("stress", self.atoms, _atomic=True)
            else:
                # Wavefunction dumps use the dedicated route.
                result = self._run_periodic(self.atoms, forces=want_forces)
            if want_stress:
                voigt = np.asarray(result["stress_voigt"], dtype=float)
                self.results["stress"] = voigt * Hartree / Bohr**3
        else:
            result = self._run(self.atoms, gradient=want_forces)

        self.results["energy"] = result["energy"] * Hartree
        if "stability" in result:
            self.results["stability"] = result["stability"]
        self.results["free_energy"] = result.get("free_energy", result["energy"]) * Hartree
        if not periodic:
            if "dipole" in properties and "dipole" not in result:
                # Orbital-dependent energies obtain their analytic correlated
                # electric derivative from the common property driver.
                moment = self.get_feature_report("dipole", self.atoms, units="atomic")
                result["dipole"] = moment["dipole"]
            if "charges" in properties and "mulliken" not in result:
                # The common property route checks whether the selected model
                # provides the required density, including correlated models.
                result["mulliken"] = self.get_feature_report("charges", self.atoms, units="atomic")["mulliken"]
            if "dipole" in result:
                self.results["dipole"] = np.asarray(result["dipole"], dtype=float) * Bohr
            if "mulliken" in result:
                self.results["charges"] = np.asarray(result["mulliken"], dtype=float)
        if want_forces:
            if periodic:
                self.results["forces"] = (
                    np.asarray(result["forces"], dtype=float) * Hartree / Bohr
                )
            else:
                gradient = np.asarray(result["gradient"], dtype=float)
                self.results["forces"] = -gradient * Hartree / Bohr
        # The diagnostic blocks follow the unit system of the calculator.
        atomic = self._calculator_units() == "atomic"
        blocks = result if atomic else _to_ase_units(result)
        if self.parameters.wavefunction:
            self.results["wavefunction"] = blocks["wavefunction"]
        if self.parameters.nbo:
            self.results["nbo"] = blocks["nbo"]
        # Diagnostic components of the energy, in the unit system named by
        # the "units" entry of the block.
        self.results["scf"] = {
            key: blocks[key]
            for key in (
                "e_nuc",
                "e_elec",
                "e_xc",
                "e_nlc",
                "e_dispersion",
                "iterations",
                "scf_iterations",
                "energy_components",
                "s_squared",
                "n_alpha",
                "n_beta",
                "unrestricted",
                "occupation_tracking",
                "occupations",
                "fixed_occupation_residual",
                "spin_square_model",
                "n_core_electrons",
            )
            if key in result
        }
        self.results["scf"]["units"] = "atomic (Hartree)" if atomic else "ASE (eV)"
        if periodic and "force_method" in result:
            self.results["force_method"] = result["force_method"]
        if periodic and "force_energy" in result:
            self.results["force_energy"] = result["force_energy"]

    def get_hessian(self, atoms=None):
        """Return the Cartesian Hessian in eV/Angstrom^2, computed lazily.

        For fully periodic ``Atoms`` these are the Gamma-point force constants
        of the cell (analytic coupled-perturbed response). Molecular calls
        preserve the selected RKS/UKS/ROKS/GKS reference and double-hybrid
        target through the common analytic driver; unsupported combinations
        raise the driver's named error without a numerical fallback."""
        atoms = atoms if atoms is not None else self.atoms
        key = self._key(atoms)
        if self._hessian is not None and self._hessian_key == key:
            return self._hessian.copy()
        if self._is_periodic(atoms):
            # Gamma-point force constants: every image of an atom moves.
            result = self._periodic_report("phonons", atoms, _atomic=True)
            native._require_analytic_hessian(result)
            hessian = np.asarray(result["force_constants"], dtype=float) * Hartree / (Bohr * Bohr)
        else:
            # The common driver preserves explicit reference constraints,
            # complex sources and correlated targets in this lazy getter too.
            result = self.get_feature_report("hessian", atoms, units="atomic")
            native._require_analytic_hessian(result)
            hessian = np.asarray(result["hessian"], dtype=float) * Hartree / (Bohr * Bohr)
        self._hessian, self._hessian_key = hessian, key
        return hessian.copy()

    @staticmethod
    def _copy_vibrations(result):
        return {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in result.items()
        }

    def get_dynamic_polarizability(self, atoms=None, *, omega=0.0, damping=0.0,
                                   imaginary=False, response_tol=1e-10,
                                   response_max_iter=500, solvent=None, eps=None, **controls):
        """Molecular alpha in bohr^3; omega/damping in Hartree.

        Return the native report unchanged so convergence diagnostics and
        both parts of the complex tensor retain the same conventions.
        """
        atoms = atoms if atoms is not None else self.atoms
        if self._is_periodic(atoms):
            raise PropertyNotImplementedError("periodic dipole response requires the DFPT field driver")
        numbers, positions = self._geometry(atoms)
        options = self._molecular_driver_controls("dynamic-polarizability")
        if "damping" in options:
            options["scf_damping"] = options.pop("damping")
        options.update(controls)
        configured_solvent = options.pop("solvent", None)
        configured_eps = options.pop("eps", None)
        return native.dynamic_polarizability(
            numbers, positions, omega=omega, damping=damping,
            imaginary=imaginary, response_tol=response_tol,
            response_max_iter=response_max_iter,
            solvent=configured_solvent if solvent is None else solvent,
            eps=configured_eps if eps is None else eps,
            **options,
        )

    def _static_electric_report(self, feature, atoms, options, controls, reference_field):
        """Share the reference builder and cross the ASE unit boundary once."""
        for canonical, legacy, default in [
            ("response_tolerance", "response_tol", 1e-10),
            ("response_max_iterations", "response_max_iter", 500),
        ]:
            if canonical in controls and options.get(canonical, default) != default:
                raise ValueError(f"use either {legacy} or {canonical}")
            if canonical not in controls and options.get(canonical, default) == default:
                configured = self.parameters.get(canonical)
                if configured is not None:
                    options[canonical] = configured
        merged = {key: value for key, value in options.items() if value is not None}
        merged.update(controls)
        if reference_field is not None:
            if "field" in merged:
                raise ValueError("use either reference_field or field")
            merged["field"] = reference_field
        mode = self._unit_mode(merged.copy())
        report = self.get_feature_report(feature, atoms, **merged)
        return native._electric_compat_report(report,
            None if mode == "atomic" else "ASE units; see unit_labels")

    def get_electric_field_taylor(self, atoms=None, *, direction=(0.0, 0.0, 1.0), order=3,
                                response_tol=1e-10, response_max_iter=500,
                                solvent=None, eps=None, reference_field=None, **controls):
        """Analytic dipole Taylor coefficients on the shared reference and solver.

        Fixed populations use isospectral rotations. Default ASE units give
        field directions in eV/(e Angstrom) and dipoles in e Angstrom;
        units="atomic" gives the native report. Coefficients include 1/n!.
        """
        options = dict(field_direction=direction, response_order=order,
                       response_tolerance=response_tol, response_max_iterations=response_max_iter,
                       solvent=solvent, eps=eps)
        return self._static_electric_report("field-taylor", atoms, options, controls, reference_field)

    def get_mixed_electric_field_response(self, atoms=None, *, directions,
                                          response_tol=1e-10, response_max_iter=500,
                                          solvent=None, eps=None, reference_field=None, **controls):
        """Static mixed dipole derivative from the common reference/response driver.

        Directions carry the field units of the selected interface; the path
        parameter is dimensionless. No factorial scaling is applied.
        """
        options = dict(field_directions=directions, response_tolerance=response_tol,
                       response_max_iterations=response_max_iter, solvent=solvent, eps=eps)
        return self._static_electric_report("field-mixed", atoms, options, controls, reference_field)

    def get_static_electric_tensor(self, atoms=None, *, order=2, assembly="projector",
                                  response_tol=1e-10, response_max_iter=500,
                                  solvent=None, eps=None, reference_field=None, **controls):
        """Complete independently ordered static beta/gamma on the shared driver.

        assembly="variational" selects the three-solve 2n+1 beta contraction.
        Fixed populations retain every differently populated orbital pair.
        Per-entry unit_labels are authoritative on both unit modes.
        Parent ROHF/ROKS and full complex GHF/GKS use the common stationary
        energy contraction. Integer GKS also supports a common PT2 weight
        and reports parent, target and generalized pair tensors.
        """
        if order not in (2, 3):
            raise ValueError("static tensor order is 2 (beta) or 3 (gamma)")
        options = dict(tensor_assembly=assembly, response_tolerance=response_tol,
                       response_max_iterations=response_max_iter, solvent=solvent, eps=eps)
        return self._static_electric_report("beta" if order == 2 else "gamma", atoms,
                                           options, controls, reference_field)

    def get_beta(self, atoms=None, **options):
        """Complete static first hyperpolarizability, without factorial scaling."""
        return self.get_static_electric_tensor(atoms, order=2, **options)

    def get_gamma(self, atoms=None, **options):
        """Complete static second hyperpolarizability, without factorial scaling."""
        return self.get_static_electric_tensor(atoms, order=3, **options)

    def get_stability(self, atoms=None, *, solvent=None, eps=None, **controls):
        """Collinear real/imaginary orbital stability; atomic-unit report."""
        atoms = atoms if atoms is not None else self.atoms
        if self._is_periodic(atoms):
            raise PropertyNotImplementedError("periodic stability requires its Bloch orbital Hessian")
        numbers, positions = self._geometry(atoms)
        options = self._native_options(allow_reference_controls=True)
        options.pop("nbo")
        controls.setdefault("unrestricted", bool(self.parameters.unrestricted))
        controls.setdefault("follow_unstable", bool(self.parameters.stabilize))
        return native.stability(numbers, positions, solvent=solvent,
            eps=eps, **controls, **options)

    def get_stab_following(self, atoms=None, **controls):
        """Follow internal orbital instabilities and return the final atomic-unit report."""
        return self.get_stability(atoms, follow_unstable=True, **controls)

    def get_dynamic_electric_tensor(self, atoms=None, *, input_frequencies=None,
                                    process=None, frequencies=None, damping=0.0,
                                    tensor_order=None,
                                    imaginary=False, solvent=None, eps=None,
                                    response_tol=1e-10, response_max_iter=500):
        """Complex alpha/beta/gamma; frequencies/damping in eV, or atomic units.

        Select units="atomic" for Hartree inputs and atomic tensor outputs.
        """
        numbers, positions, options = self._molecular_response_geometry(atoms)
        return native.dynamic_electric_tensor(
            numbers, positions, input_frequencies=input_frequencies,
            process=process, frequencies=frequencies, damping=damping, tensor_order=tensor_order,
            imaginary=imaginary, solvent=solvent, eps=eps,
            response_tol=response_tol, response_max_iter=response_max_iter,
            **options,
        )

    def get_dynamic_electric_process(self, atoms=None, *, process, frequencies, damping=0.0, **options):
        """Nonlinear optical preset; frequencies/damping in eV, or units="atomic"."""
        return self.get_dynamic_electric_tensor(atoms, process=process,
            frequencies=frequencies, damping=damping, **options)

    def get_dynamic_beta(self, atoms=None, *, input_frequencies=None, process=None,
                         frequencies=None, damping=0.0, **options):
        """Full beta from explicit legs or shg/sfg/dfg/rectification/electro-optic.

        Frequencies/damping use eV, or Hartree with units="atomic".
        """
        if input_frequencies is not None and len(input_frequencies) != 2:
            raise ValueError("get_dynamic_beta requires two input frequency pairs")
        return self.get_dynamic_electric_tensor(atoms, input_frequencies=input_frequencies,
            process=process, frequencies=frequencies, damping=damping, tensor_order=2, **options)

    def get_dynamic_gamma(self, atoms=None, *, input_frequencies=None, process=None,
                          frequencies=None, damping=0.0, **options):
        """Full gamma from explicit legs or a cubic optical preset.

        Presets: thg, efish, field-induced-rectification,
        two-photon-absorption, optical-kerr, dc-kerr and four-wave-mixing.
        Frequencies/damping use eV, or Hartree with units="atomic". The
        result is the ordered molecular cubic dipole susceptibility.
        """
        if input_frequencies is not None and len(input_frequencies) != 3:
            raise ValueError("get_dynamic_gamma requires three input frequency pairs")
        return self.get_dynamic_electric_tensor(atoms, input_frequencies=input_frequencies,
            process=process, frequencies=frequencies, damping=damping, tensor_order=3, **options)

    def get_orbital_energy_gradients(self, atoms=None, *, coordinates=None,
                                     degeneracy_tolerance=1e-8, unrestricted=False,
                                     reference_field=None, response_tol=1e-10,
                                     response_max_iter=500):
        """Relaxed orbital gradients/projected degenerate blocks in atomic units."""
        numbers, positions, options = self._molecular_response_geometry(atoms)
        return native.orbital_energy_gradients(numbers,positions,coordinates=coordinates,
            degeneracy_tolerance=degeneracy_tolerance,unrestricted=unrestricted,
            reference_field=reference_field,response_tol=response_tol,
            response_max_iter=response_max_iter,**options)

    def get_frontier_gap_gradient(self, atoms=None, *, coordinates=None,
                                  degeneracy_tolerance=1e-8, unrestricted=False,
                                  reference_field=None, response_tol=1e-10,
                                  response_max_iter=500):
        """Per-spin neutral-reference orbital gaps and Hartree/Bohr gradients."""
        numbers, positions, options = self._molecular_response_geometry(atoms)
        return native.frontier_gap_gradient(numbers,positions,coordinates=coordinates,
            degeneracy_tolerance=degeneracy_tolerance,unrestricted=unrestricted,
            reference_field=reference_field,response_tol=response_tol,
            response_max_iter=response_max_iter,**options)

    def _magnetic_geometry(self, atoms):
        atoms = atoms if atoms is not None else self.atoms
        if atoms is None:
            raise ValueError("magnetic response requires Atoms")
        if self._is_periodic(atoms):
            raise PropertyNotImplementedError("periodic fields require the dimensional covariant field driver")
        if self.parameters.stabilize:
            raise PropertyNotImplementedError("this magnetic getter does not yet accept a stabilized reference")
        numbers, positions = self._geometry(atoms)
        options = self._native_options(allow_reference_controls=True)
        options.pop("nbo")
        options.pop("dispersion")
        options["unrestricted"] = bool(self.parameters.unrestricted)
        return numbers, positions, options

    def get_magnetic_response(self, atoms=None, *, gauge="giao", origin=None,
                              response_tol=1e-10, response_max_iter=500, response_restart=40):
        """Spin orbital dP/dB in reduced atomic units; origin in Bohr."""
        numbers, positions, options = self._magnetic_geometry(atoms)
        return native.magnetic_response(numbers, positions, gauge=gauge, origin=origin,
            response_tol=response_tol, response_max_iter=response_max_iter,
            response_restart=response_restart, **options)

    def get_nmr(self, atoms=None, *, gauge="giao", origin=None,
                response_tol=1e-10, response_max_iter=500, response_restart=40):
        """Orbital shieldings in ppm with raw residuals; origin in Bohr."""
        numbers, positions, options = self._magnetic_geometry(atoms)
        return native.nmr(numbers, positions, gauge=gauge, origin=origin,
            response_tol=response_tol, response_max_iter=response_max_iter,
            response_restart=response_restart, **options)

    def get_nuclear_taylor(self, atoms=None, *, direction, order=4,
                           response_tol=1e-10, response_max_iter=500):
        """Analytic nuclear energy coefficients and diagnostics in atomic units.

        ASE positions are converted from Angstrom. Direction displacements
        are explicitly in Bohr, as in the native response report.
        """
        atoms = atoms if atoms is not None else self.atoms
        if self._is_periodic(atoms):
            raise PropertyNotImplementedError("periodic nuclear Taylor response requires the DFPT geometry driver")
        numbers, positions = self._geometry(atoms)
        options = self._native_options()
        options.pop("nbo")
        return native.nuclear_taylor(
            numbers, positions, direction, order=order, response_tol=response_tol,
            response_max_iter=response_max_iter, **options,
        )

    def get_mixed_nuclear_response(self, atoms=None, *, directions,
                                  response_tol=1e-10, response_max_iter=500):
        """Mixed nuclear energy derivative; directions in Bohr, result in Hartree."""
        atoms = atoms if atoms is not None else self.atoms
        if self._is_periodic(atoms):
            raise PropertyNotImplementedError("periodic mixed nuclear response requires the DFPT geometry driver")
        numbers, positions = self._geometry(atoms)
        options = self._native_options()
        options.pop("nbo")
        return native.mixed_nuclear_response(
            numbers, positions, directions, response_tol=response_tol,
            response_max_iter=response_max_iter, **options,
        )

    def _molecular_response_geometry(self, atoms, *, common=False):
        atoms = atoms if atoms is not None else self.atoms
        if self._is_periodic(atoms):
            raise PropertyNotImplementedError("periodic mixed geometry/field response requires the DFPT driver")
        numbers, positions = self._geometry(atoms)
        if common:
            options = self._molecular_driver_controls()
        else:
            options = self._native_options()
            options.pop("nbo")
        return numbers, positions, options

    def get_molecular_taylor(self, atoms=None, *, nuclear_direction,
                             electric_direction, order=4, reference_field=None,
                             response_tol=1e-10, response_max_iter=500):
        """Analytic molecular jets; nuclear paths in Bohr and fields in atomic units."""
        numbers, positions, options = self._molecular_response_geometry(atoms)
        return native.molecular_taylor(
            numbers, positions, nuclear_direction, electric_direction, order=order,
            reference_field=reference_field, response_tol=response_tol,
            response_max_iter=response_max_iter, **options,
        )

    def get_mixed_molecular_response(self, atoms=None, *, nuclear_directions,
                                     electric_directions, reference_field=None,
                                     response_tol=1e-10, response_max_iter=500):
        """Actual mixed energy/dipole derivatives with physical residuals."""
        numbers, positions, options = self._molecular_response_geometry(atoms)
        return native.mixed_molecular_response(
            numbers, positions, nuclear_directions, electric_directions,
            reference_field=reference_field, response_tol=response_tol,
            response_max_iter=response_max_iter, **options,
        )

    def get_static_spectroscopic_derivative(self, atoms=None, *, nuclear_direction,
                                            electric_order=2, reference_field=None,
                                            response_tol=1e-10, response_max_iter=500):
        """Directional APT/Raman/hyper-Raman on the common molecular reference.

        Integer RKS/UKS correlated targets include full orbital and PT2
        amplitude response through energy order three/four. Nuclear paths
        use Angstrom by default, Bohr in atomic mode. All ordered entries
        and raw permutation/response diagnostics are retained. Parent
        ROHF/ROKS and full complex GHF/GKS share the stationary energy
        connection; their constrained PT2 and dynamic routes remain separate.
        """
        numbers, positions, options = self._molecular_response_geometry(atoms, common=True)
        if reference_field is not None:
            options["reference_field"] = reference_field
        options.update(response_tol=response_tol, response_max_iter=response_max_iter)
        return native.static_spectroscopic_derivative(
            numbers, positions, nuclear_direction, electric_order=electric_order, **options,
        )

    def get_analytic_force_constants(self, atoms=None, *, order=3, patterns=None,
                                     coordinates=None, reference_field=None,
                                     response_tol=1e-10, response_max_iter=500, **controls):
        """Complete analytic cubic/quartic Cartesian, block or pattern tensor.

        Integer RKS/UKS double hybrids include full orbital/PT2 amplitude
        response and separate target/SS/OS/dispersion tensors. Controls and
        results follow the calculator unit mode and common report labels.
        """
        for key, value, default in (("response_tolerance", response_tol, 1e-10),
                                    ("response_max_iterations", response_max_iter, 500)):
            if key in controls and value != default:
                raise ValueError(f"use either {key} or its compatibility alias")
            controls.setdefault(key, value)
        report = self.get_feature_report("force-constants", atoms, response_order=order,
            patterns=patterns, coordinates=coordinates, reference_field=reference_field, **controls)
        return native._electric_compat_report(report, report["response_units"])

    def get_anharmonic(self, atoms=None, *, mode_ids=None, include_raman=False,
                       equilibrium_tol=1e-4, resonance_tol=1e-8,
                       resonance_coupling_factor=1.0, max_basis_states=1024,
                       max_polyad_states=256, response_tol=1e-10, response_max_iter=500,
                       method="retained-polyad"):
        """Analytic rectilinear anharmonic bands, IR km/mol and Raman Angstrom^4/amu."""
        numbers, positions, options = self._molecular_response_geometry(atoms)
        return native.anharmonic(numbers, positions, mode_ids=mode_ids, include_raman=include_raman,
            equilibrium_tol=equilibrium_tol, resonance_tol=resonance_tol,
            resonance_coupling_factor=resonance_coupling_factor, max_basis_states=max_basis_states,
            max_polyad_states=max_polyad_states, response_tol=response_tol,
            response_max_iter=response_max_iter, method=method, **options)

    def get_vibrations(self, atoms=None):
        """Return frequencies (cm^-1), IR intensities, and projection metadata."""
        atoms = atoms if atoms is not None else self.atoms
        if self._is_periodic(atoms):
            raise PropertyNotImplementedError(
                "molecular vibrational analysis is not defined for periodic Atoms; use get_phonons, get_phonon_bands or get_phonon_thermo"
            )
        key = self._key(atoms)
        if self._vibrations is not None and self._vibrations_key == key:
            return self._copy_vibrations(self._vibrations)
        numbers, positions = self._geometry(atoms)
        native_result = native.vibrations(numbers, positions, **self._native_options())
        result = {
            "frequencies": np.asarray(native_result["frequencies"], dtype=float),
            "ir_intensities": np.asarray(native_result["ir_intensities"], dtype=float),
            "n_imaginary": native_result["n_imaginary"],
            "n_projected": native_result["n_projected"],
            "method": native_result["method"],
        }
        self._vibrations, self._vibrations_key = result, key
        return self._copy_vibrations(result)

    # ------------------------------------------------------------------
    # Common calculation driver
    #
    # These getters call the same Rust driver as `dft_rs.run_feature` and
    # `dft-rs <feature>`. Geometry and cell are converted from Angstrom to Bohr; the returned
    # report is converted to eV and Angstrom wherever its "units" entry names
    # Hartree or Bohr (see the module docstring; `units="atomic"`
    # returns it unchanged).
    # Calculator parameters supply the reference controls and keyword
    # arguments override them.
    # ------------------------------------------------------------------

    _DRIVER_REFERENCE_PARAMETERS = (
        "basis", "ecp", "xc", "dispersion", "density_fit", "charge",
        "multiplicity", "unrestricted", "stabilize", "grid_level",
        "grid_preset", "grid_angular_scheme", "grid_weight_cutoff",
        "vv10_grid_level", "max_iter", "conv_tol", "conv_tol_grad",
        "conv_tol_dp", "incore_mb", "guess", "level_shift", "damping",
        "diis_size", "adiis_until_error",
    )
    _EXTENDED_SCF_PARAMETERS = (
        "level_shift", "damping", "diis_size", "use_adiis", "adiis_until_error",
        "guess", "electronic_temperature",
    )

    def _changed_scf_parameters(self):
        return tuple(key for key in self._EXTENDED_SCF_PARAMETERS
                     if self.parameters.get(key) != self.default_parameters[key])

    def _molecular_driver_controls(self, feature=None, *, dirac_species=False):
        """Atomic reference controls shared by common and adapted getters."""
        merged = {}
        for key in self._DRIVER_REFERENCE_PARAMETERS:
            if key in self._EXTENDED_SCF_PARAMETERS and self.parameters.get(key) == self.default_parameters[key]:
                continue
            if key == "adiis_until_error" and self.parameters.get(key) is None:
                continue
            value = self._parameter(key)
            if value is not None:
                merged[key] = value
        adiis = self.parameters.get("use_adiis", True)
        if not isinstance(adiis, (bool, np.bool_)):
            raise ValueError("use_adiis must be Boolean")
        if not adiis:
            merged["no_adiis"] = True
        temperature = float(self.parameters.get("electronic_temperature") or 0.0)
        if temperature != 0.0:
            merged["electronic_temperature"] = temperature
        if feature == "dirac-coulomb-scf" or dirac_species:
            applicable = {"basis", "charge", "max_iter", "conv_tol", "conv_tol_grad",
                          "conv_tol_dp", "damping", "diis_size"}
            merged = {key: value for key, value in merged.items()
                      if key in applicable or key not in self.default_parameters
                      or self.parameters.get(key) != self.default_parameters[key]}
        merged.update(self._extra_controls(feature))
        return merged

    def get_feature_report(self, feature, atoms=None, **controls):
        """Run a registered driver calculation for ``atoms``."""
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError("an Atoms object is required")
        if self._is_periodic(atoms):
            return self._periodic_report(feature, atoms, **controls)
        mode = self._unit_mode(controls)
        numbers, positions = self._geometry(atoms)
        if feature == "mcd" and controls.get(
            "mcd_model", self.parameters.get("mcd_model")
        ) in ("thermal-manifold", "thermal-complex-manifold"):
            # Supplied multiplet operators do not consume an SCF reference.
            # Per-call inapplicable controls still reach strict driver checks.
            # The common getter wrapper already applies calculator resources.
            merged = self._extra_controls(feature)
        else:
            dirac_species = feature == "multicomponent" and controls.get(
                "species_electronic_model", self.parameters.get("species_electronic_model"),
            ) == "dirac-coulomb"
            merged = self._molecular_driver_controls(feature, dirac_species=dirac_species)
        self._merge_report_controls(merged, controls, mode, feature)
        report = native.run_feature(feature, numbers, positions, **merged)
        return report if mode == "atomic" else _to_ase_units(report)

    def get_energy_report(self, atoms=None, **controls):
        """Total energy and components in Hartree."""
        return self.get_feature_report("energy", atoms, **controls)

    def get_free_energy_report(self, atoms=None, **controls):
        """Compatibility getter for electronic Mermin free energy; see get_mermin_free_energy."""
        return self.get_feature_report("free-energy", atoms, **controls)

    def get_mermin_free_energy(self, atoms=None, **controls):
        """Electronic A=E-T_e*S_e; electronic_temperature in K, result eV by default.

        units="atomic" returns Hartree. Nuclear RRHO and pV are excluded.
        ROKS and ROHF fix both spin counts. GKS and GHF fix the total count.
        Set gradient=True for the analytic nuclear derivative with a
        supported provider.
        RKS and UKS also accept gibbs=True for equilibrium RRHO
        thermochemistry on the same reference. Temperature is K and pressure is Pa.
        """
        return self.get_feature_report("mermin-free-energy", atoms, **controls)

    def get_gibbs_free_energy(self, atoms=None, **controls):
        """Ideal-gas RRHO G=H-T*S from the analytic Hessian, eV per molecule.

        temperature is Kelvin, pressure is Pa; units="atomic" returns
        Hartree. Uses the same harmonic model as get_frequencies_report.
        Constrained ROHF/ROKS/GHF/GKS reuse their analytic Hessian provider.
        electronic_degeneracy defaults to one for fixed spinors. Free-molecule
        Eckart modes, positive internal frequencies and convergence are required;
        provider restrictions and bounded acceptance remain in response.md.
        """
        return self.get_feature_report("gibbs-free-energy", atoms, **controls)

    def get_gradient_report(self, atoms=None, **controls):
        """Analytic nuclear gradient in Hartree/Bohr."""
        return self.get_feature_report("gradient", atoms, **controls)

    def get_charges_report(self, atoms=None, **controls):
        """Mulliken/Loewdin charges; reference selects auto/rks/uks/roks/gks and HF aliases."""
        return self.get_feature_report("charges", atoms, **controls)

    def get_natural_bond(self, atoms=None, **controls):
        """Natural bond orbital analysis: natural charges and configurations,
        Wiberg indices, the natural Lewis structure, NLMOs and second-order
        donor-acceptor energies (eV in ASE mode, Hartree in atomic mode).
        Optional natural-orbital controls include ``nbo_natural_orbitals``,
        ``nbo_spin_density`` and ``nbo_corresponding_orbitals``.
        ``nbo_spin_density_points`` uses Angstrom in ASE mode and Bohr in
        atomic mode."""
        return self.get_feature_report("natural-bond", atoms, **controls)

    def get_igmh(self, atoms=None, **controls):
        """Sample molecular IGMH fields on points or an affine grid.

        Coordinates and step vectors use Angstrom in ASE mode and Bohr in
        atomic mode. The report converts density and gradient units accordingly.
        """
        return self.get_feature_report("igmh", atoms, **controls)

    def get_periodic_natural_bond(self, atoms=None, **controls):
        """Natural bond orbital analysis of the periodic cell on the
        Born-von Karman supercell of the k mesh (energies in eV in ASE mode,
        Hartree in atomic mode)."""
        return self._periodic_report("periodic-natural-bond", atoms, **controls)

    def get_natural_shielding(self, atoms=None, **controls):
        """Natural chemical shielding analysis (ppm) over natural localized
        molecular orbitals."""
        return self.get_feature_report("natural-shielding", atoms, **controls)

    def get_natural_coupling(self, atoms=None, **controls):
        """Natural J-coupling analysis of a closed-shell molecule over
        natural localized molecular orbitals."""
        return self.get_feature_report("natural-coupling", atoms, **controls)

    def get_natural_energy_decomposition(self, atoms=None, **controls):
        """Natural energy decomposition of fragment interactions
        (``fragments`` per atom; eV in ASE mode, Hartree in atomic mode)."""
        return self.get_feature_report("natural-energy-decomposition", atoms, **controls)

    def get_sapt_analysis(self, atoms=None, **controls):
        """Intermolecular perturbation analysis with `fragments` per atom.

        `sapt_variant` accepts `"response"`, `"sapt0"`, `"sapt0-s2"`,
        `"ssapt0"`, `"sapt-30"`, `"sapt2+3"` and `"sapt-ks-fdds"`. Energies use eV by default or Hartree
        with `units="atomic"`. For optional `sapt_regularization_eta` with
        `"sapt0"` or `"sapt-ks-fdds"`, use inverse Angstrom squared or inverse Bohr squared
        in atomic mode. The optional `sapt_correlation_terms` and
        `sapt_relaxed_induction30` reports follow the same energy units."""
        return self.get_feature_report("sapt-analysis", atoms, **controls)

    def get_dipole_report(self, atoms=None, **controls):
        """Dipole in e*Bohr and Debye about the reported origin."""
        return self.get_feature_report("dipole", atoms, **controls)

    def get_orbitals_report(self, atoms=None, **controls):
        """Orbital energies and occupations; reference selects auto/rks/uks/roks/gks and HF aliases."""
        return self.get_feature_report("orbitals", atoms, **controls)

    def write_molden(self, atoms=None, *, path, **controls):
        """Write a Molden file of the converged reference."""
        return self.get_feature_report("molden", atoms, path=str(path), **controls)

    def write_orbitals(self, atoms=None, *, path, **controls):
        """Write the JSON orbital report."""
        return self.get_feature_report("orbitals-save", atoms, path=str(path), **controls)

    def get_molecular_multipole(self, atoms=None, **controls):
        """Cartesian, traceless and spherical multipoles (e*Bohr^rank)."""
        return self.get_feature_report("molecular-multipole", atoms, **controls)

    def get_molecular_medium_optics(self, atoms=None, **controls):
        """Molecular dielectric and polarization-resolved slab optics."""
        return self.get_feature_report("molecular-medium-optics", atoms, **controls)

    def get_quadrupole_polarizability(self, atoms=None, **controls):
        """Electric quadrupole response in five symmetric traceless gradient modes."""
        return self.get_feature_report("quadrupole-polarizability", atoms, **controls)

    def get_multipole_polarizability(self, atoms=None, **controls):
        """Static or frequency-resolved Racah multipole response for rank 1 to 5."""
        return self.get_feature_report("multipole-polarizability", atoms, **controls)

    def get_conceptual_dft(self, atoms=None, **controls):
        """Conceptual DFT descriptors (Hartree-based)."""
        return self.get_feature_report("conceptual-dft", atoms, **controls)

    def get_spin_conceptual_dft(self, atoms=None, **controls):
        """Spin-channel conceptual DFT descriptors."""
        return self.get_feature_report("spin-conceptual-dft", atoms, **controls)

    def get_hessian_report(self, atoms=None, **controls):
        """Analytic Hessian report in Hartree/Bohr^2 (atomic units).

        Positive electronic_temperature selects the Mermin Hessian for
        supported ROKS and GKS providers. Read the thermal response and
        number diagnostics from the report.
        Optional ``bfield`` and ``field`` select the complex London nuclear
        response with fixed laboratory fields. Residuals are returned before
        any intrinsic tensor permutation reuse; correlated PT2 and extra
        Hamiltonian providers retain their documented restrictions.
        """
        return self.get_feature_report("hessian", atoms, **controls)

    def get_frequencies_report(self, atoms=None, **controls):
        """Frequencies, optional IR, normal modes and thermochemistry.

        Constrained references reuse their analytic Hessian provider and raw
        residuals. vibrational_projection=auto retains all Cartesian modes
        under a fixed laboratory source; eckart/cartesian override projection.
        IR is null without an atomic polar tensor. electronic_degeneracy
        defaults to one for fixed spinors. See response.md for the precise
        correlated-reference and bounded acceptance scope.
        """
        return self.get_feature_report("frequencies", atoms, **controls)

    def get_vibrational_spectra(self, atoms=None, **controls):
        """Analytic IR intensities and Raman activities."""
        return self.get_feature_report("ir-raman", atoms, **controls)

    def get_polarizability_report(self, atoms=None, **controls):
        """Static polarizability report (Bohr^3 and Angstrom^3)."""
        return self.get_feature_report("polarizability", atoms, **controls)

    def get_multipole_gradient(self, atoms=None, **controls):
        """Analytic multipole nuclear derivatives."""
        return self.get_feature_report("multipole-gradient", atoms, **controls)

    def get_response_derivatives(self, atoms=None, **controls):
        """U^F, dP/dF and dP/dR in atomic units (AO basis)."""
        if controls.get("path") is not None:
            controls["path"] = str(controls["path"])
        return self.get_feature_report("response-save", atoms, **controls)

    def get_third_derivative_2n1(self, atoms=None, **controls):
        """Cartesian third derivatives of the energy by the 2n+1 rule."""
        return self.get_feature_report("third-derivative-2n1", atoms, **controls)

    def get_cubic_fc_along(self, atoms=None, **controls):
        """Cubic force constants contracted with a displacement pattern.

        Optional ``bfield``/``field`` use complex London nuclear response.
        """
        return self.get_feature_report("cubic-fc-along", atoms, **controls)

    def get_quartic_fc_along(self, atoms=None, **controls):
        """Quartic force constants contracted with two displacement patterns.

        Optional ``bfield``/``field`` use complex London nuclear response.
        """
        return self.get_feature_report("quartic-fc-along", atoms, **controls)

    def get_streamed_gradient(self, atoms=None, **controls):
        """Analytic gradient without a stored integral tensor."""
        return self.get_feature_report("streamed-gradient", atoms, **controls)

    def get_excitations(self, atoms=None, **controls):
        """Collinear TDA/RPA, common-orbital reference-spin TDA or
        full-spin GKS TDA poles. Spinor states have no exact total-spin label.
        Request TDA transition orbitals with ``td_nto=True``.
        ``td_difference_density=True`` gives unrelaxed attachment and
        detachment orbitals for RKS, UKS and GKS."""
        return self.get_feature_report("excitations", atoms, **controls)

    def get_ecd_spectrum(self, atoms=None, **controls):
        """Broadened absorption and circular-dichroism spectra."""
        return self.get_feature_report("ecd-spectrum", atoms, **controls)

    def get_td_state_gradient(self, atoms=None, **controls):
        """Analytic total excited-state gradient (Hartree/Bohr).
        HF common-orbital doublet/quartet and full-spin HF subsets are accepted.
        Moving semilocal spinor verification and correlated pole derivatives remain pending."""
        return self.get_feature_report("td-state-gradient", atoms, **controls)

    def get_hyperfine(self, atoms=None, **controls):
        """Static contact, spin-dipolar and total hyperfine tensors.

        MHz/per-nuclear-g and optional isotope-scaled MHz/gauss labels are
        preserved. Native reference/model restrictions apply, including the
        explicit single-Kramers-pair approximation for GKS/GHF. The default
        report is static; nuclear_order=1/2 selects the separate bounded
        HF-collinear nuclear-A adapter. This is not vibrational averaging.
        """
        return self.get_feature_report("hyperfine", atoms, **controls)

    def get_nmr_shielding(self, atoms=None, **controls):
        """Nuclear magnetic shielding tensors (ppm)."""
        return self.get_feature_report("nmr-shielding", atoms, **controls)

    def get_nmr_spectrum(self, atoms=None, **controls):
        """Isotropic NMR spectrum (ppm); nmr_spin_model="exact" retains scalar spin mixing.
        Uses the shared native controls, transition areas and spin residuals.
        """
        return self.get_feature_report("nmr-spectrum", atoms, **controls)

    def get_spin_spin_couplings(self, atoms=None, **controls):
        """Ramsey tensors, including UKS/GKS full-spin mixed terms.
        Selects free response by default; fixed-spin-direction clamps orientation.
        ROKS and open-shell PT2 remain separate providers.
        """
        return self.get_feature_report("spin-spin-couplings", atoms, **controls)

    def get_fermi_contact_couplings(self, atoms=None, **controls):
        """Fermi-contact part of the indirect nuclear spin-spin couplings."""
        return self.get_feature_report("fermi-contact-couplings", atoms, **controls)

    def get_exafs(self, atoms=None, **controls):
        """Single-scattering EXAFS from tabulated scattering functions."""
        return self.get_feature_report("exafs", atoms, **controls)

    def get_xanes(self, atoms=None, **controls):
        """Core-level absorption from core-valence-separated TDA."""
        return self.get_feature_report("xanes", atoms, **controls)

    def get_nexafs(self, atoms=None, **controls):
        """Polarized molecular near-edge absorption."""
        return self.get_feature_report("nexafs", atoms, **controls)

    def get_constrained_dft(self, atoms=None, **controls):
        """AO population or spin constraints, with optional forces, relaxation
        and static electric response."""
        return self.get_feature_report("constrained-dft", atoms, **controls)

    def get_dft_plus_u(self, atoms=None, **controls):
        """Molecular DFT+U. Nested Hubbard site parameters are always in Hartree."""
        return self.get_feature_report("dft-plus-u", atoms, **controls)

    def get_pz_sic(self, atoms=None, **controls):
        """Perdew-Zunger SIC with occupied orbital optimization."""
        return self.get_feature_report("pz-sic", atoms, **controls)

    def get_flo_sic_fixed(self, atoms=None, **controls):
        """Return frozen UKS Fermi-Lowdin SIC with fixed descriptors in Bohr."""
        return self.get_feature_report("flo-sic-fixed", atoms, **controls)

    def get_car_parrinello_md(self, atoms=None, **controls):
        """Microcanonical molecular extended-Lagrangian trajectory."""
        return self.get_feature_report("car-parrinello-md", atoms, **controls)

    def get_magnetic_quadrupole(self, atoms=None, **controls):
        """Orbital and Pauli magnetic quadrupole."""
        return self.get_feature_report("magnetic-quadrupole", atoms, **controls)

    def get_magnetizability(self, atoms=None, **controls):
        """Magnetizability tensor with a common gauge origin."""
        return self.get_feature_report("magnetizability", atoms, **controls)

    def get_franck_condon(self, atoms=None, **controls):
        """Franck-Condon progression in the vertical-gradient model."""
        return self.get_feature_report("franck-condon", atoms, **controls)

    def get_excited_optimization(self, atoms=None, **controls):
        """Relax a supported excited surface with its analytic gradient.
        ``td_gradient_route="path"`` enables compatible paths; general spinor
        optimization and robust crossing/subspace tracking remain pending."""
        return self.get_feature_report("excited-optimization", atoms, **controls)

    def get_optical_activity(self, atoms=None, **controls):
        """Optical rotation tensors and specific rotation."""
        return self.get_feature_report("optical-activity", atoms, **controls)

    def get_broken_symmetry_exchange(self, atoms=None, **controls):
        """Heisenberg exchange coupling from a broken-symmetry determinant."""
        return self.get_feature_report("broken-symmetry-exchange", atoms, **controls)

    def get_vcd(self, atoms=None, **controls):
        """Vibrational circular dichroism rotational strengths."""
        return self.get_feature_report("vcd", atoms, **controls)

    def get_spin_susceptibility(self, atoms=None, **controls):
        """Static spin susceptibility of a Fermi-Dirac ensemble."""
        return self.get_feature_report("spin-susceptibility", atoms, **controls)

    def get_spin_spin_zfs(self, atoms=None, *, nuclear_order=0, **controls):
        """Direct spin-spin D and optional analytic HF/UHF nuclear derivatives.

        Requires multiplicity >= 3; SOC and spin purification are excluded.
        Native units are Hartree/Bohr; ASE mode converts D, its gradient and
        Hessian to eV, eV/Angstrom and eV/Angstrom^2. The atom-major xyz
        ordering and raw diagnostic labels are preserved. Unrequested
        derivatives are None. See native.spin_spin_zfs for model restrictions.
        """
        return self.get_feature_report("spin-spin-zfs", atoms, nuclear_order=nuclear_order, **controls)

    def get_g_tensor(self, atoms=None, **controls):
        """Static electronic g tensor (dimensionless), with ppm shift terms.

        Uses the native open-shell reference/model restrictions. ROKS/ROHF
        have a dedicated common-origin route; GKS/GHF require an explicit
        single-Kramers-pair approximation. ``nuclear_order=1/2`` is a
        separate integer all-electron HF/UHF London route, with Cartesian
        gradient/Hessian shapes [3N,3,3]/[3N,3N,3,3]; the Hessian is null
        for order one. Default ASE units are inverse Angstrom and inverse
        Angstrom squared; atomic mode retains inverse Bohr units. DFT,
        RO/spinor nuclear response, nonzero electric field and stability
        following remain unsupported. This is not experimental-spectrum
        validation or a complete many-electron SOC model.
        """
        return self.get_feature_report("g-tensor", atoms, **controls)

    def get_roks(self, atoms=None, **controls):
        """Restricted open-shell Kohn-Sham energy and gradient."""
        return self.get_feature_report("roks", atoms, **controls)

    def get_roks_hessian(self, atoms=None, **controls):
        """Analytic Hessian and harmonic frequencies of the restricted
        open-shell energy."""
        return self.get_feature_report("roks-hessian", atoms, **controls)

    def get_roks_polarizability(self, atoms=None, **controls):
        """Dipole moment and static polarizability of the restricted
        open-shell reference."""
        return self.get_feature_report("roks-polarizability", atoms, **controls)

    def get_divide_conquer(self, atoms=None, **controls):
        """Divide-and-conquer Kohn-Sham energy and gradient."""
        return self.get_feature_report("dc", atoms, **controls)

    def get_mrsf(self, atoms=None, **controls):
        """Mixed-reference spin-flip singlets and triplets."""
        return self.get_feature_report("mrsf", atoms, **controls)

    def get_mrsf_state(self, atoms=None, **controls):
        """Analytic gradient of one mixed-reference spin-flip state."""
        return self.get_feature_report("mrsf-state", atoms, **controls)

    def get_mrsf_hessian(self, atoms=None, **controls):
        """Analytic Hessian and harmonic frequencies of one mixed-reference
        spin-flip state."""
        return self.get_feature_report("mrsf-hessian", atoms, **controls)

    def get_spin_orbit(self, atoms=None, **controls):
        """One-electron spin-orbit coupling at the selected molecular reference.

        With reference="gks", the generated operator enters the
        self-consistent spin-covariant GKS equations.
        """
        return self.get_feature_report("spin-orbit", atoms, **controls)

    def get_spinor_polarizability(self, atoms=None, *, frequencies, **controls):
        """Dynamic polarizability of the self-consistent spin-orbit spinor
        state at ``frequencies``."""
        return self.get_feature_report("spinor-polarizability", atoms, frequencies=frequencies, **controls)

    def get_magnetic_anisotropy(self, atoms=None, **controls):
        """Magnetic anisotropy energies of several spin axes."""
        return self.get_feature_report("magnetic-anisotropy", atoms, **controls)

    def get_magnon(self, atoms=None, **controls):
        """Transverse spin excitation energies (spin-flip TDA)."""
        return self.get_feature_report("magnon", atoms, **controls)

    def get_mcd(self, atoms=None, **controls):
        """Faraday A and B terms of magnetic circular dichroism."""
        return self.get_feature_report("mcd", atoms, **controls)

    def get_td_hessian(self, atoms=None, **controls):
        """Analytic isolated-state excited Hessian (Hartree/Bohr squared).
        HF flag doublet/quartet and full-spin HF subsets have complete Cartesian
        acceptance. Their reports contain tensors without harmonic postprocessing;
        semilocal spinor verification and correlated pole derivatives remain pending."""
        return self.get_feature_report("td-hessian", atoms, **controls)

    def get_td_ecd_giao(self, atoms=None, **controls):
        """Rotatory strengths with London orbitals in the length gauge."""
        return self.get_feature_report("td-ecd-giao", atoms, **controls)

    def get_roa(self, atoms=None, **controls):
        """Raman optical activity invariants and circular intensity differences."""
        return self.get_feature_report("roa", atoms, **controls)

    def get_hypermagnetizability(self, atoms=None, **controls):
        """Static hypermagnetizability, polarizability and magnetizability."""
        return self.get_feature_report("hypermagnetizability", atoms, **controls)

    def get_cotton_mouton(self, atoms=None, **controls):
        """Cotton-Mouton constant per molecule at a temperature."""
        return self.get_feature_report("cotton-mouton", atoms, **controls)

    def get_magneto_optical(self, atoms=None, **controls):
        """Magneto-optical tensor at a photon energy (spin-orbit or external field)."""
        return self.get_feature_report("magneto-optical", atoms, **controls)

    def get_magnetic_field_scf(self, atoms=None, **controls):
        """Self-consistent field in a finite uniform magnetic field."""
        return self.get_feature_report("bfield", atoms, **controls)

    def get_magnetic_field_gradient(self, atoms=None, **controls):
        """Energy and analytic gradient in a finite uniform magnetic field."""
        return self.get_feature_report("bfield-gradient", atoms, **controls)

    def get_magnetoelectric(self, atoms=None, **controls):
        """Linear magnetoelectric response to first order in spin-orbit coupling."""
        return self.get_feature_report("magnetoelectric", atoms, **controls)

    _PERIODIC_REFERENCE_PARAMETERS = (
        "basis", "ecp", "xc", "charge", "multiplicity", "unrestricted", "k_mesh",
        "overlap_cutoff", "exchange_cutoff", "grid_level", "grid_preset",
        "grid_angular_scheme", "grid_weight_cutoff", "grid_shell", "max_iter",
        "conv_tol", "conv_tol_grad", "conv_tol_dp", "level_shift", "damping",
        "diis_size", "adiis_until_error", "guess",
    )

    def _periodic_report(self, feature, atoms=None, *, _atomic=False, **controls):
        mode = self._unit_mode(controls)
        if atoms is None:
            atoms = self.atoms
        dimensional = feature in (
            "periodic-energy", "stress", "dos", "periodic-field-scf", "periodic-spinor-scf",
            "periodic-magnetic-scf", "lab-frame-spin-spiral", "dmi",
        )
        if atoms is None or not (any(atoms.get_pbc()) if dimensional else all(atoms.get_pbc())):
            raise ValueError(f"{feature} needs " + ("at least one periodic axis" if dimensional else "a fully periodic Atoms object"))
        numbers, positions = self._geometry(atoms)
        merged = {}
        reference = controls.get("reference", self.parameters.get("reference"))
        spinor = str(reference).strip().lower() in {"gks", "ghf"}
        dimensional_defaults = (
            feature in ("periodic-spinor-scf", "periodic-magnetic-scf", "lab-frame-spin-spiral", "dmi")
            or (dimensional and not all(atoms.get_pbc()))
        )
        adiis_disabled = bool(controls.get("no_adiis", not self.parameters.get("use_adiis", True)))
        for key in self._PERIODIC_REFERENCE_PARAMETERS:
            # Preserve an explicit cutoff so the reference can validate it.
            if (key == "adiis_until_error" and self.parameters.get(key) is None
                    and (spinor or adiis_disabled or dimensional_defaults)):
                continue
            if (key == "grid_shell" and dimensional_defaults
                    and self.parameters[key] == self.default_parameters[key]):
                continue
            value = self._k_mesh() if key == "k_mesh" else self._parameter(key)
            if value is not None:
                merged[key] = value
        if not self.parameters.get("use_adiis", True):
            merged["no_adiis"] = True
        temperature = float(self.parameters.get("electronic_temperature") or 0.0)
        if temperature != 0.0:
            merged["electronic_temperature"] = temperature
        # A dispersion model is passed on, so that a calculation without its
        # contribution refuses it by name instead of ignoring it.
        dispersion = str(self.parameters.get("dispersion") or "auto").lower()
        if dispersion not in ("auto", "none"):
            merged["dispersion"] = dispersion
        merged.update(self._extra_controls(feature))
        self._merge_report_controls(merged, controls, mode, feature)
        cell = (atoms.cell.array / Bohr).tolist()
        report = native.run_feature(feature, numbers, positions, cell=cell, pbc=atoms.get_pbc().tolist(), **merged)
        return report if _atomic or mode == "atomic" else _to_ase_units(report)

    def get_periodic_energy_report(self, atoms=None, **controls):
        """Periodic energy per cell (Hartree) and optional forces (Hartree/Bohr)."""
        return self._periodic_report("periodic-energy", atoms, **controls)

    def get_periodic_stress_report(self, atoms=None, **controls):
        """Affine stress normalized by the active periodic measure.

        Collinear commensurate Fourier fields keep reduced wave indices
        and Cartesian amplitudes fixed in the derivative.
        """
        return self._periodic_report("stress", atoms, **controls)

    def get_opt_fixed_cell(self, atoms=None, **controls):
        """Fixed-cell relaxation report (Bohr positions, Hartree energies)."""
        return self._periodic_report("opt-fixed-cell", atoms, **controls)

    def get_cell_opt(self, atoms=None, **controls):
        """Variable-cell relaxation report: relaxed ``cell`` rows and
        ``positions`` in Bohr, enthalpy in Hartree, stress in Hartree/Bohr^3."""
        return self._periodic_report("cell-opt", atoms, **controls)

    def get_relaxed_atoms(self, atoms=None, *, cell=False, **controls):
        """Relax the atoms (``cell=False``) or atoms and cell (``cell=True``)
        and return a new ``Atoms`` in Angstrom; the optimization report (ASE
        units) is stored in ``relaxed.info["dft_rs_report"]``."""
        if atoms is None:
            atoms = self.atoms
        mode = self._unit_mode(controls)
        report = self._periodic_report(
            "cell-opt" if cell else "opt-fixed-cell", atoms, _atomic=True, units=mode, **controls
        )
        relaxed = atoms.copy()
        if "cell" in report:
            relaxed.set_cell(np.asarray(report["cell"], dtype=float) * Bohr, scale_atoms=False)
        relaxed.set_positions(np.asarray(report["positions"], dtype=float) * Bohr)
        relaxed.info["dft_rs_report"] = report if mode == "atomic" else _to_ase_units(report)
        return relaxed

    def get_phonons(self, atoms=None, **controls):
        """Gamma-point phonons from analytic force constants (cm^-1).
        Optional field selects the stationary finite-bias Berry enthalpy;
        it is eV/Angstrom by default, atomic field units with units='atomic'."""
        return self._periodic_report("phonons", atoms, **controls)

    def get_phonon_bands(self, atoms=None, *, supercell, path, **controls):
        """Phonon dispersion from analytic supercell force constants (cm^-1).
        lattice_flexoelectric adds fixed-mass-center internal relaxation and
        the Born-charge lattice contribution (e/Angstrom by default), with
        order-dependent coefficient units. See native.phonon_bands for scope.
        """
        return self._periodic_report("phonon-bands", atoms, supercell=supercell, path=self._path_vertices(path), **controls)

    def get_phonon_thermo(self, atoms=None, *, supercell, q_mesh, **controls):
        """Harmonic lattice thermodynamics per primitive cell (Hartree, K)."""
        return self._periodic_report("phonon-thermo", atoms, supercell=supercell, q_mesh=q_mesh, **controls)

    def get_electron_phonon(self, atoms=None, **controls):
        """Electron-phonon coupling, a2F and critical temperatures (K):
        McMillan, Allen-Dynes and isotropic Eliashberg (``tc_eliashberg_k``), and the
        nonlinear gap with Pade continuation for ``gap_temperatures_k``.
        ``scdft_pairing`` adds Gamma-vertex inputs and static K/Z. For generated
        Coulomb screening use ``scdft_screening="rpa"`` or ``"rpa-neutral"``.
        Default units are eV, eV squared for squared vertices and inverse Angstrom.
        """
        return self._periodic_report("electron-phonon", atoms, **controls)

    def get_quasi_harmonic(self, atoms=None, *, supercell, q_mesh, **controls):
        """Quasi-harmonic thermodynamics on scaled cells (Bohr^3, GPa, 1/K)."""
        return self._periodic_report("quasi-harmonic", atoms, supercell=supercell, q_mesh=q_mesh, **controls)

    def get_elastic(self, atoms=None, **controls):
        """Clamped- and relaxed-ion elastic constants (Voigt, GPa)."""
        return self._periodic_report("elastic", atoms, **controls)

    def get_elastic_fd(self, atoms=None, **controls):
        """Validation route: elastic constants from stress differences (GPa)."""
        return self._periodic_report("elastic-fd", atoms, **controls)

    def get_elastic_moduli(self, atoms=None, **controls):
        """Voigt, Reuss and Hill moduli (GPa), wave speeds and Debye temperature."""
        return self._periodic_report("elastic-moduli", atoms, **controls)

    def get_christoffel(self, atoms=None, **controls):
        """Christoffel phase velocities (m/s) along ``directions``."""
        return self._periodic_report("christoffel", atoms, **controls)

    def get_material_card(self, atoms=None, **controls):
        """Continuum material card (GPa, kg/m^3, m/s); ``card_path`` writes it."""
        return self._periodic_report("material-card", atoms, **controls)

    def get_equation_of_state(self, atoms=None, **controls):
        """Birch-Murnaghan equation of state (V0 in Bohr^3, B0 in GPa)."""
        return self._periodic_report("equation-of-state", atoms, **controls)

    def get_dielectric_tensor(self, atoms=None, **controls):
        """Electronic dielectric tensor by periodic DFPT."""
        return self._periodic_report("dielectric-tensor", atoms, **controls)

    def get_born_charges(self, atoms=None, **controls):
        """Born effective charges (e) and the electronic dielectric tensor."""
        return self._periodic_report("born-charges", atoms, **controls)

    def get_piezoelectric_tensor(self, atoms=None, **controls):
        """Proper clamped- and relaxed-ion piezoelectric tensors."""
        return self._periodic_report("piezoelectric-tensor", atoms, **controls)

    def get_berry_polarization(self, atoms=None, **controls):
        """Berry-phase polarization (modulo the polarization quanta)."""
        return self._periodic_report("berry-polarization", atoms, **controls)

    def get_crystal_ir(self, atoms=None, **controls):
        """Crystal infrared intensities and LO-TO splitting (cm^-1, km/mol)."""
        return self._periodic_report("crystal-ir", atoms, **controls)

    def get_static_dielectric(self, atoms=None, **controls):
        """Static (relaxed-ion) dielectric tensor."""
        return self._periodic_report("static-dielectric", atoms, **controls)

    def get_electronic_structure(self, atoms=None, **controls):
        """SCF bands and populations; coefficients=True adds complex Bloch orbitals and F/S/P matrices."""
        return self._periodic_report("electronic-structure", atoms, **controls)

    def get_effective_mass(self, atoms=None, *, k_point, band, **controls):
        """Band velocity and effective-mass tensor (atomic units, m_e)."""
        return self._periodic_report("effective-mass", atoms, k_point=k_point, band=band, **controls)

    def get_boltzmann_transport(self, atoms=None, *, transport_k_mesh, transport_temperatures_k, **controls):
        """Boltzmann transport in SI; optional scattering_table includes tau(E)."""
        return self._periodic_report(
            "boltzmann-transport",
            atoms,
            transport_k_mesh=transport_k_mesh,
            transport_temperatures_k=transport_temperatures_k,
            **controls,
        )

    def get_hall_transport(self, atoms=None, *, transport_k_mesh, transport_temperatures_k, **controls):
        """Ordinary weak-field Hall tensors in SI units for isolated rigid
        bands. Optional scattering_table includes energy-dependent lifetime.
        Axes: electric/current and magnetic x,y,z."""
        return self._periodic_report(
            "hall-transport", atoms, transport_k_mesh=transport_k_mesh,
            transport_temperatures_k=transport_temperatures_k, **controls,
        )

    def get_coherent_transport(self, atoms=None, *, transport_k_mesh, transport_frequencies_hartree, **controls):
        """Covariant intra/interband conductivity (S/m), including degeneracy."""
        return self._periodic_report("coherent-transport", atoms,
                                     transport_k_mesh=transport_k_mesh,
                                     transport_frequencies_hartree=transport_frequencies_hartree, **controls)

    def get_quantum_hall(self, atoms=None, **controls):
        """Non-Abelian occupied-slice Chern invariant and Hall conductance (S)."""
        return self._periodic_report("quantum-hall", atoms, **controls)

    def get_wannier_flow(self, atoms=None, **controls):
        """Hybrid Wannier charge centres and their flow."""
        return self._periodic_report("wannier-flow", atoms, **controls)

    def get_periodic_cubic_fc_along(self, atoms=None, **controls):
        """Cubic force constants contracted with a Gamma displacement pattern.
        Optional field selects the stationary finite-bias Berry enthalpy."""
        return self._periodic_report("periodic-cubic-fc-along", atoms, **controls)

    def get_periodic_quartic_fc_along(self, atoms=None, **controls):
        """Quartic force constants contracted with two Gamma displacement patterns.
        Optional field selects the stationary finite-bias Berry enthalpy."""
        return self._periodic_report("periodic-quartic-fc-along", atoms, **controls)

    def get_periodic_cubic_fc(self, atoms=None, **controls):
        """Cubic force constants for Gamma-point displacements.
        Optional field selects the stationary finite-bias Berry enthalpy."""
        return self._periodic_report("periodic-cubic-fc", atoms, **controls)

    def get_periodic_quartic_fc(self, atoms=None, **controls):
        """Quartic force constants for Gamma-point displacements.
        Optional field selects the stationary finite-bias Berry enthalpy."""
        return self._periodic_report("periodic-quartic-fc", atoms, **controls)

    def get_toec(self, atoms=None, **controls):
        """Third-order Lagrangian elastic constants at clamped and relaxed ions."""
        return self._periodic_report("toec", atoms, **controls)

    def get_foec(self, atoms=None, **controls):
        """Fourth-order Lagrangian elastic constants at clamped and relaxed ions."""
        return self._periodic_report("foec", atoms, **controls)

    def get_mode_gruneisen(self, atoms=None, **controls):
        """Gamma phonon Gruneisen response; order two uses log-volume curvature.

        Set ``gruneisen_order=2`` for ``-d gamma/d ln(V)``. Acoustic and
        unstable coefficients are undefined. See ``mode_gruneisen`` for
        clamped/relaxed-ion and stationarity controls.
        """
        return self._periodic_report("mode-gruneisen", atoms, **controls)

    def get_periodic_beta(self, atoms=None, **controls):
        """Electric susceptibility of a cell and its field derivative."""
        return self._periodic_report("periodic-beta", atoms, **controls)

    def get_periodic_raman(self, atoms=None, **controls):
        """Born charges and the displacement derivative of the susceptibility."""
        return self._periodic_report("periodic-raman", atoms, **controls)

    def get_periodic_gamma(self, atoms=None, **controls):
        """Static clamped-ion third-order electric susceptibility of a cell."""
        return self._periodic_report("periodic-gamma", atoms, **controls)

    def get_electrostriction(self, atoms=None, **controls):
        """Strain derivative of the electric susceptibility of a cell."""
        return self._periodic_report("electrostriction", atoms, **controls)

    def get_piezo_higher(self, atoms=None, **controls):
        """Piezoelectric tensor of a cell and its strain derivative."""
        return self._periodic_report("piezo-higher", atoms, **controls)

    def get_periodic_efield(self, atoms=None, **controls):
        """Self-consistent field of a cell in a finite uniform electric field."""
        return self._periodic_report("periodic-efield", atoms, **controls)

    def get_dielectric_q(self, atoms=None, **controls):
        """Density response and inverse dielectric function at commensurate wave vectors."""
        return self._periodic_report("dielectric-q", atoms, **controls)

    def get_orbital_susceptibility_q(self, atoms=None, **controls):
        """Orbital magnetic susceptibility at commensurate wave vectors."""
        return self._periodic_report("orbital-susceptibility-q", atoms, **controls)

    def get_periodic_dc(self, atoms=None, **controls):
        """Divide-and-conquer Kohn-Sham energy and forces of a cell at Gamma."""
        return self._periodic_report("periodic-dc", atoms, **controls)

    def get_dc_stress(self, atoms=None, **controls):
        """Stress of the divide-and-conquer free energy of a cell."""
        return self._periodic_report("dc-stress", atoms, **controls)

    def get_periodic_nmr_shielding(self, atoms=None, **controls):
        """Nuclear magnetic shielding in a periodic cell at commensurate wave vectors."""
        return self._periodic_report("periodic-nmr-shielding", atoms, **controls)

    def get_berry_dipole(self, atoms=None, **controls):
        """Berry curvature dipole of the Fermi surface."""
        return self._periodic_report("berry-dipole", atoms, **controls)

    def get_lab_frame_spin_spiral(self, atoms=None, *, spiral_q, **controls):
        """Energy of a commensurate flat spin spiral (two-component SCF in
        its supercell); ``spiral_q`` in reduced reciprocal coordinates."""
        return self._periodic_report("lab-frame-spin-spiral", atoms, spiral_q=spiral_q, **controls)

    def get_dmi(self, atoms=None, *, spiral_q, **controls):
        """Chirality-odd spin-orbit energy of a commensurate spin spiral."""
        return self._periodic_report("dmi", atoms, spiral_q=spiral_q, **controls)

    def get_bands(self, atoms=None, *, path, **controls):
        """Band structure along reduced k vertices ``path`` (Hartree)."""
        return self._periodic_report("bands", atoms, path=self._path_vertices(path), **controls)

    def get_periodic_spin_response(self, atoms=None, **controls):
        """Longitudinal spin-only finite-q/frequency Zeeman response per cell."""
        return self._periodic_report("periodic-spin-response", atoms, **controls)

    def get_periodic_field_scf(self, atoms=None, **controls):
        """Complex finite periodic-field SCF and optional force and stress reports."""
        return self._periodic_report("periodic-field-scf", atoms, **controls)

    def get_periodic_spinor_scf(self, atoms=None, **controls):
        """Complex spinor SCF with total-number occupations and raw residuals."""
        return self._periodic_report("periodic-spinor-scf", atoms, **controls)

    def get_periodic_magnetic_scf(self, atoms=None, **controls):
        """Uniform-flux London reference on a physical 2D/3D magnetic cell.

        magnetic_model selects finite_gamma or crystal_bz. Wires use the
        separate periodic spinor reference with uniform_magnetic_field.
        GGA and meta-GGA SCF reports include angular and spatial XC defects.
        nuclear_order=1..4 selects the relaxed finite_gamma exchange/SOC
        nuclear provider. nuclear_direction selects a scalar path; otherwise
        gradient/Hessian or probe-contracted cubic/quartic tensors are returned
        in nuclear_response. Input directions/probes are dimensionless.
        """
        return self._periodic_report("periodic-magnetic-scf", atoms, **controls)

    def get_dos(self, atoms=None, *, sigma, **controls):
        """Density of states with Gaussian broadening ``sigma`` (Hartree)."""
        return self._periodic_report("dos", atoms, sigma=sigma, **controls)

    def get_pauli_spin_susceptibility(self, atoms=None, **controls):
        """Bare Pauli thermodynamics, or ``exchange_enhanced=True`` static
        semilocal longitudinal spin response with a thermal SCF per temperature.
        The common report includes units, assumptions and raw residuals."""
        return self._periodic_report("pauli-spin-susceptibility", atoms, **controls)

    def get_dirac_coulomb_scf(self, atoms=None, **controls):
        """Finite-nucleus no-pair energy with raw SCF diagnostics.

        Select the Coulomb, Gaunt or Breit interaction with dirac_interaction.
        """
        return self.get_feature_report("dirac-coulomb-scf", atoms, **controls)

    def get_two_component_relativistic(self, atoms=None, **controls):
        """Picture-changed finite-nucleus X2C, DKH2 or matrix ZORA report."""
        return self.get_feature_report("two-component-relativistic", atoms, **controls)

    def get_dirac_dks(self, atoms=None, **controls):
        """Fixed-sector four-component Dirac–Coulomb KS and electric response."""
        return self.get_feature_report("dirac-dks", atoms, **controls)

    @staticmethod
    def get_no_pair_projector(atoms=None, **inputs):
        """Four-component H/S projector, optionally generated from Atoms.
        Default Hamiltonian and separator are eV, overlap dimensionless;
        ``units="atomic"`` uses Hartree. Balanced input is the provider's
        responsibility for supplied matrices. With dirac_integrals="rkb",
        Atoms supplies Angstrom geometry and explicit nuclear_rms_radii
        use Angstrom (Bohr with units="atomic"). dirac_integrals="rmb"
        additionally retains the magnetic small basis and complex metric;
        magnetic_field/direction are atomic magnetic units and gauge_origin
        is Angstrom, or Bohr with units="atomic". No interacting SCF is inferred.
        """
        mode = inputs.pop("units", "ase")
        if mode not in ("ase", "atomic"):
            raise ValueError("units must be 'ase' or 'atomic'")
        if atoms is not None:
            if np.asarray(atoms.pbc).any():
                raise ValueError("balanced molecular Dirac integrals require isolated Atoms")
            inputs["numbers"] = atoms.get_atomic_numbers().tolist()
            inputs["positions"] = (np.asarray(atoms.get_positions()) / Bohr).tolist()
        report = native.no_pair_projector(
            **(inputs if mode == "atomic" else _controls_to_atomic(inputs))
        )
        return report if mode == "atomic" else _to_ase_units(report)

    @staticmethod
    def get_superconducting_state(**inputs):
        """Canonical band gap or supplied-operator full Nambu density response.

        No Atoms is needed. Energies/gaps and their path directions are eV,
        squared vertices eV squared and temperature Kelvin by default.
        units="atomic" uses Hartree. pairing_model="lm2005" generates
        phonon K and dimensionless Z from the supplied vertices and band
        weights; supplied and lm2005-off-diagonal require explicit Z.
        scdft_phase_selection=True checks the canonical phase for a supplied
        reciprocal kernel and uniform Z without starting guesses. The
        condensation free energy is in eV (Hartree with units="atomic").
        pairing_model="nambu" accepts full normal and antisymmetric pairing
        matrices in eV (Hartree with units="atomic"), with dimensionless
        overlap and positive temperature at supplied chemical potential.
        Supply signed frequencies and optional frequency_damping for retarded
        collisionless response. Dimensionless probe rows are flattened
        matrices in row-major order. Optional symmetric nambu_probe_interaction
        uses the energy unit. Bare and induced susceptibilities are inverse eV
        (inverse Hartree with units="atomic") at fixed quasiparticle occupations.
        """
        mode = inputs.pop("units", "ase")
        if mode not in ("ase", "atomic"):
            raise ValueError("units must be 'ase' or 'atomic'")
        report = native.superconducting_state(
            **(inputs if mode == "atomic" else _controls_to_atomic(inputs))
        )
        return report if mode == "atomic" else _to_ase_units(report)

    @staticmethod
    def get_scdft_pairing_kernel(**inputs):
        """Pairing K and optional phonon Z on supplied bands; no Atoms.
        Default energies are eV and squared vertices eV squared; temperature
        is Kelvin. ``units="atomic"`` uses Hartree and Hartree squared.
        Supply band_weights to generate dimensionless Z and its response.
        """
        mode = inputs.pop("units", "ase")
        if mode not in ("ase", "atomic"):
            raise ValueError("units must be 'ase' or 'atomic'")
        report = native.scdft_pairing_kernel(
            **(inputs if mode == "atomic" else _controls_to_atomic(inputs))
        )
        return report if mode == "atomic" else _to_ase_units(report)

    @staticmethod
    def get_piezoelectric_strain(*, piezoelectric_stress, piezoelectric_compliance,
                               ion_response):
        """Supplied atomic-unit proper e/S to d; no Atoms or SCF, SI outputs included.

        Voigt shear is engineering strain. Both tensors must have the same
        clamped/relaxed-ion and fixed-electric-field convention. This static
        data method retains native atomic units, like the explicit SI outputs.
        """
        return native.piezoelectric_strain(piezoelectric_stress=piezoelectric_stress,
                                         piezoelectric_compliance=piezoelectric_compliance,
                                         ion_response=ion_response)

    @staticmethod
    def get_aggregate_moduli(*, phases):
        """Aggregate bounds and estimates (GPa); no ``Atoms`` needed."""
        return native.aggregate_moduli(phases=phases)

    @staticmethod
    def get_rock_physics(**inputs):
        """Dry frames (including spheroidal pores), fluid substitution, Biot
        parameters and thermal-conductivity bounds; no ``Atoms`` needed."""
        return native.rock_physics(**inputs)

    def get_multicomponent(self, atoms=None, **controls):
        """Experimental simultaneous electronic/quantum-particle SCF report.

        ``quantum_atoms`` contains zero-based quantum nuclei. Geometry and
        Gaussian exponents cross the common ASE/atomic unit boundary once;
        particle masses use electron masses. The laboratory energy retains
        center-of-mass motion. ``species_derivative_order=1``/``2`` returns
        the analytic gradient/Hessian. ``quantum_position_constraint=True``
        differentiates nuclear expectation positions; otherwise a quantum
        center derivative measures finite-basis dependence. Target positions
        use Angstrom in ASE mode and Bohr in atomic mode. The same boundary
        converts gradients, Hessians and ``species_polarizability`` once.
        For ``species_electronic_model="dirac-coulomb"``, supply
        ``nuclear_rms_radii`` in Angstrom or Bohr in atomic mode.
        This route returns the joint fixed-sector energy and raw residuals.
        ``field`` adds the total dipole. ``species_polarizability=True``
        enables static or frequency-resolved coupled dipole response.
        ``species_quadrupole_polarizability=True`` adds quadrupole and
        mixed electric response. ``species_field_gradient`` applies the
        five gradient modes in eV/e/Angstrom², or Hartree/e/Bohr² in
        atomic mode. ``origin`` sets the shared dipole and quadrupole
        origin in the selected length unit or by name.
        """
        return self.get_feature_report("multicomponent", atoms, **controls)

    def get_continuous_species(self, atoms=None, **controls):
        """Fixed-template real nuclear-charge RKS model and analytic sensitivities.

        Supply fixed_electron_count, charge_directions, charge_coordinates and
        charge_bounds. Charges are in e and parameter coordinates dimensionless;
        charge_origin uses Angstrom by default and Bohr with units='atomic'.
        Optional charge_nuclear_gradient returns template nuclear forces.
        Local Gaussian amplitudes, steric strengths and fragment_internal_energy
        use eV by default; exponents use Angstrom^-2 and environment hardness
        Angstrom^-3. Atomic mode uses Hartree and Bohr. Explicit pair masks and
        fixed integer spin populations are dimensionless. Optional probes and
        dynamic response use the common report's unit labels.
        """
        return self.get_feature_report("continuous-species", atoms, **controls)

    def get_reduced_fragment(self, atoms=None, *, model, fragment_mode="evaluate", **controls):
        """Evaluate saved candidate energy/forces with standard ASE conversion.

        ``model`` is opaque native-unit JSON text or a dict and is never unit
        converted. Atoms are the retained labels followed by the saved ionic
        role label. Atoms positions always use Angstrom, including atomic
        output mode. Field/origin controls use ASE units by default, atomic units
        in atomic mode. ``fragment_mode="evaluate"`` preserves the original
        record; ``"few-evaluate"`` uses a fixed PBE primitive record and returns
        energy/primary forces and metadata without dipole or response. The returned candidate status does not establish chemical accuracy
        or an internal-polarization model. This getter does not define masses
        for reduced-coordinate dynamics or relax frozen internal coordinates.

        ``"rigid-evaluate"`` instead requires an explicit ``rigid-static-core-v1``
        primitive recipe and atoms at its retained point-ion centers.
        ``derivative_order=0/1/2`` requests energy, analytic gradient or relaxed
        Hessian. Center gradients and Hessians convert to eV/Angstrom and
        eV/Angstrom^2; center/angle blocks convert to eV/Angstrom/radian, and
        angle blocks to eV/radian^2. Euler gradients are energy covectors,
        not Cartesian forces or world-frame torques. Recipe angles remain
        native radians and opaque model coefficients are never converted.
        The static PBE restricted route is experimental; private generated
        schemas and frozen internal-coordinate relaxation are unsupported.

        ``"scalar-evaluate"`` and ``"scalar-optimize"`` use the explicit
        frozen-scalar-action-v1 record and its first-state raw geometry.
        Flattened ``fragment_weights`` contain one simplex per saved site.
        The full joint endpoint evaluators retain site cross interactions;
        this is not a compressed potential. Weight gradients and residuals
        with Hartree dimensions convert to eV. Native-unit saved models are
        opaque. Only weights change; nuclear geometry optimization and
        reaction enthalpies are not implemented by these modes.

        ``"scalar-gradient"``, ``"scalar-geometry-optimize"`` and ``"scalar-hessian"`` select one
        explicit ``fragment_state`` and ``fragment_endpoint``. The saved
        endpoint must declare ``fixed-internal-affine-v1`` transport and
        full-atom owners. Only the raw-center coordinates move, with fixed
        internal offsets and no fragment rotation or weight interpolation.
        Gradients/forces convert to eV/Angstrom and positions to Angstrom.
        The geometry convergence flag is returned without inferring a
        vibrational minimum or reaction enthalpy.
        """
        if fragment_mode not in ("evaluate", "few-evaluate", "rigid-evaluate", "scalar-evaluate", "scalar-optimize",
                                 "scalar-gradient", "scalar-geometry-optimize", "scalar-hessian"):
            raise ValueError("unsupported saved fragment evaluation mode")
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError("an Atoms object is required")
        if self._is_periodic(atoms):
            raise PropertyNotImplementedError("saved reduced-fragment evaluation does not implement periodic boundary conditions")
        mode = self._unit_mode(controls)
        numbers, positions = self._geometry(atoms)
        # The saved model supplies its reference. The outer calculator envelope
        # resolves resources; the native call inherits its per-call overrides.
        merged = self._extra_controls("continuous-species")
        merged.update(controls if mode == "atomic"
                      else _controls_to_atomic(controls, "continuous-species"))
        merged["fragment_mode"] = fragment_mode
        report = native.evaluate_reduced_fragment(numbers, positions, model=model, **merged)
        return report if mode == "atomic" else _to_ase_units(report)

    def get_casscf(self, atoms=None, **controls):
        """Complete-active-space SCF report through the common driver.

        ``cas=[active electrons, active orbitals]`` is required. Energies and
        energy tolerances are eV in ASE mode and Hartree in atomic mode.
        """
        return self.get_feature_report("casscf", atoms, **controls)

    def get_mcpdft(self, atoms=None, **controls):
        """Multiconfiguration pair-density functional theory report.

        Takes the ``casscf`` controls plus ``on_top_functional``,
        ``pdft_lambda`` and ``pdft_method``; energies are eV in ASE mode and
        Hartree in atomic mode.
        """
        return self.get_feature_report("mcpdft", atoms, **controls)

    @staticmethod
    def get_feature_registry():
        """The registry of public calculations as data."""
        return native.feature_registry()


# Keyword arguments with a dimension of the dedicated getters. In ASE units
# (the default) they are given in eV and Angstrom and the result is
# converted from its unit labels; `units="atomic"` passes both through.
_FIELD = "hartree/e/bohr"
_SCF = {
    "conv_tol": "hartree",
    "conv_tol_grad": "hartree",
    "level_shift": "hartree",
    "adiis_until_error": "hartree",
    "exchange_cutoff": "bohr",
    "basis_cutoff": "bohr",
    "grid_cutoff": "bohr",
    "grid_weight_cutoff": "bohr^3",
}
_EWALD = {
    "alpha": "1/bohr",
    "real_cutoff": "bohr",
    "reciprocal_cutoff": "1/bohr",
    "countercharge_center": "bohr",
    "countercharge_width": "bohr",
}
_RESPONSE = {
    "q": "1/bohr",
    "omega": "hartree",
    "damping": "hartree",
    "reciprocal_cutoff": "1/bohr",
    "scalar_potential": "hartree",
}
_DEDICATED = {
    "get_periodic_basis": _SCF,
    "get_dimensional_scf": _SCF,
    "get_dimensional_field_response": {**_SCF, **_EWALD, "frequency": "hartree", "scalar_potential": "hartree"},
    "get_periodic_field_operators": {"basis_cutoff": "bohr", "wave": "1/bohr", "k_offset": "1/bohr", "frequency": "hartree"},
    "get_periodic_xc_operators": _SCF,
    "get_periodic_coulomb_operators": {"basis_cutoff": "bohr", "nuclear_widths": "bohr", **_EWALD},
    "get_periodic_gaussian_integral": {"exponents": "1/bohr^2", "source_width": "bohr", **_EWALD},
    "get_periodic_electrostatics": {"widths": "bohr", **_EWALD},
    "get_periodic_field_response": _RESPONSE,
    "get_periodic_density_response": _RESPONSE,
    "get_periodic_thermal_response": {"delta_mu": "hartree", "reciprocal_cutoff": "1/bohr"},
    "get_periodic_dielectric": _RESPONSE,
    "get_dynamic_polarizability": {"omega": "hartree", "damping": "hartree", **_SCF,
                                 "scf_damping": "dimensionless",
                                   "field": _FIELD, "reference_field": _FIELD,
                                   "gks_operator_real": "hartree", "gks_operator_imaginary": "hartree",
                                   "spinor_xc_tolerance": "hartree"},
    "get_stability": {"resolution": "hartree"},
    "get_stab_following": {"resolution": "hartree", "min_energy_drop": "hartree"},
    # Aliases delegate here so each input crosses the unit boundary once.
    "get_dynamic_electric_tensor": {"input_frequencies": "hartree", "frequencies": "hartree", "damping": "hartree"},
    "get_orbital_energy_gradients": {"degeneracy_tolerance": "hartree"},
    "get_frontier_gap_gradient": {"degeneracy_tolerance": "hartree"},
    "get_magnetic_response": {"origin": "bohr"},
    "get_nmr": {"origin": "bohr"},
    "get_nuclear_taylor": {"direction": "bohr", "reference_field": _FIELD},
    "get_mixed_nuclear_response": {"directions": "bohr"},
    "get_molecular_taylor": {"nuclear_direction": "bohr", "electric_direction": _FIELD, "reference_field": _FIELD},
    "get_mixed_molecular_response": {"nuclear_directions": "bohr", "electric_directions": _FIELD, "reference_field": _FIELD},
    "get_static_spectroscopic_derivative": {"nuclear_direction": "bohr", "reference_field": _FIELD},
    "get_anharmonic": {},
}


def _in_calculator_units(name, arguments):
    method = getattr(DftRs, name)

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        mode = self._unit_mode(kwargs)
        if mode != "atomic":
            for key, label in arguments.items():
                value = kwargs.get(key)
                if value is None:
                    continue
                if hasattr(value, "tolist"):
                    value = value.tolist()
                outcome = _converted(value, label, kwargs, inverse=True, integers=True)
                if outcome is not None:
                    kwargs[key] = outcome[0]
        result = method(self, *args, **kwargs)
        if mode == "atomic" or not isinstance(result, dict):
            return result
        return _to_ase_units(result)

    setattr(DftRs, name, wrapper)


for _name, _arguments in _DEDICATED.items():
    _in_calculator_units(_name, _arguments)
del _name, _arguments


def _with_calculator_resources(method):
    """Apply resources once around an entire getter, including legacy adapters."""
    execute = native._reporting(method)

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        explicit = any(kwargs.get(name) is not None for name in ("cpu_threads", "memory_mb"))
        explicit_logging = any(kwargs.get(name) is not None for name in ("log_file", "log_mode", "log_level"))
        limits = {name: kwargs.pop(name, self.parameters.get(name)) for name in ("cpu_threads", "memory_mb")}
        limits = {name: value for name, value in limits.items() if value is not None}
        logging = {name: kwargs.pop(name, self.parameters.get(name)) for name in ("log_file", "log_mode", "log_level")}
        logging = {name: value for name, value in logging.items() if value is not None}
        current = getattr(native._core, "_resource_usage", lambda: None)()
        configured = getattr(native._core, "_resource_scope_configured", lambda: True)()
        if current is not None and configured and not explicit:
            limits = {}
        if getattr(native._core, "_logging_active", lambda: False)() and not explicit_logging:
            logging = {}
        if not limits and not logging:
            return method(self, *args, **kwargs)
        return execute(self, *args, **kwargs, **limits, **logging)

    wrapper.__signature__ = execute.__signature__
    return wrapper


for _name, _method in list(vars(DftRs).items()):
    if isinstance(_method, staticmethod) and _name.startswith("get_"):
        setattr(DftRs, _name, staticmethod(native._reporting(_method.__func__)))
    elif callable(_method) and (_name.startswith("get_") or _name == "calculate"):
        setattr(DftRs, _name, _with_calculator_resources(_method))
del _name, _method
