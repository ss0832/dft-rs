# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Structures and calculator objects for the common driver.

``Molecule`` and ``Crystal`` are immutable structures in atomic units
(Bohr). A calculator binds a structure to reference controls and runs the
calculations of the common driver, so that a structure and a method are
stated once::

    water = Molecule.from_xyz("water.xyz")
    calc = MolecularCalculator(water, xc="b3lyp", basis="def2-svp")
    energy = calc.energy().energy
    gradient = calc.gradient().gradient

    crystal = Crystal.from_angstrom([14, 14], positions, cell)
    calc = PeriodicCalculator(crystal, xc="pbe", basis="def2-svp", k_mesh=(4, 4, 4))
    relaxed, history = calc.relax(cell=True)
    bands = calc.with_structure(relaxed).phonon_bands(supercell=(2, 2, 2), path=vertices)

Every calculation of the driver that applies to the structure is a method
named like its function in :mod:`dft_rs.native`; keyword arguments override
the stored controls for one call. Reports are :class:`dft_rs.Report`
dictionaries in the driver's atomic units, and their ``units`` entry names
every unit.
"""

from __future__ import annotations

import re
import operator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from . import native

__all__ = [
    "Molecule",
    "Crystal",
    "Calculator",
    "MolecularCalculator",
    "PeriodicCalculator",
    "calculator",
    "features",
]

_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
    "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr "
    "Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
_NUMBER_OF = {symbol.lower(): index + 1 for index, symbol in enumerate(_SYMBOLS)}


def _vector(values, name):
    out = tuple(float(v) for v in values)
    if len(out) != 3:
        raise ValueError(f"{name} needs three components")
    return out


def _cross(u, v):
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _read_xyz(path):
    """Atomic numbers, Angstrom positions and the comment line of an XYZ file."""
    lines = Path(path).read_text().splitlines()
    try:
        count = int(lines[0].split()[0])
    except (IndexError, ValueError):
        raise ValueError(f"{path}: the first line of an XYZ file is the number of atoms") from None
    if len(lines) < count + 2:
        raise ValueError(f"{path}: {count} atoms announced, {max(len(lines) - 2, 0)} given")
    numbers = []
    positions = []
    for line in lines[2 : count + 2]:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"{path}: an atom line needs a symbol and three coordinates: {line!r}")
        symbol = fields[0]
        number = int(symbol) if symbol.isdigit() else _NUMBER_OF.get(symbol.lower())
        if number is None:
            raise ValueError(f"{path}: unknown element {symbol!r}")
        numbers.append(number)
        positions.append([float(x) for x in fields[1:4]])
    return numbers, positions, lines[1]


def _displaced(positions, atom, axis, step):
    moved = [list(p) for p in positions]
    moved[atom][axis] += float(step)
    return moved


@dataclass(frozen=True)
class Molecule:
    """Integer atomic numbers (1..118) and Cartesian positions in Bohr."""

    numbers: tuple
    positions: tuple

    def __post_init__(self):
        numbers = native._atomic_numbers(self.numbers)
        positions = tuple(_vector(p, "a position") for p in self.positions)
        if len(numbers) != len(positions):
            raise ValueError(f"{len(numbers)} atomic numbers for {len(positions)} positions")
        object.__setattr__(self, "numbers", numbers)
        object.__setattr__(self, "positions", positions)

    @classmethod
    def from_angstrom(cls, numbers, positions):
        """Cartesian positions given in Angstrom."""
        scale = native.BOHR_PER_ANGSTROM
        return cls(numbers, [[scale * float(x) for x in p] for p in positions])

    @classmethod
    def from_xyz(cls, path):
        """An XYZ file (Angstrom)."""
        numbers, positions, _ = _read_xyz(path)
        return cls.from_angstrom(numbers, positions)

    @classmethod
    def from_ase(cls, atoms):
        """An ASE ``Atoms`` object (Angstrom); its cell is ignored."""
        return cls.from_angstrom(
            [int(z) for z in atoms.get_atomic_numbers()], atoms.get_positions().tolist()
        )

    def to_ase(self):
        """An ASE ``Atoms`` object (Angstrom); needs ASE."""
        from ase import Atoms

        return Atoms(numbers=list(self.numbers), positions=[list(p) for p in self.angstrom_positions()])

    @property
    def natoms(self):
        """Number of atoms in this immutable structure."""
        return len(self.numbers)

    def angstrom_positions(self):
        """Cartesian positions converted from Bohr to Angstrom."""
        scale = 1.0 / native.BOHR_PER_ANGSTROM
        return tuple(tuple(scale * x for x in p) for p in self.positions)

    def with_positions(self, positions):
        """The same atoms at other Cartesian positions (Bohr)."""
        return Molecule(self.numbers, positions)

    def displaced(self, atom, axis, step):
        """One atom moved by ``step`` Bohr along a Cartesian axis (0, 1, 2)."""
        return Molecule(self.numbers, _displaced(self.positions, atom, axis, step))


@dataclass(frozen=True)
class Crystal:
    """Integer atomic numbers (1..118), positions and lattice rows in Bohr."""

    numbers: tuple
    positions: tuple
    cell: tuple
    pbc: tuple = (True, True, True)

    def __post_init__(self):
        numbers = native._atomic_numbers(self.numbers)
        positions = tuple(_vector(p, "a position") for p in self.positions)
        cell = tuple(_vector(row, "a lattice row") for row in self.cell)
        pbc = native._periodicity(self.pbc)
        if len(numbers) != len(positions):
            raise ValueError(f"{len(numbers)} atomic numbers for {len(positions)} positions")
        if len(cell) != 3 or len(pbc) != 3:
            raise ValueError("a crystal needs three lattice rows and three periodicity flags")
        for name, value in (
            ("numbers", numbers),
            ("positions", positions),
            ("cell", cell),
            ("pbc", pbc),
        ):
            object.__setattr__(self, name, value)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_angstrom(cls, numbers, positions, cell, pbc=(True, True, True)):
        """Cartesian positions and lattice rows given in Angstrom."""
        scale = native.BOHR_PER_ANGSTROM
        return cls(
            numbers,
            [[scale * float(x) for x in p] for p in positions],
            [[scale * float(x) for x in row] for row in cell],
            pbc,
        )

    @classmethod
    def from_fractional(cls, numbers, fractional, cell, pbc=(True, True, True)):
        """Three-component fractional coordinates with lattice rows in Bohr."""
        rows = [_vector(row, "a lattice row") for row in cell]
        if len(rows) != 3:
            raise ValueError("a crystal needs three lattice rows")
        positions = [
            tuple(sum(f[i] * rows[i][axis] for i in range(3)) for axis in range(3))
            for f in (_vector(position, "a fractional position") for position in fractional)
        ]
        return cls(numbers, positions, rows, pbc)

    @classmethod
    def from_xyz(cls, path):
        """An extended XYZ file (Angstrom) whose comment line carries
        ``Lattice="ax ay az bx by bz cx cy cz"`` and optionally ``pbc="T T F"``."""
        numbers, positions, comment = _read_xyz(path)
        lattice = re.search(r'lattice="([^"]*)"', comment, flags=re.IGNORECASE)
        if lattice is None:
            raise ValueError(f'{path}: the comment line has no Lattice="..." entry')
        values = [float(x) for x in lattice.group(1).split()]
        if len(values) != 9:
            raise ValueError(f"{path}: Lattice needs nine numbers (Angstrom)")
        pbc = (True, True, True)
        flags = re.search(r'pbc="([^"]*)"', comment, flags=re.IGNORECASE)
        if flags is not None:
            words = flags.group(1).lower().split()
            if len(words) != 3 or any(word not in ("t", "f", "true", "false", "1", "0") for word in words):
                raise ValueError(f"{path}: pbc takes three T/F flags")
            pbc = tuple(word in ("t", "true", "1") for word in words)
        return cls.from_angstrom(numbers, positions, [values[0:3], values[3:6], values[6:9]], pbc)

    @classmethod
    def from_ase(cls, atoms):
        """An ASE ``Atoms`` object (Angstrom)."""
        return cls.from_angstrom(
            [int(z) for z in atoms.get_atomic_numbers()],
            atoms.get_positions().tolist(),
            atoms.cell.array.tolist(),
            [bool(flag) for flag in atoms.get_pbc()],
        )

    def to_ase(self):
        """An ASE ``Atoms`` object (Angstrom); needs ASE."""
        from ase import Atoms

        return Atoms(
            numbers=list(self.numbers),
            positions=[list(p) for p in self.angstrom_positions()],
            cell=[list(row) for row in self.angstrom_cell()],
            pbc=list(self.pbc),
        )

    # -- geometry ----------------------------------------------------------

    @property
    def natoms(self):
        """Number of atoms in the periodic unit cell."""
        return len(self.numbers)

    @property
    def volume(self):
        """Cell volume in Bohr^3."""
        return abs(_dot(self.cell[0], _cross(self.cell[1], self.cell[2])))

    def fractional_positions(self):
        """Positions in units of the lattice rows."""
        a = self.cell
        det = _dot(a[0], _cross(a[1], a[2]))
        if det == 0.0:
            raise ValueError("the lattice rows are linearly dependent")
        dual = (_cross(a[1], a[2]), _cross(a[2], a[0]), _cross(a[0], a[1]))
        return tuple(tuple(_dot(p, dual[i]) / det for i in range(3)) for p in self.positions)

    def angstrom_positions(self):
        """Unit-cell Cartesian positions in Angstrom."""
        scale = 1.0 / native.BOHR_PER_ANGSTROM
        return tuple(tuple(scale * x for x in p) for p in self.positions)

    def angstrom_cell(self):
        """Lattice vectors as rows in Angstrom."""
        scale = 1.0 / native.BOHR_PER_ANGSTROM
        return tuple(tuple(scale * x for x in row) for row in self.cell)

    # -- derived structures ------------------------------------------------

    def with_positions(self, positions):
        """The same cell with other Cartesian positions (Bohr)."""
        return Crystal(self.numbers, positions, self.cell, self.pbc)

    def with_cell(self, cell, *, scale_atoms=False):
        """Another cell (Bohr rows); ``scale_atoms`` keeps fractional coordinates."""
        if not scale_atoms:
            return Crystal(self.numbers, self.positions, cell, self.pbc)
        return Crystal.from_fractional(self.numbers, self.fractional_positions(), cell, self.pbc)

    def displaced(self, atom, axis, step):
        """One atom moved by ``step`` Bohr along a Cartesian axis (0, 1, 2)."""
        return self.with_positions(_displaced(self.positions, atom, axis, step))

    def scaled(self, factor):
        """Isotropic scaling of the cell and the positions by a linear factor."""
        factor = float(factor)
        if not factor > 0.0:
            raise ValueError("the linear scale factor must be positive")
        return Crystal(
            self.numbers,
            [[factor * x for x in p] for p in self.positions],
            [[factor * x for x in row] for row in self.cell],
            self.pbc,
        )

    def strained(self, strain):
        """Homogeneous strain ``x -> (1 + strain) x`` of cell and positions."""
        eps = [_vector(row, "a strain row") for row in strain]
        if len(eps) != 3:
            raise ValueError("the strain is a 3x3 matrix")

        def apply(v):
            return tuple(v[a] + sum(eps[a][b] * v[b] for b in range(3)) for a in range(3))

        return Crystal(
            self.numbers,
            [apply(p) for p in self.positions],
            [apply(row) for row in self.cell],
            self.pbc,
        )

    def supercell(self, repeats):
        """An ``n1 x n2 x n3`` supercell in the atom order of the driver:
        the primitive atom index runs fastest, then ``n1``, ``n2``, ``n3``.
        Repeats must be three positive integers; fractional values are rejected."""
        try:
            n = [operator.index(r) for r in repeats]
        except TypeError:
            raise ValueError("a supercell takes three positive integers") from None
        if len(n) != 3 or min(n) < 1:
            raise ValueError("a supercell takes three positive integers")
        numbers = []
        positions = []
        for k in range(n[2]):
            for j in range(n[1]):
                for i in range(n[0]):
                    shift = tuple(
                        i * self.cell[0][axis] + j * self.cell[1][axis] + k * self.cell[2][axis]
                        for axis in range(3)
                    )
                    for z, p in zip(self.numbers, self.positions):
                        numbers.append(z)
                        positions.append(tuple(p[axis] + shift[axis] for axis in range(3)))
        cell = [[n[row] * x for x in self.cell[row]] for row in range(3)]
        return Crystal(numbers, positions, cell, self.pbc)


def features(space=None):
    """The registered calculations as a list of dictionaries (``id``,
    ``title``, ``space``, ``route``, ``python``, ``ase``, ...).

    ``space`` filters by ``"molecule"``, ``"periodic"`` or ``"data"``; a
    calculation defined for molecules and crystals is listed for both."""
    if space is not None and space not in ("molecule", "periodic", "data"):
        raise ValueError("space must be molecule, periodic or data")
    entries = native.feature_registry()["features"]
    if space is None:
        return [dict(entry) for entry in entries]
    return [
        dict(entry)
        for entry in entries
        if entry.get("space") == space
        or (space in ("molecule", "periodic") and entry.get("space") == "molecule_and_periodic")
    ]


@lru_cache(maxsize=None)
def _driver_features(spaces):
    """Method name -> feature id of the driver calculations of some spaces."""
    out = {}
    for entry in native.feature_registry()["features"]:
        name = entry.get("python", "")
        if entry.get("route") == "driver" and entry.get("space") in spaces and name.isidentifier():
            out[name] = entry["id"]
    return out


class Calculator:
    """A structure with reference controls (``xc``, ``basis``, ...).

    ``run(feature, **controls)`` runs a calculation of the common driver on
    the structure. Each calculation that applies to the structure is also a
    method named like its function in :mod:`dft_rs.native`; ``features()``
    lists them. Use :class:`MolecularCalculator` or
    :class:`PeriodicCalculator`, or :func:`calculator` to pick by structure.
    """

    _spaces = ()
    _structure_type = object

    def __init__(self, structure, **controls: Any):
        """Bind a compatible immutable structure to common-driver controls."""
        if not isinstance(structure, self._structure_type):
            raise TypeError(
                f"{type(self).__name__} needs a {self._structure_type.__name__}, "
                f"got {type(structure).__name__}"
            )
        self.structure = structure
        self.controls = dict(controls)

    def __repr__(self):
        return f"{type(self).__name__}({self.structure!r}, **{self.controls!r})"

    def with_structure(self, structure):
        """The same controls on another structure."""
        return type(self)(structure, **self.controls)

    def with_controls(self, **controls: Any):
        """The same structure with updated controls (``None`` removes one)."""
        merged = {**self.controls, **controls}
        return type(self)(self.structure, **{k: v for k, v in merged.items() if v is not None})

    @classmethod
    def features(cls):
        """Method names and the driver calculations they run."""
        return dict(_driver_features(cls._spaces))

    def run(self, feature: str, **controls: Any):
        """Run a registered calculation; ``controls`` override the stored ones."""
        return native.run_feature(feature, self.structure, **{**self.controls, **controls})

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        feature = _driver_features(type(self)._spaces).get(name)
        if feature is None:
            raise AttributeError(
                f"{name!r} is not a calculation of {type(self).__name__}; see {type(self).__name__}.features()"
            )

        def method(**controls: Any):
            return self.run(feature, **controls)

        method.__name__ = name
        method.__doc__ = f"Run the `{feature}` calculation of the common driver."
        return method

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(_driver_features(type(self)._spaces)))


class MolecularCalculator(Calculator):
    """A :class:`Molecule` with reference controls: ``energy``, ``gradient``,
    ``charges``, ``dipole``, ``hessian_report``, ``excitations``, ..."""

    _spaces = ("molecule", "molecule_and_periodic")
    _structure_type = Molecule

    @property
    def molecule(self):
        """The immutable molecule bound to this calculator."""
        return self.structure

    def with_molecule(self, molecule):
        """The same controls on another molecule."""
        return self.with_structure(molecule)

    def relax(self, **controls: Any):
        """Relax the geometry with analytic gradients.

        This is the dedicated ``dft_rs.native.optimize`` route: next to the
        reference controls it understands (``xc``, ``basis``, ``charge``,
        ``multiplicity``, ``grid_level``, ...) it takes ``max_steps``,
        ``gtol`` (Hartree/Bohr), ``step_max`` (Bohr), ``history`` and
        ``trajectory``. Returns the relaxed :class:`Molecule` and the result
        (``positions``, ``energy``, ``converged``, ``iterations``,
        ``trajectory``) as a :class:`dft_rs.Report`."""
        result = native.optimize(
            self.structure.numbers, self.structure.positions, **{**self.controls, **controls}
        )
        return self.structure.with_positions(result["positions"]), native.Report(result)


class PeriodicCalculator(Calculator):
    """A :class:`Crystal` with reference controls: ``periodic_energy``,
    ``stress``, ``phonons``, ``phonon_bands``, ``quasi_harmonic``,
    ``elastic``, ``bands``, ..."""

    _spaces = ("periodic", "molecule_and_periodic")
    _structure_type = Crystal

    @property
    def crystal(self):
        """The immutable crystal bound to this calculator."""
        return self.structure

    def with_crystal(self, crystal):
        """The same controls on another crystal."""
        return self.with_structure(crystal)

    def relax(self, *, cell=False, **controls: Any):
        """Relax the atoms (``cell=False``) or atoms and cell (``cell=True``).

        Returns the relaxed :class:`Crystal` and the optimization report."""
        report = self.run("cell-opt" if cell else "opt-fixed-cell", **controls)
        relaxed = Crystal(
            self.structure.numbers,
            report["positions"],
            report.get("cell", self.structure.cell),
            self.structure.pbc,
        )
        return relaxed, report

    def scan_volumes(self, scales: Sequence[float], feature: str = "periodic-energy", **controls: Any):
        """Run one calculation on isotropically scaled copies of the crystal.

        Returns ``[(volume, report), ...]`` in the order of ``scales``."""
        out = []
        for scale in scales:
            scaled = self.structure.scaled(scale)
            out.append((scaled.volume, self.with_structure(scaled).run(feature, **controls)))
        return out


def calculator(structure, **controls: Any):
    """The calculator of a structure: molecular for a :class:`Molecule`,
    periodic for a :class:`Crystal`."""
    if isinstance(structure, Crystal):
        return PeriodicCalculator(structure, **controls)
    if isinstance(structure, Molecule):
        return MolecularCalculator(structure, **controls)
    raise TypeError(f"expected a Molecule or a Crystal, got {type(structure).__name__}")
