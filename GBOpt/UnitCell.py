# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Define conventional unit cells and exact built-in basis metadata.

The module owns crystal-structure decoration and immutable rational basis data used by
exact grain construction. Supercell enumeration and bicrystal assembly do not belong
here.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import KDTree

from GBOpt.Atom import Atom
from GBOpt.Utils.integer_linalg import (
    ExactIntegerShapeError,
    ExactIntegerTypeError,
    ExactIntegerValueError,
    as_int_array,
)


class UnitCellError(Exception):
    """Base class for exceptions in the UnitCell class"""


class UnitCellValueError(UnitCellError):
    """Exceptions raised when an incorrect value is assigned to a UnitCell attribute."""


class UnitCellTypeError(UnitCellError):
    """Exceptions raised when an incorrect type is assigned to a UnitCell attribute."""


class UnitCellRuntimeError(UnitCellError):
    """Exceptions raised when there is an error during runtime in the UnitCell class."""


def _as_unitcell_int_array(
        values: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Return exact Python integers using the shared integer validator.

    :param values: Candidate integer-valued array.
    :param shape: Required array shape.
    :param name: Input name used in validation messages.
    :return: Object-dtype array containing exact Python integers.
    :raises UnitCellTypeError: If values cannot be interpreted as exact integers.
    :raises UnitCellValueError: If values have an invalid shape or value.
    """
    try:
        return as_int_array(values, shape, name)
    except ExactIntegerTypeError as exc:
        raise UnitCellTypeError(str(exc)) from exc
    except (ExactIntegerShapeError, ExactIntegerValueError) as exc:
        raise UnitCellValueError(str(exc)) from exc


@dataclass(frozen=True, slots=True, init=False)
class RationalBasis:
    """Immutable exact fractional basis for a conventional unit cell.

    Each row in ``numerators`` is paired with the species at the same index in
    ``names``. Dividing the integer rows by ``denominator`` gives conventional-cell
    fractional coordinates in the canonical half-open interval ``[0, 1)``. Row order
    is preserved so built-in metadata follows the established structured-basis order.

    :param names: Ordered species name for every basis site.
    :param numerators: Exact coordinate numerators with shape ``(basis_size, 3)``.
    :param denominator: Positive common coordinate denominator.
    :raises UnitCellTypeError: If names or exact coordinates have invalid types.
    :raises UnitCellValueError: If the basis is empty, malformed, noncanonical, or
        contains duplicate coordinates.
    """

    names: tuple[str, ...]
    denominator: int
    _numerator_rows: tuple[tuple[int, int, int], ...] = field(repr=False)

    def __init__(
            self, names: Sequence[str], numerators: np.ndarray,
            denominator: int) -> None:
        denominator_array = _as_unitcell_int_array(
            (denominator,), (1,), "rational-basis denominator"
        )
        denominator = int(denominator_array[0])
        if denominator <= 0:
            raise UnitCellValueError(
                "The rational-basis denominator must be greater than zero."
            )

        if isinstance(names, (str, bytes)):
            raise UnitCellTypeError(
                "Rational-basis names must be a sequence of strings."
            )
        try:
            names = tuple(names)
        except TypeError as exc:
            raise UnitCellTypeError(
                "Rational-basis names must be a sequence of strings."
            ) from exc
        if any(not isinstance(name, str) for name in names):
            raise UnitCellTypeError("Every rational-basis name must be a string.")
        if not names:
            raise UnitCellValueError("A rational basis must contain at least one site.")

        try:
            numerator_array = np.asarray(numerators, dtype=object)
        except (TypeError, ValueError) as exc:
            raise UnitCellValueError(
                "Rational-basis numerators must have shape (basis_size, 3)."
            ) from exc
        if numerator_array.ndim != 2 or numerator_array.shape[1] != 3:
            raise UnitCellValueError(
                "Rational-basis numerators must have shape (basis_size, 3)."
            )
        if numerator_array.shape[0] != len(names):
            raise UnitCellValueError(
                "Rational-basis names and numerator rows must have equal lengths."
            )

        exact_numerators = _as_unitcell_int_array(
            numerator_array,
            numerator_array.shape,
            "rational-basis numerators",
        )
        normalized_rows = []
        for row in exact_numerators:
            normalized_row = []
            for value in row:
                integer = int(value)
                if integer < 0 or integer >= denominator:
                    raise UnitCellValueError(
                        "Rational-basis coordinates must lie in the half-open interval "
                        "[0, denominator)."
                    )
                normalized_row.append(integer)
            normalized_rows.append(tuple(normalized_row))

        coordinate_rows = tuple(normalized_rows)
        if len(coordinate_rows) != len(set(coordinate_rows)):
            raise UnitCellValueError(
                "Rational-basis coordinates must be unique within the conventional "
                "cell, regardless of species."
            )

        object.__setattr__(self, "names", names)
        object.__setattr__(self, "denominator", denominator)
        object.__setattr__(self, "_numerator_rows", tuple(normalized_rows))

    @property
    def numerators(self) -> np.ndarray:
        """Return a read-only copy of the exact coordinate numerators.

        :return: Object-dtype array containing exact Python integers with shape
            ``(basis_size, 3)``.
        """
        stored_copy = np.array(self._numerator_rows, dtype=object)
        stored_copy.setflags(write=False)
        readonly_view = stored_copy.view()
        readonly_view.setflags(write=False)
        return readonly_view


_RATIONAL_BASIS_SPECS = {
    "sc": (
        1,
        ((0, 0, 0, 0),),
    ),
    "bcc": (
        2,
        (
            (0, 0, 0, 0),
            (0, 1, 1, 1),
        ),
    ),
    "fcc": (
        2,
        (
            (0, 0, 0, 0),
            (0, 0, 1, 1),
            (0, 1, 0, 1),
            (0, 1, 1, 0),
        ),
    ),
    "diamond": (
        4,
        (
            (0, 0, 0, 0),
            (0, 0, 2, 2),
            (0, 2, 0, 2),
            (0, 2, 2, 0),
            (0, 1, 1, 1),
            (0, 3, 3, 1),
            (0, 3, 1, 3),
            (0, 1, 3, 3),
        ),
    ),
    "fluorite": (
        4,
        (
            (0, 0, 0, 0),
            (0, 0, 2, 2),
            (0, 2, 0, 2),
            (0, 2, 2, 0),
            (1, 1, 1, 1),
            (1, 1, 1, 3),
            (1, 1, 3, 1),
            (1, 1, 3, 3),
            (1, 3, 1, 1),
            (1, 3, 1, 3),
            (1, 3, 3, 1),
            (1, 3, 3, 3),
        ),
    ),
    "rocksalt": (
        2,
        (
            (0, 0, 0, 0),
            (0, 0, 1, 1),
            (0, 1, 0, 1),
            (0, 1, 1, 0),
            (1, 0, 0, 1),
            (1, 0, 1, 0),
            (1, 1, 0, 0),
            (1, 1, 1, 1),
        ),
    ),
    "zincblende": (
        4,
        (
            (0, 0, 0, 0),
            (0, 0, 2, 2),
            (0, 2, 0, 2),
            (0, 2, 2, 0),
            (1, 1, 1, 1),
            (1, 3, 3, 1),
            (1, 3, 1, 3),
            (1, 1, 3, 3),
        ),
    ),
}


class UnitCell:
    """
    Helper class for the managing a unit cell and the types of each atom.
    Atom positions are given as fractional coordinates. Types start at 1
    """

    __slots__ = [
        "__a0",
        "__conventional",
        "__ideal_bond_lengths",
        "__primitive",
        "__radius",
        "__ratio",
        "__rational_basis",
        "__reciprocal",
        "__type_map",
        "__unit_cell",
    ]

    # TODO: Basis might be needed for more complicated structures.
    def __init__(self):
        self.__unit_cell = []
        self.__primitive = np.zeros((3, 3))
        self.__a0 = 1.0
        self.__radius = 0.0
        self.__reciprocal = np.zeros((3, 3))
        self.__ideal_bond_lengths = {}
        self.__ratio = {1: 1}
        self.__type_map = {}
        self.__rational_basis = None

    def init_by_structure(
            self, structure: str, a0: float, atoms: str | Sequence[str],
            type_map: dict[str, int] | dict[int, str] = None) -> None:
        """
        Initialize the UnitCell by crystal structure.

        :param structure: The name of the crystal structure. Currently limited to fcc,
            bcc, sc, diamond, fluorite, rocksalt, and zincblende. Other structures can
            be added upon request.
        :param a0: The lattice parameter in Angstroms.
        :param atoms: Ordered atom types required by the selected structure. A single
            string is required for monoatomic structures; a two-item sequence is
            required for fluorite, rocksalt, and zincblende.
        :param type_map: Optional. Sets the type mapping for the atoms in the unit cell.
            Note that the mapping requires sequential values starting from 1. Optional,
            defaults to setting the atom types based on the order of their appearance in
            "atoms."
        :return: None.
        :raises UnitCellTypeError: If structure or atoms have invalid types.
        :raises UnitCellValueError: If the number of atom types does not match the
            selected structure.
        :raises AtomValueError: If an atom type is not a recognized element symbol.
        :raises NotImplementedError: Exception raised if the specified structure has not
            been implemented.
        """
        if not isinstance(structure, str):
            raise UnitCellTypeError("The crystal structure must be specified as a string.")

        try:
            denominator, basis_spec = _RATIONAL_BASIS_SPECS[structure]
        except KeyError as exc:
            raise NotImplementedError(
                f"Lattice structure {structure} not recognized/implemented"
            ) from exc

        if isinstance(atoms, str):
            atoms = (atoms,)
        elif isinstance(atoms, Sequence) and not isinstance(atoms, bytes):
            atoms = tuple(atoms)
        else:
            raise UnitCellTypeError(
                "atoms must be a species string or an ordered sequence of strings."
            )
        if any(not isinstance(atom, str) for atom in atoms):
            raise UnitCellTypeError("Every atom type must be specified as a string.")

        required_atom_count = max(entry[0] for entry in basis_spec) + 1
        if len(atoms) != required_atom_count:
            suffix = "s" if required_atom_count != 1 else ""
            raise UnitCellValueError(
                f"The {structure!r} crystal structure requires exactly "
                f"{required_atom_count} atom type{suffix}."
            )

        rational_basis = RationalBasis(
            names=tuple(atoms[entry[0]] for entry in basis_spec),
            numerators=np.array([entry[1:] for entry in basis_spec], dtype=object),
            denominator=denominator,
        )
        unit_cell = [
            Atom(
                name,
                *(float(value) / denominator for value in row),
            )
            for name, row in zip(
                rational_basis.names,
                rational_basis.numerators,
            )
        ]

        self.__a0 = a0
        if type_map is None:
            unique_atoms = []
            seen = set()
            for atom in atoms:
                if atom not in seen:
                    seen.add(atom)
                    unique_atoms.append(atom)
            self.type_map = {
                atom_type: i + 1 for i, atom_type in enumerate(unique_atoms)
            }
        else:
            self.type_map = type_map
        self.__conventional = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ]
        )
        self.__ratio = {1: 1}

        if structure == "fcc":
            self.__radius = math.sqrt(2) * 0.25
            self.__primitive = np.array(
                [
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0]
                ]
            )
            self.__ideal_bond_lengths = {
                (1, 1): unit_cell[0].position.distance(unit_cell[1].position)
            }
        elif structure == "bcc":
            self.__radius = math.sqrt(3) * 0.25
            self.__primitive = np.array(
                [
                    [1.0, 1.0, -1.0],
                    [1.0, -1.0, 1.0],
                    [-1.0, 1.0, 1.0]
                ]
            )
            self.__ideal_bond_lengths = {
                (1, 1): unit_cell[0].position.distance(unit_cell[1].position)
            }
        elif structure == "sc":
            self.__radius = 0.5
            # multiply by 2 here since we multiply by half the lattice parameter later
            self.__primitive = 2 * np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ]
            )
            self.__ideal_bond_lengths = {(1, 1): self.__a0}
        elif structure == "diamond":
            self.__radius = math.sqrt(3) * 0.125
            self.__primitive = np.array(
                [
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0]
                ]
            )
            self.__ideal_bond_lengths = {
                (1, 1): unit_cell[0].position.distance(unit_cell[4].position)
            }
        elif structure == "fluorite":
            self.__radius = math.sqrt(3) * 0.125
            self.__primitive = np.array(
                [
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0]
                ]
            )
            self.__ideal_bond_lengths = {
                (1, 1): unit_cell[0].position.distance(unit_cell[1].position),
                (1, 2): unit_cell[0].position.distance(unit_cell[4].position),
                (2, 2): unit_cell[4].position.distance(unit_cell[5].position)
            }
            self.__ratio = {1: 1, 2: 2}
        elif structure == "rocksalt":
            self.__radius = 0.25
            self.__primitive = np.array(
                [
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0]
                ]
            )
            self.__ideal_bond_lengths = {
                (1, 1): unit_cell[0].position.distance(unit_cell[1].position),
                (1, 2): unit_cell[0].position.distance(unit_cell[4].position),
                (2, 2): unit_cell[4].position.distance(unit_cell[5].position)
            }
            self.__ratio = {1: 1, 2: 1}
        elif structure == "zincblende":
            self.__radius = math.sqrt(3) * 0.125
            self.__primitive = np.array(
                [
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0]
                ]
            )
            self.__ideal_bond_lengths = {
                (1, 1): unit_cell[0].position.distance(unit_cell[1].position),
                (1, 2): unit_cell[0].position.distance(unit_cell[4].position),
                (2, 2): unit_cell[4].position.distance(unit_cell[6].position)
            }
            self.__ratio = {1: 1, 2: 1}
        else:
            raise UnitCellRuntimeError(
                "The validated rational-basis specification has no matching "
                f"structure initialization branch: {structure!r}."
            )
        for i in range(len(unit_cell)):
            unit_cell[i]["position"] *= a0
        self.__radius *= self.__a0
        self.__primitive *= self.__a0 / 2.0
        self.__conventional *= self.__a0
        vol = np.dot(self.__primitive[0], np.cross(
            self.__primitive[1], self.__primitive[2]))
        self.__reciprocal = np.array(
            [
                np.cross(
                    self.__primitive[(i+1) % 3],
                    self.__primitive[(i+2) % 3]
                )
                for i in range(len(self.__primitive))
            ]
        ) / vol
        self.__ideal_bond_lengths = {
            key: value * self.__a0 for key, value in self.__ideal_bond_lengths.items()
        }
        self.__unit_cell = unit_cell
        self.__rational_basis = rational_basis

    def init_by_custom(self, unit_cell: np.ndarray,
                       unit_cell_types: str | Sequence[str], a0: float,
                       conventional: np.ndarray, reciprocal: np.ndarray,
                       ideal_bond_lengths: dict, ratio: dict[int, int] = {1: 1},
                       type_map: dict[int, str] | dict[str, int] = None) -> None:
        """
        Initialize the UnitCell with a custom-built lattice.

        :param unit_cell: The fractional coordinates of the atom positions in the unit
            cell.
        :param unit_cell_types: Either an int (all atoms have the same type) or a
            Sequence (list, tuple) defining the types of the atoms in the unit cell. The
            atom types are assigned to the atoms in the same order given in the unit
            cell.
        :param a0: The lattice parameter in Angstroms.
        :param conventional: The conventional lattice vectors of the lattice. Requires a
            (3,3) shape.
        :param reciprocal: The reciprocal lattice vectors of the lattice. Requires a
            (3,3) shape.
        :param ideal_bond_lengths: The ideal bond lengths of the system. One bond length
            for each unique pair of atoms is needed.
        :param ratio: The ratio of atoms in the system. Must be a dict of ints.
            Optional, defaults to {1: 1}.
        :param type_map: Optional. Sets the type mapping for the atoms in the unit cell.
            Note that the mapping requires sequential values starting from 1. Optional,
            defaults to setting the atom types based on the order of their appearance in
            "unit_cell_types."
        :raises UnitCellValueError: Exception raised when the reciprocal shape is not
            (3,3).
        """
        self.__a0 = a0
        if len(unit_cell_types) != len(unit_cell):
            raise UnitCellValueError(
                "Length mismatch between unit cell types and unit cell")
        else:
            cell_types = unit_cell_types

        if not isinstance(ratio, dict):
            raise UnitCellTypeError(
                "The 'ratio' dict must be a dict of positive ints."
            )
        for key, val in ratio.items():
            if not isinstance(key, int) or not isinstance(val, int):
                raise UnitCellTypeError(
                    "Key and value in the 'ratio' dict must be positive ints"
                )
            if key < 1 or val < 1:
                raise UnitCellValueError(
                    "Key and value in the 'ratio' duct must be positive ints"
                )
        ratio_sum = sum([val for val in ratio.values()])
        if not len(unit_cell) % ratio_sum == 0:
            raise UnitCellValueError(
                f"The number of atoms ({len(unit_cell)}) in the UnitCell must be an "
                f"integer multiple of the ratio ({ratio_sum})."
            )

        self.__unit_cell = [
            Atom(t, x, y, z)
            for (t, (x, y, z)) in zip(cell_types, unit_cell)
        ]
        self.__rational_basis = None

        if not isinstance(conventional, np.ndarray):
            conventional = np.array(reciprocal)
        if conventional.shape != (3, 3):
            raise UnitCellValueError(
                "Incorrect shape for conventional vectors. Must be (3,3)")
        self.__conventional = conventional

        if not isinstance(reciprocal, np.ndarray):
            reciprocal = np.array(reciprocal)
        if reciprocal.shape != (3, 3):
            raise UnitCellValueError(
                "Incorrect shape for reciprocal vectors. Must be (3,3)")
        self.__reciprocal = reciprocal

        self.__ideal_bond_lengths = ideal_bond_lengths

        self.__ratio = ratio

        if type_map is None:
            self.type_map = {
                name: i + 1
                for i, name in enumerate(dict.fromkeys(unit_cell_types))
            }
        else:
            self.type_map = type_map

    def positions(self) -> np.ndarray:
        """Returns the positions of the atoms in the UnitCell."""
        return self.__a0 * np.vstack([a.position.asarray() for a in self.__unit_cell])

    def names(self, *, asint=False) -> np.ndarray:
        """Returns an array containing the types of atoms in the UnitCell."""
        if asint:
            names = [a.name for a in self.__unit_cell]
            return np.hstack([self.type_map[name] for name in names], dtype=int)
        else:
            return np.hstack([a.name for a in self.__unit_cell])

    def types(self) -> np.ndarray:
        """
        Returns an array assigning a "type" number to each unique atom type.
        """
        names = np.array([a["name"] for a in self.__unit_cell])
        converted_names = np.array([self.type_map[name] for name in names])

        return converted_names

    @property
    def reciprocal(self) -> np.ndarray:
        """Returns the reciprocal lattice for the defined UnitCell."""
        return self.__reciprocal

    @property
    def a0(self) -> float:
        """Returns the lattice parameter for the defined UnitCell."""
        return self.__a0

    @a0.setter
    def a0(self, value: float) -> None:
        if value <= 0:
            raise UnitCellValueError("Invalid value for a0. Must be > 0.")
        self.__primitive /= self.__a0 / 2.0
        self.__conventional /= self.__a0
        self.__radius /= self.__a0
        self.__ideal_bond_lengths = {
            key: value / self.__a0 for key, value in self.__ideal_bond_lengths.items()
        }
        self.__a0 = value
        self.__primitive *= self.__a0 / 2.0
        self.__conventional *= self.a0
        self.__radius *= self.__a0
        vol = np.dot(self.__primitive[0], np.cross(
            self.__primitive[1], self.__primitive[2]))
        self.__reciprocal = np.array(
            [
                np.cross(
                    self.__primitive[(i+1) % 3],
                    self.__primitive[(i+2) % 3]
                )
                for i in range(len(self.__primitive))
            ]
        ) / vol
        self.__ideal_bond_lengths = {
            key: value * self.__a0 for key, value in self.__ideal_bond_lengths.items()
        }

    @property
    def radius(self) -> float:
        return self.__radius

    @property
    def unit_cell(self) -> np.ndarray:
        return self.__unit_cell

    @property
    def primitive(self) -> np.ndarray:
        return self.__primitive

    @property
    def conventional(self) -> np.ndarray:
        return self.__conventional

    @property
    def rational_basis(self) -> RationalBasis | None:
        """Return exact built-in basis metadata, when authoritatively defined.

        Built-in structures provide exact rational metadata. Custom and
        uninitialized cells return ``None`` because arbitrary floating-point
        coordinates are not rationalized approximately.

        :return: Immutable exact basis metadata for a built-in structure, or ``None``
            for a custom or uninitialized cell.
        """
        return self.__rational_basis

    def asarray(self) -> np.ndarray:
        """
        Gives the unit cell as a structured numpy array with the atom type and position.

        :return: Structured numpy array with atom type and position.
        """
        return np.array(
            [tuple(atom["name", "x", "y", "z"]) for atom in self.__unit_cell],
            dtype=Atom.atom_dtype
        )

    @property
    def ideal_bond_lengths(self) -> dict:
        """Returns the ideal bond lengths for the defined UnitCell."""
        return self.__ideal_bond_lengths

    @property
    def ratio(self) -> dict[int, int]:
        """Returns the ratios of the different atom types in the defined UnitCell."""
        return self.__ratio

    @property
    def type_map(self) -> dict[str, int]:
        """
        Returns the string-to-int map that is used to assign types/strings to each
        atom.
        """
        return self.__type_map

    @type_map.setter
    def type_map(self, value: dict[str, int] | dict[int, str]) -> None:
        """
        Sets the string-to-int map that is used to assign types/strings to each atom.
        """
        if not isinstance(value, dict):
            raise UnitCellTypeError(
                "type_map must be a dict with an int or str as the key, and a str or "
                "int as the value, respectively. Example: {1: 'H', 2: 'He'} or "
                "{'H': 1, 'He': 2}"
            )
        if (
            all([isinstance(key, str) for key in value.keys()]) and
            all([isinstance(val, int) for val in value.values()])
        ):
            if not min(value.values()) == 1:
                raise UnitCellValueError(
                    "The minimum integer value for the type map must be equal to 1.")
            if not all(
                [val1 + 1 == val2
                 for val1, val2 in zip(
                     sorted(list(value.values()))[:-1], sorted(list(value.values()))[1:]
                 )
                 ]
            ):
                raise UnitCellValueError(
                    "Integer values must be sequential starting from 1."
                )
            self.__type_map = value
        elif (
            all([isinstance(key, int) for key in value.keys()]) and
            all([isinstance(val, str) for val in value.values()])
        ):
            if not min(value.keys()) == 1:
                raise UnitCellValueError(
                    "The minimum integer value for the type map must be equal to 1.")
            if not all(
                [val1 + 1 == val2
                 for val1, val2 in zip(
                     sorted(list(value.keys()))[:-1], sorted(list(value.keys()))[1:]
                 )
                 ]
            ):
                raise UnitCellValueError(
                    "Integer values must be sequential starting from 1."
                )
            self.__type_map = {val: key for key, val in value.items()}
        else:
            raise UnitCellTypeError(
                "type_map must be a dict with an int or str as the key, and a str or "
                "int as the value, respectively. Example: {1: 'H', 2: 'He'} or "
                "{'H': 1, 'He': 2}"
            )

    def nn_distance(self, nn, atom_type=None, *, max_attempts: int = 10) -> float:
        def generate_atom_sphere(rcut: float) -> np.ndarray:
            max_repeats = int(
                np.ceil(rcut / np.min(np.linalg.norm(self.__conventional, axis=1)))) + 1
            range_vals = np.arange(-max_repeats, max_repeats + 1)
            grid = np.array(np.meshgrid(range_vals, range_vals,
                            range_vals, indexing="ij")).reshape(3, -1).T

            cell_origins = grid @ self.__conventional
            basis = self.positions()
            types = self.names(asint=True)
            supercell = (cell_origins[:, np.newaxis, :] +
                         basis[np.newaxis, :, :]).reshape(-1, 3)
            types = np.tile(types, cell_origins.shape[0])

            distances = np.linalg.norm(supercell - basis[0], axis=1)
            mask = distances <= rcut
            return supercell[mask], types[mask]

        rcut = self.__a0
        step = self.__a0 / 2
        if isinstance(atom_type, str):
            atom_type = self.__type_map[atom_type]
        if atom_type is None:
            ref_pos = self.positions()[0]
        else:
            ref_pos = self.positions()[np.argmax(self.names(asint=True) == atom_type)]
        for _ in range(max_attempts):
            # pdb.set_trace()
            supercell, supercell_types = generate_atom_sphere(rcut)
            if atom_type is not None:
                mask = supercell_types == atom_type
                supercell = supercell[mask]
                supercell_types = supercell_types[mask]
            kdtree = KDTree(supercell)
            distances, indices = kdtree.query(ref_pos, k=len(supercell))
            # We round everything to reduce floating point errors, and we only take the
            # distances that are non-zero (the first one is the self-distance)
            distances = np.round(distances[1:], decimals=8)

            unique_dists = np.unique(distances)
            if len(unique_dists) >= nn + 1:
                return unique_dists[nn - 1]

            rcut += step
        raise UnitCellRuntimeError(
            f"Could not find {nn} unique neighbor distances within {rcut=}")

    def __repr__(self):
        structure_info = f"UnitCell with {len(self.__unit_cell)} " + \
            f"atom{"s" if len(self.__unit_cell) != 1 else ""}"
        lattice_info = f"Lattice parameter (a0): {self.__a0:.3f} Å"
        radius_info = f"Radius: {self.__radius:.3f} Å"
        atom_info = ", ".join(
            [f"'{atom.atom_name}': {atom.position.x:.3f}, {atom.position.y:.3f}, {atom.position.z:.3f}"
             for atom in self.__unit_cell]
        )
        reciprocal_info = f"Reciprocal lattice:\n{self.__reciprocal}"

        return (f"{structure_info}\n"
                f"{lattice_info}\n"
                f"{radius_info}\n"
                f"Atoms: [{atom_info}]\n"
                f"{reciprocal_info}")
