import copy
import math
import numbers
import warnings
from typing import Sequence

import numpy as np

from GBOpt.Atom import Atom


class UnitCellError(Exception):
    """Base class for exceptions in the UnitCell class"""
    pass


class UnitCellValueError(UnitCellError):
    """Exceptions raised in the UnitCell class when an incorrect value is given."""
    pass


class UnitCell:
    """
    Helper class for the managing a unit cell and the types of each atom.
    Atom positions are given as fractional coordinates. Types start at 1
    """

    # TODO: Basis might be needed for more complicated structures.
    def __init__(self):
        self.__unit_cell = []
        self.__primitive = np.zeros((3, 3))
        self.__a0 = 1.0
        self.__radius = 0.0
        self.__reciprocal = np.zeros((3, 3))

    def init_by_structure(self, structure: str, a0: float) -> None:
        """
        Initialize the UnitCell by crystal structure.

        :param structure: The name of the crystal structure. Currently limited to fcc,
            bcc, sc, diamond, fluorite, rocksalt, and zincblende. Other structures can
            be added upon request.
        :param a0: The lattice parameter in Angstroms.
        :raises NotImplementedError: Exception raised if the specified structure has not
            been implemented.
        """
        self.__a0 = a0
        if structure == 'fcc':
            unit_cell = [
                Atom(1, 1, 0.0, 0.0, 0.0),
                Atom(2, 1, 0.0, 0.5, 0.5),
                Atom(3, 1, 0.5, 0.0, 0.5),
                Atom(4, 1, 0.5, 0.5, 0.0)
            ]
            self.__radius = math.sqrt(2) * 0.25
            self.__primitive = np.array(
                [
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0]
                ]
            )
        elif structure == 'bcc':
            unit_cell = [
                Atom(1, 1, 0.0, 0.0, 0.0),
                Atom(2, 1, 0.5, 0.5, 0.5)
            ]
            self.__radius = math.sqrt(3) * 0.25
            self.__primitive = np.array(
                [
                    [1.0, 1.0, -1.0],
                    [1.0, -1.0, 1.0],
                    [-1.0, 1.0, 1.0]
                ]
            )
        elif structure == 'sc':
            unit_cell = [Atom(1, 1, 0.0, 0.0, 0.0)]
            self.__radius = 0.5
            # multiply by 2 here since we multiply by half the lattice parameter later
            self.__primitive = 2 * np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ]
            )
        elif structure == 'diamond':
            unit_cell = [
                Atom(1, 1, 0, 0, 0),
                Atom(2, 1, 0, 0.5, 0.5),
                Atom(3, 1, 0.5, 0, 0.5),
                Atom(4, 1, 0.5, 0.5, 0),
                Atom(5, 1, 0.25, 0.25, 0.25),
                Atom(6, 1, 0.75, 0.75, 0.25),
                Atom(7, 1, 0.75, 0.25, 0.75),
                Atom(8, 1, 0.25, 0.75, 0.75)
            ]
            self.__radius = math.sqrt(3) * 0.125
            self.__primitive = np.array(
                [
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0]
                ]
            )
        elif structure == 'fluorite':
            unit_cell = [
                Atom(1, 1, 0, 0, 0),
                Atom(2, 1, 0, 0.5, 0.5),
                Atom(3, 1, 0.5, 0, 0.5),
                Atom(4, 1, 0.5, 0.5, 0),
                Atom(5, 2, 0.25, 0.25, 0.25),
                Atom(6, 2, 0.25, 0.25, 0.75),
                Atom(7, 2, 0.25, 0.75, 0.25),
                Atom(8, 2, 0.25, 0.75, 0.75),
                Atom(9, 2, 0.75, 0.25, 0.25),
                Atom(10, 2, 0.75, 0.25, 0.75),
                Atom(11, 2, 0.75, 0.75, 0.25),
                Atom(12, 2, 0.75, 0.75, 0.75)
            ]
            self.__radius = math.sqrt(3) * 0.125
            self.__primitive = np.array(
                [
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0]
                ]
            )
        elif structure == 'rocksalt':
            unit_cell = [
                Atom(1, 1, 0, 0, 0),
                Atom(2, 1, 0, 0.5, 0.5),
                Atom(3, 1, 0.5, 0, 0.5),
                Atom(4, 1, 0.5, 0.5, 0),
                Atom(5, 2, 0, 0, 0.5),
                Atom(6, 2, 0, 0.5, 0),
                Atom(7, 2, 0.5, 0, 0),
                Atom(8, 2, 0.5, 0.5, 0.5)
            ]
            self.__radius = 0.25
            self.__primitive = np.array(
                [
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0]
                ]
            )
        elif structure == 'zincblende':
            unit_cell = [
                Atom(1, 1, 0, 0, 0),
                Atom(2, 1, 0, 0.5, 0.5),
                Atom(3, 1, 0.5, 0, 0.5),
                Atom(4, 1, 0.5, 0.5, 0),
                Atom(5, 2, 0.25, 0.25, 0.25),
                Atom(6, 2, 0.75, 0.75, 0.25),
                Atom(7, 2, 0.75, 0.25, 0.75),
                Atom(8, 2, 0.25, 0.75, 0.75)
            ]
            self.__radius = math.sqrt(3) * 0.125
            self.__primitive = np.array(
                [
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0]
                ]
            )
        else:
            raise NotImplementedError(
                f"Lattice structure {structure} not recognized/implemented")
        self.__unit_cell = unit_cell
        self.__radius *= self.__a0
        self.__primitive *= self.__a0 / 2.0
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

    def init_by_custom(self, unit_cell: np.ndarray,
                       unit_cell_types: int | Sequence[numbers.Number], a0: float,
                       reciprocal: np.ndarray) -> None:
        """
        Initialize the UnitCell with a custom-built lattice.

        :param unit_cell: The fractional coordinates of the atom positions in the unit
            cell.
        :param unit_cell_types: Either an int (all atoms have the same type) or a
            Sequence (list, tuple) defining the types of the atoms in the unit cell. The
            atom types are assigned to the atoms in the same order given in the unit
            cell.
        :param a0: The lattice parameter in Angstroms.
        :param reciprocal: The reciprocal lattice vectors of the lattice. Requires a
            (3,3) shape.
        :raises UnitCellValueError: Exception raised when the reciprocal shape is not
            (3,3).
        """
        self.__a0 = a0
        if isinstance(unit_cell_types, int):
            if unit_cell_types != 1:
                warnings.warn("All types set to 1.")
            cell_types = np.ones(len(unit_cell), dtype=int)
        else:
            min_types = min(unit_cell_types)
            if min_types != 1:
                warnings.warn(f"Types shifted by {-(min_types-1)}")
                cell_types = [uct - (min_types - 1) for uct in unit_cell_types]
            else:
                cell_types = unit_cell_types

        self.__unit_cell = [
            Atom(i+1, t, x, y, z)
            for i, (t, (x, y, z)) in enumerate(zip(cell_types, unit_cell))
        ]
        if not isinstance(reciprocal, np.ndarray):
            reciprocal = np.array(reciprocal)
        if reciprocal.shape != (3, 3):
            raise UnitCellValueError(
                "Incorrect shape for reciprocal vectors. Must be (3,3)")
        self.__reciprocal = reciprocal

    def positions(self) -> np.ndarray:
        """Returns the positions of the atoms in the UnitCell."""
        return self.__a0 * np.vstack([[a.position.x, a.position.y, a.position.z] for a in self.__unit_cell])

    def types(self) -> np.ndarray:
        """Returns an array containing the types of atoms in the UnitCell."""
        return np.hstack([a.atom_type for a in self.__unit_cell])

    @property
    def reciprocal(self) -> np.ndarray:
        """Returns the reciprocal lattice for the defined UnitCell."""
        return copy.copy(self.__reciprocal)

    @property
    def a0(self) -> float:
        """Returns the lattice parameter for the defined UnitCell."""
        return self.__a0

    @a0.setter
    def a0(self, value: float) -> None:
        if value <= 0:
            raise UnitCellValueError("Invalid value for a0. Must be > 0.")
        self.__primitive /= self.__a0 / 2.0
        self.__radius /= self.__a0
        self.__a0 = value
        self.__primitive *= self.__a0 / 2.0
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

    @property
    def radius(self) -> float:
        return self.__radius

    @property
    def unit_cell(self) -> np.ndarray:
        return self.__unit_cell

    @property
    def primitive(self) -> np.ndarray:
        return self.__primitive

    def __repr__(self):
        structure_info = f"UnitCell with {len(self.__unit_cell)} " + \
            f"atom{'s' if len(self.__unit_cell) != 1 else ''}"
        lattice_info = f"Lattice parameter (a0): {self.__a0:.3f} Å"
        radius_info = f"Radius: {self.__radius:.3f} Å"
        atom_info = ", ".join(
            [f"{atom.id} ({atom.atom_type}): {atom.position.x:.3f}, {atom.position.y:.3f}, {atom.position.z:.3f}"
             for atom in self.__unit_cell]
        )
        reciprocal_info = f"Reciprocal lattice:\n{self.__reciprocal}"

        return (f"{structure_info}\n"
                f"{lattice_info}\n"
                f"{radius_info}\n"
                f"Atoms: [{atom_info}]\n"
                f"{reciprocal_info}")
