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

    def init_by_structure(self, structure: str, a0: float) -> None:
        """
        Initialize the UnitCell by the crystal structure.

        :param structure: The name of the crystal structure. Currently limited to fcc,
            bcc, sc, diamond, fluorite, rocksalt, and zincblende. Other structures can
            be added upon request.
        :param a0: The lattice parameter in Angstroms.
        TODO: Figure out why the LLM used the [1,1,1] vector, as this is not a
            reciprocal lattice vector
        """
        self.a0 = a0
        if structure == 'fcc':
            unit_cell = [
                Atom(1, 1, 0.0, 0.0, 0.0),
                Atom(2, 1, 0.0, 0.5, 0.5),
                Atom(3, 1, 0.5, 0.0, 0.5),
                Atom(4, 1, 0.5, 0.5, 0.0)
            ]
            self.radius = math.sqrt(2) * 0.25
            self.reciprocal = np.array(
                [
                    [1.0, 1.0, 1.0],
                    [-1.0, 1.0, 1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, 1.0, -1.0]
                ]
            )
        elif structure == 'bcc':
            unit_cell = [
                Atom(1, 1, 0.0, 0.0, 0.0),
                Atom(2, 1, 0.5, 0.5, 0.5)
            ]
            self.radius = math.sqrt(3) * 0.25
            self.reciprocal = np.array(
                [
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0]
                ]
            )
        elif structure == 'sc':
            unit_cell = [Atom(1, 1, 0.0, 0.0, 0.0)]
            self.radius = 0.5
            self.reciprocal = np.array(
                [
                    [1.0, 1.0, 1.0],
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
            self.radius = math.sqrt(3) * 0.125
            self.reciprocal = np.array(
                [
                    [1.0, 1.0, 1.0],
                    [-1.0, 1.0, 1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, 1.0, -1.0]
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
            self.radius = math.sqrt(3) * 0.125
            self.reciprocal = np.array(
                [
                    [1.0, 1.0, 1.0],
                    [-1.0, 1.0, 1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, 1.0, -1.0]
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
            self.radius = 0.25
            self.reciprocal = np.array(
                [
                    [1.0, 1.0, 1.0],
                    [-1.0, 1.0, 1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, 1.0, -1.0]
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
            self.radius = math.sqrt(3) * 0.125
            self.reciprocal = np.array(
                [
                    [1.0, 1.0, 1.0],
                    [-1.0, 1.0, 1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, 1.0, -1.0]
                ]
            )
        else:
            raise NotImplementedError(
                f"Lattice structure {structure} not recognized/implemented")
        self.unit_cell = unit_cell
        self.radius *= self.a0
        self.reciprocal /= self.a0

    def init_by_custom(
        self,
        unit_cell: np.ndarray,
        unit_cell_types: int | Sequence[numbers.Number],
        a0: float,
        reciprocal: np.ndarray
    ) -> None:
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
            (4,3) shape, with the first vector equal to [1,1,1].
        """
        self.a0 = a0
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

        self.unit_cell = [
            Atom(i+1, t, x, y, z)
            for i, (t, (x, y, z)) in enumerate(zip(cell_types, unit_cell))
        ]
        if reciprocal.shape != (4, 3):
            raise UnitCellValueError(
                "Incorrect shape for reciprocal vectors. Must be (4,3)")
        self.reciprocal = reciprocal

    def positions(self):
        """Returns the positions of the atoms in the UnitCell."""
        return self.a0 * np.vstack([[a.position.x, a.position.y, a.position.z] for a in self.unit_cell])

    def types(self):
        """Returns an array containing the types of atoms in the UnitCell."""
        return np.hstack([a.atom_type for a in self.unit_cell])

    def reciprocal_lattice(self):
        """Returns the reciprocal lattice for the defined UnitCell."""
        return self.reciprocal
