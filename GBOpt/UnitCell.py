import math
from typing import Sequence, Tuple, Union

import numpy as np

from GBOpt import Atom


class UnitCellError(Exception):
    """Base class for exceptions in the UnitCell class"""
    pass


class UnitCellValueError(UnitCellError):
    """Exceptions raised when an incorrect value is assigned to a UnitCell attribute."""
    pass


class UnitCellTypeError(UnitCellError):
    """Exceptions raised when an incorrect type is assigned to a UnitCell attribute."""
    pass


class UnitCell:
    """
    Helper class for the managing a unit cell and the types of each atom.
    Atom positions are given as fractional coordinates. Types start at 1
    """

    __slots__ = ['__unit_cell', '__primitive', '__a0',
                 '__radius', '__reciprocal', '__ideal_bond_lengths']

    # TODO: Basis might be needed for more complicated structures.
    def __init__(self):
        self.__unit_cell = []
        self.__primitive = np.zeros((3, 3))
        self.__a0 = 1.0
        self.__radius = 0.0
        self.__reciprocal = np.zeros((3, 3))
        self.__ideal_bond_lengths = {}

    def init_by_structure(self, structure: str, a0: float, atoms: Union[str, Tuple[str, ...]]) -> None:
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
        if not isinstance(atoms, tuple) and not isinstance(atoms, list):
            atoms = (atoms,)
        if structure == 'fcc':
            unit_cell = [
                Atom(atoms[0], 0.0, 0.0, 0.0),
                Atom(atoms[0], 0.0, 0.5, 0.5),
                Atom(atoms[0], 0.5, 0.0, 0.5),
                Atom(atoms[0], 0.5, 0.5, 0.0)
            ]
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
        elif structure == 'bcc':
            unit_cell = [
                Atom(atoms[0], 0.0, 0.0, 0.0),
                Atom(atoms[0], 0.5, 0.5, 0.5)
            ]
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
        elif structure == 'sc':
            unit_cell = [Atom(atoms[0], 0.0, 0.0, 0.0)]
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
        elif structure == 'diamond':
            unit_cell = [
                Atom(atoms[0], 0, 0, 0),
                Atom(atoms[0], 0, 0.5, 0.5),
                Atom(atoms[0], 0.5, 0, 0.5),
                Atom(atoms[0], 0.5, 0.5, 0),
                Atom(atoms[0], 0.25, 0.25, 0.25),
                Atom(atoms[0], 0.75, 0.75, 0.25),
                Atom(atoms[0], 0.75, 0.25, 0.75),
                Atom(atoms[0], 0.25, 0.75, 0.75)
            ]
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
        elif structure == 'fluorite':
            if len(atoms) != 2:
                raise UnitCellValueError(
                    "The specified crystal structure requires 2 atom types.")
            unit_cell = [
                Atom(atoms[0], 0, 0, 0),
                Atom(atoms[0], 0, 0.5, 0.5),
                Atom(atoms[0], 0.5, 0, 0.5),
                Atom(atoms[0], 0.5, 0.5, 0),
                Atom(atoms[1], 0.25, 0.25, 0.25),
                Atom(atoms[1], 0.25, 0.25, 0.75),
                Atom(atoms[1], 0.25, 0.75, 0.25),
                Atom(atoms[1], 0.25, 0.75, 0.75),
                Atom(atoms[1], 0.75, 0.25, 0.25),
                Atom(atoms[1], 0.75, 0.25, 0.75),
                Atom(atoms[1], 0.75, 0.75, 0.25),
                Atom(atoms[1], 0.75, 0.75, 0.75)
            ]
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
        elif structure == 'rocksalt':
            if len(atoms) != 2:
                raise UnitCellValueError(
                    "The specified crystal structure requires 2 atom types.")

            unit_cell = [
                Atom(atoms[0], 0, 0, 0),
                Atom(atoms[0], 0, 0.5, 0.5),
                Atom(atoms[0], 0.5, 0, 0.5),
                Atom(atoms[0], 0.5, 0.5, 0),
                Atom(atoms[1], 0, 0, 0.5),
                Atom(atoms[1], 0, 0.5, 0),
                Atom(atoms[1], 0.5, 0, 0),
                Atom(atoms[1], 0.5, 0.5, 0.5)
            ]
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
        elif structure == 'zincblende':
            if len(atoms) != 2:
                raise UnitCellValueError(
                    "The specified crystal structure requires 2 atom types.")
            unit_cell = [
                Atom(atoms[0], 0, 0, 0),
                Atom(atoms[0], 0, 0.5, 0.5),
                Atom(atoms[0], 0.5, 0, 0.5),
                Atom(atoms[0], 0.5, 0.5, 0),
                Atom(atoms[1], 0.25, 0.25, 0.25),
                Atom(atoms[1], 0.75, 0.75, 0.25),
                Atom(atoms[1], 0.75, 0.25, 0.75),
                Atom(atoms[1], 0.25, 0.75, 0.75)
            ]
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
        else:
            raise NotImplementedError(
                f"Lattice structure {structure} not recognized/implemented")
        for i in range(len(unit_cell)):
            unit_cell[i]['position'] *= a0
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
        self.__ideal_bond_lengths = {
            key: value * self.__a0 for key, value in self.__ideal_bond_lengths.items()
        }

    def init_by_custom(self, unit_cell: np.ndarray,
                       unit_cell_types: str | Sequence[str], a0: float,
                       reciprocal: np.ndarray, ideal_bond_lengths: dict) -> None:
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
        :param ideal_bond_lengths: The ideal bond lengths of the system. One bond length
            for each unique pair of atoms is needed.
        :raises UnitCellValueError: Exception raised when the reciprocal shape is not
            (3,3).
        """
        self.__a0 = a0
        if len(unit_cell_types) != len(unit_cell):
            raise UnitCellValueError(
                "Length mismatch between unit cell types and unit cell")
        else:
            cell_types = unit_cell_types

        self.__unit_cell = [
            Atom(t, x, y, z)
            for (t, (x, y, z)) in zip(cell_types, unit_cell)
        ]
        if not isinstance(reciprocal, np.ndarray):
            reciprocal = np.array(reciprocal)
        if reciprocal.shape != (3, 3):
            raise UnitCellValueError(
                "Incorrect shape for reciprocal vectors. Must be (3,3)")
        self.__reciprocal = reciprocal

        self.__ideal_bond_lengths = ideal_bond_lengths

    def positions(self) -> np.ndarray:
        """Returns the positions of the atoms in the UnitCell."""
        return self.__a0 * np.vstack([a.position.asarray() for a in self.__unit_cell])

    def names(self, *, asint=False) -> np.ndarray:
        """Returns an array containing the types of atoms in the UnitCell."""
        if asint:
            names = [a.name for a in self.__unit_cell]
            name_int_dict = {name: idx + 1 for idx, name in enumerate(set(names))}
            return np.hstack([name_int_dict[name] for name in names], dtype=int)
        else:
            return np.hstack([a.name for a in self.__unit_cell])

    def types(self) -> np.ndarray:
        """
        Returns an array assigning a 'type' number to each unique atom type.
        """
        names = np.array([a['name'] for a in self.__unit_cell])
        _, converted_names = np.unique(names, return_inverse=True)
        return converted_names + 1  # starts the indexing from 1

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
        self.__radius /= self.__a0
        self.__ideal_bond_lengths = {
            key: value / self.__a0 for key, value in self.__ideal_bond_lengths.items()
        }
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

    def asarray(self) -> np.ndarray:
        """
        Gives the unit cell as a structured numpy array with the atom type and position.

        :return: Structured numpy array with atom type and position.
        """
        return np.array(
            [tuple(atom['name', 'x', 'y', 'z']) for atom in self.__unit_cell],
            dtype=Atom.atom_dtype
        )

    @property
    def ideal_bond_lengths(self) -> dict:
        """Returns the ideal bond lengths for the defined UnitCell."""
        return self.__ideal_bond_lengths

    def __repr__(self):
        structure_info = f"UnitCell with {len(self.__unit_cell)} " + \
            f"atom{'s' if len(self.__unit_cell) != 1 else ''}"
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
