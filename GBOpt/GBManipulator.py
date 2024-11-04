import math
import multiprocessing as mp
import warnings
from os.path import isfile
from typing import Union

import numpy as np
from numba import float64, jit
from scipy.spatial import Delaunay, KDTree, cKDTree

from GBOpt.Atom import Atom
from GBOpt.GBMaker import GBMaker
from GBOpt.UnitCell import UnitCell

# TODO: Generalize to interfaces, not just GBs


class GBManipulatorError(Exception):
    """Base class for exceptions in the GBManipulator class."""
    pass


class GBManipulatorValueError(GBManipulatorError):
    """
    Exception raised in the GBManipulator class when an invalid value is assigned to a
    GBManipulator attribute.
    """
    pass


class ParentError(Exception):
    """Base class for exceptions in the Parent class."""
    pass


class ParentValueError(ParentError):
    """
    Exception raised in the Parent class when an invalid value is assigned to a Parent
    attribute.
    """
    pass


class ParentFileNotFoundError(ParentError):
    """
    Exception raised in the Parent class when the snapshot file is not found.
    """
    pass


class ParentCorruptedSnapshotError(ParentError):
    """
    Exception raised in the Parent class when an error occurs while reading a snapshot.
    """
    pass


class ParentSnapshotMissingDataError(ParentError):
    """
    Exception raised when data is missing from a snapshot that is otherwise formatted
    correctly.
    """
    pass


class ParentsProxyError(Exception):
    """Base class for exceptions in the ParentsProxy class."""
    pass


class ParentsProxyValueError(ParentsProxyError):
    """
    Exception raised in the ParentsProxy class when an invalid value is assigned to a
    ParentsProxy attribute.
    """
    pass


class ParentsProxyIndexError(ParentsProxyError):
    """
    Exception raised in the ParentsProxy class when an invalid index is used. Valid
    indices are 0 and 1.
    """
    pass


class ParentsProxyTypeError(ParentsProxyError):
    """
    Exception raised in the ParentsProxy class when an invalid type is assigned to a
    ParentsProxy attribute.
    """
    pass


class Parent:
    """
    Class containing the information about a parent of a new structure.

    :param system: A GBMaker instance or string containing the filename of a LAMMPS dump
        file.
    :param unit_cell: Required only if GB is specified using a LAMMPS dump file. Gives
        the nominal unit cell of the system.
    :param gb_thickness: Thickness of the GB region, optional, defaults to 10.
    """
    __num_to_name = {val: key for key, val in Atom._numbers.items()}

    def __init__(
        self,
        system: Union[GBMaker, str],
        *,
        unit_cell: UnitCell = None,
        gb_thickness: float = 10,
        type_dict: dict = {}
    ) -> None:
        if isinstance(system, GBMaker):
            self.__init_by_gbmaker(system)
        else:
            if gb_thickness is None:  # defaults to 10 if passed in as None.
                gb_thickness = 10
            self.__init_by_snapshot(system, unit_cell, gb_thickness, type_dict)
        # self.__whole_system = np.hstack((self.__left_grain, self.__right_grain))
        left_x_max = max(self.__left_grain['x'])
        right_x_min = min(self.__right_grain['x'])
        left_cut = left_x_max - self.__gb_thickness / 2.0
        right_cut = right_x_min + self.__gb_thickness / 2.0
        left_gb_indices = self.__left_grain['x'] > left_cut
        right_gb_indices = self.__right_grain['x'] < right_cut
        left_gb = self.__left_grain[left_gb_indices]
        right_gb = self.__right_grain[right_gb_indices]
        self.__gb_atoms = np.hstack((left_gb, right_gb))
        # TODO: make this a more robust calculation, rather than assuming the GB is in the middle of the system.
        self.__GBpos = self.__whole_system[
            np.where(
                np.logical_and(
                    self.__whole_system['x'] >= self.__box_dims[0, 1] /
                    2 - self.__gb_thickness/2,
                    self.__whole_system['x'] <= self.__box_dims[0, 1] /
                    2 + self.__gb_thickness/2
                )
            )
        ]

    def __init_by_gbmaker(self, system: GBMaker) -> None:
        """
        Method for initializing the Parent using a GBMaker instance.

        :param system: The GBMaker instance.
        """
        self.__right_grain = system.right_grain
        self.__left_grain = system.left_grain
        self.__whole_system = system.whole_system
        self.__y_dim = system.y_dim
        self.__z_dim = system.z_dim
        self.__gb_thickness = system.gb_thickness
        self.__unit_cell = system.unit_cell
        self.__atom_radius = system.radius
        self.__box_dims = system.box_dims
        # We do not use GB.x_dim because this is limited to a single grain of the GB,
        # not the entire system.
        self.__x_dim = self.__box_dims[0][1] - self.__box_dims[0][0]

    def __init_by_snapshot(
        self,
        system_file: str,
        unit_cell: UnitCell,
        gb_thickness: float,
        type_dict: dict
    ) -> None:
        """
        Method for initializing the Parent using a LAMMPS dump file.

        :param system_file: Filename of the LAMMPS dump file.
        :param unit_cell: Nominal unit cell of the bulk structure.
        :param gb_thickness: Thickness of the GB region, given in angstroms.
        :param type_dict: Conversion from type number to type name, optional. Note that
            if this is not provided and the snapshot does not indicate the atom names,
            atom names are assumed started from 'H'.
        :raises ParentValueError: Exception raised if unit_cell is not passed in.
        :raises ParentFileNotFoundError: Exception raised if the specified file is not
            found.
        :raises ParentCorruptedSnapshotError: Exception raised if the dump file is not
            formatted correctly
        :raises ParentSnapshotMissingDataError: Exception raised if the dump file is
            otherwise formatted correctly, but is missing required data.
        """

        if not unit_cell:
            raise ParentValueError("Unit cell must be specified for snapshots")
        self.__unit_cell = unit_cell
        self.__gb_thickness = gb_thickness
        if not isfile(system_file):
            raise ParentFileNotFoundError(f"{system_file} does not exist.")
        skip_rows = 0
        with open(system_file) as f:
            line = f.readline()
            skip_rows += 1
            # skip to the box bounds
            while not line.startswith("ITEM: BOX BOUNDS"):
                line = f.readline()
                skip_rows += 1
                if not line:
                    raise ParentCorruptedSnapshotError(
                        f"Box bounds not found in {system_file}")
            skip_rows += 3
            if len(line.split()) == 6:  # orthogonal box
                x_dims = [float(i) for i in f.readline().split()]
                y_dims = [float(i) for i in f.readline().split()]
                z_dims = [float(i) for i in f.readline().split()]
            elif len(line.split()) == 9:  # triclinic box, restricted format
                xline = f.readline().split()
                yline = f.readline().split()
                zline = f.readline().split()
                x_dims, _ = ([float(i) for i in xline[0:2]], float(xline[2]))
                y_dims, _ = ([float(i) for i in yline[0:2]], float(yline[2]))
                z_dims, _ = ([float(i) for i in zline[0:2]], float(zline[2]))
            elif len(line.split()) == 8:  # triclinic box, general format
                xline = f.readline().split()
                yline = f.readline().split()
                zline = f.readline().split()
                origin = np.empty((3,))
                A, origin[0] = (np.array([float(i)
                                for i in xline[0:3]]), float(xline[3]))
                B, origin[1] = (np.array([float(i)
                                for i in xline[0:3]]), float(xline[3]))
                C, origin[2] = (np.array([float(i)
                                for i in xline[0:3]]), float(xline[3]))

                a = np.array([np.linalg.norm(A), 0, 0])
                Ahat = A / a[0]
                b = np.array([np.dot(B, Ahat), np.cross(Ahat, B), 0])
                AxB = np.cross(A, B)
                AxBhat = AxB/np.linalg.norm(AxB)
                c = np.array([np.dot(C, Ahat), np.dot(
                    C, np.cross(AxBhat, Ahat)), np.abs(np.dot(C, AxBhat))])

                x_dims = [origin[0], origin[0] + a[0]]
                y_dims = [origin[1], origin[1] + a[1] + b[1]]
                z_dims = [origin[2], origin[2] + c[2]]
            else:
                raise ParentCorruptedSnapshotError(
                    f"Box bounds corrupted in {system_file}")
            if not (x_dims or y_dims or z_dims) or len(x_dims) != 2 or \
                    len(y_dims) != 2 or len(z_dims) != 2:
                raise ParentCorruptedSnapshotError(
                    f"Box bounds corrupted in {system_file}")
            self.__box_dims = np.array([x_dims, y_dims, z_dims])
            self.__x_dim = x_dims[1] - x_dims[0]
            self.__y_dim = y_dims[1] - y_dims[0]
            self.__z_dim = z_dims[1] - z_dims[0]
            # TODO: Need a more robust calculation of where the GB is located. This calculation is duplicated.
            grain_cutoff = (x_dims[1] - x_dims[0]) / 2
            line = f.readline()
            skip_rows += 1
            while not line.startswith("ITEM: ATOMS"):
                line = f.readline()
                skip_rows += 1
                if not line:
                    raise ParentCorruptedSnapshotError(
                        f"Atoms not found in {system_file}")
            atom_attributes = line.split()[2:]
            required_attributes = ['type', 'x', 'y', 'z']

            if not all([i in atom_attributes for i in required_attributes]):
                raise ParentSnapshotMissingDataError(
                    f"One or more required attributes are missing.\n"
                    f"Required: {required_attributes}, "
                    f"available: {atom_attributes}")
            required_attribute_indices = {attr: atom_attributes.index(
                attr) for attr in required_attributes}

            typelabel_in_attrs = 'typelabel' in atom_attributes
            if typelabel_in_attrs:
                required_attribute_indices['typelabel'] = atom_attributes.index(
                    'typelabel')
            col_indices = [required_attribute_indices['typelabel'] if typelabel_in_attrs else required_attribute_indices['type'],
                           required_attribute_indices['x'], required_attribute_indices['y'], required_attribute_indices['z']]

            def convert_type(value):
                if typelabel_in_attrs:
                    return value
                else:
                    return self.__num_to_name[int(value)]
            max_rows = 0
            line = f.readline()  # read the next line to move the file pointer ahead.
            while not line.startswith("ITEM"):
                line = f.readline()
                max_rows += 1
                if not line:
                    break

        self.__whole_system = np.loadtxt(system_file, skiprows=skip_rows, max_rows=max_rows, converters={
            col_indices[0]: convert_type}, usecols=tuple(col_indices), dtype=Atom.atom_dtype)
        self.__left_grain = self.__whole_system[self.__whole_system['x'] < grain_cutoff]
        self.__right_grain = self.__whole_system[self.__whole_system['x']
                                                 >= grain_cutoff]

    # Getters
    @property
    def left_grain(self) -> np.ndarray:
        return self.__left_grain

    @property
    def right_grain(self) -> np.ndarray:
        return self.__right_grain

    @property
    def whole_system(self) -> np.ndarray:
        return self.__whole_system

    @property
    def gb_atoms(self) -> np.ndarray:
        return self.__gb_atoms

    @property
    def unit_cell(self) -> UnitCell:
        return self.__unit_cell

    @property
    def gb_thickness(self) -> float:
        return self.__gb_thickness

    @property
    def box_dims(self) -> np.ndarray:
        return self.__box_dims

    @property
    def x_dim(self) -> float:
        return self.__x_dim

    @property
    def y_dim(self) -> float:
        return self.__y_dim

    @property
    def z_dim(self) -> float:
        return self.__z_dim


class _ParentsProxy:
    """
    Class for allowing for access to parents in the GBManipulator class by index.

    :param manipulator: The instance of GBManipulator that the ParentsProxy class acts
        for.
    """

    def __init__(self, manipulator) -> None:
        self.__manipulator = manipulator

    def __getitem__(self, index) -> Parent:
        return self.__manipulator._GBManipulator__parents[index]

    def __setitem__(self, index, value) -> None:
        """
        Method allowing for setting the parents of the GBManipulator class by index.

        :param index: The index to assign to. Valid values are 0 and 1, and the 0th
            index must be assigned to first.
        :param value: The Parent instance to assign to the GBManipulator parents
            attribute.
        :raises ParentsProxyIndexError: Exception raised when an index other than 0 or 1
            is passed in.
        :raises ParentsProxyTypeError: Exception raised when an incorrect type is passed
            in as to the parents attribute.
        :raises ParentsProxyValueError: Exception raised when attempting to assign to
            the second parent first. As most mutators act on the first parent, assigning
            to the first value is required.
        """
        if index not in (0, 1):  # Only valid values are 0 and 1: max of 2 parents.
            raise ParentsProxyIndexError("Index out of range. Index must be 0 or 1.")

        if not (value is None or isinstance(value, Parent)):
            raise ParentsProxyTypeError("Value must be None or a instance of Parent")

        # Since most of the manipulators act on the first parent, we make sure that
        # assignments are made first to index 0.
        if index == 1 and self.__manipulator._GBManipulator__parents[0] is None:
            raise ParentsProxyValueError("parents[0] must be assigned to first.")

        parents = self.__manipulator._GBManipulator__parents[:]
        parents[index] = value
        self.__manipulator._GBManipulator__parents = parents

    def __len__(self) -> int:
        """
        Method for returning the length of the parents list. This value should always be
        2, even if no parents are assigned.

        :return: 2, the length of the parents attribute.
        """
        return len(self.__manipulator._GBManipulator__parents)


@jit(float64(float64, float64), nopython=True, cache=True)
def _gaussian(x: float, sigma: float = 0.02) -> float:
    """
    Calculates a Gaussian-smeared delta function at *x* given a standard deviation of
    *sigma*.

    :param x: Where to calculate the Gaussian-smeared delta function.
    :param sigma: Standard deviation of the Gaussian-smeared delta function, optional,
        defaults to 0.02.
    :return: Value of the Gaussian-smeared delta function at x.
    """
    prefactor = 1 / (sigma * np.sqrt(2 * np.pi))
    return prefactor * np.exp(-x * x / (2 * sigma * sigma))


@jit(nopython=True, cache=True)
def _calculate_fingerprint_vector(atom, neighs, NB, V, Btype, Delta, Rmax):
    """
    Calculates the fingerprint for *atom* as described in Lyakhov *et al.*,
    Computer Phys. Comm. 181 (2010) 1623-1632 (Eq. 4).

    :param np.ndarray atom: The atom we are calculating the fingerprint for.
    :param np.ndarray neighs: list of Atom containing the neighbors to **atom**.
    :param int NB: The number of atoms of type B neighbor to **atom**.
    :param float V: The volume of the unit cell in angstroms**3.
    :param int Btype: The type of neighbors we are interested in.
    :param float Delta: The discretization length for Rs in angstroms.
    :param float Rmax: The maximum distance from the *atom* to another atom to
        calculate the fingerprint.
    :return: The vector containing the fingerprint for *atom*.
    """
    Rs = np.arange(0, Rmax+Delta, Delta)

    fingerprint_vector = np.zeros_like(Rs)
    for idx, R in enumerate(Rs):
        local_sum = 0
        for neigh in neighs:
            if neigh[0] == Btype:
                diff = atom[1:] - neigh[1:]
                # Rij = np.linalg.norm(atom[1:] - neigh[1:])
                distance = np.sqrt(np.dot(diff, diff))
                delta = _gaussian(R-distance, 0.02)
                local_sum += delta / \
                    (4 * np.pi * distance * distance * (NB / V) * Delta)
                # pdb.set_trace()
        fingerprint_vector[idx] = local_sum - 1

    return fingerprint_vector


@jit(nopython=True, cache=True, parallel=True)
def _calculate_local_order(atom, neighs, unit_cell_types, unit_cell_a0, N, Delta, Rmax):
    """
    Calculates the local order parameter following Lyakhov *et al.*, Computer Phys.
    Comm. 181 (2010) 1623-1632 (Eq. 5).

    :param np.ndarray atom: Atom we are calculating the local order for.
    :param np.ndarray neighs: Neighbors of *atom*.
    :param np.ndarray unit_cell_types: The types of the atoms in the unit cell.
    :param float unit_cell_a0: The lattice parameter.
    :param int N: The number of atoms in the unit cell.
    :param float Delta: Bin size to calculate the fingerprint vector.
    :param float Rmax: Maximum distance from *atom* to consider as a neighbor to
        *atom* in angstroms.
    :return: The local order parameter for *atom* based on its neighbors.
    """
    local_sum = 0
    atom_types = np.unique(neighs[:, 0])
    V = unit_cell_a0**3
    prefactor = Delta / (N * (V/N)**(1/3))
    for Btype in atom_types:
        NB = np.sum(unit_cell_types == Btype)
        fingerprint = _calculate_fingerprint_vector(
            atom, neighs, NB, V, Btype, Delta, Rmax)
        local_sum += NB * prefactor * np.dot(fingerprint, fingerprint)
    return np.sqrt(local_sum)


def _create_neighbor_list(rcut: float, pos: np.ndarray) -> list:
    """
    Creates a neighbor list using a KDTree.

    :param rcut: Cutoff distance for considering an atom a neighbor to another.
    :param pos: The array of atom positions.
    :return: The neighbor list for the atoms in **pos**
    """
    kdtree = cKDTree(pos)
    neighbor_list = kdtree.query_ball_tree(kdtree, r=rcut)
    # Remove an atom from using itself as a neighbor.
    for i, neighbor in enumerate(neighbor_list):
        neighbor.remove(i)
    return neighbor_list


class GBManipulator:
    """
    Class to manipulate atoms in the grain boundary region.

    :param system1: The GBMaker instance containing the generated GB or the filename
        containing the name of the LAMMPS dump file. First parent.
    :param system2: The GBMaker instance containing the generated GB or the filename
        containing the name of the LAMMPS dump file for the second parent, optional,
        defaults to None.
    :param unit_cell: The unit cell of the system. Required if GB1 or GB2 is a LAMMPS
        dump file.
    :param gb_thickness: Thickness of the GB region, optional, defaults to 10.
    :param seed: The seed for random number generation, optional, defaults to None
        (automatically seeded).
    """

    def __init__(
        self,
        system1: Union[GBMaker, str],
        system2: Union[GBMaker, str] = None,
        *,
        gb_thickness: float = None,
        unit_cell: UnitCell = None,
        seed: int = None
    ) -> None:
        # initialize the random number generator
        if not seed:
            self.__rng = np.random.default_rng()
        else:
            self.__rng = np.random.default_rng(seed=seed)

        self.__parents = [None, None]

        if not system2:
            # Some mutators require two parents, so we set __one_parent to True so we do
            # not attempt to perform those in the case that only one GB is passed in.
            self.__one_parent = True
            self.__set_parents(system1, unit_cell=unit_cell,
                               gb_thickness=gb_thickness)
        else:
            self.__one_parent = False
            self.__set_parents(system1, system2, unit_cell=unit_cell,
                               gb_thickness=gb_thickness)
        self.__num_processes = mp.cpu_count() // 2 or 1

    def __set_parents(
        self,
        system1: Union[GBMaker, str],
        system2: Union[GBMaker, str] = None,
        *,
        unit_cell=None,
        gb_thickness=None
    ) -> None:
        """
        Method to assign the parent(s) that will create the child(ren).

        :param system1: The first parent.
        :param system2: The second parent, optional, defaults to None.
        :param unit_cell: Keyword argument. The nominal unit cell of the bulk structure,
            optional, defaults to None. Required only when system1 is of type str.
        :param gb_thickness: Keyword argument. The thickness of the GB region, optional,
            defaults to None. Note that if None is passed to the Parent class
            constructor, a value of 10 is assigned.
        """
        self.__parents[0] = Parent(
            system1, unit_cell=unit_cell, gb_thickness=gb_thickness)
        if system2 is not None:
            # If there are 2 parents, with the first one being of type GBMaker, and
            # unit_cell has not been passed in, we assume that the unit cell from the
            # GBMaker instance applies to the second system.
            if isinstance(system1, GBMaker) and isinstance(system2, str):
                if unit_cell is None:
                    unit_cell = system1.unit_cell
                if gb_thickness is None:
                    gb_thickness = system1.gb_thickness
            self.__parents[1] = Parent(
                system2, unit_cell=unit_cell, gb_thickness=gb_thickness)

    # TODO: Swap to use Atom class if it can be vectorized for each of these mutators.

    def translate_right_grain(self, dy: float, dz: float) -> np.ndarray:
        """
        Displace the right grain in the plane of the GB by (0, dy, dz).

        :param dy: Displacement in y direction in angstroms.
        :param dz: Displacement in z direction in angstroms.
        :return: Atom positions after translation of the right grain.
        """
        if not self.__one_parent:
            warnings.warn("Grain translation only occurring based on parent 1.")
        parent = self.__parents[0]
        updated_right_grain = np.copy(parent.right_grain)
        # Displace all atoms in the right grain by [0, dy, dz]. We modulo by the
        # grain dimensions so atoms do not exceed the original boundary conditions
        # updated_right_grain[:, 2] = (
        # updated_right_grain[:, 2] + dy) % parent.y_dim
        # updated_right_grain[:, 3] = (
        #     updated_right_grain[:, 3] + dz) % parent.z_dim
        updated_right_grain['y'] = (updated_right_grain['y'] + dy) % parent.y_dim
        updated_right_grain['z'] = (updated_right_grain['z'] + dz) % parent.z_dim

        return np.hstack((self.__parents[0].left_grain, updated_right_grain))

    def slice_and_merge(self) -> np.ndarray:
        """
        Given two GB systems, merge them by cutting them at the same location and
        swapping one slice with the same slice in the other system.

        :return: Atom positions after merging the slices.
        """
        # TODO: Make the slice a randomly oriented, randomly placed plane, rather than a
        # randomly placed x-oriented plane. Would need a check that the maximum
        # deviation from the x axis isn't too high though.
        if self.__one_parent:
            raise GBManipulatorValueError(
                "Unable to slice and merge with only one parent.")
        parent1 = self.__parents[0]
        parent2 = self.__parents[1]
        pos1 = parent1.whole_system
        pos2 = parent2.whole_system
        # Limit the slice site to be a quarter of the gb width from the GB itself.
        # TODO: use a more robust calculation for GB position.
        # Note that this is the third time this has been calculated.
        slice_pos = (parent1.box_dims[0, 1] - parent1.box_dims[0, 0]) / 2.0 + \
            parent1.gb_thickness * (-0.25 + 0.5*self.__rng.random())
        pos1 = pos1[pos1['x'] < slice_pos]
        pos2 = pos2[pos2['x'] >= slice_pos]
        new_positions = np.hstack((pos1, pos2))

        return new_positions

    def remove_atoms(
        self,
        *,
        gb_fraction: float = None,
        num_to_remove: int = None
    ) -> np.ndarray:
        """
        Removes *gb_fraction* of atoms or *num_to_remove* atom(s) in the GB region. Uses
        the local order parameter method of Lyakhov *et al.*, Computer Phys. Comm. 181
        (2010) 1623-1632.

        One of the following parameters must be specified.
        :param gb_fraction: Keyword argument. The fraction of atoms in the GB plane to
            remove. Must be less than 25% of the total number of atoms in the GB region.
        :param num_to_remove: Keyword argument. The specific number of atoms to remove.
            Maximum is 25% of the total number of atoms in the GB region.
        :return: Atom positions after atom removal.
        """
        if not gb_fraction and not num_to_remove:
            raise GBManipulatorValueError(
                "gb_fraction or num_to_remove must be specified.")
        # TODO: Include logic to maintain stoichiometry/charge neutrality (as desired)
        if not self.__one_parent:
            warnings.warn("Atom removal only occurring based on parent 1.")
        parent = self.__parents[0]
        # We use the array format because numba/jit has issues with strings.
        atoms = Atom.asarray(parent.whole_system)
        if gb_fraction is not None and (gb_fraction <= 0 or gb_fraction > 0.25):
            raise GBManipulatorValueError("Invalid value for gb_fraction ("
                                          f"{gb_fraction=}). Must be 0 < gb_fraction "
                                          "<= 0.25")
        positions = atoms[:, 1:]
        gb_atom_indices = np.where(
            np.logical_and(
                positions[:, 0] >= (parent.box_dims[0, 1] -
                                    parent.box_dims[0, 0])/2 - parent.gb_thickness/2,
                positions[:, 0] <= (parent.box_dims[0, 1] -
                                    parent.box_dims[0, 0])/2 + parent.gb_thickness/2
            )
        )[0]

        gb_atoms = positions[gb_atom_indices]
        if num_to_remove is None:
            num_to_remove = int(gb_fraction * len(gb_atoms))
            if num_to_remove == 0:
                warnings.warn(
                    "Calculated fraction of atoms to remove is 0 "
                    f"(int({gb_fraction}*{len(gb_atoms)} = 0)"
                )
                return atoms

        # TODO: use a more robust calculation than '6' for the cutoff distance. Base it
        # off the crystal structure?
        neighbor_list = _create_neighbor_list(6, positions)

        args_list = [
            (
                atoms[atom_idx],
                atoms[neighbor_list[atom_idx]],
                parent.unit_cell.names(asint=True),
                parent.unit_cell.a0,
                len(parent.unit_cell.unit_cell),
                0.05,
                15
            )
            for idx, atom_idx in enumerate(gb_atom_indices)
        ]
        with mp.Pool(self.__num_processes) as pool:
            order = pool.starmap(_calculate_local_order, args_list)
        order = np.array(order)

        # We want the probabilities to be inversely proportional to the order parameter.
        # Higher order values should be more 'stable' against removal than low order
        # values. We give small probabilities to the higher order values just to allow
        # for variety in the calculations.
        probabilities = max(order) - order + min(order)
        probabilities = probabilities / np.sum(probabilities, dtype=float)

        indices_to_remove = self.__rng.choice(
            gb_atom_indices, num_to_remove, replace=False, p=probabilities)
        pos = np.delete(parent.whole_system, indices_to_remove, axis=0)
        return pos

    def insert_atoms(
        self,
        *,
        fill_fraction: float = None,
        num_to_insert: int = None,
        method: str = 'delaunay',
        keep_stoichiometry: bool = True
    ) -> np.ndarray:
        """
        Inserts **fraction** atoms in the GB at empty lattice sites. 'Empty' sites are
        determined through Delaunay triangulation (method='Delaunay') or through a grid
        with a resolution of 1 angstrom (method='grid').

        One of the following parameters must be specified.
        :param fill_fraction: Keyword argument. The fraction of empty lattice sites to
            fill. Must be less than or equal to 25% of the total number of atoms in the
            GB slab.
        :param num_to_insert: Keyword argument. The number of atoms to insert. Must be
            less than or equal to 25% of the total number of atoms in the GB slab.

        :param method: Keyword argument, optional, defaults to 'delaunay'. The method to
            use. Must be either 'delaunay' or 'grid.'
        :param keep_stoichiometry: Keyword argument, optional, defaults to True. Flag
            specifying whether or not to keep stoichiometric ratios in the system with
            the added atoms.
        :raises GBManipulatorValueError: Exception raised if an invalid method is
            specified.
        :return: Atom positions after atom insertion.
        """
        if not fill_fraction and not num_to_insert:
            raise GBManipulatorValueError(
                "fill_fraction or num_to_insert must be specified.")

        if not self.__one_parent:
            warnings.warn("Atom insertion only occurring based on parent 1.")
        parent = self.__parents[0]
        gb_atoms = Atom.asarray(parent.gb_atoms)

        if fill_fraction is not None and (fill_fraction <= 0 or fill_fraction > 0.25):
            raise GBManipulatorValueError("Invalid value for fill_fraction ("
                                          f"{fill_fraction=}). Must be 0 < "
                                          "fill_fraction <= 0.25")

        if (num_to_insert is not None and
            (
                        num_to_insert < 1 or
                        num_to_insert > int(0.25 * len(gb_atoms))
                    )
            ):
            raise GBManipulatorValueError(
                "Invalid num_to_insert value. Must be >= 1, and must be less than or "
                "equal to 25% of the total number of atoms in the GB region")

        def Delaunay_approach(
            gb_atoms: np.ndarray,
            atom_radius: float,
            num_to_insert: int
        ) -> np.ndarray:
            """
            Delaunay triangulation approach for inserting atoms. Potential insertion
            sites are the circumcenters of the tetrahedra.

            :param gb_atoms: Array of atom positions where we are considering inserting
                new atoms.
            :param atom_radius: The radius of an atom.
            :param num_to_insert: The number of atoms to insert.
            :return: The sites at which new atoms are inserted.
            """
            # Delaunay triangulation approach
            triangulation = Delaunay(gb_atoms)
            # ijk is for the 3x3 transformation matrix triangulation.transform[:, :3, :]
            # ik is for the offset vector triangulation.transform[:, 3, :], and ij is
            # the resulting circumcenter coordinates
            circumcenters = -np.einsum(
                'ijk,ik->ij',
                triangulation.transform[:, :3, :],
                triangulation.transform[:, 3, :]
            )
            volumes = np.abs(np.linalg.det(triangulation.transform[:, :3, :]))
            volume_threshold = 1e-8
            valid_mask = (volumes > volume_threshold) & ~np.isnan(
                circumcenters).any(axis=1)
            valid_circumcenters = circumcenters[valid_mask]
            valid_simplices = triangulation.simplices[valid_mask, 0]
            sphere_radii = np.linalg.norm(
                gb_atoms[valid_simplices] - valid_circumcenters, axis=1)
            interstitial_radii = sphere_radii - atom_radius
            probabilities = interstitial_radii / np.sum(interstitial_radii)
            probabilities = probabilities / np.sum(probabilities)  # normalize
            assert abs(1 - np.sum(probabilities)
                       ) < 1e-8, "Probabilities are not normalized!"
            num_sites = len(circumcenters)

            if num_to_insert is None:
                num_to_insert = int(fill_fraction * num_sites)

            if num_to_insert == 0:
                warnings.warn("Calculated fraction of atoms to insert is 0: "
                              f"int({fill_fraction}*{len(gb_atoms)}) = 0"
                              )
            indices = self.__rng.choice(
                list(range(len(valid_circumcenters))),
                num_to_insert,
                replace=False,
                p=probabilities
            )
            return valid_circumcenters[indices]

        def grid_approach(
            gb_atoms: np.ndarray,
            atom_radius: float,
            num_to_insert: int,
        ) -> np.ndarray:
            """
            Grid approach for inserting atoms. Potential insertion sites are on a 1x1x1
            Angstrom grid where sites must be at least *atom_radius* away.

            :param gb_atoms: Array of atom positions where we are considering inserting
                new atoms.
            :param atom_radius: The radius of an atom.
            :param num_to_insert: The number of atoms to insert.

            :return: The sites at which new atoms are inserted.
            """
            # Grid approach
            max_x, max_y, max_z = gb_atoms.max(axis=0)
            min_x, min_y, min_z = gb_atoms.min(axis=0)
            X, Y, Z = np.meshgrid(
                np.arange(np.floor(min_x), np.ceil(max_x) + 1),
                np.arange(np.floor(min_y), np.ceil(max_y) + 1),
                np.arange(np.floor(min_z), np.ceil(max_z) + 1),
                indexing='ij'
            )
            sites = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
            GB_tree = KDTree(gb_atoms)
            sites_tree = KDTree(sites)
            indices_to_remove = GB_tree.query_ball_tree(sites_tree, atom_radius)
            indices_to_remove = list(set(
                [i for sublist in indices_to_remove for i in sublist]))
            filtered_sites = np.delete(sites, indices_to_remove, axis=0)

            distances, _ = GB_tree.query(filtered_sites, k=1)
            probabilities = distances / np.sum(distances)
            probabilities = probabilities / np.sum(probabilities)  # normalize
            assert abs(1 - np.sum(probabilities)
                       ) < 1e-8, "Probabilities are not normalized!"
            num_sites = len(filtered_sites)

            if num_to_insert is None:
                num_to_insert = int(fill_fraction * num_sites)

            if num_to_insert == 0:
                warnings.warn("Calculated fraction of atoms to insert is 0: "
                              f"int({fill_fraction}*{len(gb_atoms)}) = 0"
                              )

            indices = self.__rng.choice(num_sites,
                                        num_to_insert,
                                        replace=False,
                                        p=probabilities
                                        )
            return filtered_sites[indices]

        available_types = set(parent.unit_cell.names())
        type_counts = {}
        for i in available_types:
            type_counts[i] = np.sum(parent.unit_cell.names() == i)
        type_ratios = {key: int(value / math.gcd(*type_counts.values()))
                       for key, value in type_counts.items()}
        total_ratio = sum(type_ratios.values())

        # Calculate the insertion sites using the specified approach.
        if method == 'delaunay':
            new_pos = Delaunay_approach(
                gb_atoms[:, 1:], parent.unit_cell.radius, num_to_insert)
        elif method == 'grid':
            new_pos = grid_approach(
                gb_atoms[:, 1:], parent.unit_cell.radius, num_to_insert)
        else:
            raise GBManipulatorValueError(f"Unrecognized insert_atoms method: {method}")

        # The number of extra atoms needed to maintain stoichiometry.
        extra = len(new_pos) % total_ratio
        if extra != 0:
            new_pos = new_pos[:-extra]
        num_elements = [int(len(new_pos) * (r / total_ratio))
                        for r in type_ratios.values()]
        new_types = np.array([val for val, num in zip(
            available_types, num_elements) for _ in range(num)])
        self.__rng.shuffle(new_types)
        new_atoms = np.empty((len(new_pos),), dtype=Atom.atom_dtype)
        new_atoms['x'] = new_pos[:, 0]
        new_atoms['y'] = new_pos[:, 1]
        new_atoms['z'] = new_pos[:, 2]
        new_atoms['name'] = new_types
        return np.hstack((parent.whole_system, new_atoms))

    def displace_along_soft_modes(self) -> np.ndarray:
        """
        Displace atoms along soft phonon modes.

        :raises NotImplementedError: Not currently implemented.
        :return: Atom positions after displacement.
        """
        pos = self.__parents[0].whole_system
        raise NotImplementedError("This mutator has not been implemented yet.")
        return pos

    def apply_group_symmetry(self, group: str) -> np.ndarray:
        """
        Apply the specified group symmetry to the GB region.

        :param group: One of the 230 crystallographic space groups.
        :raises NotImplementedError: Not currently implemented.
        :return: Atoms positions after applying group symmetry.
        """

        pos = self.__parents[0].whole_system
        raise NotImplementedError("This mutator has not been implemented yet.")
        return pos

    # Getter and setter methods for the parents
    @property
    def parents(self) -> list:
        return _ParentsProxy(self)

    @parents.setter
    def parents(self, value) -> None:
        if not isinstance(value, list) or len(value) != 2:
            raise GBManipulatorValueError(
                "The parents attribute must be a list with exactly 2 elements.")

        if any(not (v is None or isinstance(v, Parent)) for v in value):
            raise GBManipulatorValueError(
                "Both items in the parents list must be None or instances of Parent")

        self.__parents = value


if __name__ == "__main__":
    theta = math.radians(36.869898)
    GB = GBMaker(a0=1.0, structure='fcc', gb_thickness=10.0,
                 misorientation=[theta, 0, 0, 0, -theta/2], repeat_factor=4)
    GBManip = GBManipulator(GB)
    GB.write_lammps(GB.gb, GB.box_dims, 'test1.dat')
    GB.write_lammps(GBManip.translate_right_grain(
        0.5, 1.0), GB.box_dims, 'test2.dat')
