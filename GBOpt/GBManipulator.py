import math
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

    :param GB: A GBMaker instance or string containing the filename of a LAMMPS dump
        file.
    :param unit_cell: Required only if GB is specified using a LAMMPS dump file. Gives
        the nominal unit cell of the system.
    :param gb_thickness: Thickness of the GB region, optional, defaults to 10.
    """

    def __init__(self, GB: Union[GBMaker, str], *, unit_cell: UnitCell = None, gb_thickness: float = 10) -> None:
        if isinstance(GB, GBMaker):
            self.__init_by_gbmaker(GB)
        else:
            if gb_thickness is None:  # defaults to 10 if passed in as None.
                gb_thickness = 10
            self.__init_by_snapshot(GB, unit_cell, gb_thickness)
        self.__whole_gb = np.vstack((self.__left_grain, self.__right_grain))
        # TODO: make this a more robust calculation, rather than assuming the GB is
        # in the middle of the system.
        self.__GBpos = self.__whole_gb[
            np.where(
                np.logical_and(
                    self.__whole_gb[:, 0] >= self.__box_dims[0, 1] /
                    2 - self.__gb_thickness/2,
                    self.__whole_gb[:, 0] <= self.__box_dims[0, 1] /
                    2 + self.__gb_thickness/2
                )
            )
        ]

    def __init_by_gbmaker(self, GB: GBMaker) -> None:
        """
        Method for initializing the Parent using a GBMaker instance.

        :param GB: The GBMaker instance.
        """
        self.__right_grain = GB.right_grain
        self.__left_grain = GB.left_grain
        self.__y_dim = GB.y_dim
        self.__z_dim = GB.z_dim
        self.__gb_thickness = GB.gb_thickness
        self.__unit_cell = GB.unit_cell
        self.__radius = GB.radius
        self.__box_dims = GB.box_dims
        # We do not use GB.x_dim because this is limited to a single grain of the GB,
        # not the entire system.
        self.__x_dim = self.__box_dims[0][1] - self.__box_dims[0][0]

    def __init_by_snapshot(self, GB: str, unit_cell: UnitCell, gb_thickness: float) -> None:
        """
        Method for initializing the Parent using a LAMMPS dump file.

        :param GB: Filename of the LAMMPS dump file.
        :param unit_cell: Nominal unit cell of the bulk structure.
        :param gb_thickness: Thickness of the GB region, given in angstroms.
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
        if not isfile(GB):
            raise ParentFileNotFoundError(f"{GB} does not exist.")
        with open(GB) as f:
            line = f.readline()
            # skip to the box bounds
            while not line.startswith("ITEM: BOX BOUNDS"):
                line = f.readline()
                if not line:
                    raise ParentCorruptedSnapshotError(
                        f"Box bounds not found in {GB}")
            if len(line.split()) == 6:  # orthogonal box
                x_dims = [float(i) for i in f.readline().split()]
                y_dims = [float(i) for i in f.readline().split()]
                z_dims = [float(i) for i in f.readline().split()]
            elif len(line.split()) == 9:  # triclinic box
                xline = f.readline().split()
                yline = f.readline().split()
                zline = f.readline().split()
                x_dims, _ = ([float(i)
                              for i in xline[0:2]], float(xline[2]))
                y_dims, _ = ([float(i)
                              for i in yline[0:2]], float(yline[2]))
                z_dims, _ = ([float(i)
                              for i in zline[0:2]], float(zline[2]))
            else:
                raise ParentCorruptedSnapshotError(f"Box bounds corrupted in {GB}")
            if not (x_dims or y_dims or z_dims) or len(x_dims) != 2 or \
                    len(y_dims) != 2 or len(z_dims) != 2:
                raise ParentCorruptedSnapshotError(f"Box bounds corrupted in {GB}")
            self.__box_dims = np.array([x_dims, y_dims, z_dims])
            self.__x_dim = x_dims[1] - x_dims[0]
            self.__y_dim = y_dims[1] - y_dims[0]
            self.__z_dim = z_dims[1] - z_dims[0]
            # TODO: Need a more robust calculation of where the GB is located.
            grain_cutoff = (x_dims[1] - x_dims[0]) / 2
            line = f.readline()
            while not line.startswith("ITEM: ATOMS"):
                line = f.readline()
                if not line:
                    raise ParentCorruptedSnapshotError(f"Atoms not found in {GB}")
            atom_attributes = line.split()[2:]
            required = ['id', 'type', 'x', 'y', 'z']

            if not all([i in atom_attributes for i in required]):
                raise ParentSnapshotMissingDataError(
                    f"One or more required attributes are missing.\n"
                    f"Required: {required}, "
                    f"available: {atom_attributes}")
            reqd_index = {attr: atom_attributes.index(attr) for attr in required}
            left_grain = []
            right_grain = []
            line = f.readline()  # read the next line to move the file pointer ahead.
            while not line.startswith("ITEM:"):
                line = line.split()
                if not line:
                    break
                atom = Atom(int(line[reqd_index['id']]),
                            int(line[reqd_index['type']]),
                            float(line[reqd_index['x']]),
                            float(line[reqd_index['y']]),
                            float(line[reqd_index['z']])
                            )
                # TODO: Swap to use Atom class.
                if atom.position.x < grain_cutoff:
                    left_grain.append(atom['type', 'x', 'y', 'z'])
                    # left_grain.append(atom)
                else:
                    right_grain.append(atom['type', 'x', 'y', 'z'])
                    # right_grain.append(atom)
                line = f.readline()
            self.__left_grain = np.array(left_grain)
            self.__right_grain = np.array(right_grain)

    # Getters
    @property
    def left_grain(self) -> np.ndarray:
        return self.__left_grain

    @property
    def right_grain(self) -> np.ndarray:
        return self.__right_grain

    @property
    def whole_gb(self) -> np.ndarray:
        return self.__whole_gb

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
    Calculates a Gaussian-smeared delta function at **x** given a standard deviation
    of **sigma**.

    :param x: where to calculate the Gaussian-smeared delta function.
    :param sigma: Standard deviation of the Gaussian-smeared delta
        function, optional, defaults to 0.02.
    :return: Value of the Gaussian-smeared delta function at x.
    """
    prefactor = 1 / (sigma * np.sqrt(2 * np.pi))
    return prefactor * np.exp(-x * x / (2 * sigma * sigma))


@jit(nopython=True, cache=True)
def _calculate_fingerprint_vector(atom, neighs, NB, V, Btype, Delta, Rmax):
    """
    Calculates the fingerprint for **atom** as described in Lyakhov *et al.*,
    Computer Phys. Comm. 181 (2010) 1623-1632 (Eq. 4).

    :param np.ndarray atom: The atom we are calculating the fingerprint for.
    :param np.ndarray neighs: list of Atom containing the neighbors to **atom**.
    :param int NB: The number of atoms of type B neighbor to **atom**.
    :param float V: The volume of the unit cell in angstroms**3.
    :param int Btype: The type of neighbors we are interested in.
    :param float Delta: The discretization length for Rs in angstroms.
    :param float Rmax: The maximum distance from atom to calculate the fingerprint.
    :return: The vector containing the fingerprint for **atom**.
    """
    Rs = np.arange(0, Rmax+Delta, Delta)

    fingerprint_vector = np.zeros_like(Rs)
    for idx, R in enumerate(Rs):
        local_sum = 0
        for neigh in neighs:
            # if neigh['type'] == Btype # TODO: Swap to use Atom class
            if neigh[0] == Btype:
                # Rij = atom['position'].distance(neigh['position']) # TODO: Swap to use Atom class
                diff = atom[1:] - neigh[1:]
                # Rij = np.linalg.norm(atom[1:] - neigh[1:])
                Rij = np.sqrt(np.dot(diff, diff))
                delta = _gaussian(R-Rij, 0.02)
                local_sum += delta / \
                    (4 * np.pi * Rij * Rij * (NB / V) * Delta)
                # pdb.set_trace()
        fingerprint_vector[idx] = local_sum - 1

    return fingerprint_vector


@jit(nopython=True, cache=True)
def _calculate_local_order(atom, neighs, unit_cell_types, unit_cell_a0, N, Delta, Rmax):
    """
    Calculates the local order parameter following Lyakhov *et al.*, Computer Phys.
    Comm. 181 (2010) 1623-1632 (Eq. 5).

    :param np.ndarray atom: Atom we are calculating the local order for.
    :param np.ndarray neighs: Neighbors of **atom**.
    :param np.ndarray unit_cell_types: The types of the atoms in the unit cell.
    :param float unit_cell_a0: The lattice parameter.
    :param int N: The number of atoms in the unit cell.
    :param float Delta: Bin size to calculate the fingerprint vector, optional, defaults
        to 0.05.
    :param float Rmax: Maximum distance from **atom** to consider as a neighbor to
        **atom** in angstroms, optional, defaults to 10.
    :return: The local order parameter for **atom** based on its neighbors.
    """
    local_sum = 0
    # atom_types = set([a['type'] for a in neighs]) # TODO: Swap to use Atom class

    atom_types = np.unique(neighs[:, 0])
    V = unit_cell_a0**3
    unit_cell_types = unit_cell_types
    prefactor = Delta / (N * (V/N)**(1/3))
    for Btype in atom_types:
        NB = np.sum(unit_cell_types == Btype)
        fingerprint = _calculate_fingerprint_vector(
            atom, neighs, NB, V, Btype, Delta, Rmax)
        local_sum += NB * prefactor * np.dot(fingerprint, fingerprint)
    return np.sqrt(local_sum)


class GBManipulator:
    """
    Class to manipulate atoms in the grain boundary region.

    :param GB1: The GBMaker instance containing the generated GB or the filename
        containing the name of the LAMMPS dump file. First parent.
    :param GB2: The GBMaker instance containing the generated GB or the filename
        containing the name of the LAMMPS dump file for the second parent, optional,
        defaults to None.
    :param unit_cell: The unit cell of the system. Required if GB1 or GB2 is a LAMMPS
        dump file.
    :param gb_thickness: Thickness of the GB region, optional, defaults to 10.
    :param seed: The seed for random number generation, optional, defaults to None
        (automatically seeded).
    """

    def __init__(self, GB1: Union[GBMaker, str], GB2: Union[GBMaker, str] = None, *, gb_thickness: float = None, unit_cell: UnitCell = None, seed: int = None) -> None:
        # initialize the random number generator
        if not seed:
            self.__rng = np.random.default_rng()
        else:
            self.__rng = np.random.default_rng(seed=seed)

        self.__parents = [None, None]

        if not GB2:
            # Some mutators require two parents, so we set __one_parent to True so we do
            # not attempt to perform those in the case that only one GB is passed in.
            self.__one_parent = True
            self.__set_parents(GB1, unit_cell=unit_cell,
                               gb_thickness=gb_thickness)
        else:
            self.__one_parent = False
            self.__set_parents(GB1, GB2, unit_cell=unit_cell,
                               gb_thickness=gb_thickness)

    def __set_parents(self, GB1: Union[GBMaker, str], GB2: Union[GBMaker, str] = None, unit_cell=None, gb_thickness=None) -> None:
        """
        Method to assign the parent(s) that will create the child(ren).

        :param GB1: The first parent.
        :param GB2: The second parent, optional, defaults to None.
        :param unit_cell: The nominal unit cell of the bulk structure, optional,
            defaults to None. Required only when GB1 is of type str.
        :param gb_thickness: The thickness of the GB region, optional, defaults to None.
            Note that if None is passed to the Parent class constructor, a value of 10
            is assigned.
        """
        self.__parents[0] = Parent(GB1, unit_cell=unit_cell, gb_thickness=gb_thickness)
        if GB2 is not None:
            # If there are 2 parents, with the first one being of type GBMaker, and
            # unit_cell has not been passed in, we assume that the unit cell from the
            # GBMaker instance applies to the second system.
            if isinstance(GB1, GBMaker) and isinstance(GB2, str):
                if unit_cell is None:
                    unit_cell = GB1.unit_cell
                if gb_thickness is None:
                    gb_thickness = GB1.gb_thickness
            self.__parents[1] = Parent(
                GB2, unit_cell=unit_cell, gb_thickness=gb_thickness)

    def __create_neighbor_list(self, rcut: float, pos: np.ndarray) -> list:
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

    # TODO: Swap to use Atom class if it can be vectorized for each of these mutators.
    def translate_right_grain(self, dy: float, dz: float) -> np.ndarray:
        """
        Displace the right grain in the plane of the GB by (0, dy, dz).

        :param dy: Displacement in y direction in angstroms.
        :param dz: Displacement in z direction in angstroms.
        :return: Atom positions after translation of the right grain.
        """
        if not self.__one_parent:
            warnings.warn("Grain translation only occuring based on parent 1.")
        parent = self.__parents[0]
        updated_right_grain = parent.right_grain
        # Displace all atoms in the right grain by [0, dy, dz]. We modulo by the
        # grain dimensions so atoms do not exceed the original boundary conditions
        updated_right_grain[:, 2] = (
            updated_right_grain[:, 2] + dy) % parent.y_dim
        updated_right_grain[:, 3] = (
            updated_right_grain[:, 3] + dz) % parent.z_dim

        return np.vstack((self.__parents[0].left_grain, updated_right_grain))

    def slice_and_merge(self) -> np.ndarray:
        """
        Given two GB systems, merge them by cutting them at the same location and
        swapping one slice with the same slice in the other system.

        :return: Atom positions after merging the slices.
        """
        # TODO: Make the slice a randomly oriented, randomly placed plane, rather than a
        # randomly placed x-oriented plane. Would need a check that the maximum
        # deviataion from the x axis isn't too high though.
        if self.__one_parent:
            raise GBManipulatorValueError(
                "Unable to slice and merge with only one parent.")
        parent1 = self.__parents[0]
        parent2 = self.__parents[1]
        pos1 = parent1.whole_gb
        pos2 = parent2.whole_gb
        # Limit the slice site to be a quarter of the gb width from the GB itself.
        # TODO: use a more robust calculation for GB position.
        slice_pos = (parent1.box_dims[0, 1] - parent1.box_dims[0, 0]) / 2.0 + \
            parent1.gb_thickness * (-0.25 + 0.5*self.__rng.random())
        pos1 = pos1[pos1[:, 1] < slice_pos]
        pos2 = pos2[pos2[:, 1] >= slice_pos]
        new_positions = np.vstack((pos1, pos2))

        return new_positions

    def remove_atoms(self, fraction: float) -> np.ndarray:
        """
        Removes **fraction** of atoms in the GB slab. Uses the local order parameter
        method of Lyakhov *et al.*, Computer Phys. Comm. 181 (2010) 1623-1632.

        :param fraction: The fraction of atoms in the GB plane to remove. Must be
            less than 25% of the total number of atoms in the GB slab.
        :return: Atom positions after atom removal.
        """
        # TODO: Include logic to maintain stoichiometry/charge neutrality (as desired)
        if not self.__one_parent:
            warnings.warn("Atom removal only occuring based on parent 1.")
        parent = self.__parents[0]
        atoms = parent.whole_gb
        if fraction <= 0 or fraction > 0.25:
            raise GBManipulatorValueError("Invalid value for fraction ("
                                          f"{fraction=}). Must be 0 < fraction <= 0.25")
        positions = atoms[:, 1:]
        GB_slab_indices = np.where(
            np.logical_and(
                positions[:, 0] >= (parent.box_dims[0, 1] -
                                    parent.box_dims[0, 0])/2 - parent.gb_thickness/2,
                positions[:, 0] <= (parent.box_dims[0, 1] -
                                    parent.box_dims[0, 0])/2 + parent.gb_thickness/2
            )
        )[0]

        GB_slab = positions[GB_slab_indices]
        num_to_remove = int(fraction * len(GB_slab))
        if num_to_remove == 0:
            warnings.warn(
                "Calculated fraction of atoms to remove is 0 "
                f"(int({fraction}*{len(GB_slab)} = 0)"
            )
            return atoms

        # TODO: use a more robust calculation than '6' for the cutoff distance. Base it
        # off the crystal structure?
        neighbor_list = self.__create_neighbor_list(6, positions)
        # unit_cell = self.__parents[0].unit_cell
        # N = len(unit_cell.unit_cell)
        # V = unit_cell.a0**3
        # atom_types = np.array(list(set(unit_cell.types())))
        # NBs = np.array([np.sum(unit_cell.types() == Btype) for Btype in atom_types])

        # with multiprocessing.Pool() as pool:
        #     calc_order_partial = partial(
        #         self.__calculate_local_order, atom_types=atom_types, V=V, N=N, NBs=NBs, Delta=0.05, Rmax=15)
        #     order = pool.starmap(calc_order_partial[(atom, atoms[neigh_indices])] for atom_idx, atom in enumerate(
        #         GB_slab) for neigh_indices in neighbor_list[atom_idx])
        order = np.zeros(len(GB_slab_indices))
        for idx, atom_idx in enumerate(GB_slab_indices):
            atom = atoms[atom_idx]
            neigh_indices = neighbor_list[atom_idx]
            order[idx] = _calculate_local_order(
                atom,
                atoms[neigh_indices],
                parent.unit_cell.types(),
                parent.unit_cell.a0,
                len(parent.unit_cell.unit_cell),
                Delta=0.05,
                Rmax=15
            )

        # We want the probabilities to be inversely proportional to the order parameter.
        # Higher order values should be more 'stable' against removal than low order
        # values. We give small probabilities to the higher order values just to allow
        # for variety in the calculations.
        probabilities = max(order) - order + min(order)
        probabilities = probabilities / np.sum(probabilities, dtype=float)

        indices_to_remove = self.__rng.choice(
            GB_slab_indices, num_to_remove, replace=False, p=probabilities)
        pos = np.delete(atoms, indices_to_remove, axis=0)
        return pos

    def insert_atoms(self, fraction: float, *, method: str) -> np.ndarray:
        """
        Inserts **fraction** atoms in the GB at empty lattice sites. 'Empty' sites are
        determined through Delaunay triangulation (method='Delaunay') or through a grid
        with a resolution of 1 angstrom (method='grid').

        :param fraction: The fraction of empty lattice sites to fill. Must be less
            than or equal to 25% of the total number of atoms in the GB slab.
        :param method: The method to use. Must be either 'delaunay' or 'grid'
        :raises GBManipulatorValueError: Exception raised if an invalid method is
            specified.
        :return: Atom positions after atom insertion.
        """
        # TODO: Logic to maintain stoichiometry/charge neutrality needed.
        if fraction <= 0 or fraction > 0.25:
            raise GBManipulatorValueError("Invalid value for fraction ("
                                          f"{fraction=}). Must be 0 < fraction <= 0.25")

        if not self.__one_parent:
            warnings.warn("Atom insertion only occuring based on parent 1.")
        parent = self.__parents[0]
        pos = parent.whole_gb[:, 1:]
        GB_slab_indices = np.where(
            np.logical_and(
                pos[:, 0] >= (parent.box_dims[0, 1] - parent.box_dims[0, 0]
                              ) / 2 - parent.gb_thickness / 2,
                pos[:, 0] <= (parent.box_dims[0, 1] - parent.box_dims[0, 0]
                              ) / 2 + parent.gb_thickness / 2
            )
        )
        GB_slab = pos[GB_slab_indices]

        def Delaunay_approach(GB_slab: np.ndarray, atom_radius: float) -> np.ndarray:
            """
            Delaunay triangulation approach for inserting atoms. Potential insertion
            sites are the circumcenters of the tetrahedra.

            :param GB_slab: Array of atom positions where we are considering inserting
                new atoms.
            :param atom_radius: The radius of an atom.
            :return: The sites at which new atoms are inserted.
            """
            # Delaunay triangulation approach
            triangulation = Delaunay(GB_slab)
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
                GB_slab[valid_simplices] - valid_circumcenters, axis=1)
            interstitial_radii = sphere_radii - atom_radius
            probabilities = interstitial_radii / np.sum(interstitial_radii)
            probabilities = probabilities / np.sum(probabilities)  # normalize
            assert abs(1 - np.sum(probabilities)
                       ) < 1e-8, "Probabilities are not normalized!"
            num_sites = len(circumcenters)

            num_to_insert = int(fraction * num_sites)
            if num_to_insert == 0:
                warnings.warn("Calculated fraction of atoms to insert is 0: "
                              f"int({fraction}*{len(GB_slab)}) = 0"
                              )
                return GB_slab
            indices = self.__rng.choice(
                list(range(len(valid_circumcenters))),
                num_to_insert,
                replace=False,
                p=probabilities
            )
            return valid_circumcenters[indices]

        def grid_approach(GB_slab: np.ndarray, atom_radius: float) -> np.ndarray:
            """
            Grid approach for inserting atoms. Potential insertion sites are on a 1x1x1
            Angstrom grid where sites must be at least **atom_radius** away.

            :param GB_slab: Array of atom positions where we are considering inserting
                new atoms.
            :param atom_radius: The radius of an atom.
            :return: The sites at which new atoms are inserted.
            """
            # Grid approach
            max_x, max_y, max_z = GB_slab.max(axis=0)
            min_x, min_y, min_z = GB_slab.min(axis=0)
            X, Y, Z = np.meshgrid(
                np.arange(np.floor(min_x), np.ceil(max_x) + 1),
                np.arange(np.floor(min_y), np.ceil(max_y) + 1),
                np.arange(np.floor(min_z), np.ceil(max_z) + 1),
                indexing='ij'
            )
            sites = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
            GB_tree = KDTree(GB_slab)
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

            num_to_insert = int(fraction * num_sites)
            if num_to_insert == 0:
                warnings.warn("Calculate fraction of atoms to insert is 0: "
                              f"int({fraction}*{len(GB_slab)}) = 0"
                              )
                return GB_slab
            indices = self.__rng.choice(num_sites,
                                        num_to_insert,
                                        replace=False,
                                        p=probabilities
                                        )
            return filtered_sites[indices]

        if method == 'delaunay':
            return np.vstack((pos, Delaunay_approach(GB_slab, parent.unit_cell.radius)))
        elif method == 'grid':
            return np.vstack((pos, grid_approach(GB_slab, parent.unit_cell.radius)))
        else:
            raise GBManipulatorValueError(f"Unrecognized insert_atoms method: {method}")

    def displace_along_soft_modes(self) -> np.ndarray:
        """
        Displace atoms along soft phonon modes.

        :raises NotImplementedError: Not currently implemented.
        :return: Atom positions after displacement.
        """
        pos = self.__parents[0].whole_gb
        raise NotImplementedError("This mutator has not been implemented yet.")
        return pos

    def apply_group_symmetry(self, group: str) -> np.ndarray:
        """
        Apply the specified group symmetry to the GB region.

        :param group: One of the 230 crystallographic space groups.
        :raises NotImplementedError: Not currently implemented.
        :return: Atoms positions after applying group symmetry.
        """

        pos = self.__parents[0].whole_gb
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
                 misorientation=[theta, 0, 0, 0, 0], repeat_factor=4)
    GBManip = GBManipulator(GB)
    GB.write_lammps(GB.gb, GB.box_dims, 'test1.dat')
    GB.write_lammps(GBManip.translate_right_grain(
        0.5, 1.0), GB.box_dims, 'test2.dat')
    # GBManip.rng = np.random.default_rng(seed=100)
    # i = 1
    # num_shifts = 5
    # positions = []
    # for dy in np.arange(0, 3.61 + 3.61/num_shifts, 3.61/num_shifts):
    #     for dz in np.arange(0, 3.61+3.61/num_shifts, 3.61/num_shifts):
    #         positions.append(GBManip.translate_right_grain(dy, dz))
    #         box_dims = np.array(
    #             [
    #                 [
    #                     -GB.vacuum_thickness - min(positions[i-1][:, 0]),
    #                     GB.vacuum_thickness + max(positions[i-1][:, 0])
    #                 ],
    #                 GB.box_dims[1],
    #                 GB.box_dims[2]
    #             ]
    #         )
    #         GB.write_lammps(positions[i-1], box_dims, f"CSL_GB_{i}.dat")
    #         i += 1

    # positions.append(GBManip.slice_and_merge(
    #     positions[0], positions[num_shifts+1]))
    # GB.write_lammps(positions[i-1], box_dims, f"CSL_GB_{i}.dat")
    # i += 1

    # positions.append(GBManip.remove_atoms(positions[0], 0.25))
    # positions.append(GBManip.insert_atoms(
    #     positions[0], 0.25, method='Delaunay'))
    # positions.append(GBManip.insert_atoms(positions[0], 0.25, method='grid'))
