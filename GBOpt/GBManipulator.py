import math
import warnings
from copy import deepcopy
from os.path import isfile

import numpy as np
from scipy.spatial import Delaunay, KDTree

from GBOpt.Atom import Atom
from GBOpt.GBMaker import GBMaker
from GBOpt.UnitCell import UnitCell

# TODO: Generalize to interfaces, not just GBs


class GBManipulatorError(Exception):
    pass


class GBManipulatorValueError(GBManipulatorError):
    pass


class CorruptedSnapshotError(Exception):
    pass


class MissingDataError(Exception):
    pass


class GBManipulator:
    """
    Class to manipulate atoms in the grain boundary region.
    :param GBMaker GB: The GBMaker instance containing the generated GB
    """

    def __init__(self, GB: GBMaker = None, snapshot: str = None, gb_thickness: float = 60, unit_cell: UnitCell = None, a0: float = None) -> None:
        self.rng = np.random.default_rng()
        if GB is not None and snapshot is not None:
            raise GBManipulatorValueError(
                "Cannot initialize GBManipulator with GBMaker and a snapshot")

        if GB is not None:  # initialize from GBMaker
            self.right_grain = GB.right_grain
            self.left_grain = GB.left_grain
            self.grain_ydim = GB.grain_ydim
            self.grain_zdim = GB.grain_zdim
            self.gb_thickness = GB.gb_thickness
            self.unit_cell = GB.unit_cell
            self.radius = GB.radius
            self.box_dims = GB.box_dims
        else:  # initialize from LAMMPS dump file
            if unit_cell is None:
                raise GBManipulatorValueError("unit_cell must be specified.")
            if a0 is None:
                raise GBManipulatorValueError("a0 must be specified")
            self.unit_cell = unit_cell
            self.radius = self.unit_cell.radius * a0
            self.gb_thickness = gb_thickness
            self.initialize_from_snapshot(snapshot)

    def initialize_from_snapshot(self, snapshot: str) -> None:
        if not isfile(snapshot):
            raise FileNotFoundError(f"{snapshot} does not exist.")
        with open(snapshot) as f:
            line = f.readline()
            # skip to the box bounds
            while not line.startswith("ITEM: BOX BOUNDS"):
                line = f.readline()
                if not line:
                    raise CorruptedSnapshotError(
                        f"Box bounds not found in {snapshot}")
            if len(line.split()) == 6:
                x_dims = [float(i) for i in f.readline().split()]
                y_dims = [float(i) for i in f.readline().split()]
                z_dims = [float(i) for i in f.readline().split()]
            elif len(line.split()) == 9:
                xline = f.readline().split()
                yline = f.readline().split()
                zline = f.readline().split()
                x_dims, x_tilt = ([float(i)
                                  for i in xline[0:2]], float(xline[2]))
                y_dims, y_tilt = ([float(i)
                                  for i in yline[0:2]], float(yline[2]))
                z_dims, z_tilt = ([float(i)
                                  for i in zline[0:2]], float(zline[2]))
            else:
                raise CorruptedSnapshotError(
                    f"Box bounds corrupted in {snapshot}")
            self.box_dims = np.array([x_dims, y_dims, z_dims])
            self.grain_ydim = y_dims[1] - y_dims[0]
            self.grain_zdim = z_dims[1] - z_dims[0]
            grain_cutoff = (x_dims[1] - x_dims[0]) / 2
            line = f.readline()
            while not line.startswith("ITEM: ATOMS"):
                line = f.readline()
                if not line:
                    raise CorruptedSnapshotError(
                        f"Atoms not found in {snapshot}")
            atom_attributes = line.split()[2:]
            required = ['id', 'type', 'x', 'y', 'z']

            if not all([i in atom_attributes for i in required]):
                raise MissingDataError(f"One or more required attributes are missing.\n"
                                       f"Required: {required}, "
                                       f"available: {atom_attributes}")
            reqd_index = {attr: atom_attributes.index(
                attr) for attr in required}
            left_grain = []
            right_grain = []
            while not line.startswith("ITEM:"):
                line = f.readline().split()
                if not line:
                    break
                atom = Atom(int(line[reqd_index['id']]), int(line[reqd_index['type']]),
                            float(line([reqd_index['x']])), float(
                                line([reqd_index['y']])),
                            float(line([reqd_index['z']]))
                            )
                if atom.position.x < grain_cutoff:
                    left_grain.append(atom)
                else:
                    right_grain.append(atom)
            self.left_grain = left_grain
            self.right_grain = right_grain

    def translate_right_grain(self, dy: float, dz: float):
        """
        Displace the right grain in the plane of the GB by dy, dz
        :param float dy: Displacement in y direction (angstroms).
        :param float dz: Displacement in z direction (angstroms).
        :return np.ndarray: Atom positions after translation.
        """

        # Displace all atoms in the right grain by [0, dy, dz]. We modulo by the
        # grain dimensions so atoms do not exceed the original boundary conditions
        updated_right_grain = deepcopy(GB.right_grain)
        updated_right_grain[:, 1] = (
            updated_right_grain[:, 1] + dy) % GB.grain_ydim
        updated_right_grain[:, 2] = (
            updated_right_grain[:, 2] + dz) % GB.grain_zdim

        return np.vstack((GB.left_grain, updated_right_grain))

    def slice_and_merge(self, pos1, pos2):
        """
        Given two GB systems, merge them by cutting them at the same location and
        swapping one slice with the same slice in the other system.
        :param np.ndarray pos1: The positions of atoms for parent 1.
        :param np.ndarray pos2: The positions of atoms for parent 2.
        :return np.ndarray: Atom positions after merging the slices.
        TODO: Make the slice a randomly oriented, randomly placed plane, rather than a
        randomly placed x-oriented plane.
        """

        slice_pos = GB.gb_thickness * (0.25 + 0.5*np.random.rand())
        pos1 = pos1[pos1[0] < slice_pos]
        pos2 = pos2[pos2[0] >= slice_pos]
        new_positions = np.vstack((pos1, pos2))

        return new_positions

    def gaussian(self, x: float, sigma: float = 0.02):
        """
        Calculates a Gaussian-smeared delta function at _x_ given a standard deviation
        of _sigma_ = 0.02.
        :param float x: where to calculate the Gaussian-smeared delta function.
        :optional param float sigma: Standard deviation of the Gaussian-smeared delta
            function. Default value is 0.02.
        :return float: Value of the Gaussian-smeared delta function at x.
        """
        return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-x * x / (2 * sigma * sigma))

    def calculate_fingerprint_vector(self, atom, neighs, NB, V, Btype, Delta, Rmax):
        """
        Calculates the fingerprint for _atom_ as described in Lyakhov _et al._, Computer
        Phys. Comm. 181 (2010) 1623-1632 (Eq. 4).
        :param Atom atom: atom we are calculating the fingerprint for.
        :param list(Atom) neighs: list of Atom containing the neighbors to _atom_.
        :param int NB: The number of atoms of type B neighbor to _atom_.
        :param float V: The volume of the unit cell in angstroms**3.
        :param int Btype: The type of neighbors we are interested in.
        :param float Delta: The discretization length for Rs in angstroms.
        :return np.ndarray: The vector containing the fingerprint for _atom_.
        """
        Rs = np.arange(0, Rmax+Delta, Delta)

        fingerprint_vector = np.zeros_like(Rs)
        for idx, R in enumerate(Rs):
            local_sum = 0
            for neigh in neighs:
                if neigh.atom_type is Btype:
                    Rij = np.linalg.norm(atom.position-neigh.position)
                    delta = self.gaussian(R-Rij)
                    local_sum += delta / \
                        (4 * np.pi * Rij * Rij * (NB / V) * Delta)
            fingerprint_vector[idx] = local_sum - 1

        return fingerprint_vector

    def calculate_local_order(self, atom, neighs, Delta=0.05, Rmax=10):
        """
        Calculates the local order parameter foloowing Lyakhov _et al._, Computer Phys.
        Comm. 181 (2010) 1623-1632 (Eq. 5).
        :param Atom atom: Atom we are calculating the local order for.
        :param list(Atom) atom: List of Atom containing the neighbors to _atom_.
        :optional param float Delta=0.05: Bin size to calculate the fingerprint vector.
        :optional param float Rmax=10.0: Maximum distance from _atom_ to consider as a
            neighbor to _atom_.
        :return float: The local order parameter for _atom_ based on its neighbors.
        """
        local_sum = 0
        atom_types = set([a.atom_type for a in neighs])
        unit_cell = GB.unit_cell
        N = len(unit_cell.unit_cell)
        for Btype in atom_types:
            NB = np.sum(
                [1 if atom_type == Btype else 0 for atom_type in unit_cell.types])
            V = GB.a**3
            fingerprint = self.calculate_fingerprint_vector(
                atom, neighs, NB, V, Btype, Delta, Rmax)
            local_sum += NB / N * Delta / \
                (V/N)**(1/3) * np.dot(fingerprint, fingerprint)
        return np.sqrt(local_sum)

    def remove_atoms(self, pos, fraction: float):
        """
        Removes _fraction_ of atoms in the GB slab.
        :param np.ndarray pos: The positions of the atoms in the parent.
        :param float fraction: The fraction of atoms in the GB plane to remove. Must be
            less than 25% of the total number of atoms in the GB slab.
        :return np.ndarray: Atom positions after atom removal
        TODO: Only remove atoms from the GB region
        """
        if fraction <= 0 or fraction > 0.25:
            raise ValueError("Invalid value for fraction ("
                             f"{fraction=}). Must be 0 < fraction <= 0.25")

        # GB_slab = GBpos[GBpos[:,0] < ]
        num_to_remove = int(fraction * len(pos))
        if num_to_remove == 0:
            warnings.warn(
                "Calculated fraction of atoms to remove is 0 "
                f"(int({fraction}*{len(pos)}=0)"
            )
            return pos

        # region neighbor list calculation
        rcut = 15
        rcut_sq = rcut * rcut

        ncellx = int(GB.gb_thickness / rcut) + 1
        ncelly = int(GB.grain_ydim / rcut) + 1
        ncellz = int(GB.grain_zdim / rcut) + 1

        lcellx = GB.gb_thickness / ncellx
        lcelly = GB.grain_ydim / ncelly
        lcellz = GB.grain_ydim / ncellz

        n_atoms_per_cell = max(
            int(len(pos) / (ncellx * ncelly * ncellz)), 100)

        icell = np.zeros((ncellx, ncelly, ncellz), dtype=int)
        pcell = np.full(
            (ncellx, ncelly, ncellz, n_atoms_per_cell), -1, dtype=int)
        neighbor_list = np.full((n_atoms_per_cell, len(pos)), -1, dtype=int)

        for i, atom in enumerate(pos):
            idx = int(atom[0] / lcellx)
            idy = int(atom[1] / lcelly)
            idz = int(atom[2] / lcellz)

            idx = min(max(idx, 0), ncellx - 1)
            idy = min(max(idy, 0), ncelly - 1)
            idz = min(max(idz, 0), ncellz - 1)

            icell[idx, idy, idz] += 1
            pcell[idx, idy, idz, icell[idx, idy, idz] - 1] = i

        for i in range(ncellx):
            for j in range(ncelly):
                for k in range(ncellz):
                    for l in range(icell[i, j, k]):
                        id = pcell[i, j, k, l]
                        for ii in range(-1, 2):
                            for jj in range(-1, 2):
                                for kk in range(-1, 2):
                                    ia = i + ii
                                    ja = (j + jj) % ncelly
                                    ka = (k + kk) % ncellz

                                    if ia < 0 or ia >= ncellx:
                                        continue

                                    for m in range(icell[ia, ja, ka]):
                                        jd = pcell[ia, ja, ka, m]
                                        if jd <= id:
                                            continue
                                        rxij = pos[id][0] - pos[jd][0]
                                        ryij = pos[id][1] - pos[jd][1]
                                        rzij = pos[id][2] - pos[jd][2]

                                        ryij -= round(ryij /
                                                      GB.grain_ydim) * GB.grain_ydim
                                        rzij -= round(rzij /
                                                      GB.grain_zdim) * GB.grain_zdim

                                        drij_sq = rxij ** 2 + ryij ** 2 + rzij ** 2

                                        if drij_sq > rcut_sq:
                                            continue

                                        neighbor_list[0, id] += 1
                                        if neighbor_list[0, id] > n_atoms_per_cell:
                                            n_atoms_per_cell += 100
                                            neighbor_list = np.pad(
                                                neighbor_list, ((0, 100), (0, 0)), 'constant', constant_values=-1)
                                        neighbor_list[neighbor_list[0,
                                                                    id], id] = jd

                                        neighbor_list[0, jd] += 1
                                        if neighbor_list[0, jd] >= n_atoms_per_cell:
                                            n_atoms_per_cell += 100
                                            neighbor_list = np.pad(
                                                neighbor_list, ((0, 100), (0, 0)), 'constant', constant_values=-1)
                                        neighbor_list[neighbor_list[0,
                                                                    jd], jd] = id
        # endregion

        order = np.zeros(len(pos))
        for idx, atom in enumerate(pos):
            neighs = neighbor_list[idx]
            order[idx] = self.calculate_local_order(atom, neighs, Rmax=rcut)
        probabilities = max(order) - order + min(order)
        probabilities = probabilities / np.sum(probabilities, dtype=float)

        indices_to_remove = self.rng.choice(
            pos, num_to_remove, replace=False, p=probabilities)
        new_GB = np.delete(pos, indices_to_remove)
        return new_GB

    def insert_atoms(self, pos, fraction: float):
        """
        Inserts _fraction_ atoms in the GB at lattice sites. Empty sites are assumed to
        have a resolution of 1 angstrom.
        :param np.ndarray pos: The positions of atoms in the parent.
        :param float fraction: The fraction of empty lattice sites to fill. Must be less
            than or equal to 25% of the total number of atoms in the GB slab.
        :return np.ndarray: Atom positions after atom insertion.
        TODO: Compare Delaunay Triangulation vs 1 angstrom grid.
        TODO: Only insert atoms in the GB region.
        """
        if fraction <= 0 or fraction > 0.25:
            raise ValueError("Invalid value for fraction ("
                             f"{fraction=}). Must be 0 < fraction <= 0.25")

        def Delaunay_approach():
            # Delaunay triangulation approach
            triangulation = Delaunay(pos)
            circumcenters = np.einsum(
                'ijk,ik->ij', triangulation.transform[:, :3, :], triangulation.transform[:, 3, :])
            sphere_radii = np.linalg.norm(
                pos[triangulation.simplices[:, 0]] - circumcenters, axis=1)
            interstitial_radii = sphere_radii - GB.radius
            probabilities = interstitial_radii / np.sum(interstitial_radii)
            assert abs(1 - np.sum(probabilities)
                       ) < 1e-8, "Probabilities are not normalized!"
            num_sites = len(circumcenters)
            print(
                f"Found {num_sites} available insertion sites (Delaunay method).")

            indices = self.rng.choice(num_sites, int(
                fraction*num_sites), replace=False, p=probabilities)
            new_GB = np.vstack([pos, circumcenters[indices]])

            # testing the calculated circumcenters
            import matplotlib.pyplot as plt

            # https://github.com/PyCQA/pyflakes/issues/180
            from mpl_toolkits.mplot3d import Axes3D
            del Axes3D

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                       color='blue', label='Atoms')
            ax.scatter(circumcenters[:, 0], circumcenters[:, 1],
                       circumcenters[:, 2], color='red', label='Circumcenters')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.legend()
            plt.show()

            return new_GB

        def grid_approach():
            # Grid approach
            max_x, max_y, max_z = pos.max(axis=0)
            X, Y, Z = np.meshgrid(
                np.arange(0, max_x + 1),
                np.arange(0, max_y + 1),
                np.arange(0, max_z + 1)
            )
            sites = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
            tree = KDTree(pos)
            indices_to_remove = tree.query_ball_point(sites, r=GB.radius)
            indices_to_remove = set(
                [index for sublist in indices_to_remove for index in sublist])
            filtered_sites = np.delete(sites, list(indices_to_remove), axis=0)
            distances, _ = tree.query(filtered_sites)
            probabilities = distances / np.sum(distances)
            num_sites = len(filtered_sites)
            print(
                f"Found {num_sites} available insertion sites (grid method).")

            indices = self.rng.choice(num_sites, int(
                fraction * num_sites), replace=False, p=probabilities)
            new_GB = np.vstack([pos, filtered_sites[indices]])
            return new_GB

        GB_Delaunay = Delaunay_approach()
        GB_grid = grid_approach()

        return GB_Delaunay, GB_grid

    def displace_along_soft_modes(self, pos):
        """
        Displace atoms along soft phonon modes.
        :param np.ndarray pos: The positions of the GB atoms in the parent.
        :return np.ndarray: Atom positions after displacement.
        """

        raise NotImplementedError("This mutator has not been implemented yet.")

    def apply_group_symmetry(self, pos, group):
        """
        Apply the specified group symmetry to the GB region.
        :param np.ndarray pos: The positions of the GB atoms in the parent.
        :param str group: One of the 230 crystallographic space groups.
        :return np.ndarray: Atoms positions after applying group symmetry.
        """

        raise NotImplementedError("This mutator has not been implemented yet.")


if __name__ == "__main__":
    theta = math.radians(36.869898)
    GB = GBMaker(lattice_parameter=3.61, structure='fcc', gb_thickness=0.0,
                 misorientation=[theta, 0, 0], repeat_factor=4)
    GBManip = GBManipulator(GB)
    GBManip.rng = np.random.default_rng(seed=100)
    i = 1
    num_shifts = 20
    positions = []
    for dy in np.arange(0, 3.61 + 3.61/num_shifts, 3.61/num_shifts):
        # for dz in np.arange(0, 3.61+3.61/num_shifts, 3.61/num_shifts):
        positions.append(GBManip.translate_right_grain(dy, 0))
        box_dims = np.array(
            [
                [
                    -GB.vacuum_thickness - min(positions[i-1][:, 0]),
                    GB.vacuum_thickness + max(positions[i-1][:, 0])
                ],
                GB.box_dims[1],
                GB.box_dims[2]
            ]
        )
        GB.write_lammps(positions[i-1], box_dims, f"CSL_GB_{i}.dat")
        i += 1

    # positions.append(GBManip.slice_and_merge(
    #     positions[0], positions[num_shifts+1]))
    # GB.write_lammps(positions[i-1], box_dims, f"CSL_GB_{i}.dat")
    # i += 1

    # positions.append(GBManip.remove_atoms(positions[0]))
