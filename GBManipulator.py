import math
import warnings
from copy import deepcopy

import numpy as np
from scipy.spatial import Delaunay, KDTree

from GBMaker import GBMaker

# TODO: Generalize to interfaces, not just GBs


class GBManipulator:
    """
    Class to manipulate atoms in the grain boundary region.
    :param GBMaker GB: The GBMaker instance containing the generated GB
    """

    def __init__(self, GB: GBMaker):
        self.rng = np.random.default_rng()
        pass

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

    def slice_and_merge(self, GB1, GB2):
        """
        Given two GB systems, merge them by cutting them at the same location and
        swapping one slice with the same slice in the other system.
        :param np.ndarray GB1: The positions of GB atoms for parent 1.
        :param np.ndarray GB2: The positions of GB atoms for parent 2.
        :return np.ndarray: Atom positions after merging the slices.
        TODO: Make the slice a randomly oriented, randomly placed plane, rather than a
        randomly placed x-oriented plane.
        """

        slice_pos = GB.gb_thickness * (0.25 + 0.5*np.random.rand())
        pos1 = GB1[GB1[0] < slice_pos]
        pos2 = GB2[GB2[0] >= slice_pos]
        new_positions = np.vstack((pos1, pos2))

        return new_positions

    def remove_atoms(self, GBpos, fraction: float):
        """
        Removes _fraction_ of atoms at the GB plane.
        :param np.ndarray GBpos: The positions of the GB atoms in the parent.
        :param float fraction: The fraction of atoms in the GB plane to remove. Must be
            less than 25% of the total number of atoms in the GB slab.
        :return np.ndarray: Atom positions after atom removal
        """
        if fraction <= 0 or fraction > 0.25:
            raise ValueError("Invalid value for fraction ("
                             f"{fraction=}). Must be 0 < fraction <= 0.25")

        num_to_remove = int(fraction * len(GBpos))
        if num_to_remove == 0:
            warnings.warn(
                "Calculated fraction of atoms to remove is 0 "
                f"(int({fraction}*{len(GBpos)}=0)"
            )
            return GBpos

        probabilities = np.ones(num_to_remove)

        def fingerprint(distances, Rmax, delta_R, sigma):
            R_bins = np.arange(0, Rmax+delta_R, delta_R)
            F_R = {}
            for key in distances.keys():
                F_R[key] = np.zeros_like(R_bins)
                for Rij in distances[key]:
                    F_R[key] += gaussian(R_bins-Rij, sigma)
            return R_bins, F_R

        def gaussian(x: float, sigma: float = 0.03):
            return 1/(sigma * np.sqrt(2*np.pi))*np.exp(-x*x/(2*sigma*sigma))

        def lattice(a: float, structure: str):
            if structure == 'fcc':
                lattice_vectors = np.array(
                    [
                        [0, 0, 0],
                        [0.5, 0.5, 0],
                        [0.5, 0, 0.5],
                        [0, 0.5, 0.5]
                    ]
                )
                types = [1, 1, 1, 1]
            elif structure == 'bcc':
                lattice_vectors = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
                types = [1, 1]
            elif structure == 'sc':
                lattice_vectors = np.array([[0, 0, 0]])
                types = [1]
            elif structure == 'diamond':
                lattice_vectors = np.array(
                    [
                        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                        [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [
                            0.75, 0.25, 0.75],
                        [0.25, 0.75, 0.75]
                    ]
                )
                types = [1, 1, 1, 1, 1, 1, 1, 1]
            elif structure == 'fluorite':
                lattice_vectors = np.array(
                    [
                        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                        [0.25, 0.25, 0.25], [0.25, 0.25, 0.75], [
                            0.25, 0.75, 0.25],
                        [0.25, 0.75, 0.75], [0.75, 0.25, 0.25], [
                            0.75, 0.25, 0.75],
                        [0.75, 0.75, 0.25], [0.75, 0.75, 0.75]
                    ]
                )
                types = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
            elif structure == 'rocksalt':
                lattice_vectors = np.array(
                    [
                        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                        [0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0], [0.5, 0.5, 0.5]
                    ]
                )
                types = [1, 1, 1, 1, 2, 2, 2, 2]
            elif structure == 'zincblende':
                lattice_vectors = np.array(
                    [
                        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                        [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [
                            0.75, 0.25, 0.75],
                        [0.25, 0.75, 0.75]
                    ]
                )
                types = [1, 1, 1, 1, 2, 2, 2, 2]
            else:
                raise NotImplementedError(
                    f"Lattice structure {structure} not yet implemented.")
            return lattice_vectors * a, types

        def calculate_distances(positions, types, Rmax):
            n_atoms = len(positions)
            n_types = len(set(types))

            # Initialize the distances dict
            distances = {}
            for i in range(1, n_types+1):
                for j in range(1, n_types+1):
                    distances[i, j] = []
            # Calculate the distances, store the ones below the threshold
            for i in range(n_atoms):
                for j in range(i+1, n_atoms):
                    Rij = np.linalg.norm(positions[i]-positions[j])
                    if Rij < Rmax:
                        distances[types[i], types[j]].append(Rij)
            return distances

        def local_order(F_R, delta_R, V, N):
            local_order = 0
            for key in F_R.keys():
                local_order += np.sum(F_R[key]**2) * \
                    delta_R / (V/N)**(1/3) * len(F_R[key]/N)
            return np.sqrt(local_order)

        def calculate_order_param(atom, Rmax, delta_R, sigma, a0, structure):
            V = a0 ** 3
            positions, types = lattice(a0, structure)
            N = len(positions)
            distances = calculate_distances(positions, types, Rmax)
            R_bins, F_Rs = fingerprint(distances, Rmax, delta_R, sigma)
            return local_order(F_Rs, delta_R, V, N)
        Rmax = 10.0
        delta_R = 0.01
        sigma = 0.003
        positions, types = lattice(3.61, 'fcc')
        distances = calculate_distances(positions, types, Rmax)
        R_bins, F_Rs = fingerprint(distances, Rmax, delta_R, sigma)

        indices_to_remove = self.rng.choice(
            GBpos, num_to_remove, replace=False, p=probabilities)
        new_GB = np.delete(GBpos, indices_to_remove)
        return new_GB

    def insert_atoms(self, GBpos, fraction: float):
        """
        Inserts _fraction_ atoms in the GB at lattice sites. Empty sites are assumed to
        have a resolution of 1 angstrom.
        TODO: Compare Delaunay Triangulation vs 1 angstrom grid.
        :param np.ndarray GBpos: The positions of the GB atoms in the parent.
        :param float fraction: The fraction of empty lattice sites to fill. Must be less
            than or equal to 25% of the total number of atoms in the GB slab.
        :return np.ndarray: Atom positions after atom insertion.
        """
        if fraction <= 0 or fraction > 0.25:
            raise ValueError("Invalid value for fraction ("
                             f"{fraction=}). Must be 0 < fraction <= 0.25")

        def Delaunay_approach():
            # Delaunay triangulation approach
            triangulation = Delaunay(GBpos)
            circumcenters = np.einsum(
                'ijk,ik->ij', triangulation.transform[:, :3, :], triangulation.transform[:, 3, :])
            sphere_radii = np.linalg.norm(
                GB[triangulation.simplices[:, 0]] - circumcenters, axis=1)
            interstitial_radii = sphere_radii - GB.radius
            probabilities = interstitial_radii / np.sum(interstitial_radii)
            assert abs(1 - np.sum(probabilities)
                       ) < 1e-8, "Probabilities are not normalized!"
            num_sites = len(circumcenters)
            print(
                f"Found {num_sites} available insertion sites (Delaunay method).")

            indices = self.rng.choice(num_sites, int(
                fraction*num_sites), replace=False, p=probabilities)
            new_GB = np.vstack([GBpos, circumcenters[indices]])

            # testing the calculated circumcenters
            import matplotlib.pyplot as plt

            # https://github.com/PyCQA/pyflakes/issues/180
            from mpl_toolkits.mplot3d import Axes3D
            del Axes3D

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(GBpos[:, 0], GBpos[:, 1], GBpos[:, 2],
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
            max_x, max_y, max_z = GBpos.max(axis=0)
            X, Y, Z = np.meshgrid(
                np.arange(0, max_x + 1),
                np.arange(0, max_y + 1),
                np.arange(0, max_z + 1)
            )
            sites = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
            tree = KDTree(GBpos)
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
            new_GB = np.vstack([GBpos, filtered_sites[indices]])
            return new_GB

    def displace_along_soft_modes(self, GBpos):
        """
        Displace atoms along soft phonon modes.
        :param np.ndarray GBpos: The positions of the GB atoms in the parent.
        :return np.ndarray: Atom positions after displacement.
        """

        raise NotImplementedError("This mutator has not been implemented yet.")

    def apply_group_symmetry(self, GBpos, group):
        """
        Apply the specified group symmetry to the GB region.
        :param np.ndarray GBpos: The positions of the GB atoms in the parent.
        :param str group: One of the 230 crystallographic space groups.
        :return np.ndarray: Atoms positions after applying group symmetry.
        """

        raise NotImplementedError("This mutator has not been implemented yet.")


if __name__ == "__main__":
    theta = math.radians(36.869898)
    GB = GBMaker(lattice_parameter=3.61,        gb_thickness=0.0,
                 misorientation=[theta, 0, 0],        repeat_factor=4)
    GBManip = GBManipulator(GB)
    i = 1
    for dy in np.arange(0, 3.61, 3.61/10):
        for dz in np.arange(0, 3.61, 3.61/10):
            positions = GBManip.translate_right_grain(dy, dz)
            box_dims = np.array(
                [
                    [
                        -GB.vacuum_thickness - min(positions[:, 0]),
                        GB.vacuum_thickness + max(positions[:, 0])
                    ],
                    GB.box_dims[1],
                    GB.box_dims[2]
                ]
            )
            GB.write_lammps(positions, box_dims, f"CSL_GB_{i}.dat")
            i += 1
