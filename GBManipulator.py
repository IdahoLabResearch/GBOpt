import math
from copy import deepcopy

import numpy as np

from GBMaker import GBMaker


class GBManipulator:
    """
    Class to manipulate atoms in the grain boundary region
    :param GBMaker GB: The GBMaker instance containing the generated GB
    """

    def __init__(self, GB: GBMaker):
        pass

    def translate_right_grain(self, dy: float, dz: float):
        """
        Displace the right grain in the plane of the GB by dy, dz
        :param float dy: Displacement in y direction (angstroms)
        :param float dz: Displacement in z direction (angstroms)
        :return np.ndarray: Atom positions after translation.
        """
        if i == 0:
            filename = "CSL_GB.dat"
        else:
            filename = f"CSL_GB_{i}.dat"
        # Displace all atoms in the right grain by [0, dy, dz]. We modulo by the
        # grain dimensions so atoms do not exceed the original boundary conditions
        updated_right_grain = deepcopy(GB.right_grain)
        updated_right_grain[:, 1] = (
            updated_right_grain[:, 1] + dy) % GB.grain_ydim
        updated_right_grain[:, 2] = (
            updated_right_grain[:, 2] + dz) % GB.grain_zdim

        return np.vstack((GB.left_grain, updated_right_grain))

    def merge_slices(self, GB1, GB2):
        """
        Given two GB systems, merge them by cutting them at the same location and
        swapping one slice with the same slice in the other system.
        :param np.ndarray GB1: The positions of atoms for parent 1
        :param np.ndarray GB2: The positions of atoms for parent 2
        :return np.ndarray: Atom positions after merging the slices.
        """

        slice_pos = GB.gb_thickness * (0.25 + 0.5*np.random.rand())
        pos1 = GB1[GB1[0] < slice_pos]
        pos2 = GB2[GB2[0] >= slice_pos]
        new_positions = np.vstack((pos1, pos2))

        return new_positions

    def remove_atoms(self, GB, fraction: float):
        """
        Removes _fraction_ of atoms at the GB plane
        :param np.ndarray GB: The positions of the atoms in the parent.
        :param float fraction: The fraction of atoms in the GB plane to remove
        :return np.ndarray: Atom positions after atom removal
        """
        if fraction < 0 or fraction > 0.25:
            raise ValueError("Invalid value for fraction ("
                             f"{fraction=}). Must be between 0 and .25")
        if fraction == 0:
            return GB
        else:
            raise NotImplementedError(
                "This mutator has not been implemented yet")

    def insert_atoms(self, GB, fraction: float):
        """
        Inserts _fraction_ atoms in the GB at lattice sites. Empty sites are assumed to
        have a resolution of 1 angstrom.
        :param np.ndarray GB: The positions of the atoms in the parent.
        :param float fraction: The fraction of empty lattice sites to fill
            (0 <= fraction <= 0.25)
        :return np.ndarray: Atom positions after atom insertion.
        """
        if fraction < 0 or fraction > 0.25:
            raise ValueError("Invalid value for fraction ("
                             f"{fraction=}). Must be between 0 and 0.25")
        if fraction == 0:
            return GB
        else:
            raise NotImplementedError(
                "This mutator has not been implemented yet")

    def displace_along_soft_modes(self, GB):
        """
        Displace atoms along soft phonon modes.
        :param np.ndarray GB: The positions of the atoms in the parent.
        :return np.ndarray: Atom positions after displacement.
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
