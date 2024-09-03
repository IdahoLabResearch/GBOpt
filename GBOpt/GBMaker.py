import math

import numpy as np
from scipy.spatial import transform

from GBOpt.UnitCell import UnitCell


class GBMaker:

    def __init__(self, lattice_parameter, structure, gb_thickness,
                 misorientation, repeat_factor=5, gb_id=0):
        """
        Class to create a GB structure based on user defined parameters.
        The GB normal is aligned along the x-axis.
        :param float lattice_parameter: Crystal lattice parameter (A)
        :param str structure: Crystal structure string ['fcc', 'bcc', 'sc', 'diamond',
            'fluorite', 'zincblende']
        :param float gb_thickness: The width of the GB region (A)
        :param np.ndarray misorientation: Misorientation angles (alpha, beta, gamma,
            theta,phi)
        """
        self.grain_xdim = 60  # Default value, can be changed in the future
        self.vacuum_thickness = 10   # Default value, can be changed in the future
        self.ID = gb_id
        self.a = lattice_parameter
        self.misorientation = np.asarray(misorientation)
        self.gb_thickness = gb_thickness
        self.structure = structure
        self.repeat_factor = repeat_factor
        self.spacing = self.get_interplanar_spacing()
        self.grain_ydim = self.repeat_factor*self.spacing['y']
        self.grain_zdim = self.repeat_factor*self.spacing['z']

        self.unit_cell = UnitCell()
        self.unit_cell.init_by_structure(self.structure, self.a)
        self.radius = lattice_parameter * self.unit_cell.radius
        self.generate_left_grain()
        self.generate_right_grain()
        self.box_dims = np.array(
            [
                [-self.vacuum_thickness, 2*self.grain_xdim + self.vacuum_thickness],
                [0, self.grain_ydim],
                [0, self.grain_zdim]
            ]
        )

    def get_interplanar_spacing(self):
        """Currently only works for CSL with just misorientation,
        inclination dependence needs to be implemented"""
        R = transform.Rotation.from_euler(
            'ZXZ', self.misorientation, degrees=False).as_matrix()
        print(R)

        # Reciprocal lattice vectors for FCC
        # TODO: Make this an attribute related to the unit cell!
        G_vectors = self.unit_cell.reciprocal
        # 1 / self.a * \
        #     np.array([[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
        #              [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]])

        # Transform reciprocal lattice vectors
        G_prime_vectors = np.dot(G_vectors, R).T
        # Calculate lattice plane spacing along x, y, z
        spacing = {}
        for i, axis in enumerate(['x', 'y', 'z']):
            projection = np.abs(G_prime_vectors[i])
            spacing[axis] = 1 / projection.min()
        return spacing

    def get_supercell(self, corners):
        cells = corners[:, np.newaxis, :] + \
            self.unit_cell.positions()[np.newaxis, :, :]
        return cells.reshape(-1, 3)

    def get_points_inside_box(self, points, box_dim):
        """
        :param np.ndarray points: points to check
        :param list box_dim: dimensions of box
                             (x_min,y_min,z_min,x_max,y_max,z_max)
        """
        x_min, y_min, z_min, x_max, y_max, z_max = box_dim
        x_slice = points[np.where(np.logical_and(
            points[:, 0] >= x_min, points[:, 0] <= x_max))]
        y_slice = x_slice[np.where(np.logical_and(
            x_slice[:, 1] >= y_min, x_slice[:, 1] < y_max))]
        z_slice = y_slice[np.where(np.logical_and(
            y_slice[:, 2] >= z_min, y_slice[:, 2] <= z_max))]
        return z_slice

    def generate_left_grain(self):
        body_diagonal = np.linalg.norm(
            [self.grain_xdim, self.grain_ydim, self.grain_zdim])
        body_diagonal -= body_diagonal % self.a
        X = np.arange(-body_diagonal, body_diagonal, self.a)

        corners = np.vstack(np.meshgrid(X, X, X)).reshape(3, -1).T
        atoms = self.get_supercell(corners)

        self.left_grain = self.get_points_inside_box(
            atoms,
            [0, 0, 0, self.grain_xdim, self.grain_ydim, self.grain_zdim])

    def generate_right_grain(self):
        body_diagonal = np.linalg.norm(
            [2*self.grain_xdim, self.grain_ydim, self.grain_zdim])
        body_diagonal -= body_diagonal % self.a
        X = np.arange(-body_diagonal, body_diagonal, self.a)

        corners = np.vstack(np.meshgrid(X, X, X)).reshape(3, -1).T
        atoms = self.get_supercell(corners)

        R = transform.Rotation.from_euler(
            'ZXZ', self.misorientation, degrees=False).as_matrix()
        atoms = np.dot(atoms, R)

        atoms += np.amax(self.left_grain[:, 0])
        self.right_grain = self.get_points_inside_box(
            atoms,
            [self.grain_xdim, 0, 0, 2*self.grain_xdim, self.grain_ydim + 1, self.grain_zdim + 1])

    def write_lammps(self, positions, box_sizes, file_name):
        """
        Writes the atom positions with the given box dimensions to a LAMMPS input file.
        :param np.ndarray positions: The positions of the atoms.
        :param np.ndarray box_sizes: 3x2 array containing the min and max dimensions for
            each of the x, y, and z dimensions
        :param str filename: The filename to save the data
        """

        # Write LAMMPS data file
        with open(file_name, 'w') as fdata:
            # First line is a comment line
            fdata.write('Crystalline Cu atoms\n\n')

            # --- Header ---#
            # Specify number of atoms and atom types
            fdata.write('{} atoms\n'.format(len(positions)))
            fdata.write('{} atom types\n'.format(1))
            # Specify box dimensions
            fdata.write('{} {} xlo xhi\n'.format(
                box_sizes[0][0], box_sizes[0][1]))
            fdata.write('{} {} ylo yhi\n'.format(
                box_sizes[1][0], box_sizes[1][1]))
            fdata.write('{} {} zlo zhi\n'.format(
                box_sizes[2][0], box_sizes[2][1]))
            fdata.write('\n')

            # Atoms section
            fdata.write('Atoms\n\n')

            # Write each position
            for i, pos in enumerate(positions):
                fdata.write('{} 1 {} {} {}\n'.format(i+1, *pos))

    def run_optimization(self):
        print(self.ID)


if __name__ == '__main__':
    theta = math.radians(5)
    G = GBMaker(lattice_parameter=3.61, gb_thickness=0.0,
                misorientation=[theta, 0, 0], repeat_factor=4)
    G.run_optimization()
