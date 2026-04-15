# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import filecmp
import math
import os
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from GBOpt.Atom import Atom, AtomValueError
from GBOpt.GBMaker import GBMaker, GBMakerTypeError, GBMakerValueError
from GBOpt.UnitCell import UnitCell


class TestGBMaker(unittest.TestCase):
    def setUp(self):
        # Common test setup
        self.a0 = 3.61
        self.structure = "fcc"
        self.atom_types = "Cu"
        self.gb_thickness = 10.0
        theta = math.radians(36.869898)
        self.misorientation = np.array(
            [theta, 0.0, 0.0, 0.0, -theta / 2.0])
        self.repeat_factor = 6
        self.x_dim_min = 60.0
        self.vacuum = 10.0
        self.gb_id = 0
        self.interaction_distance = 10
        self.gbm = GBMaker(self.a0, self.structure, self.gb_thickness,
                           self.misorientation, self.atom_types,
                           repeat_factor=self.repeat_factor, x_dim_min=self.x_dim_min,
                           vacuum=self.vacuum,
                           interaction_distance=self.interaction_distance,
                           gb_id=self.gb_id)

    def test_initialization(self):
        # Test that all values are set correctly at initialization
        self.assertEqual(self.gbm.a0, self.a0)
        self.assertEqual(self.gbm.structure, self.structure)
        self.assertEqual(self.gbm.gb_thickness, self.gb_thickness)
        np.testing.assert_array_equal(
            self.gbm.misorientation, self.misorientation)
        self.assertEqual(self.gbm.repeat_factor, [
                         self.repeat_factor, self.repeat_factor])
        self.assertEqual(self.gbm.x_dim_min, self.x_dim_min)
        self.assertEqual(self.gbm.vacuum_thickness, self.vacuum)
        self.assertEqual(self.gbm.id, self.gb_id)
        unit_cell = UnitCell()
        unit_cell.init_by_structure(self.structure, self.a0, self.atom_types)
        self.assertTrue(repr(self.gbm.unit_cell) == repr(unit_cell))
        self.assertEqual(self.gbm.interaction_distance, self.interaction_distance)
        self.assertTrue(self.gbm.whole_system.shape[0] > 0)
        self.assertTrue(isinstance(self.gbm.whole_system, np.ndarray))
        self.assertIsNotNone(self.gbm.left_grain)
        self.assertIsNotNone(self.gbm.right_grain)
        self.assertEqual(len(self.gbm.whole_system[0]), 4)
        left_grain = self.gbm.left_grain
        right_grain = self.gbm.right_grain
        system = self.gbm.whole_system
        self.assertEqual(
            left_grain.shape[0] + right_grain.shape[0], system.shape[0])

    # Tests for invalid values
    def test_invalid_a0_type(self):
        with self.assertRaises(GBMakerTypeError):
            self.gbm.a0 = "invalid"

    def test_invalid_a0_value(self):
        with self.assertRaises(GBMakerValueError):
            self.gbm.a0 = -5.0

    def test_invalid_misorientation_length(self):
        with self.assertRaises(GBMakerValueError):
            self.gbm.misorientation = np.array([0.1, 0.2])

    def test_invalid_misorientation_type(self):
        with self.assertRaises(GBMakerTypeError):
            self.gbm.misorientation = "invalid"

    def test_invalid_structure_type(self):
        with self.assertRaises(GBMakerTypeError):
            self.gbm.structure = 123

    def test_invalid_structure_value(self):
        with self.assertRaises(GBMakerValueError):
            self.gbm.structure = "invalid_structure"

    def test_invalid_values_raise_exceptions(self):
        with self.assertRaises(GBMakerValueError):
            GBMaker(-1.0, self.structure,
                    self.gb_thickness, self.misorientation, self.atom_types)
        with self.assertRaises(GBMakerValueError):
            GBMaker(self.a0, "invalid_structure",
                    self.gb_thickness, self.misorientation, self.atom_types)
        with self.assertRaises(GBMakerValueError):
            GBMaker(self.a0, self.structure,
                    self.gb_thickness, np.array([0.1, 0.2]), self.atom_types)
        with self.assertRaises(GBMakerValueError):
            GBMaker(self.a0, self.structure, -5.0,
                    self.misorientation, self.atom_types)
        with self.assertRaises(AtomValueError):
            GBMaker(self.a0, self.structure, self.gb_thickness,
                    self.misorientation, "Invalid")

    # Tests for additional getters
    def test_additional_getters(self):
        self.assertGreater(self.gbm.y_dim, 0)
        self.assertGreater(self.gbm.z_dim, 0)
        self.assertGreater(self.gbm.radius, 0)

    def test_box_dimensions(self):
        box_dims = self.gbm.box_dims
        self.assertTrue(isinstance(box_dims, np.ndarray))
        self.assertEqual(box_dims.shape, (3, 2))

    # Tests for public methods
    def test_get_supercell(self):
        corners = np.array([[0, 0, 0], [10, 10, 10]])
        supercell = self.gbm.get_supercell(corners)
        self.assertTrue(isinstance(supercell, np.ndarray))
        self.assertGreater(supercell.shape[0], 0)

    def test_update_spacing(self):
        initial_spacing = self.gbm.spacing
        theta = math.radians(22.619865)
        self.gbm.misorientation = np.array([theta, 0.0, 0.0, 0.0, -theta / 2.0])
        self.assertNotEqual(initial_spacing, self.gbm.spacing)

    def test_write_lammps(self):
        atoms = self.gbm.whole_system
        box_sizes = self.gbm.box_dims
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            self.gbm.write_lammps(temp_file.name, atoms, box_sizes)
            with open(temp_file.name, "r") as f:
                content = f.readlines()
            self.assertGreater(len(content), 0)
            self.assertIn("atoms", content[2].lower())
            self.assertIn("atom types", content[3].lower())

    # Tests for setters
    def test_box_dimensions_after_updates(self):
        original_box_dims = self.gbm.box_dims.copy()
        self.gbm.x_dim_min = 80.0
        self.assertFalse(np.allclose(original_box_dims, self.gbm.box_dims))

    def test_misorientation_spacing(self):
        original_spacing = self.gbm.spacing.copy()
        theta = math.radians(22.619865)
        self.gbm.misorientation = np.array([theta, 0.0, 0.0, 0.0, -theta / 2.0])
        self.assertNotEqual(original_spacing, self.gbm.spacing)

    def test_setters_update_properties(self):
        self.gbm.interaction_distance = 6
        self.assertEqual(self.gbm.interaction_distance, 6)
        self.assertGreater(self.gbm.y_dim, 2*self.gbm.interaction_distance)
        self.assertGreater(self.gbm.z_dim, 2*self.gbm.interaction_distance)

        self.gbm.a0 = 4.0
        self.assertEqual(self.gbm.a0, 4.0)

        self.gbm.structure = "bcc"
        self.assertEqual(self.gbm.structure, "bcc")

        with self.assertRaises(GBMakerValueError):
            self.gbm.structure = "fluorite"

        self.gbm.gb_thickness = 12.0
        self.assertEqual(self.gbm.gb_thickness, 12.0)

        with self.assertWarns(UserWarning):
            self.gbm.misorientation = np.array([0.3, 0.4, 0.5, 0.6, 0.7])

        theta = math.radians(90-36.869898)
        new_misorientation = np.array([theta, 0.0, 0.0, 0.0, -theta / 2.0])
        self.gbm.misorientation = new_misorientation
        np.testing.assert_array_equal(
            self.gbm.misorientation, new_misorientation)

        with self.assertRaises(GBMakerValueError):
            self.gbm.repeat_factor = [-2, -1]

        self.gbm.repeat_factor = 6
        self.assertEqual(self.gbm.repeat_factor, [6, 6])

        with self.assertWarns(UserWarning):
            self.gbm.repeat_factor = 1
            self.gbm.repeat_factor = [1, 1]

        with self.assertRaises(GBMakerValueError):
            self.gbm.repeat_factor = [1.5, 2.0]

        self.gbm.x_dim_min = 80.0
        self.assertEqual(self.gbm.x_dim_min, 80.0)

        self.gbm.vacuum_thickness = 15.0
        self.assertEqual(self.gbm.vacuum_thickness, 15.0)

        self.gbm.id = 2
        self.assertEqual(self.gbm.id, 2)

    def test_thin_thick_box_dimensions(self):
        # Thin box
        self.gbm.x_dim_min = 5.0
        self.assertGreaterEqual(self.gbm.box_dims[0][1], 5.0)

        # Thick box
        self.gbm.vacuum_thickness = 50.0
        self.assertGreater(self.gbm.box_dims[0][1], 50.0)

    # Tests for private methods
    def test_approximate_rotation_matrix_as_int(self):
        rotation_matrix = np.array([[0.70710678, 0.5, 0.5],
                                    [0.70710678, -0.5, -0.5],
                                    [0.0, 0.70710678, -0.70710678]])

        approx_matrix = self.gbm._GBMaker__approximate_rotation_matrix_as_int(
            rotation_matrix)

        expected_matrix = np.array([[7,  5,  5],
                                    [7, -5, -5],
                                    [0,  1, -1]])

        np.testing.assert_array_equal(approx_matrix, expected_matrix)

    def test_calculate_periodic_spacing_logic(self):
        with patch.object(GBMaker, "_GBMaker__calculate_periodic_spacing", return_value={"x": 5.0, "y": 10.0, "z": 15.0}):
            self.gbm.update_spacing()
            self.assertEqual(self.gbm.spacing["x"], 5.0)
            self.assertEqual(self.gbm.spacing["y"], 10.0)
            self.assertEqual(self.gbm.spacing["z"], 15.0)

    # Tests for warnings
    def test_repeat_factor_warning(self):
        with self.assertWarns(UserWarning):
            gbm = GBMaker(self.a0, self.structure, self.gb_thickness,
                          self.misorientation, self.atom_types, repeat_factor=[2, 3],
                          x_dim_min=self.x_dim_min, vacuum=self.vacuum,
                          interaction_distance=30,
                          gb_id=self.gb_id)

        with self.assertWarns(UserWarning):
            gbm.interaction_distance = 32

    # Additional tests
    # Output data file format is as expected.
    def test_lammps_file_formatting(self):
        atoms = np.array([("Cu", 0.0, 0.0, 0.0), ("H", 1.0, 1.0, 1.0)],
                         dtype=Atom.atom_dtype)
        box_sizes = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            self.gbm.write_lammps(temp_file.name, atoms, box_sizes)
            with open(temp_file.name, "r") as f:
                content = f.readlines()
            self.assertEqual(content[2].strip(), "2 atoms")
            self.assertEqual(content[3].strip(), "2 atom types")

        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            self.gbm.write_lammps(temp_file.name, atoms, box_sizes, type_as_int=False)
            with open(temp_file.name, "r") as f:
                content = f.readlines()

            self.assertEqual(content[8].strip(), "Atom Type Labels")
            self.assertEqual(content[10].strip(), "1 Cu")
            self.assertEqual(content[11].strip(), "2 H")

    def test_lammps_file_formatting_with_charge(self):
        atoms = np.array(
            [
                ('U', 0.0, 0.0, 0.0),
                ('O', 0.25, 0.25, 0.25),
                ('O', 0.25, 0.25, 0.75)
            ],
            dtype=Atom.atom_dtype
        )
        charges = {'U': 2.4, 'O': -1.2}
        box_sizes = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            self.gbm.write_lammps(temp_file.name, atoms, box_sizes, charges=charges)
            with open(temp_file.name, 'r') as f:
                content = f.readlines()
            self.assertEqual(content[2].strip(), '3 atoms')
            self.assertEqual(content[3].strip(), '2 atom types')
            self.assertEqual(content[15].strip(),
                             '1 U 2.400000 0.000000 0.000000 0.000000')
            self.assertEqual(content[16].strip(),
                             '2 O -1.200000 0.250000 0.250000 0.250000')
            self.assertEqual(content[17].strip(),
                             '3 O -1.200000 0.250000 0.250000 0.750000')

        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            self.gbm.write_lammps(temp_file.name, atoms, box_sizes,
                                  charges=charges, type_as_int=True)
            with open(temp_file.name, 'r') as f:
                content = f.readlines()
            self.assertEqual(content[2].strip(), '3 atoms')
            self.assertEqual(content[3].strip(), '2 atom types')
            self.assertEqual(content[10].strip(),
                             '1 2 2.400000 0.000000 0.000000 0.000000')
            self.assertEqual(content[11].strip(),
                             '2 1 -1.200000 0.250000 0.250000 0.250000')
            self.assertEqual(content[12].strip(),
                             '3 1 -1.200000 0.250000 0.250000 0.750000')

    def test_data_integrity_in_gb(self):
        left_grain = self.gbm.left_grain
        right_grain = self.gbm.right_grain
        system = self.gbm.whole_system

        self.assertEqual(
            left_grain.shape[0] + right_grain.shape[0], system.shape[0])

    def test_inconsistent_data_raises_exceptions(self):
        with self.assertRaises(GBMakerValueError):
            GBMaker(self.a0, self.structure, -5.0,
                    self.misorientation, self.atom_types)  # Negative thickness

    def test_single_grain_creation(self):
        gbm_single = GBMaker(3.54, "fcc", 5.0,
                             np.array([0, 0, 0, 0, 0]), "Cu",
                             repeat_factor=6, x_dim_min=10,
                             vacuum=10,
                             interaction_distance=5
                             )
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            gbm_single.write_lammps(temp_file.name)
            self.assertTrue(
                filecmp.cmp(temp_file.name, "./tests/gold/fcc_Cu.txt", shallow=False))


class TestGBMakerIntRotationHelpers(unittest.TestCase):
    def setUp(self):
        a0 = 3.61
        theta = math.radians(36.868698)
        misorientation = np.array([theta, 0.0, 0.0, 0.0, -theta / 2.0])
        self.gbm = GBMaker(a0, "fcc", 10.0, misorientation, "Cu")

    # __reduce_integer_row
    def test_reduce_integer_row_basic(self):
        row = np.array([4, 6, 2])
        result = self.gbm._GBMaker__reduce_integer_row(row)
        np.testing.assert_array_equal(result, np.array([2, 3, 1]))

    def test_reduce_integer_row_already_reduced(self):
        row = np.array([1, 2, 3])
        result = self.gbm._GBMaker__reduce_integer_row(row)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_reduce_integer_row_all_zeros(self):
        row = np.array([0, 0, 0])
        result = self.gbm._GBMaker__reduce_integer_row(row)
        np.testing.assert_array_equal(result, np.array([0, 0, 0]))

    def test_reduce_integer_row_with_negatives(self):
        row = np.array([-4, 6, -2])
        result = self.gbm._GBMaker__reduce_integer_row(row)
        np.testing.assert_array_equal(result, np.array([-2, 3, -1]))

    # __row_angle_error_deg
    def test_row_angle_error_parallel(self):
        ref = np.array([1.0, 0.0, 0.0])
        cand = np.array([5, 0, 0])
        err = self.gbm._GBMaker__row_angle_error_deg(ref, cand)
        self.assertAlmostEqual(err, 0.0, places=10)

    def test_row_angle_error_perpendicular(self):
        ref = np.array([1.0, 0.0, 0.0])
        cand = np.array([0, 1, 0])
        err = self.gbm._GBMaker__row_angle_error_deg(ref, cand)
        self.assertAlmostEqual(err, 90.0, places=10)

    def test_row_angle_error_antiparallel(self):
        ref = np.array([1.0, 0.0, 0.0])
        cand = np.array([-1, 0, 0])
        err = self.gbm._GBMaker__row_angle_error_deg(ref, cand)
        self.assertAlmostEqual(err, 180, places=10)

    def test_row_angle_error_zero_vector(self):
        ref = np.array([1.0, 0.0, 0.0])
        cand = np.array([0, 0, 0])
        err = self.gbm._GBMaker__row_angle_error_deg(ref, cand)
        self.assertEqual(err, 180.0)

    # __approximate_rotation_matrix_as_int - row error tolerance
    def test_approx_matrix_sigma5_within_tolerance(self):
        """Sigma5 [001] 36.87 - all rows must be within 0.5 of the float matrix."""
        from scipy.spatial.transform import Rotation
        theta = math.radians(36.869898)
        R = Rotation.from_euler("z", theta).as_matrix()
        approx = self.gbm._GBMaker__approximate_rotation_matrix_as_int(R)
        for ref_row, approx_row in zip(R, approx.astype(float)):
            err = self.gbm._GBMaker__row_angle_error_deg(ref_row, approx_row)
            self.assertLessEqual(err, 0.5)


class TestGBMakerTriclinic(unittest.TestCase):
    def setUp(self):
        a0 = 3.61
        theta = math.radians(36.869898)
        misorientation = np.array([theta, 0.0, 0.0, 0.0, -theta / 2.0])
        self.gbm = GBMaker(a0, "fcc", 10.0, misorientation, "Cu",
                           repeat_factor=6, x_dim_min=60.0, vacuum=10.0,
                           interaction_distance=10)

    def test_triclinic_writes_tilt_line(self):
        with tempfile.NamedTemporaryFile(delete=True) as f:
            self.gbm.write_lammps(f.name, triclinic=True)
            with open(f.name) as fread:
                content = fread.read()
        self.assertIn("xy xz yz", content)

    def test_non_triclinic_no_tilt_line(self):
        with tempfile.NamedTemporaryFile(delete=True) as f:
            self.gbm.write_lammps(f.name)
            with open(f.name) as fread:
                content = fread.read()
        self.assertNotIn("xy xz yz", content)

    def test_csl_boundary_zero_tilt(self):
        # Sigma5 is a CSL boundary - period vectors lie exactly on axis directions, so
        # all tilt factors should be negligibly small
        xy, xz, yz, _ = self.gbm._GBMaker__get_triclinic_params()
        self.assertAlmostEqual(xy, 0.0, places=3)
        self.assertAlmostEqual(xz, 0.0, places=3)
        self.assertAlmostEqual(yz, 0.0, places=3)

    def test_triclinic_gbmanipulator_reads_back(self):
        from GBOpt.GBManipulator import Parent
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dat", mode="w") as f:
            fname = f.name
        try:
            self.gbm.write_lammps(fname, triclinic=True)
            # Should not raise - GBManipulator must parse the xy xz yz line cleanly
            parent = Parent(fname, unit_cell=self.gbm.unit_cell,
                            gb_thickness=self.gbm.gb_thickness)
            self.assertIsNotNone(parent)
        finally:
            os.unlink(fname)


class TestGBMakerNonCommutingBoundaries(unittest.TestCase):
    """Regression tests for boundaries where R_incl and R_mis do not commute.

    When R_incl = Rz(phi) @ Ry(theta) is a pure z-rotation (theta=0, as for [001]
    or [110] boundary normals) and R_mis is also a z-rotation (e.g. symmetric tilt
    about [001]), the two matrices commute and the order R_mis @ R_incl vs R_incl @
    R_mis is irrelevant.  For boundaries whose normal has a nonzero z-component (e.g.
    [111] or [112]), theta != 0 and the matrices generally do NOT commute.
    """

    def setUp(self):
        self.a0 = 3.61
        self.structure = "fcc"
        self.atom_types = "Cu"
        self.gb_thickness = self.a0

        # Sigma3 (111) coherent twin boundary.
        # Derived from orientation matrices (rows = crystal directions for lab x,y,z)
        # as given by Olmsted et al. doi: 10.1016/j.actamat.2009.04.007.
        #   P = [[2,2,2], [1,-1,0], [1,1,-2]]
        #   Q = [[2,2,2], [-1,1,0], [-1,-1,2]]
        # Misorientation: 180 deg rotation about [1,1,1] -> ZXZ = [3pi/4, arccos(-1/3), pi/4]
        # Inclination: boundary normal [1,1,1]           -> theta=pi/4, phi=-arctan(1/sqrt(2))
        self.sigma3_111_180deg = np.array(
            [
                3 * np.pi / 4,
                np.arccos(-1 / 3),
                np.pi / 4,
                np.pi / 4,
                -np.arctan(1 / np.sqrt(2)),
            ]
        )

        # Same physical boundary, alternative representation: 60 deg about [1,1,1].
        # ZXZ Euler angles for 60 deg rotation about [1,1,1]:
        #   alpha=arctan(2), beta=arccos(2/3), gamma=arctan(-1/2)
        self.sigma3_111_60deg = np.array(
            [
                np.arctan(2),
                np.arccos(2 / 3),
                np.arctan(-1 / 2),
                np.pi / 4,
                -np.arctan(1 / np.sqrt(2)),
            ]
        )

        # Sigma7 (111) twist boundary.
        # Rotation angle theta = 2*arctan(sqrt(3)/5); exact: cos(theta)=11/14, sin(theta)=5*sqrt(3)/14.
        # R_mis = [[6/7,-2/7,3/7],[3/7,6/7,-2/7],[-2/7,3/7,6/7]]
        # ZXZ Euler angles: alpha=arctan(3/2), beta=arccos(6/7), gamma=arctan(-2/3)
        # Same boundary-plane inclination as Sigma3 (111): theta=pi/4, phi=-arctan(1/sqrt(2)).
        # Note: the right-grain y-direction period (row [2,11,-13], norm=7*sqrt(6)~17.15*a0)
        # exceeds the default 15*a0 threshold, so a non-periodic warning is expected.
        self.sigma7_111 = np.array(
            [
                np.arctan(3 / 2),
                np.arccos(6 / 7),
                np.arctan(-2 / 3),
                np.pi / 4,
                -np.arctan(1 / np.sqrt(2)),
            ]
        )

    def _make_gb(self, misorientation, **kwargs):
        """Construct a small GB with fast defaults."""
        defaults = dict(repeat_factor=2, x_dim_min=50, interaction_distance=5)
        defaults.update(kwargs)
        return GBMaker(
            self.a0,
            self.structure,
            self.gb_thickness,
            misorientation,
            self.atom_types,
            **defaults,
        )

    # ------------------------------------------------------------------
    # Sigma3 (111) -- 180 deg misorientation representation
    # ------------------------------------------------------------------

    def test_sigma3_111_construction_succeeds(self):
        """Regression: Sigma3 (111) must not raise ValueError during construction."""
        gbm = self._make_gb(self.sigma3_111_180deg)
        self.assertGreater(gbm.left_grain.shape[0], 0)
        self.assertGreater(gbm.right_grain.shape[0], 0)
        self.assertEqual(
            gbm.left_grain.shape[0] + gbm.right_grain.shape[0],
            gbm.whole_system.shape[0],
        )

    def test_sigma3_111_no_non_periodic_warning(self):
        """Sigma3 (111) is a well-defined CSL: no non-periodic boundary warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._make_gb(self.sigma3_111_180deg)
        non_periodic = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and "non-periodic" in w.message.args[0].lower()
        ]
        self.assertEqual(
            len(non_periodic), 0, f"Unexpected non-periodic warning(s): {non_periodic}"
        )

    def test_sigma3_111_spacing(self):
        """Sigma3 (111) spacings must match the expected (111) CSL periodicities."""
        gbm = self._make_gb(self.sigma3_111_180deg)
        s = gbm.spacing
        # Boundary normal [1,1,1]:            period = a0*sqrt(3)
        # In-plane direction [-1,2,-1]/[1,-2,1]: period = a0*sqrt(6)
        # In-plane direction [-1,0,1]/[1,0,-1]:  period = a0*sqrt(2)
        self.assertAlmostEqual(s["x"]["left"], self.a0 * np.sqrt(3), places=5)
        self.assertAlmostEqual(s["x"]["right"], self.a0 * np.sqrt(3), places=5)
        self.assertAlmostEqual(s["y"], self.a0 * np.sqrt(6), places=5)
        self.assertAlmostEqual(s["z"], self.a0 * np.sqrt(2), places=5)

    def test_sigma3_111_x_dim_reasonable(self):
        """x_dim must follow from the CSL x-spacing, not be thousands of Angstroms."""
        gbm = self._make_gb(self.sigma3_111_180deg, x_dim_min=50)
        # With spacing_x = a0*sqrt(3) ~ 6.25 A and x_dim_min=50,
        # each grain is ceil(50/6.25)*6.25 ~ 50 A, so x_dim ~ 100 A.
        # Before the fix, right_x was thousands of Angstroms.
        x_spacing = self.a0 * np.sqrt(3)
        expected_grain_x = math.ceil(50 / x_spacing) * x_spacing
        self.assertAlmostEqual(gbm.x_dim, 2 * expected_grain_x, places=5)

    def test_sigma3_111_via_setter(self):
        """Same regression applies when misorientation is changed via the setter."""
        theta = math.radians(36.869898)
        gbm = self._make_gb(
            np.array([theta, 0.0, 0.0, 0.0, -theta / 2.0]), repeat_factor=(2, 3))
        gbm.misorientation = self.sigma3_111_180deg
        # Confirm the grain was actually rebuilt, not just the spacing dict updated.
        self.assertGreater(gbm.left_grain.shape[0], 0)
        self.assertGreater(gbm.right_grain.shape[0], 0)
        s = gbm.spacing
        self.assertAlmostEqual(s["x"]["left"], self.a0 * np.sqrt(3), places=5)
        self.assertAlmostEqual(s["x"]["right"], self.a0 * np.sqrt(3), places=5)
        self.assertAlmostEqual(s["y"], self.a0 * np.sqrt(6), places=5)
        self.assertAlmostEqual(s["z"], self.a0 * np.sqrt(2), places=5)

    # ------------------------------------------------------------------
    # Sigma3 (111) -- 60 deg misorientation representation
    # ------------------------------------------------------------------

    def test_sigma3_111_60deg_construction_succeeds(self):
        """Sigma3 (111) described as 60 deg about [111] must also construct cleanly."""
        gbm = self._make_gb(self.sigma3_111_60deg)
        self.assertGreater(gbm.left_grain.shape[0], 0)
        self.assertGreater(gbm.right_grain.shape[0], 0)

    def test_sigma3_111_60deg_no_non_periodic_warning(self):
        """60 deg Sigma3 (111) representation: no non-periodic boundary warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._make_gb(self.sigma3_111_60deg)
        non_periodic = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and "non-periodic" in w.message.args[0].lower()
        ]
        self.assertEqual(
            len(non_periodic), 0, f"Unexpected non-periodic warning(s): {non_periodic}"
        )

    def test_sigma3_111_60deg_spacing(self):
        """60 deg Sigma3 (111): spacings must match the same (111) CSL periodicities."""
        gbm = self._make_gb(self.sigma3_111_60deg)
        s = gbm.spacing
        self.assertAlmostEqual(s["x"]["left"], self.a0 * np.sqrt(3), places=5)
        self.assertAlmostEqual(s["x"]["right"], self.a0 * np.sqrt(3), places=5)
        self.assertAlmostEqual(s["y"], self.a0 * np.sqrt(6), places=5)
        self.assertAlmostEqual(s["z"], self.a0 * np.sqrt(2), places=5)

    # ------------------------------------------------------------------
    # Self-validating guards (Gap 1 & 2)
    # ------------------------------------------------------------------

    @staticmethod
    def _Rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

    @staticmethod
    def _Rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

    @staticmethod
    def _Ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

    def _decompose(self, mis):
        """Return (R_mis, R_incl) from a 5-element misorientation array."""
        alpha, beta, gamma, theta, phi = mis
        R_mis = self._Rz(alpha) @ self._Rx(beta) @ self._Rz(gamma)
        R_incl = self._Rz(phi) @ self._Ry(theta)
        return R_mis, R_incl

    def test_matrices_do_not_commute(self):
        """Guard: sigma3_111_180deg must produce genuinely non-commuting R_mis and R_incl.

        If the two matrices happened to commute, the spacing tests would pass even
        with the wrong multiplication order, defeating their value as regressions.
        """
        R_mis, R_incl = self._decompose(self.sigma3_111_180deg)
        self.assertFalse(
            np.allclose(R_mis @ R_incl, R_incl @ R_mis),
            "sigma3_111_180deg R_mis and R_incl must not commute; "
            "if they did, the wrong multiplication order would not be detected.",
        )

    def test_spacing_uses_correct_matrix_order(self):
        """The right-grain periodic spacing must come from R_incl @ R_mis, not R_mis @ R_incl.

        For Sigma3 (111), row 0 of R_incl @ R_mis is [1,1,1]/sqrt(3), giving
        x-spacing a0*sqrt(3).  The reversed product R_mis @ R_incl has a different
        row 0 with irrational mixing of sqrt(2)/sqrt(3)/sqrt(6), so it does NOT
        simplify to [1,1,1]/sqrt(3).  This test documents the invariant directly,
        independent of GBMaker internals.
        """
        R_mis, R_incl = self._decompose(self.sigma3_111_180deg)
        correct_row0 = (R_incl @ R_mis)[0]
        self.assertTrue(
            np.allclose(correct_row0, np.array([1, 1, 1]) / np.sqrt(3)),
            f"Row 0 of R_incl @ R_mis should be [1,1,1]/sqrt(3), got {correct_row0}",
        )
        wrong_row0 = (R_mis @ R_incl)[0]
        self.assertFalse(
            np.allclose(wrong_row0, np.array([1, 1, 1]) / np.sqrt(3)),
            "Row 0 of R_mis @ R_incl must NOT equal [1,1,1]/sqrt(3); "
            "if it did, the wrong order would be indistinguishable from the correct one.",
        )

    def test_decompose_matches_gbmaker(self):
        """_decompose must mirror GBMaker's internal rotation construction.

        Two independent checks, one per matrix:

        R_incl: for (111) inclination, Rz(phi) @ Ry(theta) must place [1,1,1]/sqrt(3)
        in row 0.  A wrong R_incl order (e.g. Ry @ Rz) would produce a different row.

        R_mis: for 180-deg rotation about [111], ZXZ Euler [3pi/4, arccos(-1/3), pi/4]
        must yield [[-1/3, 2/3, 2/3], ...].  A wrong Euler series (e.g. ZYZ) would
        produce a different matrix and R_mis[0,0] would not equal -1/3.
        """
        R_mis, R_incl = self._decompose(self.sigma3_111_180deg)
        # R_incl check: row 0 must be [1,1,1]/sqrt(3).
        expected_row0 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        self.assertTrue(
            np.allclose(R_incl[0], expected_row0, atol=1e-10),
            f"_decompose R_incl[0] = {R_incl[0]}; expected [1,1,1]/sqrt(3). "
            "Wrong R_incl construction order would produce a different row.",
        )
        # R_mis check: R_mis[0,0] must be -1/3 for 180-deg rotation about [111].
        self.assertAlmostEqual(
            R_mis[0, 0],
            -1 / 3,
            places=5,
            msg=f"R_mis[0,0]={R_mis[0, 0]}; expected -1/3. "
            "Wrong Euler series in _decompose (e.g. ZYZ vs ZXZ) would produce a different value.",
        )

    # ------------------------------------------------------------------
    # Sigma7 (111) -- different R_mis, same inclination (Gap 3)
    # ------------------------------------------------------------------

    def test_sigma7_111_construction_succeeds(self):
        """Regression: Sigma7 (111) must construct without ValueError."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            gbm = self._make_gb(self.sigma7_111)
        self.assertGreater(gbm.left_grain.shape[0], 0)
        self.assertGreater(gbm.right_grain.shape[0], 0)

    def test_sigma7_111_x_spacing_correct(self):
        """Sigma7 (111): x-spacing (boundary normal [111]) must be a0*sqrt(3) for both grains."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            gbm = self._make_gb(self.sigma7_111)
        s = gbm.spacing
        self.assertAlmostEqual(s["x"]["left"], self.a0 * np.sqrt(3), places=5)
        self.assertAlmostEqual(s["x"]["right"], self.a0 * np.sqrt(3), places=5)

    def test_sigma7_111_non_periodic_warning_expected(self):
        """Sigma7 (111) right-grain y-period (~17.15*a0) exceeds the default 15*a0 threshold.

        The right-grain y-direction row is [2,11,-13] with norm 7*sqrt(6)~17.15*a0,
        which exceeds the 15*a0 threshold.  The z-direction row [-3,-5,8] has norm
        7*sqrt(2)~9.90*a0, which does not.  Exactly one non-periodic warning is expected.
        A non-periodic UserWarning is therefore correct behaviour, not an error.
        """
        # GBMaker.__calculate_periodic_spacing calls warnings.simplefilter("once", ...)
        # without a catch_warnings guard, which populates the per-module
        # __warningregistry__.  catch_warnings restores warnings.filters on exit but
        # does NOT clear __warningregistry__, so if another Sigma7 test ran first the
        # "once" filter would suppress the warning here.  Clear it explicitly.
        mod = sys.modules.get("GBOpt.GBMaker")
        if mod is not None and hasattr(mod, "__warningregistry__"):
            mod.__warningregistry__.clear()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._make_gb(self.sigma7_111)
        non_periodic = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and "non-periodic" in w.message.args[0].lower()
        ]
        self.assertEqual(
            len(non_periodic),
            1,
            f"Expected exactly 1 non-periodic UserWarning for Sigma7 (111), "
            f"got {len(non_periodic)}: {non_periodic}",
        )


if __name__ == "__main__":
    unittest.main()
