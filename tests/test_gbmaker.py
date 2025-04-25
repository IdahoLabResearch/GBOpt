import filecmp
import math
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pytest

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
        self.assertGreaterEqual(self.gbm.x_dim, 80.0)

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

        expected_matrix = np.array([[27720, 19601, 19601],
                                    [27720, -19601, -19601],
                                    [0, 1, -1]])

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

    def test_non_periodic_boundary_warning(self):
        with self.assertWarns(UserWarning):
            self.gbm._GBMaker__approximate_rotation_matrix_as_int(
                np.array([[0.123456789, 0.56789123, -0.918273645], [-0.135792468, 0.246813579, 0.1], [0.159283746, -0.2, 0.1]]))

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

    @pytest.mark.known_bug
    def test_single_grain_creation(self):
        gbm_single = GBMaker(3.54, "fcc", 5.0,
                             np.array([0, 0, 0, 0, 0]), "Cu",
                             repeat_factor=6, x_dim_min=10,
                             vacuum=10,
                             interaction_distance=5
                             )
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            gbm_single.write_lammps(temp_file.name)
            # This test _will_ fail until we address #39
            self.assertFalse(
                filecmp.cmp(temp_file.name, "./tests/gold/fcc_Cu.txt", shallow=False))


if __name__ == "__main__":
    unittest.main()
