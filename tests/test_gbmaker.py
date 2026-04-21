# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import filecmp
import math
import os
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from scipy.spatial import KDTree

from GBOpt.Atom import Atom, AtomValueError
from GBOpt.GBMaker import (
    GBMaker,
    GBMakerTypeError,
    GBMakerValueError,
    wrap_reduced_coordinate,
)
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
        self.assertEqual(self.gbm.epsilon, 1e-10)
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

    def test_invalid_epsilon_type(self):
        with self.assertRaises(GBMakerTypeError):
            self.gbm.epsilon = "invalid"

    def test_invalid_epsilon_value(self):
        with self.assertRaises(GBMakerValueError):
            self.gbm.epsilon = -1e-10

        with self.assertRaises(GBMakerValueError):
            self.gbm.epsilon = 0.0

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

    def test_wrap_reduced_coordinate_preserves_exact_thresholds(self):
        tol = 1e-10
        coords = np.array([tol / 2, tol, 1.0 - tol, 1.0 - tol / 2])
        wrapped = wrap_reduced_coordinate(coords, tol=tol)
        expected = np.array([0.0, tol, 1.0 - tol, 0.0])
        np.testing.assert_allclose(wrapped, expected, atol=1e-15, rtol=0.0)

    def test_wrap_reduced_coordinate_scalar_and_0d_inputs(self):
        for coord in (0.375, np.array(0.375)):
            wrapped = wrap_reduced_coordinate(coord, tol=1e-10)
            self.assertEqual(wrapped.shape, ())
            np.testing.assert_allclose(wrapped, 0.375, atol=1e-15, rtol=0.0)

    def test_wrap_reduced_coordinate_preserves_multidimensional_shape(self):
        coords = np.array([[0.25, 1.2], [-0.2, 2.75]])
        wrapped = wrap_reduced_coordinate(coords, tol=1e-10)
        self.assertEqual(wrapped.shape, coords.shape)
        expected = np.array([[0.25, 0.2], [0.8, 0.75]])
        np.testing.assert_allclose(wrapped, expected, atol=1e-15, rtol=0.0)

    def test_wrap_reduced_coordinate_wraps_multiple_periods_away(self):
        coords = np.array([2.2, -3.7, 4.125, -5.875])
        wrapped = wrap_reduced_coordinate(coords, tol=1e-10)
        expected = np.array([0.2, 0.3, 0.125, 0.125])
        np.testing.assert_allclose(wrapped, expected, atol=1e-15, rtol=0.0)

    def test_wrap_reduced_coordinate_negative_tolerance_raises_gbmaker_error(self):
        with self.assertRaises(GBMakerValueError):
            wrap_reduced_coordinate(np.array([0.25]), tol=-1e-10)

    def test_wrap_reduced_coordinate_non_finite_tolerance_raises_gbmaker_error(self):
        for tol in (np.nan, np.inf, -np.inf):
            with self.subTest(tol=tol):
                with self.assertRaises(GBMakerValueError):
                    wrap_reduced_coordinate(np.array([0.25]), tol=tol)

    def test_reduced_coordinate_tolerance_scales_with_basis_length(self):
        basis_vector = np.array([3.0, 4.0, 0.0])
        self.gbm.epsilon = 2e-8

        tol = self.gbm._GBMaker__reduced_coordinate_tolerance(basis_vector)

        self.assertAlmostEqual(tol, 4e-9, delta=1e-18)

    def test_reduced_coordinate_tolerance_rejects_zero_length_basis_vector(self):
        with self.assertRaises(GBMakerValueError):
            self.gbm._GBMaker__reduced_coordinate_tolerance(np.zeros(3))

    def test_reduced_coordinate_tolerance_rejects_non_finite_basis_vectors(self):
        for basis_vector in (
            np.array([np.nan, 0.0, 0.0]),
            np.array([np.inf, 0.0, 0.0]),
            np.array([-np.inf, 0.0, 0.0]),
            np.array([1.0, np.nan, 2.0]),
            np.array([1.0, np.inf, 2.0]),
        ):
            with self.subTest(basis_vector=basis_vector):
                with self.assertRaises(GBMakerValueError):
                    self.gbm._GBMaker__reduced_coordinate_tolerance(basis_vector)

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
        np.testing.assert_array_equal(self.gbm.misorientation, new_misorientation)

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

    def test_epsilon_custom_init(self):
        gbm = GBMaker(self.a0, self.structure, self.gb_thickness, self.misorientation,
                      self.atom_types, epsilon=1e-5, interaction_distance=3)
        self.assertEqual(gbm.epsilon, 1e-5)

    def test_epsilon_setter(self):
        self.gbm.epsilon = 1e-8
        self.assertEqual(self.gbm.epsilon, 1e-8)

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

    def test_epsilon_controls_boundary_atom_inclusion(self):
        # An atom at x=0.0 with x_min=1e-12 straddles the lower boundary. With
        # epsilon=1e-10: 0.0 >= 1e-12 - 1e-10 =-9.9e-11 -> included. With epsilon=1e-13:
        # 0.0 <= 1e-12 - 1e-13 = 9e-13 -> excluded. Exercises the setter's effect on
        # __get_points_inside_box directly.
        boundary_atom = np.array(
            [("Cu", 0.0, 5.0, 5.0)], dtype=Atom.atom_dtype)
        box_dim = [1e-12, 0.0, 0.0, 10.0, 10.0, 10.0]

        self.gbm.epsilon = 1e-10
        result_large = self.gbm._GBMaker__get_points_inside_box(boundary_atom, box_dim)
        self.assertEqual(len(result_large), 1)

        self.gbm.epsilon = 1e-13  # too narrow; x=0.0 falls velow x_min - epsilon
        result_small = self.gbm._GBMaker__get_points_inside_box(boundary_atom, box_dim)
        self.assertEqual(len(result_small), 0)

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
            self.gbm.write_lammps(temp_file.name, atoms,
                                  box_sizes, type_as_int=False)
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
        self.gbm = GBMaker(a0, "fcc", 10.0, misorientation,
                           "Cu", interaction_distance=3)

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


class TestGBMakerPeriodRowOrientationHelpers(unittest.TestCase):
    def setUp(self):
        a0 = 3.61
        theta = math.radians(36.868698)
        misorientation = np.array([theta, 0.0, 0.0, 0.0, -theta / 2.0])
        self.gbm = GBMaker(
            a0, "fcc", 10.0, misorientation, "Cu", interaction_distance=3.0
        )

    def test_orient_period_rows_preserves_aligned_rows(self):
        R_grain = self.gbm._GBMaker__R_left
        approx = self.gbm._GBMaker__approximate_rotation_matrix_as_int(R_grain)
        oriented = self.gbm._GBMaker__orient_period_rows(R_grain, approx)

        np.testing.assert_array_equal(oriented, approx)
        self.assertFalse(np.shares_memory(approx, oriented))

    def test_orient_period_rows_flips_antiparallel_rows(self):
        R_grain = np.eye(3)
        approx = np.array([[2, 0, 0], [0, -3, 0], [0, 0, -4]])

        oriented = self.gbm._GBMaker__orient_period_rows(R_grain, approx)

        np.testing.assert_array_equal(
            oriented, np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]])
        )
        np.testing.assert_array_equal(approx, np.array(
            [[2, 0, 0], [0, -3, 0], [0, 0, -4]]))


class TestGBMakerScaledPeriodicBasisVector(unittest.TestCase):
    def setUp(self):
        a0 = 3.61
        theta = math.radians(36.868698)
        misorientation = np.array([theta, 0.0, 0.0, 0.0, -theta / 2.0])
        self.gbm = GBMaker(
            a0, "fcc", 10.0, misorientation, "Cu", interaction_distance=3.0
        )

    def test_scaled_periodic_basis_vector_scales_selected_axis_projection(self):
        period_vector = np.array([2.0, -1.0, 0.5])

        scaled = self.gbm._GBMaker__scaled_periodic_basis_vector(
            period_vector, 10.0, 0
        )

        np.testing.assert_allclose(
            scaled, np.array([10.0, -5.0, 2.5]), atol=1e-12, rtol=0.0
        )
        np.testing.assert_allclose(
            period_vector, np.array([2.0, -1.0, 0.5]), atol=0.0, rtol=0.0
        )

    def test_scaled_periodic_basis_vector_accepts_nonzero_projection_on_nonzero_axis(self):
        period_vector = np.array([0.0, 1e-10, 0.0])

        scaled = self.gbm._GBMaker__scaled_periodic_basis_vector(
            period_vector, 10.0, 1
        )

        np.testing.assert_allclose(
            scaled, np.array([0.0, 10.0, 0.0]), atol=1e-12, rtol=0.0
        )

    def test_scaled_periodic_basis_vector_scales_axis_index_two(self):
        period_vector = np.array([1.5, -0.75, 3.0])

        scaled = self.gbm._GBMaker__scaled_periodic_basis_vector(
            period_vector, 12.0, 2
        )

        np.testing.assert_allclose(
            scaled, np.array([6.0, -3.0, 12.0]), atol=1e-12, rtol=0.0
        )

    def test_scaled_periodic_basis_vector_scales_negative_selected_axis_projection(self):
        period_vector = np.array([1.5, -0.75, -3.0])

        scaled = self.gbm._GBMaker__scaled_periodic_basis_vector(
            period_vector, 12.0, 2
        )

        np.testing.assert_allclose(
            scaled, np.array([-6.0, 3.0, 12.0]), atol=1e-12, rtol=0.0
        )

    def test_scaled_periodic_basis_vector_rejects_zero_axis_projection(self):
        with self.assertRaises(GBMakerValueError):
            self.gbm._GBMaker__scaled_periodic_basis_vector(
                np.array([1.0, 2.0, 0.0]), 10.0, 2
            )

    def test_scaled_periodic_basis_vector_rejects_invalid_axis_index(self):
        with self.assertRaises(GBMakerValueError):
            self.gbm._GBMaker__scaled_periodic_basis_vector(
                np.array([1.0, 2.0, 3.0]), 10.0, 3
            )

    def test_scaled_periodic_basis_vector_rejects_non_integral_axis_index_types(self):
        for axis_index in (np.float64(1.0), True):
            with self.subTest(axis_index=axis_index):
                with self.assertRaises(GBMakerValueError):
                    self.gbm._GBMaker__scaled_periodic_basis_vector(
                        np.array([1.0, 2.0, 3.0]), 10.0, axis_index
                    )

    def test_scaled_periodic_basis_vector_accepts_numpy_integer_axis_index(self):
        scaled = self.gbm._GBMaker__scaled_periodic_basis_vector(
            np.array([1.0, 2.0, 3.0]), 10.0, np.int64(1)
        )

        np.testing.assert_allclose(
            scaled, np.array([5.0, 10.0, 15.0]), atol=1e-12, rtol=0.0
        )

    def test_scaled_periodic_basis_vector_rejects_non_positive_box_length(self):
        for box_length in (0.0, -1.0):
            with self.subTest(box_length=box_length):
                with self.assertRaises(GBMakerValueError):
                    self.gbm._GBMaker__scaled_periodic_basis_vector(
                        np.array([1.0, 2.0, 3.0]), box_length, 0
                    )

    def test_scaled_periodic_basis_vector_rejects_non_finite_box_length(self):
        for box_length in (np.nan, np.inf, -np.inf):
            with self.subTest(box_length=box_length):
                with self.assertRaises(GBMakerValueError):
                    self.gbm._GBMaker__scaled_periodic_basis_vector(
                        np.array([1.0, 2.0, 3.0]), box_length, 0
                    )

    def test_scaled_periodic_basis_vector_rejects_boolean_box_length(self):
        for box_length in (True, np.bool_(True)):
            with self.subTest(box_length=box_length):
                with self.assertRaises(GBMakerTypeError):
                    self.gbm._GBMaker__scaled_periodic_basis_vector(
                        np.array([1.0, 2.0, 3.0]), box_length, 0
                    )

    def test_scaled_periodic_basis_vector_rejects_complex_box_length(self):
        with self.assertRaises(GBMakerTypeError):
            self.gbm._GBMaker__scaled_periodic_basis_vector(
                np.array([1.0, 2.0, 3.0]), 10.0 + 0.0j, 0
            )

    def test_scaled_periodic_basis_vector_rejects_invalid_vectors(self):
        invalid_vectors = (
            np.array([np.nan, 1.0, 1.0]),
            np.array([1.0, 2.0]),
        )

        for period_vector in invalid_vectors:
            with self.subTest(period_vector=period_vector):
                with self.assertRaises(GBMakerValueError):
                    self.gbm._GBMaker__scaled_periodic_basis_vector(
                        period_vector, 10.0, 0
                    )

    def test_scaled_periodic_basis_vector_rejects_non_numeric_period_vector(self):
        with self.assertRaises(GBMakerTypeError):
            self.gbm._GBMaker__scaled_periodic_basis_vector(
                np.array(["a", "b", "c"]), 10.0, 0
            )

    def test_scaled_periodic_basis_vector_rejects_non_finite_scaled_vector(self):
        with self.assertRaises(GBMakerValueError):
            self.gbm._GBMaker__scaled_periodic_basis_vector(
                np.array([1e-308, 1e308, 0.0]), 1e308, 0
            )

    def test_scaled_periodic_basis_vector_rejects_boolean_period_vector(self):
        for period_vector in (True, np.bool_(True)):
            with self.subTest(period_vector=period_vector):
                with self.assertRaises(GBMakerTypeError):
                    self.gbm._GBMaker__scaled_periodic_basis_vector(
                        period_vector, 10.0, 0
                    )

    def test_scaled_periodic_basis_vector_rejects_complex_array_like_period_vector(self):
        with self.assertRaises(GBMakerTypeError):
            self.gbm._GBMaker__scaled_periodic_basis_vector(
                np.array([1.0 + 0.0j, 2.0, 3.0]), 10.0, 0
            )

    def test_scaled_periodic_basis_vector_rejects_boolean_array_like_period_vector(self):
        with self.assertRaises(GBMakerTypeError):
            self.gbm._GBMaker__scaled_periodic_basis_vector(
                np.array([True, False, True]), 10.0, 0
            )


class TestGBMakerBoxPeriodicBasis(unittest.TestCase):
    def setUp(self):
        self.gbm = object.__new__(GBMaker)
        self.gbm._GBMaker__epsilon = 1e-10
        self.gbm._GBMaker__y_dim = 12.0
        self.gbm._GBMaker__z_dim = 15.0
        self.gbm._GBMaker__inplane_periodic = (True, False)

    def test_box_periodic_basis_scales_orthogonal_basis_and_zeros_nonperiodic_axis(self):
        primitive_periods = np.array([[0.0, 3.0, 0.0], [0.0, 0.0, 5.0]])

        basis = self.gbm._GBMaker__box_periodic_basis(primitive_periods)

        np.testing.assert_allclose(
            basis,
            np.array([[0.0, 12.0, 0.0], [0.0, 0.0, 0.0]]),
            atol=1e-12,
            rtol=0.0,
        )

    def test_box_periodic_basis_preserves_tilted_components_while_matching_axis_lengths(self):
        self.gbm._GBMaker__inplane_periodic = (True, True)
        primitive_periods = np.array([[2.0, 4.0, -1.0], [-3.0, 1.5, 5.0]])

        basis = self.gbm._GBMaker__box_periodic_basis(primitive_periods)

        np.testing.assert_allclose(
            basis,
            np.array([[6.0, 12.0, -3.0], [-9.0, 4.5, 15.0]]),
            atol=1e-12,
            rtol=0.0,
        )

    def test_box_periodic_basis_rejects_near_zero_selected_axis_projection(self):
        self.gbm._GBMaker__inplane_periodic = (True, True)
        primitive_periods = np.array([[1.0, 1e-12, 0.0], [0.0, 0.0, 5.0]])

        with self.assertRaises(GBMakerValueError):
            self.gbm._GBMaker__box_periodic_basis(primitive_periods)


class TestGBMakerSelectionBasisVectors(unittest.TestCase):
    def setUp(self):
        self.gbm = object.__new__(GBMaker)
        self.gbm._GBMaker__epsilon = 1e-10
        self.gbm._GBMaker__y_dim = 12.0
        self.gbm._GBMaker__z_dim = 15.0

    def test_selection_basis_vectors_uses_periodic_box_basis_for_both_axes(self):
        self.gbm._GBMaker__inplane_periodic = (True, True)
        primitive_periods = np.array([[2.0, 4.0, -1.0], [-3.0, 1.5, 5.0]])

        basis = self.gbm._GBMaker__selection_basis_vectors(primitive_periods)

        np.testing.assert_allclose(
            basis,
            np.array([[6.0, 12.0, -3.0], [-9.0, 4.5, 15.0]]),
            atol=1e-12,
            rtol=0.0,
        )

    def test_selection_basis_vectors_uses_cartesian_unit_vector_for_nonperiodic_y(self):
        self.gbm._GBMaker__inplane_periodic = (False, True)
        primitive_periods = np.array([[2.0, 4.0, -1.0], [-3.0, 1.5, 5.0]])

        basis = self.gbm._GBMaker__selection_basis_vectors(primitive_periods)

        np.testing.assert_allclose(
            basis,
            np.array([[0.0, 1.0, 0.0], [-9.0, 4.5, 15.0]]),
            atol=1e-12,
            rtol=0.0,
        )

    def test_selection_basis_vectors_uses_cartesian_unit_vector_for_nonperiodic_z(self):
        self.gbm._GBMaker__inplane_periodic = (True, False)
        primitive_periods = np.array([[2.0, 4.0, -1.0], [-3.0, 1.5, 5.0]])

        basis = self.gbm._GBMaker__selection_basis_vectors(primitive_periods)

        np.testing.assert_allclose(
            basis,
            np.array([[6.0, 12.0, -3.0], [0.0, 0.0, 1.0]]),
            atol=1e-12,
            rtol=0.0,
        )


class TestGBMakerBoxCoordinateConversions(unittest.TestCase):
    def setUp(self):
        self.gbm = object.__new__(GBMaker)
        self.gbm._GBMaker__epsilon = 1e-10

    def test_reduced_box_coordinates_handles_orthorhombic_basis(self):
        box_basis = np.array([[0.0, 12.0, 0.0], [0.0, 0.0, 15.0]])
        cartesian = np.array([[1.25, 6.0, 3.75], [4.5, 3.0, 12.0]])

        reduced = self.gbm._GBMaker__reduced_box_coordinates(cartesian, box_basis)

        np.testing.assert_allclose(
            reduced,
            np.array([[1.25, 0.5, 0.25], [4.5, 0.25, 0.8]]),
            atol=1e-12,
            rtol=0.0,
        )

    def test_cartesian_from_box_coordinates_handles_tilted_basis(self):
        box_basis = np.array([[6.0, 12.0, -3.0], [-9.0, 4.5, 15.0]])
        box_coordinates = np.array([[1.5, 0.25, 0.75], [-2.0, 0.5, 0.2]])

        cartesian = self.gbm._GBMaker__cartesian_from_box_coordinates(
            box_coordinates, box_basis
        )

        np.testing.assert_allclose(
            cartesian,
            np.array([[-3.75, 6.375, 10.5], [-0.8, 6.9, 1.5]]),
            atol=1e-12,
            rtol=0.0,
        )

    def test_reduced_box_coordinates_and_cartesian_from_box_coordinates_round_trip(self):
        box_basis = np.array([[6.0, 12.0, -3.0], [-9.0, 4.5, 15.0]])
        cartesian = np.array(
            [[-3.75, 6.375, 10.5], [-0.8, 6.9, 1.5], [2.25, 0.0, 7.5]]
        )

        reduced = self.gbm._GBMaker__reduced_box_coordinates(cartesian, box_basis)
        reconstructed = self.gbm._GBMaker__cartesian_from_box_coordinates(
            reduced, box_basis
        )

        np.testing.assert_allclose(reconstructed, cartesian, atol=1e-12, rtol=0.0)

    def test_reduced_box_coordinates_preserves_vectorized_shape(self):
        box_basis = np.array([[6.0, 12.0, -3.0], [-9.0, 4.5, 15.0]])
        cartesian = np.array(
            [
                [[-3.75, 6.375, 10.5], [-0.8, 6.9, 1.5]],
                [[2.25, 0.0, 7.5], [1.5, 8.25, 0.0]],
            ]
        )

        reduced = self.gbm._GBMaker__reduced_box_coordinates(cartesian, box_basis)
        reconstructed = self.gbm._GBMaker__cartesian_from_box_coordinates(
            reduced, box_basis
        )

        self.assertEqual(reduced.shape, cartesian.shape)
        np.testing.assert_allclose(reconstructed, cartesian, atol=1e-12, rtol=0.0)


class TestGBMakerAssertUniquePositions(unittest.TestCase):
    def setUp(self):
        self.gbm = object.__new__(GBMaker)
        self.gbm.epsilon = 1e-10

    def test_assert_unique_positions_passes_for_distinct_positions(self):
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.gbm._GBMaker__assert_unique_positions(positions)

    def test_assert_unique_positions_raises_for_exact_duplicate(self):
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        with self.assertRaises(GBMakerValueError):
            self.gbm._GBMaker__assert_unique_positions(positions)

    def test_assert_unique_positions_raises_for_positions_within_epsilon(self):
        eps = self.gbm.epsilon
        positions = np.array([[0.0, 0.0, 0.0], [0.4*eps, 0.0, 0.0]])
        with self.assertRaises(GBMakerValueError):
            self.gbm._GBMaker__assert_unique_positions(positions)


class TestGBMakerClipAtomsToCartesianBox(unittest.TestCase):
    def setUp(self):
        self.gbm = object.__new__(GBMaker)
        self.gbm._GBMaker__epsilon = 1e-10
        self.gbm._GBMaker__y_dim = 12.0
        self.gbm._GBMaker__z_dim = 15.0
        self.gbm._GBMaker__inplane_periodic = (False, False)

    def test_clip_atoms_to_cartesian_box_respects_lower_bound_epsilon_on_x(self):
        atoms = np.array(
            [
                ("Cu", -5e-11, 1.0, 1.0),
                ("Cu", -2e-10, 1.0, 1.0),
            ],
            dtype=Atom.atom_dtype,
        )

        clipped = self.gbm._GBMaker__clip_atoms_to_cartesian_box(atoms, np.array([0.0, 5.0]))

        self.assertEqual(len(clipped), 1)
        self.assertAlmostEqual(clipped["x"][0], -5e-11, delta=1e-15)

    def test_clip_atoms_to_cartesian_box_clamps_small_negative_nonperiodic_coordinates(self):
        atoms = np.array(
            [("Cu", 1.0, -5e-11, -2e-11)],
            dtype=Atom.atom_dtype,
        )

        clipped = self.gbm._GBMaker__clip_atoms_to_cartesian_box(atoms, np.array([0.0, 5.0]))

        self.assertEqual(len(clipped), 1)
        np.testing.assert_allclose(
            np.array([clipped["y"][0], clipped["z"][0]]),
            np.array([0.0, 0.0]),
            atol=1e-15,
            rtol=0.0,
        )

    def test_clip_atoms_to_cartesian_box_excludes_values_above_nonperiodic_dims(self):
        atoms = np.array(
            [
                ("Cu", 1.0, 12.0, 1.0),
                ("Cu", 1.0, 1.0, 15.0),
                ("Cu", 1.0, 11.999, 14.999),
            ],
            dtype=Atom.atom_dtype,
        )

        clipped = self.gbm._GBMaker__clip_atoms_to_cartesian_box(atoms, np.array([0.0, 5.0]))

        self.assertEqual(len(clipped), 1)
        np.testing.assert_allclose(
            np.array([clipped["y"][0], clipped["z"][0]]),
            np.array([11.999, 14.999]),
            atol=1e-15,
            rtol=0.0,
        )


class TestGBMakerDeduplicatePositions(unittest.TestCase):
    def setUp(self):
        self.gbm = object.__new__(GBMaker)
        self.gbm._GBMaker__epsilon = 1e-10

    def test_deduplicate_positions_preserves_first_occurrence_order(self):
        atoms = np.array(
            [
                ("Cu", 0.0, 0.0, 0.0),
                ("Cu", 1.0, 0.0, 0.0),
                ("Cu", 0.0, 0.0, 0.0),
                ("Cu", 2.0, 0.0, 0.0),
                ("Cu", 1.0, 0.0, 0.0),
            ],
            dtype=Atom.atom_dtype,
        )

        deduplicated = self.gbm._GBMaker__deduplicate_positions(atoms)

        self.assertEqual(len(deduplicated), 3)
        np.testing.assert_allclose(
            np.column_stack((deduplicated["x"], deduplicated["y"], deduplicated["z"])),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            atol=1e-15,
            rtol=0.0,
        )

    def test_deduplicate_positions_collapses_positions_within_epsilon(self):
        eps = self.gbm._GBMaker__epsilon
        atoms = np.array(
            [
                ("Cu", 0.0, 0.0, 0.0),
                ("Cu", 0.4 * eps, 0.0, 0.0),
            ],
            dtype=Atom.atom_dtype,
        )

        deduplicated = self.gbm._GBMaker__deduplicate_positions(atoms)

        self.assertEqual(len(deduplicated), 1)
        np.testing.assert_allclose(
            np.array([deduplicated["x"][0], deduplicated["y"][0], deduplicated["z"][0]]),
            np.array([0.0, 0.0, 0.0]),
            atol=1e-15,
            rtol=0.0,
        )

    def test_deduplicate_positions_preserves_positions_above_epsilon(self):
        eps = self.gbm._GBMaker__epsilon
        atoms = np.array(
            [
                ("Cu", 0.0, 0.0, 0.0),
                ("Cu", 1.6 * eps, 0.0, 0.0),
            ],
            dtype=Atom.atom_dtype,
        )

        deduplicated = self.gbm._GBMaker__deduplicate_positions(atoms)

        self.assertEqual(len(deduplicated), 2)
        np.testing.assert_allclose(
            np.column_stack((deduplicated["x"], deduplicated["y"], deduplicated["z"])),
            np.array([[0.0, 0.0, 0.0], [1.6 * eps, 0.0, 0.0]]),
            atol=1e-15,
            rtol=0.0,
        )


class TestGBMakerSelectAtomsInBoxBasis(unittest.TestCase):
    def setUp(self):
        self.gbm = object.__new__(GBMaker)
        self.gbm._GBMaker__epsilon = 1e-10
        self.gbm._GBMaker__y_dim = 12.0
        self.gbm._GBMaker__z_dim = 15.0
        self.x_bounds = np.array([0.0, 5.0])

    def test_select_atoms_in_box_basis_removes_duplicate_periodic_faces_in_orthorhombic_box(self):
        self.gbm._GBMaker__inplane_periodic = (True, True)
        primitive_periods = np.array([[0.0, 3.0, 0.0], [0.0, 0.0, 5.0]])
        atoms = np.array(
            [
                ("Cu", 1.0, 0.0, 7.5),
                ("Cu", 1.0, 12.0, 7.5),
                ("Cu", 1.0, 6.0, 0.0),
                ("Cu", 1.0, 6.0, 15.0),
            ],
            dtype=Atom.atom_dtype,
        )

        selected = self.gbm._GBMaker__select_atoms_in_box_basis(
            atoms, primitive_periods, self.x_bounds
        )

        self.assertEqual(len(selected), 2)
        positions = np.column_stack((selected["x"], selected["y"], selected["z"]))
        self.gbm._GBMaker__assert_unique_positions(positions)
        np.testing.assert_allclose(
            positions,
            np.array([[1.0, 0.0, 7.5], [1.0, 6.0, 0.0]]),
            atol=1e-12,
            rtol=0.0,
        )

    def test_select_atoms_in_box_basis_keeps_one_tilted_periodic_representative_and_wraps_x(self):
        self.gbm._GBMaker__inplane_periodic = (True, True)
        primitive_periods = np.array([[2.0, 4.0, -1.0], [-3.0, 1.5, 5.0]])
        box_basis = self.gbm._GBMaker__selection_basis_vectors(primitive_periods)
        box_coordinates = np.array(
            [
                [4.0, 0.25, 0.5],
                [-2.0, 1.25, 0.5],
                [1.0, 0.5, 1.0],
                [4.0, 1.25, 0.5],
            ]
        )
        atoms = np.array(
            [
                ("Cu", *position)
                for position in self.gbm._GBMaker__cartesian_from_box_coordinates(
                    box_coordinates, box_basis
                )
            ],
            dtype=Atom.atom_dtype,
        )

        selected = self.gbm._GBMaker__select_atoms_in_box_basis(
            atoms, primitive_periods, self.x_bounds
        )

        self.assertEqual(len(selected), 2)
        positions = np.column_stack((selected["x"], selected["y"], selected["z"]))
        self.gbm._GBMaker__assert_unique_positions(positions)
        self.assertTrue(np.all(positions[:, 0] >= self.x_bounds[0] - self.gbm.epsilon))
        self.assertTrue(np.all(positions[:, 0] < self.x_bounds[1]))
        np.testing.assert_allclose(
            np.sort(positions[:, 0]),
            np.array([1.0, 4.0]),
            atol=1e-12,
            rtol=0.0,
        )
        reduced = self.gbm._GBMaker__reduced_box_coordinates(positions, box_basis)
        y_tol = self.gbm._GBMaker__reduced_coordinate_tolerance(box_basis[0])
        z_tol = self.gbm._GBMaker__reduced_coordinate_tolerance(box_basis[1])
        self.assertTrue(np.all(reduced[:, 1] >= -y_tol))
        self.assertTrue(np.all(reduced[:, 1] < 1.0 + y_tol))
        self.assertTrue(np.all(reduced[:, 2] >= -z_tol))
        self.assertTrue(np.all(reduced[:, 2] < 1.0 + z_tol))

    def test_select_atoms_in_box_basis_handles_mixed_periodic_and_nonperiodic_axes(self):
        self.gbm._GBMaker__inplane_periodic = (True, False)
        primitive_periods = np.array([[2.0, 4.0, -1.0], [0.0, 0.0, 5.0]])
        box_basis = self.gbm._GBMaker__selection_basis_vectors(primitive_periods)
        box_coordinates = np.array(
            [
                [1.0, 0.0, -5e-11],
                [4.0, 1.0, 0.0],
                [1.0, 0.5, 15.0],
                [5.5, 0.5, 0.0],
            ]
        )
        atoms = np.array(
            [
                ("Cu", *position)
                for position in self.gbm._GBMaker__cartesian_from_box_coordinates(
                    box_coordinates, box_basis
                )
            ],
            dtype=Atom.atom_dtype,
        )

        selected = self.gbm._GBMaker__select_atoms_in_box_basis(
            atoms, primitive_periods, self.x_bounds
        )

        self.assertEqual(len(selected), 2)
        positions = np.column_stack((selected["x"], selected["y"], selected["z"]))
        self.gbm._GBMaker__assert_unique_positions(positions)
        self.assertTrue(np.all(positions[:, 0] >= self.x_bounds[0] - self.gbm.epsilon))
        self.assertTrue(np.all(positions[:, 0] < self.x_bounds[1]))
        reduced = self.gbm._GBMaker__reduced_box_coordinates(positions, box_basis)
        self.assertTrue(np.all(reduced[:, 2] >= 0.0))
        self.assertTrue(np.all(reduced[:, 2] < self.gbm._GBMaker__z_dim))
        y_tol = self.gbm._GBMaker__reduced_coordinate_tolerance(box_basis[0])
        self.assertTrue(np.all(reduced[:, 1] >= -y_tol))
        self.assertTrue(np.all(reduced[:, 1] < 1.0 + y_tol))


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

    def test_orient_period_rows_wiring_feeds_triclinic_output(self):
        cases = [
            (
                "left",
                np.array([[2, 0, 0], [1, 3, 0], [4, 0, 5]], dtype=object),
                np.array([[2, 0, 0], [1, 1, 0], [9, 0, 2]], dtype=object),
            ),
            (
                "right",
                np.array([[2, 0, 0], [1, 1, 0], [9, 0, 2]], dtype=object),
                np.array([[2, 0, 0], [1, 3, 0], [4, 0, 5]], dtype=object),
            ),
        ]

        def expected_tilt(R_grain, approx):
            g_y = approx[1].astype(float)
            g_z = approx[2].astype(float)
            A2_lab = R_grain @ (g_y * self.gbm.a0)
            A3_lab = R_grain @ (g_z * self.gbm.a0)
            theta = -math.atan2(float(A2_lab[2]), float(A2_lab[1]))
            ct, st = math.cos(theta), math.sin(theta)
            return np.array(
                [
                    float(A2_lab[0]),
                    float(A3_lab[0]),
                    float(ct * A3_lab[1] - st * A3_lab[2]),
                    theta,
                ],
                dtype=float,
            )

        for selected_branch, fake_left, fake_right in cases:
            with self.subTest(selected_branch=selected_branch):
                def fake_orient(_self, R_grain, approx):
                    if np.allclose(R_grain, self.gbm._GBMaker__R_left):
                        return fake_left.copy()
                    if np.allclose(R_grain, self.gbm._GBMaker__R_right):
                        return fake_right.copy()
                    raise AssertionError(
                        "Unexpected grain matrix passed to __orient_period_rows"
                    )

                with patch.object(
                    GBMaker, "_GBMaker__orient_period_rows", autospec=True
                ) as mock_orient:
                    mock_orient.side_effect = fake_orient
                    self.gbm.update_spacing(threshold=1e6)
                    self.assertEqual(mock_orient.call_count, 2)

                np.testing.assert_array_equal(
                    self.gbm._GBMaker__R_left_approx, fake_left
                )
                np.testing.assert_array_equal(
                    self.gbm._GBMaker__R_right_approx, fake_right
                )

                selected_R = (
                    self.gbm._GBMaker__R_left
                    if selected_branch == "left"
                    else self.gbm._GBMaker__R_right
                )
                selected_approx = fake_left if selected_branch == "left" else fake_right
                other_R = (
                    self.gbm._GBMaker__R_right
                    if selected_branch == "left"
                    else self.gbm._GBMaker__R_left
                )
                other_approx = fake_right if selected_branch == "left" else fake_left

                actual = np.array(
                    self.gbm._GBMaker__get_triclinic_params(), dtype=float)
                expected_selected = expected_tilt(selected_R, selected_approx)
                expected_other = expected_tilt(other_R, other_approx)

                np.testing.assert_allclose(
                    actual, expected_selected, atol=1e-12, rtol=0.0
                )
                self.assertFalse(np.allclose(
                    actual, expected_other, atol=1e-12, rtol=0.0))

                with tempfile.NamedTemporaryFile(delete=True) as f:
                    self.gbm.write_lammps(f.name, triclinic=True)
                    with open(f.name) as fread:
                        content = fread.readlines()

                tilt_line = next(
                    line.strip() for line in content if line.endswith("xy xz yz\n")
                )
                self.assertEqual(
                    tilt_line,
                    f"{expected_selected[0]:.6f} {expected_selected[1]:.6f} "
                    f"{expected_selected[2]:.6f} xy xz yz",
                )


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
    # Stacking-test helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _layer_xs(atoms, atol=1e-2):
        xs = np.sort(atoms["x"])
        layers = [xs[0]]
        for x in xs[1:]:
            if abs(x - layers[-1]) > atol:
                layers.append(x)
        return np.array(layers)

    @staticmethod
    def _yz_points(atoms, x0, atol=1e-2):
        pts = atoms[np.isclose(atoms["x"], x0, atol=atol)]
        return np.column_stack([pts["y"], pts["z"]])

    @staticmethod
    def _planes_coincide(ref, cand, ydim, zdim, tol=0.1):
        """
        Return True if candidate and reference share the same y,z positions (same
        stacking type). KDTree boxsize applies the minimum-image convention so atoms
        straddling a periodic boundary are correctly identified as coincident.
        tol=0.1 Å is well above FP noise (~1e-10 Å) and well below the minimum DSC shift
        between different plane types (~0.85 Å for Cu).
        """
        if len(ref) != len(cand):
            return False
        ref_w = np.column_stack([ref[:, 0] % ydim, ref[:, 1] % zdim])
        cand_w = np.column_stack([cand[:, 0] % ydim, cand[:, 1] % zdim])
        tree = KDTree(ref_w, boxsize=[ydim, zdim])
        dists, _ = tree.query(cand_w, k=1)
        return np.all(dists < tol)

    def _assert_interface_stacking(self, gbm, d_spacing):
        left_layers = self._layer_xs(gbm.left_grain)
        right_layers = self._layer_xs(gbm.right_grain)
        interface_gap = right_layers[0] - left_layers[-1]
        terminal = self._yz_points(gbm.left_grain, left_layers[-1])
        right_1 = self._yz_points(gbm.right_grain, right_layers[0])

        self.assertAlmostEqual(
            interface_gap,
            d_spacing,
            delta=d_spacing * 0.05,
            msg=f"Interface x-gap {interface_gap:.4f} Å should equal "
            f"d_spacing = {d_spacing:.4f} Å.  A gap of ~0 means both grains "
            f"placed a plane at the same x-coordinate.",
        )
        self.assertFalse(
            self._planes_coincide(terminal, right_1, gbm.y_dim, gbm.z_dim),
            "Terminal left-grain plane and first right-grain plane share the "
            "same in-plane y,z positions (same stacking type). The interface "
            "should have adjacent planes of different types (e.g. C then A), "
            "not duplicate same-type planes (e.g. C then C).",
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

    # ------------------------------------------------------------------
    # Sigma3 (111) -- interfacial stacking regression
    # ------------------------------------------------------------------

    def test_sigma3_111_no_duplicate_plane_at_interface(self):
        """
        Sigma3 (111) coherent twin (180 deg representation): terminal left-grain plane
        and first right-grain plane must be separated by exactly one d_111 spacing and
        be of different stacking type.
        """

        gbm = self._make_gb(self.sigma3_111_180deg, repeat_factor=(2, 3))
        self._assert_interface_stacking(gbm, d_spacing=self.a0 / np.sqrt(3))

    def test_sigma3_111_60deg_no_duplicate_plane_at_interface(self):
        """
        Sigma3 (111) coherent twin (60 deg representation): same interface stacking
        invariant as the 180 deg representation
        """
        gbm = self._make_gb(self.sigma3_111_60deg, repeat_factor=(2, 3))
        self._assert_interface_stacking(gbm, d_spacing=self.a0 / np.sqrt(3))


if __name__ == "__main__":
    unittest.main()
