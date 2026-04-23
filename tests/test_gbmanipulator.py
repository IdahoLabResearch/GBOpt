# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import filecmp
import math
import os
import tempfile
import unittest
import warnings

import numpy as np
import pytest

from GBOpt.GBMaker import GBMaker

# from unittest.mock import patch
from GBOpt.GBManipulator import (
    GBManipulator,
    GBManipulatorValueError,
    Parent,
    ParentCorruptedFileError,
    ParentFileMissingDataError,
    ParentFileNotFoundError,
    ParentsProxyIndexError,
    ParentsProxyTypeError,
    ParentsProxyValueError,
    ParentValueError,
    _ParentsProxy,
)
from GBOpt.UnitCell import UnitCell


def structured_array_equal(array1, array2):
    if array1.dtype != array2.dtype:
        return False

    for field in array1.dtype.names:
        if np.issubdtype(array1[field].dtype, np.number):
            if not np.allclose(array1[field], array2[field]):
                return False
        else:
            if not np.array_equal(array1[field], array2[field]):
                return False

    return True


class TestGBManipulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        theta = math.radians(36.869898)
        cls.seed = 100
        cls.tilt = GBMaker(
            a0=1.0,
            structure='fcc',
            gb_thickness=10.0,
            misorientation=[theta, 0, 0, 0, -theta/2],
            atom_types='Cu',
            interaction_distance=1,
            repeat_factor=(2, 5)
        )
        cls.twist = GBMaker(
            a0=1.0,
            structure='fcc',
            gb_thickness=10.0,
            misorientation=[0, theta, 0, 0, 0],
            atom_types='Cu',
            interaction_distance=1,
            repeat_factor=2
        )
        cls.manipulator_tilt = GBManipulator(cls.tilt, seed=cls.seed)
        cls.manipulator_twist = GBManipulator(cls.twist, seed=cls.seed)

    def setUp(self):
        self.a0 = 1.0
        self.structure = 'fcc'
        self.gb_thickness = 10.0
        self.atom_types = 'Cu'
        self.misorientation = [math.radians(36.869898), 0, 0, 0, 0]
        self.file1 = "tests/inputs/basic_dump_test1.txt"
        self.file2 = "tests/inputs/basic_dump_test2.txt"
        self.seed = 100

    def test_init_with_one_gbmaker_parent(self):
        self.assertIsNotNone(self.manipulator_tilt.parents[0])
        self.assertIsNone(self.manipulator_tilt.parents[1])
        self.assertTrue(self.manipulator_tilt._GBManipulator__one_parent)

    def test_init_with_two_gbmaker_parents(self):
        manipulator = GBManipulator(self.tilt, self.tilt)
        self.assertIsNotNone(manipulator.parents[0])
        self.assertIsNotNone(manipulator.parents[1])

    def test_init_with_one_snapshot(self):
        unit_cell = UnitCell()
        unit_cell.init_by_structure(self.structure, self.a0, self.atom_types)
        gb_thickness = 10
        manipulator = GBManipulator(
            self.file1, unit_cell=unit_cell, gb_thickness=gb_thickness)
        self.assertIsNotNone(manipulator.parents[0])
        self.assertIsNone(manipulator.parents[1])

    def test_init_with_two_snapshots(self):
        unit_cell = UnitCell()
        unit_cell.init_by_structure(self.structure, self.a0, self.atom_types)
        gb_thickness = 10
        manipulator = GBManipulator(
            self.file1, self.file2, unit_cell=unit_cell, gb_thickness=gb_thickness)
        self.assertIsNotNone(manipulator.parents[0])
        self.assertIsNotNone(manipulator.parents[1])

    def test_init_with_mixed_input(self):
        unit_cell = UnitCell()
        unit_cell.init_by_structure(self.structure, self.a0, self.atom_types)
        gb_thickness = 10
        manipulator = GBManipulator(
            self.tilt, self.file1, unit_cell=unit_cell, gb_thickness=gb_thickness)
        self.assertIsNotNone(manipulator.parents[0])
        self.assertIsNotNone(manipulator.parents[1])

        manipulator2 = GBManipulator(
            self.file1, self.tilt, unit_cell=unit_cell, gb_thickness=gb_thickness)
        self.assertIsNotNone(manipulator2.parents[0])
        self.assertIsNotNone(manipulator2.parents[1])

        # test with GBMaker instance and file, without unit cell or gb_thickness
        manipulator3 = GBManipulator(self.tilt, self.file1)
        self.assertEqual(
            manipulator3.parents[0].unit_cell, manipulator3.parents[1].unit_cell)
        self.assertEqual(
            manipulator3.parents[0].gb_thickness, manipulator3.parents[1].gb_thickness)

    def test_grain_translation(self):
        new_system = self.manipulator_tilt.translate_right_grain(1.0, 0.5)
        self.assertTrue(np.allclose(self.tilt.whole_system['x'], new_system['x']))
        self.assertTrue(not np.allclose(self.tilt.whole_system['y'], new_system['y']))
        self.assertTrue(not np.allclose(self.tilt.whole_system['z'], new_system['z']))
        self.assertTrue(all(self.tilt.whole_system['name'] == new_system['name']))

    def test_grain_translation_warning(self):
        manipulator = GBManipulator(self.tilt, self.tilt)
        with self.assertWarns(UserWarning):
            _ = manipulator.translate_right_grain(1.0, 1.0)

    def test_slice_and_merge(self):
        manipulator = GBManipulator(self.tilt, self.tilt, seed=self.seed)
        new_system = manipulator.slice_and_merge()
        self.assertFalse(all(self.tilt.whole_system == new_system))

    def test_slice_and_merge_error(self):
        with self.assertRaises(GBManipulatorValueError):
            _ = self.manipulator_tilt.slice_and_merge()

    def test_remove_atoms(self):
        new_system = self.manipulator_tilt.remove_atoms(gb_fraction=0.10)
        self.assertGreater(len(self.tilt.whole_system), len(new_system))

    def test_remove_atoms_fraction_error(self):
        with self.assertRaises(GBManipulatorValueError):
            _ = self.manipulator_tilt.remove_atoms(gb_fraction=0.50)

    def test_remove_atoms_2_parent_warning(self):
        manipulator = GBManipulator(self.tilt, self.tilt, seed=self.seed)
        with self.assertWarns(UserWarning):
            _ = manipulator.remove_atoms(gb_fraction=0.10)
            _ = manipulator.remove_atoms(gb_fraction=0.10)

    def test_remove_atoms_calculated_fraction_warning(self):
        with self.assertWarns(UserWarning):
            _ = self.manipulator_tilt.remove_atoms(gb_fraction=1e-7)

    def test_remove_atoms_with_specific_number(self):
        new_system = self.manipulator_tilt.remove_atoms(num_to_remove=1)
        self.assertEqual(len(self.tilt.whole_system)-1, len(new_system))

    def test_remove_atoms_with_stoichiometry(self):
        theta = math.radians(36.869898)
        mis = [theta, 0, 0, 0, -theta/2]
        GB = GBMaker(5.454, "fluorite", 5.454, mis, ["U", "O"], repeat_factor=(2, 6))
        gbm = GBManipulator(GB)
        new_system = gbm.remove_atoms(num_to_remove=1, keep_ratio=True)
        self.assertEqual(len(GB.whole_system)-3, len(new_system))

    def test_insert_atoms(self):
        new_system_delaunay = self.manipulator_tilt.insert_atoms(
            fill_fraction=0.10, method='delaunay')
        self.assertGreater(len(new_system_delaunay), len(self.tilt.whole_system))
        new_system_grid = self.manipulator_tilt.insert_atoms(
            fill_fraction=0.10, method='grid')
        self.assertGreater(len(new_system_grid), len(self.tilt.whole_system))

    def test_insert_atoms_fraction_error(self):
        with self.assertRaises(GBManipulatorValueError):
            _ = self.manipulator_tilt.insert_atoms(
                fill_fraction=0.50, method='delaunay')

    def test_insert_atoms_2_parent_warning(self):
        manipulator = GBManipulator(self.tilt, self.tilt, seed=self.seed)
        with self.assertWarns(UserWarning):
            _ = manipulator.insert_atoms(fill_fraction=0.10, method='delaunay')

    def test_insert_atoms_calculated_fraction_warning(self):
        with self.assertWarns(UserWarning):
            _ = self.manipulator_tilt.insert_atoms(
                fill_fraction=1e-7, method='delaunay')

        with self.assertWarns(UserWarning):
            _ = self.manipulator_tilt.insert_atoms(
                fill_fraction=1e-7, method='grid')

    def test_insert_atoms_invalid_method(self):
        with self.assertRaises(GBManipulatorValueError):
            _ = self.manipulator_tilt.insert_atoms(fill_fraction=0.10, method='invalid')

    def test_insert_atoms_with_specific_number(self):
        new_system_delaunay = self.manipulator_tilt.insert_atoms(
            method='delaunay', num_to_insert=1)
        self.assertEqual(len(self.tilt.whole_system) + 1, len(new_system_delaunay))
        new_system_grid = self.manipulator_tilt.insert_atoms(
            method='grid', num_to_insert=1)
        self.assertEqual(len(self.tilt.whole_system) + 1, len(new_system_grid))

    def test_insert_atoms_with_stoichiometry(self):
        theta = math.radians(36.869898)
        mis = [theta, 0, 0, 0, -theta/2]
        GB = GBMaker(5.454, "fluorite", 5.454, mis, ["U", "O"], repeat_factor=(2, 6))
        gbm = GBManipulator(GB)
        new_system = gbm.insert_atoms(num_to_insert=1, keep_ratio=True)
        self.assertEqual(len(GB.whole_system)+3, len(new_system))

    def test_type_preservation_with_numeric_roundtrip(self):
        theta = math.radians(36.869898)
        gb = GBMaker(
            a0=3.0,
            structure="rocksalt",
            gb_thickness=5.0,
            misorientation=[theta, 0, 0, 0, -theta / 2.0],
            atom_types=("Na", "Cl"),
            repeat_factor=(2, 2),
            x_dim_min=5.0,
            vacuum=2.0,
            interaction_distance=4.0
        )
        expected_types = {"Na", "Cl"}
        base_names = gb.whole_system["name"]
        self.assertEqual(set(base_names), expected_types)

        def roundtrip_names(atoms):
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                gb.write_lammps(
                    temp_file.name,
                    atoms,
                    gb.box_dims,
                    type_as_int=True,
                )
                loaded = GBManipulator(
                    temp_file.name,
                    unit_cell=gb.unit_cell,
                    gb_thickness=gb.gb_thickness,
                    seed=self.seed,
                )
                return loaded.parents[0].whole_system["name"]

        manipulator = GBManipulator(gb, seed=self.seed)

        translated = manipulator.translate_right_grain(0.1, 0.2)
        self.assertTrue(np.array_equal(translated["name"], base_names))
        self.assertEqual(set(roundtrip_names(translated)), expected_types)

        removed = manipulator.remove_atoms(num_to_remove=1, keep_ratio=True)
        self.assertEqual(set(roundtrip_names(removed)), expected_types)

        inserted = manipulator.insert_atoms(
            num_to_insert=1,
            method="grid",
            keep_ratio=True,
        )
        self.assertTrue(
            np.array_equal(inserted["name"][:len(base_names)], base_names)
        )
        self.assertEqual(set(roundtrip_names(inserted)), expected_types)

        manipulator_two = GBManipulator(gb, gb, seed=self.seed)
        sliced = manipulator_two.slice_and_merge()
        self.assertEqual(set(roundtrip_names(sliced)), expected_types)

        soft_mode_displaced = manipulator.displace_along_soft_modes()
        self.assertTrue(np.array_equal(soft_mode_displaced["name"], base_names))
        self.assertEqual(set(roundtrip_names(soft_mode_displaced)), expected_types)

    @pytest.mark.slow
    def test_displace_along_soft_modes_base(self):
        # test the base case
        child = self.manipulator_tilt.displace_along_soft_modes()
        self.assertEqual(len(child), 1)
        self.assertFalse(structured_array_equal(
            child[0], self.manipulator_tilt.parents[0].whole_system))

    @pytest.mark.slow
    def test_displace_along_soft_modes_with_displacement_threshold(self):
        # test the case with a displacement threshold specified
        child = self.manipulator_tilt.displace_along_soft_modes(1.0)
        self.assertEqual(len(child), 1)
        self.assertFalse(structured_array_equal(
            child[0], self.manipulator_tilt.parents[0].whole_system))

    @pytest.mark.slow
    def test_displace_along_soft_modes_diff_mesh(self):
        # test differing mesh size
        child = self.manipulator_tilt.displace_along_soft_modes(mesh_size=2)
        self.assertEqual(len(child), 1)
        self.assertFalse(structured_array_equal(
            child[0], self.manipulator_tilt.parents[0].whole_system))

    @pytest.mark.slow
    def test_displace_along_soft_modes_num_q_vecs(self):
        # test number of q vectors
        child = self.manipulator_tilt.displace_along_soft_modes(num_q=20)
        self.assertEqual(len(child), 1)
        self.assertFalse(structured_array_equal(
            child[0], self.manipulator_tilt.parents[0].whole_system))

    @pytest.mark.slow
    def test_displace_along_soft_modes_num_child_structures(self):
        # test number of child structures
        children = self.manipulator_tilt.displace_along_soft_modes(num_children=2)
        self.assertEqual(len(children), 2)
        self.assertFalse(structured_array_equal(
            children[0], self.manipulator_tilt.parents[0].whole_system))
        self.assertFalse(structured_array_equal(
            children[1], self.manipulator_tilt.parents[0].whole_system))
        self.assertFalse(structured_array_equal(children[0], children[1]))

    @pytest.mark.slow
    def test_displace_along_soft_modes_simple_case(self):
        # While we end up using the indicated file for the actual atomic configuration,
        # that configuration was developed using this set of parameters
        GB = GBMaker(
            3.54, 'fcc', 5.0, np.array([0, 0, 0, 0, 0]), atom_types='Cu',
            repeat_factor=6, x_dim_min=10, vacuum=10, interaction_distance=5
        )
        manipulator = GBManipulator(
            './tests/inputs/Cu_single_crystal_with_displaced_atom.txt', unit_cell=GB.unit_cell, gb_thickness=5)
        child1 = manipulator.displace_along_soft_modes()[0]
        child2 = manipulator.displace_along_soft_modes(subtract_displacement=True)[0]
        self.assertFalse(structured_array_equal(child1, child2))

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            GB.write_lammps(temp_file.name, child1, GB.box_dims)
            self.assertTrue(
                filecmp.cmp(
                    temp_file.name,
                    './tests/gold/soft_phonon_mode_displacement_added.txt',
                    shallow=False
                )
            )

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            GB.write_lammps(temp_file.name, child2, GB.box_dims)
            self.assertTrue(
                filecmp.cmp(
                    temp_file.name,
                    './tests/gold/soft_phonon_mode_displacement_subtracted.txt',
                    shallow=False
                )
            )

    def test_apply_group_symmetry(self):
        manipulator = GBManipulator(self.tilt, seed=self.seed)
        with self.assertRaises(NotImplementedError):
            _ = manipulator.apply_group_symmetry("group")

    def test_parents_getter(self):
        manipulator = GBManipulator(self.tilt)
        parents = manipulator.parents
        self.assertTrue(isinstance(parents, _ParentsProxy))
        self.assertTrue(isinstance(parents[0], Parent))
        self.assertIsNone(parents[1])

    def test_parents_setter(self):
        manipulator = GBManipulator(self.tilt)
        self.assertIsNone(manipulator.parents[1])
        manipulator.parents[1] = Parent(self.tilt)
        self.assertIsNotNone(manipulator.parents[1])
        manipulator.parents = [Parent(self.file1, unit_cell=self.tilt.unit_cell),
                               Parent(self.file2, unit_cell=self.tilt.unit_cell)]
        self.assertFalse(None in manipulator.parents)

        with self.assertRaises(GBManipulatorValueError):
            manipulator.parents = Parent(self.file1, unit_cell=self.tilt.unit_cell)

        with self.assertRaises(GBManipulatorValueError):
            manipulator.parents = [Parent(self.file1, unit_cell=self.tilt.unit_cell), 1]

    def test_write_lammps_after_manipulate(self):
        manipulator1 = GBManipulator(self.tilt)
        manipulator2 = GBManipulator(self.tilt, self.tilt)
        p1 = manipulator1.translate_right_grain(1, 1)
        p2 = manipulator2.slice_and_merge()
        p3 = manipulator1.insert_atoms(fill_fraction=0.2, method='delaunay')
        # p4 = manipulator1.remove_atoms(0.2)
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            self.tilt.write_lammps(temp_file.name, p1, self.tilt.box_dims)
            self.tilt.write_lammps(temp_file.name, p2, self.tilt.box_dims)
            self.tilt.write_lammps(temp_file.name, p3, self.tilt.box_dims)
            # self.tilt.write_lammps(temp_file.name, p4, self.tilt.box_dims)

    def test_created_gbs(self):
        manipulator = GBManipulator(self.tilt, self.tilt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            p1 = manipulator.translate_right_grain(1, 1)
            p2 = manipulator.insert_atoms(fill_fraction=0.2, method='delaunay')
        p3 = manipulator.slice_and_merge()
        # p4 = manipulator.remove_atoms(0.2)
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            self.tilt.write_lammps(temp_file.name, type_as_int=True)
            self.assertTrue(filecmp.cmp("./tests/gold/sigma5_tilt.txt",
                            temp_file.name, shallow=False))
            self.twist.write_lammps(temp_file.name, type_as_int=True)
            self.assertTrue(filecmp.cmp("./tests/gold/sigma5_twist.txt",
                            temp_file.name, shallow=False))


class TestParent(unittest.TestCase):
    def setUp(self):
        self.unit_cell = UnitCell()
        self.unit_cell.init_by_structure('fcc', 1.0, 'Cu')
        self.GB = GBMaker(
            a0=1.0,
            structure='fcc',
            gb_thickness=10.0,
            misorientation=[math.radians(36.869898), 0, 0, 0, 0],
            atom_types='Cu',
            repeat_factor=2,
            interaction_distance=1
        )
        self.parent = Parent(self.GB)
        self.file = 'tests/inputs/basic_dump_test1.txt'

    def test_parent_init(self):
        parent1 = Parent(self.GB)
        self.assertGreater(len(parent1.left_grain), 0)
        self.assertGreater(len(parent1.right_grain), 0)
        self.assertEqual(len(parent1.left_grain) +
                         len(parent1.right_grain), len(parent1.whole_system))

        parent2 = Parent(
            self.file, unit_cell=self.unit_cell, gb_thickness=20)
        self.assertGreater(len(parent2.left_grain), 0)
        self.assertGreater(len(parent2.right_grain), 0)
        self.assertEqual(len(parent2.left_grain) +
                         len(parent2.right_grain), len(parent2.whole_system))
        self.assertEqual(parent2.gb_thickness, 20)
        self.assertEqual(parent2.whole_system[0]['name'], 'H')

    def test_parent_getters(self):
        parent = Parent(self.GB)
        self.assertEqual(repr(parent.unit_cell), repr(self.GB.unit_cell))
        self.assertEqual(parent.gb_thickness, self.GB.gb_thickness)
        self.assertTrue(np.allclose(parent.box_dims, self.GB.box_dims))
        # Parent calculates x_dim differently than GB
        self.assertNotEqual(parent.x_dim, self.GB.x_dim)
        self.assertEqual(parent.y_dim, self.GB.y_dim)
        self.assertEqual(parent.z_dim, self.GB.z_dim)

        x_gb = parent._Parent__gb_plane_x
        left_cut = x_gb - parent.gb_thickness / 2.0
        right_cut = x_gb + parent.gb_thickness / 2.0
        left_gb = parent.left_grain[parent.left_grain["x"] > left_cut]
        right_gb = parent.right_grain[parent.right_grain["x"] < right_cut]
        np.testing.assert_array_equal(parent.gb_atoms, np.hstack((left_gb, right_gb)))
        np.testing.assert_array_equal(parent.left_grain, self.GB.left_grain)
        np.testing.assert_array_equal(parent.right_grain, self.GB.right_grain)
        np.testing.assert_array_equal(parent.whole_system, self.GB.whole_system)
        self.assertEqual(parent.unit_cell, self.GB.unit_cell)

    def test_parent_snapshot_init_errors(self):
        with self.assertRaises(ParentValueError):
            _ = Parent(self.file)
        with self.assertRaises(ParentFileNotFoundError):
            _ = Parent("tests/inputs/file_not_found.txt", unit_cell=self.unit_cell)
        with self.assertRaises(ParentCorruptedFileError):
            _ = Parent("tests/inputs/file_without_box_bounds.txt",
                       unit_cell=self.unit_cell)
        with self.assertRaises(ParentCorruptedFileError):
            _ = Parent("tests/inputs/file_with_invalid_box_bounds.txt",
                       unit_cell=self.unit_cell)
        with self.assertRaises(ParentCorruptedFileError):
            _ = Parent("tests/inputs/file_with_invalid_box_bounds2.txt",
                       unit_cell=self.unit_cell)
        with self.assertRaises(ParentCorruptedFileError):
            _ = Parent("tests/inputs/file_without_atoms.txt",
                       unit_cell=self.unit_cell)
        with self.assertRaises(ParentFileMissingDataError):
            _ = Parent("tests/inputs/file_missing_required_info.txt",
                       unit_cell=self.unit_cell)

    def test_read_lammps_with_typelabel(self):
        parent = Parent("tests/inputs/lammps_dump_with_typelabel_test.txt",
                        unit_cell=self.unit_cell, gb_thickness=20)
        self.assertEqual(parent.whole_system[0]['name'], 'Cu')

    def test_read_lammps_input(self):
        uc = UnitCell()
        uc.init_by_structure('fcc', 3.54, 'Cu')
        parent1 = Parent("tests/inputs/lammps_input_with_labels.txt",
                         unit_cell=uc, gb_thickness=10)
        parent2 = Parent("tests/inputs/lammps_input_without_labels.txt",
                         unit_cell=uc, gb_thickness=10)
        self.assertEqual(len(parent1.whole_system), 792)
        self.assertTrue(np.isclose(parent1.whole_system[0]['x'], 1.77))
        self.assertTrue(np.isclose(parent1.whole_system[0]['y'], 1.77))
        self.assertTrue(np.isclose(parent1.whole_system[0]['z'], 3.54))
        self.assertTrue(np.allclose(parent1.box_dims, np.array(
            [[-10, 30], [0, 21.24], [0, 21.24]])))
        self.assertTrue(all(parent1.whole_system['name'] == 'Cu'))

        self.assertEqual(len(parent2.whole_system), 792)
        self.assertTrue(np.isclose(parent2.whole_system[0]['x'], 1.77))
        self.assertTrue(np.isclose(parent2.whole_system[0]['y'], 1.77))
        self.assertTrue(np.isclose(parent2.whole_system[0]['z'], 3.54))
        self.assertTrue(np.allclose(parent2.box_dims, np.array(
            [[-10, 30], [0, 21.24], [0, 21.24]])))
        self.assertTrue(all(parent2.whole_system['name'] == 'H'))

    def test_read_lammps_input_multiple_atom_types(self):
        uc = UnitCell()
        uc.init_by_structure('fcc', 3.54, 'Cu')
        parent = Parent(
            "tests/inputs/lammps_input_multiple_atom_types.txt",
            unit_cell=uc, gb_thickness=10)
        self.assertTrue(
            all(np.unique(parent.whole_system['name']) == np.array(['Cu', 'Fe', 'Ni'])))

    def test_read_lammps_input_errors(self):
        uc = UnitCell()
        uc.init_by_structure('fcc', 354, 'Cu')
        with self.assertRaises(ParentCorruptedFileError):
            _ = Parent(
                "tests/inputs/lammps_input_multiple_atom_types_missing_labels.txt",
                unit_cell=uc)

        with self.assertRaises(ParentCorruptedFileError):
            _ = Parent(
                "tests/inputs/lammps_input_multiple_atom_types_wrong_num_types.txt",
                unit_cell=uc)

    def test_unknown_file_type(self):
        with self.assertRaises(ParentValueError):
            _ = Parent('tests/inputs/unknown_file_type.txt',
                       unit_cell=self.unit_cell, gb_thickness=20)

    def test_file_too_short(self):
        with self.assertRaises(ParentValueError):
            _ = Parent("tests/inputs/file_too_short.txt",
                       unit_cell=self.unit_cell, gb_thickness=20)


class TestParentGBRegion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        theta = 2 * np.arctan(1 / 3)
        mis = np.array([theta, 0, 0, np.pi / 4, -np.arctan(1 / np.sqrt(2))])
        kwargs = dict(atom_types="Si", interaction_distance=6.0,
                      vacuum=0, repeat_factor=(2, 3))
        probe = GBMaker(5.431, "diamond", 5.431, mis, **kwargs)
        gb_thickness = 2 * max(probe.spacing["x"]["left"], probe.spacing["x"]["right"])
        cls.gbm = GBMaker(5.431, "diamond", gb_thickness, mis, **kwargs)
        cls.parent = Parent(cls.gbm)
        cls.d_hkl = max(probe.spacing["x"]["left"], probe.spacing["x"]["right"])

    def test_gb_indices_lie_within_symmetric_window(self):
        parent = self.parent
        x_gb = parent._Parent__gb_plane_x
        half = parent.gb_thickness / 2.0
        xs = parent.whole_system["x"][parent.gb_indices]
        self.assertGreater(len(xs), 0)
        self.assertTrue(np.all(xs > x_gb - half))
        self.assertTrue(np.all(xs < x_gb + half))

    def test_gb_plane_x_from_gbmaker_matches_source(self):
        self.assertAlmostEqual(self.parent._Parent__gb_plane_x,
                               self.gbm.gb_plane_x, places=10)

    def test_file_path_gb_plane_x_near_gbmaker_value(self):
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            path = f.name
        try:
            self.gbm.write_lammps(path, type_as_int=True)
            unit_cell = UnitCell()
            unit_cell.init_by_structure("diamond", 5.431, "Si")
            parent_file = Parent(path, unit_cell=unit_cell,
                                 gb_thickness=self.gbm.gb_thickness)
            self.assertAlmostEqual(
                parent_file._Parent__gb_plane_x, self.gbm.gb_plane_x,
                delta=self.d_hkl,
                msg="File-path gb_plane_x should be within one d_hkl of GBMaker value"
            )
        finally:
            os.remove(path)

    def test_gbpos_atoms_lie_within_gb_indices_window(self):
        parent = self.parent
        x_gb = parent._Parent__gb_plane_x
        half = parent.gb_thickness / 2.0
        gbpos_xs = parent._Parent__GBpos["x"]
        self.assertTrue(np.all(gbpos_xs >= x_gb - half))
        self.assertTrue(np.all(gbpos_xs <= x_gb + half))


class TestParentProxy(unittest.TestCase):
    def setUp(self):
        self.unit_cell = UnitCell()
        self.unit_cell.init_by_structure('fcc', 1.0, 'Cu')
        self.manipulator = GBManipulator(
            'tests/inputs/basic_dump_test1.txt', unit_cell=self.unit_cell)
        self.parents_proxy = _ParentsProxy(self.manipulator)

    def test_getitem(self):
        parent = self.parents_proxy[0]
        self.assertIsInstance(parent, Parent)

    def test_setitem(self):
        new_parent = Parent(GBMaker(a0=3.61, structure='fcc', gb_thickness=10.0, misorientation=[
            math.radians(36.869898), 0, 0, 0, 0], atom_types='Cu', repeat_factor=2, interaction_distance=1))
        self.parents_proxy[0] = new_parent
        self.assertIs(self.parents_proxy[0], new_parent)

    def test_len(self):
        self.assertEqual(len(self.parents_proxy), 2)

    def test_setitem_errors(self):
        self.parents_proxy[0] = None
        new_parent = Parent(GBMaker(a0=1.0, structure='fcc', gb_thickness=10.0, misorientation=[
            math.radians(36.869898), 0, 0, 0, 0], atom_types='Cu', repeat_factor=2, interaction_distance=1))
        with self.assertRaises(ParentsProxyValueError):
            self.parents_proxy[1] = new_parent
        self.parents_proxy[0] = new_parent
        with self.assertRaises(ParentsProxyIndexError):
            self.parents_proxy[2] = new_parent
        with self.assertRaises(ParentsProxyTypeError):
            self.parents_proxy[0] = 1


if __name__ == '__main__':
    unittest.main()
