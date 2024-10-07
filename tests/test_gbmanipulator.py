import math
import tempfile
import unittest

import numpy as np

from GBOpt.GBMaker import GBMaker

# from unittest.mock import patch
from GBOpt.GBManipulator import (
    GBManipulator,
    GBManipulatorValueError,
    Parent,
    ParentCorruptedSnapshotError,
    ParentFileNotFoundError,
    ParentSnapshotMissingDataError,
    ParentsProxyIndexError,
    ParentsProxyTypeError,
    ParentsProxyValueError,
    ParentValueError,
    _ParentsProxy,
)
from GBOpt.UnitCell import UnitCell


class TestGBManipulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = GBMaker(a0=1.0, structure='fcc', gb_thickness=10.0, misorientation=[
            math.radians(36.869898), 0, 0, 0, 0], repeat_factor=1)

    def setUp(self):
        self.a0 = 1.0
        self.structure = 'fcc'
        self.gb_thickness = 10.0
        self.misorientation = [math.radians(36.869898), 0, 0, 0, 0]
        self.file1 = "tests/inputs/test1.txt"
        self.file2 = "tests/inputs/test2.txt"
        self.seed = 100

    def test_init_with_one_gbmaker_parent(self):
        manipulator = GBManipulator(self.system)
        self.assertIsNotNone(manipulator.parents[0])
        self.assertIsNone(manipulator.parents[1])
        self.assertTrue(manipulator._GBManipulator__one_parent)

    def test_init_with_two_gbmaker_parents(self):
        manipulator = GBManipulator(self.system, self.system)
        self.assertIsNotNone(manipulator.parents[0])
        self.assertIsNotNone(manipulator.parents[1])

    def test_init_with_one_snapshot(self):
        unit_cell = UnitCell()
        unit_cell.init_by_structure(self.structure, self.a0)
        gb_thickness = 10
        manipulator = GBManipulator(
            self.file1, unit_cell=unit_cell, gb_thickness=gb_thickness)
        self.assertIsNotNone(manipulator.parents[0])
        self.assertIsNone(manipulator.parents[1])

    def test_init_with_two_snapshots(self):
        unit_cell = UnitCell()
        unit_cell.init_by_structure(self.structure, self.a0)
        gb_thickness = 10
        manipulator = GBManipulator(
            self.file1, self.file2, unit_cell=unit_cell, gb_thickness=gb_thickness)
        self.assertIsNotNone(manipulator.parents[0])
        self.assertIsNotNone(manipulator.parents[1])

    def test_init_with_mixed_input(self):
        unit_cell = UnitCell()
        unit_cell.init_by_structure(self.structure, self.a0)
        gb_thickness = 10
        manipulator = GBManipulator(
            self.system, self.file1, unit_cell=unit_cell, gb_thickness=gb_thickness)
        self.assertIsNotNone(manipulator.parents[0])
        self.assertIsNotNone(manipulator.parents[1])

        manipulator2 = GBManipulator(
            self.file1, self.system, unit_cell=unit_cell, gb_thickness=gb_thickness)
        self.assertIsNotNone(manipulator2.parents[0])
        self.assertIsNotNone(manipulator2.parents[1])

    def test_grain_translation(self):
        manipulator = GBManipulator(self.system)
        new_system = manipulator.translate_right_grain(1.0, 1.0)
        self.assertTrue(not np.allclose(self.system.gb, new_system))

    def test_grain_translation_warning(self):
        manipulator = GBManipulator(self.system, self.system)
        with self.assertWarns(UserWarning):
            _ = manipulator.translate_right_grain(1.0, 1.0)

    def test_slice_and_merge(self):
        manipulator = GBManipulator(self.system, self.system, seed=self.seed)
        new_system = manipulator.slice_and_merge()
        self.assertFalse(np.allclose(self.system.gb, new_system))

    def test_slice_and_merge_error(self):
        manipulator = GBManipulator(self.system, seed=self.seed)
        with self.assertRaises(GBManipulatorValueError):
            _ = manipulator.slice_and_merge()

    @unittest.skip("Currently get hung on generating the neighbor list.")
    def test_remove_atoms(self):
        manipulator = GBManipulator(self.system, seed=self.seed)
        new_system = manipulator.remove_atoms(0.10)
        self.assertGreater(len(self.system.gb), len(new_system))

    @unittest.skip("Currently get hung on generating the neighbor list.")
    def test_remove_atoms_fraction_error(self):
        manipulator = GBManipulator(self.system, seed=self.seed)
        with self.assertRaises(GBManipulatorValueError):
            _ = manipulator.remove_atoms(0.50)

    @unittest.skip("Currently get hung on generating the neighbor list.")
    def test_remove_atoms_2_parent_warning(self):
        manipulator = GBManipulator(self.system, self.system, seed=self.seed)
        with self.assertWarns(UserWarning):
            _ = manipulator.remove_atoms(0.10)

    @unittest.skip("Currently get hung on generating the neighbor list.")
    def test_remove_atoms_calculated_fraction_warning(self):
        manipulator = GBManipulator(self.system, seed=self.seed)
        with self.assertWarns(UserWarning):
            _ = manipulator.remove_atoms(0.0000001)

    @unittest.skip("Currently get hung on generating the neighbor list.")
    def test_remove_atoms_with_specific_number(self):
        manipulator = GBManipulator(self.system, seed=self.seed)
        new_system = manipulator.remove_atoms(0.10, num_to_remove=1)
        self.assertEqual(len(self.system.gb)-1, len(new_system))

    def test_insert_atoms(self):
        manipulator = GBManipulator(self.system, seed=self.seed)
        new_system1 = manipulator.insert_atoms(fill_fraction=0.10, method='delaunay')
        self.assertGreater(len(new_system1), len(self.system.gb))
        new_system2 = manipulator.insert_atoms(fill_fraction=0.10, method='grid')
        self.assertGreater(len(new_system2), len(self.system.gb))

    def test_insert_atoms_fraction_error(self):
        manipulator = GBManipulator(self.system, seed=self.seed)
        with self.assertRaises(GBManipulatorValueError):
            _ = manipulator.insert_atoms(fill_fraction=0.50, method='delaunay')

    def test_insert_atoms_2_parent_warning(self):
        manipulator = GBManipulator(self.system, self.system, seed=self.seed)
        with self.assertWarns(UserWarning):
            _ = manipulator.insert_atoms(fill_fraction=0.10, method='delaunay')

    def test_insert_atoms_calculated_fraction_warning(self):
        manipulator = GBManipulator(self.system, seed=self.seed)
        with self.assertWarns(UserWarning):
            _ = manipulator.insert_atoms(fill_fraction=0.0000001, method='delaunay')

    def test_insert_atoms_invalid_method(self):
        manipulator = GBManipulator(self.system, seed=self.seed)
        with self.assertRaises(GBManipulatorValueError):
            _ = manipulator.insert_atoms(fill_fraction=0.10, method='invalid')

    def test_insert_atoms_with_specific_number(self):
        manipulator = GBManipulator(self.system, seed=self.seed)
        new_system1 = manipulator.insert_atoms(method='delaunay', num_to_insert=1)
        self.assertEqual(len(self.system.gb) + 1, len(new_system1))
        new_system2 = manipulator.insert_atoms(method='grid', num_to_insert=1)
        self.assertEqual(len(self.system.gb) + 1, len(new_system2))

    def test_displace_along_soft_modes(self):
        manipulator = GBManipulator(self.system, seed=self.seed)
        with self.assertRaises(NotImplementedError):
            _ = manipulator.displace_along_soft_modes()

    def test_apply_group_symmetry(self):
        manipulator = GBManipulator(self.system, seed=self.seed)
        with self.assertRaises(NotImplementedError):
            _ = manipulator.apply_group_symmetry("group")

    def test_parents_getter(self):
        manipulator = GBManipulator(self.system)
        parents = manipulator.parents
        self.assertTrue(isinstance(parents, _ParentsProxy))
        self.assertTrue(isinstance(parents[0], Parent))
        self.assertIsNone(parents[1])

    def test_parents_setter(self):
        manipulator = GBManipulator(self.system)
        self.assertIsNone(manipulator.parents[1])
        manipulator.parents[1] = Parent(self.system)
        self.assertIsNotNone(manipulator.parents[1])
        manipulator.parents = [Parent(self.file1, unit_cell=self.system.unit_cell),
                               Parent(self.file2, unit_cell=self.system.unit_cell)]
        self.assertFalse(None in manipulator.parents)

    def test_write_lammps_after_manipulate(self):
        manipulator = GBManipulator(self.system, self.system)
        p1 = manipulator.translate_right_grain(1, 1)
        p2 = manipulator.slice_and_merge()
        p3 = manipulator.insert_atoms(fill_fraction=0.2, method='delaunay')
        # p4 = manipulator.remove_atoms(0.2)
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            self.system.write_lammps(p1, self.system.box_dims, temp_file.name)
            self.system.write_lammps(p2, self.system.box_dims, temp_file.name)
            self.system.write_lammps(p3, self.system.box_dims, temp_file.name)
            # self.system.write_lammps(p4, self.system.box_dims, temp_file.name)


class TestParent(unittest.TestCase):
    def setUp(self):
        self.unit_cell = UnitCell()
        self.unit_cell.init_by_structure('fcc', 1.0)
        self.system = GBMaker(a0=1.0, structure='fcc', gb_thickness=10.0, misorientation=[
            math.radians(36.869898), 0, 0, 0, 0], repeat_factor=1)
        self.parent = Parent(self.system)
        self.file = 'tests/inputs/test1.txt'

    def test_parent_init(self):
        parent1 = Parent(self.system)
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

    def test_parent_getters(self):
        parent = Parent(self.system)
        self.assertEqual(repr(parent.unit_cell), repr(self.system.unit_cell))
        self.assertEqual(parent.gb_thickness, self.system.gb_thickness)
        self.assertTrue(np.allclose(parent.box_dims, self.system.box_dims))
        # Parent calculates x_dim differently than GB
        self.assertNotEqual(parent.x_dim, self.system.x_dim)
        self.assertEqual(parent.y_dim, self.system.y_dim)
        self.assertEqual(parent.z_dim, self.system.z_dim)

    def test_parent_snapshot_init_errors(self):
        with self.assertRaises(ParentValueError):
            _ = Parent(self.file)
        with self.assertRaises(ParentFileNotFoundError):
            _ = Parent("tests/inputs/file_not_found.txt", unit_cell=self.unit_cell)
        with self.assertRaises(ParentCorruptedSnapshotError):
            _ = Parent("tests/inputs/file_without_box_bounds.txt",
                       unit_cell=self.unit_cell)
        with self.assertRaises(ParentCorruptedSnapshotError):
            _ = Parent("tests/inputs/file_with_invalid_box_bounds.txt",
                       unit_cell=self.unit_cell)
        with self.assertRaises(ParentCorruptedSnapshotError):
            _ = Parent("tests/inputs/file_with_invalid_box_bounds2.txt",
                       unit_cell=self.unit_cell)
        with self.assertRaises(ParentCorruptedSnapshotError):
            _ = Parent("tests/inputs/file_without_atoms.txt",
                       unit_cell=self.unit_cell)
        with self.assertRaises(ParentSnapshotMissingDataError):
            _ = Parent("tests/inputs/file_missing_required_info.txt",
                       unit_cell=self.unit_cell)


class TestParentProxy(unittest.TestCase):
    def setUp(self):
        self.unit_cell = UnitCell()
        self.unit_cell.init_by_structure('fcc', 1.0)
        self.manipulator = GBManipulator(
            'tests/inputs/test1.txt', unit_cell=self.unit_cell)
        self.parents_proxy = _ParentsProxy(self.manipulator)

    def test_getitem(self):
        parent = self.parents_proxy[0]
        self.assertIsInstance(parent, Parent)

    def test_setitem(self):
        new_parent = Parent(GBMaker(a0=3.61, structure='fcc', gb_thickness=10.0, misorientation=[
            math.radians(36.869898), 0, 0, 0, 0], repeat_factor=1))
        self.parents_proxy[0] = new_parent
        self.assertIs(self.parents_proxy[0], new_parent)

    def test_len(self):
        self.assertEqual(len(self.parents_proxy), 2)

    def test_setitem_errors(self):
        self.parents_proxy[0] = None
        new_parent = Parent(GBMaker(a0=1.0, structure='fcc', gb_thickness=10.0, misorientation=[
            math.radians(36.869898), 0, 0, 0, 0], repeat_factor=1))
        with self.assertRaises(ParentsProxyValueError):
            self.parents_proxy[1] = new_parent
        self.parents_proxy[0] = new_parent
        with self.assertRaises(ParentsProxyIndexError):
            self.parents_proxy[2] = new_parent
        with self.assertRaises(ParentsProxyTypeError):
            self.parents_proxy[0] = 1


if __name__ == '__main__':
    unittest.main()
