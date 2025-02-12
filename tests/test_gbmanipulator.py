import filecmp
import math
import tempfile
import unittest
import warnings

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
        theta = math.radians(36.869898)
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

    def setUp(self):
        self.a0 = 1.0
        self.structure = 'fcc'
        self.gb_thickness = 10.0
        self.atom_types = 'Cu'
        self.misorientation = [math.radians(36.869898), 0, 0, 0, 0]
        self.file1 = "tests/inputs/test1.txt"
        self.file2 = "tests/inputs/test2.txt"
        self.seed = 100

    def test_init_with_one_gbmaker_parent(self):
        manipulator = GBManipulator(self.tilt)
        self.assertIsNotNone(manipulator.parents[0])
        self.assertIsNone(manipulator.parents[1])
        self.assertTrue(manipulator._GBManipulator__one_parent)

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
        manipulator = GBManipulator(self.tilt)
        new_system = manipulator.translate_right_grain(1.0, 0.5)
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
        manipulator = GBManipulator(self.tilt, seed=self.seed)
        with self.assertRaises(GBManipulatorValueError):
            _ = manipulator.slice_and_merge()

    def test_remove_atoms(self):
        manipulator = GBManipulator(self.tilt, seed=self.seed)
        new_system = manipulator.remove_atoms(gb_fraction=0.10)
        self.assertGreater(len(self.tilt.whole_system), len(new_system))

    def test_remove_atoms_fraction_error(self):
        manipulator = GBManipulator(self.tilt, seed=self.seed)
        with self.assertRaises(GBManipulatorValueError):
            _ = manipulator.remove_atoms(gb_fraction=0.50)

    def test_remove_atoms_2_parent_warning(self):
        manipulator = GBManipulator(self.tilt, self.tilt, seed=self.seed)
        with self.assertWarns(UserWarning):
            _ = manipulator.remove_atoms(gb_fraction=0.10)

    def test_remove_atoms_calculated_fraction_warning(self):
        manipulator = GBManipulator(self.tilt, seed=self.seed)
        with self.assertWarns(UserWarning):
            _ = manipulator.remove_atoms(gb_fraction=0.0000001)

    def test_remove_atoms_with_specific_number(self):
        manipulator = GBManipulator(self.tilt, seed=self.seed)
        new_system = manipulator.remove_atoms(num_to_remove=1)
        self.assertEqual(len(self.tilt.whole_system)-1, len(new_system))

    def test_insert_atoms(self):
        manipulator = GBManipulator(self.tilt, seed=self.seed)
        new_system1 = manipulator.insert_atoms(fill_fraction=0.10, method='delaunay')
        self.assertGreater(len(new_system1), len(self.tilt.whole_system))
        new_system2 = manipulator.insert_atoms(fill_fraction=0.10, method='grid')
        self.assertGreater(len(new_system2), len(self.tilt.whole_system))

    def test_insert_atoms_fraction_error(self):
        manipulator = GBManipulator(self.tilt, seed=self.seed)
        with self.assertRaises(GBManipulatorValueError):
            _ = manipulator.insert_atoms(fill_fraction=0.50, method='delaunay')

    def test_insert_atoms_2_parent_warning(self):
        manipulator = GBManipulator(self.tilt, self.tilt, seed=self.seed)
        with self.assertWarns(UserWarning):
            _ = manipulator.insert_atoms(fill_fraction=0.10, method='delaunay')

    def test_insert_atoms_calculated_fraction_warning(self):
        manipulator = GBManipulator(self.tilt, seed=self.seed)
        with self.assertWarns(UserWarning):
            _ = manipulator.insert_atoms(fill_fraction=0.0000001, method='delaunay')

        with self.assertWarns(UserWarning):
            _ = manipulator.insert_atoms(fill_fraction=0.0000001, method='grid')

    def test_insert_atoms_invalid_method(self):
        manipulator = GBManipulator(self.tilt, seed=self.seed)
        with self.assertRaises(GBManipulatorValueError):
            _ = manipulator.insert_atoms(fill_fraction=0.10, method='invalid')

    def test_insert_atoms_with_specific_number(self):
        manipulator = GBManipulator(self.tilt, seed=self.seed)
        new_system1 = manipulator.insert_atoms(method='delaunay', num_to_insert=1)
        self.assertEqual(len(self.tilt.whole_system) + 1, len(new_system1))
        new_system2 = manipulator.insert_atoms(method='delaunay', num_to_insert=1)
        self.assertEqual(len(self.tilt.whole_system) + 1, len(new_system2))

    def test_displace_along_soft_modes(self):
        manipulator = GBManipulator(self.tilt, seed=self.seed)
        with self.assertRaises(NotImplementedError):
            _ = manipulator.displace_along_soft_modes()

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
        self.GB = GBMaker(a0=1.0, structure='fcc', gb_thickness=10.0, misorientation=[
            math.radians(36.869898), 0, 0, 0, 0], atom_types='Cu', repeat_factor=1)
        self.parent = Parent(self.GB)
        self.file = 'tests/inputs/test1.txt'

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

        left_x_max = max(parent.left_grain['x'])
        right_x_min = min(parent.right_grain['x'])
        left_cut = left_x_max - parent.gb_thickness / 2.0
        right_cut = right_x_min + parent.gb_thickness / 2.0
        left_gb_indices = parent.left_grain['x'] > left_cut
        right_gb_indices = parent.right_grain['x'] < right_cut
        left_gb = parent.left_grain[left_gb_indices]
        right_gb = parent.right_grain[right_gb_indices]
        self.assertTrue(all(parent.gb_atoms == np.hstack((left_gb, right_gb))))
        self.assertTrue(all(parent.left_grain == self.GB.left_grain))
        self.assertTrue(all(parent.right_grain == self.GB.right_grain))
        self.assertTrue(all(parent.whole_system == self.GB.whole_system))
        self.assertEqual(parent.unit_cell, self.GB.unit_cell)

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

    def test_read_lammps_with_typelabel(self):
        parent = Parent("tests/inputs/test3.txt",
                        unit_cell=self.unit_cell, gb_thickness=20)
        self.assertEqual(parent.whole_system[0]['name'], 'Cu')


class TestParentProxy(unittest.TestCase):
    def setUp(self):
        self.unit_cell = UnitCell()
        self.unit_cell.init_by_structure('fcc', 1.0, 'Cu')
        self.manipulator = GBManipulator(
            'tests/inputs/test1.txt', unit_cell=self.unit_cell)
        self.parents_proxy = _ParentsProxy(self.manipulator)

    def test_getitem(self):
        parent = self.parents_proxy[0]
        self.assertIsInstance(parent, Parent)

    def test_setitem(self):
        new_parent = Parent(GBMaker(a0=3.61, structure='fcc', gb_thickness=10.0, misorientation=[
            math.radians(36.869898), 0, 0, 0, 0], atom_types='Cu', repeat_factor=1))
        self.parents_proxy[0] = new_parent
        self.assertIs(self.parents_proxy[0], new_parent)

    def test_len(self):
        self.assertEqual(len(self.parents_proxy), 2)

    def test_setitem_errors(self):
        self.parents_proxy[0] = None
        new_parent = Parent(GBMaker(a0=1.0, structure='fcc', gb_thickness=10.0, misorientation=[
            math.radians(36.869898), 0, 0, 0, 0], atom_types='Cu', repeat_factor=1))
        with self.assertRaises(ParentsProxyValueError):
            self.parents_proxy[1] = new_parent
        self.parents_proxy[0] = new_parent
        with self.assertRaises(ParentsProxyIndexError):
            self.parents_proxy[2] = new_parent
        with self.assertRaises(ParentsProxyTypeError):
            self.parents_proxy[0] = 1


if __name__ == '__main__':
    unittest.main()
