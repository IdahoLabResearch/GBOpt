import math
import unittest

import numpy as np

from GBOpt.UnitCell import UnitCell, UnitCellValueError


class TestUnitCell(unittest.TestCase):

    def test_fcc_initialization(self):
        cell = UnitCell()
        cell.init_by_structure(structure='fcc', a0=1.0)
        self.assertEqual(cell.a0, 1.0)
        self.assertEqual(cell.radius, math.sqrt(2) * 0.25)
        self.assertEqual(len(cell.unit_cell), 4)
        positions = cell.positions()
        self.assertTrue(
            np.allclose(
                positions,
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0]
                    ]
                )
            )
        )
        types = cell.types()
        self.assertEqual(len(set(types)), 1)
        self.assertEqual(len(positions), len(types))
        reciprocal = cell.reciprocal
        self.assertTrue(
            np.allclose(
                reciprocal,
                np.array(
                    [
                        [-1.0, 1.0, 1.0],
                        [1.0, -1.0, 1.0],
                        [1.0, 1.0, -1.0]
                    ]
                )
            )
        )

    def test_bcc_initialization(self):
        cell = UnitCell()
        cell.init_by_structure(structure='bcc', a0=1.0)
        self.assertEqual(cell.a0, 1.0)
        self.assertEqual(cell.radius, math.sqrt(3) * 0.25)
        self.assertEqual(len(cell.unit_cell), 2)
        positions = cell.positions()
        self.assertTrue(
            np.allclose(
                positions,
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.5, 0.5, 0.5]
                    ]
                )
            )
        )
        types = cell.types()
        self.assertEqual(len(set(types)), 1)
        self.assertEqual(len(positions), len(types))
        reciprocal = cell.reciprocal
        self.assertTrue(
            np.allclose(
                reciprocal,
                np.array(
                    [
                        [1.0, 1.0, 0.0],
                        [1.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0]
                    ]
                )
            )
        )

    def test_sc_initialization(self):
        cell = UnitCell()
        cell.init_by_structure(structure='sc', a0=1.0)
        self.assertEqual(cell.a0, 1.0)
        self.assertEqual(cell.radius, 0.5)
        self.assertEqual(len(cell.unit_cell), 1)
        positions = cell.positions()
        self.assertTrue(np.allclose(positions, np.array([[0.0, 0.0, 0.0]])))
        types = cell.types()
        self.assertEqual(len(set(types)), 1)
        self.assertEqual(len(positions), len(types))
        reciprocal = cell.reciprocal
        self.assertTrue(
            np.allclose(
                reciprocal,
                np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]
                    ]
                )
            )
        )

    def test_diamond_initialization(self):
        cell = UnitCell()
        cell.init_by_structure(structure='diamond', a0=1.0)
        self.assertEqual(cell.a0, 1.0)
        self.assertEqual(cell.radius, math.sqrt(3) * 0.125)
        self.assertEqual(len(cell.unit_cell), 8)
        positions = cell.positions()
        self.assertTrue(
            np.allclose(
                positions,
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                        [0.25, 0.25, 0.25],
                        [0.75, 0.75, 0.25],
                        [0.75, 0.25, 0.75],
                        [0.25, 0.75, 0.75]
                    ]
                )
            )
        )
        types = cell.types()
        self.assertEqual(len(set(types)), 1)
        self.assertEqual(len(positions), len(types))
        reciprocal = cell.reciprocal
        self.assertTrue(
            np.allclose(
                reciprocal,
                np.array(
                    [
                        [-1.0, 1.0, 1.0],
                        [1.0, -1.0, 1.0],
                        [1.0, 1.0, -1.0]
                    ]
                )
            )
        )

    def test_fluorite_initialization(self):
        cell = UnitCell()
        cell.init_by_structure(structure='fluorite', a0=1.0)
        self.assertEqual(cell.a0, 1.0)
        self.assertEqual(cell.radius, math.sqrt(3) * 0.125)
        self.assertEqual(len(cell.unit_cell), 12)
        positions = cell.positions()
        self.assertTrue(
            np.allclose(
                positions,
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                        [0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.75],
                        [0.25, 0.75, 0.25],
                        [0.25, 0.75, 0.75],
                        [0.75, 0.25, 0.25],
                        [0.75, 0.25, 0.75],
                        [0.75, 0.75, 0.25],
                        [0.75, 0.75, 0.75],
                    ]
                )
            )
        )
        types = cell.types()
        self.assertEqual(len(set(types)), 2)
        self.assertEqual(len(positions), len(types))
        reciprocal = cell.reciprocal
        self.assertTrue(
            np.allclose(
                reciprocal,
                np.array(
                    [
                        [-1.0, 1.0, 1.0],
                        [1.0, -1.0, 1.0],
                        [1.0, 1.0, -1.0]
                    ]
                )
            )
        )

    def test_rocksalt_initialization(self):
        cell = UnitCell()
        cell.init_by_structure(structure='rocksalt', a0=1.0)
        self.assertEqual(cell.a0, 1.0)
        self.assertEqual(cell.radius, 0.25)
        self.assertEqual(len(cell.unit_cell), 8)
        positions = cell.positions()
        self.assertTrue(
            np.allclose(
                positions,
                np.array(
                    [
                        [0, 0, 0],
                        [0, 0.5, 0.5],
                        [0.5, 0, 0.5],
                        [0.5, 0.5, 0],
                        [0, 0, 0.5],
                        [0, 0.5, 0],
                        [0.5, 0, 0],
                        [0.5, 0.5, 0.5],
                    ]
                )
            )
        )
        types = cell.types()
        self.assertEqual(len(set(types)), 2)
        self.assertEqual(len(positions), len(types))
        reciprocal = cell.reciprocal
        self.assertTrue(
            np.allclose(
                reciprocal,
                np.array(
                    [
                        [-1.0, 1.0, 1.0],
                        [1.0, -1.0, 1.0],
                        [1.0, 1.0, -1.0]
                    ]
                )
            )
        )

    def test_zincblende_initialization(self):
        cell = UnitCell()
        cell.init_by_structure(structure='zincblende', a0=1.0)
        self.assertEqual(cell.a0, 1.0)
        self.assertEqual(cell.radius, math.sqrt(3) * 0.125)
        self.assertEqual(len(cell.unit_cell), 8)
        positions = cell.positions()
        self.assertTrue(
            np.allclose(
                positions,
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                        [0.25, 0.25, 0.25],
                        [0.75, 0.75, 0.25],
                        [0.75, 0.25, 0.75],
                        [0.25, 0.75, 0.75],
                    ]
                )
            )
        )
        types = cell.types()
        self.assertEqual(len(set(types)), 2)
        self.assertEqual(len(positions), len(types))
        reciprocal = cell.reciprocal
        self.assertTrue(
            np.allclose(
                reciprocal,
                np.array(
                    [
                        [-1.0, 1.0, 1.0],
                        [1.0, -1.0, 1.0],
                        [1.0, 1.0, -1.0]
                    ]
                )
            )
        )

    def test_custom_initialization(self):
        cell = UnitCell()
        custom_unit_cell = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        custom_reciprocal = np.array([[1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        cell.init_by_custom(unit_cell=custom_unit_cell,
                            unit_cell_types=[1, 2],
                            a0=1.0,
                            reciprocal=custom_reciprocal
                            )
        self.assertEqual(cell.a0, 1.0)
        self.assertEqual(len(cell.unit_cell), 2)
        self.assertEqual(cell.unit_cell[0].atom_type, 1)
        self.assertEqual(cell.unit_cell[1].atom_type, 2)
        positions = cell.positions()
        self.assertTrue(np.allclose(positions, custom_unit_cell))
        types = cell.types()
        self.assertEqual(len(set(types)), 2)
        self.assertEqual(len(positions), len(types))
        reciprocal = cell.reciprocal
        self.assertTrue(
            np.allclose(
                reciprocal,
                np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]
                    ]
                )
            )
        )

    def test_type_warning(self):
        cell = UnitCell()
        custom_unit_cell = np.array([[0.1, 0.2, 0.3]])
        custom_reciprocal = np.array([[1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with self.assertWarns(Warning):
            cell.init_by_custom(unit_cell=custom_unit_cell,
                                unit_cell_types=2,
                                a0=1.0,
                                reciprocal=custom_reciprocal
                                )

    def test_type_shift_warning(self):
        cell = UnitCell()
        custom_unit_cell = np.array([[0.1, 0.2, 0.3]])
        custom_reciprocal = np.array([[1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with self.assertWarns(Warning):
            cell.init_by_custom(unit_cell=custom_unit_cell,
                                unit_cell_types=[2],
                                a0=1.0,
                                reciprocal=custom_reciprocal
                                )

    def test_not_implemented_structure(self):
        cell = UnitCell()
        with self.assertRaises(NotImplementedError):
            cell.init_by_structure(structure='hexagonal', a0=1.0)

    def test_reciprocal_shape(self):
        cell = UnitCell()
        custom_unit_cell = np.array([[0.1, 0.2, 0.3]])
        custom_reciprocal = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with self.assertRaises(UnitCellValueError):
            cell.init_by_custom(unit_cell=custom_unit_cell,
                                unit_cell_types=1,
                                a0=1.0,
                                reciprocal=custom_reciprocal
                                )

    def test_repr(self):
        cell = UnitCell()
        self.assertEqual(
            repr(cell),
            "UnitCell with 0 atoms\nLattice parameter (a0): 1.000 Å\nRadius: 0.000 Å\n"
            "Atoms: []\nReciprocal lattice:\n[[0. 0. 0.]\n [0. 0. 0.]\n [0. 0. 0.]]")
        cell.init_by_structure('sc', 1.0)
        self.assertEqual(
            repr(cell),
            "UnitCell with 1 atom\nLattice parameter (a0): 1.000 Å\nRadius: 0.500 Å\n"
            "Atoms: [1 (1): 0.000, 0.000, 0.000]\nReciprocal lattice:\n[[1. 0. 0.]\n [0. 1. 0.]\n [0. 0. 1.]]"
        )

        cell.init_by_custom(unit_cell=[[0, 0, 0]],
                            unit_cell_types=1,
                            a0=1.0,
                            reciprocal=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(
            repr(cell),
            "UnitCell with 1 atom\nLattice parameter (a0): 1.000 Å\nRadius: 0.500 Å\n"
            "Atoms: [1 (1): 0.000, 0.000, 0.000]\nReciprocal lattice:\n[[1. 0. 0.]\n [0. 1. 0.]\n [0. 0. 1.]]"
        )

    def test_init(self):
        cell = UnitCell()
        self.assertEqual(cell.unit_cell, [])
        self.assertTrue(np.allclose(cell.primitive, np.zeros((3, 3))))
        self.assertEqual(cell.a0, 1.0)
        self.assertEqual(cell.radius, 0.0)
        self.assertTrue(np.allclose(cell.reciprocal, np.zeros((3, 3))))

    def test_setter(self):
        cell = UnitCell()
        cell.init_by_structure('fcc', 1.0)
        with self.assertRaises(UnitCellValueError):
            cell.a0 = -1.0
        cell.a0 = 2.0
        self.assertTrue(np.allclose(cell.primitive, np.array(
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]])))
        self.assertEqual(cell.radius, math.sqrt(2) * 0.25 * 2)
        self.assertTrue(np.allclose(cell.reciprocal, np.array(
            [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])))


if __name__ == '__main__':
    unittest.main()
