import math
import unittest

import numpy as np

from GBOpt.Atom import Atom
from GBOpt.UnitCell import UnitCell, UnitCellTypeError, UnitCellValueError


class TestUnitCell(unittest.TestCase):

    def test_fcc_initialization(self):
        cell = UnitCell()
        cell.init_by_structure(structure='fcc', a0=1.0, atoms='Cu')
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
        names = cell.names()
        self.assertEqual(len(set(names)), 1)
        self.assertEqual(len(positions), len(names))
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
        self.assertTrue(list(cell.ideal_bond_lengths.keys()) == [(1, 1)])
        self.assertTrue(math.isclose(cell.ideal_bond_lengths[(1, 1)], 1 / np.sqrt(2)))
        self.assertEqual(cell.ratio, {1: 1})

    def test_bcc_initialization(self):
        cell = UnitCell()
        cell.init_by_structure(structure='bcc', a0=1.0, atoms='Fe')
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
        names = cell.names()
        self.assertEqual(len(set(names)), 1)
        self.assertEqual(len(positions), len(names))
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
        self.assertTrue(list(cell.ideal_bond_lengths.keys()) == [(1, 1)])
        self.assertTrue(math.isclose(cell.ideal_bond_lengths[(1, 1)], np.sqrt(3) / 2))
        self.assertEqual(cell.ratio, {1: 1})

    def test_sc_initialization(self):
        cell = UnitCell()
        cell.init_by_structure(structure='sc', a0=1.0, atoms='H')
        self.assertEqual(cell.a0, 1.0)
        self.assertEqual(cell.radius, 0.5)
        self.assertEqual(len(cell.unit_cell), 1)
        positions = cell.positions()
        self.assertTrue(np.allclose(positions, np.array([[0.0, 0.0, 0.0]])))
        names = cell.names()
        self.assertEqual(len(set(names)), 1)
        self.assertEqual(len(positions), len(names))
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
        self.assertTrue(list(cell.ideal_bond_lengths.keys()) == [(1, 1)])
        self.assertTrue(math.isclose(cell.ideal_bond_lengths[(1, 1)], 1))
        self.assertEqual(cell.ratio, {1: 1})

    def test_diamond_initialization(self):
        cell = UnitCell()
        cell.init_by_structure(structure='diamond', a0=1.0, atoms='C')
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
        names = cell.names()
        self.assertEqual(len(set(names)), 1)
        self.assertEqual(len(positions), len(names))
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
        self.assertTrue(list(cell.ideal_bond_lengths.keys()) == [(1, 1)])
        self.assertTrue(math.isclose(cell.ideal_bond_lengths[(1, 1)], np.sqrt(3) / 4))
        self.assertEqual(cell.ratio, {1: 1})

    def test_fluorite_initialization(self):
        cell = UnitCell()
        cell.init_by_structure(structure='fluorite', a0=1.0, atoms=['Ca', 'F'])
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
        names = cell.names()
        self.assertEqual(len(set(names)), 2)
        for i in range(4):
            self.assertEqual(names[i], 'Ca')
        for i in range(4, 12):
            self.assertEqual(names[i], 'F')
        self.assertEqual(len(positions), len(names))
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
        ideal_fluorite_bonds = {
            (1, 1): 1 / np.sqrt(2),
            (1, 2): np.sqrt(3) / 4,
            (2, 2): 0.5
        }
        self.assertTrue(cell.ideal_bond_lengths.keys() == ideal_fluorite_bonds.keys())
        self.assertTrue(all([math.isclose(cell.ideal_bond_lengths[i],
                        ideal_fluorite_bonds[i]) for i in ideal_fluorite_bonds.keys()]))
        self.assertEqual(cell.ratio, {1: 1, 2: 2})

    def test_rocksalt_initialization(self):
        cell = UnitCell()
        cell.init_by_structure(structure='rocksalt', a0=1.0, atoms=['Na', 'Cl'])
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
        names = cell.names()
        self.assertEqual(len(set(names)), 2)
        for i in range(4):
            self.assertEqual(names[i], 'Na')
        for i in range(4, 8):
            self.assertEqual(names[i], 'Cl')
        self.assertEqual(len(positions), len(names))
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
        ideal_rocksalt_bonds = {
            (1, 1): 1 / np.sqrt(2),
            (1, 2): 0.5,
            (2, 2): 1 / np.sqrt(2)
        }
        self.assertTrue(cell.ideal_bond_lengths.keys() == ideal_rocksalt_bonds.keys())
        self.assertTrue(all([math.isclose(cell.ideal_bond_lengths[i],
                        ideal_rocksalt_bonds[i]) for i in ideal_rocksalt_bonds.keys()]))
        self.assertEqual(cell.ratio, {1: 1, 2: 1})

    def test_zincblende_initialization(self):
        cell = UnitCell()
        cell.init_by_structure(structure='zincblende', a0=1.0, atoms=['Zn', 'S'])
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
        names = cell.names()
        self.assertEqual(len(set(names)), 2)
        for i in range(4):
            self.assertEqual(names[i], 'Zn')
        for i in range(4, 8):
            self.assertEqual(names[i], 'S')
        self.assertEqual(len(positions), len(names))
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
        ideal_zincblende_bonds = {
            (1, 1): 1 / np.sqrt(2),
            (1, 2): np.sqrt(3) / 4,
            (2, 2): 1 / np.sqrt(2)
        }
        self.assertTrue(cell.ideal_bond_lengths.keys() == ideal_zincblende_bonds.keys())
        self.assertTrue(all([math.isclose(cell.ideal_bond_lengths[i],
                        ideal_zincblende_bonds[i]) for i in ideal_zincblende_bonds.keys()]))
        self.assertEqual(cell.ratio, {1: 1, 2: 1})

    def test_custom_initialization(self):
        cell = UnitCell()
        custom_unit_cell = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        custom_reciprocal = np.array([[1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        custom_ideal_bonds = {
            (1, 1): 1,
            (1, 2): 1,
            (2, 2): 1
        }
        custom_ratio = {1: 1, 2: 1}
        custom_type_map = {'C': 1, 'H': 2}
        with self.assertRaises(UnitCellValueError):
            cell.init_by_custom(unit_cell=custom_unit_cell,
                                unit_cell_types='H',
                                a0=1.0,
                                reciprocal=custom_reciprocal,
                                ideal_bond_lengths=custom_ideal_bonds
                                )
        with self.assertRaises(UnitCellValueError):
            cell.init_by_custom(unit_cell=custom_unit_cell,
                                unit_cell_types=['H', 'C'],
                                a0=1.0,
                                reciprocal=custom_reciprocal,
                                ideal_bond_lengths=custom_ideal_bonds,
                                ratio={1: 0, 2: 2}
                                )
        with self.assertRaises(UnitCellValueError):
            cell.init_by_custom(unit_cell=custom_unit_cell,
                                unit_cell_types=['H', 'C'],
                                a0=1.0,
                                reciprocal=custom_reciprocal,
                                ideal_bond_lengths=custom_ideal_bonds,
                                ratio={0: 1, 1: 1}
                                )
        with self.assertRaises(UnitCellTypeError):
            cell.init_by_custom(unit_cell=custom_unit_cell,
                                unit_cell_types=['H', 'C'],
                                a0=1.0,
                                reciprocal=custom_reciprocal,
                                ideal_bond_lengths=custom_ideal_bonds,
                                ratio="Error"
                                )
        with self.assertRaises(UnitCellTypeError):
            cell.init_by_custom(unit_cell=custom_unit_cell,
                                unit_cell_types=['H', 'C'],
                                a0=1.0,
                                reciprocal=custom_reciprocal,
                                ideal_bond_lengths=custom_ideal_bonds,
                                ratio="Error"
                                )
        cell.init_by_custom(unit_cell=custom_unit_cell,
                            unit_cell_types=['H', 'C'],
                            a0=1.0,
                            reciprocal=custom_reciprocal,
                            ideal_bond_lengths=custom_ideal_bonds,
                            ratio=custom_ratio
                            )
        self.assertEqual(cell.a0, 1.0)
        self.assertEqual(len(cell.unit_cell), 2)
        names = cell.names()
        self.assertEqual(names[0], 'H')
        self.assertEqual(names[1], 'C')
        positions = cell.positions()
        self.assertTrue(np.allclose(positions, custom_unit_cell))
        names = cell.names()
        self.assertEqual(len(set(names)), 2)
        self.assertEqual(len(positions), len(names))
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
        self.assertTrue(cell.ideal_bond_lengths.keys() == custom_ideal_bonds.keys())
        self.assertTrue(all([math.isclose(cell.ideal_bond_lengths[i],
                        custom_ideal_bonds[i]) for i in custom_ideal_bonds.keys()]))
        self.assertEqual(cell.ratio, custom_ratio)
        self.assertEqual(cell.type_map, {'H': 1, 'C': 2})
        cell.type_map = custom_type_map
        self.assertEqual(cell.type_map, custom_type_map)

    def test_not_implemented_structure(self):
        cell = UnitCell()
        with self.assertRaises(NotImplementedError):
            cell.init_by_structure(structure='notimplemented', a0=1.0, atoms='H')

    def test_inits_with_multiple_atoms_one_atom_type(self):
        cell = UnitCell()
        with self.assertRaises(UnitCellValueError):
            cell.init_by_structure('fluorite', 1.0, 'Ca')
        with self.assertRaises(UnitCellValueError):
            cell.init_by_structure('rocksalt', 1.0, 'Na')
        with self.assertRaises(UnitCellValueError):
            cell.init_by_structure('zincblende', 1.0, 'Zn')

    def test_reciprocal_shape(self):
        cell = UnitCell()
        custom_unit_cell = np.array([[0.1, 0.2, 0.3]])
        custom_reciprocal = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        custom_ideal_bonds = {
            (1, 1): 1,
            (1, 2): 1,
            (2, 2): 1
        }
        with self.assertRaises(UnitCellValueError):
            cell.init_by_custom(unit_cell=custom_unit_cell,
                                unit_cell_types='H',
                                a0=1.0,
                                reciprocal=custom_reciprocal,
                                ideal_bond_lengths=custom_ideal_bonds
                                )

    def test_lattice_parameter(self):
        cell = UnitCell()
        cell.init_by_structure(structure='fcc', a0=3.54, atoms='Cu')
        cu_unit_cell = [
            Atom('Cu', 0.0, 0.0, 0.0),
            Atom('Cu', 0.0, 1.77, 1.77),
            Atom('Cu', 1.77, 0.0, 1.77),
            Atom('Cu', 1.77, 1.77, 0.0)
        ]
        cu_positions = [i['position'] for i in cu_unit_cell]
        cell_positions = [i['position'] for i in cell.unit_cell]
        self.assertEqual(cell_positions, cu_positions)

    def test_repr(self):
        cell = UnitCell()
        self.assertEqual(
            repr(cell),
            "UnitCell with 0 atoms\nLattice parameter (a0): 1.000 Å\nRadius: 0.000 Å\n"
            "Atoms: []\nReciprocal lattice:\n[[0. 0. 0.]\n [0. 0. 0.]\n [0. 0. 0.]]")
        cell.init_by_structure('sc', 1.0, 'H')
        self.assertEqual(
            repr(cell),
            "UnitCell with 1 atom\nLattice parameter (a0): 1.000 Å\nRadius: 0.500 Å\n"
            "Atoms: ['H': 0.000, 0.000, 0.000]\nReciprocal lattice:\n[[1. 0. 0.]\n [0. 1. 0.]\n [0. 0. 1.]]"
        )
        custom_ideal_bonds = {
            (1, 1): 1,
            (1, 2): 1,
            (2, 2): 1
        }
        cell.init_by_custom(unit_cell=[[0, 0, 0]],
                            unit_cell_types='H',
                            a0=1.0,
                            reciprocal=[[1.0, 0.0, 0.0], [
                                0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                            ideal_bond_lengths=custom_ideal_bonds)
        self.assertEqual(
            repr(cell),
            "UnitCell with 1 atom\nLattice parameter (a0): 1.000 Å\nRadius: 0.500 Å\n"
            "Atoms: ['H': 0.000, 0.000, 0.000]\nReciprocal lattice:\n[[1. 0. 0.]\n [0. 1. 0.]\n [0. 0. 1.]]"
        )

    def test_asarray(self):
        cell = UnitCell()
        cell.init_by_structure('fcc', 1.0, 'Cu')
        expected = np.array([('Cu', 0.0, 0.0, 0.0),
                             ('Cu', 0.0, 0.5, 0.5),
                             ('Cu', 0.5, 0.0, 0.5),
                             ('Cu', 0.5, 0.5, 0.0)], dtype=Atom.atom_dtype)
        result = cell.asarray()
        self.assertTrue(all(result == expected))

        np.testing.assert_array_equal(result['name'], expected['name'])
        np.testing.assert_array_equal(result['x'], expected['x'])
        np.testing.assert_array_equal(result['y'], expected['y'])
        np.testing.assert_array_equal(result['z'], expected['z'])

    def test_init(self):
        cell = UnitCell()
        self.assertEqual(cell.unit_cell, [])
        self.assertTrue(np.allclose(cell.primitive, np.zeros((3, 3))))
        self.assertEqual(cell.a0, 1.0)
        self.assertEqual(cell.radius, 0.0)
        self.assertTrue(np.allclose(cell.reciprocal, np.zeros((3, 3))))
        self.assertEqual(cell.type_map, {})

    def test_names_as_ints(self):
        cell = UnitCell()
        cell.init_by_structure(structure='fcc', a0=1.0, atoms='Cu')
        self.assertTrue(np.allclose(cell.names(asint=True),
                        np.array([1, 1, 1, 1], dtype=int)))

        cell.init_by_structure(
            "fluorite", 5.52, ["Ca", "F"], type_map={"Ca": 2, "F": 1})
        self.assertTrue(all(cell.names(asint=True) == [
                        2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]))

    def test_setter(self):
        cell = UnitCell()
        cell.init_by_structure('fcc', 1.0, 'Ni')
        with self.assertRaises(UnitCellValueError):
            cell.a0 = -1.0
        cell.a0 = 2.0
        self.assertTrue(np.allclose(cell.primitive, np.array(
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]])))
        self.assertEqual(cell.radius, math.sqrt(2) * 0.25 * 2)
        self.assertTrue(np.allclose(cell.reciprocal, np.array(
            [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])))

    def test_fcc_nn_calculations(self):
        cell = UnitCell()
        cell.init_by_structure("fcc", 1.0, "Cu")
        self.assertAlmostEqual(cell.nn_distance(1), 1/np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(2), 1)
        self.assertAlmostEqual(cell.nn_distance(3), np.sqrt(6)/2)
        self.assertAlmostEqual(cell.nn_distance(4), np.sqrt(2))

        cell.a0 = 3.54
        self.assertAlmostEqual(cell.nn_distance(1), cell.a0 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(2), cell.a0)
        self.assertAlmostEqual(cell.nn_distance(3), cell.a0 * np.sqrt(6) / 2)
        self.assertAlmostEqual(cell.nn_distance(4), cell.a0 * np.sqrt(2))

    def test_bcc_nn_calculations(self):
        cell = UnitCell()
        cell.init_by_structure("bcc", 1.0, "Fe")
        self.assertAlmostEqual(cell.nn_distance(1), np.sqrt(3) / 2)
        self.assertAlmostEqual(cell.nn_distance(2), 1)
        self.assertAlmostEqual(cell.nn_distance(3), np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(4), np.sqrt(11) / 2)

        cell.a0 = 2.86
        self.assertAlmostEqual(cell.nn_distance(1), cell.a0 * np.sqrt(3) / 2)
        self.assertAlmostEqual(cell.nn_distance(2), cell.a0)
        self.assertAlmostEqual(cell.nn_distance(3), cell.a0 * np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(4), cell.a0 * np.sqrt(11) / 2)

    def test_sc_nn_calculations(self):
        cell = UnitCell()
        cell.init_by_structure("sc", 1.0, "Po")
        self.assertAlmostEqual(cell.nn_distance(1), 1)
        self.assertAlmostEqual(cell.nn_distance(2), np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(3), np.sqrt(3))
        self.assertAlmostEqual(cell.nn_distance(4), 2)

        cell.a0 = 3.345
        self.assertAlmostEqual(cell.nn_distance(1), cell.a0)
        self.assertAlmostEqual(cell.nn_distance(2), cell.a0 * np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(3), cell.a0 * np.sqrt(3))
        self.assertAlmostEqual(cell.nn_distance(4), cell.a0 * 2)

    def test_type_map_property(self):
        cell = UnitCell()
        cell.init_by_structure("fcc", 1.0, "Cu")

        self.assertEqual(cell.type_map, {"Cu": 1})

        cell.init_by_structure("fluorite", 2.0, ["U", "O"])
        self.assertEqual(cell.type_map, {"U": 1, "O": 2})
        cell.type_map = {"O": 1, "U": 2}
        self.assertEqual(cell.type_map, {"U": 2, "O": 1})
        cell.type_map = {1: "U", 2: "O"}
        self.assertEqual(cell.type_map, {"U": 1, "O": 2})

        with self.assertRaises(UnitCellValueError):
            cell.type_map = {2: "U", 3: "O"}

        with self.assertRaises(UnitCellValueError):
            cell.type_map = {"O": 2, "U": 3}

        with self.assertRaises(UnitCellTypeError):
            cell.type_map = "Error"

    def test_types_method(self):
        cell = UnitCell()
        cell.init_by_structure("fluorite", 5.52, ["Ca", "F"])
        self.assertTrue(all(cell.types() == [1, 2]))
        cell.type_map = {"F": 1, "Ca": 2}
        self.assertTrue(all(cell.types() == [2, 1]))


if __name__ == '__main__':
    unittest.main()
