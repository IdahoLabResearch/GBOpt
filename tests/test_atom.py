import unittest

import numpy as np

from GBOpt.Atom import Atom, AtomKeyError, AtomTypeError, AtomValueError
from GBOpt.Position import Position, PositionValueError


class TestAtom(unittest.TestCase):

    def test_initialization(self):
        atom = Atom("H", 0.0, 0.0, 0.0)
        self.assertEqual(atom.atom_name, "H")
        self.assertEqual(atom.position, Position(0.0, 0.0, 0.0))
        self.assertEqual(atom.properties, {})

    def test_class_properties(self):
        self.assertEqual(
            Atom._numbers,
            {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
             "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
             "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
             "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
             "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
             "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
             "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
             "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
             "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
             "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
             "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
             "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
             "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,
             "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99,
             "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105,
             "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111,
             "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117,
             "Og": 118})
        self.assertEqual(
            Atom._r_covs,
            {"H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76,
             "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41,
             "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
             "K": 2.03, "Ca": 1.76, "Sc": 1.7, "Ti": 1.6, "V": 1.53, "Cr": 1.39,
             "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
             "Ga": 1.22, "Ge": 1.2, "As": 1.19, "Se": 1.2, "Br": 1.2, "Kr": 1.16,
             "Rb": 2.2, "Sr": 1.95, "Y": 1.9, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54,
             "Tc": 1.47, "Ru": 1.46, "Rh": 1.52, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
             "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.4,
             "Cs": 2.44, "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01,
             "Pm": 1.99, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96, "Tb": 1.94, "Dy": 1.92,
             "Ho": 1.92, "Er": 1.89, "Tm": 1.9, "Yb": 1.87, "Lu": 1.87, "Hf": 1.75,
             "Ta": 1.7, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36,
             "Au": 1.36, "Hg": 1.32, "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "Po": 1.4,
             "At": 1.5, "Rn": 1.5, "Fr": 2.6, "Ra": 2.21, "Ac": 2.15, "Th": 2.06,
             "Pa": 2, "U": 1.96, "Np": 1.9, "Pu": 1.87, "Am": 1.8, "Cm": 1.69,
             "Bk": None, "Cf": None, "Es": None, "Fm": None, "Md": None, "No": None,
             "Lr": None, "Rf": None, "Db": None, "Sg": None, "Bh": None, "Hs": None,
             "Mt": None, "Ds": None, "Rg": None, "Cn": None, "Nh": None, "Fl": None,
             "Mc": None, "Lv": None, "Ts": None, "Og": None})
        self.assertEqual(
            Atom._valences,
            {"H": 1, "He": 0.5, "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 3, "O": 2,
             "F": 1, "Ne": 0.5, "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 3, "S": 2,
             "Cl": 1, "Ar": 0, "K": 1, "Ca": 2, "Sc": 3, "Ti": 4, "V": 4, "Cr": 3,
             "Mn": 4, "Fe": 3, "Co": 3, "Ni": 2, "Cu": 2, "Zn": 2, "Ga": 3, "Ge": 4,
             "As": 3, "Se": 2, "Br": 1, "Kr": 0.5, "Rb": 1, "Sr": 2, "Y": 3,
             "Zr": 4, "Nb": 5, "Mo": 4, "Tc": 4, "Ru": 4, "Rh": 4, "Pd": 4, "Ah": 1,
             "Cd": 2, "In": 3, "Sn": 4, "Sb": 3, "Te": 2, "I": 1, "Xe": 0.5,
             "Cs": 1, "Ba": 2, "La": 3, "Ce": 4, "Pr": 3, "Nd": 3, "Pm": 3, "Sm": 3,
             "Eu": 3, "Gd": 3, "Tb": 3, "Dy": 3, "Ho": 3, "Er": 3, "Tm": 3, "Yb": 3,
             "Lu": 3, "Hf": 4, "Ta": 5, "W": 4, "Re": 4, "Os": 4, "Ir": 4, "Pt": 4,
             "Au": 1, "Hg": 2, "Tl": 3, "Pb": 4, "Bi": 3, "Po": 2, "At": 1,
             "Rn": 0.5, "Fr": 1, "Ra": 2, "Ac": 3, "Th": 4, "Pa": 4, "U": 4,
             "Np": 4, "Pu": 4, "Am": 4, "Cm": 4, "Bk": 4, "Cf": 4, "Es": 4, "Fm": 4,
             "Md": 4, "No": 4, "Lr": None, "Rf": None, "Db": None, "Sg": None,
             "Bh": None, "Hs": None, "Mt": None, "Ds": None, "Rg": None, "Cn": None,
             "Nh": None, "Fl": None, "Mc": None, "Lv": None, "Ts": None, "Og": None})
        self.assertEqual(
            Atom._valence_electrons,
            {"H": 1, "He": 2, "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 5,
             "O": 6, "F": 7, "Ne": 8, "Na": 1, "Mg": 2, "Al": 3, "Si": 4,
             "P": 5, "S": 6, "Cl": 7, "Ar": 8, "K": 1, "Ca": 2, "Sc": 2,
             "Ti": 2, "V": 2, "Cr": 1, "Mn": 2, "Fe": 2, "Co": 2, "Ni": 2,
             "Cu": 1, "Zn": 2, "Ga": 3, "Ge": 4, "As": 5, "Se": 6, "Br": 7,
             "Kr": 8, "Rb": 1, "Sr": 2, "Y": 2, "Zr": 2, "Nb": 1, "Mo": 1,
             "Tc": 2, "Ru": 1, "Rh": 1, "Pd": 10, "Ag": 1, "Cd": 2,
             "In": 3, "Sn": 4, "Sb": 5, "Te": 6, "I": 7, "Xe": 8, "Cs": 1,
             "Ba": 2, "La": 2, "Ce": 2, "Pr": 2, "Nd": 2, "Pm": 2, "Sm": 2,
             "Eu": 2, "Gd": 2, "Tb": 2, "Dy": 2, "Ho": 2, "Er": 2, "Tm": 2,
             "Yb": 2, "Lu": 2, "Hf": 2, "Ta": 2, "W": 2, "Re": 2, "Os": 2,
             "Ir": 2, "Pt": 1, "Au": 1, "Hg": 2, "Tl": 3, "Pb": 4, "Bi": 5,
             "Po": 6, "At": 7, "Rn": 8, "Fr": 1, "Ra": 2, "Ac": 2, "Th": 2,
             "Pa": 2, "U": 2, "Np": 2, "Pu": 2, "Am": 2, "Cm": 2, "Bk": 2,
             "Cf": 2, "Es": 2, "Fm": 2, "Md": 2, "No": 2, "Lr": 3, "Rf": 2,
             "Db": 2, "Sg": 2, "Bh": 2, "Hs": 2, "Mt": 2, "Ds": 1, "Rg": 1,
             "Cn": 2, "Nh": 3, "Fl": 4, "Mc": 5, "Lv": 6, "Ts": 7, "Og": 8})

    def test_conversion_methods(self):
        atoms = np.empty((100, ), dtype=Atom.atom_dtype)
        atoms["x"] = np.random.default_rng().random((100,))
        atoms["y"] = np.random.default_rng().random((100,))
        atoms["z"] = np.random.default_rng().random((100,))
        atoms["name"] = np.array(["Cu"] * len(atoms))
        converted_atoms = Atom.as_Atomlist(atoms)
        self.assertTrue(all([isinstance(atom, Atom) for atom in converted_atoms]))
        self.assertFalse(all([isinstance(atom, Atom) for atom in atoms]))

        converted_atoms2 = Atom.as_array(atoms)
        positions = np.vstack((atoms["x"], atoms["y"], atoms["z"])).T
        self.assertEqual(converted_atoms2.shape, (100, 4))
        np.testing.assert_array_almost_equal(
            converted_atoms2[:, 1:], positions)

        converted_atoms3 = Atom.as_recarray(positions, atoms["name"])
        self.assertTrue(converted_atoms3.dtype == Atom.atom_dtype)
        self.assertTrue(all([i == j] for i, j in zip(atoms, converted_atoms3)))

    def test_setters(self):
        atom = Atom("H", 0.0, 0.0, 0.0)
        atom.atom_name = "He"
        self.assertEqual(atom.atom_name, "He")
        atom.name = "Li"
        self.assertEqual(atom.name, "Li")
        self.assertEqual(atom.position, Position(0.0, 0.0, 0.0))

        atom.position = Position(1.0, 1.0, 1.0)
        self.assertEqual(atom.position, Position(1.0, 1.0, 1.0))

        atom.position = [3, 2, 1]
        self.assertEqual(atom.position, Position(3, 2, 1))

    def test_invalid_initialization(self):
        with self.assertRaises(AtomTypeError):
            Atom(1, 0.0, 0.0, 0.0)
        with self.assertRaises(PositionValueError):
            Atom("H", 0.0, 0.0, "0.0")
        with self.assertRaises(AtomValueError):
            Atom("invalid", 0.0, 0.0, 0.0)

    def test_invalid_setters(self):
        atom = Atom("H", 0.0, 0.0, 0.0)
        with self.assertRaises(AtomTypeError):
            atom.atom_name = 1
        with self.assertRaises(AtomTypeError):
            atom.position = "Position"

        with self.assertRaises(AtomTypeError):
            atom.position = [1.0, "Hello", 1.0]

        with self.assertRaises(AtomTypeError):
            atom.position = [0.0, 0.0]

    def test_getitem(self):
        atom = Atom("H", 0.0, 0.0, 0.0)
        self.assertEqual(atom["atom_name"], "H")
        self.assertEqual(atom["name"], "H")
        self.assertEqual(atom["position"], Position(0.0, 0.0, 0.0))
        with self.assertRaises(AtomKeyError):
            _ = atom["nonexistent"]
        self.assertEqual(atom["x"], 0.0)
        self.assertEqual(atom["y"], 0.0)
        self.assertEqual(atom["z"], 0.0)

        self.assertEqual(atom["name", "position"], ["H", Position(0.0, 0.0, 0.0)])

    def test_get(self):
        atom = Atom("H", 1.0, 2.0, 3.0)
        self.assertEqual(atom.get("atom_name"), "H")
        self.assertEqual(atom.get("name"), "H")
        self.assertEqual(atom.get("position"), Position(1.0, 2.0, 3.0))
        self.assertEqual(atom.get("x"), 1.0)
        self.assertEqual(atom.get("y"), 2.0)
        self.assertEqual(atom.get("z"), 3.0)
        self.assertEqual(atom.get("r_cov"), 0.31)
        self.assertEqual(atom.get("valence"), 1)
        self.assertEqual(atom.get("valence_electrons"), 1)
        self.assertEqual(atom.get("number"), 1)
        with self.assertRaises(AtomKeyError):
            _ = atom.get("nonexistant")

    def test_set(self):
        atom = Atom("H", 0.0, 0.0, 0.0)
        atom.set("name", "He")
        self.assertEqual(atom["name"], "He")
        self.assertEqual(atom["number"], 2)

        atom.set("position", Position(1.0, 2.0, 3.0))
        self.assertEqual(atom["position"], Position(1.0, 2.0, 3.0))

        atom.set("position", np.array([0.0, 0.0, 0.0]))
        self.assertEqual(atom["position"], Position(0.0, 0.0, 0.0))

        with self.assertRaises(AtomTypeError):
            atom.set("position", [1, "oops", 0])

        with self.assertRaises(AtomTypeError):
            atom.set("position", {"x": 0, "y": 1, "z": 2})

        atom.set("x", -1.0)
        atom.set("y", -2.0)
        atom.set("z", -3.0)
        self.assertEqual(atom["position"], Position(-1.0, -2.0, -3.0))

        atom.set("r_cov", 0.5)
        self.assertEqual(atom["r_cov"], 0.5)

        with self.assertRaises(AtomValueError):
            atom.set("r_cov", -0.5)

        atom.set("valence", 3)
        self.assertEqual(atom["valence"], 3)

        with self.assertRaises(AtomValueError):
            atom.set("valence_electrons", 4)

        atom.set("name", "Ti")
        atom.set("valence_electrons", 6)

        with self.assertRaises(AtomKeyError):
            atom.set("number", 10)

        atom.set("size", 100)
        self.assertEqual(atom["size"], 100)

    def test_setitem(self):
        atom = Atom("H", 0.0, 0.0, 0.0)
        atom["atom_name"] = "He"
        self.assertEqual(atom.atom_name, "He")

        atom["name"] = "Li"
        self.assertEqual(atom.name, atom.atom_name)

        atom["position"] = Position(1.0, 1.0, 1.0)
        self.assertEqual(atom.position, Position(1.0, 1.0, 1.0))

        atom["x"] = -1.0
        self.assertEqual(atom.position, Position(-1.0, 1.0, 1.0))
        atom["y"] = -2.0
        self.assertEqual(atom.position, Position(-1.0, -2.0, 1.0))
        atom["z"] = -3.0
        self.assertEqual(atom.position, Position(-1.0, -2.0, -3.0))

        with self.assertRaises(AtomKeyError):
            atom["number"] = 10

    def test_properties(self):
        atom = Atom("H", 0.0, 0.0, 0.0)
        self.assertEqual(atom["number"], 1)
        self.assertEqual(atom["r_cov"], 0.31)
        self.assertEqual(atom["valence"], 1)
        self.assertEqual(atom["valence_electrons"], 1)
        atom.set("color", "red")
        self.assertEqual(atom.get("color"), "red")
        self.assertEqual(atom["color"], "red")
        atom["size"] = 10
        self.assertEqual(atom["size"], 10)
        self.assertEqual(atom.get("size"), 10)

    def test_iteration(self):
        atom = Atom("H", 0.0, 0.0, 0.0)
        atom.set("color", "red")
        atom.set("size", 10)
        properties = dict(atom)
        expected_properties = {
            "name": "H",
            "position": Position(0.0, 0.0, 0.0),
            "color": "red",
            "size": 10
        }
        self.assertEqual(properties, expected_properties)

    def test_repr(self):
        atom = Atom("H", 0.0, 0.0, 0.0)
        self.assertEqual(
            repr(atom),
            "Atom(atom_name='H', position=(0.0, 0.0, 0.0), properties={})")
        atom.set("color", "red")
        self.assertEqual(
            repr(atom),
            "Atom(atom_name='H', position=(0.0, 0.0, 0.0), properties={'color': 'red'})")


if __name__ == "__main__":
    unittest.main()
