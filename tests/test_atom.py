import unittest

from GBOpt.Atom import Atom, AtomKeyError, AtomTypeError, AtomValueError
from GBOpt.Position import Position, PositionValueError


class TestAtom(unittest.TestCase):

    def test_initialization(self):
        atom = Atom(1, 1, 0.0, 0.0, 0.0)
        self.assertEqual(atom.id, 1)
        self.assertEqual(atom.atom_type, 1)
        self.assertEqual(atom.position, Position(0.0, 0.0, 0.0))
        self.assertEqual(atom._properties, {})

    def test_setters(self):
        atom = Atom(1, 1, 0.0, 0.0, 0.0)
        atom.id = 2
        atom.atom_type = 2
        atom.position = Position(1.0, 1.0, 1.0)
        self.assertEqual(atom.id, 2)
        self.assertEqual(atom.atom_type, 2)
        self.assertEqual(atom.position, Position(1.0, 1.0, 1.0))

    def test_invalid_initialization(self):
        with self.assertRaises(AtomTypeError):
            Atom('1', 1, 0.0, 0.0, 0.0)
        with self.assertRaises(PositionValueError):
            Atom(1, 1, 0.0, 0.0, '0.0')
        with self.assertRaises(AtomValueError):
            Atom(0, 1, 0.0, 0.0, 0.0)
        with self.assertRaises(AtomValueError):
            Atom(1, 0, 0.0, 0.0, 0.0)

    def test_invalid_setters(self):
        atom = Atom(1, 1, 0.0, 0.0, 0.0)
        with self.assertRaises(AtomTypeError):
            atom.id = '2'
        with self.assertRaises(AtomValueError):
            atom.id = -1
        with self.assertRaises(AtomTypeError):
            atom.atom_type = '2'
        with self.assertRaises(AtomValueError):
            atom.atom_type = -1
        with self.assertRaises(AtomTypeError):
            atom.position = 'Position'

    def test_getitem(self):
        atom = Atom(1, 1, 0.0, 0.0, 0.0)
        self.assertEqual(atom['id'], 1)
        self.assertEqual(atom['atom_type'], 1)
        self.assertEqual(atom['position'], Position(0.0, 0.0, 0.0))
        with self.assertRaises(AtomKeyError):
            _ = atom['nonexistent']

    def test_setitem(self):
        atom = Atom(1, 1, 0.0, 0.0, 0.0)
        atom['id'] = 2
        self.assertEqual(atom.id, 2)
        atom['atom_type'] = 2
        self.assertEqual(atom.atom_type, 2)
        atom['position'] = Position(1.0, 1.0, 1.0)
        self.assertEqual(atom.position, Position(1.0, 1.0, 1.0))
        with self.assertRaises(AtomTypeError):
            atom['position'] = 'Position'

    def test_properties(self):
        atom = Atom(1, 1, 0.0, 0.0, 0.0)
        atom.set('color', 'red')
        self.assertEqual(atom.get('color'), 'red')
        self.assertEqual(atom['color'], 'red')
        atom['size'] = 10
        self.assertEqual(atom['size'], 10)
        self.assertEqual(atom.get('size'), 10)
        with self.assertRaises(AtomKeyError):
            _ = atom['nonexistent']

    def test_iteration(self):
        atom = Atom(1, 1, 0.0, 0.0, 0.0)
        atom.set('color', 'red')
        atom.set('size', 10)
        properties = dict(atom)
        expected_properties = {
            'id': 1,
            'type': 1,
            'position': Position(0.0, 0.0, 0.0),
            'color': 'red',
            'size': 10
        }
        self.assertEqual(properties, expected_properties)

    def test_repr(self):
        atom = Atom(1, 1, 0.0, 0.0, 0.0)
        self.assertEqual(
            repr(atom), 'Atom(id=1, atom_type=1, position=(0.0, 0.0, 0.0), properties={})')
        atom.set('color', 'red')
        self.assertEqual(repr(
            atom), "Atom(id=1, atom_type=1, position=(0.0, 0.0, 0.0), properties={'color': 'red'})")


if __name__ == '__main__':
    unittest.main()
