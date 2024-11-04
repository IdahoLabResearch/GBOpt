import math
import unittest

import numpy as np

from GBOpt.Position import (
    Position,
    PositionIndexError,
    PositionKeyError,
    PositionTypeError,
    PositionValueError,
)


class TestPosition(unittest.TestCase):

    def test_initialization(self):
        pos = Position(1.0, 2.0, 3.0)
        self.assertEqual(pos.x, 1.0)
        self.assertEqual(pos.y, 2.0)
        self.assertEqual(pos.z, 3.0)

    def test_setters(self):
        pos = Position(0.0, 0.0, 0.0)
        pos.x = 1.0
        pos.y = 2.0
        pos.z = 3.0
        self.assertEqual(pos.x, 1.0)
        self.assertEqual(pos.y, 2.0)
        self.assertEqual(pos.z, 3.0)

    def test_getitem(self):
        pos = Position(1.0, 2.0, 3.0)
        self.assertEqual(pos[0], 1.0)
        self.assertEqual(pos[1], 2.0)
        self.assertEqual(pos[2], 3.0)
        self.assertEqual(pos['x'], 1.0)
        self.assertEqual(pos['y'], 2.0)
        self.assertEqual(pos['z'], 3.0)
        with self.assertRaises(PositionIndexError):
            pos[3]
        with self.assertRaises(PositionKeyError):
            pos['a']
        with self.assertRaises(PositionTypeError):
            pos[1.2]

    def test_setitem(self):
        pos = Position(0.0, 0.0, 0.0)
        pos[0] = 1.0
        pos[1] = 2.0
        pos[2] = 3.0
        self.assertEqual(pos.x, 1.0)
        self.assertEqual(pos.y, 2.0)
        self.assertEqual(pos.z, 3.0)
        pos['x'] = 4.0
        pos['y'] = 5.0
        pos['z'] = 6.0
        self.assertEqual(pos.x, 4.0)
        self.assertEqual(pos.y, 5.0)
        self.assertEqual(pos.z, 6.0)
        with self.assertRaises(PositionIndexError):
            pos[3] = 1
        with self.assertRaises(PositionKeyError):
            pos['a'] = 1
        with self.assertRaises(PositionTypeError):
            pos[1.2] = 1

    def test_getitem_with_slice(self):
        pos = Position(1.0, 2.0, 3.0)
        self.assertEqual(pos[:], (1.0, 2.0, 3.0))
        self.assertEqual(pos[1:], (2.0, 3.0))
        self.assertEqual(pos[:2], (1.0, 2.0))
        self.assertEqual(pos[::2], (1.0, 3.0))

    def test_setitem_with_slice(self):
        pos = Position(1.0, 2.0, 3.0)
        pos[:2] = [4.0, 5.0]
        self.assertEqual(pos.x, 4.0)
        self.assertEqual(pos.y, 5.0)
        self.assertEqual(pos.z, 3.0)
        pos[::2] = [6.0, 7.0]
        self.assertEqual(pos.x, 6.0)
        self.assertEqual(pos.y, 5.0)
        self.assertEqual(pos.z, 7.0)
        with self.assertRaises(PositionValueError):
            pos[::2] = [8.0]  # Invalid length

    def test_addition(self):
        pos1 = Position(1.0, 2.0, 3.0)
        pos2 = Position(4.0, 5.0, 6.0)
        pos3 = pos1 + pos2
        self.assertEqual(pos3, Position(5.0, 7.0, 9.0))
        pos4 = [4, 5, 6]
        pos5 = pos1 + pos4
        self.assertEqual(pos5, Position(5.0, 7.0, 9.0))
        pos6 = pos4 + pos1
        self.assertEqual(pos6, Position(5.0, 7.0, 9.0))
        pos1 += pos2
        self.assertEqual(pos1, Position(5.0, 7.0, 9.0))
        pos1 += pos4
        self.assertEqual(pos1, Position(9.0, 12.0, 15.0))

    def test_subtraction(self):
        pos1 = Position(4.0, 5.0, 6.0)
        pos2 = Position(1.0, 2.0, 3.0)
        pos3 = pos1 - pos2
        self.assertEqual(pos3, Position(3.0, 3.0, 3.0))
        pos4 = [1, 2, 3]
        pos5 = pos1 - pos4
        self.assertEqual(pos5, Position(3.0, 3.0, 3.0))
        pos6 = pos4 - pos1
        self.assertEqual(pos6, Position(-3.0, -3.0, -3.0))
        pos1 -= pos2
        self.assertEqual(pos1, Position(3.0, 3.0, 3.0))
        pos1 -= pos4
        self.assertEqual(pos1, Position(2.0, 1.0, 0.0))

    def test_multiplication(self):
        pos = Position(1.0, 2.0, 3.0)
        pos2 = pos * 2
        self.assertEqual(pos2, Position(2.0, 4.0, 6.0))
        pos3 = 2 * pos
        self.assertEqual(pos3, Position(2.0, 4.0, 6.0))
        with self.assertRaises(PositionTypeError):
            pos * pos2
        pos *= 2
        self.assertEqual(pos, Position(2.0, 4.0, 6.0))

    def test_division(self):
        pos = Position(2.0, 4.0, 6.0)
        pos2 = pos / 2
        self.assertEqual(pos2, Position(1.0, 2.0, 3.0))
        with self.assertRaises(PositionValueError):
            pos / 0
        pos /= 2
        self.assertEqual(pos, Position(1.0, 2.0, 3.0))

    def test_distance(self):
        pos1 = Position(0.0, 0.0, 0.0)
        pos2 = Position(1.0, 1.0, 1.0)
        pos3 = [1, 1, 1]
        pos4 = 1
        self.assertAlmostEqual(pos1.distance(pos2), math.sqrt(3))
        self.assertAlmostEqual(pos1.distance(pos3), math.sqrt(3))
        with self.assertRaises(PositionTypeError):
            dist = pos1.distance(pos4)
            del dist

    def test_invalid_initialization(self):
        with self.assertRaises(PositionValueError):
            Position(1.0, 'a', 3.0)

    def test_invalid_setitem(self):
        pos = Position(1.0, 2.0, 3.0)
        with self.assertRaises(PositionKeyError):
            pos['a'] = 1.0
        with self.assertRaises(PositionIndexError):
            pos[3] = 1.0

    def test_invalid_operations(self):
        pos = Position(1.0, 2.0, 3.0)
        with self.assertRaises(PositionTypeError):
            pos + "string"
        with self.assertRaises(PositionTypeError):
            pos + [1]
        with self.assertRaises(PositionTypeError):
            (1) + pos
        with self.assertRaises(PositionTypeError):
            pos += 1
        with self.assertRaises(PositionTypeError):
            pos - "string"
        with self.assertRaises(PositionTypeError):
            (1) - pos
        with self.assertRaises(PositionTypeError):
            pos - [1]
        with self.assertRaises(PositionTypeError):
            pos -= np.array([1])
        with self.assertRaises(PositionTypeError):
            pos * "string"
        with self.assertRaises(PositionTypeError):
            pos *= pos
        with self.assertRaises(PositionTypeError):
            pos *= "string"
        with self.assertRaises(PositionTypeError):
            pos / "string"
        with self.assertRaises(PositionValueError):
            pos / 0
        with self.assertRaises(PositionValueError):
            pos /= 0
        with self.assertRaises(PositionTypeError):
            pos /= "string"

    def test_boolean_operations(self):
        pos = Position(1.0, 2.0, 3.0)
        pos2 = Position(1.0, 2.0, 3.0)
        self.assertTrue(pos == pos2)

    def test_asarray(self):
        pos = Position(1.0, 0.0, 0.0)
        self.assertTrue(isinstance(pos.asarray(), np.ndarray))

    def test_repr(self):
        pos = Position(1.0, 1.0, 1.0)
        self.assertEqual(repr(pos), "Position(x=1.0, y=1.0, z=1.0)")
        pos2 = Position(4.0, 5.0, 6.0)
        self.assertEqual(eval(repr(pos2)), pos2)


if __name__ == '__main__':
    unittest.main()
