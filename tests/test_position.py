import math
import unittest

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

    def test_subtraction(self):
        pos1 = Position(4.0, 5.0, 6.0)
        pos2 = Position(1.0, 2.0, 3.0)
        pos3 = pos1 - pos2
        self.assertEqual(pos3, Position(3.0, 3.0, 3.0))

    def test_multiplication(self):
        pos = Position(1.0, 2.0, 3.0)
        pos2 = pos * 2
        self.assertEqual(pos2, Position(2.0, 4.0, 6.0))

    def test_division(self):
        pos = Position(2.0, 4.0, 6.0)
        pos2 = pos / 2
        self.assertEqual(pos2, Position(1.0, 2.0, 3.0))

    def test_distance(self):
        pos1 = Position(0.0, 0.0, 0.0)
        pos2 = Position(1.0, 1.0, 1.0)
        self.assertAlmostEqual(pos1.distance(pos2), math.sqrt(3))

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
            pos - "string"
        with self.assertRaises(PositionTypeError):
            pos * "string"
        with self.assertRaises(PositionTypeError):
            pos / "string"
        with self.assertRaises(PositionValueError):
            pos / 0


if __name__ == '__main__':
    unittest.main()
