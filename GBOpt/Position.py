from __future__ import annotations

import math
import numbers
from typing import Iterator, Sequence, Tuple, Union

import numpy as np


class PositionError(Exception):
    """Base class for errors in the Position class."""
    pass


class PositionIndexError(PositionError):
    """Exception raised when an invalid index is used to access a Position attribute."""
    pass


class PositionKeyError(PositionError):
    """Exception raised when an invalid key is used to access a Position attribute."""
    pass


class PositionTypeError(PositionError):
    """Exception raised when an invalid type is assigned to a Position attribute."""
    pass


class PositionValueError(PositionError):
    """Exception raised when an invalid value is assigned to a Position attribute."""
    pass


class Position:
    """Represents a point in 3D space."""

    __slots__ = ['__x', '__y', '__z']

    def __init__(self, x: float, y: float, z: float) -> None:
        self.__x: float = self.__validate_coordinate('x', x)
        self.__y: float = self.__validate_coordinate('y', y)
        self.__z: float = self.__validate_coordinate('z', z)

    @staticmethod
    def __validate_coordinate(coordinate: str, value: float) -> float:
        """
        Validates a coordinate value.

        :param coordinate: The name of the coordinate being validated ('x', 'y', or 'z').
        :param value: The value to validate.
        :raises PositionValueError: Exception raised if a non-number is passed in.
        :return: The validated value.
        """
        if not isinstance(value, numbers.Number) and not isinstance(value, np.number):
            raise PositionValueError("Position value must be a number.")
        return float(value)

    # Standard getters and setters. Setters validate the values.
    @property
    def x(self) -> float:
        return self.__x

    @x.setter
    def x(self, value: float) -> None:
        self.__x = self.__validate_coordinate('x', value)

    @property
    def y(self) -> float:
        return self.__y

    @y.setter
    def y(self, value: float) -> None:
        self.__y = self.__validate_coordinate('y', value)

    @property
    def z(self) -> float:
        return self.__z

    @z.setter
    def z(self, value: float) -> None:
        self.__z = self.__validate_coordinate('z', value)

    def __iter__(self) -> Iterator[float]:
        """Allows for iteration over the coordinates in the order (x, y, z)."""
        return iter((self.__x, self.__y, self.__z))

    def __getitem__(self, index: Union[int, str, slice]) -> Union[float, Tuple[float, ...]]:
        """Allows indexed access to the coordinates."""
        if isinstance(index, slice):
            return tuple(self)[index]
        elif isinstance(index, int):
            if index == 0:
                return self.__x
            elif index == 1:
                return self.__y
            elif index == 2:
                return self.__z
            else:
                raise PositionIndexError("Index out of range")
        elif isinstance(index, str):
            if index == 'x':
                return self.__x
            elif index == 'y':
                return self.__y
            elif index == 'z':
                return self.__z
            else:
                raise PositionKeyError(f"Invalid key: {index}")
        else:
            raise PositionTypeError("Invalid argument type")

    def __setitem__(self, index: Union[int, str, slice], value: float) -> None:
        if isinstance(index, slice):
            indices = range(*index.indices(3))
            if len(value) != len(indices):
                raise PositionValueError(
                    "Value length does not match slice length")
            for idx, val in zip(indices, value):
                self[idx] = val
        elif isinstance(index, int):
            if index == 0:
                self.x = value
            elif index == 1:
                self.y = value
            elif index == 2:
                self.z = value
            else:
                raise PositionIndexError("Index out of range")
        elif isinstance(index, str):
            if index == 'x':
                self.x = value
            elif index == 'y':
                self.y = value
            elif index == 'z':
                self.z = value
            else:
                raise PositionKeyError(f"Invalid key: {index}")
        else:
            raise PositionTypeError("Invalid argument type")

    def distance(self, p: Union[Position, Sequence[numbers.Number], np.ndarray]) -> float:
        """
        Calculates the distance between this Position and the specified point.

        :param p: The point to calculate the distance to
        :raises PositionTypeError: Exception raised when an invalid sequence is passed
            in.
        :return: The Euclidean distance between this Position and the specified point.
        """
        if isinstance(p, Position):
            x, y, z = p
        elif isinstance(p, (list, tuple, np.ndarray)) and len(p) == 3:
            x, y, z = p
        else:
            raise PositionTypeError(
                "Argument must be a Position instance, a sequence, or a numpy array "
                "with 3 numeric elements.")
        return math.sqrt((self.__x - x) ** 2 + (self.__y - y) ** 2 + (self.__z - z) ** 2)

    def asarray(self) -> np.ndarray:
        """ Method to convert the Position to a numpy array"""
        return np.array([self.__x, self.__y, self.__z])

    def asarray(self) -> np.ndarray:
        """ Method to convert the Position to a numpy array"""
        return np.array([self.x, self.y, self.z])

    # Dunder methods relevant to Positions
    def __add__(self, other: Union[Position, Sequence[numbers.Number], np.ndarray]) -> Position:
        if isinstance(other, Position):
            return Position(self.__x + other.x, self.__y + other.y, self.__z + other.z)
        elif isinstance(other, (list, tuple, np.ndarray)) and len(other) == 3:
            return Position(self.__x + other[0], self.__y + other[1], self.__z + other[2])
        else:
            raise PositionTypeError(
                "Operand must be a Position instance or a sequence with 3 elements.")

    def __radd__(self, other: Union[Position, Sequence[numbers.Number], np.ndarray]) -> Position:
        return self.__add__(other)

    def __sub__(self, other: Union[Position, Sequence[numbers.Number], np.ndarray]) -> Position:
        if isinstance(other, Position):
            return Position(self.__x - other.x, self.__y - other.y, self.__z - other.z)
        elif isinstance(other, (Sequence, np.ndarray)) and len(other) == 3:
            return Position(self.__x - other[0], self.__y - other[1], self.__z - other[2])
        else:
            raise PositionTypeError(
                "Operand must be a Position instance or a sequence with 3 elements.")

    def __rsub__(self, other: Union[Position, Sequence[numbers.Number], np.ndarray]) -> Position:
        if isinstance(other, (Sequence, np.ndarray)) and len(other) == 3:
            return Position(other[0] - self.__x, other[1] - self.__y, other[2] - self.__z)
        else:
            raise PositionTypeError(
                "Operand must be a sequence with 3 elements.")

    def __mul__(self, other: numbers.Number) -> Position:
        if isinstance(other, numbers.Number):
            return Position(self.__x * other, self.__y * other, self.__z * other)
        else:
            raise PositionTypeError("Operand must be a numeric type")

    def __rmul__(self, other: numbers.Number) -> Position:
        return self.__mul__(other)

    def __truediv__(self, other: numbers.Number) -> Position:
        if isinstance(other, numbers.Number):
            if other == 0:
                raise PositionValueError("Cannot divide by 0.")
            return Position(self.__x / other, self.__y / other, self.__z / other)
        else:
            raise PositionTypeError("Operand must be a numeric type")

    def __iadd__(self, other: Union[Position, Sequence[numbers.Number], np.ndarray]) -> Position:
        if isinstance(other, Position):
            self.__x += other.x
            self.__y += other.y
            self.__z += other.z
        elif isinstance(other, (Sequence, np.ndarray)) and len(other) == 3:
            self.__x += other[0]
            self.__y += other[1]
            self.__z += other[2]
        else:
            raise PositionTypeError(
                "Operand must be a Position instance or a sequence with 3 elements.")

        return self

    def __isub__(self, other: Union[Position, Sequence[numbers.Number], np.ndarray]) -> Position:
        if isinstance(other, Position):
            self.__x -= other.x
            self.__y -= other.y
            self.__z -= other.z
        elif isinstance(other, (Sequence, np.ndarray)) and len(other) == 3:
            self.__x -= other[0]
            self.__y -= other[1]
            self.__z -= other[2]
        else:
            raise PositionTypeError(
                "Operand must be a Position instance or a sequence with 3 elements.")

        return self

    def __imul__(self, other: numbers.Number) -> Position:
        if isinstance(other, numbers.Number):
            self.__x *= other
            self.__y *= other
            self.__z *= other
            return self
        else:
            raise PositionTypeError("Operand must be a numeric type")

    def __itruediv__(self, other: numbers.Number) -> Position:
        if isinstance(other, numbers.Number):
            if other == 0:
                raise PositionValueError("Cannot divide by 0.")
            self.__x /= other
            self.__y /= other
            self.__z /= other
            return self
        else:
            raise PositionTypeError("Operand must be a numeric type")

    def __eq__(self, other: Position) -> bool:
        return self.__x == other.x and self.__y == other.y and self.__z == other.z

    def __repr__(self) -> str:
        return f"Position(x={self.__x}, y={self.__y}, z={self.__z})"
