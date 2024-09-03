from collections.abc import Sequence
from numbers import Number
from typing import Any, Dict, Iterator, Tuple, Union

import numpy as np

from GBOpt.Position import Position


class AtomError(Exception):
    """Base class for errors in the Atom class."""
    pass


class AtomValueError(AtomError):
    """Exception raised when an invalid value is assigned to an Atom attribute."""
    pass


class AtomKeyError(AtomError):
    """Exception raised when an invalid Atom attribute is requested."""
    pass


class AtomTypeError(AtomError):
    """Exception raised when an invalid type is assigned to an Atom attribute."""
    pass


class Atom:
    """Represents an atom with an ID, type, position, and other properties."""

    def __init__(self, id: int, atom_type: int, x: float, y: float, z: float):
        self._id: int = self._validate_value('id', id, int, positive=True)
        self._atom_type: int = self._validate_value(
            'atom_type', atom_type, int, positive=True)
        self._position: Position = Position(x, y, z)
        self._properties: Dict[str, Any] = {}

    @staticmethod
    def _validate_value(attribute: str, value: Any, expected_types: Union[type, Tuple[type, ...]], positive: bool = False) -> Any:
        """
        Validates an attribute value.

        :param attribute: The name of the attribute being validated.
        :param value: The value to validate.
        :param expected_type: The expected type(s) for the value.
        :param positive: Whether or not the value must be positive (> 0).
        :raises AtomTypeError: Exception raised when an incorrect type is given for a
            parameter.
        "raises AtomValueError: Exception raised when an invalid value is given for a
            parameter.
        :return: The validated value.
        """
        if not isinstance(expected_types, tuple):
            expected_types = (expected_types,)

        if not any(isinstance(value, t) for t in expected_types) and not isinstance(value, np.generic):
            expected_type_names = ', '.join(t.__name__ for t in expected_types)
            raise AtomTypeError(f"The {attribute} must be of type "
                                f"{expected_type_names} or a compatible NumPy type.")

        if positive and isinstance(value, Number) and value <= 0:
            raise AtomValueError(f"The {attribute} must be a positive value.")
        return value

    # Standard getters and setters. In every case except position a deepcopy of the
    # value is passed in to prevent inadvertent changes to parameters. If a deepcopy of
    # position needs to be passed, a helper method needs to be written where the
    # assignment actually takes place.
    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        self._id = self._validate_value('id', value, int, positive=True)

    @property
    def atom_type(self) -> int:
        return self._atom_type

    @atom_type.setter
    def atom_type(self, value: int) -> None:
        self._atom_type = self._validate_value(
            'atom_type', value, int, positive=True)

    @property
    def position(self) -> Position:
        return self._position

    @position.setter
    def position(self, value: Union[Position, Sequence, np.ndarray]) -> None:
        if isinstance(value, Sequence) and not isinstance(value, str):
            if not all(isinstance(item, Number) for item in value):
                raise AtomTypeError(
                    "All elements of 'position' must be numeric.")

            if len(value) != 3:
                raise AtomTypeError(
                    "'position' must be a sequence of length 3.")
            value = Position(*value)
        elif not isinstance(value, (Position, np.ndarray)):
            raise AtomTypeError(
                "Value for 'position' must be a Position instance, a numeric sequence of length 3, or a NumPy array of length 3.")
        self._position = value

    def get(self, key: str) -> Any:
        """Gets a property by key."""
        if key == 'id':
            return self._id
        elif key in ['type', 'atom_type']:
            return self._atom_type
        elif key == 'position':
            return self._position
        elif key in ['x', 'y', 'z']:
            return self._position[key]
        else:
            try:
                return self._properties[key]
            except KeyError:
                raise AtomKeyError(f"No property found for key: {key}")

    def set(self, key: str, value: Any) -> None:
        """Sets a property by key."""
        if key == 'id':
            self.id = value
        elif key in ['type', 'atom_type']:
            self.atom_type = value
        elif key in ['x', 'y', 'z']:
            self.position[key] = value
        elif key == 'position':
            if isinstance(value, Position):
                self.position = value
            elif isinstance(value, (list, tuple, np.ndarray)) and len(value) == 3:
                if not all(isinstance(item, Number) for item in value):
                    raise AtomTypeError(
                        "All elements of 'position' must be numeric.")
                self.position = Position(*value)
            else:
                raise AtomTypeError(
                    "Value for 'position' must be a Position instance, a numeric sequence of length 3, or a NumPy array of length 3.")
        else:
            self._properties[key] = value

    def __getitem__(self, keys: Union[str, Tuple[str, ...]]) -> Any:
        """Allows dictionary-like access to properties."""
        if not isinstance(keys, tuple):
            keys = (keys,)

        attrs = [None] * len(keys)
        for idx, k in enumerate(keys):
            if k in ['id', 'atom_type', 'position']:
                attrs[idx] = getattr(self, k)
            elif k == 'type':
                attrs[idx] = self.atom_type
            elif k in ['x', 'y', 'z']:
                attrs[idx] = getattr(self.position, k)
            elif k in self._properties:
                attrs[idx] = self.get(k)
            else:
                raise AtomKeyError(f"No property found for key: {k}.")
        if len(attrs) == 1:
            return attrs[0]
        else:
            return attrs

    def __setitem__(self, key: str, value: Any) -> None:
        """Allows dictionary-like setting of properties."""
        if key in ['id', 'atom_type', 'position']:
            setattr(self, key, value)
        elif key == 'type':
            setattr(self, 'atom_type', value)
        elif key in ['x', 'y', 'z']:
            setattr(self._position, key, value)
        else:
            self.set(key, value)

    def __iter__(self) -> Iterator:
        """Allows iteration over atom properties."""
        yield 'id', self._id
        yield 'type', self._atom_type
        yield 'position', self._position
        yield from self._properties.items()

    def __repr__(self) -> str:
        """Returns a string representation of the Atom object."""
        return (f"Atom(id={self.id}, atom_type={self.atom_type}, "
                f"position=({self.position.x}, {self.position.y}, "
                f"{self.position.z}), "
                f"properties={self._properties})"
                )
