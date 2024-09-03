from numbers import Number
from typing import Any, Dict, Iterator

from numpy import generic as np_generic

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
    def _validate_value(attribute: str, value: Any, expected_type: type, positive: bool = False) -> Any:
        """
        Validates an attribute value.

        :param attribute: The name of the attribute being validated.
        :param value: The value to validate.
        :param expected_type: The expected type for the value.
        :param positive: Whether or not the value must be positive (> 0).
        :return: The validated value.
        """
        if not isinstance(value, expected_type) and not isinstance(value, np_generic):
            raise AtomTypeError(f"The {attribute} must be of type "
                                f"{expected_type.__name__} or a compatible NumPy type.")
        if issubclass(expected_type, Number) and not isinstance(value, Number):
            raise AtomTypeError(f"The {attribute} must be a number.")
        if positive and value <= 0:
            raise AtomValueError(f"The {attribute} must be a positive value.")
        return value

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
    def position(self, value: Position) -> None:
        self._position = self._validate_value('position', value, Position)

    def get(self, key: str) -> Any:
        """Gets a property by key."""
        if key == 'id':
            return self._id
        elif key in ['type', 'atom_type']:
            return self._atom_type
        elif key == 'position':
            return self._position
        else:
            try:
                return self._properties[key]
            except KeyError:
                raise AtomKeyError(f"No property found for key: {key}")

    def set(self, key: str, value) -> None:
        """Sets a property by key."""
        if key == 'id':
            self.id = value
        elif key in ['type', 'atom_type']:
            self.atom_type = value
        elif key == 'position':
            if isinstance(value, Position):
                self.position = value
            else:
                raise AtomTypeError(
                    "Value for 'position' must be a Position instance.")
        else:
            self._properties[key] = value

    def __getitem__(self, key: str) -> Any:
        """Allows dictionary-like access to properties."""
        if key in ['id', 'atom_type', 'position']:
            return getattr(self, key)
        elif key == 'type':
            return self.atom_type
        elif key in self._properties:
            return self.get(key)
        else:
            raise AtomKeyError(f"No property found for key: {key}.")

    def __setitem__(self, key: str, value: Any) -> None:
        """Allows dictionary-like setting of properties."""
        if key in ['id', 'atom_type', 'position']:
            setattr(self, key, value)
        elif key == 'type':
            setattr(self, 'atom_type', value)
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
