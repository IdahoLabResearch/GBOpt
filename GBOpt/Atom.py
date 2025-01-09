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
    """Exception raised when an invalid Atom key is requested."""
    pass


class AtomTypeError(AtomError):
    """Exception raised when an invalid type is assigned to an Atom attribute."""
    pass


class Atom:
    """
    Represents an atom with an ID, type, position, and other properties.
    Source for covalent radius is Cordero **et al.**. Source for valences is the USPEX
    appendix: "9.9 Table of default chemical valences used in USPEX". The number of
    valence electrons can be generally determined for the main group elements (not
    transition metals or the lanthanides/actinides) with the ones digit of the group
    number. For the transition metals and lanthanides/actinides, values between 2 and 10
     are allowed. In most cases, 2 is used. Valence electrons for these elements are
     taken from WolframAlpha, but the general rule for the number of valence electrons
     can be found at https://www.wikihow.com/Find-Valence-Electrons.


    :param atom_name: The name of the Atom (from the periodic table)
    :param x, y, z: The coordinates of the Atom.
    :param  **kwargs: See below.

    :Keyword Arguments
        Any valid string and any valid alphanumeric value can be assigned with keyword
        arguments. For example, you could assign an atom to a specific numbered unit
        cell:
        ```
        Atom('H', 0, 0, 0, unit_cell=1)
        ```
        Or give an atom a 'greeting':
        ```
        Atom('H', 0, 0, 0, greeting='hello')
        ```
    """

    # Numpy dtype for use in structured arrays
    atom_dtype = np.dtype([
        ('name', 'U2'),
        ('x', float),
        ('y', float),
        ('z', float),
    ])

    _numbers = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
                'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
                'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
                'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
                'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
                'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
                'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
                'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
                'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
                'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
                'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
                'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
                'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
                'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99,
                'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
                'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111,
                'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117,
                'Og': 118}

    _r_covs = {'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76,
               'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41,
               'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
               'K': 2.03, 'Ca': 1.76, 'Sc': 1.7, 'Ti': 1.6, 'V': 1.53, 'Cr': 1.39,
               'Mn': 1.39, 'Fe': 1.32, 'Co': 1.26, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22,
               'Ga': 1.22, 'Ge': 1.2, 'As': 1.19, 'Se': 1.2, 'Br': 1.2, 'Kr': 1.16,
               'Rb': 2.2, 'Sr': 1.95, 'Y': 1.9, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54,
               'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.52, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44,
               'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.4,
               'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01,
               'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96, 'Tb': 1.94, 'Dy': 1.92,
               'Ho': 1.92, 'Er': 1.89, 'Tm': 1.9, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75,
               'Ta': 1.7, 'W': 1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36,
               'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.4,
               'At': 1.5, 'Rn': 1.5, 'Fr': 2.6, 'Ra': 2.21, 'Ac': 2.15, 'Th': 2.06,
               'Pa': 2, 'U': 1.96, 'Np': 1.9, 'Pu': 1.87, 'Am': 1.8, 'Cm': 1.69,
               'Bk': None, 'Cf': None, 'Es': None, 'Fm': None, 'Md': None, 'No': None,
               'Lr': None, 'Rf': None, 'Db': None, 'Sg': None, 'Bh': None, 'Hs': None,
               'Mt': None, 'Ds': None, 'Rg': None, 'Cn': None, 'Nh': None, 'Fl': None,
               'Mc': None, 'Lv': None, 'Ts': None, 'Og': None}

    _valences = {'H': 1, 'He': 0.5, 'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 3, 'O': 2,
                 'F': 1, 'Ne': 0.5, 'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 3, 'S': 2,
                 'Cl': 1, 'Ar': 0, 'K': 1, 'Ca': 2, 'Sc': 3, 'Ti': 4, 'V': 4, 'Cr': 3,
                 'Mn': 4, 'Fe': 3, 'Co': 3, 'Ni': 2, 'Cu': 2, 'Zn': 2, 'Ga': 3, 'Ge': 4,
                 'As': 3, 'Se': 2, 'Br': 1, 'Kr': 0.5, 'Rb': 1, 'Sr': 2, 'Y': 3,
                 'Zr': 4, 'Nb': 5, 'Mo': 4, 'Tc': 4, 'Ru': 4, 'Rh': 4, 'Pd': 4, 'Ah': 1,
                 'Cd': 2, 'In': 3, 'Sn': 4, 'Sb': 3, 'Te': 2, 'I': 1, 'Xe': 0.5,
                 'Cs': 1, 'Ba': 2, 'La': 3, 'Ce': 4, 'Pr': 3, 'Nd': 3, 'Pm': 3, 'Sm': 3,
                 'Eu': 3, 'Gd': 3, 'Tb': 3, 'Dy': 3, 'Ho': 3, 'Er': 3, 'Tm': 3, 'Yb': 3,
                 'Lu': 3, 'Hf': 4, 'Ta': 5, 'W': 4, 'Re': 4, 'Os': 4, 'Ir': 4, 'Pt': 4,
                 'Au': 1, 'Hg': 2, 'Tl': 3, 'Pb': 4, 'Bi': 3, 'Po': 2, 'At': 1,
                 'Rn': 0.5, 'Fr': 1, 'Ra': 2, 'Ac': 3, 'Th': 4, 'Pa': 4, 'U': 4,
                 'Np': 4, 'Pu': 4, 'Am': 4, 'Cm': 4, 'Bk': 4, 'Cf': 4, 'Es': 4, 'Fm': 4,
                 'Md': 4, 'No': 4, 'Lr': None, 'Rf': None, 'Db': None, 'Sg': None,
                 'Bh': None, 'Hs': None, 'Mt': None, 'Ds': None, 'Rg': None, 'Cn': None,
                 'Nh': None, 'Fl': None, 'Mc': None, 'Lv': None, 'Ts': None, 'Og': None}

    _valence_electrons = {'H': 1, 'He': 2, 'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5,
                          'O': 6, 'F': 7, 'Ne': 8, 'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4,
                          'P': 5, 'S': 6, 'Cl': 7, 'Ar': 8, 'K': 1, 'Ca': 2, 'Sc': 2,
                          'Ti': 2, 'V': 2, 'Cr': 1, 'Mn': 2, 'Fe': 2, 'Co': 2, 'Ni': 2,
                          'Cu': 1, 'Zn': 2, 'Ga': 3, 'Ge': 4, 'As': 5, 'Se': 6, 'Br': 7,
                          'Kr': 8, 'Rb': 1, 'Sr': 2, 'Y': 2, 'Zr': 2, 'Nb': 1, 'Mo': 1,
                          'Tc': 2, 'Ru': 1, 'Rh': 1, 'Pd': 10, 'Ag': 1, 'Cd': 2,
                          'In': 3, 'Sn': 4, 'Sb': 5, 'Te': 6, 'I': 7, 'Xe': 8, 'Cs': 1,
                          'Ba': 2, 'La': 2, 'Ce': 2, 'Pr': 2, 'Nd': 2, 'Pm': 2, 'Sm': 2,
                          'Eu': 2, 'Gd': 2, 'Tb': 2, 'Dy': 2, 'Ho': 2, 'Er': 2, 'Tm': 2,
                          'Yb': 2, 'Lu': 2, 'Hf': 2, 'Ta': 2, 'W': 2, 'Re': 2, 'Os': 2,
                          'Ir': 2, 'Pt': 1, 'Au': 1, 'Hg': 2, 'Tl': 3, 'Pb': 4, 'Bi': 5,
                          'Po': 6, 'At': 7, 'Rn': 8, 'Fr': 1, 'Ra': 2, 'Ac': 2, 'Th': 2,
                          'Pa': 2, 'U': 2, 'Np': 2, 'Pu': 2, 'Am': 2, 'Cm': 2, 'Bk': 2,
                          'Cf': 2, 'Es': 2, 'Fm': 2, 'Md': 2, 'No': 2, 'Lr': 3, 'Rf': 2,
                          'Db': 2, 'Sg': 2, 'Bh': 2, 'Hs': 2, 'Mt': 2, 'Ds': 1, 'Rg': 1,
                          'Cn': 2, 'Nh': 3, 'Fl': 4, 'Mc': 5, 'Lv': 6, 'Ts': 7, 'Og': 8}

    __slots__ = ['__atom_name', '__position', '__properties',
                 '__rcov_set_by_user', '__valence_set_by_user',
                 '__valence_electrons_set_by_user']

    def __init__(self, atom_name: str, x: float, y: float, z: float, **kwargs):
        self.__atom_name = self.__validate_value('atom_name', atom_name, str)
        self.__position: Position = Position(x, y, z)
        self.__properties: Dict[str, Any] = kwargs
        self.__rcov_set_by_user = False
        self.__valence_set_by_user = False
        self.__valence_electrons_set_by_user = False

    @staticmethod
    def asAtom(atoms: np.ndarray) -> list:
        converted = [None] * len(atoms)
        for i, atom in enumerate(atoms):
            converted[i] = Atom(*atom)

        return converted

    @staticmethod
    def asarray(atoms: np.ndarray) -> np.ndarray:
        converted = np.empty((len(atoms), 4))
        names = atoms['name']
        names_dict = {name: idx + 1 for idx, name in enumerate(set(names))}
        converted_names = np.array([names_dict[name] for name in names])
        positions = np.vstack((atoms['x'], atoms['y'], atoms['z'])).T
        converted = np.hstack((converted_names[:, np.newaxis], positions))

        return converted

    @staticmethod
    def __validate_value(attribute: str, value: Any, expected_types: Union[type, Tuple[type, ...]], positive: bool = False) -> Any:
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

        if attribute == "atom_name" and value not in [
                'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
                'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn',
                'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
                'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
                'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
                'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
                'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At',
                'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
                'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs',
                'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']:
            raise AtomValueError(f"Invalid atom type name: {value}.")
        return value

    # Standard getters and setters.
    @property
    def atom_name(self) -> str:
        return self.__atom_name

    @atom_name.setter
    def atom_name(self, value: str) -> None:
        self.__atom_name = self.__validate_value('atom_name', value, str)
        self.__rcov_set_by_user = False
        self.__valence_set_by_user = False
        self.__valence_electrons_set_by_user = False

    @property
    def name(self) -> str:
        return self.__atom_name

    @name.setter
    def name(self, value: str) -> None:
        self.__atom_name = self.__validate_value('atom_name', value, str)
        self.__rcov_set_by_user = False
        self.__valence_set_by_user = False
        self.__valence_electrons_set_by_user = False

    @property
    def position(self) -> Position:
        return self.__position

    @property
    def properties(self) -> Dict[str, Any]:
        return self.__properties

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
        self.__position = value

    @property
    def properties(self) -> Dict[str, Any]:
        return self.__properties

    def get(self, key: str) -> Any:
        """Gets a property by key."""
        if key in ['atom_name', 'name']:
            return self.__atom_name
        elif key == 'position':
            return self.__position
        elif key in ['x', 'y', 'z']:
            return self.__position[key]
        elif key == 'r_cov' and not self.__rcov_set_by_user:
            return self._r_covs[self.__atom_name]
        elif key == 'valence' and not self.__valence_set_by_user:
            return self._valences[self.__atom_name]
        elif key == 'valence_electrons' and not self.__valence_electrons_set_by_user:
            return self._valence_electrons[self.__atom_name]
        elif key == 'number':
            return self._numbers[self.__atom_name]
        else:
            try:
                return self.__properties[key]
            except KeyError:
                raise AtomKeyError(f"No property found for key: {key}")

    def set(self, key: str, value: Any) -> None:
        """Sets a property by key."""
        if key in ['number']:
            raise AtomKeyError(
                f'{key} is set by other attributes and cannot be modified directly.')
        if key in ['atom_name', 'name']:
            self.atom_name = value
            self.__rcov_set_by_user = False
            self.__valence_set_by_user = False
            self.__valence_electrons_set_by_user = False
        elif key == 'r_cov':
            self.properties['r_cov'] = self.__validate_value(
                'r_cov', value, Number, positive=True)
            self.__rcov_set_by_user = True
        elif key == 'valence':
            self.properties['valence'] = self.__validate_value(
                'valence', value, Number, positive=True)
            self.__valence_set_by_user = True
        elif key == 'valence_electrons':
            if self.__atom_name in ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                                    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                                    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'In',
                                    'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'Tl', 'Pb',
                                    'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Nh', 'Fl', 'Mc',
                                    'Lv', 'Ts', 'Og']:
                raise AtomValueError(
                    "Only transition metals and the lanthanides/actinides can have "
                    "variable valence electrons.")
            self.__properties['valence_electrons'] = self.__validate_value(
                'valence_electrons', value, int, positive=True)
            self.__valence_electrons_set_by_user = True
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
            self.__properties[key] = value

    def __getitem__(self, keys: Union[str, Tuple[str, ...]]) -> Any:
        """Allows dictionary-like access to properties."""
        if not isinstance(keys, tuple):
            keys = (keys,)

        attrs = [None] * len(keys)
        for idx, k in enumerate(keys):
            if k in ['atom_name', 'position']:
                attrs[idx] = getattr(self, k)
            elif k == 'name':
                attrs[idx] = getattr(self, 'atom_name')
            elif k in ['x', 'y', 'z']:
                attrs[idx] = getattr(self.position, k)
            elif k == 'r_cov' and not self.__rcov_set_by_user:
                attrs[idx] = self._r_covs[self.__atom_name]
            elif k == 'valence' and not self.__valence_set_by_user:
                attrs[idx] = self._valences[self.__atom_name]
            elif k == 'valence_electrons' and not self.__valence_electrons_set_by_user:
                attrs[idx] = self._valence_electrons[self.__atom_name]
            elif k == 'number':
                attrs[idx] = self._numbers[self.__atom_name]
            elif k in self.__properties:
                attrs[idx] = self.get(k)
            else:
                raise AtomKeyError(f"No property found for key: {k}.")
        if len(attrs) == 1:
            return attrs[0]
        else:
            return attrs

    def __setitem__(self, key: str, value: Any) -> None:
        """Allows dictionary-like setting of properties."""
        if key in ['number']:
            raise AtomKeyError(
                f'{key} is set by other attributes and cannot be modified directly.')
        elif key in ['atom_name', 'position']:
            setattr(self, key, value)
        elif key == 'name':
            setattr(self, 'atom_name', value)
        elif key in ['x', 'y', 'z']:
            setattr(self.__position, key, value)
        else:
            self.set(key, value)

    def __iter__(self) -> Iterator:
        """Allows iteration over atom properties."""
        yield 'name', self.__atom_name
        yield 'position', self.__position
        yield from self.__properties.items()

    def __repr__(self) -> str:
        """Returns a string representation of the Atom object."""
        return (f"Atom(atom_name='{self.atom_name}', "
                f"position=({self.position.x}, {self.position.y}, "
                f"{self.position.z}), "
                f"properties={self.__properties})"
                )
