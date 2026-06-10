# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Exact integer validation and small integer matrix helpers.

Thin wrappers over GBOpt.Utils.integer_normal_forms that translate
``ExactNormalFormError`` into the crystallography package exception hierarchy. All
functions here operate on exact Python-integer arithmetic and reject non-integer or
rank-deficient inputs rather than silently rounding.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike

import GBOpt.Utils.integer_linalg as ilinalg
from GBOpt.Utils.integer_linalg import ExactIntegerError

from .types import CrystallographyValueError

_T = TypeVar("_T")


def _translate_exact_error(func: Callable[..., _T], *args: object, **kwargs: object) -> _T:
    """Call an exact integer-linalg helper and translate its validation errors.

    :param func: Exact integer-linalg helper to call.
    :param args: Positional arguments forwarded to ``func``.
    :param kwargs: Keyword arguments forwarded to ``func``.
    :return: Result returned by ``func``.
    :raises CrystallographyValueError: If ``func`` raises ``ExactIntegerError``.
    """
    try:
        return func(*args, **kwargs)
    except ExactIntegerError as exc:
        raise CrystallographyValueError(str(exc)) from exc


def as_int_array(array: ArrayLike, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Return ``array`` as a shape-checked object array of Python integers.

    :param array: Input array-like to validate and convert.
    :param shape: Required shape that ``A`` must conform to; either a 1D tuple ``(n,)``
        or 2D tuple ``(m, n)``.
    :param name: Name used in error messages.
    :return: Object-dtype ndarray of Python integers.
    """
    return _translate_exact_error(ilinalg.as_int_array, array, shape, name)


def as_int_vector(values: ArrayLike, length: int, name: str) -> tuple[int, ...]:
    """Return an exact integer tuple from a 1D array-like input.

    Delegates to ``as_int_array`` with ``shape=(length,)`` and converts to a plain
    Python tuple of ints for use in contexts that require hashable sequences.

    :param values: 1D array-like of integer-valued entries.
    :param length: Expected number of elements.
    :param name: Name used in error messages.
    :return: Tuple of Python ints.
    """
    return _translate_exact_error(ilinalg.as_int_vector, values, length, name)


def as_positive_int(value: object, name: str) -> int:
    """Return a positive Python integer from an integer scalar.

    Boolean values are rejected even though ``bool`` is a subclass of ``int``.

    :param value: Candidate Python or NumPy integer scalar.
    :param name: Parameter name used in validation error messages.
    :return: Validated value converted to a Python ``int``.
    :raises CrystallographyValueError: If ``value`` is not an integer scalar or is
        less than or equal to zero.
    """
    if isinstance(value, (bool, np.bool_)):
        raise CrystallographyValueError(
            f"{name} must be a positive integer; got {value!r}."
        )

    try:
        result = operator.index(value)  # type: ignore[ty:invalid-argument-type]
    except TypeError as exc:
        raise CrystallographyValueError(
            f"{name} must be a positive integer; got {value!r}."
        ) from exc

    if result <= 0:
        raise CrystallographyValueError(
            f"{name} must be a positive integer; got {value!r}."
        )

    return int(result)


def row_gcd_reduce(row: np.ndarray) -> np.ndarray:
    """Divide an integer-valued row by the GCD of its absolute components.

    :param row: One-dimensional integer-valued row.
    :return: GCD-reduced row as object-dtype array of Python integers. An all-zero row
        is returned unchanged.
    """
    return _translate_exact_error(ilinalg.row_gcd_reduce, row)


def dot_int(x: ArrayLike, y: ArrayLike) -> int:
    """Return the exact integer dot product of two equal-length vectors.

    :param x: First array-like vector.
    :param y: Second array-like vector.
    :return: Exact Python-int dot product.
    """
    return _translate_exact_error(ilinalg.dot_int, x, y)


def cross_int3(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Return the exact integer cross product of two length-3 vectors.

    :param x: First length-3 array-like vector.
    :param y: Second length-3 array-like vector.
    :return: Length-3 object-dtype cross product.
    """
    return _translate_exact_error(ilinalg.cross_int3, x, y)


def integer_det3(matrix: ArrayLike) -> int:
    """Compute the exact determinant of a 3 by 3 integer matrix.

    :param matrix: 3 by 3 integer-valued matrix.
    :return: Exact Python-int determinant.
    """
    return _translate_exact_error(ilinalg.det3_int, matrix)


def integer_adj3(matrix: ArrayLike) -> list[list[int]]:
    """Compute the exact adjugate of a 3 by 3 integer matrix.

    :param matrix: 3 by 3 integer-valued matrix.
    :return: 3 by 3 list-of-lists representing ``adj(matrix)``.
    """
    return _translate_exact_error(ilinalg.adjugate3_int, matrix)


__all__ = [
    "as_int_array",
    "as_int_vector",
    "as_positive_int",
    "row_gcd_reduce",
    "dot_int",
    "cross_int3",
    "integer_det3",
    "integer_adj3",
]
