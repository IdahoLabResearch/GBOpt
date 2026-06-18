# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Exact integer validation and small integer matrix helpers.

Thin wrappers over GBOpt.Utils.integer_normal_forms that translate
ExactNormalFormError into the crystallography package exception hierarchy.
All functions here operate on exact Python-integer arithmetic and reject
non-integer or rank-deficient inputs rather than silently rounding.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from GBOpt.Utils import integer_normal_forms as inf
from GBOpt.Utils.integer_normal_forms import ExactNormalFormError

from .types import CrystallographyValueError


def as_int_array(A: ArrayLike, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Return *A* as a shape-checked object array of Python integers.

    Thin wrapper over ``integer_normal_forms._as_int_matrix`` that translates
    ``ExactNormalFormError`` to ``CrystallographyValueError`` for consistency with this
    module's exception hierarchy.

    :param A: Input array-like of any shape.
    :param shape: Expected shape (1D or 2D).
    :param name: Name used in error messages.
    :return: Object-dtype ndarray of Python integers.
    :raises CrystallographyValueError: If shape mismatches or any entry is non-integer.
    """
    try:
        return inf._as_int_matrix(A, shape, name)
    except ExactNormalFormError as exc:
        raise CrystallographyValueError(str(exc)) from exc


def as_int_vector(values: ArrayLike, length: int, name: str) -> tuple[int, ...]:
    """Return an exact integer tuple from a 1D array-like input.

    Delegates to ``_as_int_array`` with ``shape=(length,)`` and converts to a
    plain Python tuple of ints for use in contexts that require hashable sequences.

    :param values: 1D array-like of integer-valued entries.
    :param length: Expected number of elements.
    :param name: Name used in error messages.
    :return: Tuple of Python ints.
    :raises CrystallographyValueError: If length mismatches or any entry is non-integer.
    """
    arr = as_int_array(values, (length,), name)
    return tuple(int(v) for v in arr)


def row_gcd_reduce_int(row: np.ndarray) -> np.ndarray:
    """Divide an integer-valued row by the GCD of its absolute components.

    :param row: One-dimensional integer-valued row.
    :return: GCD-reduced row as object-dtype array of Python integers.
    :raises CrystallographyValueError: If ``row`` is not a one-dimensional integer row.
    """
    try:
        return inf._row_gcd_reduce(row)
    except ExactNormalFormError as exc:
        raise CrystallographyValueError(str(exc)) from exc


def row_gcd_reduce_float(row: np.ndarray) -> np.ndarray:
    """Divide an integer-valued row by the GCD of its absolute components.

    GBMaker's rotation-matrix pipeline expects float arrays, so this wrapper casts the
    results of ``row_gcd_reduce_int`` to float dtype.

    :param row: One-dimensional integer-valued row.
    :return: GCD-reduced row as float dtype.
    :raises CrystallographyValueError: If ``row`` is not a one-dimensional integer row.
    """
    return row_gcd_reduce_int(row).astype(float)


def assert_integer_rows(M: np.ndarray, name: str) -> None:
    """Raise if any row of ``M`` is not close to integer-valued.

    :param M: Matrix whose rows should be integer-valued within tolerance.
    :param name: Name used in error messages.
    :raises CrystallographyValueError: If any row contains non-integer values.
    """
    for i, row in enumerate(M):
        if not np.allclose(row, np.round(row), atol=1e-9, rtol=0.0):
            raise CrystallographyValueError(
                f"{name} row {i} {row} is not integer-valued. "
            )


def integer_det3(M) -> int:
    """Compute the determinant of a 3 by 3 integer matrix.

    The input is validated as exactly integer-valued before delegating to the
    shared normal-form determinant helper.

    :param M: 3 by 3 array-like with integer-valued entries.
    :return: Integer determinant.
    :raises CrystallographyValueError: If shape or integer validation fails.
    """
    try:
        return inf._int_det3(inf._as_int_matrix(M, (3, 3), "M"))
    except ExactNormalFormError as exc:
        raise CrystallographyValueError(str(exc)) from exc


def integer_adj3(M) -> list:
    """Compute the adjugate of a 3 by 3 integer matrix.

    :param M: 3 by 3 array-like with integer-valued entries.
    :return: 3 by 3 list-of-lists representing adj(M).
    :raises CrystallographyValueError: If shape or integer validation fails.
    """
    try:
        return inf._int_adj3(M)
    except ExactNormalFormError as exc:
        raise CrystallographyValueError(str(exc)) from exc


__all__ = [
    "as_int_array",
    "as_int_vector",
    "row_gcd_reduce_int",
    "row_gcd_reduce_float",
    "assert_integer_rows",
    "integer_det3",
    "integer_adj3",
]
