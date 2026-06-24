# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Exact integer linear-algebra utilities.

This module provides reusable exact integer validation and small-vector/matrix
operations. It intentionally avoids fixed-width NumPy integer arithmetic by returning
object-dtype arrays containing Python ``int`` values.
"""

from __future__ import annotations

import math
import operator

import numpy as np
from numpy.typing import ArrayLike


class ExactIntegerError(Exception):
    """Base exception for exact integer linear-algebra failures."""


class ExactIntegerShapeError(ExactIntegerError):
    """Raised when an exact integer input has the wrong shape."""


class ExactIntegerValueError(ExactIntegerError, ValueError):
    """Raised when an exact integer input has an invalid value."""


class ExactIntegerTypeError(ExactIntegerError, TypeError):
    """Raised when an exact integer input has an invalid type."""


def _as_flat_int_array(
    values: ArrayLike,
    name: str,
    *,
    length: int | None = None,
) -> np.ndarray:
    """Return values flattened as an object array of exact Python integers.

    :param values: Array-like input to flatten, validate, and convert.
    :param name: Name used in error messages.
    :param length: Required flattened length, or ``None`` to accept any length. Keyword
        argument, optional, defaults to ``None``.
    :return: One-dimensional object-dtype ndarray of Python int values.
    :raises ExactIntegerTypeError: If values cannot be converted to an object array.
    :raises ExactIntegerShapeError: If length is supplied and the flattened input length
        does not match.
    """
    try:
        flat = np.asarray(values, dtype=object).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ExactIntegerTypeError(
            f"{name} cannot be converted to an array: {exc}"
        ) from exc

    if length is not None and flat.size != length:
        raise ExactIntegerShapeError(
            f"{name} must flatten to length {length}; got {flat.size}."
        )

    out = np.empty((flat.size,), dtype=object)
    for index, value in enumerate(flat):
        out[index] = _coerce_exact_int(value, f"{name}[{index}]")
    return out


def _coerce_exact_int(value: object, name: str) -> int:
    """Return ``value`` as a Python ``int`` after exact integer validation.

    :param value: Scalar value to validate and convert.
    :param name: Name used in error messages.
    :return: ``value`` converted to a Python ``int``.
    :raises ExactIntegerTypeError: If ``value`` is boolean, string-like, or cannot be
        converted to an integer scalar.
    :raises ExactIntegerValueError: If ``value`` is non-finite or not exactly
        integer-valued.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ExactIntegerTypeError(f"{name}={value!r} is not an integer.")

    if isinstance(value, (str, bytes)):
        raise ExactIntegerTypeError(f"{name}={value!r} is not an integer.")

    try:
        return int(operator.index(value))  # type: ignore[ty:invalid-argument-type]
    except TypeError:
        pass

    try:
        finite = np.isfinite(value)  # type: ignore[ty:no-matching-overload]
    except TypeError:
        finite = True

    if finite is not True and not bool(finite):
        raise ExactIntegerValueError(f"{name}={value!r} is not finite.")

    try:
        integer = int(value)  # type: ignore[ty:invalid-argument-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExactIntegerTypeError(f"{name}={value!r} is not an integer.") from exc

    if value != integer:
        raise ExactIntegerValueError(
            f"{name}={value!r} is not exactly integer-valued."
        )

    return integer


def as_int_array(array: ArrayLike, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Return a shape-checked object array of exact Python integers.

    :param array: Array-like input to validate.
    :param shape: Expected shape of array.
    :param name: Name used in error messages.
    :return: Object-dtype ndarray of Python int values with shape shape.
    :raises ExactIntegerTypeError: If array cannot be converted to an object array.
    :raises ExactIntegerShapeError: If array does not have shape shape.
    """
    try:
        arr = np.asarray(array, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ExactIntegerTypeError(
            f"{name} cannot be converted to an array: {exc}"
        ) from exc

    if arr.shape != shape:
        raise ExactIntegerShapeError(
            f"{name} must have shape {shape}; got {arr.shape}."
        )

    out = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        out[index] = _coerce_exact_int(arr[index], f"{name}{index}")
    return out


def as_int_vector(values: ArrayLike, length: int, name: str) -> tuple[int, ...]:
    """Return a fixed-length tuple of exact Python integers.

    :param values: One-dimensional array-like input to validate.
    :param length: Required vector length.
    :param name: Name used in error messages.
    :return: Tuple of Python ``int`` values.
    """
    arr = as_int_array(values, (length,), name)
    return tuple(int(value) for value in arr)


def identity_int(n: int) -> np.ndarray:
    """Return an ``n`` by ``n`` identity matrix with Python integer entries.

    :param n: Matrix dimension.
    :return: ``n`` by ``n`` object-dtype identity matrix.
    :raises ExactIntegerValueError: If ``n`` is negative.
    """
    dim = _coerce_exact_int(n, "n")
    if dim < 0:
        raise ExactIntegerValueError(f"n must be nonnegative; got {dim}.")
    return np.eye(dim, dtype=object)


def row_gcd_reduce(row: ArrayLike) -> np.ndarray:
    """Divide a one-dimensional integer row by its component GCD.

    Rows that are already primitive, and all-zero rows, are returned unchanged except
    for conversion to object dtype with Python ``int`` entries.

    :param row: One-dimensional integer-valued array-like input.
    :return: Object-dtype row with the common component GCD divided out.
    :raises ExactIntegerShapeError: If ``row`` is not one-dimensional.
    """
    arr = np.asarray(row)
    if arr.ndim != 1:
        raise ExactIntegerShapeError(
            f"row must be a 1D integer-valued array; got shape {arr.shape}."
        )

    ints = [
        _coerce_exact_int(value, f"row[{index}]") for index, value in enumerate(arr)
    ]
    gcd_value = math.gcd(*(abs(value) for value in ints))
    if gcd_value <= 1:
        return np.array(ints, dtype=object)
    return np.array([value // gcd_value for value in ints], dtype=object)


def dot_int(x: ArrayLike, y: ArrayLike) -> int:
    """Return the exact integer dot product of two equal-length vectors.

    Both inputs are flattened before validation. Unlike direct ``int(value)``
    conversion, this function rejects non-integral floats and booleans rather than
    silently truncating them.

    :param x: First array-like vector.
    :param y: Second array-like vector.
    :return: Exact Python-int dot product.
    :raises ExactIntegerShapeError: If ``x`` and ``y`` have different flattened lengths.
    """
    x_vals = _as_flat_int_array(x, "x")
    y_vals = _as_flat_int_array(y, "y")

    if x_vals.size != y_vals.size:
        raise ExactIntegerShapeError(
            f"Dot product requires equal lengths; got {x_vals.size} and {y_vals.size}."
        )

    return sum(int(xi) * int(yi) for xi, yi in zip(x_vals, y_vals))


def cross_int3(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Return the exact integer cross product of two length-3 vectors.

    Both inputs are flattened before validation. The result is returned as an
    object-dtype ndarray to preserve Python-int arithmetic for large entries.

    :param x: First length-3 array-like vector.
    :param y: Second length-3 array-like vector.
    :return: Length-3 object-dtype array containing x cross y.
    :raises ExactIntegerTypeError: If either input cannot be converted to an object
        array.
    :raises ExactIntegerShapeError: If either input does not flatten to exactly three
        entries.
    """
    try:
        x_flat = np.asarray(x, dtype=object).reshape(-1)
        y_flat = np.asarray(y, dtype=object).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ExactIntegerTypeError(
            f"Cross product inputs cannot be converted to arrays: {exc}"
        ) from exc

    if x_flat.size != 3 or y_flat.size != 3:
        raise ExactIntegerShapeError(
            f"Cross product requires length-3 vectors; got {x_flat.size} "
            f"and {y_flat.size}."
        )

    x0, x1, x2 = (int(value) for value in _as_flat_int_array(x_flat, "x", length=3))
    y0, y1, y2 = (int(value) for value in _as_flat_int_array(y_flat, "y", length=3))

    return np.array(
        [
            x1 * y2 - x2 * y1,
            x2 * y0 - x0 * y2,
            x0 * y1 - x1 * y0,
        ],
        dtype=object,
    )


def det3_int_checked(matrix: np.ndarray) -> int:
    """Return the exact determinant of a validated 3 by 3 integer matrix.

    This is a trusted helper for hot internal paths. It does not validate shape or
    integer-valuedness; callers must validate with ``as_int_array`` before calling.

    :param matrix: Validated 3 by 3 integer-valued matrix.
    :return: Exact Python-int determinant.
    """
    return int(
        matrix[0, 0] * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1])
        - matrix[0, 1] * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0])
        + matrix[0, 2] * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0])
    )


def det3_int(matrix: ArrayLike) -> int:
    """Return the exact determinant of a 3 by 3 integer matrix.

    :param matrix: 3 by 3 integer-valued array-like input.
    :return: Exact Python-int determinant.
    """
    int_matrix = as_int_array(matrix, (3, 3), "matrix")
    return det3_int_checked(int_matrix)


def adjugate3_int(matrix: ArrayLike) -> list[list[int]]:
    """Return the exact adjugate of a 3 by 3 integer matrix.

    The adjugate is returned as a Python list-of-lists so row-vector products such as
    ``n @ adj`` can stay in pure Python-integer arithmetic when callers convert as
    needed.

    :param matrix: 3 by 3 integer-valued array-like input.
    :return: 3 by 3 list-of-lists representing ``adj(matrix)``.
    """
    a = as_int_array(matrix, (3, 3), "matrix")

    a00, a01, a02 = int(a[0, 0]), int(a[0, 1]), int(a[0, 2])
    a10, a11, a12 = int(a[1, 0]), int(a[1, 1]), int(a[1, 2])
    a20, a21, a22 = int(a[2, 0]), int(a[2, 1]), int(a[2, 2])

    return [
        [
            a11 * a22 - a12 * a21,
            a02 * a21 - a01 * a22,
            a01 * a12 - a02 * a11,
        ],
        [
            a12 * a20 - a10 * a22,
            a00 * a22 - a02 * a20,
            a02 * a10 - a00 * a12,
        ],
        [
            a10 * a21 - a11 * a20,
            a01 * a20 - a00 * a21,
            a00 * a11 - a01 * a10,
        ],
    ]


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return the GCD and Bezout coefficients for two integers.

    The result ``(g, x, y)`` satisfies ``x*a + y*b == g`` with ``g == gcd(abs(a),
    abs(b))``. Signed inputs are handled directly.

    :param a: First integer scalar.
    :param b: Second integer scalar.
    :return: ``(g, x, y)``, where ``g`` is nonnegative and ``x*a + y*b == g``.
    """
    old_r = _coerce_exact_int(a, "a")
    r = _coerce_exact_int(b, "b")
    old_s, s = 1, 0
    old_t, t = 0, 1

    while r:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t

    if old_r < 0:
        old_r, old_s, old_t = -old_r, -old_s, -old_t

    return old_r, old_s, old_t


__all__ = [
    "ExactIntegerError",
    "ExactIntegerShapeError",
    "ExactIntegerTypeError",
    "ExactIntegerValueError",
    "as_int_array",
    "as_int_vector",
    "identity_int",
    "row_gcd_reduce",
    "dot_int",
    "cross_int3",
    "det3_int",
    "det3_int_checked",
    "adjugate3_int",
    "extended_gcd",
]
