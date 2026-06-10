# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Quaternion normalization and conversion to scaled rotations.

Constructs exact ``ScaledRotation`` objects from integer quaternions and provides
helpers for recovering integer quaternions from unit quaternion inputs. Returned
``ScaledRotation`` objects use the project row-vector convention documented on
``ScaledRotation`` in types.py.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from ._guards import _require_cubic
from .integer import as_int_vector, as_positive_int
from .types import (
    CrystallographyValueError,
    Int4,
    ScaledRotation,
)


def normalize_integer_quaternion(quat: ArrayLike) -> Int4:
    """Return the canonical primitive representative of an integer quaternion.

    :param quat: Integer quaternion in Hamilton scalar-first order ``[w, x, y, z]``.
    :return: Primitive quaternion with common factors removed and a deterministic sign
        convention: the first nonzero component is always positive.
    :raises CrystallographyValueError: If ``quat`` is the zero quaternion after integer
        conversion.
    """
    values = list(as_int_vector(quat, 4, "quat"))
    gcd_value = math.gcd(*map(abs, values))
    if gcd_value == 0:
        raise CrystallographyValueError("quat is the zero quaternion.")
    values = [value // gcd_value for value in values]
    if tuple(values) < (0, 0, 0, 0):
        values = [-value for value in values]
    a, b, c, d = values
    return a, b, c, d


def quaternion_to_scaled_rotation(
    quat: ArrayLike,
    *,
    canonicalize: bool = True,
    lattice_metric: np.ndarray | None = None,
) -> ScaledRotation:
    """Construct the exact Euler-Rodrigues scaled rotation for an integer quaternion.

    :param quat: Integer quaternion in Hamilton scalar-first order ``[w, x, y, z]``.
    :param canonicalize: If ``True``, reduce common factors and apply the canonical sign
        convention before constructing the rotation. Set to ``False`` only when testing
        that a specific non-primitive quaternion produces the expected scaled
        denominator without GCD reduction. Keyword argument, optional, defaults to
        ``True``.
    :param lattice_metric: Reserved non-cubic metric hook; only ``None`` is currently
        supported. Keyword argument, optional, defaults to ``None``.
    :return: Exact scaled rotation in the project row-vector convention.
    :raises CrystallographyValueError: If ``quat`` is the zero quaternion.
    """
    _require_cubic(lattice_metric)
    if canonicalize:
        w, x, y, z = normalize_integer_quaternion(quat)
    else:
        w, x, y, z = as_int_vector(quat, 4, "quat")

    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    denominator = ww + xx + yy + zz
    if denominator == 0:
        raise CrystallographyValueError("quat is the zero quaternion.")

    # Euler-Rodrigues formula
    xy, wz, xz, wy, yz, wx = x * y, w * z, x * z, w * y, y * z, w * x
    numerator_matrix = np.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=object,
    )
    return ScaledRotation(
        denominator=denominator,
        matrix=numerator_matrix,
        source="quaternion",
        quaternion=(w, x, y, z),
    )


def integer_quaternion_from_unit(quat: ArrayLike, *, max_denominator: int = 10001) -> Int4:
    """Recover a primitive integer quaternion proportional to a unit quaternion.

    :param quat: Unit quaternion in Hamilton scalar-first order ``[w, x, y, z]``.
    :param max_denominator: Maximum rational denominator used when recovering integer
        ratios from floating-point components. Keyword argument, optional, defaults to
        ``10001``.
    :return: Primitive integer quaternion.
    :raises CrystallographyValueError: If the input is not length 4, is zero, or has
        non-finite values; max_denominator is not a positive integer; or the recovered
        integer quaternion does not match the supplied unit quaternion.
    """
    arr = np.asarray(quat, dtype=float)
    max_denominator = as_positive_int(max_denominator, "max_denominator")
    if arr.shape != (4,):
        raise CrystallographyValueError(f"quat must have shape (4,); got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise CrystallographyValueError("quat must contain only finite values.")

    # 1e-14: numerical zero floor (well below float64 machine epsilon ~2.2e-16 times
    # typical quaternion magnitudes); components smaller than this are treated as
    # exactly zero before computing ratios relative to the largest-magnitude component.
    zero_tol = 1e-14
    abs_arr = np.abs(arr)
    if not np.any(abs_arr > zero_tol):
        raise CrystallographyValueError("quat is the zero quaternion.")

    reference_index = int(np.argmax(abs_arr))
    reference = float(arr[reference_index])

    fractions = [
        Fraction(0, 1)
        if abs(float(v)) <= zero_tol
        else Fraction(float(v) / reference).limit_denominator(max_denominator)
        for v in arr
    ]

    denominator_lcm = math.lcm(*(value.denominator for value in fractions))
    int_components = [int(value * denominator_lcm) for value in fractions]
    int_quat = normalize_integer_quaternion(tuple(int_components))

    int_quat_arr = np.asarray(int_quat, dtype=float)
    recovered = int_quat_arr / np.linalg.norm(int_quat_arr)

    # arr may not be exactly unit-norm due to floating-point; renormalize defensively
    target = arr / np.linalg.norm(arr)
    if np.dot(recovered, target) < 0:
        recovered = -recovered

    # 1e-9: reconstruction accuracy tolerance; verifies that the recovered integer
    # quaternion, when renormalized, reproduces the original unit quaternion direction
    # to within acceptable float64 round-trip error.
    if not np.allclose(recovered, target, atol=1e-9, rtol=0.0):
        raise CrystallographyValueError(
            "Recovered integer quaternion does not match the supplied unit quaternion."
        )

    return int_quat


def quaternion_to_rotation_matrix(quat: ArrayLike) -> np.ndarray:
    """Convert a quaternion ``[w, x, y, z]`` to a 3 by 3 rotation matrix.

    Delegates to ``scipy.spatial.transform.Rotation`` using scalar-last order
    internally; the reordering is handled here so callers always use Hamilton
    scalar-first order. Integer quaternions are normalized internally, and unit
    quaternions are accepted unchanged.

    :param quat: Quaternion in Hamilton scalar-first order ``[w, x, y, z]``, with shape
        ``(4,)``. Non-unit inputs must be integer-valued.
    :return: Rotation matrix ``R`` of shape ``(3, 3)`` such that ``v_rotated = v @ R.T``
        under the row-vector convention.
    :raises CrystallographyValueError: If the shape is invalid, the quaternion is
        non-finite or zero, or a non-unit quaternion is not integer-valued.
    """
    quat_array = np.asarray(quat, dtype=float)
    if quat_array.shape != (4,):
        raise CrystallographyValueError(
            f"Quaternion must be a 1-D array of length 4; got shape {quat_array.shape}."
        )
    norm = float(np.linalg.norm(quat_array))
    if not np.isfinite(norm) or norm == 0.0:
        raise CrystallographyValueError("Quaternion must be non-zero and finite.")
    # 1e-12: tight tolerance for unit-norm check; properly normalized quaternions
    # from scipy or explicit normalization should be within a few ULPs of 1.
    if not np.isclose(norm, 1.0, atol=1e-12, rtol=0.0):
        # 1e-9: looser tolerance for integer-valuedness; integer-valued floats
        # can carry small rounding errors from upstream arithmetic.
        if not np.allclose(quat_array, np.round(quat_array), atol=1e-9, rtol=0.0):
            raise CrystallographyValueError(
                f"Quaternion components must be integer-valued; got {quat_array}. "
                "Non-unit quaternions must be exact integer quaternions."
            )
        int_q = np.round(quat_array).astype(int)
        quat_array = int_q.astype(float) / np.sqrt(float(np.dot(int_q, int_q)))

    scalar_last = quat_array[[1, 2, 3, 0]]
    return Rotation.from_quat(scalar_last).as_matrix()


__all__ = [
    "normalize_integer_quaternion",
    "quaternion_to_scaled_rotation",
    "integer_quaternion_from_unit",
    "quaternion_to_rotation_matrix",
]
