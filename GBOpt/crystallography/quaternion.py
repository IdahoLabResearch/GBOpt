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


def _validated_float_quaternion(quat: ArrayLike) -> np.ndarray:
    """Return a validated finite floating-point quaternion.

    :param quat: Candidate quaternion in Hamilton scalar-first order ``[w, x, y, z]``.
    :return: Finite ``float64`` array with shape ``(4,)``.
    :raises CrystallographyValueError: If ``quat`` cannot be converted to floating
        point, does not have shape ``(4,)``, or contains a non-finite value.
    """
    try:
        arr = np.asarray(quat, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CrystallographyValueError(
            "quat must be a finite four-component quaternion."
        ) from exc

    if arr.shape != (4,):
        raise CrystallographyValueError(
            f"quat must have shape (4,); got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise CrystallographyValueError(
            "quat must contain only finite values."
        )

    return arr


def _integer_quaternion_candidate_from_unit(
    quat: ArrayLike,
    *,
    max_denominator: int = 10001,
) -> Int4:
    """Return a bounded-denominator integer candidate for a quaternion direction.

    The candidate is primitive and uses the canonical quaternion sign convention.
    This helper deliberately performs no reconstruction-accuracy check. Callers are
    responsible for applying the error metric and tolerance appropriate to their
    operation.

    :param quat: Quaternion direction in Hamilton scalar-first order
        ``[w, x, y, z]``. The input need not be exactly unit length.
    :param max_denominator: Maximum denominator used when rationalizing component
        ratios. Keyword argument, optional, defaults to ``10001``.
    :return: Primitive canonical integer quaternion.
    :raises CrystallographyValueError: If ``quat`` cannot be converted to a finite
        four-component vector, is effectively zero, or ``max_denominator`` is not a
        positive integer.
    """
    try:
        arr = np.asarray(quat, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CrystallographyValueError(
            "quat must be a finite four-component quaternion."
        ) from exc

    max_denominator = as_positive_int(
        max_denominator,
        "max_denominator",
    )

    if arr.shape != (4,):
        raise CrystallographyValueError(
            f"quat must have shape (4,); got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise CrystallographyValueError(
            "quat must contain only finite values."
        )

    # Components below this numerical floor are treated as exact zeros when
    # constructing ratios relative to the largest-magnitude component.
    zero_tol = 1.0e-14
    abs_arr = np.abs(arr)
    if not np.any(abs_arr > zero_tol):
        raise CrystallographyValueError(
            "quat is the zero quaternion."
        )

    reference_index = int(np.argmax(abs_arr))
    reference = float(arr[reference_index])

    fractions = [
        (
            Fraction(0, 1)
            if abs(float(component)) <= zero_tol
            else Fraction(float(component) / reference).limit_denominator(
                max_denominator
            )
        )
        for component in arr
    ]

    denominator_lcm = math.lcm(
        *(fraction.denominator for fraction in fractions)
    )
    components = tuple(
        int(fraction * denominator_lcm)
        for fraction in fractions
    )

    return normalize_integer_quaternion(components)


def integer_quaternion_from_unit(
    quat: ArrayLike,
    *,
    max_denominator: int = 10001,
) -> Int4:
    """Recover a primitive integer quaternion proportional to a unit quaternion.

    This public convenience function constructs a bounded-denominator candidate and
    verifies that its normalized quaternion direction matches the supplied direction
    within a fixed componentwise tolerance. Callers that require a different error
    metric should use their own operation-specific policy around the private candidate
    constructor.

    :param quat: Unit quaternion in Hamilton scalar-first order ``[w, x, y, z]``.
    :param max_denominator: Maximum rational denominator used when recovering integer
        ratios from floating-point components. Keyword argument, optional, defaults to
        ``10001``.
    :return: Primitive canonical integer quaternion.
    :raises CrystallographyValueError: If validation or rationalization fails, or if
        the recovered integer quaternion does not reproduce the supplied quaternion
        direction within the fixed public-function tolerance.
    """
    int_quat = _integer_quaternion_candidate_from_unit(
        quat,
        max_denominator=max_denominator,
    )

    # Candidate construction has already validated conversion, shape, finiteness,
    # and nonzero magnitude.
    target = np.asarray(quat, dtype=np.float64)
    target /= np.linalg.norm(target)

    int_quat_array = np.asarray(int_quat, dtype=np.float64)
    recovered = int_quat_array / np.linalg.norm(int_quat_array)

    if float(np.dot(recovered, target)) < 0.0:
        recovered = -recovered

    if not np.allclose(
        recovered,
        target,
        atol=1.0e-9,
        rtol=0.0,
    ):
        raise CrystallographyValueError(
            "Recovered integer quaternion does not match the supplied "
            "unit quaternion."
        )

    return int_quat


def _unit_integer_quaternion(values: tuple[int, int, int, int]) -> np.ndarray:
    """Return a scale-safe unit vector for a nonzero integer quaternion.

    The integer components are divided by their largest absolute value before
    normalization to avoid unnecessary floating-point overflow.

    :param values: Integer quaternion in Hamilton scalar-first order ``(w, x, y, z)``.
    :return: Unit-length ``float64`` quaternion with shape ``(4,)``.
    :raises CrystallographyValueError: If ``values`` is the zero quaternion.
    """
    scale = max(map(abs, values))
    if scale == 0:
        raise CrystallographyValueError(
            "Quaternion must be non-zero."
        )

    scaled = np.fromiter(
        (value / scale for value in values),
        dtype=np.float64,
        count=4,
    )
    return scaled / np.linalg.norm(scaled)


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
    try:
        int_quat = as_int_vector(quat, 4, "quat")
    except CrystallographyValueError:
        quat_array = _validated_float_quaternion(quat)
        norm = float(np.linalg.norm(quat_array))
        if norm == 0.0:
            raise CrystallographyValueError(
                "Quaternion must be non-zero."
            )
        if not np.isclose(norm, 1.0, atol=1.0e-12, rtol=0.0):
            raise CrystallographyValueError(
                "Non-unit quaternions must be exact integer quaternions."
            )
        quat_array = quat_array / norm
    else:
        quat_array = _unit_integer_quaternion(int_quat)

    return Rotation.from_quat(quat_array[[1, 2, 3, 0]]).as_matrix()


__all__ = [
    "normalize_integer_quaternion",
    "quaternion_to_scaled_rotation",
    "integer_quaternion_from_unit",
    "quaternion_to_rotation_matrix",
]
