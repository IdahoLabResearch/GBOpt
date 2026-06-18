# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Quaternion normalization and conversion to scaled rotations.

Constructs exact ScaledRotation objects from integer quaternions and provides
helpers for recovering integer quaternions from unit quaternion inputs.
Returned ScaledRotation objects use the project row-vector convention
documented on ScaledRotation in types.py.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from .integer import as_int_vector
from .rotation import assert_scaled_rotation
from .types import (
    CrystallographyNotImplementedError,
    CrystallographyValueError,
    Int4,
    ScaledRotation,
)


def normalize_integer_quaternion(q: tuple) -> Int4:
    """Return the canonical primitive representative of an integer quaternion.

    :param q: Integer quaternion in Hamilton scalar-first order ``[w, x, y, z]``.
    :return: Primitive quaternion with common factors removed and a deterministic
        sign convention.
    :raises CrystallographyValueError: If ``q`` is not length 4, contains non-integers,
        or is the zero quaternion.
    """
    values = list(as_int_vector(q, 4, "q"))
    gcd_value = 0
    for value in values:
        gcd_value = math.gcd(gcd_value, abs(value))
    if gcd_value == 0:
        raise CrystallographyValueError("q is the zero quaternion.")
    values = [value // gcd_value for value in values]
    if tuple(values) < (0, 0, 0, 0):
        values = [-value for value in values]
    a, b, c, d = values
    return a, b, c, d


def quaternion_to_scaled_rotation(
    q: tuple,
    *,
    canonicalize: bool = True,
    lattice_metric: np.ndarray | None = None,
) -> ScaledRotation:
    """Construct the exact Euler-Rodrigues scaled rotation for an integer quaternion.

    :param q: Integer quaternion in Hamilton scalar-first order ``[w, x, y, z]``.
    :param canonicalize: If true, reduce common factors and apply the canonical
        sign convention before constructing the rotation.
    :param lattice_metric: Reserved non-cubic metric hook; only ``None`` is
        currently supported.
    :return: Exact scaled rotation in the project row-vector convention.
    :raises CrystallographyValueError: If the quaternion is invalid.
    :raises CrystallographyNotImplementedError: If ``lattice_metric`` is supplied.
    """
    _reject_non_cubic_metric(lattice_metric)
    if canonicalize:
        quat = normalize_integer_quaternion(q)
    else:
        w, x, y, z = as_int_vector(q, 4, "q")
        quat = (w, x, y, z)

    w, x, y, z = quat
    N = w * w + x * x + y * y + z * z
    if N == 0:
        raise CrystallographyValueError("q is the zero quaternion.")

    M = np.array(
        [
            [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z],
        ],
        dtype=object,
    )
    rotation = ScaledRotation(N=N, M=M, source="quaternion", quaternion=quat)
    assert_scaled_rotation(rotation)
    return rotation


def integer_quaternion_from_unit(q: ArrayLike, *, max_denominator: int = 10001) -> Int4:
    """Recover a primitive integer quaternion proportional to a unit quaternion.

    :param q: Unit quaternion in Hamilton scalar-first order ``[w, x, y, z]``.
    :param max_denominator: Maximum rational denominator used when recovering
        integer ratios from floating-point components.
    :return: Primitive integer quaternion.
    :raises CrystallographyValueError: If the input is not length 4 or is zero, or the
        recovered integer quaternion does not match the supplied unit quaternion.
    """
    arr = np.asarray(q, dtype=float)
    if arr.shape != (4,):
        raise CrystallographyValueError(f"q must have shape (4,); got {arr.shape}.")
    nonzero = [i for i, value in enumerate(arr) if abs(float(value)) > 1e-14]
    if not nonzero:
        raise CrystallographyValueError("q is the zero quaternion.")
    ref = nonzero[0]
    fractions: list[Fraction] = []
    for value in arr:
        if abs(float(value)) <= 1e-14:
            fractions.append(Fraction(0, 1))
        else:
            fractions.append(
                Fraction(float(value) / float(arr[ref])).limit_denominator(
                    max_denominator
                )
            )
    denominator_lcm = 1
    for value in fractions:
        denominator_lcm = math.lcm(denominator_lcm, value.denominator)
    ints = [int(value * denominator_lcm) for value in fractions]
    int_quat = normalize_integer_quaternion(tuple(ints))

    mag = math.sqrt(sum(i * i for i in int_quat))
    recovered = np.array([i / mag for i in int_quat], dtype=float)

    target = arr / np.linalg.norm(arr)
    if np.dot(recovered, target) < 0:
        recovered = -recovered

    if not np.allclose(recovered, target, atol=1e-9, rtol=0.0):
        raise CrystallographyValueError(
            "Recovered integer quaternion does not match the supplied unit quaternion."
        )

    return int_quat


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion [w, x, y, z] to a 3x3 rotation matrix.

    Delegates to ``scipy.spatial.transform.Rotation`` using scalar-last order
    internally; the reordering is handled here so callers always use Hamilton
    scalar-first order. Integer quaternions are normalized internally; unit
    quaternions are accepted unchanged.

    :param quat: Quaternion in Hamilton scalar-first order [w, x, y, z],
        shape (4,). Non-unit inputs must be integer-valued.
    :returns: Rotation matrix ``R`` of shape (3, 3) such that
        ``v_rotated = v @ R.T`` (row-vector convention).
    :raises CrystallographyValueError: If the shape is invalid, the quaternion is
        non-finite or zero, or a non-unit quaternion is not integer-valued.
    """
    q = np.asarray(quat, dtype=float)
    if q.shape != (4,):
        raise CrystallographyValueError(
            f"Quaternion must be a 1-D array of length 4; got shape {q.shape}."
        )
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm == 0.0:
        raise CrystallographyValueError("Quaternion must be non-zero and finite.")
    if not np.isclose(norm, 1.0, atol=1e-12, rtol=0.0):
        if not np.allclose(q, np.round(q), atol=1e-9, rtol=0.0):
            raise CrystallographyValueError(
                f"Quaternion components must be integer-valued; got {q}. "
                "Non-unit quaternions must be exact integer quaternions."
            )
        int_q = np.round(q).astype(int)
        q = q / np.sqrt(float(np.dot(int_q, int_q)))
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _reject_non_cubic_metric(metric: np.ndarray | None) -> None:
    """Reject non-cubic lattice metrics reserved for a later extension.

    ``metric`` is intended to represent a future 3 by 3 lattice metric tensor
    for non-cubic crystals. Exact CSL support is currently implemented only
    for the implicit cubic identity metric, so callers must pass ``None``.

    NOTE: This is a temporary guard, and it (and its companion method in `rotation.py`
    and `plane.py`) should be centralized properly when fully implemented.
    """
    if metric is not None:
        raise CrystallographyNotImplementedError(
            "non-cubic lattice metrics are not implemented"
        )


__all__ = [
    "normalize_integer_quaternion",
    "quaternion_to_scaled_rotation",
    "integer_quaternion_from_unit",
    "quaternion_to_rotation_matrix",
]
