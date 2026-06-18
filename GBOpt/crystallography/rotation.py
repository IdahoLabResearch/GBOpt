# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Scaled-rotation validation and convention helpers.

Validates ScaledRotation objects and provides convention-shifting utilities.
The project row-vector convention is documented on ScaledRotation in types.py;
functions here enforce and convert between row and column conventions without
duplicating that documentation.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .integer import as_int_array, integer_det3, row_gcd_reduce_int
from .types import (
    CrystallographyDivisibilityError,
    CrystallographyNotImplementedError,
    CrystallographyValueError,
    ScaledRotation,
)


def assert_scaled_rotation(rotation: ScaledRotation) -> None:
    """Raise if ``rotation`` fails exact scaled-rotation identities.

    :param rotation: Scaled rotation to verify.
    :raises CrystallographyValueError: If ``M @ M.T != N**2 I`` or ``det(M) != N**3``.
    """
    N = int(rotation.N)
    M = np.asarray(rotation.M, dtype=object)
    gram = M @ M.T
    expected = (N * N) * np.eye(3, dtype=object)
    if not np.array_equal(gram, expected):
        raise CrystallographyValueError("scaled rotation is not exactly orthogonal.")
    det = integer_det3(M)
    if det != N ** 3:
        raise CrystallographyValueError(
            f"scaled rotation determinant {det} does not equal N^3={N ** 3}."
        )


def validate_scaled_rotation_matrix(
    M_in: np.ndarray,
    N: int | None = None,
    *,
    source: Literal["matrix", "five_dof", "quaternion"] = "matrix",
    reduce_common_factor: bool = False,
    lattice_metric: np.ndarray | None = None,
) -> ScaledRotation:
    """Validate a user-supplied integer scaled rotation matrix.

    :param M_in: 3 by 3 integer numerator matrix for a scaled rotation ``M / N``.
    :param N: Optional positive denominator. When omitted, it is derived from
        ``M_in @ M_in.T``.
    :param source: Label describing the input source stored on the result.
    :param reduce_common_factor: If true, remove a common factor from ``M_in``
        and ``N`` before validation.
    :param lattice_metric: Reserved non-cubic lattice metric hook; only ``None``
        is currently supported.
    :return: Validated scaled rotation.
    :raises CrystallographyValueError: If the matrix is not an exact proper rotation.
    """
    _reject_non_cubic_metric(lattice_metric)
    M = as_int_array(M_in, (3, 3), "M_in")
    expected_N = None if N is None else int(N)
    if N is not None and expected_N != N:
        raise CrystallographyValueError(f"N must be an integer; got {N!r}.")

    if reduce_common_factor:
        gcd_value = math.gcd(*[abs(int(v)) for v in M.flat])
        if gcd_value > 1:
            if expected_N is not None and expected_N % gcd_value != 0:
                raise CrystallographyValueError(
                    "common matrix factor does not divide the supplied N."
                )
            M = np.array(
                [int(value) // gcd_value for value in M.flat],
                dtype=object,
            ).reshape(3, 3)
            if expected_N is not None:
                expected_N //= gcd_value

    gram = M @ M.T
    diagonal = [int(gram[i, i]) for i in range(3)]
    if diagonal[0] <= 0 or len(set(diagonal)) != 1:
        raise CrystallographyValueError(
            "M @ M.T does not have equal positive diagonal entries."
        )
    for i in range(3):
        for j in range(3):
            if i != j and gram[i, j] != 0:
                raise CrystallographyValueError(
                    "M @ M.T has nonzero off-diagonal entries.")
    derived_N = math.isqrt(diagonal[0])
    if derived_N * derived_N != diagonal[0]:
        raise CrystallographyValueError("M @ M.T diagonal is not a perfect square.")
    det = integer_det3(M)
    if det != derived_N ** 3:
        raise CrystallographyValueError(
            f"det(M)={det} does not equal N^3={derived_N ** 3}."
        )
    if expected_N is not None and expected_N != derived_N:
        raise CrystallographyValueError(
            f"supplied N={expected_N} does not match derived N={derived_N}."
        )
    return ScaledRotation(N=derived_N, M=M, source=source)


def transpose_rotation_convention(rotation: ScaledRotation) -> ScaledRotation:
    """Return the transposed convention of a scaled rotation.

    :param rotation: Scaled rotation to transpose.
    :returns: A validated scaled rotation with numerator ``rotation.M.T`` and the same
        denominator.
    :raises CrystallographyValueError: If the transposed rotation fails validation.
    """
    return validate_scaled_rotation_matrix(
        np.asarray(rotation.M, dtype=object).T,
        N=rotation.N, source=rotation.source
    )


def scaled_row_image(
    row: ArrayLike,
    rotation: ScaledRotation,
    *,
    require_divisible: bool = True,
) -> np.ndarray:
    """Apply a row-convention scaled rotation to an integer row.

    The project row-vector convention is::

        image = row @ rotation.M / rotation.N

    When ``require_divisible`` is true, the image must be exactly integer-valued. When
    it is false, the integer numerator is GCD-reduced and returned as the primitive
    integer direction of the rational image.

    :param row: Integer row vector of length 3.
    :param rotation: Row-convention scaled rotation.
    :param require_divisible: Whether to require exact divisibility by ``rotation.N``.
        Optional. Defaults to True.
    :returns: Integer image row.
    :raises CrystallographyValueError: If ``row`` is invalid, ``rotation`` is invalid,
        or exact divisibility is required but unavailable.
    """
    row_obj = as_int_array(row, (3,), "row")
    checked = validate_scaled_rotation_matrix(
        rotation.M,
        N=rotation.N,
        source=rotation.source,
    )

    numerator = row_obj @ np.asarray(checked.M, dtype=object)
    denominator = int(checked.N)

    if all(int(value) % denominator == 0 for value in numerator):
        return np.array(
            [int(value) // denominator for value in numerator],
            dtype=object,
        )

    if require_divisible:
        raise CrystallographyDivisibilityError(
            "scaled row image is not integer-valued under the supplied rotation."
        )

    return row_gcd_reduce_int(numerator)


def _reject_non_cubic_metric(metric: np.ndarray | None) -> None:
    """Reject non-cubic lattice metrics reserved for a later extension.

    ``metric`` is intended to represent a future 3 by 3 lattice metric tensor
    for non-cubic crystals. Exact CSL support is currently implemented only
    for the implicit cubic identity metric, so callers must pass ``None``.

    NOTE: This is a temporary guard, and it (and its companion method in `quaternion.py`
    and `plane.py`) should be centralized properly when fully implemented.
    """
    if metric is not None:
        raise CrystallographyNotImplementedError(
            "non-cubic lattice metrics are not implemented"
        )


__all__ = [
    "assert_scaled_rotation",
    "validate_scaled_rotation_matrix",
    "transpose_rotation_convention",
    "scaled_row_image",
]
