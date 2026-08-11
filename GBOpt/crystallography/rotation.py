# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Scaled-rotation validation and convention helpers.

Validates ``ScaledRotation`` objects and provides convention-shifting utilities. The
project row-vector convention is documented on ``ScaledRotation`` in types.py; functions
here enforce and convert between row and column conventions without duplicating that
documentation.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._guards import _require_cubic
from .integer import as_int_array, as_positive_int, integer_det3, row_gcd_reduce
from .types import (
    CrystallographyDivisibilityError,
    CrystallographyValueError,
    ScaledRotation,
)


def _minimal_integral_row_pair(
    direction: np.ndarray,
    row_rotation: ScaledRotation,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct one exactly paired integer row under a scaled rotation.

    ``row_rotation`` represents the row-vector mapping ``q = p @ M / N``. The supplied
    direction is first GCD-reduced to a primitive integer row. That primitive row is
    then multiplied by the smallest positive integer scale for which every component of
    its rotated image is divisible by ``N``. The returned rows therefore satisfy
    ``image_row == paired_row @ M / N`` exactly in integer arithmetic.

    :param direction: Integer-valued row direction. It may be non-primitive, but it must
        be one-dimensional and valid for :func:`row_gcd_reduce`.
    :param row_rotation: Valid exact row-convention scaled rotation ``M / N``.
    :return: ``(paired_row, image_row)`` as object-dtype integer arrays. ``paired_row``
        is the minimally scaled primitive input direction and ``image_row`` is its exact
        integral image under ``row_rotation``.
    :raises CrystallographyValueError: If ``direction`` is not a valid integer row or if
        the computed scale fails to produce an exactly divisible image, which indicates
        an internal inconsistency in the pairing calculation.
    """
    primitive = row_gcd_reduce(np.asarray(direction, dtype=object))
    numerator_matrix = np.asarray(row_rotation.matrix, dtype=object)
    denominator = int(row_rotation.denominator)
    image_numerator = primitive @ numerator_matrix

    common_divisor = math.gcd(
        denominator,
        *(abs(int(component)) for component in image_numerator),
    )
    scale = denominator // common_divisor
    paired_row = scale * primitive
    scaled_image = paired_row @ numerator_matrix

    if any(int(component) % denominator for component in scaled_image):
        raise CrystallographyValueError(
            "Internal error while constructing exact paired P/Q rows."
        )

    image_row = np.array(
        [int(component) // denominator for component in scaled_image],
        dtype=object,
    )
    return paired_row, image_row


def _scaled_row_images(
    rows: np.ndarray,
    rotation: ScaledRotation,
    *,
    allow_inexact: tuple[bool, bool, bool],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a row-convention scaled rotation to exactly three integer rows.

    This batched companion to ``scaled_row_image`` validates the rotation once and then
    applies ``row @ rotation.matrix / rotation.denominator`` to each input row. Exactly
    divisible images are returned directly. For an inexact image, the corresponding
    ``allow_inexact`` flag determines whether the primitive numerator direction is
    returned or an exception is raised.

    :param rows: 3 by 3 integer-valued row matrix whose rows are transformed
        independently.
    :param rotation: Row-convention scaled rotation to apply.
    :param allow_inexact: Three Boolean flags controlling whether each respective row
        may return a primitive numerator direction instead of an exactly divisible
        image. Keyword argument.
    :return: Three object-dtype integer arrays containing the transformed rows in input
        order.
    :raises CrystallographyValueError: If ``rows`` or ``rotation`` fails exact
        validation.
    :raises CrystallographyDivisibilityError: If a row image is not exactly
        integer-valued and its corresponding ``allow_inexact`` flag is ``False``.
    """
    int_rows = as_int_array(rows, (len(rows), 3), "rows")
    assert_scaled_rotation(rotation)

    matrix = np.asarray(rotation.matrix, dtype=object)
    denominator = int(rotation.denominator)

    images: list[np.ndarray] = []
    for row, row_allow_inexact in zip(int_rows, allow_inexact, strict=True):
        numerator = row @ matrix
        numerator_ints = [int(v) for v in numerator]

        if all(v % denominator == 0 for v in numerator_ints):
            images.append(
                np.array([v // denominator for v in numerator_ints], dtype=object)
            )
            continue

        if not row_allow_inexact:
            raise CrystallographyDivisibilityError(
                "Scaled row image is not integer-valued under the supplied rotation."
            )

        images.append(row_gcd_reduce(numerator))
    q1, q2, q3 = images
    return q1, q2, q3


def assert_scaled_rotation(rotation: ScaledRotation) -> None:
    """Raise if ``rotation`` fails exact scaled-rotation identities.

    Checks two necessary conditions for a proper scaled rotation ``M / N``:

    * ``M @ M.T == N**2 * I``
    * ``det(M) == N**3``

    Both checks use exact integer arithmetic.

    :param rotation: Scaled rotation to verify.
    :raises CrystallographyValueError: If ``M @ M.T != N**2 * I`` or
        ``det(M) != N**3``.
    """
    denom = rotation.denominator
    matrix = rotation.matrix

    denom_sq = denom * denom
    denom_cubed = denom_sq * denom

    gram = matrix @ matrix.T
    expected = denom_sq * np.eye(3, dtype=object)
    if not np.array_equal(gram, expected):
        raise CrystallographyValueError("Scaled rotation is not exactly orthogonal.")

    det = integer_det3(matrix)
    if det != denom_cubed:
        raise CrystallographyValueError(
            f"Scaled rotation determinant {det} does not equal denom^3={denom_cubed}."
        )


def validate_scaled_rotation_matrix(
    input_matrix: np.ndarray,
    *,
    denominator: int | None = None,
    source: Literal["matrix", "five_dof", "quaternion"] = "matrix",
    reduce_common_factor: bool = False,
    lattice_metric: np.ndarray | None = None,
) -> ScaledRotation:
    """Validate an integer scaled-rotation numerator and return a ``ScaledRotation``.

    The denominator ``N`` is derived as ``sqrt(diagonal(M @ M.T))`` when not supplied.
    The matrix must have equal positive diagonal entries and zero off-diagonal entries
    in its Gram product, and ``det(M)`` must equal ``N**3``.

    :param input_matrix: 3 by 3 integer numerator matrix ``M`` for a scaled rotation ``M
        / N``.
    :param denominator: Expected positive denominator ``N``. Keyword argument, optional,
        defaults to ``None``.
    :param source: Label stored on the returned ``ScaledRotation``. Keyword argument,
        optional, defaults to ``"matrix"``.
    :param reduce_common_factor: If ``True``, divide out the GCD of all matrix entries
        from both ``input_matrix`` and ``denominator`` before validation. Keyword
        argument, optional, defaults to ``False``.
    :param lattice_metric: Reserved non-cubic lattice metric hook; only ``None`` is
        currently supported. Keyword argument, optional, defaults to ``None``.
    :return: Validated ``ScaledRotation`` with ``source`` set and ``quaternion=None``.
    :raises CrystallographyValueError: If a common matrix factor does not divide the
        supplied denominator, the Gram product is not a positive scalar multiple of the
        identity, the Gram diagonal is not a perfect square, the determinant is
        inconsistent with the derived denominator, or the supplied denominator does not
        match the derived denominator.
    :raises CrystallographyNotImplementedError: If ``lattice_metric`` is not ``None``.
    """
    _require_cubic(lattice_metric)
    int_matrix = as_int_array(input_matrix, (3, 3), "input_matrix")
    expected_denom = (
        None
        if denominator is None
        else as_positive_int(denominator, "denominator")
    )

    if reduce_common_factor:
        gcd_value = math.gcd(*(abs(int(v)) for v in int_matrix.flat))
        if gcd_value > 1:
            if expected_denom is not None and expected_denom % gcd_value != 0:
                raise CrystallographyValueError(
                    "Common matrix factor does not divide the supplied denominator."
                )
            int_matrix = int_matrix // gcd_value
            if expected_denom is not None:
                expected_denom //= gcd_value

    gram = int_matrix @ int_matrix.T
    diag_value = int(gram[0, 0])
    if diag_value <= 0 or not np.array_equal(gram, diag_value * np.eye(3, dtype=object)):
        raise CrystallographyValueError(
            "int_matrix @ int_matrix.T is not a positive scalar multiple of the identity."
        )
    derived_denom = math.isqrt(diag_value)
    if derived_denom * derived_denom != diag_value:
        raise CrystallographyValueError(
            "int_matrix @ int_matrix.T diagonal is not a perfect square."
        )
    det = integer_det3(int_matrix)
    derived_denom_cubed = derived_denom ** 3
    if det != derived_denom_cubed:
        raise CrystallographyValueError(
            f"det(int_matrix)={det} does not equal denominator^3={derived_denom_cubed}."
        )
    if expected_denom is not None and expected_denom != derived_denom:
        raise CrystallographyValueError(
            f"Supplied denominator={expected_denom} does not match derived "
            f"denominator={derived_denom}."
        )
    return ScaledRotation(denominator=derived_denom, matrix=int_matrix, source=source)


def transpose_rotation_convention(rotation: ScaledRotation) -> ScaledRotation:
    """Return a new ``ScaledRotation`` in the transposed convention.

    The project convention stores the rotation as a row-vector multiplier,
    ``q_row = p_row @ M / N``. Column-vector CSL routines require ``M.T``. This function
    converts between the two by returning a validated ``ScaledRotation`` with numerator
    ``rotation.matrix.T`` and the same denominator and source.

    :param rotation: Row-convention scaled rotation to transpose.
    :return: Validated scaled rotation with numerator ``rotation.matrix.T``, the same
        denominator, and the same source.
    :raises CrystallographyValueError: If ``rotation`` does not satisfy the exact
        scaled-rotation orthogonality and determinant identities.
    """
    return validate_scaled_rotation_matrix(
        np.asarray(rotation.matrix, dtype=object).T,
        denominator=rotation.denominator,
        source=rotation.source,
    )


def scaled_row_image(
    row: ArrayLike,
    rotation: ScaledRotation,
    *,
    allow_inexact: bool = False,
    validate_rotation: bool = True,
) -> np.ndarray:
    """Apply a row-convention scaled rotation to an integer row vector.

    Computes ``row @ M / N`` exactly. When the result is integer-valued, it is returned
    directly. When it is not integer-valued, behavior depends on ``allow_inexact``.

    :param row: Integer row vector of length 3.
    :param rotation: Row-convention scaled rotation to apply.
    :param allow_inexact: If ``False``, raise when the image is not exactly
        integer-valued. If ``True``, GCD-reduce the integer numerator and return it as
        the primitive integer direction of the rational image. Keyword argument,
        optional, defaults to ``False``.
    :param validate_rotation: If ``True``, verify the exact scaled-rotation identities
        before applying the row image. Keyword argument, optional, defaults to ``True``.
    :return: Exact integer image row when division is exact, or a GCD-reduced primitive
        integer direction when ``allow_inexact=True`` and exact division fails. The
        returned array is object dtype.
    :raises CrystallographyDivisibilityError: If the image is not exactly integer-valued
        and ``allow_inexact=False``.
    :raises CrystallographyValueError: If ``row`` is not an exact integer vector of
        length three or, when ``validate_rotation`` is true, ``rotation`` fails the
        exact scaled-rotation identities.
    """
    if validate_rotation:
        assert_scaled_rotation(rotation)

    row_obj = as_int_array(row, (3,), "row")
    numerator = row_obj @ rotation.matrix
    denominator = rotation.denominator

    pairs = [divmod(int(value), denominator) for value in numerator]
    quotients = [quotient for quotient, _remainder in pairs]
    remainders = [remainder for _quotient, remainder in pairs]

    if all(remainder == 0 for remainder in remainders):
        return np.array(quotients, dtype=object)

    if not allow_inexact:
        raise CrystallographyDivisibilityError(
            "Scaled row image is not integer-valued under the supplied rotation."
        )

    # For an inexact rational image, the primitive integer direction is represented by
    # the reduced numerator; the denominator is a scalar.
    return row_gcd_reduce(numerator)


__all__ = [
    "assert_scaled_rotation",
    "validate_scaled_rotation_matrix",
    "transpose_rotation_convention",
    "scaled_row_image",
]
