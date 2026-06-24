# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Exact integer supercell enumeration for ``GBMaker`` coherent-boundary construction.

This module converts canonical crystallographic orientation rows into integer supercell
matrices and enumerates conventional-cell origins inside repeated supercells. It is
``GBMaker``-facing glue, not core CSL/PQ/plane arithmetic.

TODO: Move this module into ``GBOpt.GBMaker`` when ``GBMaker`` is split into a package.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from GBOpt.crystallography.integer import (
    as_int_array,
    cross_int3,
    integer_adj3,
    integer_det3,
    row_gcd_reduce,
)
from GBOpt.crystallography.types import CrystallographyValueError


def _integer_membership(
    origin,
    adj_S: list,
    det_S: int,
    repeat_x: int,
    repeat_y: int,
    repeat_z: int,
) -> bool:
    """Return whether an integer conventional-cell origin lies inside a repeated
    supercell.

    Computes fractional supercell coordinates as integer numerators via ``origin @
    adj(S)``. If ``det(S)`` is negative, the numerators are sign-flipped before checking
    ``0 <= u_num[i] < repeat[i] * abs(det(S))`` for each axis.

    This helper supports both signs of ``det_S``; production callers currently guarantee
    ``det_S > 0``.

    :param origin: Integer 3-vector giving the candidate conventional-cell origin.
    :param adj_S: Adjugate of ``S`` as a 3 by 3 list-of-lists from ``integer_adj3``.
    :param det_S: Integer determinant of ``S`` from ``integer_det3``.
    :param repeat_x: Number of repeats along the boundary-normal direction.
    :param repeat_y: Number of repeats along the first in-plane direction.
    :param repeat_z: Number of repeats along the second in-plane direction.
    :return: ``True`` if ``origin`` lies inside the repeated supercell.
    """
    abs_det = abs(det_S)
    # u_num[j] = sum_k origin[k] * adj_S[k][j]   (row-vector @ matrix)
    u_num = [sum(int(origin[k]) * adj_S[k][j] for k in range(3)) for j in range(3)]
    if det_S < 0:
        u_num = [-u for u in u_num]
    return (
        0 <= u_num[0] < repeat_x * abs_det
        and 0 <= u_num[1] < repeat_y * abs_det
        and 0 <= u_num[2] < repeat_z * abs_det
    )


def build_supercell_matrix(P: np.ndarray) -> np.ndarray:
    """Build the integer supercell matrix ``S = [s0; s1; s2]`` from canonical ``P``.

    For a canonical orientation matrix ``P`` whose rows have already been GCD-reduced
    and made right-handed by ``canonicalize_pq``, ``s1 = P[1]``, ``s2 = P[2]``, and ``s0
    = P[0]``. This relies on ``P[0]`` equaling ``gcd_reduce(cross(P[1], P[2]))``.

    :param P: 3 by 3 canonical orientation matrix with integer-valued rows.
    :return: 3 by 3 integer ndarray ``S`` with rows ``[s0, s1, s2]``.
    :raises ValueError: If ``P`` cannot be converted to an exact 3 by 3 integer matrix,
        ``S`` is singular, or ``P[0]`` does not equal ``gcd_reduce(cross(P[1], P[2]))``.
    """
    try:
        supercell_obj = as_int_array(P, (3, 3), "P (supercell matrix)")
    except CrystallographyValueError as exc:
        raise ValueError(str(exc)) from exc

    supercell = np.asarray(supercell_obj, dtype=int)
    det_S = integer_det3(supercell)
    if det_S == 0:
        raise ValueError(
            f"Supercell matrix S derived from P is singular (det_S=0). "
            f"P = {supercell_obj.tolist()}. The in-plane rows P[1], P[2] must be "
            "linearly independent."
        )
    expected_s0 = row_gcd_reduce(
        np.array(cross_int3(supercell[1], supercell[2]), dtype=object)
    )
    if not np.array_equal(expected_s0.astype(int), supercell[0]):
        raise ValueError(
            f"P[0]={supercell[0].tolist()} does not equal gcd_reduce(cross(P[1], P[2]))"
            f"={expected_s0.tolist()}; P must be canonical and right-handed."
        )
    return supercell


def enumerate_supercell_origins(
    supercell: np.ndarray,
    repeat_x: int,
    repeat_y: int,
    repeat_z: int,
) -> np.ndarray:
    """Enumerate all integer conventional-cell origins inside the repeated supercell.

    The repeated supercell is spanned by ``repeat_x * s0``, ``repeat_y * s1``, and
    ``repeat_z * s2``. Candidates are drawn from the integer bounding box of the eight
    parallelepiped corners, padded by one lattice step. Membership is tested with
    ``_integer_membership``, so no floating-point selection is used.

    :param supercell: 3 by 3 integer supercell matrix with rows ``s0``, ``s1``, and
        ``s2``.
    :param repeat_x: Number of repeats along ``s0``.
    :param repeat_y: Number of repeats along ``s1``.
    :param repeat_z: Number of repeats along ``s2``.
    :return: Array of shape ``(N, 3)`` of accepted integer origins, where ``N ==
        repeat_x * repeat_y * repeat_z * abs(det(S))``.
    :raises ValueError: If ``supercell`` cannot be converted to an exact integer matrix,
        ``supercell`` is singular or has negative determinant, a repeat count is not a
        positive integer, or the accepted count does not match the expected value.
    """
    try:
        int_supercell = as_int_array(supercell, (3, 3), "S")
    except CrystallographyValueError as exc:
        raise ValueError(str(exc)) from exc

    for name, value in (
        ("repeat_x", repeat_x),
        ("repeat_y", repeat_y),
        ("repeat_z", repeat_z),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be a positive integer; got {value}.")
        if int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value}.")
    repeat_x, repeat_y, repeat_z = int(repeat_x), int(repeat_y), int(repeat_z)

    det_S = integer_det3(int_supercell)
    if det_S == 0:
        raise ValueError("enumerate_supercell_origins requires non-singular S.")
    if det_S < 0:
        raise ValueError(
            "S must have positive determinant; ensure P was produced by "
            "canonicalize_pq with right-handed orientation rows. "
            f"Got det(S)={det_S}, S={int_supercell.tolist()}."
        )
    adj_S = integer_adj3(int_supercell)

    # Bounding box from the 8 parallelepiped corners
    corners = np.array(
        [
            i * repeat_x * int_supercell[0]
            + j * repeat_y * int_supercell[1]
            + k * repeat_z * int_supercell[2]
            for i in (0, 1)
            for j in (0, 1)
            for k in (0, 1)
        ],
        dtype=object,
    )
    lower_bound = [int(corners[:, d].min()) - 1 for d in range(3)]
    upper_bound = [int(corners[:, d].max()) + 1 for d in range(3)]

    ranges = [range(lower_bound[d], upper_bound[d] + 1) for d in range(3)]
    candidates = list(product(*ranges))

    accepted = [
        tuple(row)
        for row in candidates
        if _integer_membership(row, adj_S, det_S, repeat_x, repeat_y, repeat_z)
    ]

    expected = repeat_x * repeat_y * repeat_z * abs(det_S)
    if len(accepted) != expected:
        raise ValueError(
            f"enumerate_supercell_origins: expected {expected} origins "
            f"(repeat={repeat_x},{repeat_y},{repeat_z}, |det|={abs(det_S)}) "
            f"but got {len(accepted)}.  supercell = {int_supercell.tolist()}"
        )
    return np.array(accepted, dtype=int)


def supercell_axis_numerators(
    supercell: np.ndarray,
    origins: np.ndarray,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Return exact supercell-coordinate numerators for integer origins.

    For a supercell matrix ``S``, an integer origin has reduced supercell coordinates
    ``u = origin @ inv(S)``. This function returns the selected numerator column of
    ``origin @ adj(S)``, leaving the common denominator ``det(S)`` implicit. Distinct
    numerator values identify fine integer layers along the selected supercell axis.

    :param supercell: 3 by 3 integer supercell matrix ``S``.
    :param origins: Integer conventional-cell origins with shape ``(N, 3)``.
    :param axis: Supercell coordinate axis to return. Must be 0, 1, or 2.
    :return: Integer numerator coordinates parallel to ``origins``.
    :raises ValueError: If inputs cannot be converted to exact integer arrays, if
        ``axis`` is invalid, or if ``supercell`` is singular.
    """
    int_supercell = as_int_array(supercell, (3, 3), "S")

    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis!r}.")
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis!r}.")

    origins_arr = np.asarray(origins, dtype=object)
    if origins_arr.ndim != 2 or origins_arr.shape[1] != 3:
        raise ValueError(f"origins must have shape (N, 3); got {origins_arr.shape}.")

    int_origins = as_int_array(origins_arr, origins_arr.shape, "origins")

    det_S = integer_det3(int_supercell)
    if det_S == 0:
        raise ValueError("supercell_axis_numerators requires non-singular S.")

    adj_S = np.array(integer_adj3(int_supercell), dtype=object)
    numerators = int_origins @ adj_S[:, axis]

    if det_S < 0:
        numerators = -numerators

    return np.asarray([int(value) for value in numerators], dtype=object)


__all__ = [
    "build_supercell_matrix",
    "enumerate_supercell_origins",
    "supercell_axis_numerators",
]
