# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Exact integer supercell enumeration for  GBMaker's coherent boundary construction
path.

This module converts canonical crystallographic orientation rows into integer supercell
matrices and enumerates conventional-cell origins inside repeated supercells. It is
GBMaker-facing glue, not core CSL/PQ/plane arithmetic.

TODO: Move this module into GBOpt.GBMaker when GBMaker is split into a package.
"""

from __future__ import annotations

import numpy as np

from GBOpt.crystallography.integer import (
    assert_integer_rows,
    integer_adj3,
    integer_det3,
)
from GBOpt.Utils import integer_normal_forms as inf
from GBOpt.Utils.integer_normal_forms import ExactNormalFormError


def _integer_membership(
    n,
    adj_S: list,
    det_S: int,
    repeat_x: int,
    repeat_y: int,
    repeat_z: int,
) -> bool:
    """Test whether integer conventional-cell origin *n* lies inside the repeated
    supercell.

    Fractional supercell coordinates are ``u = n @ S^-1 = (n @ adj(S)) / det(S)``.
    Origin *n* is accepted when ``0 <= u[i] < repeat[i]`` for each axis, which in
    integer arithmetic becomes ``0 <= u_num[i] < repeat[i] * |det(S)|``, where
    ``u_num = n @ adj(S)`` (sign-flipped when ``det(S) < 0`` so the inequality
    direction is preserved).

    The exact construction path always produces ``det(S) > 0``; the sign-flip
    branch supports the general helper for testing purposes.

    :param n: Integer 3-vector (conventional-cell coordinates of the origin).
    :param adj_S: Adjugate of S as a 3x3 list-of-lists (from ``integer_adj3``).
    :param det_S: Integer determinant of S (from ``integer_det3``).
    :param repeat_x: Number of repeats along the x (boundary-normal) direction.
    :param repeat_y: Number of repeats along the y (in-plane) direction.
    :param repeat_z: Number of repeats along the z (in-plane) direction.
    :return: True if *n* is accepted.
    """
    abs_det = abs(det_S)
    # u_num[j] = sum_k n[k] * adj_S[k][j]   (row-vector @ matrix)
    u_num = [
        sum(int(n[k]) * adj_S[k][j] for k in range(3))
        for j in range(3)
    ]
    if det_S < 0:
        u_num = [-u for u in u_num]
    return (
        0 <= u_num[0] < repeat_x * abs_det
        and 0 <= u_num[1] < repeat_y * abs_det
        and 0 <= u_num[2] < repeat_z * abs_det
    )


def build_supercell_matrix(P: np.ndarray) -> np.ndarray:
    """Build the integer supercell matrix *S* = [s0; s1; s2] from canonical P.

    For a canonical orientation matrix P whose rows have already been
    GCD-reduced and made right-handed by ``canonicalize_pq``:
    - ``s1 = P[1]`` -- in-plane period along lab y (integer Miller indices)
    - ``s2 = P[2]`` -- in-plane period along lab z
    - ``s0 = gcd_reduce(cross(s1, s2))`` -- boundary-normal stacking period

    For canonical right-handed P, ``cross(s1, s2)`` is parallel to ``P[0]``
    and after GCD reduction equals it exactly, so *S* = P.

    A clear error is raised if S is non-integer or singular (det = 0).

    :param P: 3x3 canonical orientation matrix (integer-valued rows).
    :return: 3x3 integer ndarray S with rows [s0, s1, s2].
    :raises BoundarySpecError: If P rows are not integer-valued.
    :raises ValueError: If the resulting S is singular (det = 0).
    """
    assert_integer_rows(P, "P (supercell matrix)")
    S = np.round(P).astype(int)
    det = integer_det3(S)
    if det == 0:
        raise ValueError(
            f"Supercell matrix S derived from P is singular (det=0). "
            f"P = {P.tolist()}. The in-plane rows P[1], P[2] must be "
            "linearly independent."
        )
    return S


def enumerate_supercell_origins(
    S: np.ndarray,
    repeat_x: int,
    repeat_y: int,
    repeat_z: int,
) -> np.ndarray:
    """Enumerate all integer conventional-cell origins inside the repeated supercell.

    The repeated supercell is spanned by ``repeat_x*s0``, ``repeat_y*s1``,
    ``repeat_z*s2``.  Candidates are drawn from the integer bounding box of the
    8 parallelepiped corners, padded by one lattice step.  Membership is tested
    with ``_integer_membership`` -- no floating-point selection is used.

    :param S: 3x3 integer supercell matrix (rows = s0, s1, s2).
    :param repeat_x: Number of repeats along s0.
    :param repeat_y: Number of repeats along s1.
    :param repeat_z: Number of repeats along s2.
    :return: Array of shape (N, 3) of accepted integer origins, where
        ``N == repeat_x * repeat_y * repeat_z * abs(det(S))``.
    :raises ValueError: If ``S`` is singular, a repeat count is not positive,
        or the accepted count does not match the expected value.
    """
    try:
        S_int = inf._as_int_matrix(S, (3, 3), "S")
    except ExactNormalFormError as exc:
        raise ValueError(str(exc)) from exc
    repeats = []
    for name, value in (
        ("repeat_x", repeat_x),
        ("repeat_y", repeat_y),
        ("repeat_z", repeat_z),
    ):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a positive integer; got {value!r}.")
        if not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"{name} must be a positive integer; got {value!r}."
            )
        repeat = int(value)
        if repeat <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}.")
        repeats.append(repeat)
    repeat_x, repeat_y, repeat_z = repeats

    s0 = S_int[0]
    s1 = S_int[1]
    s2 = S_int[2]
    det_S = integer_det3(S_int)
    if det_S == 0:
        raise ValueError("enumerate_supercell_origins requires non-singular S.")
    adj_S = integer_adj3(S_int)

    # Bounding box from the 8 parallelepiped corners
    corners = np.array([
        i * repeat_x * s0 + j * repeat_y * s1 + k * repeat_z * s2
        for i in (0, 1) for j in (0, 1) for k in (0, 1)
    ], dtype=object)
    lo = np.array([int(value) for value in corners.min(axis=0) - 1])
    hi = np.array([int(value) for value in corners.max(axis=0) + 1])

    ranges = [np.arange(lo[d], hi[d] + 1) for d in range(3)]
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)

    accepted = [
        tuple(row)
        for row in grid
        if _integer_membership(row, adj_S, det_S, repeat_x, repeat_y, repeat_z)
    ]

    expected = repeat_x * repeat_y * repeat_z * abs(det_S)
    if len(accepted) != expected:
        raise ValueError(
            f"enumerate_supercell_origins: expected {expected} origins "
            f"(repeat={repeat_x},{repeat_y},{repeat_z}, |det|={abs(det_S)}) "
            f"but got {len(accepted)}.  S = {S_int.tolist()}"
        )
    return np.array(accepted, dtype=int)


__all__ = [
    "build_supercell_matrix",
    "enumerate_supercell_origins",
]
