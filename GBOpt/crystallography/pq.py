# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""P/Q matrix canonicalization and row-rotation recovery.

Provides canonical forms for grain orientation matrices and recovers exact scaled
rotations from paired P/Q row matrices. Does not import ``BoundaryEmbedding``, boundary
specs, or embedding construction; those concerns belong in embedding.py and boundary.py.
"""

from __future__ import annotations

import math

import numpy as np

from GBOpt.Utils.integer_linalg import cross_int3, dot_int

from .integer import (
    as_int_array,
    integer_adj3,
    integer_det3,
    row_gcd_reduce,
)
from .reduction import gauss_reduce_2d, gauss_reduce_2d_paired
from .rotation import validate_scaled_rotation_matrix
from .types import CrystallographyValueError, ScaledRotation


def _first_nonzero_sign(row: np.ndarray) -> int:
    """Return the sign of the first nonzero component.

    :param row: One-dimensional row vector.
    :return: ``1`` for positive, ``-1`` for negative, or ``0`` for all-zero.
    """
    return next((1 if v > 0 else -1 for v in row if v != 0), 0)


def _canonical_inplane_key(row: np.ndarray) -> tuple[int, tuple[int, ...]]:
    """Return the deterministic sort key for an in-plane orientation row.

    :param row: Integer-valued in-plane row.
    :return: ``(norm_squared, canonical_sign_tuple)`` used for row ordering. The row
        sign is normalized so that the lex comparison is independent of sign convention:
        ``[-1, 2, 0]`` and ``[1, -2, 0]`` produce the same key.
    """
    row_int = np.asarray(row, dtype=object)
    if _first_nonzero_sign(row_int) < 0:
        row_int = -row_int
    return dot_int(row_int, row_int), tuple(int(v) for v in row_int)


def _det_sign(rows: list[np.ndarray]) -> int:
    """Return the sign of the scalar triple product ``rows[0] . (rows[1] x rows[2])``.

    :param rows: Three integer-valued one-dimensional arrays: boundary normal, first
        in-plane direction, and second in-plane direction.
    :return: ``1``, ``-1``, or ``0`` for positive, negative, or zero triple product.
    """
    triple = dot_int(rows[0], cross_int3(rows[1], rows[2]))
    if triple > 0:
        return 1
    if triple < 0:
        return -1
    return 0


def _orient_rows_by_primary(
    primary_rows: list[np.ndarray],
    paired_rows: list[np.ndarray] | None = None,
) -> None:
    """Canonicalize in-plane row order, signs, and handedness in place.

    ``primary_rows`` determines all canonicalization decisions. The two in-plane rows
    are ordered deterministically using ``_canonical_inplane_key``; row 0 and row 1 are
    sign-normalized so their first nonzero components are positive; row 2 absorbs the
    compensating sign flips; and the final row-2 sign is adjusted so the primary matrix
    is right-handed.

    When ``paired_rows`` is supplied, the same row swaps and sign flips are applied to
    preserve row-by-row correspondence with ``primary_rows``. The paired rows do not
    influence ordering, sign, or determinant decisions.

    :param primary_rows: Three integer-valued row arrays used as the canonicalization
        authority. Mutated in place.
    :param paired_rows: Optional three integer-valued row arrays paired row-by-row with
        ``primary_rows``. When supplied, mutated in place with the same row swaps and
        sign flips. Keyword argument, optional, defaults to ``None``.
    :return: ``None``. The supplied row lists are modified in place.
    """
    def swap_rows(i: int, j: int) -> None:
        """Swap two primary rows and their paired rows, if present.

        :param i: First row index to swap.
        :param j: Second row index to swap.
        """
        primary_rows[i], primary_rows[j] = primary_rows[j], primary_rows[i]
        if paired_rows is not None:
            paired_rows[i], paired_rows[j] = paired_rows[j], paired_rows[i]

    def flip_rows(*indices: int) -> None:
        """Flip the sign of selected primary rows and their paired rows, if present.

        :param indices: Row indices whose signs should be flipped.
        """
        for index in indices:
            primary_rows[index] = -primary_rows[index]
            if paired_rows is not None:
                paired_rows[index] = -paired_rows[index]

    if _canonical_inplane_key(primary_rows[1]) < _canonical_inplane_key(primary_rows[2]):
        swap_rows(1, 2)

    if _first_nonzero_sign(primary_rows[0]) < 0:
        flip_rows(0, 2)

    if _first_nonzero_sign(primary_rows[1]) < 0:
        flip_rows(1, 2)

    if _det_sign(primary_rows) < 0:
        flip_rows(2)


def _canonicalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return the canonical form of a single 3 by 3 orientation matrix.

    Sign convention:

    - Row 0, the boundary normal, has positive first nonzero component. Row 2 absorbs
      the compensating negation so the determinant is preserved.
    - Row 1, the first in-plane direction, has positive first nonzero component. Row 2
      again absorbs the compensating negation.
    - Row 2 has no independent sign convention; its sign is fully derived, followed by a
      final determinant check to ensure right-handedness.

    :param matrix: 3 by 3 integer-valued orientation matrix.
    :return: Canonical object-dtype matrix with GCD-reduced rows.
    """
    rows = [row_gcd_reduce(matrix[i]) for i in range(3)]

    r1, r2 = gauss_reduce_2d(rows[1], rows[2])
    rows[1] = row_gcd_reduce(r1)
    rows[2] = row_gcd_reduce(r2)

    _orient_rows_by_primary(rows)

    return np.array(rows, dtype=object)


def _assert_no_zero_rows(P: np.ndarray, Q: np.ndarray) -> None:
    """Raise if either canonical orientation matrix contains a zero row.

    :param P: Canonical left-grain orientation matrix.
    :param Q: Canonical right-grain orientation matrix.
    :raises CrystallographyValueError: If any row of ``P`` or ``Q`` is the zero vector.
    """
    for name, matrix in (("P", P), ("Q", Q)):
        zero_rows = ~np.any(matrix, axis=1)
        if np.any(zero_rows):
            raise CrystallographyValueError(
                f"Canonical {name} contains a zero row; check that input rows "
                "are nonzero integer Miller indices."
            )


def canonicalize_pq(
    P: np.ndarray,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical forms of the ``P`` and ``Q`` orientation matrices.

    Canonicalization rules: rows must be integer-valued, each row is divided by the GCD
    of its absolute components, matrices are made right-handed, row 0 is the boundary
    normal, rows 1-2 form a deterministic Gauss-reduced in-plane basis, row 1 receives
    the larger canonical key, and first nonzero components of rows 0 and 1 are positive.

    :param P: Row-wise orientation matrix for the left grain, shape ``(3, 3)``.
    :param Q: Row-wise orientation matrix for the right grain, shape ``(3, 3)``.
    :return: ``(P_canon, Q_canon)``, canonicalized orientation matrices as object-dtype
        integer matrices.
    """
    P_int = as_int_array(P, (3, 3), "P")
    Q_int = as_int_array(Q, (3, 3), "Q")

    P_canon, Q_canon = (_canonicalize_matrix(M) for M in (P_int, Q_int))
    _assert_no_zero_rows(P_canon, Q_canon)

    return P_canon, Q_canon


def canonicalize_pq_paired(
    P: np.ndarray,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalize orientation rows for a paired ``P``/``Q`` bicrystal.

    Applies the same row operations, including GCD reduction, Gauss reduction, ordering,
    and sign fixing, to both ``P`` and ``Q`` simultaneously while preserving row-by-row
    correspondence. Two ``P``/``Q`` pairs that represent the same boundary with
    different row scalings, sign conventions, or in-plane basis orderings produce
    identical output.

    This is not a canonical representative of the physical grain-boundary equivalence
    class: grain exchange, crystal symmetry, and translation equivalences are not
    resolved.

    :param P: 3 by 3 integer-valued reference-grain rows.
    :param Q: 3 by 3 integer-valued rows paired with ``P``.
    :return: Canonical ``(P, Q)`` with row correspondence preserved as object-dtype
        integer matrices.
    """
    P_int = as_int_array(P, (3, 3), "P")
    Q_int = as_int_array(Q, (3, 3), "Q")

    # In-plane rows are reduced as a paired lattice operation so Q follows P's row ops.
    p_rows = [row_gcd_reduce(P_int[0]), P_int[1], P_int[2]]
    q_rows = [row_gcd_reduce(Q_int[0]), Q_int[1], Q_int[2]]

    p1, p2, q1, q2 = gauss_reduce_2d_paired(p_rows[1], p_rows[2], q_rows[1], q_rows[2])
    # GCD-reduce each in-plane row independently after Gauss reduction, matching
    # _canonicalize_matrix, so scaled-but-equivalent inputs produce identical canonical
    # output (e.g. [2,0,0] -> [1,0,0]).  Independent reduction is valid because
    # direction indices carry no meaningful scaling.
    p_rows[1] = row_gcd_reduce(p1)
    p_rows[2] = row_gcd_reduce(p2)
    q_rows[1] = row_gcd_reduce(q1)
    q_rows[2] = row_gcd_reduce(q2)

    # p_rows[1] is the larger key, or the larger in-plane vector.
    _orient_rows_by_primary(p_rows, q_rows)

    P_canon = as_int_array(p_rows, (3, 3), "P")
    Q_canon = as_int_array(q_rows, (3, 3), "Q")
    _assert_no_zero_rows(P_canon, Q_canon)

    return P_canon, Q_canon


def recover_exact_row_rotation_from_paired_pq(
    P: np.ndarray, Q: np.ndarray
) -> ScaledRotation:
    """Recover an exact row-convention scaled rotation from paired ``P``/``Q`` rows.

    The rotation is recovered as ``R = inv(P) @ Q``, computed exactly as ``adj(P) @ Q /
    det(P)`` to avoid floating-point inversion. The rational entries are rescaled by
    their least-common-multiple denominator to yield an integer numerator matrix and
    positive integer denominator, which are passed to
    ``validate_scaled_rotation_matrix``.

    :param P: 3 by 3 integer ``P`` matrix whose rows define the reference grain.
    :param Q: 3 by 3 integer ``Q`` matrix paired row-by-row with ``P``.
    :return: Validated scaled rotation mapping ``P`` rows to ``Q`` rows.
    :raises CrystallographyValueError: If ``P`` is singular or the paired rows do not
        recover an exact proper rotation.
    """
    P_int = as_int_array(P, (3, 3), "P")
    Q_int = as_int_array(Q, (3, 3), "Q")
    det_P = integer_det3(P_int)
    if det_P == 0:
        raise CrystallographyValueError(
            "Cannot recover rotation from singular P matrix."
        )
    adj_P = np.asarray(integer_adj3(P_int), dtype=object)
    numerator = adj_P @ Q_int
    denominator = int(det_P)
    if denominator < 0:
        numerator = -numerator
        denominator = -denominator

    entry_dens = [
        denominator // math.gcd(abs(int(value)), denominator)
        for value in numerator.flat
    ]
    scale = math.lcm(*entry_dens)

    numerator_matrix = np.array(
        [int(value) * scale // denominator for value in numerator.flat],
        dtype=object,
    ).reshape(3, 3)

    try:
        return validate_scaled_rotation_matrix(numerator_matrix, denominator=scale)
    except CrystallographyValueError as exc:
        raise CrystallographyValueError(
            "P/Q paired rows do not recover an exact proper rotation."
        ) from exc


__all__ = [
    "canonicalize_pq",
    "canonicalize_pq_paired",
    "recover_exact_row_rotation_from_paired_pq",
]
