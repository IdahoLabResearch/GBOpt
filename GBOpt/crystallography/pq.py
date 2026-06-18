# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""P/Q matrix canonicalization and row-rotation recovery.

Provides canonical forms for grain orientation matrices and recovers exact
scaled rotations from paired P/Q row matrices. Does not import BoundaryEmbedding,
boundary specs, or embedding construction; those concerns belong in embedding.py
and boundary.py.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from GBOpt.Utils.integer_normal_forms import _dot_int

from .integer import assert_integer_rows, integer_adj3, integer_det3, row_gcd_reduce_int
from .reduction import gauss_reduce_2d, gauss_reduce_2d_paired
from .rotation import validate_scaled_rotation_matrix
from .types import CrystallographyValueError, ScaledRotation


def _first_nonzero_sign(row: np.ndarray) -> int:
    """Return the sign of the first nonzero component.

    :param row: One-dimensional row vector.
    :return: ``1`` for positive, ``-1`` for negative, or ``0`` for all-zero.
    """
    for v in row:
        if v != 0:
            return 1 if v > 0 else -1
    return 0


def _canonical_inplane_key(row: np.ndarray) -> tuple[int, tuple[int, ...]]:
    """Return the deterministic sort key for an in-plane orientation row.

    :param row: Integer-valued in-plane row.
    :return: ``(norm_squared, canonical_sign_tuple)`` used for row ordering.
    """
    row_int = np.asarray(row, dtype=int)
    if _first_nonzero_sign(row_int) < 0:
        row_int = -row_int
    return _dot_int(row_int, row_int), tuple(int(v) for v in row_int)


def _canonicalize_matrix(M: np.ndarray) -> np.ndarray:
    """Return the canonical form of a single 3x3 orientation matrix.

    Sign convention:
    - Row 0 (boundary normal): first nonzero component positive.
      Compensating negation of row 2 preserves the determinant.
    - Row 1 (first in-plane direction): first nonzero component positive.
      Compensating negation of row 2 preserves the determinant.
    - Row 2 sign: absorbs all compensating negations above, then a final
      determinant check ensures right-handedness.  Row 2 is never given
      an independent sign convention; its sign is fully derived.

    :param M: 3 by 3 integer-valued orientation matrix.
    :return: Canonical float matrix with GCD-reduced rows.
    :raises CrystallographyValueError: If row reduction rejects a row.
    """
    # GCD-reduce each row
    rows = [row_gcd_reduce_int(M[i]) for i in range(3)]

    # Gauss-reduce the in-plane rows (1 and 2), then GCD-reduce each.
    # The reduction may swap rows, which can flip the determinant.
    r1, r2 = gauss_reduce_2d(rows[1], rows[2])
    rows[1] = row_gcd_reduce_int(r1)
    rows[2] = row_gcd_reduce_int(r2)

    # Deterministic in-plane row ordering: put the row with the larger
    # (norm_sq, canonical_sign_tuple) key into rows[1].  "Canonical sign"
    # negates a row whose first nonzero component is negative so that the lex
    # comparison is independent of sign convention.  Sorting here -- before any
    # sign fixing -- ensures equivalent in-plane bases that differ only by row
    # order produce the same canonical matrix.
    if _canonical_inplane_key(rows[1]) < _canonical_inplane_key(rows[2]):
        rows[1], rows[2] = rows[2], rows[1]

    # Fix row 0 sign: negate rows 0 AND 2 together so det is unchanged.
    if _first_nonzero_sign(rows[0]) < 0:
        rows[0] = -rows[0]
        rows[2] = -rows[2]

    # Fix row 1 sign: negate rows 1 AND 2 together so det is unchanged.
    if _first_nonzero_sign(rows[1]) < 0:
        rows[1] = -rows[1]
        rows[2] = -rows[2]

    # Final right-handedness check using integer triple product to avoid
    # float comparison.  Gauss reduction may have swapped rows (flipping
    # det), and the two compensating negations above may have cancelled.
    det_sign = int(np.dot(
        rows[0].astype(int),
        np.cross(rows[1].astype(int), rows[2].astype(int)),
    ))
    if det_sign < 0:
        rows[2] = -rows[2]

    return np.array(rows, dtype=float)


def canonicalize_pq_paired(
    P: np.ndarray,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalize the orientation convention for an ordered P/Q bicrystal.

    P is treated as the reference grain and Q is transformed only by paired
    row swaps/sign flips so that P/Q row correspondence is preserved. This is
    not a canonical representative of the physical grain-boundary equivalence
    class (grain exchange, crystal symmetry, and translation equivalences are
    not handled).

    :param P: 3 by 3 integer-valued reference-grain rows.
    :param Q: 3 by 3 integer-valued rows paired with ``P``.
    :return: Canonical ``(P, Q)`` with row correspondence preserved.
    :raises CrystallographyValueError: If either input contains non-integer or zero rows.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    assert_integer_rows(P, "P")
    assert_integer_rows(Q, "Q")

    p_rows = [
        row_gcd_reduce_int(P[0]),
        np.round(P[1]).astype(int).astype(float),
        np.round(P[2]).astype(int).astype(float),
    ]
    q_rows = [
        row_gcd_reduce_int(Q[0]),
        np.round(Q[1]).astype(int).astype(float),
        np.round(Q[2]).astype(int).astype(float),
    ]

    p1, p2, q1, q2 = gauss_reduce_2d_paired(
        p_rows[1], p_rows[2], q_rows[1], q_rows[2]
    )
    # GCD-reduce each in-plane row independently after Gauss reduction,
    # matching _canonicalize_matrix, so scaled-but-equivalent inputs produce
    # identical canonical output (e.g. [2,0,0] -> [1,0,0]).  Independent
    # reduction is valid because direction indices carry no meaningful scaling.
    p_rows[1] = row_gcd_reduce_int(p1)
    p_rows[2] = row_gcd_reduce_int(p2)
    q_rows[1] = row_gcd_reduce_int(q1)
    q_rows[2] = row_gcd_reduce_int(q2)

    if _canonical_inplane_key(p_rows[1]) < _canonical_inplane_key(p_rows[2]):
        p_rows[1], p_rows[2] = p_rows[2], p_rows[1]
        q_rows[1], q_rows[2] = q_rows[2], q_rows[1]

    if _first_nonzero_sign(p_rows[0]) < 0:
        p_rows[0] = -p_rows[0]
        p_rows[2] = -p_rows[2]
        q_rows[0] = -q_rows[0]
        q_rows[2] = -q_rows[2]

    if _first_nonzero_sign(p_rows[1]) < 0:
        p_rows[1] = -p_rows[1]
        p_rows[2] = -p_rows[2]
        q_rows[1] = -q_rows[1]
        q_rows[2] = -q_rows[2]

    det_sign = int(np.dot(
        p_rows[0].astype(int),
        np.cross(p_rows[1].astype(int), p_rows[2].astype(int)),
    ))
    if det_sign < 0:
        p_rows[2] = -p_rows[2]
        q_rows[2] = -q_rows[2]

    P_canon = np.array(p_rows, dtype=float)
    Q_canon = np.array(q_rows, dtype=float)
    for name, M in [("P", P_canon), ("Q", Q_canon)]:
        if any(not np.any(row) for row in M):
            raise CrystallographyValueError(
                f"Canonical {name} contains a zero row; check that input rows "
                "are nonzero integer Miller indices."
            )
    return P_canon, Q_canon


def recover_exact_row_rotation_from_paired_pq(
    P: np.ndarray, Q: np.ndarray
) -> ScaledRotation:
    """Recover exact row-convention scaled rotation from paired P/Q rows.

    :param P: 3 by 3 integer P matrix whose rows define the reference grain.
    :param Q: 3 by 3 integer Q matrix paired row-by-row with ``P``.
    :return: Validated scaled rotation mapping P rows to Q rows.
    :raises CrystallographyValueError: If P is singular or the paired rows do not
        recover an exact proper rotation.
    """
    assert_integer_rows(P, "P")
    assert_integer_rows(Q, "Q")
    P_int = np.round(P).astype(object)
    Q_int = np.round(Q).astype(object)
    det_P = integer_det3(P_int)
    if det_P == 0:
        raise CrystallographyValueError(
            "Cannot recover rotation from singular P matrix.")
    adj_P = np.asarray(integer_adj3(P_int), dtype=object)
    numerator = adj_P @ Q_int
    denominator = int(det_P)
    if denominator < 0:
        numerator = -numerator
        denominator = -denominator

    fractions = [
        Fraction(int(value), denominator)
        for value in np.asarray(numerator, dtype=object).flat
    ]
    scale = 1
    for value in fractions:
        scale = math.lcm(scale, value.denominator)
    M = np.array(
        [int(value * scale) for value in fractions],
        dtype=object,
    ).reshape(3, 3)
    try:
        return validate_scaled_rotation_matrix(M, N=scale)
    except CrystallographyValueError as exc:
        raise CrystallographyValueError(
            "P/Q paired rows do not recover an exact proper rotation."
        ) from exc


def canonicalize_pq(
    P: np.ndarray,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical forms of the P and Q orientation matrices.

    Canonicalization rules:
    - rows must be integer-valued (within 1e-9 absolute tolerance, rtol=0)
    - each row is divided by the GCD of its absolute components
    - matrices are right-handed (positive determinant after normalization)
    - row 0 is the boundary normal
    - rows 1-2 form a deterministic Gauss-reduced in-plane basis; the row with the
        larger (norm_sq, canonical-sign lex) key is placed in row 1
    - sign convention: first nonzero component of rows 0 and 1 is positive
    - equivalent inputs (same directions, different scalings, sign flips, or in-plane
        row ordering) canonicalize identically

    :param P: Row-wise orientation matrix for the left grain, shape (3, 3).
    :param Q: Row-wise orientation matrix for the right grain, shape (3, 3).
    :returns: ``(P_canon, Q_canon)`` -- canonicalized orientation matrices.
    :raises CrystallographyValueError: If any row of P or Q is not integer-valued, or
        if canonicalization produces a zero row.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    assert_integer_rows(P, "P")
    assert_integer_rows(Q, "Q")
    P_canon = _canonicalize_matrix(P)
    Q_canon = _canonicalize_matrix(Q)
    for name, M in [("P", P_canon), ("Q", Q_canon)]:
        if any(not np.any(row) for row in M):
            raise CrystallographyValueError(
                f"Canonical {name} contains a zero row; check that input rows "
                "are nonzero integer Miller indices."
            )
    return P_canon, Q_canon


__all__ = [
    "canonicalize_pq",
    "canonicalize_pq_paired",
    "recover_exact_row_rotation_from_paired_pq"
]
