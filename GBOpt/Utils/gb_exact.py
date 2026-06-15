# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Exact-solver utilities for canonical P/Q bicrystal construction."""

import math

import numpy as np

from GBOpt.BoundarySpec import BoundaryEmbedding, BoundarySpecError, PQSpec


# ---------------------------------------------------------------------------
# Internal helpers for canonicalization
# ---------------------------------------------------------------------------

def _row_gcd_reduce(row: np.ndarray) -> np.ndarray:
    """Divide an integer-valued row by the GCD of its absolute components."""
    ints = np.round(row).astype(int)
    running_gcd = 0
    for component in ints:
        running_gcd = math.gcd(running_gcd, int(abs(component)))
    if running_gcd <= 1:
        return ints.astype(float)
    return (ints // running_gcd).astype(float)


def _assert_integer_rows(M: np.ndarray, name: str) -> None:
    """Raise BoundarySpecError if any row of M is not close to integer-valued."""
    for i, row in enumerate(M):
        if not np.allclose(row, np.round(row), atol=1e-9, rtol=0.0):
            raise BoundarySpecError(
                f"{name} row {i} {row} is not integer-valued. "
                "P/Q rows must be integer Miller indices."
            )


def _first_nonzero_sign(row: np.ndarray) -> int:
    """Return +1 if the first nonzero component of row is positive, -1 if negative, 0 if all zero."""
    for v in row:
        if v != 0:
            return 1 if v > 0 else -1
    return 0


def _gauss_reduce_2d(
    v1: np.ndarray, v2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Lagrange 2D lattice reduction using integer arithmetic; returns (shorter, longer).

    Inputs are expected to be integer-valued (as produced by _row_gcd_reduce).
    Integer dot products and integer rounding are used throughout to avoid
    floating-point precision loss for large Miller indices.

    .. note::
        Shortness is measured by the Euclidean inner product, which assumes an
        orthonormal basis. For non-cubic systems, pass vectors in Cartesian
        coordinates; passing raw Miller-index vectors in a non-cubic cell will
        give an incorrect reduced basis.
    """
    import warnings

    a = np.round(v1).astype(int)
    b = np.round(v2).astype(int)
    for _ in range(200):
        aa = int(np.dot(a, a))
        bb = int(np.dot(b, b))
        if bb < aa:
            a, b = b, a
            aa = bb
        if aa == 0:
            break
        ab = int(np.dot(a, b))
        t = (ab + aa // 2) // aa
        if t == 0:
            break
        b = b - t * a
    else:
        warnings.warn(
            "Gauss reduction did not converge in 200 iterations; result may be unreduced",
            stacklevel=3,
        )
    return a, b


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
    """
    # GCD-reduce each row
    rows = [_row_gcd_reduce(M[i]) for i in range(3)]

    # Gauss-reduce the in-plane rows (1 and 2), then GCD-reduce each.
    # The reduction may swap rows, which can flip the determinant.
    r1, r2 = _gauss_reduce_2d(rows[1], rows[2])
    rows[1] = _row_gcd_reduce(r1)
    rows[2] = _row_gcd_reduce(r2)

    # Deterministic in-plane row ordering: put the row with the larger
    # (norm_sq, canonical_sign_tuple) key into rows[1].  "Canonical sign"
    # negates a row whose first nonzero component is negative so that the lex
    # comparison is independent of sign convention.  Sorting here — before any
    # sign fixing — ensures equivalent in-plane bases that differ only by row
    # order produce the same canonical matrix.
    def _inplane_key(row: np.ndarray) -> tuple:
        rv = row.astype(int)
        if _first_nonzero_sign(rv) < 0:
            rv = -rv
        return (int(np.dot(rv, rv)), tuple(rv.tolist()))

    if _inplane_key(rows[1]) < _inplane_key(rows[2]):
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
    - rows 1-2 form a deterministic Gauss-reduced in-plane basis; the row
      with the larger (norm_sq, canonical-sign lex) key is placed in row 1
    - sign convention: first nonzero component of rows 0 and 1 is positive
    - equivalent inputs (same directions, different scalings, sign flips, or
      in-plane row ordering) canonicalize identically

    :param P: Row-wise orientation matrix for the left grain, shape (3, 3).
    :param Q: Row-wise orientation matrix for the right grain, shape (3, 3).
    :returns: ``(P_canon, Q_canon)`` — canonicalized orientation matrices.
    :raises BoundarySpecError: If any row of P or Q is not integer-valued, or
        if canonicalization produces a zero row.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    _assert_integer_rows(P, "P")
    _assert_integer_rows(Q, "Q")
    P_canon = _canonicalize_matrix(P)
    Q_canon = _canonicalize_matrix(Q)
    for name, M in [("P", P_canon), ("Q", Q_canon)]:
        if any(not np.any(row) for row in M):
            raise BoundarySpecError(
                f"Canonical {name} contains a zero row; check that input rows "
                "are nonzero integer Miller indices."
            )
    return P_canon, Q_canon


def pq_spec_to_embedding(spec: PQSpec) -> BoundaryEmbedding:
    """Convert a validated PQSpec to a BoundaryEmbedding.

    :param spec: A validated PQSpec.
    :returns: Canonical embedding with ``exact=True``, ``coherent=True``,
        ``source="pq"``. P and Q are in canonical form. R_left and R_right
        are derived by normalizing each row of canonical P and Q to unit length.
        Equivalent PQSpecs (differing only by row scaling, sign convention, or
        in-plane basis choice) always produce identical BoundaryEmbeddings.
    :raises BoundarySpecError: If P or Q rows are not integer-valued, produce
        a zero row after canonicalization, or do not form a proper rotation
        matrix after row-normalization.
    """
    P_canon, Q_canon = canonicalize_pq(
        np.asarray(spec.P, dtype=float),
        np.asarray(spec.Q, dtype=float),
    )
    R_left = P_canon / np.linalg.norm(P_canon, axis=1, keepdims=True)
    R_right = Q_canon / np.linalg.norm(Q_canon, axis=1, keepdims=True)
    for r_name, R in [("R_left", R_left), ("R_right", R_right)]:
        if not (np.allclose(R @ R.T, np.eye(3), atol=1e-10)
                and abs(np.linalg.det(R) - 1.0) < 1e-10):
            raise BoundarySpecError(
                f"{r_name} derived from P/Q is not a proper rotation matrix "
                "(R @ R.T != I or det != 1). Ensure P/Q rows are mutually "
                "orthogonal integer Miller directions."
            )
    return BoundaryEmbedding(
        P=P_canon,
        Q=Q_canon,
        R_left=R_left,
        R_right=R_right,
        exact=True,
        coherent=True,
        source="pq",
    )
