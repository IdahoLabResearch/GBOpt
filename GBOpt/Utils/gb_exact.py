# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Exact-solver utilities for canonical P/Q bicrystal construction."""

import math
import numpy as np

from GBOpt.BoundarySpec import BoundaryEmbedding, BoundarySpecError, PQSpec
from GBOpt.Utils.exact_csl import (
    ExactCSLError,
    csl_from_scaled_rotation,
    inplane_basis_from_csl,
    integer_quaternion_from_unit,
    pq_from_csl_plane,
    quaternion_to_scaled_rotation,
    validate_scaled_rotation_matrix,
)


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



# ---------------------------------------------------------------------------
# Integer membership kernel for exact supercell construction
# ---------------------------------------------------------------------------

def _require_int_matrix(M, name: str) -> list:
    """Validate that every entry of a 3x3 matrix is integer-valued and return as a list-of-lists of Python ints.

    Raises ``ValueError`` if any entry deviates from an integer by more than 1e-9.

    :param M: 3x3 array-like.
    :param name: Name used in the error message.
    :return: 3x3 list-of-lists of Python int.
    :raises ValueError: If any entry is not integer-valued.
    """
    result = []
    for i in range(3):
        row = []
        for j in range(3):
            v = float(M[i][j])
            rounded = round(v)
            if abs(v - rounded) > 1e-9:
                raise ValueError(
                    f"{name}[{i}][{j}] = {v} is not integer-valued "
                    f"(deviation {abs(v - rounded):.2e} > 1e-9)."
                )
            row.append(int(rounded))
        result.append(row)
    return result


def _int_det3(M) -> int:
    """Compute the determinant of a 3x3 integer matrix using pure-Python-int arithmetic.

    :param M: 3x3 array-like with integer-valued entries.
    :return: Integer determinant.
    :raises ValueError: If any entry is not integer-valued.
    """
    a = _require_int_matrix(M, "M")
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def _int_adj3(M) -> list:
    """Compute the adjugate (transpose of the cofactor matrix) of a 3x3 integer matrix.

    Returns a Python list-of-lists so that ``n @ adj`` (where n is a
    3-element Python list of ints) stays in pure-integer arithmetic.

    :param M: 3x3 array-like with integer-valued entries.
    :return: 3x3 list-of-lists representing adj(M).
    :raises ValueError: If any entry is not integer-valued.
    """
    a = _require_int_matrix(M, "M")

    def _cofactor(ri, ci):
        rows = [r for r in range(3) if r != ri]
        cols = [c for c in range(3) if c != ci]
        minor = (a[rows[0]][cols[0]] * a[rows[1]][cols[1]]
                 - a[rows[0]][cols[1]] * a[rows[1]][cols[0]])
        return minor if (ri + ci) % 2 == 0 else -minor

    # adj[i][j] = cofactor(j, i)  — transpose of the cofactor matrix
    return [[_cofactor(j, i) for j in range(3)] for i in range(3)]


def _integer_membership(
    n,
    adj_S: list,
    det_S: int,
    repeat_x: int,
    repeat_y: int,
    repeat_z: int,
) -> bool:
    """Test whether integer conventional-cell origin *n* lies inside the repeated supercell.

    Fractional supercell coordinates are ``u = n @ S^-1 = (n @ adj(S)) / det(S)``.
    Origin *n* is accepted when ``0 <= u[i] < repeat[i]`` for each axis, which in
    integer arithmetic becomes ``0 <= u_num[i] < repeat[i] * |det(S)|``, where
    ``u_num = n @ adj(S)`` (sign-flipped when ``det(S) < 0`` so the inequality
    direction is preserved).

    The exact construction path always produces ``det(S) > 0``; the sign-flip
    branch supports the general helper for testing purposes.

    :param n: Integer 3-vector (conventional-cell coordinates of the origin).
    :param adj_S: Adjugate of S as a 3x3 list-of-lists (from ``_int_adj3``).
    :param det_S: Integer determinant of S (from ``_int_det3``).
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

    For a canonical (GCD-reduced, right-handed) orientation matrix P:
    - ``s1 = P[1]`` — in-plane period along lab y (integer Miller indices)
    - ``s2 = P[2]`` — in-plane period along lab z
    - ``s0 = gcd_reduce(cross(s1, s2))`` — boundary-normal stacking period

    For canonical right-handed P, ``cross(s1, s2)`` is parallel to ``P[0]``
    and after GCD reduction equals it exactly, so *S* = P.

    A clear error is raised if S is non-integer or singular (det = 0).

    :param P: 3x3 canonical orientation matrix (integer-valued rows).
    :return: 3x3 integer ndarray S with rows [s0, s1, s2].
    :raises BoundarySpecError: If P rows are not integer-valued.
    :raises ValueError: If the resulting S is singular (det = 0).
    """
    _assert_integer_rows(P, "P (supercell matrix)")
    S = np.round(P).astype(int)
    det = _int_det3(S)
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
    with ``_integer_membership`` — no floating-point selection is used.

    :param S: 3x3 integer supercell matrix (rows = s0, s1, s2).
    :param repeat_x: Number of repeats along s0.
    :param repeat_y: Number of repeats along s1.
    :param repeat_z: Number of repeats along s2.
    :return: Array of shape (N, 3) of accepted integer origins, where
        ``N == repeat_x * repeat_y * repeat_z * abs(det(S))``.
    :raises ValueError: If the accepted count does not match the expected value.
    """
    S_int = np.round(S).astype(int)
    s0 = S_int[0]
    s1 = S_int[1]
    s2 = S_int[2]
    det_S = _int_det3(S_int)
    adj_S = _int_adj3(S_int)

    # Bounding box from the 8 parallelepiped corners
    corners = np.array([
        i * repeat_x * s0 + j * repeat_y * s1 + k * repeat_z * s2
        for i in (0, 1) for j in (0, 1) for k in (0, 1)
    ], dtype=int)
    lo = corners.min(axis=0) - 1
    hi = corners.max(axis=0) + 1

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


def validate_and_normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    """Validate that quat is an integer quaternion and return its normalized form.

    A rotation quaternion encodes a rotation by angle theta about a unit axis
    n_hat = (nx, ny, nz) in Hamilton scalar-first order ``[w, x, y, z]``,
    where ``w = cos(theta/2)`` and ``(x, y, z) = sin(theta/2) * n_hat``. For a CSL
    grain boundary the rotation angle is rational, so the quaternion
    components can be exact integers; the unit quaternion is obtained by
    dividing by the norm.

    :param quat: Candidate integer quaternion in Hamilton order [w, x, y, z],
        shape (4,).
    :returns: Normalized (unit-length) quaternion as ``np.ndarray`` of shape
        (4,) with dtype float.
    :raises BoundarySpecError: If any component is non-integer or the
        quaternion is zero.
    """
    arr = np.asarray(quat, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != 4:
        raise BoundarySpecError(
            f"Quaternion must be a 1-D array of length 4; got shape {arr.shape}."
        )
    if not np.allclose(arr, np.round(arr), atol=1e-9, rtol=0.0):
        raise BoundarySpecError(
            f"Quaternion components must be integer-valued; got {arr}. "
            "CSLExactSpec requires an integer quaternion [a, b, c, d]."
        )
    int_q = np.round(arr).astype(int)
    norm_sq = int(np.dot(int_q, int_q))
    if norm_sq == 0:
        raise BoundarySpecError(
            "Quaternion is the zero vector; a non-zero integer quaternion is required."
        )
    return arr / np.sqrt(float(norm_sq))


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion [w, x, y, z] to a 3x3 rotation matrix.

    Delegates to ``scipy.spatial.transform.Rotation`` using scalar-last order
    internally; the reordering is handled here so callers always use Hamilton
    scalar-first order. The quaternion must already be normalized — call
    ``validate_and_normalize_quaternion`` first.

    :param quat: Normalized unit quaternion in Hamilton scalar-first order
        [w, x, y, z], shape (4,).
    :returns: Rotation matrix ``R`` of shape (3, 3) such that
        ``v_rotated = v @ R.T`` (row-vector convention).
    """
    q = np.asarray(quat, dtype=float)
    if q.shape != (4,):
        raise BoundarySpecError(
            f"Quaternion must be a 1-D array of length 4; got shape {q.shape}."
        )
    if np.allclose(q, np.round(q), atol=1e-9, rtol=0.0):
        q_int = tuple(int(v) for v in np.round(q).astype(int))
    else:
        try:
            q_int = integer_quaternion_from_unit(q)
        except ExactCSLError as exc:
            raise BoundarySpecError(str(exc)) from exc
    try:
        scaled = quaternion_to_scaled_rotation(q_int)
    except ExactCSLError as exc:
        raise BoundarySpecError(str(exc)) from exc
    return np.asarray(scaled.M, dtype=float) / float(scaled.N)


def validate_sigma(quat: np.ndarray, sigma: int) -> None:
    """Validate that sigma derived from quat matches the user-supplied value.

    Sigma (Sigma) is the reciprocal density of coincidence sites for a CSL grain
    boundary. For an integer quaternion ``q = [w, x, y, z]``, sigma is the
    odd part of ``N = w^2 + x^2 + y^2 + z^2`` — that is, ``N`` divided by its
    largest power-of-2 factor. For example: ``q = [2, 0, 0, 1]`` gives
    ``N = 5``, so ``sigma = 5``; ``q = [3, 0, 0, 1]`` gives ``N = 10 = 2*5``,
    so ``sigma = 5``. Validation is an exact integer equality check.

    :param quat: Integer quaternion (unnormalized) in Hamilton order
        [w, x, y, z], shape (4,).
    :param sigma: Expected sigma value to validate against.
    :raises BoundarySpecError: If the derived sigma does not match.
    """
    arr = np.asarray(quat, dtype=float)
    if arr.shape != (4,):
        raise BoundarySpecError(
            f"Quaternion must be a 1-D array of length 4; got shape {arr.shape}."
        )
    if not np.allclose(arr, np.round(arr), atol=1e-9, rtol=0.0):
        raise BoundarySpecError(
            f"Quaternion components must be integer-valued; got {arr}. "
            "CSLExactSpec requires an integer quaternion [a, b, c, d]."
        )
    int_q = np.round(arr).astype(int)
    if int(np.dot(int_q, int_q)) == 0:
        raise BoundarySpecError(
            "Quaternion is zero; sigma cannot be derived from a zero quaternion."
        )
    try:
        rot = quaternion_to_scaled_rotation(tuple(int(x) for x in int_q))
        csl = csl_from_scaled_rotation(rot)
    except ExactCSLError as exc:
        raise BoundarySpecError(str(exc)) from exc

    derived = csl.sigma
    if derived != int(sigma):
        raise BoundarySpecError(
            f"Sigma mismatch: quaternion {int_q.tolist()} gives "
            f"sigma={derived}, but sigma={sigma} was provided."
        )



def _recover_sigma_from_rotation(R: np.ndarray, max_sigma: int = 10001) -> int:
    """Return N such that N*R is an integer matrix (N = sigma or a multiple).

    For a rotation matrix produced from an integer quaternion [w,x,y,z] with
    norm-squared N = w^2+x^2+y^2+z^2, every entry of R is a rational number whose
    denominator (in lowest terms) divides N.  We recover N as the LCM of the
    denominators of all matrix entries, using ``Fraction.limit_denominator`` to
    convert each floating-point entry to its exact rational form.

    This is O(9) — one Fraction call per matrix entry — rather than a linear
    search up to max_sigma.

    :param R: 3x3 rotation matrix whose entries are rationals with denominator <= max_sigma.
    :param max_sigma: Upper bound passed to ``limit_denominator``; the search
        raises if the true denominator exceeds this value.
    :return: LCM of all entry denominators = the common scaling factor N.
    :raises BoundarySpecError: If any entry's denominator exceeds max_sigma, which
        indicates R was not produced from an integer quaternion with sigma < max_sigma.
    """
    from fractions import Fraction
    from math import lcm

    denom = 1
    for entry in R.flat:
        if abs(entry) > 1e-10:
            frac = Fraction(entry).limit_denominator(max_sigma)
            denom = lcm(denom, frac.denominator)
    if denom == 0:
        raise BoundarySpecError(
            "Could not recover sigma from rotation matrix (all entries near zero)."
        )
    return denom


def _ext_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return (g, x, y) with x*a + y*b = g = gcd(|a|, |b|)."""
    old_r, r = a, b
    old_s, s = 1, 0
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
    g = abs(old_r)
    x = old_s if old_r >= 0 else -old_s
    y = (g - x * a) // b if b else 0
    return g, x, y


def _plane_null_basis(
    plane_int: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a **primitive** integer basis for the null space of plane_int.

    Finds e1, e2 in Z^3 such that:

    * ``plane_int * e1 = 0`` and ``plane_int * e2 = 0`` (both in-plane), and
    * ``e1 x e2 = plane_int`` (the basis spans the *full* integer plane lattice,
      not a coarser sublattice).

    The construction applies unimodular column operations (Smith-Normal-Form
    style) to ``[h, k, l]`` until it becomes ``[g, 0, 0]`` with g = gcd = 1,
    tracking the transformations in V in GL₃(Z).  Because V is unimodular,
    columns 1 and 2 of V are exactly the primitive null vectors.

    The cross-product criterion ``e1 x e2 = ±plane_int`` ensures the 2-D
    integer lattice in the plane is covered without gaps — the previous
    axis-aligned cross-product formula could return vectors whose span had
    index > 1 (e.g. plane [5,2,3] gave index-5 sublattice, causing
    ``solve_inplane_csl`` to miss short CSL vectors).
    """
    vec = np.array(
        [int(plane_int[0]), int(plane_int[1]), int(plane_int[2])], dtype=int
    )
    V = np.eye(3, dtype=int)

    for i in range(2):
        # Move the first nonzero entry to position i (column pivot).
        nz = next((j for j in range(i, 3) if vec[j] != 0), None)
        if nz is None:
            break
        if nz != i:
            vec[[i, nz]] = vec[[nz, i]]
            V[:, [i, nz]] = V[:, [nz, i]]
        # Zero out vec[j] for every j > i via extended GCD.
        for j in range(i + 1, 3):
            if vec[j] == 0:
                continue
            g, a, b = _ext_gcd(int(vec[i]), int(vec[j]))
            c, d = int(vec[i]) // g, int(vec[j]) // g
            old_i, old_j = V[:, i].copy(), V[:, j].copy()
            V[:, i] = a * old_i + b * old_j        # new col i absorbs the gcd
            V[:, j] = -d * old_i + c * old_j       # new col j becomes 0
            vec[i] = g
            vec[j] = 0

    # After reduction vec == [1, 0, 0] (since gcd(h,k,l) = 1 for primitive plane).
    # V is unimodular, so [h,k,l] @ V[:,1:] == [0,0] and V[:,1:] is primitive.
    e1 = V[:, 1].astype(float)
    e2 = V[:, 2].astype(float)
    # Guarantee e1 x e2 = +plane_int (not −plane_int) for a consistent orientation.
    if np.dot(np.cross(e1, e2), plane_int) < 0:
        e2 = -e2
    return e1, e2


def solve_inplane_csl(
    axis: np.ndarray,
    plane: np.ndarray,
    R: np.ndarray,
    max_exact_atoms: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the exact in-plane CSL basis from rotation axis, boundary plane, and R.

    A CSL (Coincidence Site Lattice) vector is an integer lattice vector v of
    grain 1 that is also a lattice vector of grain 2, i.e. ``v @ R`` is an
    integer vector (row-vector convention).  This function finds the two
    shortest such vectors that lie in the boundary plane ``plane * v = 0``.

    The search works in the 2-D integer lattice spanned by the two null-space
    basis vectors of ``plane`` (see ``_plane_null_basis``).  Every candidate
    ``v = s*e1 + t*e2`` is tested against the CSL condition
    ``(v @ M_int) % N == 0`` where ``M_int = round(N*R)`` and N is the sigma
    value recovered from R.

    :param axis: Integer Miller rotation axis [u v w] (used for documentation
        only; the CSL condition is derived solely from R).
    :param plane: Integer Miller boundary-plane normal [h k l] in grain 1's
        crystal frame.
    :param R: Exact rotation matrix from ``quaternion_to_rotation_matrix``.
    :param max_exact_atoms: Guard on cell size.  Raises if the area of the
        in-plane CSL unit cell (``|v1 x v2|``) exceeds this value, which would
        produce an impractically large simulation cell.
    :return: ``(v1, v2)`` — two linearly independent in-plane CSL vectors,
        before Gauss reduction.  Pass to ``reduce_2d_basis`` to get the
        shortest basis.
    :raises BoundarySpecError: If fewer than two independent in-plane CSL
        vectors are found within the search range, or if the cell exceeds
        ``max_exact_atoms``.
    """
    plane_int = _row_gcd_reduce(np.round(plane).astype(int))
    try:
        N = _recover_sigma_from_rotation(R)
        M_int = np.round(N * np.asarray(R, dtype=float)).astype(int)
        row_rotation = validate_scaled_rotation_matrix(M_int.T, N=N)
        csl = csl_from_scaled_rotation(row_rotation)
        inplane = inplane_basis_from_csl(csl.basis_hnf, tuple(int(x) for x in plane_int))
    except ExactCSLError as exc:
        raise BoundarySpecError(str(exc)) from exc

    v1 = np.asarray(inplane.basis[:, 0], dtype=float)
    v2 = np.asarray(inplane.basis[:, 1], dtype=float)
    area = np.linalg.norm(np.cross(v1, v2))
    if area > max_exact_atoms:
        raise BoundarySpecError(
            f"Exact in-plane CSL cell area ({area:.1f}) exceeds max_exact_atoms="
            f"{max_exact_atoms}.  Use mode='approximate' or increase the limit."
        )
    return v1, v2


def reduce_2d_basis(
    v1: np.ndarray,
    v2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Gauss-Lagrange 2-D lattice reduction to a pair of in-plane vectors.

    Returns the shortest basis for the lattice spanned by v1 and v2, with the
    shorter vector first.  Delegates to the internal ``_gauss_reduce_2d`` helper.

    :param v1: First in-plane basis vector (integer-valued).
    :param v2: Second in-plane basis vector (integer-valued).
    :return: ``(r1, r2)`` — reduced basis, ``||r1|| <= ||r2||``.
    """
    r1, r2 = _gauss_reduce_2d(v1, v2)
    return r1.astype(float), r2.astype(float)


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


def csl_spec_to_embedding(spec, max_exact_atoms: int = 10_000) -> BoundaryEmbedding:
    """Convert a validated CSLExactSpec to a BoundaryEmbedding.

    **How P and Q are constructed.**  In GBMaker's convention each row of a
    grain's orientation matrix records which crystal Miller direction aligns
    with the corresponding lab axis: row 0 = lab x (boundary normal), row 1 =
    lab y, row 2 = lab z.

    For grain 1 we fix the boundary normal (``plane``) as row 0 and fill rows
    1–2 with the two cross-product null-basis vectors of that plane (see
    ``_plane_null_basis``).  For grain 2 each row is obtained by applying the
    misorientation matrix M_int to the corresponding integer row of P and
    GCD-reducing the result::

        Q[row i] = gcd_reduce(P[row i] @ M_int)

    where ``M_int = round(N * R)`` and N is recovered from R.  This formula
    is equivalent to rotating each lab axis from grain 1's crystal frame into
    grain 2's crystal frame — exactly what R_right encodes.  After
    ``canonicalize_pq`` the resulting matrices are identical to what a
    ``PQSpec`` with the same boundary would produce, enabling the cross-format
    round-trip test.

    :param spec: A ``CSLExactSpec`` instance (quat is required).
    :param max_exact_atoms: Passed to ``solve_inplane_csl`` as the cell-size
        guard.  Raises ``BoundarySpecError`` if the in-plane CSL cell would be
        larger than this.
    :return: ``BoundaryEmbedding`` with ``exact=True``, ``coherent=True``,
        ``source="csl"``.
    :raises BoundarySpecError: On invalid quaternion, sigma mismatch, missing
        CSL for the given plane, or cell too large.
    """
    if spec.quat is None:
        raise BoundarySpecError("CSLExactSpec.quat is required.")

    quat_arr = np.asarray(spec.quat, dtype=float)
    quat_norm = validate_and_normalize_quaternion(quat_arr)

    try:
        rot = quaternion_to_scaled_rotation(tuple(int(x) for x in np.round(quat_arr).astype(int)))
        csl = csl_from_scaled_rotation(rot)
    except ExactCSLError as exc:
        raise BoundarySpecError(str(exc)) from exc

    if spec.sigma is not None and csl.sigma != int(spec.sigma):
        raise BoundarySpecError(
            f"Sigma mismatch: quaternion {np.round(quat_arr).astype(int).tolist()} "
            f"gives sigma={csl.sigma}, but sigma={spec.sigma} was provided."
        )

    plane_int = _row_gcd_reduce(np.round(np.asarray(spec.plane, dtype=float)).astype(int))
    plane_col = np.asarray(plane_int, dtype=object)
    image = rot.M @ plane_col
    preserves_plane = (
        all(int(value) % rot.N == 0 for value in image)
        and np.array_equal(
            _row_gcd_reduce(np.array([int(value) // rot.N for value in image], dtype=int)),
            plane_int,
        )
    )
    if preserves_plane:
        try:
            inplane = inplane_basis_from_csl(csl.basis_hnf, tuple(int(x) for x in plane_int))
            P_raw, Q_raw = pq_from_csl_plane(rot, inplane)
            P_canon, Q_canon = canonicalize_pq(P_raw, Q_raw)
        except ExactCSLError as exc:
            raise BoundarySpecError(str(exc)) from exc
        R_exact = np.asarray(rot.M, dtype=float) / float(rot.N)
        return BoundaryEmbedding(
            P=P_canon,
            Q=Q_canon,
            R_left=R_exact,
            R_right=R_exact,
            exact=True,
            coherent=True,
            source="csl",
        )

    R = quaternion_to_rotation_matrix(quat_norm)
    N = _recover_sigma_from_rotation(R)
    M_int = np.round(N * R).astype(int)

    # Find the minimal in-plane CSL basis (raises if none exists or cell is too large).
    # Use the shortest CSL in-plane vector as e1 so the max_exact_atoms guard
    # applies to the same basis that P will actually use.
    v1, v2 = solve_inplane_csl(
        np.asarray(spec.axis, dtype=float),
        np.asarray(spec.plane, dtype=float),
        R,
        max_exact_atoms=max_exact_atoms,
    )
    r1, _r2 = reduce_2d_basis(v1, v2)
    e1 = _row_gcd_reduce(r1)
    # e2 = plane_int x e1 is orthogonal to both, keeping P rows mutually
    # orthogonal so the normalized rows form a proper rotation matrix.
    e2 = _row_gcd_reduce(np.cross(plane_int, e1).astype(int))
    P = np.array([
        plane_int.astype(float),
        e1.astype(float),
        e2.astype(float),
    ])

    # Build Q: rotate each lab axis from grain 1 into grain 2's crystal frame.
    Q = np.array([
        _row_gcd_reduce(P[i].astype(int) @ M_int).astype(float)
        for i in range(3)
    ])

    P_canon, Q_canon = canonicalize_pq(P, Q)

    # Re-check the cell size using the actual constructed matrices.
    # solve_inplane_csl guards |v1 x v2| (the CSL in-plane area), but e2 is
    # defined as plane x e1 rather than the second CSL vector, so det(P)
    # equals |plane|^2*|e1|^2, which can exceed the CSL area by a factor of
    # |plane|^2 for non-(100) boundaries.
    det_P = abs(_int_det3(np.round(P_canon).astype(int)))
    det_Q = abs(_int_det3(np.round(Q_canon).astype(int)))
    if max(det_P, det_Q) > max_exact_atoms:
        raise BoundarySpecError(
            f"CSL supercell exceeds max_exact_atoms={max_exact_atoms}: "
            f"|det(P)|={det_P}, |det(Q)|={det_Q}. "
            "Use mode='approximate' or increase the limit."
        )

    R_left = P_canon / np.linalg.norm(P_canon, axis=1, keepdims=True)
    R_right = Q_canon / np.linalg.norm(Q_canon, axis=1, keepdims=True)
    for r_name, Rm in [("R_left", R_left), ("R_right", R_right)]:
        if not (np.allclose(Rm @ Rm.T, np.eye(3), atol=1e-10)
                and abs(np.linalg.det(Rm) - 1.0) < 1e-10):
            raise BoundarySpecError(
                f"{r_name} derived from CSLExactSpec is not a proper rotation matrix "
                "(R @ R.T ≠ I or det ≠ 1). Check that axis, plane, and quat are "
                "mutually consistent."
            )

    return BoundaryEmbedding(
        P=P_canon,
        Q=Q_canon,
        R_left=R_left,
        R_right=R_right,
        exact=True,
        coherent=True,
        source="csl",
    )


def csl_approx_spec_to_embedding(spec) -> BoundaryEmbedding:
    """Convert a CSLApproxSpec to a BoundaryEmbedding using the approximate path.

    Constructs floating-point R_left and R_right from the given plane and
    axis/angle misorientation.  P and Q are set to None (no exact integer
    matrices are available).

    R_left is built so that its first row is the unit boundary-plane normal.
    The remaining two rows are completed via Gram-Schmidt using the two
    non-dominant axis-aligned unit vectors, giving a proper rotation.
    R_right = R_left @ R_mis, where R_mis is the rotation about the given
    axis by angle_deg.

    :param spec: A ``CSLApproxSpec`` instance.
    :return: ``BoundaryEmbedding`` with ``exact=False``, ``coherent=True``,
        ``source="csl"``.
    """
    from scipy.spatial.transform import Rotation

    plane = np.asarray(spec.plane, dtype=float)
    plane_unit = plane / np.linalg.norm(plane)

    axis = np.asarray(spec.axis, dtype=float)
    axis_unit = axis / np.linalg.norm(axis)
    angle_rad = float(spec.angle_deg) * np.pi / 180.0
    R_mis = Rotation.from_rotvec(axis_unit * angle_rad).as_matrix()

    # Build R_left: row 0 = plane unit normal; rows 1–2 = orthogonal in-plane
    # directions.  e2 = plane x e1 is orthogonal to both by construction.
    plane_int = _row_gcd_reduce(np.round(plane).astype(int))
    e1, _ = _plane_null_basis(plane_int)
    e1 = _row_gcd_reduce(e1)
    e2 = _row_gcd_reduce(np.cross(plane_int, e1).astype(int))
    e1_unit = e1.astype(float) / np.linalg.norm(e1)
    e2_unit = e2.astype(float) / np.linalg.norm(e2)

    R_left = np.array([plane_unit, e1_unit, e2_unit])
    R_right = R_left @ R_mis

    return BoundaryEmbedding(
        P=None,
        Q=None,
        R_left=R_left,
        R_right=R_right,
        exact=False,
        coherent=True,
        source="csl",
    )


