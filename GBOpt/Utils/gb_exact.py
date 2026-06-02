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



def validate_and_normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    """Validate that quat is an integer quaternion and return its normalized form.

    A rotation quaternion encodes a rotation by angle θ about a unit axis
    n̂ = (nx, ny, nz) in Hamilton scalar-first order ``[w, x, y, z]``,
    where ``w = cos(θ/2)`` and ``(x, y, z) = sin(θ/2) · n̂``. For a CSL
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
    """Convert a unit quaternion [w, x, y, z] to a 3×3 rotation matrix.

    Delegates to ``scipy.spatial.transform.Rotation`` using scalar-last order
    internally; the reordering is handled here so callers always use Hamilton
    scalar-first order. The quaternion must already be normalized — call
    ``validate_and_normalize_quaternion`` first.

    :param quat: Normalized unit quaternion in Hamilton scalar-first order
        [w, x, y, z], shape (4,).
    :returns: Rotation matrix ``R`` of shape (3, 3) such that
        ``v_rotated = v @ R.T`` (row-vector convention).
    """
    from scipy.spatial.transform import Rotation

    q = np.asarray(quat, dtype=float)
    # scipy uses scalar-last order [x, y, z, w]
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def validate_sigma(quat: np.ndarray, sigma: int) -> None:
    """Validate that sigma derived from quat matches the user-supplied value.

    Sigma (Σ) is the reciprocal density of coincidence sites for a CSL grain
    boundary. For an integer quaternion ``q = [w, x, y, z]``, sigma is the
    odd part of ``N = w² + x² + y² + z²`` — that is, ``N`` divided by its
    largest power-of-2 factor. For example: ``q = [2, 0, 0, 1]`` gives
    ``N = 5``, so ``sigma = 5``; ``q = [3, 0, 0, 1]`` gives ``N = 10 = 2·5``,
    so ``sigma = 5``. Validation is an exact integer equality check.

    :param quat: Integer quaternion (unnormalized) in Hamilton order
        [w, x, y, z], shape (4,).
    :param sigma: Expected sigma value to validate against.
    :raises BoundarySpecError: If the derived sigma does not match.
    """
    int_q = np.round(np.asarray(quat, dtype=float)).astype(int)
    norm_sq = int(np.dot(int_q, int_q))
    if norm_sq == 0:
        raise BoundarySpecError(
            "Quaternion is zero; sigma cannot be derived from a zero quaternion."
        )
    # Divide out the largest power of 2 to obtain the odd sigma value.
    while norm_sq % 2 == 0:
        norm_sq //= 2
    derived = norm_sq
    if derived != int(sigma):
        raise BoundarySpecError(
            f"Sigma mismatch: quaternion {int_q.tolist()} gives "
            f"sigma={derived}, but sigma={sigma} was provided."
        )



def _recover_sigma_from_rotation(R: np.ndarray, max_sigma: int = 10001) -> int:
    """Return N such that N·R is an integer matrix (N = sigma or a multiple).

    For a rotation matrix produced from an integer quaternion [w,x,y,z] with
    norm-squared N = w²+x²+y²+z², every entry of R is a rational number whose
    denominator (in lowest terms) divides N.  We recover N as the LCM of the
    denominators of all matrix entries, using ``Fraction.limit_denominator`` to
    convert each floating-point entry to its exact rational form.

    This is O(9) — one Fraction call per matrix entry — rather than a linear
    search up to max_sigma.

    :param R: 3×3 rotation matrix whose entries are rationals with denominator ≤ max_sigma.
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


def _plane_null_basis(
    plane_int: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two linearly independent integer vectors orthogonal to plane_int.

    For a plane normal [h, k, l] the two basis vectors are cross products of
    [h,k,l] with the two axis-aligned unit vectors that are **not** the dominant
    axis.  Because the dominant component (largest |value|) of [h,k,l] is
    non-zero, it always survives in the cross-product result, guaranteeing
    neither basis vector is the zero vector.

    The three cases and their cross products are::

        dominant axis = x  →  e1 = [h,k,l]×ŷ = [-l, 0,  h]
                               e2 = [h,k,l]×ẑ = [ k,-h,  0]
        dominant axis = y  →  e1 = [h,k,l]×x̂ = [ 0, l, -k]
                               e2 = [h,k,l]×ẑ = [ k,-h,  0]
        dominant axis = z  →  e1 = [h,k,l]×x̂ = [ 0, l, -k]
                               e2 = [h,k,l]×ŷ = [-l, 0,  h]

    All results are exact integers.
    """
    h, k, l = int(plane_int[0]), int(plane_int[1]), int(plane_int[2])
    idx = int(np.argmax([abs(h), abs(k), abs(l)]))
    if idx == 0:
        e1 = np.array([-l,  0,  h])   # [h,k,l] × ŷ
        e2 = np.array([ k, -h,  0])   # [h,k,l] × ẑ
    elif idx == 1:
        e1 = np.array([ 0,  l, -k])   # [h,k,l] × x̂
        e2 = np.array([ k, -h,  0])   # [h,k,l] × ẑ
    else:
        e1 = np.array([ 0,  l, -k])   # [h,k,l] × x̂
        e2 = np.array([-l,  0,  h])   # [h,k,l] × ŷ
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
    shortest such vectors that lie in the boundary plane ``plane · v = 0``.

    The search works in the 2-D integer lattice spanned by the two null-space
    basis vectors of ``plane`` (see ``_plane_null_basis``).  Every candidate
    ``v = s·e1 + t·e2`` is tested against the CSL condition
    ``(v @ M_int) % N == 0`` where ``M_int = round(N·R)`` and N is the sigma
    value recovered from R.

    :param axis: Integer Miller rotation axis [u v w] (used for documentation
        only; the CSL condition is derived solely from R).
    :param plane: Integer Miller boundary-plane normal [h k l] in grain 1's
        crystal frame.
    :param R: Exact rotation matrix from ``quaternion_to_rotation_matrix``.
    :param max_exact_atoms: Guard on cell size.  Raises if the area of the
        in-plane CSL unit cell (``|v1 × v2|``) exceeds this value, which would
        produce an impractically large simulation cell.
    :return: ``(v1, v2)`` — two linearly independent in-plane CSL vectors,
        before Gauss reduction.  Pass to ``reduce_2d_basis`` to get the
        shortest basis.
    :raises BoundarySpecError: If fewer than two independent in-plane CSL
        vectors are found within the search range, or if the cell exceeds
        ``max_exact_atoms``.
    """
    plane_int = _row_gcd_reduce(np.round(plane).astype(int))
    N = _recover_sigma_from_rotation(R)
    M_int = np.round(N * R).astype(int)

    e1, e2 = _plane_null_basis(plane_int)

    # Search the 2-D in-plane integer lattice for CSL vectors.
    # The bound N+2 is sufficient: in-plane CSL vectors have components of
    # order sqrt(N) in the worst case, so s,t ≤ N covers all primitive vectors.
    search_range = max(N + 2, 10)
    candidates = []
    for s in range(-search_range, search_range + 1):
        for t in range(-search_range, search_range + 1):
            if s == 0 and t == 0:
                continue
            v = s * e1 + t * e2
            if np.all((v @ M_int) % N == 0):
                candidates.append(v.copy())

    if len(candidates) < 2:
        raise BoundarySpecError(
            "Could not find two independent in-plane CSL vectors for the given "
            f"plane={plane_int.tolist()} and rotation matrix.  Verify that the "
            "quaternion defines a valid CSL boundary for this plane."
        )

    # Pick the shortest vector, then the shortest independent from it.
    candidates.sort(key=lambda v: int(np.dot(v, v)))
    v1 = candidates[0]
    v2 = next(
        (c for c in candidates[1:] if np.any(np.cross(v1, c) != 0)),
        None,
    )
    if v2 is None:
        raise BoundarySpecError(
            "All in-plane CSL vectors found are collinear; could not form a 2-D basis."
        )

    area = np.linalg.norm(np.cross(v1, v2))
    if area > max_exact_atoms:
        raise BoundarySpecError(
            f"Exact in-plane CSL cell area ({area:.1f}) exceeds max_exact_atoms="
            f"{max_exact_atoms}.  Use mode='approximate' or increase the limit."
        )

    return v1.astype(float), v2.astype(float)


def reduce_2d_basis(
    v1: np.ndarray,
    v2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Gauss-Lagrange 2-D lattice reduction to a pair of in-plane vectors.

    Returns the shortest basis for the lattice spanned by v1 and v2, with the
    shorter vector first.  Delegates to the internal ``_gauss_reduce_2d`` helper.

    :param v1: First in-plane basis vector (integer-valued).
    :param v2: Second in-plane basis vector (integer-valued).
    :return: ``(r1, r2)`` — reduced basis, ``‖r1‖ ≤ ‖r2‖``.
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

    where ``M_int = round(N · R)`` and N is recovered from R.  This formula
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

    if spec.sigma is not None:
        validate_sigma(quat_arr, spec.sigma)

    R = quaternion_to_rotation_matrix(quat_norm)
    N = _recover_sigma_from_rotation(R)
    M_int = np.round(N * R).astype(int)

    # Validate that a CSL exists for this plane — raises if not.
    solve_inplane_csl(
        np.asarray(spec.axis, dtype=float),
        np.asarray(spec.plane, dtype=float),
        R,
        max_exact_atoms=max_exact_atoms,
    )

    # Build P: row 0 = boundary normal; rows 1–2 = orthogonal in-plane basis.
    # e1 is one null-basis vector (orthogonal to plane_int).
    # e2 = plane_int × e1 is then orthogonal to BOTH plane_int and e1, ensuring
    # the three rows form a proper rotation when normalized.
    plane_int = _row_gcd_reduce(np.round(np.asarray(spec.plane, dtype=float)).astype(int))
    e1, _ = _plane_null_basis(plane_int)
    e1 = _row_gcd_reduce(e1)
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
    # directions.  e2 = plane × e1 is orthogonal to both by construction.
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


