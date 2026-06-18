# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Exact integer normal-form helpers for small CSL matrices."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


class ExactNormalFormError(Exception):
    """Base exception for integer normal-form computation failures."""


@dataclass(frozen=True)
class SmithNormalForm:
    """Smith normal-form decomposition ``U @ A @ V == D``.

    A dataclass rather than a plain tuple so callers can use named fields
    ``U``, ``D``, and ``V`` unambiguously.
    """

    U: np.ndarray
    D: np.ndarray
    V: np.ndarray


def _as_int_matrix(A: ArrayLike, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Return *A* as a shape-checked object array of Python integers.

    Accepts any shape, not just 2D: pass ``shape=(n,)`` for a 1D vector.
    """
    arr = np.asarray(A)
    if arr.shape != shape:
        raise ExactNormalFormError(f"{name} must have shape {shape}; got {arr.shape}.")
    out = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        value = arr[index]
        if isinstance(value, (bool, np.bool_)):
            raise ExactNormalFormError(
                f"{name}{index}={value!r} is not an integer."
            )
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ExactNormalFormError(
                f"{name}{index}={value!r} is not an integer."
            ) from exc
        if value != integer:
            raise ExactNormalFormError(
                f"{name}{index}={value!r} is not exactly integer-valued."
            )
        out[index] = integer
    return out


def _identity(n: int) -> np.ndarray:
    """Return an ``n`` by ``n`` identity matrix with Python-int entries.

    Uses ``dtype=object`` so matrix products remain exact Python integers
    and never silently overflow ``int64``.
    """
    return np.eye(n, dtype=object)


def _row_gcd_reduce(row: ArrayLike) -> np.ndarray:
    """Divide an integer-valued row by the GCD of its absolute components.

    Returns an ``object``-dtype array of Python integers so the result can be
    cast by callers to any desired numeric type (``.astype(float)``, etc.).
    Rows that are already primitive (gcd == 1) or all-zero are returned unchanged.
    """
    arr = np.asarray(row)
    if arr.ndim != 1:
        raise ExactNormalFormError(
            f"row must be a 1D integer-valued array; got shape {arr.shape}."
        )
    ints = []
    for index, value in enumerate(arr):
        if isinstance(value, (bool, np.bool_)):
            raise ExactNormalFormError(
                f"row[{index}]={value!r} is not an integer."
            )
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ExactNormalFormError(
                f"row[{index}]={value!r} is not an integer."
            ) from exc
        if value != integer:
            raise ExactNormalFormError(
                f"row[{index}]={value!r} is not exactly integer-valued."
            )
        ints.append(integer)
    gcd = 0
    for v in ints:
        gcd = math.gcd(gcd, abs(v))
    if gcd <= 1:
        return np.array(ints, dtype=object)
    return np.array([v // gcd for v in ints], dtype=object)


def _dot_int(x: ArrayLike, y: ArrayLike) -> int:
    """Return the exact integer dot product of two equal-length vectors."""
    x_vals = [int(value) for value in np.asarray(x).flat]
    y_vals = [int(value) for value in np.asarray(y).flat]
    if len(x_vals) != len(y_vals):
        raise ExactNormalFormError(
            f"dot product requires equal lengths; got {len(x_vals)} and {len(y_vals)}."
        )
    return sum(xi * yi for xi, yi in zip(x_vals, y_vals))


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return ``(g, x, y)`` with ``x*a + y*b == g == gcd(abs(a), abs(b))``.

    Operates directly on signed inputs so Python's floor-division semantics
    handle negative values correctly without independent sign fixes for x and y.
    A single guard at the end ensures the returned gcd is non-negative.
    """
    old_r, r = int(a), int(b)
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    if old_r < 0:
        old_r, old_s, old_t = -old_r, -old_s, -old_t
    return old_r, old_s, old_t


def _elimination_transform(a: int, b: int) -> tuple[int, int, int, int]:
    """Return coefficients that combine a pivot vector with one to eliminate.

    The returned ``(p0, s0, p1, s1)`` maps
    ``primary, secondary`` to
    ``p0 * primary + s0 * secondary, p1 * primary + s1 * secondary``.
    For entries ``a`` and ``b`` at the active coordinate, the transformed
    secondary entry is zero while the primary entry becomes ``gcd(a, b)``.
    """
    if a == 0:
        raise ExactNormalFormError("cannot eliminate with a zero pivot.")
    if b % a == 0:
        return 1, 0, -(b // a), 1
    gcd_value, primary_coeff, secondary_coeff = _extended_gcd(a, b)
    return (
        primary_coeff,
        secondary_coeff,
        -b // gcd_value,
        a // gcd_value,
    )


def _cross_int3(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Return the exact Python-int cross product of two length-3 vectors."""
    x_vals = [int(value) for value in np.asarray(x, dtype=object).flat]
    y_vals = [int(value) for value in np.asarray(y, dtype=object).flat]
    if len(x_vals) != 3 or len(y_vals) != 3:
        raise ExactNormalFormError(
            f"cross product requires length-3 vectors; got {len(x_vals)} "
            f"and {len(y_vals)}."
        )
    return np.array(
        [
            x_vals[1] * y_vals[2] - x_vals[2] * y_vals[1],
            x_vals[2] * y_vals[0] - x_vals[0] * y_vals[2],
            x_vals[0] * y_vals[1] - x_vals[1] * y_vals[0],
        ],
        dtype=object,
    )


def primitive_integer_null_basis_3d(covector: ArrayLike) -> np.ndarray:
    """Return primitive integer columns spanning ``covector @ x == 0``.

    The returned 3 by 2 matrix has columns ``e1`` and ``e2``.  They are
    generated by unimodular column operations, so they span the full integer
    null lattice, not a sublattice.  The orientation is chosen so
    ``cross(e1, e2)`` points along ``covector``.
    """
    vec = _as_int_matrix(covector, (3,), "covector")
    if not any(int(value) for value in vec):
        raise ExactNormalFormError("covector must not be the zero vector.")

    reduced = vec.copy()
    transform = _identity(3)
    for i in range(2):
        nonzero = next((j for j in range(i, 3) if reduced[j] != 0), None)
        if nonzero is None:
            break
        if nonzero != i:
            reduced[[i, nonzero]] = reduced[[nonzero, i]]
            transform[:, [i, nonzero]] = transform[:, [nonzero, i]]
        for j in range(i + 1, 3):
            if reduced[j] == 0:
                continue
            g, a, b = _extended_gcd(int(reduced[i]), int(reduced[j]))
            c = int(reduced[i]) // g
            d = int(reduced[j]) // g
            old_i = transform[:, i].copy()
            old_j = transform[:, j].copy()
            transform[:, i] = a * old_i + b * old_j
            transform[:, j] = -d * old_i + c * old_j
            reduced[i] = g
            reduced[j] = 0

    basis = transform[:, 1:3].astype(object)
    cross = _cross_int3(basis[:, 0], basis[:, 1])
    if _dot_int(cross, vec) < 0:
        basis[:, 1] = -basis[:, 1]
    return basis


def _int_det3(A: np.ndarray) -> int:
    """Return the exact determinant of a 3 by 3 integer matrix."""
    return int(
        A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
        - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
        + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
    )


def _int_adj3(M: ArrayLike) -> list:
    """Compute the adjugate (transpose of cofactor matrix) of a 3 by 3 integer matrix.

    Returns a Python list-of-lists so that ``n @ adj`` (where n is a 3-element
    list of Python ints) stays in pure-integer arithmetic.

    :param M: 3 by 3 array-like with integer-valued entries.
    :return: 3 by 3 list-of-lists representing adj(M).
    :raises ExactNormalFormError: If any entry is not integer-valued.
    """
    a = _as_int_matrix(M, (3, 3), "M")

    def _cofactor(ri: int, ci: int) -> int:
        rows = [r for r in range(3) if r != ri]
        cols = [c for c in range(3) if c != ci]
        minor = int(a[rows[0], cols[0]]) * int(a[rows[1], cols[1]]) - \
            int(a[rows[0], cols[1]]) * int(a[rows[1], cols[0]])
        return minor if (ri + ci) % 2 == 0 else -minor

    # adj[i][j] = cofactor(j, i)  -- transpose of the cofactor matrix
    return [[_cofactor(j, i) for j in range(3)] for i in range(3)]


def _select_pivot(A: np.ndarray, start: int) -> tuple[int, int] | None:
    """Return the smallest nonzero pivot position in ``A[start:, start:]``.

    Hard-coded for 3 by 3 matrices.
    """
    best: tuple[int, int, int] | None = None
    for i in range(start, 3):
        for j in range(start, 3):
            if A[i, j] != 0:
                value = abs(int(A[i, j]))
                if best is None or value < best[0]:
                    best = (value, i, j)
    if best is None:
        return None
    return best[1], best[2]


def smith_normal_form_3x3(A: ArrayLike) -> SmithNormalForm:
    """Return exact ``U, D, V`` with ``U @ A @ V == D`` for a 3 by 3 matrix.

    The implementation tracks every Euclidean row and column operation with
    Python integers.  No fixed-width NumPy integer arithmetic is used.
    """
    original = _as_int_matrix(A, (3, 3), "A")
    D = original.copy()
    U = _identity(3)
    V = _identity(3)

    def swap_rows(i: int, j: int) -> None:
        if i == j:
            return
        D[[i, j], :] = D[[j, i], :]
        U[[i, j], :] = U[[j, i], :]

    def swap_cols(i: int, j: int) -> None:
        if i == j:
            return
        D[:, [i, j]] = D[:, [j, i]]
        V[:, [i, j]] = V[:, [j, i]]

    for k in range(3):
        pivot = _select_pivot(D, k)
        if pivot is None:
            break
        swap_rows(k, pivot[0])
        swap_cols(k, pivot[1])

        while True:
            changed = True
            while changed:
                changed = False

                for i in range(3):
                    if i == k or D[i, k] == 0:
                        continue
                    a = int(D[k, k])
                    b = int(D[i, k])
                    p0, s0, p1, s1 = _elimination_transform(a, b)
                    row_k = D[k, :].copy()
                    row_i = D[i, :].copy()
                    u_k = U[k, :].copy()
                    u_i = U[i, :].copy()
                    D[k, :] = p0 * row_k + s0 * row_i
                    D[i, :] = p1 * row_k + s1 * row_i
                    U[k, :] = p0 * u_k + s0 * u_i
                    U[i, :] = p1 * u_k + s1 * u_i
                    changed = True

                for j in range(3):
                    if j == k or D[k, j] == 0:
                        continue
                    a = int(D[k, k])
                    b = int(D[k, j])
                    p0, s0, p1, s1 = _elimination_transform(a, b)
                    col_k = D[:, k].copy()
                    col_j = D[:, j].copy()
                    v_k = V[:, k].copy()
                    v_j = V[:, j].copy()
                    D[:, k] = p0 * col_k + s0 * col_j
                    D[:, j] = p1 * col_k + s1 * col_j
                    V[:, k] = p0 * v_k + s0 * v_j
                    V[:, j] = p1 * v_k + s1 * v_j
                    changed = True

            if D[k, k] < 0:
                D[k, :] = -D[k, :]
                U[k, :] = -U[k, :]

            divisor = int(D[k, k])
            nondivisible: tuple[int, int] | None = None
            if divisor != 0:
                for i in range(k + 1, 3):
                    for j in range(k + 1, 3):
                        if int(D[i, j]) % divisor != 0:
                            nondivisible = (i, j)
                            break
                    if nondivisible is not None:
                        break
            if nondivisible is None:
                break

            j = nondivisible[1]
            D[:, k] = D[:, k] + D[:, j]
            V[:, k] = V[:, k] + V[:, j]

    for i in range(3):
        if D[i, i] < 0:
            D[i, :] = -D[i, :]
            U[i, :] = -U[i, :]

    if not np.array_equal(U @ original @ V, D):
        raise ExactNormalFormError("internal SNF check failed: U @ A @ V != D.")
    for i in range(3):
        for j in range(3):
            if i != j and D[i, j] != 0:
                raise ExactNormalFormError(
                    "internal SNF check failed: D is not diagonal."
                )
    diagonal = [abs(int(D[i, i])) for i in range(3)]
    for i in range(2):
        if diagonal[i] != 0 and diagonal[i + 1] % diagonal[i] != 0:
            raise ExactNormalFormError(
                f"internal SNF check failed: {diagonal[i]} does not divide "
                f"{diagonal[i + 1]}."
            )
    return SmithNormalForm(U=U, D=D, V=V)


def column_hnf_3x3(A: ArrayLike) -> np.ndarray:
    """Return canonical lower column-HNF for a full-rank 3 by 3 lattice.

    The result H is lower triangular with positive diagonal and satisfies
    ``0 <= H[j, i] < H[j, j]`` for all ``j > i``.

    :param A: Full-rank 3 by 3 integer matrix.
    :return: Lower column-HNF matrix with Python-integer entries.
    :raises ExactNormalFormError: If A is singular or the postcondition fails.
    """
    H = _as_int_matrix(A, (3, 3), "A")
    if _int_det3(H) == 0:
        raise ExactNormalFormError("column_hnf_3x3 requires a full-rank matrix.")

    # Triangularize: for each diagonal position i, eliminate H[i, j] for j > i.
    for i in range(3):
        nonzero = next((j for j in range(i, 3) if H[i, j] != 0), None)
        if nonzero is None:
            raise ExactNormalFormError("column_hnf_3x3 requires a full-rank matrix.")
        if nonzero != i:
            H[:, [i, nonzero]] = H[:, [nonzero, i]]

        while True:
            changed = False
            for j in range(i + 1, 3):
                if H[i, j] == 0:
                    continue
                a = int(H[i, i])
                b = int(H[i, j])
                p0, s0, p1, s1 = _elimination_transform(a, b)
                col_i = H[:, i].copy()
                col_j = H[:, j].copy()
                H[:, i] = p0 * col_i + s0 * col_j
                H[:, j] = p1 * col_i + s1 * col_j
                changed = True
            if not changed:
                break

        if H[i, i] < 0:
            H[:, i] = -H[:, i]

    # Reduce below-diagonal entries to canonical range [0, H[j, j]).
    # Ascending j order is required: reducing column j modifies column i only
    # through H[j', j] (j' < j), which are zero in the lower-triangular matrix,
    # so earlier reductions are not disturbed.  Descending order would re-dirty
    # entries already reduced via nonzero H[j', j] values.
    for i in range(3):
        for j in range(i + 1, 3):
            diagonal = int(H[j, j])
            quotient = int(H[j, i]) // diagonal
            H[:, i] = H[:, i] - quotient * H[:, j]

    # Postcondition: lower triangular, positive diagonal, canonical residues.
    for j in range(3):
        if H[j, j] <= 0:
            raise ExactNormalFormError(
                f"column_hnf_3x3 postcondition: diagonal H[{j},{j}]={H[j, j]} <= 0."
            )
        for i in range(j):
            if H[j, i] < 0 or H[j, i] >= H[j, j]:
                raise ExactNormalFormError(
                    f"column_hnf_3x3 postcondition: H[{j},{i}]={H[j, i]} "
                    f"not in [0, {H[j, j]})."
                )
        for i in range(j + 1, 3):
            if H[j, i] != 0:
                raise ExactNormalFormError(
                    f"column_hnf_3x3 postcondition: H[{j},{i}]={H[j, i]} != 0 "
                    "(upper-triangle entry non-zero)."
                )

    return H


def hnf_2d_supercells(n: int) -> list[np.ndarray]:
    """Return all 2 by 2 lower-HNF matrices with determinant ``n``.

    The matrices have the form ``[[a, 0], [c, b]]``, where
    ``a * b == n`` and ``0 <= c < b``.  The count is ``sigma(n)``,
    the sum of positive divisors of ``n``.

    These enumerate index-``n`` 2D sublattices/supercells in HNF form.
    Further point-group symmetry reduction, if desired, must be done separately.

    :param n: Positive integer supercell index.
    :return: List of 2 by 2 lower-HNF arrays with Python-integer entries.
    :raises ExactNormalFormError: If ``n`` is not a positive integer.
    """
    if isinstance(n, (bool, np.bool_)):
        raise ExactNormalFormError(f"n must be a positive integer; got {n!r}.")

    try:
        index = operator.index(n)
    except TypeError as exc:
        raise ExactNormalFormError(
            f"n must be a positive integer; got {n!r}."
        ) from exc

    if index < 1:
        raise ExactNormalFormError(f"n must be a positive integer; got {n!r}.")

    hnfs: list[np.ndarray] = []
    for a in range(1, index + 1):
        if index % a == 0:
            b = index // a
            for c in range(b):
                hnfs.append(np.array([[a, 0], [c, b]], dtype=object))
    return hnfs
