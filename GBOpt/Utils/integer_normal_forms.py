# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Exact integer normal-form helpers for small CSL matrices."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike


class ExactNormalFormError(Exception):
    """Base exception for integer normal-form computation failures."""


@dataclass(frozen=True)
class SmithNormalForm:
    """Smith normal-form decomposition ``U @ A @ V == D``."""

    U: np.ndarray
    D: np.ndarray
    V: np.ndarray


def _as_int_matrix(A: ArrayLike, shape: tuple[int, int], name: str) -> np.ndarray:
    """Return *A* as a shape-checked object array of Python integers."""
    arr = np.asarray(A)
    if arr.shape != shape:
        raise ExactNormalFormError(f"{name} must have shape {shape}; got {arr.shape}.")
    out = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        value = arr[index]
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
    """Return an ``n`` by ``n`` identity matrix with Python-int entries."""
    return np.eye(n, dtype=object)


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return ``(g, x, y)`` with ``x*a + y*b == g == gcd(abs(a), abs(b))``."""
    a = int(a)
    b = int(b)
    old_r, r = abs(a), abs(b)
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    x = old_s if a >= 0 else -old_s
    y = old_t if b >= 0 else -old_t
    return old_r, x, y


def _det3(A: np.ndarray) -> int:
    """Return the exact determinant of a 3 by 3 integer matrix."""
    return int(
        A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
        - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
        + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
    )


def _select_pivot(A: np.ndarray, start: int) -> tuple[int, int] | None:
    """Return the smallest nonzero pivot position in ``A[start:, start:]``."""
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
                    g, s, t = _extended_gcd(a, b)
                    row_k = D[k, :].copy()
                    row_i = D[i, :].copy()
                    u_k = U[k, :].copy()
                    u_i = U[i, :].copy()
                    D[k, :] = s * row_k + t * row_i
                    D[i, :] = (-b // g) * row_k + (a // g) * row_i
                    U[k, :] = s * u_k + t * u_i
                    U[i, :] = (-b // g) * u_k + (a // g) * u_i
                    changed = True

                for j in range(3):
                    if j == k or D[k, j] == 0:
                        continue
                    a = int(D[k, k])
                    b = int(D[k, j])
                    g, s, t = _extended_gcd(a, b)
                    col_k = D[:, k].copy()
                    col_j = D[:, j].copy()
                    v_k = V[:, k].copy()
                    v_j = V[:, j].copy()
                    D[:, k] = s * col_k + t * col_j
                    D[:, j] = (-b // g) * col_k + (a // g) * col_j
                    V[:, k] = s * v_k + t * v_j
                    V[:, j] = (-b // g) * v_k + (a // g) * v_j
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
                raise ExactNormalFormError("internal SNF check failed: D is not diagonal.")
    diagonal = [abs(int(D[i, i])) for i in range(3)]
    for i in range(2):
        if diagonal[i] != 0 and diagonal[i + 1] % diagonal[i] != 0:
            raise ExactNormalFormError(
                f"internal SNF check failed: {diagonal[i]} does not divide "
                f"{diagonal[i + 1]}."
            )
    return SmithNormalForm(U=U, D=D, V=V)


def _column_hnf_3x3_fallback(A: np.ndarray) -> np.ndarray:
    """Return lower column-HNF using only unimodular column operations."""
    H = A.copy()
    if _det3(H) == 0:
        raise ExactNormalFormError("column_hnf_3x3 requires a full-rank matrix.")

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
                g, s, t = _extended_gcd(a, b)
                col_i = H[:, i].copy()
                col_j = H[:, j].copy()
                H[:, i] = s * col_i + t * col_j
                H[:, j] = (-b // g) * col_i + (a // g) * col_j
                changed = True
            if not changed:
                break

        if H[i, i] < 0:
            H[:, i] = -H[:, i]

    for i in range(3):
        if H[i, i] <= 0:
            H[:, i] = -H[:, i]
        for j in range(2, i, -1):
            diagonal = int(H[j, j])
            quotient = math.floor(int(H[j, i]) / diagonal)
            H[:, i] = H[:, i] - quotient * H[:, j]

    if _det3(H) < 0:
        H[:, 0] = -H[:, 0]
        for j in range(2, 0, -1):
            diagonal = int(H[j, j])
            quotient = math.floor(int(H[j, 0]) / diagonal)
            H[:, 0] = H[:, 0] - quotient * H[:, j]

    return H.astype(object)


def _column_hnf_3x3_sympy(A: np.ndarray) -> np.ndarray | None:
    """Return SymPy column-HNF when SymPy is importable, otherwise ``None``."""
    try:
        import sympy as sp
        from sympy.matrices.normalforms import hermite_normal_form
    except ImportError:
        return None

    matrix = sp.Matrix([[int(A[i, j]) for j in range(3)] for i in range(3)])
    try:
        hnf = matrix.hermite_normal_form(col_wise=True)
    except (AttributeError, TypeError):
        hnf = hermite_normal_form(matrix.T).T
    out = np.array([[int(hnf[i, j]) for j in range(3)] for i in range(3)], dtype=object)
    if _det3(out) < 0:
        out[:, 0] = -out[:, 0]
    return _column_hnf_3x3_fallback(out)


def column_hnf_3x3(A: ArrayLike) -> np.ndarray:
    """Return canonical lower column-HNF for a full-rank 3 by 3 lattice."""
    matrix = _as_int_matrix(A, (3, 3), "A")
    if _det3(matrix) == 0:
        raise ExactNormalFormError("column_hnf_3x3 requires a full-rank matrix.")
    sympy_result = _column_hnf_3x3_sympy(matrix)
    if sympy_result is not None:
        return sympy_result
    return _column_hnf_3x3_fallback(matrix)


def saturate_column_lattice_3x3(A: ArrayLike) -> np.ndarray:
    """Return the primitive saturation of a full-rank 3 by 3 column lattice.

    A full-rank sublattice of ``Z^3`` spans all of ``Q^3``, so its saturation
    in ``Z^3`` is the ambient integer lattice itself.
    """
    matrix = _as_int_matrix(A, (3, 3), "A")
    if _det3(matrix) == 0:
        raise ExactNormalFormError(
            "saturate_column_lattice_3x3 currently requires full rank."
        )
    return _identity(3)


def hnf_2d_supercells(n: int) -> list[np.ndarray]:
    """Return all 2 by 2 lower-HNF matrices with determinant ``n``."""
    try:
        index = int(n)
    except (TypeError, ValueError) as exc:
        raise ExactNormalFormError(f"n must be a positive integer; got {n!r}.") from exc
    if index != n or index < 1:
        raise ExactNormalFormError(f"n must be a positive integer; got {n!r}.")

    hnfs: list[np.ndarray] = []
    for a in range(1, index + 1):
        if index % a != 0:
            continue
        b = index // a
        for c in range(b):
            hnfs.append(np.array([[a, 0], [c, b]], dtype=object))
    return hnfs
