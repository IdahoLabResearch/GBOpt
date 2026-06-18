# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Exact lattice-reduction algorithms.

Implements Gauss-Lagrange 2D reduction and Lenstra-Lenstra-Lovasz 3D
reduction over integer lattices. All helpers operate on exact integer
arithmetic and reject non-integer inputs. LLL reduction improves basis
quality but is not required for CSL correctness; the exact default is
no post-reduction.
"""

from __future__ import annotations

import math
import warnings
from fractions import Fraction

import numpy as np

from GBOpt.Utils.integer_normal_forms import _dot_int

from .integer import as_int_array, integer_det3
from .types import CrystallographyValueError


class GaussReductionWarning(UserWarning):
    """Issued when Gauss reduction does not converge."""


def gauss_reduce_2d_paired(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gauss-reduce P in-plane rows while applying the same row ops to Q.

    Inputs must be 1D integer-valued arrays. P rows must have matching length,
    and Q rows must have matching length.
    The integer contract is enforced by the caller (``_canonicalize_pq_paired``
    calls ``assert_integer_rows`` on P and Q before slicing into rows).
    Dot products use ``_dot_int`` to avoid int64 overflow for large entries.

    Issues a :class:`GaussReductionWarning` and returns the best result found if the
    algorithm does not converge within 200 iterations.

    :param p1: First in-plane row of P.
    :param p2: Second in-plane row of P.
    :param q1: First in-plane row of Q (paired with p1).
    :param q2: Second in-plane row of Q (paired with p2).
    :return: ``(p1_reduced, p2_reduced, q1_reduced, q2_reduced)``.
    :raises CrystallographyValueError: If any input is not a 1D array or if the
        input arrays have incompatible shapes.
    """
    for name, v in [("p1", p1), ("p2", p2), ("q1", q1), ("q2", q2)]:
        arr = np.asarray(v)
        if arr.ndim != 1:
            raise CrystallographyValueError(
                f"{name} must be a 1D array; got shape {arr.shape}."
            )
    p_lengths = {np.asarray(v).shape[0] for v in (p1, p2)}
    q_lengths = {np.asarray(v).shape[0] for v in (q1, q2)}
    if len(p_lengths) != 1 or len(q_lengths) != 1:
        raise CrystallographyValueError(
            "P rows must match each other and "
            "Q rows must match each other; got lengths "
            f"{[np.asarray(v).shape[0] for v in (p1, p2, q1, q2)]}."
        )
    a = as_int_array(p1, (p1.shape[0],), "p1").astype(int)
    b = as_int_array(p2, (p2.shape[0],), "p2").astype(int)
    qa = as_int_array(q1, (q1.shape[0],), "q1").astype(int)
    qb = as_int_array(q2, (q2.shape[0],), "q2").astype(int)
    for _ in range(200):
        aa = _dot_int(a, a)
        bb = _dot_int(b, b)
        if bb < aa:
            a, b = b, a
            qa, qb = qb, qa
            aa = bb
        if aa == 0:
            break
        ab = _dot_int(a, b)
        t = (ab + aa // 2) // aa
        if t == 0:
            break
        b = b - t * a
        qb = qb - t * qa
    else:
        warnings.warn(
            "Convergence not reached in 200 iterations; result may be unreduced",
            category=GaussReductionWarning,
            stacklevel=3,
        )

    if _dot_int(b, b) < _dot_int(a, a):
        a, b = b, a
        qa, qb = qb, qa
    return a, b, qa, qb


def gauss_reduce_2d(
    v1: np.ndarray, v2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Lagrange 2D lattice reduction using integer arithmetic.

    Returns ``(shorter, longer)`` for integer-valued input vectors. This is a
    thin wrapper over the paired reducer; dummy Q rows let the paired function
    remain the single implementation of the row-operation loop.

    :param v1: First integer-valued in-plane basis vector.
    :param v2: Second integer-valued in-plane basis vector.
    :return: Reduced basis ``(shorter, longer)``.
    :raises CrystallographyValueError: If either vector is not one-dimensional.
    """
    q1 = np.zeros(1, dtype=int)
    q2 = np.zeros(1, dtype=int)
    a, b, _qa, _qb = gauss_reduce_2d_paired(v1, v2, q1, q2)
    return a, b


def lll_reduce(B: np.ndarray, delta: float = 0.75) -> np.ndarray:
    """Return an LLL-reduced basis spanning the same lattice as the columns of B.

    Applies the Lenstra-Lenstra-Lovasz algorithm to find a basis of short,
    nearly-orthogonal vectors.  For a 3-column input the algorithm terminates
    in O(log ||B||) iterations.

    The output columns are the same integer lattice as the input but with
    shorter, more orthogonal basis vectors. All updates are exact integer
    column operations, so the output spans the same lattice as the input.

    :param B: Full-rank 3 by 3 integer matrix whose columns are basis vectors.
    :param delta: Lovasz condition parameter in ``(0.25, 1.0]``. Larger
        values impose a stricter Lovasz condition and usually produce a more
        reduced basis, possibly with more swaps. ``delta=0.75`` is the
        classical Lenstra-Lenstra-Lovasz choice.
    :return: LLL-reduced 3 by 3 integer matrix (object dtype).
    :raises CrystallographyValueError: If ``delta`` is out of range or B is singular.
    """
    try:
        delta_fraction = Fraction(delta).limit_denominator()
    except TypeError as exc:
        raise CrystallographyValueError("delta must be numeric.") from exc
    if not (Fraction(1, 4) < delta_fraction <= Fraction(1, 1)):
        raise CrystallographyValueError("delta must be in the interval (0.25, 1.0].")
    M = as_int_array(B, (3, 3), "B")
    if integer_det3(M) == 0:
        raise CrystallographyValueError(
            "lll_reduce requires a full-rank (non-singular) basis."
        )

    basis = M.copy()

    def _nearest_integer(value: Fraction) -> int:
        """Return floor(value + 1/2), with ties rounded upward."""
        return math.floor(value + Fraction(1, 2))

    def _gram_schmidt_coefficients():
        """Return exact Gram-Schmidt mu coefficients and squared norms."""
        gram = [
            [_dot_int(basis[:, i], basis[:, j]) for j in range(3)]
            for i in range(3)
        ]
        mu = [[Fraction(0, 1) for _ in range(3)] for _ in range(3)]
        norm = [Fraction(0, 1) for _ in range(3)]
        for i in range(3):
            norm_i = Fraction(gram[i][i], 1)
            for j in range(i):
                numerator = Fraction(gram[i][j], 1)
                for ell in range(j):
                    numerator -= mu[i][ell] * mu[j][ell] * norm[ell]
                if norm[j] == 0:
                    raise CrystallographyValueError(
                        "lll_reduce requires a full-rank (non-singular) basis."
                    )
                mu[i][j] = numerator / norm[j]
                norm_i -= mu[i][j] * mu[i][j] * norm[j]
            if norm_i <= 0:
                raise CrystallographyValueError(
                    "lll_reduce requires a full-rank (non-singular) basis."
                )
            norm[i] = norm_i
        return mu, norm

    k = 1
    while k < 3:
        mu, norm = _gram_schmidt_coefficients()

        # Size reduction: replace b_k with b_k - round(mu[k,j]) * b_j.
        for j in range(k - 1, -1, -1):
            r = _nearest_integer(mu[k][j])
            if r != 0:
                basis[:, k] = basis[:, k] - r * basis[:, j]
                mu, norm = _gram_schmidt_coefficients()

        # Lovasz condition: check whether columns k and k-1 should swap.
        if norm[k] >= (delta_fraction - mu[k][k - 1] ** 2) * norm[k - 1]:
            k += 1
        else:
            basis[:, [k, k - 1]] = basis[:, [k - 1, k]]
            k = max(k - 1, 1)

    return basis.astype(object)


__all__ = [
    "GaussReductionWarning",
    "gauss_reduce_2d_paired",
    "gauss_reduce_2d",
    "lll_reduce"
]
