# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Exact lattice-reduction algorithms.

Implements Gauss-Lagrange 2D reduction and Lenstra-Lenstra-Lovasz 3D reduction over
integer lattices. All helpers operate on exact integer arithmetic and reject non-integer
inputs. LLL reduction improves basis quality but is not required for CSL correctness;
the exact default is no post-reduction.
"""

from __future__ import annotations

import math
import warnings
from fractions import Fraction

import numpy as np

from GBOpt.Utils.integer_linalg import dot_int

from .integer import as_int_array, integer_det3
from .types import CrystallographyValueError


class GaussReductionWarning(UserWarning):
    """Issued when Gauss reduction does not converge."""


MAX_GAUSS_REDUCTION_STEPS = 200
_MIN_LLL_DELTA = Fraction(1, 4)
_MAX_LLL_DELTA = Fraction(1, 1)
_ZERO = Fraction(0, 1)
_HALF = Fraction(1, 2)


def _as_int_1d(vector: np.ndarray, name: str) -> np.ndarray:
    """Return vector as a validated one-dimensional object array of integers.

    :param vector: Array-like vector to validate.
    :param name: Name used in error messages.
    :return: One-dimensional exact integer ndarray.
    :raises CrystallographyValueError: If vector is not one-dimensional.
    """
    arr = np.asarray(vector)
    if arr.ndim != 1:
        raise CrystallographyValueError(
            f"{name} must be a 1D array; got shape {arr.shape}."
        )
    return as_int_array(arr, (arr.shape[0],), name)


def _as_public_int_array(array: np.ndarray) -> np.ndarray:
    """Return array with a NumPy integer dtype when values fit, else object dtype.

    Reduction arithmetic is performed with object arrays to preserve exact Python-int
    behavior. Public callers historically received NumPy integer arrays, so this helper
    converts back when doing so is safe.

    :param array: Exact integer array to convert for public return.
    :return: NumPy integer array when all values fit in np.int_, otherwise object-dtype
        array with the same shape.
    """
    info = np.iinfo(np.int_)
    values = [int(value) for value in np.asarray(array, dtype=object).flat]
    if all(info.min <= value <= info.max for value in values):
        return np.array(values, dtype=int).reshape(array.shape)
    return np.array(values, dtype=object).reshape(array.shape)


def gauss_reduce_2d_paired(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gauss-reduce ``P`` in-plane rows while applying the same row operations to ``Q``.

    All four inputs must be one-dimensional integer-valued arrays. Dot products use
    ``dot_int`` to avoid fixed-width integer overflow for large entries. ``P`` rows must
    have equal length, and ``Q`` rows must have equal length; ``P`` and ``Q`` row
    lengths may differ from each other.

    Issues ``GaussReductionWarning`` and returns the best result found if the algorithm
    does not converge within ``MAX_GAUSS_REDUCTION_STEPS`` iterations.

    :param p1: First in-plane row of ``P``.
    :param p2: Second in-plane row of ``P``.
    :param q1: First in-plane row of ``Q`` paired with ``p1``.
    :param q2: Second in-plane row of ``Q`` paired with ``p2``.
    :return: ``(shorter_p, longer_p, paired_q_for_shorter, paired_q_for_longer)``, with
        ``Q`` rows reordered to match ``P``.
    :raises CrystallographyValueError: If any input is not one-dimensional or if paired
        row lengths are incompatible.
    """
    validated = {}
    for name, vector in [("p1", p1), ("p2", p2), ("q1", q1), ("q2", q2)]:
        validated[name] = _as_int_1d(vector, name)

    p_lengths = {validated[name].shape[0] for name in ("p1", "p2")}
    q_lengths = {validated[name].shape[0] for name in ("q1", "q2")}
    if len(p_lengths) != 1 or len(q_lengths) != 1:
        raise CrystallographyValueError(
            "P rows must match each other and "
            "Q rows must match each other; got lengths "
            f"{[validated[name].shape[0] for name in ('p1', 'p2', 'q1', 'q2')]}."
        )

    a, b = validated["p1"], validated["p2"]
    qa, qb = validated["q1"], validated["q2"]

    for _ in range(MAX_GAUSS_REDUCTION_STEPS):
        aa = dot_int(a, a)
        bb = dot_int(b, b)

        if bb < aa:
            a, b = b, a
            qa, qb = qb, qa
            aa = bb

        if aa == 0:
            break

        ab = dot_int(a, b)
        reduction_coeff = (ab + aa // 2) // aa
        if reduction_coeff == 0:
            break

        b = b - reduction_coeff * a
        qb = qb - reduction_coeff * qa
    else:
        warnings.warn(
            "Convergence not reached in "
            f"{MAX_GAUSS_REDUCTION_STEPS} iterations; result may be unreduced",
            category=GaussReductionWarning,
            stacklevel=3,
        )

    if dot_int(b, b) < dot_int(a, a):
        a, b = b, a
        qa, qb = qb, qa

    return (
        _as_public_int_array(a),
        _as_public_int_array(b),
        _as_public_int_array(qa),
        _as_public_int_array(qb),
    )


def gauss_reduce_2d(v1: np.ndarray, v2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Lagrange 2D lattice reduction using integer arithmetic.

    Returns ``(shorter, longer)`` for integer-valued input vectors. This is a thin
    wrapper over the paired reducer; dummy Q rows let the paired function remain the
    single implementation of the row-operation loop.

    :param v1: First integer-valued in-plane basis vector.
    :param v2: Second integer-valued in-plane basis vector.
    :return: Reduced basis ``(shorter, longer)``.
    """
    q1 = np.zeros_like(v1, dtype=object)
    q2 = np.zeros_like(v2, dtype=object)
    a, b, _qa, _qb = gauss_reduce_2d_paired(v1, v2, q1, q2)
    return a, b


def lll_reduce(basis_input: np.ndarray, delta: float = 0.75) -> np.ndarray:
    """Return an LLL-reduced basis spanning the same lattice as the columns of ``B``.

    Applies the Lenstra-Lenstra-Lovasz algorithm to find a basis of short, nearly
    orthogonal vectors. For a 3-column input, the algorithm terminates in ``O(log
    ||B||)`` iterations.

    The output columns span the same integer lattice as the input but usually have
    shorter, more orthogonal basis vectors.

    :param basis_input: Full-rank 3 by 3 integer matrix whose columns are basis vectors.
    :param delta: Lovasz condition parameter in ``(0.25, 1.0]``. Larger values impose a
        stricter Lovasz condition and usually produce a more reduced basis, possibly
        with more swaps. ``delta=0.75`` is the classical Lenstra-Lenstra-Lovasz choice.
        Keyword argument, optional, defaults to ``0.75``.
    :return: LLL-reduced 3 by 3 integer matrix with object dtype.
    :raises CrystallographyValueError: If ``delta`` is out of range or ``basis_input``
        is singular.
    """
    try:
        delta_fraction = Fraction(delta).limit_denominator(1_000_000)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CrystallographyValueError("delta must be finite numeric.") from exc
    if not (_MIN_LLL_DELTA < delta_fraction <= _MAX_LLL_DELTA):
        raise CrystallographyValueError("delta must be in the interval (0.25, 1.0].")
    int_basis = as_int_array(basis_input, (3, 3), "basis_input")
    if integer_det3(int_basis) == 0:
        raise CrystallographyValueError(
            "lll_reduce requires a full-rank (non-singular) basis."
        )

    basis = int_basis.copy()

    def _nearest_integer(value: Fraction) -> int:
        """Return floor(value + 1/2), with ties rounded upward.

        :param value: Fraction to round.
        :return: Nearest integer using upward tie handling.
        """
        return math.floor(value + _HALF)

    def _gram_schmidt_coefficients() -> tuple[list[list[Fraction]], list[Fraction]]:
        """Return exact Gram-Schmidt coefficients and squared norms.

        :return: Pair ``(mu, norm)``, where ``mu`` is the lower Gram-Schmidt coefficient
            table and ``norm`` contains exact squared orthogonalized lengths.
        :raises CrystallographyValueError: If the basis is rank deficient during
            Gram-Schmidt computation.
        """
        gram = [[0 for _ in range(3)] for _ in range(3)]
        for i in range(3):
            for j in range(i, 3):
                value = dot_int(basis[:, i], basis[:, j])
                gram[i][j] = value
                gram[j][i] = value

        mu = [[_ZERO for _ in range(3)] for _ in range(3)]
        norm = [_ZERO for _ in range(3)]
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
            # Negative values shouldn't happen for valid input
            if norm_i <= 0:
                raise CrystallographyValueError(
                    "lll_reduce requires a full-rank (non-singular) basis."
                )
            norm[i] = norm_i
        return mu, norm

    # pivot index; LLL works left to right, reducing column k against all prior columns
    k = 1
    while k < 3:
        mu, norm = _gram_schmidt_coefficients()

        # Size reduction: replace b_k with b_k - round(mu[k,j]) * b_j. Norms are
        # unchanged by size reduction; only mu[k][ell] for ell <= j need updating, so we
        # avoid a full recomputation after each step.
        for j in range(k - 1, -1, -1):
            # nearest integer to mu[k][j]; subtracted to size-reduce column k
            r = _nearest_integer(mu[k][j])
            if r != 0:
                basis[:, k] = basis[:, k] - r * basis[:, j]
                # update prior mu[k][ell] coefficients in-place using the Gram-Schmidt
                # recurrence
                for ell in range(j):
                    mu[k][ell] -= r * mu[j][ell]
                mu[k][j] -= r  # mu[j][j] is implicitly 1 in the standard convention

        # Lovasz condition: check whether columns k and k-1 should swap.
        if norm[k] >= (delta_fraction - mu[k][k - 1] ** 2) * norm[k - 1]:
            k += 1
        else:
            tmp = basis[:, k].copy()
            basis[:, k] = basis[:, k - 1]
            basis[:, k - 1] = tmp
            k = max(k - 1, 1)

    return basis


__all__ = [
    "GaussReductionWarning",
    "gauss_reduce_2d_paired",
    "gauss_reduce_2d",
    "lll_reduce",
]
