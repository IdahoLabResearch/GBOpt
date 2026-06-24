# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Exact CSL and DSC lattice construction.

Constructs coincidence site lattices and displacement shift complete lattices from
validated ``ScaledRotation`` objects using Smith normal form decomposition and column
Hermite normal form canonicalization. Does not know about boundary specs, P/Q matrices,
``GBMaker``, or embedding construction.
"""

from __future__ import annotations

import math

import numpy as np

from GBOpt.Utils.integer_normal_forms import (
    ExactNormalFormError,
    column_hnf_3x3,
    smith_normal_form_3x3,
)

from .integer import as_int_array, as_int_vector, integer_adj3, integer_det3
from .reduction import lll_reduce
from .types import (
    CoincidenceCheck,
    CrystallographyBackendError,
    CrystallographyNotImplementedError,
    CrystallographyValueError,
    CSLResult,
    DSCBasis,
    Int3,
    ReductionMode,
    ScaledRotation,
    SmithDiagnostics,
)

_VALID_REDUCTION_MODES = ("none", "lll")


def sigma_from_snf_diagonal(denominator: int, diagonal: Int3) -> tuple[int, Int3]:
    """Derive true Sigma and kernel moduli from an SNF diagonal and scale ``N``.

    Each kernel modulus ``N / gcd(d_i, N)`` counts how many lattice planes along SNF
    axis ``i`` are lost to the coincidence condition; their product is the CSL index
    sigma.

    :param denominator: Positive scaled-rotation denominator.
    :param diagonal: Three Smith normal-form diagonal entries.
    :return: ``(sigma, (m1, m2, m3))`` where ``sigma`` is the CSL index and each ``mi``
        is the per-axis kernel modulus ``denominator / gcd(diagonal[i], denominator)``.
    :raises CrystallographyValueError: If ``denominator`` is not positive.
    """
    denom = int(denominator)
    if denom <= 0:
        raise CrystallographyValueError(
            f"denominator must be positive; got {denom}."
        )
    diag = as_int_vector(diagonal, 3, "diagonal")
    m1, m2, m3 = (
        denom // math.gcd(abs(d), denom)
        for d in diag
    )
    sigma = m1 * m2 * m3

    return sigma, (m1, m2, m3)


def csl_from_scaled_rotation(
    rotation: ScaledRotation,
    *,
    post_reduce: ReductionMode = "none",
) -> CSLResult:
    """Construct a canonical CSL basis from an exact scaled rotation.

    :param rotation: Validated ``ScaledRotation`` from
        ``quaternion_to_scaled_rotation``.
    :param post_reduce: Post-reduction mode. ``"none"`` returns the raw SNF-derived
        basis; ``"lll"`` applies Lenstra-Lenstra-Lovasz reduction to yield shorter basis
        vectors. Keyword argument, optional, defaults to ``"none"``.
    :return: Complete ``CSLResult`` whose ``basis_hnf`` field is always the canonical
        column-HNF basis and whose ``basis`` field reflects the requested
        ``post_reduce`` mode.
    :raises CrystallographyValueError: If ``post_reduce`` is unknown or if exact
        verification of the constructed CSL basis fails.
    :raises CrystallographyBackendError: If the internal SNF or HNF computation fails.
    """

    if post_reduce not in _VALID_REDUCTION_MODES:
        expected = ", ".join(repr(mode) for mode in _VALID_REDUCTION_MODES)
        raise CrystallographyValueError(
            f"unknown post_reduce mode {post_reduce}. Expected one of "
            f"{expected}"
        )

    int_matrix = as_int_array(rotation.matrix, (3, 3), "rotation.matrix")
    try:
        snf = smith_normal_form_3x3(int_matrix)
        diagonal: Int3 = (int(snf.D[0, 0]), int(snf.D[1, 1]), int(snf.D[2, 2]))
        sigma, kernel_moduli = sigma_from_snf_diagonal(
            rotation.denominator, diagonal
        )
        raw_basis = snf.V * np.asarray(kernel_moduli, dtype=object)
        basis_hnf = column_hnf_3x3(raw_basis)
    except ExactNormalFormError as exc:
        raise CrystallographyBackendError(str(exc)) from exc

    check = verify_coincidence_basis(rotation, basis_hnf, sigma=sigma)
    if not check.ok:
        raise CrystallographyValueError(
            "constructed CSL basis failed exact verification."
        )

    exposed_basis = lll_reduce(basis_hnf) if post_reduce == "lll" else raw_basis
    diagnostics = SmithDiagnostics(
        diagonal=diagonal,
        kernel_moduli=kernel_moduli,
    )
    result = CSLResult(
        rotation=rotation,
        sigma=sigma,
        basis=exposed_basis,
        basis_hnf=basis_hnf,
        diagnostics=diagnostics,
    )
    return result


def dsc_basis(
    csl_basis: np.ndarray,
    sigma: int,
    *,
    lattice_basis: np.ndarray | None = None,
) -> DSCBasis:
    """Return the cubic DSC basis as an integer numerator and denominator.

    The displacement shift complete lattice is the finest lattice of which both crystal
    lattices are sublattices. For a cubic CSL with basis ``C``, it equals ``adj(C) /
    sigma``.

    :param csl_basis: 3 by 3 integer CSL basis ``C`` in column convention, with each
        column representing a basis vector.
    :param sigma: Expected CSL index, equal to ``abs(det(C))``.
    :param lattice_basis: Reserved non-cubic lattice basis hook; only ``None`` is
        currently supported. Keyword argument, optional, defaults to ``None``.
    :return: Rational DSC basis represented as ``numerator / denominator``.
    :raises CrystallographyValueError: If ``sigma`` is invalid or inconsistent with
        ``csl_basis``.
    :raises CrystallographyNotImplementedError: If ``lattice_basis`` is supplied.
    """
    if lattice_basis is not None:
        raise CrystallographyNotImplementedError(
            "non-cubic lattice bases are not implemented"
        )
    int_matrix = as_int_array(csl_basis, (3, 3), "csl_basis")
    if not isinstance(sigma, (int, np.integer)) or sigma <= 0:
        raise CrystallographyValueError(
            f"sigma must be a positive integer; got {sigma}."
        )
    sigma = int(sigma)
    det = integer_det3(int_matrix)
    abs_det = abs(det)
    if abs_det != sigma:
        raise CrystallographyValueError(
            f"|det(csl_basis)|={abs_det} does not equal sigma={sigma}."
        )
    numerator = np.array(integer_adj3(int_matrix), dtype=object)
    if det < 0:
        numerator = -numerator
    if not np.array_equal(int_matrix @ numerator, sigma * np.eye(3, dtype=object)):
        raise CrystallographyValueError("DSC adjugate check failed.")
    return DSCBasis(numerator=numerator, denominator=sigma, sigma=sigma)


def verify_coincidence_basis(
    rotation: ScaledRotation,
    csl_basis: np.ndarray,
    *,
    sigma: int | None = None,
) -> CoincidenceCheck:
    """Check exact CSL membership and optional determinant index.

    A basis vector ``v`` is in the CSL iff ``rotation.matrix @ v`` is congruent to zero
    modulo ``rotation.denominator``. This condition is checked column-wise across all
    three columns of ``csl_basis``.

    :param rotation: Validated scaled rotation. Its integer numerator matrix and
        denominator define the membership test ``rotation.matrix @ csl_basis == 0``
        modulo ``rotation.denominator``.
    :param csl_basis: 3 by 3 integer basis to check.
    :param sigma: Expected CSL index. Keyword argument, optional, defaults to ``None``.
        When provided, ``abs(det(csl_basis)) == sigma`` is checked alongside the
        residual test, and ``CoincidenceCheck.ok`` is ``False`` if either condition
        fails.
    :return: ``CoincidenceCheck`` containing residual and determinant diagnostics.
    :raises CrystallographyValueError: If the rotation denominator or ``sigma`` is
        invalid.
    """
    int_rotation = as_int_array(rotation.matrix, (3, 3), "rotation.matrix")
    int_basis = as_int_array(csl_basis, (3, 3), "csl_basis")
    denominator = int(rotation.denominator)
    if denominator <= 0:
        raise CrystallographyValueError(
            f"rotation.denominator must be positive; got {rotation.denominator}."
        )
    if sigma is not None:
        if not isinstance(sigma, (int, np.integer)) or sigma <= 0:
            raise CrystallographyValueError(
                f"sigma must be a positive integer; got {sigma}."
            )
        sigma = int(sigma)
    residual = (int_rotation @ int_basis) % denominator
    det_basis = abs(integer_det3(int_basis))
    ok = bool(np.all(residual == 0))
    if sigma is not None:
        ok = ok and det_basis == sigma

    return CoincidenceCheck(
        ok=ok,
        residual_mod_N=residual,
        det_basis=det_basis,
        sigma=sigma,
    )


__all__ = [
    "sigma_from_snf_diagonal",
    "csl_from_scaled_rotation",
    "dsc_basis",
    "verify_coincidence_basis",
]
