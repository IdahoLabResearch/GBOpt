# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Exact CSL and DSC lattice construction.

Constructs coincidence site lattices and displacement shift complete lattices
from validated ScaledRotation objects using Smith normal form decomposition
and column Hermite normal form canonicalization. Does not know about boundary
specs, P/Q matrices, GBMaker, or embedding construction.
"""

from __future__ import annotations

import math

import numpy as np

from GBOpt.Utils import integer_normal_forms as inf
from GBOpt.Utils.integer_normal_forms import (
    ExactNormalFormError,
    column_hnf_3x3,
    smith_normal_form_3x3,
)

from .integer import as_int_array, as_int_vector, integer_det3
from .reduction import lll_reduce
from .rotation import assert_scaled_rotation
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


def sigma_from_snf_diagonal(N: int, diagonal: tuple) -> tuple[int, Int3]:
    """Derive true Sigma and kernel moduli from an SNF diagonal and scale ``N``.

    :param N: Positive scaled-rotation denominator.
    :param diagonal: Three Smith normal-form diagonal entries.
    :return: ``(sigma, kernel_moduli)``.
    :raises CrystallographyValueError: If ``N`` or the derived moduli are not positive.
    """
    scale = int(N)
    if scale <= 0:
        raise CrystallographyValueError(f"N must be positive; got {N!r}.")
    diag = as_int_vector(diagonal, 3, "diagonal")
    moduli: list[int] = []
    sigma = 1
    for value in diag:
        modulus = scale // math.gcd(abs(int(value)), scale)
        if modulus <= 0:
            raise CrystallographyValueError(
                "SNF-derived kernel modulus is not positive.")
        moduli.append(modulus)
        sigma *= modulus
    if sigma <= 0:
        raise CrystallographyValueError("SNF-derived sigma is not positive.")

    m1, m2, m3 = moduli
    return sigma, (m1, m2, m3)


def csl_from_scaled_rotation(
    rotation: ScaledRotation,
    *,
    post_reduce: ReductionMode = "none",
) -> CSLResult:
    """Construct a canonical CSL basis from an exact scaled rotation.

    :param rotation: Validated scaled rotation (from ``quaternion_to_scaled_rotation``).
    :param post_reduce: Optional post-reduction mode. ``"none"`` returns the raw
        SNF-derived basis; ``"lll"`` applies Lenstra-Lenstra-Lovasz reduction to
        yield shorter basis vectors.
    :return: Complete CSL construction result.
    :raises CrystallographyValueError: On invalid input or if exact verification fails.
    :raises CrystallographyBackendError: If the internal SNF or HNF computation fails.
    """
    if post_reduce not in ("none", "lll"):
        raise CrystallographyValueError(f"unknown post_reduce mode {post_reduce!r}.")

    M = as_int_array(rotation.M, (3, 3), "rotation.M")
    checked_rotation = ScaledRotation(
        N=int(rotation.N),
        M=M,
        source=rotation.source,
        quaternion=rotation.quaternion,
    )
    assert_scaled_rotation(checked_rotation)
    try:
        snf = smith_normal_form_3x3(M)
        diagonal: Int3 = (int(snf.D[0, 0]), int(snf.D[1, 1]), int(snf.D[2, 2]))
        sigma, kernel_moduli = sigma_from_snf_diagonal(checked_rotation.N, diagonal)
        scales = np.diag(np.array(kernel_moduli, dtype=object))
        raw_basis = snf.V @ scales
        basis_hnf = column_hnf_3x3(raw_basis)
    except ExactNormalFormError as exc:
        raise CrystallographyBackendError(str(exc)) from exc

    exposed_basis = lll_reduce(basis_hnf) if post_reduce == "lll" else raw_basis
    diagnostics = SmithDiagnostics(
        diagonal=diagonal,
        kernel_moduli=kernel_moduli,
    )
    result = CSLResult(
        rotation=checked_rotation,
        sigma=sigma,
        basis=exposed_basis,
        basis_hnf=basis_hnf,
        diagnostics=diagnostics,
    )
    check = verify_coincidence_basis(checked_rotation, basis_hnf, sigma=sigma)
    if not check.ok:
        raise CrystallographyValueError(
            "constructed CSL basis failed exact verification.")
    return result


def dsc_basis(
    csl_basis: np.ndarray,
    sigma: int,
    *,
    lattice_basis: np.ndarray | None = None,
) -> DSCBasis:
    """Return the cubic DSC basis numerator ``adj(C)`` and denominator Sigma.

    :param csl_basis: 3 by 3 integer CSL basis ``C``.
    :param sigma: Expected CSL index, equal to ``abs(det(C))``.
    :param lattice_basis: Reserved non-cubic lattice basis hook; only ``None``
        is currently supported.
    :return: Rational DSC basis represented as ``numerator / denominator``.
    :raises CrystallographyValueError: If sigma is invalid or inconsistent with
        ``csl_basis``.
    :raises CrystallographyNotImplementedError: If ``lattice_basis`` is supplied.
    """
    if lattice_basis is not None:
        raise CrystallographyNotImplementedError(
            "non-cubic lattice bases are not implemented"
        )
    C = as_int_array(csl_basis, (3, 3), "csl_basis")
    sigma_int = int(sigma)
    if sigma_int != sigma or sigma_int <= 0:
        raise CrystallographyValueError(
            f"sigma must be a positive integer; got {sigma!r}.")
    det = integer_det3(C)
    if abs(det) != sigma_int:
        raise CrystallographyValueError(
            f"|det(csl_basis)|={abs(det)} does not equal sigma={sigma_int}."
        )
    numerator = np.array(inf._int_adj3(C), dtype=object)
    if det < 0:
        numerator = -numerator
    if not np.array_equal(C @ numerator, sigma_int * np.eye(3, dtype=object)):
        raise CrystallographyValueError("DSC adjugate check failed.")
    return DSCBasis(numerator=numerator, denominator=sigma_int, sigma=sigma_int)


def verify_coincidence_basis(
    rotation: ScaledRotation,
    csl_basis: np.ndarray,
    *,
    sigma: int | None = None,
) -> CoincidenceCheck:
    """Check exact CSL membership and optional determinant index.

    :param rotation: Validated scaled rotation ``M / N``.
    :param csl_basis: 3 by 3 integer basis to check.
    :param sigma: Optional expected determinant index.
    :return: Residual and determinant diagnostics.
    :raises CrystallographyValueError: If the rotation denominator or sigma is invalid.
    """
    M = as_int_array(rotation.M, (3, 3), "rotation.M")
    C = as_int_array(csl_basis, (3, 3), "csl_basis")
    N = int(rotation.N)
    if N <= 0:
        raise CrystallographyValueError(
            f"rotation.N must be positive; got {rotation.N!r}.")
    residual = (M @ C) % N
    det_basis = abs(integer_det3(C))
    ok = bool(np.all(residual == 0))
    if sigma is not None:
        expected = int(sigma)
        if expected != sigma or expected <= 0:
            raise CrystallographyValueError(
                f"sigma must be a positive integer; got {sigma!r}."
            )
        ok = ok and det_basis == expected
    else:
        expected = None
    return CoincidenceCheck(
        ok=ok,
        residual_mod_N=residual.astype(object),
        det_basis=det_basis,
        sigma=expected,
    )


__all__ = [
    "sigma_from_snf_diagonal",
    "csl_from_scaled_rotation",
    "dsc_basis",
    "verify_coincidence_basis",
]
