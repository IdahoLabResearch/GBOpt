# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Shared data types, type aliases, and exceptions for crystallography operations.

Contains dataclasses, named tuples, type aliases, and the exception hierarchy
used across the crystallography package. No arithmetic or validation logic
belongs here; this module is a pure data-definition layer imported by all
other crystallography modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Int3 = tuple[int, int, int]
Int4 = tuple[int, int, int, int]
ReductionMode = Literal["none", "lll"]


class CrystallographyError(Exception):
    """Base for all crystallography errors."""


class CrystallographyValueError(CrystallographyError, ValueError):
    """Invalid input to a crystallography function."""


class CrystallographyBackendError(CrystallographyError):
    """Computation failed in an exact normal-form routine."""


class CrystallographyDivisibilityError(CrystallographyValueError):
    """A rational exact result was not integer-valued when required."""


class CrystallographyNotImplementedError(CrystallographyError, NotImplementedError):
    """Operation is defined but not yet implemented."""


@dataclass(frozen=True)
class ScaledRotation:
    """Exact scaled rotation ``R = M / N``.

    Project convention:
    ``M / N`` is the row-vector multiplier used by P/Q embeddings::

        q_row = p_row @ M / N

    Column-vector CSL routines must receive the transposed numerator ``M.T``.

    :param N: Positive integer denominator.
    :param M: 3 by 3 integer numerator matrix in row-vector convention.
    :param source: Input source used to construct the rotation.
    :param quaternion: Primitive Hamilton-order integer quaternion when the
        rotation came from quaternion input; otherwise ``None``.
    """

    N: int
    M: np.ndarray
    source: Literal["quaternion", "matrix", "five_dof"]
    quaternion: Int4 | None = None


@dataclass(frozen=True)
class SmithDiagnostics:
    """Smith normal-form diagnostics for a scaled rotation.

    Only the domain-specific derived quantities are stored here.  The full
    ``U``, ``D``, ``V`` matrices from the SNF decomposition are available as
    the local ``snf`` variable inside ``csl_from_scaled_rotation`` but are not
    carried forward since no downstream computation uses them.

    :param diagonal: The SNF diagonal entries ``(d0, d1, d2)``.
    :param kernel_moduli: Per-axis moduli ``N / gcd(di, N)`` used to derive sigma.
    """

    diagonal: Int3
    kernel_moduli: Int3


@dataclass(frozen=True)
class CSLResult:
    """Complete CSL construction result.

    :param rotation: Validated scaled rotation used to build the CSL.
    :param sigma: CSL index.
    :param basis: Exposed 3 by 3 CSL basis, optionally post-reduced.
    :param basis_hnf: Canonical column-HNF CSL basis used for verification.
    :param diagnostics: Smith normal-form derived diagnostic quantities.
    """

    rotation: ScaledRotation
    sigma: int
    basis: np.ndarray
    basis_hnf: np.ndarray
    diagnostics: SmithDiagnostics


@dataclass(frozen=True)
class InPlaneBasis:
    """Primitive in-plane CSL basis and its CSL-column coefficients.

    :param basis: 3 by 2 integer matrix whose columns are in-plane CSL vectors.
    :param coefficients: 3 by 2 coefficient matrix mapping CSL basis columns
        to ``basis``.
    :param plane_covector: Primitive integer plane normal.
    """

    basis: np.ndarray
    coefficients: np.ndarray
    plane_covector: Int3


@dataclass(frozen=True)
class DSCBasis:
    """Rational DSC basis represented by an integer numerator and denominator.

    :param numerator: 3 by 3 integer numerator matrix for the DSC basis.
    :param denominator: Positive denominator; equals ``sigma`` for cubic CSLs.
    :param sigma: CSL index associated with this DSC basis.
    """

    numerator: np.ndarray
    denominator: int
    sigma: int


@dataclass(frozen=True)
class CoincidenceCheck:
    """Exact coincidence-lattice membership check result.

    :param ok: True when every tested basis vector is coincident and, if
        supplied, the determinant matches ``sigma``.
    :param residual_mod_N: Exact residual matrix ``(M @ C) % N``.
    :param det_basis: Absolute determinant of the checked basis.
    :param sigma: Expected sigma used for determinant validation, or ``None``.
    """

    ok: bool
    residual_mod_N: np.ndarray
    det_basis: int
    sigma: int | None


__all__ = [
    "Int3",
    "Int4",
    "ReductionMode",
    "CrystallographyError",
    "CrystallographyValueError",
    "CrystallographyBackendError",
    "CrystallographyDivisibilityError",
    "CrystallographyNotImplementedError",
    "ScaledRotation",
    "SmithDiagnostics",
    "CSLResult",
    "InPlaneBasis",
    "DSCBasis",
    "CoincidenceCheck",
]
