# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

ConstructionMode = Literal["exact", "prefer_exact", "approximate"]


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class BoundarySpecError(Exception):
    """Base class for all boundary-spec validation failures."""


class BoundarySpecTypeError(BoundarySpecError, TypeError):
    """Raised when a boundary-spec field has the wrong type."""


class BoundarySpecValueError(BoundarySpecError, ValueError):
    """Raised when a boundary-spec field has an invalid value."""


# ---------------------------------------------------------------------------
# Boundary-format dataclasses
# ---------------------------------------------------------------------------

def _validate_pq_matrix(m, name: str) -> None:
    """Raise BoundarySpecError if m is not a valid 3x3 non-singular finite matrix.

    Called from PQSpec.__post_init__ for each of P and Q.
    """
    try:
        arr = np.asarray(m, dtype=float)
    except (ValueError, TypeError) as e:
        raise BoundarySpecTypeError(
            f"PQSpec.{name} cannot be converted to a numeric array: {e}"
        ) from e
    if arr.ndim != 2 or arr.shape != (3, 3):
        raise BoundarySpecValueError(
            f"PQSpec.{name} must be a 3x3 matrix; got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise BoundarySpecValueError(
            f"PQSpec.{name} contains non-finite entries (NaN or inf)"
        )
    if abs(np.linalg.det(arr)) < 1e-12:
        raise BoundarySpecValueError(
            f"PQSpec.{name} is singular (determinant ~= 0)"
        )


@dataclass(frozen=True)
class PQSpec:
    P: Sequence[Sequence[int | float]]
    Q: Sequence[Sequence[int | float]]

    def __post_init__(self):
        _validate_pq_matrix(self.P, "P")
        _validate_pq_matrix(self.Q, "Q")


# ---------------------------------------------------------------------------
# Internal canonical boundary embedding
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundaryEmbedding:
    """Canonical internal representation produced by every input adapter.

    P and Q are the exact row-wise orientation matrices (None for
    approximate-only paths). R_left and R_right are floating-point rotation
    matrices matching GBMaker's internal convention. exact and coherent flag
    the construction path and interface type. source names the originating
    format ("pq", "csl", "five_dof").
    """
    P: np.ndarray | None
    Q: np.ndarray | None
    R_left: np.ndarray
    R_right: np.ndarray
    exact: bool
    coherent: bool
    source: str
