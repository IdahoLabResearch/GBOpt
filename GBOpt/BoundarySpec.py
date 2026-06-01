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

@dataclass(frozen=True)
class FiveDOFSpec:
    params: Sequence[float]


@dataclass(frozen=True)
class PQSpec:
    P: Sequence[Sequence[int | float]]
    Q: Sequence[Sequence[int | float]]


@dataclass(frozen=True)
class _CSLSpecBase:
    axis: Sequence[int]
    plane: Sequence[int]
    sigma: int | None = None


@dataclass(frozen=True)
class CSLExactSpec(_CSLSpecBase):
    quat: Sequence[int] = None  # required; integer quaternion


@dataclass(frozen=True)
class CSLApproxSpec(_CSLSpecBase):
    angle_deg: float = None


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
