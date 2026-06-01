# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from dataclasses import dataclass
from typing import Literal, Sequence

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
