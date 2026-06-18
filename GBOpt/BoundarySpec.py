# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from dataclasses import dataclass
import math
import operator
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


class BoundarySpecOrthogonalityError(BoundarySpecError):
    """Raised when a derived rotation matrix is not orthogonal (R @ R.T != I).

    Distinct from the base ``BoundarySpecError`` so callers can catch it
    specifically without matching on error-message text.
    """


# ---------------------------------------------------------------------------
# Boundary-format dataclasses
# ---------------------------------------------------------------------------

def _is_bool_like(value) -> bool:
    """Return True for Python or NumPy boolean scalars."""
    return isinstance(value, (bool, np.bool_))


def _reject_bool_values(values, name: str) -> None:
    """Reject booleans before numeric conversion treats them as 0/1."""
    arr = np.asarray(values, dtype=object)
    if any(_is_bool_like(value) for value in arr.flat):
        raise BoundarySpecTypeError(f"{name} must not contain boolean values.")


@dataclass(frozen=True)
class FiveDOFSpec:
    """Five-degree-of-freedom boundary specified by Euler-angle parameters.

    :param params: Five-element sequence ``[alpha, beta, gamma, theta, phi]``
        where ``alpha``, ``beta``, and ``gamma`` are ZXZ Euler angles (radians)
        defining the misorientation rotation between the two grains; ``theta``
        is an additional inclination rotation about the y-axis (the tilt angle,
        radians); and ``phi`` is an additional inclination rotation about the
        z-axis (radians).  All entries must be finite floats.
    """

    params: Sequence[float]

    def __post_init__(self):
        _reject_bool_values(self.params, "FiveDOFSpec.params")
        try:
            arr = np.asarray(self.params, dtype=float)
        except (ValueError, TypeError) as e:
            raise BoundarySpecTypeError(
                f"FiveDOFSpec.params cannot be converted to a numeric array: {e}"
            ) from e
        if arr.ndim != 1 or arr.shape[0] != 5:
            raise BoundarySpecValueError(
                "FiveDOFSpec.params must be a 5-element sequence of "
                f"[alpha, beta, gamma, theta, phi]; got shape {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise BoundarySpecValueError(
                "FiveDOFSpec.params contains non-finite entries (NaN or inf)"
            )
        object.__setattr__(self, "params", tuple(float(value) for value in arr))


def _validate_nonzero_int_vector(
    seq, name: str, length: int = 3
) -> tuple[int, ...]:
    """Return ``seq`` as an integer tuple after non-zero vector validation."""
    _reject_bool_values(seq, name)
    try:
        arr = np.asarray(seq, dtype=float)
    except (ValueError, TypeError) as e:
        raise BoundarySpecTypeError(
            f"{name} cannot be converted to a numeric array: {e}"
        ) from e
    if arr.ndim != 1 or arr.shape[0] != length:
        raise BoundarySpecValueError(
            f"{name} must be a {length}-element integer sequence; got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise BoundarySpecValueError(
            f"{name} contains non-finite entries (NaN or inf)"
        )
    if not np.allclose(arr, np.round(arr), atol=1e-9, rtol=0.0):
        raise BoundarySpecValueError(
            f"{name} must have integer-valued components; got {arr.tolist()}"
        )
    if not np.any(np.round(arr).astype(int)):
        raise BoundarySpecValueError(f"{name} must not be all-zero")
    return tuple(int(value) for value in np.round(arr).astype(int))


def _validate_pq_matrix(m, name: str) -> tuple[tuple[float, ...], ...]:
    """Raise BoundarySpecError if m is not a valid 3x3 non-singular finite matrix.

    Called from PQSpec.__post_init__ for each of P and Q.
    """
    _reject_bool_values(m, f"PQSpec.{name}")
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
    return tuple(tuple(float(value) for value in row) for row in arr)


def _sigma_from_integer_quaternion(quat: Sequence[int]) -> int:
    """Return the Sigma value implied by a primitive integer quaternion."""
    values = [int(value) for value in quat]
    gcd_value = 0
    for value in values:
        gcd_value = math.gcd(gcd_value, abs(value))
    values = [value // gcd_value for value in values]
    norm_sq = sum(value * value for value in values)
    while norm_sq > 1 and norm_sq % 2 == 0:
        norm_sq //= 2
    return int(norm_sq)


def _validate_positive_integer(value, name: str) -> int:
    """Return a positive integer scalar while rejecting booleans."""
    if _is_bool_like(value):
        raise BoundarySpecTypeError(f"{name} must not be boolean.")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise BoundarySpecValueError(
            f"{name} must be a positive integer"
        ) from exc
    if integer <= 0:
        raise BoundarySpecValueError(f"{name} must be a positive integer")
    return int(integer)


@dataclass(frozen=True)
class PQSpec:
    P: Sequence[Sequence[int | float]]
    Q: Sequence[Sequence[int | float]]
    basis_mode: Literal["primitive", "supplied"] = "primitive"

    def __post_init__(self):
        P = _validate_pq_matrix(self.P, "P")
        Q = _validate_pq_matrix(self.Q, "Q")
        if self.basis_mode not in ("primitive", "supplied"):
            raise BoundarySpecValueError(
                "PQSpec.basis_mode must be either 'primitive' or 'supplied'; "
                f"got {self.basis_mode!r}"
            )
        object.__setattr__(self, "P", P)
        object.__setattr__(self, "Q", Q)


@dataclass(frozen=True, kw_only=True)
class _CSLSpecBase:
    axis: Sequence[int]
    plane: Sequence[int]
    sigma: int | None = None

    def __post_init__(self):
        axis = _validate_nonzero_int_vector(self.axis, "axis")
        plane = _validate_nonzero_int_vector(self.plane, "plane")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "plane", plane)
        if self.sigma is not None:
            sigma = _validate_positive_integer(self.sigma, "sigma")
            object.__setattr__(self, "sigma", sigma)


@dataclass(frozen=True, kw_only=True)
class CSLExactSpec(_CSLSpecBase):
    """Exact CSL boundary specified by axis, plane, and integer quaternion.

    :param axis: Rotation axis as integer Miller indices [u v w], e.g.
        ``[0, 0, 1]`` for a [001]-axis tilt boundary.
    :param plane: Boundary-plane normal as integer Miller indices [h k l] in
        grain 1's crystal frame, e.g. ``[1, 0, 0]`` for a (100) boundary plane.
    :param quat: Integer quaternion in Hamilton scalar-first order
        ``[w, x, y, z]`` where w = cos(theta/2) and (x, y, z) =
        sin(theta/2)*n_hat. All four components must be integers (the actual
        unit quaternion is derived by dividing by the norm). The identity
        quaternion is valid and represents a zero-misorientation Sigma 1 case.
        Example: Sigma5 [001] 53.13 deg tilt: ``quat=[2, 0, 0, 1]``.
    :param sigma: Optional sigma value for the CSL boundary (e.g. ``5`` for Sigma5).
        When provided it is validated against the quaternion; mismatches raise
        ``BoundarySpecError``.  Sigma equals the odd part of w^2+x^2+y^2+z^2.
    """

    # default=None is required by Python dataclass inheritance rules: once the
    # parent class (_CSLSpecBase) has any field with a default (sigma=None),
    # every subclass field must also carry a default.  The __post_init__ below
    # turns the None sentinel into a hard error, making quat effectively required.
    quat: Sequence[int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.quat is None:
            raise BoundarySpecValueError("CSLExactSpec.quat is required.")
        quat = _validate_nonzero_int_vector(self.quat, "quat", length=4)
        object.__setattr__(self, "quat", quat)
        quat_vec = np.array(quat[1:], dtype=float)
        quat_vec_norm = np.linalg.norm(quat_vec)
        if quat_vec_norm > 1e-10:
            # The vector part must be parallel to the user-supplied axis field.
            axis_vec = np.asarray(self.axis, dtype=float)
            cross = np.cross(quat_vec, axis_vec)
            denom = quat_vec_norm * np.linalg.norm(axis_vec)
            if np.linalg.norm(cross) > 1e-9 * denom:
                raise BoundarySpecValueError(
                    f"Quaternion vector part {quat_vec.tolist()} is not parallel to "
                    f"axis {list(self.axis)}. The axis must match the rotation axis "
                    "encoded in the quaternion (components [x, y, z] in Hamilton "
                    "order)."
                )
        if self.sigma is not None:
            derived_sigma = _sigma_from_integer_quaternion(quat)
            if derived_sigma != self.sigma:
                raise BoundarySpecValueError(
                    f"Sigma mismatch: quaternion {list(quat)} gives "
                    f"sigma={derived_sigma}, but sigma={self.sigma} was provided."
                )


@dataclass(frozen=True, kw_only=True)
class CSLApproxSpec(_CSLSpecBase):
    """Approximate CSL boundary specified by rotation axis, boundary plane, and angle.

    :param axis: Rotation axis as integer Miller indices [u v w].
    :param plane: Boundary-plane normal as integer Miller indices [h k l] in
        grain 1's crystal frame.
    :param angle_deg: Misorientation angle in degrees (required).
    :param sigma: Optional nominal sigma value (informational only; not validated
        against the angle since the construction is approximate).
    """

    # default=None required by dataclass inheritance; __post_init__ enforces it.
    angle_deg: float = None

    def __post_init__(self):
        super().__post_init__()
        if self.angle_deg is None:
            raise BoundarySpecValueError("CSLApproxSpec.angle_deg is required.")
        if _is_bool_like(self.angle_deg):
            raise BoundarySpecTypeError(
                "CSLApproxSpec.angle_deg must not be boolean."
            )
        try:
            val = float(self.angle_deg)
        except (TypeError, ValueError) as e:
            raise BoundarySpecTypeError(
                "CSLApproxSpec.angle_deg must be a finite float; "
                f"got {self.angle_deg!r}"
            ) from e
        if not np.isfinite(val):
            raise BoundarySpecValueError(
                f"CSLApproxSpec.angle_deg must be finite; got {val}"
            )
        object.__setattr__(self, "angle_deg", val)


# ---------------------------------------------------------------------------
# Internal canonical boundary embedding
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrimitiveCellMetadata:
    """Primitive boundary-defining cell metadata for exact embeddings.

    Stores boundary-topology quantities produced by the CSL or PQ adapter, not
    crystal structure data. ``UnitCell`` holds the atomic basis and lattice
    parameter; this class holds boundary-level geometry (area indices, rotation
    denominator, boundary-plane normal, and sigma-type information) that
    ``UnitCell`` has no concept of.
    """

    basis_mode: Literal["primitive", "supplied"]
    supplied_area_index: int
    primitive_area_index: int
    reduction_index: int
    plane: tuple[int, int, int]
    rotation_denominator: int
    conventional_cell_multiplier: int


@dataclass(frozen=True)
class BoundaryEmbedding:
    """Canonical internal representation produced by every input adapter.

    P and Q are the exact row-wise orientation matrices (None for
    approximate-only paths). R_left and R_right are floating-point rotation
    matrices matching GBMaker's internal convention. exact and coherent flag the
    construction path and interface type. source names the originating format
    ("pq", "csl", "five_dof").
    """
    P: np.ndarray | None
    Q: np.ndarray | None
    R_left: np.ndarray
    R_right: np.ndarray
    exact: bool
    coherent: bool
    source: str
    metadata: PrimitiveCellMetadata | None = None
