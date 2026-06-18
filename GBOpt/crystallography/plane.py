# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Plane-specific crystallographic lattice operations.

Operates on Miller plane covectors and CSL bases to find in-plane lattice
vectors, null-space bases, and supercell enumerations. Consumes raw CSL
basis arrays from csl.py but does not import csl.py directly; the interface
between the two modules is plain numpy arrays. Plane-preservation logic
for scaled rotations also lives here since it requires primitive plane
normalization.
"""


from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from GBOpt.Utils.integer_normal_forms import (
    ExactNormalFormError,
    _dot_int,
    hnf_2d_supercells,
    primitive_integer_null_basis_3d,
)

from .integer import as_int_array, as_int_vector, row_gcd_reduce_int
from .rotation import scaled_row_image
from .types import (
    CrystallographyDivisibilityError,
    CrystallographyNotImplementedError,
    CrystallographyValueError,
    InPlaneBasis,
    Int3,
    ScaledRotation,
)


def primitive_plane(plane_covector: ArrayLike) -> Int3:
    """Return a primitive integer boundary-plane covector.

    :param plane_covector: Integer Miller plane normal ``[h, k, l]``.
    :return: GCD-reduced covector with the original sign convention preserved.
    :raises CrystallographyValueError: If the covector is not length 3, contains
        non-integers, or is the zero vector.
    """
    vec = list(as_int_vector(plane_covector, 3, "plane_covector"))
    gcd_value = math.gcd(*[abs(v) for v in vec])
    if gcd_value == 0:
        raise CrystallographyValueError("plane covector h must not be the zero vector.")
    a, b, c = tuple(value // gcd_value for value in vec)
    return a, b, c


def plane_null_basis(
    plane_int: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a primitive integer basis for the null space of ``plane_int``.

    Finds ``e1``, ``e2`` in Z^3 such that:

    * ``dot(plane_int, e1) == 0`` and ``dot(plane_int, e2) == 0`` (both vectors lie in
        the boundary plane), and
    * ``cross(e1, e2) == plane_int`` (the basis spans the full integer plane lattice
        without gaps).

    The construction applies unimodular column operations to ``[h, k, l]`` until it
    becomes ``[1, 0, 0]``, tracking the transformations in V in GL_3(Z).  Because V is
    unimodular, columns 1 and 2 of V are exactly the primitive null vectors.

    :param plane_int: Primitive boundary-plane normal as an integer 3-vector. Must be
        non-zero and GCD-reduced (``gcd(|h|, |k|, |l|) == 1``). Callers are responsible
        for reducing via ``_row_gcd_reduce`` first.
    :return: ``(e1, e2)`` as float arrays.
    :raises CrystallographyValueError: If ``plane_int`` is the zero vector or not
        primitive.
    """
    vec = np.array(
        [int(plane_int[0]), int(plane_int[1]), int(plane_int[2])], dtype=int
    )
    if not any(vec):
        raise CrystallographyValueError(
            "plane_int must not be the zero vector."
        )
    if math.gcd(math.gcd(abs(int(vec[0])), abs(int(vec[1]))), abs(int(vec[2]))) != 1:
        raise CrystallographyValueError(
            f"plane_int {vec.tolist()} is not primitive "
            "(gcd of components != 1). Call _row_gcd_reduce first."
        )

    try:
        basis = primitive_integer_null_basis_3d(vec)
    except ExactNormalFormError as exc:
        raise CrystallographyValueError(str(exc)) from exc
    return basis[:, 0].astype(float), basis[:, 1].astype(float)


def inplane_area_index(P: np.ndarray) -> int:
    """Return the integer area index of P's in-plane rows in P[0]'s plane.

    Validates that P[1] and P[2] are actually in the plane defined by P[0]
    (i.e., ``dot(P[1], plane) == 0`` and ``dot(P[2], plane) == 0``).  Also
    validates that all entries are close to integers before rounding.

    :param P: 3x3 integer-valued orientation matrix (row 0 = plane normal,
        rows 1-2 = in-plane vectors).
    :return: Positive integer area index.
    :raises CrystallographyValueError: If P[0] is zero, P[1]/P[2] are not in-plane,
        rows are not integer-valued, or the area index is zero.
    """
    P_arr = np.asarray(P, dtype=float)
    if not np.allclose(P_arr, np.round(P_arr), atol=1e-9, rtol=0.0):
        raise CrystallographyValueError(
            "P rows must be integer-valued for area-index computation; "
            f"got non-integer entries in P={P_arr.tolist()}."
        )
    P_int = np.round(P_arr).astype(int)
    plane = row_gcd_reduce_int(P_int[0])
    denom = _dot_int(plane, plane)
    if denom == 0:
        raise CrystallographyValueError(
            "Cannot compute area index for a zero boundary plane.")
    for row_idx in (1, 2):
        proj = _dot_int(P_int[row_idx], plane)
        if proj != 0:
            raise CrystallographyValueError(
                f"P row {row_idx} {P_int[row_idx].tolist()} is not in the boundary "
                f"plane {plane.tolist()} (dot product = {proj}, expected 0). "
                "P[1] and P[2] must be integer lattice vectors lying in the plane "
                "defined by the primitive normal P[0]."
            )
    cross = np.cross(P_int[1], P_int[2])
    numer = abs(_dot_int(cross, plane))
    if numer % denom != 0:
        raise CrystallographyValueError(
            "In-plane rows do not define an integer area index for the boundary plane."
        )
    index = numer // denom
    if index == 0:
        raise CrystallographyValueError(
            "In-plane area index is zero; P[1] and P[2] may be parallel or zero."
        )
    return int(index)


def inplane_basis_from_csl(
    csl_basis: np.ndarray,
    plane_covector: tuple,
    *,
    lattice_metric: np.ndarray | None = None,
) -> InPlaneBasis:
    """Find two CSL vectors lying in an integer boundary plane.

    :param csl_basis: 3 by 3 integer CSL basis.
    :param plane_covector: Integer Miller plane normal ``[h, k, l]`` defining
        ``plane_covector @ v == 0``.
    :param lattice_metric: Reserved non-cubic metric hook; only ``None`` is
        currently supported.
    :return: In-plane CSL basis and diagnostic coefficient data.
    :raises CrystallographyValueError: If inputs are invalid or the projected basis is
        rank deficient.
    """
    _reject_non_cubic_metric(lattice_metric)
    C = as_int_array(csl_basis, (3, 3), "csl_basis")
    plane = primitive_plane(plane_covector)
    h_vec = np.array(plane, dtype=object)
    d = C.T @ h_vec
    coeffs = primitive_integer_null_basis_3d(d)

    basis = C @ coeffs
    basis = _verify_projected_basis(basis)
    residual = h_vec @ basis
    if residual[0] != 0 or residual[1] != 0:
        raise CrystallographyValueError(
            "constructed in-plane basis is not in the plane.")
    return InPlaneBasis(
        basis=basis,
        coefficients=coeffs,
        plane_covector=plane,
    )


def enumerate_inplane_hnf_supercells(
    inplane_basis: np.ndarray,
    index: int,
) -> list[np.ndarray]:
    """Return all index-``n`` supercells of an in-plane CSL basis.

    :param inplane_basis: 3 by 2 integer in-plane basis.
    :param index: Positive supercell index.
    :return: List of 3 by 2 integer supercell bases.
    :raises CrystallographyValueError: If ``inplane_basis`` is not integer-valued.
    """
    basis = as_int_array(inplane_basis, (3, 2), "inplane_basis")
    return [basis @ H for H in hnf_2d_supercells(index)]


def rotation_preserves_plane(
    rotation: ScaledRotation,
    plane: ArrayLike,
    *,
    allow_antiparallel: bool = False,
) -> bool:
    """Return whether a row-convention rotation preserves a boundary plane.

    :param rotation: Row-convention scaled rotation.
    :param plane: Integer boundary-plane covector.
    :param allow_antiparallel: If true, accept the opposite primitive normal.
    :returns: True when ``plane @ M / N`` maps to the same primitive plane.
    :raises CrystallographyValueError: If ``plane`` is invalid.
    """
    plane_int = np.asarray(primitive_plane(plane), dtype=object)

    try:
        image = scaled_row_image(
            plane_int,
            rotation,
            require_divisible=True,
        )
    except CrystallographyDivisibilityError:
        return False

    image_plane = np.asarray(primitive_plane(image), dtype=object)

    if np.array_equal(image_plane, plane_int):
        return True
    if allow_antiparallel and np.array_equal(image_plane, -plane_int):
        return True
    return False


def _reject_non_cubic_metric(metric: np.ndarray | None) -> None:
    """Reject non-cubic lattice metrics reserved for a later extension.

    ``metric`` is intended to represent a future 3 by 3 lattice metric tensor
    for non-cubic crystals. Exact CSL support is currently implemented only
    for the implicit cubic identity metric, so callers must pass ``None``.

    NOTE: This is a temporary guard, and it (and it's companion method in `rotation.py`
    and `quaternion.py`) should be centralized properly when fully implemented.
    """
    if metric is not None:
        raise CrystallographyNotImplementedError(
            "non-cubic lattice metrics are not implemented"
        )


def _verify_projected_basis(basis: np.ndarray) -> np.ndarray:
    """Validate and return a rank-two projected basis.

    :param basis: 3 by 2 integer-valued matrix whose columns are proposed
        in-plane CSL vectors.
    :return: Object-dtype copy of ``basis``.
    :raises CrystallographyValueError: If the two vectors are linearly dependent.
    """
    cross = np.cross(basis[:, 0].astype(int), basis[:, 1].astype(int))
    if not any(cross):
        raise CrystallographyValueError("in-plane CSL vectors are linearly dependent.")
    return basis.astype(object)


__all__ = [
    "primitive_plane",
    "plane_null_basis",
    "inplane_area_index",
    "inplane_basis_from_csl",
    "enumerate_inplane_hnf_supercells",
    "rotation_preserves_plane",
]
