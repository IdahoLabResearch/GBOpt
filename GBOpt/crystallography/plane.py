# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Plane-specific crystallographic lattice operations.

Operates on Miller plane covectors and CSL bases to find in-plane lattice vectors and
null-space bases. Consumes raw CSL basis arrays from csl.py but does not import csl.py
directly; the interface between the two modules is plain numpy arrays.
Plane-preservation logic for scaled rotations also lives here since it requires
primitive plane normalization.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from GBOpt.Utils.integer_normal_forms import (
    ExactNormalFormError,
    primitive_integer_null_basis_3d,
)

from ._guards import _require_cubic
from .integer import as_int_array, as_int_vector, cross_int3, dot_int, row_gcd_reduce
from .rotation import scaled_row_image
from .types import (
    CrystallographyDivisibilityError,
    CrystallographyValueError,
    InPlaneBasis,
    Int3,
    ScaledRotation,
)


def primitive_plane(plane_covector: ArrayLike) -> Int3:
    """Return a primitive integer boundary-plane covector.

    :param plane_covector: Integer Miller plane normal ``[h, k, l]``.
    :return: GCD-reduced covector with the original sign convention preserved.
    :raises CrystallographyValueError: If the covector is the zero vector.
    """
    vec = as_int_array(plane_covector, (3,), "plane_covector")
    if not any(vec):
        raise CrystallographyValueError("plane_covector must not be the zero vector.")
    reduced = row_gcd_reduce(vec)
    h, k, l = tuple(int(v) for v in reduced)
    return h, k, l


def plane_null_basis(
    plane_int: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a primitive integer basis for the null space of ``plane_int``.

    Finds ``e1``, ``e2`` in Z^3 such that:

    * ``dot(plane_int, e1) == 0`` and ``dot(plane_int, e2) == 0`` (both vectors lie in
        the boundary plane), and
    * ``cross(e1, e2) == plane_int`` (the basis spans the full integer plane lattice
        without gaps).

    Delegates to ``primitive_integer_null_basis_3d`` and returns its two null basis
    columns as exact integer object arrays.

    :param plane_int: Primitive boundary-plane normal as an integer 3-vector. Must be
        nonzero and GCD-reduced (``gcd(|h|, |k|, |l|) == 1``). Callers are responsible
        for reducing via ``row_gcd_reduce`` first.
    :return: ``(e1, e2)`` as object arrays.
    :raises CrystallographyValueError: If ``plane_int`` is the zero vector or not
        primitive.
    """
    plane_arr = as_int_vector(plane_int, 3, "plane_int")
    if not any(plane_arr):
        raise CrystallographyValueError("plane_int must not be the zero vector.")
    if math.gcd(*map(abs, plane_arr)) != 1:
        raise CrystallographyValueError(
            f"plane_int {plane_arr} is not primitive "
            "(gcd of components != 1). Call row_gcd_reduce first."
        )

    try:
        basis = primitive_integer_null_basis_3d(plane_arr)
    except ExactNormalFormError as exc:
        raise CrystallographyValueError(str(exc)) from exc
    return (
        basis[:, 0].astype(object, copy=False),
        basis[:, 1].astype(object, copy=False),
    )


def inplane_area_index(matrix: ArrayLike) -> int:
    """Return the integer area index of ``P``'s in-plane rows in ``P[0]``'s plane.

    The area index is computed as ``dot(cross(matrix[1], matrix[2]), plane_normal) /
    dot(plane_normal, plane_normal)``. It equals the area of the in-plane parallelogram
    measured in units of the primitive plane-normal length.

    :param matrix: 3 by 3 integer-valued orientation matrix, where row 0 is the plane
        normal and rows 1-2 are in-plane vectors.
    :return: Positive integer area index.
    :raises CrystallographyValueError: If ``matrix[0]`` is zero, ``matrix[1]`` or
        ``matrix[2]`` is not in the boundary plane, the projected area is not divisible
        by the plane norm squared, or the area index is zero.
    """
    int_matrix = as_int_array(matrix, (3, 3), "matrix")

    plane_normal = row_gcd_reduce(int_matrix[0])
    plane_norm_sq = dot_int(plane_normal, plane_normal)
    if plane_norm_sq == 0:
        raise CrystallographyValueError(
            "Cannot compute area index for a zero boundary plane."
        )

    for row_idx, row in enumerate(int_matrix[1:], start=1):
        proj = dot_int(row, plane_normal)
        if proj != 0:
            raise CrystallographyValueError(
                f"Matrix row {row_idx} {row.tolist()} is not in the "
                f"boundary plane {plane_normal.tolist()} (dot product = {proj}, "
                "expected 0). matrix[1] and matrix[2] must be integer lattice vectors "
                "lying in the plane defined by the primitive normal matrix[0]."
            )

    cross = cross_int3(int_matrix[1], int_matrix[2])
    projected_area = abs(dot_int(cross, plane_normal))
    if projected_area % plane_norm_sq != 0:
        raise CrystallographyValueError(
            "In-plane rows do not define an integer area index for the boundary plane."
        )

    index = projected_area // plane_norm_sq
    if index == 0:
        raise CrystallographyValueError(
            "In-plane area index is zero; matrix[1] and matrix[2] may be parallel or "
            "zero."
        )
    return int(index)


def inplane_basis_from_csl(
    csl_basis: np.ndarray,
    plane_covector: ArrayLike,
    *,
    lattice_metric: np.ndarray | None = None,
) -> InPlaneBasis:
    """Find two CSL vectors lying in an integer boundary plane.

    Projects the plane covector onto the CSL column basis to obtain integer coordinates,
    computes the null space of those coordinates to find CSL column combinations whose
    3D images lie in the plane, then reconstructs the full 3D vectors.

    :param csl_basis: 3 by 3 integer CSL basis.
    :param plane_covector: Integer Miller plane normal ``[h, k, l]`` defining
        ``plane_covector @ v == 0``.
    :param lattice_metric: Reserved non-cubic metric hook; only ``None`` is currently
        supported. Keyword argument, optional, defaults to ``None``.
    :return: In-plane CSL basis and diagnostic coefficient data.
    :raises CrystallographyValueError: If projected null-basis construction fails, the
        constructed in-plane CSL vectors are linearly dependent, or the constructed
        basis is not in the plane.
    :raises CrystallographyNotImplementedError: If ``lattice_metric`` is not ``None``.
    """
    _require_cubic(lattice_metric)
    int_basis = as_int_array(csl_basis, (3, 3), "csl_basis")
    plane_prim = primitive_plane(plane_covector)
    plane_cov_obj = np.array(plane_prim, dtype=object)
    csl_projections = int_basis.T @ plane_cov_obj
    try:
        null_coeffs = primitive_integer_null_basis_3d(csl_projections)
    except ExactNormalFormError as exc:
        raise CrystallographyValueError(str(exc)) from exc

    basis = (int_basis @ null_coeffs).astype(object, copy=False)

    cross = cross_int3(basis[:, 0], basis[:, 1])
    if not any(cross):
        raise CrystallographyValueError("in-plane CSL vectors are linearly dependent.")

    if np.any(plane_cov_obj @ basis != 0):
        raise CrystallographyValueError(
            "Constructed in-plane basis is not in the plane."
        )
    return InPlaneBasis(
        basis=basis,
        coefficients=null_coeffs,
        plane_covector=plane_prim,
    )


def rotation_preserves_plane(
    rotation: ScaledRotation,
    plane: ArrayLike,
    *,
    allow_antiparallel: bool = False,
) -> bool:
    """Return whether a row-convention rotation preserves a boundary plane.

    The rotation preserves the plane when ``plane @ M / N`` is exactly integer-valued
    and GCD-reduces to the same primitive covector as ``plane``, where ``M`` and ``N``
    are the numerator matrix and denominator of ``rotation``.

    :param rotation: Row-convention scaled rotation.
    :param plane: Integer boundary-plane covector.
    :param allow_antiparallel: If ``True``, accept the opposite primitive normal.
        Keyword argument, optional, defaults to ``False``.
    :return: ``True`` when the rotation maps the plane to itself, or to its opposite if
        ``allow_antiparallel`` is ``True``.
    :raises CrystallographyValueError: If ``plane`` is not a valid nonzero integer
        three-vector or if an exact rotation validation failure occurs.
    """
    plane_int = np.asarray(primitive_plane(plane), dtype=object)

    try:
        image = scaled_row_image(
            plane_int,
            rotation,
            allow_inexact=False,
        )
    except CrystallographyDivisibilityError:
        return False

    image_plane = primitive_plane(image)

    target = tuple(int(v) for v in plane_int)
    opposite = tuple(-v for v in target)
    return image_plane == target or (allow_antiparallel and image_plane == opposite)


__all__ = [
    "primitive_plane",
    "plane_null_basis",
    "inplane_area_index",
    "inplane_basis_from_csl",
    "rotation_preserves_plane",
]
