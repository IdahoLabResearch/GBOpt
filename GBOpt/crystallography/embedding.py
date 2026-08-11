# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Construct ``BoundaryEmbedding`` objects from normalized crystallographic data.

Functions here accept already-validated crystallographic inputs (scaled rotations, CSL
bases, plane covectors, P/Q matrices) and return ``BoundaryEmbedding`` objects.
Boundary-spec parsing and user-facing validation belong in boundary.py; CSL arithmetic
belongs in csl.py.
"""
from typing import Literal

import numpy as np

from GBOpt.BoundarySpec import (
    BoundaryEmbedding,
    BoundarySpecError,
    BoundarySpecOrthogonalityError,
    PrimitiveCellMetadata,
)
from GBOpt.Utils.integer_linalg import cross_int3, dot_int

from .csl import csl_from_scaled_rotation
from .integer import as_int_array, integer_det3, row_gcd_reduce
from .plane import inplane_area_index, inplane_basis_from_csl
from .pq import canonicalize_pq, canonicalize_pq_paired
from .reduction import gauss_reduce_2d
from .rotation import (
    _scaled_row_images,
    transpose_rotation_convention,
)
from .types import CrystallographyValueError, InPlaneBasis, ScaledRotation


def _as_float_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return matrix as a float ndarray for numerical rotation operations.

    :param matrix: Input matrix to convert to a float array.
    :return: Float ndarray view or copy produced by np.asarray.
    """
    return np.asarray(matrix, dtype=float)


def _as_object_int_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    """Return matrix as an exact object-dtype integer matrix.

    :param matrix: Input matrix to convert to an object-dtype integer array.
    :param name: Name used in validation error messages.
    :return: 3 by 3 exact integer ndarray.
    """
    return as_int_array(matrix, (3, 3), name)


def _csl_inplane(row_rotation: ScaledRotation, plane_int: np.ndarray) -> InPlaneBasis:
    """Build a CSL and return its in-plane basis for a given plane.

    Transposes ``row_rotation`` to column convention, constructs the CSL, and returns
    the in-plane basis for ``plane_int``.

    :param row_rotation: Exact row-convention scaled rotation.
    :param plane_int: Primitive boundary-plane normal as a length-3 integer array.
    :return: In-plane CSL basis for the given plane.
    :raises BoundarySpecError: If the rotation or plane is invalid, or no CSL in-plane
        basis exists for the given inputs.
    """
    try:
        column_rotation = transpose_rotation_convention(row_rotation)
        csl = csl_from_scaled_rotation(column_rotation)
        return inplane_basis_from_csl(csl.basis_hnf, tuple(int(x) for x in plane_int))
    except CrystallographyValueError as exc:
        raise BoundarySpecError(str(exc)) from exc


def _normalize_rotation_rows(matrix: np.ndarray) -> np.ndarray:
    """Return a copy of M with each row normalized to unit length.

    :param matrix: 3 by 3 float matrix whose rows are to be normalized.
    :return: 3 by 3 float matrix with unit-length rows.
    :raises BoundarySpecError: If any row is the zero row (norm == 0).
    """
    matrix_float = _as_float_matrix(matrix)
    norms = np.linalg.norm(matrix_float, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise BoundarySpecError(f"Zero row present in matrix: {matrix}")
    return matrix_float / norms


def _assert_proper_rotation_rows(R_left: np.ndarray, R_right: np.ndarray) -> None:
    """Raise if either rotation matrix is not a proper rotation.

    :param R_left: Left-grain rotation matrix.
    :param R_right: Right-grain rotation matrix.
    :raises BoundarySpecOrthogonalityError: If either matrix is not orthogonal or has
        determinant not equal to ``1``.
    """
    identity = np.eye(3)
    for name, R in (("R_left", R_left), ("R_right", R_right)):
        orthogonal = np.allclose(R @ R.T, identity, atol=1e-10)
        proper = abs(np.linalg.det(R) - 1) < 1e-10
        if not orthogonal:
            raise BoundarySpecOrthogonalityError(
                f"{name} is not an orthogonal rotation matrix (R @ R.T != I)."
            )
        if not proper:
            raise BoundarySpecOrthogonalityError(
                f"{name} is not a proper rotation matrix (det(R) != 1)."
            )


def primitive_metadata(
    *,
    basis_mode: Literal["primitive", "supplied"],
    primitive_area_index: int,
    plane: np.ndarray,
    rotation_denominator: int,
    input_area_index: int | None = None,
    orientation_area_index: int | None = None,
) -> PrimitiveCellMetadata:
    """Build primitive-cell metadata for an exact boundary embedding.

    The metadata separates primitive CSL topology from orientation-row bookkeeping.
    ``primitive_area_index`` is the minimal CSL in-plane area for the exact rotation and
    boundary plane. ``input_area_index`` records the area of caller-provided ``P`` rows
    when available. ``orientation_area_index`` records the area of the returned
    ``BoundaryEmbedding.P`` rows when useful, but it is descriptive only and is not
    required to be an integer multiple of ``primitive_area_index``.

    :param basis_mode: ``"primitive"`` for a primitive CSL embedding or ``"supplied"``
        for caller-supplied ``P``/``Q`` rows. Keyword argument, required.
    :param primitive_area_index: Minimal in-plane CSL area index for the exact rotation
        and boundary plane. Keyword argument, required.
    :param plane: Primitive boundary-plane normal as a length-3 integer array. Keyword
        argument, required.
    :param rotation_denominator: Denominator ``N`` of the exact scaled rotation
        ``R = M / N`` associated with this boundary. Keyword argument, required.
    :param input_area_index: In-plane area index of caller-provided ``P`` rows, when
        available. Must be an integer multiple of ``primitive_area_index`` when
        supplied. Keyword argument, optional, defaults to ``None``.
    :param orientation_area_index: In-plane area index of the returned
        ``BoundaryEmbedding.P`` orientation rows, when useful to report. This value is
        not required to be related by divisibility to ``primitive_area_index``. Keyword
        argument, optional, defaults to ``None``.
    :return: Boundary metadata attached to ``BoundaryEmbedding``.
    :raises BoundarySpecError: If an area index is not positive, if
        ``input_area_index`` is supplied but is not an integer multiple of
        ``primitive_area_index``, or if metadata construction fails validation.
    """
    if basis_mode not in ("primitive", "supplied"):
        raise BoundarySpecError(
            f"basis_mode must be 'primitive' or 'supplied'; got {basis_mode!r}."
        )

    if primitive_area_index <= 0:
        raise BoundarySpecError(
            f"primitive_area_index must be positive; got {primitive_area_index}."
        )

    input_reduction_index = None
    if input_area_index is not None:
        if input_area_index <= 0:
            raise BoundarySpecError(
                f"input_area_index must be positive; got {input_area_index}."
            )
        if input_area_index % primitive_area_index != 0:
            raise BoundarySpecError(
                "input_area_index must be an integer multiple of primitive_area_index "
                "when reporting primitive-cell metadata; got "
                f"{input_area_index=}, {primitive_area_index=}."
            )
        input_reduction_index = input_area_index // primitive_area_index

    if orientation_area_index is not None and orientation_area_index <= 0:
        raise BoundarySpecError(
            f"orientation_area_index must be positive; got {orientation_area_index}."
        )

    h, k, l = (int(plane[0]), int(plane[1]), int(plane[2]))

    return PrimitiveCellMetadata(
        basis_mode=basis_mode,
        input_area_index=input_area_index,
        primitive_area_index=primitive_area_index,
        input_reduction_index=input_reduction_index,
        orientation_area_index=orientation_area_index,
        plane=(h, k, l),
        rotation_denominator=int(rotation_denominator),
        conventional_cell_multiplier=int(2 * primitive_area_index),
    )


def embedding_from_pq(
    P_canon: np.ndarray,
    Q_canon: np.ndarray,
    *,
    source: str,
    metadata=None,
) -> BoundaryEmbedding:
    """Build a ``BoundaryEmbedding`` from canonical P and Q matrices.

    :param P_canon: Canonical left-grain orientation matrix.
    :param Q_canon: Canonical right-grain orientation matrix.
    :param source: String label describing the upstream boundary spec type (e.g.
        ``"pq"``, ``"csl"``), stored on the returned ``BoundaryEmbedding``. Keyword
        argument, required.
    :param metadata: Primitive-cell metadata to attach to the embedding, or ``None`` if
        no metadata is available. Keyword argument, optional, defaults to ``None``.
    :return: ``BoundaryEmbedding`` with ``exact=True``, ``coherent=True``, and
        ``source`` as supplied. ``R_left`` and ``R_right`` are constructed by
        normalizing each row of ``P_canon`` and ``Q_canon`` to unit length respectively.
    """
    R_left = _normalize_rotation_rows(P_canon)
    R_right = _normalize_rotation_rows(Q_canon)
    _assert_proper_rotation_rows(R_left, R_right)
    return BoundaryEmbedding(
        P=P_canon,
        Q=Q_canon,
        R_left=R_left,
        R_right=R_right,
        exact=True,
        coherent=True,
        source=source,
        metadata=metadata,
    )


def embedding_from_rotation_rows(
    R_left: np.ndarray,
    R_right: np.ndarray,
    *,
    source: str,
    coherent: bool = True,
) -> BoundaryEmbedding:
    """Build an approximate ``BoundaryEmbedding`` from proper floating-point rotations.

    This helper is used by approximate construction paths that have no exact ``P``/``Q``
    matrices but still need validated left/right rotation frames.

    :param R_left: Floating-point left-grain rotation matrix.
    :param R_right: Floating-point right-grain rotation matrix.
    :param source: String label describing the upstream boundary spec type, stored on
        the returned ``BoundaryEmbedding``. Keyword argument, required.
    :param coherent: Whether the approximate embedding represents a coherent boundary
        construction. Keyword argument, optional, defaults to ``True``.
    :return: Approximate ``BoundaryEmbedding`` with ``P=None``, ``Q=None``, and
        ``exact=False``.
    :raises BoundarySpecOrthogonalityError: If either rotation matrix is not orthogonal
        or has determinant not equal to ``1``.
    """
    R_left = _as_float_matrix(R_left)
    R_right = _as_float_matrix(R_right)
    _assert_proper_rotation_rows(R_left, R_right)

    return BoundaryEmbedding(
        P=None,
        Q=None,
        R_left=R_left,
        R_right=R_right,
        exact=False,
        coherent=coherent,
        source=source,
    )


def primitive_embedding_from_row_rotation(
    row_rotation: ScaledRotation,
    plane: np.ndarray,
    *,
    source: str,
    input_area_index: int | None = None,
    max_exact_atoms: int | None = None,
) -> BoundaryEmbedding:
    """Build a primitive paired P/Q embedding from a row-convention rotation.

    The CSL is built from the column-convention transpose of ``row_rotation`` because
    ``csl_from_scaled_rotation`` expects column-vector convention. The boundary-normal
    image row ``q0`` is computed with ``allow_inexact=True`` because non-preserving
    planes map to a rational direction that is GCD-reduced to its primitive integer
    representative; the two in-plane image rows ``q1`` and ``q2`` require exact
    divisibility and raise if the CSL membership check fails.

    :param row_rotation: Exact row-convention scaled rotation.
    :param plane: Boundary-plane normal in the reference grain.
    :param source: String label describing the upstream boundary spec type, stored on
        the returned ``BoundaryEmbedding``. Keyword argument, required.
    :param input_area_index: In-plane area index of the caller-provided ``P`` rows, when
        available. Keyword argument, optional, defaults to ``None``.
    :param max_exact_atoms: Upper bound on the primitive in-plane area index. Keyword
        argument, optional, defaults to ``None``.
    :return: Exact coherent ``BoundaryEmbedding`` with primitive-cell metadata.
    :raises BoundarySpecError: If the computed primitive area index exceeds
        ``max_exact_atoms``.
    """
    plane_int = row_gcd_reduce(np.asarray(plane, dtype=object))
    inplane = _csl_inplane(row_rotation, plane_int)

    p1 = inplane.basis[:, 0]
    p2 = inplane.basis[:, 1]
    q0, q1, q2 = _scaled_row_images(
        np.array([plane_int, p1, p2], dtype=object),
        row_rotation,
        allow_inexact=(True, False, False),
    )

    P_int = np.array([plane_int, p1, p2], dtype=object)
    Q_int = np.array([q0, q1, q2], dtype=object)

    primitive_area_index = inplane_area_index(P_int)
    if max_exact_atoms is not None and primitive_area_index > max_exact_atoms:
        raise BoundarySpecError(
            f"Exact in-plane CSL area index ({primitive_area_index}) exceeds "
            f"{max_exact_atoms=}. Use mode='approximate' or increase the limit."
        )

    P_canon, Q_canon = canonicalize_pq_paired(P_int, Q_int)
    orientation_area_index = inplane_area_index(P_canon)

    metadata = primitive_metadata(
        basis_mode="primitive",
        input_area_index=input_area_index,
        primitive_area_index=primitive_area_index,
        orientation_area_index=orientation_area_index,
        plane=plane_int,
        rotation_denominator=int(row_rotation.denominator),
    )

    return embedding_from_pq(P_canon, Q_canon, source=source, metadata=metadata)


def orthogonal_embedding_from_row_rotation_and_plane(
    row_rotation: ScaledRotation,
    plane_normal: np.ndarray,
    *,
    source: str,
    input_area_index: int | None = None,
    max_exact_atoms: int | None = None,
) -> BoundaryEmbedding:
    """Build a BoundaryEmbedding whose P rows are mutually orthogonal.

    Row 0 is the primitive boundary-plane normal. Row 1 is chosen from the in-plane CSL
    basis after Gauss reduction. Row 2 is ``cross(plane_normal, row1)``, giving an
    orthogonal P-frame. The corresponding Q rows are constructed exactly from the
    row-rotation numerator and GCD-reduced as integer directions.

    The returned metadata records both the primitive CSL area and the larger orthogonal
    embedding area when the orthogonal fallback expands the returned cell.

    :param row_rotation: Exact row-convention scaled rotation.
    :param plane_normal: Primitive boundary-plane normal as a length-3 integer array.
    :param source: String label describing the upstream boundary spec type, stored on
        the returned ``BoundaryEmbedding``. Keyword argument, required.
    :param input_area_index: In-plane area index of the caller-provided ``P`` rows, when
        available. Keyword argument, optional, defaults to ``None``.
    :param max_exact_atoms: Upper bound on cell size used for raw in-plane CSL area and
        post-canonicalization determinant checks. Keyword argument, optional, defaults
        to ``None``.
    :return: Exact coherent ``BoundaryEmbedding`` with primitive-cell metadata.
    :raises BoundarySpecError: If the raw squared in-plane CSL cell area exceeds
        ``max_exact_atoms**2``, or if ``max(abs(det(P)), abs(det(Q)))`` exceeds
        ``max_exact_atoms`` after canonicalization.
    """
    plane_int = row_gcd_reduce(np.asarray(plane_normal, dtype=object))
    inplane = _csl_inplane(row_rotation, plane_int)

    v1 = inplane.basis[:, 0]
    v2 = inplane.basis[:, 1]

    primitive_P = np.array([plane_int, v1, v2], dtype=object)
    primitive_area_index = inplane_area_index(primitive_P)

    cross = cross_int3(v1, v2)
    area_sq = dot_int(cross, cross)
    if max_exact_atoms is not None and area_sq > max_exact_atoms**2:
        raise BoundarySpecError(
            f"Exact squared in-plane CSL cell area ({area_sq}) exceeds "
            f"{max_exact_atoms**2=}. Use mode='approximate' or increase the limit."
        )

    r1, _ = gauss_reduce_2d(v1, v2)
    e1 = row_gcd_reduce(r1)
    e2 = row_gcd_reduce(np.array(cross_int3(plane_int, e1), dtype=object))

    M_int = np.asarray(row_rotation.matrix, dtype=object)

    P_int = np.array([plane_int, e1, e2], dtype=object)
    Q_int = np.array(
        [row_gcd_reduce(np.asarray(row, dtype=object) @ M_int) for row in P_int],
        dtype=object,
    )

    P_canon, Q_canon = canonicalize_pq(P_int, Q_int)

    # Validate the returned orientation frames before computing optional metadata
    # from those rows. This preserves BoundarySpecOrthogonalityError for malformed
    # canonical rows.
    R_left = _normalize_rotation_rows(P_canon)
    R_right = _normalize_rotation_rows(Q_canon)
    _assert_proper_rotation_rows(R_left, R_right)

    orientation_area_index = inplane_area_index(P_canon)

    if max_exact_atoms is not None:
        P_check = as_int_array(P_canon, (3, 3), "P_canon")
        Q_check = as_int_array(Q_canon, (3, 3), "Q_canon")
        det_P = abs(integer_det3(P_check))
        det_Q = abs(integer_det3(Q_check))
        if max(det_P, det_Q) > max_exact_atoms:
            raise BoundarySpecError(
                f"CSL supercell exceeds {max_exact_atoms=}: "
                f"|det(P)|={det_P}, |det(Q)|={det_Q}."
            )

    metadata = primitive_metadata(
        basis_mode="primitive",
        input_area_index=input_area_index,
        primitive_area_index=primitive_area_index,
        orientation_area_index=orientation_area_index,
        plane=plane_int,
        rotation_denominator=int(row_rotation.denominator),
    )

    return BoundaryEmbedding(
        P=P_canon,
        Q=Q_canon,
        R_left=R_left,
        R_right=R_right,
        exact=True,
        coherent=True,
        source=source,
        metadata=metadata,
    )


__all__ = [
    "primitive_metadata",
    "embedding_from_pq",
    "embedding_from_rotation_rows",
    "primitive_embedding_from_row_rotation",
    "orthogonal_embedding_from_row_rotation_and_plane",
]
