# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Construct ``BoundaryEmbedding`` objects from normalized crystallographic data.

Functions here accept already-validated crystallographic inputs (scaled rotations, CSL
bases, plane covectors, P/Q matrices) and return ``BoundaryEmbedding`` objects.
Boundary-spec parsing and user-facing validation belong in boundary.py; CSL arithmetic
belongs in csl.py.
"""
from __future__ import annotations

import math

import numpy as np

from GBOpt.BoundarySpec import (
    BoundaryEmbedding,
    BoundarySpecError,
    BoundarySpecOrthogonalityError,
    PrimitiveCellMetadata,
)

from .csl import csl_from_scaled_rotation
from .integer import (
    as_int_array,
    as_positive_int,
    cross_int3,
    dot_int,
    integer_det3,
    row_gcd_reduce,
)
from .orientation import validate_orientation_matrix
from .plane import inplane_area_index, inplane_basis_from_csl, rotation_preserves_plane
from .pq import (
    canonicalize_pq_paired,
    recover_exact_row_rotation_from_paired_pq,
)
from .reduction import gauss_reduce_2d
from .rotation import (
    _minimal_integral_row_pair,
    _scaled_row_images,
    transpose_rotation_convention,
)
from .types import (
    CrystallographyError,
    CrystallographyValueError,
    CSLResult,
    InPlaneBasis,
    ScaledRotation,
)


def _validated_exact_orientation_rows(
    matrix: object,
    name: str,
) -> np.ndarray:
    """Return a validated exact integer row-orientation matrix.

    The candidate is converted to a 3 by 3 object-dtype array of Python integers and
    validated entirely with exact integer arithmetic. Each row must be nonzero, all
    row pairs must have exact dot product zero, and the exact determinant must be
    positive.

    Unlike :func:`validate_orientation_matrix`, this helper does not convert the matrix
    to floating point or normalize its rows. It is intended for exact crystallographic
    direction matrices whose components may exceed the floating-point range.

    :param matrix: Candidate 3 by 3 integer-valued row-orientation matrix.
    :param name: Human-readable matrix name used in validation error messages.
    :return: Object-dtype 3 by 3 NumPy array containing Python integers.
    :raises CrystallographyValueError: If ``matrix`` has the wrong shape, contains a
        non-integer entry or zero row, has nonorthogonal rows, or is not right-handed.
    """
    rows = as_int_array(matrix, (3, 3), name)

    for row_index, row in enumerate(rows):
        if all(int(value) == 0 for value in row):
            raise CrystallographyValueError(
                f"{name} row {row_index} must be nonzero."
            )

    for first, second in ((0, 1), (0, 2), (1, 2)):
        exact_dot = dot_int(rows[first], rows[second])
        if exact_dot != 0:
            raise CrystallographyValueError(
                f"{name} rows {first} and {second} have exact dot product "
                f"{exact_dot}; expected 0."
            )

    determinant = integer_det3(rows)
    if determinant <= 0:
        raise CrystallographyValueError(
            f"{name} must be right-handed; exact determinant is {determinant}."
        )

    return rows


def _enforce_primitive_area_index_limit(
    primitive_area_index: int,
    *,
    max_primitive_area_index: int | None,
) -> None:
    """Enforce the primitive in-plane CSL area-index limit.

    :param primitive_area_index: Primitive in-plane CSL area index to check.
    :param max_primitive_area_index: Maximum permitted primitive area index, or ``None``
        to disable the limit. Keyword argument.
    :return: ``None``.
    :raises CrystallographyValueError: If ``max_primitive_area_index`` is not a positive
        integer when supplied.
    :raises BoundarySpecError: If ``primitive_area_index`` exceeds the configured limit.
    """
    if max_primitive_area_index is None:
        return

    limit = as_positive_int(
        max_primitive_area_index,
        "max_primitive_area_index",
    )
    if primitive_area_index > limit:
        raise BoundarySpecError(
            "Primitive CSL area index exceeds "
            f"max_primitive_area_index={limit}: "
            f"primitive_area_index={primitive_area_index}."
        )


def _enforce_pq_determinant_limit(
    P: object,
    Q: object,
    *,
    max_pq_determinant: int | None,
) -> None:
    """Enforce the absolute determinant limit on exact P/Q matrices.

    :param P: Candidate exact left-grain 3 by 3 integer orientation matrix.
    :param Q: Candidate exact right-grain 3 by 3 integer orientation matrix.
    :param max_pq_determinant: Maximum permitted value of ``abs(det(P))`` and
        ``abs(det(Q))``, or ``None`` to disable the limit. Keyword argument.
    :return: ``None``.
    :raises CrystallographyValueError: If the limit is invalid or either matrix fails
        exact 3 by 3 integer validation.
    :raises BoundarySpecError: If either absolute determinant exceeds the configured
        limit.
    """
    if max_pq_determinant is None:
        return

    limit = as_positive_int(
        max_pq_determinant,
        "max_pq_determinant",
    )
    P_int = as_int_array(P, (3, 3), "P")
    Q_int = as_int_array(Q, (3, 3), "Q")
    det_P = abs(integer_det3(P_int))
    det_Q = abs(integer_det3(Q_int))

    if max(det_P, det_Q) > limit:
        raise BoundarySpecError(
            "Exact P/Q determinant exceeds "
            f"max_pq_determinant={limit}: "
            f"|det(P)|={det_P}, |det(Q)|={det_Q}."
        )


def _paired_pq_from_direction_rows(
    direction_rows: np.ndarray,
    row_rotation: ScaledRotation,
    *,
    primitive_area_index: int,
    max_pq_determinant: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build exact paired P/Q orientation matrices from reference directions.

    Each row in ``direction_rows`` is paired with its exact image under
    ``row_rotation``. If necessary, the second row of both matrices is enlarged by the
    smallest common factor that makes the resulting in-plane area index divisible by
    ``primitive_area_index``.

    All orientation-frame validation is performed with exact integer arithmetic.

    :param direction_rows: Three integer-valued reference directions arranged as a 3 by
        3 row matrix.
    :param row_rotation: Exact row-convention scaled rotation.
    :param primitive_area_index: Positive primitive CSL in-plane area index. Keyword
        argument.
    :param max_pq_determinant: Maximum permitted value of ``abs(det(P))`` and
        ``abs(det(Q))``. Keyword argument, optional, defaults to ``None``.
    :return: Exact object-dtype integer matrices ``(P, Q)``.
    :raises CrystallographyValueError: If an argument is malformed, the completed rows
        are not proper exact orientation frames, or exact rotation recovery fails.
    :raises BoundarySpecError: If ``abs(det(P))`` or ``abs(det(Q))`` exceeds
        ``max_pq_determinant``.
    """
    direction_rows = as_int_array(
        direction_rows,
        (3, 3),
        "direction_rows",
    )
    primitive_area_index = as_positive_int(
        primitive_area_index,
        "primitive_area_index",
    )

    paired_rows = [
        _minimal_integral_row_pair(row, row_rotation)
        for row in direction_rows
    ]
    P = np.array([pair[0] for pair in paired_rows], dtype=object)
    Q = np.array([pair[1] for pair in paired_rows], dtype=object)

    input_area_index = inplane_area_index(P)
    area_factor = primitive_area_index // math.gcd(
        input_area_index,
        primitive_area_index,
    )
    if area_factor > 1:
        P[1] *= area_factor
        Q[1] *= area_factor

    P = _validated_exact_orientation_rows(P, "P")
    Q = _validated_exact_orientation_rows(Q, "Q")

    recovered = recover_exact_row_rotation_from_paired_pq(P, Q)
    recovered_numerator = (
        np.asarray(recovered.matrix, dtype=object)
        * row_rotation.denominator
    )
    expected_numerator = (
        np.asarray(row_rotation.matrix, dtype=object)
        * recovered.denominator
    )
    if not np.array_equal(recovered_numerator, expected_numerator):
        raise CrystallographyValueError(
            "Internal error: exact paired P/Q rows changed the recovered "
            "rotation."
        )

    _enforce_pq_determinant_limit(P, Q, max_pq_determinant=max_pq_determinant)

    return P, Q


def _validated_rotation_rows(
    matrix: np.ndarray,
    name: str,
) -> np.ndarray:
    """Return normalized proper rotation rows with embedding-layer exceptions.

    Floating-point row-orientation validation is delegated to
    ``validate_orientation_matrix``. This keeps normalization, shape validation,
    finite-value validation, orthogonality checks, and handedness checks in the
    orientation module while preserving the exception type expected by embedding
    callers.

    :param matrix: Candidate 3 by 3 row-orientation matrix. Rows may have arbitrary
        nonzero magnitudes but must be finite, mutually orthogonal after normalization,
        and right-handed.
    :param name: Human-readable matrix name included in validation error messages.
    :return: Normalized ``float64`` row-orientation matrix with shape ``(3, 3)``.
    :raises BoundarySpecOrthogonalityError: If ``matrix`` has the wrong shape, contains
        non-finite or zero rows, is not orthogonal after row normalization, or is not
        right-handed.
    """
    try:
        return validate_orientation_matrix(matrix, name)
    except CrystallographyValueError as exc:
        message = str(exc)
        if "right-handed" in message:
            message = f"{name} is not a proper rotation matrix: {message}"
        elif "orthogonal" in message:
            message = f"{name} is not an orthogonal rotation matrix: {message}"
        raise BoundarySpecOrthogonalityError(message) from exc


def _csl_from_row_rotation(row_rotation: ScaledRotation) -> CSLResult:
    """Construct the column-convention CSL for a row-convention rotation.

    :param row_rotation: Exact row-convention scaled rotation.
    :return: Canonical CSL constructed from the column-convention transpose of
        ``row_rotation``.
    :raises BoundarySpecError: If rotation conversion or CSL construction rejects the
        supplied rotation.
    """
    try:
        column_rotation = transpose_rotation_convention(row_rotation)
        return csl_from_scaled_rotation(column_rotation)
    except CrystallographyError as exc:
        raise BoundarySpecError(str(exc)) from exc


def _inplane_from_csl(csl: CSLResult, plane_int: np.ndarray) -> InPlaneBasis:
    """Return the in-plane basis of a precomputed CSL.

    ``csl`` must correspond to the column-convention transpose of the row rotation used
    by the eventual embedding constructor. Keeping this operation separate allows an
    exactification caller that already needs the CSL sigma to reuse the same exact CSL
    construction during embedding.

    :param csl: Precomputed CSL in column-vector convention.
    :param plane_int: Primitive boundary-plane normal as a length-3 integer array.
    :return: In-plane CSL basis for ``plane_int``.
    :raises BoundarySpecError: If no valid in-plane CSL basis exists for the supplied
        CSL and plane.
    """
    try:
        return inplane_basis_from_csl(
            csl.basis_hnf,
            tuple(int(value) for value in plane_int),
        )
    except CrystallographyValueError as exc:
        raise BoundarySpecError(str(exc)) from exc


def _inplane_from_row_rotation(row_rotation: ScaledRotation, plane_int: np.ndarray) -> InPlaneBasis:
    """Build a CSL and return its in-plane basis for a given plane.

    :param row_rotation: Exact row-convention scaled rotation.
    :param plane_int: Primitive boundary-plane normal as a length-3 integer array.
    :return: In-plane CSL basis for the given plane.
    :raises BoundarySpecError: If CSL construction or in-plane basis construction fails.
    """
    csl = _csl_from_row_rotation(row_rotation)
    return _inplane_from_csl(csl, plane_int)


def embedding_from_pq(
    P_canon: np.ndarray,
    Q_canon: np.ndarray,
    *,
    source: str,
    metadata: PrimitiveCellMetadata | None = None,
    max_pq_determinant: int | None = None,
) -> BoundaryEmbedding:
    """Build a ``BoundaryEmbedding`` from canonical P and Q matrices.

    :param P_canon: Canonical left-grain orientation matrix.
    :param Q_canon: Canonical right-grain orientation matrix.
    :param source: String label describing the upstream boundary spec type (e.g.
        ``"pq"``, ``"csl"``), stored on the returned ``BoundaryEmbedding``. Keyword
        argument, required.
    :param metadata: Primitive-cell metadata to attach to the embedding, or ``None`` if
        no metadata is available. Keyword argument, optional, defaults to ``None``.
    :param max_pq_determinant: Bounds abs(det(P)) and abs(det(Q)) for both the selected
        exact embedding and the final exactly paired P/Q matrices. Keyword argument,
        optional, defaults to ``None``.
    :return: ``BoundaryEmbedding`` with ``exact=True``, ``coherent=True``, and
        ``source`` as supplied. ``R_left`` and ``R_right`` are constructed by
        normalizing and validating the rows of ``P_canon`` and ``Q_canon``.
    :raises CrystallographyValueError: If ``max_pq_determinant`` is invalid or either
        P/Q matrix fails exact integer validation.
    :raises BoundarySpecError: If either P/Q determinant exceeds ``max_pq_determinant``.
    :raises BoundarySpecOrthogonalityError: If the normalized P or Q rows do not form a
        proper rotation frame.
    """
    _enforce_pq_determinant_limit(
        P_canon,
        Q_canon,
        max_pq_determinant=max_pq_determinant,
    )

    R_left = _validated_rotation_rows(P_canon, "R_left")
    R_right = _validated_rotation_rows(Q_canon, "R_right")

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
    :return: Approximate ``BoundaryEmbedding`` with normalized ``R_left`` and
        ``R_right``, ``P=None``, ``Q=None``, and ``exact=False``.
    :raises BoundarySpecOrthogonalityError: If either matrix is malformed, non-finite,
        degenerate, nonorthogonal after row normalization, or not right-handed.
    """
    R_left = _validated_rotation_rows(R_left, "R_left")
    R_right = _validated_rotation_rows(R_right, "R_right")

    return BoundaryEmbedding(
        P=None,
        Q=None,
        R_left=R_left,
        R_right=R_right,
        exact=False,
        coherent=coherent,
        source=source,
    )


def _primitive_embedding_from_inplane(
    row_rotation: ScaledRotation,
    plane_int: np.ndarray,
    inplane: InPlaneBasis,
    *,
    source: str,
    input_area_index: int | None = None,
    max_primitive_area_index: int | None = None,
    max_pq_determinant: int | None = None,
) -> BoundaryEmbedding:
    """Build a primitive exact embedding from a precomputed in-plane CSL basis.

    ``plane_int`` and ``inplane`` must describe the same primitive boundary plane. The
    boundary-normal image may be rational and is reduced to its primitive integer
    direction. The two in-plane rows are CSL vectors and therefore must have exact
    integer images under ``row_rotation``.

    The minimal in-plane CSL area index is calculated before P/Q canonicalization and
    stored as ``PrimitiveCellMetadata.primitive_area_index``. The area index of the
    returned canonical P rows is stored separately as ``orientation_area_index``.

    :param row_rotation: Exact row-convention scaled rotation.
    :param plane_int: Primitive integer boundary-plane normal in the reference grain.
    :param inplane: In-plane basis derived from the column-convention CSL associated
        with ``row_rotation`` and ``plane_int``.
    :param source: Label identifying the upstream boundary representation. Keyword
        argument, required.
    :param input_area_index: In-plane area index of caller-provided P rows, when
        available. Keyword argument, optional, defaults to ``None``.
    :param max_primitive_area_index: Maximum permitted minimal in-plane CSL area index.
        ``None`` disables this limit. Keyword argument, optional, defaults to ``None``.
    :param max_pq_determinant: Maximum permitted absolute determinant of each returned
        exact P/Q matrix. ``None`` disables this limit. Keyword argument, optional,
        defaults to ``None``.
    :return: Exact coherent primitive ``BoundaryEmbedding`` with primitive-cell
        metadata.
    :raises CrystallographyValueError: If exact integer input, area-index calculation,
        or limit validation fails.
    :raises CrystallographyDivisibilityError: If an in-plane CSL direction does not have
        an exactly integral image under ``row_rotation``.
    :raises BoundarySpecError: If an exact-cell limit is exceeded or the resulting P/Q
        matrices cannot form a valid embedding.
    """
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
    _enforce_primitive_area_index_limit(
        primitive_area_index,
        max_primitive_area_index=max_primitive_area_index,
    )

    P_canon, Q_canon = canonicalize_pq_paired(P_int, Q_int)
    orientation_area_index = inplane_area_index(P_canon)

    metadata = PrimitiveCellMetadata(
        basis_mode="primitive",
        input_area_index=input_area_index,
        primitive_area_index=primitive_area_index,
        orientation_area_index=orientation_area_index,
        plane=tuple(int(value) for value in plane_int),
        rotation_denominator=row_rotation.denominator,
    )

    return embedding_from_pq(
        P_canon,
        Q_canon,
        source=source,
        metadata=metadata,
        max_pq_determinant=max_pq_determinant,
    )


def primitive_embedding_from_row_rotation(
    row_rotation: ScaledRotation,
    plane: np.ndarray,
    *,
    source: str,
    input_area_index: int | None = None,
    max_primitive_area_index: int | None = None,
    max_pq_determinant: int | None = None,
) -> BoundaryEmbedding:
    """Build a primitive paired P/Q embedding from a row-convention rotation.

    The CSL is built from the column-convention transpose of ``row_rotation`` because
    ``csl_from_scaled_rotation`` expects column-vector convention. The boundary-normal
    image row may be rational and is reduced to its primitive integer direction; both
    in-plane rows are CSL vectors and therefore require exact divisibility.

    :param row_rotation: Exact row-convention scaled rotation.
    :param plane: Boundary-plane normal in the reference grain.
    :param source: String label describing the upstream boundary spec type, stored on
        the returned ``BoundaryEmbedding``. Keyword argument, required.
    :param input_area_index: In-plane area index of caller-provided ``P`` rows, when
        available. Keyword argument, optional, defaults to ``None``.
    :param max_primitive_area_index: Bounds the minimal in-plane CSL area index. Keyword
            argument, optional, defaults to ``None``.
    :param max_pq_determinant: Bounds abs(det(P)) and abs(det(Q)) for both the selected
        exact embedding and the final exactly paired P/Q matrices. Keyword argument,
        optional, defaults to ``None``.
    :return: Exact coherent ``BoundaryEmbedding`` with primitive-cell metadata.
    :raises CrystallographyValueError: If the plane, exact integer data, or an exact
        construction limit is malformed.
    :raises BoundarySpecError: If CSL or in-plane construction fails, an exact-cell
        limit is exceeded, or proper P/Q orientation frames cannot be constructed.
    """
    plane_int = row_gcd_reduce(np.asarray(plane, dtype=object))
    inplane = _inplane_from_row_rotation(row_rotation, plane_int)
    return _primitive_embedding_from_inplane(
        row_rotation,
        plane_int,
        inplane,
        source=source,
        input_area_index=input_area_index,
        max_primitive_area_index=max_primitive_area_index,
        max_pq_determinant=max_pq_determinant,
    )


def _orthogonal_embedding_from_inplane(
    row_rotation: ScaledRotation,
    plane_int: np.ndarray,
    inplane: InPlaneBasis,
    *,
    source: str,
    input_area_index: int | None = None,
    max_primitive_area_index: int | None = None,
    max_pq_determinant: int | None = None,
) -> BoundaryEmbedding:
    """Build an orthogonal exact embedding from a precomputed in-plane CSL basis.

    The minimal in-plane CSL cell is used only to determine ``primitive_area_index``. A
    Gauss-reduced CSL direction is selected for the first in-plane orientation row, and
    the second is constructed as its exact cross product with ``plane_int``. This
    produces a mutually orthogonal reference-grain frame. Corresponding right-grain
    directions are constructed from the exact row-rotation numerator and canonicalized
    as a paired P/Q representation.

    The returned orthogonal orientation cell may differ in area from the minimal CSL
    cell. These values are recorded separately as ``primitive_area_index`` and
    ``orientation_area_index``.

    :param row_rotation: Exact row-convention scaled rotation.
    :param plane_int: Primitive integer boundary-plane normal in the reference grain.
    :param inplane: In-plane basis derived from the column-convention CSL associated
        with ``row_rotation`` and ``plane_int``.
    :param source: Label identifying the upstream boundary representation. Keyword
        argument, required.
    :param input_area_index: In-plane area index of caller-provided P rows, when
        available. Keyword argument, optional, defaults to ``None``.
    :param max_primitive_area_index: Maximum permitted minimal in-plane CSL area index.
        ``None`` disables this limit. Keyword argument, optional, defaults to ``None``.
    :param max_pq_determinant: Maximum permitted absolute determinant of each returned
        exact P/Q matrix. ``None`` disables this limit. Keyword argument, optional,
        defaults to ``None``.
    :return: Exact coherent orthogonal ``BoundaryEmbedding`` with primitive-cell
        metadata.
    :raises CrystallographyValueError: If exact integer validation, basis reduction, or
        area-index construction fails.
    :raises BoundarySpecError: If an exact-cell limit is exceeded or the resulting P/Q
        matrices cannot form a valid embedding.
    """
    v1 = inplane.basis[:, 0]
    v2 = inplane.basis[:, 1]

    primitive_P = np.array([plane_int, v1, v2], dtype=object)
    primitive_area_index = inplane_area_index(primitive_P)
    _enforce_primitive_area_index_limit(
        primitive_area_index,
        max_primitive_area_index=max_primitive_area_index,
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

    P_canon, Q_canon = canonicalize_pq_paired(P_int, Q_int)

    orientation_area_index = inplane_area_index(P_canon)

    metadata = PrimitiveCellMetadata(
        basis_mode="primitive",
        input_area_index=input_area_index,
        primitive_area_index=primitive_area_index,
        orientation_area_index=orientation_area_index,
        plane=tuple(int(value) for value in plane_int),
        rotation_denominator=row_rotation.denominator,
    )

    return embedding_from_pq(
        P_canon,
        Q_canon,
        source=source,
        metadata=metadata,
        max_pq_determinant=max_pq_determinant,
    )


def orthogonal_embedding_from_row_rotation_and_plane(
    row_rotation: ScaledRotation,
    plane_normal: np.ndarray,
    *,
    source: str,
    input_area_index: int | None = None,
    max_primitive_area_index: int | None = None,
    max_pq_determinant: int | None = None,
) -> BoundaryEmbedding:
    """Build a ``BoundaryEmbedding`` whose P rows are mutually orthogonal.

    Row 0 is the primitive boundary-plane normal. Row 1 is chosen from a Gauss-reduced
    in-plane CSL basis. Row 2 is ``cross(plane_normal, row1)``, giving an orthogonal
    P-frame. Corresponding Q rows are constructed exactly from the row-rotation
    numerator and GCD-reduced as integer directions.

    The returned metadata records both the primitive CSL area and the larger orthogonal
    embedding area when the orthogonal construction expands the returned cell.

    :param row_rotation: Exact row-convention scaled rotation.
    :param plane_normal: Boundary-plane normal in the reference grain.
    :param source: String label describing the upstream boundary spec type, stored on
        the returned ``BoundaryEmbedding``. Keyword argument, required.
    :param input_area_index: In-plane area index of caller-provided ``P`` rows, when
        available. Keyword argument, optional, defaults to ``None``.
    :param max_primitive_area_index: Bounds the minimal in-plane CSL area index. Keyword
            argument, optional, defaults to ``None``.
    :param max_pq_determinant: Bounds abs(det(P)) and abs(det(Q)) for both the selected
        exact embedding and the final exactly paired P/Q matrices. Keyword argument,
        optional, defaults to ``None``.
    :return: Exact coherent ``BoundaryEmbedding`` with primitive-cell metadata.
    :raises CrystallographyValueError: If the plane, exact integer data, or an exact
        construction limit is malformed.
    :raises BoundarySpecError: If CSL or in-plane construction fails, an exact-cell
        limit is exceeded, or proper P/Q orientation frames cannot be constructed.
    """
    plane_int = row_gcd_reduce(np.asarray(plane_normal, dtype=object))
    inplane = _inplane_from_row_rotation(row_rotation, plane_int)
    return _orthogonal_embedding_from_inplane(
        row_rotation,
        plane_int,
        inplane,
        source=source,
        input_area_index=input_area_index,
        max_primitive_area_index=max_primitive_area_index,
        max_pq_determinant=max_pq_determinant,
    )


def _exact_embedding_from_precomputed_csl(
    row_rotation: ScaledRotation,
    plane_int: np.ndarray,
    csl: CSLResult,
    *,
    source: str,
    input_area_index: int | None = None,
    max_primitive_area_index: int | None = None,
    max_pq_determinant: int | None = None,
) -> BoundaryEmbedding:
    """Select an exact embedding path using an already-constructed CSL.

    ``csl`` must have been constructed from the column-convention transpose of
    ``row_rotation``. The in-plane basis is computed once and reused if a
    plane-preserving primitive construction must fall back to the orthogonal path.

    :param row_rotation: Exact row-convention scaled rotation.
    :param plane_int: Primitive boundary-plane normal in the reference grain.
    :param csl: CSL constructed from the column-convention transpose of
        ``row_rotation``.
    :param source: Label identifying the upstream boundary representation. Keyword
        argument, required.
    :param input_area_index: In-plane area index of caller-supplied ``P`` rows, when
        available. Keyword argument, optional, defaults to ``None``.
    :param max_primitive_area_index: Bounds the minimal in-plane CSL area index. Keyword
            argument, optional, defaults to ``None``.
    :param max_pq_determinant: Bounds abs(det(P)) and abs(det(Q)) for both the selected
        exact embedding and the final exactly paired P/Q matrices. Keyword argument,
        optional, defaults to ``None``.
    :return: Exact coherent ``BoundaryEmbedding`` constructed through the primitive path
        when possible, otherwise through the orthogonal path.
    :raises CrystallographyValueError: If exact plane, area, or P/Q construction data is
        invalid.
    :raises BoundarySpecError: If in-plane construction fails, an exact-cell limit is
        exceeded, or neither embedding path can construct proper orientation frames.
    """
    inplane = _inplane_from_csl(csl, plane_int)

    if rotation_preserves_plane(
        row_rotation,
        plane_int,
        allow_antiparallel=True
    ):
        try:
            return _primitive_embedding_from_inplane(
                row_rotation,
                plane_int,
                inplane,
                source=source,
                input_area_index=input_area_index,
                max_primitive_area_index=max_primitive_area_index,
                max_pq_determinant=max_pq_determinant,
            )
        except BoundarySpecOrthogonalityError:
            pass

    return _orthogonal_embedding_from_inplane(
        row_rotation,
        plane_int,
        inplane,
        source=source,
        input_area_index=input_area_index,
        max_primitive_area_index=max_primitive_area_index,
        max_pq_determinant=max_pq_determinant,
    )


def exact_embedding_from_row_rotation_and_plane(
    row_rotation: ScaledRotation,
    plane: np.ndarray,
    *,
    source: str,
    input_area_index: int | None = None,
    max_primitive_area_index: int | None = None,
    max_pq_determinant: int | None = None,
) -> BoundaryEmbedding:
    """Select an exact embedding path for a row rotation and boundary plane.

    The CSL and its in-plane basis are each constructed once. A primitive embedding is
    attempted when ``row_rotation`` preserves the supplied plane. If the primitive rows
    do not form proper orthogonal orientation frames, construction falls back to the
    orthogonal path while reusing the same in-plane basis. Rotations that do not
    preserve the plane use the orthogonal path directly.

    Cell-size errors raised by either construction path are propagated rather than
    causing a fallback.

    :param row_rotation: Exact row-convention scaled rotation.
    :param plane: Integer-valued boundary-plane normal in the reference grain.
    :param source: Label identifying the upstream boundary representation. Keyword
        argument, required.
    :param input_area_index: In-plane area index of caller-supplied ``P`` rows, when
        available. Keyword argument, optional, defaults to ``None``.
    :param max_primitive_area_index: Bounds the minimal in-plane CSL area index. Keyword
            argument, optional, defaults to ``None``.
    :param max_pq_determinant: Bounds abs(det(P)) and abs(det(Q)) for both the selected
        exact embedding and the final exactly paired P/Q matrices. Keyword argument,
        optional, defaults to ``None``.
    :return: Exact coherent ``BoundaryEmbedding`` constructed through the primitive path
        when possible, otherwise through the orthogonal path.
    :raises CrystallographyValueError: If the plane, exact integer data, or an exact
        construction limit is malformed.
    :raises BoundarySpecError: If CSL or in-plane construction fails, an exact-cell
        limit is exceeded, or proper P/Q orientation frames cannot be constructed.
    """
    plane_int = row_gcd_reduce(np.asarray(plane, dtype=object))
    csl = _csl_from_row_rotation(row_rotation)
    return _exact_embedding_from_precomputed_csl(
        row_rotation,
        plane_int,
        csl,
        source=source,
        input_area_index=input_area_index,
        max_primitive_area_index=max_primitive_area_index,
        max_pq_determinant=max_pq_determinant,
    )


__all__ = [
    "embedding_from_pq",
    "embedding_from_rotation_rows",
    "primitive_embedding_from_row_rotation",
    "orthogonal_embedding_from_row_rotation_and_plane",
    "exact_embedding_from_row_rotation_and_plane",
]
