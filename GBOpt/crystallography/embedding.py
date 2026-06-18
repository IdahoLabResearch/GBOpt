# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Construct BoundaryEmbedding objects from normalized crystallographic data.

Functions here accept already-validated crystallographic inputs (scaled rotations, CSL
bases, plane covectors, P/Q matrices) and return BoundaryEmbedding objects.
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
from GBOpt.Utils.integer_normal_forms import _cross_int3

from .csl import csl_from_scaled_rotation
from .integer import integer_det3, row_gcd_reduce_int
from .plane import inplane_area_index, inplane_basis_from_csl
from .pq import canonicalize_pq, canonicalize_pq_paired
from .reduction import gauss_reduce_2d
from .rotation import scaled_row_image, validate_scaled_rotation_matrix
from .types import CrystallographyValueError, ScaledRotation


def primitive_metadata(
    *,
    basis_mode: Literal["primitive", "supplied"],
    supplied_area_index: int,
    primitive_area_index: int,
    plane: np.ndarray,
    rotation_denominator: int,
) -> PrimitiveCellMetadata:
    """Build primitive-cell metadata for an exact boundary embedding.

    :param basis_mode: ``"primitive"`` or ``"supplied"``.
    :param supplied_area_index: Area index represented by the supplied P rows.
    :param primitive_area_index: Minimal primitive in-plane area index.
    :param plane: Primitive boundary-plane normal.
    :param rotation_denominator: Denominator of the recovered scaled rotation.
    :return: Boundary metadata attached to ``BoundaryEmbedding``.
    :raises BoundarySpecError: If the supplied area is not an integer multiple
        of the primitive area.
    """
    if supplied_area_index % primitive_area_index == 0:
        reduction_index = supplied_area_index // primitive_area_index
    else:
        raise BoundarySpecError(
            "supplied_area_index must be an integer multiple of "
            "primitive_area_index when reporting primitive-cell metadata; "
            f"got supplied_area_index={supplied_area_index}, "
            f"primitive_area_index={primitive_area_index}."
        )
    h, k, l = (int(x) for x in plane)
    return PrimitiveCellMetadata(
        basis_mode=basis_mode,
        supplied_area_index=int(supplied_area_index),
        primitive_area_index=int(primitive_area_index),
        reduction_index=int(reduction_index),
        plane=(h, k, l),
        rotation_denominator=int(rotation_denominator),
        conventional_cell_multiplier=int(2 * primitive_area_index),
    )


def normalize_rotation_rows(M: np.ndarray) -> np.ndarray:
    """Return a copy of M with each row normalized to unit length.

    :param M: 3 by 3 float matrix whose rows are to be normalized.
    :return: 3 by 3 float matrix with unit-length rows.
    """
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def assert_proper_rotation_rows(R_left: np.ndarray, R_right: np.ndarray) -> None:
    """Raise if either rotation matrix is not a proper rotation.

    :param R_left: Left grain rotation matrix.
    :param R_right: Right grain rotation matrix.
    :raises BoundarySpecOrthogonalityError: If either matrix is not orthogonal or has
        determinant not equal to 1.
    """
    for name, R in [("R_left", R_left), ("R_right", R_right)]:
        if not (np.allclose(R @ R.T, np.eye(3), atol=1e-10)
                and abs(np.linalg.det(R) - 1.0) < 1e-10):
            raise BoundarySpecOrthogonalityError(
                f"{name} is not a proper rotation matrix "
                "(R @ R.T != I or det != 1)."
            )


def embedding_from_pq(
    P_canon: np.ndarray,
    Q_canon: np.ndarray,
    *,
    source: str,
    metadata=None,
) -> BoundaryEmbedding:
    """Build a BoundaryEmbedding from canonical P and Q matrices.

    :param P_canon: Canonical left-grain orientation matrix.
    :param Q_canon: Canonical right-grain orientation matrix.
    :param source: Boundary source label stored on the embedding.
    :param metadata: Optional primitive-cell metadata.
    :return: Exact coherent BoundaryEmbedding.
    :raises BoundarySpecError: If the normalized rows are not proper rotations.
    """
    R_left = normalize_rotation_rows(P_canon)
    R_right = normalize_rotation_rows(Q_canon)
    assert_proper_rotation_rows(R_left, R_right)
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


def orthogonal_embedding_from_row_rotation_and_plane(
    row_rotation: ScaledRotation,
    plane_int: np.ndarray,
    *,
    source: str,
    max_exact_atoms: int | None = None,
) -> BoundaryEmbedding:
    """Build an orthogonal BoundaryEmbedding from a row rotation and plane.

    Constructs e1 from the shortest reduced CSL in-plane vector and e2 via
    cross product, making P rows mutually orthogonal. Verifies that e2 is a
    CSL vector before returning an exact embedding; raises if no exact
    orthogonal embedding exists for the given plane and rotation.

    :param row_rotation: Exact row-convention scaled rotation.
    :param plane_int: Primitive boundary-plane normal.
    :param source: Boundary source label stored on the embedding.
    :param max_exact_atoms: Optional guard on cell size.
    :return: Exact coherent BoundaryEmbedding.
    :raises BoundarySpecError: If CSL construction fails, the cell is too
        large, the resulting matrices are not proper rotations, or e2 is
        not a CSL vector.
    """
    try:
        column_rotation = validate_scaled_rotation_matrix(
            np.asarray(row_rotation.M, dtype=object).T,
            N=row_rotation.N,
        )
        fallback_csl = csl_from_scaled_rotation(column_rotation)
        inplane = inplane_basis_from_csl(
            fallback_csl.basis_hnf,
            tuple(int(x) for x in plane_int),
        )
    except CrystallographyValueError as exc:
        raise BoundarySpecError(str(exc)) from exc

    v1 = np.asarray(inplane.basis[:, 0], dtype=float)
    v2 = np.asarray(inplane.basis[:, 1], dtype=float)
    area = np.linalg.norm(np.array(_cross_int3(v1, v2), dtype=object))
    if max_exact_atoms is not None and area > max_exact_atoms:
        raise BoundarySpecError(
            f"Exact in-plane CSL cell area ({area:.1f}) exceeds "
            f"max_exact_atoms={max_exact_atoms}. Use mode='approximate' or "
            "increase the limit."
        )

    r1, _r2 = gauss_reduce_2d(v1, v2)
    e1 = row_gcd_reduce_int(r1)
    e2 = row_gcd_reduce_int(np.array(_cross_int3(plane_int, e1), dtype=object))
    P = np.array([
        plane_int.astype(float),
        e1.astype(float),
        e2.astype(float),
    ])
    M_int = np.asarray(row_rotation.M, dtype=object)
    Q = np.array(
        [row_gcd_reduce_int(np.asarray(P[i], dtype=object) @ M_int)
         for i in range(3)]
    )
    P_canon, Q_canon = canonicalize_pq(P, Q)

    det_P = abs(integer_det3(np.round(P_canon).astype(int)))
    det_Q = abs(integer_det3(np.round(Q_canon).astype(int)))
    if max_exact_atoms is not None and max(det_P, det_Q) > max_exact_atoms:
        raise BoundarySpecError(
            f"CSL supercell exceeds max_exact_atoms={max_exact_atoms}: "
            f"|det(P)|={det_P}, |det(Q)|={det_Q}."
        )

    M_obj = np.asarray(row_rotation.M, dtype=object)
    N = row_rotation.N
    e2_canon = np.round(P_canon[2]).astype(int)
    e2_residual = np.asarray(e2_canon, dtype=object) @ M_obj % N
    if not np.all(e2_residual == 0):
        raise BoundarySpecError(
            f"Orthogonal fallback e2={e2_canon.tolist()} is not a CSL vector "
            f"for plane={plane_int.tolist()} (residual mod {N} = "
            f"{e2_residual.tolist()}). No exact orthogonal embedding exists "
            "for this plane and rotation."
        )

    return embedding_from_pq(P_canon, Q_canon, source=source)


def primitive_embedding_from_row_rotation(
    row_rotation: ScaledRotation,
    plane: np.ndarray,
    *,
    source: str,
    supplied_area_index: int | None = None,
    max_exact_atoms: int | None = None,
) -> BoundaryEmbedding:
    """Build a primitive paired P/Q embedding from a row-convention rotation.

    :param row_rotation: Exact row-convention scaled rotation.
    :param plane: Boundary-plane normal in the reference grain.
    :param source: Boundary source label stored on the embedding.
    :param supplied_area_index: Optional area index of the user-supplied cell.
    :param max_exact_atoms: Optional guard on primitive in-plane area index.
    :return: Exact coherent ``BoundaryEmbedding`` with primitive-cell metadata.
    """
    plane_int = row_gcd_reduce_int(np.asarray(plane, dtype=int)).astype(int)
    try:
        column_rotation = validate_scaled_rotation_matrix(
            np.asarray(row_rotation.M, dtype=object).T,
            N=row_rotation.N,
        )
        csl = csl_from_scaled_rotation(column_rotation)
        inplane = inplane_basis_from_csl(
            csl.basis_hnf,
            tuple(int(x) for x in plane_int),
        )
    except CrystallographyValueError as exc:
        raise BoundarySpecError(str(exc)) from exc

    p1 = np.asarray(inplane.basis[:, 0], dtype=int)
    p2 = np.asarray(inplane.basis[:, 1], dtype=int)
    q0 = scaled_row_image(plane_int, row_rotation, require_divisible=False)
    q1 = scaled_row_image(p1, row_rotation, require_divisible=True)
    q2 = scaled_row_image(p2, row_rotation, require_divisible=True)

    P_raw = np.array([plane_int, p1, p2], dtype=float)
    Q_raw = np.array([q0, q1, q2], dtype=float)
    P_canon, Q_canon = canonicalize_pq_paired(P_raw, Q_raw)
    primitive_area_index = inplane_area_index(P_canon)
    if max_exact_atoms is not None and primitive_area_index > max_exact_atoms:
        raise BoundarySpecError(
            f"Exact in-plane CSL area index ({primitive_area_index}) exceeds "
            f"max_exact_atoms={max_exact_atoms}. Use mode='approximate' or "
            "increase the limit."
        )

    supplied_index = (
        primitive_area_index
        if supplied_area_index is None
        else int(supplied_area_index)
    )
    metadata = primitive_metadata(
        basis_mode="primitive",
        supplied_area_index=supplied_index,
        primitive_area_index=primitive_area_index,
        plane=plane_int,
        rotation_denominator=int(row_rotation.N),
    )

    return embedding_from_pq(P_canon, Q_canon, source=source, metadata=metadata)


__all__ = [
    "primitive_metadata",
    "primitive_embedding_from_row_rotation",
    "embedding_from_pq",
    "orthogonal_embedding_from_row_rotation_and_plane",
]
