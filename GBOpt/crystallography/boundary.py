# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""User-facing adapters from boundary specs to ``BoundaryEmbedding`` objects.

Parses and validates boundary specs beyond dataclass validation, calls lower-level
crystallography utilities, and returns ``BoundaryEmbedding`` objects. This module is
orchestration only: CSL arithmetic belongs in ``csl.py``, plane operations in
``plane.py``, P/Q canonicalization in ``pq.py``, and embedding construction in
``embedding.py``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from GBOpt.BoundarySpec import (
    BoundaryEmbedding,
    BoundarySpecError,
    BoundarySpecOrthogonalityError,
    CSLApproxSpec,
    CSLExactSpec,
    FiveDOFSpec,
    PQSpec,
    PrimitiveCellMetadata,
)

from ._limits import (
    DEFAULT_MAX_PQ_DETERMINANT,
    DEFAULT_MAX_PRIMITIVE_AREA_INDEX,
)
from .csl import csl_from_scaled_rotation
from .embedding import (
    _exact_embedding_from_precomputed_csl,
    embedding_from_pq,
    embedding_from_rotation_rows,
    orthogonal_embedding_from_row_rotation_and_plane,
    primitive_embedding_from_row_rotation,
)
from .integer import cross_int3, row_gcd_reduce
from .orientation import orientation_matrices_from_five_dof
from .plane import inplane_area_index, plane_null_basis
from .pq import (
    canonicalize_pq_paired,
    recover_exact_row_rotation_from_paired_pq,
)
from .quaternion import quaternion_to_scaled_rotation
from .rotation import transpose_rotation_convention
from .types import CrystallographyError, CrystallographyValueError


def pq_spec_to_embedding(
    spec: PQSpec,
    *,
    max_primitive_area_index: int | None = None,
    max_pq_determinant: int | None = None,
) -> BoundaryEmbedding:
    """Convert a validated ``PQSpec`` to a ``BoundaryEmbedding``.

    ``spec.basis_mode`` controls how the P/Q matrices are interpreted.

    In ``"primitive"`` mode, the paired rows must recover an exact row-convention scaled
    rotation. The adapter reconstructs the primitive in-plane CSL embedding for that
    rotation and boundary plane. If that primitive representation cannot form proper
    orthogonal orientation rows, an exact orthogonal embedding is used.

    In ``"supplied"`` mode, the caller's paired basis rows are canonicalized without
    primitive-cell reconstruction. Metadata is attached when an exact row rotation can
    be recovered. An exact supplied-mode embedding does not imply that rotation recovery
    succeeded; metadata is ``None`` when it did not.

    :param spec: Validated ``PQSpec`` instance.
    :param max_primitive_area_index: Maximum permitted minimal in-plane CSL area index
        during primitive reconstruction. This limit is not applied in ``"supplied"``
        mode. Keyword argument, optional, defaults to ``None``.
    :param max_pq_determinant: Maximum permitted absolute determinant of each returned
        exact P/Q matrix. Keyword argument, optional, defaults to ``None``.
    :return: Canonical embedding with ``exact=True``, ``coherent=True``, and
        ``source="pq"``.
    :raises BoundarySpecError: If primitive mode cannot recover an exact paired
        rotation, if supplied-mode P/Q rows do not form proper rotations after row
        normalization, if an exact-cell limit is invalid, or if an exact-cell limit is
        exceeded.
    """
    P_int = np.asarray(spec.P, dtype=object)
    Q_int = np.asarray(spec.Q, dtype=object)

    if spec.basis_mode == "primitive":
        try:
            row_rotation = recover_exact_row_rotation_from_paired_pq(P_int, Q_int)
        except CrystallographyValueError as exc:
            raise BoundarySpecError(
                "basis_mode='primitive' requires paired P/Q rows that recover an exact "
                "proper row rotation. Use basis_mode='supplied' when P and Q should be "
                "used as supplied basis rows instead."
            ) from exc

        try:
            input_area_index = inplane_area_index(P_int)
            plane_int = row_gcd_reduce(P_int[0])
        except CrystallographyError as exc:
            raise BoundarySpecError(str(exc)) from exc

        try:
            try:
                return primitive_embedding_from_row_rotation(
                    row_rotation,
                    plane_int,
                    source="pq",
                    input_area_index=input_area_index,
                    max_primitive_area_index=max_primitive_area_index,
                    max_pq_determinant=max_pq_determinant,
                )
            except BoundarySpecOrthogonalityError:
                return orthogonal_embedding_from_row_rotation_and_plane(
                    row_rotation,
                    plane_int,
                    source="pq",
                    input_area_index=input_area_index,
                    max_primitive_area_index=max_primitive_area_index,
                    max_pq_determinant=max_pq_determinant,
                )
        except CrystallographyError as exc:
            raise BoundarySpecError(str(exc)) from exc

    # basis_mode == "supplied"
    # doccheck: ignore=DOC115[CrystallographyValueError]
    #   PQSpec validation guarantees exact nonsingular 3-by-3 integer matrices with no
    #   zero rows
    P_canon, Q_canon = canonicalize_pq_paired(P_int, Q_int)

    metadata = None
    try:
        row_rotation = recover_exact_row_rotation_from_paired_pq(P_int, Q_int)
    except CrystallographyValueError:
        row_rotation = None

    if row_rotation is not None:
        try:
            input_area_index = inplane_area_index(P_int)
            orientation_area_index = inplane_area_index(P_canon)
            plane = tuple(int(value) for value in row_gcd_reduce(P_canon[0]))
        except CrystallographyError as exc:
            raise BoundarySpecError(str(exc)) from exc

        metadata = PrimitiveCellMetadata(
            basis_mode="supplied",
            input_area_index=input_area_index,
            primitive_area_index=orientation_area_index,
            orientation_area_index=orientation_area_index,
            plane=plane,
            rotation_denominator=row_rotation.denominator,
        )

    try:
        # For basis_mode="supplied", max_pq_determinant applies to the returned P/Q
        # matrices; max_primitive_area_index does not apply because supplied mode
        # intentionally does not replace the supplied rows with a reconstructed
        # primitive CSL embedding.
        return embedding_from_pq(
            P_canon,
            Q_canon,
            source="pq",
            metadata=metadata,
            max_pq_determinant=max_pq_determinant,
        )
    except BoundarySpecOrthogonalityError as exc:
        raise BoundarySpecError(
            "R_left or R_right derived from P/Q is not a proper rotation matrix. Ensure "
            "P/Q rows are mutually orthogonal integer Miller directions."
        ) from exc
    except CrystallographyValueError as exc:
        raise BoundarySpecError(str(exc)) from exc


def csl_exact_spec_to_embedding(
    spec: CSLExactSpec,
    *,
    max_primitive_area_index: int = DEFAULT_MAX_PRIMITIVE_AREA_INDEX,
    max_pq_determinant: int = DEFAULT_MAX_PQ_DETERMINANT,
) -> BoundaryEmbedding:
    """Convert a validated exact CSL specification to an exact embedding.

    The specification's integer quaternion is converted to an exact row-convention
    scaled rotation. Its column-convention transpose is then used to construct the CSL.
    When ``spec.sigma`` is supplied, it is checked against the Sigma value derived from
    that CSL.

    Embedding-path selection reuses the already-constructed CSL. A primitive embedding
    is attempted for a plane-preserving rotation; otherwise, or when the primitive rows
    are not orthogonal, an exact orthogonal embedding is selected.

    :param spec: Validated exact CSL boundary specification.
    :param max_primitive_area_index: Maximum permitted minimal in-plane CSL area index.
        Keyword argument, optional, defaults to ``DEFAULT_MAX_PRIMITIVE_AREA_INDEX``.
    :param max_pq_determinant: Maximum permitted absolute determinant of each returned
        exact P/Q matrix. Keyword argument, optional, defaults to
        ``DEFAULT_MAX_PQ_DETERMINANT``.
    :return: Exact coherent ``BoundaryEmbedding`` with ``source="csl"`` and
        primitive-cell metadata.
    :raises BoundarySpecError: If quaternion or CSL construction fails, ``spec.sigma``
        does not match the derived Sigma value, exact embedding construction fails, or
        an exact-cell limit is exceeded.
    """
    try:
        row_rotation = quaternion_to_scaled_rotation(tuple(spec.quat))
        column_rotation = transpose_rotation_convention(row_rotation)
        csl = csl_from_scaled_rotation(column_rotation)
    except CrystallographyError as exc:
        raise BoundarySpecError(str(exc)) from exc

    if spec.sigma is not None and csl.sigma != spec.sigma:
        raise BoundarySpecError(
            f"Sigma mismatch: quaternion {list(row_rotation.quaternion)} "
            f"gives csl_sigma={csl.sigma}, but spec.sigma={spec.sigma} "
            "was provided."
        )

    # doccheck: ignore=DOC115[CrystallographyValueError]
    #   spec.plane is a validated nonzero integer three-vector
    plane_int = row_gcd_reduce(
        np.asarray(spec.plane, dtype=object)
    )

    try:
        return _exact_embedding_from_precomputed_csl(
            row_rotation,
            plane_int,
            csl,
            source="csl",
            max_primitive_area_index=max_primitive_area_index,
            max_pq_determinant=max_pq_determinant,
        )
    except CrystallographyError as exc:
        raise BoundarySpecError(str(exc)) from exc


def csl_approx_spec_to_embedding(spec: CSLApproxSpec) -> BoundaryEmbedding:
    """Convert a ``CSLApproxSpec`` to an approximate ``BoundaryEmbedding``.

    Floating-point ``R_left`` and ``R_right`` matrices are constructed from the
    specified boundary plane and axis-angle misorientation. ``P`` and ``Q`` are ``None``
    because this path does not construct exact integer orientation matrices.

    ``R_left`` is formed with the unit boundary-plane normal as its first row, an
    integer null-basis direction as its second row, and their cross-product complement
    as its third row. The right-grain orientation is then computed as ``R_left @
    R_mis``.

    :param spec: Validated approximate CSL boundary specification.
    :return: ``BoundaryEmbedding`` with ``exact=False``, ``coherent=False``, and
        ``source="csl"``.
    :raises BoundarySpecError: If the validated plane cannot be converted into a proper
        approximate orientation embedding.
    """
    plane = np.asarray(spec.plane, dtype=float)
    plane_unit = plane / np.linalg.norm(plane)

    axis = np.asarray(spec.axis, dtype=float)
    axis_unit = axis / np.linalg.norm(axis)
    R_mis = Rotation.from_rotvec(axis_unit * np.deg2rad(spec.angle_deg)).as_matrix()

    try:
        plane_int = row_gcd_reduce(np.asarray(spec.plane, dtype=object))
        e1, _unused = plane_null_basis(plane_int)
        e2 = row_gcd_reduce(
            np.asarray(cross_int3(plane_int, e1), dtype=object)
        )
    except CrystallographyError as exc:
        raise BoundarySpecError(str(exc)) from exc

    e1_float = e1.astype(float)
    e2_float = e2.astype(float)

    R_left = np.array(
        [
            plane_unit,
            e1_float / np.linalg.norm(e1_float),
            e2_float / np.linalg.norm(e2_float),
        ]
    )
    R_right = R_left @ R_mis

    return embedding_from_rotation_rows(R_left, R_right, source="csl", coherent=False)


def five_dof_spec_to_embedding(spec: FiveDOFSpec) -> BoundaryEmbedding:
    """Convert a validated ``FiveDOFSpec`` to an approximate embedding.

    The validated five-DOF parameters are converted to floating-point left- and
    right-grain row-orientation matrices by ``orientation_matrices_from_five_dof``.
    Construction and final validation of the embedding are then delegated to
    ``embedding_from_rotation_rows``.

    This adapter intentionally does not attempt exactification. Rationalization of
    five-DOF values into exact integer P/Q matrices belongs to the separate
    exactification path.

    :param spec: Validated five-DOF boundary specification containing ``[alpha, beta,
        gamma, theta, phi]`` in radians.
    :return: Approximate ``BoundaryEmbedding`` with ``P=None``, ``Q=None``,
        ``exact=False``, ``coherent=False``, and ``source="five_dof"``.
    :raises BoundarySpecError: If the five-DOF values cannot be converted into proper
        floating-point orientation matrices.
    """
    try:
        R_left, R_right = orientation_matrices_from_five_dof(spec.params)
    except CrystallographyValueError as exc:
        raise BoundarySpecError(
            f"Invalid five-DOF orientation: {exc}"
        ) from exc

    return embedding_from_rotation_rows(
        R_left,
        R_right,
        source="five_dof",
        coherent=False,
    )


def primitive_bicrystal_atom_count(
    embedding: BoundaryEmbedding,
    atoms_per_conventional_cell: int,
) -> int:
    """Return the primitive boundary-defining bicrystal atom count.

    The count is separate from the expanded ``GBMaker`` simulation cell size and equals
    ``2 * primitive_area_index * atoms_per_conventional_cell``.

    :param embedding: Exact embedding with primitive-cell metadata.
    :param atoms_per_conventional_cell: Number of atoms in one conventional unit cell
        for the target structure.
    :return: Total atom count across both grains in the primitive boundary-defining
        bicrystal.
    :raises BoundarySpecError: If metadata is missing or ``atoms_per_conventional_cell``
        is not a positive integer.
    """
    if embedding.metadata is None:
        raise BoundarySpecError(
            "BoundaryEmbedding has no primitive-cell metadata to report."
        )
    if (
        isinstance(atoms_per_conventional_cell, (bool, np.bool_))
        or not isinstance(atoms_per_conventional_cell, (int, np.integer))
        or atoms_per_conventional_cell <= 0
    ):
        raise BoundarySpecError(
            "atoms_per_conventional_cell should be an integer, got "
            f"{atoms_per_conventional_cell}."
        )
    atoms = int(atoms_per_conventional_cell)

    return int(embedding.metadata.conventional_cell_multiplier * atoms)


__all__ = [
    "pq_spec_to_embedding",
    "csl_exact_spec_to_embedding",
    "csl_approx_spec_to_embedding",
    "five_dof_spec_to_embedding",
    "primitive_bicrystal_atom_count",
]
