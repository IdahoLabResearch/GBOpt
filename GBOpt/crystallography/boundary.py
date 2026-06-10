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
)
from GBOpt.Utils.integer_linalg import cross_int3

from .csl import csl_from_scaled_rotation
from .embedding import (
    embedding_from_pq,
    embedding_from_rotation_rows,
    exact_embedding_from_row_rotation_and_plane,
    orthogonal_embedding_from_row_rotation_and_plane,
    primitive_embedding_from_row_rotation,
    primitive_metadata,
)
from .integer import row_gcd_reduce
from .orientation import orientation_matrices_from_five_dof
from .plane import inplane_area_index, plane_null_basis
from .pq import (
    canonicalize_pq_paired,
    recover_exact_row_rotation_from_paired_pq,
)
from .quaternion import quaternion_to_scaled_rotation
from .types import CrystallographyValueError


def pq_spec_to_embedding(spec: PQSpec) -> BoundaryEmbedding:
    """Convert a validated PQSpec to a BoundaryEmbedding.

    ``spec.basis_mode`` controls how P/Q are interpreted:

    * ``"primitive"`` requires paired P/Q rows that recover an exact row-convention
      scaled rotation. The adapter attempts to rebuild the primitive in-plane CSL
      embedding for that rotation and boundary plane. If the primitive in-plane CSL
      basis cannot be represented as proper orthogonal GBMaker rotation rows, an exact
      orthogonal embedding is used instead.
    * ``"supplied"`` treats P/Q as the caller's supplied paired basis rows. Row
      correspondence is preserved during canonicalization. Metadata is attached when an
      exact row rotation can be recovered from the paired rows. In supplied mode,
      ``exact=True`` means the returned P/Q define exact integer orientation rows and
      proper left/right rotations. It does not imply that an exact paired row rotation
      could be recovered; metadata is ``None`` when recovery is unavailable.

    :param spec: Validated ``PQSpec`` instance.
    :return: Canonical embedding with ``exact=True``, ``coherent=True``, and
        ``source="pq"``.
    :raises BoundarySpecError: If primitive mode cannot recover an exact paired
        rotation, or if supplied-mode P/Q rows do not form proper rotations after row
        normalization.
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

        input_area_index = inplane_area_index(P_int)
        plane_int = row_gcd_reduce(P_int[0])

        try:
            return primitive_embedding_from_row_rotation(
                row_rotation,
                plane_int,
                source="pq",
                input_area_index=input_area_index,
            )
        except BoundarySpecOrthogonalityError:
            return orthogonal_embedding_from_row_rotation_and_plane(
                row_rotation,
                plane_int,
                source="pq",
                input_area_index=input_area_index,
            )

    # basis_mode == "supplied"
    P_canon, Q_canon = canonicalize_pq_paired(P_int, Q_int)

    metadata = None
    try:
        row_rotation = recover_exact_row_rotation_from_paired_pq(P_int, Q_int)
    except CrystallographyValueError:
        row_rotation = None

    if row_rotation is not None:
        input_area_index = inplane_area_index(P_int)
        orientation_area_index = inplane_area_index(P_canon)

        metadata = primitive_metadata(
            basis_mode="supplied",
            input_area_index=input_area_index,
            primitive_area_index=orientation_area_index,
            orientation_area_index=orientation_area_index,
            plane=row_gcd_reduce(P_canon[0]),
            rotation_denominator=int(row_rotation.denominator),
        )

    try:
        return embedding_from_pq(
            P_canon,
            Q_canon,
            source="pq",
            metadata=metadata,
        )
    except BoundarySpecOrthogonalityError as exc:
        raise BoundarySpecError(
            "R_left or R_right derived from P/Q is not a proper rotation matrix. Ensure "
            "P/Q rows are mutually orthogonal integer Miller directions."
        ) from exc


def csl_exact_spec_to_embedding(
    spec: CSLExactSpec, *, max_exact_atoms: int = 10_000
) -> BoundaryEmbedding:
    """Convert a validated ``CSLExactSpec`` to a ``BoundaryEmbedding``.

    Assumes cubic symmetry.

    Embedding path selection: if the rotation preserves the boundary plane, a primitive
    paired embedding is attempted first via ``primitive_embedding_from_row_rotation``.
    If that raises ``BoundarySpecOrthogonalityError``, or if the rotation does not
    preserve the plane, the orthogonal fallback
    ``orthogonal_embedding_from_row_rotation_and_plane`` is used instead.

    :param spec: A ``CSLExactSpec`` instance.
    :param max_exact_atoms: Cell-size guard passed to the primitive and orthogonal
        embedding constructors. Keyword argument, optional, defaults to ``10_000``.
    :return: ``BoundaryEmbedding`` with ``exact=True``, ``coherent=True``, and
        ``source="csl"``.
    :raises BoundarySpecError: If quaternion conversion fails, or if the provided
        ``sigma`` does not match the CSL sigma derived from the quaternion.
    """
    try:
        rot = quaternion_to_scaled_rotation(tuple(spec.quat))
    except CrystallographyValueError as exc:
        raise BoundarySpecError(str(exc)) from exc

    if spec.sigma is not None:
        quat_int = rot.quaternion
        csl_sigma = csl_from_scaled_rotation(rot).sigma
        if csl_sigma != int(spec.sigma):
            raise BoundarySpecError(
                f"Sigma mismatch: quaternion "
                f"{list(quat_int)} "  # type: ignore[ty:invalid-argument-type]
                f"gives {csl_sigma=}, but {spec.sigma=} was provided."
            )

    return exact_embedding_from_row_rotation_and_plane(
        rot,
        spec.plane,
        source="csl",
        max_exact_atoms=max_exact_atoms,
    )


def csl_approx_spec_to_embedding(spec: CSLApproxSpec) -> BoundaryEmbedding:
    """Convert a ``CSLApproxSpec`` to a ``BoundaryEmbedding`` using the approximate
    path.

    Constructs floating-point ``R_left`` and ``R_right`` from the given plane and
    axis/angle misorientation. ``P`` and ``Q`` are set to ``None`` because no exact
    integer matrices are available.

    ``R_left`` is built so that its first row is the unit boundary-plane normal. The
    remaining two rows are an integer null-basis direction of the plane and its
    cross-product complement, giving a proper rotation. ``R_right = R_left @ R_mis``,
    where ``R_mis`` is the rotation about the given axis by ``angle_deg``.

    :param spec: A ``CSLApproxSpec`` instance.
    :return: ``BoundaryEmbedding`` with ``exact=False``, ``coherent=False``, and
        ``source="csl"``.
    """
    plane = np.asarray(spec.plane, dtype=float)
    plane_unit = plane / np.linalg.norm(plane)

    axis = np.asarray(spec.axis, dtype=float)
    axis_unit = axis / np.linalg.norm(axis)
    R_mis = Rotation.from_rotvec(axis_unit * np.deg2rad(spec.angle_deg)).as_matrix()

    plane_int = row_gcd_reduce(np.asarray(spec.plane, dtype=object))
    e1, _unused = plane_null_basis(plane_int)
    e2 = row_gcd_reduce(np.asarray(cross_int3(plane_int, e1), dtype=object))

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

    The boundary adapter translates the validated five-DOF specification into
    floating-point row-orientation matrices through
    :func:`orientation_matrices_from_five_dof`, then delegates ``BoundaryEmbedding``
    construction and final rotation validation to :func:`embedding_from_rotation_rows`.

    Exactification is intentionally not attempted here. Five-DOF rationalization into
    exact integer P/Q matrices belongs to the separate exactification path.

    :param spec: Validated five-DOF boundary specification containing ``[alpha, beta,
        gamma, theta, phi]`` in radians.
    :return: Approximate ``BoundaryEmbedding`` with ``P=None``, ``Q=None``,
        ``exact=False``, ``coherent=False``, and ``source="five_dof"``.
    :raises BoundarySpecError: If the five-DOF values cannot be translated into proper
        floating-point orientation matrices.
    :raises BoundarySpecOrthogonalityError: If the translated left or right rows do not
        form a proper orientation matrix during embedding construction.
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
