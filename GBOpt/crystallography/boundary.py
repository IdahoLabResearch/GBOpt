# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""User-facing adapters from boundary specs to BoundaryEmbedding objects.

Parses and validates boundary specs beyond dataclass validation, calls
lower-level crystallography utilities, and returns BoundaryEmbedding objects.
Orchestration only: CSL arithmetic belongs in csl.py, plane operations in
plane.py, P/Q canonicalization in pq.py, and embedding construction in
embedding.py. User-facing warnings such as the primitive-fallback warning
also live here.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from GBOpt.BoundarySpec import (
    BoundaryEmbedding,
    BoundarySpecError,
    BoundarySpecOrthogonalityError,
    PQSpec,
)
from GBOpt.Utils.integer_normal_forms import _cross_int3

from .csl import csl_from_scaled_rotation
from .embedding import (
    orthogonal_embedding_from_row_rotation_and_plane,
    primitive_embedding_from_row_rotation,
    primitive_metadata,
)
from .integer import row_gcd_reduce_int
from .plane import inplane_area_index, plane_null_basis
from .pq import (
    canonicalize_pq,
    canonicalize_pq_paired,
    recover_exact_row_rotation_from_paired_pq,
)
from .quaternion import quaternion_to_scaled_rotation
from .rotation import validate_scaled_rotation_matrix
from .types import CrystallographyValueError


def pq_spec_to_embedding(spec: PQSpec) -> BoundaryEmbedding:
    """Convert a validated PQSpec to a BoundaryEmbedding.

    ``validated`` means ``PQSpec.__post_init__`` has already converted the
    user input to finite, non-singular 3 by 3 matrices and checked
    ``basis_mode``. This adapter performs the exact P/Q canonicalization and
    rotation checks that require the exact utility layer.

    :param spec: Validated ``PQSpec`` instance.
    :returns: Canonical embedding with ``exact=True``, ``coherent=True``,
        ``source="pq"``. P and Q are in canonical form. R_left and R_right
        are derived by normalizing each row of canonical P and Q to unit length.
        Equivalent PQSpecs (differing only by row scaling, sign convention, or
        in-plane basis choice) always produce identical BoundaryEmbeddings.
    :raises BoundarySpecError: If P or Q rows are not integer-valued, produce
        a zero row after canonicalization, or do not form a proper rotation
        matrix after row-normalization.
    """
    P_raw = np.asarray(spec.P, dtype=float)
    Q_raw = np.asarray(spec.Q, dtype=float)
    row_rotation = None
    if spec.basis_mode == "primitive":
        try:
            row_rotation = recover_exact_row_rotation_from_paired_pq(P_raw, Q_raw)
        except CrystallographyValueError as exc:
            # Existing public PQSpec examples often provide orientation rows
            # rather than paired supercell rows (for example P=I and scaled Q
            # direction rows).  Those cannot define A=inv(P)@Q as a proper
            # rotation, so preserve the legacy supplied-basis behavior.
            _warn_pq_primitive_fallback(spec, exc)
            row_rotation = None
        if row_rotation is not None:
            try:
                supplied_area_index = inplane_area_index(P_raw)
                return primitive_embedding_from_row_rotation(
                    row_rotation,
                    row_gcd_reduce_int(np.round(P_raw[0]).astype(int)),
                    source="pq",
                    supplied_area_index=supplied_area_index,
                )
            except BoundarySpecError as exc:
                _warn_pq_primitive_fallback(spec, exc)

    if spec.basis_mode == "supplied" or row_rotation is not None:
        P_canon, Q_canon = canonicalize_pq_paired(P_raw, Q_raw)
    else:
        P_canon, Q_canon = canonicalize_pq(P_raw, Q_raw)
    R_left = P_canon / np.linalg.norm(P_canon, axis=1, keepdims=True)
    R_right = Q_canon / np.linalg.norm(Q_canon, axis=1, keepdims=True)
    for r_name, R in [("R_left", R_left), ("R_right", R_right)]:
        if not (np.allclose(R @ R.T, np.eye(3), atol=1e-10)
                and abs(np.linalg.det(R) - 1.0) < 1e-10):
            raise BoundarySpecError(
                f"{r_name} derived from P/Q is not a proper rotation matrix "
                "(R @ R.T != I or det != 1). Ensure P/Q rows are mutually "
                "orthogonal integer Miller directions."
            )
    metadata = None
    if row_rotation is None:
        try:
            row_rotation = recover_exact_row_rotation_from_paired_pq(P_raw, Q_raw)
        except CrystallographyValueError:
            row_rotation = None
    if row_rotation is not None:
        supplied_area_index = inplane_area_index(P_canon)
        metadata = primitive_metadata(
            basis_mode="supplied",
            supplied_area_index=supplied_area_index,
            primitive_area_index=supplied_area_index,
            plane=row_gcd_reduce_int(np.round(P_canon[0]).astype(int)),
            rotation_denominator=int(row_rotation.N),
        )

    return BoundaryEmbedding(
        P=P_canon,
        Q=Q_canon,
        R_left=R_left,
        R_right=R_right,
        exact=True,
        coherent=True,
        source="pq",
        metadata=metadata,
    )


def csl_spec_to_embedding(spec, max_exact_atoms: int = 10_000) -> BoundaryEmbedding:
    """Convert a validated CSLExactSpec to a BoundaryEmbedding.

    **How P and Q are constructed.**  In GBMaker's convention each row of a
    grain's orientation matrix records which crystal Miller direction aligns
    with the corresponding lab axis: row 0 = lab x (boundary normal), row 1 =
    lab y, row 2 = lab z.

    For grain 1 we fix the boundary normal (``plane``) as row 0 and fill rows
    1-2 with the two cross-product null-basis vectors of that plane (see
    ``plane_null_basis``).  For grain 2 each row is obtained by applying the
    misorientation matrix M_int to the corresponding integer row of P and
    GCD-reducing the result::

        Q[row i] = gcd_reduce(P[row i] @ M_int)

    where ``M_int`` and ``N`` are carried from the exact scaled rotation.  This
    formula is equivalent to rotating each lab axis from grain 1's crystal frame
    into grain 2's crystal frame -- exactly what R_right encodes.  After
    ``canonicalize_pq`` the resulting matrices are identical to what a ``PQSpec``
    with the same boundary would produce, enabling the cross-format round-trip
    test.

    :param spec: A ``CSLExactSpec`` instance (quat is required).
    :param max_exact_atoms: Cell-size guard. Raises ``BoundarySpecError`` if
        the in-plane CSL cell would be larger than this.
    :return: ``BoundaryEmbedding`` with ``exact=True``, ``coherent=True``,
        ``source="csl"``.
    :raises BoundarySpecError: On invalid quaternion, sigma mismatch, missing
        CSL for the given plane, or cell too large.
    """
    if spec.quat is None:
        raise BoundarySpecError("CSLExactSpec.quat is required.")

    try:
        rot = quaternion_to_scaled_rotation(tuple(spec.quat))
        csl = csl_from_scaled_rotation(rot)
    except CrystallographyValueError as exc:
        raise BoundarySpecError(str(exc)) from exc
    quat_int = rot.quaternion if rot.quaternion is not None else tuple(spec.quat)

    if spec.sigma is not None and csl.sigma != int(spec.sigma):
        raise BoundarySpecError(
            f"Sigma mismatch: quaternion {list(quat_int)} "
            f"gives sigma={csl.sigma}, but sigma={spec.sigma} was provided."
        )

    plane_int = row_gcd_reduce_int(np.asarray(spec.plane, dtype=object))
    try:
        row_rotation = validate_scaled_rotation_matrix(
            rot.M,
            N=rot.N,
            reduce_common_factor=True,
        )
    except CrystallographyValueError as exc:
        raise BoundarySpecError(str(exc)) from exc

    plane_row = np.asarray(plane_int, dtype=object)
    image = plane_row @ np.asarray(row_rotation.M, dtype=object)
    preserves_plane = (
        all(int(value) % row_rotation.N == 0 for value in image)
        and np.array_equal(
            row_gcd_reduce_int(
                np.array([int(value) // row_rotation.N for value in image], dtype=int)
            ),
            plane_int,
        )
    )
    if preserves_plane:
        primitive_embedding = None
        try:
            primitive_embedding = primitive_embedding_from_row_rotation(
                row_rotation,
                plane_int,
                source="csl",
                max_exact_atoms=max_exact_atoms,
            )
        except BoundarySpecOrthogonalityError:
            # Some plane-preserving rotations (e.g. Sigma 3 [111]) have a
            # primitive in-plane CSL basis that is not an orthogonal lab y/z
            # pair.  GBMaker's exact path requires proper row-orthogonal
            # rotation matrices, so those cases fall through to the orthogonal
            # construction below.
            primitive_embedding = None
        if primitive_embedding is not None:
            return primitive_embedding

    return orthogonal_embedding_from_row_rotation_and_plane(
        row_rotation,
        plane_int,
        source="csl",
        max_exact_atoms=max_exact_atoms
    )


def csl_approx_spec_to_embedding(spec) -> BoundaryEmbedding:
    """Convert a CSLApproxSpec to a BoundaryEmbedding using the approximate path.

    Constructs floating-point R_left and R_right from the given plane and
    axis/angle misorientation.  P and Q are set to None (no exact integer
    matrices are available).

    R_left is built so that its first row is the unit boundary-plane normal.
    The remaining two rows are completed via Gram-Schmidt using the two
    non-dominant axis-aligned unit vectors, giving a proper rotation.
    R_right = R_left @ R_mis, where R_mis is the rotation about the given
    axis by angle_deg.

    :param spec: A ``CSLApproxSpec`` instance.
    :return: ``BoundaryEmbedding`` with ``exact=False``, ``coherent=True``,
        ``source="csl"``.
    """
    plane = np.asarray(spec.plane, dtype=float)
    plane_unit = plane / np.linalg.norm(plane)

    axis = np.asarray(spec.axis, dtype=float)
    axis_unit = axis / np.linalg.norm(axis)
    angle_rad = float(spec.angle_deg) * np.pi / 180.0
    R_mis = Rotation.from_rotvec(axis_unit * angle_rad).as_matrix()

    # Build R_left: row 0 = plane unit normal; rows 1-2 = orthogonal in-plane
    # directions.  e2 = plane x e1 is orthogonal to both by construction.
    plane_int = row_gcd_reduce_int(np.asarray(plane, dtype=int))
    e1, _ = plane_null_basis(plane_int)
    e1 = row_gcd_reduce_int(e1)
    e2 = row_gcd_reduce_int(np.array(_cross_int3(plane_int, e1), dtype=object))
    e1_unit = e1.astype(float) / np.linalg.norm(e1)
    e2_unit = e2.astype(float) / np.linalg.norm(e2)

    R_left = np.array([plane_unit, e1_unit, e2_unit])
    R_right = R_left @ R_mis

    return BoundaryEmbedding(
        P=None,
        Q=None,
        R_left=R_left,
        R_right=R_right,
        exact=False,
        coherent=True,
        source="csl",
    )


def primitive_bicrystal_atom_count(
    embedding: BoundaryEmbedding,
    atoms_per_conventional_cell: int,
) -> int:
    """Return the primitive boundary-defining bicrystal atom count.

    The count is separate from the expanded GBMaker simulation cell size:
    ``2 * primitive_area_index * atoms_per_conventional_cell``.

    :param embedding: Exact embedding with primitive-cell metadata.
    :param atoms_per_conventional_cell: Number of atoms in one conventional
        unit cell for the target structure.
    :return: Primitive bicrystal atom count.
    :raises BoundarySpecError: If metadata is missing or the atom count is not
        a positive integer.
    """
    if embedding.metadata is None:
        raise BoundarySpecError(
            "BoundaryEmbedding has no primitive-cell metadata to report."
        )
    atoms = int(atoms_per_conventional_cell)
    if atoms <= 0 or atoms != atoms_per_conventional_cell:
        raise BoundarySpecError(
            "atoms_per_conventional_cell must be a positive integer."
        )
    return int(embedding.metadata.conventional_cell_multiplier * atoms)


def _warn_pq_primitive_fallback(spec: PQSpec, reason: Exception) -> None:
    """Warn that primitive PQ reconstruction failed before supplied fallback.

    :param spec: PQ specification that requested primitive reconstruction.
    :param reason: Error that caused fallback to supplied-basis canonicalization.
    """
    P_list = np.asarray(spec.P, dtype=float).tolist()
    Q_list = np.asarray(spec.Q, dtype=float).tolist()
    warnings.warn(
        "PQSpec with basis_mode='primitive' could not reconstruct a primitive "
        "in-plane CSL basis; falling back to supplied-basis canonicalization. "
        f"P={P_list}, Q={Q_list}. Reason: {reason}",
        UserWarning,
        stacklevel=3,
    )


__all__ = [
    "pq_spec_to_embedding",
    "csl_spec_to_embedding",
    "csl_approx_spec_to_embedding",
    "primitive_bicrystal_atom_count",
]
