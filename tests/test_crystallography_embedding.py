# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import numpy as np
import pytest

from GBOpt.BoundarySpec import (
    BoundaryEmbedding,
    BoundarySpecError,
    BoundarySpecOrthogonalityError,
)
from GBOpt.crystallography.embedding import (
    embedding_from_pq,
    orthogonal_embedding_from_row_rotation_and_plane,
    primitive_embedding_from_row_rotation,
    primitive_metadata,
)

# ---------------------------------------------------------------------------
# primitive_metadata
# ---------------------------------------------------------------------------


def test_primitive_metadata_valid_returns_metadata():
    result = primitive_metadata(
        basis_mode="primitive",
        supplied_area_index=10,
        primitive_area_index=5,
        plane=np.array([0, 0, 1]),
        rotation_denominator=5,
    )
    assert result.basis_mode == "primitive"
    assert result.supplied_area_index == 10
    assert result.primitive_area_index == 5
    assert result.reduction_index == 2
    assert result.plane == (0, 0, 1)
    assert result.rotation_denominator == 5
    assert result.conventional_cell_multiplier == 10


def test_primitive_metadata_supplied_mode():
    result = primitive_metadata(
        basis_mode="supplied",
        supplied_area_index=5,
        primitive_area_index=5,
        plane=np.array([0, 0, 1]),
        rotation_denominator=5,
    )
    assert result.basis_mode == "supplied"
    assert result.reduction_index == 1


def test_primitive_metadata_requires_divisible_area_indices():
    with pytest.raises(BoundarySpecError, match="integer multiple"):
        primitive_metadata(
            basis_mode="primitive",
            supplied_area_index=7,
            primitive_area_index=5,
            plane=np.array([0, 0, 1]),
            rotation_denominator=10,
        )


def test_primitive_metadata_conventional_cell_multiplier_is_twice_primitive():
    result = primitive_metadata(
        basis_mode="primitive",
        supplied_area_index=5,
        primitive_area_index=5,
        plane=np.array([1, 0, 0]),
        rotation_denominator=5,
    )
    assert result.conventional_cell_multiplier == 2 * result.primitive_area_index


# ---------------------------------------------------------------------------
# embedding_from_pq
# ---------------------------------------------------------------------------

def test_embedding_from_pq_returns_boundary_embedding():
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    Q = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    result = embedding_from_pq(P, Q, source="pq")
    assert isinstance(result, BoundaryEmbedding)


def test_embedding_from_pq_preserves_P_and_Q():
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    Q = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    result = embedding_from_pq(P, Q, source="pq")

    assert result.P is not None
    assert result.Q is not None
    np.testing.assert_array_equal(result.P, P)
    np.testing.assert_array_equal(result.Q, Q)


def test_embedding_from_pq_flags():
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    Q = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    result = embedding_from_pq(P, Q, source="pq")
    assert result.exact is True
    assert result.coherent is True
    assert result.source == "pq"
    assert result.metadata is None


def test_embedding_from_pq_R_left_R_right_are_normalized():
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    Q = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    result = embedding_from_pq(P, Q, source="pq")
    np.testing.assert_allclose(
        np.linalg.norm(result.R_left, axis=1), np.ones(3), atol=1e-12
    )
    np.testing.assert_allclose(
        np.linalg.norm(result.R_right, axis=1), np.ones(3), atol=1e-12
    )


def test_embedding_from_pq_non_orthogonal_raises():
    P = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
    Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    with pytest.raises(BoundarySpecError):
        embedding_from_pq(P, Q, source="pq")


def test_embedding_from_pq_passes_metadata():
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    Q = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    meta = primitive_metadata(
        basis_mode="primitive",
        supplied_area_index=1,
        primitive_area_index=1,
        plane=np.array([0, 0, 1]),
        rotation_denominator=1,
    )
    result = embedding_from_pq(P, Q, source="pq", metadata=meta)
    assert result.metadata is meta


# ---------------------------------------------------------------------------
# primitive_embedding_from_row_rotation
# ---------------------------------------------------------------------------

def test_primitive_embedding_exact_and_coherent(sigma5_53deg_rotation):
    plane = np.array([0, 0, 1])
    result = primitive_embedding_from_row_rotation(
        sigma5_53deg_rotation, plane, source="csl"
    )
    assert result.exact is True
    assert result.coherent is True


def test_primitive_embedding_source(sigma5_53deg_rotation):
    plane = np.array([0, 0, 1])
    result = primitive_embedding_from_row_rotation(
        sigma5_53deg_rotation, plane, source="csl"
    )
    assert result.source == "csl"


def test_primitive_embedding_proper_rotations(sigma5_53deg_rotation):
    plane = np.array([0, 0, 1])
    result = primitive_embedding_from_row_rotation(
        sigma5_53deg_rotation, plane, source="csl"
    )
    for R in (result.R_left, result.R_right):
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


def test_primitive_embedding_has_metadata(sigma5_53deg_rotation):
    plane = np.array([0, 0, 1])
    result = primitive_embedding_from_row_rotation(
        sigma5_53deg_rotation, plane, source="csl"
    )
    assert result.metadata is not None
    assert result.metadata.basis_mode == "primitive"


def test_primitive_embedding_max_exact_atoms_raises(sigma5_53deg_rotation):
    plane = np.array([0, 0, 1])
    with pytest.raises(BoundarySpecError, match="max_exact_atoms"):
        primitive_embedding_from_row_rotation(
            sigma5_53deg_rotation, plane, source="csl", max_exact_atoms=1
        )


def test_primitive_embedding_orthogonality_error_raised_for_sigma3_111(
    sigma3_111_rotation,
):
    # Sigma3 [111] plane is known to produce a non-orthogonal primitive basis,
    # triggering BoundarySpecOrthogonalityError
    plane = np.array([1, 1, 1])
    with pytest.raises(BoundarySpecOrthogonalityError):
        primitive_embedding_from_row_rotation(
            sigma3_111_rotation, plane, source="csl"
        )


# ---------------------------------------------------------------------------
# orthogonal_embedding_from_row_rotation_and_plane
# ---------------------------------------------------------------------------

def test_orthogonal_embedding_exact_and_coherent(sigma5_53deg_rotation):
    plane = np.array([1, 0, 0])
    result = orthogonal_embedding_from_row_rotation_and_plane(
        sigma5_53deg_rotation, plane, source="csl"
    )
    assert result.exact is True
    assert result.coherent is True


def test_orthogonal_embedding_proper_rotations(sigma5_53deg_rotation):
    plane = np.array([1, 0, 0])
    result = orthogonal_embedding_from_row_rotation_and_plane(
        sigma5_53deg_rotation, plane, source="csl"
    )
    for R in (result.R_left, result.R_right):
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


def test_orthogonal_embedding_sigma3_111_proper_rotations(sigma3_111_rotation):
    # Sigma3 [111] is the case that requires the orthogonal fallback
    plane = np.array([1, 1, 1])
    result = orthogonal_embedding_from_row_rotation_and_plane(
        sigma3_111_rotation, plane, source="csl"
    )
    for R in (result.R_left, result.R_right):
        np.testing.assert_allclose(
            R @ R.T, np.eye(3), atol=1e-10,
            err_msg="R is not orthogonal for Sigma3 [111] boundary"
        )
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


def test_orthogonal_embedding_max_exact_atoms_raises(sigma5_53deg_rotation):
    plane = np.array([1, 0, 0])
    with pytest.raises(BoundarySpecError, match="max_exact_atoms"):
        orthogonal_embedding_from_row_rotation_and_plane(
            sigma5_53deg_rotation, plane, source="csl", max_exact_atoms=1
        )


def test_orthogonal_embedding_e2_is_csl_member(sigma5_53deg_rotation):
    """e2 = cross(plane, e1) must be a CSL vector for exact=True to be valid."""
    plane = np.array([1, 0, 0])
    result = orthogonal_embedding_from_row_rotation_and_plane(
        sigma5_53deg_rotation, plane, source="csl"
    )
    assert result.P is not None
    e2 = np.round(result.P[2]).astype(int)
    M = np.asarray(sigma5_53deg_rotation.M, dtype=object)
    N = sigma5_53deg_rotation.N
    residual = M @ np.asarray(e2, dtype=object) % N
    np.testing.assert_array_equal(
        residual, np.zeros(3, dtype=object),
        err_msg=f"e2={e2} is not a CSL vector: residual mod {N} = {residual}",
    )


def test_orthogonal_embedding_sigma3_111_e2_is_csl_member(sigma3_111_rotation):
    """e2 = cross(plane, e1) is a CSL vector for Sigma3 [111]; embedding is exact=True."""
    plane = np.array([1, 1, 1])
    result = orthogonal_embedding_from_row_rotation_and_plane(
        sigma3_111_rotation, plane, source="csl"
    )
    assert result.exact is True
    assert result.P is not None
    e2_canon = np.round(result.P[2]).astype(int)
    M_obj = np.asarray(sigma3_111_rotation.M, dtype=object)
    N = sigma3_111_rotation.N
    residual = np.asarray(e2_canon, dtype=object) @ M_obj % N
    np.testing.assert_array_equal(
        residual, np.zeros(3, dtype=object),
        err_msg=f"e2={e2_canon} is not a CSL vector for Sigma3 [111]: residual mod {N} = {residual}",
    )


def test_orthogonal_embedding_raises_when_e2_not_csl_member(sigma5_53deg_rotation):
    """If e2 is not a CSL vector the function must raise BoundarySpecError."""
    from unittest.mock import patch

    from GBOpt.crystallography import pq

    plane = np.array([1, 0, 0])
    # [1, 1, 0] is not a CSL vector for sigma5_53deg_rotation on plane [1,0,0]
    non_csl_e2 = np.array([1., 1., 0.])

    original_canonicalize = pq.canonicalize_pq

    def patched_canonicalize(P, Q):
        P_c, Q_c = original_canonicalize(P, Q)
        P_c = P_c.copy()
        P_c[2] = non_csl_e2
        return P_c, Q_c

    with patch(
        "GBOpt.crystallography.embedding.canonicalize_pq",
        side_effect=patched_canonicalize,
    ):
        with pytest.raises(BoundarySpecError, match="not a CSL vector"):
            orthogonal_embedding_from_row_rotation_and_plane(
                sigma5_53deg_rotation, plane, source="csl"
            )
