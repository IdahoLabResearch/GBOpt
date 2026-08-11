# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
from typing import Any

import numpy as np
import pytest

from GBOpt.BoundarySpec import (
    BoundaryEmbedding,
    BoundarySpecError,
    BoundarySpecOrthogonalityError,
)
from GBOpt.crystallography.embedding import (
    embedding_from_pq,
    embedding_from_rotation_rows,
    orthogonal_embedding_from_row_rotation_and_plane,
    primitive_embedding_from_row_rotation,
    primitive_metadata,
)
from GBOpt.crystallography.integer import as_int_array

# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture
def identity_orientation_rows():
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    Q = P.copy()
    return P, Q


@pytest.fixture
def sigma5_53deg_rotation():
    """Sigma5 [001] 53.13 deg scaled rotation -- quaternion (2, 0, 0, 1), N=5."""
    from GBOpt.crystallography.quaternion import quaternion_to_scaled_rotation
    return quaternion_to_scaled_rotation((2, 0, 0, 1))


@pytest.fixture
def sigma5_36deg_rotation():
    """Sigma5 [001] 36.87 deg scaled rotation -- quaternion (3, 0, 0, 1), N=10."""
    from GBOpt.crystallography.quaternion import quaternion_to_scaled_rotation
    return quaternion_to_scaled_rotation((3, 0, 0, 1))


@pytest.fixture
def sigma3_111_rotation():
    """Sigma3 [111] 60 deg twin scaled rotation -- quaternion (1, 1, 1, 0), N=3."""
    from GBOpt.crystallography.quaternion import quaternion_to_scaled_rotation
    return quaternion_to_scaled_rotation((1, 1, 1, 0))


# --------------------------------------------------------------------------------------
# Assertion helpers
# --------------------------------------------------------------------------------------


def _assert_proper_rotation_pair(embedding, *, atol=1e-10):
    for label, R in (("R_left", embedding.R_left), ("R_right", embedding.R_right)):
        np.testing.assert_allclose(
            R @ R.T,
            np.eye(3),
            atol=atol,
            rtol=0.0,
            err_msg=f"{label} is not orthogonal",
        )
        assert np.isclose(np.linalg.det(R), 1.0, atol=atol, rtol=0.0)


# --------------------------------------------------------------------------------------
# primitive_metadata
# --------------------------------------------------------------------------------------


def test_primitive_metadata_valid_returns_metadata():
    result = primitive_metadata(
        basis_mode="primitive",
        input_area_index=10,
        primitive_area_index=5,
        orientation_area_index=5,
        plane=np.array([0, 0, 1]),
        rotation_denominator=5,
    )

    assert result.basis_mode == "primitive"
    assert result.input_area_index == 10
    assert result.primitive_area_index == 5
    assert result.orientation_area_index == 5
    assert result.input_reduction_index == 2
    assert result.plane == (0, 0, 1)
    assert result.rotation_denominator == 5
    assert result.conventional_cell_multiplier == 10


def test_primitive_metadata_supplied_mode_records_input_area_as_single_reduction():
    result = primitive_metadata(
        basis_mode="supplied",
        input_area_index=5,
        primitive_area_index=5,
        orientation_area_index=5,
        plane=np.array([0, 0, 1]),
        rotation_denominator=5,
    )

    assert result.basis_mode == "supplied"
    assert result.input_area_index == 5
    assert result.primitive_area_index == 5
    assert result.orientation_area_index == 5
    assert result.input_reduction_index == 1
    assert result.conventional_cell_multiplier == 10


def test_primitive_metadata_requires_divisible_input_area_index():
    with pytest.raises(BoundarySpecError, match="integer multiple"):
        primitive_metadata(
            basis_mode="primitive",
            input_area_index=7,
            primitive_area_index=5,
            orientation_area_index=1,
            plane=np.array([0, 0, 1]),
            rotation_denominator=5,
        )


def test_primitive_metadata_allows_orientation_area_not_multiple_of_primitive_area():
    result = primitive_metadata(
        basis_mode="primitive",
        input_area_index=None,
        primitive_area_index=5,
        orientation_area_index=1,
        plane=np.array([0, 0, 1]),
        rotation_denominator=5,
    )

    assert result.primitive_area_index == 5
    assert result.orientation_area_index == 1
    assert result.input_area_index is None
    assert result.input_reduction_index is None


def test_primitive_metadata_conventional_cell_multiplier_is_twice_primitive():
    result = primitive_metadata(
        basis_mode="primitive",
        input_area_index=None,
        primitive_area_index=5,
        orientation_area_index=5,
        plane=np.array([1, 0, 0]),
        rotation_denominator=5,
    )

    assert result.conventional_cell_multiplier == 2 * result.primitive_area_index


@pytest.mark.parametrize("basis_mode", ["orthogonal", "", None])
def test_primitive_metadata_rejects_invalid_basis_mode(basis_mode):
    with pytest.raises(BoundarySpecError, match="basis_mode"):
        primitive_metadata(
            basis_mode=basis_mode,
            primitive_area_index=1,
            plane=np.array([0, 0, 1]),
            rotation_denominator=1,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param(
            {"primitive_area_index": 0},
            "primitive_area_index",
            id="primitive-area-index-zero",
        ),
        pytest.param(
            {"primitive_area_index": -1},
            "primitive_area_index",
            id="primitive-area-index-negative",
        ),
        pytest.param(
            {"input_area_index": 0},
            "input_area_index",
            id="input-area-index-zero",
        ),
        pytest.param(
            {"input_area_index": -1},
            "input_area_index",
            id="input-area-index-negative",
        ),
        pytest.param(
            {"orientation_area_index": 0},
            "orientation_area_index",
            id="orientation-area-index-zero",
        ),
        pytest.param(
            {"orientation_area_index": -1},
            "orientation_area_index",
            id="orientation-area-index-negative",
        ),
    ],
)
def test_primitive_metadata_rejects_nonpositive_metadata_values(
    kwargs: dict[str, Any],
    match: str,
):
    params: dict[str, Any] = {
        "basis_mode": "primitive",
        "primitive_area_index": 1,
        "plane": np.array([0, 0, 1]),
        "rotation_denominator": 1,
    }
    params.update(kwargs)

    with pytest.raises(BoundarySpecError, match=match):
        primitive_metadata(**params)


# --------------------------------------------------------------------------------------
# embedding_from_pq
# --------------------------------------------------------------------------------------


def test_embedding_from_pq_builds_exact_coherent_embedding_from_canonical_rows(
    identity_orientation_rows,
):
    P, Q = identity_orientation_rows
    result = embedding_from_pq(P, Q, source="pq")

    assert isinstance(result, BoundaryEmbedding)
    assert result.exact is True
    assert result.coherent is True
    assert result.source == "pq"
    assert result.metadata is None
    assert result.P is not None
    assert result.Q is not None
    np.testing.assert_array_equal(result.P, P)
    np.testing.assert_array_equal(result.Q, Q)
    _assert_proper_rotation_pair(result, atol=1e-12)


def test_embedding_from_pq_non_orthogonal_raises(identity_orientation_rows):
    _, Q = identity_orientation_rows
    P = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    with pytest.raises(BoundarySpecOrthogonalityError, match="orthogonal"):
        embedding_from_pq(P, Q, source="pq")


def test_embedding_from_pq_passes_metadata(identity_orientation_rows):
    P, Q = identity_orientation_rows
    meta = primitive_metadata(
        basis_mode="primitive",
        input_area_index=None,
        primitive_area_index=1,
        orientation_area_index=1,
        plane=np.array([0, 0, 1]),
        rotation_denominator=1,
    )
    result = embedding_from_pq(P, Q, source="pq", metadata=meta)
    assert result.metadata is meta


# --------------------------------------------------------------------------------------
# embedding_from_rotation_rows
# --------------------------------------------------------------------------------------


def test_embedding_from_rotation_rows_builds_approximate_embedding():
    R_left = np.eye(3)
    R_right = np.eye(3)

    result = embedding_from_rotation_rows(
        R_left,
        R_right,
        source="csl",
        coherent=False,
    )

    assert result.P is None
    assert result.Q is None
    assert result.exact is False
    assert result.coherent is False
    assert result.source == "csl"
    np.testing.assert_allclose(result.R_left, R_left)
    np.testing.assert_allclose(result.R_right, R_right)


def test_embedding_from_rotation_rows_rejects_improper_rotation():
    R_left = np.eye(3)
    R_right = np.diag([-1.0, 1.0, 1.0])

    with pytest.raises(BoundarySpecOrthogonalityError, match="proper"):
        embedding_from_rotation_rows(R_left, R_right, source="csl")


def test_embedding_from_rotation_rows_rejects_non_orthogonal_rotation():
    R_left = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R_right = np.eye(3)

    with pytest.raises(BoundarySpecOrthogonalityError, match="orthogonal"):
        embedding_from_rotation_rows(R_left, R_right, source="csl")


# --------------------------------------------------------------------------------------
# primitive_embedding_from_row_rotation
# --------------------------------------------------------------------------------------


def test_primitive_embedding_exact_and_coherent(sigma5_53deg_rotation):
    plane = np.array([0, 0, 1])
    result = primitive_embedding_from_row_rotation(
        sigma5_53deg_rotation, plane, source="csl"
    )
    assert result.exact is True
    assert result.coherent is True


def test_primitive_embedding_preserves_source_label(sigma5_53deg_rotation):
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
    _assert_proper_rotation_pair(result)


def test_primitive_embedding_records_primitive_metadata(sigma5_53deg_rotation):
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
    plane = np.array([1, 1, 1])
    with pytest.raises(BoundarySpecOrthogonalityError):
        primitive_embedding_from_row_rotation(sigma3_111_rotation, plane, source="csl")


def test_primitive_embedding_pq_rows_satisfy_row_rotation_contract(
    sigma5_53deg_rotation,
):
    result = primitive_embedding_from_row_rotation(
        sigma5_53deg_rotation,
        np.array([0, 0, 1]),
        source="csl",
    )

    assert result.P is not None
    assert result.Q is not None

    M = np.asarray(sigma5_53deg_rotation.matrix, dtype=object)
    N = int(sigma5_53deg_rotation.denominator)

    for p_row, q_row in zip(result.P, result.Q):
        numerator = np.asarray(p_row, dtype=object) @ M
        assert all(int(v) % N == 0 for v in numerator)
        image = np.array([int(v) // N for v in numerator], dtype=object)
        np.testing.assert_array_equal(image, q_row)


def test_primitive_embedding_records_input_area_reduction_index(
    sigma5_53deg_rotation,
):
    plane = np.array([0, 0, 1])

    baseline = primitive_embedding_from_row_rotation(
        sigma5_53deg_rotation,
        plane,
        source="csl",
    )
    assert baseline.metadata is not None
    primitive_area_index = baseline.metadata.primitive_area_index
    input_area_index = 3 * primitive_area_index

    result = primitive_embedding_from_row_rotation(
        sigma5_53deg_rotation,
        plane,
        source="csl",
        input_area_index=input_area_index,
    )

    assert result.metadata is not None
    assert result.metadata.input_area_index == input_area_index
    assert result.metadata.primitive_area_index == primitive_area_index
    assert result.metadata.input_reduction_index == 3


# --------------------------------------------------------------------------------------
# orthogonal_embedding_from_row_rotation_and_plane
# --------------------------------------------------------------------------------------


def test_orthogonal_embedding_exact_and_coherent(sigma5_53deg_rotation):
    plane = np.array([1, 0, 0])
    result = orthogonal_embedding_from_row_rotation_and_plane(
        sigma5_53deg_rotation, plane, source="csl"
    )
    assert result.exact is True
    assert result.coherent is True


def test_orthogonal_embedding_sigma5_rows_normalize_to_proper_rotations(
    sigma5_53deg_rotation,
):
    plane = np.array([1, 0, 0])
    result = orthogonal_embedding_from_row_rotation_and_plane(
        sigma5_53deg_rotation, plane, source="csl"
    )
    _assert_proper_rotation_pair(result)


def test_orthogonal_embedding_sigma3_111_proper_rotations(sigma3_111_rotation):
    plane = np.array([1, 1, 1])
    result = orthogonal_embedding_from_row_rotation_and_plane(
        sigma3_111_rotation, plane, source="csl"
    )
    _assert_proper_rotation_pair(result)


def test_orthogonal_embedding_sigma3_111_records_primitive_and_orientation_area_metadata(
    sigma3_111_rotation,
):
    plane = np.array([1, 1, 1])
    result = orthogonal_embedding_from_row_rotation_and_plane(
        sigma3_111_rotation, plane, source="csl"
    )

    assert result.metadata is not None
    assert result.metadata.basis_mode == "primitive"
    assert result.metadata.input_area_index is None
    assert result.metadata.input_reduction_index is None
    assert result.metadata.primitive_area_index == 3
    assert result.metadata.orientation_area_index == 2
    assert result.metadata.conventional_cell_multiplier == 6
    assert result.metadata.plane == (1, 1, 1)
    assert result.metadata.rotation_denominator == 3


def test_orthogonal_embedding_max_exact_atoms_raises(sigma5_53deg_rotation):
    plane = np.array([1, 0, 0])
    with pytest.raises(BoundarySpecError, match="max_exact_atoms"):
        orthogonal_embedding_from_row_rotation_and_plane(
            sigma5_53deg_rotation, plane, source="csl", max_exact_atoms=1
        )


def test_orthogonal_embedding_returns_exact_integer_pq_rows(sigma5_53deg_rotation):
    plane = np.array([1, 0, 0])
    result = orthogonal_embedding_from_row_rotation_and_plane(
        sigma5_53deg_rotation, plane, source="csl"
    )

    assert result.P is not None
    assert result.Q is not None
    assert result.P.dtype == object
    assert result.Q.dtype == object

    P = as_int_array(result.P, (3, 3), "P")
    Q = as_int_array(result.Q, (3, 3), "Q")

    np.testing.assert_array_equal(P, result.P)
    np.testing.assert_array_equal(Q, result.Q)


def test_orthogonal_embedding_rows_are_orthogonal_integer_directions(
    sigma5_53deg_rotation,
):
    plane = np.array([1, 0, 0])
    result = orthogonal_embedding_from_row_rotation_and_plane(
        sigma5_53deg_rotation, plane, source="csl"
    )

    assert result.P is not None
    assert result.Q is not None

    P = as_int_array(result.P, (3, 3), "P")
    Q = as_int_array(result.Q, (3, 3), "Q")

    P_gram = P @ P.T
    Q_gram = Q @ Q.T

    np.testing.assert_array_equal(P_gram, np.diag(np.diag(P_gram)))
    np.testing.assert_array_equal(Q_gram, np.diag(np.diag(Q_gram)))


def test_orthogonal_embedding_records_input_area_reduction_index(
    sigma5_53deg_rotation,
):
    plane = np.array([1, 0, 0])

    baseline = orthogonal_embedding_from_row_rotation_and_plane(
        sigma5_53deg_rotation,
        plane,
        source="csl",
    )
    assert baseline.metadata is not None
    primitive_area_index = baseline.metadata.primitive_area_index
    input_area_index = 4 * primitive_area_index

    result = orthogonal_embedding_from_row_rotation_and_plane(
        sigma5_53deg_rotation,
        plane,
        source="csl",
        input_area_index=input_area_index,
    )

    assert result.metadata is not None
    assert result.metadata.input_area_index == input_area_index
    assert result.metadata.primitive_area_index == primitive_area_index
    assert result.metadata.input_reduction_index == 4
