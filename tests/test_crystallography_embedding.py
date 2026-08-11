# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import numpy as np
import pytest

import GBOpt.crystallography.embedding as embedding_module
from GBOpt.BoundarySpec import (
    BoundaryEmbedding,
    BoundarySpecError,
    BoundarySpecOrthogonalityError,
    PrimitiveCellMetadata,
)
from GBOpt.crystallography.csl import csl_from_scaled_rotation
from GBOpt.crystallography.embedding import (
    _exact_embedding_from_precomputed_csl,
    _paired_pq_from_direction_rows,
    _validated_exact_orientation_rows,
    embedding_from_pq,
    embedding_from_rotation_rows,
    exact_embedding_from_row_rotation_and_plane,
    orthogonal_embedding_from_row_rotation_and_plane,
    primitive_embedding_from_row_rotation,
)
from GBOpt.crystallography.integer import as_int_array
from GBOpt.crystallography.plane import inplane_area_index
from GBOpt.crystallography.pq import recover_exact_row_rotation_from_paired_pq
from GBOpt.crystallography.quaternion import quaternion_to_scaled_rotation
from GBOpt.crystallography.rotation import transpose_rotation_convention
from GBOpt.crystallography.types import CrystallographyValueError

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
    return quaternion_to_scaled_rotation((2, 0, 0, 1))


@pytest.fixture
def sigma5_36deg_rotation():
    """Sigma5 [001] 36.87 deg scaled rotation -- quaternion (3, 0, 0, 1), N=10."""
    return quaternion_to_scaled_rotation((3, 0, 0, 1))


@pytest.fixture
def sigma3_111_rotation():
    """Sigma3 [111] 60 deg twin scaled rotation -- quaternion (1, 1, 1, 0), N=3."""
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
# _validated_exact_orientation_rows
# --------------------------------------------------------------------------------------


def test_validated_exact_orientation_rows_preserves_large_exact_rows():
    large_value = 10**400
    matrix = np.array(
        [
            [large_value, 0, 0],
            [0, large_value + 1, 0],
            [0, 0, 1],
        ],
        dtype=object,
    )

    result = _validated_exact_orientation_rows(matrix, "P")

    assert result.shape == (3, 3)
    assert result.dtype == object
    assert all(type(value) is int for value in result.flat)
    np.testing.assert_array_equal(result, matrix)


@pytest.mark.parametrize(
    ("matrix", "match"),
    [
        pytest.param(
            np.eye(2, dtype=object),
            "shape",
            id="wrong-shape",
        ),
        pytest.param(
            [
                [1, 0, 0],
                [0, 1.5, 0],
                [0, 0, 1],
            ],
            "exactly integer-valued",
            id="noninteger-entry",
        ),
    ],
)
def test_validated_exact_orientation_rows_rejects_malformed_matrix(matrix, match):
    with pytest.raises(CrystallographyValueError, match=match):
        _validated_exact_orientation_rows(matrix, "P")


def test_validated_exact_orientation_rows_rejects_zero_row():
    matrix = np.array(
        [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
        ],
        dtype=object,
    )

    with pytest.raises(
        CrystallographyValueError,
        match="P row 1 must be nonzero",
    ):
        _validated_exact_orientation_rows(matrix, "P")


def test_validated_exact_orientation_rows_rejects_nonorthogonal_rows():
    matrix = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
        ],
        dtype=object,
    )

    with pytest.raises(
        CrystallographyValueError,
        match=r"rows 0 and 1 have exact dot product 1",
    ):
        _validated_exact_orientation_rows(matrix, "P")


def test_validated_exact_orientation_rows_rejects_left_handed_frame():
    matrix = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1],
        ],
        dtype=object,
    )

    with pytest.raises(
        CrystallographyValueError,
        match=r"right-handed.*exact determinant is -1",
    ):
        _validated_exact_orientation_rows(matrix, "P")


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
    meta = PrimitiveCellMetadata(
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


def test_primitive_embedding_generates_successfully(sigma5_53deg_rotation):
    plane = np.array([0, 0, 1])
    result = primitive_embedding_from_row_rotation(
        sigma5_53deg_rotation, plane, source="csl"
    )
    assert result.exact is True
    assert result.coherent is True
    assert result.source == "csl"
    _assert_proper_rotation_pair(result)
    assert result.metadata is not None
    assert result.metadata.basis_mode == "primitive"


def test_primitive_embedding_enforces_primitive_area_limit(sigma5_53deg_rotation):
    with pytest.raises(BoundarySpecError, match="max_primitive_area_index=4"):
        primitive_embedding_from_row_rotation(
            sigma5_53deg_rotation,
            np.array([0, 0, 1]),
            source="csl",
            max_primitive_area_index=4,
        )


def test_primitive_embedding_enforces_pq_determinant_limit(sigma5_53deg_rotation):
    with pytest.raises(BoundarySpecError, match="max_pq_determinant=4"):
        primitive_embedding_from_row_rotation(
            sigma5_53deg_rotation,
            np.array([0, 0, 1]),
            source="csl",
            max_pq_determinant=4,
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


def test_primitive_embedding_records_input_area_reduction_index(sigma5_53deg_rotation):
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


def test_orthogonal_embedding_generates_successfully(sigma5_53deg_rotation):
    plane = np.array([1, 0, 0])
    result = orthogonal_embedding_from_row_rotation_and_plane(
        sigma5_53deg_rotation, plane, source="csl"
    )
    assert result.exact is True
    assert result.coherent is True
    _assert_proper_rotation_pair(result)

    assert result.P is not None
    assert result.Q is not None
    assert result.P.dtype == object
    assert result.Q.dtype == object

    P = as_int_array(result.P, (3, 3), "P")
    Q = as_int_array(result.Q, (3, 3), "Q")

    np.testing.assert_array_equal(P, result.P)
    np.testing.assert_array_equal(Q, result.Q)


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


def test_orthogonal_embedding_enforces_primitive_area_limit(sigma5_53deg_rotation):
    with pytest.raises(BoundarySpecError, match="max_primitive_area_index=4"):
        orthogonal_embedding_from_row_rotation_and_plane(
            sigma5_53deg_rotation,
            np.array([1, 0, 0]),
            source="csl",
            max_primitive_area_index=4,
        )


def test_orthogonal_embedding_enforces_pq_determinant_limit(sigma5_53deg_rotation):
    with pytest.raises(BoundarySpecError, match="max_pq_determinant=24"):
        orthogonal_embedding_from_row_rotation_and_plane(
            sigma5_53deg_rotation,
            np.array([1, 0, 0]),
            source="csl",
            max_pq_determinant=24,
        )


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


def test_orthogonal_embedding_records_input_area_reduction_index(sigma5_53deg_rotation):
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


def test_orthogonal_embedding_preserves_sigma3_misorientation(sigma3_111_rotation):
    result = orthogonal_embedding_from_row_rotation_and_plane(
        sigma3_111_rotation,
        np.array([1, 1, 1]),
        source="csl",
    )

    expected_rotation = (
        np.asarray(sigma3_111_rotation.matrix, dtype=np.float64)
        / sigma3_111_rotation.denominator
    )

    np.testing.assert_allclose(
        result.R_left.T @ result.R_right,
        expected_rotation,
        atol=1.0e-12,
        rtol=0.0,
    )


def test_orthogonal_embedding_preserves_sigma3_row_image_directions(
    sigma3_111_rotation,
):
    result = orthogonal_embedding_from_row_rotation_and_plane(
        sigma3_111_rotation,
        np.array([1, 1, 1]),
        source="csl",
    )

    assert result.P is not None
    assert result.Q is not None

    numerator_matrix = np.asarray(
        sigma3_111_rotation.matrix,
        dtype=object,
    )

    for p_row, q_row in zip(result.P, result.Q, strict=True):
        image_numerator = (
            np.asarray(p_row, dtype=object) @ numerator_matrix
        )
        q_row = np.asarray(q_row, dtype=object)

        np.testing.assert_array_equal(
            np.cross(
                np.asarray(image_numerator, dtype=np.int64),
                np.asarray(q_row, dtype=np.int64),
            ),
            np.zeros(3, dtype=np.int64),
        )
        assert int(image_numerator @ q_row) > 0


# --------------------------------------------------------------------------------------
# _paired_pq_from_direction_rows
# --------------------------------------------------------------------------------------


def test_paired_pq_from_direction_rows_builds_expected_sigma5_pair(
    sigma5_53deg_rotation,
):
    P, Q = _paired_pq_from_direction_rows(
        np.eye(3, dtype=object),
        sigma5_53deg_rotation,
        primitive_area_index=5,
    )

    expected_P = np.array(
        [
            [5, 0, 0],
            [0, 5, 0],
            [0, 0, 1],
        ],
        dtype=object,
    )
    expected_Q = np.array(
        [
            [3, -4, 0],
            [4, 3, 0],
            [0, 0, 1],
        ],
        dtype=object,
    )

    np.testing.assert_array_equal(P, expected_P)
    np.testing.assert_array_equal(Q, expected_Q)


def test_paired_pq_from_direction_rows_preserves_exact_rotation(sigma5_53deg_rotation):
    P, Q = _paired_pq_from_direction_rows(
        np.eye(3, dtype=object),
        sigma5_53deg_rotation,
        primitive_area_index=5,
    )

    recovered = recover_exact_row_rotation_from_paired_pq(P, Q)

    np.testing.assert_array_equal(
        np.asarray(recovered.matrix, dtype=object)
        * sigma5_53deg_rotation.denominator,
        np.asarray(sigma5_53deg_rotation.matrix, dtype=object)
        * recovered.denominator,
    )


def test_paired_pq_from_direction_rows_enlarges_rows_as_exact_pairs():
    identity_rotation = quaternion_to_scaled_rotation((1, 0, 0, 0))

    P, Q = _paired_pq_from_direction_rows(
        np.eye(3, dtype=object),
        identity_rotation,
        primitive_area_index=3,
    )

    expected = np.array(
        [
            [1, 0, 0],
            [0, 3, 0],
            [0, 0, 1],
        ],
        dtype=object,
    )

    np.testing.assert_array_equal(P, expected)
    np.testing.assert_array_equal(Q, expected)
    assert inplane_area_index(P) == 3


@pytest.mark.parametrize(
    ("direction_rows", "match"),
    [
        pytest.param(
            np.eye(2, dtype=object),
            "shape",
            id="wrong-shape",
        ),
        pytest.param(
            [
                [1, 0, 0],
                [0, 1.5, 0],
                [0, 0, 1],
            ],
            "exactly integer-valued",
            id="noninteger-entry",
        ),
    ],
)
def test_paired_pq_from_direction_rows_rejects_malformed_direction_rows(
    direction_rows,
    match,
    sigma5_53deg_rotation,
):
    with pytest.raises(CrystallographyValueError, match=match):
        _paired_pq_from_direction_rows(
            direction_rows,
            sigma5_53deg_rotation,
            primitive_area_index=5,
        )


@pytest.mark.parametrize(
    "primitive_area_index",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="boolean"),
        pytest.param(1.0, id="float"),
    ],
)
def test_paired_pq_from_direction_rows_rejects_invalid_primitive_area_index(
    primitive_area_index,
    sigma5_53deg_rotation,
):
    with pytest.raises(
        CrystallographyValueError,
        match="primitive_area_index must be a positive integer",
    ):
        _paired_pq_from_direction_rows(
            np.eye(3, dtype=object),
            sigma5_53deg_rotation,
            primitive_area_index=primitive_area_index,
        )


def test_paired_pq_from_direction_rows_enforces_final_pq_determinant_limit():
    identity_rotation = quaternion_to_scaled_rotation((1, 0, 0, 0))

    with pytest.raises(
        BoundarySpecError,
        match=(
            r"max_pq_determinant=2.*"
            r"\|det\(P\)\|=3.*"
            r"\|det\(Q\)\|=3"
        ),
    ):
        _paired_pq_from_direction_rows(
            np.eye(3, dtype=object),
            identity_rotation,
            primitive_area_index=3,
            max_pq_determinant=2,
        )


# --------------------------------------------------------------------------------------
# exact_embedding_from_row_rotation_and_plane
# --------------------------------------------------------------------------------------


def test_exact_embedding_uses_primitive_path_for_preserved_plane(sigma5_53deg_rotation):
    result = exact_embedding_from_row_rotation_and_plane(
        sigma5_53deg_rotation,
        np.array([0, 0, 1]),
        source="csl",
    )

    assert result.metadata is not None
    assert result.metadata.plane == (0, 0, 1)
    assert result.metadata.primitive_area_index == 5


def test_exact_embedding_falls_back_when_primitive_rows_are_not_orthogonal(
    monkeypatch,
    sigma5_53deg_rotation,
):
    expected = object()

    def reject_primitive(*args, **kwargs):
        raise BoundarySpecOrthogonalityError("not orthogonal")

    def return_orthogonal(*args, **kwargs):
        return expected

    monkeypatch.setattr(
        embedding_module,
        "_primitive_embedding_from_inplane",
        reject_primitive,
    )
    monkeypatch.setattr(
        embedding_module,
        "_orthogonal_embedding_from_inplane",
        return_orthogonal,
    )

    result = exact_embedding_from_row_rotation_and_plane(
        sigma5_53deg_rotation,
        np.array([0, 0, 1]),
        source="csl",
    )

    assert result is expected


def test_exact_embedding_does_not_fallback_on_cell_size_error(
    monkeypatch,
    sigma5_53deg_rotation,
):
    def reject_primitive(*args, **kwargs):
        raise BoundarySpecError("cell too large")

    monkeypatch.setattr(
        embedding_module,
        "_primitive_embedding_from_inplane",
        reject_primitive,
    )

    with pytest.raises(BoundarySpecError, match="cell too large"):
        exact_embedding_from_row_rotation_and_plane(
            sigma5_53deg_rotation,
            np.array([0, 0, 1]),
            source="csl",
        )


def test_exact_embedding_uses_primitive_path_for_antiparallel_plane(monkeypatch):
    # A 180-degree rotation about y maps [1, 0, 0] to [-1, 0, 0].
    row_rotation = quaternion_to_scaled_rotation((0, 0, 1, 0))
    expected = object()

    def return_primitive(*args, **kwargs):
        return expected

    def reject_orthogonal(*args, **kwargs):
        pytest.fail(
            "The orthogonal path should not be used when the plane normal "
            "is preserved up to sign."
        )

    monkeypatch.setattr(
        embedding_module,
        "_primitive_embedding_from_inplane",
        return_primitive,
    )
    monkeypatch.setattr(
        embedding_module,
        "_orthogonal_embedding_from_inplane",
        reject_orthogonal,
    )

    result = exact_embedding_from_row_rotation_and_plane(
        row_rotation,
        np.array([1, 0, 0]),
        source="csl",
    )

    assert result is expected


def test_exact_embedding_constructs_primitive_antiparallel_plane():
    row_rotation = quaternion_to_scaled_rotation((0, 0, 1, 0))

    result = exact_embedding_from_row_rotation_and_plane(
        row_rotation,
        np.array([1, 0, 0]),
        source="csl",
    )

    assert result.exact is True
    assert result.coherent is True
    assert result.metadata is not None
    assert result.metadata.basis_mode == "primitive"
    assert result.metadata.plane == (1, 0, 0)

    expected_rotation = (
        np.asarray(row_rotation.matrix, dtype=np.float64)
        / row_rotation.denominator
    )
    np.testing.assert_allclose(
        result.R_left.T @ result.R_right,
        expected_rotation,
        atol=1.0e-12,
        rtol=0.0,
    )


def test_precomputed_csl_path_matches_public_exact_selector(sigma5_53deg_rotation):
    plane = np.array([1, 0, 0], dtype=object)
    column_rotation = transpose_rotation_convention(
        sigma5_53deg_rotation
    )
    csl = csl_from_scaled_rotation(column_rotation)

    expected = exact_embedding_from_row_rotation_and_plane(
        sigma5_53deg_rotation,
        plane,
        source="csl",
    )
    actual = _exact_embedding_from_precomputed_csl(
        sigma5_53deg_rotation,
        plane,
        csl,
        source="csl",
    )

    np.testing.assert_array_equal(actual.P, expected.P)
    np.testing.assert_array_equal(actual.Q, expected.Q)
    np.testing.assert_allclose(actual.R_left, expected.R_left)
    np.testing.assert_allclose(actual.R_right, expected.R_right)
    assert actual.metadata == expected.metadata
