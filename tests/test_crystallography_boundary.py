# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
import numpy as np
import pytest

from GBOpt.BoundarySpec import (
    BoundaryEmbedding,
    BoundarySpecError,
    CSLApproxSpec,
    CSLExactSpec,
    FiveDOFSpec,
    PQSpec,
)
from GBOpt.crystallography.boundary import (
    csl_approx_spec_to_embedding,
    csl_exact_spec_to_embedding,
    five_dof_spec_to_embedding,
    pq_spec_to_embedding,
    primitive_bicrystal_atom_count,
)
from GBOpt.crystallography.integer import as_int_array, integer_det3
from GBOpt.crystallography.pq import (
    recover_exact_row_rotation_from_paired_pq,
)
from GBOpt.crystallography.quaternion import quaternion_to_scaled_rotation

# --------------------------------------------------------------------------------------
# Shared test data
# --------------------------------------------------------------------------------------

SIGMA5_36_P = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
SIGMA5_36_Q = ((4, -3, 0), (3, 4, 0), (0, 0, 1))
SIGMA5_36_PRIMITIVE_INPUT_P = ((10, 0, 0), (0, 10, 0), (0, 0, 1))
SIGMA5_36_PRIMITIVE_INPUT_Q = ((8, -6, 0), (6, 8, 0), (0, 0, 1))
SIGMA5_TWIST_LEGACY_P = ((0, 0, 1), (3, 1, 0), (-1, 3, 0))
SIGMA5_TWIST_LEGACY_Q = ((0, 0, 1), (3, -1, 0), (1, 3, 0))
SIGMA5_TWIST_PRIMITIVE_P = ((0, 0, 1), (1, 2, 0), (-2, 1, 0))
SIGMA5_TWIST_PRIMITIVE_Q = ((0, 0, 1), (2, 1, 0), (-1, 2, 0))
SIGMA5_TWIST_ORTHOGONAL_FALLBACK_P = ((0, 0, 1), (2, -1, 0), (1, 2, 0))
SIGMA5_TWIST_ORTHOGONAL_FALLBACK_Q = ((0, 0, 1), (2, 1, 0), (-1, 2, 0))


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture
def identity_pq_spec():
    return PQSpec(
        P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        Q=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )


@pytest.fixture
def sigma5_supplied_pq_spec():
    return PQSpec(P=list(SIGMA5_36_P), Q=list(SIGMA5_36_Q), basis_mode="supplied")


@pytest.fixture
def approx_sigma5_spec():
    return CSLApproxSpec(axis=[0, 0, 1], plane=[1, 0, 0], angle_deg=36.87)


@pytest.fixture
def sigma5_twist_csl_embedding():
    spec = CSLExactSpec(axis=[0, 0, 1], plane=[0, 0, 1], quat=[3, 0, 0, 1])
    emb = csl_exact_spec_to_embedding(spec)
    assert emb.metadata is not None
    return emb


# --------------------------------------------------------------------------------------
# Assertion helpers
# --------------------------------------------------------------------------------------


def _assert_embedding_flags(emb, *, exact, coherent, source):
    assert isinstance(emb, BoundaryEmbedding)
    assert emb.exact is exact
    assert emb.coherent is coherent
    assert emb.source == source


def _assert_proper_rotation_matrix(R, *, atol=1e-10, label="rotation"):
    np.testing.assert_allclose(
        R @ R.T,
        np.eye(3),
        atol=atol,
        rtol=0.0,
        err_msg=f"{label} is not orthogonal",
    )
    assert np.isclose(
        np.linalg.det(R),
        1.0,
        atol=atol,
        rtol=0.0,
    ), f"{label} determinant is not +1"


def _assert_proper_rotation_pair(emb, *, atol=1e-10):
    _assert_proper_rotation_matrix(emb.R_left, atol=atol, label="R_left")
    _assert_proper_rotation_matrix(emb.R_right, atol=atol, label="R_right")


# --------------------------------------------------------------------------------------
# pq_spec_to_embedding - basic embedding behavior
# --------------------------------------------------------------------------------------


def test_pq_spec_to_embedding_flags_and_source(identity_pq_spec):
    emb = pq_spec_to_embedding(identity_pq_spec)
    _assert_embedding_flags(emb, exact=True, coherent=True, source="pq")


def test_pq_spec_to_embedding_identity_r_left_r_right(identity_pq_spec):
    emb = pq_spec_to_embedding(identity_pq_spec)
    np.testing.assert_allclose(emb.R_left, np.eye(3), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(emb.R_right, np.eye(3), atol=1e-12, rtol=0.0)


def test_pq_spec_to_embedding_sigma5_r_right_matches_expected(
    sigma5_supplied_pq_spec,
):
    embedding = pq_spec_to_embedding(sigma5_supplied_pq_spec)

    expected = np.array(
        [
            [4.0 / 5.0, -3.0 / 5.0, 0.0],
            [3.0 / 5.0, 4.0 / 5.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_allclose(
        embedding.R_right,
        expected,
        atol=1.0e-12,
        rtol=0.0,
    )


def test_pq_spec_to_embedding_sigma5_r_right_is_proper_rotation(
    sigma5_supplied_pq_spec,
):
    emb = pq_spec_to_embedding(sigma5_supplied_pq_spec)
    _assert_proper_rotation_matrix(emb.R_right, atol=1e-12, label="R_right")


def test_pq_spec_to_embedding_row_norms_are_unit():
    spec = PQSpec(
        P=[[2, 1, 0], [0, 0, 1], [1, -2, 0]],
        Q=[[2, -1, 0], [0, 0, 1], [-1, -2, 0]],
    )
    emb = pq_spec_to_embedding(spec)
    np.testing.assert_allclose(
        np.linalg.norm(emb.R_left, axis=1), np.ones(3), atol=1e-12
    )
    np.testing.assert_allclose(
        np.linalg.norm(emb.R_right, axis=1), np.ones(3), atol=1e-12
    )


def test_pq_spec_to_embedding_non_orthogonal_raises():
    spec = PQSpec(
        P=[[1, 0, 0], [1, 1, 0], [0, 0, 1]],
        Q=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        basis_mode="supplied",
    )

    with pytest.raises(
        BoundarySpecError,
        match=r"R_left or R_right derived from P/Q is not a proper rotation matrix",
    ):
        pq_spec_to_embedding(spec)


# --------------------------------------------------------------------------------------
# pq_spec_to_embedding - primitive basis mode
# --------------------------------------------------------------------------------------


def test_pq_spec_primitive_legacy_sigma5_twist_area_index_5():
    emb = pq_spec_to_embedding(
        PQSpec(P=list(SIGMA5_TWIST_LEGACY_P), Q=list(SIGMA5_TWIST_LEGACY_Q))
    )
    assert emb.P is not None
    assert emb.Q is not None
    assert integer_det3(emb.P) == 5
    assert integer_det3(emb.Q) == 5
    assert emb.metadata is not None
    assert emb.metadata.basis_mode == "primitive"
    assert emb.metadata.input_area_index == 10
    assert emb.metadata.primitive_area_index == 5
    assert emb.metadata.orientation_area_index == 5
    assert emb.metadata.input_reduction_index == 2


def test_pq_spec_primitive_and_legacy_sigma5_twist_match():
    emb_legacy = pq_spec_to_embedding(
        PQSpec(P=list(SIGMA5_TWIST_LEGACY_P), Q=list(SIGMA5_TWIST_LEGACY_Q))
    )
    emb_primitive = pq_spec_to_embedding(
        PQSpec(P=list(SIGMA5_TWIST_PRIMITIVE_P), Q=list(SIGMA5_TWIST_PRIMITIVE_Q))
    )

    assert emb_legacy.P is not None
    assert emb_primitive.P is not None
    assert emb_legacy.Q is not None
    assert emb_primitive.Q is not None
    np.testing.assert_array_equal(emb_legacy.P, emb_primitive.P)
    np.testing.assert_array_equal(emb_legacy.Q, emb_primitive.Q)
    np.testing.assert_allclose(emb_legacy.R_left, emb_primitive.R_left, atol=1e-12)
    np.testing.assert_allclose(emb_legacy.R_right, emb_primitive.R_right, atol=1e-12)


@pytest.mark.parametrize(
    "P,Q,expected_plane",
    [
        (
            [[2, 1, 0], [0, 0, 1], [1, -2, 0]],
            [[2, -1, 0], [0, 0, 1], [-1, -2, 0]],
            (2, 1, 0),
        ),
        (
            [[3, 1, 0], [0, 0, 1], [1, -3, 0]],
            [[3, -1, 0], [0, 0, 1], [-1, -3, 0]],
            (3, 1, 0),
        ),
    ],
)
def test_pq_spec_primitive_tilt_boundaries_preserve_plane(P, Q, expected_plane):
    emb = pq_spec_to_embedding(PQSpec(P=P, Q=Q))
    assert emb.metadata is not None
    assert emb.metadata.plane == expected_plane
    assert emb.P is not None
    np.testing.assert_array_equal(emb.P[0].astype(int), expected_plane)


def test_pq_spec_primitive_raises_when_rotation_recovery_fails():
    spec = PQSpec(
        P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        Q=[[4, -3, 0], [3, 4, 0], [0, 0, 1]],
    )

    with pytest.raises(
        BoundarySpecError,
        match="basis_mode='primitive' requires paired P/Q rows",
    ):
        pq_spec_to_embedding(spec)


def test_pq_spec_primitive_raises_when_primitive_embedding_fails(monkeypatch):
    def raise_primitive_failure(*args, **kwargs):
        raise BoundarySpecError("forced primitive failure")

    monkeypatch.setattr(
        "GBOpt.crystallography.boundary.primitive_embedding_from_row_rotation",
        raise_primitive_failure,
    )

    spec = PQSpec(P=list(SIGMA5_TWIST_PRIMITIVE_P), Q=list(SIGMA5_TWIST_PRIMITIVE_Q))

    with pytest.raises(BoundarySpecError, match="forced primitive failure"):
        pq_spec_to_embedding(spec)


def test_pq_spec_primitive_orthogonal_fallback_preserves_input_reduction_metadata():
    spec = PQSpec(
        P=SIGMA5_36_PRIMITIVE_INPUT_P,
        Q=SIGMA5_36_PRIMITIVE_INPUT_Q,
        basis_mode="primitive",
    )

    emb = pq_spec_to_embedding(spec)

    assert emb.P is not None
    assert emb.Q is not None
    np.testing.assert_array_equal(emb.P, np.array(SIGMA5_36_P, dtype=object))
    np.testing.assert_array_equal(emb.Q, np.array(SIGMA5_36_Q, dtype=object))

    assert emb.metadata is not None
    assert emb.metadata.basis_mode == "primitive"

    assert emb.metadata.input_area_index == 10
    assert emb.metadata.primitive_area_index == 5
    assert emb.metadata.input_reduction_index == 2

    assert emb.metadata.orientation_area_index == 1

    assert emb.metadata.plane == (1, 0, 0)
    assert emb.metadata.rotation_denominator == 5
    assert emb.metadata.conventional_cell_multiplier == 10

    assert (
        emb.metadata.input_reduction_index
        == emb.metadata.input_area_index // emb.metadata.primitive_area_index
    )


# --------------------------------------------------------------------------------------
# pq_spec_to_embedding - supplied basis mode
# --------------------------------------------------------------------------------------


def test_pq_spec_supplied_mode_preserves_area_index_10():
    emb = pq_spec_to_embedding(
        PQSpec(
            P=list(SIGMA5_TWIST_LEGACY_P),
            Q=list(SIGMA5_TWIST_LEGACY_Q),
            basis_mode="supplied",
        )
    )
    assert emb.P is not None
    assert emb.Q is not None
    assert integer_det3(emb.P) == 10
    assert integer_det3(emb.Q) == 10
    assert emb.metadata is not None
    assert emb.metadata.basis_mode == "supplied"
    assert emb.metadata.input_area_index == 10
    assert emb.metadata.primitive_area_index == 10
    assert emb.metadata.orientation_area_index == 10
    assert emb.metadata.input_reduction_index == 1


def test_pq_spec_supplied_mode_preserves_paired_row_rotation():
    transform = np.array([[1, 1], [0, 1]], dtype=int)
    P = np.array(SIGMA5_TWIST_PRIMITIVE_P, dtype=float)
    Q = np.array(SIGMA5_TWIST_PRIMITIVE_Q, dtype=float)
    P_transformed = P.copy()
    Q_transformed = Q.copy()
    P_transformed[1:] = transform @ P[1:]
    Q_transformed[1:] = transform @ Q[1:]

    expected_rotation = recover_exact_row_rotation_from_paired_pq(
        P_transformed, Q_transformed
    )
    emb = pq_spec_to_embedding(
        PQSpec(
            P=P_transformed.tolist(),
            Q=Q_transformed.tolist(),
            basis_mode="supplied",
        )
    )
    assert emb.P is not None
    assert emb.Q is not None
    actual_rotation = recover_exact_row_rotation_from_paired_pq(emb.P, emb.Q)
    assert actual_rotation.denominator == expected_rotation.denominator
    np.testing.assert_array_equal(actual_rotation.matrix, expected_rotation.matrix)


def test_pq_spec_supplied_mode_does_not_fabricate_primitive_metadata_for_unrecoverable_rotation():
    spec = PQSpec(
        P=[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        Q=[
            [1, 0, 0],
            [0, 1, 1],
            [0, -1, 1],
        ],
        basis_mode="supplied",
    )

    emb = pq_spec_to_embedding(spec)

    _assert_embedding_flags(emb, exact=True, coherent=True, source="pq")
    _assert_proper_rotation_pair(emb)

    assert emb.P is not None
    assert emb.Q is not None
    assert emb.metadata is None


# --------------------------------------------------------------------------------------
# csl_exact_spec_to_embedding - basic exact embeddings
# --------------------------------------------------------------------------------------


def test_csl_exact_spec_to_embedding_flags_and_proper_rotations():
    spec = CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 0, 0, 1])
    emb = csl_exact_spec_to_embedding(spec)

    _assert_embedding_flags(emb, exact=True, coherent=True, source="csl")
    _assert_proper_rotation_pair(emb)


def test_csl_exact_spec_to_embedding_identity_quaternion_zero_misorientation():
    spec = CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[1, 0, 0, 0], sigma=1)
    emb = csl_exact_spec_to_embedding(spec)
    assert emb.exact is True
    np.testing.assert_allclose(emb.R_left, emb.R_right, atol=1e-12)


def test_csl_exact_spec_to_embedding_cross_format_round_trip():
    spec_csl = CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 0, 0, 1])
    spec_pq = PQSpec(P=list(SIGMA5_36_P), Q=list(SIGMA5_36_Q), basis_mode="supplied")
    emb_csl = csl_exact_spec_to_embedding(spec_csl)
    emb_pq = pq_spec_to_embedding(spec_pq)

    assert emb_csl.P is not None
    assert emb_csl.Q is not None
    assert emb_pq.P is not None
    assert emb_pq.Q is not None
    np.testing.assert_array_equal(emb_csl.P, emb_pq.P)
    np.testing.assert_array_equal(emb_csl.Q, emb_pq.Q)


def test_csl_exact_spec_to_embedding_sigma3_111_proper_rotations():
    spec = CSLExactSpec(axis=[1, 1, 1], plane=[1, 1, 1], quat=[3, 1, 1, 1])
    emb = csl_exact_spec_to_embedding(spec)
    _assert_proper_rotation_pair(emb)


def test_csl_exact_spec_to_embedding_twist_matches_primitive_pqspec():
    spec_csl = CSLExactSpec(axis=[0, 0, 1], plane=[0, 0, 1], quat=[3, 0, 0, 1])
    spec_pq = PQSpec(P=list(SIGMA5_TWIST_LEGACY_P), Q=list(SIGMA5_TWIST_LEGACY_Q))
    emb_csl = csl_exact_spec_to_embedding(spec_csl)
    emb_pq = pq_spec_to_embedding(spec_pq)

    assert emb_csl.P is not None
    assert emb_csl.Q is not None
    assert emb_pq.P is not None
    assert emb_pq.Q is not None
    np.testing.assert_array_equal(emb_csl.P, emb_pq.P)
    np.testing.assert_array_equal(emb_csl.Q, emb_pq.Q)
    assert integer_det3(emb_csl.P) == 5
    assert integer_det3(emb_csl.Q) == 5


def test_csl_exact_spec_to_embedding_twist_rotations_derive_from_final_pq():
    spec = CSLExactSpec(axis=[0, 0, 1], plane=[0, 0, 1], quat=[3, 0, 0, 1])
    emb = csl_exact_spec_to_embedding(spec)
    assert emb.P is not None
    assert emb.Q is not None
    P_float = np.asarray(emb.P, dtype=float)
    Q_float = np.asarray(emb.Q, dtype=float)
    expected_left = P_float / np.linalg.norm(
        np.asarray(P_float, dtype=float), axis=1, keepdims=True
    )
    expected_right = Q_float / np.linalg.norm(
        np.asarray(Q_float, dtype=float), axis=1, keepdims=True
    )
    np.testing.assert_allclose(emb.R_left, expected_left, atol=1e-12)
    np.testing.assert_allclose(emb.R_right, expected_right, atol=1e-12)
    assert not np.allclose(emb.R_left, emb.R_right, atol=1e-12)
    assert emb.metadata is not None
    assert emb.metadata.basis_mode == "primitive"


def test_csl_exact_spec_to_embedding_large_denominator_stays_exact():
    spec = CSLExactSpec(axis=[128, 1, 1], plane=[1, 0, 0], quat=[128, 128, 1, 1])
    emb = csl_exact_spec_to_embedding(spec, max_exact_atoms=10**9)
    assert emb.exact is True
    assert emb.P is not None
    assert emb.Q is not None
    for R in (emb.R_left, emb.R_right):
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    np.testing.assert_array_equal(emb.P.astype(int), np.eye(3, dtype=int))
    expected_Q = np.array([[16383, 0, 256], [256, 0, -16383], [0, 1, 0]], dtype=int)
    np.testing.assert_array_equal(emb.Q.astype(int), expected_Q)


# --------------------------------------------------------------------------------------
# csl_exact_spec_to_embedding - fallback paths
# --------------------------------------------------------------------------------------


def test_csl_exact_spec_uses_orthogonal_embedding_when_rotation_does_not_preserve_plane(
    monkeypatch,
):
    def fail_if_primitive_builder_is_called(*args, **kwargs):
        raise AssertionError(
            "primitive embedding should not be attempted when the rotation does not "
            "preserve the boundary plane"
        )

    monkeypatch.setattr(
        "GBOpt.crystallography.boundary.primitive_embedding_from_row_rotation",
        fail_if_primitive_builder_is_called,
    )

    spec = CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 0, 0, 1])

    emb = csl_exact_spec_to_embedding(spec)

    _assert_embedding_flags(emb, exact=True, coherent=True, source="csl")
    _assert_proper_rotation_pair(emb)

    assert emb.P is not None
    assert emb.Q is not None
    np.testing.assert_array_equal(emb.P, np.array(SIGMA5_36_P, dtype=object))
    np.testing.assert_array_equal(emb.Q, np.array(SIGMA5_36_Q, dtype=object))

    assert emb.metadata is not None
    assert emb.metadata.basis_mode == "primitive"
    assert emb.metadata.input_area_index is None
    assert emb.metadata.primitive_area_index == 5
    assert emb.metadata.input_reduction_index is None
    assert emb.metadata.orientation_area_index == 1
    assert emb.metadata.plane == (1, 0, 0)
    assert emb.metadata.rotation_denominator == 10
    assert emb.metadata.conventional_cell_multiplier == 10


# --------------------------------------------------------------------------------------
# Row-rotation convention regressions
# --------------------------------------------------------------------------------------


def test_csl_embedding_recovers_row_rotation_not_transpose():
    q = (2, 0, 0, 1)
    spec = CSLExactSpec(axis=(0, 0, 1), plane=(0, 0, 1), quat=q, sigma=5)
    rot = quaternion_to_scaled_rotation(q)

    emb = csl_exact_spec_to_embedding(spec)

    assert emb.P is not None
    assert emb.Q is not None

    recovered = recover_exact_row_rotation_from_paired_pq(emb.P, emb.Q)

    assert recovered.denominator == rot.denominator
    np.testing.assert_array_equal(recovered.matrix, rot.matrix)
    assert not np.array_equal(recovered.matrix, rot.matrix.T)


def test_csl_embedding_pq_rows_satisfy_row_rotation_contract():
    q = (2, 0, 0, 1)
    spec = CSLExactSpec(axis=(0, 0, 1), plane=(0, 0, 1), quat=q, sigma=5)
    rot = quaternion_to_scaled_rotation(q)

    emb = csl_exact_spec_to_embedding(spec)

    assert emb.P is not None
    assert emb.Q is not None

    P_int = emb.P.astype(object)
    Q_int = emb.Q.astype(object)

    for p_row, q_row in zip(P_int, Q_int):
        numerator = p_row @ rot.matrix

        assert all(int(v) % rot.denominator == 0 for v in numerator)

        image = np.array(
            [int(v) // rot.denominator for v in numerator],
            dtype=object,
        )
        np.testing.assert_array_equal(image, q_row)


# --------------------------------------------------------------------------------------
# Exact CSL P/Q pinning
# --------------------------------------------------------------------------------------


def test_sigma5_twist_csl_embedding_pins_expected_primitive_pq_rows():
    spec = CSLExactSpec(axis=(0, 0, 1), plane=(0, 0, 1), quat=(2, 0, 0, 1), sigma=5)
    embedding = csl_exact_spec_to_embedding(spec)
    assert embedding.P is not None
    assert embedding.Q is not None

    expected_P = np.array([[0, 0, 1], [2, 1, 0], [-1, 2, 0]], dtype=int)
    expected_Q = np.array([[0, 0, 1], [2, -1, 0], [1, 2, 0]], dtype=int)
    np.testing.assert_array_equal(as_int_array(embedding.P, (3, 3), "P"), expected_P)
    np.testing.assert_array_equal(as_int_array(embedding.Q, (3, 3), "Q"), expected_Q)

    assert embedding.exact is True
    assert embedding.metadata is not None
    assert embedding.metadata.basis_mode == "primitive"


# --------------------------------------------------------------------------------------
# csl_approx_spec_to_embedding
# --------------------------------------------------------------------------------------


def test_csl_approx_spec_to_embedding_flags(approx_sigma5_spec):
    emb = csl_approx_spec_to_embedding(approx_sigma5_spec)
    _assert_embedding_flags(emb, exact=False, coherent=False, source="csl")


def test_csl_approx_spec_to_embedding_P_Q_are_none(approx_sigma5_spec):
    emb = csl_approx_spec_to_embedding(approx_sigma5_spec)
    assert emb.P is None
    assert emb.Q is None


def test_csl_approx_spec_to_embedding_proper_rotations(approx_sigma5_spec):
    emb = csl_approx_spec_to_embedding(approx_sigma5_spec)
    _assert_proper_rotation_pair(emb)


def test_csl_approx_spec_to_embedding_zero_angle_gives_same_rotations():
    spec = CSLApproxSpec(axis=[0, 0, 1], plane=[1, 0, 0], angle_deg=0.0)
    emb = csl_approx_spec_to_embedding(spec)
    np.testing.assert_allclose(emb.R_left, emb.R_right, atol=1e-12, rtol=0.0)


def test_csl_approx_spec_to_embedding_is_approximate_and_incoherent():
    spec = CSLApproxSpec(
        axis=[1, 0, 0],
        plane=[1, 1, 1],
        angle_deg=15.0,
    )

    emb = csl_approx_spec_to_embedding(spec)

    assert emb.exact is False
    assert emb.coherent is False
    assert emb.source == "csl"
    assert emb.P is None
    assert emb.Q is None

    expected_plane_unit = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)

    np.testing.assert_allclose(
        emb.R_left[0],
        expected_plane_unit,
        atol=1e-12,
        rtol=0.0,
    )

    _assert_proper_rotation_pair(emb)


# --------------------------------------------------------------------------------------
# five_dof_spec_to_embedding
# --------------------------------------------------------------------------------------


def test_five_dof_spec_to_embedding_delegates_to_rotation_embedding(
    monkeypatch,
):
    spec = FiveDOFSpec([0, 0, 0, 0, 0])
    expected_left = np.eye(3)
    expected_right = np.eye(3)
    expected_embedding = object()

    monkeypatch.setattr(
        "GBOpt.crystallography.boundary.orientation_matrices_from_five_dof",
        lambda params: (expected_left, expected_right),
    )

    calls = {}

    def fake_embedding_from_rotation_rows(
        R_left,
        R_right,
        *,
        source,
        coherent,
    ):
        calls.update(
            R_left=R_left,
            R_right=R_right,
            source=source,
            coherent=coherent,
        )
        return expected_embedding

    monkeypatch.setattr(
        "GBOpt.crystallography.boundary.embedding_from_rotation_rows",
        fake_embedding_from_rotation_rows,
    )

    result = five_dof_spec_to_embedding(spec)

    assert result is expected_embedding
    assert calls["R_left"] is expected_left
    assert calls["R_right"] is expected_right
    assert calls["source"] == "five_dof"
    assert calls["coherent"] is False


# --------------------------------------------------------------------------------------
# primitive_bicrystal_atom_count
# --------------------------------------------------------------------------------------


def test_primitive_bicrystal_atom_count_sigma5_fluorite(sigma5_twist_csl_embedding):
    assert primitive_bicrystal_atom_count(sigma5_twist_csl_embedding, 12) == 120


@pytest.mark.parametrize(
    "atoms_per_conventional_cell",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(1.5, id="float"),
        pytest.param(True, id="bool"),
        pytest.param(None, id="none"),
    ],
)
def test_primitive_bicrystal_atom_count_rejects_invalid_atoms_per_conventional_cell(
    sigma5_twist_csl_embedding,
    atoms_per_conventional_cell,
):
    with pytest.raises(
        BoundarySpecError,
        match=r"atoms_per_conventional_cell should be an integer",
    ):
        primitive_bicrystal_atom_count(
            sigma5_twist_csl_embedding,
            atoms_per_conventional_cell,
        )


def test_primitive_bicrystal_atom_count_missing_metadata_raises():
    emb = BoundaryEmbedding(
        P=None,
        Q=None,
        R_left=np.eye(3),
        R_right=np.eye(3),
        exact=False,
        coherent=False,
        source="five_dof",
    )

    with pytest.raises(
        BoundarySpecError,
        match=r"BoundaryEmbedding has no primitive-cell metadata to report",
    ):
        primitive_bicrystal_atom_count(emb, 12)
