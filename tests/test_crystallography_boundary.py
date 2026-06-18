# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from unittest.mock import patch

import numpy as np
import pytest

from GBOpt.BoundarySpec import (
    BoundaryEmbedding,
    BoundarySpecError,
    BoundarySpecOrthogonalityError,
    CSLApproxSpec,
    CSLExactSpec,
    PQSpec,
)
from GBOpt.crystallography import (
    csl_approx_spec_to_embedding,
    csl_spec_to_embedding,
    pq_spec_to_embedding,
    primitive_bicrystal_atom_count,
)
from GBOpt.crystallography.integer import integer_det3
from GBOpt.crystallography.pq import (
    canonicalize_pq,
    recover_exact_row_rotation_from_paired_pq,
)
from GBOpt.crystallography.quaternion import quaternion_to_scaled_rotation

SIGMA5_36_P = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
SIGMA5_36_Q = ((4, -3, 0), (3, 4, 0), (0, 0, 1))
SIGMA5_TWIST_LEGACY_P = ((0, 0, 1), (3, 1, 0), (-1, 3, 0))
SIGMA5_TWIST_LEGACY_Q = ((0, 0, 1), (3, -1, 0), (1, 3, 0))
SIGMA5_TWIST_PRIMITIVE_P = ((0, 0, 1), (1, 2, 0), (-2, 1, 0))
SIGMA5_TWIST_PRIMITIVE_Q = ((0, 0, 1), (2, 1, 0), (-1, 2, 0))


# ---------------------------------------------------------------------------
# pq_spec_to_embedding
# ---------------------------------------------------------------------------

def test_pq_spec_to_embedding_flags_and_source():
    spec = PQSpec(
        P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        Q=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    emb = pq_spec_to_embedding(spec)
    assert isinstance(emb, BoundaryEmbedding)
    assert emb.exact is True
    assert emb.coherent is True
    assert emb.source == "pq"


def test_pq_spec_to_embedding_identity_r_left_r_right():
    spec = PQSpec(
        P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        Q=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    emb = pq_spec_to_embedding(spec)
    np.testing.assert_array_almost_equal(emb.R_left, np.eye(3))
    np.testing.assert_array_almost_equal(emb.R_right, np.eye(3))


def test_pq_spec_to_embedding_sigma5_r_right_matches_canonical():
    P_raw = np.array(SIGMA5_36_P, dtype=float)
    Q_raw = np.array(SIGMA5_36_Q, dtype=float)
    spec = PQSpec(P=P_raw.tolist(), Q=Q_raw.tolist(), basis_mode="supplied")
    emb = pq_spec_to_embedding(spec)
    _, Q_c = canonicalize_pq(P_raw, Q_raw)
    R_right_expected = Q_c / np.linalg.norm(Q_c, axis=1, keepdims=True)
    np.testing.assert_array_almost_equal(emb.R_right, R_right_expected)


def test_pq_spec_to_embedding_sigma5_r_right_is_proper_rotation():
    spec = PQSpec(
        P=list(SIGMA5_36_P),
        Q=list(SIGMA5_36_Q),
        basis_mode="supplied",
    )
    emb = pq_spec_to_embedding(spec)
    assert abs(np.linalg.det(emb.R_right) - 1.0) < 1e-12
    np.testing.assert_array_almost_equal(emb.R_right @ emb.R_right.T, np.eye(3))


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
    with pytest.raises(BoundarySpecError):
        pq_spec_to_embedding(spec)


# ---------------------------------------------------------------------------
# pq_spec_to_embedding -- primitive basis mode
# ---------------------------------------------------------------------------

def test_pq_spec_primitive_legacy_sigma5_twist_area_index_5():
    emb = pq_spec_to_embedding(
        PQSpec(P=list(SIGMA5_TWIST_LEGACY_P), Q=list(SIGMA5_TWIST_LEGACY_Q))
    )
    assert integer_det3(emb.P) == 5
    assert integer_det3(emb.Q) == 5
    assert emb.metadata is not None
    assert emb.metadata.basis_mode == "primitive"
    assert emb.metadata.supplied_area_index == 10
    assert emb.metadata.primitive_area_index == 5
    assert emb.metadata.reduction_index == 2


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


@pytest.mark.parametrize("P,Q,expected_plane", [
    ([[2, 1, 0], [0, 0, 1], [1, -2, 0]],
     [[2, -1, 0], [0, 0, 1], [-1, -2, 0]], (2, 1, 0)),
    ([[3, 1, 0], [0, 0, 1], [1, -3, 0]],
     [[3, -1, 0], [0, 0, 1], [-1, -3, 0]], (3, 1, 0)),
])
def test_pq_spec_primitive_tilt_boundaries_preserve_plane(P, Q, expected_plane):
    emb = pq_spec_to_embedding(PQSpec(P=P, Q=Q))
    assert emb.metadata is not None
    assert emb.metadata.plane == expected_plane
    assert emb.P is not None
    np.testing.assert_array_equal(emb.P[0].astype(int), expected_plane)


def test_pq_spec_supplied_mode_preserves_area_index_10():
    emb = pq_spec_to_embedding(
        PQSpec(
            P=list(SIGMA5_TWIST_LEGACY_P),
            Q=list(SIGMA5_TWIST_LEGACY_Q),
            basis_mode="supplied",
        )
    )
    assert integer_det3(emb.P) == 10
    assert integer_det3(emb.Q) == 10
    assert emb.metadata is not None
    assert emb.metadata.basis_mode == "supplied"
    assert emb.metadata.supplied_area_index == 10
    assert emb.metadata.primitive_area_index == 10


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
    assert actual_rotation.N == expected_rotation.N
    np.testing.assert_array_equal(actual_rotation.M, expected_rotation.M)


def test_pq_spec_primitive_warns_when_rotation_recovery_falls_back():
    spec = PQSpec(
        P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        Q=[[4, -3, 0], [3, 4, 0], [0, 0, 1]],
    )
    with pytest.warns(UserWarning, match="basis_mode='primitive'.*falling back"):
        emb = pq_spec_to_embedding(spec)
    assert emb.metadata is None
    assert integer_det3(emb.Q) == 25


def test_pq_spec_primitive_warns_when_embedding_reconstruction_falls_back():

    spec = PQSpec(
        P=list(SIGMA5_TWIST_PRIMITIVE_P),
        Q=list(SIGMA5_TWIST_PRIMITIVE_Q),
    )
    with patch(
        "GBOpt.crystallography.boundary.primitive_embedding_from_row_rotation",
        side_effect=BoundarySpecError("forced primitive failure"),
    ):
        with pytest.warns(UserWarning, match="forced primitive failure"):
            emb = pq_spec_to_embedding(spec)
    assert emb.metadata is not None
    assert emb.metadata.basis_mode == "supplied"


# ---------------------------------------------------------------------------
# csl_spec_to_embedding
# ---------------------------------------------------------------------------

def test_csl_spec_to_embedding_flags_and_proper_rotations():
    spec = CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 0, 0, 1])
    emb = csl_spec_to_embedding(spec)
    assert isinstance(emb, BoundaryEmbedding)
    assert emb.exact is True
    assert emb.coherent is True
    assert emb.source == "csl"
    for R in (emb.R_left, emb.R_right):
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


def test_csl_spec_to_embedding_identity_quaternion_zero_misorientation():
    spec = CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[1, 0, 0, 0], sigma=1)
    emb = csl_spec_to_embedding(spec)
    assert emb.exact is True
    np.testing.assert_allclose(emb.R_left, emb.R_right, atol=1e-12)


def test_csl_spec_to_embedding_cross_format_round_trip():
    spec_csl = CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 0, 0, 1])
    spec_pq = PQSpec(P=list(SIGMA5_36_P), Q=list(SIGMA5_36_Q), basis_mode="supplied")
    emb_csl = csl_spec_to_embedding(spec_csl)
    emb_pq = pq_spec_to_embedding(spec_pq)

    assert emb_csl.P is not None
    assert emb_csl.Q is not None
    assert emb_pq.P is not None
    assert emb_pq.Q is not None
    np.testing.assert_array_equal(emb_csl.P, emb_pq.P)
    np.testing.assert_array_equal(emb_csl.Q, emb_pq.Q)


def test_csl_spec_to_embedding_non_preserving_plane_fallback():
    spec = CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 0, 0, 1])
    emb = csl_spec_to_embedding(spec)
    assert emb.exact is True
    assert emb.coherent is True


def test_csl_spec_to_embedding_sigma3_111_proper_rotations():
    spec = CSLExactSpec(axis=[1, 1, 1], plane=[1, 1, 1], quat=[3, 1, 1, 1])
    emb = csl_spec_to_embedding(spec)
    for label, R in [("R_left", emb.R_left), ("R_right", emb.R_right)]:
        np.testing.assert_allclose(
            R @ R.T, np.eye(3), atol=1e-10,
            err_msg=f"{label} is not orthogonal for Sigma3 [111] boundary"
        )
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


def test_csl_spec_to_embedding_orthogonality_error_falls_back():
    spec = CSLExactSpec(axis=[0, 0, 1], plane=[0, 0, 1], quat=[3, 0, 0, 1])
    with patch(
        "GBOpt.crystallography.boundary.primitive_embedding_from_row_rotation",
        side_effect=BoundarySpecOrthogonalityError("forced orthogonality failure"),
    ) as primitive_builder:
        emb = csl_spec_to_embedding(spec)
    primitive_builder.assert_called_once()
    assert emb.exact is True
    assert emb.coherent is True
    for R in (emb.R_left, emb.R_right):
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


def test_csl_spec_to_embedding_twist_matches_primitive_pqspec():
    spec_csl = CSLExactSpec(axis=[0, 0, 1], plane=[0, 0, 1], quat=[3, 0, 0, 1])
    spec_pq = PQSpec(P=list(SIGMA5_TWIST_LEGACY_P), Q=list(SIGMA5_TWIST_LEGACY_Q))
    emb_csl = csl_spec_to_embedding(spec_csl)
    emb_pq = pq_spec_to_embedding(spec_pq)

    assert emb_csl.P is not None
    assert emb_csl.Q is not None
    assert emb_pq.P is not None
    assert emb_pq.Q is not None
    np.testing.assert_array_equal(emb_csl.P, emb_pq.P)
    np.testing.assert_array_equal(emb_csl.Q, emb_pq.Q)
    assert integer_det3(emb_csl.P) == 5
    assert integer_det3(emb_csl.Q) == 5


def test_csl_spec_to_embedding_twist_rotations_derive_from_final_pq():
    spec = CSLExactSpec(axis=[0, 0, 1], plane=[0, 0, 1], quat=[3, 0, 0, 1])
    emb = csl_spec_to_embedding(spec)
    assert emb.P is not None
    assert emb.Q is not None
    expected_left = emb.P / np.linalg.norm(emb.P, axis=1, keepdims=True)
    expected_right = emb.Q / np.linalg.norm(emb.Q, axis=1, keepdims=True)
    np.testing.assert_allclose(emb.R_left, expected_left, atol=1e-12)
    np.testing.assert_allclose(emb.R_right, expected_right, atol=1e-12)
    assert not np.allclose(emb.R_left, emb.R_right, atol=1e-12)
    assert emb.metadata is not None
    assert emb.metadata.basis_mode == "primitive"


def test_csl_spec_to_embedding_sigma_mismatch_raises():
    with pytest.raises(BoundarySpecError, match="[Ss]igma"):
        csl_spec_to_embedding(
            CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[2, 0, 0, 1], sigma=3)
        )


def test_csl_spec_to_embedding_missing_quat_raises():
    with pytest.raises(BoundarySpecError):
        csl_spec_to_embedding(CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0]))


def test_csl_spec_to_embedding_large_denominator_stays_exact():
    spec = CSLExactSpec(axis=[128, 1, 1], plane=[1, 0, 0], quat=[128, 128, 1, 1])
    emb = csl_spec_to_embedding(spec, max_exact_atoms=10**9)
    assert emb.exact is True
    assert emb.P is not None
    assert emb.Q is not None
    for R in (emb.R_left, emb.R_right):
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    np.testing.assert_array_equal(emb.P.astype(int), np.eye(3, dtype=int))
    expected_Q = np.array(
        [[16383, 0, 256], [256, 0, -16383], [0, 1, 0]], dtype=int
    )
    np.testing.assert_array_equal(emb.Q.astype(int), expected_Q)


# ---------------------------------------------------------------------------
# Row-rotation convention (deferred from Phase 3 step 19 and csl tests)
# ---------------------------------------------------------------------------

def test_csl_embedding_uses_row_rotation_convention():
    q = (2, 0, 0, 1)
    spec = CSLExactSpec(axis=(0, 0, 1), plane=(0, 0, 1), quat=q, sigma=5)
    rot = quaternion_to_scaled_rotation(q)
    emb = csl_spec_to_embedding(spec)
    assert emb.P is not None
    assert emb.Q is not None

    recovered = recover_exact_row_rotation_from_paired_pq(emb.P, emb.Q)
    assert recovered.N == rot.N
    np.testing.assert_array_equal(recovered.M, rot.M)
    assert not np.array_equal(recovered.M, rot.M.T)

    P_int = emb.P.astype(object)
    Q_int = emb.Q.astype(object)
    for p_row, q_row in zip(P_int, Q_int):
        numerator = p_row @ rot.M
        assert all(int(v) % rot.N == 0 for v in numerator)
        image = np.array([int(v) // rot.N for v in numerator], dtype=object)
        np.testing.assert_array_equal(image, q_row)


# ---------------------------------------------------------------------------
# Round-trip P/Q pinning (deferred from csl tests)
# ---------------------------------------------------------------------------

def test_round_trip_embedding_pins_expected_pq_and_flags():
    spec = CSLExactSpec(axis=(0, 0, 1), plane=(0, 0, 1), quat=(2, 0, 0, 1), sigma=5)
    embedding = csl_spec_to_embedding(spec)
    assert embedding.P is not None
    assert embedding.Q is not None

    expected_P = np.array([[0, 0, 1], [2, 1, 0], [-1, 2, 0]], dtype=int)
    expected_Q = np.array([[0, 0, 1], [2, -1, 0], [1, 2, 0]], dtype=int)
    np.testing.assert_array_equal(np.round(embedding.P).astype(int), expected_P)
    np.testing.assert_array_equal(np.round(embedding.Q).astype(int), expected_Q)

    assert embedding.exact is True
    assert embedding.metadata is not None
    assert embedding.metadata.basis_mode == "primitive"


# ---------------------------------------------------------------------------
# csl_approx_spec_to_embedding
# ---------------------------------------------------------------------------

def test_csl_approx_spec_to_embedding_flags():
    spec = CSLApproxSpec(axis=[0, 0, 1], plane=[1, 0, 0], angle_deg=36.87)
    emb = csl_approx_spec_to_embedding(spec)
    assert emb.exact is False
    assert emb.coherent is True
    assert emb.source == "csl"


def test_csl_approx_spec_to_embedding_P_Q_are_none():
    spec = CSLApproxSpec(axis=[0, 0, 1], plane=[1, 0, 0], angle_deg=36.87)
    emb = csl_approx_spec_to_embedding(spec)
    assert emb.P is None
    assert emb.Q is None


def test_csl_approx_spec_to_embedding_proper_rotations():
    spec = CSLApproxSpec(axis=[0, 0, 1], plane=[1, 0, 0], angle_deg=36.87)
    emb = csl_approx_spec_to_embedding(spec)
    for R in (emb.R_left, emb.R_right):
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


def test_csl_approx_spec_to_embedding_zero_angle_gives_same_rotations():
    spec = CSLApproxSpec(axis=[0, 0, 1], plane=[1, 0, 0], angle_deg=0.0)
    emb = csl_approx_spec_to_embedding(spec)
    np.testing.assert_allclose(emb.R_left, emb.R_right, atol=1e-12)


# ---------------------------------------------------------------------------
# primitive_bicrystal_atom_count
# ---------------------------------------------------------------------------

def test_primitive_bicrystal_atom_count_sigma5_fluorite():
    spec = CSLExactSpec(axis=[0, 0, 1], plane=[0, 0, 1], quat=[3, 0, 0, 1])
    emb = csl_spec_to_embedding(spec)
    assert emb.metadata is not None
    assert primitive_bicrystal_atom_count(emb, 12) == 120


def test_primitive_bicrystal_atom_count_missing_metadata_raises():
    emb = BoundaryEmbedding(
        P=None, Q=None,
        R_left=np.eye(3), R_right=np.eye(3),
        exact=False, coherent=False, source="five_dof",
    )
    with pytest.raises(BoundarySpecError):
        primitive_bicrystal_atom_count(emb, 12)

# ---------------------------------------------------------------------------
# CSLExactSpec validation
# ---------------------------------------------------------------------------


def test_csl_exact_spec_valid_instantiates():
    CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[2, 0, 0, 1])


def test_csl_exact_spec_identity_quaternion_instantiates():
    spec = CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[1, 0, 0, 0], sigma=1)
    assert spec.quat is not None
    assert list(spec.quat) == [1, 0, 0, 0]


def test_csl_exact_spec_missing_quat_raises():
    with pytest.raises(BoundarySpecError):
        CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0])


def test_csl_exact_spec_malformed_quat_raises():
    with pytest.raises(BoundarySpecError):
        CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[2, 0, 1])


def test_csl_exact_spec_non_integer_plane_raises():
    with pytest.raises(BoundarySpecError):
        CSLExactSpec(axis=[0, 0, 1], plane=[1.5, 0, 0], quat=[
                     2, 0, 0, 1])  # type: ignore[arg-type]


def test_csl_exact_spec_non_integer_axis_raises():
    with pytest.raises(BoundarySpecError):
        CSLExactSpec(axis=[0.7, 0, 1], plane=[1, 0, 0], quat=[
                     2, 0, 0, 1])  # type: ignore[arg-type]


def test_csl_exact_spec_zero_axis_raises():
    with pytest.raises(BoundarySpecError):
        CSLExactSpec(axis=[0, 0, 0], plane=[1, 0, 0], quat=[2, 0, 0, 1])


def test_csl_exact_spec_zero_plane_raises():
    with pytest.raises(BoundarySpecError):
        CSLExactSpec(axis=[0, 0, 1], plane=[0, 0, 0], quat=[2, 0, 0, 1])


def test_csl_exact_spec_axis_quat_mismatch_raises():
    # quat [2, 0, 0, 1] encodes rotation about [0, 0, 1]; axis=[1, 0, 0] is wrong.
    with pytest.raises(BoundarySpecError):
        CSLExactSpec(axis=[1, 0, 0], plane=[1, 0, 0], quat=[2, 0, 0, 1])


def test_csl_exact_spec_sigma_mismatch_raises():
    # quat [2, 0, 0, 1] -> sigma=5, not 3
    with pytest.raises(BoundarySpecError):
        CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[2, 0, 0, 1], sigma=3)
