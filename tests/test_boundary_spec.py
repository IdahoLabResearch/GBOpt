# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from typing import get_args

import numpy as np
import pytest
from GBOpt.BoundarySpec import (
    BoundaryEmbedding,
    BoundarySpecError,
    BoundarySpecTypeError,
    BoundarySpecValueError,
    CSLApproxSpec,
    CSLExactSpec,
    ConstructionMode,
    FiveDOFSpec,
    PQSpec,
    PrimitiveCellMetadata,
    _CSLSpecBase,
)


VALID_PARAMS = [0.1, 0.2, 0.3, 45.0, 30.0]
IDENTITY_P = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
IDENTITY_Q = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
SWAPPED_Q = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
ROTATED_Q = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]

VALID_AXIS = [0, 0, 1]
VALID_PLANE = [1, 0, 0]
VALID_QUAT = [3, 0, 0, 1]
VALID_ANGLE_DEG = 36.87


def _csl_kwargs(**overrides):
    kwargs = {"axis": VALID_AXIS, "plane": VALID_PLANE}
    kwargs.update(overrides)
    return kwargs


def _make_embedding(
    P=None,
    Q=None,
    exact=True,
    coherent=True,
    source="pq",
    metadata=None,
):
    R = np.eye(3)
    return BoundaryEmbedding(
        P=P,
        Q=Q,
        R_left=R,
        R_right=R,
        exact=exact,
        coherent=coherent,
        source=source,
        metadata=metadata,
    )


@pytest.fixture
def csl_base_kwargs():
    return _csl_kwargs()


@pytest.fixture
def csl_exact_kwargs():
    return _csl_kwargs(quat=VALID_QUAT)


@pytest.fixture
def csl_approx_kwargs():
    return _csl_kwargs(angle_deg=VALID_ANGLE_DEG)


def test_public_names_importable():
    assert PQSpec is not None
    assert BoundarySpecError is not None
    assert PrimitiveCellMetadata is not None


def test_construction_mode_values():
    assert set(get_args(ConstructionMode)) == {"exact", "prefer_exact", "approximate"}


def test_five_dof_spec_stores_params():
    spec = FiveDOFSpec(params=VALID_PARAMS)

    assert list(spec.params) == VALID_PARAMS


@pytest.mark.parametrize(
    ("params", "error_type"),
    [
        ([0.1, 0.2, 0.3], BoundarySpecValueError),
        (["a", "b", "c", "d", "e"], BoundarySpecTypeError),
        ([0.1, float("nan"), 0.3, 45.0, 30.0], BoundarySpecValueError),
        ([0.1, float("inf"), 0.3, 45.0, 30.0], BoundarySpecValueError),
    ],
    ids=["wrong_length", "non_numeric", "nan", "inf"],
)
def test_five_dof_spec_rejects_invalid_params(params, error_type):
    with pytest.raises(error_type):
        FiveDOFSpec(params=params)


def test_five_dof_spec_copies_mutable_input():
    params = list(VALID_PARAMS)
    spec = FiveDOFSpec(params=params)

    params[0] = 99.0

    assert spec.params[0] == pytest.approx(VALID_PARAMS[0], abs=1e-12, rel=0)


def test_five_dof_spec_rejects_boolean_params():
    with pytest.raises(BoundarySpecTypeError):
        FiveDOFSpec(params=[0.1, True, 0.3, 45.0, 30.0])


@pytest.mark.parametrize("q_matrix", [SWAPPED_Q, ROTATED_Q])
def test_pq_spec_stores_valid_matrices(q_matrix):
    spec = PQSpec(P=IDENTITY_P, Q=q_matrix)

    np.testing.assert_allclose(spec.P, IDENTITY_P, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(spec.Q, q_matrix, atol=0.0, rtol=0.0)
    assert spec.basis_mode == "primitive"


def test_pq_spec_stores_supplied_basis_mode():
    spec = PQSpec(P=IDENTITY_P, Q=IDENTITY_Q, basis_mode="supplied")

    assert spec.basis_mode == "supplied"


def test_pq_spec_rejects_invalid_basis_mode():
    with pytest.raises(BoundarySpecValueError):
        PQSpec(P=IDENTITY_P, Q=IDENTITY_Q, basis_mode="literal")


@pytest.mark.parametrize(
    "p_matrix",
    [
        [[1, 0, 0], [0, 1, 0]],
        [[1, 0, 0], [0, float("nan"), 0], [0, 0, 1]],
        [[1, 0, 0], [0, float("inf"), 0], [0, 0, 1]],
        [[1, 0, 0], [1, 0, 0], [0, 0, 1]],
    ],
    ids=["wrong_shape", "nan_entry", "inf_entry", "singular"],
)
def test_pq_spec_rejects_invalid_p_matrix(p_matrix):
    with pytest.raises(BoundarySpecError):
        PQSpec(P=p_matrix, Q=IDENTITY_Q)


@pytest.mark.parametrize(
    "q_matrix",
    [
        [[1, 0, 0], [0, 1, 0]],
        [[1, 0, 0], [0, float("nan"), 0], [0, 0, 1]],
        [[1, 0, 0], [0, float("inf"), 0], [0, 0, 1]],
        [[1, 0, 0], [1, 0, 0], [0, 0, 1]],
    ],
    ids=["wrong_shape", "nan_entry", "inf_entry", "singular"],
)
def test_pq_spec_rejects_invalid_q_matrix(q_matrix):
    with pytest.raises(BoundarySpecError):
        PQSpec(P=IDENTITY_P, Q=q_matrix)


def test_pq_spec_copies_mutable_inputs():
    P = [row[:] for row in IDENTITY_P]
    Q = [row[:] for row in ROTATED_Q]
    spec = PQSpec(P=P, Q=Q)

    P[0][0] = 9
    Q[0][0] = 9

    np.testing.assert_allclose(spec.P, IDENTITY_P, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(spec.Q, ROTATED_Q, atol=0.0, rtol=0.0)


def test_csl_base_defaults_sigma_to_none(csl_base_kwargs):
    spec = _CSLSpecBase(**csl_base_kwargs)

    assert spec.sigma is None


def test_csl_base_stores_sigma(csl_base_kwargs):
    spec = _CSLSpecBase(**csl_base_kwargs, sigma=5)

    assert spec.sigma == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        _csl_kwargs(axis=[0, 0, 0]),
        _csl_kwargs(plane=[1.5, 0.0, 0.0]),
        _csl_kwargs(sigma=0),
        _csl_kwargs(axis=[0, float("nan"), 1]),
    ],
    ids=["zero_axis", "non_integer_plane", "non_positive_sigma", "nan_axis"],
)
def test_csl_base_rejects_invalid_fields(kwargs):
    with pytest.raises(BoundarySpecValueError):
        _CSLSpecBase(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        _csl_kwargs(axis=[0, True, 1]),
        _csl_kwargs(plane=[False, 0, 1]),
        _csl_kwargs(sigma=np.bool_(True)),
    ],
    ids=["bool_axis", "bool_plane", "bool_sigma"],
)
def test_csl_base_rejects_boolean_fields(kwargs):
    with pytest.raises(BoundarySpecTypeError):
        _CSLSpecBase(**kwargs)


def test_csl_base_copies_mutable_axis_and_plane():
    axis = [0, 0, 1]
    plane = [1, 0, 0]
    spec = _CSLSpecBase(axis=axis, plane=plane, sigma=np.int64(5))

    axis[2] = 9
    plane[0] = 9

    assert spec.axis == (0, 0, 1)
    assert spec.plane == (1, 0, 0)
    assert spec.sigma == 5


def test_csl_exact_spec_stores_quat_and_sigma(csl_exact_kwargs):
    spec = CSLExactSpec(**csl_exact_kwargs, sigma=5)

    assert list(spec.quat) == VALID_QUAT
    assert spec.sigma == 5


def test_csl_exact_spec_allows_identity_quaternion():
    spec = CSLExactSpec(axis=VALID_AXIS, plane=VALID_PLANE, quat=[1, 0, 0, 0])

    assert list(spec.quat) == [1, 0, 0, 0]


def test_csl_exact_spec_rejects_sigma_mismatch_at_construction():
    with pytest.raises(BoundarySpecValueError, match="Sigma mismatch"):
        CSLExactSpec(
            axis=[0, 0, 1],
            plane=[1, 0, 0],
            quat=[2, 0, 0, 1],
            sigma=3,
        )


def test_csl_exact_spec_copies_mutable_quat():
    quat = [3, 0, 0, 1]
    spec = CSLExactSpec(axis=VALID_AXIS, plane=VALID_PLANE, quat=quat, sigma=5)

    quat[0] = 9

    assert spec.quat == (3, 0, 0, 1)


@pytest.mark.parametrize(
    ("cls", "args"),
    [
        (CSLExactSpec, (VALID_AXIS, VALID_PLANE, VALID_QUAT)),
        (CSLApproxSpec, (VALID_AXIS, VALID_PLANE, VALID_ANGLE_DEG)),
    ],
    ids=["exact", "approx"],
)
def test_csl_specs_reject_positional_construction(cls, args):
    with pytest.raises(TypeError):
        cls(*args)


@pytest.mark.parametrize(
    "kwargs",
    [
        _csl_kwargs(),
        _csl_kwargs(quat=[1.5, 0.0, 0.0, 1.0]),
        _csl_kwargs(quat=[1, 0, 0]),
        _csl_kwargs(axis=[1, 0, 0], plane=[0, 0, 1], quat=VALID_QUAT),
        _csl_kwargs(axis=[0, 0, 0], quat=VALID_QUAT),
    ],
    ids=[
        "missing_quat",
        "non_integer_quat",
        "wrong_quat_length",
        "axis_quat_mismatch",
        "zero_axis",
    ],
)
def test_csl_exact_spec_rejects_invalid_fields(kwargs):
    with pytest.raises(BoundarySpecValueError):
        CSLExactSpec(**kwargs)


def test_csl_approx_spec_stores_angle_and_sigma(csl_approx_kwargs):
    spec = CSLApproxSpec(**csl_approx_kwargs, sigma=5)

    assert spec.angle_deg == pytest.approx(VALID_ANGLE_DEG, abs=1e-12, rel=0)
    assert spec.sigma == 5


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        (_csl_kwargs(), BoundarySpecValueError),
        (_csl_kwargs(angle_deg="not_a_number"), BoundarySpecTypeError),
        (_csl_kwargs(angle_deg=float("nan")), BoundarySpecValueError),
        (_csl_kwargs(angle_deg=float("inf")), BoundarySpecValueError),
        (
            _csl_kwargs(plane=[0, 0, 0], angle_deg=VALID_ANGLE_DEG),
            BoundarySpecValueError,
        ),
    ],
    ids=["missing_angle", "non_numeric_angle", "nan_angle", "inf_angle", "zero_plane"],
)
def test_csl_approx_spec_rejects_invalid_fields(kwargs, error_type):
    with pytest.raises(error_type):
        CSLApproxSpec(**kwargs)


def test_csl_approx_spec_rejects_boolean_angle(csl_base_kwargs):
    with pytest.raises(BoundarySpecTypeError):
        CSLApproxSpec(**csl_base_kwargs, angle_deg=np.bool_(True))


def test_boundary_embedding_stores_arrays_and_default_metadata():
    P = np.array(IDENTITY_P, dtype=float)
    Q = np.array(SWAPPED_Q, dtype=float)
    emb = _make_embedding(P=P, Q=Q)

    np.testing.assert_allclose(emb.P, P, atol=1e-15, rtol=0)
    np.testing.assert_allclose(emb.Q, Q, atol=1e-15, rtol=0)
    np.testing.assert_allclose(emb.R_left, np.eye(3), atol=1e-15, rtol=0)
    np.testing.assert_allclose(emb.R_right, np.eye(3), atol=1e-15, rtol=0)
    assert emb.metadata is None


def test_boundary_embedding_stores_approximate_none_pq():
    emb = _make_embedding(
        P=None, Q=None, exact=False, coherent=False, source="five_dof"
    )

    assert emb.P is None
    assert emb.Q is None
    assert emb.exact is False
    assert emb.coherent is False
    assert emb.source == "five_dof"


def test_boundary_embedding_stores_source():
    emb = _make_embedding(source="csl")

    assert emb.source == "csl"


def test_boundary_embedding_round_trips_primitive_metadata():
    metadata = PrimitiveCellMetadata(
        basis_mode="primitive",
        supplied_area_index=10,
        primitive_area_index=5,
        reduction_index=2,
        plane=(0, 0, 1),
        rotation_denominator=10,
        conventional_cell_multiplier=10,
    )

    emb = _make_embedding(metadata=metadata)

    assert emb.metadata == metadata
    assert emb.metadata.plane == (0, 0, 1)
