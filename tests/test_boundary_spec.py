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


@pytest.mark.parametrize("q_matrix", [SWAPPED_Q, ROTATED_Q])
def test_pq_spec_stores_valid_matrices(q_matrix):
    spec = PQSpec(P=IDENTITY_P, Q=q_matrix)

    assert spec.P == IDENTITY_P
    assert spec.Q == q_matrix
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


def test_csl_exact_spec_stores_quat_and_sigma(csl_exact_kwargs):
    spec = CSLExactSpec(**csl_exact_kwargs, sigma=5)

    assert list(spec.quat) == VALID_QUAT
    assert spec.sigma == 5


def test_csl_exact_spec_allows_identity_quaternion():
    spec = CSLExactSpec(axis=VALID_AXIS, plane=VALID_PLANE, quat=[1, 0, 0, 0])

    assert list(spec.quat) == [1, 0, 0, 0]


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
