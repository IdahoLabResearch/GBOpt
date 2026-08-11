# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import numpy as np
import pytest

from GBOpt.BoundarySpec import (
    BoundaryEmbedding,
    BoundarySpecTypeError,
    BoundarySpecValueError,
    CSLApproxSpec,
    CSLExactSpec,
    FiveDOFSpec,
    PQSpec,
    PrimitiveCellMetadata,
)

# --------------------------------------------------------------------------------------
# Shared test data
# --------------------------------------------------------------------------------------

VALID_PARAMS = [0.1, 0.2, 0.3, 45.0, 30.0]

IDENTITY_P = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
IDENTITY_Q = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
SWAPPED_Q = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
ROTATED_Q = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]

IDENTITY_INT = np.eye(3, dtype=object)
IDENTITY_FLOAT = np.eye(3, dtype=float)

VALID_AXIS = [0, 0, 1]
VALID_PLANE = [1, 0, 0]
VALID_QUAT = [3, 0, 0, 1]
VALID_ANGLE_DEG = 36.87

_MISSING = object()

# --------------------------------------------------------------------------------------
# Parametrized invalid inputs
# --------------------------------------------------------------------------------------

INVALID_PQ_MATRICES = [
    pytest.param(
        [[1, 0, 0], [0, 1, 0]],
        BoundarySpecValueError,
        r"must have shape \(3, 3\)",
        id="wrong_shape",
    ),
    pytest.param(
        [[1.5, 0, 0], [0, 1, 0], [0, 0, 1]],
        BoundarySpecValueError,
        r"not exactly integer-valued",
        id="non_integer",
    ),
    pytest.param(
        [[True, 0, 0], [0, 1, 0], [0, 0, 1]],
        BoundarySpecTypeError,
        r"is not an integer",
        id="boolean_entry",
    ),
    pytest.param(
        [[1, 0, 0], [0, float("nan"), 0], [0, 0, 1]],
        BoundarySpecValueError,
        r"not finite",
        id="nan",
    ),
    pytest.param(
        [[1, 0, 0], [0, float("inf"), 0], [0, 0, 1]],
        BoundarySpecValueError,
        r"not finite",
        id="inf",
    ),
    pytest.param(
        [[1, 0, 0], [1, 0, 0], [0, 0, 1]],
        BoundarySpecValueError,
        r"is singular",
        id="singular",
    ),
]


# --------------------------------------------------------------------------------------
# Helpers and fixtures
# --------------------------------------------------------------------------------------


def _csl_kwargs(**overrides):
    kwargs = {"axis": VALID_AXIS, "plane": VALID_PLANE}
    kwargs.update(overrides)
    return kwargs


def _metadata_kwargs(**overrides):
    kwargs = {
        "basis_mode": "primitive",
        "input_area_index": 10,
        "primitive_area_index": 5,
        "orientation_area_index": 5,
        "plane": (0, 0, 1),
        "rotation_denominator": 10,
    }
    kwargs.update(overrides)
    return kwargs


def _make_embedding(
    *,
    P=_MISSING,
    Q=_MISSING,
    R_left=_MISSING,
    R_right=_MISSING,
    exact=True,
    coherent=True,
    source="pq",
    metadata=None,
):
    return BoundaryEmbedding(
        P=IDENTITY_INT.copy() if P is _MISSING else P,
        Q=IDENTITY_INT.copy() if Q is _MISSING else Q,
        R_left=IDENTITY_FLOAT.copy() if R_left is _MISSING else R_left,
        R_right=IDENTITY_FLOAT.copy() if R_right is _MISSING else R_right,
        exact=exact,
        coherent=coherent,
        source=source,
        metadata=metadata,
    )


def _make_exact_spec(**overrides):
    kwargs = _csl_kwargs(quat=VALID_QUAT)
    kwargs.update(overrides)
    return CSLExactSpec(**kwargs)


def _make_approx_spec(**overrides):
    kwargs = _csl_kwargs(angle_deg=VALID_ANGLE_DEG)
    kwargs.update(overrides)
    return CSLApproxSpec(**kwargs)


@pytest.fixture
def csl_base_kwargs():
    return _csl_kwargs()


@pytest.fixture
def csl_exact_kwargs():
    return _csl_kwargs(quat=VALID_QUAT)


@pytest.fixture
def csl_approx_kwargs():
    return _csl_kwargs(angle_deg=VALID_ANGLE_DEG)


# --------------------------------------------------------------------------------------
# FiveDOFSpec
# --------------------------------------------------------------------------------------


def test_five_dof_spec_stores_params():
    spec = FiveDOFSpec(params=VALID_PARAMS)

    assert list(spec.params) == VALID_PARAMS


@pytest.mark.parametrize(
    ("params", "error_type", "message"),
    [
        (
            [0.1, 0.2, 0.3],
            BoundarySpecValueError,
            r"FiveDOFSpec\.params must have shape \(5,\)",
        ),
        (
            ["a", "b", "c", "d", "e"],
            BoundarySpecTypeError,
            r"FiveDOFSpec\.params cannot be converted to a numeric array",
        ),
        (
            [0.1, float("nan"), 0.3, 45.0, 30.0],
            BoundarySpecValueError,
            r"FiveDOFSpec\.params contains non-finite entries",
        ),
        (
            [0.1, float("inf"), 0.3, 45.0, 30.0],
            BoundarySpecValueError,
            r"FiveDOFSpec\.params contains non-finite entries",
        ),
    ],
    ids=["wrong_length", "non_numeric", "nan", "inf"],
)
def test_five_dof_spec_rejects_invalid_params(params, error_type, message):
    with pytest.raises(error_type, match=message):
        FiveDOFSpec(params=params)


def test_five_dof_spec_copies_mutable_input():
    params = list(VALID_PARAMS)
    spec = FiveDOFSpec(params=params)

    params[0] = 99.0

    assert spec.params[0] == pytest.approx(VALID_PARAMS[0], abs=1e-12, rel=0)


def test_five_dof_spec_rejects_boolean_params():
    with pytest.raises(BoundarySpecTypeError, match="must not be boolean"):
        FiveDOFSpec(params=[0.1, True, 0.3, 45.0, 30.0])


# --------------------------------------------------------------------------------------
# PQSpec
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("q_matrix", [SWAPPED_Q, ROTATED_Q])
def test_pq_spec_accepts_non_identity_q_and_defaults_to_primitive_mode(q_matrix):
    spec = PQSpec(P=IDENTITY_P, Q=q_matrix)

    np.testing.assert_allclose(spec.P, IDENTITY_P, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(spec.Q, q_matrix, atol=0.0, rtol=0.0)
    assert spec.basis_mode == "primitive"


def test_pq_spec_stores_supplied_basis_mode():
    spec = PQSpec(P=IDENTITY_P, Q=IDENTITY_Q, basis_mode="supplied")

    assert spec.basis_mode == "supplied"


def test_pq_spec_rejects_invalid_basis_mode():
    with pytest.raises(BoundarySpecValueError, match="PQSpec\\.basis_mode"):
        PQSpec(
            P=IDENTITY_P,
            Q=IDENTITY_Q,
            basis_mode="literal",  # type: ignore[ty:invalid-argument-type]
        )


@pytest.mark.parametrize("field_name", ["P", "Q"])
@pytest.mark.parametrize(
    ("bad_matrix", "error_type", "message"),
    INVALID_PQ_MATRICES,
)
def test_pq_spec_rejects_invalid_pq_matrix(field_name, bad_matrix, error_type, message):
    kwargs = {
        "P": IDENTITY_INT,
        "Q": IDENTITY_INT,
    }
    kwargs[field_name] = bad_matrix

    with pytest.raises(error_type, match=rf"PQSpec\.{field_name}.*{message}"):
        PQSpec(**kwargs)


def test_pq_spec_copies_mutable_inputs():
    P = [row[:] for row in IDENTITY_P]
    Q = [row[:] for row in ROTATED_Q]
    spec = PQSpec(P=P, Q=Q)

    P[0][0] = 9
    Q[0][0] = 9

    np.testing.assert_allclose(spec.P, IDENTITY_P, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(spec.Q, ROTATED_Q, atol=0.0, rtol=0.0)


# --------------------------------------------------------------------------------------
# CSL spec shared behavior
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [_make_exact_spec, _make_approx_spec],
    ids=["exact", "approx"],
)
def test_csl_specs_default_sigma_to_none(factory):
    spec = factory()

    assert spec.sigma is None


@pytest.mark.parametrize(
    "factory",
    [_make_exact_spec, _make_approx_spec],
    ids=["exact", "approx"],
)
def test_csl_specs_copy_mutable_axis_and_plane(factory):
    axis = [0, 0, 1]
    plane = [1, 0, 0]

    spec = factory(axis=axis, plane=plane, sigma=5)

    axis[2] = 9
    plane[0] = 9

    assert spec.axis == (0, 0, 1)
    assert spec.plane == (1, 0, 0)
    assert spec.sigma == 5


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        (
            {"axis": [0, 0, 0]},
            BoundarySpecValueError,
            r"axis must not be all-zero",
        ),
        (
            {"axis": [0, float("nan"), 1]},
            BoundarySpecValueError,
            r"axis\(1,\)=nan is not finite",
        ),
        (
            {"axis": [0, True, 1]},
            BoundarySpecTypeError,
            r"axis\(1,\)=True is not an integer",
        ),
        (
            {"axis": [0, 0]},
            BoundarySpecValueError,
            r"axis must have shape \(3,\); got \(2,\)\.",
        ),
        (
            {"plane": [1, 0]},
            BoundarySpecValueError,
            r"plane must have shape \(3,\); got \(2,\)\.",
        ),
        (
            {"plane": [0, 0, 0]},
            BoundarySpecValueError,
            r"plane must not be all-zero",
        ),
        (
            {"plane": [1.5, 0.0, 0.0]},
            BoundarySpecValueError,
            r"plane\(0,\)=1\.5 is not exactly integer-valued",
        ),
        (
            {"plane": [False, 0, 1]},
            BoundarySpecTypeError,
            r"plane\(0,\)=False is not an integer",
        ),
        (
            {"sigma": 0},
            BoundarySpecValueError,
            r"sigma must be a positive integer",
        ),
        (
            {"sigma": np.bool_(True)},
            BoundarySpecTypeError,
            r"sigma must not be boolean",
        ),
    ],
    ids=[
        "zero_axis",
        "nan_axis",
        "bool_axis",
        "wrong_axis_shape",
        "wrong_plane_shape",
        "zero_plane",
        "non_integer_plane",
        "bool_plane",
        "non_positive_sigma",
        "bool_sigma",
    ],
)
@pytest.mark.parametrize(
    "factory",
    [_make_exact_spec, _make_approx_spec, ],
    ids=["exact", "approx"],
)
def test_csl_specs_reject_invalid_shared_fields(
    factory,
    overrides,
    error_type,
    message,
):
    with pytest.raises(error_type, match=message):
        factory(**overrides)


@pytest.mark.parametrize(
    ("cls", "args"),
    [
        (CSLExactSpec, (VALID_AXIS, VALID_PLANE, VALID_QUAT)),
        (CSLApproxSpec, (VALID_AXIS, VALID_PLANE, VALID_ANGLE_DEG)),
    ],
    ids=["exact", "approx"],
)
def test_csl_specs_require_keyword_arguments(cls, args):
    with pytest.raises(TypeError):
        cls(*args)


# --------------------------------------------------------------------------------------
# CSLExactSpec
# --------------------------------------------------------------------------------------


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
    ("kwargs", "error_type", "message"),
    [
        pytest.param(
            _csl_kwargs(),
            BoundarySpecValueError,
            r"CSLExactSpec\.quat is required",
            id="missing_quat",
        ),
        pytest.param(
            _csl_kwargs(quat=[1.5, 0.0, 0.0, 1.0]),
            BoundarySpecValueError,
            r"quat\(0,\)=1\.5 is not exactly integer-valued",
            id="non_integer_quat",
        ),
        pytest.param(
            _csl_kwargs(quat=[True, 0, 0, 1]),
            BoundarySpecTypeError,
            r"quat\(0,\)=True is not an integer",
            id="boolean_quat",
        ),
        pytest.param(
            _csl_kwargs(quat=[1, 0, 0]),
            BoundarySpecValueError,
            r"quat must have shape \(4,\); got \(3,\)\.",
            id="wrong_quat_length",
        ),
        pytest.param(
            _csl_kwargs(quat=[0, 0, 0, 0]),
            BoundarySpecValueError,
            r"quat must not be all-zero",
            id="zero_quat",
        ),
        pytest.param(
            _csl_kwargs(axis=[1, 0, 0], plane=[0, 0, 1], quat=VALID_QUAT),
            BoundarySpecValueError,
            r"is not parallel to axis",
            id="axis_quat_mismatch",
        ),
        pytest.param(
            _csl_kwargs(axis=[0, 0, 0], quat=VALID_QUAT),
            BoundarySpecValueError,
            r"axis must not be all-zero",
            id="zero_axis",
        ),
    ],
)
def test_csl_exact_spec_rejects_invalid_fields(kwargs, error_type, message):
    with pytest.raises(error_type, match=message):
        CSLExactSpec(**kwargs)


def test_csl_exact_spec_reduces_nonprimitive_quaternion():
    spec = CSLExactSpec(axis=VALID_AXIS, plane=VALID_PLANE, quat=[6, 0, 0, 2])

    assert spec.quat == (3, 0, 0, 1)


# --------------------------------------------------------------------------------------
# CSLApproxSpec
# --------------------------------------------------------------------------------------


def test_csl_approx_spec_stores_angle_and_sigma(csl_approx_kwargs):
    spec = CSLApproxSpec(**csl_approx_kwargs, sigma=5)

    assert spec.angle_deg == pytest.approx(VALID_ANGLE_DEG, abs=1e-12, rel=0)
    assert spec.sigma == 5


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        (
            _csl_kwargs(),
            BoundarySpecValueError,
            r"CSLApproxSpec\.angle_deg is required",
        ),
        (
            _csl_kwargs(angle_deg="not_a_number"),
            BoundarySpecTypeError,
            r"CSLApproxSpec\.angle_deg must be a finite float",
        ),
        (
            _csl_kwargs(angle_deg=float("nan")),
            BoundarySpecValueError,
            r"CSLApproxSpec\.angle_deg must be finite",
        ),
        (
            _csl_kwargs(angle_deg=float("inf")),
            BoundarySpecValueError,
            r"CSLApproxSpec\.angle_deg must be finite",
        ),
        (
            _csl_kwargs(plane=[0, 0, 0], angle_deg=VALID_ANGLE_DEG),
            BoundarySpecValueError,
            r"plane must not be all-zero",
        ),
    ],
    ids=["missing_angle", "non_numeric_angle", "nan_angle", "inf_angle", "zero_plane"],
)
def test_csl_approx_spec_rejects_invalid_fields(kwargs, error_type, message):
    with pytest.raises(error_type, match=message):
        CSLApproxSpec(**kwargs)


def test_csl_approx_spec_rejects_boolean_angle(csl_base_kwargs):
    with pytest.raises(BoundarySpecTypeError, match="must not be boolean"):
        CSLApproxSpec(
            **csl_base_kwargs,
            angle_deg=np.bool_(True),  # type: ignore[ty:invalid-argument-type]
        )


# --------------------------------------------------------------------------------------
# PrimitiveCellMetadata
# --------------------------------------------------------------------------------------


def test_primitive_metadata_allows_orientation_area_not_multiple_of_primitive_area():
    metadata = PrimitiveCellMetadata(
        **_metadata_kwargs(
            primitive_area_index=5,
            orientation_area_index=1,
        )
    )

    assert metadata.primitive_area_index == 5
    assert metadata.orientation_area_index == 1


def test_primitive_metadata_normalizes_valid_fields():
    metadata = PrimitiveCellMetadata(
        **_metadata_kwargs(
            input_area_index=np.int64(10),
            primitive_area_index=np.int64(5),
            orientation_area_index=np.int64(5),
            plane=[0, 0, 1],
            rotation_denominator=np.int64(10),
        )
    )

    assert metadata.basis_mode == "primitive"
    assert metadata.input_area_index == 10
    assert metadata.primitive_area_index == 5
    assert metadata.input_reduction_index == 2
    assert metadata.orientation_area_index == 5
    assert metadata.plane == (0, 0, 1)
    assert metadata.rotation_denominator == 10
    assert metadata.conventional_cell_multiplier == 10


def test_primitive_cell_metadata_derives_none_reduction_without_input_area():
    metadata = PrimitiveCellMetadata(
        basis_mode="primitive",
        primitive_area_index=5,
        orientation_area_index=1,
        plane=(0, 0, 1),
        rotation_denominator=10,
    )

    assert metadata.input_area_index is None
    assert metadata.input_reduction_index is None
    assert metadata.conventional_cell_multiplier == 10


def test_primitive_metadata_supplied_mode_records_input_area_as_single_reduction():
    result = PrimitiveCellMetadata(
        basis_mode="supplied",
        input_area_index=5,
        primitive_area_index=5,
        orientation_area_index=5,
        plane=(0, 0, 1),
        rotation_denominator=5,
    )

    assert result.basis_mode == "supplied"
    assert result.input_area_index == 5
    assert result.primitive_area_index == 5
    assert result.orientation_area_index == 5
    assert result.input_reduction_index == 1
    assert result.conventional_cell_multiplier == 10


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        pytest.param(
            {"basis_mode": "orthogonal"},
            BoundarySpecValueError,
            r"PrimitiveCellMetadata\.basis_mode",
            id="invalid-basis-mode",
        ),
        pytest.param(
            {"primitive_area_index": 0},
            BoundarySpecValueError,
            r"PrimitiveCellMetadata\.primitive_area_index "
            r"must be a positive integer",
            id="zero-primitive-area-index",
        ),
        pytest.param(
            {"primitive_area_index": -1},
            BoundarySpecValueError,
            r"PrimitiveCellMetadata\.primitive_area_index "
            r"must be a positive integer",
            id="negative-primitive-area-index",
        ),
        pytest.param(
            {"primitive_area_index": 1.5},
            BoundarySpecValueError,
            r"PrimitiveCellMetadata\.primitive_area_index "
            r"must be a positive integer",
            id="noninteger-primitive-area-index",
        ),
        pytest.param(
            {"primitive_area_index": np.bool_(True)},
            BoundarySpecTypeError,
            r"PrimitiveCellMetadata\.primitive_area_index "
            r"must not be boolean",
            id="boolean-primitive-area-index",
        ),
        pytest.param(
            {"input_area_index": 0},
            BoundarySpecValueError,
            r"PrimitiveCellMetadata\.input_area_index "
            r"must be a positive integer",
            id="zero-input-area-index",
        ),
        pytest.param(
            {"input_area_index": -1},
            BoundarySpecValueError,
            r"PrimitiveCellMetadata\.input_area_index "
            r"must be a positive integer",
            id="negative-input-area-index",
        ),
        pytest.param(
            {"input_area_index": 7},
            BoundarySpecValueError,
            r"PrimitiveCellMetadata\.input_area_index "
            r"must be an integer multiple of primitive_area_index",
            id="indivisible-input-area-index",
        ),
        pytest.param(
            {"orientation_area_index": 0},
            BoundarySpecValueError,
            r"PrimitiveCellMetadata\.orientation_area_index "
            r"must be a positive integer",
            id="zero-orientation-area-index",
        ),
        pytest.param(
            {"orientation_area_index": -1},
            BoundarySpecValueError,
            r"PrimitiveCellMetadata\.orientation_area_index "
            r"must be a positive integer",
            id="negative-orientation-area-index",
        ),
        pytest.param(
            {"orientation_area_index": 1.5},
            BoundarySpecValueError,
            r"PrimitiveCellMetadata\.orientation_area_index "
            r"must be a positive integer",
            id="noninteger-orientation-area-index",
        ),
        pytest.param(
            {"orientation_area_index": np.bool_(True)},
            BoundarySpecTypeError,
            r"PrimitiveCellMetadata\.orientation_area_index "
            r"must not be boolean",
            id="boolean-orientation-area-index",
        ),
        pytest.param(
            {"plane": [1, 0]},
            BoundarySpecValueError,
            r"PrimitiveCellMetadata\.plane must have shape \(3,\)",
            id="short-plane",
        ),
        pytest.param(
            {"plane": [0, 0, 0]},
            BoundarySpecValueError,
            r"PrimitiveCellMetadata\.plane must not be all-zero",
            id="zero-plane",
        ),
        pytest.param(
            {"plane": [1.5, 0, 0]},
            BoundarySpecValueError,
            r"PrimitiveCellMetadata\.plane\(0,\)=1\.5 "
            r"is not exactly integer-valued",
            id="noninteger-plane",
        ),
        pytest.param(
            {"plane": [True, 0, 0]},
            BoundarySpecTypeError,
            r"PrimitiveCellMetadata\.plane\(0,\)=True "
            r"is not an integer",
            id="boolean-plane",
        ),
        pytest.param(
            {"rotation_denominator": 0},
            BoundarySpecValueError,
            r"PrimitiveCellMetadata\.rotation_denominator "
            r"must be a positive integer",
            id="zero-rotation-denominator",
        ),
        pytest.param(
            {"rotation_denominator": 1.5},
            BoundarySpecValueError,
            r"PrimitiveCellMetadata\.rotation_denominator "
            r"must be a positive integer",
            id="noninteger-rotation-denominator",
        ),
        pytest.param(
            {"rotation_denominator": True},
            BoundarySpecTypeError,
            r"PrimitiveCellMetadata\.rotation_denominator "
            r"must not be boolean",
            id="boolean-rotation-denominator",
        ),
    ],
)
def test_primitive_metadata_rejects_invalid_fields(
    overrides,
    error_type,
    message,
):
    with pytest.raises(error_type, match=message):
        PrimitiveCellMetadata(**_metadata_kwargs(**overrides))


# --------------------------------------------------------------------------------------
# BoundaryEmbedding
# --------------------------------------------------------------------------------------


def test_boundary_embedding_converts_pq_to_integer_arrays_and_defaults_metadata():
    P = np.array(IDENTITY_P, dtype=float)
    Q = np.array(SWAPPED_Q, dtype=float)

    emb = _make_embedding(P=P, Q=Q)

    np.testing.assert_array_equal(emb.P, np.asarray(P, dtype=object))
    np.testing.assert_array_equal(emb.Q, np.asarray(Q, dtype=object))
    np.testing.assert_allclose(emb.R_left, np.eye(3), atol=1e-15, rtol=0)
    np.testing.assert_allclose(emb.R_right, np.eye(3), atol=1e-15, rtol=0)
    assert emb.metadata is None
    assert emb.P.dtype == object
    assert emb.Q.dtype == object
    assert emb.exact is True
    assert emb.coherent is True
    assert emb.source == "pq"


def test_boundary_embedding_stores_approximate_none_pq():
    emb = _make_embedding(
        P=None, Q=None, exact=False, coherent=False, source="five_dof"
    )

    assert emb.P is None
    assert emb.Q is None
    assert emb.exact is False
    assert emb.coherent is False
    assert emb.source == "five_dof"


def test_boundary_embedding_accepts_supported_source():
    emb = _make_embedding(source="csl")

    assert emb.source == "csl"


def test_boundary_embedding_stores_primitive_metadata():
    metadata = PrimitiveCellMetadata(
        basis_mode="primitive",
        input_area_index=10,
        primitive_area_index=5,
        orientation_area_index=5,
        plane=(0, 0, 1),
        rotation_denominator=10,
    )

    emb = _make_embedding(metadata=metadata)

    assert emb.metadata is metadata


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        pytest.param(
            {"source": "unknown"},
            BoundarySpecValueError,
            r"source must be one of",
            id="invalid_source",
        ),
        pytest.param(
            {"P": None, "exact": True},
            BoundarySpecValueError,
            r"Exact BoundaryEmbedding requires P and Q matrices",
            id="exact_missing_p",
        ),
        pytest.param(
            {"Q": None, "exact": True},
            BoundarySpecValueError,
            r"Exact BoundaryEmbedding requires P and Q matrices",
            id="exact_missing_q",
        ),
        pytest.param(
            {"exact": 1},
            BoundarySpecTypeError,
            r"exact must be a bool",
            id="non_bool_exact",
        ),
        pytest.param(
            {"coherent": "yes"},
            BoundarySpecTypeError,
            r"coherent must be a bool",
            id="non_bool_coherent",
        ),
        pytest.param(
            {"metadata": object()},
            BoundarySpecTypeError,
            r"metadata must be a PrimitiveCellMetadata instance or None",
            id="invalid_metadata_type",
        ),
        pytest.param(
            {"R_left": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]},
            BoundarySpecValueError,
            r"R_left must have shape \(3, 3\)",
            id="malformed_r_left_shape",
        ),
        pytest.param(
            {
                "R_right": [
                    [1.0, 0.0, 0.0],
                    [0.0, float("nan"), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            },
            BoundarySpecValueError,
            r"R_right contains non-finite entries",
            id="non_finite_r_right",
        ),
        pytest.param(
            {"P": [[1, 0, 0], [0, 1, 0]]},
            BoundarySpecValueError,
            r"P must have shape \(3, 3\)",
            id="malformed_p_shape",
        ),
        pytest.param(
            {"Q": [[1.5, 0, 0], [0, 1, 0], [0, 0, 1]]},
            BoundarySpecValueError,
            r"Q\(0, 0\)=1\.5 is not exactly integer-valued",
            id="non_integer_q",
        ),
        pytest.param(
            {"P": [[True, 0, 0], [0, 1, 0], [0, 0, 1]]},
            BoundarySpecTypeError,
            r"P\(0, 0\)=True is not an integer",
            id="boolean_p",
        ),
    ],
)
def test_boundary_embedding_rejects_invalid_inputs(overrides, error_type, message):
    with pytest.raises(error_type, match=message):
        _make_embedding(**overrides)


def test_boundary_embedding_copies_and_freezes_arrays():
    P = np.eye(3, dtype=object)
    Q = np.eye(3, dtype=object)
    R_left = np.eye(3, dtype=float)
    R_right = np.eye(3, dtype=float)

    emb = _make_embedding(P=P, Q=Q, R_left=R_left, R_right=R_right)

    P[0, 0] = 9
    Q[1, 1] = 9
    R_left[0, 0] = 9.0
    R_right[1, 1] = 9.0

    np.testing.assert_array_equal(emb.P, IDENTITY_INT)
    np.testing.assert_array_equal(emb.Q, IDENTITY_INT)
    np.testing.assert_array_equal(emb.R_left, IDENTITY_FLOAT)
    np.testing.assert_array_equal(emb.R_right, IDENTITY_FLOAT)

    assert emb.P is not P
    assert emb.Q is not Q
    assert emb.R_left is not R_left
    assert emb.R_right is not R_right

    assert not emb.P.flags.writeable
    assert not emb.Q.flags.writeable
    assert not emb.R_left.flags.writeable
    assert not emb.R_right.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        emb.P[0, 0] = 2

    with pytest.raises(ValueError, match="read-only"):
        emb.R_left[0, 0] = 2.0
