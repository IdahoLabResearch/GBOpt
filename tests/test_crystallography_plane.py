# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
import numpy as np
import pytest

from GBOpt.crystallography.plane import (
    inplane_area_index,
    inplane_basis_from_csl,
    plane_null_basis,
    primitive_plane,
    rotation_preserves_plane,
)
from GBOpt.crystallography.quaternion import quaternion_to_scaled_rotation
from GBOpt.crystallography.types import (
    CrystallographyNotImplementedError,
    CrystallographyValueError,
)
from GBOpt.Utils.integer_linalg import cross_int3

# Sigma5 [001] 53.13 deg -- quat [2, 0, 0, 1], N=5
# Rotation preserves [0, 0, 1] plane but not [1, 0, 0]
SIGMA5_QUAT = (2, 0, 0, 1)


def _sigma5_001_csl_hnf():
    return np.array([[1, 0, 0], [2, 5, 0], [0, 0, 1]], dtype=object)


def _assert_plane_null_basis_is_primitive_and_oriented(plane) -> None:
    p = np.array(plane, dtype=object)
    e1, e2 = plane_null_basis(p)

    assert e1.dtype == object
    assert e2.dtype == object
    assert int(p @ e1) == 0
    assert int(p @ e2) == 0
    np.testing.assert_array_equal(cross_int3(e1, e2), p)


# ---------------------------------------------------------------------------
# primitive_plane
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("plane_covector", "expected"),
    [
        pytest.param([1, 2, 3], (1, 2, 3), id="already-primitive"),
        pytest.param([2, 4, 6], (1, 2, 3), id="reduces-nonprimitive"),
        pytest.param([-2, 4, -6], (-1, 2, -3), id="preserves-sign-convention"),
    ],
)
def test_primitive_plane_returns_gcd_reduced_int3_tuple(plane_covector, expected):
    result = primitive_plane(plane_covector)

    assert result == expected
    assert isinstance(result, tuple)


@pytest.mark.parametrize(
    ("plane_covector", "match"),
    [
        pytest.param([0, 0, 0], "zero vector", id="zero-vector"),
        pytest.param([1.5, 0, 0], "integer-valued", id="non-integer-float"),
        pytest.param([True, 0, 0], "not an integer", id="bool"),
        pytest.param([np.nan, 0, 0], "not finite", id="nan"),
        pytest.param([1, 0], "shape", id="wrong-length"),
    ],
)
def test_primitive_plane_rejects_invalid_covector(plane_covector, match):
    with pytest.raises(CrystallographyValueError, match=match):
        primitive_plane(plane_covector)


# ---------------------------------------------------------------------------
# plane_null_basis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plane",
    [
        pytest.param((1, 0, 0), id="100"),
        pytest.param((1, 1, 1), id="111"),
        pytest.param((5, 2, 3), id="523"),
        pytest.param((-1, 2, -1), id="negative-components"),
    ],
)
def test_plane_null_basis_returns_oriented_primitive_basis(plane):
    _assert_plane_null_basis_is_primitive_and_oriented(plane)


@pytest.mark.parametrize(
    ("plane", "match"),
    [
        pytest.param([0, 0, 0], "zero vector", id="zero-vector"),
        pytest.param([2, 0, 0], "not primitive", id="non-primitive"),
        pytest.param([1.5, 0, 0], "integer-valued", id="non-integer-float"),
        pytest.param([True, 0, 0], "not an integer", id="bool"),
        pytest.param([1, 0], "shape", id="wrong-length"),
    ],
)
def test_plane_null_basis_rejects_invalid_plane(plane, match):
    with pytest.raises(CrystallographyValueError, match=match):
        plane_null_basis(plane)


# ---------------------------------------------------------------------------
# inplane_basis_from_csl
# These are direct unit tests for plane.inplane_basis_from_csl using fixed CSL bases.
# Rotation-derived CSL integration coverage lives in test_crystallography_csl.py.
# ---------------------------------------------------------------------------


def test_inplane_basis_from_csl_vectors_are_in_plane():
    csl_basis = _sigma5_001_csl_hnf()
    result = inplane_basis_from_csl(csl_basis, (0, 0, 1))
    h = np.array([0, 0, 1], dtype=object)
    assert int(h @ result.basis[:, 0]) == 0
    assert int(h @ result.basis[:, 1]) == 0


def test_inplane_basis_from_csl_returns_basis_coefficients_and_plane_covector():
    csl_basis = _sigma5_001_csl_hnf()

    result = inplane_basis_from_csl(csl_basis, (0, 0, 1))

    np.testing.assert_array_equal(result.basis, csl_basis @ result.coefficients)
    assert result.plane_covector == (0, 0, 1)
    np.testing.assert_array_equal(
        cross_int3(result.basis[:, 0], result.basis[:, 1]),
        np.array([0, 0, 5], dtype=object),
    )


@pytest.mark.parametrize(
    ("csl_basis", "match"),
    [
        pytest.param(
            np.array([[1.5, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
            "integer-valued",
            id="non-integer-basis",
        ),
        pytest.param(
            np.eye(2, dtype=object),
            "shape",
            id="wrong-shape",
        ),
    ],
)
def test_inplane_basis_from_csl_rejects_invalid_csl_basis(csl_basis, match):
    with pytest.raises(CrystallographyValueError, match=match):
        inplane_basis_from_csl(csl_basis, (0, 0, 1))


def test_inplane_basis_from_csl_lattice_metric_raises():
    csl_basis = _sigma5_001_csl_hnf()

    with pytest.raises(CrystallographyNotImplementedError, match="non-cubic"):
        inplane_basis_from_csl(csl_basis, (0, 0, 1), lattice_metric=np.eye(3))


@pytest.mark.parametrize(
    ("plane_covector", "match"),
    [
        pytest.param([0, 0, 0], "zero vector", id="zero-plane"),
        pytest.param([1.5, 0, 0], "integer-valued", id="non-integer-float"),
        pytest.param([True, 0, 0], "not an integer", id="bool"),
        pytest.param([1, 0], "shape", id="wrong-length"),
    ],
)
def test_inplane_basis_from_csl_rejects_invalid_plane_covector(
    plane_covector,
    match,
):
    with pytest.raises(CrystallographyValueError, match=match):
        inplane_basis_from_csl(_sigma5_001_csl_hnf(), plane_covector)


def test_inplane_basis_from_csl_linearly_dependent_basis_raises():
    csl_basis = np.array(
        [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
        ],
        dtype=object,
    )

    with pytest.raises(CrystallographyValueError, match="linearly dependent"):
        inplane_basis_from_csl(csl_basis, (0, 0, 1))


# ---------------------------------------------------------------------------
# inplane_area_index
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("matrix", "match"),
    [
        pytest.param(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            "zero boundary plane",
            id="zero-boundary-plane",
        ),
        pytest.param(
            [[0, 0, 1], [1, 0, 0], [0, 1, 1]],
            "not in the boundary plane",
            id="out-of-plane-row",
        ),
        pytest.param(
            [[0, 0, 1], [1.5, 0, 0], [0, 1, 0]],
            "integer-valued",
            id="non-integer-entry",
        ),
        pytest.param(
            [[0, 0, 1], [1, 0, 0], [2, 0, 0]],
            "area index is zero",
            id="parallel-inplane-rows",
        ),
        pytest.param(
            [[0, 0, 1], [1, 0, 0]],
            "shape",
            id="wrong-shape",
        ),
    ],
)
def test_inplane_area_index_rejects_invalid_orientation_matrix(matrix, match):
    P = np.array(matrix, dtype=float)

    with pytest.raises(CrystallographyValueError, match=match):
        inplane_area_index(P)


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        pytest.param(
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            1,
            id="primitive-plane-normal",
        ),
        pytest.param(
            [[0, 0, 2], [1, 0, 0], [0, 1, 0]],
            1,
            id="nonprimitive-plane-normal",
        ),
        pytest.param(
            [[0, 0, 1], [1, 2, 0], [-2, 1, 0]],
            5,
            id="sigma5-positive-orientation",
        ),
        pytest.param(
            [[0, 0, 1], [-2, 1, 0], [1, 2, 0]],
            5,
            id="sigma5-reversed-orientation",
        ),
    ],
)
def test_inplane_area_index_returns_positive_area_index(matrix, expected):
    P = np.array(matrix, dtype=float)

    assert inplane_area_index(P) == expected


# ---------------------------------------------------------------------------
# rotation_preserves_plane
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plane",
    [
        pytest.param([0, 0, 1], id="primitive-plane"),
        pytest.param([0, 0, 2], id="nonprimitive-plane"),
    ],
)
def test_rotation_preserves_plane_preserved_returns_true(plane):
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)

    assert rotation_preserves_plane(rot, plane) is True


def test_rotation_preserves_plane_not_preserved_returns_false():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    # [001] rotation does not preserve [1, 0, 0]
    assert rotation_preserves_plane(rot, [1, 0, 0]) is False


def test_rotation_preserves_plane_antiparallel_false_by_default():
    # A 180 deg rotation about [0, 1, 0] maps [1, 0, 0] to [-1, 0, 0].
    rot = quaternion_to_scaled_rotation((0, 0, 1, 0))
    assert rotation_preserves_plane(rot, [1, 0, 0], allow_antiparallel=False) is False


def test_rotation_preserves_plane_antiparallel_true_accepts_opposite():
    rot = quaternion_to_scaled_rotation((0, 0, 1, 0))
    assert rotation_preserves_plane(rot, [1, 0, 0], allow_antiparallel=True) is True


def test_rotation_preserves_plane_invalid_plane_raises():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)

    with pytest.raises(CrystallographyValueError, match="zero vector"):
        rotation_preserves_plane(rot, [0, 0, 0])
