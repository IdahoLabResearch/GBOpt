# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import math

import numpy as np
import pytest

from GBOpt.crystallography.plane import (
    enumerate_inplane_hnf_supercells,
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
from GBOpt.Utils.integer_normal_forms import (
    _cross_int3,
    _dot_int,
    primitive_integer_null_basis_3d,
)

# Sigma5 [001] 53.13 deg -- quat [2, 0, 0, 1], N=5
# Rotation preserves [0, 0, 1] plane but not [1, 0, 0]
SIGMA5_QUAT = (2, 0, 0, 1)


# ---------------------------------------------------------------------------
# primitive_plane
# ---------------------------------------------------------------------------

def test_primitive_plane_already_primitive():
    assert primitive_plane([1, 2, 3]) == (1, 2, 3)


def test_primitive_plane_reduces_nonprimitive():
    assert primitive_plane([2, 4, 6]) == (1, 2, 3)


def test_primitive_plane_preserves_sign_convention():
    assert primitive_plane([-1, 2, -3]) == (-1, 2, -3)


def test_primitive_plane_returns_tuple():
    result = primitive_plane([1, 0, 0])
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_primitive_plane_zero_vector_raises():
    with pytest.raises(CrystallographyValueError):
        primitive_plane([0, 0, 0])


def test_primitive_plane_non_integer_raises():
    with pytest.raises(CrystallographyValueError):
        primitive_plane([1.5, 0, 0])


def test_primitive_plane_wrong_length_raises():
    with pytest.raises(CrystallographyValueError):
        primitive_plane([1, 0])


# ---------------------------------------------------------------------------
# plane_null_basis
# ---------------------------------------------------------------------------

def _check_null_basis(plane: list) -> None:
    """Assert e1, e2 are in-plane and cross(e1, e2) == +plane."""
    p = np.array(plane, dtype=int)
    e1, e2 = plane_null_basis(p)
    e1i = np.round(e1).astype(int)
    e2i = np.round(e2).astype(int)
    assert np.dot(p, e1i) == 0, f"e1 not in plane {plane}: dot={np.dot(p, e1i)}"
    assert np.dot(p, e2i) == 0, f"e2 not in plane {plane}: dot={np.dot(p, e2i)}"
    cross = np.cross(e1i, e2i)
    assert np.array_equal(cross, p), (
        f"cross(e1, e2)={cross.tolist()} != plane={plane}"
    )


def test_plane_null_basis_100():
    _check_null_basis([1, 0, 0])


def test_plane_null_basis_111():
    _check_null_basis([1, 1, 1])


def test_plane_null_basis_523():
    _check_null_basis([5, 2, 3])


def test_plane_null_basis_negative_components():
    _check_null_basis([-1, 2, -1])


def test_plane_null_basis_zero_plane_raises():
    with pytest.raises(CrystallographyValueError, match="zero vector"):
        plane_null_basis(np.array([0, 0, 0]))


def test_plane_null_basis_non_primitive_raises():
    with pytest.raises(CrystallographyValueError, match="not primitive"):
        plane_null_basis(np.array([2, 0, 0]))


# ---------------------------------------------------------------------------
# Tests that verify that `primitive_integer_null_basis_3d` returns a saturated
# primitive basis, which is what inplane_basis_from_csl relies on directly
# ---------------------------------------------------------------------------

SATURATION_COVECTORS = [
    (1, 2, 3),                      # primitive, all nonzero
    (5, 2, 3),                      # physical regression case
    (2, 4, 6),                      # nonprimitive, all nonzero
    (0, 4, 6),                      # one zero component
    (4, 0, 6),                      # one zero component
    (4, 6, 0),                      # one zero component
    (10**20, 10**20 + 1, 1),        # Python-int overflow regression
]


@pytest.mark.parametrize("covector", SATURATION_COVECTORS)
def test_primitive_integer_null_basis_null_vectors(covector):
    cov = np.array(covector, dtype=object)
    basis = primitive_integer_null_basis_3d(cov)
    assert _dot_int(cov, basis[:, 0]) == 0
    assert _dot_int(cov, basis[:, 1]) == 0


@pytest.mark.parametrize("covector", SATURATION_COVECTORS)
def test_primitive_integer_null_basis_is_saturated(covector):

    cov = np.array(covector, dtype=object)
    g = math.gcd(*[abs(int(v)) for v in cov])
    primitive_cov = np.array([int(v) // g for v in cov], dtype=object)
    basis = primitive_integer_null_basis_3d(cov)
    cross = _cross_int3(basis[:, 0], basis[:, 1])
    np.testing.assert_array_equal(cross, primitive_cov)


# ---------------------------------------------------------------------------
# inplane_basis_from_csl
#
# NOTE: Full inplane_basis_from_csl tests that depend on csl_from_scaled_rotation
# are deferred to test_crystallography_csl.py, since csl.py has not been
# refactored yet. The INPLANE_EXACT_CSL_SCENARIOS and test_exact_csl_inplane_basis
# tests from test_exact.py will be migrated there.
# ---------------------------------------------------------------------------

def test_inplane_basis_from_csl_vectors_are_in_plane():
    # Use the Sigma5 [001] HNF basis directly
    csl_basis = np.array([[1, 0, 0], [2, 5, 0], [0, 0, 1]], dtype=object)
    result = inplane_basis_from_csl(csl_basis, (0, 0, 1))
    h = np.array([0, 0, 1], dtype=object)
    assert int(h @ result.basis[:, 0]) == 0
    assert int(h @ result.basis[:, 1]) == 0


def test_inplane_basis_from_csl_vectors_are_linearly_independent():
    csl_basis = np.array([[1, 0, 0], [2, 5, 0], [0, 0, 1]], dtype=object)
    result = inplane_basis_from_csl(csl_basis, (0, 0, 1))
    cross = np.cross(
        result.basis[:, 0].astype(int),
        result.basis[:, 1].astype(int),
    )
    assert any(v != 0 for v in cross)


def test_inplane_basis_from_csl_non_integer_basis_raises():
    csl_basis = np.array([[1.5, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    with pytest.raises(CrystallographyValueError):
        inplane_basis_from_csl(csl_basis, (0, 0, 1))


def test_inplane_basis_from_csl_lattice_metric_raises():
    csl_basis = np.array([[1, 0, 0], [2, 5, 0], [0, 0, 1]], dtype=object)
    with pytest.raises(CrystallographyNotImplementedError):
        inplane_basis_from_csl(csl_basis, (0, 0, 1), lattice_metric=np.eye(3))


# ---------------------------------------------------------------------------
# enumerate_inplane_hnf_supercells
# ---------------------------------------------------------------------------

def test_enumerate_inplane_hnf_supercells_count():
    # index n has sigma(n) supercells (sum of divisors)
    basis = np.array([[1, 0], [0, 1], [0, 0]], dtype=object)
    assert len(enumerate_inplane_hnf_supercells(basis, 1)) == 1
    assert len(enumerate_inplane_hnf_supercells(basis, 2)) == 3
    assert len(enumerate_inplane_hnf_supercells(basis, 6)) == 12


def test_enumerate_inplane_hnf_supercells_shape():
    basis = np.array([[1, 0], [0, 1], [0, 0]], dtype=object)
    supercells = enumerate_inplane_hnf_supercells(basis, 2)
    for sc in supercells:
        assert sc.shape == (3, 2)


def test_enumerate_inplane_hnf_supercells_non_integer_raises():
    basis = np.array([[1.5, 0], [0, 1], [0, 0]], dtype=float)
    with pytest.raises(CrystallographyValueError):
        enumerate_inplane_hnf_supercells(basis, 2)


# ---------------------------------------------------------------------------
# inplane_area_index
# ---------------------------------------------------------------------------

def test_inplane_area_index_identity_plane_100():
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    assert inplane_area_index(P) == 1


def test_inplane_area_index_sigma5_csl():
    P = np.array([[0, 0, 1], [1, 2, 0], [-2, 1, 0]], dtype=float)
    assert inplane_area_index(P) == 5


def test_inplane_area_index_out_of_plane_row_raises():
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 1]], dtype=float)
    with pytest.raises(CrystallographyValueError, match="not in the boundary plane"):
        inplane_area_index(P)


def test_inplane_area_index_non_integer_raises():
    P = np.array([[0, 0, 1], [1.5, 0, 0], [0, 1, 0]], dtype=float)
    with pytest.raises(CrystallographyValueError, match="integer-valued"):
        inplane_area_index(P)


def test_inplane_area_index_zero_area_raises():
    P = np.array([[0, 0, 1], [1, 0, 0], [2, 0, 0]], dtype=float)
    with pytest.raises(CrystallographyValueError):
        inplane_area_index(P)


# ---------------------------------------------------------------------------
# rotation_preserves_plane
# ---------------------------------------------------------------------------

def test_rotation_preserves_plane_preserved_returns_true():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    # [001] rotation preserves the [001] plane
    assert rotation_preserves_plane(rot, [0, 0, 1]) is True


def test_rotation_preserves_plane_not_preserved_returns_false():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    # [001] rotation does not preserve [1, 0, 0]
    assert rotation_preserves_plane(rot, [1, 0, 0]) is False


def test_rotation_preserves_plane_antiparallel_false_by_default():
    # Identity rotation negated -- construct a rotation that maps plane to -plane
    # Use a 180 deg rotation about [0,1,0]: maps [1,0,0] -> [-1,0,0]
    rot = quaternion_to_scaled_rotation((0, 0, 1, 0))
    assert rotation_preserves_plane(rot, [1, 0, 0], allow_antiparallel=False) is False


def test_rotation_preserves_plane_antiparallel_true_accepts_opposite():
    rot = quaternion_to_scaled_rotation((0, 0, 1, 0))
    assert rotation_preserves_plane(rot, [1, 0, 0], allow_antiparallel=True) is True


def test_rotation_preserves_plane_invalid_plane_raises():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    with pytest.raises(CrystallographyValueError):
        rotation_preserves_plane(rot, [0, 0, 0])
