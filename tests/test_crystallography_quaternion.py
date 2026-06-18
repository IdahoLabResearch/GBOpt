# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import math

import numpy as np
import pytest

from GBOpt.crystallography.quaternion import (
    integer_quaternion_from_unit,
    normalize_integer_quaternion,
    quaternion_to_rotation_matrix,
    quaternion_to_scaled_rotation,
)
from GBOpt.crystallography.rotation import assert_scaled_rotation
from GBOpt.crystallography.types import (
    CrystallographyNotImplementedError,
    CrystallographyValueError,
)

# Sigma5 [001] 53.13 deg -- quat [2, 0, 0, 1], N=5
# M = [[3, -4, 0], [4, 3, 0], [0, 0, 5]]
SIGMA5_QUAT = (2, 0, 0, 1)
SIGMA5_N = 5
SIGMA5_M = np.array([[3, -4, 0], [4, 3, 0], [0, 0, 5]], dtype=object)

# Sigma5 [001] 36.87 deg -- quat [3, 0, 0, 1], N=10
SIGMA5_36_R = np.array([
    [4/5, -3/5, 0],
    [3/5,  4/5, 0],
    [0,    0,   1],
])

# Sigma5 [001] 53.13 deg unit quaternion
SIGMA5_QUAT_NORM = np.array([2, 0, 0, 1], dtype=float) / np.sqrt(5)
SIGMA5_R_EXPECTED = np.array([
    [3/5, -4/5, 0],
    [4/5,  3/5, 0],
    [0,    0,   1],
])


# ---------------------------------------------------------------------------
# normalize_integer_quaternion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q_in,q_expected", [
    ((4, 0, 0, 2), (2, 0, 0, 1)),
    ((0, 0, 0, -3), (0, 0, 0, 1)),
    ((-2, 0, 0, -1), (2, 0, 0, 1)),
])
def test_normalize_integer_quaternion(q_in, q_expected):
    assert normalize_integer_quaternion(q_in) == q_expected


def test_normalize_integer_quaternion_zero_raises():
    with pytest.raises(CrystallographyValueError):
        normalize_integer_quaternion((0, 0, 0, 0))


# ---------------------------------------------------------------------------
# quaternion_to_scaled_rotation
# ---------------------------------------------------------------------------

def test_quaternion_to_scaled_rotation_correct_N():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    assert rot.N == SIGMA5_N


def test_quaternion_to_scaled_rotation_correct_M():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    np.testing.assert_array_equal(rot.M, SIGMA5_M)


def test_quaternion_to_scaled_rotation_source_is_quaternion():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    assert rot.source == "quaternion"


def test_quaternion_to_scaled_rotation_quaternion_field_set():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    assert rot.quaternion == SIGMA5_QUAT


def test_quaternion_to_scaled_rotation_passes_validation():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    assert_scaled_rotation(rot)  # should not raise


def test_quaternion_to_scaled_rotation_canonicalize_reduces_common_factors():
    # (4, 0, 0, 2) should canonicalize to (2, 0, 0, 1) before constructing M
    rot_raw = quaternion_to_scaled_rotation((4, 0, 0, 2), canonicalize=True)
    rot_canonical = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    assert rot_raw.N == rot_canonical.N
    np.testing.assert_array_equal(rot_raw.M, rot_canonical.M)


def test_quaternion_to_scaled_rotation_no_canonicalize_uses_raw():
    # Without canonicalization (4,0,0,2) gives N=20, not N=5
    rot = quaternion_to_scaled_rotation((4, 0, 0, 2), canonicalize=False)
    assert rot.N == 20


def test_quaternion_to_scaled_rotation_zero_quaternion_raises():
    with pytest.raises(CrystallographyValueError):
        quaternion_to_scaled_rotation((0, 0, 0, 0))


def test_quaternion_to_scaled_rotation_lattice_metric_raises():
    with pytest.raises(CrystallographyNotImplementedError):
        quaternion_to_scaled_rotation(
            SIGMA5_QUAT, lattice_metric=np.eye(3)
        )


# ---------------------------------------------------------------------------
# integer_quaternion_from_unit
# ---------------------------------------------------------------------------

def test_integer_quaternion_from_unit_valid_returns_correct_quaternion():
    result = integer_quaternion_from_unit(SIGMA5_QUAT_NORM)
    assert result == SIGMA5_QUAT


def test_integer_quaternion_from_unit_wrong_shape_raises():
    with pytest.raises(CrystallographyValueError):
        integer_quaternion_from_unit(np.array([1.0, 0.0, 0.0]))


def test_integer_quaternion_from_unit_zero_quaternion_raises():
    with pytest.raises(CrystallographyValueError):
        integer_quaternion_from_unit(np.array([0.0, 0.0, 0.0, 0.0]))


def test_integer_quaternion_from_unit_mismatch_raises():
    # A unit quaternion that cannot be recovered as an integer quaternion
    # within the default max_denominator should raise
    irrational = np.array([np.sqrt(2), 1.0, 1.0, 1.0], dtype=float)
    irrational /= np.linalg.norm(irrational)
    with pytest.raises(CrystallographyValueError):
        integer_quaternion_from_unit(irrational, max_denominator=2)


def test_integer_quaternion_from_unit_result_is_primitive():
    result = integer_quaternion_from_unit(SIGMA5_QUAT_NORM)
    gcd = math.gcd(*[abs(v) for v in result])
    assert gcd == 1


# ---------------------------------------------------------------------------
# quaternion_to_rotation_matrix
# ---------------------------------------------------------------------------

def test_quaternion_to_rotation_matrix_sigma5_matches_expected():
    R = quaternion_to_rotation_matrix(SIGMA5_QUAT_NORM)
    np.testing.assert_allclose(R, SIGMA5_R_EXPECTED, atol=1e-12)


def test_quaternion_to_rotation_matrix_integer_quaternion_normalized_internally():
    R = quaternion_to_rotation_matrix(np.array([2, 0, 0, 1]))
    np.testing.assert_allclose(R, SIGMA5_R_EXPECTED, atol=1e-12)


def test_quaternion_to_rotation_matrix_integer_valued_float_normalized_internally():
    R = quaternion_to_rotation_matrix(np.array([2.0, 0.0, 0.0, 1.0]))
    np.testing.assert_allclose(R, SIGMA5_R_EXPECTED, atol=1e-12)


def test_quaternion_to_rotation_matrix_output_is_proper_rotation():
    R = quaternion_to_rotation_matrix(SIGMA5_QUAT_NORM)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12


def test_quaternion_to_rotation_matrix_identity_quaternion_gives_identity():
    R = quaternion_to_rotation_matrix(np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


def test_quaternion_to_rotation_matrix_sigma5_36deg():
    q = np.array([3, 0, 0, 1], dtype=float) / np.sqrt(10)
    R = quaternion_to_rotation_matrix(q)
    np.testing.assert_allclose(R, SIGMA5_36_R, atol=1e-12)


@pytest.mark.parametrize("quat", [
    [1, 0, 0],
    [[1, 0, 0, 0]],
    np.ones((2, 2)),
])
def test_quaternion_to_rotation_matrix_invalid_shape_raises(quat):
    with pytest.raises(CrystallographyValueError):
        quaternion_to_rotation_matrix(quat)


def test_quaternion_to_rotation_matrix_non_finite_raises():
    with pytest.raises(CrystallographyValueError, match="finite"):
        quaternion_to_rotation_matrix(np.array([np.nan, 0, 0, 1]))


def test_quaternion_to_rotation_matrix_non_integer_non_unit_raises():
    with pytest.raises(CrystallographyValueError, match="integer-valued"):
        quaternion_to_rotation_matrix(np.array([1.5, 0, 0, 1]))
