# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import math

import numpy as np
import pytest

from GBOpt.crystallography.quaternion import quaternion_to_scaled_rotation
from GBOpt.crystallography.rotation import (
    assert_scaled_rotation,
    scaled_row_image,
    transpose_rotation_convention,
    validate_scaled_rotation_matrix,
)
from GBOpt.crystallography.types import (
    CrystallographyDivisibilityError,
    CrystallographyValueError,
    ScaledRotation,
)

# Sigma5 [001] 53.13 deg -- quat [2, 0, 0, 1], N=5
# M = [[3, -4, 0], [4, 3, 0], [0, 0, 5]]
SIGMA5_QUAT = (2, 0, 0, 1)


# ---------------------------------------------------------------------------
# assert_scaled_rotation
# ---------------------------------------------------------------------------

def test_assert_scaled_rotation_valid_passes_silently():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    assert_scaled_rotation(rot)  # should not raise


def test_assert_scaled_rotation_non_orthogonal_raises():
    M = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=object)
    rot = ScaledRotation(N=1, M=M, source="matrix")
    with pytest.raises(CrystallographyValueError, match="orthogonal"):
        assert_scaled_rotation(rot)


def test_assert_scaled_rotation_wrong_determinant_raises():
    # M = 2*I is orthogonal scaled by 2 but det(M) = 8 != N^3 = 1
    M = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=object)
    rot = ScaledRotation(N=1, M=M, source="matrix")
    with pytest.raises(CrystallographyValueError):
        assert_scaled_rotation(rot)


# ---------------------------------------------------------------------------
# validate_scaled_rotation_matrix
# ---------------------------------------------------------------------------

def test_validate_scaled_rotation_matrix_valid_returns_scaled_rotation():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    result = validate_scaled_rotation_matrix(rot.M, N=rot.N, source="matrix")
    assert isinstance(result, ScaledRotation)
    assert result.N == rot.N
    np.testing.assert_array_equal(result.M, rot.M)


def test_validate_scaled_rotation_matrix_wrong_shape_raises():
    M = np.array([[1, 0], [0, 1]], dtype=object)
    with pytest.raises(CrystallographyValueError):
        validate_scaled_rotation_matrix(M)


def test_validate_scaled_rotation_matrix_non_integer_raises():
    M = np.array([[1.5, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    with pytest.raises(CrystallographyValueError):
        validate_scaled_rotation_matrix(M)


def test_validate_scaled_rotation_matrix_non_orthogonal_raises():
    M = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=object)
    with pytest.raises(CrystallographyValueError):
        validate_scaled_rotation_matrix(M)


def test_validate_scaled_rotation_matrix_determinant_mismatch_raises():
    # Construct M with correct gram but wrong det by negating a row
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    M_bad = rot.M.copy()
    M_bad[0] = -M_bad[0]
    with pytest.raises(CrystallographyValueError):
        validate_scaled_rotation_matrix(M_bad, N=rot.N)


def test_validate_scaled_rotation_matrix_supplied_N_matches_derived():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    result = validate_scaled_rotation_matrix(rot.M, N=5, source="matrix")
    assert result.N == 5


def test_validate_scaled_rotation_matrix_supplied_N_mismatch_raises():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    with pytest.raises(CrystallographyValueError, match="does not match"):
        validate_scaled_rotation_matrix(rot.M, N=3, source="matrix")


def test_validate_scaled_rotation_matrix_reduce_common_factor():
    # 2 * Sigma5 M with N=10 should reduce to the standard Sigma5 rotation
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    M_scaled = np.array(rot.M * 2, dtype=object)
    result = validate_scaled_rotation_matrix(
        M_scaled, N=10, source="matrix", reduce_common_factor=True
    )
    assert result.N == rot.N
    np.testing.assert_array_equal(result.M, rot.M)


def test_validate_scaled_rotation_matrix_reduce_common_factor_indivisible_N_raises():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    M_scaled = np.array(rot.M * 2, dtype=object)
    with pytest.raises(CrystallographyValueError):
        validate_scaled_rotation_matrix(
            M_scaled, N=7, source="matrix", reduce_common_factor=True
        )


@pytest.mark.parametrize("source", ["matrix", "five_dof", "quaternion"])
def test_validate_scaled_rotation_matrix_source_stored_correctly(source):
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    result = validate_scaled_rotation_matrix(rot.M, N=rot.N, source=source)
    assert result.source == source


# ---------------------------------------------------------------------------
# transpose_rotation_convention
# ---------------------------------------------------------------------------

def test_transpose_rotation_convention_returns_transposed_M():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    transposed = transpose_rotation_convention(rot)
    np.testing.assert_array_equal(transposed.M, rot.M.T)


def test_transpose_rotation_convention_same_N():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    transposed = transpose_rotation_convention(rot)
    assert transposed.N == rot.N


def test_transpose_rotation_convention_result_is_valid():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    transposed = transpose_rotation_convention(rot)
    assert_scaled_rotation(transposed)  # should not raise


def test_transpose_rotation_convention_nontrivial():
    # For a non-trivial rotation M and M.T must differ
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    transposed = transpose_rotation_convention(rot)
    assert not np.array_equal(transposed.M, rot.M)


# ---------------------------------------------------------------------------
# scaled_row_image
# ---------------------------------------------------------------------------

def test_scaled_row_image_divisible_returns_integer_image():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    # Plane [0, 0, 1] is preserved by a [001] rotation
    row = np.array([0, 0, 1], dtype=object)
    result = scaled_row_image(row, rot)
    np.testing.assert_array_equal(result, row)


def test_scaled_row_image_require_divisible_raises_when_not_divisible():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    # [1, 0, 0] is not preserved by this rotation
    row = np.array([1, 0, 0], dtype=object)
    with pytest.raises(CrystallographyDivisibilityError):
        scaled_row_image(row, rot, require_divisible=True)


def test_scaled_row_image_not_require_divisible_returns_gcd_reduced():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    row = np.array([1, 0, 0], dtype=object)
    result = scaled_row_image(row, rot, require_divisible=False)
    # Result should be integer-valued and GCD-reduced
    assert result.dtype == object

    gcd = math.gcd(*[abs(int(v)) for v in result])
    assert gcd == 1


def test_scaled_row_image_invalid_row_raises():
    rot = quaternion_to_scaled_rotation(SIGMA5_QUAT)
    with pytest.raises(CrystallographyValueError):
        scaled_row_image(np.array([[1, 0, 0]], dtype=object), rot)
