# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
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
    CrystallographyNotImplementedError,
    CrystallographyValueError,
    ScaledRotation,
)

# Sigma5 [001] 53.13 deg -- quat [2, 0, 0, 1], denominator=5
# M = [[3, -4, 0], [4, 3, 0], [0, 0, 5]]
SIGMA5_QUAT = (2, 0, 0, 1)


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture
def sigma5_rotation():
    return quaternion_to_scaled_rotation(SIGMA5_QUAT)


# --------------------------------------------------------------------------------------
# assert_scaled_rotation
# --------------------------------------------------------------------------------------


def test_assert_scaled_rotation_valid_passes_silently(sigma5_rotation):
    assert assert_scaled_rotation(sigma5_rotation) is None


def test_assert_scaled_rotation_non_orthogonal_raises():
    M = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=object)
    rot = ScaledRotation(denominator=1, matrix=M, source="matrix")

    with pytest.raises(CrystallographyValueError, match="orthogonal"):
        assert_scaled_rotation(rot)


def test_assert_scaled_rotation_improper_rotation_raises():
    M = np.diag([-1, 1, 1]).astype(object)
    rot = ScaledRotation(denominator=1, matrix=M, source="matrix")

    with pytest.raises(CrystallographyValueError, match="determinant -1"):
        assert_scaled_rotation(rot)


# --------------------------------------------------------------------------------------
# validate_scaled_rotation_matrix
# --------------------------------------------------------------------------------------


def test_validate_scaled_rotation_matrix_valid_returns_scaled_rotation(sigma5_rotation):
    result = validate_scaled_rotation_matrix(
        sigma5_rotation.matrix,
        denominator=sigma5_rotation.denominator,
        source="matrix",
    )

    assert isinstance(result, ScaledRotation)
    assert result.denominator == sigma5_rotation.denominator
    assert result.source == "matrix"
    np.testing.assert_array_equal(result.matrix, sigma5_rotation.matrix)


def test_validate_scaled_rotation_matrix_wrong_shape_raises():
    M = np.array([[1, 0], [0, 1]], dtype=object)

    with pytest.raises(CrystallographyValueError, match="input_matrix must have shape"):
        validate_scaled_rotation_matrix(M)


def test_validate_scaled_rotation_matrix_non_integer_raises():
    M = np.array([[1.5, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

    with pytest.raises(CrystallographyValueError, match="exactly integer-valued"):
        validate_scaled_rotation_matrix(M)


def test_validate_scaled_rotation_matrix_non_orthogonal_raises():
    M = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=object)

    with pytest.raises(
        CrystallographyValueError,
        match="positive scalar multiple of the identity",
    ):
        validate_scaled_rotation_matrix(M)


def test_validate_scaled_rotation_matrix_determinant_mismatch_raises(sigma5_rotation):
    M_bad = sigma5_rotation.matrix.copy()
    M_bad[0] = -M_bad[0]

    with pytest.raises(CrystallographyValueError, match=r"det\(int_matrix\)"):
        validate_scaled_rotation_matrix(
            M_bad,
            denominator=sigma5_rotation.denominator,
        )


def test_validate_scaled_rotation_matrix_supplied_denominator_mismatch_raises(
    sigma5_rotation,
):
    with pytest.raises(CrystallographyValueError, match="does not match"):
        validate_scaled_rotation_matrix(
            sigma5_rotation.matrix,
            denominator=3,
            source="matrix",
        )


@pytest.mark.parametrize(
    "denominator",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(1.5, id="float"),
        pytest.param(True, id="bool"),
    ],
)
def test_validate_scaled_rotation_matrix_rejects_invalid_denominator(
    sigma5_rotation,
    denominator,
):
    with pytest.raises(CrystallographyValueError, match="positive integer"):
        validate_scaled_rotation_matrix(
            sigma5_rotation.matrix,
            denominator=denominator,
        )


def test_validate_scaled_rotation_matrix_reduces_common_factor_when_requested(
    sigma5_rotation,
):
    M_scaled = np.array(sigma5_rotation.matrix * 2, dtype=object)

    result = validate_scaled_rotation_matrix(
        M_scaled,
        denominator=10,
        source="matrix",
        reduce_common_factor=True,
    )

    assert result.denominator == sigma5_rotation.denominator
    assert result.source == "matrix"
    np.testing.assert_array_equal(result.matrix, sigma5_rotation.matrix)


def test_validate_scaled_rotation_matrix_reduce_common_factor_indivisible_denominator_raises(
    sigma5_rotation,
):
    M_scaled = np.array(sigma5_rotation.matrix * 2, dtype=object)

    with pytest.raises(CrystallographyValueError, match="Common matrix factor"):
        validate_scaled_rotation_matrix(
            M_scaled,
            denominator=7,
            source="matrix",
            reduce_common_factor=True,
        )


def test_validate_scaled_rotation_matrix_lattice_metric_raises_not_implemented(
    sigma5_rotation,
):
    with pytest.raises(CrystallographyNotImplementedError, match="non-cubic"):
        validate_scaled_rotation_matrix(
            sigma5_rotation.matrix,
            lattice_metric=np.eye(3),
        )


@pytest.mark.parametrize("source", ["matrix", "five_dof", "quaternion"])
def test_validate_scaled_rotation_matrix_source_stored_correctly(
    sigma5_rotation,
    source,
):
    result = validate_scaled_rotation_matrix(
        sigma5_rotation.matrix,
        denominator=sigma5_rotation.denominator,
        source=source,
    )

    assert result.source == source


# --------------------------------------------------------------------------------------
# transpose_rotation_convention
# --------------------------------------------------------------------------------------


def test_transpose_rotation_convention_transposes_matrix_and_preserves_metadata(
    sigma5_rotation,
):
    transposed = transpose_rotation_convention(sigma5_rotation)

    np.testing.assert_array_equal(transposed.matrix, sigma5_rotation.matrix.T)
    assert transposed.denominator == sigma5_rotation.denominator
    assert transposed.source == sigma5_rotation.source
    assert_scaled_rotation(transposed)


# --------------------------------------------------------------------------------------
# scaled_row_image
# --------------------------------------------------------------------------------------


def test_scaled_row_image_divisible_returns_integer_image(sigma5_rotation):
    row = np.array([0, 0, 1], dtype=object)

    result = scaled_row_image(row, sigma5_rotation)

    np.testing.assert_array_equal(result, row)


def test_scaled_row_image_require_divisible_raises_when_not_divisible(sigma5_rotation):
    row = np.array([1, 0, 0], dtype=object)

    with pytest.raises(CrystallographyDivisibilityError, match="not integer-valued"):
        scaled_row_image(row, sigma5_rotation, allow_inexact=False)


def test_scaled_row_image_allow_inexact_returns_gcd_reduced_direction(sigma5_rotation):
    row = np.array([1, 0, 0], dtype=object)

    result = scaled_row_image(row, sigma5_rotation, allow_inexact=True)

    np.testing.assert_array_equal(result, np.array([3, -4, 0], dtype=object))
    assert result.dtype == object


def test_scaled_row_image_validate_rotation_false_skips_rotation_validation():
    invalid_rot = ScaledRotation(
        denominator=1,
        matrix=np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=object),
        source="matrix",
    )

    result = scaled_row_image(
        np.array([1, 0, 0], dtype=object),
        invalid_rot,
        validate_rotation=False,
    )

    np.testing.assert_array_equal(result, np.array([1, 1, 0], dtype=object))


def test_scaled_row_image_invalid_row_raises(sigma5_rotation):
    with pytest.raises(CrystallographyValueError, match="row must have shape"):
        scaled_row_image(np.array([[1, 0, 0]], dtype=object), sigma5_rotation)
