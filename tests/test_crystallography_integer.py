# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import numpy as np
import pytest

from GBOpt.crystallography.integer import (
    as_int_array,
    as_int_vector,
    assert_integer_rows,
    integer_adj3,
    integer_det3,
    row_gcd_reduce_float,
    row_gcd_reduce_int,
)
from GBOpt.crystallography.types import CrystallographyValueError

INVALID_INTEGER_MATRICES = [
    [[1.1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [["not-an-int", 0, 0], [0, 1, 0], [0, 0, 1]],
    [[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[1, 0], [0, 1]],
]

# TODO: Move to test_utils_integer_normal_forms.py
# def test_integer_normal_form_helper_rejects_non_integer_entries(self):
#     with pytest.raises(ExactNormalFormError, match="integer-valued"):
#         _inf_row_gcd_reduce(np.array([2.5, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# integer_det3
# ---------------------------------------------------------------------------

def test_determinant_identity():
    assert integer_det3([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == 1


def test_determinant_known_positive():
    assert integer_det3([[1, 0, 0], [0, 2, 0], [0, 0, 3]]) == 6


def test_determinant_known_negative():
    assert integer_det3([[0, 1, 0], [1, 0, 0], [0, 0, 1]]) == -1


def test_determinant_sigma5_right_S():
    # Sigma5 36.87 deg right grain: Q = [[4,-3,0],[3,4,0],[0,0,1]], det = 25
    assert integer_det3([[4, -3, 0], [3, 4, 0], [0, 0, 1]]) == 25


@pytest.mark.parametrize("matrix", INVALID_INTEGER_MATRICES)
def test_determinant_invalid_input_raises(matrix):
    with pytest.raises(CrystallographyValueError):
        integer_det3(matrix)

# ---------------------------------------------------------------------------
# integer_adj3
# ---------------------------------------------------------------------------


def test_adjoint_times_M_equals_det_times_I():
    M = [[4, -3, 0], [3, 4, 0], [0, 0, 1]]
    det = integer_det3(M)
    adj = integer_adj3(M)
    product = np.array(M) @ np.array(adj)
    np.testing.assert_array_equal(product, det * np.eye(3, dtype=int))


def test_identity_adjoint_is_identity():
    adj = integer_adj3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert adj == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


@pytest.mark.parametrize("matrix", INVALID_INTEGER_MATRICES)
def test_adjoint_invalid_input_raises(matrix):
    with pytest.raises(CrystallographyValueError):
        integer_adj3(matrix)


# ---------------------------------------------------------------------------
# row_gcd_reduce_int
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("row,expected", [
    ([6, -9, 0], [2, -3, 0]),
    ([-4, -8, 0], [-1, -2, 0]),
    ([0, 0, 0], [0, 0, 0]),
    ([5, 0, 0], [1, 0, 0]),
])
def test_row_gcd_reduce_reduces_by_common_component_gcd(row, expected):
    np.testing.assert_array_equal(row_gcd_reduce_int(np.array(row)), expected)


def test_row_gcd_reduce_rejects_non_integer_entries():
    with pytest.raises(CrystallographyValueError, match="integer-valued"):
        row_gcd_reduce_int(np.array([2.5, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# as_int_array
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("A,shape", [
    ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], (3, 3)),
    ([1, 2, 3], (3,)),
])
def test_as_int_array_valid_input_returns_object_dtype(A, shape):
    result = as_int_array(A, shape, "A")
    assert result.shape == shape
    assert result.dtype == object
    assert all(isinstance(v, int) for v in result.flat)


@pytest.mark.parametrize("A,shape", [
    ([[1, 0], [0, 1]], (3, 3)),
    ([1, 2], (3,)),
])
def test_as_int_array_wrong_shape_raises(A, shape):
    with pytest.raises(CrystallographyValueError):
        as_int_array(A, shape, "A")


@pytest.mark.parametrize("A,shape", [
    ([[1.5, 0, 0], [0, 1, 0], [0, 0, 1]], (3, 3)),
    (["not-an-int", 0, 0], (3,)),
    ([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]], (3, 3)),
])
def test_as_int_array_non_integer_input_raises(A, shape):
    with pytest.raises(CrystallographyValueError):
        as_int_array(A, shape, "A")

# ---------------------------------------------------------------------------
# as_int_vector
# ---------------------------------------------------------------------------


def test_as_int_vector_valid_input_returns_tuple_of_ints():
    result = as_int_vector([1, 2, 3], 3, "v")
    assert isinstance(result, tuple)
    assert all(isinstance(v, int) for v in result)
    assert result == (1, 2, 3)


def test_as_int_vector_wrong_length_raises():
    with pytest.raises(CrystallographyValueError):
        as_int_vector([1, 2], 3, "v")


def test_as_int_vector_non_integer_raises():
    with pytest.raises(CrystallographyValueError):
        as_int_vector([1.5, 2, 3], 3, "v")


# ---------------------------------------------------------------------------
# row_gcd_reduce_float
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("row", [
    [6, -9, 0],
    [-4, -8, 0],
    [0, 0, 0],
    [5, 0, 0],
])
def test_row_gcd_reduce_float_returns_float_dtype(row):
    result = row_gcd_reduce_float(np.array(row))
    assert result.dtype == float


@pytest.mark.parametrize("row", [
    [6, -9, 0],
    [-4, -8, 0],
    [0, 0, 0],
    [5, 0, 0],
])
def test_row_gcd_reduce_float_matches_int_values(row):
    int_result = row_gcd_reduce_int(np.array(row))
    float_result = row_gcd_reduce_float(np.array(row))
    np.testing.assert_array_equal(float_result, int_result.astype(float))


# ---------------------------------------------------------------------------
# assert_integer_rows
# ---------------------------------------------------------------------------

def test_assert_integer_rows_passes_silently_on_integer_input():
    M = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    assert_integer_rows(M, "M")  # should not raise


@pytest.mark.parametrize("row_idx,bad_value", [
    (0, 1.5),
    (1, 0.1),
    (2, -2.7),
])
def test_assert_integer_rows_raises_on_non_integer_row(row_idx, bad_value):
    M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    M[row_idx, 0] = bad_value
    with pytest.raises(CrystallographyValueError):
        assert_integer_rows(M, "M")
