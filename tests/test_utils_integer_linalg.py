# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for GBOpt.Utils.integer_linalg public exact integer helpers."""

import math

import numpy as np
import pytest

from GBOpt.Utils.integer_linalg import (
    ExactIntegerError,
    ExactIntegerShapeError,
    ExactIntegerTypeError,
    ExactIntegerValueError,
    adjugate3_int,
    as_int_array,
    as_int_vector,
    cross_int3,
    det3_int,
    det3_int_checked,
    dot_int,
    extended_gcd,
    identity_int,
    row_gcd_reduce,
)


# ---------------------------------------------------------------------------
# as_int_array / as_int_vector
# ---------------------------------------------------------------------------
def test_as_int_array_valid_input_returns_object_dtype_python_ints():
    result = as_int_array([[1, np.int64(2)], [3.0, 4]], (2, 2), "A")

    assert result.shape == (2, 2)
    assert result.dtype == object
    assert all(isinstance(value, int) for value in result.flat)
    np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]], dtype=object))


def test_as_int_array_wrong_shape_raises_shape_error():
    with pytest.raises(ExactIntegerShapeError, match="shape"):
        as_int_array([[1, 2], [3, 4]], (3, 3), "A")


@pytest.mark.parametrize(
    "bad_value,error_type",
    [
        (True, ExactIntegerTypeError),
        (np.bool_(False), ExactIntegerTypeError),
        ("not-an-int", ExactIntegerTypeError),
        (1.5, ExactIntegerValueError),
        (np.nan, ExactIntegerValueError),
    ],
)
def test_as_int_array_rejects_non_exact_integer_entries(bad_value, error_type):
    with pytest.raises(error_type):
        as_int_array([bad_value], (1,), "A")


def test_as_int_vector_valid_input_returns_tuple_of_ints():
    result = as_int_vector(np.array([1, 2, 3], dtype=np.int64), 3, "v")

    assert result == (1, 2, 3)
    assert all(isinstance(value, int) for value in result)


def test_as_int_vector_wrong_length_raises_shape_error():
    with pytest.raises(ExactIntegerShapeError, match="shape"):
        as_int_vector([1, 2], 3, "v")


# ---------------------------------------------------------------------------
# identity_int
# ---------------------------------------------------------------------------
def test_identity_int_returns_object_identity():
    result = identity_int(3)

    assert result.dtype == object
    np.testing.assert_array_equal(result, np.eye(3, dtype=object))


@pytest.mark.parametrize("bad_n", [True, 1.5, -1])
def test_identity_int_rejects_invalid_dimension(bad_n):
    with pytest.raises(ExactIntegerError):
        identity_int(bad_n)


# ---------------------------------------------------------------------------
# row_gcd_reduce
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "row,expected",
    [
        ([6, -9, 0], [2, -3, 0]),
        ([-4, -8, 0], [-1, -2, 0]),
        ([0, 0, 0], [0, 0, 0]),
        ([5, 0, 0], [1, 0, 0]),
        ([1, 2, 3], [1, 2, 3]),
    ],
)
def test_row_gcd_reduce_reduces_by_common_component_gcd(row, expected):
    np.testing.assert_array_equal(row_gcd_reduce(row), np.array(expected, dtype=object))


def test_row_gcd_reduce_rejects_non_1d_input():
    with pytest.raises(ExactIntegerShapeError, match="1D"):
        row_gcd_reduce([[1, 2, 3]])


def test_row_gcd_reduce_rejects_non_integer_entry():
    with pytest.raises(ExactIntegerValueError, match="integer-valued"):
        row_gcd_reduce([2.5, 0, 0])


# ---------------------------------------------------------------------------
# dot_int
# ---------------------------------------------------------------------------
def test_dot_int_positive_negative_and_large_entries():
    assert dot_int([1, 2, 3], [4, 5, 6]) == 32
    assert dot_int([-1, 2, -3], [4, -5, 6]) == -32

    big = 10**20
    assert dot_int([big, big], [big, big]) == 2 * big**2


def test_dot_int_flattens_inputs():
    assert dot_int([[1, 2, 3]], np.array([[4], [5], [6]])) == 32


def test_dot_int_shape_mismatch_raises():
    with pytest.raises(ExactIntegerShapeError, match="equal lengths"):
        dot_int([1, 2], [1, 2, 3])


@pytest.mark.parametrize("bad_value", [True, np.bool_(True), "1", 1.5])
def test_dot_int_rejects_non_exact_integer_without_truncation(bad_value):
    with pytest.raises(ExactIntegerError):
        dot_int([bad_value], [2])


# ---------------------------------------------------------------------------
# cross_int3
# ---------------------------------------------------------------------------
def test_cross_int3_standard_basis():
    result = cross_int3([1, 0, 0], [0, 1, 0])

    assert result.dtype == object
    np.testing.assert_array_equal(result, np.array([0, 0, 1], dtype=object))


def test_cross_int3_large_entries_uses_python_ints():
    big = 10**20
    result = cross_int3([big, 0, 0], [0, big, 0])

    np.testing.assert_array_equal(result, np.array([0, 0, big**2], dtype=object))
    assert isinstance(result[2], int)


@pytest.mark.parametrize(
    "x,y",
    [
        ([1, 0], [0, 1, 0]),
        ([1, 0, 0], [0, 1]),
    ],
)
def test_cross_int3_wrong_length_raises(x, y):
    with pytest.raises(ExactIntegerShapeError, match="length-3"):
        cross_int3(x, y)


@pytest.mark.parametrize("bad_value", [True, np.bool_(True), "1", 1.5])
def test_cross_int3_rejects_non_exact_integer_without_truncation(bad_value):
    with pytest.raises(ExactIntegerError):
        cross_int3([bad_value, 0, 0], [0, 1, 0])


# ---------------------------------------------------------------------------
# det3_int / det3_int_checked / adjugate3_int
# ---------------------------------------------------------------------------
def test_det3_int_identity_and_known_values():
    assert det3_int(np.eye(3, dtype=object)) == 1
    assert det3_int([[1, 0, 0], [0, 2, 0], [0, 0, 3]]) == 6
    assert det3_int([[0, 1, 0], [1, 0, 0], [0, 0, 1]]) == -1


def test_det3_int_checked_uses_prevalidated_matrix():
    matrix = as_int_array([[4, -3, 0], [3, 4, 0], [0, 0, 1]], (3, 3), "matrix")

    assert det3_int_checked(matrix) == 25


@pytest.mark.parametrize(
    "bad_matrix",
    [
        [[1, 0], [0, 1]],
        [[1.5, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[True, 0, 0], [0, 1, 0], [0, 0, 1]],
    ],
)
def test_det3_int_rejects_invalid_input(bad_matrix):
    with pytest.raises(ExactIntegerError):
        det3_int(bad_matrix)


def test_adjugate3_int_identity_is_identity():
    assert adjugate3_int(np.eye(3, dtype=object)) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_adjugate3_int_satisfies_matrix_adjugate_identity():
    matrix = np.array([[4, -3, 0], [3, 4, 0], [0, 0, 1]], dtype=object)
    adj = np.array(adjugate3_int(matrix), dtype=object)
    det = det3_int(matrix)

    np.testing.assert_array_equal(matrix @ adj, det * np.eye(3, dtype=object))


@pytest.mark.parametrize(
    "bad_matrix",
    [
        [[1, 0], [0, 1]],
        [[1.5, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[False, 0, 0], [0, 1, 0], [0, 0, 1]],
    ],
)
def test_adjugate3_int_rejects_invalid_input(bad_matrix):
    with pytest.raises(ExactIntegerError):
        adjugate3_int(bad_matrix)


# ---------------------------------------------------------------------------
# extended_gcd
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "a,b",
    [
        (30, 12),
        (-30, 12),
        (30, -12),
        (-30, -12),
        (0, 5),
        (5, 0),
        (0, 0),
    ],
)
def test_extended_gcd_bezout_identity(a, b):
    gcd_value, x, y = extended_gcd(a, b)

    assert gcd_value == math.gcd(abs(a), abs(b))
    assert x * a + y * b == gcd_value
    assert gcd_value >= 0


@pytest.mark.parametrize("bad_value", [True, np.bool_(True), 1.5, "a"])
def test_extended_gcd_rejects_non_exact_integer_inputs(bad_value):
    with pytest.raises(ExactIntegerError):
        extended_gcd(bad_value, 1)
