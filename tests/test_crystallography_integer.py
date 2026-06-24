# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
import numpy as np
import pytest

from GBOpt.crystallography.integer import (
    as_int_array,
    as_int_vector,
    cross_int3,
    dot_int,
    integer_adj3,
    integer_det3,
    row_gcd_reduce,
)
from GBOpt.crystallography.types import CrystallographyValueError


def test_as_int_array_valid_input_returns_object_dtype_python_ints():
    result = as_int_array([[1, np.int64(2)], [3.0, 4]], (2, 2), "A")

    assert result.shape == (2, 2)
    assert result.dtype == object
    assert all(isinstance(value, int) for value in result.flat)
    np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]], dtype=object))


def test_as_int_array_wrong_shape_raises_crystallography_value_error():
    with pytest.raises(CrystallographyValueError, match="shape"):
        as_int_array([[1, 0], [0, 1]], (3, 3), "A")


def test_as_int_array_non_integer_input_raises_crystallography_value_error():
    with pytest.raises(CrystallographyValueError, match="exactly integer-valued"):
        as_int_array([1.5, 2, 3], (3,), "A")


def test_as_int_vector_valid_input_returns_tuple_of_python_ints():
    result = as_int_vector([1, np.int64(2), 3.0], 3, "v")

    assert result == (1, 2, 3)
    assert all(isinstance(value, int) for value in result)


def test_as_int_vector_wrong_length_raises_crystallography_value_error():
    with pytest.raises(CrystallographyValueError, match="shape"):
        as_int_vector([1, 2], 3, "v")


def test_as_int_vector_non_integer_input_raises_crystallography_value_error():
    with pytest.raises(CrystallographyValueError, match="exactly integer-valued"):
        as_int_vector([1.5, 2, 3], 3, "v")


def test_row_gcd_reduce_reduces_by_common_component_gcd():
    np.testing.assert_array_equal(
        row_gcd_reduce(np.array([6, -9, 0])),
        np.array([2, -3, 0], dtype=object),
    )


def test_row_gcd_reduce_rejects_non_integer_entries():
    with pytest.raises(CrystallographyValueError, match="integer-valued"):
        row_gcd_reduce(np.array([2.5, 0, 0]))


def test_dot_int_returns_exact_integer_dot_product():
    assert dot_int([1, 2, 3], [4, 5, 6]) == 32


def test_dot_int_rejects_unequal_lengths():
    with pytest.raises(CrystallographyValueError, match="equal lengths"):
        dot_int([1, 2], [3])


def test_cross_int3_returns_exact_integer_cross_product():
    np.testing.assert_array_equal(
        cross_int3([1, 0, 0], [0, 1, 0]),
        np.array([0, 0, 1], dtype=object),
    )


def test_cross_int3_rejects_wrong_length_vectors():
    with pytest.raises(CrystallographyValueError, match="length-3"):
        cross_int3([1, 0], [0, 1, 0])


def test_integer_det3_returns_expected_determinant():
    assert integer_det3([[4, -3, 0], [3, 4, 0], [0, 0, 1]]) == 25


def test_integer_det3_invalid_input_raises_crystallography_value_error():
    with pytest.raises(CrystallographyValueError, match="shape"):
        integer_det3([[1, 0], [0, 1]])


def test_integer_adj3_satisfies_matrix_adjugate_identity():
    matrix = np.array([[4, -3, 0], [3, 4, 0], [0, 0, 1]], dtype=object)
    determinant = integer_det3(matrix)
    adjugate = np.array(integer_adj3(matrix), dtype=object)

    np.testing.assert_array_equal(
        matrix @ adjugate,
        determinant * np.eye(3, dtype=object),
    )


def test_integer_adj3_invalid_input_raises_crystallography_value_error():
    with pytest.raises(CrystallographyValueError, match="shape"):
        integer_adj3([[1, 0], [0, 1]])
