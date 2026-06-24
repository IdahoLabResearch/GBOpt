# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for GBOpt.Utils.integer_normal_forms public normal-form routines."""

import math

import numpy as np
import pytest

from GBOpt.Utils.integer_linalg import adjugate3_int, cross_int3, det3_int, dot_int
from GBOpt.Utils.integer_normal_forms import (
    ExactNormalFormError,
    SmithNormalForm,
    column_hnf_3x3,
    primitive_integer_null_basis_3d,
    smith_normal_form_3x3,
)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _assert_python_int_array(array: np.ndarray, shape: tuple[int, ...]) -> None:
    """Assert an array has the public exact-integer representation."""
    assert array.shape == shape
    assert array.dtype == object
    assert all(isinstance(value, int) for value in array.flat)


def _assert_integer_unimodular_column_transform(
    source: np.ndarray,
    target: np.ndarray,
) -> None:
    """Assert ``target == source @ transform`` for integer unimodular transform."""
    source = np.asarray(source, dtype=object)
    target = np.asarray(target, dtype=object)
    determinant = det3_int(source)
    assert determinant != 0

    transform_numerator = np.asarray(adjugate3_int(source), dtype=object) @ target
    assert all(int(value) % determinant == 0 for value in transform_numerator.flat)

    transform = np.array(
        [int(value) // determinant for value in transform_numerator.flat],
        dtype=object,
    ).reshape(3, 3)
    _assert_python_int_array(transform, (3, 3))
    assert abs(det3_int(transform)) == 1
    np.testing.assert_array_equal(source @ transform, target)


def _assert_hnf_postconditions(H: np.ndarray) -> None:
    """Assert ``H`` satisfies the public lower column-HNF contract."""
    _assert_python_int_array(H, (3, 3))

    for row in range(3):
        diagonal = int(H[row, row])
        assert diagonal > 0

        for column in range(row):
            assert 0 <= int(H[row, column]) < diagonal

        for column in range(row + 1, 3):
            assert int(H[row, column]) == 0


def _assert_snf_postconditions(A: np.ndarray, snf: SmithNormalForm) -> None:
    """Assert a result satisfies the complete public SNF contract."""
    assert isinstance(snf, SmithNormalForm)

    for matrix in (snf.U, snf.D, snf.V):
        _assert_python_int_array(matrix, (3, 3))

    assert abs(det3_int(snf.U)) == 1
    assert abs(det3_int(snf.V)) == 1
    np.testing.assert_array_equal(snf.U @ A @ snf.V, snf.D)

    for row in range(3):
        for column in range(3):
            if row != column:
                assert int(snf.D[row, column]) == 0

    diagonal = tuple(int(snf.D[index, index]) for index in range(3))
    assert all(value >= 0 for value in diagonal)

    for left, right in zip(diagonal, diagonal[1:]):
        if left == 0:
            assert right == 0
        else:
            assert right % left == 0


INVALID_MATRICES = [
    pytest.param(
        [[1, 0], [0, 1]],
        "must have shape",
        id="wrong-shape",
    ),
    pytest.param(
        [[1.5, 0, 0], [0, 1, 0], [0, 0, 1]],
        "exactly integer-valued",
        id="nonintegral-entry",
    ),
    pytest.param(
        [[True, 0, 0], [0, 1, 0], [0, 0, 1]],
        "not an integer",
        id="boolean-entry",
    ),
]


# --------------------------------------------------------------------------------------
# smith_normal_form_3x3
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "matrix,expected_diagonal",
    [
        pytest.param(
            [[2, 0, 0], [0, 6, 0], [0, 0, 15]],
            (1, 6, 30),
            id="couples-coprime-factors",
        ),
        pytest.param(
            np.eye(3, dtype=object),
            (1, 1, 1),
            id="identity",
        ),
        pytest.param(
            np.zeros((3, 3), dtype=object),
            (0, 0, 0),
            id="rank-zero",
        ),
        pytest.param(
            [[1, 2, 3], [2, 4, 6], [0, 0, 0]],
            (1, 0, 0),
            id="rank-one",
        ),
        pytest.param(
            [[2, 4, 0], [0, 6, 0], [0, 0, 0]],
            (2, 6, 0),
            id="rank-two",
        ),
        pytest.param(
            [[-2, 0, 0], [0, 6, 0], [0, 0, -15]],
            (1, 6, 30),
            id="normalizes-negative-diagonal",
        ),
        pytest.param(
            [[0, 2, 0], [0, 0, 3], [5, 0, 0]],
            (1, 1, 30),
            id="swaps-pivot-row-and-column",
        ),
        pytest.param(
            [[10**20, 0, 0], [0, 10**20 + 1, 0], [0, 0, 1]],
            (1, 1, 10**20 * (10**20 + 1)),
            id="python-integers-beyond-int64",
        ),
    ],
)
def test_snf_returns_canonical_decomposition(matrix, expected_diagonal):
    A = np.asarray(matrix, dtype=object)

    snf = smith_normal_form_3x3(A)

    diagonal = tuple(int(snf.D[index, index]) for index in range(3))
    assert diagonal == expected_diagonal
    _assert_snf_postconditions(A, snf)


@pytest.mark.parametrize("bad_matrix,error_match", INVALID_MATRICES)
def test_snf_invalid_input_raises_exact_normal_form_error(
    bad_matrix,
    error_match,
):
    with pytest.raises(ExactNormalFormError, match=error_match):
        smith_normal_form_3x3(bad_matrix)


# --------------------------------------------------------------------------------------
# column_hnf_3x3
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "matrix,expected",
    [
        pytest.param(
            [[1, -4, -1], [2, 5, -3], [-1, 3, -1]],
            [[1, 0, 0], [0, 1, 0], [22, 2, 27]],
            id="canonicalizes-interdependent-reductions",
        ),
        pytest.param(
            [[3, 0, 0], [-7, 5, 0], [2, 1, 4]],
            [[3, 0, 0], [3, 5, 0], [0, 1, 4]],
            id="reduces-negative-below-diagonal-entries",
        ),
        pytest.param(
            np.eye(3, dtype=object),
            np.eye(3, dtype=object),
            id="identity",
        ),
        pytest.param(
            [[1, 2, 0], [0, 5, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 5, 0], [0, 0, 1]],
            id="sigma-five",
        ),
        pytest.param(
            [[2, 3, 1], [0, 4, 2], [1, 0, 5]],
            [[1, 0, 0], [0, 2, 0], [11, 15, 21]],
            id="general-full-rank-matrix",
        ),
        pytest.param(
            [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
            np.eye(3, dtype=object),
            id="swaps-leading-zero-column",
        ),
        pytest.param(
            [[-2, 0, 0], [0, -3, 0], [0, 0, -5]],
            [[2, 0, 0], [0, 3, 0], [0, 0, 5]],
            id="normalizes-negative-diagonal",
        ),
        pytest.param(
            [[10**20, 1, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [1, 10**20, 0], [0, 0, 1]],
            id="python-integers-beyond-int64",
        ),
    ],
)
def test_hnf_returns_canonical_lower_column_form(matrix, expected):
    A = np.asarray(matrix, dtype=object)

    H = column_hnf_3x3(A)

    np.testing.assert_array_equal(H, np.asarray(expected, dtype=object))
    _assert_hnf_postconditions(H)
    _assert_integer_unimodular_column_transform(A, H)


@pytest.mark.parametrize("bad_matrix,error_match", INVALID_MATRICES)
def test_hnf_invalid_input_raises_exact_normal_form_error(
    bad_matrix,
    error_match,
):
    with pytest.raises(ExactNormalFormError, match=error_match):
        column_hnf_3x3(bad_matrix)


def test_hnf_singular_matrix_raises_exact_normal_form_error():
    matrix = [[1, 2, 3], [2, 4, 6], [0, 0, 0]]

    with pytest.raises(ExactNormalFormError, match="full-rank"):
        column_hnf_3x3(matrix)


# --------------------------------------------------------------------------------------
# primitive_integer_null_basis_3d
# --------------------------------------------------------------------------------------


NULL_BASIS_CASES = [
    pytest.param((1, 2, 3), id="primitive-positive"),
    pytest.param((2, 4, 6), id="nonprimitive-positive"),
    pytest.param((0, 4, 6), id="zero-first-component"),
    pytest.param((4, 0, 6), id="zero-second-component"),
    pytest.param((4, 6, 0), id="zero-third-component"),
    pytest.param((-1, 2, -3), id="signed-covector"),
    pytest.param((0, 0, -7), id="axis-aligned-covector"),
    pytest.param(
        (10**20, 10**20 + 1, 1),
        id="python-integers-beyond-int64",
    ),
]


@pytest.mark.parametrize("covector", NULL_BASIS_CASES)
def test_primitive_integer_null_basis_returns_saturated_oriented_basis(covector):
    cov = np.asarray(covector, dtype=object)
    component_gcd = math.gcd(*(abs(int(value)) for value in cov))
    primitive_covector = np.array(
        [int(value) // component_gcd for value in cov],
        dtype=object,
    )

    basis = primitive_integer_null_basis_3d(cov)

    _assert_python_int_array(basis, (3, 2))
    assert dot_int(cov, basis[:, 0]) == 0
    assert dot_int(cov, basis[:, 1]) == 0
    np.testing.assert_array_equal(
        cross_int3(basis[:, 0], basis[:, 1]),
        primitive_covector,
    )


def test_primitive_integer_null_basis_zero_covector_raises():
    with pytest.raises(ExactNormalFormError, match="zero vector"):
        primitive_integer_null_basis_3d([0, 0, 0])


@pytest.mark.parametrize(
    "bad_covector,error_match",
    [
        pytest.param([1, 0], "must have shape", id="wrong-shape"),
        pytest.param(
            [1.5, 0, 0],
            "exactly integer-valued",
            id="nonintegral-entry",
        ),
        pytest.param([True, 0, 0], "not an integer", id="boolean-entry"),
    ],
)
def test_primitive_integer_null_basis_invalid_covector_raises(
    bad_covector,
    error_match,
):
    with pytest.raises(ExactNormalFormError, match=error_match):
        primitive_integer_null_basis_3d(bad_covector)
