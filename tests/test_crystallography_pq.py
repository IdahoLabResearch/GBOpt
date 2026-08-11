# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
import math

import numpy as np
import pytest
from crystallography_fixtures import SIGMA5_TWIST_PRIMITIVE_P, SIGMA5_TWIST_PRIMITIVE_Q

from GBOpt.crystallography.pq import (
    canonicalize_pq_paired,
    recover_exact_row_rotation_from_paired_pq,
)
from GBOpt.crystallography.rotation import assert_scaled_rotation
from GBOpt.crystallography.types import CrystallographyValueError
from GBOpt.Utils.integer_linalg import det3_int

IDENTITY_PQ = np.eye(3, dtype=float)

SIGMA5_Q = np.array([[3, 1, 0], [0, 0, 1], [-1, 3, 0]], dtype=float)


def _assert_rows_are_primitive(matrix):
    for row in matrix:
        assert math.gcd(*map(abs, row)) == 1


def _sigma5_twist_primitive_pair():
    return (
        np.array(SIGMA5_TWIST_PRIMITIVE_P, dtype=float),
        np.array(SIGMA5_TWIST_PRIMITIVE_Q, dtype=float),
    )


# --------------------------------------------------------------------------------------
# canonicalize_pq_paired
# --------------------------------------------------------------------------------------


def test_canonicalize_pq_paired_reduces_nonprimitive_rows():
    P = np.array([[0, 0, 2], [2, 0, 0], [0, 3, 0]], dtype=float)
    Q = np.array([[0, 0, 4], [4, 0, 0], [0, 6, 0]], dtype=float)

    P_c, Q_c = canonicalize_pq_paired(P, Q)

    np.testing.assert_array_equal(
        P_c,
        np.array(
            [
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        Q_c,
        np.array(
            [
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=object,
        ),
    )


def test_canonicalize_pq_paired_preserves_paired_inplane_basis():
    P = np.array([[0, 0, 1], [2, 1, 0], [-1, 2, 0]], dtype=float)
    Q = np.array([[0, 0, 1], [1, 2, 0], [-2, 1, 0]], dtype=float)

    P_c, Q_c = canonicalize_pq_paired(P, Q)

    assert det3_int(P_c) > 0
    assert det3_int(Q_c) > 0

    assert np.dot(P_c[0], P_c[1]) == 0
    assert np.dot(P_c[0], P_c[2]) == 0
    assert np.dot(Q_c[0], Q_c[1]) == 0
    assert np.dot(Q_c[0], Q_c[2]) == 0

    _assert_rows_are_primitive(P_c)
    _assert_rows_are_primitive(Q_c)


def test_canonicalize_pq_paired_idempotent():
    P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    Q = np.array([[3, 1, 0], [0, 0, 1], [1, -3, 0]], dtype=float)
    P1, Q1 = canonicalize_pq_paired(P, Q)
    P2, Q2 = canonicalize_pq_paired(P1, Q1)
    np.testing.assert_array_equal(P1, P2)
    np.testing.assert_array_equal(Q1, Q2)


def test_canonicalize_pq_paired_scaled_equivalent_inplane_rows():
    P_a = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    P_b = np.array([[0, 0, 1], [2, 0, 0], [0, 3, 0]], dtype=float)
    Q = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)

    P_a_c, Q_a_c = canonicalize_pq_paired(P_a, Q)
    P_b_c, Q_b_c = canonicalize_pq_paired(P_b, Q.copy())

    np.testing.assert_array_equal(P_a_c, P_b_c)
    np.testing.assert_array_equal(Q_a_c, Q_b_c)


@pytest.mark.parametrize(
    ("P", "Q"),
    [
        pytest.param(
            np.array([[1.5, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
            IDENTITY_PQ,
            id="noninteger-P",
        ),
        pytest.param(
            IDENTITY_PQ,
            np.array([[1, 0, 0], [0, 1.5, 0], [0, 0, 1]], dtype=float),
            id="noninteger-Q",
        ),
    ],
)
def test_canonicalize_pq_paired_rejects_invalid_values(P, Q):
    with pytest.raises(CrystallographyValueError, match="integer-valued"):
        canonicalize_pq_paired(P, Q)


def test_canonicalize_pq_paired_rejects_zero_rows():
    P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    Q = IDENTITY_PQ

    with pytest.raises(CrystallographyValueError, match="zero row"):
        canonicalize_pq_paired(P, Q)


# --------------------------------------------------------------------------------------
# recover_exact_row_rotation_from_paired_pq
# --------------------------------------------------------------------------------------


def test_recover_exact_row_rotation_returns_sigma5_denominator():
    P, Q = _sigma5_twist_primitive_pair()
    rot = recover_exact_row_rotation_from_paired_pq(P, Q)
    assert rot.denominator == 5


def test_recover_exact_row_rotation_satisfies_row_rotation_contract():
    P, Q = _sigma5_twist_primitive_pair()
    rot = recover_exact_row_rotation_from_paired_pq(P, Q)

    P_int = np.asarray(P, dtype=object)
    Q_int = np.asarray(Q, dtype=object)
    M = np.asarray(rot.matrix, dtype=object)
    N = int(rot.denominator)

    for p_row, q_row in zip(P_int, Q_int):
        np.testing.assert_array_equal(p_row @ M, N * q_row)


def test_recover_exact_row_rotation_result_is_valid_scaled_rotation():
    P, Q = _sigma5_twist_primitive_pair()
    rot = recover_exact_row_rotation_from_paired_pq(P, Q)
    assert_scaled_rotation(rot)  # should not raise


@pytest.mark.parametrize(
    ("P", "Q"),
    [
        pytest.param(
            np.array([[1.5, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
            IDENTITY_PQ,
            id="noninteger-P",
        ),
        pytest.param(
            IDENTITY_PQ,
            np.array([[1, 0, 0], [0, 1.5, 0], [0, 0, 1]], dtype=float),
            id="noninteger-Q",
        ),
    ],
)
def test_recover_exact_row_rotation_rejects_invalid_values(P, Q):
    with pytest.raises(CrystallographyValueError, match="integer-valued"):
        recover_exact_row_rotation_from_paired_pq(P, Q)


def test_recover_exact_row_rotation_rejects_singular_p():
    P = np.array(
        [
            [1, 0, 0],
            [2, 0, 0],
            [0, 1, 0],
        ],
        dtype=float,
    )
    Q = np.eye(3, dtype=float)

    with pytest.raises(CrystallographyValueError, match="singular"):
        recover_exact_row_rotation_from_paired_pq(P, Q)


def test_recover_exact_row_rotation_rejects_non_proper_rotation():
    P = np.eye(3, dtype=float)
    Q = np.diag([-1, 1, 1]).astype(float)

    with pytest.raises(CrystallographyValueError, match="exact proper rotation"):
        recover_exact_row_rotation_from_paired_pq(P, Q)


def test_recover_exact_row_rotation_normalizes_negative_det_p():
    P = np.diag([-1, 1, 1]).astype(float)
    Q = P.copy()

    rot = recover_exact_row_rotation_from_paired_pq(P, Q)

    assert rot.denominator == 1
    np.testing.assert_array_equal(rot.matrix, np.eye(3, dtype=object))


# --------------------------------------------------------------------------------------
# Shared validation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(canonicalize_pq_paired, id="canonicalize-pq-paired"),
        pytest.param(recover_exact_row_rotation_from_paired_pq, id="recover-rotation"),
    ],
)
def test_pq_functions_reject_wrong_shape(func):
    with pytest.raises(CrystallographyValueError, match="shape"):
        func(np.eye(2, dtype=float), IDENTITY_PQ)
