# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import numpy as np
import pytest

import GBOpt.crystallography.reduction as reduction_module
from GBOpt.crystallography.integer import dot_int, integer_adj3, integer_det3
from GBOpt.crystallography.reduction import (
    GaussReductionWarning,
    gauss_reduce_2d,
    gauss_reduce_2d_paired,
    lll_reduce,
)
from GBOpt.crystallography.types import CrystallographyValueError

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _assert_integer_unimodular_column_transform(source, target):
    """Assert ``target == source @ U`` for integer unimodular U."""

    source = np.asarray(source, dtype=object)
    target = np.asarray(target, dtype=object)
    det = integer_det3(source)
    adj = np.asarray(integer_adj3(source), dtype=object)
    transform_num = adj @ target
    assert det != 0
    assert all(int(v) % det == 0 for v in transform_num.flat)
    transform = np.array(
        [int(v) // det for v in transform_num.flat], dtype=object
    ).reshape(3, 3)
    assert abs(integer_det3(transform)) == 1
    np.testing.assert_array_equal(source @ transform, target)


def _max_column_norm_sq(matrix):
    return max(dot_int(matrix[:, i], matrix[:, i]) for i in range(3))


# --------------------------------------------------------------------------------------
# gauss_reduce_2d
# --------------------------------------------------------------------------------------


def test_gauss_reduce_2d_returns_shorter_vector_first():
    r1, r2 = gauss_reduce_2d(
        np.array([0, 5, 0], dtype=float),
        np.array([0, 0, 1], dtype=float),
    )

    assert dot_int(r1, r1) <= dot_int(r2, r2)


def test_gauss_reduce_2d_preserves_inplane_area():
    v1 = np.array([0, 5, 0])
    v2 = np.array([0, 0, 1])

    r1, r2 = gauss_reduce_2d(v1, v2)

    normal_before = np.cross(v1, v2)
    normal_after = np.cross(r1, r2)
    magnitude_before = dot_int(normal_before, normal_before)
    magnitude_after = dot_int(normal_after, normal_after)
    assert magnitude_after == magnitude_before


def test_gauss_reduce_2d_iteration_limit_warns_and_returns_rows(monkeypatch):
    monkeypatch.setattr(reduction_module, "MAX_GAUSS_REDUCTION_STEPS", 1)

    with pytest.warns(GaussReductionWarning, match="Convergence not reached"):
        p1, p2 = reduction_module.gauss_reduce_2d(
            np.array([1, 0, 0]),
            np.array([2, 1, 0]),
        )

    np.testing.assert_array_equal(p1, np.array([1, 0, 0]))
    np.testing.assert_array_equal(p2, np.array([0, 1, 0]))


@pytest.mark.parametrize(
    ("v1", "v2", "match"),
    [
        pytest.param(
            np.array([[1, 0, 0]]),
            np.array([0, 1, 0]),
            "1D array",
            id="v1-not-1d",
        ),
        pytest.param(
            np.array([1, 0, 0]),
            np.array([[0, 1, 0]]),
            "1D array",
            id="v2-not-1d",
        ),
        pytest.param(
            np.array([1, 0, 0]),
            np.array([0, 1]),
            "match each other",
            id="length-mismatch",
        ),
        pytest.param(
            np.array([1.5, 0, 0]),
            np.array([0, 1, 0]),
            "integer-valued",
            id="v1-non-integer",
        ),
        pytest.param(
            np.array([1, 0, 0]),
            np.array([0, 1.5, 0]),
            "integer-valued",
            id="v2-non-integer",
        ),
    ],
)
def test_gauss_reduce_2d_rejects_invalid_vectors(v1, v2, match):
    with pytest.raises(CrystallographyValueError, match=match):
        gauss_reduce_2d(v1, v2)


def test_gauss_reduce_2d_already_reduced_unchanged():
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])

    r1, r2 = gauss_reduce_2d(v1, v2)

    np.testing.assert_array_equal(r1, v1)
    np.testing.assert_array_equal(r2, v2)


# --------------------------------------------------------------------------------------
# gauss_reduce_2d_paired
# --------------------------------------------------------------------------------------


def test_gauss_reduce_2d_paired_returns_shorter_vector_first():
    p1 = np.array([0, 5, 0])
    p2 = np.array([0, 0, 1])
    q1 = np.array([0, 1, 0])
    q2 = np.array([0, 0, 1])

    r1, r2, _, _ = gauss_reduce_2d_paired(p1, p2, q1, q2)

    assert dot_int(r1, r1) <= dot_int(r2, r2)


def test_gauss_reduce_2d_paired_iteration_limit_warns_and_returns_rows(monkeypatch):
    monkeypatch.setattr(reduction_module, "MAX_GAUSS_REDUCTION_STEPS", 1)

    with pytest.warns(GaussReductionWarning, match="Convergence not reached"):
        p1, p2, q1, q2 = reduction_module.gauss_reduce_2d_paired(
            np.array([1, 0, 0]),
            np.array([2, 1, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
        )

    np.testing.assert_array_equal(p1, np.array([1, 0, 0]))
    np.testing.assert_array_equal(p2, np.array([0, 1, 0]))
    np.testing.assert_array_equal(q1, np.array([0, 1, 0]))
    np.testing.assert_array_equal(q2, np.array([0, -2, 1]))


def test_gauss_reduce_2d_paired_negative_projection_large():
    p1 = np.array([3, 0, 0])
    p2 = np.array([-10, 1, 0])
    q1 = np.array([1, 0, 0])
    q2 = np.array([0, 1, 0])

    a, b, qa, qb = gauss_reduce_2d_paired(p1, p2, q1, q2)

    np.testing.assert_array_equal(a, np.array([-1, 1, 0]))
    np.testing.assert_array_equal(b, np.array([2, 1, 0]))
    np.testing.assert_array_equal(qa, np.array([3, 1, 0]))
    np.testing.assert_array_equal(qb, np.array([4, 1, 0]))


def test_gauss_reduce_2d_paired_same_ops_applied_to_q():
    p1 = np.array([5, 0, 0])
    p2 = np.array([3, 4, 0])

    a, b, qa, qb = gauss_reduce_2d_paired(p1, p2, p1.copy(), p2.copy())

    np.testing.assert_array_equal(qa, a)
    np.testing.assert_array_equal(qb, b)


def test_gauss_reduce_2d_paired_dependent_vectors_return_zero_shortest_row():
    p1 = np.array([1, 2, 0])
    p2 = np.array([2, 4, 0])
    q1 = np.array([1, 0, 0])
    q2 = np.array([2, 0, 0])

    a, b, qa, qb = gauss_reduce_2d_paired(p1, p2, q1, q2)

    np.testing.assert_array_equal(a, np.array([0, 0, 0]))
    np.testing.assert_array_equal(b, np.array([1, 2, 0]))
    np.testing.assert_array_equal(qa, np.array([0, 0, 0]))
    np.testing.assert_array_equal(qb, np.array([1, 0, 0]))


@pytest.mark.parametrize(
    ("p1", "p2", "q1", "q2", "match"),
    [
        pytest.param(
            np.array([1, 0, 0]),
            np.array([0, 1]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            "match each other",
            id="p-length-mismatch",
        ),
        pytest.param(
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1]),
            "match each other",
            id="q-length-mismatch",
        ),
        pytest.param(
            np.array([[1, 0, 0]]),
            np.array([0, 1, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            "1D array",
            id="p1-not-1d",
        ),
        pytest.param(
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([[1, 0, 0]]),
            np.array([0, 1, 0]),
            "1D array",
            id="q1-not-1d",
        ),
        pytest.param(
            np.array([1.5, 0, 0]),
            np.array([0, 1, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            "integer-valued",
            id="non-integer-p",
        ),
        pytest.param(
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([1.5, 0, 0]),
            np.array([0, 1, 0]),
            "integer-valued",
            id="non-integer-q",
        ),
    ],
)
def test_gauss_reduce_2d_paired_rejects_invalid_vectors(p1, p2, q1, q2, match):
    with pytest.raises(CrystallographyValueError, match=match):
        gauss_reduce_2d_paired(p1, p2, q1, q2)


def test_gauss_reduce_2d_paired_allows_q_row_length_to_differ_from_p_rows():
    p1 = np.array([5, 0, 0])
    p2 = np.array([3, 4, 0])
    q1 = np.array([1, 0])
    q2 = np.array([0, 1])

    a, b, qa, qb = gauss_reduce_2d_paired(p1, p2, q1, q2)

    assert a.shape == (3,)
    assert b.shape == (3,)
    assert qa.shape == (2,)
    assert qb.shape == (2,)


def test_gauss_reduce_2d_paired_already_reduced_unchanged():
    p1 = np.array([1, 0, 0])
    p2 = np.array([0, 1, 0])
    q1 = np.array([1, 0, 0])
    q2 = np.array([0, 1, 0])

    r1, r2, s1, s2 = gauss_reduce_2d_paired(p1, p2, q1, q2)

    np.testing.assert_array_equal(r1, p1)
    np.testing.assert_array_equal(r2, p2)
    np.testing.assert_array_equal(s1, q1)
    np.testing.assert_array_equal(s2, q2)


# --------------------------------------------------------------------------------------
# lll_reduce
# --------------------------------------------------------------------------------------


def test_lll_reduce_identity_unchanged():
    basis = np.eye(3, dtype=object)

    reduced = lll_reduce(basis)

    np.testing.assert_array_equal(reduced, basis)
    assert reduced.dtype == object


def test_lll_reduce_shortens_skewed_basis_and_preserves_lattice():
    basis = np.array([[1, 10, 0], [0, 1, 0], [0, 0, 1]], dtype=object)

    reduced = lll_reduce(basis)

    assert _max_column_norm_sq(reduced) < _max_column_norm_sq(basis)
    _assert_integer_unimodular_column_transform(basis, reduced)


def test_lll_reduce_large_integer_basis_preserves_lattice():
    big = 10**12
    basis = np.array([[1, big, 0], [0, 1, 0], [0, 0, 1]], dtype=object)

    reduced = lll_reduce(basis)

    _assert_integer_unimodular_column_transform(basis, reduced)
    assert _max_column_norm_sq(reduced) < big * big


@pytest.mark.parametrize(
    "delta",
    [
        pytest.param(0.25, id="closed-lower-bound"),
        pytest.param(0.0, id="zero"),
        pytest.param(1.01, id="above-one"),
        pytest.param(object(), id="non-numeric-object"),
        pytest.param(np.nan, id="nan"),
    ],
)
def test_lll_reduce_rejects_invalid_delta(delta):
    with pytest.raises(CrystallographyValueError, match="delta"):
        lll_reduce(np.eye(3, dtype=object), delta=delta)


def test_lll_reduce_accepts_delta_one():
    basis = np.eye(3, dtype=object)

    reduced = lll_reduce(basis, delta=1.0)

    np.testing.assert_array_equal(reduced, basis)


@pytest.mark.parametrize(
    ("basis", "match"),
    [
        pytest.param(np.eye(2, dtype=object), "shape", id="wrong-shape"),
        pytest.param(np.eye(3, dtype=float) * 1.5, "integer-valued", id="non-integer"),
        pytest.param(
            [[True, 0, 0], [0, 1, 0], [0, 0, 1]],
            "not an integer",
            id="bool",
        ),
        pytest.param(
            np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0]], dtype=object),
            "full-rank",
            id="singular",
        ),
    ],
)
def test_lll_reduce_rejects_invalid_basis_input(basis, match):
    with pytest.raises(CrystallographyValueError, match=match):
        lll_reduce(basis)
