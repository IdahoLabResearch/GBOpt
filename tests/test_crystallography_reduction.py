# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from unittest.mock import patch

import numpy as np
import pytest

from GBOpt.crystallography.csl import csl_from_scaled_rotation
from GBOpt.crystallography.integer import integer_det3
from GBOpt.crystallography.quaternion import quaternion_to_scaled_rotation
from GBOpt.crystallography.reduction import (
    GaussReductionWarning,
    gauss_reduce_2d,
    gauss_reduce_2d_paired,
    lll_reduce,
)
from GBOpt.crystallography.types import CrystallographyValueError

# ---------------------------------------------------------------------------
# gauss_reduce_2d
# ---------------------------------------------------------------------------


def test_gauss_reduce_2d_shorter_vector_is_first():
    v1 = np.array([0, 5, 0], dtype=float)
    v2 = np.array([0, 0, 1], dtype=float)
    r1, r2 = gauss_reduce_2d(v1, v2)
    assert np.linalg.norm(r1) <= np.linalg.norm(r2) + 1e-10


def test_gauss_reduce_2d_spans_same_lattice():
    # Area |v1xv2| must be preserved after reduction.
    v1 = np.array([0, 5, 0], dtype=float)
    v2 = np.array([0, 0, 1], dtype=float)
    r1, r2 = gauss_reduce_2d(v1, v2)
    area_before = np.linalg.norm(np.cross(v1, v2))
    area_after = np.linalg.norm(np.cross(r1, r2))
    assert abs(area_before - area_after) < 1e-10


def test_gauss_reduce_2d_iteration_limit_warns_and_returns_rows():
    # Force the loop to exhaust without a break so the warning branch is
    # covered without constructing enormous worst-case integer inputs.
    with patch("GBOpt.crystallography.reduction.range", return_value=[0], create=True):
        with pytest.warns(GaussReductionWarning):
            p1, p2 = gauss_reduce_2d(
                np.array([1, 0, 0]),
                np.array([2, 1, 0])
            )

    np.testing.assert_array_equal(p1, np.array([1, 0, 0]))
    np.testing.assert_array_equal(p2, np.array([0, 1, 0]))


def test_gauss_reduce_2d_non_1d_raises():
    with pytest.raises(CrystallographyValueError, match="1D array"):
        gauss_reduce_2d(
            np.array([[1, 0, 0]]),  # 2D
            np.array([0, 1, 0])
        )


def test_gauss_reduce_2d_already_reduced_unchanged():
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    r1, r2 = gauss_reduce_2d(v1, v2)
    np.testing.assert_array_equal(r1, v1)
    np.testing.assert_array_equal(r2, v2)


def test_gauss_reduce_2d_non_integer_input_raises():
    with pytest.raises(CrystallographyValueError):
        gauss_reduce_2d(
            np.array([1.5, 0, 0]),
            np.array([0, 1, 0]),
        )

# ---------------------------------------------------------------------------
# gauss_reduce_2d_paired
# ---------------------------------------------------------------------------


def test_gauss_reduce_2d_paired_shorter_vector_is_first():
    p1 = np.array([0, 5, 0])
    p2 = np.array([0, 0, 1])
    q1 = np.array([0, 1, 0])
    q2 = np.array([0, 0, 1])

    r1, r2, _, _ = gauss_reduce_2d_paired(p1, p2, q1, q2)
    assert np.linalg.norm(r1) <= np.linalg.norm(r2) + 1e-10


def test_gauss_reduce_2d_paired_iteration_limit_warns_and_returns_rows():
    # Force the loop to exhaust without a break so the warning branch is
    # covered without constructing enormous worst-case integer inputs.
    with patch("GBOpt.crystallography.reduction.range", return_value=[0], create=True):
        with pytest.warns(GaussReductionWarning):
            p1, p2, q1, q2 = gauss_reduce_2d_paired(
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
    # ab < 0 and |ab| > aa//2: reduction must still converge.
    p1 = np.array([3, 0, 0])
    p2 = np.array([-10, 1, 0])
    q1 = np.array([1, 0, 0])
    q2 = np.array([0, 1, 0])
    a, b, qa, qb = gauss_reduce_2d_paired(p1, p2, q1, q2)
    np.testing.assert_array_equal(a, np.array([-1, 1, 0]))
    np.testing.assert_array_equal(b, np.array([2, 1, 0]))
    np.testing.assert_array_equal(qa, np.array([3, 1, 0]))
    np.testing.assert_array_equal(qb, np.array([4, 1, 0]))
    assert np.linalg.matrix_rank(np.array([a, b])) == 2


def test_gauss_reduce_2d_paired_same_ops_applied_to_q():
    # Verify Q rows get exactly the same row operations as P rows.
    p1 = np.array([5, 0, 0])
    p2 = np.array([3, 4, 0])
    # Set Q equal to P; after paired reduction Q must equal the reduced P.
    a, b, qa, qb = gauss_reduce_2d_paired(p1, p2, p1.copy(), p2.copy())
    assert np.array_equal(a, qa)
    assert np.array_equal(b, qb)


def test_gauss_reduce_2d_paired_dependent_vectors_produce_zero_row():
    p1 = np.array([1, 2, 0])
    p2 = np.array([2, 4, 0])   # parallel to p1
    q1 = np.array([1, 0, 0])
    q2 = np.array([2, 0, 0])
    a, b, _qa, _qb = gauss_reduce_2d_paired(p1, p2, q1, q2)
    assert np.allclose(b, 0) or np.allclose(a, 0)


def test_gauss_reduce_2d_paired_shape_mismatch_raises():
    with pytest.raises(CrystallographyValueError, match="match each other"):
        gauss_reduce_2d_paired(
            np.array([1, 0, 0]),
            np.array([0, 1]),     # wrong length
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
        )


def test_gauss_reduce_2d_paired_non_1d_raises():
    with pytest.raises(CrystallographyValueError, match="1D array"):
        gauss_reduce_2d_paired(
            np.array([[1, 0, 0]]),  # 2D
            np.array([0, 1, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
        )


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


def test_gauss_reduce_2d_paired_non_integer_p_raises():
    with pytest.raises(CrystallographyValueError):
        gauss_reduce_2d_paired(
            np.array([1.5, 0, 0]),
            np.array([0, 1, 0]),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
        )


def test_gauss_reduce_2d_paired_non_integer_q_raises():
    with pytest.raises(CrystallographyValueError):
        gauss_reduce_2d_paired(
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([1.5, 0, 0]),
            np.array([0, 1, 0]),
        )

# ---------------------------------------------------------------------------
# lll_reduce
# ---------------------------------------------------------------------------


def _assert_integer_unimodular_column_transform(source, target):
    """Assert ``target == source @ U`` for integer unimodular U."""
    from GBOpt.crystallography.integer import integer_adj3
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


def test_lll_reduce_identity_unchanged():
    B = np.eye(3, dtype=object)
    R = lll_reduce(B)
    det = int(np.round(float(np.linalg.det(np.asarray(R, dtype=float)))))
    assert abs(det) == 1
    assert np.allclose(np.asarray(R, dtype=float), np.eye(3), atol=1e-9)


def test_lll_reduce_actually_reduces():
    B = np.array([[1, 10, 0], [0, 1, 0], [0, 0, 1]], dtype=object)
    R = lll_reduce(B)
    input_norms = [np.linalg.norm(B[:, i].astype(float)) for i in range(3)]
    output_norms = [np.linalg.norm(np.asarray(R[:, i], dtype=float)) for i in range(3)]
    assert max(output_norms) < max(input_norms)
    _assert_integer_unimodular_column_transform(B, R)


def test_lll_reduce_same_lattice_as_csl_hnf():
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))
    csl = csl_from_scaled_rotation(rot)
    csl_lll = csl_from_scaled_rotation(rot, post_reduce="lll")
    assert csl_lll.sigma == csl.sigma
    _assert_integer_unimodular_column_transform(csl.basis_hnf, csl_lll.basis)


def test_lll_reduce_large_integer_basis_preserves_lattice():
    big = 10**12
    B = np.array([[1, big, 0], [0, 1, 0], [0, 0, 1]], dtype=object)
    R = lll_reduce(B)
    _assert_integer_unimodular_column_transform(B, R)
    from GBOpt.Utils.integer_normal_forms import _dot_int
    assert max(_dot_int(R[:, i], R[:, i]) for i in range(3)) < big * big


def test_lll_reduce_singular_raises():
    with pytest.raises(CrystallographyValueError, match="full-rank"):
        lll_reduce(np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0]], dtype=object))
