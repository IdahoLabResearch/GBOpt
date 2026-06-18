# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for GBOpt.Utils.integer_normal_forms low-level helpers.

Migrated from test_exact.py. These tests cover functions directly in
integer_normal_forms and have no crystallography package dependencies.
"""

import numpy as np
import pytest

from GBOpt.Utils.integer_normal_forms import (
    ExactNormalFormError,
    _cross_int3,
    _dot_int,
    _int_adj3,
    _int_det3,
    column_hnf_3x3,
    hnf_2d_supercells,
    primitive_integer_null_basis_3d,
    smith_normal_form_3x3,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_integer_unimodular_column_transform(source, target):
    """Assert ``target == source @ U`` for integer unimodular U."""
    source = np.asarray(source, dtype=object)
    target = np.asarray(target, dtype=object)
    det = _int_det3(source)
    adj = np.asarray(_int_adj3(source), dtype=object)
    transform_num = adj @ target
    assert det != 0
    assert all(int(v) % det == 0 for v in transform_num.flat)
    transform = np.array(
        [int(v) // det for v in transform_num.flat], dtype=object
    ).reshape(3, 3)
    assert abs(_int_det3(transform)) == 1
    np.testing.assert_array_equal(source @ transform, target)


def _hnf_postcondition(H: np.ndarray) -> None:
    """Assert H satisfies column-HNF postconditions."""
    H = np.asarray(H, dtype=object)
    for j in range(3):
        assert int(H[j, j]) > 0, f"diagonal H[{j},{j}]={H[j, j]} not positive"
        for i in range(j):
            assert 0 <= int(H[j, i]) < int(H[j, j]), (
                f"H[{j},{i}]={H[j, i]} not in [0, {H[j, j]})"
            )
        for i in range(j + 1, 3):
            assert int(H[j, i]) == 0, f"upper-triangle H[{j},{i}]={H[j, i]} != 0"


# ---------------------------------------------------------------------------
# _dot_int
# ---------------------------------------------------------------------------

def test_dot_int_positive():
    assert _dot_int([1, 2, 3], [4, 5, 6]) == 32


def test_dot_int_negative():
    assert _dot_int([-1, 2, -3], [4, -5, 6]) == -32


def test_dot_int_zero():
    assert _dot_int([1, 0, 0], [0, 1, 0]) == 0


def test_dot_int_large_entries():
    # Verify Python-int arithmetic avoids int64 overflow.
    big = 10 ** 10
    assert _dot_int([big, big], [big, big]) == 2 * big ** 2


# ---------------------------------------------------------------------------
# hnf_2d_supercells
# ---------------------------------------------------------------------------

def test_hnf_2d_supercells_count_and_det():
    hnfs_2 = hnf_2d_supercells(2)
    assert len(hnfs_2) == 3
    for H in hnfs_2:
        det = int(H[0, 0]) * int(H[1, 1]) - int(H[0, 1]) * int(H[1, 0])
        assert det == 2

    assert len(hnf_2d_supercells(6)) == 12


@pytest.mark.parametrize("index", [1, 2, 6])
def test_hnf_2d_supercells_are_canonical_and_unique(index):
    hnfs = hnf_2d_supercells(index)
    seen = set()

    for H in hnfs:
        assert int(H[0, 0]) > 0
        assert int(H[1, 1]) > 0
        assert int(H[0, 1]) == 0
        assert 0 <= int(H[1, 0]) < int(H[1, 1])
        det = int(H[0, 0]) * int(H[1, 1])
        assert det == index
        seen.add(tuple(int(v) for v in H.flat))

    assert len(seen) == len(hnfs)


def test_hnf_2d_supercells_rejects_numpy_bool():
    # numpy booleans are a common accidental input (e.g. from array indexing); the
    # function must reject them at runtime even though type checkers already flag them
    # statically.
    with pytest.raises(ExactNormalFormError, match="positive integer"):
        hnf_2d_supercells(np.bool_(True))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# smith_normal_form_3x3
# ---------------------------------------------------------------------------

def test_snf_known_diagonal_couples_coprime_factors():
    """Deterministic SNF oracle."""
    A = np.array([[2, 0, 0], [0, 6, 0], [0, 0, 15]], dtype=object)

    snf = smith_normal_form_3x3(A)

    np.testing.assert_array_equal(
        np.diag(snf.D).astype(int),
        np.array([1, 6, 30], dtype=int),
    )
    np.testing.assert_array_equal(snf.U @ A @ snf.V, snf.D)


# ---------------------------------------------------------------------------
# column_hnf_3x3
# ---------------------------------------------------------------------------

def test_hnf_counterexample_reduction_order():
    """Concrete matrix that exposed the descending-reduction-order bug.

    Before the fix, the reduction loop used range(2, i, -1) (descending),
    which caused H[2,0] to be re-dirtied after it was first reduced, yielding
    H[2,0]=-5 and violating the canonical residue condition.
    """
    A = np.array([[1, -4, -1], [2, 5, -3], [-1, 3, -1]], dtype=object)
    H = column_hnf_3x3(A)
    _hnf_postcondition(H)
    assert abs(_int_det3(H)) == abs(_int_det3(A))


def test_hnf_negative_offdiagonal():
    """Matrix whose triangularization produces a negative off-diagonal entry."""
    A = np.array([[3, 0, 0], [-7, 5, 0], [2, 1, 4]], dtype=object)
    H = column_hnf_3x3(A)
    _hnf_postcondition(H)
    assert abs(_int_det3(H)) == abs(_int_det3(A))


def test_hnf_known_answer_identity():
    """Identity matrix is already in HNF."""
    A = np.eye(3, dtype=object)
    H = column_hnf_3x3(A)
    assert np.array_equal(H, A)
    _hnf_postcondition(H)


def test_hnf_known_answer_sigma5():
    """Sigma5 CSL matrix reduces to the expected HNF."""
    A = np.array([[1, 2, 0], [0, 5, 0], [0, 0, 1]], dtype=object)
    H = column_hnf_3x3(A)
    _hnf_postcondition(H)
    assert abs(_int_det3(H)) == 5
    expected = np.array([[1, 0, 0], [0, 5, 0], [0, 0, 1]], dtype=object)
    assert np.array_equal(H, expected)


def test_hnf_round_trip_unimodular():
    """column_hnf_3x3(A) == A @ V for some unimodular integer matrix V."""
    A = np.array([[2, 3, 1], [0, 4, 2], [1, 0, 5]], dtype=object)
    H = column_hnf_3x3(A)
    _hnf_postcondition(H)
    A_float = np.asarray(A, dtype=float)
    H_float = np.asarray(H, dtype=float)
    V_float = np.linalg.solve(A_float, H_float)
    V_int = np.round(V_float).astype(int)
    assert np.allclose(V_float, V_int, atol=1e-9), "V is not integer-valued"
    assert abs(int(round(np.linalg.det(V_float)))) == 1, "V is not unimodular"
    assert np.array_equal(
        np.asarray(A, dtype=int) @ V_int,
        np.asarray(H, dtype=int),
    )


# ---------------------------------------------------------------------------
# primitive_integer_null_basis_3d (_cross_int3 overflow regression)
# ---------------------------------------------------------------------------

def test_primitive_integer_null_basis_uses_python_int_cross_product():
    """Large-integer covector must not overflow int64 in the cross product."""
    covector = np.array([10**20, 10**20 + 1, 1], dtype=object)

    basis = primitive_integer_null_basis_3d(covector)

    assert _dot_int(covector, basis[:, 0]) == 0
    assert _dot_int(covector, basis[:, 1]) == 0
    np.testing.assert_array_equal(_cross_int3(basis[:, 0], basis[:, 1]), covector)
