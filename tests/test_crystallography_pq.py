# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import math

import numpy as np
import pytest
from crystallography_fixtures import SIGMA5_TWIST_PRIMITIVE_P, SIGMA5_TWIST_PRIMITIVE_Q

from GBOpt.crystallography.pq import (
    canonicalize_pq,
    canonicalize_pq_paired,
    recover_exact_row_rotation_from_paired_pq,
)
from GBOpt.crystallography.rotation import assert_scaled_rotation
from GBOpt.crystallography.types import CrystallographyValueError


def _make_identity_pair():
    I = np.eye(3, dtype=float)
    return I.copy(), I.copy()


# ---------------------------------------------------------------------------
# canonicalize_pq
# ---------------------------------------------------------------------------

def test_canonicalize_pq_idempotent_identity():
    P, Q = _make_identity_pair()
    P1, Q1 = canonicalize_pq(P, Q)
    P2, Q2 = canonicalize_pq(P1, Q1)
    np.testing.assert_array_equal(P1, P2)
    np.testing.assert_array_equal(Q1, Q2)


def test_canonicalize_pq_idempotent_sigma5():
    P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    Q = np.array([[3, 1, 0], [0, 0, 1], [-1, 3, 0]], dtype=float)
    P1, Q1 = canonicalize_pq(P, Q)
    P2, Q2 = canonicalize_pq(P1, Q1)
    np.testing.assert_array_equal(P1, P2)
    np.testing.assert_array_equal(Q1, Q2)


def test_canonicalize_pq_row_scaling_equivalent():
    P_base = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    P_scaled = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 2]], dtype=float)
    Q = np.eye(3, dtype=float)
    P_canon_base, _ = canonicalize_pq(P_base, Q)
    P_canon_scaled, _ = canonicalize_pq(P_scaled, Q)
    np.testing.assert_array_equal(P_canon_base, P_canon_scaled)


def test_canonicalize_pq_inplane_basis_equivalent():
    P_a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    P_b = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]], dtype=float)
    Q = np.eye(3, dtype=float)
    Pa, _ = canonicalize_pq(P_a, Q)
    Pb, _ = canonicalize_pq(P_b, Q)
    np.testing.assert_array_equal(Pa, Pb)


def test_canonicalize_pq_inplane_basis_longer_combination():
    P_a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    P_b = np.array([[1, 0, 0], [0, 1, 0], [0, 2, 1]], dtype=float)
    Q = np.eye(3, dtype=float)
    Pa, _ = canonicalize_pq(P_a, Q)
    Pb, _ = canonicalize_pq(P_b, Q)
    np.testing.assert_array_equal(Pa, Pb)


def test_canonicalize_pq_inplane_row_sign_equivalent():
    P_a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    P_b = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    Q = np.eye(3, dtype=float)
    Pa, _ = canonicalize_pq(P_a, Q)
    Pb, _ = canonicalize_pq(P_b, Q)
    np.testing.assert_array_equal(Pa, Pb)


def test_canonicalize_pq_boundary_normal_sign_equivalent():
    P_a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    P_b = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
    Q = np.eye(3, dtype=float)
    Pa, _ = canonicalize_pq(P_a, Q)
    Pb, _ = canonicalize_pq(P_b, Q)
    np.testing.assert_array_equal(Pa, Pb)


def test_canonicalize_pq_output_right_handed_identity():
    P, Q = _make_identity_pair()
    Pc, Qc = canonicalize_pq(P, Q)
    assert np.linalg.det(Pc) > 0
    assert np.linalg.det(Qc) > 0


def test_canonicalize_pq_output_right_handed_sigma5():
    P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    Q = np.array([[3, 1, 0], [0, 0, 1], [-1, 3, 0]], dtype=float)
    Pc, Qc = canonicalize_pq(P, Q)
    assert np.linalg.det(Pc) > 0
    assert np.linalg.det(Qc) > 0


def test_canonicalize_pq_output_right_handed_after_sign_fixes():
    P = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
    Q = np.eye(3, dtype=float)
    Pc, _ = canonicalize_pq(P, Q)
    assert np.linalg.det(Pc) > 0


def test_canonicalize_pq_swapped_inplane_rows_same_canonical():
    P_a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    P_b = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
    Q = np.eye(3, dtype=float)
    Pa, _ = canonicalize_pq(P_a, Q)
    Pb, _ = canonicalize_pq(P_b, Q)
    np.testing.assert_array_equal(Pa, Pb)


def test_canonicalize_pq_swapped_sigma5_inplane_rows_same_canonical():
    P = np.eye(3, dtype=float)
    Q_a = np.array([[3, 1, 0], [0, 0, 1], [-1, 3, 0]], dtype=float)
    Q_b = np.array([[3, 1, 0], [-1, 3, 0], [0, 0, 1]], dtype=float)
    _, Qa = canonicalize_pq(P, Q_a)
    _, Qb = canonicalize_pq(P, Q_b)
    np.testing.assert_array_equal(Qa, Qb)


def test_canonicalize_pq_rational_row_raises():
    P_rational = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]], dtype=float)
    Q = np.eye(3, dtype=float)
    with pytest.raises(CrystallographyValueError):
        canonicalize_pq(P_rational, Q)


def test_canonicalize_pq_mixed_rational_row_raises():
    P_bad = np.array([[1, 0, 0], [0, 1.5, 0], [0, 0, 1]], dtype=float)
    Q = np.eye(3, dtype=float)
    with pytest.raises(CrystallographyValueError):
        canonicalize_pq(P_bad, Q)


def test_canonicalize_pq_large_noninteger_row_raises():
    P_large = np.array([[100000.4, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    Q = np.eye(3, dtype=float)
    with pytest.raises(CrystallographyValueError):
        canonicalize_pq(P_large, Q)


# ---------------------------------------------------------------------------
# canonicalize_pq_paired
# ---------------------------------------------------------------------------

def test_canonicalize_pq_paired_nonprimitive_inplane_rows_are_reduced():

    P = np.array([[0, 0, 1], [2, 0, 0], [0, 2, 0]], dtype=float)
    Q = np.array([[0, 0, 1], [4, 0, 0], [0, 6, 0]], dtype=float)
    P_c, Q_c = canonicalize_pq_paired(P, Q)
    for M, name in [(P_c, "P"), (Q_c, "Q")]:
        for i, row in enumerate(M):
            ints = np.round(row).astype(int)
            gcd = 0
            for v in ints:
                gcd = math.gcd(gcd, abs(int(v)))
            assert gcd == 1, f"{name} row {i} {row} has GCD {gcd}, expected 1"
        assert np.dot(M[0].astype(int), M[1].astype(int)) == 0
        assert np.dot(M[0].astype(int), M[2].astype(int)) == 0


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
    Pa, _ = canonicalize_pq_paired(P_a, Q)
    Pb, _ = canonicalize_pq_paired(P_b, Q.copy())
    np.testing.assert_array_equal(Pa, Pb)


@pytest.mark.parametrize("P,Q", [
    (np.array([[0.5, 0, 0], [0, 1, 0], [0, 0, 1]],
     dtype=float), np.eye(3, dtype=float)),
    (np.eye(3, dtype=float), np.array(
        [[0.5, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)),
    (np.array([[1, 0, 0], [0, 1.5, 0], [0, 0, 1]],
     dtype=float), np.eye(3, dtype=float)),
])
def test_canonicalize_pq_paired_non_integer_raises(P, Q):
    with pytest.raises(CrystallographyValueError):
        canonicalize_pq_paired(P, Q)


# ---------------------------------------------------------------------------
# recover_exact_row_rotation_from_paired_pq
# ---------------------------------------------------------------------------

def test_recover_exact_row_rotation_sigma5_correct_N():
    P = np.array(SIGMA5_TWIST_PRIMITIVE_P, dtype=float)
    Q = np.array(SIGMA5_TWIST_PRIMITIVE_Q, dtype=float)
    rot = recover_exact_row_rotation_from_paired_pq(P, Q)
    assert rot.N == 5


def test_recover_exact_row_rotation_satisfies_row_rotation_contract():
    P = np.array(SIGMA5_TWIST_PRIMITIVE_P, dtype=float)
    Q = np.array(SIGMA5_TWIST_PRIMITIVE_Q, dtype=float)
    rot = recover_exact_row_rotation_from_paired_pq(P, Q)

    P_int = np.round(P).astype(object)
    Q_int = np.round(Q).astype(object)
    M = np.asarray(rot.M, dtype=object)
    N = int(rot.N)

    for p_row, q_row in zip(P_int, Q_int):
        np.testing.assert_array_equal(p_row @ M, N * q_row)


def test_recover_exact_row_rotation_result_is_valid_scaled_rotation():
    P = np.array(SIGMA5_TWIST_PRIMITIVE_P, dtype=float)
    Q = np.array(SIGMA5_TWIST_PRIMITIVE_Q, dtype=float)
    rot = recover_exact_row_rotation_from_paired_pq(P, Q)
    assert_scaled_rotation(rot)  # should not raise


def test_recover_exact_row_rotation_singular_P_raises():
    P = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    Q = np.eye(3, dtype=float)
    with pytest.raises(CrystallographyValueError, match="singular"):
        recover_exact_row_rotation_from_paired_pq(P, Q)


def test_recover_exact_row_rotation_non_integer_P_raises():
    P = np.array([[1.5, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    Q = np.eye(3, dtype=float)
    with pytest.raises(CrystallographyValueError):
        recover_exact_row_rotation_from_paired_pq(P, Q)


def test_recover_exact_row_rotation_non_integer_Q_raises():
    P = np.eye(3, dtype=float)
    Q = np.array([[1.5, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    with pytest.raises(CrystallographyValueError):
        recover_exact_row_rotation_from_paired_pq(P, Q)


def test_recover_exact_row_rotation_non_proper_rotation_raises():
    # P and Q that don't encode a proper rotation
    P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    Q = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=float)
    with pytest.raises(CrystallographyValueError):
        recover_exact_row_rotation_from_paired_pq(P, Q)
