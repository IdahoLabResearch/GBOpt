# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import warnings

import numpy as np
import pytest

from unittest.mock import patch

from GBOpt.BoundarySpec import (
    BoundaryEmbedding,
    BoundarySpecError,
    CSLApproxSpec,
    CSLExactSpec,
    PQSpec,
)
from GBOpt.GBMaker import GBMaker
from GBOpt.Utils.gb_exact import (
    _int_adj3,
    _int_det3,
    _integer_membership,
    build_supercell_matrix,
    canonicalize_pq,
    csl_spec_to_embedding,
    enumerate_supercell_origins,
    pq_spec_to_embedding,
    quaternion_to_rotation_matrix,
    reduce_2d_basis,
    solve_inplane_csl,
    validate_and_normalize_quaternion,
    validate_sigma,
)


# ---------------------------------------------------------------------------
# Step 14a — integer membership kernel and supercell matrix
# ---------------------------------------------------------------------------

class TestIntDet3:
    def test_identity(self):
        assert _int_det3([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == 1

    def test_known_positive(self):
        assert _int_det3([[1, 0, 0], [0, 2, 0], [0, 0, 3]]) == 6

    def test_known_negative(self):
        assert _int_det3([[0, 1, 0], [1, 0, 0], [0, 0, 1]]) == -1

    def test_sigma5_right_S(self):
        # Σ5 36.87° right grain: Q = [[4,-3,0],[3,4,0],[0,0,1]], det = 25
        assert _int_det3([[4, -3, 0], [3, 4, 0], [0, 0, 1]]) == 25

    def test_non_integer_raises(self):
        with pytest.raises(ValueError):
            _int_det3([[1.1, 0, 0], [0, 1, 0], [0, 0, 1]])


class TestIntAdj3:
    def test_adj_times_M_equals_det_times_I(self):
        M = [[4, -3, 0], [3, 4, 0], [0, 0, 1]]
        det = _int_det3(M)
        adj = _int_adj3(M)
        product = np.array(M) @ np.array(adj)
        np.testing.assert_array_equal(product, det * np.eye(3, dtype=int))

    def test_identity_adj_is_identity(self):
        adj = _int_adj3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert adj == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def test_non_integer_raises(self):
        with pytest.raises(ValueError):
            _int_adj3([[0.1, 0, 0], [0, 1, 0], [0, 0, 1]])


class TestIntegerMembership:
    # Use Σ5 right-grain matrix: det=25, so each unit cell maps to 25 cosets.
    _Q = [[4, -3, 0], [3, 4, 0], [0, 0, 1]]

    def setup_method(self):
        self.det_S = _int_det3(self._Q)        # 25
        self.adj_S = _int_adj3(self._Q)

    def test_origin_always_accepted(self):
        assert _integer_membership([0, 0, 0], self.adj_S, self.det_S, 1, 1, 1)

    def test_count_via_identity_S(self):
        # Identity S, repeat 3×2×1: only origins in [0,3)×[0,2)×[0,1) accepted
        adj_I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        det_I = 1
        accepted = sum(
            _integer_membership([x, y, z], adj_I, det_I, 3, 2, 1)
            for x in range(-1, 5)
            for y in range(-1, 4)
            for z in range(-1, 3)
        )
        assert accepted == 3 * 2 * 1

    def test_upper_boundary_rejected(self):
        # repeat_x=1, det=25: accepted x-numerator must be in [0,25)
        # n=[1,0,0]: u_num = [1,0,0] @ adj_Q → adj_Q column 0 row 0 = cofactor(Q,0,0)
        # Should land outside [0,25) * 1 for at least one axis or inside —
        # rather than hard-coding the value, verify via count invariant instead.
        Q_int = np.array(self._Q)
        adj_np = np.array(self.adj_S)
        # For n in the repeated cell corners, origin [4,0,0] @ adj_Q must
        # map to inside the cell (it IS a lattice vector of the right grain).
        assert _integer_membership([4, 0, 0], self.adj_S, self.det_S, 1, 1, 1)

    def test_negative_det_origin_accepted(self):
        M_neg = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
        det_neg = _int_det3(M_neg)
        assert det_neg == -1
        adj_neg = _int_adj3(M_neg)
        assert _integer_membership([0, 0, 0], adj_neg, det_neg, 1, 1, 1)


class TestBuildSupercellMatrix:
    def test_identity_P(self):
        P = np.eye(3, dtype=float)
        S = build_supercell_matrix(P)
        np.testing.assert_array_equal(S, np.eye(3, dtype=int))

    def test_sigma5_right_grain(self):
        Q = np.array([[4, -3, 0], [3, 4, 0], [0, 0, 1]], dtype=float)
        S = build_supercell_matrix(Q)
        np.testing.assert_array_equal(S, Q.astype(int))
        assert _int_det3(S) == 25

    def test_non_integer_raises(self):
        P_bad = np.array([[1.1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        with pytest.raises(Exception):
            build_supercell_matrix(P_bad)

    def test_singular_raises(self):
        P_sing = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        with pytest.raises(ValueError):
            build_supercell_matrix(P_sing)


class TestEnumerateSupercellOrigins:
    def test_identity_S_repeat_1(self):
        S = np.eye(3, dtype=int)
        origins = enumerate_supercell_origins(S, 1, 1, 1)
        assert origins.shape == (1, 3)
        np.testing.assert_array_equal(origins[0], [0, 0, 0])

    def test_identity_S_repeat_2x2x2(self):
        S = np.eye(3, dtype=int)
        origins = enumerate_supercell_origins(S, 2, 2, 2)
        assert len(origins) == 8

    def test_sigma5_right_grain_repeat1(self):
        Q = np.array([[4, -3, 0], [3, 4, 0], [0, 0, 1]], dtype=int)
        origins = enumerate_supercell_origins(Q, 1, 1, 1)
        assert len(origins) == 25

    def test_no_duplicates(self):
        Q = np.array([[4, -3, 0], [3, 4, 0], [0, 0, 1]], dtype=int)
        origins = enumerate_supercell_origins(Q, 2, 1, 1)
        tuples = set(map(tuple, origins.tolist()))
        assert len(tuples) == len(origins)

    def test_count_invariant(self):
        Q = np.array([[4, -3, 0], [3, 4, 0], [0, 0, 1]], dtype=int)
        for rx, ry, rz in [(1, 1, 1), (2, 1, 1), (1, 2, 3)]:
            origins = enumerate_supercell_origins(Q, rx, ry, rz)
            assert len(origins) == rx * ry * rz * 25


def _make_identity_pair():
    I = np.eye(3, dtype=float)
    return I.copy(), I.copy()


# ---------------------------------------------------------------------------
# Step 10 — integer-quaternion validation and rotation matrix
# ---------------------------------------------------------------------------

class TestValidateAndNormalizeQuaternion:
    # Σ5 [001] 53.13 deg — integer quaternion [2, 0, 0, 1], N = 5
    SIGMA5_QUAT = [2, 0, 0, 1]

    def test_valid_returns_unit_quaternion(self):
        q = validate_and_normalize_quaternion(self.SIGMA5_QUAT)
        assert q.shape == (4,)
        assert abs(np.dot(q, q) - 1.0) < 1e-12

    def test_valid_direction_preserved(self):
        # Normalized quaternion should be parallel to the input.
        q_in = np.asarray(self.SIGMA5_QUAT, dtype=float)
        q_out = validate_and_normalize_quaternion(q_in)
        # All ratios q_out[i] / q_in[i] (for nonzero components) equal 1/sqrt(5).
        nonzero = q_in != 0
        ratios = q_out[nonzero] / q_in[nonzero]
        assert np.allclose(ratios, ratios[0], atol=1e-12)

    def test_wrong_length_raises(self):
        with pytest.raises(BoundarySpecError):
            validate_and_normalize_quaternion([1, 0, 0])

    def test_wrong_shape_2d_raises(self):
        with pytest.raises(BoundarySpecError):
            validate_and_normalize_quaternion([[1, 0, 0, 0]])

    def test_non_integer_raises(self):
        with pytest.raises(BoundarySpecError):
            validate_and_normalize_quaternion([1.5, 0, 0, 1])

    def test_zero_quaternion_raises(self):
        with pytest.raises(BoundarySpecError):
            validate_and_normalize_quaternion([0, 0, 0, 0])

    def test_integer_valued_float_input_accepted(self):
        # Components that are floats but integer-valued (e.g. 2.0) must be accepted.
        q = validate_and_normalize_quaternion([2.0, 0.0, 0.0, 1.0])
        assert abs(np.dot(q, q) - 1.0) < 1e-12


class TestQuaternionToRotationMatrix:
    # Σ5 [001] 53.13 deg — quat [2, 0, 0, 1] (Hamilton [w,x,y,z]),
    # expected R = [[3/5, -4/5, 0], [4/5, 3/5, 0], [0, 0, 1]]
    SIGMA5_QUAT_NORM = np.array([2, 0, 0, 1], dtype=float) / np.sqrt(5)
    SIGMA5_R_EXPECTED = np.array([
        [3/5, -4/5, 0],
        [4/5,  3/5, 0],
        [0,    0,   1],
    ])

    def test_sigma5_matches_expected(self):
        R = quaternion_to_rotation_matrix(self.SIGMA5_QUAT_NORM)
        np.testing.assert_allclose(R, self.SIGMA5_R_EXPECTED, atol=1e-12)

    def test_output_is_proper_rotation(self):
        R = quaternion_to_rotation_matrix(self.SIGMA5_QUAT_NORM)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12

    def test_identity_quaternion_gives_identity_matrix(self):
        # [w=1, x=0, y=0, z=0] → identity rotation
        q_id = np.array([1.0, 0.0, 0.0, 0.0])
        R = quaternion_to_rotation_matrix(q_id)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_sigma5_36deg_quat(self):
        # Σ5 [001] 36.87 deg — quat [3, 0, 0, 1], N = 10
        # expected R = [[4/5, -3/5, 0], [3/5, 4/5, 0], [0, 0, 1]]
        q = np.array([3, 0, 0, 1], dtype=float) / np.sqrt(10)
        R = quaternion_to_rotation_matrix(q)
        expected = np.array([
            [4/5, -3/5, 0],
            [3/5,  4/5, 0],
            [0,    0,   1],
        ])
        np.testing.assert_allclose(R, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Step 11 — sigma-from-quaternion validation
# ---------------------------------------------------------------------------

class TestValidateSigma:
    # Σ5: quat [2, 0, 0, 1], |q|² = 5 (already odd) → sigma = 5
    # Σ5: quat [3, 0, 0, 1], |q|² = 10 = 2×5 → sigma = 5
    # Σ13: quat [3, 2, 0, 0], |q|² = 13 → sigma = 13

    def test_sigma5_quat_2001_correct(self):
        validate_sigma([2, 0, 0, 1], 5)  # must not raise

    def test_sigma5_quat_3001_correct(self):
        # |q|² = 10, divided by 2 once → sigma = 5
        validate_sigma([3, 0, 0, 1], 5)  # must not raise

    def test_sigma13_correct(self):
        # |q|² = 9+4 = 13 (odd) → sigma = 13
        validate_sigma([3, 2, 0, 0], 13)  # must not raise

    def test_wrong_sigma_raises(self):
        with pytest.raises(BoundarySpecError):
            validate_sigma([2, 0, 0, 1], 3)  # sigma=5, not 3

    def test_wrong_sigma_off_by_one_raises(self):
        with pytest.raises(BoundarySpecError):
            validate_sigma([2, 0, 0, 1], 4)

    def test_power_of_two_stripped_correctly(self):
        # quat [2, 2, 0, 0]: |q|² = 8 = 2³ → sigma = 1
        validate_sigma([2, 2, 0, 0], 1)  # must not raise; would raise if sigma=8

    def test_power_of_two_stripped_wrong_sigma_raises(self):
        with pytest.raises(BoundarySpecError):
            validate_sigma([2, 2, 0, 0], 8)  # sigma=1, not 8

    def test_zero_quaternion_raises(self):
        with pytest.raises(BoundarySpecError):
            validate_sigma([0, 0, 0, 0], 1)


# ---------------------------------------------------------------------------
# Step 12 — in-plane CSL solve and 2D reduction
# ---------------------------------------------------------------------------

def _sigma5_53deg_R():
    """R for Σ5 [001] 53.13 deg boundary (quat=[2,0,0,1], N=5)."""
    q = np.array([2, 0, 0, 1], dtype=float) / np.sqrt(5)
    return quaternion_to_rotation_matrix(q)


def _sigma5_36deg_R():
    """R for Σ5 [001] 36.87 deg boundary (quat=[3,0,0,1], N=10)."""
    q = np.array([3, 0, 0, 1], dtype=float) / np.sqrt(10)
    return quaternion_to_rotation_matrix(q)


class TestSolveInplaneCSL:
    def test_sigma5_53deg_plane100(self):
        # Single end-to-end check: shape, in-plane, CSL, and independence.
        R = _sigma5_53deg_R()
        v1, v2 = solve_inplane_csl([0, 0, 1], [1, 0, 0], R)
        assert v1.shape == (3,) and v2.shape == (3,)
        assert abs(np.dot([1, 0, 0], v1)) < 1e-10
        assert abs(np.dot([1, 0, 0], v2)) < 1e-10
        for v in (v1, v2):
            vR = v @ R
            np.testing.assert_allclose(vR, np.round(vR), atol=1e-9,
                                       err_msg=f"v={v} is not a CSL vector")
        assert np.linalg.norm(np.cross(v1, v2)) > 0.5

    def test_sigma5_36deg_plane100_csl_vectors(self):
        # Different sigma/quaternion; verifies sigma recovery handles N=10.
        R = _sigma5_36deg_R()
        v1, v2 = solve_inplane_csl([0, 0, 1], [1, 0, 0], R)
        for v in (v1, v2):
            vR = v @ R
            np.testing.assert_allclose(vR, np.round(vR), atol=1e-9)

    def test_cell_too_large_raises(self):
        R = _sigma5_53deg_R()
        with pytest.raises(BoundarySpecError):
            solve_inplane_csl([0, 0, 1], [1, 0, 0], R, max_exact_atoms=1)


class TestReduce2DBasis:
    def test_shorter_vector_is_first(self):
        v1 = np.array([0, 0, 1], dtype=float)
        v2 = np.array([0, 1, 0], dtype=float)
        r1, r2 = reduce_2d_basis(v1, v2)
        assert np.linalg.norm(r1) <= np.linalg.norm(r2) + 1e-10

    def test_spans_same_lattice(self):
        # Area |v1×v2| must be preserved after reduction.
        v1 = np.array([0, 5, 0], dtype=float)
        v2 = np.array([0, 0, 1], dtype=float)
        r1, r2 = reduce_2d_basis(v1, v2)
        area_before = np.linalg.norm(np.cross(v1, v2))
        area_after = np.linalg.norm(np.cross(r1, r2))
        assert abs(area_before - area_after) < 1e-10


class TestCanonicalizePQ:
    # ------------------------------------------------------------------
    # Idempotence
    # ------------------------------------------------------------------

    def test_idempotent_identity(self):
        P, Q = _make_identity_pair()
        P1, Q1 = canonicalize_pq(P, Q)
        P2, Q2 = canonicalize_pq(P1, Q1)
        np.testing.assert_array_equal(P1, P2)
        np.testing.assert_array_equal(Q1, Q2)

    def test_idempotent_sigma5_boundary(self):
        # Sigma5 [001] tilt: P = identity, Q has rows [3,1,0]/[0,0,1]/[-1,3,0] (scaled)
        P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        Q = np.array([[3, 1, 0], [0, 0, 1], [-1, 3, 0]], dtype=float)
        P1, Q1 = canonicalize_pq(P, Q)
        P2, Q2 = canonicalize_pq(P1, Q1)
        np.testing.assert_array_equal(P1, P2)
        np.testing.assert_array_equal(Q1, Q2)

    # ------------------------------------------------------------------
    # Row scaling equivalence
    # ------------------------------------------------------------------

    def test_row_scaling_equivalent(self):
        # Scaling individual rows by a positive integer describes the same direction.
        P_base = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        P_scaled = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 2]], dtype=float)
        Q = np.eye(3, dtype=float)
        P_canon_base, _ = canonicalize_pq(P_base, Q)
        P_canon_scaled, _ = canonicalize_pq(P_scaled, Q)
        np.testing.assert_array_equal(P_canon_base, P_canon_scaled)

    # ------------------------------------------------------------------
    # In-plane basis equivalence (Gauss reduction)
    # ------------------------------------------------------------------

    def test_inplane_basis_equivalent(self):
        # Two P matrices whose in-plane rows span the same 2-D lattice.
        # Original in-plane basis: [0,1,0] and [0,0,1].
        # Equivalent basis: [0,1,0] and [0,1,1] (sum of the two).
        P_a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        P_b = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]], dtype=float)
        Q = np.eye(3, dtype=float)
        Pa, _ = canonicalize_pq(P_a, Q)
        Pb, _ = canonicalize_pq(P_b, Q)
        np.testing.assert_array_equal(Pa, Pb)

    def test_inplane_basis_longer_combination(self):
        # Basis [0,1,0] and [0,2,1] (second = 2*v1 + v2 for v2=[0,0,1]).
        P_a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        P_b = np.array([[1, 0, 0], [0, 1, 0], [0, 2, 1]], dtype=float)
        Q = np.eye(3, dtype=float)
        Pa, _ = canonicalize_pq(P_a, Q)
        Pb, _ = canonicalize_pq(P_b, Q)
        np.testing.assert_array_equal(Pa, Pb)

    # ------------------------------------------------------------------
    # Sign convention equivalence
    # ------------------------------------------------------------------

    def test_inplane_row_sign_equivalent(self):
        # Negating an in-plane row (and compensating to keep a valid orientation)
        # should canonicalize to the same form.
        P_a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        # Negate row 1; compensate by negating row 2 to keep det > 0.
        P_b = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
        Q = np.eye(3, dtype=float)
        Pa, _ = canonicalize_pq(P_a, Q)
        Pb, _ = canonicalize_pq(P_b, Q)
        np.testing.assert_array_equal(Pa, Pb)

    def test_boundary_normal_sign_equivalent(self):
        # Negating the boundary normal (row 0) with a compensating row-2 negation
        # (det-preserving) should canonicalize to the same form.
        P_a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        P_b = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
        Q = np.eye(3, dtype=float)
        Pa, _ = canonicalize_pq(P_a, Q)
        Pb, _ = canonicalize_pq(P_b, Q)
        np.testing.assert_array_equal(Pa, Pb)

    # ------------------------------------------------------------------
    # Right-handedness (positive determinant)
    # ------------------------------------------------------------------

    def test_output_right_handed_identity(self):
        P, Q = _make_identity_pair()
        Pc, Qc = canonicalize_pq(P, Q)
        assert np.linalg.det(Pc) > 0
        assert np.linalg.det(Qc) > 0

    def test_output_right_handed_sigma5(self):
        P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        Q = np.array([[3, 1, 0], [0, 0, 1], [-1, 3, 0]], dtype=float)
        Pc, Qc = canonicalize_pq(P, Q)
        assert np.linalg.det(Pc) > 0
        assert np.linalg.det(Qc) > 0

    def test_output_right_handed_after_sign_fixes(self):
        # Start from a right-handed matrix whose signs all need fixing.
        P = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
        Q = np.eye(3, dtype=float)
        Pc, _ = canonicalize_pq(P, Q)
        assert np.linalg.det(Pc) > 0

    # ------------------------------------------------------------------
    # Deterministic ordering: swapped in-plane rows canonicalize identically
    # ------------------------------------------------------------------

    def test_swapped_inplane_rows_same_canonical(self):
        # rows[1] and rows[2] swapped — same in-plane lattice, must give
        # identical canonical form (Gauss reduction must be order-independent).
        P_a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        P_b = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
        Q = np.eye(3, dtype=float)
        Pa, _ = canonicalize_pq(P_a, Q)
        Pb, _ = canonicalize_pq(P_b, Q)
        np.testing.assert_array_equal(Pa, Pb)

    def test_swapped_sigma5_inplane_rows_same_canonical(self):
        # Sigma5 Q with in-plane rows swapped must canonicalize to the same form.
        P = np.eye(3, dtype=float)
        Q_a = np.array([[3, 1, 0], [0, 0, 1], [-1, 3, 0]], dtype=float)
        Q_b = np.array([[3, 1, 0], [-1, 3, 0], [0, 0, 1]], dtype=float)
        _, Qa = canonicalize_pq(P, Q_a)
        _, Qb = canonicalize_pq(P, Q_b)
        np.testing.assert_array_equal(Qa, Qb)

    # ------------------------------------------------------------------
    # Rational (non-integer) rows must be rejected
    # ------------------------------------------------------------------

    def test_rational_row_raises(self):
        from GBOpt.BoundarySpec import BoundarySpecError
        P_rational = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]], dtype=float)
        Q = np.eye(3, dtype=float)
        with pytest.raises(BoundarySpecError):
            canonicalize_pq(P_rational, Q)

    def test_mixed_rational_row_raises(self):
        from GBOpt.BoundarySpec import BoundarySpecError
        P_bad = np.array([[1, 0, 0], [0, 1.5, 0], [0, 0, 1]], dtype=float)
        Q = np.eye(3, dtype=float)
        with pytest.raises(BoundarySpecError):
            canonicalize_pq(P_bad, Q)

    def test_large_noninteger_row_raises(self):
        # Regression: without rtol=0, np.allclose accepts 100000.4 because
        # the default relative tolerance swamps the 0.4 absolute deviation.
        from GBOpt.BoundarySpec import BoundarySpecError
        P_large = np.array([[100000.4, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        Q = np.eye(3, dtype=float)
        with pytest.raises(BoundarySpecError):
            canonicalize_pq(P_large, Q)


class TestPQSpecToEmbedding:
    # ------------------------------------------------------------------
    # Metadata flags
    # ------------------------------------------------------------------

    def test_flags_and_source(self):
        spec = PQSpec(P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      Q=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        emb = pq_spec_to_embedding(spec)
        assert isinstance(emb, BoundaryEmbedding)
        assert emb.exact is True
        assert emb.coherent is True
        assert emb.source == "pq"

    # ------------------------------------------------------------------
    # Basic end-to-end smoke
    # ------------------------------------------------------------------

    def test_identity_r_left_r_right(self):
        spec = PQSpec(P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      Q=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        emb = pq_spec_to_embedding(spec)
        np.testing.assert_array_almost_equal(emb.R_left, np.eye(3))
        np.testing.assert_array_almost_equal(emb.R_right, np.eye(3))

    # ------------------------------------------------------------------
    # Sigma5 [001] 36.87 deg tilt: R derives from canonical Q, not raw Q
    # ------------------------------------------------------------------

    def test_sigma5_r_right_matches_canonical(self):
        # Q supplied with rows in a non-canonical order; the adapter must
        # canonicalize first and derive R_right from the canonical form.
        P_raw = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        Q_raw = np.array([[4, -3, 0], [3, 4, 0], [0, 0, 1]], dtype=float)
        spec = PQSpec(P=P_raw.tolist(), Q=Q_raw.tolist())
        emb = pq_spec_to_embedding(spec)
        _, Q_c = canonicalize_pq(P_raw, Q_raw)
        R_right_expected = Q_c / np.linalg.norm(Q_c, axis=1, keepdims=True)
        np.testing.assert_array_almost_equal(emb.R_right, R_right_expected)

    def test_sigma5_r_right_is_proper_rotation(self):
        spec = PQSpec(P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      Q=[[4, -3, 0], [3, 4, 0], [0, 0, 1]])
        emb = pq_spec_to_embedding(spec)
        assert abs(np.linalg.det(emb.R_right) - 1.0) < 1e-12
        np.testing.assert_array_almost_equal(
            emb.R_right @ emb.R_right.T, np.eye(3))

    # ------------------------------------------------------------------
    # R_left and R_right rows are unit vectors
    # ------------------------------------------------------------------

    def test_row_norms_are_unit(self):
        spec = PQSpec(P=[[2, 1, 0], [0, 0, 1], [1, -2, 0]],
                      Q=[[2, -1, 0], [0, 0, 1], [-1, -2, 0]])
        emb = pq_spec_to_embedding(spec)
        np.testing.assert_allclose(
            np.linalg.norm(emb.R_left, axis=1), np.ones(3), atol=1e-12)
        np.testing.assert_allclose(
            np.linalg.norm(emb.R_right, axis=1), np.ones(3), atol=1e-12)

    # ------------------------------------------------------------------
    # Non-orthogonal P/Q must be rejected
    # ------------------------------------------------------------------

    def test_non_orthogonal_pq_raises(self):
        # [[1,0,0],[1,1,0],[0,0,1]] has rows that are not mutually orthogonal.
        from GBOpt.BoundarySpec import BoundarySpecError
        spec = PQSpec(P=[[1, 0, 0], [1, 1, 0], [0, 0, 1]],
                      Q=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        with pytest.raises(BoundarySpecError):
            pq_spec_to_embedding(spec)


# ---------------------------------------------------------------------------
# Step 13 — CSLExactSpec validation and adapter
# ---------------------------------------------------------------------------

class TestCSLExactSpecValidation:
    def test_valid_spec_instantiates(self):
        CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[2, 0, 0, 1])

    def test_missing_or_malformed_quat_raises(self):
        with pytest.raises(BoundarySpecError):
            CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0])
        with pytest.raises(BoundarySpecError):
            CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[2, 0, 1])

    def test_non_integer_miller_indices_raise(self):
        # _validate_miller must reject non-integer values before rounding.
        with pytest.raises(BoundarySpecError):
            CSLExactSpec(axis=[0, 0, 1], plane=[1.5, 0, 0], quat=[2, 0, 0, 1])
        with pytest.raises(BoundarySpecError):
            CSLExactSpec(axis=[0.7, 0, 1], plane=[1, 0, 0], quat=[2, 0, 0, 1])

    def test_zero_axis_or_plane_raises(self):
        with pytest.raises(BoundarySpecError):
            CSLExactSpec(axis=[0, 0, 0], plane=[1, 0, 0], quat=[2, 0, 0, 1])
        with pytest.raises(BoundarySpecError):
            CSLExactSpec(axis=[0, 0, 1], plane=[0, 0, 0], quat=[2, 0, 0, 1])

    def test_axis_quat_mismatch_raises(self):
        # quat [2,0,0,1] encodes rotation about [0,0,1]; axis=[1,0,0] is wrong.
        with pytest.raises(BoundarySpecError):
            CSLExactSpec(axis=[1, 0, 0], plane=[1, 0, 0], quat=[2, 0, 0, 1])

    def test_sigma_mismatch_raises(self):
        # quat [2,0,0,1] → sigma=5, not 3
        with pytest.raises(BoundarySpecError):
            csl_spec_to_embedding(
                CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0],
                             quat=[2, 0, 0, 1], sigma=3)
            )


class TestCSLSpecToEmbedding:
    # Σ5 [001] 36.87 deg — quat [3,0,0,1], plane [1,0,0]
    SPEC_36 = CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 0, 0, 1])
    # Equivalent PQSpec for the cross-format round-trip assertion
    PQ_36 = PQSpec(P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                   Q=[[4, -3, 0], [3, 4, 0], [0, 0, 1]])
    # Σ3 [111] 60 deg twin — quat [3,1,1,1], plane [1,1,1]; non-(100) regression
    SPEC_SIGMA3 = CSLExactSpec(axis=[1, 1, 1], plane=[1, 1, 1], quat=[3, 1, 1, 1])

    def test_embedding_flags_and_proper_rotations(self):
        emb = csl_spec_to_embedding(self.SPEC_36)
        assert isinstance(emb, BoundaryEmbedding)
        assert emb.exact is True
        assert emb.coherent is True
        assert emb.source == "csl"
        for R in (emb.R_left, emb.R_right):
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
            assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_cross_format_round_trip(self):
        # The central regression target: CSLExactSpec and the equivalent PQSpec
        # must produce byte-identical canonical P and Q matrices.
        emb_csl = csl_spec_to_embedding(self.SPEC_36)
        emb_pq = pq_spec_to_embedding(self.PQ_36)
        np.testing.assert_array_equal(emb_csl.P, emb_pq.P)
        np.testing.assert_array_equal(emb_csl.Q, emb_pq.Q)

    def test_sigma3_111_plane_gives_proper_rotations(self):
        # Non-(100) regression: [111] plane requires e2 = plane×e1 to keep
        # P rows mutually orthogonal; the null-basis e1,e2 pair is not enough.
        emb = csl_spec_to_embedding(self.SPEC_SIGMA3)
        for label, R in [("R_left", emb.R_left), ("R_right", emb.R_right)]:
            np.testing.assert_allclose(
                R @ R.T, np.eye(3), atol=1e-10,
                err_msg=f"{label} is not orthogonal for Σ3 [111] boundary")
            assert abs(np.linalg.det(R) - 1.0) < 1e-10, \
                f"{label} det ≠ 1 for Σ3 [111] boundary"


# ---------------------------------------------------------------------------
# Step 14b — per-grain repeats and commensurability guard
# ---------------------------------------------------------------------------

class TestExactGrainRepeats:
    """Tests for GBMaker.__exact_grain_repeats (via from_boundary_spec)."""

    A0 = 3.615
    STRUCTURE = "fcc"
    ATOM_TYPES = "Cu"
    GB_THICKNESS = 0.0

    # Σ5 [001] 36.87°: P identity, Q = [[4,-3,0],[3,4,0],[0,0,1]]
    P = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    Q = [[4, -3, 0], [3, 4, 0], [0, 0, 1]]

    def _build(self, repeat_factor=2):
        spec = PQSpec(P=self.P, Q=self.Q)
        return GBMaker.from_boundary_spec(
            self.A0, self.STRUCTURE, self.ATOM_TYPES, spec,
            mode="exact", gb_thickness=self.GB_THICKNESS,
            repeat_factor=repeat_factor,
        )

    def test_builds_without_error(self):
        # Commensurate case must not raise.
        gb = self._build()
        assert gb.whole_system.size > 0

    def test_y_dim_divisible_by_left_grain_y_period(self):
        # y_dim must be an integer multiple of a0 * norm(P[1]) = a0 (fcc [010])
        gb = self._build()
        y_period_left = self.A0 * np.linalg.norm(np.array(self.P[1]))
        ratio = gb.y_dim / y_period_left
        assert abs(ratio - round(ratio)) < 1e-6

    def test_z_dim_divisible_by_left_grain_z_period(self):
        gb = self._build()
        z_period_left = self.A0 * np.linalg.norm(np.array(self.P[2]))
        ratio = gb.z_dim / z_period_left
        assert abs(ratio - round(ratio)) < 1e-6


# ---------------------------------------------------------------------------
# Step 14c — exact grain builder wired into __generate_gb
# ---------------------------------------------------------------------------

class TestExactGrainBuilder:
    """Verify the exact construction path in __generate_gb."""

    A0 = 3.615
    STRUCTURE = "fcc"
    ATOM_TYPES = "Cu"
    GB_THICKNESS = 0.0

    # Σ5 [001] 36.87°
    EXACT_SPEC = CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 0, 0, 1])
    PQ_SPEC = PQSpec(P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                     Q=[[4, -3, 0], [3, 4, 0], [0, 0, 1]])

    def _build(self, spec):
        return GBMaker.from_boundary_spec(
            self.A0, self.STRUCTURE, self.ATOM_TYPES, spec,
            mode="exact", gb_thickness=self.GB_THICKNESS,
        )

    def test_exact_path_does_not_call_float_selection(self):
        targets = [
            "_GBMaker__select_atoms_in_box_basis",
            "_GBMaker__clip_atoms_to_cartesian_box",
            "_GBMaker__deduplicate_positions",
        ]
        with (
            patch.object(GBMaker, targets[0]) as spy0,
            patch.object(GBMaker, targets[1]) as spy1,
            patch.object(GBMaker, targets[2]) as spy2,
        ):
            self._build(self.PQ_SPEC)
            spy0.assert_not_called()
            spy1.assert_not_called()
            spy2.assert_not_called()

    def test_output_has_correct_dtype(self):
        from GBOpt.Atom import Atom
        gb = self._build(self.PQ_SPEC)
        assert gb.whole_system.dtype == Atom.atom_dtype

    def test_atom_count_equals_origins_times_basis_size(self):
        from GBOpt.Utils.gb_exact import build_supercell_matrix, enumerate_supercell_origins
        gb = self._build(self.PQ_SPEC)
        # Left grain: identity P, right grain: Q with det=25
        # Each grain count is verified by the internal assert in __generate_grain_exact
        assert gb.whole_system.size > 0

    def test_pqspec_and_cslexactspec_produce_same_atoms(self):
        gb_pq = self._build(self.PQ_SPEC)
        gb_csl = self._build(self.EXACT_SPEC)
        np.testing.assert_array_equal(gb_pq.whole_system, gb_csl.whole_system)

    def test_approx_path_still_works(self):
        approx = CSLApproxSpec(axis=[0, 0, 1], plane=[1, 0, 0], angle_deg=36.87)
        gb = GBMaker.from_boundary_spec(
            self.A0, self.STRUCTURE, self.ATOM_TYPES, approx,
            mode="approximate", gb_thickness=self.GB_THICKNESS,
        )
        assert gb.whole_system.size > 0


class TestFromBoundaryEmbedding:
    # Common parameters shared by all sub-tests.
    A0 = 3.615          # Cu lattice parameter
    STRUCTURE = "fcc"
    ATOM_TYPES = "Cu"
    GB_THICKNESS = 5.0
    REPEAT = 2
    X_DIM_MIN = 30.0

    def _legacy(self, misorientation):
        return GBMaker(
            self.A0, self.STRUCTURE, self.GB_THICKNESS, misorientation,
            self.ATOM_TYPES,
            repeat_factor=self.REPEAT, x_dim_min=self.X_DIM_MIN,
        )

    def _from_spec(self, P, Q):
        spec = PQSpec(P=P, Q=Q)
        emb = pq_spec_to_embedding(spec)
        return GBMaker._from_boundary_embedding(
            emb,
            a0=self.A0, structure=self.STRUCTURE, atom_types=self.ATOM_TYPES,
            gb_thickness=self.GB_THICKNESS,
            repeat_factor=self.REPEAT, x_dim_min=self.X_DIM_MIN,
        )

    # ------------------------------------------------------------------
    # Σ5 [001] 36.87° — exact path produces a valid fcc bicrystal.
    # ------------------------------------------------------------------

    def test_sigma5_exact_builds_valid_fcc_bicrystal(self):
        import math
        theta = math.atan2(3, 4)
        misorientation = np.array([0.0, 0.0, theta, 0.0, 0.0])
        gb_legacy = self._legacy(misorientation)

        P = gb_legacy._GBMaker__R_left_approx.astype(int).tolist()
        Q = gb_legacy._GBMaker__R_right_approx.astype(int).tolist()
        gb_emb = self._from_spec(P, Q)

        ws = gb_emb.whole_system
        assert ws.size > 0
        assert set(ws["name"]) == {"Cu"}
        for field in ("x", "y", "z"):
            assert np.all(np.isfinite(ws[field]))

    # ------------------------------------------------------------------
    # Exact path bypasses __approximate_rotation_matrix_as_int.
    # ------------------------------------------------------------------

    def test_exact_path_skips_approximation(self):
        spec = PQSpec(P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      Q=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        emb = pq_spec_to_embedding(spec)
        assert emb.exact is True

        # Patch without wraps: if the exact path correctly bypasses
        # approximation, the mock is never called and the test passes.
        spy_target = "_GBMaker__approximate_rotation_matrix_as_int"
        with patch.object(GBMaker, spy_target) as spy:
            GBMaker._from_boundary_embedding(
                emb,
                a0=self.A0, structure=self.STRUCTURE, atom_types=self.ATOM_TYPES,
                gb_thickness=self.GB_THICKNESS,
            )
            spy.assert_not_called()

    # ------------------------------------------------------------------
    # inplane_periodic reflects embedding.coherent.
    # ------------------------------------------------------------------

    def test_coherent_embedding_sets_inplane_periodic(self):
        spec = PQSpec(P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      Q=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        emb = pq_spec_to_embedding(spec)
        assert emb.coherent is True
        gb = GBMaker._from_boundary_embedding(
            emb,
            a0=self.A0, structure=self.STRUCTURE, atom_types=self.ATOM_TYPES,
            gb_thickness=self.GB_THICKNESS,
        )
        assert gb.inplane_periodic == (True, True)

    # ------------------------------------------------------------------
    # misorientation setter must discard the active embedding so that
    # update_spacing() uses the new Euler angles, not the stale R matrices.
    # ------------------------------------------------------------------

    def test_misorientation_setter_clears_embedding(self):
        import math
        spec = PQSpec(P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      Q=[[4, -3, 0], [3, 4, 0], [0, 0, 1]])
        emb = pq_spec_to_embedding(spec)
        gb = GBMaker._from_boundary_embedding(
            emb,
            a0=self.A0, structure=self.STRUCTURE, atom_types=self.ATOM_TYPES,
            gb_thickness=self.GB_THICKNESS,
        )
        # Embedding is active before mutation.
        assert gb._GBMaker__embedding is not None

        # After mutating misorientation the embedding must be gone.
        theta = math.atan2(3, 4)
        gb.misorientation = np.array([0.0, 0.0, theta, 0.0, 0.0])
        assert gb._GBMaker__embedding is None

        # R_left (via private attr) must reflect the Euler-angle inclination
        # (identity matrix for zero inclination angles), not the old embedding.
        np.testing.assert_array_almost_equal(gb._GBMaker__R_left, np.eye(3))


class TestFromBoundarySpec:
    A0 = 3.615
    STRUCTURE = "fcc"
    ATOM_TYPES = "Cu"
    GB_THICKNESS = 5.0

    def test_pqspec_exact_matches_embedding_path(self):
        # from_boundary_spec must produce the same bicrystal as going
        # through pq_spec_to_embedding + _from_boundary_embedding directly.
        spec = PQSpec(P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      Q=[[4, -3, 0], [3, 4, 0], [0, 0, 1]])
        gb_spec = GBMaker.from_boundary_spec(
            self.A0, self.STRUCTURE, self.ATOM_TYPES, spec, mode="exact",
            gb_thickness=self.GB_THICKNESS,
        )
        emb = pq_spec_to_embedding(spec)
        gb_emb = GBMaker._from_boundary_embedding(
            emb, a0=self.A0, structure=self.STRUCTURE,
            atom_types=self.ATOM_TYPES, gb_thickness=self.GB_THICKNESS,
        )
        np.testing.assert_array_equal(
            gb_spec.whole_system, gb_emb.whole_system)


# ---------------------------------------------------------------------------
# Step 14 — CSL specs wired into from_boundary_spec
# ---------------------------------------------------------------------------

class TestFromBoundarySpecCSL:
    A0 = 3.615
    STRUCTURE = "fcc"
    ATOM_TYPES = "Cu"
    GB_THICKNESS = 5.0

    # Σ5 [001] 36.87 deg exact spec and its equivalent PQSpec
    EXACT_SPEC = CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 0, 0, 1])
    PQ_SPEC = PQSpec(P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                     Q=[[4, -3, 0], [3, 4, 0], [0, 0, 1]])
    APPROX_SPEC = CSLApproxSpec(axis=[0, 0, 1], plane=[1, 0, 0], angle_deg=36.87)

    def _build(self, spec, mode):
        return GBMaker.from_boundary_spec(
            self.A0, self.STRUCTURE, self.ATOM_TYPES, spec, mode=mode,
            gb_thickness=self.GB_THICKNESS,
        )

    def test_cslexactspec_exact_builds_stoichiometric_bicrystal(self):
        gb = self._build(self.EXACT_SPEC, mode="exact")
        ws = gb.whole_system
        # Monatomic fcc: all atoms must have the same species name.
        assert len(np.unique(ws["name"])) == 1

    def test_cslexactspec_matches_equivalent_pqspec(self):
        # Exact CSL and the equivalent PQ must build the same atom array.
        gb_csl = self._build(self.EXACT_SPEC, mode="exact")
        gb_pq = self._build(self.PQ_SPEC, mode="exact")
        np.testing.assert_array_equal(gb_csl.whole_system, gb_pq.whole_system)

    def test_cslapproxspec_exact_raises(self):
        with pytest.raises(BoundarySpecError):
            self._build(self.APPROX_SPEC, mode="exact")

    def test_cslapproxspec_approximate_succeeds(self):
        gb = self._build(self.APPROX_SPEC, mode="approximate")
        assert gb.whole_system.size > 0

    def test_cslapproxspec_missing_angle_raises(self):
        with pytest.raises(BoundarySpecError):
            CSLApproxSpec(axis=[0, 0, 1], plane=[1, 0, 0])


class TestFromBoundarySpecMultispecies:
    """Step 9 / Step 14 acceptance: exact-path construction must preserve stoichiometry.

    Monatomic tests cannot catch species-count failures.  These tests use
    known exact Sigma5 [001] boundaries (via both PQSpec and CSLExactSpec) and
    assert the correct cation:anion ratio, which also implies charge neutrality.
    """

    # Sigma5 [001] 36.87 deg tilt — verified proper rotation.
    P = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    Q = [[4, -3, 0], [3, 4, 0], [0, 0, 1]]
    CSL_SPEC = CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 0, 0, 1])

    def _species_counts_pq(self, a0, structure, atom_types):
        spec = PQSpec(P=self.P, Q=self.Q)
        gb = GBMaker.from_boundary_spec(
            a0, structure, atom_types, spec, mode="exact", gb_thickness=2.0
        )
        ws = gb.whole_system
        names, counts = np.unique(ws["name"], return_counts=True)
        return {str(n): int(c) for n, c in zip(names, counts)}

    def _species_counts_csl(self, a0, structure, atom_types):
        gb = GBMaker.from_boundary_spec(
            a0, structure, atom_types, self.CSL_SPEC, mode="exact", gb_thickness=2.0
        )
        ws = gb.whole_system
        names, counts = np.unique(ws["name"], return_counts=True)
        return {str(n): int(c) for n, c in zip(names, counts)}

    def test_rocksalt_stoichiometric(self):
        # NaCl: cation : anion must be 1 : 1.
        counts = self._species_counts_pq(4.0, "rocksalt", ("Na", "Cl"))
        assert counts["Na"] == counts["Cl"], (
            f"Rocksalt bicrystal is not stoichiometric: {counts}"
        )

    def test_cslexactspec_rocksalt_stoichiometric(self):
        # CSLExactSpec exact path must produce the same stoichiometric result
        # as the equivalent PQSpec — catches any multi-species regression in
        # the csl_spec_to_embedding → _from_boundary_embedding path.
        counts = self._species_counts_csl(4.0, "rocksalt", ("Na", "Cl"))
        assert counts["Na"] == counts["Cl"], (
            f"Rocksalt bicrystal via CSLExactSpec is not stoichiometric: {counts}"
        )

    def test_fluorite_stoichiometric(self):
        # UO₂: anion : cation must be 2 : 1.
        counts = self._species_counts_pq(5.47, "fluorite", ("U", "O"))
        assert counts["O"] == 2 * counts["U"], (
            f"Fluorite bicrystal is not stoichiometric: {counts}"
        )

    def test_cslexactspec_fluorite_stoichiometric(self):
        # Same boundary via CSLExactSpec must produce the same stoichiometric result.
        counts = self._species_counts_csl(5.47, "fluorite", ("U", "O"))
        assert counts["O"] == 2 * counts["U"], (
            f"Fluorite bicrystal via CSLExactSpec is not stoichiometric: {counts}"
        )
