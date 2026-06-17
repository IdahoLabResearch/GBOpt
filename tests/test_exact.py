# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import warnings

import numpy as np
import pytest

from unittest.mock import patch

from GBOpt.BoundarySpec import (
    BoundaryEmbedding,
    BoundarySpecError,
    BoundarySpecOrthogonalityError,
    CSLApproxSpec,
    CSLExactSpec,
    PQSpec,
)
from GBOpt.GBMaker import GBMaker
from GBOpt.Utils.exact import (
    ExactCSLValueError,
    csl_from_scaled_rotation,
    dsc_basis,
    inplane_basis_from_csl,
    lll_reduce,
    normalize_integer_quaternion,
    pq_from_csl_plane,
    quaternion_to_scaled_rotation,
    verify_coincidence_basis,
)
from GBOpt.Utils.exact import (
    _canonicalize_pq_paired,
    _dot_int,
    _gauss_reduce_2d_paired,
    _inplane_area_index,
    _int_adj3,
    _int_det3,
    _integer_membership,
    _plane_null_basis,
    _primitive_metadata,
    _recover_row_rotation_from_pq,
    _row_gcd_reduce,
    build_supercell_matrix,
    canonicalize_pq,
    csl_spec_to_embedding,
    enumerate_supercell_origins,
    primitive_bicrystal_atom_count,
    pq_spec_to_embedding,
    quaternion_to_rotation_matrix,
    reduce_2d_basis,
    solve_inplane_csl,
    validate_and_normalize_quaternion,
    validate_sigma,
)

from GBOpt.Utils.integer_normal_forms import (
    _det3,
    column_hnf_3x3,
    hnf_2d_supercells,
    smith_normal_form_3x3,
)


# ---------------------------------------------------------------------------
# Integer membership kernel and supercell matrix
# ---------------------------------------------------------------------------

INVALID_INTEGER_MATRICES = [
    [[1.1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [["not-an-int", 0, 0], [0, 1, 0], [0, 0, 1]],
    [[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[1, 0], [0, 1]],
]


class TestIntDet3:
    def test_identity(self):
        assert _int_det3([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == 1

    def test_known_positive(self):
        assert _int_det3([[1, 0, 0], [0, 2, 0], [0, 0, 3]]) == 6

    def test_known_negative(self):
        assert _int_det3([[0, 1, 0], [1, 0, 0], [0, 0, 1]]) == -1

    def test_sigma5_right_S(self):
        # Sigma5 36.87 deg right grain: Q = [[4,-3,0],[3,4,0],[0,0,1]], det = 25
        assert _int_det3([[4, -3, 0], [3, 4, 0], [0, 0, 1]]) == 25

    @pytest.mark.parametrize("matrix", INVALID_INTEGER_MATRICES)
    def test_invalid_input_raises(self, matrix):
        with pytest.raises(ValueError):
            _int_det3(matrix)


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

    @pytest.mark.parametrize("matrix", INVALID_INTEGER_MATRICES)
    def test_invalid_input_raises(self, matrix):
        with pytest.raises(ValueError):
            _int_adj3(matrix)


class TestIntegerMembership:
    # Use Sigma5 right-grain matrix: det=25, so each unit cell maps to 25 cosets.
    _Q = [[4, -3, 0], [3, 4, 0], [0, 0, 1]]

    def setup_method(self):
        self.det_S = _int_det3(self._Q)        # 25
        self.adj_S = _int_adj3(self._Q)

    def test_origin_always_accepted(self):
        assert _integer_membership([0, 0, 0], self.adj_S, self.det_S, 1, 1, 1)

    def test_count_via_identity_S(self):
        # Identity S, repeat 3x2x1: only origins in [0,3)x[0,2)x[0,1) accepted
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
        # repeat_z=1 rejects origins whose z numerator lands exactly on the
        # exclusive upper bound.
        assert not _integer_membership([0, 0, 1], self.adj_S, self.det_S, 1, 1, 1)

    def test_negative_det_origin_sign_normalization_accepts_origin(self):
        # Exact production supercells are positive-determinant, but the helper
        # normalizes signs for general integer matrices.
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

    @pytest.mark.parametrize("P_bad", [
        np.array([[1.1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        np.array([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        np.array([[1, 0], [0, 1]], dtype=float),
    ])
    def test_invalid_input_raises(self, P_bad):
        with pytest.raises((BoundarySpecError, ValueError)):
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
# Integer-quaternion validation and rotation matrix
# ---------------------------------------------------------------------------

class TestValidateAndNormalizeQuaternion:
    # Sigma5 [001] 53.13 deg -- integer quaternion [2, 0, 0, 1], N = 5
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

    @pytest.mark.parametrize("quat", [
        [1, 0, 0],
        [[1, 0, 0, 0]],
        np.ones((2, 2)),
        np.ones((1, 4, 1)),
    ])
    def test_wrong_shape_raises(self, quat):
        with pytest.raises(BoundarySpecError):
            validate_and_normalize_quaternion(quat)

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
    # Sigma5 [001] 53.13 deg -- quat [2, 0, 0, 1] (Hamilton [w,x,y,z]),
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
        # [w=1, x=0, y=0, z=0] -> identity rotation
        q_id = np.array([1.0, 0.0, 0.0, 0.0])
        R = quaternion_to_rotation_matrix(q_id)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_sigma5_36deg_quat(self):
        # Sigma5 [001] 36.87 deg -- quat [3, 0, 0, 1], N = 10
        # expected R = [[4/5, -3/5, 0], [3/5, 4/5, 0], [0, 0, 1]]
        q = np.array([3, 0, 0, 1], dtype=float) / np.sqrt(10)
        R = quaternion_to_rotation_matrix(q)
        expected = np.array([
            [4/5, -3/5, 0],
            [3/5,  4/5, 0],
            [0,    0,   1],
        ])
        np.testing.assert_allclose(R, expected, atol=1e-12)

    @pytest.mark.parametrize("quat", [
        [1, 0, 0],
        [[1, 0, 0, 0]],
        np.ones((2, 2)),
    ])
    def test_invalid_shape_raises(self, quat):
        with pytest.raises(BoundarySpecError):
            quaternion_to_rotation_matrix(quat)

    @pytest.mark.parametrize("quat", [
        [2, 0, 0, 1],
        [np.nan, 0, 0, 1],
    ])
    def test_non_unit_quaternion_raises(self, quat):
        with pytest.raises(BoundarySpecError, match="normalized"):
            quaternion_to_rotation_matrix(quat)


# ---------------------------------------------------------------------------
# Sigma-from-quaternion validation
# ---------------------------------------------------------------------------

class TestValidateSigma:
    # Sigma5: quat [2, 0, 0, 1], |q|^2 = 5 (already odd) -> sigma = 5
    # Sigma5: quat [3, 0, 0, 1], |q|^2 = 10 = 2x5 -> sigma = 5
    # Sigma13: quat [3, 2, 0, 0], |q|^2 = 13 -> sigma = 13

    def test_sigma5_quat_2001_correct(self):
        validate_sigma([2, 0, 0, 1], 5)  # must not raise

    def test_sigma5_quat_3001_correct(self):
        # |q|^2 = 10, divided by 2 once -> sigma = 5
        validate_sigma([3, 0, 0, 1], 5)  # must not raise

    def test_sigma13_correct(self):
        # |q|^2 = 9+4 = 13 (odd) -> sigma = 13
        validate_sigma([3, 2, 0, 0], 13)  # must not raise

    @pytest.mark.parametrize("wrong_sigma", [3, 4])
    def test_wrong_sigma_raises(self, wrong_sigma):
        with pytest.raises(BoundarySpecError):
            validate_sigma([2, 0, 0, 1], wrong_sigma)

    def test_power_of_two_stripped_correctly(self):
        # quat [2, 2, 0, 0]: |q|^2 = 8 = 2^3 -> sigma = 1
        validate_sigma([2, 2, 0, 0], 1)  # must not raise; would raise if sigma=8

    def test_power_of_two_stripped_wrong_sigma_raises(self):
        with pytest.raises(BoundarySpecError):
            validate_sigma([2, 2, 0, 0], 8)  # sigma=1, not 8

    def test_zero_quaternion_raises(self):
        with pytest.raises(BoundarySpecError):
            validate_sigma([0, 0, 0, 0], 1)

    @pytest.mark.parametrize("quat", [
        (2, 0, 0, 1),
        (3, 0, 0, 1),
        (1, 1, 1, 1),
        (1, 2, 3, 4),
    ])
    def test_matches_csl_derived_sigma(self, quat):
        rot = quaternion_to_scaled_rotation(quat)
        csl = csl_from_scaled_rotation(rot)
        validate_sigma(quat, csl.sigma)
        with pytest.raises(BoundarySpecError):
            validate_sigma(quat, csl.sigma + 1)


# ---------------------------------------------------------------------------
# In-plane CSL solve and 2D reduction
# ---------------------------------------------------------------------------

def _sigma5_53deg_R():
    """R for Sigma5 [001] 53.13 deg boundary (quat=[2,0,0,1], N=5)."""
    q = np.array([2, 0, 0, 1], dtype=float) / np.sqrt(5)
    return quaternion_to_rotation_matrix(q)


def _sigma5_36deg_R():
    """R for Sigma5 [001] 36.87 deg boundary (quat=[3,0,0,1], N=10)."""
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

    def test_sigma5_36deg_plane523_primitive_basis(self):
        # General planes must use the full primitive plane lattice, not a
        # strict sublattice, so the in-plane CSL area stays minimal.
        R = _sigma5_36deg_R()
        v1, v2 = solve_inplane_csl(
            [0, 0, 1], [5, 2, 3], R, max_exact_atoms=100
        )
        area = np.linalg.norm(np.cross(v1, v2))
        assert area < 40.0, (
            f"CSL cell area ({area:.4f}) too large; non-primitive null basis suspected"
        )
        # Both returned vectors must lie in the plane [5,2,3]
        for v in (v1, v2):
            assert abs(np.dot([5, 2, 3], v)) < 1e-9, f"{v} not in-plane for [5,2,3]"

    def test_sigma5_36deg_plane210_primitive_basis(self):
        # The primitive [2,1,0] plane lattice contains [0,0,1], which is also a
        # CSL vector for the [001] rotation.
        R = _sigma5_36deg_R()
        v1, v2 = solve_inplane_csl([0, 0, 1], [2, 1, 0], R)
        r1, _r2 = reduce_2d_basis(v1, v2)
        # Shortest in-plane CSL vector must be primitive (length 1, i.e. [0,0,+/-1])
        assert abs(np.linalg.norm(r1) - 1.0) < 1e-9, (
            f"Expected primitive in-plane CSL vector of length 1, got {r1} "
            f"(norm={np.linalg.norm(r1):.4f})"
        )
        area = np.linalg.norm(np.cross(v1, v2))
        assert area < 15.0, (
            f"CSL cell area ({area:.4f}) too large; non-primitive basis suspected"
        )
        v1b, v2b = solve_inplane_csl(
            [0, 0, 1], [2, 1, 0], R, max_exact_atoms=20
        )
        assert np.linalg.norm(np.cross(v1b, v2b)) < 15.0


class TestReduce2DBasis:
    def test_shorter_vector_is_first(self):
        v1 = np.array([0, 5, 0], dtype=float)
        v2 = np.array([0, 0, 1], dtype=float)
        r1, r2 = reduce_2d_basis(v1, v2)
        assert np.linalg.norm(r1) <= np.linalg.norm(r2) + 1e-10

    def test_spans_same_lattice(self):
        # Area |v1xv2| must be preserved after reduction.
        v1 = np.array([0, 5, 0], dtype=float)
        v2 = np.array([0, 0, 1], dtype=float)
        r1, r2 = reduce_2d_basis(v1, v2)
        area_before = np.linalg.norm(np.cross(v1, v2))
        area_after = np.linalg.norm(np.cross(r1, r2))
        assert abs(area_before - area_after) < 1e-10

    def test_paired_reduction_iteration_limit_warns_and_returns_rows(self):
        # Force the loop to exhaust without a break so the warning branch is
        # covered without constructing enormous worst-case integer inputs.
        with patch("GBOpt.Utils.exact.range", return_value=[0], create=True):
            with pytest.warns(UserWarning, match="Paired Gauss reduction"):
                p1, p2, q1, q2 = _gauss_reduce_2d_paired(
                    np.array([1, 0, 0]),
                    np.array([2, 1, 0]),
                    np.array([0, 1, 0]),
                    np.array([0, 0, 1]),
                )

        np.testing.assert_array_equal(p1, np.array([1, 0, 0]))
        np.testing.assert_array_equal(p2, np.array([0, 1, 0]))
        np.testing.assert_array_equal(q1, np.array([0, 1, 0]))
        np.testing.assert_array_equal(q2, np.array([0, -2, 1]))


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
        # rows[1] and rows[2] swapped -- same in-plane lattice, must give
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
        # Use rtol=0 so the absolute tolerance controls large entries.
        from GBOpt.BoundarySpec import BoundarySpecError
        P_large = np.array([[100000.4, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        Q = np.eye(3, dtype=float)
        with pytest.raises(BoundarySpecError):
            canonicalize_pq(P_large, Q)


class TestCanonicalizePQPaired:
    """Tests for _canonicalize_pq_paired robustness and consistency."""

    def test_nonprimitive_inplane_rows_are_reduced(self):
        # Non-primitive in-plane rows must be GCD-reduced to match
        # _canonicalize_matrix behavior; [2,0,0] -> [1,0,0].
        import math
        P = np.array([[1, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=float)
        Q = np.array([[1, 0, 0], [4, 0, 0], [0, 6, 0]], dtype=float)
        P_c, Q_c = _canonicalize_pq_paired(P, Q)
        for M, name in [(P_c, "P"), (Q_c, "Q")]:
            for i, row in enumerate(M):
                ints = np.round(row).astype(int)
                gcd = 0
                for v in ints:
                    gcd = math.gcd(gcd, abs(int(v)))
                assert gcd == 1, f"{name} row {i} {row} has GCD {gcd}, expected 1"

    def test_idempotent(self):
        # Canonicalization must be idempotent: applying it twice gives the same result.
        P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        Q = np.array([[3, 1, 0], [0, 0, 1], [1, -3, 0]], dtype=float)
        P1, Q1 = _canonicalize_pq_paired(P, Q)
        P2, Q2 = _canonicalize_pq_paired(P1, Q1)
        np.testing.assert_array_equal(P1, P2)
        np.testing.assert_array_equal(Q1, Q2)

    def test_scaled_equivalent_inplane_rows_canonicalize_identically(self):
        # P with in-plane rows [1,0,0]/[0,1,0] and [2,0,0]/[0,3,0] must
        # produce the same canonical P (after GCD reduction).
        P_a = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        P_b = np.array([[1, 0, 0], [2, 0, 0], [0, 3, 0]], dtype=float)
        Q = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        Pa, _ = _canonicalize_pq_paired(P_a, Q)
        Pb, _ = _canonicalize_pq_paired(P_b, Q)
        np.testing.assert_array_equal(Pa, Pb)


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


class TestPQSpecPrimitiveBasis:
    LEGACY_TWIST_P = [[0, 0, 1], [3, 1, 0], [-1, 3, 0]]
    LEGACY_TWIST_Q = [[0, 0, 1], [3, -1, 0], [1, 3, 0]]
    PRIMITIVE_TWIST_P = [[0, 0, 1], [1, 2, 0], [-2, 1, 0]]
    PRIMITIVE_TWIST_Q = [[0, 0, 1], [2, 1, 0], [-1, 2, 0]]
    TILT_210_P = [[2, 1, 0], [0, 0, 1], [1, -2, 0]]
    TILT_210_Q = [[2, -1, 0], [0, 0, 1], [-1, -2, 0]]
    TILT_310_P = [[3, 1, 0], [0, 0, 1], [1, -3, 0]]
    TILT_310_Q = [[3, -1, 0], [0, 0, 1], [-1, -3, 0]]

    def test_legacy_sigma5_twist_defaults_to_primitive_area_index_5(self):
        emb = pq_spec_to_embedding(
            PQSpec(P=self.LEGACY_TWIST_P, Q=self.LEGACY_TWIST_Q)
        )

        assert _int_det3(emb.P) == 5
        assert _int_det3(emb.Q) == 5
        assert emb.metadata is not None
        assert emb.metadata.basis_mode == "primitive"
        assert emb.metadata.supplied_area_index == 10
        assert emb.metadata.primitive_area_index == 5
        assert emb.metadata.reduction_index == 2

    def test_primitive_and_legacy_sigma5_twist_match_by_default(self):
        emb_legacy = pq_spec_to_embedding(
            PQSpec(P=self.LEGACY_TWIST_P, Q=self.LEGACY_TWIST_Q)
        )
        emb_primitive = pq_spec_to_embedding(
            PQSpec(P=self.PRIMITIVE_TWIST_P, Q=self.PRIMITIVE_TWIST_Q)
        )

        np.testing.assert_array_equal(emb_legacy.P, emb_primitive.P)
        np.testing.assert_array_equal(emb_legacy.Q, emb_primitive.Q)
        np.testing.assert_allclose(
            emb_legacy.R_left, emb_primitive.R_left, atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(
            emb_legacy.R_right, emb_primitive.R_right, atol=1e-12, rtol=0
        )

    @pytest.mark.parametrize("P,Q,expected_plane", [
        (TILT_210_P, TILT_210_Q, (2, 1, 0)),
        (TILT_310_P, TILT_310_Q, (3, 1, 0)),
    ])
    def test_sigma5_tilt_boundaries_preserve_boundary_plane(
        self, P, Q, expected_plane
    ):
        emb = pq_spec_to_embedding(PQSpec(P=P, Q=Q))

        assert emb.metadata is not None
        assert emb.metadata.plane == expected_plane
        np.testing.assert_array_equal(emb.P[0].astype(int), expected_plane)

    def test_supplied_basis_mode_preserves_area_index_10(self):
        emb = pq_spec_to_embedding(
            PQSpec(
                P=self.LEGACY_TWIST_P,
                Q=self.LEGACY_TWIST_Q,
                basis_mode="supplied",
            )
        )

        assert _int_det3(emb.P) == 10
        assert _int_det3(emb.Q) == 10
        assert emb.metadata is not None
        assert emb.metadata.basis_mode == "supplied"
        assert emb.metadata.supplied_area_index == 10
        assert emb.metadata.primitive_area_index == 10

    def test_supplied_basis_mode_preserves_paired_row_rotation(self):
        transform = np.array([[1, 1], [0, 1]], dtype=int)
        P = np.array(self.PRIMITIVE_TWIST_P, dtype=float)
        Q = np.array(self.PRIMITIVE_TWIST_Q, dtype=float)
        P_transformed = P.copy()
        Q_transformed = Q.copy()
        P_transformed[1:] = transform @ P[1:]
        Q_transformed[1:] = transform @ Q[1:]

        expected_rotation = _recover_row_rotation_from_pq(
            P_transformed, Q_transformed
        )
        emb = pq_spec_to_embedding(
            PQSpec(
                P=P_transformed.tolist(),
                Q=Q_transformed.tolist(),
                basis_mode="supplied",
            )
        )
        actual_rotation = _recover_row_rotation_from_pq(emb.P, emb.Q)

        assert actual_rotation.N == expected_rotation.N
        np.testing.assert_array_equal(actual_rotation.M, expected_rotation.M)

    def test_primitive_mode_warns_when_rotation_recovery_falls_back(self):
        spec = PQSpec(
            P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            Q=[[4, -3, 0], [3, 4, 0], [0, 0, 1]],
        )

        with pytest.warns(UserWarning, match="basis_mode='primitive'.*falling back"):
            emb = pq_spec_to_embedding(spec)

        assert emb.metadata is None
        assert _int_det3(emb.Q) == 25

    def test_primitive_mode_warns_when_embedding_reconstruction_falls_back(self):
        spec = PQSpec(
            P=self.PRIMITIVE_TWIST_P,
            Q=self.PRIMITIVE_TWIST_Q,
        )

        with patch(
            "GBOpt.Utils.exact._primitive_embedding_from_row_rotation",
            side_effect=BoundarySpecError("forced primitive failure"),
        ):
            with pytest.warns(UserWarning, match="forced primitive failure"):
                emb = pq_spec_to_embedding(spec)

        assert emb.metadata is not None
        assert emb.metadata.basis_mode == "supplied"


# ---------------------------------------------------------------------------
# CSLExactSpec validation and adapter
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
        # quat [2,0,0,1] -> sigma=5, not 3
        with pytest.raises(BoundarySpecError):
            csl_spec_to_embedding(
                CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0],
                             quat=[2, 0, 0, 1], sigma=3)
            )


class TestCSLSpecToEmbedding:
    # Sigma5 [001] 36.87 deg -- quat [3,0,0,1], plane [1,0,0]
    SPEC_36 = CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 0, 0, 1])
    SPEC_TWIST = CSLExactSpec(axis=[0, 0, 1], plane=[0, 0, 1], quat=[3, 0, 0, 1])
    # Equivalent PQSpec for the cross-format round-trip assertion
    PQ_36 = PQSpec(P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                   Q=[[4, -3, 0], [3, 4, 0], [0, 0, 1]])
    PQ_TWIST = PQSpec(P=[[0, 0, 1], [3, 1, 0], [-1, 3, 0]],
                      Q=[[0, 0, 1], [3, -1, 0], [1, 3, 0]])
    # Sigma3 [111] 60 deg twin -- quat [3,1,1,1], plane [1,1,1].
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
        # CSLExactSpec and the equivalent PQSpec
        # must produce byte-identical canonical P and Q matrices.
        emb_csl = csl_spec_to_embedding(self.SPEC_36)
        emb_pq = pq_spec_to_embedding(self.PQ_36)
        np.testing.assert_array_equal(emb_csl.P, emb_pq.P)
        np.testing.assert_array_equal(emb_csl.Q, emb_pq.Q)

    def test_non_preserving_plane_fallback_builds_embedding(self):
        # The [100] plane is not invariant under a [001] Sigma5 rotation, so
        # this exercises the fallback path that builds P/Q after an in-plane
        # CSL solve.
        emb = csl_spec_to_embedding(self.SPEC_36)
        assert emb.exact is True
        assert emb.coherent is True

    def test_sigma3_111_plane_gives_proper_rotations(self):
        # The [111] plane requires e2 = planexe1 to keep
        # P rows mutually orthogonal; the null-basis e1,e2 pair is not enough.
        emb = csl_spec_to_embedding(self.SPEC_SIGMA3)
        for label, R in [("R_left", emb.R_left), ("R_right", emb.R_right)]:
            np.testing.assert_allclose(
                R @ R.T, np.eye(3), atol=1e-10,
                err_msg=f"{label} is not orthogonal for Sigma3 [111] boundary")
            assert abs(np.linalg.det(R) - 1.0) < 1e-10, \
                f"{label} det != 1 for Sigma3 [111] boundary"

    def test_orthogonality_error_falls_back_to_orthogonal_embedding(self):
        with patch(
            "GBOpt.Utils.exact._primitive_embedding_from_row_rotation",
            side_effect=BoundarySpecOrthogonalityError("forced orthogonality failure"),
        ) as primitive_builder:
            emb = csl_spec_to_embedding(self.SPEC_TWIST)

        primitive_builder.assert_called_once()
        assert emb.exact is True
        assert emb.coherent is True
        for R in (emb.R_left, emb.R_right):
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10, rtol=0)
            assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_twist_matches_equivalent_primitive_pqspec(self):
        emb_csl = csl_spec_to_embedding(self.SPEC_TWIST)
        emb_pq = pq_spec_to_embedding(self.PQ_TWIST)

        np.testing.assert_array_equal(emb_csl.P, emb_pq.P)
        np.testing.assert_array_equal(emb_csl.Q, emb_pq.Q)
        assert _int_det3(emb_csl.P) == 5
        assert _int_det3(emb_csl.Q) == 5

    def test_twist_rotations_derive_from_final_paired_pq(self):
        emb = csl_spec_to_embedding(self.SPEC_TWIST)
        expected_left = emb.P / np.linalg.norm(emb.P, axis=1, keepdims=True)
        expected_right = emb.Q / np.linalg.norm(emb.Q, axis=1, keepdims=True)

        np.testing.assert_allclose(emb.R_left, expected_left, atol=1e-12, rtol=0)
        np.testing.assert_allclose(emb.R_right, expected_right, atol=1e-12, rtol=0)
        assert not np.allclose(emb.R_left, emb.R_right, atol=1e-12, rtol=0)
        for label, R in [("R_left", emb.R_left), ("R_right", emb.R_right)]:
            np.testing.assert_allclose(
                R @ R.T,
                np.eye(3),
                atol=1e-10,
                rtol=0,
                err_msg=f"{label} is not orthogonal for primitive Sigma5 twist",
            )
            assert abs(np.linalg.det(R) - 1.0) < 1e-10
        assert emb.metadata is not None
        assert emb.metadata.basis_mode == "primitive"
        assert emb.metadata.primitive_area_index == 5


class TestPrimitiveCellReporting:
    TWIST_SPEC = CSLExactSpec(axis=[0, 0, 1], plane=[0, 0, 1], quat=[3, 0, 0, 1])

    def test_sigma5_fluorite_primitive_bicrystal_atom_count(self):
        emb = csl_spec_to_embedding(self.TWIST_SPEC)

        assert emb.metadata is not None
        conventional_cells = emb.metadata.conventional_cell_multiplier
        species_counts = {
            "U": conventional_cells * 4,
            "O": conventional_cells * 8,
        }

        assert primitive_bicrystal_atom_count(emb, 12) == 120
        assert species_counts == {"U": 40, "O": 80}
        assert species_counts["O"] == 2 * species_counts["U"]

    def test_reporting_requires_metadata(self):
        emb = BoundaryEmbedding(
            P=None,
            Q=None,
            R_left=np.eye(3),
            R_right=np.eye(3),
            exact=False,
            coherent=False,
            source="five_dof",
        )

        with pytest.raises(BoundarySpecError):
            primitive_bicrystal_atom_count(emb, 12)

    def test_primitive_metadata_requires_divisible_area_indices(self):
        # The supplied area index cannot be reduced to the primitive area by an
        # integer factor, so the metadata would report an inconsistent cell.
        with pytest.raises(BoundarySpecError, match="integer multiple"):
            _primitive_metadata(
                basis_mode="primitive",
                supplied_area_index=7,
                primitive_area_index=5,
                plane=np.array([0, 0, 1]),
                rotation_denominator=10,
            )

    def test_primitive_metadata_does_not_shrink_expanded_gbmaker_cell(self):
        emb = csl_spec_to_embedding(self.TWIST_SPEC)
        primitive_atoms = primitive_bicrystal_atom_count(emb, 12)
        a0 = 5.47
        repeat_factor = [2, 3]
        x_dim_min = 30.0
        interaction_distance = 11.0

        gb = GBMaker.from_boundary_spec(
            a0,
            "fluorite",
            ("U", "O"),
            self.TWIST_SPEC,
            mode="exact",
            gb_thickness=0.0,
            repeat_factor=repeat_factor,
            x_dim_min=x_dim_min,
            vacuum=0.0,
            interaction_distance=interaction_distance,
        )

        y_period = a0 * np.linalg.norm(emb.P[1])
        z_period = a0 * np.linalg.norm(emb.P[2])
        assert gb.whole_system.size > primitive_atoms
        assert gb._GBMaker__left_x >= x_dim_min - 1e-9
        assert gb._GBMaker__right_x >= x_dim_min - 1e-9
        assert gb.y_dim >= repeat_factor[0] * y_period - 1e-9
        assert gb.z_dim >= repeat_factor[1] * z_period - 1e-9
        assert gb.y_dim >= 2.0 * interaction_distance - 1e-9
        assert gb.z_dim >= 2.0 * interaction_distance - 1e-9


# ---------------------------------------------------------------------------
# Per-grain repeats and commensurability guard
# ---------------------------------------------------------------------------

class TestExactGrainRepeats:
    """Tests for GBMaker.__exact_grain_repeats (via from_boundary_spec)."""

    A0 = 3.615
    STRUCTURE = "fcc"
    ATOM_TYPES = "Cu"
    GB_THICKNESS = 0.0

    # Sigma5 [001] 36.87 deg: P identity, Q = [[4,-3,0],[3,4,0],[0,0,1]]
    P = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    Q = [[4, -3, 0], [3, 4, 0], [0, 0, 1]]

    def _build(self, repeat_factor=2):
        spec = PQSpec(P=self.P, Q=self.Q)
        return GBMaker.from_boundary_spec(
            self.A0, self.STRUCTURE, self.ATOM_TYPES, spec,
            mode="exact", gb_thickness=self.GB_THICKNESS,
            repeat_factor=repeat_factor,
        )

    def test_builds_and_dims_are_divisible_by_left_grain_periods(self):
        gb = self._build()
        assert gb.whole_system.size > 0

        y_period_left = self.A0 * np.linalg.norm(np.array(self.P[1]))
        z_period_left = self.A0 * np.linalg.norm(np.array(self.P[2]))
        y_ratio = gb.y_dim / y_period_left
        z_ratio = gb.z_dim / z_period_left
        assert abs(y_ratio - round(y_ratio)) < 1e-6
        assert abs(z_ratio - round(z_ratio)) < 1e-6


# ---------------------------------------------------------------------------
# Exact grain builder wired into __generate_gb
# ---------------------------------------------------------------------------

class TestExactGrainBuilder:
    """Verify the exact construction path in __generate_gb."""

    A0 = 3.615
    STRUCTURE = "fcc"
    ATOM_TYPES = "Cu"
    GB_THICKNESS = 0.0

    # Sigma5 [001] 36.87 deg
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

    def test_output_dtype_and_atom_count_are_valid(self):
        from GBOpt.Atom import Atom
        gb = self._build(self.PQ_SPEC)
        assert gb.whole_system.dtype == Atom.atom_dtype
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
    # Sigma5 [001] 36.87 deg -- exact path produces a valid fcc bicrystal.
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
        # (identity matrix for zero inclination angles), not the stale embedding.
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
# CSL specs wired into from_boundary_spec
# ---------------------------------------------------------------------------

class TestFromBoundarySpecCSL:
    A0 = 3.615
    STRUCTURE = "fcc"
    ATOM_TYPES = "Cu"
    GB_THICKNESS = 5.0

    # Sigma5 [001] 36.87 deg exact spec and its equivalent PQSpec
    EXACT_SPEC = CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 0, 0, 1])
    PQ_SPEC = PQSpec(P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                     Q=[[4, -3, 0], [3, 4, 0], [0, 0, 1]])
    APPROX_SPEC = CSLApproxSpec(axis=[0, 0, 1], plane=[1, 0, 0], angle_deg=36.87)

    def _build(self, spec, mode):
        return GBMaker.from_boundary_spec(
            self.A0, self.STRUCTURE, self.ATOM_TYPES, spec, mode=mode,
            gb_thickness=self.GB_THICKNESS,
        )

    def test_cslexactspec_exact_builds_monatomic_bicrystal(self):
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

    @pytest.mark.parametrize("kwargs", [
        {"axis": [0, 0, 1], "plane": [1, 0, 0]},
        {"axis": [0, 0, 1], "plane": [1, 0, 0], "angle_deg": np.nan},
        {"axis": [0, 0, 1], "plane": [1, 0, 0], "angle_deg": np.inf},
        {"axis": [0, 0], "plane": [1, 0, 0], "angle_deg": 36.87},
        {"axis": [0, 0, 0], "plane": [1, 0, 0], "angle_deg": 36.87},
        {"axis": [0, 0, 1], "plane": [1, 0], "angle_deg": 36.87},
        {"axis": [0, 0, 1], "plane": [0, 0, 0], "angle_deg": 36.87},
    ])
    def test_cslapproxspec_invalid_inputs_raise(self, kwargs):
        with pytest.raises(BoundarySpecError):
            CSLApproxSpec(**kwargs)


class TestFromBoundarySpecMultispecies:
    """Exact-path construction must preserve multispecies stoichiometry.

    Monatomic tests cannot catch species-count failures.  These tests use
    known exact Sigma5 [001] boundaries (via both PQSpec and CSLExactSpec) and
    assert the correct cation:anion ratio, which also implies charge neutrality.
    """

    # Sigma5 [001] 36.87 deg tilt -- verified proper rotation.
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
        # CSLExactSpec exact path must produce the same stoichiometric result as
        # the equivalent PQSpec; catches multi-species failures in the
        # csl_spec_to_embedding -> _from_boundary_embedding path.
        counts = self._species_counts_csl(4.0, "rocksalt", ("Na", "Cl"))
        assert counts["Na"] == counts["Cl"], (
            f"Rocksalt bicrystal via CSLExactSpec is not stoichiometric: {counts}"
        )

    def test_fluorite_stoichiometric(self):
        # UO2: anion : cation must be 2 : 1.
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


# ---------------------------------------------------------------------------
# Box bounds and vacuum=0 periodic gap
# ---------------------------------------------------------------------------

class TestExactPathBoxBounds:
    """All atoms from the exact builder must lie within the simulation box."""

    @pytest.mark.parametrize("spec,a0,structure,atom_types", [
        (
            PQSpec(P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                   Q=[[4, -3, 0], [3, 4, 0], [0, 0, 1]]),
            3.615, "fcc", "Cu",
        ),
        (
            CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 0, 0, 1]),
            3.615, "fcc", "Cu",
        ),
        (
            PQSpec(P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                   Q=[[4, -3, 0], [3, 4, 0], [0, 0, 1]]),
            5.47, "fluorite", ("U", "O"),
        ),
    ])
    def test_atoms_within_yz_box(self, spec, a0, structure, atom_types):
        gb = GBMaker.from_boundary_spec(
            a0, structure, atom_types, spec, mode="exact",
        )
        ws = gb.whole_system
        tol = 1e-4
        assert np.all(ws["y"] >= -tol), f"y underflow: min={ws['y'].min():.6f}"
        assert np.all(ws["y"] < gb.y_dim + tol), (
            f"y overflow: max={ws['y'].max():.6f} > y_dim={gb.y_dim:.6f}"
        )
        assert np.all(ws["z"] >= -tol), f"z underflow: min={ws['z'].min():.6f}"
        assert np.all(ws["z"] < gb.z_dim + tol), (
            f"z overflow: max={ws['z'].max():.6f} > z_dim={gb.z_dim:.6f}"
        )

    def test_exact_left_grain_does_not_overflow_gb_plane(self):
        """Olmsted GB 1 covers left-grain high-x basis offset overflow."""
        spec = PQSpec(
            P=[[3, 1, 0], [0, 0, 2], [1, -3, 0]],
            Q=[[3, 1, 0], [0, 0, -2], [-1, 3, 0]],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            gb = GBMaker.from_boundary_spec(
                3.52,
                "fcc",
                "Ni",
                spec,
                mode="exact",
                gb_thickness=0.0,
                repeat_factor=1,
                x_dim_min=20.0,
                vacuum=0.0,
                interaction_distance=5.0,
            )

        tol = 1e-4 * gb.a0
        left_max_x = float(np.max(gb.left_grain["x"]))
        right_min_x = float(np.min(gb.right_grain["x"]))
        assert left_max_x <= gb.gb_plane_x + tol, (
            f"left grain overflows GB plane: max_x={left_max_x:.6f}, "
            f"gb_plane_x={gb.gb_plane_x:.6f}"
        )
        assert right_min_x >= gb.gb_plane_x - tol, (
            f"right grain underflows GB plane: min_x={right_min_x:.6f}, "
            f"gb_plane_x={gb.gb_plane_x:.6f}"
        )

    @pytest.mark.parametrize("spec,a0,structure,atom_types,kwargs", [
        (
            PQSpec(P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                   Q=[[4, -3, 0], [3, 4, 0], [0, 0, 1]]),
            3.615, "fcc", "Cu", {},
        ),
        (
            PQSpec(P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                   Q=[[4, -3, 0], [3, 4, 0], [0, 0, 1]]),
            5.47, "fluorite", ("U", "O"), {},
        ),
        # Small slab: repeat_x=1 for the right grain, so the coarse n0 grouping
        # would leave atoms outside the box because the only n0 group can't be
        # removed.  Fine u_num_0 labels allow suffix trimming within the group.
        (
            PQSpec(P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                   Q=[[4, -3, 0], [3, 4, 0], [0, 0, 1]]),
            5.47, "fluorite", ("U", "O"),
            {"x_dim_min": 20, "interaction_distance": 1, "repeat_factor": 2},
        ),
    ])
    def test_vacuum0_atoms_within_x_box(self, spec, a0, structure, atom_types, kwargs):
        gb = GBMaker.from_boundary_spec(
            a0, structure, atom_types, spec, mode="exact", vacuum=0, **kwargs,
        )
        ws = gb.whole_system
        tol = 1e-4
        x_dim = gb._GBMaker__x_dim
        assert np.all(ws["x"] >= -tol), f"x underflow: min={ws['x'].min():.6f}"
        assert np.all(ws["x"] < x_dim + tol), (
            f"x overflow: max={ws['x'].max():.6f} > x_dim={x_dim:.6f}"
        )

    def test_vacuum0_zhang_sigma53_atoms_within_x_box(self):
        """Regression for a fluorite exact build with basis-offset x leakage."""
        from zhang2021_boundaries import BOUNDARIES

        entry = BOUNDARIES["sigma53_100_0_7_2bar_0_2bar_7_STGB"]
        spec = PQSpec(P=entry["P"], Q=entry["Q"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            gb = GBMaker.from_boundary_spec(
                5.454,
                "fluorite",
                ("U", "O"),
                spec,
                mode="exact",
                gb_thickness=0.0,
                vacuum=0.0,
                repeat_factor=[1, 1],
                x_dim_min=20,
                interaction_distance=1.0,
            )

        ws = gb.whole_system
        tol = 1e-8
        assert np.min(ws["x"]) >= -tol
        assert np.max(ws["x"]) < gb.x_dim - tol

        central_gap = float(np.min(gb.right_grain["x"]) - np.max(gb.left_grain["x"]))
        periodic_gap = float(
            (gb.x_dim - np.max(gb.right_grain["x"]))
            + np.min(gb.left_grain["x"])
        )
        assert periodic_gap >= central_gap - tol

        names, counts = np.unique(ws["name"], return_counts=True)
        species_counts = {str(name): int(count) for name, count in zip(names, counts)}
        assert species_counts["O"] == 2 * species_counts["U"]

    def test_vacuum0_periodic_gap_matches_central_gap(self):
        """Exact vacuum=0 build: gap at PBC boundary equals gap at the central GB.

        Fine u_num_0 x-layer labels allow the gap-equalization loop to remove
        only the minimal crystallographic slices needed to bring both grains
        within the simulation box, so the periodic gap converges to the central
        gap rather than overshooting by a full x-period.
        """
        spec = PQSpec(P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      Q=[[4, -3, 0], [3, 4, 0], [0, 0, 1]])
        gb = GBMaker.from_boundary_spec(
            3.615, "fcc", "Cu", spec, mode="exact", vacuum=0,
        )
        x_dim = gb._GBMaker__x_dim
        rg = gb._GBMaker__right_grain
        lg = gb._GBMaker__left_grain
        left_max_x = np.max(lg["x"])
        right_min_x = np.min(rg["x"])
        central_gap = right_min_x - left_max_x
        right_max_x = np.max(rg["x"])
        left_min_x = np.min(lg["x"])
        periodic_gap = (x_dim - right_max_x) + left_min_x
        assert right_max_x < x_dim + 1e-4, (
            f"vacuum=0 right grain overflows box: max_x={right_max_x:.4f} "
            f"> x_dim={x_dim:.4f}"
        )
        assert abs(periodic_gap - central_gap) < 0.1, (
            f"vacuum=0 periodic_gap ({periodic_gap:.4f}) != central_gap "
            f"({central_gap:.4f})"
        )


class TestExactPathNoCoincidentAtoms:
    """Exact-path bicrystals must have no coincident left/right interface atoms.

    Multi-atom-basis structures (fluorite, rocksalt) can produce right-grain atoms
    whose rotated fractional positions land inside the left grain's spatial domain.
    The low-x fine-layer removal must eliminate those atoms while preserving
    stoichiometry in both grains and the whole system.
    """

    @pytest.mark.parametrize("spec,a0,structure,atom_types", [
        (
            PQSpec(P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                   Q=[[4, -3, 0], [3, 4, 0], [0, 0, 1]]),
            5.47, "fluorite", ("U", "O"),
        ),
        (
            CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 0, 0, 1]),
            5.47, "fluorite", ("U", "O"),
        ),
    ])
    def test_no_coincident_interface_atoms(self, spec, a0, structure, atom_types):
        from scipy.spatial import cKDTree
        gb = GBMaker.from_boundary_spec(
            a0, structure, atom_types, spec, mode="exact", vacuum=0,
        )
        lg = gb._GBMaker__left_grain
        rg = gb._GBMaker__right_grain
        L = np.column_stack([lg["x"], lg["y"], lg["z"]])
        R = np.column_stack([rg["x"], rg["y"], rg["z"]])
        tree = cKDTree(R)
        dists, _ = tree.query(L, k=1)
        assert dists.min() > 1e-4, (
            f"Coincident left/right atoms detected: "
            f"{(dists < 1e-4).sum()} pairs at zero distance"
        )

    @pytest.mark.parametrize("spec,a0,structure,atom_types", [
        (
            PQSpec(P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                   Q=[[4, -3, 0], [3, 4, 0], [0, 0, 1]]),
            5.47, "fluorite", ("U", "O"),
        ),
        (
            CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 0, 0, 1]),
            5.47, "fluorite", ("U", "O"),
        ),
    ])
    def test_stoichiometry_preserved_after_low_x_removal(
        self, spec, a0, structure, atom_types
    ):
        gb = GBMaker.from_boundary_spec(
            a0, structure, atom_types, spec, mode="exact", vacuum=0,
        )
        rg = gb._GBMaker__right_grain
        ws = gb.whole_system
        u_rg = (rg["name"] == "U").sum()
        o_rg = (rg["name"] == "O").sum()
        u_ws = (ws["name"] == "U").sum()
        o_ws = (ws["name"] == "O").sum()
        assert u_rg > 0, "Right grain has no U atoms"
        assert o_rg == 2 * u_rg, (
            f"Right grain stoichiometry broken: {u_rg} U, {o_rg} O (expected 2:1)"
        )
        assert o_ws == 2 * u_ws, (
            f"Whole-system stoichiometry broken: {u_ws} U, {o_ws} O (expected 2:1)"
        )


class TestRowGCDReduce:
    @pytest.mark.parametrize("row,expected", [
        ([6, -9, 0], [2, -3, 0]),
        ([-4, -8, 0], [-1, -2, 0]),
        ([0, 0, 0], [0, 0, 0]),
        ([5, 0, 0], [1, 0, 0]),
    ])
    def test_reduces_by_common_component_gcd(self, row, expected):
        np.testing.assert_array_equal(_row_gcd_reduce(np.array(row)), expected)


class TestDotInt:
    def test_positive(self):
        assert _dot_int([1, 2, 3], [4, 5, 6]) == 32

    def test_negative(self):
        assert _dot_int([-1, 2, -3], [4, -5, 6]) == -32

    def test_zero(self):
        assert _dot_int([1, 0, 0], [0, 1, 0]) == 0

    def test_large_entries(self):
        # Verify Python-int arithmetic avoids int64 overflow.
        big = 10 ** 10
        assert _dot_int([big, big], [big, big]) == 2 * big ** 2


# ---------------------------------------------------------------------------
# _gauss_reduce_2d_paired
# ---------------------------------------------------------------------------

class TestGaussReduce2dPaired:
    def test_negative_projection_large(self):
        # ab < 0 and |ab| > aa//2: reduction must still converge.
        p1 = np.array([3, 0, 0])
        p2 = np.array([-10, 1, 0])
        q1 = np.array([1, 0, 0])
        q2 = np.array([0, 1, 0])
        a, b, qa, qb = _gauss_reduce_2d_paired(p1, p2, q1, q2)
        # Both output vectors must still span the same lattice as the inputs.
        # Verify by checking that the same unimodular operations were applied.
        assert np.linalg.matrix_rank(np.array([a, b])) == 2

    def test_same_ops_applied_to_q(self):
        # Verify Q rows get exactly the same row operations as P rows.
        p1 = np.array([5, 0, 0])
        p2 = np.array([3, 4, 0])
        # Set Q equal to P; after paired reduction Q must equal the reduced P.
        a, b, qa, qb = _gauss_reduce_2d_paired(p1, p2, p1.copy(), p2.copy())
        assert np.array_equal(a, qa)
        assert np.array_equal(b, qb)

    def test_dependent_vectors_produce_zero_row(self):
        p1 = np.array([1, 2, 0])
        p2 = np.array([2, 4, 0])   # parallel to p1
        q1 = np.array([1, 0, 0])
        q2 = np.array([2, 0, 0])
        a, b, _qa, _qb = _gauss_reduce_2d_paired(p1, p2, q1, q2)
        assert np.allclose(b, 0) or np.allclose(a, 0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(BoundarySpecError, match="same length"):
            _gauss_reduce_2d_paired(
                np.array([1, 0, 0]),
                np.array([0, 1]),     # wrong length
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
            )

    def test_non_1d_raises(self):
        with pytest.raises(BoundarySpecError, match="1D array"):
            _gauss_reduce_2d_paired(
                np.array([[1, 0, 0]]),  # 2D
                np.array([0, 1, 0]),
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
            )


# ---------------------------------------------------------------------------
# _plane_null_basis
# ---------------------------------------------------------------------------

def _check_null_basis(plane: list) -> None:
    """Assert e1, e2 are in-plane and cross(e1, e2) == +plane."""
    p = np.array(plane, dtype=int)
    e1, e2 = _plane_null_basis(p)
    e1i = np.round(e1).astype(int)
    e2i = np.round(e2).astype(int)
    assert np.dot(p, e1i) == 0, f"e1 not in plane {plane}: dot={np.dot(p, e1i)}"
    assert np.dot(p, e2i) == 0, f"e2 not in plane {plane}: dot={np.dot(p, e2i)}"
    cross = np.cross(e1i, e2i)
    assert np.array_equal(cross, p), (
        f"cross(e1, e2)={cross.tolist()} != plane={plane}"
    )


class TestPlaneNullBasis:
    def test_100(self):
        _check_null_basis([1, 0, 0])

    def test_111(self):
        _check_null_basis([1, 1, 1])

    def test_523(self):
        _check_null_basis([5, 2, 3])

    def test_zero_plane_raises(self):
        with pytest.raises(ValueError, match="zero vector"):
            _plane_null_basis(np.array([0, 0, 0]))

    def test_non_primitive_raises(self):
        with pytest.raises(ValueError, match="not primitive"):
            _plane_null_basis(np.array([2, 0, 0]))

    def test_negative_components(self):
        _check_null_basis([-1, 2, -1])


# ---------------------------------------------------------------------------
# _inplane_area_index
# ---------------------------------------------------------------------------

class TestInplaneAreaIndex:
    def test_identity_plane_100(self):
        # P[0] = [0,0,1], P[1] = [1,0,0], P[2] = [0,1,0]: index = 1
        P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
        assert _inplane_area_index(P) == 1

    def test_sigma5_csl_index(self):
        # Sigma5 [001] in-plane: P = [[0,0,1],[1,2,0],[-2,1,0]], index = 5
        P = np.array([[0, 0, 1], [1, 2, 0], [-2, 1, 0]], dtype=float)
        assert _inplane_area_index(P) == 5

    def test_out_of_plane_row_raises(self):
        # P[2] = [0, 1, 1] is not in the [0, 0, 1] plane
        P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 1]], dtype=float)
        with pytest.raises(BoundarySpecError, match="not in the boundary plane"):
            _inplane_area_index(P)

    def test_non_integer_raises(self):
        P = np.array([[0, 0, 1], [1.5, 0, 0], [0, 1, 0]], dtype=float)
        with pytest.raises(BoundarySpecError, match="integer-valued"):
            _inplane_area_index(P)

    def test_zero_area_raises(self):
        # P[1] and P[2] parallel: zero area
        P = np.array([[0, 0, 1], [1, 0, 0], [2, 0, 0]], dtype=float)
        with pytest.raises(BoundarySpecError):
            _inplane_area_index(P)

# ---------------------------------------------------------------------------
# Exact CSL arithmetic
# ---------------------------------------------------------------------------

EXACT_CSL_SCENARIOS = [
    pytest.param(
        {
            "q": (2, 0, 0, 1),
            "plane": (0, 0, 1),
            "expected_N": 5,
            "expected_M": np.array(
                [[3, -4, 0], [4, 3, 0], [0, 0, 5]], dtype=object
            ),
            "expected_sigma": 5,
            "expected_hnf_det": 5,
            "expected_basis_hnf": np.array(
                [[1, 0, 0], [2, 5, 0], [0, 0, 1]], dtype=object
            ),
            "expected_inplane_case": 1,
            "expected_inplane_basis": np.array(
                [[1, 0], [2, 5], [0, 0]], dtype=object
            ),
        },
        id="sigma5_001",
    ),
    pytest.param(
        {
            "q": (1, 1, 1, 1),
            "plane": None,
            "expected_N": 4,
            "expected_sigma": 1,
            "expected_hnf_det": 1,
            "expected_kernel_moduli": (1, 1, 1),
        },
        id="symmetry_quaternion_sigma_one",
    ),
    pytest.param(
        {
            "q": (1, 1, 1, 0),
            "plane": (1, 1, 1),
            "expected_N": 3,
            "expected_sigma": 3,
            "expected_hnf_det": 3,
            "expected_inplane_cross_abs": np.array([3, 3, 3]),
        },
        id="sigma3_111",
    ),
]

INPLANE_EXACT_CSL_SCENARIOS = [
    scenario
    for scenario in EXACT_CSL_SCENARIOS
    if scenario.values[0]["plane"] is not None
]


def _build_exact_csl_case(case):
    rot = quaternion_to_scaled_rotation(case["q"])
    csl = csl_from_scaled_rotation(rot)
    inplane = None
    if case["plane"] is not None:
        inplane = inplane_basis_from_csl(csl.basis_hnf, case["plane"])
    return rot, csl, inplane


@pytest.mark.parametrize("case", EXACT_CSL_SCENARIOS)
def test_exact_csl_scaled_rotation(case):
    rot, _csl, _inplane = _build_exact_csl_case(case)

    assert rot.N == case["expected_N"]
    if "expected_M" in case:
        np.testing.assert_array_equal(rot.M, case["expected_M"])
    gram = rot.M @ rot.M.T
    np.testing.assert_array_equal(
        gram,
        (rot.N ** 2) * np.eye(3, dtype=object),
    )


@pytest.mark.parametrize("case", EXACT_CSL_SCENARIOS)
def test_exact_csl_basis_and_sigma(case):
    _rot, csl, _inplane = _build_exact_csl_case(case)

    assert csl.sigma == case["expected_sigma"]
    assert abs(_det3(csl.basis_hnf)) == case["expected_hnf_det"]
    if "expected_basis_hnf" in case:
        np.testing.assert_array_equal(csl.basis_hnf, case["expected_basis_hnf"])
    if "expected_kernel_moduli" in case:
        assert csl.diagnostics.kernel_moduli == case["expected_kernel_moduli"]


@pytest.mark.parametrize("case", EXACT_CSL_SCENARIOS)
def test_exact_csl_membership(case):
    rot, csl, _inplane = _build_exact_csl_case(case)

    check = verify_coincidence_basis(rot, csl.basis_hnf, sigma=case["expected_sigma"])
    assert check.ok
    np.testing.assert_array_equal(
        check.residual_mod_N,
        np.zeros((3, 3), dtype=object),
    )


@pytest.mark.parametrize("case", INPLANE_EXACT_CSL_SCENARIOS)
def test_exact_csl_inplane_basis(case):
    _rot, _csl, inplane = _build_exact_csl_case(case)

    assert inplane is not None
    h = np.array(case["plane"], dtype=object)
    assert int(h @ inplane.basis[:, 0]) == 0
    assert int(h @ inplane.basis[:, 1]) == 0
    if "expected_inplane_case" in case:
        assert inplane.case_id == case["expected_inplane_case"]
    if "expected_inplane_basis" in case:
        np.testing.assert_array_equal(inplane.basis, case["expected_inplane_basis"])
    if "expected_inplane_cross_abs" in case:
        v1 = inplane.basis[:, 0].astype(int)
        v2 = inplane.basis[:, 1].astype(int)
        np.testing.assert_array_equal(
            np.abs(np.cross(v1, v2)),
            case["expected_inplane_cross_abs"],
        )


def test_sigma5_001_dsc_and_pq_exact_division():
    case = EXACT_CSL_SCENARIOS[0].values[0]
    rot, csl, inplane = _build_exact_csl_case(case)
    P_raw, Q_raw = pq_from_csl_plane(rot, inplane)

    result = dsc_basis(csl.basis_hnf, csl.sigma)
    expected_adj = np.array([[5, 0, 0], [-2, 1, 0], [0, 0, 5]], dtype=object)
    assert result.denominator == 5
    np.testing.assert_array_equal(result.numerator, expected_adj)
    assert _det3(result.numerator) == 25

    p1 = inplane.basis[:, 0]
    p2 = inplane.basis[:, 1]
    assert np.array_equal(rot.M @ p1, rot.N * Q_raw[1])
    assert np.array_equal(rot.M @ p2, rot.N * Q_raw[2])
    assert P_raw.shape == (3, 3)
    assert Q_raw.shape == (3, 3)


def test_round_trip_pins_expected_pq():
    spec = CSLExactSpec(axis=(0, 0, 1), plane=(0, 0, 1), quat=(2, 0, 0, 1), sigma=5)
    embedding = csl_spec_to_embedding(spec)

    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))
    csl = csl_from_scaled_rotation(rot)
    inplane = inplane_basis_from_csl(csl.basis_hnf, (0, 0, 1))
    P_d, Q_d = pq_from_csl_plane(rot, inplane)
    P_d, Q_d = canonicalize_pq(P_d, Q_d)

    expected_P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=int)
    expected_Q = np.array([[0, 0, 1], [2, 1, 0], [-1, 2, 0]], dtype=int)
    expected_embedding_P = np.array([[0, 0, 1], [2, 1, 0], [-1, 2, 0]], dtype=int)
    expected_embedding_Q = np.array([[0, 0, 1], [2, -1, 0], [1, 2, 0]], dtype=int)
    P_d = np.round(P_d).astype(int)
    Q_d = np.round(Q_d).astype(int)
    P_embedding = np.round(embedding.P).astype(int)
    Q_embedding = np.round(embedding.Q).astype(int)

    assert np.array_equal(P_d, expected_P), f"Direct path P:\n{P_d}"
    assert np.array_equal(Q_d, expected_Q), f"Direct path Q:\n{Q_d}"
    assert np.array_equal(P_embedding, expected_embedding_P), (
        f"Embedding P:\n{embedding.P}"
    )
    assert np.array_equal(Q_embedding, expected_embedding_Q), (
        f"Embedding Q:\n{embedding.Q}"
    )
    assert embedding.exact is True
    assert embedding.metadata is not None
    assert embedding.metadata.basis_mode == "primitive"


@pytest.mark.parametrize("q_in, q_expected", [
    ((4, 0, 0, 2), (2, 0, 0, 1)),
    ((0, 0, 0, -3), (0, 0, 0, 1)),
    ((-2, 0, 0, -1), (2, 0, 0, 1)),
])
def test_normalize_integer_quaternion(q_in, q_expected):
    assert normalize_integer_quaternion(q_in) == q_expected


def test_normalize_zero_raises():
    with pytest.raises(ExactCSLValueError):
        normalize_integer_quaternion((0, 0, 0, 0))


def test_lll_reduce_identity_unchanged():
    """LLL reduction of an already-reduced basis (identity) returns the identity."""
    B = np.eye(3, dtype=object)
    R = lll_reduce(B)
    # Both bases should span the same lattice with det = +-1.
    det = int(np.round(float(np.linalg.det(np.asarray(R, dtype=float)))))
    assert abs(det) == 1
    assert np.allclose(np.asarray(R, dtype=float), np.eye(3), atol=1e-9)


def test_lll_reduce_actually_reduces():
    """LLL reduction produces shorter basis vectors than the input."""
    # A skewed but full-rank basis whose second column is far from orthogonal.
    B = np.array([[1, 10, 0], [0, 1, 0], [0, 0, 1]], dtype=object)
    R = lll_reduce(B)
    # Output columns must be shorter in norm than the skewed input.
    input_norms = [np.linalg.norm(B[:, i].astype(float)) for i in range(3)]
    output_norms = [np.linalg.norm(np.asarray(R[:, i], dtype=float)) for i in range(3)]
    assert max(output_norms) < max(input_norms), (
        f"LLL did not reduce: input norms {input_norms}, output norms {output_norms}"
    )
    # The lattice must be preserved (det unchanged up to sign).
    det_in = abs(int(np.round(float(np.linalg.det(B.astype(float))))))
    det_out = abs(int(np.round(float(np.linalg.det(np.asarray(R, dtype=float))))))
    assert det_in == det_out


def test_lll_reduce_same_lattice_as_csl_hnf():
    """LLL-reduced Sigma5 CSL basis spans the same lattice as the HNF basis."""
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))
    csl = csl_from_scaled_rotation(rot)
    csl_lll = csl_from_scaled_rotation(rot, post_reduce="lll")
    # Both must have the same sigma (same lattice).
    assert csl_lll.sigma == csl.sigma
    # HNF basis and LLL basis must have the same absolute determinant.
    det_hnf = abs(_det3(csl.basis_hnf))
    det_lll = abs(_det3(np.asarray(csl_lll.basis, dtype=object)))
    assert det_hnf == det_lll


def test_lll_reduce_singular_raises():
    with pytest.raises(ExactCSLValueError, match="full-rank"):
        lll_reduce(np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0]], dtype=object))


def test_hnf_2d_supercells_count_and_det():
    hnfs_2 = hnf_2d_supercells(2)
    assert len(hnfs_2) == 3
    for H in hnfs_2:
        det = int(H[0, 0]) * int(H[1, 1]) - int(H[0, 1]) * int(H[1, 0])
        assert det == 2

    assert len(hnf_2d_supercells(6)) == 12


def test_snf_known_diagonal_couples_coprime_factors():
    """Deterministic SNF oracle independent of optional SymPy coverage."""
    A = np.array([[2, 0, 0], [0, 6, 0], [0, 0, 15]], dtype=object)

    snf = smith_normal_form_3x3(A)

    np.testing.assert_array_equal(
        np.diag(snf.D).astype(int),
        np.array([1, 6, 30], dtype=int),
    )
    np.testing.assert_array_equal(snf.U @ A @ snf.V, snf.D)


# ---------------------------------------------------------------------------
# column_hnf_3x3 tests
# ---------------------------------------------------------------------------

def _hnf_postcondition(H: np.ndarray) -> None:
    """Assert H satisfies column-HNF postconditions."""
    H = np.asarray(H, dtype=object)
    for j in range(3):
        assert int(H[j, j]) > 0, f"diagonal H[{j},{j}]={H[j,j]} not positive"
        for i in range(j):
            assert 0 <= int(H[j, i]) < int(H[j, j]), (
                f"H[{j},{i}]={H[j,i]} not in [0, {H[j,j]})"
            )
        for i in range(j + 1, 3):
            assert int(H[j, i]) == 0, f"upper-triangle H[{j},{i}]={H[j,i]} != 0"


def test_hnf_counterexample_reduction_order():
    """Concrete matrix that exposed the descending-reduction-order bug.

    Before the fix, the reduction loop used range(2, i, -1) (descending),
    which caused H[2,0] to be re-dirtied after it was first reduced, yielding
    H[2,0]=-5 and violating the canonical residue condition.
    """
    A = np.array([[1, -4, -1], [2, 5, -3], [-1, 3, -1]], dtype=object)
    H = column_hnf_3x3(A)
    _hnf_postcondition(H)
    assert abs(_det3(H)) == abs(_det3(A))


def test_hnf_negative_offdiagonal():
    """Matrix whose triangularization produces a negative off-diagonal entry."""
    A = np.array([[3, 0, 0], [-7, 5, 0], [2, 1, 4]], dtype=object)
    H = column_hnf_3x3(A)
    _hnf_postcondition(H)
    assert abs(_det3(H)) == abs(_det3(A))


def test_hnf_known_answer_identity():
    """Identity matrix is already in HNF."""
    A = np.eye(3, dtype=object)
    H = column_hnf_3x3(A)
    assert np.array_equal(H, A)
    _hnf_postcondition(H)


def test_hnf_known_answer_sigma5():
    """Sigma5 CSL matrix in an uncanonical form reduces to the expected HNF.

    The Sigma5 [001] CSL lattice has basis columns [1,2,0], [0,5,0], [0,0,1].
    Expressed as the row matrix [[1,0,0],[2,5,0],[0,0,1]] it is already in HNF;
    starting from the row-permuted / uncanonical form [[1,0,0],[4,5,0],[0,0,1]]
    (H[1,0]=4 violates the residue bound 0<=H[j,i]<H[j,j]=5... actually 4<5 is
    fine) -- instead use a permuted-column input that provably needs reduction.
    """
    # Upper-triangular input: columns are [1,0,0],[2,5,0],[0,0,1] read column-wise.
    # After column HNF: lower-triangular [[1,0,0],[2,5,0],[0,0,1]].
    A = np.array([[1, 2, 0], [0, 5, 0], [0, 0, 1]], dtype=object)
    H = column_hnf_3x3(A)
    _hnf_postcondition(H)
    # det must be preserved (det A = 5)
    assert abs(_det3(H)) == 5
    # The unique canonical HNF for this lattice
    expected = np.array([[1, 0, 0], [0, 5, 0], [0, 0, 1]], dtype=object)
    assert np.array_equal(H, expected)


def test_hnf_round_trip_unimodular():
    """column_hnf_3x3(A) == A @ V for some unimodular integer matrix V."""
    A = np.array([[2, 3, 1], [0, 4, 2], [1, 0, 5]], dtype=object)
    H = column_hnf_3x3(A)
    _hnf_postcondition(H)
    # Solve for V: A @ V == H => V = A^{-1} @ H
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


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("sympy"),
    reason="SymPy not installed",
)
def test_hnf_randomized_vs_sympy():
    """Randomized comparison against SymPy's hermite_normal_form (optional)."""
    import sympy as sp
    from sympy.matrices.normalforms import hermite_normal_form

    rng = np.random.default_rng(42)
    failures = []
    for _ in range(200):
        while True:
            A_raw = rng.integers(-9, 10, size=(3, 3))
            if abs(round(float(np.linalg.det(A_raw.astype(float))))) >= 1:
                break
        A = np.asarray(A_raw, dtype=object)
        H = column_hnf_3x3(A)
        try:
            _hnf_postcondition(H)
        except AssertionError as exc:
            failures.append(f"A={A_raw.tolist()}: {exc}")
            continue
        # Compare against SymPy's column HNF.
        sp_matrix = sp.Matrix([[int(A[i, j]) for j in range(3)] for i in range(3)])
        sp_hnf = hermite_normal_form(sp_matrix, col_wise=True)
        sp_array = np.array([[int(sp_hnf[i, j]) for j in range(3)] for i in range(3)],
                            dtype=object)
        if _det3(sp_array) < 0:
            sp_array[:, 0] = -sp_array[:, 0]
        if not np.array_equal(H, sp_array):
            failures.append(
                f"A={A_raw.tolist()}: GBOpt={H.tolist()} SymPy={sp_array.tolist()}"
            )
    assert not failures, "\n".join(failures)
