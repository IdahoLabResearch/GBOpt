# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from __future__ import annotations

import numpy as np
import pytest

from GBOpt.Utils.exact_csl import (
    ExactCSLNotImplementedError,
    ExactCSLValueError,
    csl_from_scaled_rotation,
    dsc_basis,
    exactify_five_dof,
    inplane_basis_from_csl,
    lll_reduce,
    normalize_integer_quaternion,
    pq_from_csl_plane,
    quaternion_to_scaled_rotation,
    verify_coincidence_basis,
)
from GBOpt.Utils.integer_normal_forms import (
    ExactNormalFormError,
    hnf_2d_supercells,
    saturate_column_lattice_3x3,
)


def _det3(A: np.ndarray) -> int:
    """Exact integer 3x3 determinant via Sarrus."""
    A = np.asarray(A, dtype=object)
    return int(
        A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
        - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
        + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
    )


class TestSigma5_001:
    """q=(2,0,0,1): verifies M, CSL basis, in-plane vectors, DSC, and P/Q."""

    def setup_method(self):
        q = (2, 0, 0, 1)
        self.rot = quaternion_to_scaled_rotation(q)
        self.csl = csl_from_scaled_rotation(self.rot)
        self.inplane = inplane_basis_from_csl(self.csl.basis_hnf, (0, 0, 1))
        self.P_raw, self.Q_raw = pq_from_csl_plane(self.rot, self.inplane)

    def test_N(self):
        assert self.rot.N == 5

    def test_M(self):
        expected = np.array([[3, -4, 0], [4, 3, 0], [0, 0, 5]], dtype=object)
        assert np.array_equal(self.rot.M, expected)

    def test_M_orthogonality(self):
        G = self.rot.M @ self.rot.M.T
        assert np.array_equal(G, (self.rot.N ** 2) * np.eye(3, dtype=object))

    def test_sigma(self):
        assert self.csl.sigma == 5

    def test_csl_det(self):
        assert abs(_det3(self.csl.basis_hnf)) == 5

    def test_csl_basis_hnf(self):
        expected = np.array([[1, 0, 0], [2, 5, 0], [0, 0, 1]], dtype=object)
        assert np.array_equal(self.csl.basis_hnf, expected)

    def test_membership(self):
        check = verify_coincidence_basis(self.rot, self.csl.basis_hnf, sigma=5)
        assert check.ok
        assert np.array_equal(check.residual_mod_N, np.zeros((3, 3), dtype=object))

    def test_inplane_case_id(self):
        assert self.inplane.case_id == 1

    def test_inplane_orthogonality(self):
        h = np.array([0, 0, 1], dtype=object)
        assert int(h @ self.inplane.basis[:, 0]) == 0
        assert int(h @ self.inplane.basis[:, 1]) == 0

    def test_inplane_basis_values(self):
        expected = np.array([[1, 0], [2, 5], [0, 0]], dtype=object)
        assert np.array_equal(self.inplane.basis, expected)

    def test_dsc(self):
        result = dsc_basis(self.csl.basis_hnf, self.csl.sigma)
        expected_adj = np.array([[5, 0, 0], [-2, 1, 0], [0, 0, 5]], dtype=object)
        assert result.denominator == 5
        assert np.array_equal(result.numerator, expected_adj)
        assert _det3(result.numerator) == 25

    def test_pq_grain_b_exact_division(self):
        p1 = self.inplane.basis[:, 0]
        p2 = self.inplane.basis[:, 1]
        N, M = self.rot.N, self.rot.M
        assert np.array_equal(M @ p1, N * self.Q_raw[1])
        assert np.array_equal(M @ p2, N * self.Q_raw[2])


class TestSymmetryQuaternionSigmaOne:
    """120 degree [111] rotation is a crystal symmetry; true Sigma=1."""

    def setup_method(self):
        q = (1, 1, 1, 1)
        self.rot = quaternion_to_scaled_rotation(q)
        self.csl = csl_from_scaled_rotation(self.rot)

    def test_N_is_4(self):
        assert self.rot.N == 4

    def test_sigma_is_1_not_4(self):
        assert self.csl.sigma == 1, (
            f"sigma={self.csl.sigma} but expected 1; "
            "Sigma must come from SNF, never from N."
        )

    def test_kernel_moduli_all_ones(self):
        assert self.csl.diagnostics.kernel_moduli == (1, 1, 1)

    def test_csl_is_whole_lattice(self):
        assert abs(_det3(self.csl.basis_hnf)) == 1

    def test_membership(self):
        check = verify_coincidence_basis(self.rot, self.csl.basis_hnf, sigma=1)
        assert check.ok


class TestSigma3_111:
    """q=(1,1,1,0): CSL det=3 and exact in-plane cross product."""

    def setup_method(self):
        q = (1, 1, 1, 0)
        self.rot = quaternion_to_scaled_rotation(q)
        self.csl = csl_from_scaled_rotation(self.rot)
        self.inplane = inplane_basis_from_csl(self.csl.basis_hnf, (1, 1, 1))

    def test_N(self):
        assert self.rot.N == 3

    def test_sigma(self):
        assert self.csl.sigma == 3

    def test_csl_det(self):
        assert abs(_det3(self.csl.basis_hnf)) == 3

    def test_inplane_orthogonality(self):
        h = np.array([1, 1, 1], dtype=object)
        assert int(h @ self.inplane.basis[:, 0]) == 0
        assert int(h @ self.inplane.basis[:, 1]) == 0

    def test_inplane_cross_product(self):
        v1 = self.inplane.basis[:, 0].astype(int)
        v2 = self.inplane.basis[:, 1].astype(int)
        cross = np.cross(v1, v2)
        assert np.array_equal(np.abs(cross), np.array([3, 3, 3])), (
            f"Expected |cross| = [3,3,3], got {cross}"
        )

    def test_membership(self):
        check = verify_coincidence_basis(self.rot, self.csl.basis_hnf, sigma=3)
        assert check.ok


def test_round_trip_pins_expected_pq():
    from GBOpt.BoundarySpec import CSLExactSpec
    from GBOpt.Utils.gb_exact import canonicalize_pq, csl_spec_to_embedding

    spec = CSLExactSpec(axis=(0, 0, 1), plane=(0, 0, 1), quat=(2, 0, 0, 1), sigma=5)
    embedding = csl_spec_to_embedding(spec)

    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))
    csl = csl_from_scaled_rotation(rot)
    inplane = inplane_basis_from_csl(csl.basis_hnf, (0, 0, 1))
    P_d, Q_d = pq_from_csl_plane(rot, inplane)
    P_d, Q_d = canonicalize_pq(P_d, Q_d)

    expected_P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=int)
    expected_Q = np.array([[0, 0, 1], [2, 1, 0], [-1, 2, 0]], dtype=int)
    P_d = np.round(P_d).astype(int)
    Q_d = np.round(Q_d).astype(int)
    P_embedding = np.round(embedding.P).astype(int)
    Q_embedding = np.round(embedding.Q).astype(int)

    assert np.array_equal(P_d, expected_P), f"Direct path P:\n{P_d}"
    assert np.array_equal(Q_d, expected_Q), f"Direct path Q:\n{Q_d}"
    assert np.array_equal(P_embedding, expected_P), f"Embedding P:\n{embedding.P}"
    assert np.array_equal(Q_embedding, expected_Q), f"Embedding Q:\n{embedding.Q}"
    assert embedding.exact is True


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


def test_exactify_five_dof_raises():
    with pytest.raises(ExactCSLNotImplementedError, match="Stage E"):
        exactify_five_dof(np.array([0.0, 0.0, 0.0, 0.6435, 0.0]))


def test_hnf_2d_supercells_count_and_det():
    hnfs_2 = hnf_2d_supercells(2)
    assert len(hnfs_2) == 3
    for H in hnfs_2:
        det = int(H[0, 0]) * int(H[1, 1]) - int(H[0, 1]) * int(H[1, 0])
        assert det == 2

    assert len(hnf_2d_supercells(6)) == 12


def test_saturate_column_lattice_full_rank_is_ambient_lattice():
    basis = np.array([[2, 0, 0], [1, 3, 0], [4, 5, 7]], dtype=object)
    saturated = saturate_column_lattice_3x3(basis)
    assert np.array_equal(saturated, np.eye(3, dtype=object))


def test_saturate_column_lattice_singular_raises():
    basis = np.array([[1, 0, 0], [2, 0, 0], [0, 0, 1]], dtype=object)
    with pytest.raises(ExactNormalFormError):
        saturate_column_lattice_3x3(basis)


def test_lll_reduce_warns_and_returns_input_unchanged():
    basis = np.array([[2, 0, 0], [1, 3, 0], [4, 5, 7]], dtype=object)
    with pytest.warns(UserWarning, match="lll_reduce is not yet implemented"):
        reduced = lll_reduce(basis)
    assert np.array_equal(reduced, basis)


def test_csl_post_reduce_lll_warns_deferred():
    rot = quaternion_to_scaled_rotation((2, 0, 0, 1))
    with pytest.warns(UserWarning, match="lll_reduce is not yet implemented"):
        result = csl_from_scaled_rotation(rot, post_reduce="lll")
    assert np.array_equal(result.basis, result.basis_hnf)
