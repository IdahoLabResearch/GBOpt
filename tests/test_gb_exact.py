# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import warnings

import numpy as np
import pytest

from unittest.mock import patch

from GBOpt.BoundarySpec import BoundaryEmbedding, BoundarySpecError, PQSpec
from GBOpt.GBMaker import GBMaker
from GBOpt.Utils.gb_exact import (
    canonicalize_pq,
    pq_spec_to_embedding,
    quaternion_to_rotation_matrix,
    validate_and_normalize_quaternion,
    validate_sigma,
)


def _make_identity_pair():
    I = np.eye(3, dtype=float)
    return I.copy(), I.copy()


# ---------------------------------------------------------------------------
# Step 10 — integer-quaternion validation and rotation matrix
# ---------------------------------------------------------------------------

class TestValidateAndNormalizeQuaternion:
    # Σ5 [001] 53.13° — integer quaternion [2, 0, 0, 1], N = 5
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
    # Σ5 [001] 53.13° — quat [2, 0, 0, 1] (Hamilton [w,x,y,z]),
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
        # Σ5 [001] 36.87° — quat [3, 0, 0, 1], N = 10
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
    # Atom-array round-trip: Sigma5 [001] 36.87 deg tilt boundary.
    # P and Q are chosen to exactly match what __approximate_rotation_matrix_as_int
    # produces for this misorientation, so the row ordering is identical.
    # ------------------------------------------------------------------

    def test_sigma5_atom_array_matches_legacy(self):
        import math
        # Use the exact angle arctan(3/4) so scipy's rotation matrix is
        # [[4/5,-3/5,0],[3/5,4/5,0],[0,0,1]] to within 1 ULP — close enough
        # that atom coordinates agree to ~1e-14 A.
        theta = math.atan2(3, 4)
        misorientation = np.array([0.0, 0.0, theta, 0.0, 0.0])
        gb_legacy = self._legacy(misorientation)

        P = gb_legacy._GBMaker__R_left_approx.astype(int).tolist()
        Q = gb_legacy._GBMaker__R_right_approx.astype(int).tolist()
        gb_emb = self._from_spec(P, Q)

        legacy_atoms = gb_legacy.whole_system
        emb_atoms = gb_emb.whole_system
        assert legacy_atoms.shape == emb_atoms.shape
        numeric_fields = [
            f for f in legacy_atoms.dtype.names
            if np.issubdtype(legacy_atoms[f].dtype, np.number)
        ]
        for field in numeric_fields:
            np.testing.assert_allclose(
                emb_atoms[field], legacy_atoms[field], atol=1e-10,
                err_msg=f"field '{field}' differs",
            )

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


class TestFromBoundarySpecMultispecies:
    """Step 9 acceptance: exact-PQ construction must preserve stoichiometry.

    Monatomic tests cannot catch species-count failures.  These tests use
    known exact Sigma5 [001] P/Q and assert the correct cation:anion ratio,
    which also implies charge neutrality for these structures.
    """

    # Sigma5 [001] 36.87 deg tilt — orthogonal integer rows, verified proper rotation.
    P = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    Q = [[4, -3, 0], [3, 4, 0], [0, 0, 1]]

    def _species_counts(self, a0, structure, atom_types):
        spec = PQSpec(P=self.P, Q=self.Q)
        gb = GBMaker.from_boundary_spec(
            a0, structure, atom_types, spec, mode="exact", gb_thickness=2.0
        )
        ws = gb.whole_system
        names, counts = np.unique(ws["name"], return_counts=True)
        return {str(n): int(c) for n, c in zip(names, counts)}

    def test_rocksalt_stoichiometric(self):
        # NaCl: cation : anion must be 1 : 1.
        counts = self._species_counts(4.0, "rocksalt", ("Na", "Cl"))
        assert counts["Na"] == counts["Cl"], (
            f"Rocksalt bicrystal is not stoichiometric: {counts}"
        )

    @pytest.mark.known_bug
    def test_fluorite_stoichiometric(self):
        # UO₂: anion : cation must be 2 : 1.
        # Known bug: GBMaker produces 12 fewer O atoms than expected (e.g.
        # 9492 O / 4752 U instead of 9504 O / 4752 U) on both the legacy and
        # exact-PQ paths.  Confirmed pre-existing; not introduced by Stage B.
        counts = self._species_counts(5.47, "fluorite", ("U", "O"))
        assert counts["O"] == 2 * counts["U"], (
            f"Fluorite bicrystal is not stoichiometric: {counts}"
        )
