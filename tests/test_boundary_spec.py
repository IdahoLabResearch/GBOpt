# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import numpy as np
import pytest
from GBOpt.BoundarySpec import (
    PQSpec,
    ConstructionMode,
    BoundarySpecError,
    BoundaryEmbedding,
)


class TestImports:
    def test_public_names_importable(self):
        assert PQSpec is not None
        assert BoundarySpecError is not None

    def test_construction_mode_values(self):
        import typing

        args = typing.get_args(ConstructionMode)
        assert set(args) == {"exact", "prefer_exact", "approximate"}


class TestPQSpec:
    def test_frozen(self):
        P = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        Q = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        spec = PQSpec(P=P, Q=Q)
        with pytest.raises((AttributeError, TypeError)):
            spec.P = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

    def test_stores_p_and_q(self):
        P = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        Q = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
        spec = PQSpec(P=P, Q=Q)
        assert spec.P == P
        assert spec.Q == Q

    def test_valid_3x3_input_passes(self):
        P = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        Q = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        spec = PQSpec(P=P, Q=Q)
        assert spec.P == P

    def test_wrong_shape_raises(self):
        P_bad = [[1, 0, 0], [0, 1, 0]]  # 2x3
        Q = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        with pytest.raises(BoundarySpecError):
            PQSpec(P=P_bad, Q=Q)

    def test_nan_entry_raises(self):
        P = [[1, 0, 0], [0, float("nan"), 0], [0, 0, 1]]
        Q = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        with pytest.raises(BoundarySpecError):
            PQSpec(P=P, Q=Q)

    def test_inf_entry_raises(self):
        P = [[1, 0, 0], [0, float("inf"), 0], [0, 0, 1]]
        Q = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        with pytest.raises(BoundarySpecError):
            PQSpec(P=P, Q=Q)

    def test_singular_matrix_raises(self):
        P_singular = [[1, 0, 0], [1, 0, 0], [0, 0, 1]]  # rows 0 and 1 identical
        Q = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        with pytest.raises(BoundarySpecError):
            PQSpec(P=P_singular, Q=Q)



class TestBoundaryEmbedding:
    def _make(self, P=None, Q=None, exact=True, coherent=True, source="pq"):
        R = np.eye(3)
        return BoundaryEmbedding(
            P=P,
            Q=Q,
            R_left=R,
            R_right=R,
            exact=exact,
            coherent=coherent,
            source=source,
        )

    def test_instantiate_with_arrays(self):
        P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        Q = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        emb = self._make(P=P, Q=Q)
        np.testing.assert_allclose(emb.P, P, atol=1e-15, rtol=0)
        np.testing.assert_allclose(emb.Q, Q, atol=1e-15, rtol=0)
        np.testing.assert_allclose(emb.R_left, np.eye(3), atol=1e-15, rtol=0)
        np.testing.assert_allclose(emb.R_right, np.eye(3), atol=1e-15, rtol=0)

    def test_instantiate_with_none_pq(self):
        emb = self._make(P=None, Q=None, exact=False, coherent=False, source="five_dof")
        assert emb.P is None
        assert emb.Q is None
        assert emb.exact is False
        assert emb.coherent is False
        assert emb.source == "five_dof"

    def test_source_stored(self):
        emb = self._make(source="csl")
        assert emb.source == "csl"
