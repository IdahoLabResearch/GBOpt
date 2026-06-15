# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import numpy as np
import pytest
from GBOpt.BoundarySpec import (
    BoundaryEmbedding,
    BoundarySpecError,
    BoundarySpecTypeError,
    BoundarySpecValueError,
    CSLApproxSpec,
    CSLExactSpec,
    ConstructionMode,
    FiveDOFSpec,
    PQSpec,
    _CSLSpecBase,
)


class TestImports:
    def test_public_names_importable(self):
        assert PQSpec is not None
        assert BoundarySpecError is not None

    def test_construction_mode_values(self):
        import typing

        args = typing.get_args(ConstructionMode)
        assert set(args) == {"exact", "prefer_exact", "approximate"}


class TestFiveDOFSpec:
    VALID_PARAMS = [0.1, 0.2, 0.3, 45.0, 30.0]

    def test_frozen(self):
        spec = FiveDOFSpec(params=self.VALID_PARAMS)
        with pytest.raises((AttributeError, TypeError)):
            spec.params = [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_stores_params(self):
        spec = FiveDOFSpec(params=self.VALID_PARAMS)
        assert list(spec.params) == self.VALID_PARAMS

    def test_valid_5_element_passes(self):
        spec = FiveDOFSpec(params=self.VALID_PARAMS)
        assert spec is not None

    def test_wrong_length_raises(self):
        with pytest.raises(BoundarySpecValueError):
            FiveDOFSpec(params=[0.1, 0.2, 0.3])

    def test_non_numeric_raises(self):
        with pytest.raises(BoundarySpecTypeError):
            FiveDOFSpec(params=["a", "b", "c", "d", "e"])

    def test_nan_raises(self):
        with pytest.raises(BoundarySpecValueError):
            FiveDOFSpec(params=[0.1, float("nan"), 0.3, 45.0, 30.0])

    def test_inf_raises(self):
        with pytest.raises(BoundarySpecValueError):
            FiveDOFSpec(params=[0.1, float("inf"), 0.3, 45.0, 30.0])


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


class TestCSLSpecBase:
    def test_frozen(self):
        spec = _CSLSpecBase(axis=[0, 0, 1], plane=[1, 0, 0])
        with pytest.raises((AttributeError, TypeError)):
            spec.axis = [1, 1, 0]

    def test_sigma_defaults_to_none(self):
        spec = _CSLSpecBase(axis=[0, 0, 1], plane=[1, 0, 0])
        assert spec.sigma is None

    def test_sigma_stored(self):
        spec = _CSLSpecBase(axis=[0, 0, 1], plane=[1, 0, 0], sigma=5)
        assert spec.sigma == 5

    def test_zero_axis_raises(self):
        with pytest.raises(BoundarySpecValueError):
            _CSLSpecBase(axis=[0, 0, 0], plane=[1, 0, 0])

    def test_non_integer_plane_raises(self):
        with pytest.raises(BoundarySpecValueError):
            _CSLSpecBase(axis=[0, 0, 1], plane=[1.5, 0.0, 0.0])

    def test_non_positive_sigma_raises(self):
        with pytest.raises(BoundarySpecValueError):
            _CSLSpecBase(axis=[0, 0, 1], plane=[1, 0, 0], sigma=0)

    def test_nan_axis_raises(self):
        with pytest.raises(BoundarySpecValueError):
            _CSLSpecBase(axis=[0, float("nan"), 1], plane=[1, 0, 0])


class TestCSLExactSpec:
    # Sigma5 [001] 36.87 deg: quat=[3,0,0,1] has vector part [0,0,1] || axis=[0,0,1].
    VALID_QUAT = [3, 0, 0, 1]

    def test_frozen(self):
        spec = CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=self.VALID_QUAT)
        with pytest.raises((AttributeError, TypeError)):
            spec.quat = [1, 0, 0, 0]

    def test_stores_quat(self):
        spec = CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=self.VALID_QUAT)
        assert list(spec.quat) == self.VALID_QUAT

    def test_sigma_optional(self):
        spec = CSLExactSpec(
            axis=[0, 0, 1], plane=[1, 0, 0], quat=self.VALID_QUAT, sigma=5
        )
        assert spec.sigma == 5

    def test_quat_none_raises(self):
        with pytest.raises(BoundarySpecValueError):
            CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0])

    def test_non_integer_quat_raises(self):
        with pytest.raises(BoundarySpecValueError):
            CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[1.5, 0.0, 0.0, 1.0])

    def test_wrong_quat_length_raises(self):
        with pytest.raises(BoundarySpecValueError):
            CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[1, 0, 0])

    def test_axis_quat_vector_mismatch_raises(self):
        # quat vector part [0, 0, 1] is not parallel to axis [1, 0, 0]
        with pytest.raises(BoundarySpecValueError):
            CSLExactSpec(axis=[1, 0, 0], plane=[0, 0, 1], quat=[3, 0, 0, 1])

    def test_zero_axis_raises(self):
        with pytest.raises(BoundarySpecValueError):
            CSLExactSpec(axis=[0, 0, 0], plane=[1, 0, 0], quat=self.VALID_QUAT)


class TestCSLApproxSpec:
    def test_frozen(self):
        spec = CSLApproxSpec(axis=[0, 0, 1], plane=[1, 0, 0], angle_deg=36.87)
        with pytest.raises((AttributeError, TypeError)):
            spec.angle_deg = 45.0

    def test_stores_angle_deg(self):
        spec = CSLApproxSpec(axis=[0, 0, 1], plane=[1, 0, 0], angle_deg=36.87)
        np.testing.assert_allclose(spec.angle_deg, 36.87, atol=1e-12, rtol=0)

    def test_sigma_optional(self):
        spec = CSLApproxSpec(
            axis=[0, 0, 1], plane=[1, 0, 0], angle_deg=36.87, sigma=5
        )
        assert spec.sigma == 5

    def test_angle_none_raises(self):
        with pytest.raises(BoundarySpecValueError):
            CSLApproxSpec(axis=[0, 0, 1], plane=[1, 0, 0])

    def test_non_numeric_angle_raises(self):
        with pytest.raises(BoundarySpecTypeError):
            CSLApproxSpec(axis=[0, 0, 1], plane=[1, 0, 0], angle_deg="not_a_number")

    def test_nan_angle_raises(self):
        with pytest.raises(BoundarySpecValueError):
            CSLApproxSpec(axis=[0, 0, 1], plane=[1, 0, 0], angle_deg=float("nan"))

    def test_inf_angle_raises(self):
        with pytest.raises(BoundarySpecValueError):
            CSLApproxSpec(axis=[0, 0, 1], plane=[1, 0, 0], angle_deg=float("inf"))

    def test_zero_plane_raises(self):
        with pytest.raises(BoundarySpecValueError):
            CSLApproxSpec(axis=[0, 0, 1], plane=[0, 0, 0], angle_deg=36.87)


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
