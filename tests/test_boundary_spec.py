# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import numpy as np
import pytest
from GBOpt.BoundarySpec import (
    FiveDOFSpec,
    PQSpec,
    CSLExactSpec,
    CSLApproxSpec,
    ConstructionMode,
    BoundarySpecError,
    BoundarySpecTypeError,
    BoundarySpecValueError,
    _CSLSpecBase,
    BoundaryEmbedding,
)


class TestImports:
    def test_all_public_names_importable(self):
        # Acceptance criterion: imports succeed
        assert FiveDOFSpec is not None
        assert PQSpec is not None
        assert CSLExactSpec is not None
        assert CSLApproxSpec is not None
        assert BoundarySpecError is not None

    def test_construction_mode_values(self):
        # ConstructionMode is a typing.Literal — check it exists and its args
        import typing

        args = typing.get_args(ConstructionMode)
        assert set(args) == {"exact", "prefer_exact", "approximate"}


class TestFiveDOFSpec:
    def test_frozen(self):
        spec = FiveDOFSpec(params=[0.1, 0.2, 0.3, 0.4, 0.5])
        with pytest.raises((AttributeError, TypeError)):
            spec.params = [1, 2, 3, 4, 5]

    def test_stores_params(self):
        p = [0.1, 0.2, 0.3, 0.4, 0.5]
        spec = FiveDOFSpec(params=p)
        np.testing.assert_allclose(list(spec.params), p, atol=1e-15, rtol=0)


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


class TestCSLExactSpec:
    def test_frozen(self):
        spec = CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 1, 0, 0])
        with pytest.raises((AttributeError, TypeError)):
            spec.quat = [1, 0, 0, 0]

    def test_stores_quat(self):
        spec = CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 1, 0, 0])
        assert spec.quat == [3, 1, 0, 0]

    def test_sigma_optional(self):
        spec = CSLExactSpec(axis=[0, 0, 1], plane=[1, 0, 0], quat=[3, 1, 0, 0], sigma=5)
        assert spec.sigma == 5

    def test_inherits_base(self):
        assert issubclass(CSLExactSpec, _CSLSpecBase)


class TestCSLApproxSpec:
    def test_frozen(self):
        spec = CSLApproxSpec(axis=[0, 0, 1], plane=[1, 0, 0], angle_deg=36.87)
        with pytest.raises((AttributeError, TypeError)):
            spec.angle_deg = 45.0

    def test_stores_angle_deg(self):
        spec = CSLApproxSpec(axis=[0, 0, 1], plane=[1, 0, 0], angle_deg=36.87)
        np.testing.assert_allclose(spec.angle_deg, 36.87, atol=1e-12, rtol=0)

    def test_inherits_base(self):
        assert issubclass(CSLApproxSpec, _CSLSpecBase)


class TestBoundarySpecErrorHierarchy:
    def test_base_is_exception(self):
        assert issubclass(BoundarySpecError, Exception)

    def test_type_error_subclasses_base(self):
        assert issubclass(BoundarySpecTypeError, BoundarySpecError)
        assert issubclass(BoundarySpecTypeError, TypeError)

    def test_value_error_subclasses_base(self):
        assert issubclass(BoundarySpecValueError, BoundarySpecError)
        assert issubclass(BoundarySpecValueError, ValueError)

    def test_independent_from_gbmaker_errors(self):
        from GBOpt.GBMaker import GBMakerError

        assert not issubclass(BoundarySpecError, GBMakerError)
        assert not issubclass(GBMakerError, BoundarySpecError)

    def test_raise_and_catch_as_base(self):
        with pytest.raises(BoundarySpecError):
            raise BoundarySpecValueError("invalid field")


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


class TestGBExactSkeleton:
    """Verify the gb_exact stub surface is present and raises NotImplementedError."""

    def test_module_importable(self):
        import GBOpt.Utils.gb_exact as gb_exact

        assert gb_exact is not None

    def test_expected_functions_present(self):
        import GBOpt.Utils.gb_exact as gb_exact

        expected = [
            "validate_and_normalize_quaternion",
            "quaternion_to_rotation_matrix",
            "validate_sigma",
            "solve_inplane_csl",
            "reduce_2d_basis",
            "canonicalize_pq",
            "exactify_five_dof",
        ]
        for name in expected:
            assert hasattr(gb_exact, name), f"gb_exact missing: {name}"

    def test_stubs_raise_not_implemented(self):
        import GBOpt.Utils.gb_exact as gb_exact

        dummy = np.zeros(4)
        dummy3 = np.zeros(3)
        dummy33 = np.eye(3)
        cases = [
            lambda: gb_exact.validate_and_normalize_quaternion(dummy),
            lambda: gb_exact.quaternion_to_rotation_matrix(dummy),
            lambda: gb_exact.validate_sigma(dummy, 5),
            lambda: gb_exact.solve_inplane_csl(dummy3, dummy3, dummy33),
            lambda: gb_exact.reduce_2d_basis(dummy3, dummy3),
            lambda: gb_exact.canonicalize_pq(dummy33, dummy33),
            lambda: gb_exact.exactify_five_dof(np.zeros(5)),
        ]
        for fn in cases:
            with pytest.raises(NotImplementedError):
                fn()
