# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

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
        assert list(spec.params) == p


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
        assert spec.angle_deg == 36.87

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
