# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
from typing import Any, cast

import numpy as np
import pytest

from GBOpt.crystallography.types import (
    CoincidenceCheck,
    CrystallographyValueError,
    CSLResult,
    DSCBasis,
    InPlaneBasis,
    ScaledRotation,
    SmithDiagnostics,
)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _identity_rotation() -> ScaledRotation:
    return ScaledRotation(
        denominator=1,
        matrix=np.eye(3, dtype=object),
        source="matrix",
    )


def _identity_smith_diagnostics() -> SmithDiagnostics:
    return SmithDiagnostics(
        diagonal=(1, 1, 1),
        kernel_moduli=(1, 1, 1),
    )


def _matrix_with_entry(value: object) -> np.ndarray:
    matrix = np.eye(3, dtype=object)
    matrix[0, 0] = value
    return matrix


def _csl_result_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "rotation": _identity_rotation(),
        "sigma": 1,
        "basis": np.eye(3, dtype=object),
        "basis_hnf": np.eye(3, dtype=object),
        "diagnostics": _identity_smith_diagnostics(),
    }
    kwargs.update(overrides)
    return kwargs


def _inplane_basis_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "basis": np.ones((3, 2), dtype=object),
        "coefficients": np.eye(3, 2, dtype=object),
        "plane_covector": (1, 1, 1),
    }
    kwargs.update(overrides)
    return kwargs


def _dsc_basis_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "numerator": np.eye(3, dtype=object),
        "denominator": 5,
        "sigma": 5,
    }
    kwargs.update(overrides)
    return kwargs


def _coincidence_check_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "ok": True,
        "residual_mod_N": np.zeros((3, 3), dtype=object),
        "det_basis": 5,
        "sigma": 5,
    }
    kwargs.update(overrides)
    return kwargs


# --------------------------------------------------------------------------------------
# ScaledRotation
# --------------------------------------------------------------------------------------


def test_scaled_rotation_normalizes_denominator_and_freezes_matrix():
    rotation = ScaledRotation(
        denominator=np.int64(1),  # type: ignore[ty:invalid-argument-type]
        matrix=np.eye(3, dtype=object),
        source="matrix",
    )

    assert rotation.denominator == 1
    assert rotation.matrix.dtype == object
    assert not rotation.matrix.flags.writeable


def test_scaled_rotation_copies_input_matrix():
    matrix = np.eye(3, dtype=object)

    rotation = ScaledRotation(
        denominator=1,
        matrix=matrix,
        source="matrix",
    )

    matrix[0, 0] = 99

    assert rotation.matrix[0, 0] == 1


def test_scaled_rotation_matrix_is_read_only_to_callers():
    rotation = _identity_rotation()

    with pytest.raises(ValueError, match="read-only"):
        rotation.matrix[0, 0] = 2


@pytest.mark.parametrize(
    "denominator",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(1.5, id="float"),
        pytest.param(True, id="bool"),
    ],
)
def test_scaled_rotation_rejects_invalid_denominator(denominator):
    with pytest.raises(CrystallographyValueError, match="denominator"):
        ScaledRotation(
            denominator=denominator,
            matrix=np.eye(3, dtype=object),
            source="matrix",
        )


def test_scaled_rotation_rejects_non_3_by_3_matrix():
    with pytest.raises(CrystallographyValueError, match="matrix must have shape"):
        ScaledRotation(
            denominator=1,
            matrix=np.ones((2, 2), dtype=object),
            source="matrix",
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(1.5, id="float"),
        pytest.param(True, id="bool"),
        pytest.param("1", id="string"),
    ],
)
def test_scaled_rotation_rejects_non_integer_matrix_entries(bad_value):
    with pytest.raises(CrystallographyValueError, match="matrix"):
        ScaledRotation(
            denominator=1,
            matrix=_matrix_with_entry(bad_value),
            source="matrix",
        )


def test_scaled_rotation_rejects_unknown_source():
    with pytest.raises(CrystallographyValueError, match="source must be one of"):
        ScaledRotation(
            denominator=1,
            matrix=np.eye(3, dtype=object),
            source="bad",  # type: ignore[ty:invalid-argument-type]
        )


def test_scaled_rotation_normalizes_quaternion_tuple():
    rotation = ScaledRotation(
        denominator=1,
        matrix=np.eye(3, dtype=object),
        source="quaternion",
        quaternion=np.array(  # type: ignore[ty:invalid-argument-type]
            [1, 0, 0, 0], dtype=np.int64
        ),
    )

    assert rotation.quaternion == (1, 0, 0, 0)


@pytest.mark.parametrize(
    ("quaternion", "match"),
    [
        pytest.param((1, 2), "quaternion", id="wrong-length"),
        pytest.param((1.5, 0, 0, 0), "quaternion", id="float-entry"),
        pytest.param((True, 0, 0, 0), "quaternion", id="bool-entry"),
    ],
)
def test_scaled_rotation_rejects_invalid_quaternion(quaternion, match):
    with pytest.raises(CrystallographyValueError, match=match):
        ScaledRotation(
            denominator=1,
            matrix=np.eye(3, dtype=object),
            source="matrix",
            quaternion=quaternion,
        )


# --------------------------------------------------------------------------------------
# SmithDiagnostics
# --------------------------------------------------------------------------------------


def test_smith_diagnostics_normalizes_numpy_integer_fields():
    diagnostics = SmithDiagnostics(
        diagonal=(0, 1, np.int64(2)),  # type: ignore[ty:invalid-argument-type]
        kernel_moduli=(1, 1, np.int64(5)),  # type: ignore[ty:invalid-argument-type]
    )

    assert diagnostics.diagonal == (0, 1, 2)
    assert diagnostics.kernel_moduli == (1, 1, 5)
    assert all(type(value) is int for value in diagnostics.diagonal)
    assert all(type(value) is int for value in diagnostics.kernel_moduli)


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        pytest.param("diagonal", (-1, 1, 2), "diagonal", id="negative-diagonal"),
        pytest.param("diagonal", (1, 2), "diagonal", id="short-diagonal"),
        pytest.param("diagonal", (1.5, 1, 2), "diagonal", id="float-diagonal"),
        pytest.param("diagonal", (True, 1, 2), "diagonal", id="bool-diagonal"),
        pytest.param(
            "kernel_moduli",
            (1, 0, 5),
            "kernel_moduli",
            id="zero-kernel-modulus",
        ),
        pytest.param(
            "kernel_moduli",
            (1, -1, 5),
            "kernel_moduli",
            id="negative-kernel-modulus",
        ),
        pytest.param(
            "kernel_moduli",
            (1, 5),
            "kernel_moduli",
            id="short-kernel-moduli",
        ),
        pytest.param(
            "kernel_moduli",
            (1.5, 1, 5),
            "kernel_moduli",
            id="float-kernel-moduli",
        ),
        pytest.param(
            "kernel_moduli",
            (True, 1, 5),
            "kernel_moduli",
            id="bool-kernel-moduli",
        ),
    ],
)
def test_smith_diagnostics_rejects_invalid_fields(field_name, value, match):
    kwargs: dict[str, Any] = {
        "diagonal": (0, 1, 2),
        "kernel_moduli": (1, 1, 5),
    }
    kwargs[field_name] = value

    with pytest.raises(CrystallographyValueError, match=match):
        SmithDiagnostics(**kwargs)


# --------------------------------------------------------------------------------------
# CSLResult
# --------------------------------------------------------------------------------------


def test_csl_result_normalizes_sigma_and_freezes_basis_arrays():
    result = CSLResult(**_csl_result_kwargs(sigma=np.int64(1)))

    assert result.sigma == 1
    assert type(result.sigma) is int
    assert not result.basis.flags.writeable
    assert not result.basis_hnf.flags.writeable


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        pytest.param("sigma", 0, "sigma", id="zero-sigma"),
        pytest.param("sigma", -1, "sigma", id="negative-sigma"),
        pytest.param("sigma", 1.5, "sigma", id="float-sigma"),
        pytest.param("sigma", True, "sigma", id="bool-sigma"),
        pytest.param(
            "basis",
            np.ones((2, 2), dtype=object),
            "basis",
            id="basis-wrong-shape",
        ),
        pytest.param(
            "basis",
            _matrix_with_entry(1.5),
            "basis",
            id="basis-non-integer",
        ),
        pytest.param(
            "basis_hnf",
            np.ones((2, 2), dtype=object),
            "basis_hnf",
            id="basis-hnf-wrong-shape",
        ),
        pytest.param(
            "basis_hnf",
            _matrix_with_entry(True),
            "basis_hnf",
            id="basis-hnf-bool-entry",
        ),
    ],
)
def test_csl_result_rejects_invalid_fields(field_name, value, match):
    with pytest.raises(CrystallographyValueError, match=match):
        CSLResult(**_csl_result_kwargs(**{field_name: value}))


# --------------------------------------------------------------------------------------
# InPlaneBasis
# --------------------------------------------------------------------------------------


def test_inplane_basis_normalizes_plane_covector_and_freezes_arrays():
    inplane = InPlaneBasis(
        **_inplane_basis_kwargs(
            plane_covector=np.array([1, 1, 1], dtype=np.int64),
        )
    )

    assert not inplane.basis.flags.writeable
    assert not inplane.coefficients.flags.writeable
    assert inplane.plane_covector == (1, 1, 1)
    assert all(type(value) is int for value in inplane.plane_covector)


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        pytest.param(
            "basis",
            np.ones((3, 3), dtype=object),
            "basis",
            id="basis-wrong-shape",
        ),
        pytest.param(
            "basis",
            np.array([[1.5, 0], [0, 1], [1, 1]], dtype=object),
            "basis",
            id="basis-non-integer",
        ),
        pytest.param(
            "basis",
            np.array([[True, 0], [0, 1], [1, 1]], dtype=object),
            "basis",
            id="basis-bool-entry",
        ),
        pytest.param(
            "coefficients",
            np.ones((2, 2), dtype=object),
            "coefficients",
            id="coefficients-wrong-shape",
        ),
        pytest.param(
            "coefficients",
            np.array([[1, 0], [0, 1], [1.5, 1]], dtype=object),
            "coefficients",
            id="coefficients-non-integer",
        ),
        pytest.param(
            "plane_covector",
            (1, 1),
            "plane_covector",
            id="plane-covector-short",
        ),
        pytest.param(
            "plane_covector",
            (1.5, 1, 1),
            "plane_covector",
            id="plane-covector-non-integer",
        ),
        pytest.param(
            "plane_covector",
            (True, 1, 1),
            "plane_covector",
            id="plane-covector-bool-entry",
        ),
    ],
)
def test_inplane_basis_rejects_invalid_fields(field_name, value, match):
    with pytest.raises(CrystallographyValueError, match=match):
        InPlaneBasis(**_inplane_basis_kwargs(**{field_name: value}))


# --------------------------------------------------------------------------------------
# DSCBasis
# --------------------------------------------------------------------------------------


def test_dsc_basis_normalizes_integer_scalars_and_freezes_numerator():
    dsc = DSCBasis(**_dsc_basis_kwargs(sigma=np.int64(5)))

    assert dsc.denominator == 5
    assert dsc.sigma == 5
    assert type(dsc.denominator) is int
    assert type(dsc.sigma) is int
    assert not dsc.numerator.flags.writeable


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        pytest.param(
            "numerator",
            np.ones((2, 2), dtype=object),
            "numerator",
            id="numerator-wrong-shape",
        ),
        pytest.param(
            "numerator",
            _matrix_with_entry(1.5),
            "numerator",
            id="numerator-non-integer",
        ),
        pytest.param("denominator", 0, "denominator", id="zero-denominator"),
        pytest.param("denominator", -1, "denominator", id="negative-denominator"),
        pytest.param("denominator", 1.5, "denominator", id="float-denominator"),
        pytest.param("denominator", True, "denominator", id="bool-denominator"),
        pytest.param("sigma", 0, "sigma", id="zero-sigma"),
        pytest.param("sigma", -1, "sigma", id="negative-sigma"),
        pytest.param("sigma", 1.5, "sigma", id="float-sigma"),
        pytest.param("sigma", True, "sigma", id="bool-sigma"),
    ],
)
def test_dsc_basis_rejects_invalid_fields(field_name, value, match):
    with pytest.raises(CrystallographyValueError, match=match):
        DSCBasis(**_dsc_basis_kwargs(**{field_name: value}))


# --------------------------------------------------------------------------------------
# CoincidenceCheck
# --------------------------------------------------------------------------------------


def test_coincidence_check_normalizes_integer_diagnostics_and_freezes_residuals():
    check = CoincidenceCheck(
        **_coincidence_check_kwargs(
            det_basis=np.int64(5),
            sigma=np.int64(5),
        )
    )

    assert check.ok is True
    assert check.det_basis == 5
    assert check.sigma == 5
    assert type(check.det_basis) is int
    assert type(check.sigma) is int
    assert not check.residual_mod_N.flags.writeable


@pytest.mark.parametrize(
    "ok",
    [
        pytest.param(1, id="one"),
        pytest.param(0, id="zero"),
        pytest.param("true", id="string"),
        pytest.param(None, id="none"),
    ],
)
def test_coincidence_check_rejects_non_bool_ok(ok):
    with pytest.raises(CrystallographyValueError, match="ok must be a bool"):
        CoincidenceCheck(**_coincidence_check_kwargs(ok=ok))


def test_coincidence_check_normalizes_numpy_bool():
    check = CoincidenceCheck(**_coincidence_check_kwargs(ok=cast(bool, np.bool_(True))))

    assert check.ok is True


def test_coincidence_check_allows_sigma_none():
    check = CoincidenceCheck(**_coincidence_check_kwargs(sigma=None))

    assert check.sigma is None


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        pytest.param(
            "residual_mod_N",
            np.zeros((2, 2), dtype=object),
            "residual_mod_N",
            id="residual-wrong-shape",
        ),
        pytest.param(
            "residual_mod_N",
            _matrix_with_entry(1.5),
            "residual_mod_N",
            id="residual-non-integer",
        ),
        pytest.param("det_basis", -1, "det_basis", id="negative-det-basis"),
        pytest.param("det_basis", 1.5, "det_basis", id="float-det-basis"),
        pytest.param("det_basis", True, "det_basis", id="bool-det-basis"),
        pytest.param("sigma", 0, "sigma", id="zero-sigma"),
        pytest.param("sigma", -1, "sigma", id="negative-sigma"),
        pytest.param("sigma", 1.5, "sigma", id="float-sigma"),
        pytest.param("sigma", True, "sigma", id="bool-sigma"),
    ],
)
def test_coincidence_check_rejects_invalid_fields(field_name, value, match):
    with pytest.raises(CrystallographyValueError, match=match):
        CoincidenceCheck(**_coincidence_check_kwargs(**{field_name: value}))


# --------------------------------------------------------------------------------------
# Equality and repr behavior
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_obj",
    [
        pytest.param(
            lambda: ScaledRotation(
                denominator=1,
                matrix=np.eye(3, dtype=object),
                source="matrix",
            ),
            id="scaled-rotation",
        ),
        pytest.param(
            lambda: CSLResult(**_csl_result_kwargs()),
            id="csl-result",
        ),
        pytest.param(
            lambda: InPlaneBasis(**_inplane_basis_kwargs()),
            id="inplane-basis",
        ),
        pytest.param(
            lambda: DSCBasis(**_dsc_basis_kwargs()),
            id="dsc-basis",
        ),
        pytest.param(
            lambda: CoincidenceCheck(**_coincidence_check_kwargs()),
            id="coincidence-check",
        ),
    ],
)
def test_array_dataclass_instances_are_identity_compared(make_obj):
    a = make_obj()
    b = make_obj()

    assert a is not b
    assert a != b


@pytest.mark.parametrize(
    ("obj", "expected_fields"),
    [
        (
            ScaledRotation(
                denominator=1,
                matrix=np.eye(3, dtype=object),
                source="matrix",
            ),
            ("matrix", "shape=(3, 3)", "dtype=object", "writeable=False"),
        ),
        (
            CSLResult(**_csl_result_kwargs()),
            ("basis", "basis_hnf", "shape=(3, 3)", "dtype=object", "writeable=False"),
        ),
        (
            InPlaneBasis(
                basis=np.ones((3, 2), dtype=object),
                coefficients=np.eye(3, 2, dtype=object),
                plane_covector=(1, 0, 0),
            ),
            (
                "basis",
                "coefficients",
                "shape=(3, 2)",
                "dtype=object",
                "writeable=False",
            ),
        ),
        (
            DSCBasis(
                numerator=np.eye(3, dtype=object),
                denominator=1,
                sigma=1,
            ),
            ("numerator", "shape=(3, 3)", "dtype=object", "writeable=False"),
        ),
        (
            CoincidenceCheck(
                ok=True,
                residual_mod_N=np.zeros((3, 3), dtype=object),
                det_basis=1,
                sigma=1,
            ),
            ("residual_mod_N", "shape=(3, 3)", "dtype=object", "writeable=False"),
        ),
    ],
)
def test_array_dataclass_repr_is_single_line_and_includes_array_metadata(
    obj,
    expected_fields,
):
    text = repr(obj)

    assert "\n" not in text
    for expected in expected_fields:
        assert expected in text
