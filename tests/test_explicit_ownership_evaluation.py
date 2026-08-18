# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Focused tests for explicit-ownership evaluator contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from GBOpt.FileGrainOwnership import GrainOwnershipError
from GBOpt.GBManipulator import GBManipulatorError, ParentError
from GBOpt._explicit_ownership_evaluation import (
    CandidateEvaluation,
    ExplicitOwnershipEvaluator,
)


def _evaluator(**overrides):
    arguments = {
        "GB": SimpleNamespace(
            unit_cell=SimpleNamespace(type_map={}),
            gb_thickness=1.0,
        ),
        "scalar_energy_func": lambda *args: None,
        "batch_energy_func": None,
        "local_random": np.random.default_rng(0),
        "penalty": 1.0e30,
    }
    arguments.update(overrides)
    return ExplicitOwnershipEvaluator(**arguments)


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        pytest.param(
            {"scalar_energy_func": None},
            TypeError,
            "scalar_energy_func must be callable",
            id="scalar-callback",
        ),
        pytest.param(
            {"batch_energy_func": object()},
            TypeError,
            "batch_energy_func must be callable or None",
            id="batch-callback",
        ),
        pytest.param(
            {"local_random": object()},
            TypeError,
            "local_random must be a numpy.random.Generator",
            id="random-generator",
        ),
        pytest.param(
            {"penalty": True},
            TypeError,
            "penalty must be a non-Boolean real scalar",
            id="boolean-penalty",
        ),
        pytest.param(
            {"penalty": "1.0"},
            TypeError,
            "penalty must be a non-Boolean real scalar",
            id="string-penalty",
        ),
        pytest.param(
            {"penalty": np.nan},
            ValueError,
            "penalty must be finite",
            id="nan-penalty",
        ),
        pytest.param(
            {"penalty": np.inf},
            ValueError,
            "penalty must be finite",
            id="infinite-penalty",
        ),
    ],
)
def test_evaluator_validates_configuration(overrides, error, match):
    with pytest.raises(error, match=match):
        _evaluator(**overrides)


def test_evaluator_requires_optimizer_supplied_penalty():
    with pytest.raises(TypeError, match="penalty"):
        ExplicitOwnershipEvaluator(
            GB=SimpleNamespace(),
            scalar_energy_func=lambda *args: None,
            batch_energy_func=None,
            local_random=np.random.default_rng(0),
        )


def test_candidate_evaluation_normalizes_python_scalar_fields():
    result = CandidateEvaluation(
        input_index=np.int64(-1),
        energy=np.float64(2.5),
        structure_path=None,
        mapping=None,
        manipulator=None,
        success=False,
        failure_reason="failed",
    )

    assert result.input_index == -1
    assert type(result.input_index) is int
    assert result.energy == pytest.approx(2.5)
    assert type(result.energy) is float


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        pytest.param(
            {"input_index": True},
            TypeError,
            "input_index must be a non-Boolean integer",
            id="boolean-index",
        ),
        pytest.param(
            {"energy": np.nan},
            ValueError,
            "energy must be finite",
            id="nonfinite-energy",
        ),
        pytest.param(
            {"success": np.bool_(False)},
            TypeError,
            "success must be a bool",
            id="numpy-success",
        ),
        pytest.param(
            {"failure_reason": None},
            ValueError,
            "failed evaluation requires a failure reason",
            id="missing-failure-reason",
        ),
        pytest.param(
            {"manipulator": object()},
            ValueError,
            "failed evaluation must not include a manipulator",
            id="failed-with-manipulator",
        ),
    ],
)
def test_failed_candidate_evaluation_rejects_incoherent_state(kwargs, error, match):
    arguments = {
        "input_index": 0,
        "energy": 1.0e30,
        "structure_path": None,
        "mapping": None,
        "manipulator": None,
        "success": False,
        "failure_reason": "failed",
    }
    arguments.update(kwargs)

    with pytest.raises(error, match=match):
        CandidateEvaluation(**arguments)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param(
            {"structure_path": None},
            "successful evaluation requires",
            id="missing-path",
        ),
        pytest.param(
            {"mapping": None},
            "successful evaluation requires",
            id="missing-mapping",
        ),
        pytest.param(
            {"manipulator": None},
            "successful evaluation requires",
            id="missing-manipulator",
        ),
        pytest.param(
            {"failure_reason": "unexpected"},
            "successful evaluation must not include a failure reason",
            id="failure-reason",
        ),
    ],
)
def test_successful_candidate_evaluation_rejects_incoherent_state(kwargs, match):
    arguments = {
        "input_index": 0,
        "energy": 1.25,
        "structure_path": "/tmp/candidate.data",
        "mapping": object(),
        "manipulator": object(),
        "success": True,
        "failure_reason": None,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=match):
        CandidateEvaluation(**arguments)


def test_invalid_structure_path_becomes_typed_failed_evaluation():
    evaluator = _evaluator(penalty=123.0)

    result = evaluator._record_result(
        input_index=4,
        mapping=object(),
        energy=1.0,
        structure_path="invalid\x00path",
    )

    assert result.success is False
    assert result.energy == pytest.approx(123.0)
    assert result.manipulator is None
    assert result.structure_path is None
    assert "invalid structure path" in result.failure_reason
    assert "ValueError" in result.failure_reason


@pytest.mark.parametrize("error_type", [ParentError, GBManipulatorError])
def test_reload_mapping_translates_reconstruction_errors(
    monkeypatch,
    error_type,
):
    evaluator = _evaluator()

    def fail_reload(*args, **kwargs):
        raise error_type("invalid reconstructed candidate")

    monkeypatch.setattr(
        "GBOpt._explicit_ownership_evaluation.reload_explicit_manipulator",
        fail_reload,
    )

    with pytest.raises(
        GrainOwnershipError,
        match="could not reconstruct a valid candidate",
    ) as exc_info:
        evaluator._reload_mapping("candidate.data", object())

    assert isinstance(exc_info.value.__cause__, error_type)
