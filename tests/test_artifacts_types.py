# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import json

import numpy as np
import pytest

from GBOpt.artifacts.types import (
    BUILTIN_PROPERTY_NAMES,
    ArtifactPin,
    ArtifactRecord,
    ArtifactStatus,
    ArtifactValueError,
    CandidatePropertyContext,
    RetentionCandidate,
    normalize_property_mapping,
    normalize_retention_value,
    retention_value_from_state,
    retention_value_to_state,
)

_ATOM_DTYPE = np.dtype([("name", "U2"), ("x", float), ("y", float), ("z", float)])


def _atoms():
    return np.array(
        [("U", 1.0, 2.0, 3.0), ("O", 4.0, 5.0, 6.0)],
        dtype=_ATOM_DTYPE,
    )


def test_retention_value_normalizes_numpy_scalars_and_nested_tuples():
    result = normalize_retention_value((np.int64(2), np.float64(3.5), np.bool_(True)))

    assert result == (2, 3.5, True)
    assert tuple(type(value) for value in result) == (int, float, bool)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_retention_value_rejects_nonfinite_numeric_values(value):
    with pytest.raises(ArtifactValueError, match="finite"):
        normalize_retention_value(value)


@pytest.mark.parametrize("value", [[1, 2], {"a": 1}, np.array([1, 2]), object()])
def test_retention_value_rejects_mutable_or_unsupported_values(value):
    with pytest.raises(ArtifactValueError, match="unsupported type"):
        normalize_retention_value(value)


def test_retention_value_state_round_trip_preserves_tuples():
    value = ("U", (2, 3.0), True)

    state = retention_value_to_state(value)
    restored = retention_value_from_state(json.loads(json.dumps(state)))

    assert restored == value
    assert isinstance(restored, tuple)
    assert isinstance(restored[1], tuple)


def test_property_mapping_is_read_only_and_lexically_ordered():
    properties = normalize_property_mapping({"z": 2, "a": 1})

    assert list(properties) == ["a", "z"]
    with pytest.raises(TypeError):
        properties["x"] = 3  # ty: ignore[invalid-assignment]


def test_property_provider_namespace_rejects_reserved_names():
    assert "objective" in BUILTIN_PROPERTY_NAMES

    with pytest.raises(ArtifactValueError, match="reserved property 'objective'"):
        normalize_property_mapping({"objective": 1.0}, reject_reserved=True)


def test_retention_candidate_injects_identity_objective_and_generation_properties():
    candidate = RetentionCandidate(
        candidate_id="GA_123_g17_c4",
        generation=np.int64(17),
        objective=np.float64(1.25),
        properties={"mass_density": 10.9},
        lineage=("GA_123_g16_c1",),
    )

    assert candidate.candidate_id == "GA_123_g17_c4"
    assert candidate.generation == 17
    assert candidate.objective == 1.25
    assert candidate.properties["candidate_id"] == candidate.candidate_id
    assert candidate.properties["generation"] == 17
    assert candidate.properties["objective"] == 1.25
    assert candidate.properties["mass_density"] == 10.9


@pytest.mark.parametrize(
    ("generation", "objective"),
    [(True, 1.0), (0, True)],
    ids=("boolean-generation", "boolean-objective"),
)
def test_retention_candidate_rejects_booleans_for_numeric_fields(generation, objective):
    with pytest.raises(ArtifactValueError, match="non-Boolean"):
        RetentionCandidate(
            candidate_id="candidate-a",
            generation=generation,
            objective=objective,
        )


def test_retention_candidate_rejects_conflicting_reserved_property():
    with pytest.raises(ArtifactValueError, match="conflicts"):
        RetentionCandidate(
            candidate_id="candidate-a",
            generation=0,
            objective=1.0,
            properties={"objective": 2.0},
        )


def test_retention_candidate_state_is_json_safe_and_round_trips():
    candidate = RetentionCandidate(
        candidate_id="candidate-a",
        generation=2,
        objective=1.0,
        properties={"composition": (("O", 2), ("U", 1))},
        lineage=("candidate-parent",),
    )

    state = json.loads(json.dumps(candidate.to_state(), sort_keys=True))
    restored = RetentionCandidate.from_state(state)

    assert restored == candidate
    assert restored.properties["composition"] == (("O", 2), ("U", 1))


def test_candidate_property_context_copies_and_freezes_physical_arrays():
    atoms = _atoms()
    box = np.array([[0.0, 10.0], [0.0, 8.0], [0.0, 6.0]])
    labels = np.array([0, 1], dtype=np.int8)

    context = CandidatePropertyContext(
        candidate_id="candidate-a",
        generation=0,
        objective=1.0,
        atoms=atoms,
        box_dims=box,
        grain_labels=labels,
        gb_plane_x=5.0,
    )
    atoms[0]["x"] = 99.0
    box[0, 1] = 99.0
    labels[0] = 1

    assert context.atoms[0]["x"] == pytest.approx(1.0)
    assert context.box_dims[0, 1] == pytest.approx(10.0)
    assert context.grain_labels is not None
    assert context.grain_labels[0] == 0
    assert not context.atoms.flags.writeable
    assert not context.box_dims.flags.writeable
    assert not context.grain_labels.flags.writeable


def test_candidate_property_context_rejects_misaligned_labels():
    with pytest.raises(ArtifactValueError, match="aligned"):
        CandidatePropertyContext(
            candidate_id="candidate-a",
            generation=0,
            objective=1.0,
            atoms=_atoms(),
            box_dims=np.array([[0.0, 10.0], [0.0, 8.0], [0.0, 6.0]]),
            grain_labels=np.array([0], dtype=np.int8),
            gb_plane_x=5.0,
        )


def test_candidate_property_context_rejects_incomplete_or_nonfinite_atom_rows():
    missing_coordinate = np.array(
        [("U", 1.0, 2.0)],
        dtype=[("name", "U2"), ("x", float), ("y", float)],
    )
    nonfinite = _atoms()
    nonfinite[1]["z"] = np.inf

    with pytest.raises(ArtifactValueError, match="structured fields"):
        CandidatePropertyContext(
            candidate_id="candidate-a",
            generation=0,
            objective=1.0,
            atoms=missing_coordinate,
            box_dims=np.array([[0.0, 10.0], [0.0, 8.0], [0.0, 6.0]]),
            gb_plane_x=5.0,
        )
    with pytest.raises(ArtifactValueError, match="coordinates must be finite"):
        CandidatePropertyContext(
            candidate_id="candidate-a",
            generation=0,
            objective=1.0,
            atoms=nonfinite,
            box_dims=np.array([[0.0, 10.0], [0.0, 8.0], [0.0, 6.0]]),
            gb_plane_x=5.0,
        )


@pytest.mark.parametrize(
    "labels",
    [np.array([0.0, 1.0]), np.array([False, True])],
    ids=("floating-point", "boolean"),
)
def test_candidate_property_context_rejects_coercive_grain_labels(labels):
    with pytest.raises(ArtifactValueError, match="integer dtype"):
        CandidatePropertyContext(
            candidate_id="candidate-a",
            generation=0,
            objective=1.0,
            atoms=_atoms(),
            box_dims=np.array([[0.0, 10.0], [0.0, 8.0], [0.0, 6.0]]),
            grain_labels=labels,
            gb_plane_x=5.0,
        )


def test_candidate_property_context_rejects_unknown_grain_label():
    with pytest.raises(ArtifactValueError, match="only left/right labels"):
        CandidatePropertyContext(
            candidate_id="candidate-a",
            generation=0,
            objective=1.0,
            atoms=_atoms(),
            box_dims=np.array([[0.0, 10.0], [0.0, 8.0], [0.0, 6.0]]),
            grain_labels=np.array([0, 2], dtype=np.int8),
            gb_plane_x=5.0,
        )


def test_candidate_property_context_rejects_boolean_box_bounds():
    with pytest.raises(ArtifactValueError, match="real numeric"):
        CandidatePropertyContext(
            candidate_id="candidate-a",
            generation=0,
            objective=1.0,
            atoms=_atoms(),
            box_dims=np.array(
                [[False, True], [False, True], [False, True]], dtype=bool
            ),
            gb_plane_x=0.5,
        )


def test_candidate_property_context_requires_plane_strictly_inside_box():
    for plane in (0.0, 10.0):
        with pytest.raises(ArtifactValueError, match="strictly inside"):
            CandidatePropertyContext(
                candidate_id="candidate-a",
                generation=0,
                objective=1.0,
                atoms=_atoms(),
                box_dims=np.array([[0.0, 10.0], [0.0, 8.0], [0.0, 6.0]]),
                gb_plane_x=plane,
            )


def test_artifact_record_status_preserves_independent_pin_and_rule_reasons():
    candidate = RetentionCandidate("candidate-a", 0, 1.0)
    record = ArtifactRecord(
        candidate=candidate,
        pins=(ArtifactPin.BEST_RESULT, ArtifactPin.BEST_RESULT),
        retention_reasons=("rule:elite", "rule:elite", "rule:density"),
    )

    assert record.pins == (ArtifactPin.BEST_RESULT,)
    assert record.retention_reasons == ("rule:density", "rule:elite")
    assert record.status is ArtifactStatus.PINNED_AND_RETAINED
