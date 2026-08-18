# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import json

import numpy as np
import pytest

from GBOpt.artifacts.policy import ArtifactPolicyError, ArtifactRetentionPolicy
from GBOpt.artifacts.rules import KeepBest, KeepIf, KeepRange
from GBOpt.artifacts.types import CandidatePropertyContext


_ATOM_DTYPE = np.dtype([("name", "U2"), ("x", float), ("y", float), ("z", float)])


def _context(candidate_id="candidate-a", objective=1.0):
    atoms = np.array(
        [
            ("U", 1.0, 2.0, 3.0),
            ("O", 4.0, 5.0, 6.0),
            ("O", 2.0, 3.0, 4.0),
        ],
        dtype=_ATOM_DTYPE,
    )
    return CandidatePropertyContext(
        candidate_id=candidate_id,
        generation=2,
        objective=objective,
        atoms=atoms,
        box_dims=np.array([[0.0, 10.0], [0.0, 8.0], [0.0, 6.0]]),
        grain_labels=np.array([0, 1, 1], dtype=np.int8),
        gb_plane_x=5.0,
    )


def _provider(context):
    return {"mass_density": 10.9, "density_class": "high"}


def test_policy_rejects_duplicate_rule_names():
    with pytest.raises(ArtifactPolicyError, match="unique"):
        ArtifactRetentionPolicy(
            rules=(
                KeepBest(name="same", property="objective", direction="min", count=1),
                KeepBest(name="same", property="objective", direction="max", count=1),
            )
        )


def test_policy_sorts_rules_by_explicit_name_for_deterministic_identity():
    a = KeepBest(name="a", property="objective", direction="min", count=1)
    z = KeepBest(name="z", property="objective", direction="max", count=1)

    first = ArtifactRetentionPolicy(rules=(z, a), prune=True)
    second = ArtifactRetentionPolicy(rules=(a, z), prune=True)

    assert first.rule_names == ("a", "z")
    assert first.to_state() == second.to_state()
    assert first.signature == second.signature


def test_property_provider_requires_explicit_version_contract():
    with pytest.raises(ArtifactPolicyError, match="property_provider_version"):
        ArtifactRetentionPolicy(property_provider=_provider)



def test_lambda_property_provider_requires_explicit_persistent_name():
    with pytest.raises(ArtifactPolicyError, match="property_provider_name"):
        ArtifactRetentionPolicy(
            property_provider=lambda context: {},
            property_provider_version="1",
        )


def test_property_provider_identity_and_config_are_in_policy_signature():
    first = ArtifactRetentionPolicy(
        property_provider=_provider,
        property_provider_version="1",
        property_provider_name="uo2_properties",
        property_provider_config={"method": "bulk_reference_v1"},
    )
    same = ArtifactRetentionPolicy(
        property_provider=_provider,
        property_provider_version="1",
        property_provider_name="uo2_properties",
        property_provider_config={"method": "bulk_reference_v1"},
    )
    changed = ArtifactRetentionPolicy(
        property_provider=_provider,
        property_provider_version="2",
        property_provider_name="uo2_properties",
        property_provider_config={"method": "bulk_reference_v1"},
    )

    assert first.signature == same.signature
    assert first.signature != changed.signature


def test_policy_state_never_serializes_callback_objects():
    policy = ArtifactRetentionPolicy(
        rules=(
            KeepIf(
                name="custom",
                predicate=lambda candidate: True,
                version="3",
                count=1,
                rank_by="objective",
                direction="min",
            ),
        ),
        property_provider=_provider,
        property_provider_version="5",
        property_provider_name="uo2_properties",
    )

    encoded = json.dumps(policy.to_state(), sort_keys=True)

    assert "<function" not in encoded
    assert "predicate" not in encoded
    assert '"version": "3"' in encoded
    assert "uo2_properties" in encoded


def test_candidate_from_context_adds_guaranteed_builtin_properties():
    policy = ArtifactRetentionPolicy.keep_all()

    candidate = policy.candidate_from_context(_context())

    assert candidate.properties["candidate_id"] == "candidate-a"
    assert candidate.properties["generation"] == 2
    assert candidate.properties["objective"] == 1.0
    assert candidate.properties["atom_count"] == 3
    assert candidate.properties["composition"] == (("O", 2), ("U", 1))
    assert candidate.properties["cell_volume"] == pytest.approx(480.0)


def test_candidate_from_context_uses_validated_relaxed_context_for_provider():
    seen = {}

    def provider(context):
        seen["writeable"] = context.atoms.flags.writeable
        seen["x"] = context.atoms[0]["x"]
        return {"mass_density": 10.9}

    policy = ArtifactRetentionPolicy(
        property_provider=provider,
        property_provider_version="1",
        property_provider_name="test_provider",
    )

    candidate = policy.candidate_from_context(_context())

    assert seen == {"writeable": False, "x": 1.0}
    assert candidate.properties["mass_density"] == 10.9


def test_malformed_property_provider_output_fails_explicitly():
    policy = ArtifactRetentionPolicy(
        property_provider=lambda context: [1, 2],
        property_provider_version="1",
        property_provider_name="bad_provider",
    )

    with pytest.raises(ArtifactPolicyError, match="provider output"):
        policy.candidate_from_context(_context())


def test_nonfinite_property_provider_value_fails_explicitly():
    policy = ArtifactRetentionPolicy(
        property_provider=lambda context: {"mass_density": np.nan},
        property_provider_version="1",
        property_provider_name="bad_provider",
    )

    with pytest.raises(ArtifactPolicyError, match="finite"):
        policy.candidate_from_context(_context())


def test_property_provider_cannot_overwrite_reserved_builtin_namespace():
    policy = ArtifactRetentionPolicy(
        property_provider=lambda context: {"objective": -99.0},
        property_provider_version="1",
        property_provider_name="bad_provider",
    )

    with pytest.raises(ArtifactPolicyError, match="reserved property 'objective'"):
        policy.candidate_from_context(_context())


def test_successful_candidate_missing_active_rule_property_fails_during_acquisition():
    policy = ArtifactRetentionPolicy(
        rules=(
            KeepBest(
                name="density",
                property="mass_density",
                direction="max",
                count=1,
            ),
        )
    )

    with pytest.raises(ArtifactPolicyError, match="mass_density"):
        policy.candidate_from_context(_context())



def test_candidate_from_context_translates_invalid_lineage_to_policy_error():
    policy = ArtifactRetentionPolicy.keep_all()

    with pytest.raises(ArtifactPolicyError, match="retention candidate state"):
        policy.candidate_from_context(_context(), lineage=("",))


def test_policy_evaluates_multiple_rules_independently():
    policy = ArtifactRetentionPolicy(
        rules=(
            KeepBest(name="elite", property="objective", direction="min", count=1),
            KeepRange(
                name="density_window",
                property="mass_density",
                minimum=10.5,
                maximum=11.0,
                count=2,
                rank_by="objective",
                direction="min",
            ),
        ),
        property_provider=_provider,
        property_provider_version="1",
        property_provider_name="uo2_properties",
    )
    candidates = [
        policy.candidate_from_context(_context("a", 2.0)),
        policy.candidate_from_context(_context("b", 1.0)),
    ]

    assert policy.evaluate(candidates) == {
        "density_window": ("b", "a"),
        "elite": ("b",),
    }


def test_keep_all_policy_is_explicitly_non_pruning_and_stable():
    first = ArtifactRetentionPolicy.keep_all()
    second = ArtifactRetentionPolicy.keep_all()

    assert first.prune is False
    assert first.rules == ()
    assert first.signature == second.signature
