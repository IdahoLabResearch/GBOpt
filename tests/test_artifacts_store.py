# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import json

import pytest

from GBOpt.artifacts.policy import ArtifactRetentionPolicy
from GBOpt.artifacts.rules import KeepBest, KeepRange
from GBOpt.artifacts.store import ArtifactStore, ArtifactStoreError
from GBOpt.artifacts.types import ArtifactPin, ArtifactStatus, RetentionCandidate


def _candidate(candidate_id, objective, **properties):
    return RetentionCandidate(candidate_id, 0, objective, properties=properties)


def _policy(*rules):
    return ArtifactRetentionPolicy(rules=rules, prune=True)


def test_operational_pin_blocks_pruning_until_released():
    store = ArtifactStore(_policy())
    store.register_candidate(_candidate("a", 1.0))

    assert store.is_prunable("a")
    store.pin("a", ArtifactPin.CANDIDATE_CHECKPOINT)
    assert not store.is_prunable("a")
    assert store.record("a").status is ArtifactStatus.PINNED
    store.release_pin("a", ArtifactPin.CANDIDATE_CHECKPOINT)
    assert store.is_prunable("a")


def test_pin_and_release_are_idempotent():
    store = ArtifactStore(_policy())
    store.register_candidate(_candidate("a", 1.0))

    store.pin("a", ArtifactPin.RUN_CHECKPOINT)
    store.pin("a", ArtifactPin.RUN_CHECKPOINT)
    store.release_pin("a", ArtifactPin.CARRYOVER_CACHE)

    assert store.pins("a") == (ArtifactPin.RUN_CHECKPOINT,)


def test_scientific_reason_blocks_pruning_until_removed():
    store = ArtifactStore(_policy())
    store.register_candidate(_candidate("a", 1.0))

    store.add_retention_reason("a", "rule:manual")
    assert not store.is_prunable("a")
    store.remove_retention_reason("a", "rule:manual")
    assert store.is_prunable("a")


def test_candidate_becomes_prunable_only_after_final_independent_reference_disappears():
    store = ArtifactStore(_policy())
    store.register_candidate(_candidate("a", 1.0))
    store.pin("a", ArtifactPin.BEST_RESULT)
    store.add_retention_reason("a", "rule:elite")
    store.add_retention_reason("a", "rule:density")

    store.remove_retention_reason("a", "rule:elite")
    assert not store.is_prunable("a")
    store.release_pin("a", ArtifactPin.BEST_RESULT)
    assert not store.is_prunable("a")
    store.remove_retention_reason("a", "rule:density")
    assert store.is_prunable("a")


def test_multiple_policy_rules_retain_same_candidate_as_independent_reasons():
    policy = _policy(
        KeepBest(name="elite", property="objective", direction="min", count=1),
        KeepRange(
            name="density",
            property="density",
            minimum=10.0,
            maximum=11.0,
            count=1,
            rank_by="objective",
            direction="min",
        ),
    )
    store = ArtifactStore(policy)

    store.register_candidate(_candidate("a", 1.0, density=10.5))

    assert store.retention_reasons("a") == ("rule:density", "rule:elite")


def test_archive_membership_eviction_removes_only_the_evicted_rule_reason():
    policy = _policy(
        KeepBest(name="elite", property="objective", direction="min", count=1),
        KeepRange(
            name="density",
            property="density",
            minimum=10.0,
            maximum=11.0,
            count=2,
            rank_by="objective",
            direction="min",
        ),
    )
    store = ArtifactStore(policy)
    store.register_candidate(_candidate("old", 2.0, density=10.5))
    store.register_candidate(_candidate("new", 1.0, density=10.5))

    assert store.retention_reasons("old") == ("rule:density",)
    assert store.retention_reasons("new") == ("rule:density", "rule:elite")
    assert not store.is_prunable("old")


def test_best_result_pin_replacement_releases_superseded_best():
    store = ArtifactStore(_policy())
    store.register_candidate(_candidate("old", 2.0))
    store.register_candidate(_candidate("new", 1.0))

    store.replace_pin(ArtifactPin.BEST_RESULT, "old")
    store.replace_pin(ArtifactPin.BEST_RESULT, "new")

    assert ArtifactPin.BEST_RESULT not in store.pins("old")
    assert store.is_prunable("old")
    assert store.pins("new") == (ArtifactPin.BEST_RESULT,)
    assert not store.is_prunable("new")


def test_policy_none_preserves_keep_all_behavior():
    store = ArtifactStore(policy=None)
    store.register_candidate(_candidate("a", 1.0))

    assert not store.pruning_enabled
    assert not store.is_prunable("a")
    assert store.prunable_candidate_ids() == ()


def test_explicit_keep_all_policy_preserves_keep_all_behavior():
    store = ArtifactStore(ArtifactRetentionPolicy.keep_all())
    store.register_candidate(_candidate("a", 1.0))

    assert not store.is_prunable("a")


def test_duplicate_registration_is_idempotent_only_for_identical_identity_state():
    store = ArtifactStore(_policy())
    candidate = _candidate("a", 1.0)

    first = store.register_candidate(candidate, source_path="eval/a.data")
    second = store.register_candidate(candidate, source_path="eval/a.data")

    assert first == second
    with pytest.raises(ArtifactStoreError, match="different state"):
        store.register_candidate(_candidate("a", 2.0), source_path="eval/a.data")
    with pytest.raises(ArtifactStoreError, match="different source path"):
        store.register_candidate(candidate, source_path="other/a.data")


def test_failed_policy_evaluation_leaves_new_candidate_unregistered():
    policy = _policy(
        KeepBest(name="density", property="density", direction="max", count=1)
    )
    store = ArtifactStore(policy)

    with pytest.raises(ArtifactStoreError, match="could not register"):
        store.register_candidate(_candidate("a", 1.0))

    assert len(store) == 0


def test_serialized_state_is_deterministic_across_registration_order():
    policy = _policy(
        KeepBest(name="elite", property="objective", direction="min", count=1)
    )
    first = ArtifactStore(policy)
    first.register_candidate(_candidate("b", 2.0), source_path="b.data")
    first.register_candidate(_candidate("a", 1.0), source_path="a.data")
    first.pin("b", ArtifactPin.CARRYOVER_CACHE)

    second = ArtifactStore(policy)
    second.register_candidate(_candidate("a", 1.0), source_path="a.data")
    second.register_candidate(_candidate("b", 2.0), source_path="b.data")
    second.pin("b", ArtifactPin.CARRYOVER_CACHE)

    assert first.to_state() == second.to_state()
    assert json.dumps(first.to_state(), sort_keys=True) == json.dumps(
        second.to_state(), sort_keys=True
    )


def test_store_state_round_trip_restores_multiple_references():
    policy = _policy(
        KeepBest(name="elite", property="objective", direction="min", count=1)
    )
    store = ArtifactStore(policy)
    store.register_candidate(_candidate("a", 1.0), source_path="a.data")
    store.register_candidate(_candidate("b", 2.0), source_path="b.data")
    store.pin("b", ArtifactPin.CARRYOVER_CACHE)
    store.add_retention_reason("b", "diagnostic")

    state = json.loads(json.dumps(store.to_state(), sort_keys=True))
    restored = ArtifactStore.from_state(state, policy=policy)

    assert restored.to_state() == store.to_state()
    assert restored.retention_reasons("b") == ("diagnostic",)
    assert restored.pins("b") == (ArtifactPin.CARRYOVER_CACHE,)


def test_store_restore_rejects_policy_signature_mismatch():
    first_policy = _policy(
        KeepBest(name="elite", property="objective", direction="min", count=1)
    )
    changed_policy = _policy(
        KeepBest(name="elite", property="objective", direction="min", count=2)
    )
    store = ArtifactStore(first_policy)
    store.register_candidate(_candidate("a", 1.0))

    with pytest.raises(ArtifactStoreError, match="signature mismatch"):
        ArtifactStore.from_state(store.to_state(), policy=changed_policy)


def test_prunable_candidate_ids_are_lexically_sorted():
    store = ArtifactStore(_policy())
    store.register_candidate(_candidate("z", 1.0))
    store.register_candidate(_candidate("a", 1.0))

    assert store.prunable_candidate_ids() == ("a", "z")


def test_retained_source_becomes_prunable_after_canonical_archive_exists():
    policy = ArtifactRetentionPolicy(
        rules=(KeepBest(name="best", property="objective", direction="min", count=1),),
        prune=True,
    )
    store = ArtifactStore(policy=policy)
    candidate = _candidate("candidate-a", 1.0)
    store.register_candidate(candidate, source_path="/tmp/source.data")

    assert store.source_is_prunable(candidate.candidate_id) is False
    store.set_archive_path(candidate.candidate_id, "/tmp/archive.data")
    assert store.source_is_prunable(candidate.candidate_id) is True


def test_best_pin_can_be_satisfied_by_canonical_archive_for_source_pruning():
    policy = ArtifactRetentionPolicy(rules=(), prune=True)
    store = ArtifactStore(policy=policy)
    candidate = _candidate("candidate-a", 1.0)
    store.register_candidate(candidate, source_path="/tmp/source.data")
    store.replace_pin(ArtifactPin.BEST_RESULT, candidate.candidate_id)

    assert store.source_is_prunable(candidate.candidate_id) is False
    store.set_archive_path(candidate.candidate_id, "/tmp/archive.data")
    assert store.source_is_prunable(candidate.candidate_id) is True


def test_archive_path_round_trips_through_store_state():
    policy = ArtifactRetentionPolicy(rules=(), prune=True)
    store = ArtifactStore(policy=policy)
    candidate = _candidate("candidate-a", 1.0)
    store.register_candidate(candidate, source_path="/tmp/source.data")
    store.set_archive_path(candidate.candidate_id, "/tmp/archive.data")

    restored = ArtifactStore.from_state(store.to_state(), policy=policy)

    assert restored.record(candidate.candidate_id).archive_path == "/tmp/archive.data"
