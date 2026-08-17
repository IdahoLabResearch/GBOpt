# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Checkpoint persistence and candidate-sidecar recovery contracts."""

import json
from pathlib import Path

import numpy as np
import pytest

from GBOpt.Checkpoint import (
    CandidateCheckpoint,
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointStore,
    CheckpointValueError,
    _wrap_batch_func_with_checkpoint,
)


def _make_state(index=1):
    return {
        "schema_version": 1,
        "minimizer": "TestMinimizer",
        "progress_unit": "step",
        "progress_index": index,
        "best_energy": -1.0,
        "best_dump": None,
        "rng_state": {},
        "run_params": {},
        "state": {"value": index},
    }


@pytest.fixture
def checkpoint_path(tmp_path):
    return tmp_path / "run.json"


@pytest.fixture
def candidate_uids():
    return ["run_i0_c0", "run_i0_c1", "run_i0_c2"]


def _fresh_candidate_checkpoint(main_path, unique_ids, *, fmt="json", iteration=0):
    suffix = ".pkl" if fmt == "pickle" else main_path.suffix
    base = main_path.with_suffix(suffix)
    path = base.with_suffix(f".iter{iteration}{base.suffix}")
    return CandidateCheckpoint(path, fmt, iteration, unique_ids)


# ---------------------------------------------------------------------------
# CheckpointStore
# ---------------------------------------------------------------------------


def test_disabled_store_is_noop(checkpoint_path):
    store = CheckpointStore.disabled()

    assert not store.enabled
    assert not store.exists
    store.save_if_due(1, lambda: _make_state())
    store.save_final(_make_state())
    store.delete()

    assert store.load() is None
    assert not checkpoint_path.exists()


def test_from_optional_returns_disabled_when_path_is_none():
    assert not CheckpointStore.from_optional(None).enabled


def test_from_optional_rejects_invalid_format(checkpoint_path):
    with pytest.raises(CheckpointValueError, match="fmt must be"):
        CheckpointStore.from_optional(checkpoint_path, fmt="invalid")


@pytest.mark.parametrize("interval", [0, -1])
def test_from_optional_rejects_nonpositive_interval(checkpoint_path, interval):
    with pytest.raises(CheckpointValueError, match="interval must be >= 1"):
        CheckpointStore.from_optional(checkpoint_path, interval=interval)


def test_save_if_due_respects_interval(checkpoint_path):
    store = CheckpointStore.from_optional(checkpoint_path, interval=3)

    store.save_if_due(1, lambda: _make_state(1))
    store.save_if_due(2, lambda: _make_state(2))
    assert not checkpoint_path.exists()

    store.save_if_due(3, lambda: _make_state(3))

    assert json.loads(checkpoint_path.read_text())["progress_index"] == 3


def test_state_callable_is_not_called_when_save_is_not_due(checkpoint_path):
    calls = 0

    def counting_state():
        nonlocal calls
        calls += 1
        return _make_state()

    store = CheckpointStore.from_optional(checkpoint_path, interval=5)
    store.save_if_due(1, counting_state)
    store.save_if_due(2, counting_state)

    assert calls == 0


def test_save_final_bypasses_interval(checkpoint_path):
    store = CheckpointStore.from_optional(checkpoint_path, interval=100)

    store.save_final(_make_state(7))

    assert json.loads(checkpoint_path.read_text())["progress_index"] == 7


def test_disabled_store_save_final_is_noop(checkpoint_path):
    CheckpointStore.disabled().save_final(_make_state())

    assert not checkpoint_path.exists()


def test_load_returns_none_when_file_is_absent(checkpoint_path):
    assert CheckpointStore.from_optional(checkpoint_path).load() is None


def test_load_returns_saved_state(checkpoint_path):
    store = CheckpointStore.from_optional(checkpoint_path)
    store.save_final(_make_state(5))

    assert store.load()["progress_index"] == 5


def test_load_rejects_corrupted_file(checkpoint_path):
    checkpoint_path.write_bytes(b"not valid json {{{")

    with pytest.raises(CheckpointError, match="Could not parse checkpoint"):
        CheckpointStore.from_optional(checkpoint_path).load()


def test_pickle_round_trip(tmp_path):
    path = tmp_path / "run.pkl"
    store = CheckpointStore.from_optional(path, fmt="pickle")
    store.save_final(_make_state(9))

    assert store.load()["progress_index"] == 9


def test_delete_removes_saved_file(checkpoint_path):
    store = CheckpointStore.from_optional(checkpoint_path)
    store.save_final(_make_state())

    store.delete()

    assert not checkpoint_path.exists()


def test_delete_is_safe_without_file(checkpoint_path):
    CheckpointStore.from_optional(checkpoint_path).delete()
    CheckpointStore.disabled().delete()

    assert not checkpoint_path.exists()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(np.array([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0], id="array"),
        pytest.param(np.float64(-2.5), -2.5, id="scalar"),
        pytest.param(Path("/some/dump.data"), "/some/dump.data", id="path"),
    ],
)
def test_json_serialization_normalizes_supported_values(
    checkpoint_path,
    value,
    expected,
):
    store = CheckpointStore.from_optional(checkpoint_path)
    state = _make_state()
    state["state"]["value"] = value

    store.save_final(state)

    assert json.loads(checkpoint_path.read_text())["state"]["value"] == expected


def test_saved_envelope_contains_schema_version(checkpoint_path):
    store = CheckpointStore.from_optional(checkpoint_path)
    store.save_final(_make_state())

    assert json.loads(checkpoint_path.read_text())["schema_version"] == 1


def test_exists_tracks_saved_file(checkpoint_path):
    store = CheckpointStore.from_optional(checkpoint_path)

    assert not store.exists
    store.save_final(_make_state())
    assert store.exists
    assert not CheckpointStore.disabled().exists


# ---------------------------------------------------------------------------
# CandidateCheckpoint
# ---------------------------------------------------------------------------


def test_candidate_checkpoint_starts_with_population_unevaluated(
    checkpoint_path,
    candidate_uids,
):
    checkpoint = _fresh_candidate_checkpoint(checkpoint_path, candidate_uids)

    assert all(not checkpoint.is_done(uid) for uid in candidate_uids)


def test_candidate_checkpoint_rejects_invalid_format(checkpoint_path, candidate_uids):
    with pytest.raises(CheckpointValueError, match="fmt must be"):
        CandidateCheckpoint(checkpoint_path, "yaml", 0, candidate_uids)


@pytest.mark.parametrize(
    "unique_ids",
    [
        pytest.param(("uid0", "uid1"), id="not-list"),
        pytest.param(["uid0", ""], id="empty-id"),
        pytest.param(["uid0", 1], id="non-string-id"),
        pytest.param(["uid0", "uid0"], id="duplicate-id"),
    ],
)
def test_candidate_checkpoint_rejects_invalid_population_identity(
    checkpoint_path,
    unique_ids,
):
    with pytest.raises(CheckpointValueError, match="unique_ids"):
        CandidateCheckpoint(checkpoint_path, "json", 0, unique_ids)


def test_record_marks_only_requested_candidate_done(checkpoint_path, candidate_uids):
    checkpoint = _fresh_candidate_checkpoint(checkpoint_path, candidate_uids)

    checkpoint.record(candidate_uids[0], 1.23, "/tmp/dump.data")

    assert checkpoint.is_done(candidate_uids[0])
    assert not checkpoint.is_done(candidate_uids[1])


def test_record_persists_result_to_sidecar(checkpoint_path, candidate_uids):
    checkpoint = _fresh_candidate_checkpoint(checkpoint_path, candidate_uids)
    expected_path = checkpoint_path.with_suffix(".iter0.json")

    checkpoint.record(candidate_uids[0], 2.5, "/tmp/d.data")

    payload = json.loads(expected_path.read_text())
    assert payload["iteration_index"] == 0
    assert payload["unique_ids"] == candidate_uids
    assert candidate_uids[0] in payload["results"]


def test_record_rejects_candidate_outside_population(checkpoint_path, candidate_uids):
    checkpoint = _fresh_candidate_checkpoint(checkpoint_path, candidate_uids)

    with pytest.raises(CheckpointValueError, match="not part of this checkpoint"):
        checkpoint.record("run_i0_c99", 1.0, "/tmp/other.data")


def test_get_result_returns_energy_and_dump(checkpoint_path, candidate_uids):
    checkpoint = _fresh_candidate_checkpoint(checkpoint_path, candidate_uids)
    checkpoint.record(candidate_uids[1], 3.14, "/tmp/out.data")

    energy, dump = checkpoint.get_result(candidate_uids[1])
    assert energy == pytest.approx(3.14)
    assert dump == "/tmp/out.data"


def test_get_result_raises_for_unevaluated_candidate(checkpoint_path, candidate_uids):
    checkpoint = _fresh_candidate_checkpoint(checkpoint_path, candidate_uids)

    with pytest.raises(CheckpointError, match="No result recorded"):
        checkpoint.get_result(candidate_uids[0])


def test_failure_result_preserves_missing_dump(checkpoint_path, candidate_uids):
    checkpoint = _fresh_candidate_checkpoint(checkpoint_path, candidate_uids)
    checkpoint.record(candidate_uids[2], 1.0e30, None)

    assert checkpoint.get_result(candidate_uids[2]) == (1.0e30, None)


def test_metadata_round_trips_without_changing_result(checkpoint_path, candidate_uids):
    checkpoint = _fresh_candidate_checkpoint(checkpoint_path, candidate_uids)
    metadata = {
        "owned_evaluation_version": 1,
        "success": False,
        "failure_reason": "lost atoms",
    }
    checkpoint.record(candidate_uids[2], 1.0e30, None, metadata=metadata)

    restored = CandidateCheckpoint.new_or_resume(
        checkpoint_path,
        "json",
        0,
        candidate_uids,
    )

    assert restored.get_result(candidate_uids[2]) == (1.0e30, None)
    assert restored.get_metadata(candidate_uids[2]) == metadata


def test_result_without_metadata_remains_supported(checkpoint_path, candidate_uids):
    checkpoint = _fresh_candidate_checkpoint(checkpoint_path, candidate_uids)
    checkpoint.record(candidate_uids[0], 1.25, "/legacy.data")

    assert checkpoint.get_metadata(candidate_uids[0]) is None


@pytest.mark.parametrize("fmt", ["json", "pickle"])
def test_new_or_resume_restores_completed_candidates(
    tmp_path,
    candidate_uids,
    fmt,
):
    main_path = tmp_path / ("run.pkl" if fmt == "pickle" else "run.json")
    checkpoint = _fresh_candidate_checkpoint(main_path, candidate_uids, fmt=fmt)
    checkpoint.record(candidate_uids[0], 1.1, "/a.data")
    checkpoint.record(candidate_uids[1], 2.2, "/b.data")

    restored = CandidateCheckpoint.new_or_resume(main_path, fmt, 0, candidate_uids)

    assert restored.is_done(candidate_uids[0])
    assert restored.is_done(candidate_uids[1])
    assert not restored.is_done(candidate_uids[2])
    assert restored.get_result(candidate_uids[0])[0] == pytest.approx(1.1)


@pytest.mark.parametrize(
    "resumed_uids",
    [
        pytest.param(
            ["run_i0_c1", "run_i0_c0", "run_i0_c2"],
            id="reordered",
        ),
        pytest.param(["run_i0_c0", "run_i0_c1"], id="missing"),
        pytest.param(
            ["run_i0_c0", "run_i0_c1", "run_i0_c2", "run_i0_c3"],
            id="extra",
        ),
        pytest.param(["other0", "other1", "other2"], id="different"),
    ],
)
def test_resume_rejects_different_candidate_population(
    checkpoint_path,
    candidate_uids,
    resumed_uids,
):
    checkpoint = _fresh_candidate_checkpoint(checkpoint_path, candidate_uids)
    checkpoint.record(candidate_uids[0], 1.0, "/x.data")

    with pytest.raises(
        CheckpointCompatibilityError,
        match="population identity",
    ):
        CandidateCheckpoint.new_or_resume(
            checkpoint_path,
            "json",
            0,
            resumed_uids,
        )


def test_resume_rejects_iteration_identity_mismatch(checkpoint_path, candidate_uids):
    checkpoint = _fresh_candidate_checkpoint(checkpoint_path, candidate_uids)
    checkpoint.record(candidate_uids[0], 1.0, "/x.data")
    source = checkpoint_path.with_suffix(".iter0.json")
    mismatched = checkpoint_path.with_suffix(".iter1.json")
    source.rename(mismatched)

    with pytest.raises(
        CheckpointCompatibilityError,
        match="iteration mismatch",
    ):
        CandidateCheckpoint.new_or_resume(
            checkpoint_path,
            "json",
            1,
            candidate_uids,
        )


def test_resume_accepts_legacy_sidecar_with_exact_population_identity(
    checkpoint_path,
    candidate_uids,
):
    sidecar = checkpoint_path.with_suffix(".iter0.json")
    sidecar.write_text(
        json.dumps(
            {
                "iteration_index": 0,
                "results": {uid: None for uid in candidate_uids},
            }
        )
    )

    restored = CandidateCheckpoint.new_or_resume(
        checkpoint_path,
        "json",
        0,
        candidate_uids,
    )

    assert all(not restored.is_done(uid) for uid in candidate_uids)


def test_resume_rejects_corrupted_sidecar(checkpoint_path, candidate_uids):
    checkpoint_path.with_suffix(".iter0.json").write_bytes(b"not json {{{{")

    with pytest.raises(CheckpointError, match="Could not parse candidate checkpoint"):
        CandidateCheckpoint.new_or_resume(
            checkpoint_path,
            "json",
            0,
            candidate_uids,
        )


def test_candidate_sidecar_uses_documented_iteration_filename(
    checkpoint_path,
    candidate_uids,
):
    checkpoint = CandidateCheckpoint.new_or_resume(
        checkpoint_path,
        "json",
        3,
        candidate_uids,
    )
    checkpoint.record(candidate_uids[0], 1.0, "/x.data")

    assert checkpoint_path.with_suffix(".iter3.json").exists()


def test_candidate_checkpoint_delete_removes_sidecar(checkpoint_path, candidate_uids):
    checkpoint = _fresh_candidate_checkpoint(checkpoint_path, candidate_uids)
    sidecar = checkpoint_path.with_suffix(".iter0.json")
    checkpoint.record(candidate_uids[0], 1.0, "/x.data")

    checkpoint.delete()
    checkpoint.delete()

    assert not sidecar.exists()


def test_new_or_resume_creates_fresh_checkpoint_when_sidecar_is_absent(
    checkpoint_path,
    candidate_uids,
):
    checkpoint = CandidateCheckpoint.new_or_resume(
        checkpoint_path,
        "json",
        0,
        candidate_uids,
    )

    assert all(not checkpoint.is_done(uid) for uid in candidate_uids)


# ---------------------------------------------------------------------------
# Batch callback adaptation
# ---------------------------------------------------------------------------


@pytest.fixture
def batch_uids():
    return ["uid0", "uid1", "uid2"]


def _batch_checkpoint(tmp_path, unique_ids):
    return CandidateCheckpoint(tmp_path / "run.iter0.json", "json", 0, unique_ids)


def _batch_results(unique_ids):
    return [
        {"energy": float(index), "final_dump": f"/dump_{uid}.data"}
        for index, uid in enumerate(unique_ids)
    ]


def test_wrapped_batch_records_results_after_batch_return(tmp_path, batch_uids):
    def plain_batch(_GB, _manips, _structs, _lineages, unique_ids):
        return _batch_results(unique_ids)

    checkpoint = _batch_checkpoint(tmp_path, batch_uids)
    wrapped = _wrap_batch_func_with_checkpoint(plain_batch, penalty=1.0e30)

    wrapped(None, [], [], [], batch_uids, checkpoint=checkpoint)

    assert all(checkpoint.is_done(uid) for uid in batch_uids)


def test_wrapped_batch_preserves_already_completed_result(tmp_path, batch_uids):
    checkpoint = _batch_checkpoint(tmp_path, batch_uids)
    checkpoint.record("uid0", 99.0, "/existing.data")

    def plain_batch(_GB, _manips, _structs, _lineages, unique_ids):
        return _batch_results(unique_ids)

    wrapped = _wrap_batch_func_with_checkpoint(plain_batch, penalty=1.0e30)
    wrapped(None, [], [], [], batch_uids, checkpoint=checkpoint)

    assert checkpoint.get_result("uid0")[0] == pytest.approx(99.0)


def test_wrapped_batch_works_without_checkpoint(batch_uids):
    def plain_batch(_GB, _manips, _structs, _lineages, unique_ids):
        return _batch_results(unique_ids)

    wrapped = _wrap_batch_func_with_checkpoint(plain_batch, penalty=1.0e30)

    assert wrapped(None, [], [], [], batch_uids) == _batch_results(batch_uids)


def test_wrapped_batch_preserves_callback_return_identity():
    expected = [{"energy": 1.0, "final_dump": "/a.data"}]

    def plain_batch(_GB, _manips, _structs, _lineages, _unique_ids):
        return expected

    wrapped = _wrap_batch_func_with_checkpoint(plain_batch, penalty=1.0e30)

    assert wrapped(None, [], [], [], ["uid0"]) is expected


def test_wrapped_batch_uses_optimizer_penalty_when_energy_is_missing(tmp_path):
    def plain_batch(_GB, _manips, _structs, _lineages, _unique_ids):
        return [{"final_dump": None}]

    checkpoint = _batch_checkpoint(tmp_path, ["uid0"])
    wrapped = _wrap_batch_func_with_checkpoint(plain_batch, penalty=1234.5)

    wrapped(None, [], [], [], ["uid0"], checkpoint=checkpoint)

    assert checkpoint.get_result("uid0") == (1234.5, None)
