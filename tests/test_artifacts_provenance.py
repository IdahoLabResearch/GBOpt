# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Focused tests for derived artifact manifests and append-only lifecycle history."""

import json

import pytest

from GBOpt.artifacts.provenance import ArtifactProvenanceError, _ArtifactProvenance
from GBOpt.artifacts.types import ArtifactPin, ArtifactRecord, RetentionCandidate


def _candidate(candidate_id, objective, *, generation=1):
    return RetentionCandidate(
        candidate_id=candidate_id,
        generation=generation,
        objective=objective,
        properties={
            "atom_count": 12,
            "composition": (("O", 2), ("U", 1)),
            "cell_volume": 120.0,
            "mass_density": 10.9 + objective,
        },
        lineage=("parent",),
    )


def _history_events(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_manifest_is_deterministic_and_contains_current_artifact_state(tmp_path):
    provenance = _ArtifactProvenance(tmp_path / "run.artifacts")
    retained = ArtifactRecord(
        candidate=_candidate("b", 2.0),
        source_path=tmp_path / "b-source.data",
        archive_path=tmp_path / "structures" / "b.data",
        pins=(ArtifactPin.BEST_RESULT,),
        retention_reasons=("rule:elite",),
    )
    transient = ArtifactRecord(
        candidate=_candidate("a", 1.0),
        source_path=tmp_path / "a-source.data",
    )
    ownership = {
        "b": {
            "atom_ids": [1, 2],
            "labels": [0, 1],
            "normal_topology": "periodic",
        }
    }

    provenance.write_manifest((retained, transient), ownership_metadata=ownership)
    first = provenance.manifest_path.read_bytes()
    provenance.write_manifest((transient, retained), ownership_metadata=ownership)
    second = provenance.manifest_path.read_bytes()

    assert second == first
    manifest = json.loads(second)
    assert manifest["version"] == 1
    assert [record["candidate_id"] for record in manifest["records"]] == ["a", "b"]
    retained_state = manifest["records"][1]
    assert retained_state["status"] == "pinned_and_retained"
    assert retained_state["retention_reasons"] == ["rule:elite"]
    assert retained_state["pins"] == ["best_result"]
    assert retained_state["properties"]["mass_density"] == pytest.approx(12.9)
    assert retained_state["ownership_metadata"] == ownership["b"]
    assert not provenance.manifest_path.with_name("manifest.json.tmp").exists()


def test_history_records_lifecycle_events_and_replay_is_idempotent(tmp_path):
    root = tmp_path / "run.artifacts"
    provenance = _ArtifactProvenance(root)
    candidate = _candidate("GA_1_g2_c3", 1.5, generation=2)
    archive_path = root / "structures" / "GA_1_g2_c3.data"
    source_path = tmp_path / "evaluations" / "GA_1_g2_c3.data"

    provenance.record_candidate_evaluated(candidate)
    provenance.record_properties_calculated(candidate)
    provenance.record_retention_reason_added(candidate.candidate_id, "rule:elite")
    provenance.record_retention_reason_removed(candidate.candidate_id, "rule:elite")
    provenance.record_archive_created(candidate.candidate_id, archive_path)
    provenance.record_source_pruned(candidate.candidate_id, source_path)
    provenance.record_archive_evicted(candidate.candidate_id, archive_path)
    provenance.record_cleanup_failed(
        "source_prune",
        source_path,
        "simulated failure",
        candidate_id=candidate.candidate_id,
    )

    replay = _ArtifactProvenance(root)
    replay.record_candidate_evaluated(candidate)
    replay.record_archive_created(candidate.candidate_id, archive_path)

    events = _history_events(provenance.history_path)
    assert len(events) == 8
    assert [event["event"] for event in events] == [
        "candidate_evaluated",
        "properties_calculated",
        "retention_reason_added",
        "retention_reason_removed",
        "archive_created",
        "source_pruned",
        "archive_evicted",
        "cleanup_failed",
    ]
    assert all(event["version"] == 1 for event in events)
    assert events[0]["evaluation_status"] == "success"
    assert events[1]["properties"]["mass_density"] == pytest.approx(12.4)


def test_existing_malformed_history_is_rejected(tmp_path):
    root = tmp_path / "run.artifacts"
    root.mkdir()
    (root / "history.jsonl").write_text('{"event":"partial"}', encoding="utf-8")

    with pytest.raises(ArtifactProvenanceError, match="incomplete trailing entry"):
        _ArtifactProvenance(root)


def test_manifest_rejects_unknown_or_non_json_ownership_metadata(tmp_path):
    provenance = _ArtifactProvenance(tmp_path / "run.artifacts")
    record = ArtifactRecord(candidate=_candidate("candidate", 1.0))

    with pytest.raises(ArtifactProvenanceError, match="unknown candidate identities"):
        provenance.write_manifest(
            (record,),
            ownership_metadata={"other": {"atom_ids": [1]}},
        )

    with pytest.raises(ArtifactProvenanceError, match="unsupported type"):
        provenance.write_manifest(
            (record,),
            ownership_metadata={"candidate": {"bad": object()}},
        )
