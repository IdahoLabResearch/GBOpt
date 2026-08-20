# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pytest

from GBOpt.artifacts import ArtifactRetentionPolicy, KeepBest
from GBOpt.GBMaker import GBMaker
from GBOpt.GBMinimizer import (
    GBMinimizerError,
    GBMinimizerValueError,
    MonteCarloMinimizer,
)

_TEST_CALCULATION_CONTEXT = {"calculator": {"name": "test-evaluator"}}


pytestmark = pytest.mark.filterwarnings(
    "ignore:File-backed Parent initialization without explicit grain ownership is "
    "deprecated.*:DeprecationWarning"
)


@pytest.fixture
def gb():
    theta = math.radians(36.869898)
    misorientation = np.array([theta, 0.0, 0.0, 0.0, -theta / 2.0])
    return GBMaker(
        3.52,
        "fcc",
        10.0,
        misorientation,
        "Ni",
        repeat_factor=(2, 5),
        x_dim_min=30.0,
        vacuum=8.0,
        interaction_distance=8.0,
    )


@pytest.fixture(autouse=True)
def _run_in_tmp_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _make_energy_func(gb, crash_after=None):
    """Return a deterministic evaluator that may fail after a fixed call count."""
    call_count = 0

    def energy_func(GB, manipulator, atom_positions, unique_id):
        nonlocal call_count
        call_count += 1
        if crash_after is not None and call_count > crash_after:
            raise RuntimeError(f"Simulated crash at call {call_count}")
        path = f"{unique_id}_{call_count}.data"
        GB.write_lammps(
            path,
            atom_positions,
            manipulator.parents[0].box_dims,
        )
        return 2.0 - call_count * 0.001, path

    return energy_func


def _make_minimizer(gb, energy_func):
    return MonteCarloMinimizer(
        gb,
        energy_func,
        ["translate_right_grain"],
        seed=0,
    )


def _make_sequence_energy_func(energies, root, *, start_index=0):
    """Return an evaluator writing deterministic relaxed files beneath ``root``."""
    values = iter(energies)
    call_count = start_index
    root.mkdir(parents=True, exist_ok=True)

    def energy_func(GB, manipulator, atom_positions, unique_id):
        nonlocal call_count
        call_count += 1
        path = root / f"{unique_id}_{call_count}.data"
        GB.write_lammps(
            str(path),
            atom_positions,
            manipulator.parents[0].box_dims,
        )
        return next(values), str(path)

    return energy_func


def test_run_mc_no_checkpoint_no_file_created(gb, tmp_path):
    mc = _make_minimizer(gb, _make_energy_func(gb))

    mc.run_MC(max_steps=2, unique_id=1)

    assert list(tmp_path.glob("*.json")) == []
    assert list(tmp_path.glob("*.pkl")) == []
    assert mc.artifact_store is None
    assert list(tmp_path.glob("*.artifacts")) == []


def test_mc_retention_prunes_superseded_accepted_source_after_commit(gb, tmp_path):
    managed_root = tmp_path / "managed"
    energy_func = _make_sequence_energy_func([2.0, 1.0], managed_root)
    policy = ArtifactRetentionPolicy(
        rules=(
            KeepBest(
                name="objective_best",
                property="objective",
                direction="min",
                count=1,
            ),
        ),
        prune=True,
    )
    mc = MonteCarloMinimizer(
        gb,
        energy_func,
        ["translate_right_grain"],
        seed=0,
        retention_policy=policy,
        calculation_context=_TEST_CALCULATION_CONTEXT,
        managed_artifact_root=managed_root,
    )
    checkpoint = tmp_path / "mc_retention.json"

    mc.run_MC(max_steps=1, unique_id=41, checkpoint_file=checkpoint)

    initial_source = managed_root / "initial41_1.data"
    current_source = managed_root / "41_2.data"
    archive = tmp_path / "mc_retention.artifacts" / "structures" / "MC_41_s1.data"
    assert not initial_source.exists()
    assert current_source.exists()
    assert archive.is_file()

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    manifest = json.loads(
        (checkpoint.with_suffix(".artifacts") / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["calculation_context"] == _TEST_CALCULATION_CONTEXT
    assert state["best_dump"] == str(archive)
    records = state["state"]["artifact_store"]["records"]
    by_id = {record["candidate"]["candidate_id"]: record for record in records}
    assert by_id["MC_41_s1"]["pins"] == ["best_result", "run_checkpoint"]
    assert by_id["MC_41_s1"]["retention_reasons"] == ["rule:objective_best"]


def test_mc_retains_rejected_scientific_result_and_prunes_its_source(gb, tmp_path):
    managed_root = tmp_path / "managed_rejected"
    energy_func = _make_sequence_energy_func([1.0, 1000.0], managed_root)
    policy = ArtifactRetentionPolicy(
        rules=(
            KeepBest(
                name="largest_objective",
                property="objective",
                direction="max",
                count=1,
            ),
        ),
        prune=True,
    )
    mc = MonteCarloMinimizer(
        gb,
        energy_func,
        ["translate_right_grain"],
        seed=0,
        retention_policy=policy,
        calculation_context=_TEST_CALCULATION_CONTEXT,
        managed_artifact_root=managed_root,
    )
    checkpoint = tmp_path / "mc_rejected.json"

    mc.run_MC(max_steps=1, unique_id=42, checkpoint_file=checkpoint)

    rejected_source = managed_root / "42_2.data"
    rejected_archive = (
        tmp_path / "mc_rejected.artifacts" / "structures" / "MC_42_s1.data"
    )
    assert not rejected_source.exists()
    assert rejected_archive.is_file()

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    records = state["state"]["artifact_store"]["records"]
    rejected = next(
        record
        for record in records
        if record["candidate"]["candidate_id"] == "MC_42_s1"
    )
    assert rejected["pins"] == []
    assert rejected["retention_reasons"] == ["rule:largest_objective"]
    assert (tmp_path / "mc_rejected.artifacts" / "manifest.json").is_file()
    assert (tmp_path / "mc_rejected.artifacts" / "history.jsonl").is_file()


def test_mc_resume_rejects_retention_policy_mismatch(gb, tmp_path):
    managed_root = tmp_path / "managed_mismatch"
    energy_func = _make_sequence_energy_func([2.0, 1.0], managed_root)
    first_policy = ArtifactRetentionPolicy(
        rules=(
            KeepBest(
                name="objective_best",
                property="objective",
                direction="min",
                count=1,
            ),
        ),
    )
    checkpoint = tmp_path / "mc_mismatch.json"
    MonteCarloMinimizer(
        gb,
        energy_func,
        ["translate_right_grain"],
        seed=0,
        retention_policy=first_policy,
    ).run_MC(max_steps=1, unique_id=43, checkpoint_file=checkpoint)

    changed_policy = ArtifactRetentionPolicy(
        rules=(
            KeepBest(
                name="objective_best",
                property="objective",
                direction="min",
                count=2,
            ),
        ),
    )
    resumed = MonteCarloMinimizer(
        gb,
        _make_energy_func(gb),
        ["translate_right_grain"],
        seed=0,
        retention_policy=changed_policy,
    )

    with pytest.raises(
        GBMinimizerError,
        match="artifact retention policy signature mismatch",
    ):
        resumed.run_MC(max_steps=2, checkpoint_file=checkpoint)


def test_mc_pruning_requires_calculation_context(gb, tmp_path):
    managed_root = tmp_path / "managed_context"
    policy = ArtifactRetentionPolicy(prune=True)

    with pytest.raises(
        GBMinimizerValueError,
        match="requires a non-empty calculation_context",
    ):
        MonteCarloMinimizer(
            gb,
            _make_energy_func(gb),
            ["translate_right_grain"],
            seed=0,
            retention_policy=policy,
            managed_artifact_root=managed_root,
        )


def test_mc_pruning_requires_checkpoint_file(gb, tmp_path):
    managed_root = tmp_path / "managed_no_checkpoint"
    policy = ArtifactRetentionPolicy(prune=True)
    mc = MonteCarloMinimizer(
        gb,
        _make_energy_func(gb),
        ["translate_right_grain"],
        seed=0,
        retention_policy=policy,
        calculation_context=_TEST_CALCULATION_CONTEXT,
        managed_artifact_root=managed_root,
    )

    with pytest.raises(GBMinimizerValueError, match="requires checkpoint_file"):
        mc.run_MC(max_steps=1, unique_id=44)


def test_mc_cleanup_failure_leaks_source_but_checkpoint_resumes(gb, tmp_path):
    managed_root = tmp_path / "managed_cleanup_failure"
    energy_func = _make_sequence_energy_func([1.0, 1000.0], managed_root)
    policy = ArtifactRetentionPolicy(prune=True)

    def failing_cleanup(_request):
        raise OSError("backend cleanup failed")

    checkpoint = tmp_path / "mc_cleanup_failure.json"
    mc = MonteCarloMinimizer(
        gb,
        energy_func,
        ["translate_right_grain"],
        seed=0,
        retention_policy=policy,
        calculation_context=_TEST_CALCULATION_CONTEXT,
        cleanup_candidate=failing_cleanup,
    )

    with pytest.warns(RuntimeWarning, match="Artifact cleanup failed"):
        mc.run_MC(max_steps=1, unique_id=45, checkpoint_file=checkpoint)

    leaked_source = managed_root / "45_2.data"
    assert checkpoint.is_file()
    assert leaked_source.is_file()

    resumed = MonteCarloMinimizer(
        gb,
        _make_sequence_energy_func([1000.0], managed_root),
        ["translate_right_grain"],
        seed=0,
        retention_policy=policy,
        calculation_context=_TEST_CALCULATION_CONTEXT,
        cleanup_candidate=lambda _request: None,
    )
    resumed.run_MC(max_steps=2, checkpoint_file=checkpoint)

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert state["progress_index"] == 2


def test_mc_prune_fresh_resume_matches_continuous_run(gb, tmp_path):
    continuous_root = tmp_path / "continuous"
    resumed_root = tmp_path / "resumed"
    policy = ArtifactRetentionPolicy(
        rules=(
            KeepBest(
                name="objective_best",
                property="objective",
                direction="min",
                count=1,
            ),
        ),
        prune=True,
    )

    continuous_checkpoint = tmp_path / "continuous.json"
    continuous = MonteCarloMinimizer(
        gb,
        _make_sequence_energy_func([4.0, 3.0, 2.0, 1.0], continuous_root),
        ["translate_right_grain"],
        seed=11,
        retention_policy=policy,
        calculation_context=_TEST_CALCULATION_CONTEXT,
        managed_artifact_root=continuous_root,
    )
    continuous_energy = continuous.run_MC(
        max_steps=3,
        unique_id=46,
        checkpoint_file=continuous_checkpoint,
    )

    resumed_checkpoint = tmp_path / "resumed.json"
    partial = MonteCarloMinimizer(
        gb,
        _make_sequence_energy_func([4.0, 3.0], resumed_root),
        ["translate_right_grain"],
        seed=11,
        retention_policy=policy,
        calculation_context=_TEST_CALCULATION_CONTEXT,
        managed_artifact_root=resumed_root,
    )
    partial.run_MC(max_steps=1, unique_id=46, checkpoint_file=resumed_checkpoint)

    partial_state = json.loads(resumed_checkpoint.read_text(encoding="utf-8"))
    for record in partial_state["state"]["artifact_store"]["records"]:
        if "run_checkpoint" in record["pins"]:
            assert Path(record["source_path"]).is_file()
        elif record["source_path"] is not None:
            assert not Path(record["source_path"]).exists()

    resumed = MonteCarloMinimizer(
        gb,
        _make_sequence_energy_func([2.0, 1.0], resumed_root, start_index=2),
        ["translate_right_grain"],
        seed=999,
        retention_policy=policy,
        calculation_context=_TEST_CALCULATION_CONTEXT,
        managed_artifact_root=resumed_root,
    )
    resumed_energy = resumed.run_MC(
        max_steps=3,
        checkpoint_file=resumed_checkpoint,
    )

    assert resumed_energy == pytest.approx(continuous_energy)
    assert resumed.GBE_vals == continuous.GBE_vals
    assert resumed.accepted_idx == continuous.accepted_idx
    assert resumed.operation_list == continuous.operation_list

    continuous_state = json.loads(
        continuous_checkpoint.read_text(encoding="utf-8")
    )["state"]
    resumed_state = json.loads(resumed_checkpoint.read_text(encoding="utf-8"))["state"]

    def normalized_records(state):
        return [
            {
                "candidate": record["candidate"],
                "pins": record["pins"],
                "retention_reasons": record["retention_reasons"],
                "has_archive": record["archive_path"] is not None,
            }
            for record in state["artifact_store"]["records"]
        ]

    assert normalized_records(resumed_state) == normalized_records(continuous_state)


def test_run_mc_checkpoint_kept_on_completion(gb, tmp_path):
    mc = _make_minimizer(gb, _make_energy_func(gb))
    checkpoint = tmp_path / "mc.json"

    mc.run_MC(max_steps=3, unique_id=2, checkpoint_file=checkpoint)

    assert checkpoint.exists()


def test_run_mc_checkpoint_file_is_valid_json(gb, tmp_path):
    mc = _make_minimizer(gb, _make_energy_func(gb, crash_after=3))
    checkpoint = tmp_path / "mc.json"

    with pytest.raises(RuntimeError):
        mc.run_MC(max_steps=10, unique_id=3, checkpoint_file=checkpoint)

    assert checkpoint.exists()
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert {
        "schema_version",
        "minimizer",
        "progress_unit",
        "progress_index",
        "best_energy",
        "rng_state",
        "run_params",
        "state",
    } <= state.keys()
    assert {
        "T",
        "rejection_count",
        "prev_gbe",
        "GBE_vals",
        "accepted_idx",
        "operation_list",
        "current_structure_dump",
    } <= state["state"].keys()
    assert state["minimizer"] == "MonteCarloMinimizer"
    assert state["progress_unit"] == "step"


def test_run_mc_checkpoint_format_pickle(gb, tmp_path):
    mc = _make_minimizer(gb, _make_energy_func(gb, crash_after=3))
    checkpoint = tmp_path / "mc.pkl"

    with pytest.raises(RuntimeError):
        mc.run_MC(
            max_steps=10,
            unique_id=4,
            checkpoint_file=checkpoint,
            checkpoint_format="pickle",
        )

    assert checkpoint.exists()
    with checkpoint.open("rb") as stream:
        state = pickle.load(stream)
    assert "progress_index" in state
    assert "GBE_vals" in state["state"]


def test_run_mc_resume_from_json(gb, tmp_path):
    mc = _make_minimizer(gb, _make_energy_func(gb, crash_after=3))
    checkpoint = tmp_path / "mc_resume.json"

    with pytest.raises(RuntimeError):
        mc.run_MC(max_steps=10, unique_id=5, checkpoint_file=checkpoint)

    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    resumed_from_step = saved["progress_index"]
    gbe_count_before_resume = len(mc.GBE_vals)
    assert resumed_from_step > 0

    mc.gb_energy_func = _make_energy_func(gb)
    mc.run_MC(max_steps=10, unique_id=5, checkpoint_file=checkpoint)

    assert checkpoint.exists()
    assert len(mc.GBE_vals) > gbe_count_before_resume


def test_run_mc_resume_from_pickle(gb, tmp_path):
    mc = _make_minimizer(gb, _make_energy_func(gb, crash_after=3))
    checkpoint = tmp_path / "mc_resume.pkl"

    with pytest.raises(RuntimeError):
        mc.run_MC(
            max_steps=10,
            unique_id=6,
            checkpoint_file=checkpoint,
            checkpoint_format="pickle",
        )

    assert checkpoint.exists()
    mc.gb_energy_func = _make_energy_func(gb)
    mc.run_MC(
        max_steps=10,
        unique_id=6,
        checkpoint_file=checkpoint,
        checkpoint_format="pickle",
    )
    assert checkpoint.exists()


def test_run_mc_corrupted_checkpoint_raises(gb, tmp_path):
    checkpoint = tmp_path / "corrupt.json"
    checkpoint.write_bytes(b"not valid json {{{")
    mc = _make_minimizer(gb, _make_energy_func(gb))

    with pytest.raises(GBMinimizerError):
        mc.run_MC(max_steps=5, unique_id=7, checkpoint_file=checkpoint)


def test_run_mc_invalid_format_raises(gb, tmp_path):
    mc = _make_minimizer(gb, _make_energy_func(gb))
    checkpoint = tmp_path / "mc.hdf5"

    with pytest.raises(GBMinimizerValueError):
        mc.run_MC(
            max_steps=5,
            unique_id=8,
            checkpoint_file=checkpoint,
            checkpoint_format="hdf5",
        )


def test_run_mc_checkpoint_interval_respected(gb, tmp_path):
    # Calls 1-5 succeed and call 6 (step 5) raises. With interval=3, only step 3
    # has been checkpointed at that point.
    mc = _make_minimizer(gb, _make_energy_func(gb, crash_after=5))
    checkpoint = tmp_path / "mc_interval.json"

    with pytest.raises(RuntimeError):
        mc.run_MC(
            max_steps=10,
            unique_id=9,
            checkpoint_file=checkpoint,
            checkpoint_interval=3,
        )

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert state["progress_index"] == 3


def test_resume_without_unique_id_restores_original_label(gb, tmp_path):
    checkpoint = tmp_path / "mc_uid.json"
    _make_minimizer(gb, _make_energy_func(gb)).run_MC(
        max_steps=2,
        unique_id=7777,
        checkpoint_file=checkpoint,
    )

    _make_minimizer(gb, _make_energy_func(gb)).run_MC(
        max_steps=4,
        checkpoint_file=checkpoint,
    )

    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["run_params"]["unique_id"] == "7777"


def test_two_fresh_runs_without_unique_id_use_different_labels(gb, tmp_path):
    checkpoint_1 = tmp_path / "mc1.json"
    checkpoint_2 = tmp_path / "mc2.json"

    _make_minimizer(gb, _make_energy_func(gb)).run_MC(
        max_steps=1,
        checkpoint_file=checkpoint_1,
    )
    _make_minimizer(gb, _make_energy_func(gb)).run_MC(
        max_steps=1,
        checkpoint_file=checkpoint_2,
    )

    uid_1 = json.loads(checkpoint_1.read_text(encoding="utf-8"))["run_params"][
        "unique_id"
    ]
    uid_2 = json.loads(checkpoint_2.read_text(encoding="utf-8"))["run_params"][
        "unique_id"
    ]
    assert uid_1 != uid_2


def test_resume_restores_cooldown_rate_from_checkpoint(gb, tmp_path):
    checkpoint = tmp_path / "mc_cr.json"
    _make_minimizer(gb, _make_energy_func(gb)).run_MC(
        max_steps=2,
        cooldown_rate=0.8,
        unique_id=1,
        checkpoint_file=checkpoint,
    )

    _make_minimizer(gb, _make_energy_func(gb)).run_MC(
        max_steps=4,
        checkpoint_file=checkpoint,
    )

    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["run_params"]["cooldown_rate"] == pytest.approx(0.8)


def test_resume_restores_min_steps_from_checkpoint(gb, tmp_path):
    checkpoint = tmp_path / "mc_ms.json"
    _make_minimizer(gb, _make_energy_func(gb)).run_MC(
        max_steps=2,
        min_steps=5,
        unique_id=2,
        checkpoint_file=checkpoint,
    )

    _make_minimizer(gb, _make_energy_func(gb)).run_MC(
        max_steps=10,
        checkpoint_file=checkpoint,
    )

    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["run_params"]["min_steps"] == 5
