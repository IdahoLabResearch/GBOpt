# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import json
import math
import os
import pickle
import tempfile
import unittest
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


pytestmark = pytest.mark.filterwarnings(
    "ignore:File-backed Parent initialization without explicit grain ownership is "
    "deprecated.*:DeprecationWarning"
)


class TestMonteCarloMinimizerCheckpointing(unittest.TestCase):

    def setUp(self):
        theta = math.radians(36.869898)
        misorientation = np.array([theta, 0.0, 0.0, 0.0, -theta / 2.0])
        self.gb = GBMaker(
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
        self.tmpdir = tempfile.TemporaryDirectory()
        self._orig_cwd = os.getcwd()
        os.chdir(self.tmpdir.name)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        self.tmpdir.cleanup()

    def _make_energy_func(self, crash_after=None):
        """Fake energy func that writes LAMMPS files and optionally raises after N calls.

        Returns slightly decreasing energies (0.001 per call) so every step is accepted
        but del_E (0.001) > E_tol (1e-4), preventing early convergence.
        """
        call_count = [0]

        def energy_func(GB, manipulator, atom_positions, unique_id):
            call_count[0] += 1
            if crash_after is not None and call_count[0] > crash_after:
                raise RuntimeError(f"Simulated crash at call {call_count[0]}")
            path = f"{unique_id}_{call_count[0]}.data"
            GB.write_lammps(path, atom_positions,
                            manipulator.parents[0].box_dims)
            return 2.0 - call_count[0] * 0.001, path

        return energy_func

    def _make_minimizer(self, energy_func):
        return MonteCarloMinimizer(
            self.gb,
            energy_func,
            ["translate_right_grain"],
            seed=0,
        )

    def _make_sequence_energy_func(self, energies, root):
        """Return an evaluator writing deterministic relaxed files beneath ``root``."""
        values = iter(energies)
        call_count = [0]
        root.mkdir(parents=True, exist_ok=True)

        def energy_func(GB, manipulator, atom_positions, unique_id):
            call_count[0] += 1
            path = root / f"{unique_id}_{call_count[0]}.data"
            GB.write_lammps(
                str(path),
                atom_positions,
                manipulator.parents[0].box_dims,
            )
            return next(values), str(path)

        return energy_func

    def test_run_mc_no_checkpoint_no_file_created(self):
        mc = self._make_minimizer(self._make_energy_func())
        mc.run_MC(max_steps=2, unique_id=1)
        json_files = list(Path(self.tmpdir.name).glob("*.json"))
        pkl_files = list(Path(self.tmpdir.name).glob("*.pkl"))
        self.assertEqual(json_files, [])
        self.assertEqual(pkl_files, [])
        self.assertIsNone(mc.artifact_store)
        self.assertEqual(list(Path(self.tmpdir.name).glob("*.artifacts")), [])

    def test_mc_retention_prunes_superseded_accepted_source_after_commit(self):
        managed_root = Path(self.tmpdir.name) / "managed"
        energy_func = self._make_sequence_energy_func([2.0, 1.0], managed_root)
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
            self.gb,
            energy_func,
            ["translate_right_grain"],
            seed=0,
            retention_policy=policy,
            managed_artifact_root=managed_root,
        )
        cp = Path(self.tmpdir.name) / "mc_retention.json"

        mc.run_MC(max_steps=1, unique_id=41, checkpoint_file=cp)

        initial_source = managed_root / "initial41_1.data"
        current_source = managed_root / "41_2.data"
        archive = (
            Path(self.tmpdir.name)
            / "mc_retention.artifacts"
            / "structures"
            / "MC_41_s1.data"
        )
        self.assertFalse(initial_source.exists())
        self.assertTrue(current_source.exists())
        self.assertTrue(archive.is_file())
        with cp.open() as stream:
            state = json.load(stream)
        self.assertEqual(state["best_dump"], str(archive))
        records = state["state"]["artifact_store"]["records"]
        by_id = {record["candidate"]["candidate_id"]: record for record in records}
        self.assertEqual(by_id["MC_41_s1"]["pins"], ["best_result", "run_checkpoint"])
        self.assertEqual(
            by_id["MC_41_s1"]["retention_reasons"], ["rule:objective_best"]
        )

    def test_mc_retains_rejected_scientific_result_and_prunes_its_source(self):
        managed_root = Path(self.tmpdir.name) / "managed_rejected"
        energy_func = self._make_sequence_energy_func([1.0, 1000.0], managed_root)
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
            self.gb,
            energy_func,
            ["translate_right_grain"],
            seed=0,
            retention_policy=policy,
            managed_artifact_root=managed_root,
        )
        cp = Path(self.tmpdir.name) / "mc_rejected.json"

        mc.run_MC(max_steps=1, unique_id=42, checkpoint_file=cp)

        rejected_source = managed_root / "42_2.data"
        rejected_archive = (
            Path(self.tmpdir.name)
            / "mc_rejected.artifacts"
            / "structures"
            / "MC_42_s1.data"
        )
        self.assertFalse(rejected_source.exists())
        self.assertTrue(rejected_archive.is_file())
        with cp.open() as stream:
            state = json.load(stream)
        records = state["state"]["artifact_store"]["records"]
        rejected = next(
            record
            for record in records
            if record["candidate"]["candidate_id"] == "MC_42_s1"
        )
        self.assertEqual(rejected["pins"], [])
        self.assertEqual(rejected["retention_reasons"], ["rule:largest_objective"])
        manifest = Path(self.tmpdir.name) / "mc_rejected.artifacts" / "manifest.json"
        history = Path(self.tmpdir.name) / "mc_rejected.artifacts" / "history.jsonl"
        self.assertTrue(manifest.is_file())
        self.assertTrue(history.is_file())

    def test_mc_resume_rejects_retention_policy_mismatch(self):
        managed_root = Path(self.tmpdir.name) / "managed_mismatch"
        energy_func = self._make_sequence_energy_func([2.0, 1.0], managed_root)
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
        cp = Path(self.tmpdir.name) / "mc_mismatch.json"
        MonteCarloMinimizer(
            self.gb,
            energy_func,
            ["translate_right_grain"],
            seed=0,
            retention_policy=first_policy,
        ).run_MC(max_steps=1, unique_id=43, checkpoint_file=cp)

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
            self.gb,
            self._make_energy_func(),
            ["translate_right_grain"],
            seed=0,
            retention_policy=changed_policy,
        )
        with self.assertRaisesRegex(
            GBMinimizerError,
            "artifact retention policy signature mismatch",
        ):
            resumed.run_MC(max_steps=2, checkpoint_file=cp)

    def test_mc_pruning_requires_checkpoint_file(self):
        managed_root = Path(self.tmpdir.name) / "managed_no_checkpoint"
        policy = ArtifactRetentionPolicy(prune=True)
        mc = MonteCarloMinimizer(
            self.gb,
            self._make_energy_func(),
            ["translate_right_grain"],
            seed=0,
            retention_policy=policy,
            managed_artifact_root=managed_root,
        )
        with self.assertRaisesRegex(
            GBMinimizerValueError,
            "requires checkpoint_file",
        ):
            mc.run_MC(max_steps=1, unique_id=44)

    def test_mc_cleanup_failure_leaks_source_but_checkpoint_resumes(self):
        managed_root = Path(self.tmpdir.name) / "managed_cleanup_failure"
        energy_func = self._make_sequence_energy_func([1.0, 1000.0], managed_root)
        policy = ArtifactRetentionPolicy(prune=True)

        def failing_cleanup(_request):
            raise OSError("backend cleanup failed")

        cp = Path(self.tmpdir.name) / "mc_cleanup_failure.json"
        mc = MonteCarloMinimizer(
            self.gb,
            energy_func,
            ["translate_right_grain"],
            seed=0,
            retention_policy=policy,
            cleanup_candidate=failing_cleanup,
        )
        with self.assertWarnsRegex(RuntimeWarning, "Artifact cleanup failed"):
            mc.run_MC(max_steps=1, unique_id=45, checkpoint_file=cp)

        leaked_source = managed_root / "45_2.data"
        self.assertTrue(cp.is_file())
        self.assertTrue(leaked_source.is_file())

        resumed = MonteCarloMinimizer(
            self.gb,
            self._make_sequence_energy_func([1000.0], managed_root),
            ["translate_right_grain"],
            seed=0,
            retention_policy=policy,
            cleanup_candidate=lambda _request: None,
        )
        resumed.run_MC(max_steps=2, checkpoint_file=cp)
        with cp.open() as stream:
            state = json.load(stream)
        self.assertEqual(state["progress_index"], 2)

    def test_run_mc_checkpoint_kept_on_completion(self):
        mc = self._make_minimizer(self._make_energy_func())
        cp = Path(self.tmpdir.name) / "mc.json"
        mc.run_MC(max_steps=3, unique_id=2, checkpoint_file=cp)
        self.assertTrue(cp.exists())

    def test_run_mc_checkpoint_file_is_valid_json(self):
        mc = self._make_minimizer(self._make_energy_func(crash_after=3))
        cp = Path(self.tmpdir.name) / "mc.json"
        with self.assertRaises(RuntimeError):
            mc.run_MC(max_steps=10, unique_id=3, checkpoint_file=cp)
        self.assertTrue(cp.exists())
        with open(cp) as f:
            state = json.load(f)
        for key in ("schema_version", "minimizer", "progress_unit",
                    "progress_index", "best_energy", "rng_state", "run_params", "state"):
            self.assertIn(key, state)
        for key in ("T", "rejection_count", "prev_gbe", "GBE_vals",
                    "accepted_idx", "operation_list", "current_structure_dump"):
            self.assertIn(key, state["state"])
        self.assertEqual(state["minimizer"], "MonteCarloMinimizer")
        self.assertEqual(state["progress_unit"], "step")

    def test_run_mc_checkpoint_format_pickle(self):
        mc = self._make_minimizer(self._make_energy_func(crash_after=3))
        cp = Path(self.tmpdir.name) / "mc.pkl"
        with self.assertRaises(RuntimeError):
            mc.run_MC(max_steps=10, unique_id=4, checkpoint_file=cp,
                      checkpoint_format="pickle")
        self.assertTrue(cp.exists())
        with open(cp, "rb") as f:
            state = pickle.load(f)
        self.assertIn("progress_index", state)
        self.assertIn("GBE_vals", state["state"])

    def test_run_mc_resume_from_json(self):
        # First run: crash after 3 calls (initial + step1 + step2 succeed; step3 raises)
        mc = self._make_minimizer(self._make_energy_func(crash_after=3))
        cp = Path(self.tmpdir.name) / "mc_resume.json"
        with self.assertRaises(RuntimeError):
            mc.run_MC(max_steps=10, unique_id=5, checkpoint_file=cp)
        with open(cp) as f:
            saved = json.load(f)
        resumed_from_step = saved["progress_index"]
        gbe_count_before_resume = len(mc.GBE_vals)
        self.assertGreater(resumed_from_step, 0)
        # Resume: swap in a non-crashing energy func and continue
        mc.gb_energy_func = self._make_energy_func()
        mc.run_MC(max_steps=10, unique_id=5, checkpoint_file=cp)
        self.assertTrue(cp.exists())
        self.assertGreater(len(mc.GBE_vals), gbe_count_before_resume)

    def test_run_mc_resume_from_pickle(self):
        mc = self._make_minimizer(self._make_energy_func(crash_after=3))
        cp = Path(self.tmpdir.name) / "mc_resume.pkl"
        with self.assertRaises(RuntimeError):
            mc.run_MC(max_steps=10, unique_id=6, checkpoint_file=cp,
                      checkpoint_format="pickle")
        self.assertTrue(cp.exists())
        mc.gb_energy_func = self._make_energy_func()
        mc.run_MC(max_steps=10, unique_id=6, checkpoint_file=cp,
                  checkpoint_format="pickle")
        self.assertTrue(cp.exists())

    def test_run_mc_corrupted_checkpoint_raises(self):
        cp = Path(self.tmpdir.name) / "corrupt.json"
        cp.write_bytes(b"not valid json {{{")
        mc = self._make_minimizer(self._make_energy_func())
        with self.assertRaises(GBMinimizerError):
            mc.run_MC(max_steps=5, unique_id=7, checkpoint_file=cp)

    def test_run_mc_invalid_format_raises(self):
        mc = self._make_minimizer(self._make_energy_func())
        cp = Path(self.tmpdir.name) / "mc.hdf5"
        with self.assertRaises(GBMinimizerValueError):
            mc.run_MC(max_steps=5, unique_id=8,
                      checkpoint_file=cp, checkpoint_format="hdf5")

    def test_run_mc_checkpoint_interval_respected(self):
        # crash_after=5: calls 1(initial),2(step1),3(step2),4(step3),5(step4) succeed;
        # call 6 (step5) raises. With interval=3, only step3 is saved (i%3==0).
        mc = self._make_minimizer(self._make_energy_func(crash_after=5))
        cp = Path(self.tmpdir.name) / "mc_interval.json"
        with self.assertRaises(RuntimeError):
            mc.run_MC(max_steps=10, unique_id=9,
                      checkpoint_file=cp, checkpoint_interval=3)
        with open(cp) as f:
            state = json.load(f)
        self.assertEqual(state["progress_index"], 3)

    def test_resume_without_unique_id_restores_original_label(self):
        """Resuming run_MC without passing unique_id continues under the
        original label stored in the checkpoint."""
        cp = Path(self.tmpdir.name) / "mc_uid.json"
        mc1 = self._make_minimizer(self._make_energy_func())
        mc1.run_MC(max_steps=2, unique_id=7777, checkpoint_file=cp)

        mc2 = self._make_minimizer(self._make_energy_func())
        mc2.run_MC(max_steps=4, checkpoint_file=cp)  # no unique_id

        with open(cp) as f:
            saved = json.load(f)
        self.assertEqual(saved["run_params"]["unique_id"], "7777")

    def test_two_fresh_runs_without_unique_id_use_different_labels(self):
        """Each fresh run_MC() call without unique_id must generate a distinct label."""
        cp1 = Path(self.tmpdir.name) / "mc1.json"
        cp2 = Path(self.tmpdir.name) / "mc2.json"
        self._make_minimizer(self._make_energy_func()).run_MC(
            max_steps=1, checkpoint_file=cp1)
        self._make_minimizer(self._make_energy_func()).run_MC(
            max_steps=1, checkpoint_file=cp2)
        with open(cp1) as f:
            uid1 = json.load(f)["run_params"]["unique_id"]
        with open(cp2) as f:
            uid2 = json.load(f)["run_params"]["unique_id"]
        self.assertNotEqual(uid1, uid2)

    def test_resume_restores_cooldown_rate_from_checkpoint(self):
        """Resuming without passing cooldown_rate uses the value from the checkpoint."""
        cp = Path(self.tmpdir.name) / "mc_cr.json"
        self._make_minimizer(self._make_energy_func()).run_MC(
            max_steps=2, cooldown_rate=0.8, unique_id=1, checkpoint_file=cp)
        # Resume without specifying cooldown_rate (default is 1.0)
        self._make_minimizer(self._make_energy_func()).run_MC(
            max_steps=4, checkpoint_file=cp)
        with open(cp) as f:
            self.assertAlmostEqual(
                json.load(f)["run_params"]["cooldown_rate"], 0.8)

    def test_resume_restores_min_steps_from_checkpoint(self):
        """Resuming without passing min_steps uses the value stored in the checkpoint."""
        cp = Path(self.tmpdir.name) / "mc_ms.json"
        self._make_minimizer(self._make_energy_func()).run_MC(
            max_steps=2, min_steps=5, unique_id=2, checkpoint_file=cp)
        self._make_minimizer(self._make_energy_func()).run_MC(
            max_steps=10, checkpoint_file=cp)
        with open(cp) as f:
            self.assertEqual(json.load(f)["run_params"]["min_steps"], 5)


if __name__ == "__main__":
    unittest.main()
