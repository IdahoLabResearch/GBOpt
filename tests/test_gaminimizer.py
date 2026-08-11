# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import json
import math
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from GBOpt.Checkpoint import CheckpointStore
from GBOpt.GBMaker import GBMaker
from GBOpt.GBMinimizer import (
    GBMinimizerError,
    GBMinimizerValueError,
    GeneticAlgorithmMinimizer,
)


class TestGeneticAlgorithmMinimizer(unittest.TestCase):

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

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run_ga_returns_best_energy_and_dump(self):
        def fake_energy_func(GB, manipulator, atom_positions, unique_id):
            dump_file = Path(self.tmpdir.name) / f"{unique_id}.data"
            GB.write_lammps(
                str(dump_file),
                atom_positions,
                manipulator.parents[0].box_dims,
            )
            energy = float(np.mean(atom_positions["x"]))
            return energy, str(dump_file)

        minimizer = GeneticAlgorithmMinimizer(
            self.gb,
            fake_energy_func,
            ["insert_atoms", "remove_atoms", "translate_right_grain"],
            seed=0,
            population_size=4,
            generations=2,
            keep_top_pct=25,
            intermediate_pct=75,
        )
        cp = Path(self.tmpdir.name) / "ga1.json"
        best_energy, best_dump = minimizer.run_GA(
            unique_id=1, checkpoint_file=cp)

        self.assertIsInstance(best_energy, float)
        self.assertTrue(Path(best_dump).exists())
        self.assertEqual(len(minimizer.GBE_vals), minimizer.generations + 1)

    def test_history_populated_after_run(self):
        def fake_energy_func(GB, manipulator, atom_positions, unique_id):
            dump_file = Path(self.tmpdir.name) / f"{unique_id}.data"
            GB.write_lammps(
                str(dump_file),
                atom_positions,
                manipulator.parents[0].box_dims,
            )
            return float(np.mean(atom_positions["x"])), str(dump_file)

        minimizer = GeneticAlgorithmMinimizer(
            self.gb,
            fake_energy_func,
            ["insert_atoms", "remove_atoms", "translate_right_grain"],
            seed=0,
            population_size=4,
            generations=2,
            keep_top_pct=25,
            intermediate_pct=75,
        )
        cp = Path(self.tmpdir.name) / "ga2.json"
        minimizer.run_GA(unique_id=2, checkpoint_file=cp)

        # One entry per generation
        self.assertEqual(len(minimizer.history), minimizer.generations)

        for gen_idx, gen_history in enumerate(minimizer.history):
            # Each generation records population_size (lineage, energy) pairs
            self.assertEqual(len(gen_history), minimizer.population_size)

            for lineage, energy in gen_history:
                # Lineage is a non-empty list of strings
                self.assertIsInstance(lineage, list)
                self.assertGreater(len(lineage), 0)
                self.assertIsInstance(lineage[0], str)
                # First element is a known operation label
                op = lineage[0]
                self.assertTrue(
                    op in {"slice_and_merge", "carryover", "START"}
                    or op.startswith("shift") or op.startswith("add")
                    or op.startswith("remove"),
                    f"Unexpected operation label: {op!r}",
                )

            # Energies in history match the corresponding GBE_vals entry
            # (GBE_vals[0] is the initial eval, so gen 0 -> GBE_vals[1])
            self.assertEqual(
                [e for _, e in gen_history],
                minimizer.GBE_vals[gen_idx + 1],
            )

    def test_failed_generation_appends_to_history(self):
        def fake_energy_func(GB, manipulator, atom_positions, unique_id):
            # Force all generation-0 candidates to fail
            if "_g0_c" in str(unique_id):
                raise RuntimeError("Simulated failure")
            dump_file = Path(self.tmpdir.name) / f"{unique_id}.data"
            GB.write_lammps(
                str(dump_file),
                atom_positions,
                manipulator.parents[0].box_dims,
            )
            return float(np.mean(atom_positions["x"])), str(dump_file)

        minimizer = GeneticAlgorithmMinimizer(
            self.gb,
            fake_energy_func,
            ["insert_atoms", "remove_atoms", "translate_right_grain"],
            seed=0,
            population_size=4,
            generations=2,
            keep_top_pct=25,
            intermediate_pct=75,
        )
        cp = Path(self.tmpdir.name) / "ga3.json"
        minimizer.run_GA(unique_id=3, checkpoint_file=cp)

        # history still has one entry per generation despite the failure
        self.assertEqual(len(minimizer.history), minimizer.generations)

        # Generation 0 failed entirely — all energies should be PENALTY
        PENALTY = 1.0e30
        failed_gen = minimizer.history[0]
        self.assertEqual(len(failed_gen), minimizer.population_size)
        for lineage, energy in failed_gen:
            self.assertEqual(energy, PENALTY)

        # Generation 1 recovered and has real energies
        recovered_gen = minimizer.history[1]
        self.assertEqual(len(recovered_gen), minimizer.population_size)
        for _, energy in recovered_gen:
            self.assertLess(energy, PENALTY)

    def test_ga_history_never_exceeds_generations(self):
        def fake_energy_func(GB, manipulator, atom_positions, unique_id):
            dump_file = Path(self.tmpdir.name) / f"{unique_id}.data"
            GB.write_lammps(
                str(dump_file),
                atom_positions,
                manipulator.parents[0].box_dims,
            )
            return float(np.mean(atom_positions["x"])), str(dump_file)

        minimizer = GeneticAlgorithmMinimizer(
            self.gb,
            fake_energy_func,
            ["insert_atoms", "remove_atoms", "translate_right_grain"],
            seed=0,
            population_size=4,
            generations=2,
            keep_top_pct=25,
            intermediate_pct=75,
        )
        cp = Path(self.tmpdir.name) / "ga_never_exceed.json"
        minimizer.run_GA(unique_id=2, checkpoint_file=cp)
        self.assertEqual(len(minimizer.history), minimizer.generations)

        minimizer.run_GA(unique_id=2, checkpoint_file=cp)
        self.assertEqual(len(minimizer.history), minimizer.generations)


class TestGeneticAlgorithmMinimizerCheckpointing(unittest.TestCase):

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

    def tearDown(self):
        self.tmpdir.cleanup()

    def _fake_energy_func(self, GB, manipulator, atom_positions, unique_id):
        dump_file = Path(self.tmpdir.name) / f"{unique_id}.data"
        GB.write_lammps(
            str(dump_file),
            atom_positions,
            manipulator.parents[0].box_dims,
        )
        return float(np.mean(atom_positions["x"])), str(dump_file)

    def _make_minimizer(self, generations=2):
        return GeneticAlgorithmMinimizer(
            self.gb,
            self._fake_energy_func,
            ["insert_atoms", "remove_atoms", "translate_right_grain"],
            seed=0,
            population_size=4,
            generations=generations,
            keep_top_pct=25,
            intermediate_pct=75,
        )

    def test_run_ga_checkpoint_kept_on_completion(self):
        cp = Path(self.tmpdir.name) / "ga.json"
        minimizer = self._make_minimizer()
        minimizer.run_GA(unique_id=10, checkpoint_file=cp)
        self.assertTrue(cp.exists())

    def test_run_ga_checkpoint_file_is_valid_json(self):
        """After a simulated crash (via mock), the checkpoint file is valid JSON."""
        from unittest.mock import patch

        from GBOpt.Checkpoint import CheckpointStore
        cp = Path(self.tmpdir.name) / "ga.json"
        minimizer = self._make_minimizer(generations=3)

        call_count = [0]
        original_save = CheckpointStore._save

        def save_then_crash(self_store, state):
            original_save(self_store, state)
            call_count[0] += 1
            if call_count[0] >= 1:
                raise RuntimeError("Simulated crash after gen-0 checkpoint")

        with patch.object(CheckpointStore, '_save', save_then_crash):
            with self.assertRaises(RuntimeError):
                minimizer.run_GA(unique_id=11, checkpoint_file=cp)

        self.assertTrue(cp.exists())
        with open(cp) as f:
            state = json.load(f)
        for key in ("schema_version", "minimizer", "progress_unit", "progress_index",
                    "best_energy", "best_dump", "rng_state", "run_params", "state"):
            self.assertIn(key, state)
        for key in ("GBE_vals", "history", "population_lineages"):
            self.assertIn(key, state["state"])
        self.assertEqual(state["progress_index"], 0)
        self.assertEqual(state["minimizer"], "GeneticAlgorithmMinimizer")

    def test_run_ga_checkpoint_format_pickle(self):
        from unittest.mock import patch

        from GBOpt.Checkpoint import CheckpointStore
        cp = Path(self.tmpdir.name) / "ga.pkl"
        minimizer = self._make_minimizer(generations=3)

        call_count = [0]
        original_save = CheckpointStore._save

        def save_then_crash(self_store, state):
            original_save(self_store, state)
            call_count[0] += 1
            if call_count[0] >= 1:
                raise RuntimeError("Simulated crash")

        with patch.object(CheckpointStore, '_save', save_then_crash):
            with self.assertRaises(RuntimeError):
                minimizer.run_GA(unique_id=12, checkpoint_file=cp,
                                 checkpoint_format="pickle")

        self.assertTrue(cp.exists())
        with open(cp, "rb") as f:
            state = pickle.load(f)
        self.assertIn("progress_index", state)

    def test_run_ga_resume_gbe_vals_not_duplicated(self):
        """Resuming from a gen-0 checkpoint adds gens 1+ without re-running gen 0."""
        cp = Path(self.tmpdir.name) / "ga_resume.json"
        # First run: 1 generation → checkpoint saved after gen 0
        minimizer = self._make_minimizer(generations=1)
        minimizer.run_GA(unique_id=13, checkpoint_file=cp)
        # Resume: total 3 generations
        minimizer2 = self._make_minimizer(generations=3)
        minimizer2.run_GA(unique_id=13, checkpoint_file=cp)
        # GBE_vals: 1 initial + 3 generations = 4 entries
        self.assertEqual(len(minimizer2.GBE_vals), minimizer2.generations + 1)

    def test_run_ga_resume_history_not_doubled(self):
        """Resumed run produces exactly `generations` history entries, not more."""
        cp = Path(self.tmpdir.name) / "ga_hist.json"
        minimizer = self._make_minimizer(generations=1)
        minimizer.run_GA(unique_id=14, checkpoint_file=cp)
        minimizer2 = self._make_minimizer(generations=3)
        minimizer2.run_GA(unique_id=14, checkpoint_file=cp)
        self.assertEqual(len(minimizer2.history), minimizer2.generations)

    def test_run_ga_corrupted_checkpoint_raises(self):
        cp = Path(self.tmpdir.name) / "corrupt.json"
        cp.write_bytes(b"not valid json {{{")
        minimizer = self._make_minimizer()
        with self.assertRaises(GBMinimizerError):
            minimizer.run_GA(unique_id=15, checkpoint_file=cp)

    def test_run_ga_invalid_format_raises(self):
        cp = Path(self.tmpdir.name) / "ga.hdf5"
        minimizer = self._make_minimizer()
        with self.assertRaises(GBMinimizerValueError):
            minimizer.run_GA(unique_id=16, checkpoint_file=cp,
                             checkpoint_format="yaml")


class TestGAIntraGenerationCheckpointing(unittest.TestCase):

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

    def tearDown(self):
        self.tmpdir.cleanup()

    def _fake_energy_func(self, GB, manipulator, atom_positions, unique_id):
        dump_file = Path(self.tmpdir.name) / f"{unique_id}.data"
        GB.write_lammps(str(dump_file), atom_positions,
                        manipulator.parents[0].box_dims)
        return float(np.mean(atom_positions["x"])), str(dump_file)

    def _make_minimizer(self, generations=2, batch_func=None):
        return GeneticAlgorithmMinimizer(
            self.gb,
            self._fake_energy_func,
            ["insert_atoms", "remove_atoms", "translate_right_grain"],
            seed=0,
            population_size=4,
            generations=generations,
            keep_top_pct=25,
            intermediate_pct=75,
            gb_batch_energy_func=batch_func,
        )

    def test_single_eval_resumes_skips_completed_candidates(self):
        """Crash after k candidates are checkpointed in gen-0; resume evaluates only N-k."""
        from GBOpt.Checkpoint import CandidateCheckpoint
        cp = Path(self.tmpdir.name) / "ga_intra.json"
        crash_after = 2

        # Crash by raising from gen_checkpoint.record() after crash_after saves.
        # This propagates out of run_GA because record() is called outside the
        # evaluation try/except block.
        original_record = CandidateCheckpoint.record
        record_calls = {"n": 0}

        def crashing_record(self_cp, unique_id, energy, dump):
            record_calls["n"] += 1
            original_record(self_cp, unique_id, energy, dump)
            if record_calls["n"] >= crash_after:
                raise RuntimeError(
                    "Simulated mid-gen crash via checkpoint record")

        CandidateCheckpoint.record = crashing_record
        minimizer = self._make_minimizer(generations=2)
        try:
            with self.assertRaises(RuntimeError):
                minimizer.run_GA(unique_id=20, checkpoint_file=cp)
        finally:
            CandidateCheckpoint.record = original_record

        # Resume: count how many gen-0 evaluations are made
        resume_calls = {"n": 0}
        original_func = self._fake_energy_func

        def tracking_func(GB, manipulator, atom_positions, unique_id):
            if "_g0_c" in str(unique_id):
                resume_calls["n"] += 1
            return original_func(GB, manipulator, atom_positions, unique_id)

        minimizer2 = GeneticAlgorithmMinimizer(
            self.gb,
            tracking_func,
            ["insert_atoms", "remove_atoms", "translate_right_grain"],
            seed=0,
            population_size=4,
            generations=2,
            keep_top_pct=25,
            intermediate_pct=75,
        )
        minimizer2.run_GA(unique_id=20, checkpoint_file=cp)

        # crash_after candidates were recorded; only the rest need re-evaluation
        self.assertEqual(resume_calls["n"],
                         minimizer2.population_size - crash_after)

    def test_gen_checkpoint_deleted_after_generation_completes(self):
        """The per-iteration checkpoint file is absent after the generation finishes."""
        from GBOpt.Checkpoint import CandidateCheckpoint
        cp = Path(self.tmpdir.name) / "ga_gencp.json"
        minimizer = self._make_minimizer()
        minimizer.run_GA(unique_id=21, checkpoint_file=cp)

        iter0_path = CandidateCheckpoint._derive_path(cp, 0)
        iter1_path = CandidateCheckpoint._derive_path(cp, 1)
        self.assertFalse(iter0_path.exists())
        self.assertFalse(iter1_path.exists())

    def test_gbe_vals_not_duplicated_after_intra_gen_resume(self):
        """Crash mid-gen via checkpoint record and resume; GBE_vals stays at generations+1."""
        from GBOpt.Checkpoint import CandidateCheckpoint
        cp = Path(self.tmpdir.name) / "ga_nodup.json"

        # Crash after the 2nd candidate in gen-0 is recorded to the iter checkpoint.
        # gen_checkpoint.record() is called outside the evaluation try/except, so
        # the exception propagates up through run_GA.
        original_record = CandidateCheckpoint.record
        record_calls = {"n": 0}

        def crashing_record(self_cp, unique_id, energy, dump):
            record_calls["n"] += 1
            original_record(self_cp, unique_id, energy, dump)
            if record_calls["n"] >= 2:
                raise RuntimeError(
                    "Simulated mid-gen crash via checkpoint record")

        CandidateCheckpoint.record = crashing_record
        minimizer = self._make_minimizer(generations=2)
        try:
            with self.assertRaises(RuntimeError):
                minimizer.run_GA(unique_id=22, checkpoint_file=cp)
        finally:
            CandidateCheckpoint.record = original_record

        minimizer2 = self._make_minimizer(generations=2)
        minimizer2.run_GA(unique_id=22, checkpoint_file=cp)
        self.assertEqual(len(minimizer2.GBE_vals), minimizer2.generations + 1)

    def test_batch_func_without_checkpoint_kwarg_auto_wrapped(self):
        """Batch func without checkpoint kwarg triggers UserWarning and still runs."""
        def batch_no_checkpoint(GB, manips, structs, lineages, unique_ids):
            results = []
            for manip, struct, uid in zip(manips, structs, unique_ids):
                dump_file = Path(self.tmpdir.name) / f"{uid}.data"
                GB.write_lammps(str(dump_file), struct,
                                manip.parents[0].box_dims)
                results.append({
                    "energy": float(np.mean(struct["x"])),
                    "final_dump": str(dump_file),
                })
            return results

        with self.assertWarns(UserWarning):
            minimizer = GeneticAlgorithmMinimizer(
                self.gb,
                self._fake_energy_func,
                ["insert_atoms", "remove_atoms", "translate_right_grain"],
                seed=0,
                population_size=4,
                generations=1,
                keep_top_pct=25,
                intermediate_pct=75,
                gb_batch_energy_func=batch_no_checkpoint,
            )

        cp = Path(self.tmpdir.name) / "ga_batch_wrap.json"
        best_energy, _ = minimizer.run_GA(unique_id=23, checkpoint_file=cp)
        self.assertIsInstance(best_energy, float)

    def test_batch_func_with_checkpoint_kwarg_not_wrapped(self):
        """Batch func that declares checkpoint= receives a CandidateCheckpoint instance."""
        from GBOpt.Checkpoint import CandidateCheckpoint
        received_checkpoints = []

        def batch_with_checkpoint(GB, manips, structs, lineages, unique_ids,
                                  checkpoint=None):
            received_checkpoints.append(checkpoint)
            results = []
            for manip, struct, uid in zip(manips, structs, unique_ids):
                dump_file = Path(self.tmpdir.name) / f"{uid}.data"
                GB.write_lammps(str(dump_file), struct,
                                manip.parents[0].box_dims)
                results.append({
                    "energy": float(np.mean(struct["x"])),
                    "final_dump": str(dump_file),
                })
            return results

        minimizer = GeneticAlgorithmMinimizer(
            self.gb,
            self._fake_energy_func,
            ["insert_atoms", "remove_atoms", "translate_right_grain"],
            seed=0,
            population_size=4,
            generations=1,
            keep_top_pct=25,
            intermediate_pct=75,
            gb_batch_energy_func=batch_with_checkpoint,
        )
        cp = Path(self.tmpdir.name) / "ga_batch_cp.json"
        minimizer.run_GA(unique_id=24, checkpoint_file=cp)

        self.assertTrue(len(received_checkpoints) > 0)
        self.assertIsInstance(received_checkpoints[0], CandidateCheckpoint)

    def test_orphaned_gen_checkpoint_cleaned_up_on_resume(self):
        """A stale iter checkpoint for a completed generation is removed when resuming."""
        from unittest.mock import patch

        from GBOpt.Checkpoint import CandidateCheckpoint, CheckpointStore
        cp = Path(self.tmpdir.name) / "ga_orphan.json"
        minimizer = self._make_minimizer(generations=2)

        call_count = [0]
        original_save = CheckpointStore._save

        def save_then_crash(self_store, state):
            original_save(self_store, state)
            call_count[0] += 1
            if call_count[0] >= 1:
                raise RuntimeError("Crash after gen-0 main checkpoint")

        with patch.object(CheckpointStore, '_save', save_then_crash):
            with self.assertRaises(RuntimeError):
                minimizer.run_GA(unique_id=25, checkpoint_file=cp)

        # Manually leave a stale iter-0 checkpoint (simulates crash before delete)
        stale = CandidateCheckpoint._derive_path(cp, 0)
        if not stale.exists():
            stale.write_text('{"iteration_index": 0, "results": {}}')

        minimizer2 = self._make_minimizer(generations=2)
        minimizer2.run_GA(unique_id=25, checkpoint_file=cp)

        self.assertFalse(stale.exists())

    def test_no_checkpoint_file_when_not_specified(self):
        """No checkpoint file is created when checkpoint_file is not provided."""
        minimizer = self._make_minimizer(generations=1)
        uid = 9999
        cp_path = Path(f"ga_checkpoint_{uid}.json")

        # Ensure it doesn't already exist
        if cp_path.exists():
            cp_path.unlink()

        minimizer.run_GA(unique_id=uid)
        self.assertFalse(cp_path.exists())

    def test_pending_paths_in_checkpoint_after_generation(self):
        """
        Gen-0 checkpoint must record .pending paths in population_checkpoint_paths (not
        population_lineages), and those files must exist on disk so a resume can load
        them.
        """
        cp = Path(self.tmpdir.name) / "ga_pending.json"
        call_count = [0]
        original_save = CheckpointStore._save

        def save_then_crash(self_store, state):
            original_save(self_store, state)
            call_count[0] += 1
            if call_count[0] >= 1:
                raise RuntimeError("Simulated interrupt at gen-0 boundary")

        minimizer = self._make_minimizer(generations=2)
        with patch.object(CheckpointStore, "_save", save_then_crash):
            with self.assertRaises(RuntimeError):
                minimizer.run_GA(unique_id=32, checkpoint_file=cp)

        with open(cp) as f:
            saved = json.load(f)

        cp_paths = saved["state"]["population_checkpoint_paths"]
        self.assertEqual(len(cp_paths), minimizer.population_size)
        for path in cp_paths:
            self.assertTrue(
                str(path).endswith(".pending"),
                f"Expected .pending path in checkpoint, got {path!r}",
            )
            self.assertTrue(
                Path(path).exists(),
                f".pending file referenced by checkpoint is missing: {path!r}",
            )
        lineages = saved["state"]["population_lineages"]
        self.assertEqual(len(lineages), minimizer.population_size)
        for lineage in lineages:
            path = lineage[1]
            self.assertFalse(
                str(path).endswith(".pending"),
                f"Provenance path must not be .pending, got {path!r}",
            )
            self.assertTrue(
                Path(path).exists(),
                f".pending file referenced by checkpoint is missing: {path!r}",
            )

    def test_candidate_reconstruction_matches_continuous_run(self):
        """
        Resumed run produces identical GBE_vals as an uninterrupted run with the same seed
        """
        TOTAL_GENS = 3
        UID_CONT = 30
        UID_INTR = 31

        # Continuous run: all TOTAL_GENS in one shot
        cp_cont = Path(self.tmpdir.name) / "ga_cont.json"
        m_cont = self._make_minimizer(generations=TOTAL_GENS)
        m_cont.run_GA(unique_id=UID_CONT, checkpoint_file=cp_cont)

        # Interrupted+resumed run: crash after the gen-0 checkpoint is written
        cp_intr = Path(self.tmpdir.name) / "ga_intr.json"
        call_count = [0]
        original_save = CheckpointStore._save

        def save_then_crash(self_store, state):
            original_save(self_store, state)
            call_count[0] += 1
            if call_count[0] >= 1:
                raise RuntimeError("Simulated interrupt at gen-0 boundary")

        m_intr1 = self._make_minimizer(generations=TOTAL_GENS)
        with patch.object(CheckpointStore, "_save", save_then_crash):
            with self.assertRaises(RuntimeError):
                m_intr1.run_GA(unique_id=UID_INTR, checkpoint_file=cp_intr)

        m_intr2 = self._make_minimizer(generations=TOTAL_GENS)
        m_intr2.run_GA(unique_id=UID_INTR, checkpoint_file=cp_intr)

        # Both runs must produce the same energy trajectory (within file-I/O rounding)
        self.assertEqual(
            len(m_cont.GBE_vals), len(m_intr2.GBE_vals),
            "GBE_vals length differs between continuous and resumed run",
        )
        for gen_idx, (cont_gen, res_gen) in enumerate(
                zip(m_cont.GBE_vals, m_intr2.GBE_vals)):
            self.assertEqual(
                len(cont_gen), len(res_gen),
                f"Population size mismatch at GBE_vals[{gen_idx}]",
            )
            for cand_idx, (cont_e, res_e) in enumerate(zip(cont_gen, res_gen)):
                self.assertAlmostEqual(
                    cont_e, res_e, delta=1e-5,
                    msg=(
                        f"Energy mismatch at GBE_vals[{gen_idx}][{cand_idx}]: "
                        f"continuous={cont_e}, resumed={res_e}"
                    ),
                )

    def test_normal_completion_then_extension_matches_continuous_run(self):
        """A run completed cleanly (generations=1), then extended to generations=3
        via its checkpoint, must produce the same GBE_vals as an uninterrupted
        3-generation run."""
        TOTAL_GENS = 3
        UID_CONT = 33
        UID_EXT = 34

        # Continuous reference run
        cp_cont = Path(self.tmpdir.name) / "ga_cont_ext.json"
        m_cont = self._make_minimizer(generations=TOTAL_GENS)
        m_cont.run_GA(unique_id=UID_CONT, checkpoint_file=cp_cont)

        # Partial run: 1 generation to clean completion
        cp_ext = Path(self.tmpdir.name) / "ga_ext.json"
        m_part = self._make_minimizer(generations=1)
        m_part.run_GA(unique_id=UID_EXT, checkpoint_file=cp_ext)

        # Extend to TOTAL_GENS from the checkpoint
        m_full = self._make_minimizer(generations=TOTAL_GENS)
        m_full.run_GA(unique_id=UID_EXT, checkpoint_file=cp_ext)

        self.assertEqual(
            len(m_cont.GBE_vals), len(m_full.GBE_vals),
            "GBE_vals length differs between continuous and extended run",
        )
        for gen_idx, (cont_gen, ext_gen) in enumerate(
                zip(m_cont.GBE_vals, m_full.GBE_vals)):
            self.assertEqual(
                len(cont_gen), len(ext_gen),
                f"Population size mismatch at GBE_vals[{gen_idx}]",
            )
            for cand_idx, (cont_e, ext_e) in enumerate(zip(cont_gen, ext_gen)):
                self.assertAlmostEqual(
                    cont_e, ext_e, delta=1e-4,
                    msg=(
                        f"Energy mismatch at GBE_vals[{gen_idx}][{cand_idx}]: "
                        f"continuous={cont_e}, extended={ext_e}"
                    ),
                )

    def test_resume_fails_loudly_on_missing_pending_file(self):
        """GA resume raises GBMinimizerError when a population_checkpoint_paths
        entry is missing, rather than silently substituting best_dump."""
        cp = Path(self.tmpdir.name) / "ga_missing.json"
        original_save = CheckpointStore._save

        def save_then_crash(self_store, state):
            original_save(self_store, state)
            raise RuntimeError("Simulated crash after gen-0 checkpoint")

        minimizer = self._make_minimizer(generations=2)
        with patch.object(CheckpointStore, "_save", save_then_crash):
            with self.assertRaises(RuntimeError):
                minimizer.run_GA(unique_id=35, checkpoint_file=cp)

        with open(cp) as f:
            saved = json.load(f)
        Path(saved["state"]["population_checkpoint_paths"][0]).unlink()

        m2 = self._make_minimizer(generations=2)
        with self.assertRaises(GBMinimizerError):
            m2.run_GA(unique_id=35, checkpoint_file=cp)


if __name__ == "__main__":
    unittest.main()
