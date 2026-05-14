# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import json
import math
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from GBOpt.GBMaker import GBMaker
from GBOpt.GBMinimizer import GBMinimizerError, GBMinimizerValueError, GeneticAlgorithmMinimizer


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

        best_energy, best_dump = minimizer.run_GA(unique_id=1)

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
        minimizer.run_GA(unique_id=2)

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
        minimizer.run_GA(unique_id=3)

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
        minimizer.run_GA(unique_id=2)
        self.assertEqual(len(minimizer.history), minimizer.generations)

        minimizer.run_GA(unique_id=2)
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

    def test_run_ga_checkpoint_deleted_on_completion(self):
        cp = Path(self.tmpdir.name) / "ga.json"
        minimizer = self._make_minimizer()
        minimizer.run_GA(unique_id=10, checkpoint_file=cp)
        self.assertFalse(cp.exists())

    def test_run_ga_checkpoint_file_is_valid_json(self):
        """After a simulated crash (via mock), the checkpoint file is valid JSON."""
        cp = Path(self.tmpdir.name) / "ga.json"
        minimizer = self._make_minimizer(generations=3)

        # Crash immediately after the first checkpoint save.
        original_save = minimizer._save_checkpoint

        def save_then_crash(checkpoint_file, checkpoint_format, state):
            original_save(checkpoint_file, checkpoint_format, state)
            raise RuntimeError("Simulated crash after gen-0 checkpoint")

        minimizer._save_checkpoint = save_then_crash

        with self.assertRaises(RuntimeError):
            minimizer.run_GA(unique_id=11, checkpoint_file=cp)

        self.assertTrue(cp.exists())
        with open(cp) as f:
            state = json.load(f)
        for key in (
            "generation", "best_energy", "best_dump", "GBE_vals",
            "history", "rng_state", "run_params", "population_lineages",
        ):
            self.assertIn(key, state)
        self.assertEqual(state["generation"], 0)

    def test_run_ga_checkpoint_format_pickle(self):
        cp = Path(self.tmpdir.name) / "ga.pkl"
        minimizer = self._make_minimizer(generations=3)
        original_save = minimizer._save_checkpoint

        def save_then_crash(checkpoint_file, checkpoint_format, state):
            original_save(checkpoint_file, checkpoint_format, state)
            raise RuntimeError("Simulated crash")

        minimizer._save_checkpoint = save_then_crash

        with self.assertRaises(RuntimeError):
            minimizer.run_GA(unique_id=12, checkpoint_file=cp,
                             checkpoint_format="pickle")

        self.assertTrue(cp.exists())
        with open(cp, "rb") as f:
            state = pickle.load(f)
        self.assertIn("generation", state)

    def test_run_ga_resume_gbe_vals_not_duplicated(self):
        """Resuming from a gen-0 checkpoint adds gens 1+ without re-running gen 0."""
        cp = Path(self.tmpdir.name) / "ga_resume.json"
        minimizer = self._make_minimizer(generations=3)

        original_save = minimizer._save_checkpoint

        def save_then_crash(checkpoint_file, checkpoint_format, state):
            original_save(checkpoint_file, checkpoint_format, state)
            raise RuntimeError("Simulated crash after gen-0 checkpoint")

        minimizer._save_checkpoint = save_then_crash

        with self.assertRaises(RuntimeError):
            minimizer.run_GA(unique_id=13, checkpoint_file=cp)

        # Restore normal save and resume
        minimizer._save_checkpoint = original_save
        minimizer.run_GA(unique_id=13, checkpoint_file=cp)

        # GBE_vals: 1 initial + 3 generations = 4 entries
        self.assertEqual(len(minimizer.GBE_vals), minimizer.generations + 1)
        self.assertFalse(cp.exists())

    def test_run_ga_resume_history_not_doubled(self):
        """Resumed run produces exactly `generations` history entries, not more."""
        cp = Path(self.tmpdir.name) / "ga_hist.json"
        minimizer = self._make_minimizer(generations=3)

        original_save = minimizer._save_checkpoint

        def save_then_crash(checkpoint_file, checkpoint_format, state):
            original_save(checkpoint_file, checkpoint_format, state)
            raise RuntimeError("Simulated crash after gen-0 checkpoint")

        minimizer._save_checkpoint = save_then_crash

        with self.assertRaises(RuntimeError):
            minimizer.run_GA(unique_id=14, checkpoint_file=cp)

        minimizer._save_checkpoint = original_save
        minimizer.run_GA(unique_id=14, checkpoint_file=cp)

        self.assertEqual(len(minimizer.history), minimizer.generations)

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


if __name__ == "__main__":
    unittest.main()
