# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import json
import math
import os
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from GBOpt.GBMaker import GBMaker
from GBOpt.GBMinimizer import GBMinimizerError, GBMinimizerValueError, MonteCarloMinimizer


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

    def test_run_mc_no_checkpoint_no_file_created(self):
        mc = self._make_minimizer(self._make_energy_func())
        mc.run_MC(max_steps=2, unique_id=1)
        json_files = list(Path(self.tmpdir.name).glob("*.json"))
        pkl_files = list(Path(self.tmpdir.name).glob("*.pkl"))
        self.assertEqual(json_files, [])
        self.assertEqual(pkl_files, [])

    def test_run_mc_checkpoint_deleted_on_completion(self):
        mc = self._make_minimizer(self._make_energy_func())
        cp = Path(self.tmpdir.name) / "mc.json"
        mc.run_MC(max_steps=3, unique_id=2, checkpoint_file=cp)
        self.assertFalse(cp.exists())

    def test_run_mc_checkpoint_file_is_valid_json(self):
        mc = self._make_minimizer(self._make_energy_func(crash_after=3))
        cp = Path(self.tmpdir.name) / "mc.json"
        with self.assertRaises(RuntimeError):
            mc.run_MC(max_steps=10, unique_id=3, checkpoint_file=cp)
        self.assertTrue(cp.exists())
        with open(cp) as f:
            state = json.load(f)
        for key in (
            "step_index", "T", "rejection_count", "min_gbe", "prev_gbe",
            "GBE_vals", "accepted_idx", "operation_list", "rng_state",
            "current_structure_dump", "run_params",
        ):
            self.assertIn(key, state)

    def test_run_mc_checkpoint_format_pickle(self):
        mc = self._make_minimizer(self._make_energy_func(crash_after=3))
        cp = Path(self.tmpdir.name) / "mc.pkl"
        with self.assertRaises(RuntimeError):
            mc.run_MC(max_steps=10, unique_id=4, checkpoint_file=cp,
                      checkpoint_format="pickle")
        self.assertTrue(cp.exists())
        with open(cp, "rb") as f:
            state = pickle.load(f)
        self.assertIn("step_index", state)
        self.assertIn("GBE_vals", state)

    def test_run_mc_resume_from_json(self):
        # First run: crash after 3 calls (initial + step1 + step2 succeed; step3 raises)
        mc = self._make_minimizer(self._make_energy_func(crash_after=3))
        cp = Path(self.tmpdir.name) / "mc_resume.json"
        with self.assertRaises(RuntimeError):
            mc.run_MC(max_steps=10, unique_id=5, checkpoint_file=cp)
        with open(cp) as f:
            saved = json.load(f)
        resumed_from_step = saved["step_index"]
        gbe_count_before_resume = len(mc.GBE_vals)
        self.assertGreater(resumed_from_step, 0)
        # Resume: swap in a non-crashing energy func and continue
        mc.gb_energy_func = self._make_energy_func()
        mc.run_MC(max_steps=10, unique_id=5, checkpoint_file=cp)
        self.assertFalse(cp.exists())
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
        self.assertFalse(cp.exists())

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
        self.assertEqual(state["step_index"], 3)


if __name__ == "__main__":
    unittest.main()
