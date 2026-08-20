# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import json
import math
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from GBOpt.BoundarySpec import CSLExactSpec
from GBOpt.artifacts import ArtifactRetentionPolicy, KeepBest
from GBOpt.Checkpoint import CheckpointStore
from GBOpt.GBMaker import GBMaker
from GBOpt.GBManipulator import (
    CompositionAwareCrossoverError,
    GBManipulator,
    GBManipulatorValueError,
)
from GBOpt.GBMinimizer import (
    GBMinimizerError,
    GBMinimizerTypeError,
    GBMinimizerValueError,
    GeneticAlgorithmMinimizer,
    Mutator,
)
from GBOpt.GrainOwnership import (
    LEFT_GRAIN_LABEL,
    RIGHT_GRAIN_LABEL,
    GrainOwnership,
)


pytestmark = pytest.mark.filterwarnings(
    "ignore:File-backed Parent initialization without explicit grain ownership is "
    "deprecated.*:DeprecationWarning"
)


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

        def crashing_record(self_cp, unique_id, energy, dump, **kwargs):
            record_calls["n"] += 1
            original_record(self_cp, unique_id, energy, dump, **kwargs)
            if record_calls["n"] >= crash_after:
                raise RuntimeError(
                    "Simulated mid-gen crash via checkpoint record")

        minimizer = self._make_minimizer(generations=2)
        with patch.object(CandidateCheckpoint, "record", crashing_record):
            with self.assertRaises(RuntimeError):
                minimizer.run_GA(unique_id=20, checkpoint_file=cp)

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

        def crashing_record(self_cp, unique_id, energy, dump, **kwargs):
            record_calls["n"] += 1
            original_record(self_cp, unique_id, energy, dump, **kwargs)
            if record_calls["n"] >= 2:
                raise RuntimeError(
                    "Simulated mid-gen crash via checkpoint record")

        minimizer = self._make_minimizer(generations=2)
        with patch.object(CandidateCheckpoint, "record", crashing_record):
            with self.assertRaises(RuntimeError):
                minimizer.run_GA(unique_id=22, checkpoint_file=cp)

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

        with self.assertWarnsRegex(
            UserWarning,
            r"does not accept a 'checkpoint' kwarg",
        ):
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


@pytest.fixture
def ga_gb():
    return GBMaker.from_boundary_spec(
        3.52,
        "fcc",
        "Ni",
        CSLExactSpec(
            axis=(0, 0, 1),
            plane=(3, 1, 0),
            quat=(3, 0, 0, 1),
            sigma=5,
        ),
        mode="exact",
        gb_thickness=10.0,
        repeat_factor=2,
        x_dim_min=10.0,
        vacuum=8.0,
        interaction_distance=3.0,
    )


@pytest.fixture
def owned_ga(tmp_path):
    gb = GBMaker.from_boundary_spec(
        3.52,
        "fcc",
        "Ni",
        CSLExactSpec(
            axis=(0, 0, 1),
            plane=(3, 1, 0),
            quat=(3, 0, 0, 1),
            sigma=5,
        ),
        mode="exact",
        gb_thickness=10.0,
        repeat_factor=2,
        x_dim_min=10.0,
        vacuum=0.0,
        interaction_distance=3.0,
    )
    seed_path = tmp_path / "owned_initial.data"
    gb.write_lammps(str(seed_path), type_as_int=False, precision=12)
    labels = np.hstack(
        (
            np.full(
                len(gb.left_grain),
                LEFT_GRAIN_LABEL,
                dtype=np.int8,
            ),
            np.full(
                len(gb.right_grain),
                RIGHT_GRAIN_LABEL,
                dtype=np.int8,
            ),
        )
    )
    ownership = GrainOwnership(
        atom_ids=np.arange(1, len(gb.whole_system) + 1),
        labels=labels,
        gb_plane_x=gb.gb_plane_x,
        inplane_periodic=gb.inplane_periodic,
        left_grain_x_bounds=(gb.box_dims[0, 0], gb.gb_plane_x),
        right_grain_x_bounds=(gb.gb_plane_x, gb.box_dims[0, 1]),
        coordinate_tolerance=gb.epsilon,
        normal_topology=gb.normal_topology,
    )
    return gb, seed_path, ownership, labels


def _write_owned_evaluator_output(
    path,
    atoms,
    box_dims,
    *,
    move_row=None,
    plane=None,
    change_species_row=None,
):
    output = np.array(atoms, copy=True)
    if move_row is not None:
        output[move_row]["x"] = float(plane) + 0.25
    if change_species_row is not None:
        output[change_species_row]["name"] = "Cu"
    order = np.arange(len(output))[::-1]
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        stream.write("Owned evaluator output\n\n")
        stream.write(f"{len(output)} atoms\n")
        stream.write(f"{len(set(output['name'].tolist()))} atom types\n")
        stream.writelines(f"{lower:.12f} {upper:.12f} {axis}lo {axis}hi\n" for axis,
                          (lower, upper) in zip("xyz", box_dims, strict=True))
        stream.write("\nAtoms\n\n")
        for row in order:
            atom = output[row]
            stream.write(
                f"{row + 1} {atom['name']} {atom['x']:.12f} "
                f"{atom['y']:.12f} {atom['z']:.12f}\n"
            )


def test_run_ga_returns_best_energy_and_dump(ga_gb, tmp_path):
    def fake_energy_func(GB, manipulator, atom_positions, unique_id):
        dump_file = tmp_path / f"{unique_id}.data"
        GB.write_lammps(
            str(dump_file),
            atom_positions,
            manipulator.parents[0].box_dims,
        )
        energy = float(np.mean(atom_positions["x"]))
        return energy, str(dump_file)

    minimizer = GeneticAlgorithmMinimizer(
        ga_gb,
        fake_energy_func,
        ["insert_atoms", "remove_atoms", "translate_right_grain"],
        seed=0,
        population_size=4,
        generations=2,
        keep_top_pct=25,
        intermediate_pct=75,
    )

    checkpoint = tmp_path / "ga1.json"
    best_energy, best_dump = minimizer.run_GA(
        unique_id=1,
        checkpoint_file=checkpoint,
    )

    assert isinstance(best_energy, float)
    assert Path(best_dump).is_file()
    assert len(minimizer.GBE_vals) == minimizer.generations + 1


@pytest.mark.parametrize(
    ("slice_and_merge_pct", "expected_crossover_slots"),
    [
        pytest.param(0.0, 0, id="mutation-only"),
        pytest.param(50.0, 1, id="default-mixed"),
        pytest.param(100.0, 3, id="crossover-only"),
    ],
)
def test_slice_and_merge_pct_controls_legacy_offspring_mix(
    ga_gb,
    tmp_path,
    slice_and_merge_pct,
    expected_crossover_slots,
):
    def fake_energy_func(GB, manipulator, atom_positions, unique_id):
        dump_file = tmp_path / f"{unique_id}.data"
        GB.write_lammps(
            str(dump_file),
            atom_positions,
            manipulator.parents[0].box_dims,
        )
        return 0.0, str(dump_file)

    minimizer = GeneticAlgorithmMinimizer(
        ga_gb,
        fake_energy_func,
        ["translate_right_grain"],
        seed=0,
        population_size=4,
        generations=2,
        keep_top_pct=25,
        intermediate_pct=100,
        slice_and_merge_pct=slice_and_merge_pct,
    )

    minimizer.run_GA(unique_id=101)

    operations = [lineage[0] for lineage, _energy in minimizer.history[1]]
    crossover_slots = sum(
        operation == "slice_and_merge"
        or operation.startswith("crossover_fallback_")
        for operation in operations
    )
    assert operations.count("carryover") == 1
    assert crossover_slots == expected_crossover_slots
    assert len(operations) == minimizer.population_size


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        pytest.param(True, GBMinimizerTypeError, id="boolean"),
        pytest.param("50", GBMinimizerTypeError, id="string"),
        pytest.param(float("nan"), GBMinimizerValueError, id="nan"),
        pytest.param(-0.1, GBMinimizerValueError, id="negative"),
        pytest.param(100.1, GBMinimizerValueError, id="above-one-hundred"),
    ],
)
def test_slice_and_merge_pct_rejects_invalid_values(ga_gb, value, error_type):
    with pytest.raises(error_type, match="slice_and_merge_pct"):
        GeneticAlgorithmMinimizer(
            ga_gb,
            lambda *_args: (0.0, None),
            ["translate_right_grain"],
            slice_and_merge_pct=value,
        )


def test_legacy_carryover_cache_skips_only_unchanged_survivor(ga_gb, tmp_path):
    evaluated_ids = []

    def fake_energy_func(GB, manipulator, atom_positions, unique_id):
        evaluated_ids.append(str(unique_id))
        dump_file = tmp_path / f"{unique_id}.data"
        GB.write_lammps(
            str(dump_file),
            atom_positions,
            manipulator.parents[0].box_dims,
        )
        return 0.0, str(dump_file)

    minimizer = GeneticAlgorithmMinimizer(
        ga_gb,
        fake_energy_func,
        ["translate_right_grain"],
        seed=0,
        population_size=4,
        generations=2,
        keep_top_pct=25,
        intermediate_pct=100,
        reuse_carryover_evaluations=True,
    )

    minimizer.run_GA(unique_id=102)

    generation_one_ids = [uid for uid in evaluated_ids if "_g1_c" in uid]
    assert generation_one_ids == [
        "GA_102_g1_c1",
        "GA_102_g1_c2",
        "GA_102_g1_c3",
    ]
    assert minimizer.history[1][0][0][0] == "carryover"
    assert minimizer.history[1][0][1] == minimizer.history[0][0][1]


def test_legacy_carryover_cache_survives_checkpoint_extension(ga_gb, tmp_path):
    checkpoint = tmp_path / "cached.json"

    def write_energy(GB, manipulator, atom_positions, unique_id):
        dump_file = tmp_path / f"{unique_id}.data"
        GB.write_lammps(
            str(dump_file),
            atom_positions,
            manipulator.parents[0].box_dims,
        )
        return 0.0, str(dump_file)

    partial = GeneticAlgorithmMinimizer(
        ga_gb,
        write_energy,
        ["translate_right_grain"],
        seed=0,
        population_size=4,
        generations=1,
        keep_top_pct=25,
        intermediate_pct=100,
        reuse_carryover_evaluations=True,
    )
    partial.run_GA(unique_id=103, checkpoint_file=checkpoint)

    resumed_ids = []

    def tracking_energy(GB, manipulator, atom_positions, unique_id):
        resumed_ids.append(str(unique_id))
        return write_energy(GB, manipulator, atom_positions, unique_id)

    resumed = GeneticAlgorithmMinimizer(
        ga_gb,
        tracking_energy,
        ["translate_right_grain"],
        seed=0,
        population_size=4,
        generations=2,
        keep_top_pct=25,
        intermediate_pct=100,
        reuse_carryover_evaluations=True,
    )
    resumed.run_GA(unique_id=103, checkpoint_file=checkpoint)

    assert resumed_ids == [
        "GA_103_g1_c1",
        "GA_103_g1_c2",
        "GA_103_g1_c3",
    ]


def test_legacy_batch_cache_preserves_full_population_indices(ga_gb, tmp_path):
    submitted_ids = []

    def batch_energy(
        GB,
        manipulators,
        structures,
        lineages,
        unique_ids,
        checkpoint=None,
    ):
        submitted_ids.extend(str(uid) for uid in unique_ids)
        results = []
        for manipulator, structure, unique_id in zip(
            manipulators,
            structures,
            unique_ids,
            strict=True,
        ):
            dump_file = tmp_path / f"{unique_id}.data"
            GB.write_lammps(
                str(dump_file),
                structure,
                manipulator.parents[0].box_dims,
            )
            results.append({"energy": 0.0, "final_dump": str(dump_file)})
        return results

    initial_path = tmp_path / "initial.data"
    ga_gb.write_lammps(
        str(initial_path),
        ga_gb.whole_system,
        ga_gb.box_dims,
    )

    def initial_energy(GB, manipulator, atom_positions, unique_id):
        return 0.0, str(initial_path)

    minimizer = GeneticAlgorithmMinimizer(
        ga_gb,
        initial_energy,
        ["translate_right_grain"],
        seed=0,
        population_size=4,
        generations=2,
        keep_top_pct=25,
        intermediate_pct=100,
        reuse_carryover_evaluations=True,
        gb_batch_energy_func=batch_energy,
    )

    minimizer.run_GA(unique_id=104)

    assert [uid for uid in submitted_ids if "_g1_c" in uid] == [
        "GA_104_g1_c1",
        "GA_104_g1_c2",
        "GA_104_g1_c3",
    ]


def test_reuse_carryover_evaluations_requires_boolean(ga_gb):
    with pytest.raises(
        GBMinimizerTypeError,
        match="reuse_carryover_evaluations",
    ):
        GeneticAlgorithmMinimizer(
            ga_gb,
            lambda *_args: (0.0, None),
            ["translate_right_grain"],
            reuse_carryover_evaluations=1,
        )


def test_history_populated_after_run(ga_gb, tmp_path):
    def fake_energy_func(GB, manipulator, atom_positions, unique_id):
        dump_file = tmp_path / f"{unique_id}.data"
        GB.write_lammps(
            str(dump_file),
            atom_positions,
            manipulator.parents[0].box_dims,
        )
        return float(np.mean(atom_positions["x"])), str(dump_file)

    minimizer = GeneticAlgorithmMinimizer(
        ga_gb,
        fake_energy_func,
        ["insert_atoms", "remove_atoms", "translate_right_grain"],
        seed=0,
        population_size=4,
        generations=2,
        keep_top_pct=25,
        intermediate_pct=75,
    )
    checkpoint = tmp_path / "ga2.json"
    minimizer.run_GA(unique_id=2, checkpoint_file=checkpoint)

    assert len(minimizer.history) == minimizer.generations

    for gen_idx, gen_history in enumerate(minimizer.history):
        assert len(gen_history) == minimizer.population_size

        for lineage, _energy in gen_history:
            assert isinstance(lineage, list)
            assert lineage
            assert isinstance(lineage[0], str)
            op = lineage[0]
            assert op in {"slice_and_merge", "carryover", "START"} or op.startswith(
                ("shift", "add", "remove")
            ), f"Unexpected operation label: {op!r}"

        assert [e for _, e in gen_history] == minimizer.GBE_vals[gen_idx + 1]


def test_failed_generation_appends_to_history(ga_gb, tmp_path):
    population_size = 4
    evaluation_index = 0

    def fake_energy_func(GB, manipulator, atom_positions, unique_id):
        nonlocal evaluation_index
        current_index = evaluation_index
        evaluation_index += 1

        # The initial evaluation is index 0. Fail exactly the first generation.
        if 1 <= current_index <= population_size:
            raise RuntimeError("Simulated failure")

        dump_file = tmp_path / f"{unique_id}.data"
        GB.write_lammps(
            str(dump_file),
            atom_positions,
            manipulator.parents[0].box_dims,
        )
        return float(np.mean(atom_positions["x"])), str(dump_file)

    minimizer = GeneticAlgorithmMinimizer(
        ga_gb,
        fake_energy_func,
        ["insert_atoms", "remove_atoms", "translate_right_grain"],
        seed=0,
        population_size=population_size,
        generations=2,
        keep_top_pct=25,
        intermediate_pct=75,
    )
    checkpoint = tmp_path / "ga3.json"
    minimizer.run_GA(unique_id=3, checkpoint_file=checkpoint)

    assert len(minimizer.history) == minimizer.generations

    penalty = 1.0e30
    failed_gen = minimizer.history[0]
    assert len(failed_gen) == minimizer.population_size
    assert all(energy == penalty for _, energy in failed_gen)

    recovered_gen = minimizer.history[1]
    assert len(recovered_gen) == minimizer.population_size
    assert all(energy < penalty for _, energy in recovered_gen)


def test_ga_history_never_exceeds_generations(ga_gb, tmp_path):
    def fake_energy_func(GB, manipulator, atom_positions, unique_id):
        dump_file = tmp_path / f"{unique_id}.data"
        GB.write_lammps(
            str(dump_file),
            atom_positions,
            manipulator.parents[0].box_dims,
        )
        return float(np.mean(atom_positions["x"])), str(dump_file)

    minimizer = GeneticAlgorithmMinimizer(
        ga_gb,
        fake_energy_func,
        ["insert_atoms", "remove_atoms", "translate_right_grain"],
        seed=0,
        population_size=4,
        generations=2,
        keep_top_pct=25,
        intermediate_pct=75,
    )
    checkpoint = tmp_path / "ga_never_exceed.json"
    minimizer.run_GA(unique_id=2, checkpoint_file=checkpoint)
    assert len(minimizer.history) == minimizer.generations

    minimizer.run_GA(unique_id=2, checkpoint_file=checkpoint)
    assert len(minimizer.history) == minimizer.generations


def test_initial_ownership_requires_file_backed_structure(owned_ga):
    gb, seed_path, ownership, _labels = owned_ga

    with pytest.raises(ValueError, match="requires an initial_structure"):
        GeneticAlgorithmMinimizer(
            gb,
            lambda *_args: (0.0, str(seed_path)),
            ["translate_right_grain"],
            initial_ownership=ownership,
        )
    with pytest.raises(TypeError, match="str or Path"):
        GeneticAlgorithmMinimizer(
            gb,
            lambda *_args: (0.0, str(seed_path)),
            ["translate_right_grain"],
            initial_structure=gb,
            initial_ownership=ownership,
        )
    with pytest.raises(TypeError, match="GrainOwnership"):
        GeneticAlgorithmMinimizer(
            gb,
            lambda *_args: (0.0, str(seed_path)),
            ["translate_right_grain"],
            initial_structure=str(seed_path),
            initial_ownership=object(),
        )


def test_variable_cell_requires_explicit_ownership_and_boolean(owned_ga):
    gb, seed_path, ownership, _labels = owned_ga

    with pytest.raises(ValueError, match="requires initial_ownership"):
        GeneticAlgorithmMinimizer(
            gb,
            lambda *_args: (0.0, str(seed_path)),
            ["translate_right_grain"],
            initial_structure=str(seed_path),
            allow_variable_cell=True,
        )
    with pytest.raises(TypeError, match="allow_variable_cell must be a Boolean"):
        GeneticAlgorithmMinimizer(
            gb,
            lambda *_args: (0.0, str(seed_path)),
            ["translate_right_grain"],
            initial_structure=str(seed_path),
            initial_ownership=ownership,
            allow_variable_cell=1,
        )


def test_translation_mutation_uses_current_parent_dimensions():
    class FixedRandom:
        def permutation(self, size):
            assert size == 1
            return np.array([0])

        def uniform(self, _lower, _upper):
            return 0.5

    class ParentStub:
        box_dims = np.asarray(
            [[0.0, 10.0], [-2.0, 18.0], [5.0, 35.0]],
            dtype=float,
        )

    class ManipulatorStub:
        def __init__(self):
            self.parents = [ParentStub()]
            self.translation = None

        def translate_right_grain(self, *, dy, dz):
            self.translation = (dy, dz)
            return np.empty(0)

    class GBStub:
        repeat_factor = (2, 5)
        # Deliberately unrelated to the current parent so the regression fails if the
        # mutation falls back to reference GBMaker dimensions.
        y_dim = 1_000.0
        z_dim = 2_000.0

    manipulator = ManipulatorStub()
    mutator = Mutator(["translate_right_grain"], manipulator)
    mutation, _candidate = mutator.mutate(
        FixedRandom(),
        GBStub(),
        manipulator,
    )

    # Current y/z lengths are 20 and 30 A. At a fixed random fraction of 0.5, repeat
    # factors (2, 5) therefore give dy=5 A and dz=3 A.
    assert manipulator.translation == pytest.approx((5.0, 3.0))
    assert mutation == "shift5.00000000dy3.00000000dz"


def test_initial_owned_manipulator_preserves_labels_and_plane(owned_ga):
    gb, seed_path, ownership, labels = owned_ga
    minimizer = GeneticAlgorithmMinimizer(
        gb,
        lambda *_args: (0.0, str(seed_path)),
        ["translate_right_grain"],
        seed=7,
        initial_structure=seed_path,
        initial_ownership=ownership,
        population_size=2,
        generations=1,
    )
    parent = minimizer.manipulator.parents[0]
    assert np.array_equal(parent.grain_labels, labels)
    assert parent.gb_plane_x == pytest.approx(gb.gb_plane_x)
    assert len(parent.left_grain) == len(gb.left_grain)
    assert len(parent.right_grain) == len(gb.right_grain)


def test_owned_evaluator_rejects_inadmissible_candidate_before_callback(owned_ga):
    gb, seed_path, ownership, _labels = owned_ga
    callback_calls = 0

    def unexpected_energy(GB, manipulator, atom_positions, unique_id):
        nonlocal callback_calls
        callback_calls += 1
        raise AssertionError("inadmissible candidate reached the evaluator callback")

    minimizer = GeneticAlgorithmMinimizer(
        gb,
        unexpected_energy,
        ["translate_right_grain"],
        seed=37,
        initial_structure=str(seed_path),
        initial_ownership=ownership,
        population_size=2,
        generations=1,
    )
    atoms = np.array(minimizer.manipulator.parents[0].whole_system, copy=True)
    atoms[0]["name"] = "Cu"

    record = minimizer._owned_evaluator.evaluate_candidate(
        minimizer.manipulator,
        atoms,
        "inadmissible",
        0,
    )

    assert callback_calls == 0
    assert not record.success
    assert record.energy == pytest.approx(1.0e30)
    assert "composition is inadmissible" in record.failure_reason


def test_owned_ga_variable_cell_reload_becomes_next_parent_geometry(owned_ga, tmp_path):
    gb, seed_path, ownership, labels = owned_ga
    initial_box = np.asarray(gb.box_dims, dtype=float)
    relaxed_box = initial_box.copy()
    relaxed_box[0] += np.asarray([-0.5, 1.0])
    relaxed_box[1] += np.asarray([-0.25, 0.75])
    relaxed_box[2] += np.asarray([-0.125, 0.625])
    submitted_boxes = []
    evaluation_index = 0

    def fake_energy(GB, manipulator, atom_positions, unique_id):
        nonlocal evaluation_index
        source_box = np.asarray(manipulator.parents[0].box_dims, dtype=float)
        submitted_boxes.append(source_box.copy())
        target_box = relaxed_box if evaluation_index == 0 else source_box
        returned_atoms = np.array(atom_positions, copy=True)
        if evaluation_index == 0:
            for axis_name, axis_index in zip("xyz", range(3), strict=True):
                source_lo, source_hi = source_box[axis_index]
                target_lo, target_hi = target_box[axis_index]
                reduced = (
                    returned_atoms[axis_name] - source_lo
                ) / (source_hi - source_lo)
                returned_atoms[axis_name] = (
                    target_lo + reduced * (target_hi - target_lo)
                )
        output = tmp_path / f"{unique_id}.data"
        _write_owned_evaluator_output(output, returned_atoms, target_box)
        evaluation_index += 1
        return 1.0, str(output)

    minimizer = GeneticAlgorithmMinimizer(
        gb,
        fake_energy,
        ["translate_right_grain"],
        seed=37,
        initial_structure=seed_path,
        initial_ownership=ownership,
        allow_variable_cell=True,
        population_size=2,
        generations=1,
        keep_top_pct=100,
        intermediate_pct=100,
    )
    minimizer.run_GA(unique_id=73)

    np.testing.assert_allclose(submitted_boxes[0], initial_box)
    assert len(submitted_boxes) == 3
    for box in submitted_boxes[1:]:
        np.testing.assert_allclose(box, relaxed_box)

    best = minimizer.best_evaluation
    assert best.success
    assert best.manipulator is not None
    parent = best.manipulator.parents[0]
    np.testing.assert_allclose(parent.box_dims, relaxed_box)
    np.testing.assert_array_equal(parent.grain_labels, labels)

    old_lo, old_hi = initial_box[0]
    new_lo, new_hi = relaxed_box[0]
    reduced_plane = (ownership.gb_plane_x - old_lo) / (old_hi - old_lo)
    expected_plane = new_lo + reduced_plane * (new_hi - new_lo)
    assert parent.gb_plane_x == pytest.approx(expected_plane)


def test_owned_ga_reordered_reload_preserves_crossing_atom_label(
    owned_ga,
    tmp_path,
):
    gb, seed_path, ownership, labels = owned_ga
    crossing_row = int(np.flatnonzero(labels == LEFT_GRAIN_LABEL)[0])

    def fake_energy(GB, manipulator, atom_positions, unique_id):
        output = tmp_path / f"{unique_id}.data"
        _write_owned_evaluator_output(
            output,
            atom_positions,
            manipulator.parents[0].box_dims,
            move_row=crossing_row,
            plane=GB.gb_plane_x,
        )
        return 1.0, str(output)

    minimizer = GeneticAlgorithmMinimizer(
        gb,
        fake_energy,
        ["translate_right_grain"],
        seed=17,
        initial_structure=str(seed_path),
        initial_ownership=ownership,
        population_size=2,
        generations=1,
        keep_top_pct=100,
        intermediate_pct=100,
    )
    minimizer.run_GA(unique_id=41)

    record = minimizer.last_generation_evaluations[0]
    assert record.success
    assert record.manipulator is not None

    parent = record.manipulator.parents[0]
    assert parent.grain_labels[crossing_row] == LEFT_GRAIN_LABEL
    assert parent.whole_system[crossing_row]["x"] > gb.gb_plane_x
    assert np.array_equal(parent.grain_labels, labels)
    assert parent.gb_plane_x == pytest.approx(gb.gb_plane_x)


@pytest.mark.parametrize(
    ("mutation", "count_delta"),
    [
        pytest.param("remove_atoms", -1, id="removal"),
        pytest.param("insert_atoms", 1, id="insertion"),
    ],
)
def test_owned_count_changing_mutation_uses_fresh_mapping(
    owned_ga,
    tmp_path,
    mutation,
    count_delta,
):
    gb, seed_path, ownership, _labels = owned_ga
    initial_count = len(gb.whole_system)

    def fake_energy(GB, manipulator, atom_positions, unique_id):
        output = tmp_path / f"{unique_id}.data"
        _write_owned_evaluator_output(
            output,
            atom_positions,
            manipulator.parents[0].box_dims,
        )
        return float(len(atom_positions)), str(output)

    minimizer = GeneticAlgorithmMinimizer(
        gb,
        fake_energy,
        [mutation],
        seed=23,
        initial_structure=seed_path,
        initial_ownership=ownership,
        population_size=2,
        generations=1,
        keep_top_pct=100,
        intermediate_pct=100,
    )
    minimizer.run_GA(unique_id=52)

    mutated = minimizer.last_generation_evaluations[1]
    expected_count = initial_count + count_delta

    assert mutated.success
    assert mutated.mapping is not None
    assert mutated.manipulator is not None
    np.testing.assert_array_equal(
        mutated.mapping.atom_ids,
        np.arange(1, expected_count + 1),
    )
    assert len(mutated.mapping.labels) == expected_count

    parent = mutated.manipulator.parents[0]
    assert len(parent.whole_system) == expected_count
    assert len(parent.grain_labels) == expected_count


def test_failed_owned_candidate_is_not_selected_as_parent(owned_ga, tmp_path):
    gb, seed_path, ownership, _labels = owned_ga
    batch_index = 0
    generation_zero_paths = []

    def scalar_energy(GB, manipulator, atom_positions, unique_id):
        output = tmp_path / f"{unique_id}.data"
        _write_owned_evaluator_output(
            output,
            atom_positions,
            manipulator.parents[0].box_dims,
        )
        return 5.0, str(output)

    def batch_energy(
        GB, manipulators, structures, lineages, unique_ids, checkpoint=None
    ):
        nonlocal batch_index
        results = []
        paths = []
        for index, (manipulator, atoms, candidate_id) in enumerate(
            zip(manipulators, structures, unique_ids, strict=True)
        ):
            output = tmp_path / f"{candidate_id}.data"
            paths.append(str(output))
            _write_owned_evaluator_output(
                output,
                atoms,
                manipulator.parents[0].box_dims,
                change_species_row=(0 if batch_index == 0 and index == 0 else None),
            )
            results.append(
                {"energy": float(index + 1), "final_dump": str(output)}
            )
        if batch_index == 0:
            generation_zero_paths.extend(paths)
        batch_index += 1
        return results

    minimizer = GeneticAlgorithmMinimizer(
        gb,
        scalar_energy,
        ["translate_right_grain"],
        gb_batch_energy_func=batch_energy,
        seed=3,
        initial_structure=seed_path,
        initial_ownership=ownership,
        population_size=2,
        generations=2,
        intermediate_pct=100,
    )
    minimizer.run_GA(unique_id=7)

    failed_path, valid_path = generation_zero_paths
    second_generation_lineages = [lineage for lineage, _ in minimizer.history[1]]
    assert all(failed_path not in lineage for lineage in second_generation_lineages)
    assert all(valid_path in lineage for lineage in second_generation_lineages)


def test_owned_scalar_evaluator_species_swap_is_failed(owned_ga, tmp_path):
    gb, seed_path, ownership, _labels = owned_ga
    evaluation_index = 0

    def fake_energy(GB, manipulator, atom_positions, unique_id):
        nonlocal evaluation_index
        output = tmp_path / f"{unique_id}.data"
        _write_owned_evaluator_output(
            output,
            atom_positions,
            manipulator.parents[0].box_dims,
            change_species_row=0 if evaluation_index == 1 else None,
        )
        evaluation_index += 1
        return 0.0, str(output)

    minimizer = GeneticAlgorithmMinimizer(
        gb,
        fake_energy,
        ["translate_right_grain"],
        seed=3,
        initial_structure=str(seed_path),
        initial_ownership=ownership,
        population_size=2,
        generations=1,
        keep_top_pct=100,
        intermediate_pct=100,
    )
    minimizer.run_GA(unique_id=8)

    failed, valid = minimizer.last_generation_evaluations
    assert not failed.success
    assert failed.energy == pytest.approx(1.0e30)
    assert failed.manipulator is None
    assert "changed species" in failed.failure_reason
    assert valid.success


def test_owned_batch_evaluator_keeps_failure_alignment(owned_ga, tmp_path):
    gb, seed_path, ownership, _labels = owned_ga

    def scalar_energy(GB, manipulator, atom_positions, unique_id):
        output = tmp_path / f"{unique_id}.data"
        _write_owned_evaluator_output(
            output,
            atom_positions,
            manipulator.parents[0].box_dims,
        )
        return 5.0, str(output)

    def batch_energy(
        GB, manipulators, structures, lineages, unique_ids, checkpoint=None
    ):
        results = []
        for index, (manipulator, atoms, candidate_id) in enumerate(
            zip(manipulators, structures, unique_ids, strict=True)
        ):
            output = tmp_path / f"{candidate_id}.data"
            _write_owned_evaluator_output(
                output,
                atoms,
                manipulator.parents[0].box_dims,
                change_species_row=0 if index == 0 else None,
            )
            results.append(
                {"energy": float(index + 1), "final_dump": str(output)}
            )
        return results

    minimizer = GeneticAlgorithmMinimizer(
        gb,
        scalar_energy,
        ["translate_right_grain"],
        gb_batch_energy_func=batch_energy,
        seed=13,
        initial_structure=str(seed_path),
        initial_ownership=ownership,
        population_size=2,
        generations=1,
        keep_top_pct=100,
        intermediate_pct=100,
    )
    best_energy, _best_path = minimizer.run_GA(unique_id=19)
    records = minimizer.last_generation_evaluations

    assert [record.input_index for record in records] == [0, 1]
    assert not records[0].success
    assert records[0].energy == pytest.approx(1.0e30)
    assert records[0].manipulator is None
    assert records[1].success
    assert records[1].energy == pytest.approx(2.0)
    assert best_energy == pytest.approx(2.0)


def test_owned_ga_carryover_and_crossover_preserve_ownership(owned_ga, tmp_path):
    gb, seed_path, ownership, _labels = owned_ga

    def fake_energy(GB, manipulator, atom_positions, unique_id):
        output = tmp_path / f"{unique_id}.data"
        _write_owned_evaluator_output(
            output,
            atom_positions,
            manipulator.parents[0].box_dims,
        )
        return float(np.mean(atom_positions["x"])), str(output)

    minimizer = GeneticAlgorithmMinimizer(
        gb,
        fake_energy,
        ["translate_right_grain"],
        seed=31,
        initial_structure=seed_path,
        initial_ownership=ownership,
        population_size=4,
        generations=2,
        keep_top_pct=25,
        intermediate_pct=100,
    )
    minimizer.run_GA(unique_id=61)

    second_generation_ops = [
        lineage[0] for lineage, _energy in minimizer.history[1]
    ]
    assert "carryover" in second_generation_ops
    assert "slice_and_merge" in second_generation_ops

    for record in minimizer.last_generation_evaluations:
        assert record.success
        assert record.mapping is not None
        assert record.manipulator is not None

        parent = record.manipulator.parents[0]
        assert parent.grain_labels is not None
        assert len(parent.grain_labels) == len(parent.whole_system)
        assert parent.gb_plane_x == pytest.approx(gb.gb_plane_x)
        assert parent.inplane_periodic == ownership.inplane_periodic
        assert parent.normal_topology is ownership.normal_topology


def test_owned_ga_uses_bounded_mutation_fallback_for_inadmissible_crossover(
    monkeypatch,
    owned_ga,
    tmp_path,
):
    gb, seed_path, ownership, _labels = owned_ga
    attempts = 0

    def reject_crossover(self, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise CompositionAwareCrossoverError("no admissible interval")

    monkeypatch.setattr(GBManipulator, "slice_and_merge", reject_crossover)

    def fake_energy(GB, manipulator, atom_positions, unique_id):
        output = tmp_path / f"{unique_id}.data"
        _write_owned_evaluator_output(
            output,
            atom_positions,
            manipulator.parents[0].box_dims,
        )
        return 1.0, str(output)

    minimizer = GeneticAlgorithmMinimizer(
        gb,
        fake_energy,
        ["translate_right_grain"],
        seed=43,
        initial_structure=seed_path,
        initial_ownership=ownership,
        population_size=4,
        generations=2,
        keep_top_pct=25,
        intermediate_pct=100,
        crossover_attempts=3,
    )
    minimizer.run_GA(unique_id=62)

    assert attempts == 6
    second_generation_ops = [lineage[0] for lineage, _energy in minimizer.history[1]]
    assert "carryover" in second_generation_ops
    assert sum(
        op.startswith("crossover_fallback_shift") for op in second_generation_ops
    ) == 1


def _owned_checkpoint_energy(tmp_path, *, relaxed_initial_box=None):
    """Return a deterministic owned evaluator writing ID-reordered artifacts."""

    def energy(GB, manipulator, atom_positions, unique_id):
        source_box = np.asarray(manipulator.parents[0].box_dims, dtype=float)
        target_box = source_box
        returned_atoms = np.array(atom_positions, copy=True)
        if relaxed_initial_box is not None and str(unique_id).startswith("GA_initial"):
            target_box = np.asarray(relaxed_initial_box, dtype=float)
            for axis_name, axis_index in zip("xyz", range(3), strict=True):
                source_lo, source_hi = source_box[axis_index]
                target_lo, target_hi = target_box[axis_index]
                reduced = (
                    returned_atoms[axis_name] - source_lo
                ) / (source_hi - source_lo)
                returned_atoms[axis_name] = target_lo + reduced * (
                    target_hi - target_lo
                )
        output = tmp_path / f"{unique_id}.data"
        _write_owned_evaluator_output(output, returned_atoms, target_box)
        value = float(
            np.mean(returned_atoms["x"])
            + 0.01 * np.mean(returned_atoms["y"])
            + 0.001 * len(returned_atoms)
        )
        return value, str(output)

    return energy


def _make_owned_checkpoint_minimizer(
    owned_ga,
    energy,
    *,
    generations,
    seed=101,
    choices=None,
    population_size=4,
    keep_top_pct=25,
    slice_and_merge_pct=50.0,
    reuse_carryover_evaluations=False,
    allow_variable_cell=False,
    batch_energy=None,
    retention_policy=None,
):
    gb, seed_path, ownership, _labels = owned_ga
    return GeneticAlgorithmMinimizer(
        gb,
        energy,
        ["translate_right_grain"] if choices is None else choices,
        seed=seed,
        initial_structure=seed_path,
        initial_ownership=ownership,
        allow_variable_cell=allow_variable_cell,
        population_size=population_size,
        generations=generations,
        keep_top_pct=keep_top_pct,
        intermediate_pct=100,
        slice_and_merge_pct=slice_and_merge_pct,
        reuse_carryover_evaluations=reuse_carryover_evaluations,
        gb_batch_energy_func=batch_energy,
        retention_policy=retention_policy,
    )


@pytest.mark.parametrize(
    ("slice_and_merge_pct", "expected_crossover_slots"),
    [
        pytest.param(0.0, 0, id="mutation-only"),
        pytest.param(100.0, 3, id="crossover-only"),
    ],
)
def test_slice_and_merge_pct_controls_owned_offspring_mix(
    owned_ga,
    tmp_path,
    slice_and_merge_pct,
    expected_crossover_slots,
):
    minimizer = _make_owned_checkpoint_minimizer(
        owned_ga,
        _owned_checkpoint_energy(tmp_path),
        generations=2,
        population_size=4,
        keep_top_pct=25,
        slice_and_merge_pct=slice_and_merge_pct,
    )

    minimizer.run_GA(unique_id=200)

    operations = [lineage[0] for lineage, _energy in minimizer.history[1]]
    crossover_slots = sum(
        operation == "slice_and_merge"
        or operation.startswith("crossover_fallback_")
        for operation in operations
    )
    assert operations.count("carryover") == 1
    assert crossover_slots == expected_crossover_slots
    assert len(operations) == minimizer.population_size


def test_owned_carryover_cache_survives_checkpoint_extension(owned_ga, tmp_path):
    checkpoint = tmp_path / "owned-cached.json"
    energy = _owned_checkpoint_energy(tmp_path)
    partial = _make_owned_checkpoint_minimizer(
        owned_ga,
        energy,
        generations=1,
        population_size=4,
        keep_top_pct=25,
        reuse_carryover_evaluations=True,
    )
    partial.run_GA(unique_id=213, checkpoint_file=checkpoint)

    resumed_ids = []

    def tracking_energy(GB, manipulator, atom_positions, unique_id):
        resumed_ids.append(str(unique_id))
        return energy(GB, manipulator, atom_positions, unique_id)

    resumed = _make_owned_checkpoint_minimizer(
        owned_ga,
        tracking_energy,
        generations=2,
        population_size=4,
        keep_top_pct=25,
        reuse_carryover_evaluations=True,
    )
    resumed.run_GA(unique_id=213, checkpoint_file=checkpoint)

    assert resumed_ids == [
        "GA_213_g1_c1",
        "GA_213_g1_c2",
        "GA_213_g1_c3",
    ]
    assert resumed.history[1][0][0][0] == "carryover"
    assert resumed.history[1][0][1] == resumed.history[0][0][1]


def test_owned_batch_cache_preserves_full_population_indices(owned_ga, tmp_path):
    scalar_energy = _owned_checkpoint_energy(tmp_path)
    submitted_ids = []

    def batch_energy(
        GB,
        manipulators,
        structures,
        lineages,
        unique_ids,
        checkpoint=None,
    ):
        submitted_ids.extend(str(uid) for uid in unique_ids)
        results = []
        for manipulator, structure, unique_id in zip(
            manipulators,
            structures,
            unique_ids,
            strict=True,
        ):
            energy, output = scalar_energy(
                GB,
                manipulator,
                structure,
                unique_id,
            )
            results.append({"energy": energy, "final_dump": output})
        return results

    minimizer = _make_owned_checkpoint_minimizer(
        owned_ga,
        scalar_energy,
        generations=2,
        population_size=4,
        keep_top_pct=25,
        reuse_carryover_evaluations=True,
        batch_energy=batch_energy,
    )

    minimizer.run_GA(unique_id=214)

    assert [uid for uid in submitted_ids if "_g1_c" in uid] == [
        "GA_214_g1_c1",
        "GA_214_g1_c2",
        "GA_214_g1_c3",
    ]


def test_owned_ga_checkpoint_json_contains_reconstruction_state(owned_ga, tmp_path):
    checkpoint = tmp_path / "owned.json"
    energy = _owned_checkpoint_energy(tmp_path)
    minimizer = _make_owned_checkpoint_minimizer(
        owned_ga,
        energy,
        generations=1,
    )

    minimizer.run_GA(unique_id=201, checkpoint_file=checkpoint)

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert state["state"]["ga_mode"] == "explicit_ownership"
    assert state["state"]["owned_checkpoint_version"] == 3
    assert state["run_params"]["crossover_surface"] == "periodic_wave"
    assert state["run_params"]["crossover_max_tilt_degrees"] == pytest.approx(5.0)
    assert state["run_params"]["crossover_attempts"] == 8
    assert state["run_params"]["composition_policy"] == [["Ni", 1]]
    assert state["progress_index"] == 0
    assert len(state["state"]["population_candidates"]) == 4
    candidate = state["state"]["population_candidates"][0]
    assert Path(candidate["structure_path"]).is_file()
    mapping = candidate["mapping"]
    assert mapping["labels"]
    assert mapping["atom_ids"] == list(range(1, len(mapping["labels"]) + 1))
    assert np.asarray(mapping["box_dims"]).shape == (3, 2)
    assert isinstance(mapping["gb_plane_x"], float)
    assert mapping["left_grain_x_bounds"]
    assert mapping["right_grain_x_bounds"]
    assert state["state"]["best_evaluation"]["mapping"] is not None
    assert len(state["state"]["last_generation_evaluations"]) == 4


def test_owned_ga_resume_matches_continuous_variable_cell_run(owned_ga, tmp_path):
    gb, _seed_path, ownership, labels = owned_ga
    initial_box = np.asarray(gb.box_dims, dtype=float)
    relaxed_box = initial_box.copy()
    relaxed_box[0] += (-0.4, 0.8)
    relaxed_box[1] += (-0.2, 0.3)
    relaxed_box[2] += (-0.1, 0.2)
    energy = _owned_checkpoint_energy(
        tmp_path,
        relaxed_initial_box=relaxed_box,
    )

    continuous = _make_owned_checkpoint_minimizer(
        owned_ga,
        energy,
        generations=3,
        allow_variable_cell=True,
    )
    continuous.run_GA(unique_id=202, checkpoint_file=tmp_path / "continuous.json")

    checkpoint = tmp_path / "resumed.json"
    partial = _make_owned_checkpoint_minimizer(
        owned_ga,
        energy,
        generations=1,
        allow_variable_cell=True,
    )
    partial.run_GA(unique_id=203, checkpoint_file=checkpoint)
    resumed = _make_owned_checkpoint_minimizer(
        owned_ga,
        energy,
        generations=3,
        allow_variable_cell=True,
    )
    resumed.run_GA(unique_id=203, checkpoint_file=checkpoint)

    assert len(resumed.GBE_vals) == 4
    assert len(resumed.history) == 3
    for continuous_generation, resumed_generation in zip(
        continuous.GBE_vals,
        resumed.GBE_vals,
        strict=True,
    ):
        np.testing.assert_allclose(continuous_generation, resumed_generation)

    old_lo, old_hi = initial_box[0]
    new_lo, new_hi = relaxed_box[0]
    expected_plane = new_lo + (
        (ownership.gb_plane_x - old_lo) / (old_hi - old_lo)
    ) * (new_hi - new_lo)
    for record in resumed.last_generation_evaluations:
        assert record.success
        assert record.manipulator is not None
        parent = record.manipulator.parents[0]
        np.testing.assert_allclose(parent.box_dims, relaxed_box)
        np.testing.assert_array_equal(parent.grain_labels, labels)
        assert parent.gb_plane_x == pytest.approx(expected_plane)


def test_failed_owned_evaluation_stays_excluded_after_resume(owned_ga, tmp_path):
    def energy(GB, manipulator, atom_positions, unique_id):
        output = tmp_path / f"{unique_id}.data"
        _write_owned_evaluator_output(
            output,
            atom_positions,
            manipulator.parents[0].box_dims,
            change_species_row=0 if str(unique_id).endswith("_g0_c0") else None,
        )
        return 0.0, str(output)

    checkpoint = tmp_path / "failed.json"
    partial = _make_owned_checkpoint_minimizer(
        owned_ga,
        energy,
        generations=1,
        population_size=2,
        keep_top_pct=50,
    )
    partial.run_GA(unique_id=204, checkpoint_file=checkpoint)
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    failed_state = saved["state"]["last_generation_evaluations"][0]
    assert failed_state["success"] is False
    assert "changed species" in failed_state["failure_reason"]

    resumed = _make_owned_checkpoint_minimizer(
        owned_ga,
        energy,
        generations=2,
        population_size=2,
        keep_top_pct=50,
    )
    resumed.run_GA(unique_id=204, checkpoint_file=checkpoint)
    failed_path = str((tmp_path / "GA_204_g0_c0.data").resolve())
    assert all(
        failed_path not in lineage
        for lineage, _energy in resumed.history[1]
    )


@pytest.mark.parametrize("mutation", ["insert_atoms", "remove_atoms"])
def test_owned_count_change_uses_fresh_mapping_after_resume(
    owned_ga,
    tmp_path,
    mutation,
):
    gb, _seed_path, _ownership, _labels = owned_ga
    energy = _owned_checkpoint_energy(tmp_path)
    checkpoint = tmp_path / f"{mutation}.json"
    partial = _make_owned_checkpoint_minimizer(
        owned_ga,
        energy,
        generations=1,
        choices=[mutation],
        population_size=2,
        keep_top_pct=0,
    )
    partial.run_GA(unique_id=205, checkpoint_file=checkpoint)
    resumed = _make_owned_checkpoint_minimizer(
        owned_ga,
        energy,
        generations=2,
        choices=[mutation],
        population_size=2,
        keep_top_pct=0,
    )
    resumed.run_GA(unique_id=205, checkpoint_file=checkpoint)

    counts = []
    for record in resumed.last_generation_evaluations:
        assert record.success
        assert record.mapping is not None
        count = record.mapping.expected_count
        counts.append(count)
        np.testing.assert_array_equal(
            record.mapping.atom_ids,
            np.arange(1, count + 1),
        )
        assert len(record.mapping.labels) == count
    assert any(count != len(gb.whole_system) for count in counts)


def test_owned_breeding_operations_preserve_ownership_after_resume(
    owned_ga,
    tmp_path,
):
    energy = _owned_checkpoint_energy(tmp_path)
    checkpoint = tmp_path / "breeding.json"
    partial = _make_owned_checkpoint_minimizer(
        owned_ga,
        energy,
        generations=1,
        population_size=4,
        keep_top_pct=25,
    )
    partial.run_GA(unique_id=206, checkpoint_file=checkpoint)
    resumed = _make_owned_checkpoint_minimizer(
        owned_ga,
        energy,
        generations=2,
        population_size=4,
        keep_top_pct=25,
    )
    resumed.run_GA(unique_id=206, checkpoint_file=checkpoint)

    operations = [lineage[0] for lineage, _energy in resumed.history[1]]
    assert "carryover" in operations
    assert "slice_and_merge" in operations
    assert any(operation.startswith("shift") for operation in operations)
    for record in resumed.last_generation_evaluations:
        assert record.success
        assert record.manipulator is not None
        parent = record.manipulator.parents[0]
        assert parent.grain_labels is not None
        assert len(parent.grain_labels) == len(parent.whole_system)


def test_owned_resume_fails_on_missing_population_artifact(owned_ga, tmp_path):
    energy = _owned_checkpoint_energy(tmp_path)
    checkpoint = tmp_path / "missing.json"
    partial = _make_owned_checkpoint_minimizer(
        owned_ga,
        energy,
        generations=1,
    )
    partial.run_GA(unique_id=207, checkpoint_file=checkpoint)
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    Path(state["state"]["population_candidates"][0]["structure_path"]).unlink()

    resumed = _make_owned_checkpoint_minimizer(
        owned_ga,
        energy,
        generations=2,
    )
    with pytest.raises(GBMinimizerError, match="owned population path"):
        resumed.run_GA(unique_id=207, checkpoint_file=checkpoint)


def test_owned_resume_rejects_invalid_checkpoint_state(owned_ga, tmp_path):
    energy = _owned_checkpoint_energy(tmp_path)
    checkpoint = tmp_path / "invalid.json"
    partial = _make_owned_checkpoint_minimizer(
        owned_ga,
        energy,
        generations=1,
    )
    partial.run_GA(unique_id=208, checkpoint_file=checkpoint)
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    state["state"]["owned_checkpoint_version"] = 999
    checkpoint.write_text(json.dumps(state), encoding="utf-8")

    resumed = _make_owned_checkpoint_minimizer(
        owned_ga,
        energy,
        generations=2,
    )
    with pytest.raises(GBMinimizerError, match="supported explicit-ownership"):
        resumed.run_GA(unique_id=208, checkpoint_file=checkpoint)


def test_owned_checkpointing_is_optional(owned_ga, tmp_path):
    energy = _owned_checkpoint_energy(tmp_path)
    checkpoint = tmp_path / "not-created.json"
    minimizer = _make_owned_checkpoint_minimizer(
        owned_ga,
        energy,
        generations=1,
    )

    minimizer.run_GA(unique_id=209)

    assert not checkpoint.exists()
    assert not list(tmp_path.glob("*.owned.pending"))


def test_owned_scalar_intra_generation_resume_skips_completed_candidates(
    owned_ga,
    tmp_path,
):
    from GBOpt.Checkpoint import CandidateCheckpoint

    energy = _owned_checkpoint_energy(tmp_path)
    checkpoint = tmp_path / "intra.json"
    original_record = CandidateCheckpoint.record
    record_count = 0

    def crashing_record(self_checkpoint, unique_id, value, dump, **kwargs):
        nonlocal record_count
        original_record(
            self_checkpoint,
            unique_id,
            value,
            dump,
            **kwargs,
        )
        record_count += 1
        if record_count == 2:
            raise RuntimeError("simulated owned mid-generation interruption")

    partial = _make_owned_checkpoint_minimizer(
        owned_ga,
        energy,
        generations=1,
        population_size=4,
    )
    with patch.object(CandidateCheckpoint, "record", crashing_record):
        with pytest.raises(RuntimeError, match="mid-generation"):
            partial.run_GA(unique_id=210, checkpoint_file=checkpoint)

    resumed_calls = 0

    def tracking_energy(GB, manipulator, atom_positions, unique_id):
        nonlocal resumed_calls
        if "_g0_c" in str(unique_id):
            resumed_calls += 1
        return energy(GB, manipulator, atom_positions, unique_id)

    resumed = _make_owned_checkpoint_minimizer(
        owned_ga,
        tracking_energy,
        generations=1,
        population_size=4,
    )
    resumed.run_GA(unique_id=210, checkpoint_file=checkpoint)

    assert resumed_calls == 2
    assert len(resumed.GBE_vals) == 2
    assert len(resumed.history) == 1


def test_owned_batch_intra_generation_resume_uses_recorded_raw_result(
    owned_ga,
    tmp_path,
):
    class SimulatedBatchInterruption(BaseException):
        pass

    scalar_energy = _owned_checkpoint_energy(tmp_path)
    checkpoint = tmp_path / "batch-intra.json"
    interrupted = False

    def interrupting_batch(
        GB,
        manipulators,
        structures,
        lineages,
        unique_ids,
        checkpoint=None,
    ):
        nonlocal interrupted
        results = []
        for manipulator, atoms, candidate_id in zip(
            manipulators,
            structures,
            unique_ids,
            strict=True,
        ):
            energy, output = scalar_energy(
                GB,
                manipulator,
                atoms,
                candidate_id,
            )
            result = {"energy": energy, "final_dump": output}
            results.append(result)
            if not interrupted:
                checkpoint.record(candidate_id, energy, output)
                interrupted = True
                raise SimulatedBatchInterruption()
        return results

    partial = _make_owned_checkpoint_minimizer(
        owned_ga,
        scalar_energy,
        generations=1,
        population_size=2,
        keep_top_pct=50,
        batch_energy=interrupting_batch,
    )
    with pytest.raises(SimulatedBatchInterruption):
        partial.run_GA(unique_id=212, checkpoint_file=checkpoint)

    resumed_uids = []

    def resumed_batch(
        GB,
        manipulators,
        structures,
        lineages,
        unique_ids,
        checkpoint=None,
    ):
        resumed_uids.extend(unique_ids)
        results = []
        for manipulator, atoms, candidate_id in zip(
            manipulators,
            structures,
            unique_ids,
            strict=True,
        ):
            energy, output = scalar_energy(
                GB,
                manipulator,
                atoms,
                candidate_id,
            )
            results.append({"energy": energy, "final_dump": output})
        return results

    resumed = _make_owned_checkpoint_minimizer(
        owned_ga,
        scalar_energy,
        generations=1,
        population_size=2,
        keep_top_pct=50,
        batch_energy=resumed_batch,
    )
    resumed.run_GA(unique_id=212, checkpoint_file=checkpoint)

    assert resumed_uids == ["GA_212_g0_c1"]
    assert all(record.success for record in resumed.last_generation_evaluations)


def test_owned_checkpoint_pickle_resume(owned_ga, tmp_path):
    energy = _owned_checkpoint_energy(tmp_path)
    checkpoint = tmp_path / "owned.pkl"
    partial = _make_owned_checkpoint_minimizer(
        owned_ga,
        energy,
        generations=1,
        population_size=2,
        keep_top_pct=50,
    )
    partial.run_GA(
        unique_id=211,
        checkpoint_file=checkpoint,
        checkpoint_format="pickle",
    )
    resumed = _make_owned_checkpoint_minimizer(
        owned_ga,
        energy,
        generations=2,
        population_size=2,
        keep_top_pct=50,
    )
    resumed.run_GA(
        unique_id=211,
        checkpoint_file=checkpoint,
        checkpoint_format="pickle",
    )

    assert len(resumed.GBE_vals) == 3
    assert len(resumed.history) == 2


def test_mutator_retries_after_infeasible_mutation():
    """An infeasible mutation should fall through to another configured choice."""

    class FakeRandom:
        def permutation(self, size):
            assert size == 2
            return np.array([0, 1])

        def uniform(self, low, high):
            assert low == 0
            assert high == 1
            return 0.5

    class Parent:
        box_dims = np.array(
            [
                [0.0, 10.0],
                [0.0, 20.0],
                [0.0, 30.0],
            ]
        )

    class Manipulator:
        parents = [Parent()]

        def __init__(self):
            self.calls = []

        def remove_atoms(self, *, num_to_remove):
            self.calls.append("remove_atoms")
            assert num_to_remove == 1
            raise GBManipulatorValueError(
                "Not enough neighbor atoms of type 2 to remove."
            )

        def translate_right_grain(self, *, dy, dz):
            self.calls.append("translate_right_grain")
            np.testing.assert_allclose(dy, 10.0)
            np.testing.assert_allclose(dz, 15.0)
            return "translated-system"

    class GB:
        repeat_factor = (1, 1)

    manipulator = Manipulator()
    mutator = Mutator(
        ["remove_atoms", "translate_right_grain"],
        manipulator,
    )

    mutation, new_system = mutator.mutate(
        local_random=FakeRandom(),
        GB=GB(),
        manipulator=manipulator,
    )

    assert manipulator.calls == [
        "remove_atoms",
        "translate_right_grain",
    ]
    assert mutation == "shift10.00000000dy15.00000000dz"
    assert new_system == "translated-system"


def test_mutator_does_not_hide_unexpected_mutation_error():
    """Only expected mutation-infeasibility errors should trigger a retry."""

    class FakeRandom:
        def permutation(self, size):
            assert size == 2
            return np.array([0, 1])

    class Manipulator:
        def __init__(self):
            self.calls = []

        def remove_atoms(self, *, num_to_remove):
            self.calls.append("remove_atoms")
            assert num_to_remove == 1
            raise RuntimeError("unexpected mutation failure")

        def translate_right_grain(self, *, dy, dz):
            self.calls.append("translate_right_grain")
            return "translated-system"

    class GB:
        repeat_factor = (1, 1)

    manipulator = Manipulator()
    mutator = Mutator(
        ["remove_atoms", "translate_right_grain"],
        manipulator,
    )

    with pytest.raises(RuntimeError, match="unexpected mutation failure"):
        mutator.mutate(
            local_random=FakeRandom(),
            GB=GB(),
            manipulator=manipulator,
        )

    assert manipulator.calls == ["remove_atoms"]


def test_mutator_fails_when_all_mutations_are_infeasible():
    """The optimizer should fail clearly when every configured mutation is infeasible."""

    class FakeRandom:
        def permutation(self, size):
            assert size == 2
            return np.array([0, 1])

        def uniform(self, low, high):
            assert low == 0
            assert high == 1
            return 0.5

    class Parent:
        box_dims = np.array(
            [
                [0.0, 10.0],
                [0.0, 20.0],
                [0.0, 30.0],
            ]
        )

    class Manipulator:
        parents = [Parent()]

        def __init__(self):
            self.calls = []

        def remove_atoms(self, *, num_to_remove):
            self.calls.append("remove_atoms")
            assert num_to_remove == 1
            raise GBManipulatorValueError("removal is infeasible")

        def translate_right_grain(self, *, dy, dz):
            self.calls.append("translate_right_grain")
            raise GBManipulatorValueError("translation is infeasible")

    class GB:
        repeat_factor = (1, 1)

    manipulator = Manipulator()
    mutator = Mutator(
        ["remove_atoms", "translate_right_grain"],
        manipulator,
    )

    with pytest.raises(
        GBMinimizerError,
        match="No configured mutation could produce a valid candidate",
    ) as exc_info:
        mutator.mutate(
            local_random=FakeRandom(),
            GB=GB(),
            manipulator=manipulator,
        )

    message = str(exc_info.value)
    assert "remove_atoms" in message
    assert "removal is infeasible" in message
    assert "translate_right_grain" in message
    assert "translation is infeasible" in message
    assert manipulator.calls == [
        "remove_atoms",
        "translate_right_grain",
    ]
    assert isinstance(exc_info.value.__cause__, GBManipulatorValueError)


if __name__ == "__main__":
    unittest.main()


def _objective_retention_policy(*, count=1, prune=True):
    return ArtifactRetentionPolicy(
        rules=(
            KeepBest(
                name="objective_elite",
                property="objective",
                direction="min",
                count=count,
            ),
        ),
        prune=prune,
    )


def test_owned_retention_prunes_sources_only_after_checkpoint_commit(owned_ga, tmp_path):
    checkpoint = tmp_path / "retained.json"
    minimizer = _make_owned_checkpoint_minimizer(
        owned_ga,
        _owned_checkpoint_energy(tmp_path),
        generations=2,
        population_size=4,
        keep_top_pct=25,
        reuse_carryover_evaluations=True,
        retention_policy=_objective_retention_policy(),
    )

    _energy, best_path = minimizer.run_GA(unique_id=301, checkpoint_file=checkpoint)

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    store_state = state["state"]["artifact_store"]
    assert store_state["policy_signature"] == minimizer.retention_policy.signature
    assert state["state"]["last_generation_evaluations"]
    assert all(
        "structure_path" not in summary and "mapping" not in summary
        for summary in state["state"]["last_generation_evaluations"]
    )
    assert Path(best_path).is_file()
    assert Path(best_path).parent == checkpoint.with_suffix(".artifacts") / "structures"
    assert len(list(tmp_path.glob("*.owned.pending"))) == minimizer.population_size

    records = store_state["records"]
    assert records
    assert any(record["retention_reasons"] for record in records)
    assert any("best_result" in record["pins"] for record in records)
    for record in records:
        source_path = record["source_path"]
        if source_path is not None:
            assert not Path(source_path).exists()
        archive_path = record["archive_path"]
        if archive_path is not None:
            assert Path(archive_path).is_file()


def test_owned_retention_archive_preserves_explicit_reconstruction_metadata(
    owned_ga,
    tmp_path,
):
    checkpoint = tmp_path / "ownership-retained.json"
    minimizer = _make_owned_checkpoint_minimizer(
        owned_ga,
        _owned_checkpoint_energy(tmp_path),
        generations=1,
        population_size=2,
        keep_top_pct=50,
        retention_policy=_objective_retention_policy(),
    )

    _energy, best_path = minimizer.run_GA(unique_id=302, checkpoint_file=checkpoint)

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    best_state = state["state"]["best_evaluation"]
    candidate_id = best_state["candidate_id"]
    assert candidate_id in state["state"]["retention_archive_mappings"]
    assert best_state["structure_path"] == best_path

    resumed = _make_owned_checkpoint_minimizer(
        owned_ga,
        _owned_checkpoint_energy(tmp_path),
        generations=1,
        population_size=2,
        keep_top_pct=50,
        retention_policy=_objective_retention_policy(),
    )
    resumed.run_GA(unique_id=302, checkpoint_file=checkpoint)
    restored = resumed.best_evaluation.manipulator.parents[0]
    np.testing.assert_array_equal(restored.grain_labels, owned_ga[3])
    assert restored.gb_plane_x == pytest.approx(best_state["mapping"]["gb_plane_x"])


def test_owned_retention_property_provider_receives_relaxed_candidate_state(
    owned_ga,
    tmp_path,
):
    observed = []

    def provider(context):
        observed.append((context.candidate_id, context.atoms.flags.writeable))
        return {"x_mean": float(np.mean(context.atoms["x"]))}

    policy = ArtifactRetentionPolicy(
        rules=(
            KeepBest(
                name="lowest_x_mean",
                property="x_mean",
                direction="min",
                count=1,
            ),
        ),
        property_provider=provider,
        property_provider_version="1",
        prune=False,
    )
    minimizer = _make_owned_checkpoint_minimizer(
        owned_ga,
        _owned_checkpoint_energy(tmp_path),
        generations=1,
        population_size=2,
        keep_top_pct=50,
        retention_policy=policy,
    )

    minimizer.run_GA(unique_id=303, checkpoint_file=tmp_path / "provider.json")

    assert observed[0][0] == "GA_initial303"
    assert any(candidate_id.startswith("GA_303_g0_c") for candidate_id, _ in observed)
    assert all(writeable is False for _candidate_id, writeable in observed)


def test_owned_retention_policy_mismatch_on_resume_fails_explicitly(owned_ga, tmp_path):
    checkpoint = tmp_path / "policy-mismatch.json"
    partial = _make_owned_checkpoint_minimizer(
        owned_ga,
        _owned_checkpoint_energy(tmp_path),
        generations=1,
        population_size=2,
        keep_top_pct=50,
        retention_policy=_objective_retention_policy(count=1),
    )
    partial.run_GA(unique_id=304, checkpoint_file=checkpoint)

    resumed = _make_owned_checkpoint_minimizer(
        owned_ga,
        _owned_checkpoint_energy(tmp_path),
        generations=2,
        population_size=2,
        keep_top_pct=50,
        retention_policy=_objective_retention_policy(count=2),
    )

    with pytest.raises(GBMinimizerError, match="policy signature mismatch"):
        resumed.run_GA(unique_id=304, checkpoint_file=checkpoint)


def test_owned_retention_checkpoint_contains_no_callback_object(owned_ga, tmp_path):
    def provider(context):
        return {"x_mean": float(np.mean(context.atoms["x"]))}

    policy = ArtifactRetentionPolicy(
        rules=(
            KeepBest(
                name="lowest_x_mean",
                property="x_mean",
                direction="min",
                count=1,
            ),
        ),
        property_provider=provider,
        property_provider_version="callback-v1",
        prune=False,
    )
    checkpoint = tmp_path / "callback.json"
    minimizer = _make_owned_checkpoint_minimizer(
        owned_ga,
        _owned_checkpoint_energy(tmp_path),
        generations=1,
        population_size=2,
        keep_top_pct=50,
        retention_policy=policy,
    )

    minimizer.run_GA(unique_id=305, checkpoint_file=checkpoint)

    text = checkpoint.read_text(encoding="utf-8")
    state = json.loads(text)
    assert "function provider" not in text
    assert state["state"]["artifact_store"]["policy_signature"] == policy.signature


def test_owned_carryover_cache_is_rebased_before_source_pruning(owned_ga, tmp_path):
    checkpoint = tmp_path / "rebased-cache.json"
    partial = _make_owned_checkpoint_minimizer(
        owned_ga,
        _owned_checkpoint_energy(tmp_path),
        generations=1,
        population_size=4,
        keep_top_pct=25,
        reuse_carryover_evaluations=True,
        retention_policy=_objective_retention_policy(),
    )
    partial.run_GA(unique_id=306, checkpoint_file=checkpoint)

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    cached = [
        item
        for item in state["state"]["population_cached_evaluations"]
        if item is not None
    ]
    assert len(cached) == 1
    assert cached[0]["structure_path"].endswith(".owned.pending")
    assert Path(cached[0]["structure_path"]).is_file()
    for record in state["state"]["artifact_store"]["records"]:
        if record["source_path"] is not None:
            assert not Path(record["source_path"]).exists()

    resumed_ids = []
    energy = _owned_checkpoint_energy(tmp_path)

    def tracking_energy(GB, manipulator, atom_positions, unique_id):
        resumed_ids.append(str(unique_id))
        return energy(GB, manipulator, atom_positions, unique_id)

    resumed = _make_owned_checkpoint_minimizer(
        owned_ga,
        tracking_energy,
        generations=2,
        population_size=4,
        keep_top_pct=25,
        reuse_carryover_evaluations=True,
        retention_policy=_objective_retention_policy(),
    )
    resumed.run_GA(unique_id=306, checkpoint_file=checkpoint)

    assert "GA_306_g1_c0" not in resumed_ids


def test_owned_pruning_requires_durable_checkpoint(owned_ga, tmp_path):
    minimizer = _make_owned_checkpoint_minimizer(
        owned_ga,
        _owned_checkpoint_energy(tmp_path),
        generations=1,
        population_size=2,
        keep_top_pct=50,
        retention_policy=_objective_retention_policy(),
    )

    with pytest.raises(GBMinimizerValueError, match="requires checkpoint_file"):
        minimizer.run_GA(unique_id=307)



def test_owned_retention_none_preserves_evaluator_sources(owned_ga, tmp_path):
    checkpoint = tmp_path / "keep-all.json"
    minimizer = _make_owned_checkpoint_minimizer(
        owned_ga,
        _owned_checkpoint_energy(tmp_path),
        generations=1,
        population_size=2,
        keep_top_pct=50,
        retention_policy=None,
    )

    _energy, best_path = minimizer.run_GA(unique_id=309, checkpoint_file=checkpoint)

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert state["state"]["artifact_store"] is None
    assert Path(best_path).parent == tmp_path
    assert not checkpoint.with_suffix(".artifacts").exists()
    evaluator_sources = sorted(tmp_path.glob("GA_*.data"))
    assert evaluator_sources
    assert all(path.is_file() for path in evaluator_sources)


def test_owned_prune_resume_matches_continuous_run(owned_ga, tmp_path):
    continuous_dir = tmp_path / "continuous-retention"
    resumed_dir = tmp_path / "resumed-retention"
    continuous_dir.mkdir()
    resumed_dir.mkdir()

    continuous_checkpoint = continuous_dir / "run.json"
    continuous = _make_owned_checkpoint_minimizer(
        owned_ga,
        _owned_checkpoint_energy(continuous_dir),
        generations=3,
        population_size=4,
        keep_top_pct=25,
        reuse_carryover_evaluations=True,
        retention_policy=_objective_retention_policy(count=2),
    )
    continuous_energy, _continuous_best_path = continuous.run_GA(
        unique_id=310,
        checkpoint_file=continuous_checkpoint,
    )

    resumed_checkpoint = resumed_dir / "run.json"
    partial = _make_owned_checkpoint_minimizer(
        owned_ga,
        _owned_checkpoint_energy(resumed_dir),
        generations=1,
        population_size=4,
        keep_top_pct=25,
        reuse_carryover_evaluations=True,
        retention_policy=_objective_retention_policy(count=2),
    )
    partial.run_GA(unique_id=310, checkpoint_file=resumed_checkpoint)
    partial_state = json.loads(resumed_checkpoint.read_text(encoding="utf-8"))
    assert all(
        record["source_path"] is None or not Path(record["source_path"]).exists()
        for record in partial_state["state"]["artifact_store"]["records"]
    )

    resumed = _make_owned_checkpoint_minimizer(
        owned_ga,
        _owned_checkpoint_energy(resumed_dir),
        generations=3,
        population_size=4,
        keep_top_pct=25,
        reuse_carryover_evaluations=True,
        retention_policy=_objective_retention_policy(count=2),
    )
    resumed_energy, _resumed_best_path = resumed.run_GA(
        unique_id=310,
        checkpoint_file=resumed_checkpoint,
    )

    assert resumed_energy == pytest.approx(continuous_energy)
    assert len(resumed.GBE_vals) == len(continuous.GBE_vals)
    for expected, actual in zip(continuous.GBE_vals, resumed.GBE_vals, strict=True):
        np.testing.assert_allclose(actual, expected)

    continuous_state = json.loads(continuous_checkpoint.read_text(encoding="utf-8"))[
        "state"
    ]
    resumed_state = json.loads(resumed_checkpoint.read_text(encoding="utf-8"))["state"]
    assert resumed_state["population_retention_lineages"] == continuous_state[
        "population_retention_lineages"
    ]
    assert resumed_state["last_generation_evaluations"] == continuous_state[
        "last_generation_evaluations"
    ]
    assert resumed_state["best_evaluation"]["candidate_id"] == continuous_state[
        "best_evaluation"
    ]["candidate_id"]
    assert resumed_state["best_evaluation"]["energy"] == pytest.approx(
        continuous_state["best_evaluation"]["energy"]
    )

    def normalized_store_records(state):
        return [
            {
                "candidate": record["candidate"],
                "pins": record["pins"],
                "retention_reasons": record["retention_reasons"],
                "has_archive": record["archive_path"] is not None,
            }
            for record in state["artifact_store"]["records"]
        ]

    assert normalized_store_records(resumed_state) == normalized_store_records(
        continuous_state
    )
    retained_ids = {
        record["candidate"]["candidate_id"]
        for record in resumed_state["artifact_store"]["records"]
        if record["archive_path"] is not None
    }
    assert retained_ids == {
        record["candidate"]["candidate_id"]
        for record in continuous_state["artifact_store"]["records"]
        if record["archive_path"] is not None
    }
    assert retained_ids
    assert all(
        Path(record["archive_path"]).is_file()
        for record in resumed_state["artifact_store"]["records"]
        if record["archive_path"] is not None
    )

def test_owned_cleanup_failure_leaks_source_but_checkpoint_remains_resumable(
    owned_ga,
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "cleanup-failure.json"
    original_unlink = Path.unlink

    def fail_evaluator_source_unlink(path, *args, **kwargs):
        if (
            path.suffix == ".data"
            and path.parent == tmp_path
            and path.name.startswith("GA_")
        ):
            raise OSError("simulated source cleanup failure")
        return original_unlink(path, *args, **kwargs)

    minimizer = _make_owned_checkpoint_minimizer(
        owned_ga,
        _owned_checkpoint_energy(tmp_path),
        generations=1,
        population_size=2,
        keep_top_pct=50,
        retention_policy=_objective_retention_policy(),
    )
    with monkeypatch.context() as patch_context:
        patch_context.setattr(Path, "unlink", fail_evaluator_source_unlink)
        with pytest.warns(RuntimeWarning, match="Artifact cleanup failed"):
            minimizer.run_GA(unique_id=308, checkpoint_file=checkpoint)

    assert checkpoint.is_file()
    resumed = _make_owned_checkpoint_minimizer(
        owned_ga,
        _owned_checkpoint_energy(tmp_path),
        generations=2,
        population_size=2,
        keep_top_pct=50,
        retention_policy=_objective_retention_policy(),
    )
    resumed.run_GA(unique_id=308, checkpoint_file=checkpoint)
    assert len(resumed.history) == 2
