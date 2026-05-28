# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from GBOpt.Checkpoint import (
    CandidateCheckpoint,
    CheckpointError,
    CheckpointStore,
    CheckpointValueError,
    _wrap_batch_func_with_checkpoint,
)


class TestCheckpointStore(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cp_path = Path(self.tmpdir.name) / "run.json"

    def tearDown(self):
        self.tmpdir.cleanup()

    def _make_state(self, index=1):
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

    # ------------------------------------------------------------------ disabled store

    def test_disabled_store_is_noop(self):
        store = CheckpointStore.disabled()
        self.assertFalse(store.enabled)
        self.assertFalse(store.exists)
        store.save_if_due(1, lambda: self._make_state())
        store.save_final(self._make_state())
        store.delete()
        self.assertIsNone(store.load())

    def test_from_optional_returns_disabled_when_path_is_none(self):
        store = CheckpointStore.from_optional(None)
        self.assertFalse(store.enabled)

    # ------------------------------------------------------------------ format validation

    def test_from_optional_invalid_format_raises(self):
        with self.assertRaises(CheckpointValueError):
            CheckpointStore.from_optional(self.cp_path, fmt="invalid")

    # ------------------------------------------------------------------ save_if_due

    def test_save_if_due_respects_interval(self):
        store = CheckpointStore.from_optional(self.cp_path, interval=3)
        store.save_if_due(1, lambda: self._make_state(1))
        self.assertFalse(self.cp_path.exists())
        store.save_if_due(2, lambda: self._make_state(2))
        self.assertFalse(self.cp_path.exists())
        store.save_if_due(3, lambda: self._make_state(3))
        self.assertTrue(self.cp_path.exists())
        with open(self.cp_path) as f:
            saved = json.load(f)
        self.assertEqual(saved["progress_index"], 3)

    def test_save_if_due_noop_when_disabled(self):
        store = CheckpointStore.disabled()
        store.save_if_due(1, lambda: self._make_state())
        self.assertFalse(self.cp_path.exists())

    def test_state_fn_not_called_when_not_due(self):
        calls = [0]

        def counting_fn():
            calls[0] += 1
            return self._make_state()

        store = CheckpointStore.from_optional(self.cp_path, interval=5)
        store.save_if_due(1, counting_fn)
        store.save_if_due(2, counting_fn)
        self.assertEqual(calls[0], 0)

    # ------------------------------------------------------------------ save_final

    def test_save_final_bypasses_interval(self):
        store = CheckpointStore.from_optional(self.cp_path, interval=100)
        store.save_final(self._make_state(7))
        self.assertTrue(self.cp_path.exists())
        with open(self.cp_path) as f:
            saved = json.load(f)
        self.assertEqual(saved["progress_index"], 7)

    def test_save_final_noop_when_disabled(self):
        store = CheckpointStore.disabled()
        store.save_final(self._make_state())
        self.assertFalse(self.cp_path.exists())

    # ------------------------------------------------------------------ load

    def test_load_returns_none_when_no_file(self):
        store = CheckpointStore.from_optional(self.cp_path)
        self.assertIsNone(store.load())

    def test_load_returns_state_when_file_exists(self):
        store = CheckpointStore.from_optional(self.cp_path)
        store.save_final(self._make_state(5))
        state = store.load()
        self.assertIsNotNone(state)
        self.assertEqual(state["progress_index"], 5)

    def test_load_raises_on_corrupted_file(self):
        self.cp_path.write_bytes(b"not valid json {{{")
        store = CheckpointStore.from_optional(self.cp_path)
        with self.assertRaises(CheckpointError):
            store.load()

    def test_load_pickle_format(self):
        pkl_path = Path(self.tmpdir.name) / "run.pkl"
        store = CheckpointStore.from_optional(pkl_path, fmt="pickle")
        store.save_final(self._make_state(9))
        state = store.load()
        self.assertEqual(state["progress_index"], 9)

    # ------------------------------------------------------------------ delete

    def test_delete_removes_file(self):
        store = CheckpointStore.from_optional(self.cp_path)
        store.save_final(self._make_state())
        self.assertTrue(self.cp_path.exists())
        store.delete()
        self.assertFalse(self.cp_path.exists())

    def test_delete_safe_when_no_file(self):
        store = CheckpointStore.from_optional(self.cp_path)
        store.delete()  # file never created; should not raise
        self.assertFalse(self.cp_path.exists())

    def test_delete_safe_when_disabled(self):
        store = CheckpointStore.disabled()
        store.delete()
        self.assertFalse(self.cp_path.exists())

    # ------------------------------------------------------------------ JSON serialization

    def test_json_serialization_handles_numpy_array(self):
        store = CheckpointStore.from_optional(self.cp_path)
        state = self._make_state()
        state["state"]["arr"] = np.array([1.0, 2.0, 3.0])
        store.save_final(state)
        with open(self.cp_path) as f:
            saved = json.load(f)
        self.assertEqual(saved["state"]["arr"], [1.0, 2.0, 3.0])

    def test_json_serialization_handles_numpy_scalar(self):
        store = CheckpointStore.from_optional(self.cp_path)
        state = self._make_state()
        state["best_energy"] = np.float64(-2.5)
        store.save_final(state)
        with open(self.cp_path) as f:
            saved = json.load(f)
        self.assertAlmostEqual(saved["best_energy"], -2.5)

    def test_json_serialization_handles_path(self):
        store = CheckpointStore.from_optional(self.cp_path)
        state = self._make_state()
        state["best_dump"] = Path("/some/dump.data")
        store.save_final(state)
        with open(self.cp_path) as f:
            saved = json.load(f)
        self.assertEqual(saved["best_dump"], "/some/dump.data")

    def test_envelope_schema_version_present(self):
        store = CheckpointStore.from_optional(self.cp_path)
        store.save_final(self._make_state())
        with open(self.cp_path) as f:
            saved = json.load(f)
        self.assertIn("schema_version", saved)
        self.assertEqual(saved["schema_version"], 1)

    # ------------------------------------------------------------------ exists property

    def test_exists_true_after_save(self):
        store = CheckpointStore.from_optional(self.cp_path)
        self.assertFalse(store.exists)
        store.save_final(self._make_state())
        self.assertTrue(store.exists)

    def test_exists_false_when_disabled(self):
        store = CheckpointStore.disabled()
        self.assertFalse(store.exists)

    def test_zero_interval_raises(self):
        with self.assertRaises(CheckpointValueError):
            CheckpointStore.from_optional(self.cp_path, interval=0)

    def test_negative_interval_raises(self):
        with self.assertRaises(CheckpointValueError):
            CheckpointStore.from_optional(self.cp_path, interval=-1)


class TestCandidateCheckpoint(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.main_cp = Path(self.tmpdir.name) / "run.json"
        self.uids = ["run_i0_c0", "run_i0_c1", "run_i0_c2"]

    def tearDown(self):
        self.tmpdir.cleanup()

    def _make_fresh(self, fmt="json"):
        return CandidateCheckpoint(
            self.main_cp.with_suffix(f".iter0{self.main_cp.suffix}"),
            fmt,
            iteration_index=0,
            unique_ids=self.uids,
        )

    def test_new_creates_fresh_checkpoint(self):
        cp = self._make_fresh()
        for uid in self.uids:
            self.assertFalse(cp.is_done(uid))

    def test_invalid_format_raises(self):
        with self.assertRaises(CheckpointValueError):
            CandidateCheckpoint(
                self.main_cp, "yaml", iteration_index=0, unique_ids=self.uids)

    def test_record_marks_done(self):
        cp = self._make_fresh()
        cp.record("run_i0_c0", 1.23, "/tmp/dump.data")
        self.assertTrue(cp.is_done("run_i0_c0"))
        self.assertFalse(cp.is_done("run_i0_c1"))

    def test_record_saves_to_disk_atomically(self):
        cp = self._make_fresh()
        iter_path = CandidateCheckpoint._derive_path(self.main_cp, 0)
        self.assertFalse(iter_path.exists())
        cp.record("run_i0_c0", 2.5, "/tmp/d.data")
        self.assertTrue(iter_path.exists())
        with open(iter_path) as f:
            payload = json.load(f)
        self.assertIn("run_i0_c0", payload["results"])

    def test_get_result_returns_energy_and_dump(self):
        cp = self._make_fresh()
        cp.record("run_i0_c1", 3.14, "/tmp/out.data")
        energy, dump = cp.get_result("run_i0_c1")
        self.assertAlmostEqual(energy, 3.14)
        self.assertEqual(dump, "/tmp/out.data")

    def test_get_result_raises_if_not_done(self):
        cp = self._make_fresh()
        with self.assertRaises(CheckpointError):
            cp.get_result("run_i0_c0")

    def test_get_result_dump_none_on_failure(self):
        cp = self._make_fresh()
        cp.record("run_i0_c2", 1.0e30, None)
        energy, dump = cp.get_result("run_i0_c2")
        self.assertIsNone(dump)

    def test_load_restores_state_json(self):
        cp = self._make_fresh()
        cp.record("run_i0_c0", 1.1, "/a.data")
        cp.record("run_i0_c1", 2.2, "/b.data")

        iter_path = CandidateCheckpoint._derive_path(self.main_cp, 0)
        restored = CandidateCheckpoint._load(iter_path, "json", 0, self.uids)

        self.assertTrue(restored.is_done("run_i0_c0"))
        self.assertTrue(restored.is_done("run_i0_c1"))
        self.assertFalse(restored.is_done("run_i0_c2"))
        self.assertAlmostEqual(restored.get_result("run_i0_c0")[0], 1.1)

    def test_load_restores_state_pickle(self):
        cp = CandidateCheckpoint(
            self.main_cp.with_suffix(".iter0.pkl"),
            "pickle",
            iteration_index=0,
            unique_ids=self.uids,
        )
        cp.record("run_i0_c2", 5.5, "/c.data")

        iter_path = self.main_cp.with_suffix(".iter0.pkl")
        restored = CandidateCheckpoint._load(
            iter_path, "pickle", 0, self.uids)
        self.assertTrue(restored.is_done("run_i0_c2"))

    def test_load_unknown_uid_treated_as_not_done(self):
        cp = self._make_fresh()
        cp.record("run_i0_c0", 1.0, "/x.data")

        iter_path = CandidateCheckpoint._derive_path(self.main_cp, 0)
        extra_uids = self.uids + ["run_i0_c99"]
        restored = CandidateCheckpoint._load(
            iter_path, "json", 0, extra_uids)
        self.assertFalse(restored.is_done("run_i0_c99"))

    def test_load_corrupted_file_raises(self):
        iter_path = CandidateCheckpoint._derive_path(self.main_cp, 0)
        iter_path.write_bytes(b"not json {{{{")
        with self.assertRaises(CheckpointError):
            CandidateCheckpoint._load(iter_path, "json", 0, self.uids)

    def test_derive_path(self):
        main = Path("/some/dir/run.json")
        self.assertEqual(
            CandidateCheckpoint._derive_path(main, 3),
            Path("/some/dir/run.iter3.json"),
        )
        main_pkl = Path("/some/dir/run.pkl")
        self.assertEqual(
            CandidateCheckpoint._derive_path(main_pkl, 0),
            Path("/some/dir/run.iter0.pkl"),
        )

    def test_delete_removes_file(self):
        cp = self._make_fresh()
        cp.record("run_i0_c0", 1.0, "/x.data")
        iter_path = CandidateCheckpoint._derive_path(self.main_cp, 0)
        self.assertTrue(iter_path.exists())
        cp.delete()
        self.assertFalse(iter_path.exists())

    def test_delete_is_safe_when_file_absent(self):
        cp = self._make_fresh()
        cp.delete()  # file never created; should not raise

    def test_new_or_resume_creates_fresh_when_no_file(self):
        cp = CandidateCheckpoint.new_or_resume(
            self.main_cp, "json", 0, self.uids)
        for uid in self.uids:
            self.assertFalse(cp.is_done(uid))

    def test_new_or_resume_loads_existing_file(self):
        cp = self._make_fresh()
        cp.record("run_i0_c1", 7.7, "/y.data")

        resumed = CandidateCheckpoint.new_or_resume(
            self.main_cp, "json", 0, self.uids)
        self.assertTrue(resumed.is_done("run_i0_c1"))
        self.assertFalse(resumed.is_done("run_i0_c0"))


class TestWrapBatchFuncWithCheckpoint(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.uids = ["uid0", "uid1", "uid2"]

    def tearDown(self):
        self.tmpdir.cleanup()

    def _make_checkpoint(self):
        path = Path(self.tmpdir.name) / "run.iter0.json"
        return CandidateCheckpoint(path, "json", 0, self.uids)

    def _make_batch_results(self, uids):
        return [{"energy": float(i), "final_dump": f"/dump_{u}.data"}
                for i, u in enumerate(uids)]

    def test_wrapped_func_records_results_post_batch(self):
        def plain_batch(GB, manips, structs, lineages, unique_ids):
            return self._make_batch_results(unique_ids)

        wrapped = _wrap_batch_func_with_checkpoint(plain_batch)
        cp = self._make_checkpoint()
        wrapped(None, [], [], [], self.uids, checkpoint=cp)

        for uid in self.uids:
            self.assertTrue(cp.is_done(uid))

    def test_wrapped_func_skips_already_done_uids(self):
        cp = self._make_checkpoint()
        cp.record("uid0", 99.0, "/existing.data")

        def plain_batch(GB, manips, structs, lineages, unique_ids):
            return self._make_batch_results(unique_ids)

        wrapped = _wrap_batch_func_with_checkpoint(plain_batch)
        wrapped(None, [], [], [], self.uids, checkpoint=cp)

        # uid0 was already done so should not be re-recorded with a new value
        energy, _ = cp.get_result("uid0")
        self.assertAlmostEqual(energy, 99.0)

    def test_wrapped_func_works_without_checkpoint(self):
        def plain_batch(GB, manips, structs, lineages, unique_ids):
            return self._make_batch_results(unique_ids)

        wrapped = _wrap_batch_func_with_checkpoint(plain_batch)
        results = wrapped(None, [], [], [], self.uids)
        self.assertEqual(len(results), 3)

    def test_wrapped_func_preserves_return_value(self):
        expected = [{"energy": 1.0, "final_dump": "/a.data"}]

        def plain_batch(GB, manips, structs, lineages, unique_ids):
            return expected

        wrapped = _wrap_batch_func_with_checkpoint(plain_batch)
        result = wrapped(None, [], [], [], ["uid0"])
        self.assertIs(result, expected)

    def test_wrapped_func_preserves_name(self):
        def my_batch_func(GB, manips, structs, lineages, unique_ids):
            return []

        wrapped = _wrap_batch_func_with_checkpoint(my_batch_func)
        self.assertEqual(wrapped.__name__, "my_batch_func")


if __name__ == "__main__":
    unittest.main()
